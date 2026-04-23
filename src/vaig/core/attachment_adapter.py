"""Attachment adapter protocol and concrete implementations (SPEC-ATT-02/03/04).

Provides adapters to read local directories and single files as attachments
for the ``vaig live`` command.  URL and archive adapters land in future sprints.

Adapters:

- ``LocalPathAdapter``  – non-git directory tree (SPEC-ATT-03)
- ``SingleFileAdapter`` – individual file (SPEC-ATT-04)

Use ``resolve_attachment(raw, name=..., cfg=...)`` to select the right adapter.
"""

import hashlib
import logging
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from vaig.core.config import AttachmentsConfig

logger = logging.getLogger(__name__)

# ── Re-use exclude globs from config — imported lazily to avoid circular import ──
_ARCHIVE_SUFFIXES: frozenset[str] = frozenset(
    {".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz", ".tar.xz", ".txz"}
)


# ── Enums & core models ───────────────────────────────────────────────────────


class AttachmentKind(StrEnum):
    """Discriminator for resolved attachment type."""

    local_path = "local_path"
    single_file = "single_file"
    archive = "archive"
    url = "url"
    git_clone = "git_clone"


class AttachmentSpec(BaseModel):
    """Resolved metadata for a single attachment source."""

    name: str | None = None
    """User-visible label (from ``--attach-name``, may be ``None``)."""
    source: str
    """Raw user input (the value passed to ``--attach``)."""
    kind: AttachmentKind
    resolved_path: Path | None = None
    """Absolute path on disk.  ``None`` for URL kind."""


@dataclass
class AttachmentFileEntry:
    """Metadata for a single file yielded by an adapter's ``list_files()``."""

    relative_path: str
    """Path relative to the adapter root."""
    size_bytes: int
    mtime: float | None
    is_symlink: bool


# ── Protocol ─────────────────────────────────────────────────────────────────


@runtime_checkable
class AttachmentAdapter(Protocol):
    """Adapter protocol for a single attachment source.

    Analogous to ``RepoAdapter`` but without a ``ref`` parameter — attachments
    are always at-a-point-in-time snapshots from the local filesystem.
    """

    spec: AttachmentSpec

    def list_files(self, cfg: "AttachmentsConfig") -> Iterable[AttachmentFileEntry]:
        """Return an iterable of every file in this attachment."""
        ...

    def fetch_bytes(
        self, relative_path: str
    ) -> bytes | Iterator[bytes]:
        """Return file content as ``bytes`` or a streaming ``Iterator[bytes]``.

        For large files (``> cfg.streaming_threshold_bytes``) implementations
        SHOULD return an iterator to avoid RAM spikes.

        Raises:
            ValueError: If *relative_path* is outside the adapter root.
            OSError: If the file cannot be read.
        """
        ...

    def fingerprint(self) -> str:
        """Return a stable content-hash seed for caching (Sprint 3).

        Derived from: kind + resolved_path + latest mtime + sorted file list
        hash.  The returned string is deterministic for the same content state.
        """
        ...


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_default_excludes() -> list[str]:
    """Return a copy of ``_DEFAULT_EXCLUDE_GLOBS`` from config."""
    from vaig.core.config import _DEFAULT_EXCLUDE_GLOBS  # lazy import

    return list(_DEFAULT_EXCLUDE_GLOBS)


def _path_matches_globs(rel_path: str, patterns: list[str]) -> bool:
    """Match *rel_path* against *patterns* using full ``**`` glob support.

    Delegates to :func:`~vaig.core.repo_pipeline._glob_match` which handles
    ``**/foo/*.txt``-style patterns correctly.
    """
    from vaig.core.repo_pipeline import _glob_match  # reuse tested implementation

    normalized = rel_path.replace(os.sep, "/")
    for pat in patterns:
        pat_norm = pat.replace(os.sep, "/")
        if _glob_match(normalized, pat_norm):
            return True
    return False


# ── LocalPathAdapter ──────────────────────────────────────────────────────────


class LocalPathAdapter:
    """Adapter for a non-git local directory tree (SPEC-ATT-03).

    Walks the directory tree respecting depth limits, exclude globs, symlink
    policy, and file-count caps from ``AttachmentsConfig``.
    """

    def __init__(self, root: Path, spec: AttachmentSpec) -> None:
        self.root = root.resolve()
        self.spec = spec
        # Populated lazily on first list_files() call to support fingerprint().
        self._listed: list[AttachmentFileEntry] | None = None
        self._cfg_ref: AttachmentsConfig | None = None  # keep last cfg for fingerprint

    # ------------------------------------------------------------------
    # AttachmentAdapter interface
    # ------------------------------------------------------------------

    def list_files(self, cfg: "AttachmentsConfig") -> Iterable[AttachmentFileEntry]:
        """Walk *self.root* and yield :class:`AttachmentFileEntry` objects.

        Respects:
        - ``cfg.max_depth`` (``-1`` = unlimited)
        - ``cfg.follow_symlinks``
        - ``cfg.use_default_excludes`` + ``cfg.extra_excludes``
        - ``cfg.max_files_per_attachment`` (+ ``cfg.unlimited_files`` bypass)
        - ``cfg.include_everything`` (bypasses excludes, depth, binary-skip)
        - ``cfg.binary_skip`` (deferred — binary detection happens in fetch_bytes)
        """
        self._cfg_ref = cfg
        entries: list[AttachmentFileEntry] = []

        include_everything: bool = getattr(cfg, "include_everything", False)
        use_default_excludes: bool = getattr(cfg, "use_default_excludes", True) and not include_everything
        extra_excludes: list[str] = list(getattr(cfg, "extra_excludes", []))
        max_depth: int = -1 if include_everything else int(getattr(cfg, "max_depth", -1))
        follow_symlinks: bool = bool(getattr(cfg, "follow_symlinks", False))
        max_files: int = int(getattr(cfg, "max_files_per_attachment", 10_000))
        unlimited_files: bool = bool(getattr(cfg, "unlimited_files", False)) or include_everything

        exclude_globs: list[str] = []
        if use_default_excludes:
            exclude_globs.extend(_get_default_excludes())
        exclude_globs.extend(extra_excludes)

        file_count = 0
        limit_logged = False

        for dirpath, dirnames, filenames in os.walk(
            self.root,
            followlinks=follow_symlinks,
        ):
            dir_p = Path(dirpath)

            # Depth check — also prune dirnames to avoid descending further
            if max_depth >= 0:
                try:
                    rel_dir = dir_p.relative_to(self.root)
                    depth = len(rel_dir.parts)
                    if depth >= max_depth:
                        dirnames.clear()
                        if depth > max_depth:
                            continue
                except ValueError:
                    pass

            # Directory-level jail check: when follow_symlinks=True, skip dirs
            # that resolve outside the adapter root to prevent escapes.
            if follow_symlinks:
                safe_dirs = []
                for d in list(dirnames):
                    sub = dir_p / d
                    try:
                        sub.resolve().relative_to(self.root)
                        safe_dirs.append(d)
                    except ValueError:
                        logger.debug("attachment: dir symlink escape blocked: %s", sub)
                dirnames[:] = safe_dirs

            # Prune excluded directories in-place so os.walk skips them entirely
            if exclude_globs and not include_everything:
                pruned = []
                for d in list(dirnames):
                    sub_rel = str((dir_p / d).relative_to(self.root)).replace(os.sep, "/")
                    if _path_matches_globs(sub_rel, exclude_globs):
                        logger.debug("attachment: pruning excluded dir: %s", sub_rel)
                    else:
                        pruned.append(d)
                dirnames[:] = pruned

            for fname in filenames:
                file_path = dir_p / fname

                # Symlink jail — even if follow_symlinks=True, reject paths escaping root
                resolved = file_path.resolve()
                try:
                    resolved.relative_to(self.root)
                except ValueError:
                    logger.debug("attachment: symlink escape blocked: %s", file_path)
                    continue

                try:
                    rel = str(file_path.relative_to(self.root))
                except ValueError:
                    continue

                # Exclude glob check on files
                if exclude_globs and _path_matches_globs(rel, exclude_globs):
                    continue

                # Stat the file; skip if size exceeds absolute cap (early gate,
                # avoids reading the file just for binary sniff)
                try:
                    stat = file_path.stat(follow_symlinks=False)
                    size_bytes = stat.st_size
                    mtime = stat.st_mtime
                except OSError:
                    size_bytes = 0
                    mtime = None

                # File count cap
                if not unlimited_files and file_count >= max_files:
                    if not limit_logged:
                        logger.warning(
                            "attachment: max_files_per_attachment=%d reached for %s — "
                            "remaining files skipped",
                            max_files,
                            self.root,
                        )
                        limit_logged = True
                    continue

                entries.append(
                    AttachmentFileEntry(
                        relative_path=rel,
                        size_bytes=size_bytes,
                        mtime=mtime,
                        is_symlink=file_path.is_symlink(),
                    )
                )
                file_count += 1

        self._listed = entries
        return entries

    def fetch_bytes(self, relative_path: str) -> bytes | Iterator[bytes]:
        """Return file content, streaming if large.

        Binary detection (``cfg.binary_skip``) is enforced here at fetch time,
        not during listing, to avoid reading every file just for sniff.

        Raises:
            ValueError: If path traversal outside root is attempted.
            ValueError: If file exceeds ``cfg.max_bytes_absolute``.
            OSError: If the file cannot be read.
        """
        from vaig.core.repo_pipeline import is_binary_file  # invariant AT-3

        cfg = self._cfg_ref
        streaming_threshold: int = int(getattr(cfg, "streaming_threshold_bytes", 2_000_000))
        max_bytes: int = int(getattr(cfg, "max_bytes_absolute", 500_000_000))
        binary_skip: bool = bool(getattr(cfg, "binary_skip", True))
        include_everything: bool = bool(getattr(cfg, "include_everything", False))

        # Normalise and jail-check
        target = (self.root / relative_path).resolve()
        try:
            target.relative_to(self.root)
        except ValueError:
            raise ValueError(
                f"Path traversal rejected: {relative_path!r} resolves outside {self.root}"
            ) from None

        size = target.stat().st_size
        if size > max_bytes:
            raise ValueError(
                f"File {relative_path!r} is {size} bytes, exceeds max_bytes_absolute={max_bytes}"
            )

        # Deferred binary sniff — only when actually fetching
        if binary_skip and not include_everything and not target.is_symlink():
            if is_binary_file(target):
                raise ValueError(
                    f"File {relative_path!r} is binary; set binary_skip=False to read it"
                )

        if size > streaming_threshold:
            return self._stream_file(target)
        return target.read_bytes()

    def fingerprint(self) -> str:
        """Stable hash seed from kind + path + mtimes + sorted file list."""
        cfg = self._cfg_ref
        if self._listed is None:
            if cfg is None:
                # Use a minimal default config for fingerprint calls before list_files
                from vaig.core.config import AttachmentsConfig as RealCfg

                cfg = RealCfg()
            list(self.list_files(cfg))

        h = hashlib.sha256()
        h.update(f"local_path:{self.root}".encode())
        for entry in sorted(self._listed or [], key=lambda e: e.relative_path):
            h.update(f"{entry.relative_path}:{entry.size_bytes}:{entry.mtime}".encode())
        return h.hexdigest()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stream_file(path: Path, chunk_size: int = 65_536) -> Iterator[bytes]:
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                yield chunk


# ── SingleFileAdapter ─────────────────────────────────────────────────────────


class SingleFileAdapter:
    """Adapter for a single file (SPEC-ATT-04)."""

    def __init__(self, path: Path, spec: AttachmentSpec) -> None:
        self.path = path.resolve()
        self.spec = spec
        self._cfg_ref: AttachmentsConfig | None = None

    # ------------------------------------------------------------------
    # AttachmentAdapter interface
    # ------------------------------------------------------------------

    def list_files(self, cfg: "AttachmentsConfig") -> Iterable[AttachmentFileEntry]:
        """Yield exactly one :class:`AttachmentFileEntry`."""
        self._cfg_ref = cfg
        try:
            stat = self.path.stat(follow_symlinks=False)
            size_bytes = stat.st_size
            mtime = stat.st_mtime
        except OSError:
            size_bytes = 0
            mtime = None

        return [
            AttachmentFileEntry(
                relative_path="",
                size_bytes=size_bytes,
                mtime=mtime,
                is_symlink=self.path.is_symlink(),
            )
        ]

    def fetch_bytes(self, relative_path: str = "") -> bytes | Iterator[bytes]:
        """Return file content, streaming if large.

        *relative_path* is ignored (there is only one file); pass ``""`` or
        the file name.  Enforces ``cfg.max_bytes_absolute``.
        """
        cfg = self._cfg_ref
        streaming_threshold: int = int(getattr(cfg, "streaming_threshold_bytes", 2_000_000))
        max_bytes: int = int(getattr(cfg, "max_bytes_absolute", 500_000_000))

        size = self.path.stat().st_size
        if size > max_bytes:
            raise ValueError(
                f"File {self.path} is {size} bytes, exceeds max_bytes_absolute={max_bytes}"
            )

        if size > streaming_threshold:
            return LocalPathAdapter._stream_file(self.path)
        return self.path.read_bytes()

    def fingerprint(self) -> str:
        """Return a stable hash seed for this file."""
        h = hashlib.sha256()
        h.update(f"single_file:{self.path}".encode())
        try:
            stat = self.path.stat()
            h.update(f"{stat.st_size}:{stat.st_mtime}".encode())
        except OSError:
            pass
        return h.hexdigest()


# ── Dispatcher ────────────────────────────────────────────────────────────────


def _is_archive(raw: str) -> bool:
    p = Path(raw)
    name = p.name.lower()
    return any(name.endswith(suffix) for suffix in _ARCHIVE_SUFFIXES)


def resolve_attachment(
    raw: str,
    *,
    name: str | None = None,
    cfg: "AttachmentsConfig",
) -> "LocalPathAdapter | SingleFileAdapter":
    """Select the right attachment adapter from a raw user-supplied string.

    Decision tree (order matters):

    1. ``http://`` / ``https://``  → :exc:`NotImplementedError` (Sprint 3)
    2. Archive suffix              → :exc:`NotImplementedError` (Sprint 2)
    3. Existing file path          → :class:`SingleFileAdapter`
    4. Existing dir with ``.git``  → :exc:`NotImplementedError` (Sprint 2)
    5. Existing dir (non-git)      → :class:`LocalPathAdapter`
    6. Otherwise                   → :exc:`ValueError`

    Args:
        raw:  Raw user input (value from ``--attach``).
        name: Optional label (from ``--attach-name``).
        cfg:  :class:`~vaig.core.config.AttachmentsConfig` instance.

    Returns:
        A concrete adapter implementing the :class:`AttachmentAdapter` protocol.
    """
    stripped = raw.strip()

    # 1. URL check
    if stripped.startswith("http://") or stripped.startswith("https://"):
        raise NotImplementedError(
            "URL attachments land in Sprint 3 (SPEC-ATT-06)"
        )

    # 2. Archive suffix check (path need not exist — spec says "suffix in set")
    if _is_archive(stripped):
        raise NotImplementedError(
            "Archive attachments land in Sprint 2 (SPEC-ATT-05)"
        )

    p = Path(stripped)

    # 3. Existing file
    if p.exists() and p.is_file():
        spec = AttachmentSpec(
            name=name,
            source=raw,
            kind=AttachmentKind.single_file,
            resolved_path=p.resolve(),
        )
        return SingleFileAdapter(p, spec)

    # 4. Existing directory — check for .git
    if p.exists() and p.is_dir():
        if (p / ".git").exists():
            raise NotImplementedError(
                "Git-clone attachments land in Sprint 2 (SPEC-ATT-07)"
            )
        spec = AttachmentSpec(
            name=name,
            source=raw,
            kind=AttachmentKind.local_path,
            resolved_path=p.resolve(),
        )
        return LocalPathAdapter(p, spec)

    # 5. Nothing matched
    raise ValueError(f"Cannot resolve attachment: {raw!r}")

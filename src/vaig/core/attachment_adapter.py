"""Attachment adapter protocol and concrete implementations (SPEC-ATT-02/03/04/05).

Provides adapters to read local directories, single files, archives, and
git repos as attachments for the ``vaig live`` command.

Adapters:

- ``LocalPathAdapter``      – non-git directory tree (SPEC-ATT-03)
- ``SingleFileAdapter``     – individual file (SPEC-ATT-04)
- ``ArchiveAttachmentAdapter``  – zip/tar archive (SPEC-ATT-05)
- ``GitCloneAttachmentAdapter`` – shallow git clone (SPEC-ATT-05)

Use ``resolve_attachment(raw, name=..., cfg=...)`` to select the right adapter.
"""

import hashlib
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
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


def _apply_cfg_filters(
    raw_entries: list[AttachmentFileEntry],
    cfg: "AttachmentsConfig",
    *,
    extra_exclude_globs: list[str] | None = None,
    adapter_label: str = "attachment",
) -> list[AttachmentFileEntry]:
    """Apply AttachmentsConfig filters shared by archive and git adapters.

    Enforces (in order):
    - ``extra_exclude_globs`` (adapter-specific, e.g. ``.git/**`` for clones)
    - ``cfg.use_default_excludes`` — adds ``_DEFAULT_EXCLUDE_GLOBS``
    - ``cfg.extra_excludes`` — user-supplied glob list
    - ``cfg.max_depth`` (``-1`` = unlimited) — based on path-part count
    - ``cfg.max_files_per_attachment`` (+ ``cfg.unlimited_files`` bypass)
    - ``cfg.include_everything`` bypasses excludes and depth, never file count
      bypass (that's ``unlimited_files``)

    This keeps ``ArchiveAttachmentAdapter`` and ``GitCloneAttachmentAdapter``
    aligned with ``LocalPathAdapter`` semantics (SPEC-ATT-03/05).
    """
    include_everything: bool = bool(getattr(cfg, "include_everything", False))
    use_default_excludes: bool = (
        bool(getattr(cfg, "use_default_excludes", True)) and not include_everything
    )
    user_extra_excludes: list[str] = list(getattr(cfg, "extra_excludes", []))
    max_depth: int = -1 if include_everything else int(getattr(cfg, "max_depth", -1))
    max_files: int = int(getattr(cfg, "max_files_per_attachment", 10_000))
    unlimited_files: bool = (
        bool(getattr(cfg, "unlimited_files", False)) or include_everything
    )

    exclude_globs: list[str] = list(extra_exclude_globs or [])
    if use_default_excludes:
        exclude_globs.extend(_get_default_excludes())
    exclude_globs.extend(user_extra_excludes)

    filtered: list[AttachmentFileEntry] = []
    file_count = 0
    limit_logged = False

    for entry in raw_entries:
        rel = entry.relative_path.replace(os.sep, "/")

        # Depth check — count path parts (a file at root has depth 1)
        if max_depth >= 0:
            depth = len([p for p in rel.split("/") if p])
            if depth > max_depth:
                continue

        # Exclude glob check
        if exclude_globs and _path_matches_globs(rel, exclude_globs):
            continue

        # File count cap
        if not unlimited_files and file_count >= max_files:
            if not limit_logged:
                logger.warning(
                    "%s: max_files_per_attachment=%d reached — remaining files skipped",
                    adapter_label,
                    max_files,
                )
                limit_logged = True
            continue

        filtered.append(entry)
        file_count += 1

    return filtered


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


# ── ArchiveAttachmentAdapter ──────────────────────────────────────────────────


class ArchiveAttachmentAdapter:
    """Adapter for zip/tar archives (SPEC-ATT-05).

    Extracts to a temporary directory, enforces path-traversal guards BEFORE
    extraction, and respects ``max_bytes_absolute`` / ``max_files`` caps from
    ``AttachmentsConfig``.
    """

    def __init__(self, archive_path: Path, spec: AttachmentSpec, cfg: "AttachmentsConfig") -> None:
        self.archive_path = archive_path.resolve()
        self.spec = spec
        self._cfg = cfg
        self._tempdir: str | None = None
        self._root: Path | None = None
        self._listed: list[AttachmentFileEntry] | None = None

    # ------------------------------------------------------------------
    # Context-manager support and cleanup
    # ------------------------------------------------------------------

    def __enter__(self) -> "ArchiveAttachmentAdapter":
        return self

    def __exit__(self, *_: object) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """Remove the temporary extraction directory."""
        if self._tempdir and os.path.isdir(self._tempdir):
            shutil.rmtree(self._tempdir, ignore_errors=True)
            logger.debug("attachment: cleaned up tempdir %s", self._tempdir)
            self._tempdir = None
            self._root = None

    # ------------------------------------------------------------------
    # AttachmentAdapter interface
    # ------------------------------------------------------------------

    def list_files(self, cfg: "AttachmentsConfig") -> Iterable[AttachmentFileEntry]:
        """Extract archive (if not yet done) and list all files."""
        self._cfg = cfg
        if self._root is None:
            self._extract(cfg)
        assert self._root is not None  # extraction succeeded or raised

        raw_entries: list[AttachmentFileEntry] = []
        for dirpath, _dirnames, filenames in os.walk(self._root):
            for fname in filenames:
                fp = Path(dirpath) / fname
                try:
                    rel = str(fp.relative_to(self._root))
                    stat = fp.stat(follow_symlinks=False)
                    raw_entries.append(
                        AttachmentFileEntry(
                            relative_path=rel,
                            size_bytes=stat.st_size,
                            mtime=stat.st_mtime,
                            is_symlink=fp.is_symlink(),
                        )
                    )
                except (OSError, ValueError):
                    continue

        entries = _apply_cfg_filters(
            raw_entries,
            cfg,
            adapter_label=f"archive:{self.archive_path.name}",
        )
        self._listed = entries
        return entries

    def fetch_bytes(self, relative_path: str) -> bytes | Iterator[bytes]:
        """Return file content from the extracted tempdir."""
        if self._root is None:
            self._extract(self._cfg)
        assert self._root is not None

        target = (self._root / relative_path).resolve()
        try:
            target.relative_to(self._root)
        except ValueError:
            raise ValueError(
                f"Path traversal rejected: {relative_path!r} resolves outside extraction root"
            ) from None

        cfg = self._cfg
        streaming_threshold: int = int(getattr(cfg, "streaming_threshold_bytes", 2_000_000))
        max_bytes: int = int(getattr(cfg, "max_bytes_absolute", 500_000_000))

        size = target.stat().st_size
        if size > max_bytes:
            raise ValueError(
                f"File {relative_path!r} is {size} bytes, exceeds max_bytes_absolute={max_bytes}"
            )

        if size > streaming_threshold:
            return LocalPathAdapter._stream_file(target)
        return target.read_bytes()

    def fingerprint(self) -> str:
        """Stable hash seed from archive path + size + mtime."""
        h = hashlib.sha256()
        h.update(f"archive:{self.archive_path}".encode())
        try:
            stat = self.archive_path.stat()
            h.update(f"{stat.st_size}:{stat.st_mtime}".encode())
        except OSError:
            pass
        return h.hexdigest()

    # ------------------------------------------------------------------
    # Internal extraction logic
    # ------------------------------------------------------------------

    def _extract(self, cfg: "AttachmentsConfig") -> None:
        """Validate members and extract to a fresh tempdir.

        On any failure, removes the partial tempdir before re-raising so we
        never leak state on the filesystem.
        """
        max_bytes: int = int(getattr(cfg, "max_bytes_absolute", 500_000_000))
        max_files: int = int(getattr(cfg, "max_files_per_attachment", 10_000))

        tempdir = tempfile.mkdtemp(prefix="vaig-att-")
        self._tempdir = tempdir
        tempdir_path = Path(tempdir).resolve()

        try:
            name_lower = self.archive_path.name.lower()
            if name_lower.endswith(".zip"):
                self._extract_zip(tempdir_path, max_bytes, max_files)
            else:
                self._extract_tar(tempdir_path, max_bytes, max_files)
        except (KeyboardInterrupt, SystemExit):
            shutil.rmtree(tempdir, ignore_errors=True)
            self._tempdir = None
            raise
        except Exception:
            # Cleanup partial extraction before propagating the error.
            shutil.rmtree(tempdir, ignore_errors=True)
            self._tempdir = None
            raise

        self._root = tempdir_path

    def _extract_zip(self, dest: Path, max_bytes: int, max_files: int) -> None:
        """Validate and extract a zip archive."""
        try:
            with zipfile.ZipFile(self.archive_path, "r") as zf:
                members = zf.infolist()

                # Guards — validate BEFORE extraction
                if len(members) > max_files:
                    raise ValueError(
                        f"Archive has {len(members)} members, exceeds max_files={max_files}"
                    )

                # Per-file size cap (prevents any single member from being a
                # decompression bomb). Total-size cap kept as secondary defence.
                for m in members:
                    if m.file_size > max_bytes:
                        raise ValueError(
                            f"Archive member {m.filename!r} uncompressed size "
                            f"{m.file_size} exceeds max_bytes_absolute={max_bytes}"
                        )

                total_size = sum(m.file_size for m in members)
                if total_size > max_bytes:
                    raise ValueError(
                        f"Archive uncompressed size {total_size} exceeds max_bytes_absolute={max_bytes}"
                    )

                for member in members:
                    self._validate_member_name(member.filename)

                zf.extractall(dest)  # noqa: S202

        except zipfile.BadZipFile as exc:
            raise ValueError(f"Invalid zip archive: {self.archive_path}") from exc

        # Post-extraction symlink jail
        self._reject_escaped_symlinks(dest)

    def _extract_tar(self, dest: Path, max_bytes: int, max_files: int) -> None:
        """Validate and extract a tar archive (any compression)."""
        try:
            with tarfile.open(self.archive_path, "r:*") as tf:
                members = tf.getmembers()

                if len(members) > max_files:
                    raise ValueError(
                        f"Archive has {len(members)} members, exceeds max_files={max_files}"
                    )

                # Per-file size cap (decompression-bomb guard)
                for m in members:
                    if m.isfile() and m.size > max_bytes:
                        raise ValueError(
                            f"Archive member {m.name!r} size {m.size} "
                            f"exceeds max_bytes_absolute={max_bytes}"
                        )

                total_size = sum(m.size for m in members if m.isfile())
                if total_size > max_bytes:
                    raise ValueError(
                        f"Archive uncompressed size {total_size} exceeds max_bytes_absolute={max_bytes}"
                    )

                for member in members:
                    self._validate_member_name(member.name)

                # Use filter="data" on Python 3.12+ for additional safety;
                # fall back gracefully on older versions.
                try:
                    tf.extractall(dest, filter="data")  # noqa: S202
                except TypeError:
                    tf.extractall(dest)  # noqa: S202  # pragma: no cover

        except tarfile.TarError as exc:
            raise ValueError(f"Invalid tar archive: {self.archive_path}") from exc

        self._reject_escaped_symlinks(dest)

    @staticmethod
    def _validate_member_name(name: str) -> None:
        """Reject absolute paths and ``..`` traversal components."""
        # Normalise separators
        normalised = name.replace("\\", "/")
        if normalised.startswith("/"):
            raise ValueError(f"Archive member has absolute path: {name!r}")
        parts = normalised.split("/")
        if ".." in parts:
            raise ValueError(f"Archive member contains '..' traversal: {name!r}")

    @staticmethod
    def _reject_escaped_symlinks(dest: Path) -> None:
        """Walk extracted tree and remove any symlinks that escape dest."""
        for dirpath, _dirs, files in os.walk(dest):
            for fname in files:
                fp = Path(dirpath) / fname
                if fp.is_symlink():
                    try:
                        fp.resolve().relative_to(dest)
                    except ValueError:
                        logger.warning(
                            "attachment: removing escaping symlink after extraction: %s", fp
                        )
                        fp.unlink(missing_ok=True)
                        raise ValueError(
                            f"Archive symlink {fp.name!r} resolves outside extraction directory"
                        ) from None


# ── GitCloneAttachmentAdapter ─────────────────────────────────────────────────

_GIT_URL_PREFIXES = ("git@", "git+", "git://")
_GIT_URL_HTTPS_SUFFIX = ".git"


def _is_git_url(raw: str) -> bool:
    """Return True if *raw* looks like a git remote URL."""
    stripped = raw.strip()
    if any(stripped.startswith(p) for p in _GIT_URL_PREFIXES):
        return True
    # https://…git (must end with .git to distinguish from plain web URLs)
    if stripped.startswith("https://") and stripped.endswith(_GIT_URL_HTTPS_SUFFIX):
        return True
    return False


class GitCloneAttachmentAdapter:
    """Adapter for shallow git clone (SPEC-ATT-05).

    Clones the remote URL with ``--depth=1`` into a temporary directory.
    Read-only: never pushes or modifies the remote.
    """

    def __init__(self, url: str, spec: AttachmentSpec, cfg: "AttachmentsConfig") -> None:
        self.url = url
        self.spec = spec
        self._cfg = cfg
        self._tempdir: str | None = None
        self._root: Path | None = None
        self._listed: list[AttachmentFileEntry] | None = None

    # ------------------------------------------------------------------
    # Context-manager support and cleanup
    # ------------------------------------------------------------------

    def __enter__(self) -> "GitCloneAttachmentAdapter":
        return self

    def __exit__(self, *_: object) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """Remove the temporary clone directory."""
        if self._tempdir and os.path.isdir(self._tempdir):
            shutil.rmtree(self._tempdir, ignore_errors=True)
            logger.debug("attachment: cleaned up git clone tempdir %s", self._tempdir)
            self._tempdir = None
            self._root = None

    # ------------------------------------------------------------------
    # AttachmentAdapter interface
    # ------------------------------------------------------------------

    def list_files(self, cfg: "AttachmentsConfig") -> Iterable[AttachmentFileEntry]:
        """Clone repo (if not yet done) and list all files."""
        self._cfg = cfg
        if self._root is None:
            self._clone()
        assert self._root is not None

        raw_entries: list[AttachmentFileEntry] = []
        for dirpath, _dirnames, filenames in os.walk(self._root):
            for fname in filenames:
                fp = Path(dirpath) / fname
                try:
                    rel = str(fp.relative_to(self._root))
                    stat = fp.stat(follow_symlinks=False)
                    raw_entries.append(
                        AttachmentFileEntry(
                            relative_path=rel,
                            size_bytes=stat.st_size,
                            mtime=stat.st_mtime,
                            is_symlink=fp.is_symlink(),
                        )
                    )
                except (OSError, ValueError):
                    continue

        # Always exclude the .git metadata directory from indexed content —
        # the clone is read-only and .git contents leak refs, packs, hooks, etc.
        entries = _apply_cfg_filters(
            raw_entries,
            cfg,
            extra_exclude_globs=[".git/**", ".git"],
            adapter_label=f"git_clone:{self.url}",
        )
        self._listed = entries
        return entries

    def fetch_bytes(self, relative_path: str) -> bytes | Iterator[bytes]:
        """Return file content from the cloned repository."""
        if self._root is None:
            self._clone()
        assert self._root is not None

        target = (self._root / relative_path).resolve()
        try:
            target.relative_to(self._root)
        except ValueError:
            raise ValueError(
                f"Path traversal rejected: {relative_path!r} resolves outside clone root"
            ) from None

        cfg = self._cfg
        streaming_threshold: int = int(getattr(cfg, "streaming_threshold_bytes", 2_000_000))
        max_bytes: int = int(getattr(cfg, "max_bytes_absolute", 500_000_000))

        size = target.stat().st_size
        if size > max_bytes:
            raise ValueError(
                f"File {relative_path!r} is {size} bytes, exceeds max_bytes_absolute={max_bytes}"
            )

        if size > streaming_threshold:
            return LocalPathAdapter._stream_file(target)
        return target.read_bytes()

    def fingerprint(self) -> str:
        """Stable hash seed from URL + cloned HEAD revision.

        Falls back to URL-only if ``git rev-parse`` is unavailable or the
        clone has not yet materialised. Including HEAD lets downstream
        caches invalidate when the remote advances even though the URL is
        unchanged.
        """
        h = hashlib.sha256()
        h.update(f"git_clone:{self.url}".encode())

        rev: str | None = None
        if self._root is not None and self._root.exists():
            try:
                result = subprocess.run(
                    ["git", "-C", str(self._root), "rev-parse", "HEAD"],
                    check=True,
                    timeout=10,
                    capture_output=True,
                )
                rev = result.stdout.decode(errors="replace").strip()
            except (subprocess.SubprocessError, FileNotFoundError, OSError):
                rev = None

        if rev:
            h.update(f"@{rev}".encode())
        return h.hexdigest()

    # ------------------------------------------------------------------
    # Internal clone logic
    # ------------------------------------------------------------------

    def _clone(self) -> None:
        """Perform a shallow clone of *self.url* into a fresh tempdir.

        On ANY failure we remove the partial tempdir so we never leak state.
        """
        tempdir = tempfile.mkdtemp(prefix="vaig-gitclone-")
        self._tempdir = tempdir
        dest = tempdir  # clone into the tempdir itself as the target

        try:
            subprocess.run(
                ["git", "clone", "--depth=1", self.url, dest],
                check=True,
                timeout=60,
                capture_output=True,
            )
        except (KeyboardInterrupt, SystemExit):
            shutil.rmtree(tempdir, ignore_errors=True)
            self._tempdir = None
            raise
        except subprocess.CalledProcessError as exc:
            shutil.rmtree(tempdir, ignore_errors=True)
            self._tempdir = None
            stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
            raise RuntimeError(
                f"git clone failed for {self.url!r}: {stderr.strip()}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            shutil.rmtree(tempdir, ignore_errors=True)
            self._tempdir = None
            raise RuntimeError(
                f"git clone timed out after 60 s for {self.url!r}"
            ) from exc
        except FileNotFoundError as exc:
            shutil.rmtree(tempdir, ignore_errors=True)
            self._tempdir = None
            raise RuntimeError(
                "git is not installed or not on PATH — cannot clone attachment"
            ) from exc
        except Exception as exc:
            shutil.rmtree(tempdir, ignore_errors=True)
            self._tempdir = None
            raise RuntimeError(
                f"Unexpected error during git clone of {self.url!r}: {exc}"
            ) from exc

        self._root = Path(dest).resolve()


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
) -> "LocalPathAdapter | SingleFileAdapter | ArchiveAttachmentAdapter | GitCloneAttachmentAdapter":
    """Select the right attachment adapter from a raw user-supplied string.

    Decision tree (order matters):

    1. Git URL (``git@…``, ``git+…``, ``https://….git``) → :class:`GitCloneAttachmentAdapter`
    2. Archive suffix                                     → :class:`ArchiveAttachmentAdapter`
    3. ``http://`` / ``https://``                         → :exc:`NotImplementedError` (Sprint 3)
    4. Existing file path                                 → :class:`SingleFileAdapter`
    5. Existing dir (non-git)                             → :class:`LocalPathAdapter`
    6. Otherwise                                          → :exc:`ValueError`

    Args:
        raw:  Raw user input (value from ``--attach``).
        name: Optional label (from ``--attach-name``).
        cfg:  :class:`~vaig.core.config.AttachmentsConfig` instance.

    Returns:
        A concrete adapter implementing the :class:`AttachmentAdapter` protocol.
    """
    stripped = raw.strip()

    # 1. Git URL check (before http check to catch https://…git)
    if _is_git_url(stripped):
        spec = AttachmentSpec(
            name=name,
            source=raw,
            kind=AttachmentKind.git_clone,
            resolved_path=None,
        )
        return GitCloneAttachmentAdapter(stripped, spec, cfg)

    # 2. Archive suffix check
    if _is_archive(stripped):
        p = Path(stripped)
        if not p.exists() or not p.is_file():
            raise ValueError(
                f"Archive attachment not found or not a file: {raw!r}"
            )
        spec = AttachmentSpec(
            name=name,
            source=raw,
            kind=AttachmentKind.archive,
            resolved_path=p.resolve(),
        )
        return ArchiveAttachmentAdapter(p, spec, cfg)

    # 3. Plain URL check
    if stripped.startswith("http://") or stripped.startswith("https://"):
        raise NotImplementedError(
            "URL attachments land in Sprint 3 (SPEC-ATT-06)"
        )

    p = Path(stripped)

    # 4. Existing file
    if p.exists() and p.is_file():
        spec = AttachmentSpec(
            name=name,
            source=raw,
            kind=AttachmentKind.single_file,
            resolved_path=p.resolve(),
        )
        return SingleFileAdapter(p, spec)

    # 5. Existing directory — check for .git (local git dir, not a clone)
    if p.exists() and p.is_dir():
        if (p / ".git").exists():
            # Treat as local path adapter (the .git dir is just metadata)
            spec = AttachmentSpec(
                name=name,
                source=raw,
                kind=AttachmentKind.local_path,
                resolved_path=p.resolve(),
            )
            return LocalPathAdapter(p, spec)
        spec = AttachmentSpec(
            name=name,
            source=raw,
            kind=AttachmentKind.local_path,
            resolved_path=p.resolve(),
        )
        return LocalPathAdapter(p, spec)

    # 6. Nothing matched
    raise ValueError(f"Cannot resolve attachment: {raw!r}")

"""Repository cache — store fetched GitHub file contents in `.vaig/repo-cache/`."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path

from vaig.core.project import ensure_project_subdir

logger = logging.getLogger(__name__)

_CACHE_SUBDIR = "repo-cache"


def _cache_path(root: Path, owner: str, repo: str, ref: str, file_path: str) -> Path:
    """Compute a deterministic cache file path for a GitHub file.

    The path is ``{root}/{owner}/{repo}/{ref}/{file_path}``.  ``file_path``
    may contain ``/`` separators which are preserved as nested directories so
    the cache mirrors the repository layout.

    For very long paths the final component is replaced with a SHA-256 digest
    to avoid hitting OS path-length limits.

    Args:
        root: Root cache directory (i.e. ``.vaig/repo-cache``).
        owner: Repository owner.
        repo: Repository name.
        ref: Branch, tag, or commit SHA.
        file_path: File path within the repository.

    Returns:
        Absolute :class:`~pathlib.Path` to the cache file.
    """
    for segment_name, segment in [("owner", owner), ("repo", repo), ("ref", ref), ("file_path", file_path)]:
        if ".." in Path(segment).parts or Path(segment).is_absolute():
            raise ValueError(f"Invalid {segment_name}: must be relative without parent traversal: {segment!r}")
    candidate = root / owner / repo / ref / file_path
    # Guard against excessively long paths (255-byte component limit on most FS)
    if len(str(candidate)) > 4096 or any(len(p) > 255 for p in candidate.parts):
        digest = hashlib.sha256(file_path.encode()).hexdigest()
        candidate = root / owner / repo / ref / digest
    return candidate


class RepoCache:
    """Disk-based cache for GitHub repository file contents.

    Files are stored under ``.vaig/repo-cache/{owner}/{repo}/{ref}/{path}``
    and mirror the repository directory structure.  Content is stored as
    UTF-8 text.

    The cache is append-only — there is no TTL or eviction.  It is intended
    for agentic workflows where the same file may be fetched multiple times
    during a single run and the upstream ref is pinned (e.g. a commit SHA).

    Example::

        cache = RepoCache()
        if not cache.has("acme", "myrepo", "main", "src/main.py"):
            content = fetch_from_github(...)
            cache.put("acme", "myrepo", "main", "src/main.py", content)
        content = cache.get("acme", "myrepo", "main", "src/main.py")
    """

    def __init__(self, root: Path | None = None) -> None:
        """Initialise the cache.

        Args:
            root: Override the cache root directory.  Defaults to the
                ``.vaig/repo-cache`` sub-directory inside the current working
                directory's project dir.
        """
        self._root: Path = root if root is not None else ensure_project_subdir(_CACHE_SUBDIR)

    # ── Public interface ──────────────────────────────────────────────────

    def has(self, owner: str, repo: str, ref: str, file_path: str) -> bool:
        """Return ``True`` if the file is present in the cache.

        Args:
            owner: Repository owner.
            repo: Repository name.
            ref: Branch, tag, or commit SHA.
            file_path: File path within the repository.
        """
        return _cache_path(self._root, owner, repo, ref, file_path).exists()

    def get(self, owner: str, repo: str, ref: str, file_path: str) -> str | None:
        """Return cached file content, or ``None`` if not cached.

        Args:
            owner: Repository owner.
            repo: Repository name.
            ref: Branch, tag, or commit SHA.
            file_path: File path within the repository.
        """
        path = _cache_path(self._root, owner, repo, ref, file_path)
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("RepoCache: failed to read %s: %s", path, exc)
            return None

    def put(self, owner: str, repo: str, ref: str, file_path: str, content: str) -> None:
        """Store file content in the cache.

        Creates parent directories as needed.  Silently logs and swallows
        ``OSError`` so a cache write failure never aborts the calling workflow.

        Args:
            owner: Repository owner.
            repo: Repository name.
            ref: Branch, tag, or commit SHA.
            file_path: File path within the repository.
            content: File content to store (UTF-8 text).
        """
        path = _cache_path(self._root, owner, repo, ref, file_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                    fh.write(content)
                os.replace(tmp_path, path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as exc:
            logger.warning("Failed to write cache file %s: %s", path, exc)

    def invalidate(self, owner: str, repo: str, ref: str, file_path: str) -> bool:
        """Remove a single cached file.

        Args:
            owner: Repository owner.
            repo: Repository name.
            ref: Branch, tag, or commit SHA.
            file_path: File path within the repository.

        Returns:
            ``True`` if the file was found and deleted, ``False`` otherwise.
        """
        path = _cache_path(self._root, owner, repo, ref, file_path)
        if path.exists():
            try:
                path.unlink()
                return True
            except OSError as exc:
                logger.warning("RepoCache: failed to delete %s: %s", path, exc)
        return False

    @property
    def root(self) -> Path:
        """Root directory of the cache."""
        return self._root

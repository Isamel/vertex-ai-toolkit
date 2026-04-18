"""WorkspaceJail — isolated temp-directory context manager for the coding pipeline.

Copies the workspace to a temporary directory before code generation so that
a failed or corrupted pipeline run never modifies the user's repository.

Usage::

    with WorkspaceJail(workspace_root, enabled=True) as jail:
        # jail.effective_path is the temp copy (or original when disabled)
        registry = build_registry(jail.effective_path)
        run_pipeline(registry)
    # On success: changes are synced back to workspace_root.
    # On exception: temp dir is discarded, workspace_root is untouched.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_DEFAULT_IGNORE_PATTERNS: list[str] = [".git", "node_modules", "__pycache__", "*.pyc"]


class WorkspaceJailError(Exception):
    """Raised when a path-traversal or symlink-escape attempt is detected."""


class WorkspaceJail:
    """Context manager that isolates pipeline execution in a temp directory.

    When *enabled* is True the workspace is copied to a fresh temporary
    directory on ``__enter__``.  The pipeline runs inside the copy; the
    original workspace is only updated on clean exit via ``sync_back``.
    If the pipeline raises an exception, the temp directory is discarded
    and the original workspace is left completely untouched.

    When *enabled* is False the context manager is a no-op: no copy is
    made and ``effective_path`` returns the original workspace root.

    Args:
        workspace_root: Path to the workspace directory to isolate.
        enabled: Whether to actually create an isolated copy.
        ignore_patterns: Glob patterns to exclude from the copy (e.g.
            ``.git``, ``node_modules``).  Defaults to
            :data:`_DEFAULT_IGNORE_PATTERNS`.
    """

    def __init__(
        self,
        workspace_root: Path,
        *,
        enabled: bool = True,
        ignore_patterns: list[str] | None = None,
    ) -> None:
        self._workspace_root = workspace_root.resolve()
        self._enabled = enabled
        self._ignore_patterns: list[str] = (
            ignore_patterns if ignore_patterns is not None else list(_DEFAULT_IGNORE_PATTERNS)
        )
        self._tmpdir: Path | None = None

    # ── Context manager ───────────────────────────────────────

    def __enter__(self) -> WorkspaceJail:
        """Copy workspace to temp dir (when enabled) and return self."""
        if not self._enabled:
            logger.debug("WorkspaceJail disabled — running in original workspace %s", self._workspace_root)
            return self

        tmp_parent = Path(tempfile.mkdtemp(prefix="vaig_jail_"))
        dest = tmp_parent / "workspace"
        logger.debug(
            "WorkspaceJail: copying %s → %s (ignore=%s)",
            self._workspace_root,
            dest,
            self._ignore_patterns,
        )
        shutil.copytree(
            self._workspace_root,
            dest,
            ignore=shutil.ignore_patterns(*self._ignore_patterns),
            symlinks=True,
        )
        self._tmpdir = tmp_parent
        logger.info("WorkspaceJail: jail ready at %s", dest)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Sync back on success; always clean up temp dir."""
        try:
            if exc_type is None and self._enabled and self._tmpdir is not None:
                self.sync_back()
        finally:
            self._cleanup()

    # ── Public API ────────────────────────────────────────────

    @property
    def effective_path(self) -> Path:
        """The path the pipeline should use for all file operations.

        Returns the temp copy when the jail is enabled and active,
        or the original workspace root otherwise.
        """
        if self._enabled and self._tmpdir is not None:
            return self._tmpdir / "workspace"
        return self._workspace_root

    def validate_path(self, path: Path) -> Path:
        """Validate that *path* stays within the jail boundary.

        Resolves the path and checks that it is a descendant of
        ``effective_path``.  Raises :class:`WorkspaceJailError` on
        path-traversal (``..``) or symlink-escape attempts.

        Args:
            path: The path to validate (may be relative to effective_path).

        Returns:
            The resolved absolute path.

        Raises:
            WorkspaceJailError: If the resolved path escapes the jail.
        """
        jail_root = self.effective_path.resolve()
        try:
            resolved = (jail_root / path).resolve()
        except (OSError, ValueError) as exc:
            raise WorkspaceJailError(
                f"Failed to resolve path {path!r}: {exc}"
            ) from exc

        try:
            resolved.relative_to(jail_root)
        except ValueError as exc:
            raise WorkspaceJailError(
                f"Path {path!r} escapes jail boundary {jail_root!r}"
            ) from exc

        return resolved

    def sync_back(self) -> None:
        """Copy modified files from the jail back to the original workspace.

        Replaces the original workspace content with the jail's content
        (excluding the original ignore patterns).  Called automatically
        on clean ``__exit__``.

        Raises:
            RuntimeError: If called when the jail is not active.
        """
        if not self._enabled or self._tmpdir is None:
            return

        src = self._tmpdir / "workspace"
        dst = self._workspace_root

        logger.debug("WorkspaceJail.sync_back: %s → %s", src, dst)

        # Remove the original workspace content and replace with the jail copy
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(
            src,
            dst,
            symlinks=True,
        )
        logger.info("WorkspaceJail.sync_back: workspace updated at %s", dst)

    # ── Internal helpers ──────────────────────────────────────

    def _cleanup(self) -> None:
        """Remove the temporary directory unconditionally."""
        if self._tmpdir is not None:
            try:
                shutil.rmtree(self._tmpdir, ignore_errors=True)
                logger.debug("WorkspaceJail: temp dir removed %s", self._tmpdir)
            except Exception:  # noqa: BLE001
                logger.warning("WorkspaceJail: failed to remove temp dir %s", self._tmpdir, exc_info=True)
            finally:
                self._tmpdir = None

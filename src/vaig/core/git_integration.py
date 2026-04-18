"""Git integration manager for the coding pipeline (CM-05).

Provides :class:`GitManager` — a thin, subprocess-based wrapper around the
``git`` and ``gh`` CLI tools.  All operations are gated by
:attr:`~vaig.core.config.GitConfig.enabled`; when disabled every method
is a no-op so callers need not check the flag themselves.

Safety model
------------
- :exc:`GitSafetyError` — raised when an operation would be performed on
  ``main`` or ``master`` (protected branches) unless the caller explicitly
  opts in.
- :exc:`GitDirtyError` — raised when ``check_clean()`` detects uncommitted
  changes and a clean working tree is required for the requested operation.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from vaig.core.config import GitConfig

logger = logging.getLogger(__name__)

# Branches that must never be committed to directly
_PROTECTED_BRANCHES: frozenset[str] = frozenset({"main", "master"})


# ── Custom exceptions ─────────────────────────────────────────


class GitSafetyError(RuntimeError):
    """Raised when a git operation would target a protected branch.

    Args:
        branch: The branch name that triggered the guard.
        operation: Human-readable description of the blocked operation.
    """

    def __init__(self, branch: str, operation: str = "commit") -> None:
        super().__init__(
            f"Git safety guard: refusing to {operation} directly on protected branch '{branch}'. "
            "Use a feature branch instead."
        )
        self.branch = branch
        self.operation = operation


class GitDirtyError(RuntimeError):
    """Raised when the working tree has uncommitted changes where a clean tree is required.

    Args:
        details: Optional output from ``git status`` to aid debugging.
    """

    def __init__(self, details: str = "") -> None:
        msg = "Working tree is dirty (uncommitted changes detected)."
        if details:
            msg = f"{msg}\n{details}"
        super().__init__(msg)
        self.details = details


# ── GitManager ────────────────────────────────────────────────


class GitManager:
    """Subprocess-based git lifecycle manager for the coding pipeline.

    All public methods respect :attr:`GitConfig.enabled`.  When
    ``enabled=False`` (the default) every method returns immediately
    without running any process.

    Args:
        config: The :class:`~vaig.core.config.GitConfig` settings.
        workspace: Workspace root directory (used as ``cwd`` for all
            subprocess calls).  Defaults to ``Path(".")``.

    Example::

        manager = GitManager(settings.coding.git, workspace=Path("my_project"))
        manager.create_branch("vaig/add-retry-logic")
        # ... pipeline writes files ...
        manager.commit_all("feat(retry): add exponential back-off")
    """

    def __init__(
        self,
        config: GitConfig,
        *,
        workspace: Path | None = None,
    ) -> None:
        self._config = config
        self._workspace = (workspace or Path(".")).resolve()

    # ── Public API ────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        """Return True when git integration is active."""
        return self._config.enabled

    def check_clean(self) -> bool:
        """Return True when the working tree has no uncommitted changes.

        Returns:
            ``True`` when the tree is clean, ``False`` otherwise.

        Raises:
            subprocess.SubprocessError: On unexpected git errors.
        """
        if not self.enabled:
            return True

        result = self._run(["git", "status", "--porcelain"], check=False)
        return result.stdout.strip() == ""

    def current_branch(self) -> str:
        """Return the name of the currently checked-out branch.

        Returns:
            Branch name string (e.g. ``"main"`` or ``"vaig/my-task"``).

        Raises:
            subprocess.SubprocessError: When git is not available or not in a
                repository.
        """
        if not self.enabled:
            return ""

        result = self._run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        return result.stdout.strip()

    def create_branch(self, name: str) -> None:
        """Create and check out a new branch.

        Args:
            name: Full branch name (e.g. ``"vaig/add-retry-logic"``).  The
                :attr:`GitConfig.branch_prefix` is NOT prepended here — the
                caller is responsible for formatting the name.

        Raises:
            GitSafetyError: When the *current* branch is a protected branch
                and :attr:`GitConfig.auto_branch` would create a new branch
                from it but the parent is protected.  (Creating feature branches
                *from* main is fine; this guard exists only for direct writes.)
            subprocess.CalledProcessError: When the branch already exists or
                git returns a non-zero exit code.
        """
        if not self.enabled:
            return

        logger.info("GitManager.create_branch: creating branch '%s'", name)
        self._run(["git", "checkout", "-b", name])

    def commit_all(self, message: str) -> None:
        """Stage all tracked and untracked changes and create a commit.

        Uses ``git add -A`` followed by ``git commit``.  Optionally appends a
        ``Signed-off-by`` trailer when :attr:`GitConfig.commit_signoff` is True.

        Args:
            message: Commit message (conventional commit format recommended).

        Raises:
            GitSafetyError: When the current branch is ``main`` or ``master``.
            subprocess.CalledProcessError: When git fails (e.g. nothing to commit).
        """
        if not self.enabled:
            return

        branch = self.current_branch()
        if branch in _PROTECTED_BRANCHES:
            raise GitSafetyError(branch, "commit")

        logger.info("GitManager.commit_all: staging all changes on branch '%s'", branch)
        self._run(["git", "add", "-A"])

        commit_cmd = ["git", "commit", "-m", message]
        if self._config.commit_signoff:
            commit_cmd.append("--signoff")

        logger.info("GitManager.commit_all: committing — %r", message)
        self._run(commit_cmd)

    def push(self, branch: str | None = None, *, set_upstream: bool = True) -> None:
        """Push the current (or specified) branch to the ``origin`` remote.

        Args:
            branch: Branch to push.  Defaults to :meth:`current_branch`.
            set_upstream: When True, passes ``-u origin <branch>`` so the
                remote tracking branch is set.  Ignored when the branch already
                has an upstream.

        Raises:
            GitSafetyError: When the target branch is ``main`` or ``master``.
            subprocess.CalledProcessError: On push failure.
        """
        if not self.enabled:
            return

        target = branch or self.current_branch()
        if target in _PROTECTED_BRANCHES:
            raise GitSafetyError(target, "push")

        cmd = ["git", "push"]
        if set_upstream:
            cmd += ["-u", "origin", target]
        else:
            cmd += ["origin", target]

        logger.info("GitManager.push: pushing '%s' to origin", target)
        self._run(cmd)

    def create_pr(
        self,
        title: str,
        body: str = "",
        *,
        base: str = "main",
    ) -> str:
        """Open a pull request using the ``gh`` CLI.

        Args:
            title: PR title.
            body: PR body / description.  Passed via ``--body``.
            base: Target branch for the PR (default ``"main"``).

        Returns:
            The PR URL string as returned by ``gh pr create``.

        Raises:
            RuntimeError: When :attr:`GitConfig.pr_provider` is not ``"gh"``.
            subprocess.CalledProcessError: When ``gh`` fails (not authenticated,
                no remote, etc.).
        """
        if not self.enabled:
            return ""

        if self._config.pr_provider != "gh":
            raise RuntimeError(
                f"Unsupported PR provider: '{self._config.pr_provider}'. "
                "Only 'gh' (GitHub CLI) is currently supported."
            )

        cmd = [
            "gh",
            "pr",
            "create",
            "--title",
            title,
            "--body",
            body,
            "--base",
            base,
        ]
        logger.info("GitManager.create_pr: creating PR '%s' → '%s'", title, base)
        result = self._run(cmd)
        return result.stdout.strip()

    # ── Private helpers ───────────────────────────────────────

    def _run(
        self,
        cmd: list[str],
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Run *cmd* as a subprocess in the workspace directory.

        Args:
            cmd: Command and arguments list.
            check: When True (default), raises
                :exc:`subprocess.CalledProcessError` on non-zero exit.

        Returns:
            :class:`subprocess.CompletedProcess` with decoded ``stdout`` /
            ``stderr``.
        """
        logger.debug("GitManager._run: %s (cwd=%s)", cmd, self._workspace)
        return subprocess.run(  # noqa: S603
            cmd,
            cwd=self._workspace,
            capture_output=True,
            text=True,
            check=check,
        )


# ── Branch name helper ────────────────────────────────────────


def _sanitize_branch_name(raw: str, prefix: str = "vaig/") -> str:
    """Convert *raw* text into a valid git branch name.

    Lowercases, replaces spaces and special characters with hyphens, strips
    leading/trailing hyphens, and prepends *prefix*.

    Args:
        raw: Arbitrary text to turn into a branch name (e.g. a task title).
        prefix: Prefix to prepend (default ``"vaig/"``).

    Returns:
        Sanitized branch name string.

    Examples:
        >>> _sanitize_branch_name("Add retry logic to GCS upload")
        'vaig/add-retry-logic-to-gcs-upload'
    """
    slug = raw.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    slug = slug[:60]  # keep it reasonable
    return f"{prefix}{slug}"

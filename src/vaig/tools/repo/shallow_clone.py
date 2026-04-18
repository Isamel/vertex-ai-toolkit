"""Shallow-clone context manager for ephemeral repository checkouts.

Provides :func:`shallow_clone` — a ``@contextmanager`` that clones a
repository to a temporary directory, yields the path, and cleans up on
exit regardless of success or failure.

Security considerations
-----------------------
- ``subprocess.run`` is always called with ``shell=False`` to prevent
  shell-injection.
- The URL is validated against ``config.allowed_repos`` (when non-empty)
  before ``git clone`` is invoked.
- All arguments are passed as a list, never as a shell string.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from vaig.core.config import GitHubConfig

logger = logging.getLogger(__name__)

_GIT_CLONE_TIMEOUT: int = 120  # seconds


def _validate_repo_url(config: GitHubConfig, url: str) -> None:
    """Raise ``ValueError`` when *url* is not in the allowed_repos allowlist.

    No-op when ``config.allowed_repos`` is empty (all repos allowed).

    Args:
        config: GitHub configuration with optional allowlist.
        url: Repository HTTPS clone URL (e.g.
            ``https://github.com/owner/repo.git``).

    Raises:
        ValueError: When the allowlist is non-empty and the repo extracted
            from *url* is not in it.
    """
    if not config.allowed_repos:
        return

    # Extract "owner/repo" from HTTPS URL variants:
    #   https://github.com/owner/repo
    #   https://github.com/owner/repo.git
    #   https://x-access-token:TOKEN@github.com/owner/repo.git
    parts = url.rstrip("/").removesuffix(".git").split("/")
    if len(parts) >= 2:
        owner_repo = "/".join(parts[-2:])
    else:
        owner_repo = url

    if owner_repo not in config.allowed_repos:
        raise ValueError(
            f"Repository '{owner_repo}' is not in the allowed_repos list. "
            "Add it to GitHubConfig.allowed_repos to permit cloning."
        )


@contextmanager
def shallow_clone(
    config: GitHubConfig,
    url: str,
    *,
    ref: str = "",
    depth: int = 1,
) -> Iterator[Path]:
    """Clone *url* to a temp directory and yield the checkout path.

    Uses ``--depth`` for a shallow clone to minimise data transfer.
    Cleans up the temp directory on exit via :class:`contextlib.ExitStack`.

    Args:
        config: GitHub configuration (used for allowlist validation).
        url: Repository HTTPS clone URL.
        ref: Branch, tag, or commit SHA.  Defaults to ``config.default_ref``.
        depth: Shallow clone depth (default: 1).

    Yields:
        :class:`~pathlib.Path` pointing to the cloned repository root.

    Raises:
        FileNotFoundError: When ``git`` is not installed.
        RuntimeError: When ``git clone`` fails.
        ValueError: When *url* is not in the allowlist.
    """
    _validate_repo_url(config, url)

    effective_ref = ref or config.default_ref

    if not shutil.which("git"):
        raise FileNotFoundError(
            "git is not installed or not on PATH. "
            "Install git to use shallow_clone."
        )

    with ExitStack() as stack:
        tmpdir = stack.enter_context(
            tempfile.TemporaryDirectory(prefix="vaig_clone_")
        )
        dest = Path(tmpdir) / "repo"

        # Detect whether ref looks like a commit SHA (40 or 7+ hex chars).
        # git clone --branch only accepts branch/tag names, not SHAs.
        _is_sha = bool(effective_ref and len(effective_ref) >= 7 and all(c in "0123456789abcdef" for c in effective_ref.lower()))

        if _is_sha:
            # For SHAs: init + remote add + fetch --depth
            dest.mkdir(parents=True, exist_ok=True)
            cmds: list[list[str]] = [
                ["git", "init", str(dest)],
                ["git", "-C", str(dest), "remote", "add", "origin", url],
                ["git", "-C", str(dest), "fetch", "--depth", str(depth), "origin", effective_ref],
                ["git", "-C", str(dest), "checkout", "FETCH_HEAD"],
            ]
        else:
            cmds = [
                [
                    "git", "clone",
                    "--depth", str(depth),
                    "--branch", effective_ref,
                    "--single-branch",
                    "--", url, str(dest),
                ],
            ]

        logger.debug("shallow_clone: running %d commands for %s@%s", len(cmds), url, effective_ref)

        try:
            for cmd in cmds:
                result = subprocess.run(  # noqa: S603
                    cmd,
                    shell=False,  # noqa: S603
                    capture_output=True,
                    text=True,
                    timeout=_GIT_CLONE_TIMEOUT,
                    check=False,
                )
                if result.returncode != 0:
                    stderr = result.stderr.strip()
                    raise RuntimeError(
                        f"git command failed (exit {result.returncode}): {stderr}"
                    )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"git clone timed out after {_GIT_CLONE_TIMEOUT}s for {url}"
            ) from exc

        logger.info("shallow_clone: cloned %s@%s to %s", url, effective_ref, dest)
        yield dest
        # ExitStack cleanup removes tmpdir automatically

"""Repository adapter pattern for GitHub, GitLab, and local filesystem (SPEC-V2-REPO-09).

Provides a unified ``RepoAdapter`` protocol and three concrete implementations:

- ``GitHubAdapter``          – GitHub REST API (owner/repo or https://github.com/...)
- ``GitLabAdapter``          – GitLab v4 API  (https://gitlab.*/...)
- ``LocalFilesystemAdapter`` – Local git clone; no HTTP, uses ``git`` binary.

Use ``get_adapter(repo_spec)`` to select the right adapter from a URL/path string.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from vaig.core.repo_pipeline import FileMeta

# ── Models ────────────────────────────────────────────────────────────────────


class CommitMeta(BaseModel):
    """Commit metadata."""

    sha: str
    message: str
    author: str
    date: str  # ISO format


# ── Protocol ──────────────────────────────────────────────────────────────────


@runtime_checkable
class RepoAdapter(Protocol):
    """Protocol for repository adapters.

    Implementations must provide uniform access to file trees, file contents,
    web links, and commit history regardless of where the repo is hosted.
    """

    def list_tree(self, ref: str, path: str = "") -> list[FileMeta]:
        """List files in a tree at the given ref and path."""
        ...

    def fetch_file(self, ref: str, path: str) -> str:
        """Fetch file content as a string."""
        ...

    def link_for(
        self,
        ref: str,
        path: str,
        line_range: tuple[int, int] | None = None,
    ) -> str:
        """Generate a web/file link to the given path at the given ref."""
        ...

    def commit_history(
        self, ref: str, path: str, limit: int = 10
    ) -> list[CommitMeta]:
        """Get recent commits affecting a path."""
        ...


# ── GitHub adapter ────────────────────────────────────────────────────────────


class GitHubAdapter:
    """GitHub adapter using the GitHub REST API.

    Auth via ``GITHUB_TOKEN`` env var or explicit *token*.

    Note: method bodies are stubbed — v1 ``gh-*`` tool infrastructure provides
    the concrete GitHub integration.  Raise ``NotImplementedError`` here so the
    caller can fall back to the v1 path.
    """

    def __init__(
        self,
        owner: str,
        repo: str,
        token: str | None = None,
    ) -> None:
        self.owner = owner
        self.repo = repo
        self.token = token or os.environ.get("GITHUB_TOKEN")

    # ------------------------------------------------------------------
    # RepoAdapter interface
    # ------------------------------------------------------------------

    def list_tree(self, ref: str, path: str = "") -> list[FileMeta]:  # noqa: ARG002
        raise NotImplementedError("GitHubAdapter: use v1 gh-* tool infrastructure")

    def fetch_file(self, ref: str, path: str) -> str:  # noqa: ARG002
        raise NotImplementedError("GitHubAdapter: use v1 gh-* tool infrastructure")

    def link_for(
        self,
        ref: str,
        path: str,
        line_range: tuple[int, int] | None = None,
    ) -> str:
        url = f"https://github.com/{self.owner}/{self.repo}/blob/{ref}/{path}"
        if line_range is not None:
            start, end = line_range
            url += f"#L{start}-L{end}"
        return url

    def commit_history(
        self, ref: str, path: str, limit: int = 10  # noqa: ARG002
    ) -> list[CommitMeta]:
        raise NotImplementedError("GitHubAdapter: use v1 gh-* tool infrastructure")


# ── GitLab adapter ────────────────────────────────────────────────────────────


class GitLabAdapter:
    """GitLab adapter using the GitLab v4 REST API.

    Auth via ``GITLAB_TOKEN`` env var or explicit *token*.
    Supports self-hosted instances via *base_url*.
    """

    def __init__(
        self,
        project_path: str,
        base_url: str = "https://gitlab.com",
        token: str | None = None,
    ) -> None:
        self.project_path = project_path  # e.g. "acme/configs"
        self.base_url = base_url.rstrip("/")
        self.token = token or os.environ.get("GITLAB_TOKEN")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        if self.token:
            return {"PRIVATE-TOKEN": self.token}
        return {}

    def _api(self, endpoint: str) -> str:
        """Build an API URL for the given endpoint relative to /api/v4/."""
        encoded = self.project_path.replace("/", "%2F")
        return f"{self.base_url}/api/v4/projects/{encoded}/{endpoint}"

    # ------------------------------------------------------------------
    # RepoAdapter interface
    # ------------------------------------------------------------------

    def list_tree(self, ref: str, path: str = "") -> list[FileMeta]:
        """List files via GitLab repository/tree API."""
        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "httpx is required for GitLabAdapter.list_tree(); install it with: pip install httpx"
            ) from e

        params: dict[str, str | int] = {"ref": ref, "recursive": "true", "per_page": 100}
        if path:
            params["path"] = path

        url = self._api("repository/tree")
        results: list[FileMeta] = []

        while url:
            try:
                resp = httpx.get(url, headers=self._headers(), params=params, timeout=10.0)
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                raise RuntimeError(f"GitLab API request failed: {exc}") from exc
            resp.raise_for_status()
            for item in resp.json():
                if item.get("type") == "blob":
                    results.append(
                        FileMeta(
                            path=item["path"],
                            size=0,  # tree listing doesn't include size
                            sha=item.get("id"),
                        )
                    )
            # GitLab paginates via Link header
            link = resp.headers.get("Link", "")
            next_url = _parse_next_link(link)
            url = next_url  # type: ignore[assignment]
            params = {}  # next URL already contains params

        return results

    def fetch_file(self, ref: str, path: str) -> str:
        """Fetch raw file content via GitLab repository/files API."""
        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "httpx is required for GitLabAdapter.fetch_file(); install it with: pip install httpx"
            ) from e

        encoded_path = path.replace("/", "%2F")
        url = self._api(f"repository/files/{encoded_path}/raw")
        try:
            resp = httpx.get(url, headers=self._headers(), params={"ref": ref}, timeout=10.0)
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            raise RuntimeError(f"GitLab API request failed: {exc}") from exc
        resp.raise_for_status()
        return resp.text

    def link_for(
        self,
        ref: str,
        path: str,
        line_range: tuple[int, int] | None = None,
    ) -> str:
        url = f"{self.base_url}/{self.project_path}/-/blob/{ref}/{path}"
        if line_range is not None:
            start, end = line_range
            url += f"#L{start}-{end}"
        return url

    def commit_history(
        self, ref: str, path: str, limit: int = 10
    ) -> list[CommitMeta]:
        """Fetch commit log via GitLab commits API."""
        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "httpx is required for GitLabAdapter.commit_history(); install it with: pip install httpx"
            ) from e

        url = self._api("repository/commits")
        params: dict[str, str | int] = {"ref_name": ref, "path": path, "per_page": limit}
        try:
            resp = httpx.get(url, headers=self._headers(), params=params, timeout=10.0)
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            raise RuntimeError(f"GitLab API request failed: {exc}") from exc
        resp.raise_for_status()
        commits = []
        for c in resp.json():
            commits.append(
                CommitMeta(
                    sha=c["id"],
                    message=c.get("title", ""),
                    author=c.get("author_name", ""),
                    date=c.get("authored_date", ""),
                )
            )
        return commits


def _parse_next_link(link_header: str) -> str | None:
    """Parse the ``next`` rel from a GitLab Link header."""
    for part in link_header.split(","):
        part = part.strip()
        if 'rel="next"' in part:
            m = re.match(r"<([^>]+)>", part)
            if m:
                return m.group(1)
    return None


# ── LocalFilesystem adapter ───────────────────────────────────────────────────


class LocalFilesystemAdapter:
    """Local filesystem adapter — no HTTP, uses ``git`` CLI.

    For ``--repo /absolute/path/to/clone``.
    ``link_for()`` returns a ``file://`` URL.
    """

    def __init__(self, root_path: str | Path) -> None:
        self.root = Path(root_path).resolve()
        if not self.root.is_dir():
            raise ValueError(f"Not a directory: {self.root}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _git(self, *args: str) -> str:
        """Run a git command inside *self.root* and return stdout."""
        result = subprocess.run(
            ["git", *args],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    # ------------------------------------------------------------------
    # RepoAdapter interface
    # ------------------------------------------------------------------

    def list_tree(self, ref: str, path: str = "") -> list[FileMeta]:
        """List files via ``git ls-tree -r -l`` (includes blob sizes)."""
        cmd = ["ls-tree", "-r", "-l", ref]
        if path:
            cmd.append(path)
        output = self._git(*cmd)
        results = []
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            # Format: <mode> <type> <hash> <size>\t<path>
            # e.g.: 100644 blob abc123 1024\tsrc/foo.py
            try:
                meta_part, file_path = line.split("\t", 1)
                parts = meta_part.split()
                # parts: [mode, type, hash, size]
                size = int(parts[3]) if len(parts) >= 4 and parts[3].isdigit() else 0
                sha = parts[2] if len(parts) >= 3 else None
            except (ValueError, IndexError):
                file_path = line
                size = 0
                sha = None
            results.append(FileMeta(path=file_path, size=size, sha=sha))
        return results

    def fetch_file(self, ref: str, path: str) -> str:
        """Fetch file content via ``git show {ref}:{path}``."""
        return self._git("show", f"{ref}:{path}")

    def link_for(
        self,
        ref: str,  # noqa: ARG002
        path: str,
        line_range: tuple[int, int] | None = None,  # noqa: ARG002
    ) -> str:
        """Return a ``file://`` URL pointing to the file on disk."""
        return f"file://{self.root / path}"

    def commit_history(
        self, ref: str, path: str, limit: int = 10
    ) -> list[CommitMeta]:
        """Fetch commit log via ``git log``."""
        fmt = "%H|%s|%an|%aI"
        output = self._git(
            "log", f"--format={fmt}", f"-n{limit}", ref, "--", path
        )
        commits = []
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 3)
            if len(parts) == 4:  # noqa: PLR2004
                sha, message, author, date = parts
                commits.append(
                    CommitMeta(sha=sha, message=message, author=author, date=date)
                )
        return commits


# ── Factory ───────────────────────────────────────────────────────────────────

_GITHUB_HOST_RE = re.compile(r"^https?://github\.com/(.+?)/(.+?)(?:\.git)?/?$")
_GITLAB_URL_RE = re.compile(r"^https?://(?P<host>[^/]*gitlab[^/]*)/(?P<path>.+?)(?:\.git)?/?$")
_OWNER_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


def get_adapter(repo_spec: str) -> RepoAdapter:
    """Select the right ``RepoAdapter`` based on a URL / path string.

    Supported schemes:

    * ``"owner/repo"``                              → :class:`GitHubAdapter`
    * ``"https://github.com/owner/repo"``           → :class:`GitHubAdapter`
    * ``"https://gitlab.com/owner/repo"``           → :class:`GitLabAdapter`
    * ``"https://gitlab.example.com/owner/repo"``   → :class:`GitLabAdapter`
    * ``"/absolute/path"``                          → :class:`LocalFilesystemAdapter`
    * ``"file:///path"``                            → :class:`LocalFilesystemAdapter`

    Raises:
        ValueError: If the scheme is not recognized.
    """
    spec = repo_spec.strip()

    # ── file:// URL ──────────────────────────────────────────────────────
    if spec.startswith("file://"):
        path = spec[len("file://"):]
        return LocalFilesystemAdapter(path)

    # ── Absolute filesystem path ─────────────────────────────────────────
    if spec.startswith("/"):
        return LocalFilesystemAdapter(spec)

    # ── https://github.com/owner/repo ────────────────────────────────────
    m = _GITHUB_HOST_RE.match(spec)
    if m:
        return GitHubAdapter(owner=m.group(1), repo=m.group(2))

    # ── https://*gitlab*/... ─────────────────────────────────────────────
    m = _GITLAB_URL_RE.match(spec)
    if m:
        host = m.group("host")
        project_path = m.group("path")
        base_url = f"https://{host}"
        return GitLabAdapter(project_path=project_path, base_url=base_url)

    # ── Bare owner/repo shorthand ─────────────────────────────────────────
    if _OWNER_REPO_RE.match(spec):
        owner, repo = spec.split("/", 1)
        return GitHubAdapter(owner=owner, repo=repo)

    # ── Nothing matched ──────────────────────────────────────────────────
    raise ValueError(
        f"Unrecognized repo spec {spec!r}. "
        "Supported schemes: GitHub (owner/repo or https://github.com/...), "
        "GitLab (https://gitlab.*/...), "
        "local filesystem (/absolute/path or file:///path)."
    )

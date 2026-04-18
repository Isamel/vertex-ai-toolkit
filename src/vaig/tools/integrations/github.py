"""GitHub repository tools — list file tree and read file contents via REST API v3."""

from __future__ import annotations

import base64
import logging
from typing import Any
from urllib.parse import quote

import httpx

from vaig.core.config import GitHubConfig
from vaig.core.prompt_defense import wrap_untrusted_content
from vaig.tools.base import ToolResult

logger = logging.getLogger(__name__)

_GITHUB_TIMEOUT: int = 15
_MAX_DIFF_BYTES: int = 500 * 1024  # 500 KB diff cap


def _auth_headers(config: GitHubConfig) -> dict[str, str]:
    """Build authorization headers from config token."""
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = config.token.get_secret_value() if config.token is not None else ""
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def repo_list_tree(
    *,
    config: GitHubConfig,
    owner: str,
    repo: str,
    ref: str = "",
    recursive: bool = True,
    filter_extensions: list[str] | None = None,
) -> ToolResult:
    """List the file tree of a GitHub repository via the Git Trees API.

    Args:
        config: GitHub configuration with optional token and api_base.
        owner: Repository owner (user or organisation).
        repo: Repository name.
        ref: Branch, tag, or commit SHA. Defaults to ``config.default_ref``
            (``"main"``).
        recursive: When ``True`` (default), fetch the full recursive tree.
        filter_extensions: Optional list of file extensions to include
            (e.g. ``[".py", ".ts"]``).  When ``None``, all files are returned.

    Returns:
        ``ToolResult`` with a newline-separated list of file paths, or an
        error message on failure.
    """
    effective_ref = ref or config.default_ref
    base = config.api_base.rstrip("/")
    url = f"{base}/repos/{owner}/{repo}/git/trees/{effective_ref}"
    params: dict[str, Any] = {}
    if recursive:
        params["recursive"] = "1"

    try:
        resp = httpx.get(
            url,
            headers=_auth_headers(config),
            params=params,
            timeout=_GITHUB_TIMEOUT,
            follow_redirects=True,
        )
    except httpx.TimeoutException:
        return ToolResult(
            output=f"GitHub request timed out after {_GITHUB_TIMEOUT}s",
            error=True,
        )
    except httpx.RequestError as exc:
        return ToolResult(output=f"GitHub connection failed: {exc}", error=True)

    if resp.status_code == 401:
        return ToolResult(output="GitHub authentication failed (401). Check token.", error=True)
    if resp.status_code == 403:
        return ToolResult(output="GitHub access denied (403). Check token permissions.", error=True)
    if resp.status_code == 404:
        return ToolResult(
            output=f"GitHub repository or ref not found: {owner}/{repo}@{effective_ref}",
            error=True,
        )
    if resp.status_code == 429:
        return ToolResult(output="GitHub rate limited (429). Try again later.", error=True)
    if resp.status_code >= 400:
        return ToolResult(
            output=f"GitHub API error: {resp.status_code}",
            error=True,
        )

    try:
        data = resp.json()
    except ValueError:
        return ToolResult(output="GitHub returned invalid JSON.", error=True)

    tree: list[dict[str, Any]] = data.get("tree", [])
    # Filter to blobs only (skip subtrees when recursive=True)
    paths = sorted(
        item["path"] for item in tree if item.get("type") == "blob"
    )

    # Apply extension filter when provided
    if filter_extensions:
        exts = {ext if ext.startswith(".") else f".{ext}" for ext in filter_extensions}
        paths = [p for p in paths if any(p.endswith(ext) for ext in exts)]

    if not paths:
        return ToolResult(output=f"No files found in {owner}/{repo}@{effective_ref}")

    return ToolResult(output="\n".join(paths))


def repo_read_file(
    *,
    config: GitHubConfig,
    owner: str,
    repo: str,
    path: str,
    ref: str = "",
) -> ToolResult:
    """Read a single file from a GitHub repository via the Contents API.

    Args:
        config: GitHub configuration with optional token and api_base.
        owner: Repository owner (user or organisation).
        repo: Repository name.
        path: File path within the repository (e.g. ``"src/main.py"``).
        ref: Branch, tag, or commit SHA. Defaults to ``config.default_ref``.

    Returns:
        ``ToolResult`` with the decoded file content as a string, or an
        error message on failure.
    """
    effective_ref = ref or config.default_ref
    base = config.api_base.rstrip("/")
    url = f"{base}/repos/{owner}/{repo}/contents/{quote(path, safe='/')}"

    try:
        resp = httpx.get(
            url,
            headers=_auth_headers(config),
            params={"ref": effective_ref},
            timeout=_GITHUB_TIMEOUT,
            follow_redirects=True,
        )
    except httpx.TimeoutException:
        return ToolResult(
            output=f"GitHub request timed out after {_GITHUB_TIMEOUT}s",
            error=True,
        )
    except httpx.RequestError as exc:
        return ToolResult(output=f"GitHub connection failed: {exc}", error=True)

    if resp.status_code == 401:
        return ToolResult(output="GitHub authentication failed (401). Check token.", error=True)
    if resp.status_code == 403:
        return ToolResult(output="GitHub access denied (403). Check token permissions.", error=True)
    if resp.status_code == 404:
        return ToolResult(
            output=f"File not found: {owner}/{repo}/{path}@{effective_ref}",
            error=True,
        )
    if resp.status_code == 429:
        return ToolResult(output="GitHub rate limited (429). Try again later.", error=True)
    if resp.status_code >= 400:
        return ToolResult(output=f"GitHub API error: {resp.status_code}", error=True)

    try:
        data = resp.json()
    except ValueError:
        return ToolResult(output="GitHub returned invalid JSON.", error=True)

    # The Contents API returns base64-encoded content for blobs
    encoding = data.get("encoding", "")
    raw_content = data.get("content", "")

    if encoding == "base64":
        try:
            content = base64.b64decode(raw_content).decode("utf-8", errors="replace")
        except Exception as exc:
            return ToolResult(
                output=f"Failed to decode file content: {exc}",
                error=True,
            )
    else:
        content = raw_content

    return ToolResult(output=content)


# ── Helpers ───────────────────────────────────────────────────


def _is_binary(content_type: str | None) -> bool:
    """Return True when *content_type* indicates binary data.

    Used to skip decoding of binary blobs returned by the Contents API.
    """
    if not content_type:
        return False
    binary_prefixes = ("image/", "audio/", "video/", "application/octet-stream")
    return any(content_type.startswith(p) for p in binary_prefixes)


def _check_allowed_repos(config: GitHubConfig, owner: str, repo: str) -> ToolResult | None:
    """Return an error ToolResult when *owner/repo* is not in the allowlist.

    Returns ``None`` when the allowlist is empty (no restriction) or the
    repo matches an entry.
    """
    if not config.allowed_repos:
        return None
    full_name = f"{owner}/{repo}"
    if full_name not in config.allowed_repos:
        return ToolResult(
            output=(
                f"Repository '{full_name}' is not in the allowed_repos list. "
                "Add it to GitHubConfig.allowed_repos to permit access."
            ),
            error=True,
        )
    return None


# ── New repo tools ────────────────────────────────────────────


def repo_search_code(
    *,
    config: GitHubConfig,
    owner: str,
    repo: str,
    query: str,
    ref: str = "",
) -> ToolResult:
    """Search code in a GitHub repository via the Search API.

    Args:
        config: GitHub configuration with optional token and api_base.
        owner: Repository owner (user or organisation).
        repo: Repository name.
        query: Search query string (GitHub code-search syntax).
        ref: Branch, tag, or commit SHA.  Defaults to ``config.default_ref``.

    Returns:
        ``ToolResult`` with a list of matching file paths and snippets, or
        an error message on failure.
    """
    if err := _check_allowed_repos(config, owner, repo):
        return err

    base = config.api_base.rstrip("/")
    # GitHub code search: repo:<owner>/<repo> <query>
    # Note: ref parameter is accepted but NOT forwarded — GitHub Code Search
    # API only searches the default branch.
    full_query = f"repo:{owner}/{repo} {query}"
    url = f"{base}/search/code"

    try:
        resp = httpx.get(
            url,
            headers=_auth_headers(config),
            # Note: GitHub Code Search API only searches the default branch.
            # The 'ref' parameter is not supported by this endpoint.
            params={"q": full_query, "per_page": "30"},
            timeout=_GITHUB_TIMEOUT,
            follow_redirects=True,
        )
    except httpx.TimeoutException:
        return ToolResult(
            output=f"GitHub request timed out after {_GITHUB_TIMEOUT}s",
            error=True,
        )
    except httpx.RequestError as exc:
        return ToolResult(output=f"GitHub connection failed: {exc}", error=True)

    if resp.status_code == 401:
        return ToolResult(output="GitHub authentication failed (401). Check token.", error=True)
    if resp.status_code == 403:
        return ToolResult(output="GitHub access denied (403). Check token permissions.", error=True)
    if resp.status_code == 404:
        return ToolResult(
            output=f"GitHub repository not found: {owner}/{repo}",
            error=True,
        )
    if resp.status_code == 429:
        return ToolResult(output="GitHub rate limited (429). Try again later.", error=True)
    if resp.status_code == 422:
        return ToolResult(
            output=f"GitHub search query invalid (422): {query}",
            error=True,
        )
    if resp.status_code >= 400:
        return ToolResult(output=f"GitHub API error: {resp.status_code}", error=True)

    try:
        data = resp.json()
    except ValueError:
        return ToolResult(output="GitHub returned invalid JSON.", error=True)

    items: list[dict[str, Any]] = data.get("items", [])
    if not items:
        return ToolResult(output=f"No code matches found for query: {query}")

    lines: list[str] = []
    for item in items:
        path = item.get("path", "")
        name = item.get("name", "")
        lines.append(f"{path} ({name})")

    raw = "\n".join(lines)
    return ToolResult(output=wrap_untrusted_content(raw))


def repo_get_commits(
    *,
    config: GitHubConfig,
    owner: str,
    repo: str,
    path: str = "",
    ref: str = "",
    limit: int = 20,
) -> ToolResult:
    """List recent commits in a GitHub repository via the Commits API.

    Args:
        config: GitHub configuration with optional token and api_base.
        owner: Repository owner (user or organisation).
        repo: Repository name.
        path: Optional file path to filter commits.
        ref: Branch, tag, or commit SHA.  Defaults to ``config.default_ref``.
        limit: Maximum number of commits to return (default: 20).

    Returns:
        ``ToolResult`` with a list of commits (SHA, author, date, message), or
        an error message on failure.
    """
    if err := _check_allowed_repos(config, owner, repo):
        return err

    effective_ref = ref or config.default_ref
    base = config.api_base.rstrip("/")
    url = f"{base}/repos/{owner}/{repo}/commits"

    # GitHub Commits API returns max 100 per page; cap to avoid ambiguity.
    params: dict[str, Any] = {"sha": effective_ref, "per_page": str(min(limit, 100))}
    if path:
        params["path"] = path

    try:
        resp = httpx.get(
            url,
            headers=_auth_headers(config),
            params=params,
            timeout=_GITHUB_TIMEOUT,
            follow_redirects=True,
        )
    except httpx.TimeoutException:
        return ToolResult(
            output=f"GitHub request timed out after {_GITHUB_TIMEOUT}s",
            error=True,
        )
    except httpx.RequestError as exc:
        return ToolResult(output=f"GitHub connection failed: {exc}", error=True)

    if resp.status_code == 401:
        return ToolResult(output="GitHub authentication failed (401). Check token.", error=True)
    if resp.status_code == 403:
        return ToolResult(output="GitHub access denied (403). Check token permissions.", error=True)
    if resp.status_code == 404:
        return ToolResult(
            output=f"GitHub repository or ref not found: {owner}/{repo}@{effective_ref}",
            error=True,
        )
    if resp.status_code == 429:
        return ToolResult(output="GitHub rate limited (429). Try again later.", error=True)
    if resp.status_code >= 400:
        return ToolResult(output=f"GitHub API error: {resp.status_code}", error=True)

    try:
        commits = resp.json()
    except ValueError:
        return ToolResult(output="GitHub returned invalid JSON.", error=True)

    if not isinstance(commits, list) or not commits:
        return ToolResult(output=f"No commits found in {owner}/{repo}@{effective_ref}")

    lines: list[str] = []
    for commit in commits[:limit]:
        sha = commit.get("sha", "")[:7]
        commit_data = commit.get("commit", {})
        author = commit_data.get("author", {})
        message = commit_data.get("message", "").split("\n")[0]
        date = author.get("date", "")
        name = author.get("name", "")
        lines.append(f"{sha} {date} {name}: {message}")

    raw = "\n".join(lines)
    return ToolResult(output=wrap_untrusted_content(raw))


def repo_diff(
    *,
    config: GitHubConfig,
    owner: str,
    repo: str,
    base: str,
    head: str,
    path: str = "",
) -> ToolResult:
    """Get a unified diff between two refs in a GitHub repository.

    Args:
        config: GitHub configuration with optional token and api_base.
        owner: Repository owner (user or organisation).
        repo: Repository name.
        base: Base ref (branch, tag, or commit SHA).
        head: Head ref (branch, tag, or commit SHA).
        path: Optional file path filter (not directly supported by GitHub compare
              API — returned diff is filtered client-side when provided).

    Returns:
        ``ToolResult`` with the unified diff text (capped at 500 KB), or an
        error message on failure.  Large diffs are truncated and annotated.
    """
    if err := _check_allowed_repos(config, owner, repo):
        return err

    api_base = config.api_base.rstrip("/")
    url = f"{api_base}/repos/{owner}/{repo}/compare/{base}...{head}"

    diff_headers = dict(_auth_headers(config))
    diff_headers["Accept"] = "application/vnd.github.v3.diff"

    try:
        resp = httpx.get(
            url,
            headers=diff_headers,
            timeout=_GITHUB_TIMEOUT,
            follow_redirects=True,
        )
    except httpx.TimeoutException:
        return ToolResult(
            output=f"GitHub request timed out after {_GITHUB_TIMEOUT}s",
            error=True,
        )
    except httpx.RequestError as exc:
        return ToolResult(output=f"GitHub connection failed: {exc}", error=True)

    if resp.status_code == 401:
        return ToolResult(output="GitHub authentication failed (401). Check token.", error=True)
    if resp.status_code == 403:
        return ToolResult(output="GitHub access denied (403). Check token permissions.", error=True)
    if resp.status_code == 404:
        return ToolResult(
            output=f"GitHub repository or refs not found: {owner}/{repo} ({base}...{head})",
            error=True,
        )
    if resp.status_code == 429:
        return ToolResult(output="GitHub rate limited (429). Try again later.", error=True)
    if resp.status_code >= 400:
        return ToolResult(output=f"GitHub API error: {resp.status_code}", error=True)

    diff_text = resp.text

    # Filter by path client-side when requested
    if path:
        sections: list[str] = []
        current: list[str] = []
        in_target = False
        for line in diff_text.splitlines(keepends=True):
            if line.startswith("diff --git"):
                if in_target and current:
                    sections.extend(current)
                current = [line]
                in_target = f" a/{path}" in line or f" b/{path}" in line
            elif in_target:
                current.append(line)
        if in_target and current:
            sections.extend(current)
        diff_text = "".join(sections)

    # 500 KB cap with truncation annotation
    truncated = False
    diff_bytes = diff_text.encode("utf-8")
    if len(diff_bytes) > _MAX_DIFF_BYTES:
        diff_text = diff_bytes[:_MAX_DIFF_BYTES].decode("utf-8", errors="replace")
        truncated = True

    if not diff_text.strip():
        return ToolResult(output=f"No diff found between {base} and {head}")

    if truncated:
        diff_text += f"\n\n[TRUNCATED: diff exceeded {_MAX_DIFF_BYTES // 1024} KB limit]"

    return ToolResult(output=wrap_untrusted_content(diff_text))

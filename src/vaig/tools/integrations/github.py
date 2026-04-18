"""GitHub repository tools — list file tree and read file contents via REST API v3."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from vaig.core.config import GitHubConfig
from vaig.tools.base import ToolResult

logger = logging.getLogger(__name__)

_GITHUB_TIMEOUT: int = 15


def _auth_headers(config: GitHubConfig) -> dict[str, str]:
    """Build authorization headers from config token."""
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if config.token:
        headers["Authorization"] = f"Bearer {config.token.get_secret_value()}"
    return headers


def repo_list_tree(
    *,
    config: GitHubConfig,
    owner: str,
    repo: str,
    ref: str = "",
    recursive: bool = True,
) -> ToolResult:
    """List the file tree of a GitHub repository via the Git Trees API.

    Args:
        config: GitHub configuration with optional token and api_base.
        owner: Repository owner (user or organisation).
        repo: Repository name.
        ref: Branch, tag, or commit SHA. Defaults to ``config.default_ref``
            (``"main"``).
        recursive: When ``True`` (default), fetch the full recursive tree.

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
    url = f"{base}/repos/{owner}/{repo}/contents/{path}"

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
        import base64

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

"""GitHub repo tool registry — factory for all GitHub repository tools."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vaig.tools.base import ToolDef, ToolParam
from vaig.tools.categories import REPO

if TYPE_CHECKING:
    from vaig.core.config import Settings

logger = logging.getLogger(__name__)


def create_github_repo_tools(settings: Settings) -> list[ToolDef]:
    """Create GitHub repository tools gated on ``settings.github.enabled``.

    Wraps all five repo tools (``repo_list_tree``, ``repo_read_file``,
    ``repo_search_code``, ``repo_get_commits``, ``repo_diff``) as
    :class:`~vaig.tools.base.ToolDef` objects.

    Returns an empty list when GitHub is not enabled.
    """
    if not settings.github.enabled:
        return []

    from vaig.tools.integrations.github import (
        repo_diff,
        repo_get_commits,
        repo_list_tree,
        repo_read_file,
        repo_search_code,
    )

    cfg = settings.github
    tools: list[ToolDef] = []

    # ── repo_list_tree ────────────────────────────────────────
    tools.append(
        ToolDef(
            name="repo_list_tree",
            description=(
                "List the file tree of a GitHub repository. "
                "Returns a sorted list of file paths (blobs only). "
                "Use to explore repository structure before reading files."
            ),
            parameters=[
                ToolParam(name="owner", type="string", description="Repository owner (user or org)", required=True),
                ToolParam(name="repo", type="string", description="Repository name", required=True),
                ToolParam(name="ref", type="string", description="Branch, tag, or commit SHA (defaults to main)", required=False),
                ToolParam(name="recursive", type="boolean", description="Fetch full recursive tree (default: true)", required=False),
            ],
            execute=lambda owner, repo, ref="", recursive=True, _cfg=cfg: repo_list_tree(
                config=_cfg, owner=owner, repo=repo, ref=ref, recursive=bool(recursive)
            ),
            categories=frozenset({REPO}),
            cacheable=True,
            cache_ttl_seconds=cfg.cache_ttl_seconds,
        )
    )

    # ── repo_read_file ────────────────────────────────────────
    tools.append(
        ToolDef(
            name="repo_read_file",
            description=(
                "Read a single file from a GitHub repository. "
                "Returns the decoded file content. "
                "Use after repo_list_tree to read specific source files."
            ),
            parameters=[
                ToolParam(name="owner", type="string", description="Repository owner", required=True),
                ToolParam(name="repo", type="string", description="Repository name", required=True),
                ToolParam(name="path", type="string", description="File path within the repository", required=True),
                ToolParam(name="ref", type="string", description="Branch, tag, or commit SHA", required=False),
            ],
            execute=lambda owner, repo, path, ref="", _cfg=cfg: repo_read_file(
                config=_cfg, owner=owner, repo=repo, path=path, ref=ref
            ),
            categories=frozenset({REPO}),
            cacheable=True,
            cache_ttl_seconds=cfg.cache_ttl_seconds,
        )
    )

    # ── repo_search_code ──────────────────────────────────────
    tools.append(
        ToolDef(
            name="repo_search_code",
            description=(
                "Search code in a GitHub repository using GitHub code-search syntax. "
                "Returns matching file paths with names. "
                "Use to locate relevant source files before reading them."
            ),
            parameters=[
                ToolParam(name="owner", type="string", description="Repository owner", required=True),
                ToolParam(name="repo", type="string", description="Repository name", required=True),
                ToolParam(name="query", type="string", description="Code search query (GitHub syntax)", required=True),
                ToolParam(name="ref", type="string", description="Branch, tag, or commit SHA", required=False),
            ],
            execute=lambda owner, repo, query, ref="", _cfg=cfg: repo_search_code(
                config=_cfg, owner=owner, repo=repo, query=query, ref=ref
            ),
            categories=frozenset({REPO}),
            cacheable=False,  # Search results change frequently
        )
    )

    # ── repo_get_commits ──────────────────────────────────────
    tools.append(
        ToolDef(
            name="repo_get_commits",
            description=(
                "List recent commits in a GitHub repository. "
                "Optionally filter by file path. "
                "Returns SHA, author, date, and commit message summary."
            ),
            parameters=[
                ToolParam(name="owner", type="string", description="Repository owner", required=True),
                ToolParam(name="repo", type="string", description="Repository name", required=True),
                ToolParam(name="path", type="string", description="Optional file path to filter commits", required=False),
                ToolParam(name="ref", type="string", description="Branch, tag, or commit SHA", required=False),
                ToolParam(name="limit", type="integer", description="Maximum commits to return (default: 20)", required=False),
            ],
            execute=lambda owner, repo, path="", ref="", limit=20, _cfg=cfg: repo_get_commits(
                config=_cfg, owner=owner, repo=repo, path=path, ref=ref, limit=int(limit)
            ),
            categories=frozenset({REPO}),
            cacheable=True,
            cache_ttl_seconds=cfg.cache_ttl_seconds,
        )
    )

    # ── repo_diff ─────────────────────────────────────────────
    tools.append(
        ToolDef(
            name="repo_diff",
            description=(
                "Get a unified diff between two refs in a GitHub repository. "
                "Diffs are capped at 500 KB and truncated with a notice when exceeded. "
                "Use to review changes between branches, tags, or commits."
            ),
            parameters=[
                ToolParam(name="owner", type="string", description="Repository owner", required=True),
                ToolParam(name="repo", type="string", description="Repository name", required=True),
                ToolParam(name="base", type="string", description="Base ref (branch, tag, or commit SHA)", required=True),
                ToolParam(name="head", type="string", description="Head ref (branch, tag, or commit SHA)", required=True),
                ToolParam(name="path", type="string", description="Optional file path to filter diff", required=False),
            ],
            execute=lambda owner, repo, base, head, path="", _cfg=cfg: repo_diff(
                config=_cfg, owner=owner, repo=repo, base=base, head=head, path=path
            ),
            categories=frozenset({REPO}),
            cacheable=False,
        )
    )

    logger.info(
        "Registered %d GitHub repo tool(s): %s",
        len(tools),
        ", ".join(t.name for t in tools),
    )
    return tools

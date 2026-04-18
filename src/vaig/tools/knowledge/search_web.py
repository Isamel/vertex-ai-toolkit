"""Web search tool using the Tavily API."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from vaig.core.exceptions import ToolExecutionError
from vaig.core.prompt_defense import wrap_untrusted_content
from vaig.tools.base import ToolResult

if TYPE_CHECKING:
    from vaig.core.config import WebSearchConfig


def search_web(
    query: str,
    config: WebSearchConfig,
    *,
    max_results: int | None = None,
) -> ToolResult:
    """Search the web using the Tavily API and return formatted results.

    Args:
        query: The search query string.
        config: Web search configuration including API key and limits.
        max_results: Override for the number of results to return.

    Returns:
        ToolResult with formatted search results wrapped with untrusted content delimiters.

    Raises:
        ToolExecutionError: If the api_key is empty or the HTTP request fails.
    """
    api_key = config.api_key.get_secret_value()
    if not api_key:
        raise ToolExecutionError(
            "search_web: api_key is required but not configured. "
            "Set VAIG_KNOWLEDGE__WEB_SEARCH__API_KEY to enable web search.",
            tool_name="search_web",
        )

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results if max_results is not None else config.max_results,
    }

    response = httpx.post("https://api.tavily.com/search", json=payload)

    if response.status_code >= 300:
        raise ToolExecutionError(
            f"search_web: Tavily API returned HTTP {response.status_code}: {response.text[:200]}",
            tool_name="search_web",
        )

    data = response.json()
    results = data.get("results", [])

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")
        lines.append(f"{i}. [{title}]({url})\n{content}")

    formatted = "\n\n".join(lines) if lines else "No results found."

    return ToolResult(output=wrap_untrusted_content(formatted))

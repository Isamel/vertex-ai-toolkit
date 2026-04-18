"""Knowledge tool factory — wires search_web, fetch_doc, search_rag_knowledge."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vaig.tools.base import ToolDef, ToolParam
from vaig.tools.categories import KNOWLEDGE

if TYPE_CHECKING:
    from vaig.core.config import Settings

logger = logging.getLogger(__name__)

#: CM-10 coding knowledge domains merged into allowed_domains when
#: ``include_coding_domains=True`` is passed to :func:`create_knowledge_tools`.
CM10_CODING_DOMAINS: list[str] = [
    "pypi.org",
    "pkg.go.dev",
    "npmjs.com",
    "docs.python.org",
    "golang.org",
]


def create_knowledge_tools(
    settings: Settings,
    *,
    include_coding_domains: bool = False,
) -> list[ToolDef]:
    """Create external knowledge tools gated by ``settings.knowledge.enabled``.

    Args:
        settings: The application settings instance.
        include_coding_domains: When True, merges CM-10 coding domains into
            the ``fetch_doc`` allowed-domains list.

    Returns:
        A list of :class:`~vaig.tools.base.ToolDef` instances.  Returns an
        empty list when ``settings.knowledge.enabled`` is False.
    """
    if not settings.knowledge.enabled:
        return []

    run_counter: list[int] = [0]
    domains = list(settings.knowledge.web_search.allowed_domains)
    if include_coding_domains:
        domains = list(set(domains) | set(CM10_CODING_DOMAINS))

    tools: list[ToolDef] = []

    # ── fetch_doc — always registered when knowledge.enabled ────────────────
    from vaig.tools.knowledge.fetch_doc import fetch_doc  # noqa: WPS433

    cfg_fetch = settings.knowledge.doc_fetch
    _domains = domains  # stable reference for closure
    tools.append(
        ToolDef(
            name="fetch_doc",
            description=(
                "Fetch and convert an HTML page from an allowed domain to Markdown. "
                "Use this to read official documentation or reference material. "
                "Only allowed domains can be fetched."
            ),
            parameters=[
                ToolParam(
                    name="url",
                    type="string",
                    description="Full URL of the page to fetch (must be on an allowed domain).",
                    required=True,
                ),
            ],
            execute=lambda url, _cfg=cfg_fetch, _d=_domains, _rc=run_counter: fetch_doc(
                url, _cfg, _d, _run_counter=_rc
            ),
            categories=frozenset({KNOWLEDGE}),
            cacheable=False,
        )
    )

    # ── search_web — only when Tavily api_key is set ─────────────────────────
    if settings.knowledge.web_search.api_key.get_secret_value():
        from vaig.tools.knowledge.search_web import search_web  # noqa: WPS433

        cfg_web = settings.knowledge.web_search
        tools.append(
            ToolDef(
                name="search_web",
                description=(
                    "Search the web using Tavily and return ranked results as Markdown. "
                    "Use this to find recent information about Kubernetes, GCP, or "
                    "other infrastructure topics not available in local tools."
                ),
                parameters=[
                    ToolParam(
                        name="query",
                        type="string",
                        description="Search query string.",
                        required=True,
                    ),
                ],
                execute=lambda query, _cfg=cfg_web: search_web(query, _cfg),
                categories=frozenset({KNOWLEDGE}),
                cacheable=False,
            )
        )

    # ── search_rag_knowledge — only when both RAG flags are enabled ──────────
    if settings.export.rag_enabled and settings.knowledge.rag.enabled:
        from vaig.core.rag import RAGKnowledgeBase  # noqa: WPS433
        from vaig.tools.knowledge.search_rag import search_rag_knowledge  # noqa: WPS433

        rag_kb = RAGKnowledgeBase(config=settings.export)
        cfg_rag = settings.knowledge.rag
        tools.append(
            ToolDef(
                name="search_rag_knowledge",
                description=(
                    "Search the internal RAG knowledge corpus for relevant chunks. "
                    "Use this to retrieve company-specific runbooks, post-mortems, "
                    "or architecture documentation."
                ),
                parameters=[
                    ToolParam(
                        name="query",
                        type="string",
                        description="Search query string.",
                        required=True,
                    ),
                ],
                execute=lambda query, _cfg=cfg_rag, _kb=rag_kb: search_rag_knowledge(
                    query, _cfg, _kb
                ),
                categories=frozenset({KNOWLEDGE}),
                cacheable=True,
                cache_ttl_seconds=300,
            )
        )

    if tools:
        logger.info(
            "Registered %d knowledge tool(s): %s",
            len(tools),
            ", ".join(t.name for t in tools),
        )

    return tools

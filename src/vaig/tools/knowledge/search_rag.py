"""RAG knowledge base search tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vaig.core.exceptions import ToolExecutionError
from vaig.core.prompt_defense import wrap_untrusted_content
from vaig.tools.base import ToolResult

if TYPE_CHECKING:
    from vaig.core.config import RagKnowledgeConfig
    from vaig.core.rag import RAGKnowledgeBase


def search_rag_knowledge(
    query: str,
    config: RagKnowledgeConfig,
    rag_kb: RAGKnowledgeBase,
) -> ToolResult:
    """Search the RAG knowledge base for relevant chunks.

    Args:
        query: The search query string.
        config: RAG knowledge configuration (enabled flag, top_k).
        rag_kb: The RAG knowledge base instance to query.

    Returns:
        ToolResult with formatted chunks wrapped with untrusted content delimiters.

    Raises:
        ToolExecutionError: If RAG knowledge retrieval is disabled in config.
    """
    if not config.enabled:
        raise ToolExecutionError(
            "search_rag_knowledge: RAG knowledge disabled",
            tool_name="search_rag_knowledge",
        )

    result = rag_kb.retrieve(query=query, top_k=config.top_k)

    if not result.chunks:
        return ToolResult(output=wrap_untrusted_content("No results found in knowledge corpus."))

    lines: list[str] = []
    for i, chunk in enumerate(result.chunks, 1):
        source_info = f" (source: {chunk.source})" if chunk.source else ""
        lines.append(f"{i}.{source_info} {chunk.text}")

    formatted = "\n\n".join(lines)
    return ToolResult(output=wrap_untrusted_content(formatted))

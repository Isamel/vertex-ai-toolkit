"""Recall similar historical cases from the memory RAG corpus.

Implements MEM-04: wraps :class:`~vaig.core.memory.memory_rag.MemoryRAGIndex`
as a Gemini-callable tool.  Returns semantically similar pattern narratives
from previous diagnostic runs so the agent can contextualise current findings
against known history.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vaig.core.prompt_defense import wrap_untrusted_content
from vaig.tools.base import ToolResult

if TYPE_CHECKING:
    from vaig.core.memory.memory_rag import MemoryRAGIndex


def recall_similar_cases(
    query: str,
    rag_index: MemoryRAGIndex,
    top_k: int = 5,
) -> ToolResult:
    """Recall narratives of historically similar cases from memory RAG.

    Args:
        query: Free-text description of the current finding or situation.
        rag_index: Initialised :class:`~vaig.core.memory.memory_rag.MemoryRAGIndex`.
        top_k: Maximum number of narrative strings to return.

    Returns:
        :class:`~vaig.tools.base.ToolResult` with the matching narratives
        formatted as a numbered list and wrapped as untrusted content, or a
        message stating no similar cases were found.
    """
    narratives = rag_index.recall(query, top_k=top_k)

    if not narratives:
        return ToolResult(output=wrap_untrusted_content("No similar historical cases found."))

    lines: list[str] = [
        f"{i}. {narrative}" for i, narrative in enumerate(narratives, 1)
    ]
    formatted = "\n\n".join(lines)
    return ToolResult(output=wrap_untrusted_content(formatted))

"""Semantic memory RAG index — builds narrative summaries from pattern entries.

Implements MEM-04: wraps :class:`~vaig.core.rag.RAGKnowledgeBase` with a
SEPARATE corpus dedicated to pattern memory narratives.  Provides
:func:`build_narrative` for converting a :class:`PatternEntry` to a
human-readable string, and :class:`MemoryRAGIndex` for ingesting and
recalling those narratives via Vertex AI RAG.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vaig.core.config import MemoryConfig
    from vaig.core.memory.models import PatternEntry
    from vaig.core.rag import RAGKnowledgeBase

logger = logging.getLogger(__name__)

__all__ = ["MemoryRAGIndex", "build_narrative"]


def build_narrative(entry: PatternEntry) -> str:
    """Build a Markdown narrative string from a :class:`PatternEntry`.

    Args:
        entry: The pattern entry to narrate.

    Returns:
        A Markdown string of at least 50 characters describing the
        pattern history.  Empty ``service``, ``title``, or ``category``
        fields are replaced with ``"unknown"``.
    """
    service = entry.service or "unknown"
    title = entry.title or "unknown"
    category = entry.category or "unknown"
    first_seen = entry.first_seen.isoformat()
    last_seen = entry.last_seen.isoformat()
    occ = entry.occurrences
    severity = entry.severity or "unknown"

    return (
        f"Service {service} had {occ} occurrence(s) of '{title}' ({category}). "
        f"First seen {first_seen}. Last seen {last_seen}. Severity: {severity}."
    )


class MemoryRAGIndex:
    """Wraps :class:`~vaig.core.rag.RAGKnowledgeBase` with a memory-specific corpus.

    Ingests pattern narratives built from :class:`PatternEntry` records into
    a Vertex AI RAG corpus whose name is configured separately from the
    knowledge RAG corpus (``config.memory_rag_corpus_name``).

    All methods are error-silent — failures are logged at WARNING level
    and the method returns a safe empty value so the live pipeline is never
    disrupted.

    Args:
        rag_kb: A pre-constructed :class:`~vaig.core.rag.RAGKnowledgeBase`
            instance.  The memory corpus name is injected at call time.
        config: ``MemoryConfig`` instance from application settings.
    """

    def __init__(self, rag_kb: RAGKnowledgeBase, config: MemoryConfig) -> None:
        self._rag_kb = rag_kb
        self._config = config

    def ingest(self, entries: list[PatternEntry]) -> int:
        """Build narratives and upsert them into the memory RAG corpus.

        Respects ``config.memory_rag_max_narratives`` — if ``entries``
        exceeds the limit, only the ``max_narratives`` most-recently-seen
        entries are ingested.

        Args:
            entries: Pattern entries to ingest.

        Returns:
            The number of narratives ingested.  Returns 0 on any error or
            when memory RAG is disabled.
        """
        try:
            if not self._config.memory_rag_enabled:
                return 0

            corpus = self._config.memory_rag_corpus_name
            if not corpus:
                logger.warning("MemoryRAGIndex.ingest: memory_rag_corpus_name is not set")
                return 0

            # Honour max_narratives — keep the most recently seen entries.
            max_n = self._config.memory_rag_max_narratives
            sorted_entries = sorted(entries, key=lambda e: e.last_seen, reverse=True)
            capped = sorted_entries[:max_n]

            narratives = [build_narrative(e) for e in capped]
            if not narratives:
                return 0

            # Delegate to rag_kb to import the narratives into the memory corpus.
            # The rag_kb exposes an internal helper we call here.
            self._rag_kb.ingest_narratives(corpus, narratives)
            return len(narratives)
        except Exception:  # noqa: BLE001
            logger.warning(
                "MemoryRAGIndex.ingest failed — memory narratives not updated",
                exc_info=True,
            )
            return 0

    def recall(self, query: str, top_k: int = 5) -> list[str]:
        """Return up to *top_k* narrative strings semantically similar to *query*.

        Uses the memory-specific RAG corpus.  Returns ``[]`` on any error.

        Args:
            query: Free-text query describing the current situation.
            top_k: Maximum number of narratives to return.

        Returns:
            A list of narrative strings (possibly empty).
        """
        try:
            if not self._config.memory_rag_enabled:
                return []

            corpus = self._config.memory_rag_corpus_name
            if not corpus:
                logger.warning("MemoryRAGIndex.recall: memory_rag_corpus_name is not set")
                return []

            result = self._rag_kb.retrieve_from_corpus(corpus, query, top_k=top_k)
            return [chunk.text for chunk in result if chunk.text]
        except Exception:  # noqa: BLE001
            logger.warning(
                "MemoryRAGIndex.recall failed — returning empty results",
                exc_info=True,
            )
            return []

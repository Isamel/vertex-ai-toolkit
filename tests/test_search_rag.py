"""Unit tests for search_rag_knowledge tool (T-06)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vaig.core.config import RagKnowledgeConfig
from vaig.core.exceptions import ToolExecutionError
from vaig.core.prompt_defense import DELIMITER_DATA_START
from vaig.core.rag import RAGKnowledgeBase, RetrievalResult, RetrievedChunk
from vaig.tools.knowledge.search_rag import search_rag_knowledge


def _make_rag_kb(chunks: list[RetrievedChunk]) -> MagicMock:
    kb = MagicMock(spec=RAGKnowledgeBase)
    kb.retrieve.return_value = RetrievalResult(chunks=chunks, query="test")
    return kb


class TestSearchRagKnowledge:
    def test_disabled_config_raises(self) -> None:
        cfg = RagKnowledgeConfig(enabled=False)
        kb = _make_rag_kb([])
        with pytest.raises(ToolExecutionError, match="RAG knowledge disabled"):
            search_rag_knowledge("query", cfg, kb)

    def test_empty_corpus_returns_no_results_message(self) -> None:
        cfg = RagKnowledgeConfig(enabled=True)
        kb = _make_rag_kb([])
        result = search_rag_knowledge("query", cfg, kb)
        assert "No results found in knowledge corpus." in result.output
        assert DELIMITER_DATA_START in result.output

    def test_non_empty_corpus_returns_chunks(self) -> None:
        cfg = RagKnowledgeConfig(enabled=True, top_k=3)
        chunks = [
            RetrievedChunk(text="chunk one", source="doc1.md"),
            RetrievedChunk(text="chunk two", source="doc2.md"),
            RetrievedChunk(text="chunk three", source=""),
        ]
        kb = _make_rag_kb(chunks)
        result = search_rag_knowledge("query", cfg, kb)
        assert "chunk one" in result.output
        assert "chunk two" in result.output
        assert "chunk three" in result.output
        assert DELIMITER_DATA_START in result.output

    def test_retrieve_called_with_correct_args(self) -> None:
        cfg = RagKnowledgeConfig(enabled=True, top_k=7)
        kb = _make_rag_kb([])
        search_rag_knowledge("my query", cfg, kb)
        kb.retrieve.assert_called_once_with(query="my query", top_k=7)

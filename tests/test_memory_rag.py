"""Tests for build_narrative and MemoryRAGIndex."""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from vaig.core.memory.memory_rag import MemoryRAGIndex, build_narrative
from vaig.core.memory.models import PatternEntry


def _entry(**kwargs) -> PatternEntry:  # type: ignore[no-untyped-def]
    defaults = dict(
        fingerprint="deadbeef01234567",
        first_seen=datetime(2024, 1, 1, tzinfo=UTC),
        last_seen=datetime(2024, 6, 1, tzinfo=UTC),
        occurrences=3,
        severity="high",
        title="CrashLoopBackOff",
        service="payments",
        category="pod-health",
    )
    defaults.update(kwargs)
    return PatternEntry(**defaults)


def _make_config(enabled: bool = True, corpus: str = "mem-corpus") -> MagicMock:
    cfg = MagicMock()
    cfg.memory_rag_enabled = enabled
    cfg.memory_rag_corpus_name = corpus
    cfg.memory_rag_max_narratives = 100
    return cfg


class TestBuildNarrative:
    def test_contains_service_and_title(self) -> None:
        entry = _entry()
        narrative = build_narrative(entry)
        assert "payments" in narrative
        assert "CrashLoopBackOff" in narrative

    def test_contains_occurrences(self) -> None:
        entry = _entry(occurrences=7)
        narrative = build_narrative(entry)
        assert "7" in narrative

    def test_unknown_fallback_for_empty_fields(self) -> None:
        entry = _entry(service="", title="", category="")
        narrative = build_narrative(entry)
        assert "unknown" in narrative

    def test_minimum_length(self) -> None:
        entry = _entry()
        assert len(build_narrative(entry)) >= 50


class TestMemoryRAGIndexIngest:
    def test_ingest_returns_zero_when_disabled(self) -> None:
        cfg = _make_config(enabled=False)
        rag_kb = MagicMock()
        index = MemoryRAGIndex(rag_kb=rag_kb, config=cfg)
        assert index.ingest([_entry()]) == 0
        rag_kb.ingest_narratives.assert_not_called()

    def test_ingest_returns_zero_when_no_corpus(self) -> None:
        cfg = _make_config(corpus="")
        rag_kb = MagicMock()
        index = MemoryRAGIndex(rag_kb=rag_kb, config=cfg)
        assert index.ingest([_entry()]) == 0

    def test_ingest_calls_ingest_narratives(self) -> None:
        cfg = _make_config()
        rag_kb = MagicMock()
        index = MemoryRAGIndex(rag_kb=rag_kb, config=cfg)
        count = index.ingest([_entry(), _entry(fingerprint="abcd1234efef5678")])
        assert count == 2
        rag_kb.ingest_narratives.assert_called_once()

    def test_ingest_caps_at_max_narratives(self) -> None:
        cfg = _make_config()
        cfg.memory_rag_max_narratives = 1
        rag_kb = MagicMock()
        index = MemoryRAGIndex(rag_kb=rag_kb, config=cfg)
        count = index.ingest([_entry(), _entry(fingerprint="aaaa1111bbbb2222")])
        assert count == 1

    def test_ingest_swallows_exceptions(self) -> None:
        cfg = _make_config()
        rag_kb = MagicMock()
        rag_kb.ingest_narratives.side_effect = RuntimeError("boom")
        index = MemoryRAGIndex(rag_kb=rag_kb, config=cfg)
        assert index.ingest([_entry()]) == 0


class TestMemoryRAGIndexRecall:
    def test_recall_returns_empty_when_disabled(self) -> None:
        cfg = _make_config(enabled=False)
        rag_kb = MagicMock()
        index = MemoryRAGIndex(rag_kb=rag_kb, config=cfg)
        assert index.recall("query") == []

    def test_recall_returns_texts(self) -> None:
        cfg = _make_config()
        chunk1 = MagicMock()
        chunk1.text = "Narrative one"
        chunk2 = MagicMock()
        chunk2.text = "Narrative two"
        rag_kb = MagicMock()
        rag_kb.retrieve_from_corpus.return_value = [chunk1, chunk2]
        index = MemoryRAGIndex(rag_kb=rag_kb, config=cfg)
        result = index.recall("OOMKilled", top_k=2)
        assert result == ["Narrative one", "Narrative two"]

    def test_recall_swallows_exceptions(self) -> None:
        cfg = _make_config()
        rag_kb = MagicMock()
        rag_kb.retrieve_from_corpus.side_effect = RuntimeError("oops")
        index = MemoryRAGIndex(rag_kb=rag_kb, config=cfg)
        assert index.recall("query") == []

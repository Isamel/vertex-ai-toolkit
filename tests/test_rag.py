"""Tests for the Vertex AI RAG Engine integration.

Covers RAGKnowledgeBase, RetrievedChunk, RetrievalResult, ExportConfig
RAG fields, and orchestrator RAG context injection.

All vertexai SDK calls are mocked — the SDK is NOT installed in CI.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from vaig.core.config import ExportConfig
from vaig.core.rag import RAGKnowledgeBase, RetrievalResult, RetrievedChunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> ExportConfig:
    """Build an ExportConfig with sensible RAG-enabled defaults."""
    defaults: dict[str, Any] = {
        "gcp_project_id": "my-project",
        "gcs_bucket": "my-bucket",
        "vertex_rag_corpus_id": "projects/123/locations/us-central1/ragCorpora/456",
        "rag_enabled": True,
        "rag_chunk_size": 512,
        "rag_chunk_overlap": 100,
    }
    defaults.update(overrides)
    return ExportConfig(**defaults)


def _make_disabled_config() -> ExportConfig:
    """Build an ExportConfig with RAG disabled."""
    return ExportConfig(
        gcp_project_id="my-project",
        gcs_bucket="my-bucket",
        rag_enabled=False,
    )


# ---------------------------------------------------------------------------
# ExportConfig — RAG fields
# ---------------------------------------------------------------------------


class TestExportConfigRAGFields:
    """Verify the three new RAG fields on ExportConfig."""

    def test_rag_enabled_defaults_false(self) -> None:
        cfg = ExportConfig()
        assert cfg.rag_enabled is False

    def test_rag_chunk_size_default(self) -> None:
        cfg = ExportConfig()
        assert cfg.rag_chunk_size == 1024

    def test_rag_chunk_overlap_default(self) -> None:
        cfg = ExportConfig()
        assert cfg.rag_chunk_overlap == 200

    def test_rag_fields_accept_overrides(self) -> None:
        cfg = ExportConfig(rag_enabled=True, rag_chunk_size=2048, rag_chunk_overlap=300)
        assert cfg.rag_enabled is True
        assert cfg.rag_chunk_size == 2048
        assert cfg.rag_chunk_overlap == 300


# ---------------------------------------------------------------------------
# RetrievedChunk
# ---------------------------------------------------------------------------


class TestRetrievedChunk:
    """Tests for the RetrievedChunk dataclass."""

    def test_minimal_chunk(self) -> None:
        chunk = RetrievedChunk(text="hello")
        assert chunk.text == "hello"
        assert chunk.score == 0.0
        assert chunk.source == ""

    def test_full_chunk(self) -> None:
        chunk = RetrievedChunk(text="data", score=0.95, source="gs://b/file.json")
        assert chunk.text == "data"
        assert chunk.score == 0.95
        assert chunk.source == "gs://b/file.json"


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------


class TestRetrievalResult:
    """Tests for the RetrievalResult dataclass."""

    def test_empty_result_has_no_results(self) -> None:
        r = RetrievalResult(query="test")
        assert r.has_results is False

    def test_non_empty_result_has_results(self) -> None:
        r = RetrievalResult(chunks=[RetrievedChunk(text="x")], query="q")
        assert r.has_results is True

    def test_format_context_empty(self) -> None:
        r = RetrievalResult()
        assert r.format_context() == ""

    def test_format_context_single_chunk(self) -> None:
        r = RetrievalResult(chunks=[RetrievedChunk(text="some text")])
        ctx = r.format_context()
        assert "[1]" in ctx
        assert "some text" in ctx

    def test_format_context_with_source(self) -> None:
        r = RetrievalResult(
            chunks=[RetrievedChunk(text="data", source="gs://bucket/f.json")]
        )
        ctx = r.format_context()
        assert "(source: gs://bucket/f.json)" in ctx

    def test_format_context_multiple_chunks(self) -> None:
        r = RetrievalResult(
            chunks=[
                RetrievedChunk(text="first"),
                RetrievedChunk(text="second"),
                RetrievedChunk(text="third"),
            ]
        )
        ctx = r.format_context()
        assert "[1]" in ctx
        assert "[2]" in ctx
        assert "[3]" in ctx
        assert "first" in ctx
        assert "third" in ctx


# ---------------------------------------------------------------------------
# RAGKnowledgeBase — is_configured / corpus_name
# ---------------------------------------------------------------------------


class TestRAGKnowledgeBaseProperties:
    """Tests for RAGKnowledgeBase properties."""

    def test_is_configured_when_enabled_with_corpus(self) -> None:
        kb = RAGKnowledgeBase(config=_make_config())
        assert kb.is_configured is True

    def test_not_configured_when_disabled(self) -> None:
        kb = RAGKnowledgeBase(config=_make_disabled_config())
        assert kb.is_configured is False

    def test_not_configured_when_no_corpus(self) -> None:
        cfg = _make_config(vertex_rag_corpus_id="", rag_enabled=True)
        kb = RAGKnowledgeBase(config=cfg)
        assert kb.is_configured is False

    def test_corpus_name_returns_config_value(self) -> None:
        cfg = _make_config()
        kb = RAGKnowledgeBase(config=cfg)
        assert kb.corpus_name == cfg.vertex_rag_corpus_id

    def test_project_falls_back_to_config(self) -> None:
        cfg = _make_config(gcp_project_id="proj-from-cfg")
        kb = RAGKnowledgeBase(config=cfg)
        assert kb._project == "proj-from-cfg"

    def test_project_override(self) -> None:
        cfg = _make_config(gcp_project_id="proj-cfg")
        kb = RAGKnowledgeBase(config=cfg, project="explicit-proj")
        assert kb._project == "explicit-proj"


# ---------------------------------------------------------------------------
# RAGKnowledgeBase — retrieve (mocked)
# ---------------------------------------------------------------------------


class TestRAGKnowledgeBaseRetrieve:
    """Tests for RAGKnowledgeBase.retrieve with mocked SDK."""

    def test_retrieve_not_configured_returns_empty(self) -> None:
        kb = RAGKnowledgeBase(config=_make_disabled_config())
        result = kb.retrieve("query")
        assert result.has_results is False
        assert result.query == "query"

    def test_retrieve_empty_query_returns_empty(self) -> None:
        kb = RAGKnowledgeBase(config=_make_config())
        result = kb.retrieve("   ")
        assert result.has_results is False

    @patch("vaig.core.rag._import_vertexai")
    @patch("vaig.core.rag._import_vertexai_rag")
    def test_retrieve_success(self, mock_rag_import: MagicMock, mock_vtx: MagicMock) -> None:
        mock_rag = MagicMock()
        mock_rag_import.return_value = mock_rag

        # Build a fake response
        fake_ctx = SimpleNamespace(text="chunk-text", distance=0.8, source_uri="gs://b/f.json")
        fake_contexts = SimpleNamespace(contexts=[fake_ctx])
        fake_response = SimpleNamespace(contexts=fake_contexts)
        mock_rag.retrieval_query.return_value = fake_response

        kb = RAGKnowledgeBase(config=_make_config())
        kb._initialized = True  # skip init
        result = kb.retrieve("my query", top_k=3)

        assert result.has_results is True
        assert len(result.chunks) == 1
        assert result.chunks[0].text == "chunk-text"
        assert result.chunks[0].score == 0.8
        assert result.chunks[0].source == "gs://b/f.json"

    @patch("vaig.core.rag._import_vertexai")
    @patch("vaig.core.rag._import_vertexai_rag")
    def test_retrieve_exception_returns_empty(
        self, mock_rag_import: MagicMock, mock_vtx: MagicMock
    ) -> None:
        mock_rag = MagicMock()
        mock_rag_import.return_value = mock_rag
        mock_rag.retrieval_query.side_effect = RuntimeError("API error")

        kb = RAGKnowledgeBase(config=_make_config())
        kb._initialized = True
        result = kb.retrieve("fail query")

        assert result.has_results is False
        assert result.query == "fail query"

    @patch("vaig.core.rag._import_vertexai")
    @patch("vaig.core.rag._import_vertexai_rag")
    def test_retrieve_empty_contexts(
        self, mock_rag_import: MagicMock, mock_vtx: MagicMock
    ) -> None:
        mock_rag = MagicMock()
        mock_rag_import.return_value = mock_rag
        mock_rag.retrieval_query.return_value = SimpleNamespace(contexts=None)

        kb = RAGKnowledgeBase(config=_make_config())
        kb._initialized = True
        result = kb.retrieve("query")

        assert result.has_results is False


# ---------------------------------------------------------------------------
# RAGKnowledgeBase — create_corpus (mocked)
# ---------------------------------------------------------------------------


class TestRAGKnowledgeBaseCreateCorpus:
    """Tests for corpus creation."""

    @patch("vaig.core.rag._import_vertexai")
    @patch("vaig.core.rag._import_vertexai_rag")
    def test_create_corpus_returns_name(
        self, mock_rag_import: MagicMock, mock_vtx: MagicMock
    ) -> None:
        mock_rag = MagicMock()
        mock_rag_import.return_value = mock_rag
        mock_rag.create_corpus.return_value = SimpleNamespace(name="corpus/new-123")

        kb = RAGKnowledgeBase(config=_make_config())
        kb._initialized = True
        name = kb.create_corpus("My Corpus", description="test")

        assert name == "corpus/new-123"
        mock_rag.create_corpus.assert_called_once_with(
            display_name="My Corpus", description="test"
        )


# ---------------------------------------------------------------------------
# RAGKnowledgeBase — delete_corpus (mocked)
# ---------------------------------------------------------------------------


class TestRAGKnowledgeBaseDeleteCorpus:
    """Tests for corpus deletion."""

    @patch("vaig.core.rag._import_vertexai")
    @patch("vaig.core.rag._import_vertexai_rag")
    def test_delete_corpus_success(
        self, mock_rag_import: MagicMock, mock_vtx: MagicMock
    ) -> None:
        mock_rag = MagicMock()
        mock_rag_import.return_value = mock_rag

        kb = RAGKnowledgeBase(config=_make_config())
        kb._initialized = True
        assert kb.delete_corpus("corpus/123") is True

    def test_delete_corpus_no_name_returns_false(self) -> None:
        cfg = _make_config(vertex_rag_corpus_id="")
        kb = RAGKnowledgeBase(config=cfg)
        assert kb.delete_corpus() is False

    @patch("vaig.core.rag._import_vertexai")
    @patch("vaig.core.rag._import_vertexai_rag")
    def test_delete_corpus_failure_returns_false(
        self, mock_rag_import: MagicMock, mock_vtx: MagicMock
    ) -> None:
        mock_rag = MagicMock()
        mock_rag_import.return_value = mock_rag
        mock_rag.delete_corpus.side_effect = RuntimeError("gone")

        kb = RAGKnowledgeBase(config=_make_config())
        kb._initialized = True
        assert kb.delete_corpus() is False


# ---------------------------------------------------------------------------
# RAGKnowledgeBase — list_corpora (mocked)
# ---------------------------------------------------------------------------


class TestRAGKnowledgeBaseListCorpora:
    """Tests for listing corpora."""

    @patch("vaig.core.rag._import_vertexai")
    @patch("vaig.core.rag._import_vertexai_rag")
    def test_list_corpora_success(
        self, mock_rag_import: MagicMock, mock_vtx: MagicMock
    ) -> None:
        mock_rag = MagicMock()
        mock_rag_import.return_value = mock_rag
        mock_rag.list_corpora.return_value = [
            SimpleNamespace(name="c/1", display_name="First"),
            SimpleNamespace(name="c/2", display_name="Second"),
        ]

        kb = RAGKnowledgeBase(config=_make_config())
        kb._initialized = True
        result = kb.list_corpora()

        assert len(result) == 2
        assert result[0]["name"] == "c/1"
        assert result[1]["display_name"] == "Second"

    @patch("vaig.core.rag._import_vertexai")
    @patch("vaig.core.rag._import_vertexai_rag")
    def test_list_corpora_failure_returns_empty(
        self, mock_rag_import: MagicMock, mock_vtx: MagicMock
    ) -> None:
        mock_rag = MagicMock()
        mock_rag_import.return_value = mock_rag
        mock_rag.list_corpora.side_effect = RuntimeError("boom")

        kb = RAGKnowledgeBase(config=_make_config())
        kb._initialized = True
        assert kb.list_corpora() == []


# ---------------------------------------------------------------------------
# RAGKnowledgeBase — ingest_reports (mocked)
# ---------------------------------------------------------------------------


class TestRAGKnowledgeBaseIngest:
    """Tests for report ingestion."""

    def test_ingest_not_configured_returns_false(self) -> None:
        kb = RAGKnowledgeBase(config=_make_disabled_config())
        assert kb.ingest_reports(["gs://b/f.json"]) is False

    def test_ingest_empty_paths_returns_false(self) -> None:
        kb = RAGKnowledgeBase(config=_make_config())
        assert kb.ingest_reports([]) is False

    @patch("vaig.core.rag._import_vertexai")
    @patch("vaig.core.rag._import_vertexai_rag")
    def test_ingest_success(self, mock_rag_import: MagicMock, mock_vtx: MagicMock) -> None:
        mock_rag = MagicMock()
        mock_rag_import.return_value = mock_rag

        cfg = _make_config()
        kb = RAGKnowledgeBase(config=cfg)
        kb._initialized = True
        result = kb.ingest_reports(["gs://b/a.json", "gs://b/b.json"])

        assert result is True
        mock_rag.import_files.assert_called_once_with(
            corpus_name=cfg.vertex_rag_corpus_id,
            paths=["gs://b/a.json", "gs://b/b.json"],
            chunk_size=cfg.rag_chunk_size,
            chunk_overlap=cfg.rag_chunk_overlap,
        )

    @patch("vaig.core.rag._import_vertexai")
    @patch("vaig.core.rag._import_vertexai_rag")
    def test_ingest_failure_returns_false(
        self, mock_rag_import: MagicMock, mock_vtx: MagicMock
    ) -> None:
        mock_rag = MagicMock()
        mock_rag_import.return_value = mock_rag
        mock_rag.import_files.side_effect = RuntimeError("fail")

        kb = RAGKnowledgeBase(config=_make_config())
        kb._initialized = True
        assert kb.ingest_reports(["gs://b/x.json"]) is False


# ---------------------------------------------------------------------------
# RAGKnowledgeBase — _ensure_initialized (mocked)
# ---------------------------------------------------------------------------


class TestRAGKnowledgeBaseInit:
    """Tests for SDK initialization."""

    @patch("vaig.core.rag._import_vertexai")
    def test_ensure_initialized_calls_vertexai_init(self, mock_vtx_import: MagicMock) -> None:
        mock_vtx = MagicMock()
        mock_vtx_import.return_value = mock_vtx

        kb = RAGKnowledgeBase(config=_make_config(), project="p1", location="us-east1")
        assert kb._initialized is False
        kb._ensure_initialized()
        assert kb._initialized is True
        mock_vtx.init.assert_called_once_with(project="p1", location="us-east1")

    @patch("vaig.core.rag._import_vertexai")
    def test_ensure_initialized_idempotent(self, mock_vtx_import: MagicMock) -> None:
        mock_vtx = MagicMock()
        mock_vtx_import.return_value = mock_vtx

        kb = RAGKnowledgeBase(config=_make_config())
        kb._ensure_initialized()
        kb._ensure_initialized()  # second call should not re-init
        assert mock_vtx.init.call_count == 1


# ---------------------------------------------------------------------------
# Orchestrator — _retrieve_rag_context
# ---------------------------------------------------------------------------


class TestOrchestratorRAGContext:
    """Tests for the orchestrator's RAG context injection helper."""

    def _make_orchestrator(self, rag_enabled: bool = False, corpus: str = "") -> Any:
        """Build a minimal Orchestrator with mocked dependencies."""
        from vaig.agents.orchestrator import Orchestrator

        mock_client = MagicMock()
        mock_settings = MagicMock()
        mock_settings.export = ExportConfig(
            gcp_project_id="proj",
            vertex_rag_corpus_id=corpus,
            rag_enabled=rag_enabled,
        )
        mock_settings.language = "en"
        return Orchestrator(client=mock_client, settings=mock_settings)

    def test_rag_disabled_returns_empty(self) -> None:
        orch = self._make_orchestrator(rag_enabled=False)
        assert orch._retrieve_rag_context("query") == ""

    def test_rag_no_corpus_returns_empty(self) -> None:
        orch = self._make_orchestrator(rag_enabled=True, corpus="")
        assert orch._retrieve_rag_context("query") == ""

    @patch("vaig.core.rag._import_vertexai")
    @patch("vaig.core.rag._import_vertexai_rag")
    def test_rag_success_injects_context(
        self, mock_rag_import: MagicMock, mock_vtx: MagicMock
    ) -> None:
        mock_rag = MagicMock()
        mock_rag_import.return_value = mock_rag

        fake_ctx = SimpleNamespace(text="historical data", distance=0.9, source_uri="gs://b/f")
        fake_response = SimpleNamespace(
            contexts=SimpleNamespace(contexts=[fake_ctx])
        )
        mock_rag.retrieval_query.return_value = fake_response

        orch = self._make_orchestrator(rag_enabled=True, corpus="corpus/1")
        result = orch._retrieve_rag_context("what happened?")

        assert "Historical Context from Past Reports" in result
        assert "historical data" in result

    def test_rag_import_error_returns_empty(self) -> None:
        orch = self._make_orchestrator(rag_enabled=True, corpus="corpus/1")
        with patch.dict("sys.modules", {"vaig.core.rag": None}):
            result = orch._retrieve_rag_context("query")
        assert result == ""

    @patch("vaig.core.rag._import_vertexai")
    @patch("vaig.core.rag._import_vertexai_rag")
    def test_rag_runtime_error_returns_empty(
        self, mock_rag_import: MagicMock, mock_vtx: MagicMock
    ) -> None:
        mock_rag = MagicMock()
        mock_rag_import.return_value = mock_rag
        mock_rag.retrieval_query.side_effect = RuntimeError("boom")

        orch = self._make_orchestrator(rag_enabled=True, corpus="corpus/1")
        result = orch._retrieve_rag_context("failing query")

        assert result == ""


# ---------------------------------------------------------------------------
# Per-Org Knowledge — resolve_corpus (SPEC-4.2, REQ-ORG-02)
# ---------------------------------------------------------------------------


class TestResolveCorpus:
    """Verify org corpus resolution: lookup, creation, caching, global fallback."""

    @patch("vaig.core.rag._import_vertexai")
    @patch("vaig.core.rag._import_vertexai_rag")
    def test_finds_existing_corpus(
        self, mock_rag_import: MagicMock, mock_vtx: MagicMock
    ) -> None:
        mock_rag = MagicMock()
        mock_rag_import.return_value = mock_rag

        mock_rag.list_corpora.return_value = [
            SimpleNamespace(name="projects/1/ragCorpora/org-1", display_name="vaig-acme"),
        ]

        cfg = _make_config()
        kb = RAGKnowledgeBase(config=cfg)
        result = kb.resolve_corpus("acme")

        assert result == "projects/1/ragCorpora/org-1"
        mock_rag.create_corpus.assert_not_called()

    @patch("vaig.core.rag._import_vertexai")
    @patch("vaig.core.rag._import_vertexai_rag")
    def test_creates_when_missing(
        self, mock_rag_import: MagicMock, mock_vtx: MagicMock
    ) -> None:
        mock_rag = MagicMock()
        mock_rag_import.return_value = mock_rag

        mock_rag.list_corpora.return_value = []
        mock_rag.create_corpus.return_value = SimpleNamespace(
            name="projects/1/ragCorpora/new-org"
        )

        cfg = _make_config()
        kb = RAGKnowledgeBase(config=cfg)
        result = kb.resolve_corpus("acme")

        assert result == "projects/1/ragCorpora/new-org"
        mock_rag.create_corpus.assert_called_once()

    @patch("vaig.core.rag._import_vertexai")
    @patch("vaig.core.rag._import_vertexai_rag")
    def test_caches_result(
        self, mock_rag_import: MagicMock, mock_vtx: MagicMock
    ) -> None:
        mock_rag = MagicMock()
        mock_rag_import.return_value = mock_rag

        mock_rag.list_corpora.return_value = [
            SimpleNamespace(name="projects/1/ragCorpora/org-1", display_name="vaig-acme"),
        ]

        cfg = _make_config()
        kb = RAGKnowledgeBase(config=cfg)
        kb.resolve_corpus("acme")
        kb.resolve_corpus("acme")

        # list_corpora should only be called once — second call hits cache.
        mock_rag.list_corpora.assert_called_once()

    def test_empty_org_id_returns_global_corpus(self) -> None:
        cfg = _make_config()
        kb = RAGKnowledgeBase(config=cfg)
        result = kb.resolve_corpus("")
        assert result == cfg.vertex_rag_corpus_id


# ---------------------------------------------------------------------------
# Per-Org Knowledge — retrieve_with_fallback (SPEC-4.2, REQ-ORG-03)
# ---------------------------------------------------------------------------


class TestRetrieveWithFallback:
    """Verify org-first + global-fallback retrieval with dedup."""

    @patch("vaig.core.rag._import_vertexai")
    @patch("vaig.core.rag._import_vertexai_rag")
    def test_org_only_when_sufficient(
        self, mock_rag_import: MagicMock, mock_vtx: MagicMock
    ) -> None:
        mock_rag = MagicMock()
        mock_rag_import.return_value = mock_rag

        # Org corpus returns 5 chunks (>= min_results=3).
        org_contexts = [
            SimpleNamespace(text=f"org-{i}", distance=0.9, source_uri=f"gs://org/{i}")
            for i in range(5)
        ]
        mock_rag.retrieval_query.return_value = SimpleNamespace(
            contexts=SimpleNamespace(contexts=org_contexts)
        )
        mock_rag.list_corpora.return_value = [
            SimpleNamespace(name="projects/1/ragCorpora/org-1", display_name="vaig-acme"),
        ]

        cfg = _make_config()
        kb = RAGKnowledgeBase(config=cfg)
        result = kb.retrieve_with_fallback("query", org_id="acme")

        assert len(result.chunks) == 5
        # Only one call — no global fallback.
        mock_rag.retrieval_query.assert_called_once()

    @patch("vaig.core.rag._import_vertexai")
    @patch("vaig.core.rag._import_vertexai_rag")
    def test_merges_org_and_global_when_insufficient(
        self, mock_rag_import: MagicMock, mock_vtx: MagicMock
    ) -> None:
        mock_rag = MagicMock()
        mock_rag_import.return_value = mock_rag

        # Org corpus returns 1 chunk (< min_results=3), global returns 4.
        org_ctx = [SimpleNamespace(text="org-0", distance=0.9, source_uri="gs://org/0")]
        global_ctx = [
            SimpleNamespace(text=f"global-{i}", distance=0.8, source_uri=f"gs://global/{i}")
            for i in range(4)
        ]
        mock_rag.retrieval_query.side_effect = [
            SimpleNamespace(contexts=SimpleNamespace(contexts=org_ctx)),
            SimpleNamespace(contexts=SimpleNamespace(contexts=global_ctx)),
        ]
        mock_rag.list_corpora.return_value = [
            SimpleNamespace(name="projects/1/ragCorpora/org-1", display_name="vaig-acme"),
        ]

        cfg = _make_config()
        kb = RAGKnowledgeBase(config=cfg)
        result = kb.retrieve_with_fallback("query", org_id="acme")

        assert len(result.chunks) == 5
        assert result.chunks[0].text == "org-0"  # Org chunks first.
        assert mock_rag.retrieval_query.call_count == 2

    @patch("vaig.core.rag._import_vertexai")
    @patch("vaig.core.rag._import_vertexai_rag")
    def test_dedupes_by_source_uri(
        self, mock_rag_import: MagicMock, mock_vtx: MagicMock
    ) -> None:
        mock_rag = MagicMock()
        mock_rag_import.return_value = mock_rag

        # Overlapping source_uri between org and global.
        org_ctx = [SimpleNamespace(text="org-dup", distance=0.9, source_uri="gs://shared/doc1")]
        global_ctx = [
            SimpleNamespace(text="global-dup", distance=0.8, source_uri="gs://shared/doc1"),
            SimpleNamespace(text="global-unique", distance=0.7, source_uri="gs://global/doc2"),
        ]
        mock_rag.retrieval_query.side_effect = [
            SimpleNamespace(contexts=SimpleNamespace(contexts=org_ctx)),
            SimpleNamespace(contexts=SimpleNamespace(contexts=global_ctx)),
        ]
        mock_rag.list_corpora.return_value = [
            SimpleNamespace(name="projects/1/ragCorpora/org-1", display_name="vaig-acme"),
        ]

        cfg = _make_config()
        kb = RAGKnowledgeBase(config=cfg)
        result = kb.retrieve_with_fallback("query", org_id="acme")

        sources = [c.source for c in result.chunks]
        assert sources == ["gs://shared/doc1", "gs://global/doc2"]
        # Org version kept, global duplicate dropped.
        assert result.chunks[0].text == "org-dup"

    @patch("vaig.core.rag._import_vertexai")
    @patch("vaig.core.rag._import_vertexai_rag")
    def test_global_only_when_no_org_id(
        self, mock_rag_import: MagicMock, mock_vtx: MagicMock
    ) -> None:
        mock_rag = MagicMock()
        mock_rag_import.return_value = mock_rag

        global_ctx = [
            SimpleNamespace(text="global-0", distance=0.8, source_uri="gs://global/0")
        ]
        mock_rag.retrieval_query.return_value = SimpleNamespace(
            contexts=SimpleNamespace(contexts=global_ctx)
        )

        cfg = _make_config()
        kb = RAGKnowledgeBase(config=cfg)
        result = kb.retrieve_with_fallback("query", org_id="")

        assert len(result.chunks) == 1
        assert result.chunks[0].text == "global-0"
        # Single call on global corpus.
        mock_rag.retrieval_query.assert_called_once()

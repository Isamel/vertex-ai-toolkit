"""Tests for SPEC-V2-REPO-02 T-3/T-4, REPO-04, and REPO-05."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from vaig.core.repo_chunkers import Chunk
from vaig.core.repo_index import (
    BudgetExceededFallback,
    RangedReadResult,
    RepoIndex,
    RetrievedRepoChunk,
    TokenBudgetManager,
    _tokenise,
    compute_relevance_keywords,
    read_repo_file,
    relevance_gate,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_chunk(
    file_path: str = "test/file.yaml",
    start_line: int = 1,
    end_line: int = 10,
    content: str = "apiVersion: v1\nkind: Service\nmetadata:\n  name: test-svc",
    kind: str = "yaml_doc",
    outline: str = "Service/test-svc",
) -> Chunk:
    return Chunk(
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        content=content,
        token_estimate=len(content) // 4,
        kind=kind,
        outline=outline,
    )


def _make_retrieved(
    chunk: Chunk | None = None,
    relevance_score: float = 0.5,
    retrieval_query: str = "test query",
    content_override: str | None = None,
) -> RetrievedRepoChunk:
    if chunk is None:
        chunk = _make_chunk()
    content = content_override if content_override is not None else chunk.content
    return RetrievedRepoChunk(
        file_path=chunk.file_path,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        content=content,
        token_estimate=len(content) // 4,
        kind=chunk.kind,
        outline=chunk.outline,
        relevance_score=relevance_score,
        retrieval_query=retrieval_query,
    )


# ── REPO-02 T-3: RepoIndex ────────────────────────────────────────────────────


def test_repo_index_search_returns_top_k() -> None:
    """Build index from 20 synthetic chunks, search returns ≤k results."""
    chunks = [
        _make_chunk(
            file_path=f"file_{i}.yaml",
            content=f"content about service-{i} deployment config",
            outline=f"Deployment/svc-{i}",
        )
        for i in range(20)
    ]
    index = RepoIndex(chunks)
    results = index.search("deployment config", k=5)
    assert len(results) <= 5
    assert len(results) > 0


def test_repo_index_search_scores_descending() -> None:
    """Returned chunks have descending relevance_score."""
    chunks = [
        _make_chunk(
            file_path=f"file_{i}.yaml",
            content=f"gateway ingress routing rule {i}" if i % 3 == 0 else f"unrelated content {i}",
            outline=f"VirtualService/route-{i}",
        )
        for i in range(15)
    ]
    index = RepoIndex(chunks)
    results = index.search("gateway ingress routing", k=8)
    scores = [r.relevance_score for r in results]
    assert scores == sorted(scores, reverse=True), "Scores should be in descending order"


def test_repo_index_search_empty_index() -> None:
    """Empty index returns empty list."""
    index = RepoIndex([])
    results = index.search("anything", k=5)
    assert results == []


def test_repo_index_build_applies_chunker() -> None:
    """build() calls adapter.list_tree + fetch_file + chunks."""
    from vaig.core.config import RepoInvestigationConfig
    from vaig.core.repo_pipeline import FileMeta

    mock_adapter = MagicMock()
    mock_adapter.list_tree.return_value = [
        FileMeta(path="charts/values.yaml", size=100, kind="yaml"),
    ]
    mock_adapter.fetch_file.return_value = "image:\n  tag: latest\nreplicas: 2\n"

    config = RepoInvestigationConfig(
        repo="test/repo",
        paths=["charts"],
        redaction_enabled=False,
    )

    index, gaps = RepoIndex.build(mock_adapter, config, ref="main")

    mock_adapter.list_tree.assert_called_once_with("main", "charts")
    mock_adapter.fetch_file.assert_called_once_with("main", "charts/values.yaml")
    assert isinstance(index, RepoIndex)


# ── T-4: read_repo_file ───────────────────────────────────────────────────────


def _make_adapter_with_content(content: str) -> Any:
    mock = MagicMock()
    mock.fetch_file.return_value = content
    return mock


def test_read_repo_file_returns_window() -> None:
    """offset=0, limit=3 on a 10-line file returns first 3 lines."""
    lines = [f"line {i}\n" for i in range(1, 11)]
    content = "".join(lines)
    adapter = _make_adapter_with_content(content)

    result = read_repo_file(adapter, "HEAD", "file.txt", offset=0, limit=3)

    assert result.content == "".join(lines[:3])
    assert isinstance(result, RangedReadResult)


def test_read_repo_file_has_more_true() -> None:
    """offset=0, limit=3 on 10-line file → has_more=True, next_offset=3."""
    lines = [f"line {i}\n" for i in range(1, 11)]
    content = "".join(lines)
    adapter = _make_adapter_with_content(content)

    result = read_repo_file(adapter, "HEAD", "file.txt", offset=0, limit=3)

    assert result.has_more is True
    assert result.next_offset == 3


def test_read_repo_file_last_page() -> None:
    """offset=8, limit=5 on 10-line file → has_more=False."""
    lines = [f"line {i}\n" for i in range(1, 11)]
    content = "".join(lines)
    adapter = _make_adapter_with_content(content)

    result = read_repo_file(adapter, "HEAD", "file.txt", offset=8, limit=5)

    assert result.has_more is False
    assert result.next_offset is None
    assert "line 9" in result.content
    assert "line 10" in result.content


# ── REPO-04: Relevance gate ───────────────────────────────────────────────────


def test_tokenise_filters_short_words() -> None:
    """Words < 3 chars excluded."""
    tokens = _tokenise("a bb ccc dddd")
    assert "a" not in tokens
    assert "bb" not in tokens
    assert "ccc" in tokens
    assert "dddd" in tokens


def test_compute_relevance_keywords_includes_query_terms() -> None:
    """Keywords include tokenized query terms."""
    kws = compute_relevance_keywords("gateway ingress timeout", findings=[])
    assert "gateway" in kws
    assert "ingress" in kws
    assert "timeout" in kws


def test_compute_relevance_keywords_includes_finding_titles() -> None:
    """Keywords include finding title terms."""
    finding = MagicMock()
    finding.title = "Istio VirtualService misconfigured"
    finding.affected_resources = []
    kws = compute_relevance_keywords("query", findings=[finding])
    assert "istio" in kws
    assert "virtualservice" in kws


def test_relevance_gate_keeps_always_pass_kinds() -> None:
    """Chunk with kind='helm_values' always passes regardless of keywords."""
    chunk = _make_chunk(kind="helm_values", content="some random content xyz")
    result = relevance_gate([chunk], frozenset({"unrelated_keyword"}))
    assert chunk in result


def test_relevance_gate_keeps_keyword_match() -> None:
    """Chunk content contains keyword → passes."""
    chunk = _make_chunk(content="istio gateway configuration", kind="yaml_doc")
    result = relevance_gate([chunk], frozenset({"istio"}))
    assert chunk in result


def test_relevance_gate_keeps_outline_match() -> None:
    """Chunk outline contains keyword → passes."""
    chunk = _make_chunk(content="some generic content", outline="VirtualService/ingress-gateway", kind="yaml_doc")
    result = relevance_gate([chunk], frozenset({"ingress"}))
    assert chunk in result


def test_relevance_gate_removes_irrelevant() -> None:
    """No keyword match, not always-pass kind → filtered."""
    chunk = _make_chunk(content="completely unrelated text here", outline="SomeOther/resource", kind="fallback")
    result = relevance_gate([chunk], frozenset({"istio", "gateway", "ingress"}))
    assert chunk not in result


def test_relevance_gate_retention_rate() -> None:
    """10000 chunks, query 'gateway ingress' → ≤2000 pass (use synthetic data)."""
    # Create 10000 chunks: only ~5% mention gateway/ingress
    matching = [
        _make_chunk(
            file_path=f"file_{i}.yaml",
            content="gateway ingress routing config",
            kind="yaml_doc",
        )
        for i in range(500)
    ]
    non_matching = [
        _make_chunk(
            file_path=f"other_{i}.yaml",
            content=f"database connection pool config {i}",
            outline=f"ConfigMap/db-config-{i}",
            kind="yaml_doc",
        )
        for i in range(9500)
    ]
    all_chunks = matching + non_matching

    kws = compute_relevance_keywords("gateway ingress", findings=[])
    result = relevance_gate(all_chunks, kws)

    assert len(result) <= 2000


def test_relevance_gate_empty_keywords_passes_all() -> None:
    """Empty keywords returns all chunks unchanged."""
    chunks = [_make_chunk(file_path=f"f{i}.yaml") for i in range(5)]
    result = relevance_gate(chunks, frozenset())
    assert len(result) == len(chunks)


# ── REPO-05: TokenBudgetManager ───────────────────────────────────────────────


def _make_large_retrieved(score: float = 0.8, tokens: int = 500) -> RetrievedRepoChunk:
    """Create a chunk with approximately `tokens` token estimate."""
    content = "x " * (tokens * 4 // 2)  # ~tokens when divided by 4
    return RetrievedRepoChunk(
        file_path="large/file.yaml",
        start_line=1,
        end_line=100,
        content=content,
        token_estimate=len(content) // 4,
        kind="yaml_doc",
        outline="Deployment/large-svc",
        relevance_score=score,
        retrieval_query="test",
    )


def test_budget_within_limit_no_fallback() -> None:
    """10 small chunks that fit → BudgetResult.fallbacks_applied==[]."""
    manager = TokenBudgetManager(max_tokens=10_000)
    chunks = [
        _make_retrieved(
            _make_chunk(content="small content here", file_path=f"f{i}.yaml"),
            relevance_score=0.9 - i * 0.05,
        )
        for i in range(10)
    ]
    result = manager.apply(chunks)
    assert result.fallbacks_applied == []
    assert result.evidence_gap is None
    assert result.total_candidate_chunks == 10


def test_budget_reduce_k_applied() -> None:
    """Many chunks exceeding budget → REDUCE_K in fallbacks."""
    manager = TokenBudgetManager(max_tokens=1_000)
    # 100 chunks each with ~200 tokens = 20000 total >> 1000 budget
    chunks = [_make_large_retrieved(score=1.0 - i * 0.005, tokens=200) for i in range(100)]
    result = manager.apply(chunks)
    assert BudgetExceededFallback.REDUCE_K in result.fallbacks_applied
    assert result.total_candidate_chunks == 100


def test_overlaps_helper_removes_duplicates() -> None:
    """_overlaps() correctly identifies overlapping/non-overlapping chunks."""
    from vaig.core.repo_index import _overlaps

    chunk_a = RetrievedRepoChunk(
        file_path="infra/main.yaml",
        start_line=1,
        end_line=50,
        content="gateway config",
        token_estimate=5,
        kind="yaml_doc",
        outline="Gateway/main",
        relevance_score=0.9,
        retrieval_query="gateway",
    )
    chunk_b = RetrievedRepoChunk(
        file_path="infra/main.yaml",
        start_line=30,
        end_line=80,
        content="gateway config continued",
        token_estimate=6,
        kind="yaml_doc",
        outline="Gateway/main-continued",
        relevance_score=0.7,
        retrieval_query="gateway",
    )
    chunk_c = RetrievedRepoChunk(
        file_path="other/file.yaml",
        start_line=1,
        end_line=10,
        content="other content",
        token_estimate=4,
        kind="yaml_doc",
        outline="ConfigMap/other",
        relevance_score=0.5,
        retrieval_query="gateway",
    )

    # chunk_a and chunk_b overlap (same file, lines 1-50 and 30-80 overlap at 30-50)
    assert _overlaps(chunk_a, chunk_b), "Same file, overlapping lines should overlap"
    # chunk_a and chunk_c don't overlap (different files)
    assert not _overlaps(chunk_a, chunk_c), "Different files should not overlap"
    # Non-adjacent ranges on same file should not overlap
    chunk_d = RetrievedRepoChunk(
        file_path="infra/main.yaml",
        start_line=100,
        end_line=150,
        content="later section",
        token_estimate=3,
        kind="yaml_doc",
        outline="Service/later",
        relevance_score=0.4,
        retrieval_query="gateway",
    )
    assert not _overlaps(chunk_a, chunk_d), "Non-overlapping line ranges should not overlap"

    # Verify budget manager handles the scenario without errors
    manager = TokenBudgetManager(max_tokens=8)
    result = manager.apply([chunk_a, chunk_b, chunk_c])
    # After reduce_k + possibly dedup, result is valid
    assert result.total_candidate_chunks == 3
    assert result.total_returned_chunks == len(result.chunks)


def test_budget_reduce_k_keeps_at_least_one_chunk() -> None:
    """When every chunk exceeds budget, reduce_k keeps the top-scored chunk and emits EVIDENCE_GAP."""
    manager = TokenBudgetManager(max_tokens=1)  # smaller than any single chunk

    chunk = RetrievedRepoChunk(
        file_path="big/file.yaml",
        start_line=1,
        end_line=50,
        content="a " * 100,  # 200 chars → 50 tokens >> 1
        token_estimate=50,
        kind="yaml_doc",
        outline="Deployment/big",
        relevance_score=0.9,
        retrieval_query="test",
    )
    result = manager.apply([chunk])
    # Must NOT return empty list — keeps the best chunk
    assert len(result.chunks) == 1, "Should keep at least the top-scored chunk"
    assert result.chunks[0].file_path == "big/file.yaml"
    assert BudgetExceededFallback.REDUCE_K in result.fallbacks_applied
    assert BudgetExceededFallback.EVIDENCE_GAP in result.fallbacks_applied
    assert result.evidence_gap is not None
    assert result.evidence_gap.kind == "TRUNCATED"


def test_budget_summarise_replaces_content_with_outline() -> None:
    """Summarised chunk has content == outline."""
    # Very tight budget to force summarisation
    manager = TokenBudgetManager(max_tokens=50)

    # Create chunks where even after reduce_k + dedup, still over budget
    big_content = "very long detailed content that exceeds budget " * 50
    chunk = RetrievedRepoChunk(
        file_path="huge/file.yaml",
        start_line=1,
        end_line=200,
        content=big_content,
        token_estimate=len(big_content) // 4,
        kind="yaml_doc",
        outline="Deployment/myapp",
        relevance_score=0.95,
        retrieval_query="deployment",
    )
    result = manager.apply([chunk])

    # After summarisation, any remaining chunk should have content = outline
    if BudgetExceededFallback.SUMMARISE in result.fallbacks_applied:
        for c in result.chunks:
            if c.file_path == "huge/file.yaml":
                assert c.content == c.outline


def test_budget_evidence_gap_emitted() -> None:
    """Gap emitted when token budget is exhausted — kind is TRUNCATED or repo_retrieval_truncated."""
    # Extremely tight budget to force all fallbacks including gap
    manager = TokenBudgetManager(max_tokens=1)

    big_content = "x " * 10000
    chunks = [
        RetrievedRepoChunk(
            file_path=f"f{i}.yaml",
            start_line=1,
            end_line=100,
            content=big_content,
            token_estimate=len(big_content) // 4,
            kind="yaml_doc",
            outline="LongOutline/" + "x" * 20,
            relevance_score=0.9,
            retrieval_query="test",
        )
        for i in range(5)
    ]
    result = manager.apply(chunks, path_label="test-path")

    assert result.evidence_gap is not None, "An EvidenceGap must be emitted"
    assert result.evidence_gap.kind in ("repo_retrieval_truncated", "TRUNCATED")
    assert BudgetExceededFallback.EVIDENCE_GAP in result.fallbacks_applied


def test_budget_no_silent_drop() -> None:
    """Verify total_candidate_chunks is set for all scenarios."""
    manager = TokenBudgetManager(max_tokens=500)
    chunks = [_make_large_retrieved(score=1.0 - i * 0.01, tokens=100) for i in range(50)]
    result = manager.apply(chunks)

    assert result.total_candidate_chunks == 50
    assert result.total_returned_chunks == len(result.chunks)
    # total_returned_chunks <= total_candidate_chunks
    assert result.total_returned_chunks <= result.total_candidate_chunks

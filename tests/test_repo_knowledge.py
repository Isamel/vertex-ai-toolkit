"""Tests for vaig.tools.repo.knowledge — RepoIndexManager and search_repo_knowledge."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from vaig.tools.repo.batch import Tier, TreeTriageReport, TriagedEntry
from vaig.tools.repo.knowledge import (
    RepoChunk,
    RepoIndexManager,
    RepoKnowledgeResult,
    search_repo_knowledge,
)

# ── Helpers ───────────────────────────────────────────────────


def _make_rag_mock(
    chunks: list | None = None,
    ingest_ok: bool = True,
) -> MagicMock:
    """Create a MagicMock for RAGKnowledgeBase."""
    rag = MagicMock()
    rag.ingest_narratives.return_value = ingest_ok
    rag.retrieve_from_corpus.return_value = chunks or []
    return rag


def _make_triage(
    *,
    owner: str = "acme",
    repo: str = "myrepo",
    ref: str = "main",
    tier1_paths: list[str] | None = None,
) -> TreeTriageReport:
    entries: list[TriagedEntry] = []
    for path in tier1_paths or []:
        entries.append(TriagedEntry(path=path, tier=Tier.TIER_1))
    return TreeTriageReport(owner=owner, repo=repo, ref=ref, entries=entries)


def _make_chunk(path: str = "src/main.py", text: str = "hello", score: float = 0.9) -> MagicMock:
    chunk = MagicMock()
    chunk.source = path
    chunk.text = text
    chunk.score = score
    return chunk


def _make_settings() -> MagicMock:
    settings = MagicMock()
    settings.export = MagicMock()
    return settings


# ── RepoChunk model ───────────────────────────────────────────


class TestRepoChunkModel:
    def test_valid_creation(self) -> None:
        chunk = RepoChunk(path="src/foo.py", content="import os", score=0.85)
        assert chunk.path == "src/foo.py"
        assert chunk.content == "import os"
        assert chunk.score == 0.85

    def test_zero_score(self) -> None:
        chunk = RepoChunk(path="README.md", content="# Readme", score=0.0)
        assert chunk.score == 0.0


# ── RepoKnowledgeResult model ─────────────────────────────────


class TestRepoKnowledgeResultModel:
    __test__ = True  # explicit — it IS a test class

    def test_valid_creation(self) -> None:
        result = RepoKnowledgeResult(
            repo="acme/myrepo",
            ref="main",
            chunks=[RepoChunk(path="a.py", content="x", score=0.5)],
            index_built_at=datetime(2026, 1, 1),
            from_cache=False,
        )
        assert result.repo == "acme/myrepo"
        assert len(result.chunks) == 1
        assert result.from_cache is False

    def test_from_cache_true(self) -> None:
        result = RepoKnowledgeResult(
            repo="acme/myrepo",
            ref="main",
            chunks=[],
            index_built_at=datetime(2026, 1, 1),
            from_cache=True,
        )
        assert result.from_cache is True

    def test_has_test_false_attribute(self) -> None:
        """RepoKnowledgeResult must have __test__ = False to avoid pytest collection."""
        assert RepoKnowledgeResult.__test__ is False  # type: ignore[attr-defined]

    def test_index_manager_has_test_false(self) -> None:
        assert RepoIndexManager.__test__ is False  # type: ignore[attr-defined]


# ── RepoIndexManager — build_index ────────────────────────────


class TestRepoIndexManagerBuildIndex:
    def test_first_call_builds_index(self, tmp_path: Path) -> None:
        rag = _make_rag_mock()
        mgr = RepoIndexManager(rag=rag, index_root=tmp_path)
        triage = _make_triage(tier1_paths=["src/main.py", "src/utils.py"])
        content = {"src/main.py": "def main(): pass", "src/utils.py": "def helper(): pass"}

        corpus_name, from_cache = mgr.build_index(
            owner="acme", repo="myrepo", ref="main",
            triage_report=triage, content_fetcher=content,
        )

        assert from_cache is False
        rag.ingest_narratives.assert_called_once()
        # corpus_name contains owner_repo and ref
        assert "acme" in corpus_name
        assert "myrepo" in corpus_name

    def test_second_call_returns_from_cache(self, tmp_path: Path) -> None:
        rag = _make_rag_mock()
        mgr = RepoIndexManager(rag=rag, index_root=tmp_path)
        triage = _make_triage(tier1_paths=["src/main.py"])
        content = {"src/main.py": "x"}

        _, first_cache = mgr.build_index(
            owner="acme", repo="myrepo", ref="main",
            triage_report=triage, content_fetcher=content,
        )
        _, second_cache = mgr.build_index(
            owner="acme", repo="myrepo", ref="main",
            triage_report=triage, content_fetcher=content,
        )

        assert first_cache is False
        assert second_cache is True
        # ingest_narratives called only once (not on cache hit)
        assert rag.ingest_narratives.call_count == 1

    def test_cache_invalidated_on_ref_change(self, tmp_path: Path) -> None:
        rag = _make_rag_mock()
        mgr = RepoIndexManager(rag=rag, index_root=tmp_path)
        triage = _make_triage(tier1_paths=["src/main.py"])
        content = {"src/main.py": "x"}

        _, first = mgr.build_index(
            owner="acme", repo="myrepo", ref="main",
            triage_report=triage, content_fetcher=content,
        )
        # Different ref — cache must be rebuilt
        triage2 = _make_triage(tier1_paths=["src/main.py"], ref="develop")
        _, second = mgr.build_index(
            owner="acme", repo="myrepo", ref="develop",
            triage_report=triage2, content_fetcher=content,
        )

        assert first is False
        assert second is False
        assert rag.ingest_narratives.call_count == 2

    def test_cache_invalidated_on_new_commits(self, tmp_path: Path) -> None:
        rag = _make_rag_mock()
        mgr = RepoIndexManager(rag=rag, index_root=tmp_path)
        triage = _make_triage(tier1_paths=["src/main.py"])
        content = {"src/main.py": "x"}

        mgr.build_index(
            owner="acme", repo="myrepo", ref="main",
            triage_report=triage, content_fetcher=content,
            latest_sha="abc123",
        )
        # New SHA detected
        _, from_cache = mgr.build_index(
            owner="acme", repo="myrepo", ref="main",
            triage_report=triage, content_fetcher=content,
            latest_sha="def456",
        )

        assert from_cache is False
        assert rag.ingest_narratives.call_count == 2

    def test_only_tier1_files_indexed(self, tmp_path: Path) -> None:
        rag = _make_rag_mock()
        mgr = RepoIndexManager(rag=rag, index_root=tmp_path)

        triage = TreeTriageReport(
            owner="acme",
            repo="myrepo",
            ref="main",
            entries=[
                TriagedEntry(path="src/main.py", tier=Tier.TIER_1),
                TriagedEntry(path="tests/test_main.py", tier=Tier.TIER_2),
                TriagedEntry(path="data/seed.json", tier=Tier.TIER_3),
            ],
        )
        content = {
            "src/main.py": "tier1 content",
            "tests/test_main.py": "tier2 content",
            "data/seed.json": "tier3 content",
        }

        mgr.build_index(
            owner="acme", repo="myrepo", ref="main",
            triage_report=triage, content_fetcher=content,
        )

        call_args = rag.ingest_narratives.call_args
        narratives: list[str] = call_args[0][1]  # second positional arg
        # Only tier1 content should appear
        all_narratives = "\n".join(narratives)
        assert "tier1 content" in all_narratives
        assert "tier2 content" not in all_narratives
        assert "tier3 content" not in all_narratives

    def test_wrap_untrusted_content_is_called(self, tmp_path: Path) -> None:
        """All indexed content MUST pass through wrap_untrusted_content."""
        rag = _make_rag_mock()
        mgr = RepoIndexManager(rag=rag, index_root=tmp_path)
        triage = _make_triage(tier1_paths=["src/main.py"])
        raw_content = "raw untrusted content here"
        content = {"src/main.py": raw_content}

        with patch("vaig.tools.repo.knowledge.wrap_untrusted_content") as mock_wrap:
            mock_wrap.return_value = "WRAPPED: " + raw_content
            mgr.build_index(
                owner="acme", repo="myrepo", ref="main",
                triage_report=triage, content_fetcher=content,
            )

        mock_wrap.assert_called_once_with(raw_content)
        # Wrapped content should be in the ingested narratives
        call_args = rag.ingest_narratives.call_args
        narratives: list[str] = call_args[0][1]
        assert any("WRAPPED:" in n for n in narratives)

    def test_built_at_recorded(self, tmp_path: Path) -> None:
        rag = _make_rag_mock()
        mgr = RepoIndexManager(rag=rag, index_root=tmp_path)
        triage = _make_triage(tier1_paths=["src/main.py"])
        content = {"src/main.py": "x"}

        before = datetime.utcnow()
        mgr.build_index(
            owner="acme", repo="myrepo", ref="main",
            triage_report=triage, content_fetcher=content,
        )
        after = datetime.utcnow()

        built = mgr.built_at("acme", "myrepo", "main")
        assert built is not None
        assert before <= built <= after

    def test_empty_triage_report(self, tmp_path: Path) -> None:
        rag = _make_rag_mock()
        mgr = RepoIndexManager(rag=rag, index_root=tmp_path)
        triage = _make_triage(tier1_paths=[])

        corpus_name, from_cache = mgr.build_index(
            owner="acme", repo="myrepo", ref="main",
            triage_report=triage, content_fetcher={},
        )

        # Should still succeed with empty narratives
        assert from_cache is False
        rag.ingest_narratives.assert_called_once()


# ── RepoIndexManager — invalidate / search ────────────────────


class TestRepoIndexManagerMisc:
    def test_invalidate_clears_cache(self, tmp_path: Path) -> None:
        rag = _make_rag_mock()
        mgr = RepoIndexManager(rag=rag, index_root=tmp_path)
        triage = _make_triage(tier1_paths=["a.py"])
        content = {"a.py": "x"}

        mgr.build_index(owner="acme", repo="r", ref="main", triage_report=triage, content_fetcher=content)
        mgr.invalidate("acme", "r", "main")

        # After invalidation, next build should rebuild
        _, from_cache = mgr.build_index(
            owner="acme", repo="r", ref="main",
            triage_report=triage, content_fetcher=content,
        )
        assert from_cache is False

    def test_search_delegates_to_rag(self, tmp_path: Path) -> None:
        expected = [_make_chunk()]
        rag = _make_rag_mock(chunks=expected)
        mgr = RepoIndexManager(rag=rag, index_root=tmp_path)

        results = mgr.search(owner="acme", repo="r", ref="main", query="hello")

        rag.retrieve_from_corpus.assert_called_once()
        assert results == expected

    def test_built_at_returns_none_before_build(self, tmp_path: Path) -> None:
        rag = _make_rag_mock()
        mgr = RepoIndexManager(rag=rag, index_root=tmp_path)
        assert mgr.built_at("acme", "r", "main") is None


# ── search_repo_knowledge tool ────────────────────────────────


class TestSearchRepoKnowledgeTool:
    def _run(self, coro):  # type: ignore[no-untyped-def]
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_returns_tool_result_with_chunks(self, tmp_path: Path) -> None:
        rag = _make_rag_mock(chunks=[_make_chunk("src/main.py", "def main(): pass", 0.9)])
        mgr = RepoIndexManager(rag=rag, index_root=tmp_path)
        triage = _make_triage(tier1_paths=["src/main.py"])
        content = {"src/main.py": "def main(): pass"}
        settings = _make_settings()

        result = self._run(
            search_repo_knowledge(
                settings=settings,
                owner="acme",
                repo="myrepo",
                query="main function",
                ref="main",
                triage_report=triage,
                content_fetcher=content,
                index_manager=mgr,
            )
        )

        assert result.error is False
        assert "main function" in result.output or "src/main.py" in result.output

    def test_empty_query_returns_error(self, tmp_path: Path) -> None:
        settings = _make_settings()
        result = self._run(
            search_repo_knowledge(
                settings=settings,
                owner="acme",
                repo="myrepo",
                query="   ",
            )
        )
        assert result.error is True

    def test_no_results_returns_non_error_message(self, tmp_path: Path) -> None:
        rag = _make_rag_mock(chunks=[])
        mgr = RepoIndexManager(rag=rag, index_root=tmp_path)
        settings = _make_settings()
        triage = _make_triage(tier1_paths=[])

        result = self._run(
            search_repo_knowledge(
                settings=settings,
                owner="acme",
                repo="myrepo",
                query="something",
                triage_report=triage,
                content_fetcher={},
                index_manager=mgr,
            )
        )

        assert result.error is False
        assert "No results" in result.output

    def test_from_cache_false_on_first_call(self, tmp_path: Path) -> None:
        chunk = _make_chunk()
        rag = _make_rag_mock(chunks=[chunk])
        mgr = RepoIndexManager(rag=rag, index_root=tmp_path)
        triage = _make_triage(tier1_paths=["src/main.py"])
        content = {"src/main.py": "hello"}
        settings = _make_settings()

        result = self._run(
            search_repo_knowledge(
                settings=settings,
                owner="acme",
                repo="myrepo",
                query="hello",
                ref="main",
                triage_report=triage,
                content_fetcher=content,
                index_manager=mgr,
            )
        )

        assert result.error is False
        assert "from_cache=False" in result.output

    def test_from_cache_true_on_second_call(self, tmp_path: Path) -> None:
        chunk = _make_chunk()
        rag = _make_rag_mock(chunks=[chunk])
        mgr = RepoIndexManager(rag=rag, index_root=tmp_path)
        triage = _make_triage(tier1_paths=["src/main.py"])
        content = {"src/main.py": "hello"}
        settings = _make_settings()

        # First call
        self._run(
            search_repo_knowledge(
                settings=settings,
                owner="acme",
                repo="myrepo",
                query="hello",
                ref="main",
                triage_report=triage,
                content_fetcher=content,
                index_manager=mgr,
            )
        )

        # Second call — should be from cache
        result = self._run(
            search_repo_knowledge(
                settings=settings,
                owner="acme",
                repo="myrepo",
                query="hello",
                ref="main",
                triage_report=triage,
                content_fetcher=content,
                index_manager=mgr,
            )
        )

        assert result.error is False
        assert "from_cache=True" in result.output

    def test_tier1_only_filtering_via_tool(self, tmp_path: Path) -> None:
        rag = _make_rag_mock(chunks=[_make_chunk()])
        mgr = RepoIndexManager(rag=rag, index_root=tmp_path)
        triage = TreeTriageReport(
            owner="acme",
            repo="myrepo",
            ref="main",
            entries=[
                TriagedEntry(path="src/main.py", tier=Tier.TIER_1),
                TriagedEntry(path="tests/test.py", tier=Tier.TIER_2),
            ],
        )
        content = {"src/main.py": "tier1", "tests/test.py": "tier2"}
        settings = _make_settings()

        with patch("vaig.tools.repo.knowledge.wrap_untrusted_content", side_effect=lambda x: x):
            self._run(
                search_repo_knowledge(
                    settings=settings,
                    owner="acme",
                    repo="myrepo",
                    query="search",
                    ref="main",
                    triage_report=triage,
                    content_fetcher=content,
                    index_manager=mgr,
                )
            )

        call_args = rag.ingest_narratives.call_args
        narratives: list[str] = call_args[0][1]
        all_text = "\n".join(narratives)
        assert "tier1" in all_text
        assert "tier2" not in all_text

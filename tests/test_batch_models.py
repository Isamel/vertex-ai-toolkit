"""Tests for batch models and chunk_file() in tools/repo/batch.py."""

from __future__ import annotations

import pytest

from vaig.tools.repo.batch import (
    BatchPlan,
    FileChunk,
    Tier,
    TreeTriageReport,
    TriagedEntry,
    chunk_file,
)

# ── Tier enum ─────────────────────────────────────────────────


class TestTier:
    def test_values(self) -> None:
        assert Tier.TIER_1 == "tier_1"
        assert Tier.TIER_2 == "tier_2"
        assert Tier.TIER_3 == "tier_3"


# ── TriagedEntry ──────────────────────────────────────────────


class TestTriagedEntry:
    def test_required_fields(self) -> None:
        entry = TriagedEntry(path="src/main.py", tier=Tier.TIER_1)
        assert entry.path == "src/main.py"
        assert entry.tier == Tier.TIER_1
        assert entry.reason == ""
        assert entry.size_bytes == 0
        assert entry.sha == ""

    def test_full_fields(self) -> None:
        entry = TriagedEntry(
            path="src/app.py",
            tier=Tier.TIER_2,
            reason="Core module",
            size_bytes=4096,
            sha="deadbeef",
        )
        assert entry.size_bytes == 4096
        assert entry.sha == "deadbeef"

    def test_invalid_tier_raises(self) -> None:
        with pytest.raises(Exception):
            TriagedEntry(path="x.py", tier="invalid_tier")  # type: ignore[arg-type]


# ── TreeTriageReport ──────────────────────────────────────────


class TestTreeTriageReport:
    def _report(self) -> TreeTriageReport:
        return TreeTriageReport(
            owner="acme",
            repo="myrepo",
            ref="main",
            entries=[
                TriagedEntry(path="a.py", tier=Tier.TIER_1),
                TriagedEntry(path="b.py", tier=Tier.TIER_1),
                TriagedEntry(path="tests/test_a.py", tier=Tier.TIER_2),
                TriagedEntry(path="config.yaml", tier=Tier.TIER_3),
            ],
            total_files=10,
            skipped_files=6,
        )

    def test_construction(self) -> None:
        report = self._report()
        assert report.owner == "acme"
        assert report.repo == "myrepo"
        assert len(report.entries) == 4

    def test_tier_properties(self) -> None:
        report = self._report()
        assert len(report.tier_1) == 2
        assert len(report.tier_2) == 1
        assert len(report.tier_3) == 1
        assert report.tier_1[0].path == "a.py"
        assert report.tier_2[0].path == "tests/test_a.py"

    def test_empty_report(self) -> None:
        report = TreeTriageReport(owner="acme", repo="myrepo")
        assert report.tier_1 == []
        assert report.tier_2 == []
        assert report.tier_3 == []
        assert report.total_files == 0
        assert report.skipped_files == 0


# ── FileChunk ─────────────────────────────────────────────────


class TestFileChunk:
    def test_required_fields(self) -> None:
        chunk = FileChunk(
            path="src/main.py",
            chunk_id=0,
            line_start=1,
            line_end=50,
            content="print('hello')\n",
        )
        assert chunk.chunk_id == 0
        assert chunk.line_start == 1
        assert chunk.line_end == 50
        assert chunk.sha == ""
        assert chunk.overlap_prev == 0

    def test_with_sha_and_overlap(self) -> None:
        chunk = FileChunk(
            path="foo.py",
            chunk_id=1,
            line_start=280,
            line_end=580,
            sha="cafebabe",
            content="x = 1\n",
            overlap_prev=20,
        )
        assert chunk.sha == "cafebabe"
        assert chunk.overlap_prev == 20


# ── BatchPlan ─────────────────────────────────────────────────


class TestBatchPlan:
    def test_empty_plan(self) -> None:
        plan = BatchPlan(owner="acme", repo="myrepo")
        assert plan.chunks == []
        assert plan.total_tokens_estimate == 0

    def test_plan_with_chunks(self) -> None:
        chunks = [
            FileChunk(path="a.py", chunk_id=0, line_start=1, line_end=10, content="x\n"),
            FileChunk(path="a.py", chunk_id=1, line_start=8, line_end=20, content="y\n"),
        ]
        plan = BatchPlan(
            owner="acme",
            repo="myrepo",
            ref="main",
            chunks=chunks,
            total_tokens_estimate=500,
        )
        assert len(plan.chunks) == 2
        assert plan.total_tokens_estimate == 500


# ── chunk_file ────────────────────────────────────────────────


class TestChunkFile:
    def test_small_file_single_chunk(self) -> None:
        content = "line\n" * 10
        chunks = chunk_file("src/small.py", content, max_lines=300)
        assert len(chunks) == 1
        assert chunks[0].chunk_id == 0
        assert chunks[0].line_start == 1
        assert chunks[0].line_end == 10
        assert chunks[0].overlap_prev == 0

    def test_large_file_multiple_chunks(self) -> None:
        content = "x = 1\n" * 600
        chunks = chunk_file("src/big.py", content, max_lines=300, overlap_lines=20)
        assert len(chunks) >= 2
        # Verify chunk IDs are sequential
        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_id == idx

    def test_overlap_in_subsequent_chunks(self) -> None:
        content = "line\n" * 400
        chunks = chunk_file("src/f.py", content, max_lines=300, overlap_lines=20)
        assert len(chunks) >= 2
        # Second chunk should have overlap_prev set
        second = chunks[1]
        assert second.overlap_prev == 20

    def test_first_chunk_no_overlap(self) -> None:
        content = "line\n" * 400
        chunks = chunk_file("src/f.py", content, max_lines=300)
        assert chunks[0].overlap_prev == 0

    def test_empty_file(self) -> None:
        chunks = chunk_file("src/empty.py", "")
        assert len(chunks) == 1
        assert chunks[0].content == ""
        assert chunks[0].line_start == 1

    def test_sha_embedded(self) -> None:
        content = "x = 1\n"
        chunks = chunk_file("src/f.py", content, sha="deadbeef")
        assert chunks[0].sha == "deadbeef"

    def test_python_boundary_snap(self) -> None:
        """Chunks for Python files should end at def/class boundaries."""
        lines: list[str] = []
        # Build 350 lines: first 200 are regular code, then a def at line 201,
        # then more code up to 350.
        for i in range(200):
            lines.append(f"    x_{i} = {i}\n")
        lines.append("def my_function():\n")  # line 201 — boundary
        for i in range(149):
            lines.append(f"    y_{i} = {i}\n")

        content = "".join(lines)
        chunks = chunk_file("module.py", content, max_lines=300, overlap_lines=10)
        assert len(chunks) >= 1
        # First chunk should end at or before the def boundary (line 201)
        first = chunks[0]
        assert first.line_end <= 201

    def test_non_python_file_no_boundary_snap(self) -> None:
        """Non-Python files use hard line splits without boundary detection."""
        content = "line\n" * 400
        chunks = chunk_file("data.txt", content, max_lines=300, overlap_lines=10)
        assert len(chunks) >= 2
        # First chunk ends exactly at max_lines
        assert chunks[0].line_end == 300

    def test_chunk_content_covers_all_lines(self) -> None:
        """Reassembled unique content should equal the original file."""
        content = "".join(f"line{i}\n" for i in range(100))
        chunks = chunk_file("src/f.py", content, max_lines=300)
        assert len(chunks) == 1
        assert chunks[0].content == content

    def test_line_numbers_are_one_based(self) -> None:
        content = "a\n" * 10
        chunks = chunk_file("src/f.py", content, max_lines=300)
        assert chunks[0].line_start == 1
        assert chunks[0].line_end == 10

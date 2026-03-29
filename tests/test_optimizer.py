"""Tests for ToolCallOptimizer and the ``vaig optimize`` CLI command.

Covers per-tool statistics, suggestion generation, redundant call
detection, empty-data handling, and CLI integration via Typer's
CliRunner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from typer.testing import CliRunner

from vaig.core.optimizer import (
    RedundantCall,
    ToolCallOptimizer,
    ToolInsights,
    ToolStats,
    _hash_args,
)
from vaig.core.tool_call_store import ToolCallStore

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write a list of dicts to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _make_record(
    tool_name: str = "kubectl_get_pods",
    run_id: str = "run-1",
    error: bool = False,
    error_message: str = "",
    duration_s: float = 0.5,
    cached: bool = False,
    tool_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a tool call record dict with defaults."""
    rec: dict[str, Any] = {
        "tool_name": tool_name,
        "tool_args": tool_args if tool_args is not None else {"namespace": "default"},
        "output": "ok",
        "output_size_bytes": 2,
        "error": error,
        "error_type": "RuntimeError" if error else "",
        "error_message": error_message,
        "duration_s": duration_s,
        "timestamp": "2026-03-29T00:00:00Z",
        "agent_name": "gatherer",
        "run_id": run_id,
        "iteration": 1,
        "cached": cached,
    }
    rec.update(kwargs)
    return rec


def _seed_store(
    tmp_path: Path,
    runs: dict[str, list[dict[str, Any]]],
    date: str = "2026-03-29",
) -> ToolCallStore:
    """Create a ToolCallStore populated with the given runs.

    Args:
        tmp_path: pytest temp directory.
        runs: Mapping of run_id → list of record dicts.
        date: Date directory name.

    Returns:
        A ToolCallStore pointed at the seeded directory.
    """
    for run_id, records in runs.items():
        _write_jsonl(
            tmp_path / "tool_results" / date / f"{run_id}.jsonl",
            records,
        )
    return ToolCallStore(base_dir=tmp_path)


# ---------------------------------------------------------------------------
# _hash_args
# ---------------------------------------------------------------------------


class TestHashArgs:
    def test_deterministic(self) -> None:
        h1 = _hash_args("tool_a", {"x": 1, "y": 2})
        h2 = _hash_args("tool_a", {"y": 2, "x": 1})
        assert h1 == h2

    def test_different_tools_produce_different_hashes(self) -> None:
        h1 = _hash_args("tool_a", {"x": 1})
        h2 = _hash_args("tool_b", {"x": 1})
        assert h1 != h2

    def test_empty_args(self) -> None:
        h = _hash_args("tool_a", {})
        assert isinstance(h, str)
        assert len(h) == 12


# ---------------------------------------------------------------------------
# ToolStats / RedundantCall / ToolInsights dataclass tests
# ---------------------------------------------------------------------------


class TestDataModels:
    def test_tool_stats_defaults(self) -> None:
        stats = ToolStats(
            call_count=10,
            failure_count=2,
            failure_rate=0.2,
            avg_duration_s=1.5,
            max_duration_s=5.0,
            cache_hit_count=3,
            cache_hit_rate=0.3,
            unique_arg_combos=4,
        )
        assert stats.common_errors == []
        assert stats.call_count == 10

    def test_redundant_call(self) -> None:
        rc = RedundantCall(
            tool_name="tool_a",
            args_hash="abc123",
            count=5,
            run_id="run-1",
        )
        assert rc.count == 5

    def test_tool_insights_defaults(self) -> None:
        insights = ToolInsights(
            total_runs=0,
            total_calls=0,
            total_duration_s=0.0,
            avg_calls_per_run=0.0,
        )
        assert insights.tools == {}
        assert insights.redundant_calls == []
        assert insights.suggestions == []


# ---------------------------------------------------------------------------
# ToolCallOptimizer.analyze()
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_empty_store_returns_zero_insights(self, tmp_path: Path) -> None:
        store = ToolCallStore(base_dir=tmp_path)
        optimizer = ToolCallOptimizer(store)
        insights = optimizer.analyze()

        assert insights.total_runs == 0
        assert insights.total_calls == 0
        assert insights.total_duration_s == 0.0

    def test_single_run_basic_stats(self, tmp_path: Path) -> None:
        store = _seed_store(tmp_path, {
            "run-1": [
                _make_record(tool_name="tool_a", duration_s=1.0),
                _make_record(tool_name="tool_a", duration_s=3.0),
                _make_record(tool_name="tool_b", duration_s=0.5),
            ],
        })
        optimizer = ToolCallOptimizer(store)
        insights = optimizer.analyze()

        assert insights.total_runs == 1
        assert insights.total_calls == 3
        assert "tool_a" in insights.tools
        assert "tool_b" in insights.tools
        assert insights.tools["tool_a"].call_count == 2
        assert insights.tools["tool_b"].call_count == 1

    def test_failure_stats(self, tmp_path: Path) -> None:
        store = _seed_store(tmp_path, {
            "run-1": [
                _make_record(tool_name="tool_a", error=True, error_message="timeout"),
                _make_record(tool_name="tool_a", error=True, error_message="timeout"),
                _make_record(tool_name="tool_a", error=False),
            ],
        })
        optimizer = ToolCallOptimizer(store)
        insights = optimizer.analyze()

        stats = insights.tools["tool_a"]
        assert stats.failure_count == 2
        assert abs(stats.failure_rate - 2.0 / 3.0) < 0.01

    def test_duration_stats(self, tmp_path: Path) -> None:
        store = _seed_store(tmp_path, {
            "run-1": [
                _make_record(tool_name="tool_a", duration_s=2.0),
                _make_record(tool_name="tool_a", duration_s=6.0),
            ],
        })
        optimizer = ToolCallOptimizer(store)
        insights = optimizer.analyze()

        stats = insights.tools["tool_a"]
        assert stats.avg_duration_s == 4.0
        assert stats.max_duration_s == 6.0

    def test_cache_hit_stats(self, tmp_path: Path) -> None:
        store = _seed_store(tmp_path, {
            "run-1": [
                _make_record(tool_name="tool_a", cached=True),
                _make_record(tool_name="tool_a", cached=False),
                _make_record(tool_name="tool_a", cached=True),
            ],
        })
        optimizer = ToolCallOptimizer(store)
        insights = optimizer.analyze()

        stats = insights.tools["tool_a"]
        assert stats.cache_hit_count == 2
        assert abs(stats.cache_hit_rate - 2.0 / 3.0) < 0.01

    def test_common_errors_top_3(self, tmp_path: Path) -> None:
        records = [
            _make_record(error=True, error_message="timeout"),
            _make_record(error=True, error_message="timeout"),
            _make_record(error=True, error_message="permission denied"),
            _make_record(error=True, error_message="not found"),
            _make_record(error=True, error_message="not found"),
            _make_record(error=True, error_message="not found"),
            _make_record(error=True, error_message="rare error"),
        ]
        store = _seed_store(tmp_path, {"run-1": records})
        optimizer = ToolCallOptimizer(store)
        insights = optimizer.analyze()

        errors = insights.tools["kubectl_get_pods"].common_errors
        assert len(errors) <= 3
        # "not found" (3x) should be first
        assert errors[0] == "not found"

    def test_redundant_call_detection(self, tmp_path: Path) -> None:
        # Same tool + same args within same run = redundant
        same_args = {"namespace": "kube-system"}
        store = _seed_store(tmp_path, {
            "run-1": [
                _make_record(tool_name="tool_a", tool_args=same_args),
                _make_record(tool_name="tool_a", tool_args=same_args),
                _make_record(tool_name="tool_a", tool_args=same_args),
            ],
        })
        optimizer = ToolCallOptimizer(store)
        insights = optimizer.analyze()

        assert len(insights.redundant_calls) == 1
        assert insights.redundant_calls[0].count == 3
        assert insights.redundant_calls[0].tool_name == "tool_a"

    def test_no_redundancy_for_different_args(self, tmp_path: Path) -> None:
        store = _seed_store(tmp_path, {
            "run-1": [
                _make_record(tool_name="tool_a", tool_args={"ns": "a"}),
                _make_record(tool_name="tool_a", tool_args={"ns": "b"}),
                _make_record(tool_name="tool_a", tool_args={"ns": "c"}),
            ],
        })
        optimizer = ToolCallOptimizer(store)
        insights = optimizer.analyze()

        # Each call has different args — no redundancies
        assert len(insights.redundant_calls) == 0

    def test_last_n_runs_limits_scope(self, tmp_path: Path) -> None:
        runs: dict[str, list[dict[str, Any]]] = {}
        for i in range(10):
            runs[f"run-{i}"] = [_make_record(run_id=f"run-{i}")]

        store = _seed_store(tmp_path, runs)
        optimizer = ToolCallOptimizer(store)

        insights_3 = optimizer.analyze(last_n_runs=3)
        insights_all = optimizer.analyze(last_n_runs=100)

        assert insights_3.total_runs <= 3
        assert insights_all.total_runs == 10

    def test_multiple_runs_aggregation(self, tmp_path: Path) -> None:
        store = _seed_store(tmp_path, {
            "run-1": [_make_record(tool_name="tool_a", run_id="run-1")],
            "run-2": [_make_record(tool_name="tool_a", run_id="run-2")],
        })
        optimizer = ToolCallOptimizer(store)
        insights = optimizer.analyze()

        assert insights.total_runs == 2
        assert insights.total_calls == 2
        assert insights.tools["tool_a"].call_count == 2

    def test_zero_duration_handled(self, tmp_path: Path) -> None:
        store = _seed_store(tmp_path, {
            "run-1": [_make_record(duration_s=0.0)],
        })
        optimizer = ToolCallOptimizer(store)
        insights = optimizer.analyze()

        assert insights.tools["kubectl_get_pods"].avg_duration_s == 0.0
        assert insights.tools["kubectl_get_pods"].max_duration_s == 0.0


# ---------------------------------------------------------------------------
# ToolCallOptimizer.suggest()
# ---------------------------------------------------------------------------


class TestSuggest:
    def test_no_data_suggestion(self) -> None:
        optimizer = ToolCallOptimizer.__new__(ToolCallOptimizer)
        insights = ToolInsights(
            total_runs=0,
            total_calls=0,
            total_duration_s=0.0,
            avg_calls_per_run=0.0,
        )
        suggestions = optimizer.suggest(insights)
        assert any("no tool call data" in s.lower() for s in suggestions)

    def test_high_failure_rate_suggestion(self) -> None:
        optimizer = ToolCallOptimizer.__new__(ToolCallOptimizer)
        insights = ToolInsights(
            total_runs=1,
            total_calls=10,
            total_duration_s=5.0,
            avg_calls_per_run=10.0,
            tools={
                "bad_tool": ToolStats(
                    call_count=10,
                    failure_count=9,
                    failure_rate=0.9,
                    avg_duration_s=0.5,
                    max_duration_s=1.0,
                    cache_hit_count=0,
                    cache_hit_rate=0.0,
                    unique_arg_combos=2,
                    common_errors=["timeout"],
                ),
            },
        )
        suggestions = optimizer.suggest(insights)
        assert any("bad_tool" in s and "90%" in s for s in suggestions)

    def test_slow_tool_suggestion(self) -> None:
        optimizer = ToolCallOptimizer.__new__(ToolCallOptimizer)
        insights = ToolInsights(
            total_runs=1,
            total_calls=5,
            total_duration_s=100.0,
            avg_calls_per_run=5.0,
            tools={
                "slow_tool": ToolStats(
                    call_count=5,
                    failure_count=0,
                    failure_rate=0.0,
                    avg_duration_s=20.0,
                    max_duration_s=30.0,
                    cache_hit_count=0,
                    cache_hit_rate=0.0,
                    unique_arg_combos=2,
                ),
            },
        )
        suggestions = optimizer.suggest(insights)
        assert any("slow_tool" in s and "20.0s" in s for s in suggestions)

    def test_low_cache_hit_suggestion(self) -> None:
        optimizer = ToolCallOptimizer.__new__(ToolCallOptimizer)
        insights = ToolInsights(
            total_runs=1,
            total_calls=15,
            total_duration_s=7.5,
            avg_calls_per_run=15.0,
            tools={
                "uncached_tool": ToolStats(
                    call_count=15,
                    failure_count=0,
                    failure_rate=0.0,
                    avg_duration_s=0.5,
                    max_duration_s=1.0,
                    cache_hit_count=0,
                    cache_hit_rate=0.0,
                    unique_arg_combos=3,
                ),
            },
        )
        suggestions = optimizer.suggest(insights)
        assert any("uncached_tool" in s and "cache" in s.lower() for s in suggestions)

    def test_redundant_call_suggestion(self) -> None:
        optimizer = ToolCallOptimizer.__new__(ToolCallOptimizer)
        insights = ToolInsights(
            total_runs=1,
            total_calls=10,
            total_duration_s=5.0,
            avg_calls_per_run=10.0,
            redundant_calls=[
                RedundantCall(
                    tool_name="dup_tool",
                    args_hash="abc",
                    count=5,
                    run_id="run-x",
                ),
            ],
        )
        suggestions = optimizer.suggest(insights)
        assert any("dup_tool" in s and "5 times" in s for s in suggestions)

    def test_no_suggestions_for_healthy_tools(self) -> None:
        optimizer = ToolCallOptimizer.__new__(ToolCallOptimizer)
        insights = ToolInsights(
            total_runs=1,
            total_calls=5,
            total_duration_s=2.0,
            avg_calls_per_run=5.0,
            tools={
                "good_tool": ToolStats(
                    call_count=5,
                    failure_count=0,
                    failure_rate=0.0,
                    avg_duration_s=0.4,
                    max_duration_s=0.8,
                    cache_hit_count=3,
                    cache_hit_rate=0.6,
                    unique_arg_combos=2,
                ),
            },
        )
        suggestions = optimizer.suggest(insights)
        assert len(suggestions) == 0

    def test_failure_rate_below_threshold_no_suggestion(self) -> None:
        """A tool with 50% failure rate (below 80%) should not trigger."""
        optimizer = ToolCallOptimizer.__new__(ToolCallOptimizer)
        insights = ToolInsights(
            total_runs=1,
            total_calls=4,
            total_duration_s=2.0,
            avg_calls_per_run=4.0,
            tools={
                "moderate_tool": ToolStats(
                    call_count=4,
                    failure_count=2,
                    failure_rate=0.5,
                    avg_duration_s=0.5,
                    max_duration_s=1.0,
                    cache_hit_count=0,
                    cache_hit_rate=0.0,
                    unique_arg_combos=2,
                ),
            },
        )
        suggestions = optimizer.suggest(insights)
        # 50% failure and <10 calls → no failure suggestion
        # But 0 cache hits with only 4 calls (<10 threshold) → no cache suggestion
        assert not any("moderate_tool" in s for s in suggestions)

    def test_redundant_count_3_or_less_no_suggestion(self) -> None:
        """Redundant calls with count ≤ 3 should not generate a suggestion."""
        optimizer = ToolCallOptimizer.__new__(ToolCallOptimizer)
        insights = ToolInsights(
            total_runs=1,
            total_calls=3,
            total_duration_s=1.5,
            avg_calls_per_run=3.0,
            redundant_calls=[
                RedundantCall(
                    tool_name="tool_a",
                    args_hash="abc",
                    count=3,
                    run_id="run-1",
                ),
            ],
        )
        suggestions = optimizer.suggest(insights)
        assert not any("tool_a" in s and "3 times" in s for s in suggestions)


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


class TestOptimizeCli:
    def test_optimize_no_data(self, tmp_path: Path) -> None:
        """CLI should handle empty store gracefully."""
        from vaig.cli.app import app

        with patch("vaig.cli.commands.optimize._get_settings") as mock_settings:
            settings = mock_settings.return_value
            settings.logging.tool_results_dir = str(tmp_path)

            result = runner.invoke(app, ["optimize"])

        assert result.exit_code == 0
        assert "0 runs" in result.output or "0" in result.output

    def test_optimize_with_data(self, tmp_path: Path) -> None:
        """CLI should display stats when data exists."""
        from vaig.cli.app import app

        # Seed some data
        records = [
            _make_record(tool_name="tool_a", duration_s=1.0),
            _make_record(tool_name="tool_b", duration_s=2.0),
        ]
        _write_jsonl(
            tmp_path / "tool_results" / "2026-03-29" / "run-1.jsonl",
            records,
        )

        with patch("vaig.cli.commands.optimize._get_settings") as mock_settings:
            settings = mock_settings.return_value
            settings.logging.tool_results_dir = str(tmp_path)

            result = runner.invoke(app, ["optimize"])

        assert result.exit_code == 0
        assert "tool_a" in result.output
        assert "tool_b" in result.output

    def test_optimize_last_flag(self, tmp_path: Path) -> None:
        """--last N should be accepted."""
        from vaig.cli.app import app

        with patch("vaig.cli.commands.optimize._get_settings") as mock_settings:
            settings = mock_settings.return_value
            settings.logging.tool_results_dir = str(tmp_path)

            result = runner.invoke(app, ["optimize", "--last", "10"])

        assert result.exit_code == 0

    def test_optimize_with_redundant_calls(self, tmp_path: Path) -> None:
        """CLI should show redundant calls section."""
        from vaig.cli.app import app

        same_args = {"namespace": "default"}
        records = [
            _make_record(tool_name="tool_a", tool_args=same_args),
            _make_record(tool_name="tool_a", tool_args=same_args),
            _make_record(tool_name="tool_a", tool_args=same_args),
            _make_record(tool_name="tool_a", tool_args=same_args),
            _make_record(tool_name="tool_a", tool_args=same_args),
        ]
        _write_jsonl(
            tmp_path / "tool_results" / "2026-03-29" / "run-1.jsonl",
            records,
        )

        with patch("vaig.cli.commands.optimize._get_settings") as mock_settings:
            settings = mock_settings.return_value
            settings.logging.tool_results_dir = str(tmp_path)

            result = runner.invoke(app, ["optimize"])

        assert result.exit_code == 0
        # Should mention redundant calls
        assert "5" in result.output  # 5 times same args

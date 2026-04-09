"""Tests for _inject_report_metadata cost/tool batch-2 behaviour in live.py.

Covers:
- orch_result with run_cost_usd + total_usage → CostMetrics populated
- tool_logger with tool_name_counts + tool_count → ToolUsageSummary populated
- Both None → no crash, cost_metrics / tool_usage remain None
- Edge cases: zero cost, empty counts, orch_result already has cost_metrics set
"""

from __future__ import annotations

from collections import Counter
from unittest.mock import MagicMock, patch

from vaig.cli.commands.live import _inject_report_metadata
from vaig.skills.service_health.schema import (
    CostMetrics,
    ExecutiveSummary,
    HealthReport,
    OverallStatus,
    ReportMetadata,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_report(**metadata_kwargs: object) -> HealthReport:
    """Return a minimal HealthReport for metadata injection tests."""
    return HealthReport(
        executive_summary=ExecutiveSummary(
            overall_status=OverallStatus.HEALTHY,
            scope="Namespace: default",
            summary_text="All good.",
            critical_count=0,
            warning_count=0,
            issues_found=0,
            services_checked=1,
        ),
        metadata=ReportMetadata(**metadata_kwargs),  # type: ignore[arg-type]
    )


def _make_orch_result(
    run_cost_usd: float | None = None,
    total_usage: dict[str, int] | None = None,
) -> MagicMock:
    """Return a MagicMock that quacks like OrchestratorResult."""
    result = MagicMock()
    result.run_cost_usd = run_cost_usd
    result.total_usage = total_usage
    return result


def _make_tool_logger(
    tool_name_counts: dict[str, int] | None = None,
    tool_count: int = 0,
) -> MagicMock:
    """Return a MagicMock that quacks like ToolCallLogger."""
    logger = MagicMock()
    logger.tool_name_counts = Counter(tool_name_counts or {})
    logger.tool_count = tool_count
    return logger


# ── Cost metrics injection ────────────────────────────────────────────────────


class TestInjectCostMetrics:
    """Tests for the cost_metrics path in _inject_report_metadata."""

    def test_populates_cost_metrics_from_orch_result(self) -> None:
        """run_cost_usd and total_usage are extracted into CostMetrics."""
        report = _make_report()
        orch = _make_orch_result(
            run_cost_usd=0.005,
            total_usage={"prompt_tokens": 1000, "completion_tokens": 500},
        )

        _inject_report_metadata(report, orch_result=orch)

        assert report.metadata.cost_metrics is not None
        assert report.metadata.cost_metrics.run_cost_usd == 0.005
        assert report.metadata.cost_metrics.total_tokens == 1500  # 1000 + 500

    def test_total_tokens_key_takes_precedence(self) -> None:
        """When total_tokens key is present it is used directly — no double-counting."""
        report = _make_report()
        orch = _make_orch_result(
            run_cost_usd=0.001,
            total_usage={"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250},
        )

        _inject_report_metadata(report, orch_result=orch)

        # total_tokens key takes precedence over summing prompt + completion
        assert report.metadata.cost_metrics is not None
        assert report.metadata.cost_metrics.total_tokens == 250

    def test_total_tokens_fallback_to_prompt_plus_completion(self) -> None:
        """Without total_tokens key, falls back to prompt_tokens + completion_tokens."""
        report = _make_report()
        orch = _make_orch_result(
            run_cost_usd=0.001,
            total_usage={"prompt_tokens": 200, "completion_tokens": 50},
        )

        _inject_report_metadata(report, orch_result=orch)

        assert report.metadata.cost_metrics is not None
        assert report.metadata.cost_metrics.total_tokens == 250

    def test_cost_only_no_total_usage(self) -> None:
        """Only run_cost_usd provided — total_tokens is None."""
        report = _make_report()
        orch = _make_orch_result(run_cost_usd=0.002, total_usage=None)

        _inject_report_metadata(report, orch_result=orch)

        assert report.metadata.cost_metrics is not None
        assert report.metadata.cost_metrics.run_cost_usd == 0.002
        assert report.metadata.cost_metrics.total_tokens is None

    def test_usage_only_no_cost(self) -> None:
        """Only total_usage provided — run_cost_usd is None but metrics still set."""
        report = _make_report()
        orch = _make_orch_result(run_cost_usd=None, total_usage={"prompt_tokens": 300, "completion_tokens": 100})

        _inject_report_metadata(report, orch_result=orch)

        assert report.metadata.cost_metrics is not None
        assert report.metadata.cost_metrics.run_cost_usd is None
        assert report.metadata.cost_metrics.total_tokens == 400

    def test_no_orch_result_leaves_cost_metrics_none(self) -> None:
        """When orch_result=None, cost_metrics remains None."""
        report = _make_report()

        _inject_report_metadata(report, orch_result=None)

        assert report.metadata.cost_metrics is None

    def test_zero_cost_is_stored(self) -> None:
        """run_cost_usd=0.0 is stored — not treated as absent."""
        report = _make_report()
        orch = _make_orch_result(run_cost_usd=0.0, total_usage={"prompt_tokens": 10})

        _inject_report_metadata(report, orch_result=orch)

        assert report.metadata.cost_metrics is not None
        assert report.metadata.cost_metrics.run_cost_usd == 0.0

    def test_does_not_overwrite_existing_cost_metrics(self) -> None:
        """If cost_metrics is already set on metadata, it is not overwritten."""
        report = _make_report()
        report.metadata.cost_metrics = CostMetrics(run_cost_usd=0.999, total_tokens=999)
        orch = _make_orch_result(run_cost_usd=0.001, total_usage={"prompt_tokens": 1})

        _inject_report_metadata(report, orch_result=orch)

        assert report.metadata.cost_metrics.run_cost_usd == 0.999  # unchanged

    def test_total_usage_empty_dict_produces_none_tokens(self) -> None:
        """Empty total_usage dict means sum is 0, which becomes None total_tokens."""
        report = _make_report()
        orch = _make_orch_result(run_cost_usd=0.001, total_usage={})

        _inject_report_metadata(report, orch_result=orch)

        # sum({}) == 0, which is falsy → total_tokens should be None
        assert report.metadata.cost_metrics is not None
        assert report.metadata.cost_metrics.total_tokens is None


# ── Tool usage injection ──────────────────────────────────────────────────────


class TestInjectToolUsage:
    """Tests for the tool_usage path in _inject_report_metadata."""

    def test_populates_tool_usage_from_tool_logger(self) -> None:
        """tool_name_counts and tool_count are extracted into ToolUsageSummary."""
        report = _make_report()
        logger = _make_tool_logger(
            tool_name_counts={"kubectl_get": 3, "kubectl_logs": 1},
            tool_count=4,
        )

        _inject_report_metadata(report, tool_logger=logger)

        assert report.metadata.tool_usage is not None
        assert report.metadata.tool_usage.tool_counts == {"kubectl_get": 3, "kubectl_logs": 1}
        assert report.metadata.tool_usage.tool_calls == 4

    def test_no_tool_logger_leaves_tool_usage_none(self) -> None:
        """When tool_logger=None, tool_usage remains None."""
        report = _make_report()

        _inject_report_metadata(report, tool_logger=None)

        assert report.metadata.tool_usage is None

    def test_empty_tool_counts_zero_steps_leaves_none(self) -> None:
        """Empty counts + zero steps → no ToolUsageSummary created."""
        report = _make_report()
        logger = _make_tool_logger(tool_name_counts={}, tool_count=0)

        _inject_report_metadata(report, tool_logger=logger)

        # Both values are falsy, so tool_usage should remain None
        assert report.metadata.tool_usage is None

    def test_counts_only_no_steps(self) -> None:
        """tool_count=0 is treated as None for tool_calls, but counts still stored."""
        report = _make_report()
        logger = _make_tool_logger(
            tool_name_counts={"shell_exec": 2},
            tool_count=0,
        )

        _inject_report_metadata(report, tool_logger=logger)

        assert report.metadata.tool_usage is not None
        assert report.metadata.tool_usage.tool_counts == {"shell_exec": 2}
        assert report.metadata.tool_usage.tool_calls is None

    def test_steps_only_no_counts(self) -> None:
        """Empty tool_counts with non-zero steps → tool_calls stored, tool_counts None."""
        report = _make_report()
        logger = _make_tool_logger(tool_name_counts={}, tool_count=5)

        _inject_report_metadata(report, tool_logger=logger)

        assert report.metadata.tool_usage is not None
        assert report.metadata.tool_usage.tool_counts is None
        assert report.metadata.tool_usage.tool_calls == 5


# ── Both params None ──────────────────────────────────────────────────────────


class TestInjectBothNone:
    """Tests for the case where both orch_result and tool_logger are None."""

    def test_no_crash_no_side_effects(self) -> None:
        """With both None, the call is a no-op for cost/tool fields."""
        report = _make_report()

        _inject_report_metadata(report, orch_result=None, tool_logger=None)

        assert report.metadata.cost_metrics is None
        assert report.metadata.tool_usage is None

    @patch("vaig.tools.gke.cost_estimation.fetch_workload_costs", return_value=None)
    def test_still_fills_basic_metadata(self, _mock_fetch: MagicMock) -> None:
        """Other fields (cluster_name, model_used) still work when cost/tool are None."""
        report = _make_report()
        gke = MagicMock()
        gke.cluster_name = "my-cluster"
        gke.project_id = "my-project"
        gke.trends = None  # Trends are opt-in (disabled by default)

        _inject_report_metadata(
            report,
            gke_config=gke,
            model_id="gemini-pro",
            orch_result=None,
            tool_logger=None,
        )

        assert report.metadata.cluster_name == "my-cluster"
        assert report.metadata.model_used == "gemini-pro"
        assert report.metadata.cost_metrics is None
        assert report.metadata.tool_usage is None


# ── Both params populated ──────────────────────────────────────────────────────


class TestInjectBothPopulated:
    """Tests for the case where both orch_result and tool_logger are provided."""

    def test_both_fields_set_correctly(self) -> None:
        """Providing both sets both cost_metrics and tool_usage."""
        report = _make_report()
        orch = _make_orch_result(run_cost_usd=0.0077, total_usage={"prompt_tokens": 500, "completion_tokens": 200})
        logger = _make_tool_logger(
            tool_name_counts={"helm_list": 1, "kubectl_get": 5},
            tool_count=6,
        )

        _inject_report_metadata(report, orch_result=orch, tool_logger=logger)

        assert report.metadata.cost_metrics is not None
        assert report.metadata.cost_metrics.run_cost_usd == 0.0077
        assert report.metadata.cost_metrics.total_tokens == 700
        assert report.metadata.tool_usage is not None
        assert report.metadata.tool_usage.tool_counts == {"helm_list": 1, "kubectl_get": 5}
        assert report.metadata.tool_usage.tool_calls == 6

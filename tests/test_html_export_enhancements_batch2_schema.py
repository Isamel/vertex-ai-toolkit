"""Tests for CostMetrics, ToolUsageSummary, and ReportMetadata batch-2 fields.

Covers:
- CostMetrics defaults and populated construction
- ToolUsageSummary defaults and populated construction
- ReportMetadata correctly embeds the new nested models
- model_dump serialisation round-trips cleanly
"""

from __future__ import annotations

from vaig.skills.service_health.schema import (
    CostMetrics,
    ReportMetadata,
    ToolUsageSummary,
)


class TestCostMetrics:
    """Unit tests for the CostMetrics Pydantic model."""

    def test_defaults_are_none(self) -> None:
        """Both fields default to None when not supplied."""
        m = CostMetrics()
        assert m.run_cost_usd is None
        assert m.total_tokens is None
        assert m.estimated_cost is None

    def test_populated_values(self) -> None:
        """Supplied values are stored correctly."""
        m = CostMetrics(run_cost_usd=0.0042, total_tokens=1500)
        assert m.run_cost_usd == 0.0042
        assert m.total_tokens == 1500

    def test_partial_cost_only(self) -> None:
        """Only run_cost_usd supplied — total_tokens stays None."""
        m = CostMetrics(run_cost_usd=0.001)
        assert m.run_cost_usd == 0.001
        assert m.total_tokens is None

    def test_partial_tokens_only(self) -> None:
        """Only total_tokens supplied — run_cost_usd stays None."""
        m = CostMetrics(total_tokens=3000)
        assert m.run_cost_usd is None
        assert m.total_tokens == 3000

    def test_serialisation_round_trip(self) -> None:
        """model_dump produces expected JSON-serialisable dict."""
        m = CostMetrics(run_cost_usd=0.000123, total_tokens=512)
        d = m.model_dump(mode="json")
        assert d == {"run_cost_usd": 0.000123, "total_tokens": 512, "estimated_cost": None}

    def test_serialisation_none_fields(self) -> None:
        """None fields appear as None in model_dump output."""
        m = CostMetrics()
        d = m.model_dump(mode="json")
        assert d["run_cost_usd"] is None
        assert d["total_tokens"] is None
        assert d["estimated_cost"] is None

    def test_zero_cost_stored(self) -> None:
        """Explicit 0.0 cost is stored (not treated as falsy/None)."""
        m = CostMetrics(run_cost_usd=0.0)
        assert m.run_cost_usd == 0.0

    def test_zero_tokens_stored(self) -> None:
        """Explicit 0 tokens is stored."""
        m = CostMetrics(total_tokens=0)
        assert m.total_tokens == 0


class TestToolUsageSummary:
    """Unit tests for the ToolUsageSummary Pydantic model."""

    def test_defaults_are_none(self) -> None:
        """Both fields default to None when not supplied."""
        t = ToolUsageSummary()
        assert t.tool_counts is None
        assert t.tool_calls is None

    def test_populated_values(self) -> None:
        """Supplied values are stored correctly."""
        t = ToolUsageSummary(tool_counts={"kubectl_get": 4, "kubectl_logs": 2}, tool_calls=6)
        assert t.tool_counts == {"kubectl_get": 4, "kubectl_logs": 2}
        assert t.tool_calls == 6

    def test_partial_counts_only(self) -> None:
        """Only tool_counts supplied — tool_calls stays None."""
        t = ToolUsageSummary(tool_counts={"shell_exec": 1})
        assert t.tool_counts == {"shell_exec": 1}
        assert t.tool_calls is None

    def test_partial_steps_only(self) -> None:
        """Only tool_calls supplied — tool_counts stays None."""
        t = ToolUsageSummary(tool_calls=10)
        assert t.tool_counts is None
        assert t.tool_calls == 10

    def test_serialisation_round_trip(self) -> None:
        """model_dump produces expected JSON-serialisable dict."""
        t = ToolUsageSummary(tool_counts={"kubectl_get": 4}, tool_calls=4)
        d = t.model_dump(mode="json")
        assert d == {"tool_counts": {"kubectl_get": 4}, "tool_calls": 4}

    def test_serialisation_none_fields(self) -> None:
        """None fields appear as None in model_dump output."""
        t = ToolUsageSummary()
        d = t.model_dump(mode="json")
        assert d["tool_counts"] is None
        assert d["tool_calls"] is None

    def test_empty_tool_counts_dict(self) -> None:
        """Empty dict is stored as-is (not coerced to None)."""
        t = ToolUsageSummary(tool_counts={})
        assert t.tool_counts == {}


class TestReportMetadataBatch2Fields:
    """Tests that ReportMetadata correctly embeds CostMetrics and ToolUsageSummary."""

    def test_defaults_are_none(self) -> None:
        """cost_metrics and tool_usage default to None."""
        meta = ReportMetadata()
        assert meta.cost_metrics is None
        assert meta.tool_usage is None

    def test_cost_metrics_populated(self) -> None:
        """cost_metrics can be set on ReportMetadata."""
        meta = ReportMetadata(
            cost_metrics=CostMetrics(run_cost_usd=0.01, total_tokens=2000)
        )
        assert meta.cost_metrics is not None
        assert meta.cost_metrics.run_cost_usd == 0.01
        assert meta.cost_metrics.total_tokens == 2000

    def test_tool_usage_populated(self) -> None:
        """tool_usage can be set on ReportMetadata."""
        meta = ReportMetadata(
            tool_usage=ToolUsageSummary(tool_counts={"kubectl_get": 3}, tool_calls=3)
        )
        assert meta.tool_usage is not None
        assert meta.tool_usage.tool_counts == {"kubectl_get": 3}
        assert meta.tool_usage.tool_calls == 3

    def test_both_fields_populated(self) -> None:
        """Both cost_metrics and tool_usage can be set together."""
        meta = ReportMetadata(
            cost_metrics=CostMetrics(run_cost_usd=0.005, total_tokens=1200),
            tool_usage=ToolUsageSummary(tool_counts={"search": 2}, tool_calls=2),
        )
        assert meta.cost_metrics is not None
        assert meta.cost_metrics.run_cost_usd == 0.005
        assert meta.tool_usage is not None
        assert meta.tool_usage.tool_calls == 2

    def test_model_dump_includes_nested_cost_metrics(self) -> None:
        """model_dump(mode='json') includes cost_metrics dict correctly."""
        meta = ReportMetadata(
            cost_metrics=CostMetrics(run_cost_usd=0.01, total_tokens=2000)
        )
        d = meta.model_dump(mode="json")
        assert d["cost_metrics"] is not None
        assert d["cost_metrics"]["run_cost_usd"] == 0.01
        assert d["cost_metrics"]["total_tokens"] == 2000

    def test_model_dump_includes_nested_tool_usage(self) -> None:
        """model_dump(mode='json') includes tool_usage dict correctly."""
        meta = ReportMetadata(
            tool_usage=ToolUsageSummary(tool_counts={"kubectl_get": 4}, tool_calls=4)
        )
        d = meta.model_dump(mode="json")
        assert d["tool_usage"] is not None
        assert d["tool_usage"]["tool_counts"] == {"kubectl_get": 4}
        assert d["tool_usage"]["tool_calls"] == 4

    def test_model_dump_none_fields_are_null(self) -> None:
        """When cost_metrics / tool_usage are None, model_dump shows null values."""
        meta = ReportMetadata()
        d = meta.model_dump(mode="json")
        assert d["cost_metrics"] is None
        assert d["tool_usage"] is None

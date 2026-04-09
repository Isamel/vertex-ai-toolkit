"""Tests for vaig.core.report_metadata — shared metadata injection."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from vaig.core.report_metadata import format_models_used, inject_report_metadata
from vaig.skills.service_health.schema import (
    CostMetrics,
    ReportMetadata,
    ToolUsageSummary,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_report(*, metadata: ReportMetadata | None = None) -> SimpleNamespace:
    """Return a minimal report-like object with a ``metadata`` attribute."""
    return SimpleNamespace(
        metadata=metadata or ReportMetadata(),
        cluster_overview=None,
    )


def _make_orch_result(
    *,
    run_cost_usd: float = 0.0,
    total_usage: dict[str, int] | None = None,
    models_used: list[str] | None = None,
) -> SimpleNamespace:
    """Return a minimal OrchestratorResult-like object."""
    return SimpleNamespace(
        run_cost_usd=run_cost_usd,
        total_usage=total_usage or {},
        models_used=models_used or [],
    )


def _make_gke_config(
    *,
    cluster_name: str = "my-cluster",
    project_id: str = "my-project",
    default_namespace: str = "production",
    trends: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        cluster_name=cluster_name,
        project_id=project_id,
        default_namespace=default_namespace,
        trends=trends,
    )


def _make_tool_logger(
    *,
    tool_name_counts: dict[str, int] | None = None,
    pipeline_tool_name_counts: dict[str, int] | None = None,
    tool_count: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        tool_name_counts=tool_name_counts or {},
        pipeline_tool_name_counts=pipeline_tool_name_counts or {},
        tool_count=tool_count,
    )


# ── format_models_used ───────────────────────────────────────────────────────


class TestFormatModelsUsed:
    def test_empty_list(self) -> None:
        assert format_models_used([]) == ""

    def test_single_model(self) -> None:
        assert format_models_used(["gemini-2.5-flash"]) == "gemini-2.5-flash"

    def test_same_model_repeated(self) -> None:
        assert format_models_used(["gemini-2.5-flash"] * 3) == "gemini-2.5-flash ×3"

    def test_multiple_distinct_models(self) -> None:
        result = format_models_used(["gemini-2.5-flash", "gemini-2.5-pro"])
        assert "gemini-2.5-flash" in result
        assert "gemini-2.5-pro" in result


# ── inject_report_metadata ───────────────────────────────────────────────────


class TestInjectReportMetadata:
    """Core metadata injection tests."""

    def test_no_metadata_attr_is_noop(self) -> None:
        """When report has no metadata attribute, function is a no-op."""
        report = SimpleNamespace()
        inject_report_metadata(report)  # should not raise

    def test_none_metadata_is_noop(self) -> None:
        """When report.metadata is None, function is a no-op."""
        report = SimpleNamespace(metadata=None)
        inject_report_metadata(report)

    def test_generated_at_always_set(self) -> None:
        """generated_at should always be set to current time."""
        report = _make_report()
        inject_report_metadata(report)
        assert report.metadata.generated_at
        assert report.metadata.generated_at.endswith("Z")

    def test_skill_version_set(self) -> None:
        """skill_version should be populated when empty."""
        report = _make_report()
        inject_report_metadata(report)
        assert report.metadata.skill_version.startswith("vaig ")

    def test_skill_version_not_overwritten(self) -> None:
        """skill_version should NOT be overwritten when already set."""
        meta = ReportMetadata(skill_version="custom v1")
        report = _make_report(metadata=meta)
        inject_report_metadata(report)
        assert report.metadata.skill_version == "custom v1"


class TestGKEConfigInjection:
    """Tests for GKE config metadata injection."""

    def test_cluster_name_and_project_id(self) -> None:
        report = _make_report()
        gke = _make_gke_config(cluster_name="prod-cluster", project_id="proj-123")
        inject_report_metadata(report, gke_config=gke)
        assert report.metadata.cluster_name == "prod-cluster"
        assert report.metadata.project_id == "proj-123"

    def test_gke_config_overwrites_hallucinated_values(self) -> None:
        """GKE config values overwrite whatever the LLM wrote."""
        meta = ReportMetadata(
            cluster_name="hallucinated-cluster",
            project_id="hallucinated-project",
        )
        report = _make_report(metadata=meta)
        gke = _make_gke_config(cluster_name="real-cluster", project_id="real-project")
        inject_report_metadata(report, gke_config=gke)
        assert report.metadata.cluster_name == "real-cluster"
        assert report.metadata.project_id == "real-project"

    def test_gke_config_none_clears_to_empty(self) -> None:
        """When gke_config attrs are None, metadata is set to empty string."""
        meta = ReportMetadata(cluster_name="old")
        report = _make_report(metadata=meta)
        gke = SimpleNamespace(
            cluster_name=None,
            project_id=None,
            default_namespace=None,
            trends=None,
        )
        inject_report_metadata(report, gke_config=gke)
        assert report.metadata.cluster_name == ""
        assert report.metadata.project_id == ""


class TestModelUsedInjection:
    """Tests for model_used metadata injection."""

    def test_model_id_used_without_orch_result(self) -> None:
        report = _make_report()
        inject_report_metadata(report, model_id="gemini-2.5-flash")
        assert report.metadata.model_used == "gemini-2.5-flash"

    def test_orch_result_models_used_preferred(self) -> None:
        """models_used from OrchestratorResult takes priority over model_id."""
        report = _make_report()
        orch = _make_orch_result(models_used=["gemini-2.5-pro", "gemini-2.5-pro"])
        inject_report_metadata(report, model_id="gemini-2.5-flash", orch_result=orch)
        assert "gemini-2.5-pro" in report.metadata.model_used

    def test_empty_models_used_falls_back_to_model_id(self) -> None:
        report = _make_report()
        orch = _make_orch_result(models_used=[])
        inject_report_metadata(report, model_id="gemini-2.5-flash", orch_result=orch)
        assert report.metadata.model_used == "gemini-2.5-flash"

    def test_no_model_info_sets_empty(self) -> None:
        report = _make_report()
        inject_report_metadata(report)
        assert report.metadata.model_used == ""


class TestCostMetricsInjection:
    """Tests for cost metrics injection."""

    def test_cost_metrics_injected(self) -> None:
        report = _make_report()
        orch = _make_orch_result(
            run_cost_usd=0.001234,
            total_usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
        inject_report_metadata(report, orch_result=orch)
        assert report.metadata.cost_metrics is not None
        assert report.metadata.cost_metrics.run_cost_usd == 0.001234
        assert report.metadata.cost_metrics.total_tokens == 150
        assert report.metadata.cost_metrics.estimated_cost == "$0.001234"

    def test_cost_metrics_not_overwritten(self) -> None:
        """Existing cost_metrics should not be overwritten."""
        existing = CostMetrics(run_cost_usd=0.999)
        meta = ReportMetadata(cost_metrics=existing)
        report = _make_report(metadata=meta)
        orch = _make_orch_result(run_cost_usd=0.001)
        inject_report_metadata(report, orch_result=orch)
        assert report.metadata.cost_metrics.run_cost_usd == 0.999

    def test_zero_cost_no_estimated_cost_string(self) -> None:
        report = _make_report()
        orch = _make_orch_result(
            run_cost_usd=0.0,
            total_usage={"total_tokens": 500},
        )
        inject_report_metadata(report, orch_result=orch)
        assert report.metadata.cost_metrics is not None
        assert report.metadata.cost_metrics.estimated_cost is None


class TestToolUsageInjection:
    """Tests for tool usage injection."""

    def test_tool_usage_injected(self) -> None:
        report = _make_report()
        tool_logger = _make_tool_logger(
            tool_name_counts={"kubectl_get": 4, "get_events": 2},
            tool_count=6,
        )
        inject_report_metadata(report, tool_logger=tool_logger)
        assert report.metadata.tool_usage is not None
        assert report.metadata.tool_usage.tool_calls == 6

    def test_tool_usage_not_overwritten(self) -> None:
        existing = ToolUsageSummary(tool_calls=99)
        meta = ReportMetadata(tool_usage=existing)
        report = _make_report(metadata=meta)
        tool_logger = _make_tool_logger(tool_count=5)
        inject_report_metadata(report, tool_logger=tool_logger)
        assert report.metadata.tool_usage.tool_calls == 99

    def test_zero_tool_count_treated_as_none(self) -> None:
        """tool_count of 0 should be treated as None (no calls)."""
        report = _make_report()
        tool_logger = _make_tool_logger(tool_count=0)
        inject_report_metadata(report, tool_logger=tool_logger)
        # No tool_counts and tool_calls is None → tool_usage stays None
        assert report.metadata.tool_usage is None

    def test_pipeline_counts_preferred(self) -> None:
        """pipeline_tool_name_counts should be preferred over tool_name_counts."""
        report = _make_report()
        tool_logger = _make_tool_logger(
            pipeline_tool_name_counts={"kubectl_get": 10},
            tool_name_counts={"kubectl_get": 5},
            tool_count=10,
        )
        inject_report_metadata(report, tool_logger=tool_logger)
        assert report.metadata.tool_usage is not None
        assert report.metadata.tool_usage.tool_counts == {"kubectl_get": 10}


class TestClusterOverviewInjection:
    """Tests for namespace row injection into cluster_overview."""

    def test_namespace_row_injected(self) -> None:
        from vaig.skills.service_health.schema import ClusterMetric

        report = _make_report()
        report.cluster_overview = [
            ClusterMetric(metric="Cluster", value="prod"),
        ]
        gke = _make_gke_config(default_namespace="production")
        inject_report_metadata(report, gke_config=gke)
        metrics = [row.metric for row in report.cluster_overview]
        assert "Namespace" in metrics
        # Namespace should be first
        assert report.cluster_overview[0].metric == "Namespace"
        assert report.cluster_overview[0].value == "production"

    def test_namespace_row_not_duplicated(self) -> None:
        from vaig.skills.service_health.schema import ClusterMetric

        report = _make_report()
        report.cluster_overview = [
            ClusterMetric(metric="Namespace", value="existing"),
            ClusterMetric(metric="Cluster", value="prod"),
        ]
        gke = _make_gke_config(default_namespace="production")
        inject_report_metadata(report, gke_config=gke)
        ns_rows = [r for r in report.cluster_overview if r.metric == "Namespace"]
        assert len(ns_rows) == 1


class TestGKECostEstimation:
    """Tests for GKE cost estimation injection."""

    def test_gke_cost_estimation_failure_sets_unsupported(self) -> None:
        """When cost estimation raises, gke_cost should be set to unsupported."""
        report = _make_report()
        gke = _make_gke_config()

        with patch(
            "vaig.tools.gke.cost_estimation.fetch_workload_costs",
            side_effect=RuntimeError("no metrics"),
        ):
            inject_report_metadata(report, gke_config=gke)

        assert report.metadata.gke_cost is not None
        assert report.metadata.gke_cost.supported is False

    def test_gke_cost_not_overwritten(self) -> None:
        """Existing gke_cost should not be overwritten."""
        from vaig.skills.service_health.schema import GKECostReport

        existing = GKECostReport(supported=True, cluster_type="autopilot")
        meta = ReportMetadata(gke_cost=existing)
        report = _make_report(metadata=meta)
        gke = _make_gke_config()
        inject_report_metadata(report, gke_config=gke)
        assert report.metadata.gke_cost.supported is True

    def test_gke_cost_reraises_system_exceptions(self) -> None:
        """KeyboardInterrupt / SystemExit must not be swallowed."""
        import pytest

        report = _make_report()
        gke = _make_gke_config()

        with patch(
            "vaig.tools.gke.cost_estimation.fetch_workload_costs",
            side_effect=KeyboardInterrupt,
        ):
            with pytest.raises(KeyboardInterrupt):
                inject_report_metadata(report, gke_config=gke)


class TestTrendsDefensiveAccess:
    """Tests for defensive attribute access on gke_config.trends."""

    def test_trends_without_enabled_attr(self) -> None:
        """A trends object missing .enabled should not raise."""
        report = _make_report()
        # trends is an object without an 'enabled' attribute
        gke = _make_gke_config(trends=object())
        with patch(
            "vaig.tools.gke.cost_estimation.fetch_workload_costs",
            side_effect=RuntimeError("skip"),
        ):
            inject_report_metadata(report, gke_config=gke)
        # trends analysis should be skipped — no crash
        assert getattr(report.metadata, "trends", None) is None

    def test_trends_enabled_false_skips_analysis(self) -> None:
        """When trends.enabled is False, trend analysis must be skipped."""
        report = _make_report()
        trends_obj = SimpleNamespace(enabled=False)
        gke = _make_gke_config(trends=trends_obj)
        with patch(
            "vaig.tools.gke.cost_estimation.fetch_workload_costs",
            side_effect=RuntimeError("skip"),
        ):
            inject_report_metadata(report, gke_config=gke)
        assert getattr(report.metadata, "trends", None) is None


class TestFullIntegration:
    """End-to-end test combining all injection paths."""

    def test_all_metadata_injected(self) -> None:
        """Verify all metadata sections are populated in a single call."""
        report = _make_report()
        report.cluster_overview = []
        gke = _make_gke_config()
        orch = _make_orch_result(
            run_cost_usd=0.05,
            total_usage={"prompt_tokens": 1000, "completion_tokens": 500},
            models_used=["gemini-2.5-flash", "gemini-2.5-flash"],
        )
        tool_logger = _make_tool_logger(
            tool_name_counts={"kubectl_get": 3},
            tool_count=3,
        )

        with patch(
            "vaig.tools.gke.cost_estimation.fetch_workload_costs",
            side_effect=RuntimeError("no metrics"),
        ):
            inject_report_metadata(
                report,
                gke_config=gke,
                model_id="gemini-2.5-flash",
                orch_result=orch,
                tool_logger=tool_logger,
            )

        meta = report.metadata
        # Header fields
        assert meta.cluster_name == "my-cluster"
        assert meta.project_id == "my-project"
        assert "gemini-2.5-flash" in meta.model_used
        assert meta.generated_at.endswith("Z")
        assert meta.skill_version.startswith("vaig ")
        # Cost metrics
        assert meta.cost_metrics is not None
        assert meta.cost_metrics.run_cost_usd == 0.05
        # Tool usage
        assert meta.tool_usage is not None
        assert meta.tool_usage.tool_calls == 3
        # GKE cost (failed → unsupported)
        assert meta.gke_cost is not None
        assert meta.gke_cost.supported is False
        # Namespace row
        assert any(
            r.metric == "Namespace" for r in report.cluster_overview
        )

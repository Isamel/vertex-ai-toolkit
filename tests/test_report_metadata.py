"""Tests for vaig.core.report_metadata — shared metadata injection."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from vaig.core.report_metadata import format_models_used, inject_report_metadata
from vaig.skills.service_health.schema import (
    CostMetrics,
    PostHocFieldStatus,
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


# ── AUDIT-07: pipeline_version and model_versions injection ──────────────────


class TestAudit07MetadataInjection:
    """Tests for AUDIT-07 — pipeline_version and model_versions population."""

    def test_pipeline_version_set_to_non_empty(self) -> None:
        """inject_report_metadata must resolve a non-empty pipeline_version."""
        report = _make_report()
        inject_report_metadata(report)
        assert report.metadata.pipeline_version
        assert report.metadata.pipeline_version != "unknown"

    def test_pipeline_version_stable_across_two_calls(self) -> None:
        """Two calls on the same machine yield the same pipeline_version."""
        r1 = _make_report()
        r2 = _make_report()
        inject_report_metadata(r1)
        inject_report_metadata(r2)
        assert r1.metadata.pipeline_version == r2.metadata.pipeline_version

    def test_model_versions_populated_from_orch_result_models_by_agent(self) -> None:
        """model_versions is built from orch_result.models_by_agent when available."""
        report = _make_report()
        orch = _make_orch_result(models_used=["gemini-2.5-pro-002"])
        orch.models_by_agent = {"health_analyzer": "gemini-2.5-pro-002"}
        inject_report_metadata(report, orch_result=orch, model_id="gemini-2.5-pro-002")
        assert report.metadata.model_versions == {"health_analyzer": "gemini-2.5-pro-002"}

    def test_model_versions_fallback_to_model_id_when_no_models_by_agent(self) -> None:
        """model_versions falls back to {'health_analyzer': model_id} when orch lacks models_by_agent."""
        report = _make_report()
        orch = _make_orch_result(models_used=["gemini-2.5-flash"])
        # models_by_agent NOT present on the SimpleNamespace
        inject_report_metadata(report, orch_result=orch, model_id="gemini-2.5-flash")
        assert report.metadata.model_versions == {"health_analyzer": "gemini-2.5-flash"}

    def test_model_versions_populated_without_orch_result(self) -> None:
        """model_versions set to {'health_analyzer': model_id} even without orch_result."""
        report = _make_report()
        inject_report_metadata(report, model_id="gemini-2.5-flash-002")
        assert report.metadata.model_versions == {"health_analyzer": "gemini-2.5-flash-002"}

    def test_model_versions_empty_when_no_model_id(self) -> None:
        """model_versions stays empty when neither orch_result nor model_id are supplied."""
        report = _make_report()
        inject_report_metadata(report)
        assert report.metadata.model_versions == {}

    def test_pipeline_version_fallback_when_git_unavailable(self) -> None:
        """pipeline_version falls back to vaig.__version__ when git is not available."""
        import subprocess  # noqa: PLC0415

        report = _make_report()
        with patch.object(subprocess, "run", side_effect=FileNotFoundError("git not found")):
            inject_report_metadata(report)
        assert report.metadata.pipeline_version
        assert report.metadata.pipeline_version != "unknown"


# ── AUDIT-15: Post-hoc field population telemetry ─────────────────────────────


class TestAudit15PostHocTelemetry:
    """Tests for AUDIT-15 — post-hoc field population telemetry."""

    def _make_full_report(self) -> SimpleNamespace:
        """Return a report-like object with ALL five post-hoc fields populated."""
        from vaig.skills.service_health.schema import (  # noqa: PLC0415
            ChangeEvent,
            EvidenceGap,
            ExternalLinks,
        )

        return SimpleNamespace(
            metadata=ReportMetadata(),
            cluster_overview=None,
            evidence_gaps=[EvidenceGap(source="kubectl_get", reason="empty_result")],
            recent_changes=[
                ChangeEvent(
                    timestamp="2026-04-20T00:00:00Z",
                    type="deployment",
                    description="deploy v1.2",
                    correlation_to_issue="may have caused the latency spike",
                )
            ],
            external_links=ExternalLinks(),
            investigation_coverage="Covered 5 of 5 hypotheses.",
        )

    def test_golden_path_all_populated(self) -> None:
        """Golden-path: all five post-hoc fields populated → all populated=True."""
        report = self._make_full_report()
        inject_report_metadata(report)
        statuses = report.metadata.post_hoc_field_status
        assert len(statuses) == 5  # noqa: PLR2004
        field_map = {s.field_name: s for s in statuses}
        for field_name in ("metadata", "evidence_gaps", "recent_changes", "external_links", "investigation_coverage"):
            assert field_name in field_map, f"Missing status for {field_name!r}"
            assert field_map[field_name].populated is True, f"{field_name!r} should be populated=True"

    def test_skipped_fields_recorded_as_not_populated(self) -> None:
        """When post-hoc fields are absent/empty, status has populated=False."""
        report = _make_report()
        # No evidence_gaps, recent_changes, external_links, investigation_coverage on report
        inject_report_metadata(report)
        statuses = report.metadata.post_hoc_field_status
        field_map = {s.field_name: s for s in statuses}
        # metadata is always populated
        assert field_map["metadata"].populated is True
        # The rest should be skipped (not present on SimpleNamespace-based report)
        for field_name in ("evidence_gaps", "recent_changes", "external_links", "investigation_coverage"):
            assert field_name in field_map, f"Missing status for {field_name!r}"
            assert field_map[field_name].populated is False

    def test_empty_list_fields_recorded_as_not_populated(self) -> None:
        """evidence_gaps=[] and recent_changes=[] are treated as not populated."""
        report = SimpleNamespace(
            metadata=ReportMetadata(),
            cluster_overview=None,
            evidence_gaps=[],
            recent_changes=[],
            external_links=None,
            investigation_coverage=None,
        )
        inject_report_metadata(report)
        field_map = {s.field_name: s for s in report.metadata.post_hoc_field_status}
        assert field_map["evidence_gaps"].populated is False
        assert field_map["recent_changes"].populated is False

    def test_exception_captured_in_status(self) -> None:
        """When recording post-hoc status for external_links raises, it is captured gracefully."""
        # Simulate a report where accessing external_links raises RuntimeError
        class _ErrReport:
            metadata = ReportMetadata()
            cluster_overview = None
            evidence_gaps: list[Any] = []
            recent_changes: list[Any] = []
            investigation_coverage = None

            @property
            def external_links(self) -> None:  # type: ignore[return]
                raise RuntimeError("link builder failed: permission denied")

        report = _ErrReport()
        # Should not raise; the error must be swallowed gracefully
        inject_report_metadata(report)
        # external_links access raised → the status entry should reflect that
        field_map = {s.field_name: s for s in report.metadata.post_hoc_field_status}
        assert "external_links" in field_map
        entry = field_map["external_links"]
        assert entry.populated is False
        assert entry.reason is not None
        assert "error:" in entry.reason
        assert len(entry.reason) <= 166  # noqa: PLR2004  # "error:" + 160 chars

    def test_post_hoc_field_status_is_list_of_correct_type(self) -> None:
        """post_hoc_field_status entries are PostHocFieldStatus instances."""
        report = _make_report()
        inject_report_metadata(report)
        for entry in report.metadata.post_hoc_field_status:
            assert isinstance(entry, PostHocFieldStatus)
            assert isinstance(entry.field_name, str)
            assert isinstance(entry.populated, bool)

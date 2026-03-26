"""Tests for schema hardening: extra fields ignored, enums coerced."""

from __future__ import annotations

from vaig.skills.service_health.schema import (
    Confidence,
    DowngradedFinding,
    EvidenceDetail,
    ExecutiveSummary,
    Finding,
    OverallStatus,
    RecommendedAction,
    RootCauseHypothesis,
    ServiceHealthStatus,
    ServiceStatus,
    Severity,
    TimelineEvent,
)


class TestExtraFieldsIgnored:
    """Unknown fields in JSON/dict input are silently ignored (not ValidationError)."""

    def test_executive_summary_ignores_extra(self) -> None:
        es = ExecutiveSummary(
            overall_status=OverallStatus.HEALTHY,
            scope="Cluster-wide",
            summary_text="All good",
            unknown_field_xyz="should be ignored",  # type: ignore[call-arg]
        )
        assert es.overall_status == OverallStatus.HEALTHY
        assert not hasattr(es, "unknown_field_xyz")

    def test_finding_ignores_extra(self) -> None:
        f = Finding(
            id="test-id",
            title="Test Finding",
            severity=Severity.HIGH,
            future_field="ignored",  # type: ignore[call-arg]
        )
        assert f.severity == Severity.HIGH
        assert not hasattr(f, "future_field")

    def test_service_status_ignores_extra(self) -> None:
        svc = ServiceStatus(
            service="my-svc",
            extra_col="ignored",  # type: ignore[call-arg]
        )
        assert svc.service == "my-svc"
        assert not hasattr(svc, "extra_col")

    def test_recommended_action_ignores_extra(self) -> None:
        action = RecommendedAction(
            priority=1,
            title="Fix it",
            new_llm_field="ignored",  # type: ignore[call-arg]
        )
        assert action.priority == 1
        assert not hasattr(action, "new_llm_field")


class TestEnumCoercion:
    """Invalid enum values are coerced to sensible defaults, not rejected."""

    def test_invalid_severity_coerced_to_info(self) -> None:
        f = Finding(id="x", title="X", severity="UNKNOWN_SEVERITY")  # type: ignore[arg-type]
        assert f.severity == Severity.INFO

    def test_severity_case_insensitive(self) -> None:
        f = Finding(id="x", title="X", severity="critical")  # type: ignore[arg-type]
        assert f.severity == Severity.CRITICAL

    def test_severity_mixed_case(self) -> None:
        f = Finding(id="x", title="X", severity="High")  # type: ignore[arg-type]
        assert f.severity == Severity.HIGH

    def test_invalid_overall_status_coerced_to_unknown(self) -> None:
        es = ExecutiveSummary(
            overall_status="BORKED",  # type: ignore[arg-type]
            scope="x",
            summary_text="x",
        )
        assert es.overall_status == OverallStatus.UNKNOWN

    def test_overall_status_case_insensitive(self) -> None:
        es = ExecutiveSummary(
            overall_status="healthy",  # type: ignore[arg-type]
            scope="x",
            summary_text="x",
        )
        assert es.overall_status == OverallStatus.HEALTHY

    def test_invalid_service_health_status_coerced_to_unknown(self) -> None:
        svc = ServiceStatus(service="svc", status="INVALID_STATUS")  # type: ignore[arg-type]
        assert svc.status == ServiceHealthStatus.UNKNOWN

    def test_service_health_status_case_insensitive(self) -> None:
        svc = ServiceStatus(service="svc", status="degraded")  # type: ignore[arg-type]
        assert svc.status == ServiceHealthStatus.DEGRADED

    def test_invalid_confidence_coerced_to_medium(self) -> None:
        f = Finding(id="x", title="X", severity=Severity.LOW, confidence="BOGUS")  # type: ignore[arg-type]
        assert f.confidence == Confidence.MEDIUM

    def test_confidence_case_insensitive(self) -> None:
        f = Finding(id="x", title="X", severity=Severity.LOW, confidence="high")  # type: ignore[arg-type]
        assert f.confidence == Confidence.HIGH

    def test_invalid_downgraded_finding_confidence_coerced(self) -> None:
        df = DowngradedFinding(title="T", original_confidence="NOPE")  # type: ignore[arg-type]
        assert df.original_confidence == Confidence.MEDIUM

    def test_invalid_root_cause_hypothesis_confidence_coerced(self) -> None:
        hyp = RootCauseHypothesis(
            finding_title="T",
            mechanism="some mechanism",
            confidence="NOPE",  # type: ignore[arg-type]
        )
        assert hyp.confidence == Confidence.MEDIUM

    def test_invalid_timeline_event_severity_coerced(self) -> None:
        ev = TimelineEvent(time="5m ago", event="something happened", severity="NOPE")  # type: ignore[arg-type]
        assert ev.severity == Severity.INFO

    def test_invalid_evidence_detail_content_type_coerced(self) -> None:
        ed = EvidenceDetail(title="T", content_type="NOPE")  # type: ignore[arg-type]
        from vaig.skills.service_health.schema import ContentType
        assert ed.content_type == ContentType.TEXT

    def test_content_type_case_insensitive(self) -> None:
        ed = EvidenceDetail(title="T", content_type="YAML")  # type: ignore[arg-type]
        from vaig.skills.service_health.schema import ContentType
        assert ed.content_type == ContentType.YAML

    def test_invalid_urgency_coerced_to_short_term(self) -> None:
        from vaig.skills.service_health.schema import ActionUrgency
        action = RecommendedAction(priority=1, title="Fix", urgency="NOPE")  # type: ignore[arg-type]
        assert action.urgency == ActionUrgency.SHORT_TERM

    def test_invalid_effort_coerced_to_medium(self) -> None:
        from vaig.skills.service_health.schema import Effort
        action = RecommendedAction(priority=1, title="Fix", effort="NOPE")  # type: ignore[arg-type]
        assert action.effort == Effort.MEDIUM


class TestServiceStatusRolloutFields:
    """ServiceStatus optional Argo Rollouts enrichment fields — Block C."""

    def test_rollout_strategy_defaults_to_none(self) -> None:
        """rollout_strategy is None when not provided — backward compatible."""
        svc = ServiceStatus(service="my-svc")
        assert svc.rollout_strategy is None

    def test_rollout_status_defaults_to_none(self) -> None:
        """rollout_status is None when not provided — backward compatible."""
        svc = ServiceStatus(service="my-svc")
        assert svc.rollout_status is None

    def test_hpa_conditions_defaults_to_empty_list(self) -> None:
        """hpa_conditions defaults to [] when not provided — backward compatible."""
        svc = ServiceStatus(service="my-svc")
        assert svc.hpa_conditions == []

    def test_rollout_strategy_canary_accepted(self) -> None:
        svc = ServiceStatus(service="my-svc", rollout_strategy="canary")
        assert svc.rollout_strategy == "canary"

    def test_rollout_strategy_blue_green_accepted(self) -> None:
        svc = ServiceStatus(service="my-svc", rollout_strategy="blue-green")
        assert svc.rollout_strategy == "blue-green"

    def test_rollout_status_accepted(self) -> None:
        for status in ("Healthy", "Progressing", "Paused", "Degraded"):
            svc = ServiceStatus(service="my-svc", rollout_status=status)
            assert svc.rollout_status == status

    def test_hpa_conditions_list_accepted(self) -> None:
        conditions = ["AbleToScale: True", "ScalingActive: False — DesiredReplicas=0"]
        svc = ServiceStatus(service="my-svc", hpa_conditions=conditions)
        assert svc.hpa_conditions == conditions

    def test_all_rollout_fields_together(self) -> None:
        svc = ServiceStatus(
            service="my-svc",
            namespace="prod",
            rollout_strategy="canary",
            rollout_status="Progressing",
            hpa_conditions=["AbleToScale: True"],
        )
        assert svc.rollout_strategy == "canary"
        assert svc.rollout_status == "Progressing"
        assert svc.hpa_conditions == ["AbleToScale: True"]

    def test_backward_compat_old_servicestatus_without_rollout_fields(self) -> None:
        """Existing ServiceStatus data without rollout fields must still validate cleanly."""
        data = {
            "service": "payment-svc",
            "namespace": "default",
            "status": "HEALTHY",
            "pods_ready": "3/3",
            "restarts_1h": "0",
            "cpu_usage": "120m",
            "memory_usage": "256Mi",
            "issues": "",
        }
        svc = ServiceStatus(**data)
        assert svc.service == "payment-svc"
        assert svc.rollout_strategy is None
        assert svc.rollout_status is None
        assert svc.hpa_conditions == []

    def test_extra_field_still_ignored_alongside_rollout_fields(self) -> None:
        """extra='ignore' still applies even when rollout fields are present."""
        svc = ServiceStatus(
            service="my-svc",
            rollout_strategy="canary",
            future_unknown_field="should be ignored",  # type: ignore[call-arg]
        )
        assert svc.rollout_strategy == "canary"
        assert not hasattr(svc, "future_unknown_field")


class TestRolloutDetailsTableRendering:
    """Verify that _render_service_status() produces the correct Rollout Details table."""

    @staticmethod
    def _make_report(services):
        from vaig.skills.service_health.schema import HealthReport
        return HealthReport(
            cluster_name="test-cluster",
            namespaces=["default"],
            service_statuses=services,
        )

    def _svc(self, **kwargs):
        from vaig.skills.service_health.schema import ServiceStatus
        defaults = {
            "service": "my-svc",
            "namespace": "default",
            "status": "HEALTHY",
            "pods_ready": "1/1",
            "restarts_1h": "0",
            "cpu_usage": "100m",
            "memory_usage": "128Mi",
            "issues": "",
        }
        defaults.update(kwargs)
        return ServiceStatus(**defaults)

    def test_rollout_details_table_includes_namespace_column(self) -> None:
        """The Rollout Details table header must include a Namespace column."""
        svc = self._svc(rollout_strategy="canary", namespace="prod")
        md = self._make_report([svc]).to_markdown()
        assert "| Service | Namespace |" in md, "Rollout Details table must include Namespace column."

    def test_rollout_details_table_shows_na_for_empty_namespace(self) -> None:
        """Namespace cell must show '—' when namespace is empty/None."""
        svc = self._svc(rollout_strategy="canary", namespace="")
        md = self._make_report([svc]).to_markdown()
        assert "| my-svc | — |" in md, "Empty namespace must render as '—' in the table."

    def test_rollout_details_table_shows_na_for_none_rollout_strategy(self) -> None:
        """Strategy cell must show 'N/A' when rollout_strategy is None but another field is set."""
        svc = self._svc(rollout_strategy=None, rollout_status="Healthy")
        md = self._make_report([svc]).to_markdown()
        assert "N/A" in md, "None rollout_strategy must render as 'N/A'."

    def test_filter_includes_service_with_only_rollout_status(self) -> None:
        """A service with only rollout_status set must still appear in the Rollout Details table."""
        svc = self._svc(rollout_strategy=None, rollout_status="Progressing")
        md = self._make_report([svc]).to_markdown()
        assert "### Rollout Details" in md, (
            "Rollout Details table must appear when rollout_status is set even if strategy is None."
        )

    def test_filter_includes_service_with_only_hpa_conditions(self) -> None:
        """A service with only hpa_conditions set must still appear in the Rollout Details table."""
        svc = self._svc(rollout_strategy=None, rollout_status=None, hpa_conditions=["ScalingLimited: True"])
        md = self._make_report([svc]).to_markdown()
        assert "### Rollout Details" in md, (
            "Rollout Details table must appear when hpa_conditions is set even if strategy is None."
        )

    def test_no_rollout_details_section_when_all_fields_empty(self) -> None:
        """Services with no rollout data must NOT produce a Rollout Details section."""
        svc = self._svc()
        md = self._make_report([svc]).to_markdown()
        assert "### Rollout Details" not in md, (
            "Rollout Details table must NOT appear when no service has rollout data."
        )

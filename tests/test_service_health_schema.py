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

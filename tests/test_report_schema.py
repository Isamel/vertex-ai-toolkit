"""Comprehensive tests for the HealthReport schema models.

Covers:
- Model creation and field defaults
- Enum validation (StrEnum correctness)
- to_markdown() rendering — all sections
- to_dict() serialisation
- model_json_schema() compatibility check
- Edge cases: empty findings, missing optionals, single-item lists
"""

from __future__ import annotations

import json
from typing import Any

from vaig.skills.service_health.schema import (
    CONTENT_TYPE_FENCE_MAP,
    ActionUrgency,
    ClusterMetric,
    Confidence,
    ContentType,
    DowngradedFinding,
    Effort,
    EvidenceDetail,
    ExecutiveSummary,
    Finding,
    HealthReport,
    ManualInvestigation,
    OverallStatus,
    RecommendedAction,
    ReportMetadata,
    RootCauseHypothesis,
    ServiceHealthStatus,
    ServiceStatus,
    Severity,
    TimelineEvent,
    _collapse_repeated_events,
    _CollapsedEvent,
    _normalize_event_text,
)

# ── Fixtures / helpers ───────────────────────────────────────


def _minimal_report() -> HealthReport:
    """Smallest valid HealthReport — only required fields."""
    return HealthReport(
        executive_summary=ExecutiveSummary(
            overall_status=OverallStatus.HEALTHY,
            scope="Cluster-wide",
            summary_text="All services are healthy.",
        ),
    )


def _full_report() -> HealthReport:
    """A realistic report with all sections populated."""
    return HealthReport(
        executive_summary=ExecutiveSummary(
            overall_status=OverallStatus.DEGRADED,
            scope="Namespace: production",
            summary_text="2 critical issues found affecting payment service.",
            services_checked=5,
            issues_found=3,
            critical_count=2,
            warning_count=1,
        ),
        cluster_overview=[
            ClusterMetric(metric="Total Pods", value="42"),
            ClusterMetric(metric="Healthy", value="38 (90%)"),
            ClusterMetric(metric="Degraded", value="3 (7%)"),
            ClusterMetric(metric="Failed", value="1 (2%)"),
            ClusterMetric(metric="Total Deployments", value="8"),
            ClusterMetric(metric="Fully Available", value="6"),
        ],
        service_statuses=[
            ServiceStatus(
                service="payment-svc",
                namespace="production",
                status=ServiceHealthStatus.DEGRADED,
                pods_ready="2/3",
                restarts_1h="5",
                cpu_usage="78%",
                memory_usage="85%",
                issues="CrashLoopBackOff on 1 pod",
            ),
            ServiceStatus(
                service="api-gateway",
                namespace="production",
                status=ServiceHealthStatus.HEALTHY,
                pods_ready="3/3",
                restarts_1h="0",
            ),
        ],
        findings=[
            Finding(
                id="crashloop-payment-svc",
                title="CrashLoopBackOff on payment-svc",
                severity=Severity.CRITICAL,
                category="pod-health",
                service="payment-svc",
                description="Pod payment-svc-abc123 is in CrashLoopBackOff.",
                root_cause="OOMKilled due to memory limit of 256Mi being exceeded.",
                evidence=[
                    "kubectl get pods: payment-svc-abc123 CrashLoopBackOff",
                    "Last terminated: OOMKilled, exit code 137",
                ],
                confidence=Confidence.CONFIRMED,
                impact="Payment processing is degraded, 33% capacity loss.",
                affected_resources=["production/pod/payment-svc-abc123"],
                remediation="Increase memory limit to 512Mi.",
            ),
            Finding(
                id="hpa-exhausted-payment",
                title="HPA at maximum replicas",
                severity=Severity.HIGH,
                category="scaling",
                service="payment-svc",
                description="HPA has scaled to max replicas (5/5) and CPU still at 92%.",
                evidence=["kubectl get hpa: payment-svc 5/5 92%/80%"],
                confidence=Confidence.HIGH,
                impact="Next traffic spike will cause latency degradation.",
                affected_resources=["production/hpa/payment-svc"],
            ),
            Finding(
                id="elevated-restarts-cart",
                title="Elevated restart count on cart-svc",
                severity=Severity.MEDIUM,
                category="pod-health",
                service="cart-svc",
                description="cart-svc has 4 restarts in the last hour.",
                evidence=["kubectl get pods: cart-svc-xyz789 restarts=4"],
                confidence=Confidence.MEDIUM,
                impact="Service is recovering but instability may worsen.",
            ),
            Finding(
                id="dns-warning-transient",
                title="Transient DNS resolution warning",
                severity=Severity.LOW,
                description="Single DNS resolution timeout for external-api.example.com.",
                evidence=["kube-dns log: timeout resolving external-api.example.com"],
            ),
            Finding(
                id="rollout-success-api",
                title="Successful rollout of api-gateway v2.3.1",
                severity=Severity.INFO,
                description="Rolling update completed successfully.",
                evidence=["kubectl rollout status: deployment/api-gateway successfully rolled out"],
            ),
        ],
        downgraded_findings=[
            DowngradedFinding(
                title="Memory pressure on node-pool-1",
                original_confidence=Confidence.HIGH,
                final_confidence=Confidence.LOW,
                reason="Verification showed memory usage is at 65%, within normal range.",
            ),
        ],
        root_cause_hypotheses=[
            RootCauseHypothesis(
                finding_title="CrashLoopBackOff on payment-svc",
                mechanism=(
                    "Payment service v2.1.0 introduced an in-memory cache that grows "
                    "unbounded. Under production load, the cache exceeds the 256Mi container "
                    "limit within 15 minutes, triggering OOMKill."
                ),
                confidence=Confidence.CONFIRMED,
                supporting_evidence=[
                    "Memory usage graph shows linear growth from pod start",
                    "Previous version v2.0.9 ran stable at ~180Mi",
                ],
                what_would_confirm="N/A",
            ),
        ],
        evidence_details=[
            EvidenceDetail(
                title="OOMKilled container state",
                description="Container last state from kubectl describe:",
                evidence_text=(
                    "Last State: Terminated\n"
                    "  Reason: OOMKilled\n"
                    "  Exit Code: 137\n"
                    "  Started: 2024-01-15T10:30:00Z\n"
                    "  Finished: 2024-01-15T10:45:12Z"
                ),
                content_type=ContentType.YAML,
            ),
        ],
        recommendations=[
            RecommendedAction(
                priority=1,
                title="Increase payment-svc memory limit",
                description="Raise memory limit from 256Mi to 512Mi.",
                urgency=ActionUrgency.IMMEDIATE,
                effort=Effort.LOW,
                command="kubectl set resources deployment/payment-svc -n production --limits=memory=512Mi",
                why="Pod is OOMKilled repeatedly, causing service degradation.",
                risk="low",
                related_findings=["crashloop-payment-svc"],
            ),
            RecommendedAction(
                priority=2,
                title="Review HPA scaling policy",
                description="Consider increasing maxReplicas or optimising CPU usage.",
                urgency=ActionUrgency.SHORT_TERM,
                effort=Effort.MEDIUM,
                command="kubectl patch hpa payment-svc -n production -p '{\"spec\":{\"maxReplicas\":10}}'",
                why="HPA is at ceiling with CPU still above target.",
                related_findings=["hpa-exhausted-payment"],
            ),
            RecommendedAction(
                priority=3,
                title="Implement bounded cache with eviction",
                urgency=ActionUrgency.LONG_TERM,
                effort=Effort.HIGH,
                why="Root cause is unbounded memory growth in the cache layer.",
                related_findings=["crashloop-payment-svc"],
            ),
        ],
        manual_investigations=[
            ManualInvestigation(
                finding_title="Intermittent latency spikes",
                reason="gcloud_monitoring_query timed out",
                investigation_steps="Run gcloud monitoring time-series list manually.",
            ),
        ],
        timeline=[
            TimelineEvent(
                time="57m ago", event="payment-svc-abc123 OOMKilled",
                severity=Severity.CRITICAL, service="payment-svc",
            ),
            TimelineEvent(
                time="45m ago", event="HPA scaled payment-svc to 5/5",
                severity=Severity.HIGH, service="payment-svc",
            ),
            TimelineEvent(
                time="30m ago", event="cart-svc-xyz789 restarted (4th time)",
                severity=Severity.MEDIUM, service="cart-svc",
            ),
            TimelineEvent(
                time="15m ago", event="api-gateway v2.3.1 rollout complete",
                severity=Severity.INFO, service="api-gateway",
            ),
        ],
        metadata=ReportMetadata(
            generated_at="2024-01-15T11:00:00Z",
            cluster_name="prod-us-central1",
            project_id="my-gcp-project",
            model_used="gemini-2.5-pro",
            skill_version="0.4.0",
        ),
    )


# ══════════════════════════════════════════════════════════════
# Enum tests
# ══════════════════════════════════════════════════════════════


class TestEnums:
    """StrEnum fields serialise as plain strings."""

    def test_overall_status_values(self) -> None:
        assert list(OverallStatus) == ["HEALTHY", "DEGRADED", "CRITICAL", "UNKNOWN"]

    def test_severity_values(self) -> None:
        assert list(Severity) == ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]

    def test_confidence_values(self) -> None:
        assert list(Confidence) == ["CONFIRMED", "HIGH", "MEDIUM", "LOW"]

    def test_effort_values(self) -> None:
        assert list(Effort) == ["LOW", "MEDIUM", "HIGH"]

    def test_action_urgency_values(self) -> None:
        assert list(ActionUrgency) == ["IMMEDIATE", "SHORT_TERM", "LONG_TERM"]

    def test_service_health_status_values(self) -> None:
        assert list(ServiceHealthStatus) == ["HEALTHY", "DEGRADED", "FAILED", "UNKNOWN"]

    def test_strenum_is_str(self) -> None:
        """StrEnum members must be usable as plain strings."""
        assert isinstance(Severity.CRITICAL, str)
        assert Severity.CRITICAL == "CRITICAL"
        assert f"level={Severity.HIGH}" == "level=HIGH"

    def test_enum_json_serialisation(self) -> None:
        """Enum values serialise as strings in JSON, not objects."""
        f = Finding(id="test", title="Test", severity=Severity.HIGH)
        d = f.model_dump()
        assert d["severity"] == "HIGH"
        assert isinstance(d["severity"], str)


# ══════════════════════════════════════════════════════════════
# Model creation & defaults
# ══════════════════════════════════════════════════════════════


class TestModelCreation:
    """Models can be created with minimal and full data."""

    def test_minimal_report(self) -> None:
        report = _minimal_report()
        assert report.executive_summary.overall_status == OverallStatus.HEALTHY
        assert report.findings == []
        assert report.recommendations == []
        assert report.timeline == []

    def test_full_report(self) -> None:
        report = _full_report()
        assert report.executive_summary.overall_status == OverallStatus.DEGRADED
        assert len(report.findings) == 5
        assert len(report.recommendations) == 3
        assert len(report.timeline) == 4

    def test_executive_summary_defaults(self) -> None:
        es = ExecutiveSummary(
            overall_status=OverallStatus.HEALTHY,
            scope="Cluster-wide",
            summary_text="OK",
        )
        assert es.services_checked == 0
        assert es.issues_found == 0
        assert es.critical_count == 0
        assert es.warning_count == 0

    def test_finding_defaults(self) -> None:
        f = Finding(id="f1", title="Test", severity=Severity.INFO)
        assert f.category == ""
        assert f.service == ""
        assert f.description == ""
        assert f.evidence == []
        assert f.confidence == Confidence.MEDIUM
        assert f.impact == ""
        assert f.affected_resources == []
        assert f.remediation is None

    def test_recommended_action_defaults(self) -> None:
        a = RecommendedAction(priority=1, title="Do thing")
        assert a.urgency == ActionUrgency.SHORT_TERM
        assert a.effort == Effort.MEDIUM
        assert a.command == ""
        assert a.expected_output == ""
        assert a.interpretation == ""
        assert a.related_findings == []

    def test_recommended_action_new_fields_roundtrip(self) -> None:
        a = RecommendedAction(
            priority=1,
            title="Restart pods",
            expected_output="deployment.apps/nginx restarted",
            interpretation="All pods should show Running within 30s.",
        )
        data = a.model_dump()
        restored = RecommendedAction.model_validate(data)
        assert restored.expected_output == "deployment.apps/nginx restarted"
        assert restored.interpretation == "All pods should show Running within 30s."

    def test_old_json_without_new_fields(self) -> None:
        """Old JSON payloads missing expected_output/interpretation still deserialize."""
        payload = {"priority": 1, "title": "Fix it", "command": "kubectl apply -f fix.yaml"}
        a = RecommendedAction.model_validate(payload)
        assert a.expected_output == ""
        assert a.interpretation == ""
        assert a.command == "kubectl apply -f fix.yaml"

    def test_report_metadata_defaults(self) -> None:
        m = ReportMetadata()
        assert m.generated_at == ""
        assert m.cluster_name == ""
        assert m.project_id == ""

    def test_service_status_defaults(self) -> None:
        s = ServiceStatus(service="my-svc")
        assert s.namespace == ""
        assert s.status == ServiceHealthStatus.UNKNOWN
        assert s.pods_ready == "N/A"
        assert s.restarts_1h == "N/A"
        assert s.cpu_usage == "N/A"
        assert s.memory_usage == "N/A"

    def test_timeline_event_default_severity(self) -> None:
        ev = TimelineEvent(time="5m ago", event="something happened")
        assert ev.severity == Severity.INFO


# ══════════════════════════════════════════════════════════════
# Validation tests
# ══════════════════════════════════════════════════════════════


class TestValidation:
    """Pydantic validation catches invalid data."""

    def test_invalid_severity_rejected(self) -> None:
        # After schema hardening, invalid enum values are coerced to defaults, not rejected.
        finding = Finding(id="x", title="X", severity="INVALID")  # type: ignore[arg-type]
        assert finding.severity == Severity.INFO  # coerced to default

    def test_invalid_overall_status_rejected(self) -> None:
        # After schema hardening, invalid enum values are coerced to defaults, not rejected.
        es = ExecutiveSummary(
            overall_status="BORKED",  # type: ignore[arg-type]
            scope="Cluster-wide",
            summary_text="nope",
        )
        assert es.overall_status == OverallStatus.UNKNOWN  # coerced to default

    def test_negative_priority_rejected(self) -> None:
        import pytest

        with pytest.raises(Exception):  # noqa: B017
            RecommendedAction(priority=0, title="Invalid")

    def test_negative_count_rejected(self) -> None:
        import pytest

        with pytest.raises(Exception):  # noqa: B017
            ExecutiveSummary(
                overall_status=OverallStatus.HEALTHY,
                scope="x",
                summary_text="x",
                services_checked=-1,
            )

    def test_invalid_service_health_status_rejected(self) -> None:
        # After schema hardening, invalid enum values are coerced to defaults, not rejected.
        svc = ServiceStatus(service="svc", status="BORKED")  # type: ignore[arg-type]
        assert svc.status == ServiceHealthStatus.UNKNOWN  # coerced to default

    def test_downgraded_finding_confidence_is_enum(self) -> None:
        df = DowngradedFinding(
            title="Test",
            original_confidence=Confidence.HIGH,
            final_confidence=Confidence.LOW,
            reason="test",
        )
        assert isinstance(df.original_confidence, Confidence)
        assert isinstance(df.final_confidence, Confidence)
        assert df.original_confidence == Confidence.HIGH
        assert df.final_confidence == Confidence.LOW

    def test_downgraded_finding_invalid_confidence_rejected(self) -> None:
        # After schema hardening, invalid enum values are coerced to defaults, not rejected.
        df = DowngradedFinding(
            title="Test",
            original_confidence="INVALID",  # type: ignore[arg-type]
        )
        assert df.original_confidence == Confidence.MEDIUM  # coerced to default


# ══════════════════════════════════════════════════════════════
# to_dict() tests
# ══════════════════════════════════════════════════════════════


class TestToDict:
    """to_dict() delegates to model_dump() correctly."""

    def test_returns_dict(self) -> None:
        report = _minimal_report()
        result = report.to_dict()
        assert isinstance(result, dict)

    def test_matches_model_dump(self) -> None:
        report = _full_report()
        assert report.to_dict() == report.model_dump()

    def test_json_serialisable(self) -> None:
        """to_dict() output must be JSON-serialisable."""
        report = _full_report()
        raw = json.dumps(report.to_dict())
        assert isinstance(raw, str)
        parsed = json.loads(raw)
        assert parsed["executive_summary"]["overall_status"] == "DEGRADED"

    def test_enums_are_strings_in_dict(self) -> None:
        report = _full_report()
        d = report.to_dict()
        assert d["executive_summary"]["overall_status"] == "DEGRADED"
        assert d["findings"][0]["severity"] == "CRITICAL"
        assert d["findings"][0]["confidence"] == "CONFIRMED"
        assert d["recommendations"][0]["urgency"] == "IMMEDIATE"
        assert d["recommendations"][0]["effort"] == "LOW"
        assert d["service_statuses"][0]["status"] == "DEGRADED"
        assert d["downgraded_findings"][0]["original_confidence"] == "HIGH"
        assert d["downgraded_findings"][0]["final_confidence"] == "LOW"


# ══════════════════════════════════════════════════════════════
# model_json_schema() tests
# ══════════════════════════════════════════════════════════════


class TestJsonSchema:
    """model_json_schema() produces a valid schema for Gemini."""

    def test_produces_dict(self) -> None:
        schema = HealthReport.model_json_schema()
        assert isinstance(schema, dict)

    def test_has_required_fields(self) -> None:
        schema = HealthReport.model_json_schema()
        assert "properties" in schema
        assert "executive_summary" in schema["properties"]

    def test_json_serialisable(self) -> None:
        """Schema must be JSON-serialisable (required by Gemini API)."""
        schema = HealthReport.model_json_schema()
        raw = json.dumps(schema)
        assert isinstance(raw, str)

    def test_contains_enum_values(self) -> None:
        """Enum fields should produce 'enum' arrays in the schema."""
        schema = HealthReport.model_json_schema()
        # The schema uses $defs for nested models; find Severity
        defs = schema.get("$defs", {})
        severity_schema = defs.get("Severity", {})
        assert "enum" in severity_schema, f"Severity not found with enum in $defs: {list(defs.keys())}"
        assert set(severity_schema["enum"]) == {"CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"}

    def test_schema_title_is_health_report(self) -> None:
        schema = HealthReport.model_json_schema()
        assert schema.get("title") == "HealthReport"

    def test_all_models_in_defs(self) -> None:
        """All sub-models should appear in $defs."""
        schema = HealthReport.model_json_schema()
        defs = schema.get("$defs", {})
        expected = {
            "ExecutiveSummary",
            "Finding",
            "RecommendedAction",
            "ReportMetadata",
            "Severity",
            "OverallStatus",
            "Confidence",
            "ContentType",
            "Effort",
            "ActionUrgency",
            "ServiceHealthStatus",
            "ClusterMetric",
            "ServiceStatus",
            "DowngradedFinding",
            "RootCauseHypothesis",
            "EvidenceDetail",
            "ManualInvestigation",
            "TimelineEvent",
        }
        actual = set(defs.keys())
        missing = expected - actual
        assert not missing, f"Missing from $defs: {missing}"

    def test_roundtrip_via_json_schema(self) -> None:
        """A report serialised via to_dict() should validate against model_validate()."""
        report = _full_report()
        d = report.to_dict()
        roundtripped = HealthReport.model_validate(d)
        assert roundtripped.executive_summary.overall_status == OverallStatus.DEGRADED
        assert len(roundtripped.findings) == 5


# ══════════════════════════════════════════════════════════════
# to_markdown() tests
# ══════════════════════════════════════════════════════════════


class TestToMarkdown:
    """to_markdown() renders all sections correctly."""

    def test_returns_string(self) -> None:
        report = _minimal_report()
        md = report.to_markdown()
        assert isinstance(md, str)

    def test_starts_with_title(self) -> None:
        report = _minimal_report()
        md = report.to_markdown()
        assert md.startswith("# Service Health Report")

    def test_executive_summary_present(self) -> None:
        report = _full_report()
        md = report.to_markdown()
        assert "## Executive Summary" in md
        assert "- **Status**: DEGRADED" in md
        assert "- **Scope**: Namespace: production" in md
        assert "- **Summary**: 2 critical issues" in md

    def test_cluster_overview_table(self) -> None:
        report = _full_report()
        md = report.to_markdown()
        assert "## Cluster Overview" in md
        assert "| Total Pods | 42 |" in md
        assert "| Healthy | 38 (90%) |" in md

    def test_cluster_overview_empty_shows_fallback(self) -> None:
        report = _minimal_report()
        md = report.to_markdown()
        assert "Cluster overview data was not collected" in md

    def test_service_status_table(self) -> None:
        report = _full_report()
        md = report.to_markdown()
        assert "## Service Status" in md
        assert "payment-svc" in md
        assert "CrashLoopBackOff on 1 pod" in md

    def test_service_status_empty_fallback(self) -> None:
        report = _minimal_report()
        md = report.to_markdown()
        assert "No service status data available." in md

    def test_findings_grouped_by_severity(self) -> None:
        report = _full_report()
        md = report.to_markdown()
        # Verify severity section headers appear
        assert "### 🔴 Critical" in md
        assert "### 🟠 High" in md
        assert "### 🟡 Medium" in md
        assert "### 🔵 Low" in md
        assert "### 🟢 Informational" in md

    def test_critical_finding_has_structured_fields(self) -> None:
        report = _full_report()
        md = report.to_markdown()
        assert "#### CrashLoopBackOff on payment-svc" in md
        assert "- **What**: Pod payment-svc-abc123" in md
        assert "- **Root Cause**: OOMKilled" in md
        assert "- **Confidence**: CONFIRMED" in md
        assert "- **Impact**: Payment processing" in md
        assert "- **Affected Resources**: production/pod/payment-svc-abc123" in md

    def test_low_finding_uses_bullet_format(self) -> None:
        report = _full_report()
        md = report.to_markdown()
        assert "- **Transient DNS resolution warning**:" in md

    def test_info_finding_uses_bullet_format(self) -> None:
        report = _full_report()
        md = report.to_markdown()
        assert "- **Successful rollout of api-gateway v2.3.1**:" in md

    def test_findings_empty_fallback(self) -> None:
        report = _minimal_report()
        md = report.to_markdown()
        assert "No findings to report." in md

    def test_downgraded_findings_table(self) -> None:
        report = _full_report()
        md = report.to_markdown()
        assert "## Downgraded Findings" in md
        assert "Memory pressure on node-pool-1" in md
        assert "HIGH" in md
        assert "LOW" in md

    def test_downgraded_findings_empty_message(self) -> None:
        report = _minimal_report()
        md = report.to_markdown()
        assert "No findings were downgraded during verification" in md

    def test_root_cause_hypotheses(self) -> None:
        report = _full_report()
        md = report.to_markdown()
        assert "## Root Cause Hypotheses" in md
        assert "#### CrashLoopBackOff on payment-svc" in md
        assert "- **Mechanism**:" in md
        assert "in-memory cache" in md

    def test_root_cause_empty_fallback(self) -> None:
        report = _minimal_report()
        md = report.to_markdown()
        assert "No root cause hypotheses to report." in md

    def test_evidence_details_section(self) -> None:
        report = _full_report()
        md = report.to_markdown()
        assert "## Evidence Details" in md
        assert "### OOMKilled container state" in md
        assert "Exit Code: 137" in md

    def test_evidence_details_omitted_when_empty(self) -> None:
        report = _minimal_report()
        md = report.to_markdown()
        assert "## Evidence Details" not in md

    def test_recommendations_grouped_by_urgency(self) -> None:
        report = _full_report()
        md = report.to_markdown()
        assert "### Immediate (next 5 minutes)" in md
        assert "### Short-term (next 1 hour)" in md
        assert "### Long-term (next sprint)" in md

    def test_recommendations_include_commands(self) -> None:
        report = _full_report()
        md = report.to_markdown()
        assert "kubectl set resources deployment/payment-svc" in md
        assert "- Why:" in md
        assert "- Risk:" in md

    def test_recommendations_empty_fallback(self) -> None:
        report = _minimal_report()
        md = report.to_markdown()
        assert "No recommended actions at this time." in md

    def test_recommendations_include_new_fields(self) -> None:
        """expected_output and interpretation appear in markdown when populated."""
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.DEGRADED,
                scope="Cluster-wide",
                summary_text="Issue.",
            ),
            recommendations=[
                RecommendedAction(
                    priority=1,
                    title="Fix memory",
                    urgency=ActionUrgency.IMMEDIATE,
                    command="kubectl set resources deploy/app --limits=memory=512Mi",
                    expected_output="deployment.apps/app resource requirements updated",
                    interpretation="Verify pods restart with new limits within 60s.",
                    why="OOMKilled",
                    risk="Rolling restart",
                ),
            ],
        )
        md = report.to_markdown()
        assert "- Expected output:" in md
        assert "deployment.apps/app resource requirements updated" in md
        assert "- Interpretation: Verify pods restart with new limits within 60s." in md

    def test_recommendations_omit_empty_new_fields(self) -> None:
        """Empty expected_output/interpretation produce no labels in markdown."""
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.DEGRADED,
                scope="Cluster-wide",
                summary_text="Issue.",
            ),
            recommendations=[
                RecommendedAction(
                    priority=1,
                    title="Fix memory",
                    urgency=ActionUrgency.IMMEDIATE,
                    command="kubectl apply -f fix.yaml",
                ),
            ],
        )
        md = report.to_markdown()
        assert "Expected output" not in md
        assert "Interpretation" not in md

    def test_manual_investigation_section(self) -> None:
        report = _full_report()
        md = report.to_markdown()
        assert "### Manual Investigation Required" in md
        assert "Intermittent latency spikes" in md

    def test_manual_investigation_omitted_when_empty(self) -> None:
        report = _minimal_report()
        md = report.to_markdown()
        assert "### Manual Investigation Required" not in md

    def test_timeline_table(self) -> None:
        report = _full_report()
        md = report.to_markdown()
        assert "## Timeline" in md
        assert "| 57m ago | payment-svc-abc123 OOMKilled | CRITICAL |" in md
        assert "| 15m ago | api-gateway v2.3.1 rollout complete | INFO |" in md

    def test_timeline_empty_fallback(self) -> None:
        report = _minimal_report()
        md = report.to_markdown()
        assert "No timeline events available." in md

    def test_section_order(self) -> None:
        """Sections must appear in the mandated order."""
        report = _full_report()
        md = report.to_markdown()
        sections = [
            "# Service Health Report",
            "## Executive Summary",
            "## Cluster Overview",
            "## Service Status",
            "## Findings",
            "## Downgraded Findings",
            "## Root Cause Hypotheses",
            "## Evidence Details",
            "## Recommended Actions",
            "### Manual Investigation Required",
            "## Timeline",
        ]
        positions = [md.index(s) for s in sections]
        assert positions == sorted(positions), "Sections are not in the mandated order"


# ══════════════════════════════════════════════════════════════
# Edge cases
# ══════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_finding(self) -> None:
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.CRITICAL,
                scope="Resource: deployment/broken in default",
                summary_text="One critical issue.",
                issues_found=1,
                critical_count=1,
            ),
            findings=[
                Finding(id="single", title="Only Finding", severity=Severity.CRITICAL),
            ],
        )
        md = report.to_markdown()
        assert "### 🔴 Critical" in md
        assert "#### Only Finding" in md
        # Other severity sections should NOT appear
        assert "### 🟠 High" not in md
        assert "### 🟡 Medium" not in md

    def test_only_info_findings(self) -> None:
        """Report with only INFO findings uses bullet format."""
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.HEALTHY,
                scope="Cluster-wide",
                summary_text="All clear.",
            ),
            findings=[
                Finding(id="i1", title="All pods healthy", severity=Severity.INFO),
                Finding(id="i2", title="Rollout complete", severity=Severity.INFO),
            ],
        )
        md = report.to_markdown()
        assert "### 🟢 Informational" in md
        assert "- **All pods healthy**" in md
        assert "#### All pods healthy" not in md  # Should NOT use heading format

    def test_finding_without_optional_fields(self) -> None:
        """Finding with only required fields still renders."""
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.DEGRADED,
                scope="Cluster-wide",
                summary_text="Issue.",
            ),
            findings=[
                Finding(id="sparse", title="Sparse Finding", severity=Severity.HIGH),
            ],
        )
        md = report.to_markdown()
        assert "#### Sparse Finding" in md
        assert "- **Confidence**: MEDIUM" in md  # default
        # Optional fields with empty values should not render
        assert "- **What**: \n" not in md
        assert "- **Root Cause**: \n" not in md

    def test_evidence_detail_with_corrected_text(self) -> None:
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.DEGRADED,
                scope="Cluster-wide",
                summary_text="Issue.",
            ),
            evidence_details=[
                EvidenceDetail(
                    title="Duplicate volume",
                    evidence_text="volumes:\n  - name: vol\n  - name: vol",
                    corrected_text="volumes:\n  - name: vol",
                ),
            ],
        )
        md = report.to_markdown()
        assert "**Corrected**:" in md
        assert "volumes:\n  - name: vol" in md

    def test_model_validate_json_roundtrip(self) -> None:
        """JSON string → model → JSON string roundtrip."""
        report = _full_report()
        json_str = report.model_dump_json()
        restored = HealthReport.model_validate_json(json_str)
        assert restored.executive_summary.overall_status == report.executive_summary.overall_status
        assert len(restored.findings) == len(report.findings)
        assert restored.to_markdown() == report.to_markdown()

    def test_recommendation_without_command(self) -> None:
        """Recommendation without a command still renders cleanly."""
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.DEGRADED,
                scope="Cluster-wide",
                summary_text="Issue.",
            ),
            recommendations=[
                RecommendedAction(
                    priority=1,
                    title="Review architecture",
                    urgency=ActionUrgency.LONG_TERM,
                    effort=Effort.HIGH,
                    why="Systemic issue requires design review.",
                ),
            ],
        )
        md = report.to_markdown()
        assert "1. Review architecture" in md
        assert "   ```" not in md  # No code block for empty command

    def test_multiple_findings_same_severity(self) -> None:
        """Multiple findings at the same severity level all render."""
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.CRITICAL,
                scope="Cluster-wide",
                summary_text="Multiple critical issues.",
            ),
            findings=[
                Finding(id="c1", title="Critical One", severity=Severity.CRITICAL),
                Finding(id="c2", title="Critical Two", severity=Severity.CRITICAL),
                Finding(id="c3", title="Critical Three", severity=Severity.CRITICAL),
            ],
        )
        md = report.to_markdown()
        assert "#### Critical One" in md
        assert "#### Critical Two" in md
        assert "#### Critical Three" in md


# ── Task 6.4: Integration tests for structured reporter output ──


class TestServiceHealthSkillSchemaIntegration:
    """Integration tests: get_agents_config schema fields + post_process_report."""

    def test_reporter_config_has_response_schema(self) -> None:
        """The health_reporter agent config must include response_schema=HealthReport."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        configs = skill.get_agents_config()
        reporter = next(c for c in configs if c["name"] == "health_reporter")
        assert reporter["response_schema"] is HealthReport

    def test_reporter_config_has_response_mime_type(self) -> None:
        """The health_reporter agent config must include response_mime_type."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        configs = skill.get_agents_config()
        reporter = next(c for c in configs if c["name"] == "health_reporter")
        assert reporter["response_mime_type"] == "application/json"

    def test_non_reporter_agents_have_no_schema(self) -> None:
        """Non-reporter agents must NOT have response_schema or response_mime_type."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        configs = skill.get_sequential_agents_config()
        non_reporters = [c for c in configs if c["name"] != "health_reporter"]
        assert len(non_reporters) == 3, "Should have 3 non-reporter agents"
        for cfg in non_reporters:
            assert "response_schema" not in cfg, f"{cfg['name']} should not have response_schema"
            assert "response_mime_type" not in cfg, f"{cfg['name']} should not have response_mime_type"

    def test_post_process_report_valid_json(self) -> None:
        """post_process_report must parse valid JSON and return Markdown."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        report = _minimal_report()
        json_str = report.model_dump_json()

        skill = ServiceHealthSkill()
        result = skill.post_process_report(json_str)

        assert "# Service Health Report" in result
        assert "## Executive Summary" in result
        assert "HEALTHY" in result

    def test_post_process_report_invalid_json_returns_raw(self) -> None:
        """post_process_report must include a warning and raw content on invalid JSON."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        raw = "This is not JSON — it's plain Markdown output"
        result = skill.post_process_report(raw)
        assert "⚠️ Report parsing failed" in result
        assert raw in result

    def test_post_process_report_malformed_json_returns_raw(self) -> None:
        """post_process_report must handle malformed JSON gracefully."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        result = skill.post_process_report('{"executive_summary": "not an object"}')
        # Should contain raw because executive_summary should be an object, not a string
        assert "not an object" in result

    def test_base_skill_post_process_is_noop(self) -> None:
        """BaseSkill.post_process_report must be a no-op (identity function)."""
        from vaig.skills.base import BaseSkill

        class DummySkill(BaseSkill):
            def get_metadata(self):
                pass  # type: ignore[override]
            def get_system_instruction(self) -> str:
                return ""
            def get_phase_prompt(self, phase, context, user_input) -> str:  # type: ignore[override]
                return ""
            def get_agents_config(self, **kwargs: Any) -> list:
                return []

        skill = DummySkill()
        assert skill.post_process_report("unchanged") == "unchanged"


class TestOrchestratorPostProcessWiring:
    """Verify the orchestrator calls skill.post_process_report on reporter output."""

    def test_orchestrator_has_post_process_call_sync(self) -> None:
        """The sync _execute_with_tools_impl path must call post_process_report."""
        import inspect

        from vaig.agents.orchestrator import Orchestrator

        source = inspect.getsource(Orchestrator._execute_with_tools_impl)
        assert "post_process_report" in source

    def test_orchestrator_has_post_process_call_async(self) -> None:
        """The async _async_execute_with_tools_impl path must call post_process_report."""
        import inspect

        from vaig.agents.orchestrator import Orchestrator

        source = inspect.getsource(Orchestrator._async_execute_with_tools_impl)
        assert "post_process_report" in source


# ── Task 6.5: E2E round-trip test (JSON → Markdown) ──


class TestJsonToMarkdownRoundTrip:
    """End-to-end: HealthReport JSON → model_validate_json → to_markdown."""

    def test_full_report_roundtrip(self) -> None:
        """A full report JSON round-trips through validation and renders complete Markdown."""
        report = _full_report()
        json_str = report.model_dump_json()

        # Simulate what post_process_report does
        restored = HealthReport.model_validate_json(json_str)
        md = restored.to_markdown()

        # Verify all major sections present
        assert "# Service Health Report" in md
        assert "## Executive Summary" in md
        assert "## Cluster Overview" in md
        assert "## Service Status" in md
        assert "## Findings" in md
        assert "## Downgraded Findings" in md
        assert "## Root Cause Hypotheses" in md
        assert "## Recommended Actions" in md
        assert "## Timeline" in md

    def test_full_report_via_skill_post_process(self) -> None:
        """A full report JSON processed through ServiceHealthSkill.post_process_report."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        report = _full_report()
        json_str = report.model_dump_json()

        skill = ServiceHealthSkill()
        md = skill.post_process_report(json_str)

        # Should be Markdown, not JSON
        assert md.startswith("# Service Health Report")
        assert "{" not in md.split("\n")[0]  # First line is NOT JSON

    def test_minimal_report_roundtrip(self) -> None:
        """A minimal report with only required fields round-trips correctly."""
        report = _minimal_report()
        json_str = report.model_dump_json()

        restored = HealthReport.model_validate_json(json_str)
        md = restored.to_markdown()

        assert "# Service Health Report" in md
        assert "HEALTHY" in md
        assert "No findings to report." in md

    def test_findings_severity_ordering_preserved(self) -> None:
        """Findings must be grouped by severity order in the Markdown output."""
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.CRITICAL,
                scope="Namespace: production",
                summary_text="Multiple issues.",
            ),
            findings=[
                Finding(id="low-1", title="Low Issue", severity=Severity.LOW),
                Finding(id="crit-1", title="Critical Issue", severity=Severity.CRITICAL),
                Finding(id="high-1", title="High Issue", severity=Severity.HIGH),
                Finding(id="info-1", title="Info Note", severity=Severity.INFO),
                Finding(id="med-1", title="Medium Issue", severity=Severity.MEDIUM),
            ],
        )
        json_str = report.model_dump_json()
        restored = HealthReport.model_validate_json(json_str)
        md = restored.to_markdown()

        # Verify severity sections appear in correct order
        crit_pos = md.find("Critical")
        high_pos = md.find("High Issue")
        med_pos = md.find("Medium Issue")
        low_pos = md.find("Low Issue")
        info_pos = md.find("Info Note")

        assert crit_pos < high_pos < med_pos < low_pos < info_pos

    def test_recommendations_urgency_ordering(self) -> None:
        """Recommendations must be grouped by urgency in the Markdown output."""
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.DEGRADED,
                scope="Cluster-wide",
                summary_text="Needs attention.",
            ),
            recommendations=[
                RecommendedAction(
                    priority=3, title="Plan migration",
                    urgency=ActionUrgency.LONG_TERM,
                ),
                RecommendedAction(
                    priority=1, title="Restart pod",
                    urgency=ActionUrgency.IMMEDIATE,
                    command="kubectl rollout restart deployment/app -n prod",
                ),
                RecommendedAction(
                    priority=2, title="Scale up",
                    urgency=ActionUrgency.SHORT_TERM,
                    command="kubectl scale deployment/app --replicas=5 -n prod",
                ),
            ],
        )
        json_str = report.model_dump_json()
        restored = HealthReport.model_validate_json(json_str)
        md = restored.to_markdown()

        # Verify urgency sections appear in correct order
        imm_pos = md.find("Immediate")
        short_pos = md.find("Short-term")
        long_pos = md.find("Long-term")

        assert imm_pos < short_pos < long_pos
        # Verify commands are present
        assert "kubectl rollout restart" in md
        assert "kubectl scale" in md

    def test_recommendations_sorted_by_priority_within_urgency(self) -> None:
        """Within the same urgency group, actions render in priority order."""
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.DEGRADED,
                scope="Cluster-wide",
                summary_text="Needs attention.",
            ),
            recommendations=[
                RecommendedAction(
                    priority=3, title="Third action",
                    urgency=ActionUrgency.IMMEDIATE,
                ),
                RecommendedAction(
                    priority=1, title="First action",
                    urgency=ActionUrgency.IMMEDIATE,
                ),
                RecommendedAction(
                    priority=2, title="Second action",
                    urgency=ActionUrgency.IMMEDIATE,
                ),
            ],
        )
        md = report.to_markdown()

        # Priority values used as numbering, sorted ascending
        first_pos = md.find("1. First action")
        second_pos = md.find("2. Second action")
        third_pos = md.find("3. Third action")

        assert first_pos != -1, "Expected '1. First action' in output"
        assert second_pos != -1, "Expected '2. Second action' in output"
        assert third_pos != -1, "Expected '3. Third action' in output"
        assert first_pos < second_pos < third_pos


# ══════════════════════════════════════════════════════════════
# Timeline grouping tests (Task 4.3)
# ══════════════════════════════════════════════════════════════


class TestTimelineGrouping:
    """Verify _render_timeline switches between grouped and flat layout."""

    def test_grouped_timeline_when_majority_have_service(self) -> None:
        """When >=50% of events have service, output is grouped by service."""
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.DEGRADED,
                scope="Cluster-wide",
                summary_text="Issues.",
            ),
            timeline=[
                TimelineEvent(time="10m ago", event="pod restarted", severity=Severity.HIGH, service="svc-a"),
                TimelineEvent(time="8m ago", event="OOM killed", severity=Severity.CRITICAL, service="svc-b"),
                TimelineEvent(time="5m ago", event="scaled up", severity=Severity.INFO, service="svc-a"),
            ],
        )
        md = report.to_markdown()
        # Grouped layout uses ### sub-headings per service
        assert "### svc-a" in md
        assert "### svc-b" in md
        # Flat table headers should NOT appear (no Service column)
        assert "| Time | Service |" not in md

    def test_flat_timeline_when_minority_have_service(self) -> None:
        """When <50% of events have service, output is a flat table."""
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.DEGRADED,
                scope="Cluster-wide",
                summary_text="Issues.",
            ),
            timeline=[
                TimelineEvent(time="10m ago", event="cluster upgrade started", severity=Severity.INFO),
                TimelineEvent(time="8m ago", event="node drained", severity=Severity.MEDIUM),
                TimelineEvent(time="5m ago", event="pod restarted", severity=Severity.HIGH, service="svc-a"),
                TimelineEvent(time="3m ago", event="DNS timeout", severity=Severity.LOW),
                TimelineEvent(time="1m ago", event="node ready", severity=Severity.INFO),
            ],
        )
        md = report.to_markdown()
        # Flat table — no ### sub-headings for services
        assert "### svc-a" not in md
        # Should have a Service column since at least one event has service
        assert "| Time | Service | Event | Severity |" in md

    def test_timeline_boundary_50_percent(self) -> None:
        """Exactly 50% with service → grouped (>=50% threshold)."""
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.DEGRADED,
                scope="Cluster-wide",
                summary_text="Issues.",
            ),
            timeline=[
                TimelineEvent(time="10m ago", event="event A", severity=Severity.INFO, service="svc-a"),
                TimelineEvent(time="5m ago", event="event B", severity=Severity.INFO),
            ],
        )
        md = report.to_markdown()
        # 50% = grouped
        assert "### svc-a" in md
        assert "### General" in md

    def test_timeline_empty_service_goes_to_general(self) -> None:
        """Events without service appear under 'General' in grouped mode."""
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.DEGRADED,
                scope="Cluster-wide",
                summary_text="Issues.",
            ),
            timeline=[
                TimelineEvent(time="10m ago", event="svc event", severity=Severity.HIGH, service="svc-a"),
                TimelineEvent(time="8m ago", event="another svc", severity=Severity.HIGH, service="svc-a"),
                TimelineEvent(time="5m ago", event="general event", severity=Severity.INFO),
            ],
        )
        md = report.to_markdown()
        assert "### General" in md
        assert "general event" in md

    def test_timeline_groups_sorted_alphabetically(self) -> None:
        """Service groups are sorted A-Z, with General last."""
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.DEGRADED,
                scope="Cluster-wide",
                summary_text="Issues.",
            ),
            timeline=[
                TimelineEvent(time="10m ago", event="z event", severity=Severity.INFO, service="zulu-svc"),
                TimelineEvent(time="8m ago", event="a event", severity=Severity.INFO, service="alpha-svc"),
                TimelineEvent(time="5m ago", event="g event", severity=Severity.INFO),
            ],
        )
        md = report.to_markdown()
        alpha_pos = md.find("### alpha-svc")
        zulu_pos = md.find("### zulu-svc")
        general_pos = md.find("### General")

        assert alpha_pos != -1
        assert zulu_pos != -1
        assert general_pos != -1
        assert alpha_pos < zulu_pos < general_pos, "Groups should be A-Z with General last"


# ══════════════════════════════════════════════════════════════
# Content type fence tests (Task 4.4)
# ══════════════════════════════════════════════════════════════


class TestContentTypeFences:
    """Verify EvidenceDetail renders language-specific code fences."""

    def _render_evidence(self, content_type: ContentType, text: str = "sample content") -> str:
        """Helper: render a report with a single EvidenceDetail and return markdown."""
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.DEGRADED,
                scope="Cluster-wide",
                summary_text="Issue.",
            ),
            evidence_details=[
                EvidenceDetail(
                    title="Test Evidence",
                    evidence_text=text,
                    content_type=content_type,
                ),
            ],
        )
        return report.to_markdown()

    def test_yaml_content_type_fence(self) -> None:
        md = self._render_evidence(ContentType.YAML)
        assert "```yaml" in md

    def test_json_content_type_fence(self) -> None:
        md = self._render_evidence(ContentType.JSON)
        assert "```json" in md

    def test_log_content_type_fence(self) -> None:
        md = self._render_evidence(ContentType.LOG)
        assert "```log" in md

    def test_text_content_type_fence(self) -> None:
        md = self._render_evidence(ContentType.TEXT)
        assert "```text" in md

    def test_command_content_type_fence(self) -> None:
        md = self._render_evidence(ContentType.COMMAND)
        assert "```bash" in md

    def test_default_content_type_is_text(self) -> None:
        """EvidenceDetail without explicit content_type defaults to TEXT → ```text."""
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.DEGRADED,
                scope="Cluster-wide",
                summary_text="Issue.",
            ),
            evidence_details=[
                EvidenceDetail(title="Default", evidence_text="some output"),
            ],
        )
        md = report.to_markdown()
        assert "```text" in md


# ══════════════════════════════════════════════════════════════
# Evidence sub-bullets tests (Task 4.5)
# ══════════════════════════════════════════════════════════════


class TestEvidenceBullets:
    """Verify _render_evidence_subbullets and evidence rendering in findings."""

    def test_single_line_evidence_as_bullet(self) -> None:
        """A single-line evidence item renders as an indented bullet."""
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.DEGRADED,
                scope="Cluster-wide",
                summary_text="Issue.",
            ),
            findings=[
                Finding(
                    id="test", title="Test Finding", severity=Severity.HIGH,
                    evidence=["kubectl get pods shows CrashLoopBackOff"],
                ),
            ],
        )
        md = report.to_markdown()
        assert "  - kubectl get pods shows CrashLoopBackOff" in md

    def test_multi_line_evidence_in_code_block(self) -> None:
        """An evidence item containing newlines is wrapped in a code block."""
        multi = "line 1\nline 2\nline 3"
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.DEGRADED,
                scope="Cluster-wide",
                summary_text="Issue.",
            ),
            findings=[
                Finding(
                    id="test", title="Test Finding", severity=Severity.HIGH,
                    evidence=[multi],
                ),
            ],
        )
        md = report.to_markdown()
        assert "  - Multi-line evidence:" in md
        assert "    ```text" in md
        assert "    line 1" in md
        assert "    line 2" in md
        assert "    line 3" in md
        assert "    ```" in md

    def test_multiple_evidence_items_as_subbullets(self) -> None:
        """Multiple evidence items each get their own sub-bullet."""
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.DEGRADED,
                scope="Cluster-wide",
                summary_text="Issue.",
            ),
            findings=[
                Finding(
                    id="test", title="Test Finding", severity=Severity.HIGH,
                    evidence=["item one", "item two", "item three"],
                ),
            ],
        )
        md = report.to_markdown()
        assert "  - item one" in md
        assert "  - item two" in md
        assert "  - item three" in md

    def test_empty_evidence_list(self) -> None:
        """Finding with no evidence renders without crash and no Evidence section."""
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.DEGRADED,
                scope="Cluster-wide",
                summary_text="Issue.",
            ),
            findings=[
                Finding(id="test", title="No Evidence Finding", severity=Severity.HIGH),
            ],
        )
        md = report.to_markdown()
        assert "#### No Evidence Finding" in md
        # Should NOT have an Evidence sub-section
        assert "- **Evidence**:" not in md


# ══════════════════════════════════════════════════════════════
# Schema validation tests (Task 4.6)
# ══════════════════════════════════════════════════════════════


class TestContentTypeSchema:
    """Schema validation for ContentType enum and backward compatibility."""

    def test_content_type_enum_values(self) -> None:
        """ContentType enum has all 6 expected values."""
        expected = {"yaml", "json", "log", "text", "command", "unknown"}
        actual = set(ContentType)
        assert actual == expected

    def test_content_type_in_json_schema(self) -> None:
        """ContentType appears in HealthReport's model_json_schema $defs."""
        schema = HealthReport.model_json_schema()
        defs = schema.get("$defs", {})
        assert "ContentType" in defs, f"ContentType missing from $defs: {sorted(defs.keys())}"
        ct_schema = defs["ContentType"]
        assert "enum" in ct_schema
        assert set(ct_schema["enum"]) == {"yaml", "json", "log", "text", "command", "unknown"}

    def test_backward_compat_old_json(self) -> None:
        """JSON without new fields (service, content_type) validates correctly."""
        old_json = {
            "executive_summary": {
                "overall_status": "HEALTHY",
                "scope": "Cluster-wide",
                "summary_text": "All good.",
            },
            "timeline": [
                {"time": "5m ago", "event": "rollout complete", "severity": "INFO"},
            ],
            "evidence_details": [
                {"title": "Test Evidence", "evidence_text": "some text"},
            ],
        }
        report = HealthReport.model_validate(old_json)
        # New fields should have defaults
        assert report.timeline[0].service == ""
        assert report.evidence_details[0].content_type == ContentType.TEXT

    def test_fence_map_covers_all_content_types(self) -> None:
        """CONTENT_TYPE_FENCE_MAP has an entry for every ContentType value."""
        for ct in ContentType:
            assert ct in CONTENT_TYPE_FENCE_MAP, f"Missing fence mapping for {ct}"


# ── to_summary() tests ──────────────────────────────────────────────


class TestToSummary:
    """Tests for HealthReport.to_summary() compact output."""

    def test_summary_degraded_with_recommendations(self) -> None:
        """DEGRADED status with recommendations shows status, counts, and first rec."""
        report = _full_report()
        summary = report.to_summary()

        assert "🟡" in summary  # DEGRADED status emoji (full_report is DEGRADED)
        assert "DEGRADED" in summary
        assert "3" in summary  # issues_found
        assert "2 critical" in summary
        assert "1 warning" in summary
        assert "Increase payment-svc memory limit" in summary  # first recommendation title

    def test_summary_healthy_no_recommendations(self) -> None:
        """HEALTHY status with no recommendations shows 'None'."""
        report = _minimal_report()
        summary = report.to_summary()

        assert "🟢" in summary
        assert "HEALTHY" in summary
        assert "Top recommendation: None" in summary
        assert "0 critical" in summary

    def test_summary_max_five_lines(self) -> None:
        """Summary output must be at most 5 lines."""
        for report_fn in (_minimal_report, _full_report):
            summary = report_fn().to_summary()
            lines = summary.strip().split("\n")
            assert len(lines) <= 5, f"Summary has {len(lines)} lines, expected ≤ 5"

    def test_summary_critical_status(self) -> None:
        """CRITICAL status uses red emoji."""
        report = _minimal_report()
        report.executive_summary.overall_status = OverallStatus.CRITICAL
        report.executive_summary.critical_count = 5
        report.executive_summary.issues_found = 5
        summary = report.to_summary()

        assert "🔴" in summary
        assert "CRITICAL" in summary
        assert "5 critical" in summary


# ══════════════════════════════════════════════════════════════
# Timeline smart collapse tests
# ══════════════════════════════════════════════════════════════


class TestTimelineCollapse:
    """Verify smart timeline event collapse (_collapse_repeated_events)."""

    # ── Unit tests for _normalize_event_text ─────────────────

    def test_normalize_strips_pod_hash(self) -> None:
        """Pod hash suffix like -59967f9ccc-4zdx6 is removed."""
        raw = "FailedCreate ReplicaSet/app-xyz-59967f9ccc-4zdx6 volume already exists"
        normalised = _normalize_event_text(raw)
        assert "59967f9ccc" not in normalised
        assert "4zdx6" not in normalised
        assert "FailedCreate" in normalised

    def test_normalize_strips_counter(self) -> None:
        """Occurrence counters like (3rd time) are removed."""
        assert _normalize_event_text("pod crashed (3rd time)") == "pod crashed"
        assert _normalize_event_text("OOMKilled (1st time)") == "OOMKilled"
        assert _normalize_event_text("evicted (22nd time)") == "evicted"

    def test_normalize_strips_iso_timestamp(self) -> None:
        """Embedded ISO timestamps are removed."""
        raw = "Backup completed at 2024-01-15T10:30:00Z successfully"
        result = _normalize_event_text(raw)
        assert "2024-01-15T10:30:00Z" not in result
        assert "Backup completed" in result

    def test_normalize_identical_after_stripping(self) -> None:
        """Two events that differ only in pod hash normalise to the same string."""
        a = "FailedCreate pod-abc-59967f9ccc-4zdx6"
        b = "FailedCreate pod-abc-71122f8bbb-9xkq1"
        assert _normalize_event_text(a) == _normalize_event_text(b)

    def test_normalize_plain_text_unchanged(self) -> None:
        """Text without any volatile tokens is returned as-is (stripped)."""
        plain = "FailedMount volume already exists"
        assert _normalize_event_text(plain) == plain

    def test_normalize_strips_ipv4_address(self) -> None:
        """IPv4 addresses are replaced with <IP> placeholder."""
        raw = "Connection refused from 10.0.0.5 to backend"
        result = _normalize_event_text(raw)
        assert "10.0.0.5" not in result
        assert "<IP>" in result
        assert "Connection refused" in result

    def test_normalize_strips_ipv4_with_port(self) -> None:
        """IPv4 addresses with port (ip:port) are replaced with <IP> placeholder."""
        raw = "Failed to connect to 192.168.1.100:8080 endpoint"
        result = _normalize_event_text(raw)
        assert "192.168.1.100" not in result
        assert "<IP>" in result
        assert "Failed to connect" in result

    def test_normalize_strips_uuid(self) -> None:
        """UUIDs are replaced with <UUID> placeholder."""
        raw = "Volume a1b2c3d4-e5f6-7890-abcd-ef1234567890 not found"
        result = _normalize_event_text(raw)
        assert "a1b2c3d4-e5f6-7890-abcd-ef1234567890" not in result
        assert "<UUID>" in result
        assert "Volume" in result

    def test_normalize_collapses_events_with_different_ips(self) -> None:
        """Two events differing only in IP address normalise to the same string."""
        a = "Connection refused from 10.0.0.1 to backend"
        b = "Connection refused from 172.16.0.5 to backend"
        assert _normalize_event_text(a) == _normalize_event_text(b)

    def test_normalize_collapses_events_with_different_uuids(self) -> None:
        """Two events differing only in UUID normalise to the same string."""
        a = "Volume a1b2c3d4-e5f6-7890-abcd-ef1234567890 not found"
        b = "Volume ffffffff-0000-1111-2222-333333333333 not found"
        assert _normalize_event_text(a) == _normalize_event_text(b)

    # ── Unit tests for _collapse_repeated_events ─────────────

    def test_identical_events_collapse_to_one(self) -> None:
        """Seven identical events → single _CollapsedEvent with count=7."""
        events = [
            TimelineEvent(time=f"{i}m ago", event="FailedCreate volume exists",
                          severity=Severity.HIGH, service="app-rs")
            for i in range(7, 0, -1)
        ]
        collapsed = _collapse_repeated_events(events)
        assert len(collapsed) == 1
        assert collapsed[0].count == 7

    def test_different_events_stay_separate(self) -> None:
        """Three events with different text remain as three separate entries."""
        events = [
            TimelineEvent(time="10m ago", event="OOMKilled", severity=Severity.CRITICAL, service="svc"),
            TimelineEvent(time="8m ago", event="FailedCreate", severity=Severity.HIGH, service="svc"),
            TimelineEvent(time="5m ago", event="BackOff", severity=Severity.MEDIUM, service="svc"),
        ]
        collapsed = _collapse_repeated_events(events)
        assert len(collapsed) == 3

    def test_mixed_some_collapse_some_dont(self) -> None:
        """Mixed events: some groups collapse, unique events stay separate."""
        events = [
            TimelineEvent(time="10m ago", event="FailedCreate volume exists",
                          severity=Severity.HIGH, service="rs"),
            TimelineEvent(time="9m ago", event="FailedCreate volume exists",
                          severity=Severity.HIGH, service="rs"),
            TimelineEvent(time="8m ago", event="FailedCreate volume exists",
                          severity=Severity.HIGH, service="rs"),
            TimelineEvent(time="5m ago", event="OOMKilled pod restarted",
                          severity=Severity.CRITICAL, service="svc"),
        ]
        collapsed = _collapse_repeated_events(events)
        assert len(collapsed) == 2
        failed_create = next(c for c in collapsed if "FailedCreate" in c.event)
        oom = next(c for c in collapsed if "OOMKilled" in c.event)
        assert failed_create.count == 3
        assert oom.count == 1

    def test_single_event_count_is_one(self) -> None:
        """A single unique event gets count=1."""
        events = [
            TimelineEvent(time="5m ago", event="pod started", severity=Severity.INFO, service="svc"),
        ]
        collapsed = _collapse_repeated_events(events)
        assert len(collapsed) == 1
        assert collapsed[0].count == 1

    def test_different_severity_prevents_collapse(self) -> None:
        """Same text but different severity → NOT collapsed."""
        events = [
            TimelineEvent(time="10m ago", event="pod restarted", severity=Severity.HIGH, service="svc"),
            TimelineEvent(time="5m ago", event="pod restarted", severity=Severity.CRITICAL, service="svc"),
        ]
        collapsed = _collapse_repeated_events(events)
        assert len(collapsed) == 2

    def test_different_service_prevents_collapse(self) -> None:
        """Same text and severity but different service → NOT collapsed."""
        events = [
            TimelineEvent(time="10m ago", event="pod restarted", severity=Severity.HIGH, service="svc-a"),
            TimelineEvent(time="5m ago", event="pod restarted", severity=Severity.HIGH, service="svc-b"),
        ]
        collapsed = _collapse_repeated_events(events)
        assert len(collapsed) == 2

    def test_interleaved_events_not_collapsed(self) -> None:
        """Non-consecutive identical events [A, B, A] → 3 separate entries (chronological order preserved)."""
        events = [
            TimelineEvent(time="10m ago", event="event A", severity=Severity.INFO, service="svc"),
            TimelineEvent(time="9m ago", event="event B", severity=Severity.INFO, service="svc"),
            TimelineEvent(time="8m ago", event="event A", severity=Severity.INFO, service="svc"),
        ]
        collapsed = _collapse_repeated_events(events)
        # [A, B, A] must stay as 3 entries — not consecutive, so no merging
        assert len(collapsed) == 3
        assert collapsed[0].event == "event A"
        assert collapsed[0].count == 1
        assert collapsed[1].event == "event B"
        assert collapsed[1].count == 1
        assert collapsed[2].event == "event A"
        assert collapsed[2].count == 1

    def test_consecutive_runs_collapse(self) -> None:
        """Consecutive identical events [A, A, B, B, A] collapse to [A×2, B×2, A]."""
        events = [
            TimelineEvent(time="10m ago", event="event A", severity=Severity.INFO, service="svc"),
            TimelineEvent(time="9m ago", event="event A", severity=Severity.INFO, service="svc"),
            TimelineEvent(time="8m ago", event="event B", severity=Severity.INFO, service="svc"),
            TimelineEvent(time="7m ago", event="event B", severity=Severity.INFO, service="svc"),
            TimelineEvent(time="6m ago", event="event A", severity=Severity.INFO, service="svc"),
        ]
        collapsed = _collapse_repeated_events(events)
        assert len(collapsed) == 3
        assert collapsed[0].event == "event A"
        assert collapsed[0].count == 2
        assert collapsed[1].event == "event B"
        assert collapsed[1].count == 2
        assert collapsed[2].event == "event A"
        assert collapsed[2].count == 1

    def test_time_range_tracked_correctly(self) -> None:
        """time_first and time_last reflect first and last occurrence."""
        events = [
            TimelineEvent(time="56m ago", event="FailedCreate", severity=Severity.HIGH, service="rs"),
            TimelineEvent(time="30m ago", event="FailedCreate", severity=Severity.HIGH, service="rs"),
            TimelineEvent(time="6m ago", event="FailedCreate", severity=Severity.HIGH, service="rs"),
        ]
        collapsed = _collapse_repeated_events(events)
        assert len(collapsed) == 1
        ce = collapsed[0]
        assert ce.time_first == "56m ago"
        assert ce.time_last == "6m ago"

    def test_normalisation_collapses_events_with_pod_hashes(self) -> None:
        """Events identical except for pod hashes should collapse."""
        events = [
            TimelineEvent(
                time="10m ago",
                event="FailedCreate app-rs-59967f9ccc-4zdx6 volume already exists",
                severity=Severity.HIGH,
                service="app-rs",
            ),
            TimelineEvent(
                time="5m ago",
                event="FailedCreate app-rs-71122f8bbb-9xkq1 volume already exists",
                severity=Severity.HIGH,
                service="app-rs",
            ),
        ]
        collapsed = _collapse_repeated_events(events)
        assert len(collapsed) == 1
        assert collapsed[0].count == 2

    # ── _CollapsedEvent.display_event property ───────────────

    def test_display_event_single_no_notation(self) -> None:
        """A count=1 event does NOT show ×N notation."""
        ce = _CollapsedEvent(
            time_first="5m ago",
            time_last="5m ago",
            event="pod started",
            normalized_event="pod started",
            severity=Severity.INFO,
            service="svc",
            count=1,
        )
        assert ce.display_event == "pod started"
        assert "×" not in ce.display_event

    def test_display_event_collapsed_shows_notation(self) -> None:
        """A count>1 event shows ×N and time range in display_event, using normalized text."""
        raw_event = "FailedCreate ReplicaSet/app-xyz-59967f9ccc-4zdx6 — volume already exists"
        norm_event = "FailedCreate ReplicaSet/app-xyz — volume already exists"
        ce = _CollapsedEvent(
            time_first="56m ago",
            time_last="6m ago",
            event=raw_event,
            normalized_event=norm_event,
            severity=Severity.HIGH,
            service="app-rs",
            count=7,
        )
        display = ce.display_event
        # Collapsed display uses normalized text, not raw
        assert "×7" in display
        assert "56m ago" in display
        assert "6m ago" in display
        assert "→" in display
        # Normalized text is used (volatile pod hash stripped)
        assert "59967f9ccc" not in display
        assert "FailedCreate ReplicaSet/app-xyz — volume already exists" in display

    def test_display_time_always_time_first(self) -> None:
        """display_time is always time_first regardless of count."""
        for count in (1, 3, 7):
            ce = _CollapsedEvent(
                time_first="56m ago",
                time_last="6m ago",
                event="event",
                normalized_event="event",
                severity=Severity.INFO,
                service="svc",
                count=count,
            )
            assert ce.display_time == "56m ago"

    # ── Integration: render output ────────────────────────────

    def test_markdown_shows_collapsed_notation(self) -> None:
        """to_markdown() shows ×N notation when identical events exist."""
        events = [
            TimelineEvent(
                time=f"{i}m ago",
                event="FailedCreate volume exists",
                severity=Severity.HIGH,
                service="app-rs",
            )
            for i in range(7, 0, -1)
        ]
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.DEGRADED,
                scope="Cluster-wide",
                summary_text="Issues.",
            ),
            timeline=events,
        )
        md = report.to_markdown()
        assert "×7" in md
        assert "→" in md

    def test_markdown_no_cross_notation_for_unique_events(self) -> None:
        """to_markdown() with all-unique events shows NO ×N notation."""
        events = [
            TimelineEvent(time="10m ago", event="OOMKilled", severity=Severity.CRITICAL, service="svc"),
            TimelineEvent(time="8m ago", event="FailedCreate", severity=Severity.HIGH, service="svc"),
            TimelineEvent(time="5m ago", event="BackOff", severity=Severity.MEDIUM, service="svc"),
        ]
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.DEGRADED,
                scope="Cluster-wide",
                summary_text="Issues.",
            ),
            timeline=events,
        )
        md = report.to_markdown()
        assert "×" not in md

    def test_existing_full_report_timeline_unaffected(self) -> None:
        """The existing _full_report() has distinct events — no collapse occurs."""
        report = _full_report()
        md = report.to_markdown()
        # All four distinct events should still appear as-is
        assert "payment-svc-abc123 OOMKilled" in md
        assert "HPA scaled payment-svc to 5/5" in md
        assert "api-gateway v2.3.1 rollout complete" in md
        # None should have been collapsed (all unique)
        assert "×" not in md

    def test_flat_timeline_collapsed_with_service_column(self) -> None:
        """Flat timeline (minority with service) still renders collapsed events."""
        events = [
            TimelineEvent(time="10m ago", event="cluster upgrade", severity=Severity.INFO),
            TimelineEvent(time="9m ago", event="cluster upgrade", severity=Severity.INFO),
            TimelineEvent(time="5m ago", event="node ready", severity=Severity.INFO),
        ]
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.DEGRADED,
                scope="Cluster-wide",
                summary_text="Issues.",
            ),
            timeline=events,
        )
        md = report.to_markdown()
        # cluster upgrade collapses to ×2
        assert "×2" in md
        # node ready appears once, no notation
        assert "node ready" in md

    def test_empty_timeline_unaffected(self) -> None:
        """Empty timeline still shows fallback message (no crash)."""
        report = _minimal_report()
        md = report.to_markdown()
        assert "No timeline events available." in md
        assert "×" not in md

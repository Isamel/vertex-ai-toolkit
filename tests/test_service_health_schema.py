"""Tests for schema hardening: extra fields ignored, enums coerced."""

from __future__ import annotations

from vaig.skills.service_health.schema import (
    Confidence,
    DowngradedFinding,
    EvidenceDetail,
    ExecutiveSummary,
    Finding,
    HealthReport,
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

    def test_invalid_root_cause_hypothesis_is_immutable(self) -> None:
        hyp = RootCauseHypothesis(
            label="Cache overflow",
            probability=0.8,
            confirms_if="Memory exceeds 256Mi within 15 min.",
            refutes_if="Memory stays below 200Mi under full load.",
        )
        import pytest
        with pytest.raises(Exception):
            hyp.label = "mutated"  # type: ignore[misc]

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
    def _render(services) -> str:
        """Call _render_service_status directly to avoid HealthReport required fields."""
        from vaig.skills.service_health.schema import ExecutiveSummary, HealthReport
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status="HEALTHY",
                scope="Cluster-wide",
                summary_text="test",
            ),
            service_statuses=services,
        )
        parts: list[str] = []
        report._render_service_status(parts)
        return "\n".join(parts)

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
        md = self._render([svc])
        assert "| Service | Namespace |" in md, "Rollout Details table must include Namespace column."

    def test_rollout_details_table_shows_na_for_empty_namespace(self) -> None:
        """Namespace cell must show '—' when namespace is empty/None."""
        svc = self._svc(rollout_strategy="canary", namespace="")
        md = self._render([svc])
        assert "| my-svc | — |" in md, "Empty namespace must render as '—' in the table."

    def test_rollout_details_table_shows_na_for_none_rollout_strategy(self) -> None:
        """Strategy cell must show 'N/A' when rollout_strategy is None but another field is set."""
        svc = self._svc(rollout_strategy=None, rollout_status="Healthy")
        md = self._render([svc])
        assert "N/A" in md, "None rollout_strategy must render as 'N/A'."

    def test_filter_includes_service_with_only_rollout_status(self) -> None:
        """A service with only rollout_status set must still appear in the Rollout Details table."""
        svc = self._svc(rollout_strategy=None, rollout_status="Progressing")
        md = self._render([svc])
        assert "### Rollout Details" in md, (
            "Rollout Details table must appear when rollout_status is set even if strategy is None."
        )

    def test_filter_includes_service_with_only_hpa_conditions(self) -> None:
        """A service with only hpa_conditions set must still appear in the Rollout Details table."""
        svc = self._svc(rollout_strategy=None, rollout_status=None, hpa_conditions=["ScalingLimited: True"])
        md = self._render([svc])
        assert "### Rollout Details" in md, (
            "Rollout Details table must appear when hpa_conditions is set even if strategy is None."
        )

    def test_no_rollout_details_section_when_all_fields_empty(self) -> None:
        """Services with no rollout data must NOT produce a Rollout Details section."""
        svc = self._svc()
        md = self._render([svc])
        assert "### Rollout Details" not in md, (
            "Rollout Details table must NOT appear when no service has rollout data."
        )


class TestRootCauseHypothesisProbabilityValidator:
    """SPEC-V2-AUDIT-02 acceptance criteria."""

    def _make_report(self, hypotheses: list) -> HealthReport:
        return HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.HEALTHY,
                scope="test",
                summary_text="test",
            ),
            root_cause_hypotheses=hypotheses,
        )

    def _hyp(self, probability: float, status: str = "open") -> RootCauseHypothesis:
        return RootCauseHypothesis(
            label="Test hypothesis",
            probability=probability,
            confirms_if="Signal X is observed.",
            refutes_if="Signal Y is absent.",
            status=status,  # type: ignore[arg-type]
        )

    def test_empty_hypotheses_always_valid(self) -> None:
        """Empty list must not trigger the validator."""
        report = self._make_report([])
        assert report.root_cause_hypotheses == []

    def test_single_hypothesis_valid_probability(self) -> None:
        """Single hypothesis with probability 0.8 must pass."""
        report = self._make_report([self._hyp(0.8)])
        assert len(report.root_cause_hypotheses) == 1

    def test_two_hypotheses_sum_exactly_on_boundary(self) -> None:
        """Sum == 0.7 is valid (lower bound)."""
        report = self._make_report([self._hyp(0.4), self._hyp(0.3)])
        assert len(report.root_cause_hypotheses) == 2

    def test_two_hypotheses_sum_at_upper_bound(self) -> None:
        """Sum == 1.0 is valid (upper bound)."""
        report = self._make_report([self._hyp(0.6), self._hyp(0.4)])
        assert len(report.root_cause_hypotheses) == 2

    def test_probability_sum_too_low_normalizes(self) -> None:
        """Sum < 1.0 is normalized proportionally; no error raised."""
        report = self._make_report([self._hyp(0.2), self._hyp(0.3)])
        total = sum(h.probability for h in report.root_cause_hypotheses)
        assert abs(total - 1.0) < 1e-3

    def test_probability_sum_too_high_normalizes(self) -> None:
        """Sum > 1.0 (e.g. Gemini returning 0.6+0.7=1.5) is normalized; no error raised."""
        report = self._make_report([self._hyp(0.6), self._hyp(0.7)])
        total = sum(h.probability for h in report.root_cause_hypotheses)
        assert abs(total - 1.0) < 1e-3
        # Relative ranking must be preserved: first hyp was lower, must stay lower
        assert report.root_cause_hypotheses[0].probability < report.root_cause_hypotheses[1].probability

    def test_all_zero_probabilities_assigns_uniform(self) -> None:
        """All-zero probabilities (degenerate case) must produce uniform-ish distribution summing to 1.0."""
        report = self._make_report([self._hyp(0.0), self._hyp(0.0), self._hyp(0.0)])
        total = sum(h.probability for h in report.root_cause_hypotheses)
        assert abs(total - 1.0) < 1e-3
        # All probabilities must be close to uniform (within rounding residual)
        probs = [h.probability for h in report.root_cause_hypotheses]
        expected = 1.0 / 3
        for p in probs:
            assert abs(p - expected) < 0.01

    def test_normalization_last_element_absorbs_rounding(self) -> None:
        """With 3+ hypotheses, rounding residual must be absorbed so sum is exactly 1.0."""
        # 3 equal hypotheses: 0.5 each → total 1.5; normalized each would be 0.3333...
        report = self._make_report([self._hyp(0.5), self._hyp(0.5), self._hyp(0.5)])
        total = sum(h.probability for h in report.root_cause_hypotheses)
        assert abs(total - 1.0) < 1e-4


class TestRootCauseHypothesisLabelTruncation:
    """label > 80 chars must be truncated, not rejected (Gemini ignores length hints)."""

    def _hyp(self, label: str) -> RootCauseHypothesis:
        return RootCauseHypothesis(
            label=label,
            probability=1.0,
            confirms_if="Signal X is observed.",
            refutes_if="Signal Y is absent.",
        )

    def test_label_within_limit_unchanged(self) -> None:
        """Labels <= 80 chars must not be modified."""
        hyp = self._hyp("Short label")
        assert hyp.label == "Short label"

    def test_label_exactly_80_unchanged(self) -> None:
        """Label of exactly 80 chars must not be modified."""
        label = "x" * 80
        hyp = self._hyp(label)
        assert hyp.label == label

    def test_label_over_limit_truncated_to_80(self) -> None:
        """Labels > 80 chars must be truncated to 80 chars (with ellipsis)."""
        long_label = "Los servicios backend están rechazando conexiones debido a problemas de configuración en el proxy."
        assert len(long_label) > 80
        hyp = self._hyp(long_label)
        assert len(hyp.label) <= 80
        assert hyp.label.endswith("…")

    def test_label_truncation_does_not_raise(self) -> None:
        """A report containing a hypothesis with an oversized label must not raise ValidationError."""
        long_label = "A" * 120
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.HEALTHY,
                scope="test",
                summary_text="test",
            ),
            root_cause_hypotheses=[self._hyp(long_label)],
        )  # must not raise
        assert len(report.root_cause_hypotheses[0].label) <= 80


class TestRootCauseHypothesisOther:
    """Misc RootCauseHypothesis constraints."""

    def _make_report(self, hypotheses: list) -> HealthReport:
        return HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.HEALTHY,
                scope="test",
                summary_text="test",
            ),
            root_cause_hypotheses=hypotheses,
        )

    def _hyp(self, probability: float, status: str = "open") -> RootCauseHypothesis:
        return RootCauseHypothesis(
            label="Test hypothesis",
            probability=probability,
            confirms_if="Signal X is observed.",
            refutes_if="Signal Y is absent.",
            status=status,  # type: ignore[arg-type]
        )

    def test_max_four_hypotheses_enforced(self) -> None:
        """max_length=4 must reject a list of 5."""
        import pytest
        from pydantic import ValidationError
        hyps = [self._hyp(0.2)] * 4 + [self._hyp(0.05)]
        with pytest.raises(ValidationError):
            self._make_report(hyps)

    def test_frozen_model_immutable(self) -> None:
        """RootCauseHypothesis must be frozen (immutable)."""
        import pytest
        hyp = self._hyp(0.8)
        with pytest.raises(Exception):
            hyp.label = "mutated"  # type: ignore[misc]

    def test_markdown_render_contains_probability_and_confirms_if(self) -> None:
        """Markdown render must show probability % and confirms_if."""
        # Use probability=1.0 so normalization leaves the value unchanged (single hypothesis).
        report = self._make_report([
            RootCauseHypothesis(
                label="Cache overflow",
                probability=1.0,
                supporting_evidence=["Memory grows linearly"],
                refuting_evidence=["CPU usage normal"],
                confirms_if="Memory exceeds 256Mi within 15 min.",
                refutes_if="Memory stays below 200Mi under full load.",
            )
        ])
        md = report.to_markdown()
        assert "Cache overflow" in md
        assert "100%" in md
        assert "Confirms if" in md
        assert "Memory exceeds 256Mi" in md

    def test_backwards_compat_empty_list_renders_unchanged(self) -> None:
        """Reports with empty root_cause_hypotheses must render the fallback text."""
        report = self._make_report([])
        md = report.to_markdown()
        assert "No root cause hypotheses to report." in md


class TestServiceStatusDegradedReasonTruncation:
    """ServiceStatus.degraded_reason is truncated instead of rejected when > 160 chars."""

    def _make_status(self, degraded_reason: str) -> ServiceStatus:
        return ServiceStatus(
            service="svc",
            health_status=ServiceHealthStatus.DEGRADED,
            degraded_reason=degraded_reason,
        )

    def test_short_reason_unchanged(self) -> None:
        """degraded_reason within 160 chars must not be modified."""
        reason = "Memory pressure on node pool."
        status = self._make_status(reason)
        assert status.degraded_reason == reason

    def test_reason_exactly_160_unchanged(self) -> None:
        """degraded_reason of exactly 160 chars must not be modified."""
        reason = "x" * 160
        status = self._make_status(reason)
        assert status.degraded_reason == reason

    def test_reason_over_limit_truncated(self) -> None:
        """degraded_reason > 160 chars must be truncated to 160 chars with ellipsis."""
        long_reason = "x" * 200
        status = self._make_status(long_reason)
        assert len(status.degraded_reason) <= 160  # type: ignore[arg-type]
        assert status.degraded_reason.endswith("…")  # type: ignore[union-attr]

    def test_reason_over_limit_does_not_raise(self) -> None:
        """An oversized degraded_reason must not raise ValidationError."""
        long_reason = "A" * 250
        status = self._make_status(long_reason)  # must not raise
        assert status.degraded_reason is not None


class TestHealthReportOverallSeverityReasonTruncation:
    """HealthReport.overall_severity_reason is truncated instead of rejected when > 240 chars."""

    def _make_report(self, overall_severity_reason: str) -> HealthReport:
        return HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.DEGRADED,
                scope="test",
                summary_text="test",
            ),
            overall_severity_reason=overall_severity_reason,
        )

    def test_short_reason_unchanged(self) -> None:
        """overall_severity_reason within 240 chars must not be modified."""
        reason = "Two HIGH findings in distinct namespaces."
        report = self._make_report(reason)
        assert report.overall_severity_reason == reason

    def test_reason_exactly_240_unchanged(self) -> None:
        """overall_severity_reason of exactly 240 chars must not be modified."""
        reason = "x" * 240
        report = self._make_report(reason)
        assert report.overall_severity_reason == reason

    def test_reason_over_limit_truncated(self) -> None:
        """overall_severity_reason > 240 chars must be truncated with ellipsis."""
        long_reason = "x" * 300
        report = self._make_report(long_reason)
        assert len(report.overall_severity_reason) <= 240  # type: ignore[arg-type]
        assert report.overall_severity_reason.endswith("…")  # type: ignore[union-attr]

    def test_reason_over_limit_does_not_raise(self) -> None:
        """An oversized overall_severity_reason must not raise ValidationError."""
        long_reason = "A" * 350
        report = self._make_report(long_reason)  # must not raise
        assert report.overall_severity_reason is not None

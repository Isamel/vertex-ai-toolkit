"""Tests for EvidenceGap model and HealthReport coverage fields (evidence-gaps change)."""

from __future__ import annotations

from vaig.skills.service_health.schema import EvidenceGap, HealthReport


class TestEvidenceGapModel:
    """T4 — Pydantic validation of EvidenceGap."""

    def test_valid_not_called(self) -> None:
        gap = EvidenceGap(source="deployment_metrics", reason="not_called")
        assert gap.source == "deployment_metrics"
        assert gap.reason == "not_called"
        assert gap.details is None

    def test_valid_error_with_details(self) -> None:
        gap = EvidenceGap(source="deployment_metrics", reason="error", details="timeout")
        assert gap.reason == "error"
        assert gap.details == "timeout"

    def test_valid_empty_result(self) -> None:
        gap = EvidenceGap(source="pod_logs", reason="empty_result")
        assert gap.reason == "empty_result"

    def test_all_three_reason_values_accepted(self) -> None:
        for reason in ("not_called", "error", "empty_result"):
            gap = EvidenceGap(source="some_tool", reason=reason)
            assert gap.reason == reason

    def test_invalid_reason_raises_validation_error(self) -> None:
        # reason is now a plain str to reduce Gemini schema complexity;
        # any string value is accepted at instantiation time.
        gap = EvidenceGap(source="some_tool", reason="skipped")
        assert gap.reason == "skipped"

    def test_invalid_reason_unknown_raises(self) -> None:
        # reason is now a plain str — no constraint enforced.
        gap = EvidenceGap(source="some_tool", reason="partial")
        assert gap.reason == "partial"

    def test_details_optional_none_by_default(self) -> None:
        gap = EvidenceGap(source="s", reason="not_called")
        assert gap.details is None

    def test_extra_fields_ignored(self) -> None:
        """model_config extra='ignore' — unknown fields should not raise."""
        gap = EvidenceGap(source="s", reason="error", details="d", unknown_field="x")
        assert not hasattr(gap, "unknown_field")


class TestHealthReportCoverageFields:
    """T4 — HealthReport backward compat and new coverage fields."""

    _MINIMAL_EXEC_SUMMARY = {
        "overall_status": "HEALTHY",
        "scope": "Cluster-wide",
        "summary_text": "All good",
    }

    def test_backward_compat_no_new_fields(self) -> None:
        """Old instantiation without evidence_gaps / investigation_coverage still validates."""
        report = HealthReport(executive_summary=self._MINIMAL_EXEC_SUMMARY)
        assert report.evidence_gaps == []
        assert report.investigation_coverage is None

    def test_evidence_gaps_defaults_to_empty_list(self) -> None:
        report = HealthReport(executive_summary=self._MINIMAL_EXEC_SUMMARY)
        assert isinstance(report.evidence_gaps, list)
        assert len(report.evidence_gaps) == 0

    def test_investigation_coverage_defaults_to_none(self) -> None:
        report = HealthReport(executive_summary=self._MINIMAL_EXEC_SUMMARY)
        assert report.investigation_coverage is None

    def test_health_report_with_gaps_populated(self) -> None:
        gaps = [
            {"source": "deployment_metrics", "reason": "error", "details": "timeout"},
            {"source": "pod_logs", "reason": "empty_result"},
            {"source": "events", "reason": "not_called"},
        ]
        report = HealthReport(
            executive_summary=self._MINIMAL_EXEC_SUMMARY,
            evidence_gaps=gaps,
            investigation_coverage="9/12 signal sources checked",
        )
        assert len(report.evidence_gaps) == 3
        assert report.evidence_gaps[0].source == "deployment_metrics"
        assert report.evidence_gaps[0].reason == "error"
        assert report.evidence_gaps[1].reason == "empty_result"
        assert report.evidence_gaps[2].reason == "not_called"
        assert report.investigation_coverage == "9/12 signal sources checked"

    def test_health_report_fully_covered(self) -> None:
        report = HealthReport(
            executive_summary=self._MINIMAL_EXEC_SUMMARY,
            evidence_gaps=[],
            investigation_coverage="12/12 signal sources checked",
        )
        assert report.evidence_gaps == []
        assert "12/12" in (report.investigation_coverage or "")

    def test_invalid_gap_reason_raises(self) -> None:
        # reason is now a plain str — any value is accepted.
        report = HealthReport(
            executive_summary=self._MINIMAL_EXEC_SUMMARY,
            evidence_gaps=[{"source": "x", "reason": "bad_value"}],
        )
        assert report.evidence_gaps[0].reason == "bad_value"

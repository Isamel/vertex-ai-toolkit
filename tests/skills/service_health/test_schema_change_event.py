"""Tests for ChangeEvent model and recent_changes field on HealthReport (change-correlation-section)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vaig.skills.service_health.schema import (
    ChangeEvent,
    ExecutiveSummary,
    HealthReport,
    OverallStatus,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _minimal_executive_summary() -> ExecutiveSummary:
    return ExecutiveSummary(
        overall_status=OverallStatus.HEALTHY,
        scope="Namespace: default",
        summary_text="All systems operational.",
    )


def _make_change_event(**kwargs) -> dict:
    defaults = {
        "timestamp": "2024-01-15T10:00:00Z",
        "type": "deployment",
        "description": "Deployed payment-svc v1.2.3",
        "correlation_to_issue": "Deployment coincides with error rate spike at T0",
    }
    defaults.update(kwargs)
    return defaults


# ── T5a: ChangeEvent validates with all required fields ──────────────────────


class TestChangeEventValidation:
    def test_valid_change_event_all_fields(self) -> None:
        data = _make_change_event()
        event = ChangeEvent(**data)
        assert event.timestamp == "2024-01-15T10:00:00Z"
        assert event.type == "deployment"
        assert event.description == "Deployed payment-svc v1.2.3"
        assert event.correlation_to_issue == "Deployment coincides with error rate spike at T0"

    def test_change_event_type_config_change(self) -> None:
        event = ChangeEvent(**_make_change_event(type="config_change"))
        assert event.type == "config_change"

    def test_change_event_type_hpa_scaling(self) -> None:
        event = ChangeEvent(**_make_change_event(type="hpa_scaling"))
        assert event.type == "hpa_scaling"

    # ── T5b: ValidationError when required field missing ─────────────────────

    def test_missing_timestamp_raises(self) -> None:
        data = _make_change_event()
        del data["timestamp"]
        with pytest.raises(ValidationError):
            ChangeEvent(**data)

    def test_missing_type_raises(self) -> None:
        data = _make_change_event()
        del data["type"]
        with pytest.raises(ValidationError):
            ChangeEvent(**data)

    def test_missing_description_raises(self) -> None:
        data = _make_change_event()
        del data["description"]
        with pytest.raises(ValidationError):
            ChangeEvent(**data)

    def test_missing_correlation_to_issue_raises(self) -> None:
        data = _make_change_event()
        del data["correlation_to_issue"]
        with pytest.raises(ValidationError):
            ChangeEvent(**data)


# ── T5c/T5d: HealthReport backward compat and populated deserialization ──────


class TestHealthReportRecentChanges:
    def test_health_report_without_recent_changes_defaults_to_empty_list(self) -> None:
        """Backward-compatible: missing recent_changes → []."""
        report = HealthReport(executive_summary=_minimal_executive_summary())
        assert report.recent_changes == []

    def test_health_report_model_validate_without_recent_changes(self) -> None:
        payload = {
            "executive_summary": {
                "overall_status": "HEALTHY",
                "scope": "Namespace: default",
                "summary_text": "OK",
            }
        }
        report = HealthReport.model_validate(payload)
        assert report.recent_changes == []

    def test_health_report_deserializes_populated_recent_changes(self) -> None:
        payload = {
            "executive_summary": {
                "overall_status": "DEGRADED",
                "scope": "Namespace: production",
                "summary_text": "Service degraded after deployment.",
            },
            "recent_changes": [
                {
                    "timestamp": "2024-01-15T09:58:00Z",
                    "type": "deployment",
                    "description": "Rolled out payment-svc v2.0.0",
                    "correlation_to_issue": "Deployment 2 min before error spike",
                }
            ],
        }
        report = HealthReport.model_validate(payload)
        assert len(report.recent_changes) == 1
        event = report.recent_changes[0]
        assert isinstance(event, ChangeEvent)
        assert event.type == "deployment"
        assert event.timestamp == "2024-01-15T09:58:00Z"

    def test_health_report_recent_changes_multiple_events(self) -> None:
        payload = {
            "executive_summary": {
                "overall_status": "CRITICAL",
                "scope": "Namespace: staging",
                "summary_text": "Multiple changes detected.",
            },
            "recent_changes": [
                _make_change_event(type="deployment"),
                _make_change_event(type="config_change", description="Updated ConfigMap"),
                _make_change_event(type="hpa_scaling", description="HPA scaled to 10 replicas"),
            ],
        }
        report = HealthReport.model_validate(payload)
        assert len(report.recent_changes) == 3
        types = [e.type for e in report.recent_changes]
        assert types == ["deployment", "config_change", "hpa_scaling"]

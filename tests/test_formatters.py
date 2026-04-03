"""Tests for shared notification formatters (SC-NH-11)."""

from __future__ import annotations

from unittest.mock import MagicMock

from vaig.integrations.formatters import (
    FormattedAlert,
    FormattedReport,
    format_alert,
    format_report_summary,
    meets_threshold,
    status_to_severity,
)

# ── Helper ───────────────────────────────────────────────────


def _make_report(
    status: str = "CRITICAL",
    summary: str = "Service degraded",
    issues: int = 3,
    critical: int = 1,
    warning: int = 2,
    scope: str = "Namespace: prod",
) -> MagicMock:
    report = MagicMock()
    report.executive_summary.overall_status.value = status
    report.executive_summary.scope = scope
    report.executive_summary.summary_text = summary
    report.executive_summary.issues_found = issues
    report.executive_summary.critical_count = critical
    report.executive_summary.warning_count = warning

    finding = MagicMock()
    finding.title = "OOMKilled pods detected"
    report.findings = [finding]
    return report


# ── meets_threshold tests ────────────────────────────────────


class TestMeetsThreshold:
    """Tests for meets_threshold() edge cases."""

    def test_critical_meets_critical_high(self) -> None:
        assert meets_threshold("CRITICAL", ["critical", "high"]) is True

    def test_high_meets_critical_high(self) -> None:
        assert meets_threshold("HIGH", ["critical", "high"]) is True

    def test_medium_below_critical_high(self) -> None:
        assert meets_threshold("MEDIUM", ["critical", "high"]) is False

    def test_info_below_all(self) -> None:
        assert meets_threshold("INFO", ["critical", "high"]) is False

    def test_empty_notify_on_returns_false(self) -> None:
        assert meets_threshold("CRITICAL", []) is False

    def test_unknown_severity_returns_false(self) -> None:
        assert meets_threshold("BANANA", ["critical"]) is False

    def test_invalid_notify_on_values_ignored(self) -> None:
        assert meets_threshold("CRITICAL", ["banana", "potato"]) is False

    def test_mixed_valid_invalid(self) -> None:
        assert meets_threshold("HIGH", ["banana", "high"]) is True

    def test_case_insensitive(self) -> None:
        assert meets_threshold("critical", ["CRITICAL"]) is True
        assert meets_threshold("Critical", ["critical"]) is True

    def test_all_severities_enabled(self) -> None:
        all_severities = ["critical", "high", "medium", "low", "info"]
        assert meets_threshold("INFO", all_severities) is True


# ── status_to_severity tests ─────────────────────────────────


class TestStatusToSeverity:
    def test_critical_maps_to_critical(self) -> None:
        assert status_to_severity("CRITICAL") == "CRITICAL"

    def test_degraded_maps_to_high(self) -> None:
        assert status_to_severity("DEGRADED") == "HIGH"

    def test_healthy_maps_to_info(self) -> None:
        assert status_to_severity("HEALTHY") == "INFO"

    def test_unknown_maps_to_medium(self) -> None:
        assert status_to_severity("UNKNOWN") == "MEDIUM"

    def test_unrecognised_maps_to_medium(self) -> None:
        assert status_to_severity("BANANA") == "MEDIUM"


# ── format_alert tests ───────────────────────────────────────


class TestFormatAlert:
    def test_returns_formatted_alert(self) -> None:
        report = _make_report(status="CRITICAL")
        alert = format_alert(report, service_name="my-api")

        assert isinstance(alert, FormattedAlert)
        assert alert.severity == "CRITICAL"
        assert alert.service_name == "my-api"
        assert alert.summary == "Service degraded"
        assert len(alert.findings) == 1

    def test_deterministic_output(self) -> None:
        """SC-NH-11: identical input → identical output."""
        report = _make_report()
        a = format_alert(report)
        b = format_alert(report)
        assert a == b

    def test_falls_back_to_scope_for_service_name(self) -> None:
        report = _make_report(scope="Namespace: staging")
        alert = format_alert(report)
        assert alert.service_name == "Namespace: staging"


# ── format_report_summary tests ──────────────────────────────


class TestFormatReportSummary:
    def test_returns_formatted_report(self) -> None:
        report = _make_report(status="DEGRADED")
        result = format_report_summary(report, execution_time=12.5)

        assert isinstance(result, FormattedReport)
        assert result.status == "DEGRADED"
        assert result.execution_time == 12.5
        assert result.issues_found == 3

    def test_deterministic_output(self) -> None:
        """SC-NH-11: identical input → identical output."""
        report = _make_report()
        a = format_report_summary(report, execution_time=5.0)
        b = format_report_summary(report, execution_time=5.0)
        assert a == b

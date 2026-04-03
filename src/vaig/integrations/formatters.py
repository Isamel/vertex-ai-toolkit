"""Shared notification formatters — severity helpers and structured data for all channels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vaig.skills.service_health.schema import HealthReport

# ── Severity constants ───────────────────────────────────────

SEVERITY_ORDER: dict[str, int] = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
    "info": 0,
}

SEVERITY_ICON: dict[str, str] = {
    "CRITICAL": "\U0001f534",
    "HIGH": "\U0001f7e0",
    "MEDIUM": "\U0001f7e1",
    "LOW": "\U0001f7e2",
    "INFO": "\U0001f535",
}

_STATUS_TO_SEVERITY: dict[str, str] = {
    "CRITICAL": "CRITICAL",
    "DEGRADED": "HIGH",
    "HEALTHY": "INFO",
    "UNKNOWN": "MEDIUM",
}


# ── Helper functions ─────────────────────────────────────────


def meets_threshold(severity: str, notify_on: list[str]) -> bool:
    """Check if severity meets the minimum notification threshold.

    The threshold is determined by the *lowest* valid severity in
    ``notify_on``.  Any severity at or above that threshold will pass.
    Invalid severities in ``notify_on`` are ignored so misconfiguration
    fails safe (no alerts) rather than fail open.
    """
    if not notify_on:
        return False

    severity_lower = severity.lower()
    if severity_lower not in SEVERITY_ORDER:
        return False

    valid = [s.lower() for s in notify_on if s.lower() in SEVERITY_ORDER]
    if not valid:
        return False

    min_threshold = min(SEVERITY_ORDER[s] for s in valid)
    return SEVERITY_ORDER[severity_lower] >= min_threshold


def status_to_severity(overall_status: str) -> str:
    """Map OverallStatus to a severity string for threshold comparison."""
    return _STATUS_TO_SEVERITY.get(overall_status.upper(), "MEDIUM")


# ── Formatted data structures ────────────────────────────────


@dataclass
class FormattedAlert:
    """Structured alert data for channel-specific rendering."""

    title: str
    severity: str
    severity_icon: str
    service_name: str
    summary: str
    findings: list[str] = field(default_factory=list)
    pagerduty_url: str | None = None


@dataclass
class FormattedReport:
    """Structured report summary data for channel-specific rendering."""

    title: str
    status: str
    status_icon: str
    scope: str
    issues_found: int
    critical_count: int
    warning_count: int
    summary: str
    findings: list[str] = field(default_factory=list)
    execution_time: float = 0.0


def format_report_summary(
    report: HealthReport,
    execution_time: float = 0.0,
) -> FormattedReport:
    """Build a FormattedReport from a HealthReport."""
    es = report.executive_summary
    severity = status_to_severity(es.overall_status.value)
    icon = SEVERITY_ICON.get(severity.upper(), "\u2753")

    findings_text = [f.title for f in report.findings[:5]] if report.findings else []

    return FormattedReport(
        title="Service Health Report",
        status=es.overall_status.value,
        status_icon=icon,
        scope=es.scope,
        issues_found=es.issues_found,
        critical_count=es.critical_count,
        warning_count=es.warning_count,
        summary=es.summary_text,
        findings=findings_text,
        execution_time=execution_time,
    )

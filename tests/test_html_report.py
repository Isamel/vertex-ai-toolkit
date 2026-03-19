"""Tests for the HTML report renderer (vaig.ui.html_report)."""

from __future__ import annotations

import re
from unittest.mock import patch

from vaig.skills.service_health.schema import (
    ActionUrgency,
    ClusterMetric,
    ExecutiveSummary,
    Finding,
    HealthReport,
    OverallStatus,
    RecommendedAction,
    ServiceHealthStatus,
    ServiceStatus,
    Severity,
    TimelineEvent,
)
from vaig.ui.html_report import render_health_report_html

# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_executive_summary(
    status: OverallStatus = OverallStatus.DEGRADED,
    summary_text: str = "Several services are experiencing issues.",
    critical: int = 2,
    warnings: int = 3,
    issues: int = 5,
    services: int = 10,
) -> ExecutiveSummary:
    return ExecutiveSummary(
        overall_status=status,
        scope="Namespace: production",
        summary_text=summary_text,
        critical_count=critical,
        warning_count=warnings,
        issues_found=issues,
        services_checked=services,
    )


def _make_minimal_report() -> HealthReport:
    """A minimal HealthReport with only the required field."""
    return HealthReport(executive_summary=_make_executive_summary())


def _make_full_report() -> HealthReport:
    """A HealthReport with all sections populated."""
    return HealthReport(
        executive_summary=_make_executive_summary(
            status=OverallStatus.CRITICAL,
            summary_text="Critical issues detected across multiple services.",
            critical=3,
            warnings=2,
            issues=5,
            services=8,
        ),
        cluster_overview=[
            ClusterMetric(metric="Nodes", value="5/5 Ready"),
            ClusterMetric(metric="Pods", value="42/45 Running"),
        ],
        service_statuses=[
            ServiceStatus(
                service="payment-svc",
                namespace="production",
                status=ServiceHealthStatus.FAILED,
                pods_ready="0/3",
                restarts_1h="15",
                issues="CrashLoopBackOff on all pods",
            ),
            ServiceStatus(
                service="auth-svc",
                namespace="production",
                status=ServiceHealthStatus.DEGRADED,
                pods_ready="2/3",
                restarts_1h="2",
            ),
            ServiceStatus(
                service="frontend",
                namespace="production",
                status=ServiceHealthStatus.HEALTHY,
                pods_ready="3/3",
                restarts_1h="0",
            ),
        ],
        findings=[
            Finding(
                id="crashloop-payment",
                title="CrashLoopBackOff on payment-svc",
                severity=Severity.CRITICAL,
                service="payment-svc",
                description="All 3 pods are crash looping.",
                root_cause="OOMKilled due to memory leak",
                evidence=["kubectl describe pod shows OOMKilled", "Restart count: 15"],
                remediation="Increase memory limits and fix leak",
            ),
            Finding(
                id="degraded-auth",
                title="auth-svc pod not ready",
                severity=Severity.HIGH,
                service="auth-svc",
                description="One pod is in Pending state.",
            ),
            Finding(
                id="config-warning",
                title="Deprecated API version in ConfigMap",
                severity=Severity.MEDIUM,
                service="frontend",
            ),
            Finding(
                id="resource-low",
                title="Low memory requests for logging-svc",
                severity=Severity.LOW,
            ),
            Finding(
                id="audit-log",
                title="Audit logging not enabled",
                severity=Severity.INFO,
            ),
        ],
        recommendations=[
            RecommendedAction(
                priority=1,
                title="Fix memory limit for payment-svc",
                description="Increase memory limit from 256Mi to 512Mi.",
                urgency=ActionUrgency.IMMEDIATE,
                command="kubectl set resources deployment/payment-svc -c payment --limits=memory=512Mi",
                why="OOMKilled is causing crash loops and all pods are down.",
                risk="Rolling restart required.",
            ),
            RecommendedAction(
                priority=2,
                title="Investigate auth-svc pod scheduling",
                urgency=ActionUrgency.SHORT_TERM,
            ),
        ],
        timeline=[
            TimelineEvent(time="5m ago", event="payment-svc pod OOMKilled", severity=Severity.CRITICAL, service="payment-svc"),
            TimelineEvent(time="10m ago", event="auth-svc pod entered Pending", severity=Severity.HIGH, service="auth-svc"),
        ],
    )


# ── HTML structure tests ──────────────────────────────────────────────────────


class TestHtmlStructure:
    """Validate that the output is valid HTML5 with the expected structure."""

    def test_starts_with_doctype(self) -> None:
        html = render_health_report_html(_make_minimal_report())
        assert html.startswith("<!DOCTYPE html>")

    def test_has_html_open_and_close(self) -> None:
        html = render_health_report_html(_make_minimal_report())
        assert "<html" in html
        assert "</html>" in html

    def test_has_head_and_body(self) -> None:
        html = render_health_report_html(_make_minimal_report())
        assert "<head>" in html
        assert "</head>" in html
        assert "<body>" in html
        assert "</body>" in html

    def test_has_inline_css_style_block(self) -> None:
        html = render_health_report_html(_make_minimal_report())
        assert "<style>" in html
        assert "</style>" in html

    def test_has_charset_meta(self) -> None:
        html = render_health_report_html(_make_minimal_report())
        assert 'charset="UTF-8"' in html or "charset=UTF-8" in html

    def test_has_viewport_meta(self) -> None:
        html = render_health_report_html(_make_minimal_report())
        assert "viewport" in html

    def test_title_contains_vaig(self) -> None:
        html = render_health_report_html(_make_minimal_report())
        assert "<title>" in html
        assert "vaig" in html.lower()


# ── Content tests ─────────────────────────────────────────────────────────────


class TestHtmlContent:
    """Validate that report data appears in the rendered HTML."""

    def test_executive_summary_text_present(self) -> None:
        report = _make_minimal_report()
        html = render_health_report_html(report)
        assert report.executive_summary.summary_text in html

    def test_executive_summary_scope_present(self) -> None:
        report = _make_minimal_report()
        html = render_health_report_html(report)
        assert report.executive_summary.scope in html

    def test_overall_status_present(self) -> None:
        report = _make_minimal_report()
        html = render_health_report_html(report)
        assert report.executive_summary.overall_status.value in html

    def test_service_names_present(self) -> None:
        report = _make_full_report()
        html = render_health_report_html(report)
        assert "payment-svc" in html
        assert "auth-svc" in html
        assert "frontend" in html

    def test_finding_titles_present(self) -> None:
        report = _make_full_report()
        html = render_health_report_html(report)
        assert "CrashLoopBackOff on payment-svc" in html
        assert "auth-svc pod not ready" in html

    def test_recommendation_command_present(self) -> None:
        report = _make_full_report()
        html = render_health_report_html(report)
        assert "kubectl set resources" in html

    def test_timeline_events_present(self) -> None:
        report = _make_full_report()
        html = render_health_report_html(report)
        assert "payment-svc pod OOMKilled" in html
        assert "auth-svc pod entered Pending" in html

    def test_cluster_metrics_present(self) -> None:
        report = _make_full_report()
        html = render_health_report_html(report)
        assert "5/5 Ready" in html
        assert "42/45 Running" in html

    def test_generation_timestamp_present(self) -> None:
        html = render_health_report_html(_make_minimal_report())
        # Should contain a UTC timestamp like "2026-03-18"
        assert re.search(r"\d{4}-\d{2}-\d{2}", html) is not None

    def test_vaig_version_present(self) -> None:
        with patch("vaig.__version__", "1.2.3-test"):
            html = render_health_report_html(_make_minimal_report())
        assert "1.2.3-test" in html


# ── Severity badge / colour tests ─────────────────────────────────────────────


class TestSeverityBadges:
    """Validate that severity levels render with their correct CSS colours."""

    SEVERITY_COLOURS = {
        Severity.CRITICAL: "#E82424",
        Severity.HIGH: "#FF9E3B",
        Severity.MEDIUM: "#E6C384",
        Severity.LOW: "#7FB4CA",
        Severity.INFO: "#727169",
    }

    def test_critical_colour_in_css(self) -> None:
        html = render_health_report_html(_make_full_report())
        assert self.SEVERITY_COLOURS[Severity.CRITICAL] in html

    def test_high_colour_in_css(self) -> None:
        html = render_health_report_html(_make_full_report())
        assert self.SEVERITY_COLOURS[Severity.HIGH] in html

    def test_medium_colour_in_css(self) -> None:
        html = render_health_report_html(_make_full_report())
        assert self.SEVERITY_COLOURS[Severity.MEDIUM] in html

    def test_low_colour_in_css(self) -> None:
        html = render_health_report_html(_make_full_report())
        assert self.SEVERITY_COLOURS[Severity.LOW] in html

    def test_info_colour_in_css(self) -> None:
        html = render_health_report_html(_make_full_report())
        assert self.SEVERITY_COLOURS[Severity.INFO] in html

    def test_all_severity_values_rendered_in_findings(self) -> None:
        report = _make_full_report()
        html = render_health_report_html(report)
        for severity in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]:
            assert severity.value in html, f"Severity {severity.value} not found in HTML"

    def test_kanagawa_bg_colour(self) -> None:
        """Verify the Kanagawa dark background colour is present."""
        html = render_health_report_html(_make_minimal_report())
        assert "#1F1F28" in html


# ── Overall status colour tests ───────────────────────────────────────────────


class TestStatusColours:
    """Validate that OverallStatus values render with correct accent colours."""

    def test_critical_status_colour(self) -> None:
        report = HealthReport(
            executive_summary=_make_executive_summary(status=OverallStatus.CRITICAL)
        )
        html = render_health_report_html(report)
        assert "#E82424" in html  # CRITICAL → red

    def test_healthy_status_colour(self) -> None:
        report = HealthReport(
            executive_summary=_make_executive_summary(status=OverallStatus.HEALTHY)
        )
        html = render_health_report_html(report)
        assert "#98BB6C" in html  # HEALTHY → green

    def test_degraded_status_colour(self) -> None:
        report = HealthReport(
            executive_summary=_make_executive_summary(status=OverallStatus.DEGRADED)
        )
        html = render_health_report_html(report)
        assert "#E6C384" in html  # DEGRADED → yellow


# ── Edge case tests ───────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases: empty collections, HTML escaping, single-field reports."""

    def test_empty_findings_shows_no_issues_message(self) -> None:
        report = HealthReport(
            executive_summary=_make_executive_summary(status=OverallStatus.HEALTHY, issues=0, critical=0, warnings=0),
            findings=[],
        )
        html = render_health_report_html(report)
        assert "No issues found" in html

    def test_empty_recommendations_section_omitted(self) -> None:
        report = HealthReport(
            executive_summary=_make_executive_summary(),
            recommendations=[],
        )
        html = render_health_report_html(report)
        # Action Plan card should not be present if no recommendations
        assert "<h2>Action Plan</h2>" not in html

    def test_empty_service_statuses_section_omitted(self) -> None:
        report = HealthReport(
            executive_summary=_make_executive_summary(),
            service_statuses=[],
        )
        html = render_health_report_html(report)
        assert "Service Status" not in html

    def test_empty_cluster_overview_section_omitted(self) -> None:
        report = HealthReport(
            executive_summary=_make_executive_summary(),
            cluster_overview=[],
        )
        html = render_health_report_html(report)
        assert "Cluster Overview" not in html

    def test_empty_timeline_section_omitted(self) -> None:
        report = HealthReport(
            executive_summary=_make_executive_summary(),
            timeline=[],
        )
        html = render_health_report_html(report)
        assert "<h2>Timeline</h2>" not in html

    def test_html_special_chars_escaped_in_summary(self) -> None:
        """<, > and & in user data must be HTML-escaped to prevent injection."""
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.UNKNOWN,
                scope="Resource: deploy/<my-app> & others",
                summary_text="Status: <unknown> & critical",
            )
        )
        html = render_health_report_html(report)
        # Raw < and > should NOT appear in the report body (other than HTML tags)
        assert "&lt;my-app&gt;" in html
        assert "&lt;unknown&gt;" in html
        assert "&amp;" in html

    def test_returns_string_type(self) -> None:
        html = render_health_report_html(_make_minimal_report())
        assert isinstance(html, str)

    def test_output_not_empty(self) -> None:
        html = render_health_report_html(_make_minimal_report())
        assert len(html) > 500

    def test_finding_with_no_optional_fields(self) -> None:
        """Finding with only required fields should render without error."""
        report = HealthReport(
            executive_summary=_make_executive_summary(),
            findings=[
                Finding(id="bare-finding", title="Bare finding", severity=Severity.INFO)
            ],
        )
        html = render_health_report_html(report)
        assert "Bare finding" in html

    def test_recommendation_with_no_command(self) -> None:
        """Recommendation without command should render without terminal block."""
        report = HealthReport(
            executive_summary=_make_executive_summary(),
            recommendations=[
                RecommendedAction(priority=1, title="Check logs", command="")
            ],
        )
        html = render_health_report_html(report)
        assert "Check logs" in html
        assert '<div class="terminal-block">' not in html

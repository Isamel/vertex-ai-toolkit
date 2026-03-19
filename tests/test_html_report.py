"""Tests for the HTML report renderer (vaig.ui.html_report).

The renderer now uses a template-based SPA: it loads spa_template.html via
importlib.resources and injects the serialised HealthReport JSON in place of
the sentinel placeholder.  Tests reflect that model.
"""

from __future__ import annotations

import json

import pytest

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
from vaig.ui.html_report import _SENTINEL, _load_template, render_health_report_html

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
            TimelineEvent(
                time="5m ago",
                event="payment-svc pod OOMKilled",
                severity=Severity.CRITICAL,
                service="payment-svc",
            ),
            TimelineEvent(
                time="10m ago",
                event="auth-svc pod entered Pending",
                severity=Severity.HIGH,
                service="auth-svc",
            ),
        ],
    )


# ── Template loading tests ────────────────────────────────────────────────────


class TestTemplateLoading:
    """Verify that the template is loaded and cached correctly."""

    def test_load_template_returns_string(self) -> None:
        template = _load_template()
        assert isinstance(template, str)

    def test_load_template_not_empty(self) -> None:
        template = _load_template()
        assert len(template) > 1000

    def test_load_template_contains_sentinel(self) -> None:
        """The raw template must contain the sentinel placeholder."""
        template = _load_template()
        assert _SENTINEL in template

    def test_load_template_is_cached(self) -> None:
        """Second call returns the same object (module-level cache)."""
        t1 = _load_template()
        t2 = _load_template()
        assert t1 is t2


# ── HTML structure tests ──────────────────────────────────────────────────────


class TestHtmlStructure:
    """Validate that the output is valid HTML5 with the expected structure."""

    def test_starts_with_doctype(self) -> None:
        result = render_health_report_html(_make_minimal_report())
        assert result.startswith("<!DOCTYPE html>")

    def test_has_html_open_and_close(self) -> None:
        result = render_health_report_html(_make_minimal_report())
        assert "<html" in result
        assert "</html>" in result

    def test_has_head_and_body(self) -> None:
        result = render_health_report_html(_make_minimal_report())
        assert "<head>" in result
        assert "</head>" in result
        assert "<body>" in result
        assert "</body>" in result

    def test_has_inline_css_style_block(self) -> None:
        result = render_health_report_html(_make_minimal_report())
        assert "<style>" in result
        assert "</style>" in result

    def test_has_charset_meta(self) -> None:
        result = render_health_report_html(_make_minimal_report())
        assert "charset" in result.lower()

    def test_has_viewport_meta(self) -> None:
        result = render_health_report_html(_make_minimal_report())
        assert "viewport" in result

    def test_has_script_block(self) -> None:
        result = render_health_report_html(_make_minimal_report())
        assert "<script>" in result
        assert "</script>" in result

    def test_returns_string_type(self) -> None:
        result = render_health_report_html(_make_minimal_report())
        assert isinstance(result, str)

    def test_output_not_empty(self) -> None:
        result = render_health_report_html(_make_minimal_report())
        assert len(result) > 500


# ── JSON injection tests ──────────────────────────────────────────────────────


class TestJsonInjection:
    """Verify that the sentinel is replaced with the serialised report JSON."""

    def test_sentinel_not_in_output(self) -> None:
        result = render_health_report_html(_make_minimal_report())
        assert _SENTINEL not in result

    def test_output_contains_report_json(self) -> None:
        """The serialised executive_summary text must appear in the output."""
        report = _make_minimal_report()
        result = render_health_report_html(report)
        assert report.executive_summary.summary_text in result

    def test_output_contains_full_report_json(self) -> None:
        """Full report data — service names and finding titles must be in output."""
        report = _make_full_report()
        result = render_health_report_html(report)
        assert "payment-svc" in result
        assert "CrashLoopBackOff on payment-svc" in result
        assert "auth-svc" in result

    def test_injected_json_is_parseable(self) -> None:
        """The JSON blob injected into the template must be valid JSON."""
        report = _make_minimal_report()
        result = render_health_report_html(report)
        # Extract the JSON blob: it replaces _SENTINEL (e.g. const REPORT_DATA = <JSON>;)
        # Find 'const REPORT_DATA = ' and extract until the next ';'
        marker = "const REPORT_DATA = "
        start = result.index(marker) + len(marker)
        end = result.index(";", start)
        json_blob = result[start:end]
        parsed = json.loads(json_blob)
        assert isinstance(parsed, dict)
        assert "executive_summary" in parsed

    def test_injected_json_matches_report(self) -> None:
        """The injected JSON must round-trip back to the same report structure."""
        report = _make_full_report()
        result = render_health_report_html(report)
        marker = "const REPORT_DATA = "
        start = result.index(marker) + len(marker)
        end = result.index(";", start)
        parsed = json.loads(result[start:end])
        assert parsed["executive_summary"]["overall_status"] == "CRITICAL"
        assert len(parsed["findings"]) == 5


# ── Script injection safety tests ────────────────────────────────────────────


class TestScriptInjectionSafety:
    """Verify that </script> in report data is safely escaped."""

    def test_closing_script_tag_in_data_is_escaped(self) -> None:
        """</script> inside report data MUST be escaped to <\\/script>."""
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.UNKNOWN,
                scope="test",
                summary_text="Injected </script><script>alert(1)</script>",
            )
        )
        result = render_health_report_html(report)
        # The raw </script> must NOT appear as-is in the inline script block
        # (it would prematurely close the <script> tag)
        # The escaped form <\/ must be present instead
        assert "<\\/script>" in result

    def test_closing_slash_sequence_escaped(self) -> None:
        """Every </ in serialised JSON becomes <\\/ in the output."""
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.UNKNOWN,
                scope="<br/>",  # has </
                summary_text="safe",
            )
        )
        result = render_health_report_html(report)
        # Find the JSON payload area — the </ from scope must be escaped
        marker = "const REPORT_DATA = "
        start = result.index(marker)
        end = result.index("</script>", start)
        payload_area = result[start:end]
        assert "</" not in payload_area


# ── Empty / minimal report tests ─────────────────────────────────────────────


class TestEmptyReport:
    """Verify that a minimal or near-empty report renders without errors."""

    def test_minimal_report_renders(self) -> None:
        result = render_health_report_html(_make_minimal_report())
        assert result.startswith("<!DOCTYPE html>")

    def test_all_optional_fields_empty_renders(self) -> None:
        report = HealthReport(
            executive_summary=_make_executive_summary(
                status=OverallStatus.HEALTHY,
                issues=0,
                critical=0,
                warnings=0,
            ),
            findings=[],
            recommendations=[],
            service_statuses=[],
            cluster_overview=[],
            timeline=[],
        )
        result = render_health_report_html(report)
        assert "<!DOCTYPE html>" in result
        assert "</html>" in result

    def test_empty_report_json_present_in_output(self) -> None:
        """Even a minimal report must inject JSON (not leave the sentinel)."""
        report = _make_minimal_report()
        result = render_health_report_html(report)
        assert _SENTINEL not in result
        assert "executive_summary" in result

    @pytest.mark.parametrize(
        "status",
        [OverallStatus.HEALTHY, OverallStatus.DEGRADED, OverallStatus.CRITICAL, OverallStatus.UNKNOWN],
    )
    def test_all_overall_statuses_render(self, status: OverallStatus) -> None:
        report = HealthReport(executive_summary=_make_executive_summary(status=status))
        result = render_health_report_html(report)
        assert status.value in result

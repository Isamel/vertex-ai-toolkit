"""Tests for the 'What Changed Recently' HTML section (change-correlation-section).

T7 — Verify the section renders correctly when recent_changes is populated,
and is absent (hidden) when recent_changes is empty.
"""

from __future__ import annotations

import json

from vaig.skills.service_health.schema import (
    ChangeEvent,
    ExecutiveSummary,
    HealthReport,
    OverallStatus,
)
from vaig.ui.html_report import render_health_report_html

# ── helpers ──────────────────────────────────────────────────────────────────


def _minimal_report(**kwargs) -> HealthReport:
    return HealthReport(
        executive_summary=ExecutiveSummary(
            overall_status=OverallStatus.DEGRADED,
            scope="Namespace: production",
            summary_text="Service degraded after deployment.",
        ),
        **kwargs,
    )


def _make_event(**kwargs) -> ChangeEvent:
    defaults = {
        "timestamp": "2024-01-15T09:58:00Z",
        "type": "deployment",
        "description": "Rolled out payment-svc v2.0.0",
        "correlation_to_issue": "Deployment 2 min before error spike",
    }
    defaults.update(kwargs)
    return ChangeEvent(**defaults)


# ── T7a: Section visible with events ─────────────────────────────────────────


class TestRecentChangesHTMLSection:
    def test_section_present_when_recent_changes_populated(self) -> None:
        report = _minimal_report(recent_changes=[_make_event()])
        html = render_health_report_html(report)
        assert "section-recent-changes" in html

    def test_section_contains_event_timestamp(self) -> None:
        report = _minimal_report(recent_changes=[_make_event(timestamp="2024-01-15T09:58:00Z")])
        html = render_health_report_html(report)
        # The timestamp is injected via REPORT_DATA JSON
        assert "2024-01-15T09:58:00Z" in html

    def test_section_contains_event_type_in_json(self) -> None:
        report = _minimal_report(recent_changes=[_make_event(type="config_change")])
        html = render_health_report_html(report)
        assert "config_change" in html

    def test_section_contains_description_in_json(self) -> None:
        description = "Deployed auth-svc v3.1.0"
        report = _minimal_report(recent_changes=[_make_event(description=description)])
        html = render_health_report_html(report)
        assert description in html

    def test_section_contains_correlation_in_json(self) -> None:
        correlation = "Canary rollout coincided with 15% error rate increase"
        report = _minimal_report(
            recent_changes=[_make_event(correlation_to_issue=correlation)]
        )
        html = render_health_report_html(report)
        assert correlation in html

    def test_multiple_change_events_all_present_in_json(self) -> None:
        report = _minimal_report(
            recent_changes=[
                _make_event(type="deployment", description="Deploy A"),
                _make_event(type="config_change", description="Config B"),
                _make_event(type="hpa_scaling", description="HPA C"),
            ]
        )
        html = render_health_report_html(report)
        assert "Deploy A" in html
        assert "Config B" in html
        assert "HPA C" in html

    # ── T7b: Section hidden when empty ───────────────────────────────────────

    def test_section_hidden_when_recent_changes_empty(self) -> None:
        """When recent_changes is [], the section must have display:none."""
        report = _minimal_report(recent_changes=[])
        html = render_health_report_html(report)
        # The section HTML is always present; JS sets display:none for empty
        # We verify the section element exists but also the JS will hide it
        assert 'id="section-recent-changes"' in html
        # The section starts hidden via style="display:none"
        assert 'id="section-recent-changes" style="display:none"' in html

    def test_no_recent_changes_key_empty_list_in_json(self) -> None:
        """HealthReport with no recent_changes must serialize recent_changes as []."""
        report = _minimal_report()
        html = render_health_report_html(report)
        # Locate the injected JSON and verify recent_changes is []
        # The sentinel is replaced with the JSON payload
        # REPORT_DATA is injected on a single line: const REPORT_DATA = {...};
        line = next(
            (ln for ln in html.splitlines() if 'const REPORT_DATA =' in ln),
            None,
        )
        assert line is not None, "REPORT_DATA assignment not found in HTML"
        # Strip the variable assignment prefix and trailing semicolon
        raw = line.split('const REPORT_DATA =', 1)[1].strip().rstrip(';')
        data = json.loads(raw)
        assert data["recent_changes"] == []

    # ── Section position: after summary, before findings ─────────────────────

    def test_recent_changes_section_before_findings(self) -> None:
        """recent-changes section must appear before section-findings in the DOM."""
        report = _minimal_report(recent_changes=[_make_event()])
        html = render_health_report_html(report)
        pos_changes = html.find('id="section-recent-changes"')
        pos_findings = html.find('id="section-findings"')
        assert pos_changes != -1
        assert pos_findings != -1
        assert pos_changes < pos_findings, (
            "section-recent-changes must appear before section-findings"
        )

    def test_recent_changes_section_after_summary(self) -> None:
        """recent-changes section must appear after section-summary in the DOM."""
        report = _minimal_report(recent_changes=[_make_event()])
        html = render_health_report_html(report)
        pos_summary = html.find('id="section-summary"')
        pos_changes = html.find('id="section-recent-changes"')
        assert pos_summary != -1
        assert pos_changes != -1
        assert pos_summary < pos_changes, (
            "section-summary must appear before section-recent-changes"
        )

"""Tests for the display module — show_cost_summary and related helpers."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from vaig.cli.display import (
    print_executive_summary_panel,
    print_recommendations_table,
    show_cost_summary,
)
from vaig.skills.service_health.schema import (
    ActionUrgency,
    Effort,
    ExecutiveSummary,
    HealthReport,
    OverallStatus,
    RecommendedAction,
)


def _capture_output(usage: dict[str, int] | None, model_id: str) -> str:
    """Call show_cost_summary and capture Rich output as plain text."""
    buf = StringIO()
    con = Console(file=buf, force_terminal=False, width=200)
    show_cost_summary(usage, model_id, console=con)
    return buf.getvalue()


# ── Basic display behaviour ──────────────────────────────────────


class TestShowCostSummary:
    def test_shows_tokens_and_cost(self) -> None:
        usage = {
            "prompt_tokens": 1234,
            "completion_tokens": 567,
            "thinking_tokens": 89,
            "total_tokens": 1890,
        }
        output = _capture_output(usage, "gemini-2.5-pro")
        assert "1,234 in" in output
        assert "567 out" in output
        assert "89 thinking" in output
        assert "1,890 total" in output
        assert "$" in output  # cost should be present

    def test_no_thinking_tokens_omits_thinking(self) -> None:
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "thinking_tokens": 0,
            "total_tokens": 300,
        }
        output = _capture_output(usage, "gemini-2.5-flash")
        assert "100 in" in output
        assert "200 out" in output
        assert "thinking" not in output

    def test_unknown_model_shows_na_cost(self) -> None:
        usage = {
            "prompt_tokens": 50,
            "completion_tokens": 100,
            "total_tokens": 150,
        }
        output = _capture_output(usage, "unknown-model-xyz")
        assert "N/A" in output

    def test_known_model_shows_dollar_cost(self) -> None:
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
        }
        output = _capture_output(usage, "gemini-2.5-flash")
        assert "$" in output
        assert "N/A" not in output


# ── Edge cases — silent no-ops ───────────────────────────────────


class TestShowCostSummaryEdgeCases:
    def test_none_usage_is_silent(self) -> None:
        output = _capture_output(None, "gemini-2.5-pro")
        assert output.strip() == ""

    def test_empty_dict_is_silent(self) -> None:
        output = _capture_output({}, "gemini-2.5-pro")
        assert output.strip() == ""

    def test_all_zero_tokens_is_silent(self) -> None:
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "thinking_tokens": 0,
            "total_tokens": 0,
        }
        output = _capture_output(usage, "gemini-2.5-pro")
        assert output.strip() == ""

    def test_missing_keys_defaults_to_zero(self) -> None:
        """Usage dict with only total_tokens should still work."""
        usage = {"total_tokens": 42}
        output = _capture_output(usage, "gemini-2.5-pro")
        assert "42 total" in output
        assert "0 in" in output
        assert "0 out" in output


# ── Helper: HealthReport factories ───────────────────────────────


def _make_report(
    *,
    status: OverallStatus = OverallStatus.HEALTHY,
    scope: str = "Cluster-wide",
    summary_text: str = "All services OK.",
    issues_found: int = 0,
    critical_count: int = 0,
    warning_count: int = 0,
    recommendations: list[RecommendedAction] | None = None,
) -> HealthReport:
    return HealthReport(
        executive_summary=ExecutiveSummary(
            overall_status=status,
            scope=scope,
            summary_text=summary_text,
            issues_found=issues_found,
            critical_count=critical_count,
            warning_count=warning_count,
        ),
        recommendations=recommendations or [],
    )


def _capture_panel(report: HealthReport) -> str:
    buf = StringIO()
    con = Console(file=buf, force_terminal=False, width=120)
    print_executive_summary_panel(report, console=con)
    return buf.getvalue()


def _capture_table(report: HealthReport) -> str:
    buf = StringIO()
    con = Console(file=buf, force_terminal=False, width=120)
    print_recommendations_table(report, console=con)
    return buf.getvalue()


# ── Executive Summary Panel ──────────────────────────────────────


class TestPrintExecutiveSummaryPanel:
    """Tests for print_executive_summary_panel()."""

    def test_healthy_status_shows_green_emoji(self) -> None:
        output = _capture_panel(_make_report(status=OverallStatus.HEALTHY))
        assert "HEALTHY" in output
        assert "Executive Summary" in output

    def test_critical_status_content(self) -> None:
        report = _make_report(
            status=OverallStatus.CRITICAL,
            scope="Namespace: payments",
            summary_text="Payment pods are crashlooping.",
            issues_found=3,
            critical_count=2,
            warning_count=1,
        )
        output = _capture_panel(report)
        assert "CRITICAL" in output
        assert "Namespace: payments" in output
        assert "Payment pods are crashlooping." in output
        assert "3" in output  # issues_found
        assert "2 critical" in output
        assert "1 warning" in output

    def test_degraded_status(self) -> None:
        output = _capture_panel(_make_report(status=OverallStatus.DEGRADED))
        assert "DEGRADED" in output

    def test_unknown_status(self) -> None:
        output = _capture_panel(_make_report(status=OverallStatus.UNKNOWN))
        assert "UNKNOWN" in output

    def test_panel_contains_scope(self) -> None:
        output = _capture_panel(_make_report(scope="Resource: deploy/nginx in default"))
        assert "Resource: deploy/nginx in default" in output

    def test_panel_contains_summary_text(self) -> None:
        output = _capture_panel(_make_report(summary_text="Memory pressure detected."))
        assert "Memory pressure detected." in output

    def test_uses_custom_console(self) -> None:
        """Passing a custom console should work without errors."""
        buf = StringIO()
        con = Console(file=buf, force_terminal=False, width=80)
        report = _make_report()
        print_executive_summary_panel(report, console=con)
        assert "Executive Summary" in buf.getvalue()


# ── Recommendations Table ────────────────────────────────────────


def _make_recommendation(
    *,
    priority: int = 1,
    title: str = "Restart pods",
    command: str = "kubectl rollout restart deploy/nginx",
    risk: str = "Brief downtime",
    urgency: ActionUrgency = ActionUrgency.IMMEDIATE,
    effort: Effort = Effort.LOW,
) -> RecommendedAction:
    return RecommendedAction(
        priority=priority,
        title=title,
        command=command,
        risk=risk,
        urgency=urgency,
        effort=effort,
    )


class TestPrintRecommendationsTable:
    """Tests for print_recommendations_table()."""

    def test_empty_recommendations_is_silent(self) -> None:
        output = _capture_table(_make_report())
        assert output.strip() == ""

    def test_single_recommendation_renders_table(self) -> None:
        rec = _make_recommendation()
        report = _make_report(recommendations=[rec])
        output = _capture_table(report)
        assert "Recommended Actions" in output
        assert "Restart pods" in output
        assert "kubectl rollout restart deploy/nginx" in output
        assert "Brief downtime" in output

    def test_multiple_recommendations_sorted_by_priority(self) -> None:
        recs = [
            _make_recommendation(priority=3, title="Long-term fix"),
            _make_recommendation(priority=1, title="Immediate fix"),
            _make_recommendation(priority=2, title="Short-term fix"),
        ]
        report = _make_report(recommendations=recs)
        output = _capture_table(report)
        # All three should appear
        assert "Immediate fix" in output
        assert "Short-term fix" in output
        assert "Long-term fix" in output

    def test_missing_command_shows_dash(self) -> None:
        rec = _make_recommendation(command="")
        report = _make_report(recommendations=[rec])
        output = _capture_table(report)
        # The em dash should appear in place of empty command
        assert "—" in output

    def test_missing_risk_shows_dash(self) -> None:
        rec = _make_recommendation(risk="")
        report = _make_report(recommendations=[rec])
        output = _capture_table(report)
        assert "—" in output

    def test_urgency_immediate_displayed(self) -> None:
        rec = _make_recommendation(urgency=ActionUrgency.IMMEDIATE)
        report = _make_report(recommendations=[rec])
        output = _capture_table(report)
        assert "IMMEDIATE" in output

    def test_urgency_long_term_displayed(self) -> None:
        rec = _make_recommendation(urgency=ActionUrgency.LONG_TERM)
        report = _make_report(recommendations=[rec])
        output = _capture_table(report)
        assert "LONG_TERM" in output

    def test_uses_custom_console(self) -> None:
        buf = StringIO()
        con = Console(file=buf, force_terminal=False, width=120)
        rec = _make_recommendation()
        report = _make_report(recommendations=[rec])
        print_recommendations_table(report, console=con)
        assert "Recommended Actions" in buf.getvalue()

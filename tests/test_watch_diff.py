"""Tests for watch mode diff computation and display.

Covers:
- ``compute_report_diff``: new, resolved, severity-changed, unchanged findings
- ``ReportDiff`` properties: ``has_changes``, ``summary_line``
- ``print_watch_diff_summary``: renders without errors, correct content
- Watch loop integration: diff summary printed on iteration 2+
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock, patch

import typer
from rich.console import Console

from vaig.cli.commands.live import _run_watch_loop
from vaig.cli.display import print_watch_diff_summary
from vaig.skills.service_health.diff import (
    ReportDiff,
    SeverityChange,
    compute_report_diff,
)
from vaig.skills.service_health.schema import (
    ExecutiveSummary,
    Finding,
    HealthReport,
    OverallStatus,
    Severity,
)

# ── Helpers ──────────────────────────────────────────────────────


def _make_finding(
    fid: str,
    title: str = "",
    severity: Severity = Severity.MEDIUM,
) -> Finding:
    """Create a minimal Finding for testing."""
    return Finding(
        id=fid,
        title=title or fid,
        severity=severity,
    )


def _make_report(*findings: Finding) -> HealthReport:
    """Create a minimal HealthReport with given findings."""
    return HealthReport(
        executive_summary=ExecutiveSummary(
            overall_status=OverallStatus.HEALTHY,
            scope="test",
            summary_text="test report",
            issues_found=len(findings),
            critical_count=0,
            warning_count=0,
        ),
        findings=list(findings),
    )


def _capture_diff_panel(diff: ReportDiff, iteration: int = 2) -> str:
    """Render print_watch_diff_summary and capture output as plain text."""
    buf = StringIO()
    con = Console(file=buf, force_terminal=False, width=120)
    print_watch_diff_summary(diff, iteration, console=con)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════
# compute_report_diff
# ══════════════════════════════════════════════════════════════


class TestComputeReportDiff:
    """Tests for the compute_report_diff function."""

    def test_new_findings_detected(self) -> None:
        """Findings present in current but not previous are NEW."""
        prev = _make_report(_make_finding("svc-a"))
        curr = _make_report(_make_finding("svc-a"), _make_finding("svc-b"))

        diff = compute_report_diff(curr, prev)

        assert len(diff.new_findings) == 1
        assert diff.new_findings[0].id == "svc-b"

    def test_resolved_findings_detected(self) -> None:
        """Findings present in previous but not current are RESOLVED."""
        prev = _make_report(_make_finding("svc-a"), _make_finding("svc-b"))
        curr = _make_report(_make_finding("svc-a"))

        diff = compute_report_diff(curr, prev)

        assert len(diff.resolved_findings) == 1
        assert diff.resolved_findings[0].id == "svc-b"

    def test_severity_change_detected(self) -> None:
        """Findings with same id but different severity are SEVERITY CHANGED."""
        prev = _make_report(_make_finding("svc-a", severity=Severity.MEDIUM))
        curr = _make_report(_make_finding("svc-a", severity=Severity.CRITICAL))

        diff = compute_report_diff(curr, prev)

        assert len(diff.severity_changes) == 1
        sc = diff.severity_changes[0]
        assert sc.finding.id == "svc-a"
        assert sc.previous_severity == "MEDIUM"
        assert sc.current_severity == "CRITICAL"

    def test_unchanged_findings(self) -> None:
        """Findings with same id and same severity are UNCHANGED."""
        finding = _make_finding("svc-a", severity=Severity.LOW)
        prev = _make_report(finding)
        curr = _make_report(finding)

        diff = compute_report_diff(curr, prev)

        assert len(diff.unchanged_findings) == 1
        assert diff.unchanged_findings[0].id == "svc-a"
        assert not diff.new_findings
        assert not diff.resolved_findings
        assert not diff.severity_changes

    def test_empty_reports(self) -> None:
        """Both reports empty produces an empty diff."""
        prev = _make_report()
        curr = _make_report()

        diff = compute_report_diff(curr, prev)

        assert not diff.new_findings
        assert not diff.resolved_findings
        assert not diff.unchanged_findings
        assert not diff.severity_changes

    def test_mixed_changes(self) -> None:
        """Combination of new, resolved, unchanged, and severity-changed."""
        prev = _make_report(
            _make_finding("stays", severity=Severity.LOW),
            _make_finding("goes-away"),
            _make_finding("changes-sev", severity=Severity.INFO),
        )
        curr = _make_report(
            _make_finding("stays", severity=Severity.LOW),
            _make_finding("brand-new", severity=Severity.HIGH),
            _make_finding("changes-sev", severity=Severity.CRITICAL),
        )

        diff = compute_report_diff(curr, prev)

        assert len(diff.new_findings) == 1
        assert diff.new_findings[0].id == "brand-new"
        assert len(diff.resolved_findings) == 1
        assert diff.resolved_findings[0].id == "goes-away"
        assert len(diff.severity_changes) == 1
        assert diff.severity_changes[0].previous_severity == "INFO"
        assert diff.severity_changes[0].current_severity == "CRITICAL"
        assert len(diff.unchanged_findings) == 1
        assert diff.unchanged_findings[0].id == "stays"


# ══════════════════════════════════════════════════════════════
# ReportDiff properties
# ══════════════════════════════════════════════════════════════


class TestReportDiffProperties:
    """Tests for has_changes and summary_line properties."""

    def test_has_changes_true_with_new(self) -> None:
        diff = ReportDiff(new_findings=[_make_finding("x")])
        assert diff.has_changes is True

    def test_has_changes_true_with_resolved(self) -> None:
        diff = ReportDiff(resolved_findings=[_make_finding("x")])
        assert diff.has_changes is True

    def test_has_changes_true_with_severity_change(self) -> None:
        sc = SeverityChange(
            finding=_make_finding("x"),
            previous_severity="LOW",
            current_severity="HIGH",
        )
        diff = ReportDiff(severity_changes=[sc])
        assert diff.has_changes is True

    def test_has_changes_false_unchanged_only(self) -> None:
        diff = ReportDiff(unchanged_findings=[_make_finding("x")])
        assert diff.has_changes is False

    def test_has_changes_false_empty(self) -> None:
        diff = ReportDiff()
        assert diff.has_changes is False

    def test_summary_line_no_changes(self) -> None:
        diff = ReportDiff()
        assert diff.summary_line == "No changes"

    def test_summary_line_new_only(self) -> None:
        diff = ReportDiff(new_findings=[_make_finding("a"), _make_finding("b")])
        assert "+2 NEW" in diff.summary_line

    def test_summary_line_resolved_only(self) -> None:
        diff = ReportDiff(resolved_findings=[_make_finding("a")])
        assert "1 RESOLVED" in diff.summary_line

    def test_summary_line_severity_changes(self) -> None:
        sc = SeverityChange(
            finding=_make_finding("x"),
            previous_severity="MEDIUM",
            current_severity="CRITICAL",
        )
        diff = ReportDiff(severity_changes=[sc])
        line = diff.summary_line
        assert "1 SEVERITY CHANGED" in line
        assert "MEDIUM\u2192CRITICAL" in line

    def test_summary_line_unchanged(self) -> None:
        diff = ReportDiff(unchanged_findings=[_make_finding("a"), _make_finding("b")])
        assert "2 UNCHANGED" in diff.summary_line

    def test_summary_line_combined(self) -> None:
        diff = ReportDiff(
            new_findings=[_make_finding("n")],
            resolved_findings=[_make_finding("r")],
            unchanged_findings=[_make_finding("u")],
        )
        line = diff.summary_line
        assert "+1 NEW" in line
        assert "1 RESOLVED" in line
        assert "1 UNCHANGED" in line
        assert " | " in line


# ══════════════════════════════════════════════════════════════
# print_watch_diff_summary
# ══════════════════════════════════════════════════════════════


class TestPrintWatchDiffSummary:
    """Tests for the display function."""

    def test_renders_without_error(self) -> None:
        """Diff panel renders without exceptions."""
        diff = ReportDiff(
            new_findings=[_make_finding("new-svc", title="New Service Down", severity=Severity.CRITICAL)],
            resolved_findings=[_make_finding("old-svc", title="Old Issue Fixed")],
            unchanged_findings=[_make_finding("stable-svc", title="Stable Service")],
        )
        output = _capture_diff_panel(diff)
        assert output  # Non-empty

    def test_changes_detected_title(self) -> None:
        """Panel title says 'Changes Detected' when diff has changes."""
        diff = ReportDiff(new_findings=[_make_finding("x")])
        output = _capture_diff_panel(diff, iteration=3)
        assert "Changes Detected" in output
        assert "Iteration #3" in output

    def test_no_changes_title(self) -> None:
        """Panel title says 'No Changes' when only unchanged findings exist."""
        diff = ReportDiff(unchanged_findings=[_make_finding("x")])
        output = _capture_diff_panel(diff, iteration=5)
        assert "No Changes" in output
        assert "Iteration #5" in output

    def test_new_findings_listed(self) -> None:
        """New findings show with severity emoji and title."""
        diff = ReportDiff(
            new_findings=[_make_finding("svc-crash", title="Payment Service Crash", severity=Severity.CRITICAL)],
        )
        output = _capture_diff_panel(diff)
        assert "NEW" in output
        assert "Payment Service Crash" in output

    def test_resolved_findings_listed(self) -> None:
        """Resolved findings show title."""
        diff = ReportDiff(
            resolved_findings=[_make_finding("old-issue", title="Memory Leak Fixed")],
        )
        output = _capture_diff_panel(diff)
        assert "RESOLVED" in output
        assert "Memory Leak Fixed" in output

    def test_severity_changes_listed(self) -> None:
        """Severity changes show old -> new severity."""
        sc = SeverityChange(
            finding=_make_finding("x", title="Disk Pressure"),
            previous_severity="MEDIUM",
            current_severity="CRITICAL",
        )
        diff = ReportDiff(severity_changes=[sc])
        output = _capture_diff_panel(diff)
        assert "SEVERITY CHANGED" in output
        assert "Disk Pressure" in output
        assert "MEDIUM" in output
        assert "CRITICAL" in output

    def test_unchanged_count_shown(self) -> None:
        """Unchanged section shows count."""
        diff = ReportDiff(
            unchanged_findings=[_make_finding("a"), _make_finding("b"), _make_finding("c")],
        )
        output = _capture_diff_panel(diff)
        assert "UNCHANGED" in output
        assert "(3)" in output

    def test_empty_diff_renders(self) -> None:
        """Empty diff produces a panel (edge case — no findings at all)."""
        diff = ReportDiff()
        output = _capture_diff_panel(diff)
        assert "No Changes" in output


# ══════════════════════════════════════════════════════════════
# Watch loop diff integration
# ══════════════════════════════════════════════════════════════


class TestWatchLoopDiffIntegration:
    """Tests that _run_watch_loop prints diff summary on iteration 2+."""

    @patch("vaig.cli.commands.live.print_watch_diff_summary")
    @patch("vaig.cli.commands.live.compute_report_diff")
    @patch("vaig.cli.commands.live.time.sleep")
    @patch("vaig.cli.commands.live.console")
    def test_diff_printed_on_iteration_2(
        self,
        mock_console: MagicMock,
        mock_sleep: MagicMock,
        mock_compute: MagicMock,
        mock_print_diff: MagicMock,
    ) -> None:
        """Diff summary is computed and printed starting at iteration 2."""
        report_1 = _make_report(_make_finding("svc-a"))
        report_2 = _make_report(_make_finding("svc-a"), _make_finding("svc-b"))

        call_count = 0

        def run_fn() -> HealthReport | None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return report_1
            return report_2

        # Allow 2 iterations, then interrupt
        sleep_calls = 0

        def sleep_side_effect(seconds: int) -> None:
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls >= 2:
                raise KeyboardInterrupt

        mock_sleep.side_effect = sleep_side_effect
        mock_compute.return_value = ReportDiff(
            new_findings=[_make_finding("svc-b")],
        )

        _run_watch_loop(
            run_fn=run_fn,
            interval=10,
            question="Check health",
        )

        # compute_report_diff should be called once (iteration 2)
        mock_compute.assert_called_once_with(report_2, report_1)
        mock_print_diff.assert_called_once()

    @patch("vaig.cli.commands.live.print_watch_diff_summary")
    @patch("vaig.cli.commands.live.compute_report_diff")
    @patch("vaig.cli.commands.live.time.sleep")
    @patch("vaig.cli.commands.live.console")
    def test_no_diff_on_iteration_1(
        self,
        mock_console: MagicMock,
        mock_sleep: MagicMock,
        mock_compute: MagicMock,
        mock_print_diff: MagicMock,
    ) -> None:
        """No diff computed or printed on iteration 1 (no previous report)."""
        report = _make_report(_make_finding("svc-a"))
        mock_sleep.side_effect = KeyboardInterrupt

        _run_watch_loop(
            run_fn=lambda: report,
            interval=10,
            question="Check health",
        )

        mock_compute.assert_not_called()
        mock_print_diff.assert_not_called()

    @patch("vaig.cli.commands.live.print_watch_diff_summary")
    @patch("vaig.cli.commands.live.compute_report_diff")
    @patch("vaig.cli.commands.live.time.sleep")
    @patch("vaig.cli.commands.live.console")
    def test_no_diff_when_run_fn_returns_none(
        self,
        mock_console: MagicMock,
        mock_sleep: MagicMock,
        mock_compute: MagicMock,
        mock_print_diff: MagicMock,
    ) -> None:
        """InfraAgent path returns None — no diff computed."""
        sleep_calls = 0

        def sleep_side_effect(seconds: int) -> None:
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls >= 2:
                raise KeyboardInterrupt

        mock_sleep.side_effect = sleep_side_effect

        _run_watch_loop(
            run_fn=lambda: None,
            interval=10,
            question="Check infra",
        )

        mock_compute.assert_not_called()
        mock_print_diff.assert_not_called()

    @patch("vaig.cli.commands.live.print_watch_diff_summary")
    @patch("vaig.cli.commands.live.compute_report_diff")
    @patch("vaig.cli.commands.live.time.sleep")
    @patch("vaig.cli.commands.live.console")
    def test_error_iteration_does_not_break_diff_tracking(
        self,
        mock_console: MagicMock,
        mock_sleep: MagicMock,
        mock_compute: MagicMock,
        mock_print_diff: MagicMock,
    ) -> None:
        """If iteration 2 raises typer.Exit, diff is skipped but previous_report is preserved."""
        report_1 = _make_report(_make_finding("svc-a"))
        report_3 = _make_report(_make_finding("svc-a"), _make_finding("svc-c"))

        call_count = 0

        def run_fn() -> HealthReport | None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return report_1
            if call_count == 2:
                raise typer.Exit(1)
            return report_3

        sleep_calls = 0

        def sleep_side_effect(seconds: int) -> None:
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls >= 3:
                raise KeyboardInterrupt

        mock_sleep.side_effect = sleep_side_effect
        mock_compute.return_value = ReportDiff(
            new_findings=[_make_finding("svc-c")],
        )

        _run_watch_loop(
            run_fn=run_fn,
            interval=10,
            question="Check health",
        )

        # Iteration 2 failed (typer.Exit) → no diff call
        # Iteration 3 succeeds → diff computed against report_1 (still the latest good one)
        assert mock_compute.call_count == 1
        mock_compute.assert_called_once_with(report_3, report_1)

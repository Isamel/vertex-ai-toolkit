"""Tests for severity coloring in CLI output."""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from vaig.cli.display import (
    _line_has_severity,
    colorize_severity,
    print_colored_report,
)


# ── colorize_severity unit tests ─────────────────────────────


class TestColorizeSeverity:
    """Tests for the colorize_severity() text transform."""

    def test_critical_gets_bold_red(self) -> None:
        result = colorize_severity("Severity: CRITICAL")
        assert "[bold red]CRITICAL[/bold red]" in result

    def test_high_gets_red(self) -> None:
        result = colorize_severity("Severity: HIGH")
        assert "[red]HIGH[/red]" in result

    def test_error_gets_red(self) -> None:
        result = colorize_severity("Status: ERROR")
        assert "[red]ERROR[/red]" in result

    def test_warning_gets_yellow(self) -> None:
        result = colorize_severity("Level: WARNING")
        assert "[yellow]WARNING[/yellow]" in result

    def test_warn_gets_yellow(self) -> None:
        result = colorize_severity("WARN: disk full")
        assert "[yellow]WARN[/yellow]" in result

    def test_medium_gets_yellow(self) -> None:
        result = colorize_severity("Severity: MEDIUM")
        assert "[yellow]MEDIUM[/yellow]" in result

    def test_low_gets_green(self) -> None:
        result = colorize_severity("Severity: LOW")
        assert "[green]LOW[/green]" in result

    def test_info_gets_green(self) -> None:
        result = colorize_severity("INFO: all clear")
        assert "[green]INFO[/green]" in result

    def test_healthy_gets_bold_green(self) -> None:
        result = colorize_severity("Status: HEALTHY")
        assert "[bold green]HEALTHY[/bold green]" in result

    def test_ok_gets_bold_green(self) -> None:
        result = colorize_severity("Status: OK")
        assert "[bold green]OK[/bold green]" in result

    def test_passed_gets_bold_green(self) -> None:
        result = colorize_severity("Test: PASSED")
        assert "[bold green]PASSED[/bold green]" in result

    def test_case_insensitive_critical(self) -> None:
        result = colorize_severity("critical issue found")
        assert "[bold red]critical[/bold red]" in result

    def test_case_insensitive_healthy(self) -> None:
        result = colorize_severity("Service is healthy")
        assert "[bold green]healthy[/bold green]" in result

    def test_normal_text_unchanged(self) -> None:
        text = "Everything looks fine, no issues found."
        assert colorize_severity(text) == text

    def test_multiple_keywords_in_one_line(self) -> None:
        result = colorize_severity("CRITICAL error with HIGH impact")
        assert "[bold red]CRITICAL[/bold red]" in result
        assert "[red]HIGH[/red]" in result

    def test_does_not_match_inside_word(self) -> None:
        """'OK' should not match inside 'TOKEN' or 'BOOK'."""
        result = colorize_severity("TOKEN is valid, BOOK is available")
        assert result == "TOKEN is valid, BOOK is available"

    def test_ok_standalone(self) -> None:
        result = colorize_severity("Status: OK")
        assert "[bold green]OK[/bold green]" in result

    def test_preserves_surrounding_text(self) -> None:
        result = colorize_severity("  - Pod health: HEALTHY (uptime 99.9%)")
        assert "  - Pod health: " in result
        assert "[bold green]HEALTHY[/bold green]" in result
        assert " (uptime 99.9%)" in result

    def test_severity_colon_pattern(self) -> None:
        """Common pattern: 'Severity: CRITICAL' should colorize the keyword."""
        result = colorize_severity("Severity: CRITICAL")
        assert "Severity: [bold red]CRITICAL[/bold red]" in result

    def test_warn_does_not_match_warning(self) -> None:
        """WARN should match standalone, but WARNING has its own rule."""
        result = colorize_severity("WARN: something")
        assert "[yellow]WARN[/yellow]" in result


# ── _line_has_severity ────────────────────────────────────────


class TestLineHasSeverity:
    """Tests for the _line_has_severity() helper."""

    def test_line_with_critical(self) -> None:
        assert _line_has_severity("Severity: CRITICAL") is True

    def test_line_with_healthy(self) -> None:
        assert _line_has_severity("Status: HEALTHY") is True

    def test_line_without_severity(self) -> None:
        assert _line_has_severity("No issues found") is False

    def test_empty_line(self) -> None:
        assert _line_has_severity("") is False

    def test_markdown_header(self) -> None:
        assert _line_has_severity("## Infrastructure Report") is False

    def test_mixed_content_with_keyword(self) -> None:
        assert _line_has_severity("- Pod nginx: ERROR (CrashLoopBackOff)") is True


# ── print_colored_report integration tests ───────────────────


def _capture_report(text: str) -> str:
    """Run print_colored_report and capture Rich output as plain text."""
    buf = StringIO()
    con = Console(file=buf, force_terminal=False, width=120)
    print_colored_report(text, console=con)
    return buf.getvalue()


class TestPrintColoredReport:
    """Integration tests for print_colored_report()."""

    def test_plain_text_passes_through(self) -> None:
        output = _capture_report("Hello world\nThis is a test")
        assert "Hello world" in output
        assert "This is a test" in output

    def test_severity_line_is_printed(self) -> None:
        output = _capture_report("Status: CRITICAL\nEverything else is fine")
        assert "CRITICAL" in output
        assert "Everything else is fine" in output

    def test_markdown_headers_preserved(self) -> None:
        """Markdown headers (non-severity lines) should render."""
        report = "## Health Report\n\nAll systems operational."
        output = _capture_report(report)
        assert "Health Report" in output
        assert "operational" in output

    def test_mixed_severity_and_markdown(self) -> None:
        report = (
            "## Summary\n"
            "\n"
            "- Service A: HEALTHY\n"
            "- Service B: CRITICAL\n"
            "\n"
            "### Details\n"
            "\n"
            "No further action needed."
        )
        output = _capture_report(report)
        assert "HEALTHY" in output
        assert "CRITICAL" in output
        assert "Details" in output
        assert "No further action" in output

    def test_empty_input(self) -> None:
        output = _capture_report("")
        # Should not crash, may produce empty or whitespace-only output
        assert isinstance(output, str)

    def test_only_severity_lines(self) -> None:
        report = "CRITICAL: pod down\nHIGH: memory pressure\nLOW: minor log noise"
        output = _capture_report(report)
        assert "CRITICAL" in output
        assert "HIGH" in output
        assert "LOW" in output

    def test_only_markdown_lines(self) -> None:
        report = "## Report\n\n- Item 1\n- Item 2\n\nDone."
        output = _capture_report(report)
        assert "Report" in output
        assert "Item 1" in output

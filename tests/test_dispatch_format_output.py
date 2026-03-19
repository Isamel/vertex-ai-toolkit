"""Tests for _dispatch_format_output, _export_html_report output-path support,
and format normalisation in live.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vaig.cli.commands.live import _dispatch_format_output, _export_html_report

# ── Minimal fake OrchestratorResult ─────────────────────────────────────────


def _make_orch_result(structured_report: Any = None, synthesized_output: str = "") -> MagicMock:
    """Return a minimal MagicMock that quacks like OrchestratorResult."""
    result = MagicMock()
    result.structured_report = structured_report
    result.synthesized_output = synthesized_output
    result.total_usage = None
    return result


def _make_consoles() -> tuple[MagicMock, MagicMock]:
    """Return (console, err_console) mock pair."""
    return MagicMock(), MagicMock()


# ── _dispatch_format_output — format normalisation ───────────────────────────


class TestDispatchFormatNormalisation:
    """format_ is normalised (stripped + lowercased) inside _dispatch_format_output."""

    def test_uppercase_html_triggers_html_export(self, tmp_path: Path) -> None:
        """'HTML' (uppercase) is treated as 'html'."""
        report = MagicMock()
        orch = _make_orch_result(structured_report=report)
        console, err_console = _make_consoles()
        out_file = tmp_path / "report.html"

        with patch("vaig.cli.commands.live._export_html_report") as mock_export:
            mock_export.return_value = True
            _dispatch_format_output(
                orch,
                format_="HTML",
                output=out_file,
                question="q",
                model_id="gemini-2.0",
                skill_name="test-skill",
                console=console,
                err_console=err_console,
            )
            mock_export.assert_called_once()

    def test_padded_html_triggers_html_export(self, tmp_path: Path) -> None:
        """'  html  ' (padded) is normalised and treated as 'html'."""
        report = MagicMock()
        orch = _make_orch_result(structured_report=report)
        console, err_console = _make_consoles()
        out_file = tmp_path / "report.html"

        with patch("vaig.cli.commands.live._export_html_report") as mock_export:
            mock_export.return_value = True
            _dispatch_format_output(
                orch,
                format_="  html  ",
                output=out_file,
                question="q",
                model_id="gemini-2.0",
                skill_name="test-skill",
                console=console,
                err_console=err_console,
            )
            mock_export.assert_called_once()

    def test_mixed_case_html_triggers_html_export(self, tmp_path: Path) -> None:
        """'Html' (mixed case) is normalised and treated as 'html'."""
        report = MagicMock()
        orch = _make_orch_result(structured_report=report)
        console, err_console = _make_consoles()

        with patch("vaig.cli.commands.live._export_html_report") as mock_export:
            mock_export.return_value = True
            _dispatch_format_output(
                orch,
                format_="Html",
                output=None,
                question="q",
                model_id="gemini-2.0",
                skill_name="test-skill",
                console=console,
                err_console=err_console,
            )
            mock_export.assert_called_once()


# ── _dispatch_format_output — fallback when structured_report is None ────────


class TestDispatchFallbackNoStructuredReport:
    """When format_='html' but structured_report is None, falls back to text export."""

    def test_fallback_to_handle_export_when_no_report(self) -> None:
        """No structured_report + html format → warning shown + text fallback called."""
        orch = _make_orch_result(structured_report=None, synthesized_output="output text")
        console, err_console = _make_consoles()

        with patch("vaig.cli.commands.live._handle_export_output") as mock_handle:
            _dispatch_format_output(
                orch,
                format_="html",
                output=None,
                question="q?",
                model_id="gemini-2.0",
                skill_name="test-skill",
                console=console,
                err_console=err_console,
            )
            # Warning must be printed to err_console
            err_console.print.assert_called_once()
            # Fallback text export must be invoked
            mock_handle.assert_called_once()

    def test_fallback_does_not_raise(self) -> None:
        """No crash when structured_report is None and format_ is html."""
        orch = _make_orch_result(structured_report=None)
        console, err_console = _make_consoles()

        with patch("vaig.cli.commands.live._handle_export_output"):
            # Should not raise
            _dispatch_format_output(
                orch,
                format_="html",
                output=None,
                question="q",
                model_id="gemini-2.0",
                skill_name="test-skill",
                console=console,
                err_console=err_console,
            )

    def test_fallback_calls_handle_export_with_format_html(self) -> None:
        """Fallback must pass format_='html' to _handle_export_output, not None."""
        orch = _make_orch_result(structured_report=None, synthesized_output="some output")
        console, err_console = _make_consoles()

        with patch("vaig.cli.commands.live._handle_export_output") as mock_handle:
            _dispatch_format_output(
                orch,
                format_="html",
                output=None,
                question="what happened?",
                model_id="gemini-2.0",
                skill_name="health-skill",
                console=console,
                err_console=err_console,
            )
            mock_handle.assert_called_once()
            _kwargs = mock_handle.call_args.kwargs
            assert _kwargs.get("format_") == "html", (
                f"Expected format_='html' in fallback call, got format_={_kwargs.get('format_')!r}"
            )


# ── _export_html_report — --output path respected ────────────────────────────


class TestExportHtmlReportOutputPath:
    """_export_html_report writes to the path supplied via --output."""

    def test_custom_output_path_is_used(self, tmp_path: Path) -> None:
        """When output is given, the HTML is written to that exact path."""
        report = MagicMock()
        console, err_console = _make_consoles()
        out_file = tmp_path / "custom-report.html"

        # render_health_report_html is imported lazily inside _export_html_report;
        # patch it at its actual source module.
        with patch(
            "vaig.ui.html_report.render_health_report_html",
            return_value="<html>stub</html>",
        ):
            _export_html_report(report, console=console, err_console=err_console, output=out_file)

        assert out_file.exists(), "Expected HTML file to be written at the custom path"
        assert out_file.read_text() == "<html>stub</html>"

    def test_timestamped_path_used_when_no_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When output is None, a timestamped filename is generated."""
        report = MagicMock()
        console, err_console = _make_consoles()

        # Change cwd to tmp_path so the auto-generated file lands there
        monkeypatch.chdir(tmp_path)

        with patch(
            "vaig.ui.html_report.render_health_report_html",
            return_value="<html>auto</html>",
        ):
            result = _export_html_report(report, console=console, err_console=err_console, output=None)

        assert result is True
        html_files = list(tmp_path.glob("vaig-report-*.html"))
        assert len(html_files) == 1, f"Expected one auto-named HTML file, found: {html_files}"


# ── _dispatch_format_output — early-exit guard ───────────────────────────────


class TestDispatchEarlyExitGuard:
    """When both format_ and output are None/empty, _dispatch_format_output returns early."""

    def test_no_format_no_output_returns_early(self) -> None:
        """format_=None and output=None → no export called, function returns immediately."""
        orch = _make_orch_result(structured_report=MagicMock(), synthesized_output="text")
        console, err_console = _make_consoles()

        with (
            patch("vaig.cli.commands.live._export_html_report") as mock_html,
            patch("vaig.cli.commands.live._handle_export_output") as mock_handle,
        ):
            _dispatch_format_output(
                orch,
                format_=None,
                output=None,
                question="q",
                model_id="gemini-2.0",
                skill_name="test-skill",
                console=console,
                err_console=err_console,
            )
            mock_html.assert_not_called()
            mock_handle.assert_not_called()

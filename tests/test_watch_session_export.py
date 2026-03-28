"""Tests for watch-session HTML export feature.

Covers:
- ``DiffTimelineEntry.to_dict()`` serialisation (with and without diff)
- ``WatchSessionData.to_dict()`` full session serialisation
- ``render_watch_session_html()`` sentinel replacement
- Watch loop accumulation and HTML export integration
- Edge cases: single iteration, empty diffs, many iterations
"""

from __future__ import annotations

import json
from unittest.mock import patch

import typer

from vaig.cli.commands.live import _run_watch_loop
from vaig.skills.service_health.diff import (
    DiffTimelineEntry,
    ReportDiff,
    SeverityChange,
    WatchSessionData,
)
from vaig.skills.service_health.schema import (
    ExecutiveSummary,
    Finding,
    HealthReport,
    OverallStatus,
    Severity,
)
from vaig.ui.html_report import (
    _SENTINEL,
    _WATCH_SENTINEL,
    render_watch_session_html,
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


# ══════════════════════════════════════════════════════════════
# DiffTimelineEntry.to_dict()
# ══════════════════════════════════════════════════════════════


class TestDiffTimelineEntryToDict:
    """Serialisation of DiffTimelineEntry."""

    def test_baseline_entry_no_diff(self) -> None:
        """Baseline entry serialises with is_baseline=True and diff=None."""
        entry = DiffTimelineEntry(
            iteration=1,
            timestamp="2025-01-15T10:00:00+00:00",
            is_baseline=True,
            diff=None,
        )
        d = entry.to_dict()

        assert d["iteration"] == 1
        assert d["timestamp"] == "2025-01-15T10:00:00+00:00"
        assert d["is_baseline"] is True
        assert d["diff"] is None

    def test_entry_with_diff(self) -> None:
        """Entry with a ReportDiff serialises all diff fields."""
        f_new = _make_finding("new-svc", severity=Severity.HIGH)
        f_resolved = _make_finding("old-svc", severity=Severity.LOW)
        f_changed = _make_finding("changed-svc", severity=Severity.CRITICAL)
        f_unchanged = _make_finding("stable-svc")

        diff = ReportDiff(
            new_findings=[f_new],
            resolved_findings=[f_resolved],
            unchanged_findings=[f_unchanged],
            severity_changes=[
                SeverityChange(
                    finding=f_changed,
                    previous_severity="MEDIUM",
                    current_severity="CRITICAL",
                )
            ],
        )
        entry = DiffTimelineEntry(
            iteration=2,
            timestamp="2025-01-15T10:05:00+00:00",
            is_baseline=False,
            diff=diff,
        )
        d = entry.to_dict()

        assert d["iteration"] == 2
        assert d["is_baseline"] is False
        assert d["diff"] is not None

        dd = d["diff"]
        assert dd["has_changes"] is True
        assert len(dd["new_findings"]) == 1
        assert dd["new_findings"][0]["id"] == "new-svc"
        assert len(dd["resolved_findings"]) == 1
        assert dd["resolved_findings"][0]["id"] == "old-svc"
        assert dd["unchanged_count"] == 1
        assert len(dd["severity_changes"]) == 1
        assert dd["severity_changes"][0]["previous_severity"] == "MEDIUM"
        assert dd["severity_changes"][0]["current_severity"] == "CRITICAL"

    def test_entry_with_empty_diff(self) -> None:
        """Entry where diff has no changes serialises correctly."""
        diff = ReportDiff()
        entry = DiffTimelineEntry(
            iteration=3,
            timestamp="2025-01-15T10:10:00+00:00",
            is_baseline=False,
            diff=diff,
        )
        d = entry.to_dict()

        assert d["diff"]["has_changes"] is False
        assert d["diff"]["new_findings"] == []
        assert d["diff"]["resolved_findings"] == []
        assert d["diff"]["severity_changes"] == []
        assert d["diff"]["unchanged_count"] == 0

    def test_summary_line_present_in_diff_dict(self) -> None:
        """The summary_line string is included in the serialised diff."""
        diff = ReportDiff(new_findings=[_make_finding("x")])
        entry = DiffTimelineEntry(
            iteration=2,
            timestamp="2025-01-15T10:05:00+00:00",
            diff=diff,
        )
        d = entry.to_dict()
        assert "summary_line" in d["diff"]
        assert "+1 NEW" in d["diff"]["summary_line"]


# ══════════════════════════════════════════════════════════════
# WatchSessionData.to_dict()
# ══════════════════════════════════════════════════════════════


class TestWatchSessionDataToDict:
    """Serialisation of the full WatchSessionData."""

    def test_full_session_serialisation(self) -> None:
        """All fields are present and correct in the serialised dict."""
        baseline = DiffTimelineEntry(
            iteration=1,
            timestamp="2025-01-15T10:00:00+00:00",
            is_baseline=True,
        )
        iter2 = DiffTimelineEntry(
            iteration=2,
            timestamp="2025-01-15T10:05:00+00:00",
            diff=ReportDiff(new_findings=[_make_finding("svc-a")]),
        )
        session = WatchSessionData(
            start_time="2025-01-15T10:00:00+00:00",
            end_time="2025-01-15T10:10:00+00:00",
            total_iterations=2,
            interval_seconds=300,
            diff_timeline=[baseline, iter2],
        )
        d = session.to_dict()

        assert d["start_time"] == "2025-01-15T10:00:00+00:00"
        assert d["end_time"] == "2025-01-15T10:10:00+00:00"
        assert d["total_iterations"] == 2
        assert d["interval_seconds"] == 300
        assert len(d["diff_timeline"]) == 2
        assert d["diff_timeline"][0]["is_baseline"] is True
        assert d["diff_timeline"][1]["diff"] is not None

    def test_empty_timeline(self) -> None:
        """A session with no timeline entries serialises cleanly."""
        session = WatchSessionData(
            start_time="2025-01-15T10:00:00+00:00",
            end_time="2025-01-15T10:00:05+00:00",
            total_iterations=0,
            interval_seconds=60,
        )
        d = session.to_dict()
        assert d["diff_timeline"] == []

    def test_json_serialisable(self) -> None:
        """to_dict() output is fully JSON-serialisable."""
        entry = DiffTimelineEntry(
            iteration=1,
            timestamp="2025-01-15T10:00:00+00:00",
            is_baseline=True,
        )
        session = WatchSessionData(
            start_time="2025-01-15T10:00:00+00:00",
            end_time="2025-01-15T10:05:00+00:00",
            total_iterations=1,
            interval_seconds=60,
            diff_timeline=[entry],
        )
        # Should not raise
        result = json.dumps(session.to_dict())
        assert isinstance(result, str)
        assert "2025-01-15" in result


# ══════════════════════════════════════════════════════════════
# render_watch_session_html()
# ══════════════════════════════════════════════════════════════


class TestRenderWatchSessionHtml:
    """Test sentinel replacement in render_watch_session_html()."""

    def test_both_sentinels_replaced(self) -> None:
        """Both the report and watch session sentinels are replaced."""
        report = _make_report(_make_finding("svc-a"))
        session = WatchSessionData(
            start_time="2025-01-15T10:00:00+00:00",
            end_time="2025-01-15T10:05:00+00:00",
            total_iterations=1,
            interval_seconds=60,
            diff_timeline=[
                DiffTimelineEntry(
                    iteration=1,
                    timestamp="2025-01-15T10:00:00+00:00",
                    is_baseline=True,
                )
            ],
        )
        html = render_watch_session_html(report, session)

        # Original sentinels should be gone
        assert _SENTINEL not in html
        assert _WATCH_SENTINEL not in html

        # Report data should be present
        assert "svc-a" in html

        # Session data should be present
        assert "total_iterations" in html
        assert "diff_timeline" in html

    def test_report_data_is_valid_json_in_output(self) -> None:
        """The injected report data is valid JSON."""
        report = _make_report(_make_finding("check-json"))
        session = WatchSessionData(
            start_time="2025-01-15T10:00:00+00:00",
            end_time="2025-01-15T10:05:00+00:00",
            total_iterations=1,
            interval_seconds=60,
        )
        html = render_watch_session_html(report, session)

        # Extract REPORT_DATA assignment
        marker = "const REPORT_DATA = "
        idx = html.index(marker) + len(marker)
        end_idx = html.index(";\n", idx)
        payload = html[idx:end_idx]
        data = json.loads(payload)
        assert data["findings"][0]["id"] == "check-json"

    def test_watch_session_is_valid_json_in_output(self) -> None:
        """The injected watch session data is valid JSON."""
        report = _make_report()
        session = WatchSessionData(
            start_time="2025-01-15T10:00:00+00:00",
            end_time="2025-01-15T10:05:00+00:00",
            total_iterations=3,
            interval_seconds=120,
        )
        html = render_watch_session_html(report, session)

        marker = "const WATCH_SESSION = "
        idx = html.index(marker) + len(marker)
        end_idx = html.index(";\n", idx)
        payload = html[idx:end_idx]
        data = json.loads(payload)
        assert data["total_iterations"] == 3
        assert data["interval_seconds"] == 120

    def test_script_tag_escaping(self) -> None:
        """Dangerous </script> sequences in data are escaped."""
        report = _make_report(
            _make_finding("xss-test", title="payload </script><script>alert(1)")
        )
        session = WatchSessionData(
            start_time="2025-01-15T10:00:00+00:00",
            end_time="2025-01-15T10:05:00+00:00",
            total_iterations=1,
            interval_seconds=60,
        )
        html = render_watch_session_html(report, session)

        # Raw </script> should NOT appear in the output
        assert "</script><script>" not in html
        # But the escaped version should be there
        assert "<\\/script>" in html


# ══════════════════════════════════════════════════════════════
# Watch loop integration
# ══════════════════════════════════════════════════════════════


class TestWatchLoopAccumulation:
    """Test that _run_watch_loop accumulates diff timeline and exports HTML."""

    def test_single_iteration_exports_html(self, tmp_path) -> None:
        """A single iteration produces a baseline-only HTML export."""
        report = _make_report(_make_finding("svc-a"))
        call_count = 0

        def run_fn():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise KeyboardInterrupt
            return report

        out_file = tmp_path / "session.html"

        with patch("vaig.cli.commands.live.time") as mock_time, \
             patch("vaig.cli.commands.live.console") as mock_console:
            mock_time.monotonic.return_value = 100.0
            mock_time.sleep.side_effect = KeyboardInterrupt

            _run_watch_loop(
                run_fn=run_fn,
                interval=60,
                question="test query",
                html_output_path=str(out_file),
            )

        assert out_file.exists()
        html = out_file.read_text()
        assert _SENTINEL not in html
        assert _WATCH_SENTINEL not in html
        assert "svc-a" in html
        assert "diff_timeline" in html

    def test_multiple_iterations_accumulate_diffs(self, tmp_path) -> None:
        """Multiple iterations produce entries in the diff timeline."""
        reports = [
            _make_report(_make_finding("svc-a")),
            _make_report(_make_finding("svc-a"), _make_finding("svc-b")),
            _make_report(_make_finding("svc-b")),  # svc-a resolved
        ]
        call_idx = 0

        def run_fn():
            nonlocal call_idx
            r = reports[call_idx] if call_idx < len(reports) else reports[-1]
            call_idx += 1
            return r

        out_file = tmp_path / "multi.html"

        sleep_count = 0

        def sleep_side_effect(_):
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 3:
                raise KeyboardInterrupt

        with patch("vaig.cli.commands.live.time") as mock_time, \
             patch("vaig.cli.commands.live.console") as mock_console:
            mock_time.monotonic.return_value = 100.0
            mock_time.sleep.side_effect = sleep_side_effect

            _run_watch_loop(
                run_fn=run_fn,
                interval=30,
                question="multi test",
                html_output_path=str(out_file),
            )

        assert out_file.exists()
        html = out_file.read_text()

        # Extract watch session JSON
        marker = "const WATCH_SESSION = "
        idx = html.index(marker) + len(marker)
        end_idx = html.index(";\n", idx)
        session = json.loads(html[idx:end_idx])

        assert session["total_iterations"] == 3
        assert len(session["diff_timeline"]) == 3
        # First is baseline
        assert session["diff_timeline"][0]["is_baseline"] is True
        assert session["diff_timeline"][0]["diff"] is None
        # Second has a new finding
        assert session["diff_timeline"][1]["is_baseline"] is False
        assert session["diff_timeline"][1]["diff"]["has_changes"] is True
        assert len(session["diff_timeline"][1]["diff"]["new_findings"]) == 1
        # Third has resolved + new
        assert session["diff_timeline"][2]["diff"]["has_changes"] is True

    def test_auto_generated_filename(self, tmp_path, monkeypatch) -> None:
        """When html_output_path is None, an auto-generated filename is used."""
        report = _make_report(_make_finding("svc-a"))

        def run_fn():
            return report

        # Change cwd so the auto-generated file ends up in tmp_path
        monkeypatch.chdir(tmp_path)

        with patch("vaig.cli.commands.live.time") as mock_time, \
             patch("vaig.cli.commands.live.console") as mock_console:
            mock_time.monotonic.return_value = 100.0
            mock_time.sleep.side_effect = KeyboardInterrupt

            _run_watch_loop(
                run_fn=run_fn,
                interval=60,
                question="auto filename test",
            )

        # Should have created a file matching vaig-watch-session-*.html
        html_files = list(tmp_path.glob("vaig-watch-session-*.html"))
        assert len(html_files) == 1
        html = html_files[0].read_text()
        assert "svc-a" in html

    def test_no_report_no_export(self, tmp_path) -> None:
        """When run_fn never returns a report, no HTML is exported."""
        def run_fn():
            return None

        out_file = tmp_path / "no-report.html"

        with patch("vaig.cli.commands.live.time") as mock_time, \
             patch("vaig.cli.commands.live.console") as mock_console:
            mock_time.monotonic.return_value = 100.0
            mock_time.sleep.side_effect = KeyboardInterrupt

            _run_watch_loop(
                run_fn=run_fn,
                interval=60,
                question="no report test",
                html_output_path=str(out_file),
            )

        # No report → no HTML file
        assert not out_file.exists()

    def test_iteration_with_error_still_continues(self, tmp_path) -> None:
        """An iteration that raises typer.Exit still accumulates timeline."""
        report = _make_report(_make_finding("survivor"))
        call_idx = 0

        def run_fn():
            nonlocal call_idx
            call_idx += 1
            if call_idx == 2:
                raise typer.Exit(1)
            return report

        out_file = tmp_path / "error-recovery.html"
        sleep_count = 0

        def sleep_side_effect(_):
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 3:
                raise KeyboardInterrupt

        with patch("vaig.cli.commands.live.time") as mock_time, \
             patch("vaig.cli.commands.live.console") as mock_console:
            mock_time.monotonic.return_value = 100.0
            mock_time.sleep.side_effect = sleep_side_effect

            _run_watch_loop(
                run_fn=run_fn,
                interval=30,
                question="error test",
                html_output_path=str(out_file),
            )

        assert out_file.exists()
        html = out_file.read_text()
        # Extract session
        marker = "const WATCH_SESSION = "
        idx = html.index(marker) + len(marker)
        end_idx = html.index(";\n", idx)
        session = json.loads(html[idx:end_idx])
        # All 3 iterations recorded even though #2 errored
        assert session["total_iterations"] == 3
        assert len(session["diff_timeline"]) == 3

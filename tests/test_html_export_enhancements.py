"""Tests for html-export-enhancements-batch1.

Covers:
- _inject_report_metadata: fills empty / N/A fields, never overwrites populated fields
- _export_html_report: --open flag triggers webbrowser.open, uses tempfile when output=None
- _export_html_report: browser failure is handled gracefully (no crash)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from vaig.cli.commands.live import _export_html_report, _inject_report_metadata
from vaig.skills.service_health.schema import ExecutiveSummary, HealthReport, OverallStatus, ReportMetadata

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_summary() -> ExecutiveSummary:
    return ExecutiveSummary(
        overall_status=OverallStatus.HEALTHY,
        scope="Namespace: default",
        summary_text="All services healthy.",
        critical_count=0,
        warning_count=0,
        issues_found=0,
        services_checked=1,
    )


def _make_report(**metadata_kwargs: str) -> HealthReport:
    """Return a minimal HealthReport with optional metadata fields set."""
    return HealthReport(
        executive_summary=_make_summary(),
        metadata=ReportMetadata(**metadata_kwargs),
    )


def _make_gke_config(cluster_name: str = "prod-cluster", project_id: str = "my-project") -> MagicMock:
    cfg = MagicMock()
    cfg.cluster_name = cluster_name
    cfg.project_id = project_id
    return cfg


def _make_consoles() -> tuple[MagicMock, MagicMock]:
    return MagicMock(), MagicMock()


# ── _inject_report_metadata ───────────────────────────────────────────────────


class TestInjectReportMetadata:
    """Unit tests for _inject_report_metadata."""

    def test_fills_empty_cluster_and_project(self) -> None:
        """Empty cluster_name / project_id are filled from gke_config."""
        report = _make_report()
        gke = _make_gke_config(cluster_name="cluster-a", project_id="proj-a")

        _inject_report_metadata(report, gke_config=gke, model_id="gemini-2.0")

        assert report.metadata.cluster_name == "cluster-a"
        assert report.metadata.project_id == "proj-a"
        assert report.metadata.model_used == "gemini-2.0"

    def test_fills_empty_model_used(self) -> None:
        """Empty model_used is filled from model_id argument."""
        report = _make_report()

        _inject_report_metadata(report, gke_config=None, model_id="gemini-pro")

        assert report.metadata.model_used == "gemini-pro"

    def test_no_overwrite_populated_cluster_name(self) -> None:
        """Existing non-empty cluster_name is preserved."""
        report = _make_report(cluster_name="existing-cluster")
        gke = _make_gke_config(cluster_name="different-cluster")

        _inject_report_metadata(report, gke_config=gke)

        assert report.metadata.cluster_name == "existing-cluster"

    def test_no_overwrite_populated_project_id(self) -> None:
        """Existing non-empty project_id is preserved."""
        report = _make_report(project_id="existing-project")
        gke = _make_gke_config(project_id="other-project")

        _inject_report_metadata(report, gke_config=gke)

        assert report.metadata.project_id == "existing-project"

    def test_no_overwrite_populated_model_used(self) -> None:
        """Existing non-empty model_used is preserved."""
        report = _make_report(model_used="already-set")

        _inject_report_metadata(report, gke_config=None, model_id="new-model")

        assert report.metadata.model_used == "already-set"

    def test_na_sentinel_overwritten_cluster(self) -> None:
        """cluster_name of 'N/A' is treated as empty and overwritten."""
        report = _make_report(cluster_name="N/A")
        gke = _make_gke_config(cluster_name="real-cluster")

        _inject_report_metadata(report, gke_config=gke)

        assert report.metadata.cluster_name == "real-cluster"

    def test_na_lowercase_sentinel_overwritten(self) -> None:
        """'n/a' (lowercase) is also treated as the N/A sentinel."""
        report = _make_report(project_id="n/a")
        gke = _make_gke_config(project_id="real-project")

        _inject_report_metadata(report, gke_config=gke)

        assert report.metadata.project_id == "real-project"

    def test_no_gke_config_no_crash(self) -> None:
        """Calling with gke_config=None should not raise."""
        report = _make_report()

        _inject_report_metadata(report, gke_config=None, model_id="m")

        assert report.metadata.model_used == "m"

    def test_no_metadata_attribute_no_crash(self) -> None:
        """If the report has no metadata attribute, no exception is raised."""
        obj = MagicMock(spec=[])  # no attributes

        # Should return silently without raising
        _inject_report_metadata(obj, gke_config=None, model_id="m")


# ── _export_html_report — open_browser ───────────────────────────────────────


class TestExportHtmlReportOpenBrowser:
    """Tests for the open_browser flag in _export_html_report."""

    def test_open_browser_calls_webbrowser_open(self, tmp_path: Path) -> None:
        """When open_browser=True, webbrowser.open is called with a file:// URL."""
        report = _make_report()
        console, err_console = _make_consoles()
        out_file = tmp_path / "report.html"

        with (
            patch("vaig.cli.commands.live.webbrowser.open", return_value=True) as mock_wb,
            patch("vaig.ui.html_report.render_health_report_html", return_value="<html/>"),
        ):
            result = _export_html_report(report, console=console, err_console=err_console, output=out_file, open_browser=True)

        assert result is True
        mock_wb.assert_called_once()
        url_arg = mock_wb.call_args[0][0]
        assert url_arg.startswith("file://")
        assert "report.html" in url_arg

    def test_open_browser_false_does_not_call_webbrowser(self, tmp_path: Path) -> None:
        """When open_browser=False (default), webbrowser.open is never called."""
        report = _make_report()
        console, err_console = _make_consoles()
        out_file = tmp_path / "report.html"

        with (
            patch("vaig.cli.commands.live.webbrowser.open") as mock_wb,
            patch("vaig.ui.html_report.render_health_report_html", return_value="<html/>"),
        ):
            _export_html_report(report, console=console, err_console=err_console, output=out_file, open_browser=False)

        mock_wb.assert_not_called()

    def test_open_browser_uses_tempfile_when_no_output(self) -> None:
        """When output=None and open_browser=True, a temp file is used."""
        report = _make_report()
        console, err_console = _make_consoles()

        captured_paths: list[Path] = []

        def fake_write(path: Path, content: str, **_kwargs: object) -> None:  # noqa: ARG001
            captured_paths.append(path)

        with (
            patch("vaig.cli.commands.live.webbrowser.open", return_value=True),
            patch("vaig.ui.html_report.render_health_report_html", return_value="<html/>"),
            patch("vaig.cli.commands.live.Path.write_text", new=fake_write),
        ):
            result = _export_html_report(report, console=console, err_console=err_console, output=None, open_browser=True)

        assert result is True
        # The function should have written to a temp file, not the cwd
        written = console.print.call_args_list
        # Check that the success message was printed (temp path contains "tmp" or is absolute)
        assert written  # at least one print happened

    def test_open_browser_tempfile_path_is_temporary(self, tmp_path: Path) -> None:
        """When output=None and open_browser=True, the generated path is in the temp dir."""
        import tempfile as _tempfile

        report = _make_report()
        console, err_console = _make_consoles()

        tmp_dir = str(_tempfile.gettempdir())
        written_paths: list[str] = []

        original_write_text = Path.write_text

        def capturing_write_text(self: Path, content: str, *args: object, **kwargs: object) -> None:
            written_paths.append(str(self))
            # Don't actually write to FS to keep tests hermetic — use original
            original_write_text(self, content, *args, **kwargs)

        with (
            patch("vaig.cli.commands.live.webbrowser.open", return_value=True),
            patch("vaig.ui.html_report.render_health_report_html", return_value="<html/>"),
            patch.object(Path, "write_text", capturing_write_text),
        ):
            _export_html_report(report, console=console, err_console=err_console, output=None, open_browser=True)

        assert len(written_paths) == 1
        assert written_paths[0].startswith(tmp_dir)

    def test_open_browser_webbrowser_returns_false_prints_warning(self, tmp_path: Path) -> None:
        """When webbrowser.open returns False, a warning is printed (no crash)."""
        report = _make_report()
        console, err_console = _make_consoles()
        out_file = tmp_path / "report.html"

        with (
            patch("vaig.cli.commands.live.webbrowser.open", return_value=False),
            patch("vaig.ui.html_report.render_health_report_html", return_value="<html/>"),
        ):
            result = _export_html_report(report, console=console, err_console=err_console, output=out_file, open_browser=True)

        assert result is True
        # A warning message should have been printed to console
        all_prints = " ".join(str(c) for c in console.print.call_args_list)
        assert "open manually" in all_prints.lower() or "could not open" in all_prints.lower()

    def test_open_browser_exception_handled_gracefully(self, tmp_path: Path) -> None:
        """When webbrowser.open raises, the function still returns True (no crash)."""
        report = _make_report()
        console, err_console = _make_consoles()
        out_file = tmp_path / "report.html"

        with (
            patch("vaig.cli.commands.live.webbrowser.open", side_effect=OSError("no browser")),
            patch("vaig.ui.html_report.render_health_report_html", return_value="<html/>"),
        ):
            result = _export_html_report(report, console=console, err_console=err_console, output=out_file, open_browser=True)

        assert result is True
        all_prints = " ".join(str(c) for c in console.print.call_args_list)
        assert "no browser" in all_prints or "open manually" in all_prints.lower()

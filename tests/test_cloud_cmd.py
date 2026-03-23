"""Tests for `vaig cloud push` CLI subcommands (cloud_cmd.py)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from vaig.cli.app import app
from vaig.cli.commands.cloud_cmd import _parse_since
from vaig.core.config import ExportConfig, Settings

runner = CliRunner()


# ── Shared fixtures ───────────────────────────────────────────


@pytest.fixture()
def _export_enabled_settings() -> Settings:
    """Settings with export enabled and fake GCP coordinates."""
    s = Settings()
    s.export = ExportConfig(
        enabled=True,
        gcp_project_id="test-project",
        bigquery_dataset="test_dataset",
        gcs_bucket="test-bucket",
        gcs_prefix="vaig/",
    )
    return s


@pytest.fixture()
def _export_disabled_settings() -> Settings:
    """Settings with export explicitly disabled."""
    s = Settings()
    s.export = ExportConfig(enabled=False)
    return s


@pytest.fixture()
def mock_exporter() -> MagicMock:
    """A mock DataExporter that returns sensible defaults for all export methods."""
    m = MagicMock(name="DataExporter")
    m.export_telemetry_to_bigquery.return_value = 5
    m.export_telemetry_to_gcs.return_value = "gs://test-bucket/telemetry/batch.jsonl"
    m.export_tool_calls_to_bigquery.return_value = 3
    m.export_tool_results_to_gcs.return_value = "gs://test-bucket/tool_results/run.jsonl"
    m.export_report_to_bigquery.return_value = True
    m.export_report_to_gcs.return_value = "gs://test-bucket/reports/run.json"
    return m


# ── _parse_since ──────────────────────────────────────────────


class TestParseSince:
    """Unit tests for the _parse_since helper."""

    def test_parse_7d(self) -> None:
        now = datetime.now(UTC)
        result = _parse_since("7d")
        expected = now - timedelta(days=7)
        assert abs((result - expected).total_seconds()) < 2  # noqa: PLR2004

    def test_parse_30d(self) -> None:
        now = datetime.now(UTC)
        result = _parse_since("30d")
        expected = now - timedelta(days=30)
        assert abs((result - expected).total_seconds()) < 2  # noqa: PLR2004

    def test_parse_1h(self) -> None:
        now = datetime.now(UTC)
        result = _parse_since("1h")
        expected = now - timedelta(hours=1)
        assert abs((result - expected).total_seconds()) < 2  # noqa: PLR2004

    def test_parse_2w(self) -> None:
        now = datetime.now(UTC)
        result = _parse_since("2w")
        expected = now - timedelta(weeks=2)
        assert abs((result - expected).total_seconds()) < 2  # noqa: PLR2004

    def test_parse_1m(self) -> None:
        now = datetime.now(UTC)
        result = _parse_since("1m")
        expected = now - timedelta(days=30)
        assert abs((result - expected).total_seconds()) < 2  # noqa: PLR2004

    def test_parse_invalid_raises(self) -> None:
        with pytest.raises(typer.BadParameter, match="Invalid time format"):
            _parse_since("7x")

    def test_parse_invalid_no_unit_raises(self) -> None:
        with pytest.raises(typer.BadParameter, match="Invalid time format"):
            _parse_since("30")

    def test_parse_invalid_empty_raises(self) -> None:
        with pytest.raises(typer.BadParameter, match="Invalid time format"):
            _parse_since("")

    def test_parse_invalid_text_raises(self) -> None:
        with pytest.raises(typer.BadParameter, match="Invalid time format"):
            _parse_since("last week")


# ── cloud status ──────────────────────────────────────────────


class TestCloudStatus:
    def test_cloud_status_stub(self) -> None:
        """vaig cloud status prints 'coming in a future release'."""
        result = runner.invoke(app, ["cloud", "status"])
        assert result.exit_code == 0
        assert "coming in a future release" in result.output


# ── cloud push help ───────────────────────────────────────────


class TestCloudPushHelp:
    def test_cloud_push_help_shows_subcommands(self) -> None:
        """vaig cloud push --help should list telemetry, tool-calls, reports."""
        result = runner.invoke(app, ["cloud", "push", "--help"])
        assert result.exit_code == 0
        assert "telemetry" in result.output
        assert "tool-calls" in result.output
        assert "reports" in result.output

    def test_cloud_help_shows_push(self) -> None:
        """vaig cloud --help should show the push sub-group."""
        result = runner.invoke(app, ["cloud", "--help"])
        assert result.exit_code == 0
        assert "push" in result.output


# ── push telemetry ────────────────────────────────────────────


class TestPushTelemetry:
    """Tests for `vaig cloud push telemetry`."""

    def test_push_telemetry_dry_run(
        self, _export_enabled_settings: Settings
    ) -> None:
        """Dry-run shows summary without actually calling the exporter."""
        fake_records: list[dict[str, Any]] = [
            {"event_type": "tool_call", "timestamp": "2026-01-01T00:00:00"},
        ] * 10

        with (
            patch("vaig.cli._helpers._get_settings", return_value=_export_enabled_settings),
            patch("vaig.core.telemetry.get_telemetry_collector") as mock_collector_factory,
        ):
            mock_collector = MagicMock()
            mock_collector.query_events.return_value = fake_records
            mock_collector_factory.return_value = mock_collector

            result = runner.invoke(app, ["cloud", "push", "telemetry", "--dry-run"])

        assert result.exit_code == 0
        assert "Dry Run" in result.output
        assert "10" in result.output
        # No exporter should be instantiated on dry-run — telemetry method not called
        assert "Exporting" not in result.output

    def test_push_telemetry_export_disabled(
        self, _export_disabled_settings: Settings
    ) -> None:
        """When export is disabled, command exits with code 1 and shows error."""
        with patch("vaig.cli._helpers._get_settings", return_value=_export_disabled_settings):
            result = runner.invoke(app, ["cloud", "push", "telemetry"])

        assert result.exit_code == 1
        assert "disabled" in result.output.lower() or "disabled" in (result.stderr or "").lower()

    def test_push_telemetry_no_records(
        self, _export_enabled_settings: Settings
    ) -> None:
        """When no records are found, a yellow warning is printed and exit_code=0."""
        with (
            patch("vaig.cli._helpers._get_settings", return_value=_export_enabled_settings),
            patch("vaig.core.telemetry.get_telemetry_collector") as mock_collector_factory,
        ):
            mock_collector = MagicMock()
            mock_collector.query_events.return_value = []
            mock_collector_factory.return_value = mock_collector

            result = runner.invoke(app, ["cloud", "push", "telemetry", "--since", "1h"])

        assert result.exit_code == 0
        assert "No telemetry" in result.output

    def test_push_telemetry_to_bigquery_only(
        self, _export_enabled_settings: Settings, mock_exporter: MagicMock
    ) -> None:
        """--dest bigquery calls only BigQuery export, not GCS."""
        fake_records = [{"event_type": "tool_call"}] * 3

        with (
            patch("vaig.cli._helpers._get_settings", return_value=_export_enabled_settings),
            patch("vaig.core.telemetry.get_telemetry_collector") as mock_collector_factory,
            patch("vaig.core.export.DataExporter", return_value=mock_exporter),
        ):
            mock_collector = MagicMock()
            mock_collector.query_events.return_value = fake_records
            mock_collector_factory.return_value = mock_collector

            result = runner.invoke(app, ["cloud", "push", "telemetry", "--dest", "bigquery"])

        assert result.exit_code == 0
        mock_exporter.export_telemetry_to_bigquery.assert_called_once()
        mock_exporter.export_telemetry_to_gcs.assert_not_called()

    def test_push_telemetry_to_gcs_only(
        self, _export_enabled_settings: Settings, mock_exporter: MagicMock
    ) -> None:
        """--dest gcs calls only GCS export, not BigQuery."""
        fake_records = [{"event_type": "tool_call"}] * 3

        with (
            patch("vaig.cli._helpers._get_settings", return_value=_export_enabled_settings),
            patch("vaig.core.telemetry.get_telemetry_collector") as mock_collector_factory,
            patch("vaig.core.export.DataExporter", return_value=mock_exporter),
        ):
            mock_collector = MagicMock()
            mock_collector.query_events.return_value = fake_records
            mock_collector_factory.return_value = mock_collector

            result = runner.invoke(app, ["cloud", "push", "telemetry", "--dest", "gcs"])

        assert result.exit_code == 0
        mock_exporter.export_telemetry_to_gcs.assert_called_once()
        mock_exporter.export_telemetry_to_bigquery.assert_not_called()

    def test_push_telemetry_invalid_destination(
        self, _export_enabled_settings: Settings
    ) -> None:
        """Invalid --dest value exits with non-zero code."""
        with patch("vaig.cli._helpers._get_settings", return_value=_export_enabled_settings):
            result = runner.invoke(app, ["cloud", "push", "telemetry", "--dest", "s3"])

        assert result.exit_code != 0


# ── push tool-calls ───────────────────────────────────────────


class TestPushToolCalls:
    """Tests for `vaig cloud push tool-calls`."""

    def test_push_tool_calls_dry_run(
        self, _export_enabled_settings: Settings
    ) -> None:
        """Dry-run shows summary without exporting."""
        fake_records = [{"tool_name": "get_pods", "run_id": "run-001"}] * 5

        with (
            patch("vaig.cli._helpers._get_settings", return_value=_export_enabled_settings),
            patch("vaig.core.tool_call_store.ToolCallStore") as mock_store_cls,
        ):
            mock_store = MagicMock()
            mock_store.read_records.return_value = fake_records
            mock_store_cls.return_value = mock_store

            result = runner.invoke(app, ["cloud", "push", "tool-calls", "--dry-run"])

        assert result.exit_code == 0
        assert "Dry Run" in result.output
        assert "5" in result.output

    def test_push_tool_calls_with_run_id(
        self, _export_enabled_settings: Settings, mock_exporter: MagicMock
    ) -> None:
        """--run-id is passed through to ToolCallStore.read_records."""
        fake_records = [{"tool_name": "get_logs", "run_id": "my-run-123"}] * 2

        with (
            patch("vaig.cli._helpers._get_settings", return_value=_export_enabled_settings),
            patch("vaig.core.tool_call_store.ToolCallStore") as mock_store_cls,
            patch("vaig.core.export.DataExporter", return_value=mock_exporter),
        ):
            mock_store = MagicMock()
            mock_store.read_records.return_value = fake_records
            mock_store_cls.return_value = mock_store

            result = runner.invoke(
                app, ["cloud", "push", "tool-calls", "--run-id", "my-run-123"]
            )

        assert result.exit_code == 0
        mock_store.read_records.assert_called_once_with(run_id="my-run-123", since=None)

    def test_push_tool_calls_export_disabled(
        self, _export_disabled_settings: Settings
    ) -> None:
        """When export is disabled, command exits with code 1."""
        with patch("vaig.cli._helpers._get_settings", return_value=_export_disabled_settings):
            result = runner.invoke(app, ["cloud", "push", "tool-calls"])

        assert result.exit_code == 1

    def test_push_tool_calls_no_records(
        self, _export_enabled_settings: Settings
    ) -> None:
        """When no records found, prints yellow warning."""
        with (
            patch("vaig.cli._helpers._get_settings", return_value=_export_enabled_settings),
            patch("vaig.core.tool_call_store.ToolCallStore") as mock_store_cls,
        ):
            mock_store = MagicMock()
            mock_store.read_records.return_value = []
            mock_store_cls.return_value = mock_store

            result = runner.invoke(app, ["cloud", "push", "tool-calls"])

        assert result.exit_code == 0
        assert "No tool call records" in result.output


# ── push reports ──────────────────────────────────────────────


class TestPushReports:
    """Tests for `vaig cloud push reports`."""

    def test_push_reports_dry_run(
        self, _export_enabled_settings: Settings, tmp_path: pytest.fixture
    ) -> None:
        """Dry-run shows report count without exporting."""
        import json
        from pathlib import Path

        # Create fake reports in a temp directory that mimics ~/.vaig/reports/
        vaig_reports = tmp_path / ".vaig" / "reports"
        vaig_reports.mkdir(parents=True)
        (vaig_reports / "run-001.json").write_text(json.dumps({"summary": "ok"}), encoding="utf-8")
        (vaig_reports / "run-002.json").write_text(json.dumps({"summary": "warn"}), encoding="utf-8")

        with (
            patch("vaig.cli._helpers._get_settings", return_value=_export_enabled_settings),
            patch.object(Path, "home", return_value=tmp_path),
        ):
            result = runner.invoke(app, ["cloud", "push", "reports", "--dry-run"])

        assert result.exit_code == 0
        assert "Dry Run" in result.output

    def test_push_reports_export_disabled(
        self, _export_disabled_settings: Settings
    ) -> None:
        """When export is disabled, command exits with code 1."""
        with patch("vaig.cli._helpers._get_settings", return_value=_export_disabled_settings):
            result = runner.invoke(app, ["cloud", "push", "reports"])

        assert result.exit_code == 1

    def test_push_reports_no_reports(
        self, _export_enabled_settings: Settings, tmp_path: pytest.fixture
    ) -> None:
        """When no reports exist, prints yellow warning."""
        from pathlib import Path

        # Point home to an empty tmp_path (no .vaig/reports dir)
        with (
            patch("vaig.cli._helpers._get_settings", return_value=_export_enabled_settings),
            patch.object(Path, "home", return_value=tmp_path),
        ):
            result = runner.invoke(app, ["cloud", "push", "reports"])

        assert result.exit_code == 0
        assert "No health reports" in result.output

"""Tests for the `vaig stats` CLI commands (show, export, clear)."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from vaig.cli.app import app
from vaig.core.config import Settings
from vaig.core.telemetry import TelemetryCollector, TelemetryEvent


runner = CliRunner()


# ── Fixtures ──────────────────────────────────────────────────
# db_path and collector are provided by conftest.py


@pytest.fixture()
def _populated_collector(collector: TelemetryCollector) -> TelemetryCollector:
    """Collector with a mix of event types already persisted."""
    collector.emit_tool_call("get_pods")
    collector.emit_tool_call("get_pods")
    collector.emit_tool_call("get_logs")
    collector.emit_api_call(
        "gemini-2.5-pro", tokens_in=500, tokens_out=200, cost_usd=0.01,
    )
    collector.emit_api_call(
        "gemini-2.5-flash", tokens_in=100, tokens_out=50, cost_usd=0.001,
    )
    collector.emit_error("TimeoutError", "timed out")
    collector.emit_cli_command("ask")
    collector.emit_skill_use("rca")
    collector.flush()
    return collector


@pytest.fixture(autouse=True)
def _mock_settings_and_collector(collector: TelemetryCollector) -> None:
    """Patch _get_settings and get_telemetry_collector for all stats tests."""
    settings = Settings()
    with (
        patch("vaig.cli.app._get_settings", return_value=settings),
        patch("vaig.core.telemetry.get_telemetry_collector", return_value=collector),
    ):
        yield


# ══════════════════════════════════════════════════════════════
# stats show
# ══════════════════════════════════════════════════════════════


class TestStatsShow:
    """Tests for `vaig stats show`."""

    def test_show_no_events(self) -> None:
        """With no events, prints a friendly message."""
        result = runner.invoke(app, ["stats", "show"])
        assert result.exit_code == 0
        assert "No telemetry events" in result.output

    def test_show_with_events(self, _populated_collector: TelemetryCollector) -> None:
        """With events, displays summary tables."""
        result = runner.invoke(app, ["stats", "show"])
        assert result.exit_code == 0
        assert "Telemetry Summary" in result.output
        assert "Events by Type" in result.output
        assert "tool_call" in result.output
        assert "api_call" in result.output
        assert "Top 10 Tools" in result.output
        assert "get_pods" in result.output
        assert "API Usage" in result.output

    def test_show_with_since_filter(
        self, _populated_collector: TelemetryCollector,
    ) -> None:
        """--since filter narrows results."""
        # All events are recent, so since=far-future should yield nothing
        result = runner.invoke(app, ["stats", "show", "--since", "2099-01-01T00:00:00"])
        assert result.exit_code == 0
        assert "No telemetry events" in result.output


# ══════════════════════════════════════════════════════════════
# stats export
# ══════════════════════════════════════════════════════════════


class TestStatsExport:
    """Tests for `vaig stats export`."""

    def test_export_no_events(self) -> None:
        """With no matching events, prints a message."""
        result = runner.invoke(app, ["stats", "export"])
        assert result.exit_code == 0
        assert "No events found" in result.output

    def test_export_jsonl(self, _populated_collector: TelemetryCollector) -> None:
        """JSONL export produces valid JSON on each line."""
        result = runner.invoke(app, ["stats", "export", "--format", "jsonl"])
        assert result.exit_code == 0

        lines = [ln for ln in result.output.strip().split("\n") if ln.strip()]
        assert len(lines) >= 1

        for line in lines:
            obj = json.loads(line)
            assert "event_type" in obj

    def test_export_csv(self, _populated_collector: TelemetryCollector) -> None:
        """CSV export has a header row and data rows."""
        result = runner.invoke(app, ["stats", "export", "--format", "csv"])
        assert result.exit_code == 0

        reader = csv.DictReader(io.StringIO(result.output))
        rows = list(reader)
        assert len(rows) >= 1
        assert "event_type" in reader.fieldnames  # type: ignore[operator]

    def test_export_to_file(
        self,
        _populated_collector: TelemetryCollector,
        tmp_path: Path,
    ) -> None:
        """--output writes to a file instead of stdout."""
        out_file = tmp_path / "export.jsonl"
        result = runner.invoke(
            app, ["stats", "export", "--format", "jsonl", "--output", str(out_file)],
        )
        assert result.exit_code == 0
        assert "Exported" in result.output
        assert out_file.exists()

        # Verify file content is valid JSONL
        content = out_file.read_text(encoding="utf-8")
        lines = [ln for ln in content.strip().split("\n") if ln.strip()]
        assert len(lines) >= 1
        for line in lines:
            json.loads(line)  # Should not raise

    def test_export_csv_to_file(
        self,
        _populated_collector: TelemetryCollector,
        tmp_path: Path,
    ) -> None:
        """CSV --output writes valid CSV to a file."""
        out_file = tmp_path / "export.csv"
        result = runner.invoke(
            app, ["stats", "export", "--format", "csv", "--output", str(out_file)],
        )
        assert result.exit_code == 0
        assert out_file.exists()

        content = out_file.read_text(encoding="utf-8")
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
        assert len(rows) >= 1

    def test_export_with_type_filter(
        self, _populated_collector: TelemetryCollector,
    ) -> None:
        """--type filters to a specific event type."""
        result = runner.invoke(
            app, ["stats", "export", "--format", "jsonl", "--type", "tool_call"],
        )
        assert result.exit_code == 0

        lines = [ln for ln in result.output.strip().split("\n") if ln.strip()]
        for line in lines:
            obj = json.loads(line)
            assert obj["event_type"] == "tool_call"

    def test_export_invalid_format(
        self, _populated_collector: TelemetryCollector,
    ) -> None:
        """Unsupported format exits with error."""
        result = runner.invoke(app, ["stats", "export", "--format", "xml"])
        assert result.exit_code == 1
        assert "Unsupported format" in result.output


# ══════════════════════════════════════════════════════════════
# stats clear
# ══════════════════════════════════════════════════════════════


class TestStatsClear:
    """Tests for `vaig stats clear`."""

    def test_clear_without_confirm(self) -> None:
        """Without --confirm, shows warning and exits without deleting."""
        result = runner.invoke(app, ["stats", "clear"])
        assert result.exit_code == 0
        assert "delete telemetry events" in result.output.lower() or "will delete" in result.output.lower()

    def test_clear_with_confirm(
        self, _populated_collector: TelemetryCollector, collector: TelemetryCollector,
    ) -> None:
        """With --confirm, deletes events and shows count."""
        # Add some old events
        for i in range(3):
            old = TelemetryEvent(
                event_type="tool_call",
                event_name=f"old_{i}",
                timestamp="2020-01-01T00:00:00+00:00",
            )
            collector._append(old)
        collector.flush()

        result = runner.invoke(app, ["stats", "clear", "--confirm"])
        assert result.exit_code == 0
        assert "Deleted" in result.output

    def test_clear_custom_days(
        self, collector: TelemetryCollector,
    ) -> None:
        """--days controls the retention period."""
        # Add old events
        for i in range(2):
            old = TelemetryEvent(
                event_type="tool_call",
                event_name=f"old_{i}",
                timestamp="2020-01-01T00:00:00+00:00",
            )
            collector._append(old)
        collector.flush()

        result = runner.invoke(app, ["stats", "clear", "--days", "7", "--confirm"])
        assert result.exit_code == 0
        assert "Deleted 2" in result.output
        assert "7 days" in result.output

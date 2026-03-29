"""Tests for the feedback command and supporting functions.

Validates:
- ``export_feedback_to_bigquery()`` on DataExporter (mock BQ)
- Run ID local persistence (``save_last_run_id`` / ``get_last_run_id``)
- ``vaig feedback`` CLI (--run-id, --last, mutual exclusivity, rating range)
- ``_prompt_feedback()`` interactive prompt (mock input)
- ALL external dependencies are mocked (no real GCP/network calls)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ── Helpers ──────────────────────────────────────────────────


def _make_export_config(**overrides: Any) -> SimpleNamespace:
    """Build a fake ExportConfig that mimics the real model."""
    defaults = {
        "enabled": True,
        "gcp_project_id": "test-project",
        "bigquery_dataset": "vaig_data",
        "gcs_bucket": "",
        "gcs_prefix": "vaig/",
        "auto_export_reports": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_settings(**overrides: Any) -> SimpleNamespace:
    """Build a fake Settings object with export enabled."""
    defaults = {
        "export": _make_export_config(),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ── DataExporter.export_feedback_to_bigquery tests ───────────


class TestExportFeedback:
    """Unit tests for DataExporter.export_feedback_to_bigquery."""

    def test_export_feedback_success(self) -> None:
        """Feedback is transformed and inserted into BQ."""
        from vaig.core.export import DataExporter

        config = _make_export_config()
        mock_bq = MagicMock()
        mock_bq.insert_rows_json.return_value = []
        exporter = DataExporter(config, bq_client=mock_bq)

        result = exporter.export_feedback_to_bigquery(
            rating=4,
            comment="Good analysis",
            run_id="20250601T120000Z",
        )
        assert result is True
        # Verify insert_rows_json was called with the feedback table
        mock_bq.insert_rows_json.assert_called_once()
        call_args = mock_bq.insert_rows_json.call_args
        table_id = call_args[0][0]
        rows = call_args[0][1]
        assert "feedback" in table_id
        assert len(rows) == 1
        assert rows[0]["rating"] == 4
        assert rows[0]["comment"] == "Good analysis"
        assert rows[0]["run_id"] == "20250601T120000Z"

    def test_export_feedback_no_project_returns_false(self) -> None:
        """Returns False when GCP project is not configured."""
        from vaig.core.export import DataExporter

        config = _make_export_config(gcp_project_id="")
        exporter = DataExporter(config)

        result = exporter.export_feedback_to_bigquery(rating=3, run_id="test-run")
        assert result is False

    def test_export_feedback_bq_error_returns_false(self) -> None:
        """Returns False on BQ insert failure."""
        from vaig.core.export import DataExporter

        config = _make_export_config()
        mock_bq = MagicMock()
        mock_bq.insert_rows_json.side_effect = RuntimeError("BQ down")
        exporter = DataExporter(config, bq_client=mock_bq)

        result = exporter.export_feedback_to_bigquery(rating=5, run_id="test-run")
        assert result is False

    def test_export_feedback_with_metadata(self) -> None:
        """Metadata dict is passed through to the transformer."""
        from vaig.core.export import DataExporter

        config = _make_export_config()
        mock_bq = MagicMock()
        mock_bq.insert_rows_json.return_value = []
        exporter = DataExporter(config, bq_client=mock_bq)

        result = exporter.export_feedback_to_bigquery(
            rating=2,
            comment="Needs work",
            run_id="20250601T120000Z",
            metadata={"source": "cli"},
        )
        assert result is True

    def test_export_feedback_clamps_rating(self) -> None:
        """Rating is clamped to 1-5 by the transformer."""
        from vaig.core.export import DataExporter

        config = _make_export_config()
        mock_bq = MagicMock()
        mock_bq.insert_rows_json.return_value = []
        exporter = DataExporter(config, bq_client=mock_bq)

        # Rating above 5 should be clamped
        exporter.export_feedback_to_bigquery(rating=10, run_id="test-run")
        rows = mock_bq.insert_rows_json.call_args[0][1]
        assert rows[0]["rating"] == 5

        mock_bq.reset_mock()
        mock_bq.insert_rows_json.return_value = []

        # Rating below 1 should be clamped
        exporter.export_feedback_to_bigquery(rating=-1, run_id="test-run")
        rows = mock_bq.insert_rows_json.call_args[0][1]
        assert rows[0]["rating"] == 1


# ── Run ID persistence tests ────────────────────────────────


class TestRunIdPersistence:
    """Tests for save_last_run_id and get_last_run_id."""

    def test_save_and_read_run_id(self, tmp_path: Path) -> None:
        """Round-trip: save then get returns the same run_id."""
        from vaig.core.export import get_last_run_id, save_last_run_id

        fake_path = tmp_path / "last_run_id"
        with patch("vaig.core.export._LAST_RUN_ID_PATH", fake_path):
            save_last_run_id("20250601T120000Z")
            assert get_last_run_id() == "20250601T120000Z"

    def test_get_returns_none_when_no_file(self, tmp_path: Path) -> None:
        """Returns None when the file doesn't exist."""
        from vaig.core.export import get_last_run_id

        fake_path = tmp_path / "nonexistent" / "last_run_id"
        with patch("vaig.core.export._LAST_RUN_ID_PATH", fake_path):
            assert get_last_run_id() is None

    def test_get_returns_none_when_file_empty(self, tmp_path: Path) -> None:
        """Returns None when the file is empty."""
        from vaig.core.export import get_last_run_id

        fake_path = tmp_path / "last_run_id"
        fake_path.write_text("", encoding="utf-8")
        with patch("vaig.core.export._LAST_RUN_ID_PATH", fake_path):
            assert get_last_run_id() is None

    def test_save_creates_parent_directory(self, tmp_path: Path) -> None:
        """save_last_run_id creates the parent directory if missing."""
        from vaig.core.export import save_last_run_id

        fake_path = tmp_path / "subdir" / "last_run_id"
        with patch("vaig.core.export._LAST_RUN_ID_PATH", fake_path):
            save_last_run_id("test-run")
            assert fake_path.exists()
            assert fake_path.read_text(encoding="utf-8") == "test-run"

    def test_save_strips_whitespace(self, tmp_path: Path) -> None:
        """save_last_run_id strips leading/trailing whitespace."""
        from vaig.core.export import get_last_run_id, save_last_run_id

        fake_path = tmp_path / "last_run_id"
        with patch("vaig.core.export._LAST_RUN_ID_PATH", fake_path):
            save_last_run_id("  run-123  \n")
            assert get_last_run_id() == "run-123"

    def test_save_handles_io_error(self, tmp_path: Path) -> None:
        """save_last_run_id silently handles I/O errors."""
        from vaig.core.export import save_last_run_id

        fake_path = tmp_path / "last_run_id"
        with patch("vaig.core.export._LAST_RUN_ID_PATH", fake_path), \
             patch.object(Path, "write_text", side_effect=OSError("disk full")):
            # Should not raise
            save_last_run_id("test-run")


# ── Feedback CLI command tests ───────────────────────────────


@pytest.fixture()
def cli_app():
    """Create a fresh Typer app with feedback registered."""
    import typer

    from vaig.cli.commands.feedback import register

    test_app = typer.Typer()
    register(test_app)
    return test_app


@pytest.fixture()
def runner():
    """Typer CLI test runner."""
    from typer.testing import CliRunner

    return CliRunner()


class TestFeedbackCommand:
    """Integration tests for the feedback CLI via typer CliRunner."""

    def test_feedback_with_run_id_success(self, cli_app: Any, runner: Any) -> None:
        """Feedback with explicit --run-id submits successfully."""
        mock_settings = _make_settings()
        with patch("vaig.cli.commands.feedback._get_settings", return_value=mock_settings), \
             patch("vaig.core.export.DataExporter.export_feedback_to_bigquery", return_value=True) as mock_export:
            res = runner.invoke(cli_app, ["--rating", "5", "--run-id", "20250601T120000Z"])
            assert res.exit_code == 0
            assert "\u2713" in res.output  # checkmark
            mock_export.assert_called_once_with(
                rating=5, comment="", run_id="20250601T120000Z",
            )

    def test_feedback_with_last_flag(self, cli_app: Any, runner: Any) -> None:
        """Feedback with --last reads the stored run ID."""
        mock_settings = _make_settings()
        with patch("vaig.cli.commands.feedback._get_settings", return_value=mock_settings), \
             patch("vaig.core.export.get_last_run_id", return_value="20250601T120000Z"), \
             patch("vaig.core.export.DataExporter.export_feedback_to_bigquery", return_value=True) as mock_export:
            res = runner.invoke(cli_app, ["--rating", "4", "--last"])
            assert res.exit_code == 0
            mock_export.assert_called_once_with(
                rating=4, comment="", run_id="20250601T120000Z",
            )

    def test_feedback_with_comment(self, cli_app: Any, runner: Any) -> None:
        """Feedback includes --comment in the export."""
        mock_settings = _make_settings()
        with patch("vaig.cli.commands.feedback._get_settings", return_value=mock_settings), \
             patch("vaig.core.export.DataExporter.export_feedback_to_bigquery", return_value=True) as mock_export:
            res = runner.invoke(
                cli_app,
                ["--rating", "3", "--run-id", "test-run", "--comment", "Needs improvement"],
            )
            assert res.exit_code == 0
            mock_export.assert_called_once_with(
                rating=3, comment="Needs improvement", run_id="test-run",
            )

    def test_mutual_exclusivity_run_id_and_last(self, cli_app: Any, runner: Any) -> None:
        """Providing both --run-id and --last is an error."""
        res = runner.invoke(
            cli_app,
            ["--rating", "5", "--run-id", "test-run", "--last"],
        )
        assert res.exit_code == 1
        assert "mutually exclusive" in res.output

    def test_neither_run_id_nor_last(self, cli_app: Any, runner: Any) -> None:
        """Providing neither --run-id nor --last is an error."""
        res = runner.invoke(cli_app, ["--rating", "5"])
        assert res.exit_code == 1
        assert "--run-id" in res.output or "--last" in res.output

    def test_last_flag_no_stored_id(self, cli_app: Any, runner: Any) -> None:
        """--last fails gracefully when no stored run ID exists."""
        with patch("vaig.core.export.get_last_run_id", return_value=None):
            res = runner.invoke(cli_app, ["--rating", "5", "--last"])
            assert res.exit_code == 1
            assert "No previous run ID" in res.output

    def test_export_disabled(self, cli_app: Any, runner: Any) -> None:
        """Feedback fails when export is disabled in config."""
        mock_settings = _make_settings(export=_make_export_config(enabled=False))
        with patch("vaig.cli.commands.feedback._get_settings", return_value=mock_settings):
            res = runner.invoke(cli_app, ["--rating", "5", "--run-id", "test-run"])
            assert res.exit_code == 1
            assert "disabled" in res.output.lower()

    def test_export_failure(self, cli_app: Any, runner: Any) -> None:
        """Feedback exits with error when BQ export fails."""
        mock_settings = _make_settings()
        with patch("vaig.cli.commands.feedback._get_settings", return_value=mock_settings), \
             patch("vaig.core.export.DataExporter.export_feedback_to_bigquery", return_value=False):
            res = runner.invoke(cli_app, ["--rating", "5", "--run-id", "test-run"])
            assert res.exit_code == 1
            assert "Failed" in res.output

    def test_rating_out_of_range_rejected_by_typer(self, cli_app: Any, runner: Any) -> None:
        """Typer rejects rating outside 1-5 range before our code runs."""
        res = runner.invoke(cli_app, ["--rating", "0", "--run-id", "test-run"])
        assert res.exit_code != 0

        res = runner.invoke(cli_app, ["--rating", "6", "--run-id", "test-run"])
        assert res.exit_code != 0

    def test_feedback_shows_stars(self, cli_app: Any, runner: Any) -> None:
        """Output contains star characters for the rating."""
        mock_settings = _make_settings()
        with patch("vaig.cli.commands.feedback._get_settings", return_value=mock_settings), \
             patch("vaig.core.export.DataExporter.export_feedback_to_bigquery", return_value=True):
            res = runner.invoke(cli_app, ["--rating", "3", "--run-id", "test-run"])
            assert res.exit_code == 0
            # 3 filled stars + 2 empty stars
            assert "\u2605\u2605\u2605\u2606\u2606" in res.output


# ── Prompt feedback tests ────────────────────────────────────


class TestPromptFeedback:
    """Tests for _prompt_feedback interactive prompt in live.py."""

    def test_prompt_skipped_when_export_disabled(self) -> None:
        """No prompt when export is disabled."""
        from vaig.cli.commands.live import _prompt_feedback

        settings = _make_settings(export=_make_export_config(enabled=False))
        # Should not raise — just returns immediately
        with patch("vaig.cli.commands.live.console") as mock_console:
            _prompt_feedback(settings)
            mock_console.input.assert_not_called()

    def test_prompt_skipped_on_empty_input(self) -> None:
        """Empty input (Enter) skips feedback."""
        from vaig.cli.commands.live import _prompt_feedback

        settings = _make_settings()
        with patch("vaig.cli.commands.live.console") as mock_console:
            mock_console.input.return_value = ""
            _prompt_feedback(settings)
            # Only the first input (rating) should be called
            assert mock_console.input.call_count == 1

    def test_prompt_skipped_on_eof(self) -> None:
        """EOFError skips feedback silently."""
        from vaig.cli.commands.live import _prompt_feedback

        settings = _make_settings()
        with patch("vaig.cli.commands.live.console") as mock_console:
            mock_console.input.side_effect = EOFError
            _prompt_feedback(settings)  # Should not raise

    def test_prompt_skipped_on_invalid_input(self) -> None:
        """Non-numeric input is silently ignored."""
        from vaig.cli.commands.live import _prompt_feedback

        settings = _make_settings()
        with patch("vaig.cli.commands.live.console") as mock_console:
            mock_console.input.return_value = "abc"
            _prompt_feedback(settings)
            assert mock_console.input.call_count == 1

    def test_prompt_skipped_on_out_of_range(self) -> None:
        """Rating outside 1-5 is silently ignored."""
        from vaig.cli.commands.live import _prompt_feedback

        settings = _make_settings()
        with patch("vaig.cli.commands.live.console") as mock_console:
            mock_console.input.return_value = "7"
            _prompt_feedback(settings)
            assert mock_console.input.call_count == 1

    def test_prompt_with_valid_rating_and_comment(self) -> None:
        """Valid rating triggers comment prompt and background export."""
        from vaig.cli.commands.live import _prompt_feedback

        settings = _make_settings()
        with patch("vaig.cli.commands.live.console") as mock_console, \
             patch("vaig.core.export.DataExporter.export_feedback_to_bigquery") as mock_export, \
             patch("vaig.cli.commands.live.threading") as mock_threading:
            mock_console.input.side_effect = ["4", "Great work!"]
            mock_console.print = MagicMock()

            # Make the thread run immediately (synchronously)
            mock_thread = MagicMock()
            mock_threading.Thread.return_value = mock_thread

            _prompt_feedback(settings, run_id="test-run-123")

            assert mock_console.input.call_count == 2
            mock_threading.Thread.assert_called_once()
            mock_thread.start.assert_called_once()

    def test_prompt_with_valid_rating_no_comment(self) -> None:
        """Valid rating with empty comment still exports."""
        from vaig.cli.commands.live import _prompt_feedback

        settings = _make_settings()
        with patch("vaig.cli.commands.live.console") as mock_console, \
             patch("vaig.cli.commands.live.threading") as mock_threading:
            mock_console.input.side_effect = ["5", ""]
            mock_console.print = MagicMock()

            mock_thread = MagicMock()
            mock_threading.Thread.return_value = mock_thread

            _prompt_feedback(settings, run_id="test-run")

            assert mock_console.input.call_count == 2
            mock_threading.Thread.assert_called_once()


# ── Auto-export run_id persistence tests ─────────────────────


class TestAutoExportRunIdPersistence:
    """Test that _auto_export_report forwards run_id to auto_export_report."""

    def test_auto_export_forwards_run_id(self) -> None:
        """_auto_export_report passes the provided run_id to auto_export_report."""
        from vaig.cli.commands.live import _auto_export_report

        settings = _make_settings()
        orch_result = SimpleNamespace(structured_report=SimpleNamespace(to_dict=lambda: {}))
        gke_config = SimpleNamespace(cluster_name="test", default_namespace="default")

        with patch("vaig.core.export.auto_export_report") as mock_export:
            _auto_export_report(settings, orch_result, gke_config, run_id="my-run-123")
            mock_export.assert_called_once()
            assert mock_export.call_args[1]["run_id"] == "my-run-123"

    def test_auto_export_generates_fallback_run_id(self) -> None:
        """_auto_export_report generates a timestamp run_id when none is provided."""
        from vaig.cli.commands.live import _auto_export_report

        settings = _make_settings()
        orch_result = SimpleNamespace(structured_report=SimpleNamespace(to_dict=lambda: {}))
        gke_config = SimpleNamespace(cluster_name="test", default_namespace="default")

        with patch("vaig.core.export.auto_export_report") as mock_export:
            _auto_export_report(settings, orch_result, gke_config)
            mock_export.assert_called_once()
            generated_run_id = mock_export.call_args[1]["run_id"]
            assert len(generated_run_id) > 0

    def test_auto_export_skips_when_disabled(self) -> None:
        """_auto_export_report does nothing when export is disabled."""
        from vaig.cli.commands.live import _auto_export_report

        settings = _make_settings(export=_make_export_config(enabled=False))
        orch_result = SimpleNamespace(structured_report=SimpleNamespace(to_dict=lambda: {}))
        gke_config = SimpleNamespace(cluster_name="test", default_namespace="default")

        with patch("vaig.core.export.auto_export_report") as mock_export:
            _auto_export_report(settings, orch_result, gke_config)
            mock_export.assert_not_called()

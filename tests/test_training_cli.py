"""Tests for train CLI commands — prepare and submit (REQ-TRAIN-07, REQ-TRAIN-08)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from vaig.cli.app import app

runner = CliRunner()


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def valid_jsonl_file(tmp_path: Path) -> Path:
    """Create a valid JSONL file with 20 entries."""
    path = tmp_path / "train.jsonl"
    with open(path, "w") as f:
        for i in range(20):
            entry = {
                "contents": [
                    {"role": "user", "parts": [{"text": f"Analyze tool_{i}"}]},
                    {"role": "model", "parts": [{"text": f"Report {i}"}]},
                ]
            }
            f.write(json.dumps(entry) + "\n")
    return path


def _mock_settings() -> MagicMock:
    """Build a mock Settings object with training and export sub-configs."""
    from vaig.core.config import ExportConfig, TrainingConfig

    settings = MagicMock()
    settings.export = ExportConfig(
        enabled=True,
        gcp_project_id="test-project",
        bigquery_dataset="test_dataset",
        gcs_bucket="test-bucket",
    )
    settings.training = TrainingConfig(
        enabled=True,
        min_examples=10,
        max_examples=100,
        min_rating=4,
    )
    return settings


# ── train --help ─────────────────────────────────────────────


class TestTrainHelp:
    """Verify CLI registration (REQ-TRAIN-08)."""

    def test_train_help_shows_subcommands(self) -> None:
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "prepare" in result.stdout
        assert "submit" in result.stdout


# ── train prepare ────────────────────────────────────────────


class TestTrainPrepare:
    """Verify prepare CLI command."""

    def test_prepare_dry_run_success(self, tmp_path: Path) -> None:
        from vaig.core.training import PrepareResult

        mock_result = PrepareResult(
            jsonl_path=tmp_path / "out.jsonl",
            total_examples=50,
            avg_rating=4.5,
            estimated_tokens=12000,
        )

        with (
            patch("vaig.cli._helpers._get_settings", return_value=_mock_settings()),
            patch("vaig.core.training.TrainingDataPreparer") as MockPreparer,
        ):
            MockPreparer.return_value.prepare.return_value = mock_result
            result = runner.invoke(app, ["train", "prepare", "--dry-run"])

        assert result.exit_code == 0
        assert "50" in result.stdout

    def test_prepare_with_output(self, tmp_path: Path) -> None:
        from vaig.core.training import PrepareResult

        output = tmp_path / "train.jsonl"
        mock_result = PrepareResult(
            jsonl_path=output,
            total_examples=25,
            avg_rating=4.2,
            estimated_tokens=8000,
        )

        with (
            patch("vaig.cli._helpers._get_settings", return_value=_mock_settings()),
            patch("vaig.core.training.TrainingDataPreparer") as MockPreparer,
        ):
            MockPreparer.return_value.prepare.return_value = mock_result
            result = runner.invoke(app, ["train", "prepare", "--output", str(output)])

        assert result.exit_code == 0

    def test_prepare_import_error(self) -> None:
        with (
            patch("vaig.cli._helpers._get_settings", return_value=_mock_settings()),
            patch(
                "vaig.core.training.TrainingDataPreparer",
                side_effect=ImportError("rag extras required"),
            ),
        ):
            result = runner.invoke(app, ["train", "prepare", "--dry-run"])

        assert result.exit_code == 1

    def test_prepare_insufficient_data_exits_1(self) -> None:
        with (
            patch("vaig.cli._helpers._get_settings", return_value=_mock_settings()),
            patch("vaig.core.training.TrainingDataPreparer") as MockPreparer,
        ):
            MockPreparer.return_value.prepare.side_effect = SystemExit(1)
            result = runner.invoke(app, ["train", "prepare"])

        assert result.exit_code == 1


# ── train submit ─────────────────────────────────────────────


class TestTrainSubmit:
    """Verify submit CLI command."""

    def test_submit_dry_run_success(self, valid_jsonl_file: Path) -> None:
        from vaig.core.training import SubmitResult

        mock_result = SubmitResult(
            job_name="dry-run",
            gcs_uri="",
            base_model="gemini-2.0-flash-001",
            status="dry-run: 20 examples",
        )

        with (
            patch("vaig.cli._helpers._get_settings", return_value=_mock_settings()),
            patch("vaig.core.training.TuningJobSubmitter") as MockSubmitter,
        ):
            MockSubmitter.return_value.submit.return_value = mock_result
            result = runner.invoke(
                app, ["train", "submit", "--input", str(valid_jsonl_file), "--dry-run"]
            )

        assert result.exit_code == 0
        assert "dry-run" in result.stdout

    def test_submit_full_flow(self, valid_jsonl_file: Path) -> None:
        from vaig.core.training import SubmitResult

        mock_result = SubmitResult(
            job_name="tuning-job-456",
            gcs_uri="gs://test-bucket/training_data/train.jsonl",
            base_model="gemini-2.0-flash-001",
            status="SUBMITTED",
        )

        with (
            patch("vaig.cli._helpers._get_settings", return_value=_mock_settings()),
            patch("vaig.core.training.TuningJobSubmitter") as MockSubmitter,
        ):
            MockSubmitter.return_value.submit.return_value = mock_result
            result = runner.invoke(
                app, ["train", "submit", "--input", str(valid_jsonl_file)]
            )

        assert result.exit_code == 0
        assert "tuning-job-456" in result.stdout

    def test_submit_missing_file_exits_1(self) -> None:
        with (
            patch("vaig.cli._helpers._get_settings", return_value=_mock_settings()),
            patch("vaig.core.training.TuningJobSubmitter") as MockSubmitter,
        ):
            MockSubmitter.return_value.submit.side_effect = FileNotFoundError("not found")
            result = runner.invoke(
                app, ["train", "submit", "--input", "/nonexistent/train.jsonl"]
            )

        assert result.exit_code == 1

    def test_submit_validation_error_exits_1(self, valid_jsonl_file: Path) -> None:
        with (
            patch("vaig.cli._helpers._get_settings", return_value=_mock_settings()),
            patch("vaig.core.training.TuningJobSubmitter") as MockSubmitter,
        ):
            MockSubmitter.return_value.submit.side_effect = ValueError("Insufficient examples")
            result = runner.invoke(
                app, ["train", "submit", "--input", str(valid_jsonl_file)]
            )

        assert result.exit_code == 1

"""Tests for TuningJobSubmitter — validation, GCS upload, and tuning submission."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vaig.core.config import ExportConfig, TrainingConfig

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def export_config() -> ExportConfig:
    return ExportConfig(
        enabled=True,
        gcp_project_id="test-project",
        bigquery_dataset="test_dataset",
        gcs_bucket="test-bucket",
        gcs_prefix="test/",
    )


@pytest.fixture()
def training_config() -> TrainingConfig:
    return TrainingConfig(
        enabled=True,
        min_examples=10,
        max_examples=100,
        base_model="gemini-2.0-flash-001",
        epochs=3,
        learning_rate_multiplier=1.0,
        gcs_staging_prefix="training_data/",
    )


@pytest.fixture()
def mock_gcs_client() -> MagicMock:
    client = MagicMock(name="gcs_client")
    mock_bucket = MagicMock(name="bucket")
    mock_blob = MagicMock(name="blob")
    client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    return client


@pytest.fixture()
def mock_genai_client() -> MagicMock:
    client = MagicMock(name="genai_client")
    mock_job = MagicMock()
    mock_job.name = "tuning-job-123"
    mock_job.state = "SUBMITTED"
    client.tunings.tune.return_value = mock_job
    return client


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


@pytest.fixture()
def small_jsonl_file(tmp_path: Path) -> Path:
    """Create a JSONL file with too few entries (3 < min_examples=10)."""
    path = tmp_path / "small.jsonl"
    with open(path, "w") as f:
        for i in range(3):
            entry = {
                "contents": [
                    {"role": "user", "parts": [{"text": f"Q{i}"}]},
                    {"role": "model", "parts": [{"text": f"A{i}"}]},
                ]
            }
            f.write(json.dumps(entry) + "\n")
    return path


# ── validate ─────────────────────────────────────────────────


class TestValidate:
    """Verify JSONL validation (REQ-TRAIN-05, REQ-TRAIN-06)."""

    def test_valid_file_returns_stats(
        self,
        export_config: ExportConfig,
        training_config: TrainingConfig,
        mock_gcs_client: MagicMock,
        mock_genai_client: MagicMock,
        valid_jsonl_file: Path,
    ) -> None:
        from vaig.core.training import TuningJobSubmitter

        submitter = TuningJobSubmitter(
            export_config, training_config,
            gcs_client=mock_gcs_client, genai_client=mock_genai_client,
        )
        stats = submitter.validate(valid_jsonl_file)

        assert stats["count"] == 20
        assert stats["valid"] is True
        assert "avg_tokens" in stats

    def test_too_few_lines_raises_value_error(
        self,
        export_config: ExportConfig,
        training_config: TrainingConfig,
        mock_gcs_client: MagicMock,
        mock_genai_client: MagicMock,
        small_jsonl_file: Path,
    ) -> None:
        from vaig.core.training import TuningJobSubmitter

        submitter = TuningJobSubmitter(
            export_config, training_config,
            gcs_client=mock_gcs_client, genai_client=mock_genai_client,
        )

        with pytest.raises(ValueError, match="Insufficient examples"):
            submitter.validate(small_jsonl_file)

    def test_missing_file_raises_file_not_found(
        self,
        export_config: ExportConfig,
        training_config: TrainingConfig,
        mock_gcs_client: MagicMock,
        mock_genai_client: MagicMock,
    ) -> None:
        from vaig.core.training import TuningJobSubmitter

        submitter = TuningJobSubmitter(
            export_config, training_config,
            gcs_client=mock_gcs_client, genai_client=mock_genai_client,
        )

        with pytest.raises(FileNotFoundError):
            submitter.validate(Path("/nonexistent/train.jsonl"))


# ── upload_to_gcs ────────────────────────────────────────────


class TestUploadToGcs:
    """Verify GCS upload logic (REQ-TRAIN-05)."""

    def test_upload_returns_gcs_uri(
        self,
        export_config: ExportConfig,
        training_config: TrainingConfig,
        mock_gcs_client: MagicMock,
        mock_genai_client: MagicMock,
        valid_jsonl_file: Path,
    ) -> None:
        from vaig.core.training import TuningJobSubmitter

        submitter = TuningJobSubmitter(
            export_config, training_config,
            gcs_client=mock_gcs_client, genai_client=mock_genai_client,
        )
        uri = submitter.upload_to_gcs(valid_jsonl_file)

        assert uri.startswith("gs://test-bucket/")
        assert "training_data/" in uri
        assert valid_jsonl_file.name in uri

    def test_upload_calls_gcs_correctly(
        self,
        export_config: ExportConfig,
        training_config: TrainingConfig,
        mock_gcs_client: MagicMock,
        mock_genai_client: MagicMock,
        valid_jsonl_file: Path,
    ) -> None:
        from vaig.core.training import TuningJobSubmitter

        submitter = TuningJobSubmitter(
            export_config, training_config,
            gcs_client=mock_gcs_client, genai_client=mock_genai_client,
        )
        submitter.upload_to_gcs(valid_jsonl_file)

        mock_gcs_client.bucket.assert_called_once_with("test-bucket")
        mock_bucket = mock_gcs_client.bucket.return_value
        mock_bucket.blob.assert_called_once()
        mock_blob = mock_bucket.blob.return_value
        mock_blob.upload_from_filename.assert_called_once_with(str(valid_jsonl_file))


# ── submit ───────────────────────────────────────────────────


class TestSubmit:
    """Verify full submit flow (REQ-TRAIN-05, REQ-TRAIN-06)."""

    def test_submit_full_flow(
        self,
        export_config: ExportConfig,
        training_config: TrainingConfig,
        mock_gcs_client: MagicMock,
        mock_genai_client: MagicMock,
        valid_jsonl_file: Path,
    ) -> None:
        from vaig.core.training import TuningJobSubmitter

        submitter = TuningJobSubmitter(
            export_config, training_config,
            gcs_client=mock_gcs_client, genai_client=mock_genai_client,
        )
        result = submitter.submit(valid_jsonl_file)

        assert result.job_name == "tuning-job-123"
        assert result.base_model == "gemini-2.0-flash-001"
        assert result.gcs_uri.startswith("gs://")
        mock_genai_client.tunings.tune.assert_called_once()

    def test_submit_dry_run_skips_upload_and_tuning(
        self,
        export_config: ExportConfig,
        training_config: TrainingConfig,
        mock_gcs_client: MagicMock,
        mock_genai_client: MagicMock,
        valid_jsonl_file: Path,
    ) -> None:
        from vaig.core.training import TuningJobSubmitter

        submitter = TuningJobSubmitter(
            export_config, training_config,
            gcs_client=mock_gcs_client, genai_client=mock_genai_client,
        )
        result = submitter.submit(valid_jsonl_file, dry_run=True)

        assert result.job_name == "dry-run"
        assert result.gcs_uri == ""
        assert "dry-run" in result.status
        # No GCS upload or tuning calls
        mock_gcs_client.bucket.assert_not_called()
        mock_genai_client.tunings.tune.assert_not_called()

    def test_submit_passes_tuning_params(
        self,
        export_config: ExportConfig,
        training_config: TrainingConfig,
        mock_gcs_client: MagicMock,
        mock_genai_client: MagicMock,
        valid_jsonl_file: Path,
    ) -> None:
        from vaig.core.training import TuningJobSubmitter

        submitter = TuningJobSubmitter(
            export_config, training_config,
            gcs_client=mock_gcs_client, genai_client=mock_genai_client,
        )
        submitter.submit(valid_jsonl_file)

        call_kwargs = mock_genai_client.tunings.tune.call_args
        assert call_kwargs.kwargs["base_model"] == "gemini-2.0-flash-001"
        assert "training_dataset" in call_kwargs.kwargs
        assert call_kwargs.kwargs["config"]["epoch_count"] == 3
        assert call_kwargs.kwargs["config"]["learning_rate_multiplier"] == 1.0

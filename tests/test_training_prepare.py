"""Tests for TrainingDataPreparer — BQ extraction and JSONL transformation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    )


@pytest.fixture()
def training_config() -> TrainingConfig:
    return TrainingConfig(
        enabled=True,
        min_examples=10,
        max_examples=100,
        min_rating=4,
    )


@pytest.fixture()
def mock_bq_client() -> MagicMock:
    return MagicMock(name="bq_client")


@pytest.fixture()
def sample_bq_rows() -> list[dict]:
    """Generate sample BQ rows for testing."""
    rows = []
    for i in range(20):
        rows.append({
            "tool_name": f"check_pod_{i}",
            "input_params": json.dumps({"namespace": "default", "pod": f"pod-{i}"}),
            "report_markdown": f"## Health Report {i}\n\nPod pod-{i} is healthy.",
            "event_type": "tool_execution",
            "cluster_name": "test-cluster",
            "namespace": "default",
            "rating": 5,
        })
    return rows


# ── _require_rag_deps guard ──────────────────────────────────


class TestRequireRagDepsGuard:
    """TrainingDataPreparer must guard on [rag] deps lazily."""

    def test_import_error_when_bigquery_missing(
        self, export_config: ExportConfig, training_config: TrainingConfig
    ) -> None:
        """Construction succeeds; ImportError fires on first BQ access."""
        from vaig.core.training import TrainingDataPreparer

        with patch(
            "vaig.core.training._require_rag_deps",
            side_effect=ImportError("Training features require the [rag] extras"),
        ):
            # Construction must NOT raise — guard is lazy
            preparer = TrainingDataPreparer(export_config, training_config)

            # Accessing BQ client (via extract_pairs) triggers the guard
            with pytest.raises(ImportError, match="rag"):
                preparer.extract_pairs(min_rating=4, max_examples=10)


# ── extract_pairs ────────────────────────────────────────────


class TestExtractPairs:
    """Verify BQ query execution and row mapping."""

    def test_extract_returns_rows(
        self,
        export_config: ExportConfig,
        training_config: TrainingConfig,
        mock_bq_client: MagicMock,
        sample_bq_rows: list[dict],
    ) -> None:
        from vaig.core.training import TrainingDataPreparer

        # Mock BQ query result
        mock_result = MagicMock()
        mock_result.__iter__.return_value = iter(sample_bq_rows)
        mock_query_job = MagicMock()
        mock_query_job.result.return_value = mock_result
        mock_bq_client.query.return_value = mock_query_job

        preparer = TrainingDataPreparer(export_config, training_config, bq_client=mock_bq_client)
        rows = preparer.extract_pairs(min_rating=4, max_examples=100)

        assert len(rows) == 20
        assert rows[0]["tool_name"] == "check_pod_0"
        mock_bq_client.query.assert_called_once()

    def test_extract_empty_result(
        self,
        export_config: ExportConfig,
        training_config: TrainingConfig,
        mock_bq_client: MagicMock,
    ) -> None:
        from vaig.core.training import TrainingDataPreparer

        mock_result = MagicMock()
        mock_result.__iter__.return_value = iter([])
        mock_query_job = MagicMock()
        mock_query_job.result.return_value = mock_result
        mock_bq_client.query.return_value = mock_query_job

        preparer = TrainingDataPreparer(export_config, training_config, bq_client=mock_bq_client)
        rows = preparer.extract_pairs(min_rating=4, max_examples=100)

        assert len(rows) == 0


# ── transform_to_jsonl ───────────────────────────────────────


class TestTransformToJsonl:
    """Verify JSONL schema compliance (REQ-TRAIN-04)."""

    def test_jsonl_schema_correctness(
        self,
        export_config: ExportConfig,
        training_config: TrainingConfig,
        mock_bq_client: MagicMock,
    ) -> None:
        from vaig.core.training import TrainingDataPreparer

        preparer = TrainingDataPreparer(export_config, training_config, bq_client=mock_bq_client)
        rows = [
            {
                "tool_name": "check_pod",
                "input_params": '{"ns":"default"}',
                "report_markdown": "## Report\n\nPod is healthy.",
                "event_type": "tool_execution",
                "cluster_name": "test-cluster",
                "namespace": "default",
                "rating": 5,
            }
        ]
        entries = preparer.transform_to_jsonl(rows)

        assert len(entries) == 1
        entry = entries[0]
        assert "contents" in entry
        assert len(entry["contents"]) == 2
        assert entry["contents"][0]["role"] == "user"
        assert entry["contents"][1]["role"] == "model"
        assert len(entry["contents"][0]["parts"]) == 1
        assert "text" in entry["contents"][0]["parts"][0]
        assert entry["contents"][1]["parts"][0]["text"] == "## Report\n\nPod is healthy."

    def test_user_text_combines_fields(
        self,
        export_config: ExportConfig,
        training_config: TrainingConfig,
        mock_bq_client: MagicMock,
    ) -> None:
        from vaig.core.training import TrainingDataPreparer

        preparer = TrainingDataPreparer(export_config, training_config, bq_client=mock_bq_client)
        rows = [
            {
                "tool_name": "check_pod",
                "input_params": '{"ns":"default"}',
                "report_markdown": "report text",
                "cluster_name": "my-cluster",
                "namespace": "my-ns",
                "rating": 5,
            }
        ]
        entries = preparer.transform_to_jsonl(rows)

        user_text = entries[0]["contents"][0]["parts"][0]["text"]
        assert "check_pod" in user_text
        assert '{"ns":"default"}' in user_text
        assert "my-cluster" in user_text
        assert "my-ns" in user_text

    def test_rows_without_report_markdown_skipped(
        self,
        export_config: ExportConfig,
        training_config: TrainingConfig,
        mock_bq_client: MagicMock,
    ) -> None:
        from vaig.core.training import TrainingDataPreparer

        preparer = TrainingDataPreparer(export_config, training_config, bq_client=mock_bq_client)
        rows = [
            {
                "tool_name": "check_pod",
                "input_params": "{}",
                "report_markdown": "",
                "cluster_name": "",
                "namespace": "",
                "rating": 5,
            },
            {
                "tool_name": "check_pod2",
                "input_params": "{}",
                "report_markdown": "valid report",
                "cluster_name": "",
                "namespace": "",
                "rating": 5,
            },
        ]
        entries = preparer.transform_to_jsonl(rows)
        assert len(entries) == 1

    def test_transform_valid_jsonl_output(
        self,
        export_config: ExportConfig,
        training_config: TrainingConfig,
        mock_bq_client: MagicMock,
    ) -> None:
        """Each entry must be valid JSON and serialize correctly."""
        from vaig.core.training import TrainingDataPreparer

        preparer = TrainingDataPreparer(export_config, training_config, bq_client=mock_bq_client)
        rows = [
            {
                "tool_name": "tool",
                "input_params": "{}",
                "report_markdown": "report",
                "cluster_name": "c",
                "namespace": "n",
                "rating": 4,
            }
        ]
        entries = preparer.transform_to_jsonl(rows)
        # Must serialize without error
        line = json.dumps(entries[0])
        parsed = json.loads(line)
        assert parsed["contents"][0]["role"] == "user"


# ── prepare (end-to-end) ─────────────────────────────────────


class TestPrepare:
    """Verify prepare orchestration (REQ-TRAIN-02, REQ-TRAIN-03)."""

    def test_prepare_writes_jsonl_file(
        self,
        export_config: ExportConfig,
        training_config: TrainingConfig,
        mock_bq_client: MagicMock,
        sample_bq_rows: list[dict],
        tmp_path: Path,
    ) -> None:
        from vaig.core.training import TrainingDataPreparer

        mock_result = MagicMock()
        mock_result.__iter__.return_value = iter(sample_bq_rows)
        mock_query_job = MagicMock()
        mock_query_job.result.return_value = mock_result
        mock_bq_client.query.return_value = mock_query_job

        preparer = TrainingDataPreparer(export_config, training_config, bq_client=mock_bq_client)
        output = tmp_path / "train.jsonl"
        result = preparer.prepare(output_path=output)

        assert output.exists()
        assert result.total_examples == 20
        assert result.jsonl_path == output

        # Verify file content
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 20
        entry = json.loads(lines[0])
        assert "contents" in entry

    def test_prepare_dry_run_no_file(
        self,
        export_config: ExportConfig,
        training_config: TrainingConfig,
        mock_bq_client: MagicMock,
        sample_bq_rows: list[dict],
        tmp_path: Path,
    ) -> None:
        from vaig.core.training import TrainingDataPreparer

        mock_result = MagicMock()
        mock_result.__iter__.return_value = iter(sample_bq_rows)
        mock_query_job = MagicMock()
        mock_query_job.result.return_value = mock_result
        mock_bq_client.query.return_value = mock_query_job

        preparer = TrainingDataPreparer(export_config, training_config, bq_client=mock_bq_client)
        output = tmp_path / "dry_run.jsonl"
        result = preparer.prepare(output_path=output, dry_run=True)

        assert not output.exists()
        assert result.total_examples == 20

    def test_prepare_insufficient_data_raises_value_error(
        self,
        export_config: ExportConfig,
        mock_bq_client: MagicMock,
    ) -> None:
        from vaig.core.training import TrainingDataPreparer

        # min_examples=50, but only 5 rows
        tc = TrainingConfig(min_examples=50)
        rows = [
            {
                "tool_name": f"tool_{i}",
                "input_params": "{}",
                "report_markdown": f"report {i}",
                "cluster_name": "c",
                "namespace": "n",
                "rating": 5,
            }
            for i in range(5)
        ]

        mock_result = MagicMock()
        mock_result.__iter__.return_value = iter(rows)
        mock_query_job = MagicMock()
        mock_query_job.result.return_value = mock_result
        mock_bq_client.query.return_value = mock_query_job

        preparer = TrainingDataPreparer(export_config, tc, bq_client=mock_bq_client)

        with pytest.raises(ValueError, match="Insufficient examples"):
            preparer.prepare()

    def test_prepare_avg_rating_calculation(
        self,
        export_config: ExportConfig,
        training_config: TrainingConfig,
        mock_bq_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        from vaig.core.training import TrainingDataPreparer

        rows = [
            {
                "tool_name": f"tool_{i}",
                "input_params": "{}",
                "report_markdown": f"report {i}",
                "cluster_name": "c",
                "namespace": "n",
                "rating": 4 + (i % 2),  # alternates 4, 5
            }
            for i in range(12)
        ]

        mock_result = MagicMock()
        mock_result.__iter__.return_value = iter(rows)
        mock_query_job = MagicMock()
        mock_query_job.result.return_value = mock_result
        mock_bq_client.query.return_value = mock_query_job

        preparer = TrainingDataPreparer(export_config, training_config, bq_client=mock_bq_client)
        result = preparer.prepare(output_path=tmp_path / "out.jsonl")

        assert result.avg_rating == 4.5

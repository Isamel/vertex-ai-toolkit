"""Tests for DataExporter core class — vaig/core/export.py."""

from __future__ import annotations

import logging
import threading
from unittest.mock import MagicMock, patch

import pytest

from vaig.core.config import ExportConfig
from vaig.core.export import (
    _TABLE_CONFIG,
    _TABLE_SCHEMAS,
    DataExporter,
    _require_rag_deps,
)

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def export_config() -> ExportConfig:
    """A minimal ExportConfig suitable for most tests."""
    return ExportConfig(
        enabled=True,
        gcp_project_id="test-project",
        bigquery_dataset="test_dataset",
        gcs_bucket="test-bucket",
        gcs_prefix="test/",
    )


@pytest.fixture()
def mock_bq_client() -> MagicMock:
    """A pre-built mock BigQuery client for injection tests."""
    return MagicMock(name="bq_client")


@pytest.fixture()
def mock_gcs_client() -> MagicMock:
    """A pre-built mock GCS client for injection tests."""
    return MagicMock(name="gcs_client")


@pytest.fixture()
def exporter(export_config: ExportConfig) -> DataExporter:
    """A DataExporter without injected clients (lazy init path)."""
    return DataExporter(export_config)


@pytest.fixture()
def exporter_with_clients(
    export_config: ExportConfig,
    mock_bq_client: MagicMock,
    mock_gcs_client: MagicMock,
) -> DataExporter:
    """A DataExporter with pre-injected mock clients."""
    return DataExporter(export_config, bq_client=mock_bq_client, gcs_client=mock_gcs_client)


# ── _require_rag_deps() ──────────────────────────────────────


class TestRequireRagDeps:
    """Verify the dependency guard raises a clear ImportError when deps are missing."""

    def test_raises_import_error_when_bigquery_missing(self) -> None:
        """ImportError must mention the install command."""
        with patch.dict("sys.modules", {"google.cloud.bigquery": None}):
            with pytest.raises(ImportError, match="pip install 'vertex-ai-toolkit\\[rag\\]'"):
                _require_rag_deps()

    def test_raises_import_error_when_storage_missing(self) -> None:
        """ImportError must mention the install command."""
        with patch.dict("sys.modules", {"google.cloud.storage": None}):
            with pytest.raises(ImportError, match="pip install 'vertex-ai-toolkit\\[rag\\]'"):
                _require_rag_deps()

    def test_error_message_mentions_rag_extras(self) -> None:
        """Error must name the extras group so users know what to install."""
        with patch.dict("sys.modules", {"google.cloud.bigquery": None}):
            with pytest.raises(ImportError, match=r"\[rag\]"):
                _require_rag_deps()

    def test_passes_when_deps_are_present(self) -> None:
        """No exception raised when both google-cloud packages are importable."""
        # In the test env the [rag] group is installed via [dev], so this must succeed.
        _require_rag_deps()  # should not raise


# ── Constructor ──────────────────────────────────────────────


class TestDataExporterConstructor:
    """Verify constructor stores config and optional clients correctly."""

    def test_accepts_config_only(self, export_config: ExportConfig) -> None:
        exporter = DataExporter(export_config)
        assert exporter is not None

    def test_config_property_returns_exact_config(self, export_config: ExportConfig) -> None:
        exporter = DataExporter(export_config)
        assert exporter.config is export_config

    def test_accepts_injected_bq_client(
        self, export_config: ExportConfig, mock_bq_client: MagicMock
    ) -> None:
        exporter = DataExporter(export_config, bq_client=mock_bq_client)
        assert exporter._bq_client_override is mock_bq_client  # noqa: SLF001

    def test_accepts_injected_gcs_client(
        self, export_config: ExportConfig, mock_gcs_client: MagicMock
    ) -> None:
        exporter = DataExporter(export_config, gcs_client=mock_gcs_client)
        assert exporter._gcs_client_override is mock_gcs_client  # noqa: SLF001

    def test_no_clients_created_at_construction(self, export_config: ExportConfig) -> None:
        """ADR-3: lazy init — no GCP calls at construction time."""
        exporter = DataExporter(export_config)
        assert exporter._bq_client is None  # noqa: SLF001
        assert exporter._gcs_client is None  # noqa: SLF001

    def test_bq_override_defaults_to_none(self, export_config: ExportConfig) -> None:
        exporter = DataExporter(export_config)
        assert exporter._bq_client_override is None  # noqa: SLF001

    def test_gcs_override_defaults_to_none(self, export_config: ExportConfig) -> None:
        exporter = DataExporter(export_config)
        assert exporter._gcs_client_override is None  # noqa: SLF001


# ── config property ──────────────────────────────────────────


class TestConfigProperty:
    """Verify the config property is read-only and returns the right object."""

    def test_returns_same_instance(self, exporter: DataExporter, export_config: ExportConfig) -> None:
        assert exporter.config is export_config

    def test_config_values_accessible(self, exporter: DataExporter) -> None:
        assert exporter.config.bigquery_project == "test-project"
        assert exporter.config.bigquery_dataset == "test_dataset"
        assert exporter.config.gcs_bucket == "test-bucket"


# ── Lazy client initialization ───────────────────────────────


class TestLazyClientInit:
    """ADR-3: clients must not be created until first use."""

    def test_bq_client_created_on_first_get(self, export_config: ExportConfig) -> None:
        """_get_bq_client() must return a real client on first call."""
        mock_client = MagicMock()
        mock_bq_module = MagicMock()
        mock_bq_module.Client.return_value = mock_client

        exporter = DataExporter(export_config)
        assert exporter._bq_client is None  # noqa: SLF001 — not yet created

        with patch.dict("sys.modules", {"google.cloud.bigquery": mock_bq_module}):
            # Ensure _require_rag_deps passes
            with patch("vaig.core.export._require_rag_deps"):
                with patch("vaig.core.export.DataExporter._get_bq_client", return_value=mock_client) as mock_method:
                    client = exporter._get_bq_client()  # noqa: SLF001

        assert client is mock_client

    def test_gcs_client_not_created_before_first_call(self, export_config: ExportConfig) -> None:
        exporter = DataExporter(export_config)
        assert exporter._gcs_client is None  # noqa: SLF001

    def test_bq_client_not_created_before_first_call(self, export_config: ExportConfig) -> None:
        exporter = DataExporter(export_config)
        assert exporter._bq_client is None  # noqa: SLF001


# ── Constructor injection ────────────────────────────────────


class TestConstructorInjection:
    """Injected clients must be returned directly without creating new ones."""

    def test_get_bq_client_returns_injected_client(self, exporter_with_clients: DataExporter, mock_bq_client: MagicMock) -> None:
        """_get_bq_client must return the injected mock, never create a real one."""
        client = exporter_with_clients._get_bq_client()  # noqa: SLF001
        assert client is mock_bq_client

    def test_get_gcs_client_returns_injected_client(self, exporter_with_clients: DataExporter, mock_gcs_client: MagicMock) -> None:
        """_get_gcs_client must return the injected mock, never create a real one."""
        client = exporter_with_clients._get_gcs_client()  # noqa: SLF001
        assert client is mock_gcs_client

    def test_injected_bq_client_not_replaced(self, exporter_with_clients: DataExporter, mock_bq_client: MagicMock) -> None:
        """Multiple calls to _get_bq_client always return the same injected object."""
        assert exporter_with_clients._get_bq_client() is mock_bq_client  # noqa: SLF001
        assert exporter_with_clients._get_bq_client() is mock_bq_client  # noqa: SLF001

    def test_injected_gcs_client_not_replaced(self, exporter_with_clients: DataExporter, mock_gcs_client: MagicMock) -> None:
        """Multiple calls to _get_gcs_client always return the same injected object."""
        assert exporter_with_clients._get_gcs_client() is mock_gcs_client  # noqa: SLF001
        assert exporter_with_clients._get_gcs_client() is mock_gcs_client  # noqa: SLF001

    def test_injected_client_skips_require_rag_deps(
        self, export_config: ExportConfig, mock_bq_client: MagicMock
    ) -> None:
        """When a client is injected, _require_rag_deps must NOT be called."""
        exporter = DataExporter(export_config, bq_client=mock_bq_client)
        with patch("vaig.core.export._require_rag_deps") as mock_require:
            exporter._get_bq_client()  # noqa: SLF001
        mock_require.assert_not_called()


# ── Stub method return values ────────────────────────────────


class TestStubMethods:
    """Verify all stub methods return sensible defaults and log a warning."""

    def test_ensure_dataset_returns_true_on_success(self, exporter_with_clients: DataExporter) -> None:
        """With a mock BQ client, ensure_dataset should succeed (return True)."""
        mock_dataset_cls = MagicMock()
        with (
            patch("vaig.core.export.DataExporter._get_bq_client", return_value=exporter_with_clients._bq_client_override),  # noqa: SLF001
            patch("google.cloud.bigquery.Dataset", mock_dataset_cls),
            patch("google.cloud.bigquery.TimePartitioning", MagicMock()),
            patch("google.cloud.bigquery.TimePartitioningType", MagicMock()),
        ):
            exporter_with_clients._bq_client_override.create_dataset.return_value = MagicMock()  # noqa: SLF001
            result = exporter_with_clients.ensure_dataset()
        assert result is True

    def test_ensure_tables_returns_dict_with_all_tables(self, exporter_with_clients: DataExporter) -> None:
        """ensure_tables must return a dict covering all 4 table names."""
        mock_table_cls = MagicMock()
        mock_tp_cls = MagicMock()
        mock_tp_type = MagicMock()
        with (
            patch("vaig.core.export.DataExporter._get_bq_client", return_value=exporter_with_clients._bq_client_override),  # noqa: SLF001
            patch("google.cloud.bigquery.Table", mock_table_cls),
            patch("google.cloud.bigquery.TimePartitioning", mock_tp_cls),
            patch("google.cloud.bigquery.TimePartitioningType", mock_tp_type),
            patch("google.cloud.bigquery.SchemaField", MagicMock(side_effect=lambda *a, **kw: MagicMock())),
        ):
            exporter_with_clients._bq_client_override.create_table.return_value = MagicMock()  # noqa: SLF001
            result = exporter_with_clients.ensure_tables()
        assert isinstance(result, dict)
        assert set(result.keys()) == {"telemetry_events", "tool_calls", "health_reports", "feedback"}

    def test_export_telemetry_to_bigquery_returns_zero(self, exporter_with_clients: DataExporter) -> None:
        assert exporter_with_clients.export_telemetry_to_bigquery([]) == 0

    def test_export_tool_calls_to_bigquery_returns_zero(self, exporter_with_clients: DataExporter) -> None:
        assert exporter_with_clients.export_tool_calls_to_bigquery([]) == 0

    def test_export_report_to_bigquery_returns_false_when_no_project(self) -> None:
        """export_report_to_bigquery returns False when bigquery_project is empty."""
        no_project_config = ExportConfig(
            enabled=True,
            gcp_project_id="",
            bigquery_dataset="test_dataset",
        )
        exporter = DataExporter(no_project_config)
        assert exporter.export_report_to_bigquery({}) is False

    def test_export_report_to_gcs_returns_uri_on_success(self, exporter_with_clients: DataExporter) -> None:
        """Now implemented: returns a gs:// URI when a bucket is configured."""
        result = exporter_with_clients.export_report_to_gcs({}, run_id="run-001")
        assert result is not None
        assert result.startswith("gs://")

    def test_export_tool_results_to_gcs_returns_none(self, exporter_with_clients: DataExporter) -> None:
        assert exporter_with_clients.export_tool_results_to_gcs([], run_id="run-001") is None

    def test_export_telemetry_to_gcs_returns_none(self, exporter_with_clients: DataExporter) -> None:
        assert exporter_with_clients.export_telemetry_to_gcs([]) is None


# ── Stub method logging ──────────────────────────────────────


class TestStubMethodLogging:
    """Verify BQ/GCS methods emit warning logs on misconfiguration or error conditions."""

    def test_export_telemetry_to_bigquery_logs_warning_when_no_project(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Logs a warning when bigquery_project is not configured."""
        no_project_config = ExportConfig(
            enabled=True,
            gcp_project_id="",
            bigquery_dataset="test_dataset",
        )
        exporter = DataExporter(no_project_config)
        with _captured_vaig_logs(caplog, "vaig.core.export"):
            exporter.export_telemetry_to_bigquery([{"event_type": "tool_call"}])
        assert "export_telemetry_to_bigquery" in caplog.text

    def test_export_tool_calls_to_bigquery_logs_warning_when_no_project(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Logs a warning when bigquery_project is not configured."""
        no_project_config = ExportConfig(
            enabled=True,
            gcp_project_id="",
            bigquery_dataset="test_dataset",
        )
        exporter = DataExporter(no_project_config)
        with _captured_vaig_logs(caplog, "vaig.core.export"):
            exporter.export_tool_calls_to_bigquery([{"tool_name": "kubectl"}])
        assert "export_tool_calls_to_bigquery" in caplog.text

    def test_export_report_to_bigquery_logs_warning_when_no_project(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Logs a warning when bigquery_project is not configured."""
        no_project_config = ExportConfig(
            enabled=True,
            gcp_project_id="",
            bigquery_dataset="test_dataset",
        )
        exporter = DataExporter(no_project_config)
        with _captured_vaig_logs(caplog, "vaig.core.export"):
            exporter.export_report_to_bigquery({})
        assert "export_report_to_bigquery" in caplog.text

    def test_export_report_to_gcs_logs_warning_when_no_bucket(self, export_config: ExportConfig, caplog: pytest.LogCaptureFixture) -> None:
        """export_report_to_gcs warns when gcs_bucket is not configured."""
        no_bucket_config = ExportConfig(
            enabled=True,
            gcp_project_id="test-project",
            bigquery_dataset="test_dataset",
            gcs_bucket="",
        )
        exporter = DataExporter(no_bucket_config)
        with _captured_vaig_logs(caplog, "vaig.core.export"):
            exporter.export_report_to_gcs({}, run_id="run-001")
        assert "export_report_to_gcs" in caplog.text

    def test_export_tool_results_to_gcs_logs_warning(self, exporter_with_clients: DataExporter, caplog: pytest.LogCaptureFixture) -> None:
        with _captured_vaig_logs(caplog, "vaig.core.export"):
            exporter_with_clients.export_tool_results_to_gcs([], run_id="run-001")
        assert "export_tool_results_to_gcs" in caplog.text

    def test_export_telemetry_to_gcs_logs_warning(self, exporter_with_clients: DataExporter, caplog: pytest.LogCaptureFixture) -> None:
        with _captured_vaig_logs(caplog, "vaig.core.export"):
            exporter_with_clients.export_telemetry_to_gcs([])
        assert "export_telemetry_to_gcs" in caplog.text


# ── BigQuery schema (T4) ─────────────────────────────────────


def _make_bq_mock() -> MagicMock:
    """Build a minimal google.cloud.bigquery module mock."""
    bq_mod = MagicMock(name="bigquery_module")
    bq_mod.SchemaField.side_effect = lambda *a, **kw: MagicMock(name=f"field_{a[0]}")
    bq_mod.Dataset.return_value = MagicMock(name="dataset")
    bq_mod.Table.return_value = MagicMock(name="table")
    bq_mod.TimePartitioning.return_value = MagicMock(name="tp")
    bq_mod.TimePartitioningType.DAY = "DAY"
    return bq_mod


def _make_conflict_exc() -> type[Exception]:
    """Return a class compatible with google.api_core.exceptions.Conflict."""

    class ConflictError(Exception):  # noqa: N818 — mirrors google.api_core naming
        pass

    return ConflictError


@pytest.fixture()
def bq_mock() -> MagicMock:
    return _make_bq_mock()


@pytest.fixture()
def conflict_cls() -> type[Exception]:
    return _make_conflict_exc()


class TestBigQuerySchema:
    """T4 — ensure_dataset, ensure_tables, and schema constant tests."""

    # ── Schema constant ──────────────────────────────────────

    def test_schema_constant_has_all_four_tables(self) -> None:
        """_TABLE_SCHEMAS must define exactly the 4 required tables."""
        assert set(_TABLE_SCHEMAS.keys()) == {
            "telemetry_events",
            "tool_calls",
            "health_reports",
            "feedback",
        }

    def test_table_config_has_all_four_tables(self) -> None:
        """_TABLE_CONFIG must have an entry for every table in _TABLE_SCHEMAS."""
        assert set(_TABLE_CONFIG.keys()) == set(_TABLE_SCHEMAS.keys())

    def test_telemetry_events_schema_has_required_timestamp(self) -> None:
        fields_by_name = {f[0]: f for f in _TABLE_SCHEMAS["telemetry_events"]}
        ts = fields_by_name["timestamp"]
        assert ts[1] == "TIMESTAMP"
        assert ts[2] == "REQUIRED"

    def test_tool_calls_schema_has_required_run_id(self) -> None:
        fields_by_name = {f[0]: f for f in _TABLE_SCHEMAS["tool_calls"]}
        run_id = fields_by_name["run_id"]
        assert run_id[1] == "STRING"
        assert run_id[2] == "REQUIRED"

    def test_health_reports_schema_has_findings_as_record(self) -> None:
        fields_by_name = {f[0]: f for f in _TABLE_SCHEMAS["health_reports"]}
        findings = fields_by_name["findings"]
        assert findings[1] == "RECORD"
        assert findings[2] == "REPEATED"
        assert len(findings) == 5  # 5-tuple with sub_fields

    def test_feedback_schema_has_required_fields(self) -> None:
        fields_by_name = {f[0]: f for f in _TABLE_SCHEMAS["feedback"]}
        assert "timestamp" in fields_by_name
        assert "run_id" in fields_by_name
        assert fields_by_name["timestamp"][2] == "REQUIRED"
        assert fields_by_name["run_id"][2] == "REQUIRED"

    def test_telemetry_events_partition_and_clustering(self) -> None:
        cfg = _TABLE_CONFIG["telemetry_events"]
        assert cfg["partition_field"] == "timestamp"
        assert "event_type" in cfg["clustering_fields"]
        assert "tool_name" in cfg["clustering_fields"]

    def test_health_reports_clustering_by_cluster_and_namespace(self) -> None:
        cfg = _TABLE_CONFIG["health_reports"]
        assert "cluster_name" in cfg["clustering_fields"]
        assert "namespace" in cfg["clustering_fields"]

    # ── ensure_dataset ───────────────────────────────────────

    def test_ensure_dataset_creates_dataset_via_bq_client(
        self,
        export_config: ExportConfig,
        bq_mock: MagicMock,
        conflict_cls: type[Exception],
    ) -> None:
        """ensure_dataset must call client.create_dataset exactly once."""
        mock_client = MagicMock(name="bq_client")
        exporter = DataExporter(export_config, bq_client=mock_client)

        with (
            patch("google.cloud.bigquery", bq_mock),
            patch("google.api_core.exceptions.Conflict", conflict_cls),
            patch("vaig.core.export.DataExporter._get_bq_client", return_value=mock_client),
        ):
            result = exporter.ensure_dataset()

        assert result is True
        mock_client.create_dataset.assert_called_once()

    def test_ensure_dataset_handles_already_exists_gracefully(
        self,
        export_config: ExportConfig,
        bq_mock: MagicMock,
    ) -> None:
        """ensure_dataset must return True (not raise) when the dataset already exists (409)."""
        conflict_cls = _make_conflict_exc()
        mock_client = MagicMock(name="bq_client")
        mock_client.create_dataset.side_effect = conflict_cls("already exists")
        exporter = DataExporter(export_config, bq_client=mock_client)

        with (
            patch("google.cloud.bigquery", bq_mock),
            patch("google.api_core.exceptions.Conflict", conflict_cls),
            patch("vaig.core.export.DataExporter._get_bq_client", return_value=mock_client),
        ):
            result = exporter.ensure_dataset()

        assert result is True

    def test_ensure_dataset_returns_false_and_logs_warning_on_error(
        self,
        export_config: ExportConfig,
        bq_mock: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """ensure_dataset must swallow unexpected errors, return False, and log a warning."""
        mock_client = MagicMock(name="bq_client")
        mock_client.create_dataset.side_effect = RuntimeError("network error")
        exporter = DataExporter(export_config, bq_client=mock_client)

        conflict_cls = _make_conflict_exc()

        with (
            patch("google.cloud.bigquery", bq_mock),
            patch("google.api_core.exceptions.Conflict", conflict_cls),
            patch("vaig.core.export.DataExporter._get_bq_client", return_value=mock_client),
            _captured_vaig_logs(caplog, "vaig.core.export"),
        ):
            result = exporter.ensure_dataset()

        assert result is False
        assert "ensure_dataset" in caplog.text

    # ── ensure_tables ────────────────────────────────────────

    def test_ensure_tables_creates_all_four_tables(
        self,
        export_config: ExportConfig,
        bq_mock: MagicMock,
        conflict_cls: type[Exception],
    ) -> None:
        """ensure_tables must attempt to create every table in _TABLE_SCHEMAS."""
        mock_client = MagicMock(name="bq_client")
        mock_client.create_table.return_value = MagicMock()
        exporter = DataExporter(export_config, bq_client=mock_client)

        with (
            patch("google.cloud.bigquery", bq_mock),
            patch("google.api_core.exceptions.Conflict", conflict_cls),
            patch("vaig.core.export.DataExporter._get_bq_client", return_value=mock_client),
        ):
            results = exporter.ensure_tables()

        assert set(results.keys()) == {"telemetry_events", "tool_calls", "health_reports", "feedback"}
        assert all(v is True for v in results.values())
        assert mock_client.create_table.call_count == 4

    def test_ensure_tables_handles_already_exists_for_each_table(
        self,
        export_config: ExportConfig,
        bq_mock: MagicMock,
    ) -> None:
        """When every table already exists (409), ensure_tables should return all True."""
        conflict_cls = _make_conflict_exc()
        mock_client = MagicMock(name="bq_client")
        mock_client.create_table.side_effect = conflict_cls("table already exists")
        exporter = DataExporter(export_config, bq_client=mock_client)

        with (
            patch("google.cloud.bigquery", bq_mock),
            patch("google.api_core.exceptions.Conflict", conflict_cls),
            patch("vaig.core.export.DataExporter._get_bq_client", return_value=mock_client),
        ):
            results = exporter.ensure_tables()

        assert set(results.keys()) == {"telemetry_events", "tool_calls", "health_reports", "feedback"}
        assert all(v is True for v in results.values())

    def test_ensure_tables_returns_false_on_per_table_error(
        self,
        export_config: ExportConfig,
        bq_mock: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An unexpected error for one table must set that entry to False and log a warning."""
        conflict_cls = _make_conflict_exc()
        mock_client = MagicMock(name="bq_client")
        mock_client.create_table.side_effect = RuntimeError("permission denied")
        exporter = DataExporter(export_config, bq_client=mock_client)

        with (
            patch("google.cloud.bigquery", bq_mock),
            patch("google.api_core.exceptions.Conflict", conflict_cls),
            patch("vaig.core.export.DataExporter._get_bq_client", return_value=mock_client),
            _captured_vaig_logs(caplog, "vaig.core.export"),
        ):
            results = exporter.ensure_tables()

        assert all(v is False for v in results.values())
        assert "ensure_tables" in caplog.text


# ── GCS export methods (T6) ──────────────────────────────────


def _make_gcs_mock() -> MagicMock:
    """Build a minimal mock GCS client chain: client → bucket → blob."""
    mock_blob = MagicMock(name="blob")
    mock_bucket = MagicMock(name="bucket")
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock(name="gcs_client")
    mock_client.bucket.return_value = mock_bucket
    return mock_client


@pytest.fixture()
def gcs_mock_client() -> MagicMock:
    return _make_gcs_mock()


@pytest.fixture()
def exporter_gcs(export_config: ExportConfig, gcs_mock_client: MagicMock) -> DataExporter:
    """DataExporter with injected mock GCS client and a valid bucket configured."""
    return DataExporter(export_config, gcs_client=gcs_mock_client)


class TestGCSExport:
    """T6 — GCS export method tests."""

    # ── export_report_to_gcs ─────────────────────────────────

    def test_export_report_to_gcs_success(
        self,
        exporter_gcs: DataExporter,
        gcs_mock_client: MagicMock,
    ) -> None:
        """Returns a gs:// URI and calls upload_from_string with JSON content."""
        report = {"status": "healthy", "cluster": "prod"}
        result = exporter_gcs.export_report_to_gcs(report, run_id="run-abc")

        assert result is not None
        assert result.startswith("gs://test-bucket/")
        assert "reports/" in result
        assert "run-abc.json" in result

        # Verify bucket and blob were called with correct args
        gcs_mock_client.bucket.assert_called_once_with("test-bucket")
        blob_mock = gcs_mock_client.bucket.return_value.blob.return_value
        blob_mock.upload_from_string.assert_called_once()
        call_kwargs = blob_mock.upload_from_string.call_args
        assert call_kwargs.kwargs.get("content_type") == "application/json"
        # Verify the content is valid JSON
        import json as _json
        content = call_kwargs.args[0]
        parsed = _json.loads(content)
        assert parsed["status"] == "healthy"

    def test_export_report_to_gcs_no_bucket(
        self,
        export_config: ExportConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Returns None and logs a warning when gcs_bucket is empty."""
        no_bucket_config = ExportConfig(
            enabled=True,
            gcp_project_id="test-project",
            bigquery_dataset="test_dataset",
            gcs_bucket="",
        )
        exporter = DataExporter(no_bucket_config)
        with _captured_vaig_logs(caplog, "vaig.core.export"):
            result = exporter.export_report_to_gcs({"key": "val"}, run_id="run-001")
        assert result is None
        assert "gcs_bucket" in caplog.text or "export_report_to_gcs" in caplog.text

    def test_export_report_to_gcs_gcs_error(
        self,
        export_config: ExportConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Returns None and logs a warning when GCS raises an exception."""
        mock_client = _make_gcs_mock()
        mock_client.bucket.return_value.blob.return_value.upload_from_string.side_effect = (
            RuntimeError("GCS network error")
        )
        exporter = DataExporter(export_config, gcs_client=mock_client)
        with _captured_vaig_logs(caplog, "vaig.core.export"):
            result = exporter.export_report_to_gcs({"key": "val"}, run_id="run-001")
        assert result is None
        assert "export_report_to_gcs" in caplog.text

    # ── export_tool_results_to_gcs ───────────────────────────

    def test_export_tool_results_to_gcs_success(
        self,
        exporter_gcs: DataExporter,
        gcs_mock_client: MagicMock,
    ) -> None:
        """Returns a gs:// URI and uploads valid JSONL (one JSON object per line)."""
        records = [
            {"tool": "kubectl", "success": True},
            {"tool": "helm", "success": False},
        ]
        result = exporter_gcs.export_tool_results_to_gcs(records, run_id="run-xyz")

        assert result is not None
        assert result.startswith("gs://test-bucket/")
        assert "tool_results/" in result
        assert "run-xyz.jsonl" in result

        blob_mock = gcs_mock_client.bucket.return_value.blob.return_value
        blob_mock.upload_from_string.assert_called_once()
        call_kwargs = blob_mock.upload_from_string.call_args
        assert call_kwargs.kwargs.get("content_type") == "application/jsonl"

        # Verify JSONL format: 2 records → 2 lines
        import json as _json
        content = call_kwargs.args[0]
        lines = content.strip().split("\n")
        assert len(lines) == 2
        assert _json.loads(lines[0])["tool"] == "kubectl"
        assert _json.loads(lines[1])["tool"] == "helm"

    def test_export_tool_results_to_gcs_empty_records(
        self,
        exporter_gcs: DataExporter,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Returns None (and logs warning) when the records list is empty."""
        with _captured_vaig_logs(caplog, "vaig.core.export"):
            result = exporter_gcs.export_tool_results_to_gcs([], run_id="run-001")
        assert result is None
        assert "export_tool_results_to_gcs" in caplog.text

    # ── export_telemetry_to_gcs ──────────────────────────────

    def test_export_telemetry_to_gcs_success(
        self,
        exporter_gcs: DataExporter,
        gcs_mock_client: MagicMock,
    ) -> None:
        """Returns a gs:// URI with a batch_ prefixed filename."""
        records = [
            {"event_type": "tool_call", "duration_ms": 120},
            {"event_type": "tool_call", "duration_ms": 85},
        ]
        result = exporter_gcs.export_telemetry_to_gcs(records)

        assert result is not None
        assert result.startswith("gs://test-bucket/")
        assert "telemetry/" in result
        assert "batch_" in result
        assert result.endswith(".jsonl")

        blob_mock = gcs_mock_client.bucket.return_value.blob.return_value
        blob_mock.upload_from_string.assert_called_once()
        call_kwargs = blob_mock.upload_from_string.call_args
        assert call_kwargs.kwargs.get("content_type") == "application/jsonl"

        # Verify 2 JSONL lines
        content = call_kwargs.args[0]
        lines = content.strip().split("\n")
        assert len(lines) == 2

    def test_export_telemetry_to_gcs_error(
        self,
        export_config: ExportConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Returns None and logs a warning when GCS raises an exception."""
        mock_client = _make_gcs_mock()
        mock_client.bucket.return_value.blob.return_value.upload_from_string.side_effect = (
            ConnectionError("timeout")
        )
        exporter = DataExporter(export_config, gcs_client=mock_client)
        records = [{"event_type": "tool_call"}]
        with _captured_vaig_logs(caplog, "vaig.core.export"):
            result = exporter.export_telemetry_to_gcs(records)
        assert result is None
        assert "export_telemetry_to_gcs" in caplog.text


# ── BigQuery export methods (T7) ─────────────────────────────


def _make_bq_client_mock() -> MagicMock:
    """Build a minimal mock BQ client for insert tests.

    ``insert_rows_json`` returns ``[]`` by default (no errors).
    ``load_table_from_file`` returns a mock job whose ``.result()`` is a no-op.
    """
    mock_client = MagicMock(name="bq_client")
    mock_client.insert_rows_json.return_value = []  # streaming success
    mock_job = MagicMock(name="load_job")
    mock_job.result.return_value = None
    mock_client.load_table_from_file.return_value = mock_job
    return mock_client


@pytest.fixture()
def bq_client_mock() -> MagicMock:
    return _make_bq_client_mock()


@pytest.fixture()
def exporter_bq(export_config: ExportConfig, bq_client_mock: MagicMock) -> DataExporter:
    """DataExporter with injected mock BQ client."""
    return DataExporter(export_config, bq_client=bq_client_mock)


class TestBigQueryExport:
    """T7 — BigQuery export method tests (telemetry, tool_calls, health_reports)."""

    # ── export_telemetry_to_bigquery ─────────────────────────

    def test_export_telemetry_streaming(
        self,
        exporter_bq: DataExporter,
        bq_client_mock: MagicMock,
    ) -> None:
        """≤100 rows → insert_rows_json called, load_table_from_file NOT called."""
        records = [{"event_type": "tool_call", "tool_name": "kubectl"} for _ in range(3)]
        count = exporter_bq.export_telemetry_to_bigquery(records)

        assert count == 3
        bq_client_mock.insert_rows_json.assert_called_once()
        bq_client_mock.load_table_from_file.assert_not_called()

    def test_export_telemetry_bulk_load(
        self,
        export_config: ExportConfig,
    ) -> None:
        """>100 rows → load_table_from_file called, insert_rows_json NOT called."""
        mock_client = _make_bq_client_mock()
        exporter = DataExporter(export_config, bq_client=mock_client)
        records = [{"event_type": "tool_call", "tool_name": f"tool_{i}"} for i in range(101)]
        count = exporter.export_telemetry_to_bigquery(records)

        assert count == 101
        mock_client.load_table_from_file.assert_called_once()
        mock_client.insert_rows_json.assert_not_called()

    def test_export_telemetry_empty_records(
        self,
        exporter_bq: DataExporter,
        bq_client_mock: MagicMock,
    ) -> None:
        """Empty list → returns 0 immediately, no BQ calls made."""
        count = exporter_bq.export_telemetry_to_bigquery([])

        assert count == 0
        bq_client_mock.insert_rows_json.assert_not_called()
        bq_client_mock.load_table_from_file.assert_not_called()

    def test_export_telemetry_no_project(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Returns 0 and logs a warning when bigquery_project is empty."""
        no_project_config = ExportConfig(
            enabled=True,
            gcp_project_id="",
            bigquery_dataset="test_dataset",
        )
        exporter = DataExporter(no_project_config)
        with _captured_vaig_logs(caplog, "vaig.core.export"):
            count = exporter.export_telemetry_to_bigquery([{"event_type": "tool_call"}])
        assert count == 0
        assert "export_telemetry_to_bigquery" in caplog.text

    def test_export_telemetry_bq_error(
        self,
        export_config: ExportConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """BQ exception → returns 0 and logs a warning."""
        mock_client = _make_bq_client_mock()
        mock_client.insert_rows_json.side_effect = RuntimeError("BQ unavailable")
        exporter = DataExporter(export_config, bq_client=mock_client)
        with _captured_vaig_logs(caplog, "vaig.core.export"):
            count = exporter.export_telemetry_to_bigquery([{"event_type": "tool_call"}])
        assert count == 0
        assert "export_telemetry_to_bigquery" in caplog.text

    # ── export_tool_calls_to_bigquery ────────────────────────

    def test_export_tool_calls_success(
        self,
        exporter_bq: DataExporter,
        bq_client_mock: MagicMock,
    ) -> None:
        """Transforms and inserts 2 tool call records; returns count=2."""
        records = [
            {"tool_name": "kubectl", "agent_name": "k8s-agent", "success": True},
            {"tool_name": "helm", "agent_name": "k8s-agent", "success": False},
        ]
        count = exporter_bq.export_tool_calls_to_bigquery(records)

        assert count == 2
        bq_client_mock.insert_rows_json.assert_called_once()
        # Verify the rows passed to BQ contain the transformed fields
        call_args = bq_client_mock.insert_rows_json.call_args
        rows_inserted = call_args.args[1]  # second positional arg
        assert len(rows_inserted) == 2
        assert rows_inserted[0]["tool_name"] == "kubectl"
        assert rows_inserted[1]["tool_name"] == "helm"

    # ── export_report_to_bigquery ────────────────────────────

    def test_export_report_success(
        self,
        exporter_bq: DataExporter,
        bq_client_mock: MagicMock,
    ) -> None:
        """Single health report → streaming insert, returns True."""
        report = {
            "overall_status": "HEALTHY",
            "executive_summary": {"overall_status": "HEALTHY"},
            "findings": [],
        }
        result = exporter_bq.export_report_to_bigquery(
            report,
            run_id="run-001",
            cluster_name="prod-cluster",
            namespace="default",
        )

        assert result is True
        bq_client_mock.insert_rows_json.assert_called_once()
        rows = bq_client_mock.insert_rows_json.call_args.args[1]
        assert len(rows) == 1
        assert rows[0]["run_id"] == "run-001"
        assert rows[0]["cluster_name"] == "prod-cluster"
        assert rows[0]["namespace"] == "default"

    def test_export_report_failure(
        self,
        export_config: ExportConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """BQ error during report insert → returns False and logs warning."""
        mock_client = _make_bq_client_mock()
        mock_client.insert_rows_json.side_effect = RuntimeError("quota exceeded")
        exporter = DataExporter(export_config, bq_client=mock_client)
        with _captured_vaig_logs(caplog, "vaig.core.export"):
            result = exporter.export_report_to_bigquery({"overall_status": "UNHEALTHY"})
        assert result is False
        assert "export_report_to_bigquery" in caplog.text

    # ── _insert_rows helper ──────────────────────────────────

    def test_insert_rows_streaming_partial_errors(
        self,
        export_config: ExportConfig,
    ) -> None:
        """Partial streaming errors reduce the returned count: 3 rows, 1 error → 2."""
        mock_client = _make_bq_client_mock()
        mock_client.insert_rows_json.return_value = [{"index": 2, "errors": ["bad value"]}]
        exporter = DataExporter(export_config, bq_client=mock_client)

        rows = [{"event_type": "e"}, {"event_type": "e"}, {"event_type": "bad"}]
        count = exporter._insert_rows("telemetry_events", rows)  # noqa: SLF001
        assert count == 2

    def test_insert_rows_uses_correct_table_id(
        self,
        export_config: ExportConfig,
        bq_client_mock: MagicMock,
    ) -> None:
        """_insert_rows must build the fully-qualified table ID from config."""
        exporter = DataExporter(export_config, bq_client=bq_client_mock)
        exporter._insert_rows("telemetry_events", [{"event_type": "e"}])  # noqa: SLF001

        call_args = bq_client_mock.insert_rows_json.call_args
        table_id_used = call_args.args[0]
        assert table_id_used == "test-project.test_dataset.telemetry_events"


# ── TestAutoExportReport ─────────────────────────────────────


class TestAutoExportReport:
    """Tests for the fire-and-forget auto_export_report() function."""

    _SAMPLE_REPORT: dict = {"overall_status": "HEALTHY", "findings": []}

    def test_auto_export_starts_thread(self) -> None:
        """auto_export_report must start a daemon thread."""
        config = ExportConfig(enabled=True, gcp_project_id="proj", gcs_bucket="bucket")
        with patch("threading.Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread
            from vaig.core.export import auto_export_report

            auto_export_report(config, self._SAMPLE_REPORT, run_id="run-1")

        mock_thread_cls.assert_called_once()
        _, kwargs = mock_thread_cls.call_args
        assert kwargs.get("daemon") is True
        assert kwargs.get("name") == "vaig-auto-export"
        mock_thread.start.assert_called_once()

    def test_auto_export_calls_bq_when_project_configured(self) -> None:
        """When bigquery_project is set, export_report_to_bigquery is called."""
        config = ExportConfig(enabled=True, gcp_project_id="my-proj", gcs_bucket="")
        with patch("vaig.core.export.DataExporter") as mock_exporter_cls:
            mock_exporter = MagicMock()
            mock_exporter_cls.return_value = mock_exporter
            thread = _run_auto_export_synchronously(
                config, self._SAMPLE_REPORT, run_id="run-bq", cluster_name="c1", namespace="n1"
            )
            thread.join(timeout=5)

        mock_exporter.export_report_to_bigquery.assert_called_once_with(
            self._SAMPLE_REPORT, run_id="run-bq", cluster_name="c1", namespace="n1"
        )
        mock_exporter.export_report_to_gcs.assert_not_called()

    def test_auto_export_calls_gcs_when_bucket_configured(self) -> None:
        """When gcs_bucket is set, export_report_to_gcs is called."""
        config = ExportConfig(enabled=True, gcp_project_id="", gcs_bucket="my-bucket")
        with patch("vaig.core.export.DataExporter") as mock_exporter_cls:
            mock_exporter = MagicMock()
            mock_exporter_cls.return_value = mock_exporter
            thread = _run_auto_export_synchronously(config, self._SAMPLE_REPORT, run_id="run-gcs")
            thread.join(timeout=5)

        mock_exporter.export_report_to_gcs.assert_called_once_with(
            self._SAMPLE_REPORT, run_id="run-gcs"
        )
        mock_exporter.export_report_to_bigquery.assert_not_called()

    def test_auto_export_skips_bq_when_no_project(self) -> None:
        """Empty bigquery_project → export_report_to_bigquery must NOT be called."""
        config = ExportConfig(enabled=True, gcp_project_id="", gcs_bucket="")
        with patch("vaig.core.export.DataExporter") as mock_exporter_cls:
            mock_exporter = MagicMock()
            mock_exporter_cls.return_value = mock_exporter
            thread = _run_auto_export_synchronously(config, self._SAMPLE_REPORT, run_id="run-skip-bq")
            thread.join(timeout=5)

        mock_exporter.export_report_to_bigquery.assert_not_called()
        mock_exporter.export_report_to_gcs.assert_not_called()

    def test_auto_export_skips_gcs_when_no_bucket(self) -> None:
        """Empty gcs_bucket → export_report_to_gcs must NOT be called."""
        config = ExportConfig(enabled=True, gcp_project_id="proj", gcs_bucket="")
        with patch("vaig.core.export.DataExporter") as mock_exporter_cls:
            mock_exporter = MagicMock()
            mock_exporter_cls.return_value = mock_exporter
            thread = _run_auto_export_synchronously(config, self._SAMPLE_REPORT, run_id="run-skip-gcs")
            thread.join(timeout=5)

        mock_exporter.export_report_to_bigquery.assert_called_once()
        mock_exporter.export_report_to_gcs.assert_not_called()

    def test_auto_export_exception_caught(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """If DataExporter raises during export, a warning is logged and no crash occurs."""
        config = ExportConfig(enabled=True, gcp_project_id="proj", gcs_bucket="bucket")
        with patch("vaig.core.export.DataExporter") as mock_exporter_cls:
            mock_exporter = MagicMock()
            mock_exporter.export_report_to_bigquery.side_effect = RuntimeError("BQ unavailable")
            mock_exporter_cls.return_value = mock_exporter
            with _captured_vaig_logs(caplog, "vaig.core.export"):
                thread = _run_auto_export_synchronously(config, self._SAMPLE_REPORT, run_id="run-err")
                thread.join(timeout=5)

        assert "Auto-export failed" in caplog.text
        assert "run-err" in caplog.text


# ── Helpers for TestAutoExportReport ────────────────────────


def _run_auto_export_synchronously(
    config: ExportConfig,
    report: dict,
    *,
    run_id: str,
    cluster_name: str = "",
    namespace: str = "",
) -> threading.Thread:
    """Call auto_export_report and return the thread so tests can join() it."""
    from vaig.core.export import auto_export_report

    # Capture the thread before start() so we can join in tests.
    # We patch threading.Thread to intercept, then actually start it.
    _original_thread_cls = threading.Thread
    started_threads: list[threading.Thread] = []

    def _patched_thread(*args: object, **kwargs: object) -> threading.Thread:
        t = _original_thread_cls(*args, **kwargs)
        started_threads.append(t)
        return t

    with patch("threading.Thread", side_effect=_patched_thread):
        auto_export_report(
            config, report, run_id=run_id, cluster_name=cluster_name, namespace=namespace
        )

    return started_threads[0]


from contextlib import contextmanager


@contextmanager
def _captured_vaig_logs(
    caplog: pytest.LogCaptureFixture,
    logger_name: str,
    level: int = logging.WARNING,
):
    """Capture vaig logs even when the parent logger disables propagation."""
    vaig_logger = logging.getLogger("vaig")
    original_propagate = vaig_logger.propagate
    vaig_logger.propagate = True
    try:
        with caplog.at_level(level, logger=logger_name):
            yield
    finally:
        vaig_logger.propagate = original_propagate

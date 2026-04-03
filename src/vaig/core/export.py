"""RAG data pipeline export — BigQuery and GCS export."""

from __future__ import annotations

import io
import json
import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.cloud import bigquery, storage

from vaig.core.config import ExportConfig, Settings
from vaig.core.export_transformers import (
    transform_feedback_record,
    transform_health_report,
    transform_telemetry_record,
    transform_tool_call_record,
)
from vaig.core.telemetry import TelemetryCollector
from vaig.core.tool_call_store import ToolCallStore

if TYPE_CHECKING:
    from vaig.skills.service_health.schema import HealthReport

logger = logging.getLogger(__name__)

# Lazy import of google.api_core.exceptions — available when [rag] deps are installed.
# Methods that already require rag deps can safely use _GCP_EXC for specific exception catches.
try:
    from google.api_core import exceptions as _gcp_exc
except ImportError:  # pragma: no cover
    _gcp_exc = None  # type: ignore[assignment]

# Exception tuple for GCP export methods. Covers GoogleAPICallError (network / quota /
# permission failures), RuntimeError (GCP SDK internal errors), ValueError (bad data),
# and OSError (I/O).  Falls back to the base Exception class when google.api_core is
# not installed so behaviour is unchanged.
_GCP_EXPORT_ERRORS: tuple[type[Exception], ...] = (
    ((_gcp_exc.GoogleAPICallError,) if _gcp_exc is not None else ())
    + (RuntimeError, ValueError, OSError)
)

_RETRYABLE_GCP_ERROR_NAMES = {
    "TooManyRequests",
    "ServiceUnavailable",
    "InternalServerError",
    "BadGateway",
    "GatewayTimeout",
}


# ── Dependency guard ─────────────────────────────────────────


def _require_rag_deps() -> None:
    """Raise ImportError with install instructions if [rag] deps are missing."""
    try:
        import google.cloud.bigquery  # noqa: F401
        import google.cloud.storage  # noqa: F401
    except ImportError:
        raise ImportError(
            "Export features require the [rag] extras. "
            "Install with: pip install 'vertex-ai-toolkit[rag]'"
        ) from None


class ExportResult:
    """Result metadata for a single export operation."""

    def __init__(
        self,
        *,
        bigquery_rows: int = 0,
        gcs_uri: str | None = None,
        success: bool = False,
        destination: str = "both",
        error: str = "",
    ) -> None:
        """Create an export result summary."""
        self.bigquery_rows = bigquery_rows
        self.gcs_uri = gcs_uri
        self.success = success
        self.destination = destination
        self.error = error

    @property
    def exported(self) -> bool:
        """Return True when any destination accepted data."""
        return self.success


def _is_transient_gcp_error(exc: Exception) -> bool:
    """Return True when *exc* looks retryable for GCP network/service failures."""
    return exc.__class__.__name__ in _RETRYABLE_GCP_ERROR_NAMES


def _coerce_report_dict(report: HealthReport | dict[str, Any]) -> dict[str, Any]:
    """Convert supported report inputs into a JSON-serializable dict."""
    if hasattr(report, "to_dict"):
        return dict(report.to_dict())
    if isinstance(report, dict):
        return report
    if hasattr(report, "model_dump"):
        return dict(report.model_dump())
    raise TypeError(f"Unsupported report type for export: {type(report).__name__}")


# ── BigQuery table schemas (plain tuples — no import at module level) ────────
#
# Each entry is a 4-tuple: (name, field_type, mode, description)
# Nested STRUCT fields use a 5-tuple: (name, field_type, mode, description, sub_fields)
# Converted to bigquery.SchemaField objects lazily inside ensure_tables().

_TABLE_SCHEMAS: dict[str, list[tuple[Any, ...]]] = {
    "telemetry_events": [
        ("timestamp", "TIMESTAMP", "REQUIRED", "Event timestamp"),
        ("event_type", "STRING", "REQUIRED", "Type of event"),
        ("tool_name", "STRING", "NULLABLE", "Name of the tool that generated the event"),
        ("agent_name", "STRING", "NULLABLE", "Name of the agent that ran the tool"),
        ("duration_ms", "FLOAT64", "NULLABLE", "Execution duration in milliseconds"),
        ("success", "BOOL", "NULLABLE", "Whether the event completed successfully"),
        ("error_message", "STRING", "NULLABLE", "Error message if the event failed"),
        ("metadata", "JSON", "NULLABLE", "Arbitrary metadata as JSON"),
        ("session_id", "STRING", "NULLABLE", "Session identifier"),
        ("run_id", "STRING", "NULLABLE", "Pipeline run identifier"),
    ],
    "tool_calls": [
        ("timestamp", "TIMESTAMP", "REQUIRED", "Call timestamp"),
        ("tool_name", "STRING", "REQUIRED", "Name of the tool called"),
        ("agent_name", "STRING", "NULLABLE", "Agent that invoked the tool"),
        ("input_params", "JSON", "NULLABLE", "Tool input parameters as JSON"),
        ("output_summary", "STRING", "NULLABLE", "Short summary of tool output"),
        ("duration_ms", "FLOAT64", "NULLABLE", "Execution duration in milliseconds"),
        ("success", "BOOL", "NULLABLE", "Whether the call succeeded"),
        ("error_message", "STRING", "NULLABLE", "Error message if the call failed"),
        ("run_id", "STRING", "REQUIRED", "Pipeline run identifier"),
        ("session_id", "STRING", "NULLABLE", "Session identifier"),
    ],
    "health_reports": [
        ("timestamp", "TIMESTAMP", "REQUIRED", "Report timestamp"),
        ("run_id", "STRING", "REQUIRED", "Pipeline run identifier"),
        ("cluster_name", "STRING", "NULLABLE", "Kubernetes cluster name"),
        ("namespace", "STRING", "NULLABLE", "Kubernetes namespace"),
        ("overall_status", "STRING", "NULLABLE", "Aggregated health status"),
        ("summary", "STRING", "NULLABLE", "Plain-text summary of the report"),
        (
            "findings",
            "RECORD",
            "REPEATED",
            "Structured health findings",
            [
                ("category", "STRING", "NULLABLE", "Finding category"),
                ("severity", "STRING", "NULLABLE", "Severity level"),
                ("title", "STRING", "NULLABLE", "Short title"),
                ("description", "STRING", "NULLABLE", "Detailed description"),
                ("recommendation", "STRING", "NULLABLE", "Remediation recommendation"),
            ],
        ),
        ("metadata", "JSON", "NULLABLE", "Arbitrary metadata as JSON"),
        ("report_markdown", "STRING", "NULLABLE", "Full report in Markdown format"),
    ],
    "feedback": [
        ("timestamp", "TIMESTAMP", "REQUIRED", "Feedback timestamp"),
        ("run_id", "STRING", "REQUIRED", "Pipeline run identifier"),
        ("rating", "INT64", "NULLABLE", "User rating (1-5)"),
        ("comment", "STRING", "NULLABLE", "Free-text feedback comment"),
        ("auto_quality_score", "FLOAT64", "NULLABLE", "Automated quality score"),
        ("report_summary", "STRING", "NULLABLE", "Summary of the associated report"),
        ("metadata", "JSON", "NULLABLE", "Arbitrary metadata as JSON"),
    ],
}

_TABLE_CONFIG: dict[str, dict[str, Any]] = {
    "telemetry_events": {
        "partition_field": "timestamp",
        "clustering_fields": ["event_type", "tool_name"],
    },
    "tool_calls": {
        "partition_field": "timestamp",
        "clustering_fields": ["tool_name", "agent_name"],
    },
    "health_reports": {
        "partition_field": "timestamp",
        "clustering_fields": ["cluster_name", "namespace"],
    },
    "feedback": {
        "partition_field": "timestamp",
        "clustering_fields": [],
    },
}


# ── DataExporter ─────────────────────────────────────────────


class DataExporter:
    """Exports vaig data to BigQuery and GCS. Thread-safe, lazy-initialized.

    All GCP clients are created on first use (lazy initialization) so that
    importing this module never fails even when the ``[rag]`` extras are not
    installed.  Pass pre-built clients via constructor injection for testing.

    Usage::

        config = settings.export
        exporter = DataExporter(config)
        exporter.ensure_dataset()
        count = exporter.export_telemetry_to_bigquery(records)
    """

    def __init__(
        self,
        config: ExportConfig,
        *,
        bq_client: bigquery.Client | None = None,
        gcs_client: storage.Client | None = None,
    ) -> None:
        """Create a DataExporter.

        Args:
            config: Export configuration (project, dataset, bucket, etc.).
            bq_client: Optional pre-built BigQuery client.  Useful for testing.
                       When provided, no new client is ever created.
            gcs_client: Optional pre-built GCS client.  Useful for testing.
                        When provided, no new client is ever created.
        """
        self._config = config
        self._bq_client_override = bq_client
        self._gcs_client_override = gcs_client
        self._bq_client: bigquery.Client | None = None
        self._gcs_client: storage.Client | None = None
        self._bq_lock = threading.Lock()
        self._gcs_lock = threading.Lock()

    # ── Public properties ────────────────────────────────────

    @property
    def config(self) -> ExportConfig:
        """Return the export configuration."""
        return self._config

    def _effective_project_id(self) -> str:
        """Return the configured GCP project ID with whitespace removed."""
        return self._config.gcp_project_id.strip()

    def _run_with_retry(self, operation_name: str, func: Any) -> Any:
        """Execute *func* with exponential backoff for transient GCP failures."""
        delays = (0.5, 1.0, 2.0)
        last_exc: Exception | None = None

        for attempt, delay in enumerate(delays, start=1):
            try:
                return func()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt == len(delays) or not _is_transient_gcp_error(exc):
                    raise
                logger.warning(
                    "%s failed with transient error (%s). Retrying in %.1fs [attempt %d/%d]",
                    operation_name,
                    exc.__class__.__name__,
                    delay,
                    attempt,
                    len(delays),
                )
                time.sleep(delay)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"{operation_name} retry loop exited unexpectedly")

    # ── Private client accessors (lazy) ─────────────────────

    def _get_bq_client(self) -> bigquery.Client:
        """Return the BigQuery client, creating it on first use.

        Uses the injected client when one was provided at construction time,
        otherwise creates a new one via ``google.cloud.bigquery.Client``.
        Thread-safe via double-checked locking on lazy initialization.

        Raises:
            ImportError: If ``google-cloud-bigquery`` is not installed.
        """
        if self._bq_client_override is not None:
            return self._bq_client_override
        if self._bq_client is None:
            with self._bq_lock:
                if self._bq_client is None:
                    _require_rag_deps()
                    from google.cloud import bigquery

                    self._bq_client = bigquery.Client(project=self._effective_project_id())
        return self._bq_client

    def _get_gcs_client(self) -> storage.Client:
        """Return the GCS client, creating it on first use.

        Uses the injected client when one was provided at construction time,
        otherwise creates a new one via ``google.cloud.storage.Client``.
        Thread-safe via double-checked locking on lazy initialization.

        Raises:
            ImportError: If ``google-cloud-storage`` is not installed.
        """
        if self._gcs_client_override is not None:
            return self._gcs_client_override
        if self._gcs_client is None:
            with self._gcs_lock:
                if self._gcs_client is None:
                    _require_rag_deps()
                    from google.cloud import storage

                    self._gcs_client = storage.Client(project=self._effective_project_id())
        return self._gcs_client

    # ── BigQuery schema management ───────────────────────────

    def _build_schema_fields(self, field_tuples: list[tuple[Any, ...]]) -> list[Any]:
        """Convert plain tuple definitions into ``bigquery.SchemaField`` objects.

        Supports nested STRUCT fields via a 5-tuple: ``(name, type, mode, desc, sub_fields)``.
        """
        from google.cloud import bigquery as bq

        fields: list[bq.SchemaField] = []
        for entry in field_tuples:
            name, field_type, mode, description = entry[0], entry[1], entry[2], entry[3]
            if len(entry) == 5:
                # RECORD (STRUCT) with nested sub-fields
                sub_fields = self._build_schema_fields(entry[4])
                fields.append(
                    bq.SchemaField(
                        name,
                        field_type,
                        mode=mode,
                        description=description,
                        fields=sub_fields,
                    )
                )
            else:
                fields.append(bq.SchemaField(name, field_type, mode=mode, description=description))
        return fields

    def ensure_dataset(self) -> bool:
        """Create the BigQuery dataset if it does not already exist.

        Returns:
            ``True`` if the dataset was created or already exists, ``False`` if
            the operation failed gracefully (error is logged as a warning).
        """
        try:
            from google.api_core.exceptions import Conflict
            from google.cloud import bigquery as bq

            client = self._get_bq_client()
            dataset_id = f"{self._effective_project_id()}.{self._config.bigquery_dataset}"
            dataset = bq.Dataset(dataset_id)
            dataset.location = "US"
            try:
                self._run_with_retry(
                    "ensure_dataset.create_dataset",
                    lambda: client.create_dataset(dataset, timeout=30),
                )
                logger.info("Created BigQuery dataset %s", dataset_id)
                return True
            except Conflict:
                logger.debug("BigQuery dataset %s already exists", dataset_id)
                return True
        except Exception:
            logger.warning(
                "ensure_dataset failed for dataset '%s' — skipping",
                self._config.bigquery_dataset,
                exc_info=True,
            )
            return False

    def ensure_tables(self) -> dict[str, bool]:
        """Create BigQuery tables with partitioning and clustering if absent.

        Creates all four tables defined in ``_TABLE_SCHEMAS``:
        ``telemetry_events``, ``tool_calls``, ``health_reports``, and ``feedback``.

        Returns:
            A mapping of ``table_name -> created`` where ``True`` means the table
            was newly created or already existed, and ``False`` means the creation
            attempt failed (error is logged as a warning).
        """
        try:
            from google.api_core.exceptions import Conflict
            from google.cloud import bigquery as bq

            client = self._get_bq_client()
        except Exception:
            logger.warning(
                "ensure_tables: failed to acquire BigQuery client — skipping all tables",
                exc_info=True,
            )
            return dict.fromkeys(_TABLE_SCHEMAS, False)

        results: dict[str, bool] = {}
        for table_name, field_tuples in _TABLE_SCHEMAS.items():
            table_id = (
                f"{self._effective_project_id()}"
                f".{self._config.bigquery_dataset}"
                f".{table_name}"
            )
            try:
                schema = self._build_schema_fields(field_tuples)
                table = bq.Table(table_id, schema=schema)

                # Time-partitioning
                cfg = _TABLE_CONFIG.get(table_name, {})
                partition_field: str | None = cfg.get("partition_field")
                if partition_field:
                    table.time_partitioning = bq.TimePartitioning(
                        type_=bq.TimePartitioningType.DAY,
                        field=partition_field,
                    )

                # Clustering
                clustering_fields: list[str] = cfg.get("clustering_fields", [])
                if clustering_fields:
                    table.clustering_fields = clustering_fields

                try:
                    self._run_with_retry(
                        f"ensure_tables.create_table[{table_name}]",
                        lambda table=table: client.create_table(table),
                    )
                    logger.info("Created BigQuery table %s", table_id)
                    results[table_name] = True
                except Conflict:
                    logger.debug("BigQuery table %s already exists", table_id)
                    results[table_name] = True
            except Exception:
                logger.warning(
                    "ensure_tables: failed to create table '%s' — skipping",
                    table_name,
                    exc_info=True,
                )
                results[table_name] = False

        return results

    # ── BigQuery insert helper ───────────────────────────────

    def _insert_rows(self, table_id: str, rows: list[dict[str, Any]]) -> int:
        """Insert rows to BigQuery.

        Uses streaming insert for small batches (≤100 rows) and a load job for
        larger batches (>100 rows) — ADR-5.

        Args:
            table_id: Unqualified table name (e.g. ``"telemetry_events"``).
            rows: List of BQ-compatible row dicts (already transformed).

        Returns:
            Number of rows successfully inserted.  On streaming insert partial
            failure, returns ``len(rows) - len(errors)``.  On load-job failure,
            raises so the caller can catch and log.
        """
        client = self._get_bq_client()
        full_table_id = (
            f"{self._effective_project_id()}"
            f".{self._config.bigquery_dataset}"
            f".{table_id}"
        )

        if len(rows) <= 100:
            # Streaming insert — best for pipeline hooks (1-50 rows)
            errors = self._run_with_retry(
                f"insert_rows_json[{table_id}]",
                lambda: client.insert_rows_json(full_table_id, rows),
            )
            if errors:
                logger.warning(
                    "BigQuery streaming insert errors for %s: %s", table_id, errors
                )
                return len(rows) - len(errors)
            return len(rows)
        else:
            # Load job — best for bulk CLI pushes (>100 rows)
            from google.cloud.bigquery import LoadJobConfig, SourceFormat

            job_config = LoadJobConfig(
                source_format=SourceFormat.NEWLINE_DELIMITED_JSON,
                write_disposition="WRITE_APPEND",
            )
            jsonl = "\n".join(json.dumps(row, default=str) for row in rows)
            job = self._run_with_retry(
                f"load_table_from_file[{table_id}]",
                lambda: client.load_table_from_file(
                    io.BytesIO(jsonl.encode("utf-8")),
                    full_table_id,
                    job_config=job_config,
                ),
            )
            self._run_with_retry(
                f"load_job.result[{table_id}]",
                job.result,
            )
            return len(rows)

    def _load_telemetry_records(self, since: datetime | None = None) -> list[dict[str, Any]]:
        """Load telemetry records from the local collector store."""
        settings = Settings()
        collector = TelemetryCollector(
            db_path=Path(settings.session.db_path).expanduser().parent / "telemetry.db",
            enabled=settings.telemetry.enabled,
            buffer_size=settings.telemetry.buffer_size,
        )
        try:
            since_iso = since.isoformat() if since is not None else None
            return collector.query_events(None, since=since_iso, limit=50_000)
        finally:
            collector.close()

    def _load_tool_call_records(
        self,
        since: datetime | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Load tool call records from the JSONL store."""
        settings = Settings()
        base_dir = Path(settings.logging.tool_results_dir).expanduser()
        store = ToolCallStore(base_dir=base_dir)
        return store.read_records(run_id=run_id, since=since if run_id is None else None)

    async def export_telemetry(
        self,
        since: datetime | None = None,
        dest: str = "both",
    ) -> ExportResult:
        """Export telemetry events since *since* to BigQuery and/or GCS."""
        records = self._load_telemetry_records(since=since)
        if not records:
            return ExportResult(destination=dest, success=False)

        bq_count = 0
        gcs_uri: str | None = None
        if dest in ("bigquery", "both"):
            bq_count = self.export_telemetry_to_bigquery(records)
        if dest in ("gcs", "both"):
            gcs_uri = self.export_telemetry_to_gcs(records)
        return ExportResult(
            bigquery_rows=bq_count,
            gcs_uri=gcs_uri,
            success=bool(bq_count or gcs_uri),
            destination=dest,
        )

    async def export_tool_calls(
        self,
        since: datetime | None = None,
        run_id: str | None = None,
        dest: str = "both",
    ) -> ExportResult:
        """Export tool call records to BigQuery and/or GCS."""
        records = self._load_tool_call_records(since=since, run_id=run_id)
        if not records:
            return ExportResult(destination=dest, success=False)

        effective_run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        bq_count = 0
        gcs_uri: str | None = None
        if dest in ("bigquery", "both"):
            bq_count = self.export_tool_calls_to_bigquery(records)
        if dest in ("gcs", "both"):
            gcs_uri = self.export_tool_results_to_gcs(records, effective_run_id)
        return ExportResult(
            bigquery_rows=bq_count,
            gcs_uri=gcs_uri,
            success=bool(bq_count or gcs_uri),
            destination=dest,
        )

    async def export_health_report(
        self,
        report: HealthReport | dict[str, Any],
        dest: str = "both",
    ) -> ExportResult:
        """Export a single health report to BigQuery and/or GCS."""
        report_dict = _coerce_report_dict(report)
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        bq_count = 0
        gcs_uri: str | None = None
        if dest in ("bigquery", "both"):
            bq_count = int(self.export_report_to_bigquery(report_dict, run_id=run_id))
        if dest in ("gcs", "both"):
            gcs_uri = self.export_report_to_gcs(report_dict, run_id)
        return ExportResult(
            bigquery_rows=bq_count,
            gcs_uri=gcs_uri,
            success=bool(bq_count or gcs_uri),
            destination=dest,
        )

    # ── BigQuery export methods ──────────────────────────────

    def export_telemetry_to_bigquery(self, records: list[dict[str, Any]]) -> int:
        """Stream telemetry rows to BigQuery.

        Transforms raw records using :func:`transform_telemetry_record` and
        inserts them into the ``telemetry_events`` table.  Uses streaming insert
        for ≤100 rows, load job for larger batches (ADR-5).

        Args:
            records: List of telemetry event dicts (as produced by TelemetryStore).

        Returns:
            Number of rows successfully inserted. Returns ``0`` on any failure.
        """
        if not records:
            return 0
        if not self._effective_project_id():
            logger.warning(
                "export_telemetry_to_bigquery: bigquery_project is not configured — skipping"
            )
            return 0
        try:
            rows = [transform_telemetry_record(r) for r in records]
            return self._insert_rows("telemetry_events", rows)
        except _GCP_EXPORT_ERRORS:
            logger.warning(
                "export_telemetry_to_bigquery: insert failed — skipping", exc_info=True
            )
            return 0

    def export_tool_calls_to_bigquery(self, records: list[dict[str, Any]]) -> int:
        """Stream tool call rows to BigQuery.

        Transforms raw records using :func:`transform_tool_call_record` and
        inserts them into the ``tool_calls`` table.  Uses streaming insert for
        ≤100 rows, load job for larger batches (ADR-5).

        Args:
            records: List of tool call dicts (as produced by ToolCallStore).

        Returns:
            Number of rows successfully inserted. Returns ``0`` on any failure.
        """
        if not records:
            return 0
        if not self._effective_project_id():
            logger.warning(
                "export_tool_calls_to_bigquery: bigquery_project is not configured — skipping"
            )
            return 0
        try:
            rows = [transform_tool_call_record(r) for r in records]
            return self._insert_rows("tool_calls", rows)
        except _GCP_EXPORT_ERRORS:
            logger.warning(
                "export_tool_calls_to_bigquery: insert failed — skipping", exc_info=True
            )
            return 0

    def export_report_to_bigquery(
        self,
        report: dict[str, Any],
        run_id: str = "",
        cluster_name: str = "",
        namespace: str = "",
    ) -> bool:
        """Insert a single health report row into BigQuery.

        Transforms the report using :func:`transform_health_report` and inserts
        it into the ``health_reports`` table via streaming insert (reports are
        always a single row).

        Args:
            report: Health report dict (as produced by HealthReport.model_dump()).
            run_id: Unique pipeline run identifier.
            cluster_name: Kubernetes cluster name (for partitioning / filtering).
            namespace: Kubernetes namespace (for partitioning / filtering).

        Returns:
            ``True`` if the row was inserted successfully, ``False`` on failure.
        """
        if not self._effective_project_id():
            logger.warning(
                "export_report_to_bigquery: bigquery_project is not configured — skipping"
            )
            return False
        try:
            row = transform_health_report(
                report,
                run_id=run_id,
                cluster_name=cluster_name,
                namespace=namespace,
            )
            inserted = self._insert_rows("health_reports", [row])
            return inserted > 0
        except _GCP_EXPORT_ERRORS:
            logger.warning(
                "export_report_to_bigquery: insert failed — skipping", exc_info=True
            )
            return False

    def export_feedback_to_bigquery(
        self,
        rating: int,
        comment: str = "",
        *,
        run_id: str,
        report_summary: str = "",
        auto_quality_score: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Insert a single feedback row into BigQuery.

        Transforms the feedback using :func:`transform_feedback_record` and
        inserts it into the ``feedback`` table via streaming insert.

        Args:
            rating: User rating (1–5).
            comment: Free-text feedback comment.
            run_id: Pipeline run identifier linking feedback to a specific run.
                Required — callers must always provide a non-empty run_id.
            report_summary: Optional summary of the associated report.
            auto_quality_score: Optional automated quality score.
            metadata: Optional arbitrary metadata dict.

        Returns:
            ``True`` if the row was inserted successfully, ``False`` on failure.
        """
        if not run_id:
            logger.warning(
                "export_feedback_to_bigquery: run_id is empty — skipping"
            )
            return False
        if not self._effective_project_id():
            logger.warning(
                "export_feedback_to_bigquery: gcp_project_id is not configured — skipping"
            )
            return False
        try:
            feedback_data: dict[str, Any] = {
                "rating": rating,
                "comment": comment,
                "report_summary": report_summary,
                "auto_quality_score": auto_quality_score,
                "metadata": metadata or {},
            }
            row = transform_feedback_record(feedback_data, run_id=run_id)
            inserted = self._insert_rows("feedback", [row])
            return inserted > 0
        except _GCP_EXPORT_ERRORS:
            logger.warning(
                "export_feedback_to_bigquery: insert failed — skipping", exc_info=True
            )
            return False

    # ── GCS export methods ───────────────────────────────────

    def export_report_to_gcs(self, report: dict[str, Any], run_id: str) -> str | None:
        """Upload a health report as a single JSON document to GCS (one JSON file per report).

        The blob is written to ``{effective_gcs_prefix}reports/{YYYY-MM-DD}/{run_id}.json``.
        Note: this uploads a single JSON document, not JSONL.

        Args:
            report: Health report dict to serialise as a JSON document.
            run_id: Unique identifier for this pipeline run (used as filename).

        Returns:
            The ``gs://`` URI of the uploaded object, or ``None`` on failure.
        """
        if not self._config.gcs_bucket:
            logger.warning("export_report_to_gcs: gcs_bucket is not configured — skipping")
            return None
        try:
            date_str = datetime.now(tz=UTC).strftime("%Y-%m-%d")
            blob_path = f"{self._config.effective_gcs_prefix}reports/{date_str}/{run_id}.json"
            content = json.dumps(report, default=str)
            client = self._get_gcs_client()
            bucket = client.bucket(self._config.gcs_bucket)
            blob = bucket.blob(blob_path)
            blob.upload_from_string(content, content_type="application/json")
            uri = f"gs://{self._config.gcs_bucket}/{blob_path}"
            logger.info("Uploaded report to %s", uri)
            return uri
        except _GCP_EXPORT_ERRORS:
            logger.warning("export_report_to_gcs: upload failed — skipping", exc_info=True)
            return None

    def export_tool_results_to_gcs(self, records: list[dict[str, Any]], run_id: str) -> str | None:
        """Upload tool call records as JSONL to GCS.

        The blob is written to ``{effective_gcs_prefix}tool_results/{YYYY-MM-DD}/{run_id}.jsonl``.

        Args:
            records: List of tool call dicts to serialise as JSONL.
            run_id: Unique identifier for this pipeline run (used as filename).

        Returns:
            The ``gs://`` URI of the uploaded object, or ``None`` on failure.
        """
        if not self._config.gcs_bucket:
            logger.warning("export_tool_results_to_gcs: gcs_bucket is not configured — skipping")
            return None
        if not records:
            logger.warning("export_tool_results_to_gcs: no records to upload — skipping")
            return None
        try:
            date_str = datetime.now(tz=UTC).strftime("%Y-%m-%d")
            blob_path = f"{self._config.effective_gcs_prefix}tool_results/{date_str}/{run_id}.jsonl"
            content = "\n".join(json.dumps(r, default=str) for r in records)
            client = self._get_gcs_client()
            bucket = client.bucket(self._config.gcs_bucket)
            blob = bucket.blob(blob_path)
            blob.upload_from_string(content, content_type="application/jsonl")
            uri = f"gs://{self._config.gcs_bucket}/{blob_path}"
            logger.info("Uploaded %d tool results to %s", len(records), uri)
            return uri
        except _GCP_EXPORT_ERRORS:
            logger.warning("export_tool_results_to_gcs: upload failed — skipping", exc_info=True)
            return None

    def export_telemetry_to_gcs(self, records: list[dict[str, Any]]) -> str | None:
        """Upload a telemetry batch as JSONL to GCS.

        The blob is written to ``{effective_gcs_prefix}telemetry/{YYYY-MM-DD}/batch_{timestamp_iso}.jsonl``.

        Args:
            records: List of telemetry event dicts to serialise as JSONL.

        Returns:
            The ``gs://`` URI of the uploaded object, or ``None`` on failure.
        """
        if not self._config.gcs_bucket:
            logger.warning("export_telemetry_to_gcs: gcs_bucket is not configured — skipping")
            return None
        if not records:
            logger.warning("export_telemetry_to_gcs: no records to upload — skipping")
            return None
        try:
            now = datetime.now(tz=UTC)
            date_str = now.strftime("%Y-%m-%d")
            timestamp_iso = now.strftime("%Y%m%dT%H%M%SZ")
            blob_path = f"{self._config.effective_gcs_prefix}telemetry/{date_str}/batch_{timestamp_iso}.jsonl"
            content = "\n".join(json.dumps(r, default=str) for r in records)
            client = self._get_gcs_client()
            bucket = client.bucket(self._config.gcs_bucket)
            blob = bucket.blob(blob_path)
            blob.upload_from_string(content, content_type="application/jsonl")
            uri = f"gs://{self._config.gcs_bucket}/{blob_path}"
            logger.info("Uploaded %d telemetry records to %s", len(records), uri)
            return uri
        except _GCP_EXPORT_ERRORS:
            logger.warning("export_telemetry_to_gcs: upload failed — skipping", exc_info=True)
            return None


# ── Run ID persistence ──────────────────────────────────────


_LAST_RUN_ID_PATH = Path("~/.vaig/last_run_id")


def save_last_run_id(run_id: str) -> None:
    """Persist the most recent run_id to ``~/.vaig/last_run_id``.

    Creates the ``~/.vaig/`` directory if it doesn't exist.  Silently
    swallows any I/O errors so callers don't need error handling.
    """
    try:
        path = _LAST_RUN_ID_PATH.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(run_id.strip(), encoding="utf-8")
    except OSError:
        logger.debug("Could not save last run_id to %s", _LAST_RUN_ID_PATH, exc_info=True)


def get_last_run_id() -> str | None:
    """Read the most recent run_id from ``~/.vaig/last_run_id``.

    Returns:
        The stored run_id string, or ``None`` if the file doesn't exist or
        is empty.
    """
    try:
        path = _LAST_RUN_ID_PATH.expanduser()
        if not path.is_file():
            return None
        content = path.read_text(encoding="utf-8").strip()
        return content or None
    except OSError:
        logger.debug("Could not read last run_id from %s", _LAST_RUN_ID_PATH, exc_info=True)
        return None


# ── Fire-and-forget auto-export hook ────────────────────────


def auto_export_report(
    config: ExportConfig,
    report: HealthReport | dict[str, Any],
    run_id: str,
    cluster_name: str = "",
    namespace: str = "",
) -> None:
    """Fire-and-forget export of a health report. Runs in background, never raises.

    Starts a daemon thread that exports *report* to BigQuery and/or GCS based
    on the provided *config*.  The thread is a daemon so it will never block
    CLI exit — the export is best-effort.

    Args:
        config: Export configuration (controls which destinations are enabled).
        report: Health report dict (e.g. ``HealthReport.to_dict()``).
        run_id: Unique pipeline run identifier (used as BQ run_id and GCS filename).
        cluster_name: Kubernetes cluster name (for BQ partitioning).
        namespace: Kubernetes namespace (for BQ partitioning).
    """
    def _export() -> None:
        try:
            exporter = DataExporter(config)
            report_dict = _coerce_report_dict(report)
            # Export to BigQuery if project configured
            if config.gcp_project_id:
                exporter.export_report_to_bigquery(
                    report_dict, run_id=run_id, cluster_name=cluster_name, namespace=namespace
                )
            # Export to GCS if bucket configured
            if config.gcs_bucket:
                exporter.export_report_to_gcs(report_dict, run_id=run_id)
            logger.info("Auto-export completed for run %s", run_id)
        except Exception:  # noqa: BLE001
            logger.warning("Auto-export failed for run %s", run_id, exc_info=True)

    thread = threading.Thread(target=_export, daemon=True, name="vaig-auto-export")
    thread.start()
    logger.debug("Auto-export started in background for run %s", run_id)

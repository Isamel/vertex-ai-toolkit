"""Audit subscriber — dual-sink (BigQuery + Cloud Logging) event adapter.

Subscribes to 8 domain events on the :class:`~vaig.core.event_bus.EventBus`,
enriches each into an audit record with identity and app metadata, buffers
records, and flushes to BigQuery (streaming insert) and Cloud Logging
(structured JSON) when the buffer fills or a session ends.

Mirrors the :class:`~vaig.core.subscribers.TelemetrySubscriber` pattern.
Each handler is wrapped in ``try/except`` so that audit failures never
propagate to the emitting code.

All ``google-cloud-*`` imports are **lazy** — the ``[audit]`` extras are
only required when ``audit.enabled: true``.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vaig.core.event_bus import EventBus
from vaig.core.events import (
    ApiCalled,
    CliCommandTracked,
    ErrorOccurred,
    RemediationExecuted,
    SessionEnded,
    SessionStarted,
    SkillUsed,
    ToolExecuted,
)
from vaig.core.identity import get_app_version, resolve_identity

if TYPE_CHECKING:
    from google.auth.credentials import Credentials
    from google.cloud.bigquery import Client as BQClient
    from google.cloud.logging import Client as LoggingClient

    from vaig.core.config import Settings

logger = logging.getLogger(__name__)

__all__ = [
    "AuditSubscriber",
]

# Maximum buffer size to prevent unbounded memory usage
_MAX_BUFFER_CAP = 1000


# ── Dependency guard ─────────────────────────────────────────


def _require_audit_deps() -> None:
    """Raise ImportError with install instructions if [audit] deps are missing."""
    try:
        import google.cloud.bigquery  # noqa: F401
        import google.cloud.logging  # noqa: F401
    except ImportError:
        raise ImportError(
            "Audit features require the [audit] extras. "
            "Install with: pip install 'vertex-ai-toolkit[audit]'"
        ) from None


# ── BigQuery schema definition ───────────────────────────────

_BQ_SCHEMA = [
    {"name": "timestamp", "type": "TIMESTAMP", "mode": "REQUIRED"},
    {"name": "event_type", "type": "STRING", "mode": "REQUIRED"},
    {"name": "os_user", "type": "STRING", "mode": "REQUIRED"},
    {"name": "gcp_user", "type": "STRING", "mode": "NULLABLE"},
    {"name": "command", "type": "STRING", "mode": "NULLABLE"},
    {"name": "skill", "type": "STRING", "mode": "NULLABLE"},
    {"name": "model", "type": "STRING", "mode": "NULLABLE"},
    {"name": "tokens_in", "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "tokens_out", "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "tokens_thinking", "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "duration_ms", "type": "FLOAT", "mode": "NULLABLE"},
    {"name": "result", "type": "STRING", "mode": "REQUIRED"},
    {"name": "error_message", "type": "STRING", "mode": "NULLABLE"},
    {"name": "app_version", "type": "STRING", "mode": "REQUIRED"},
    {"name": "session_id", "type": "STRING", "mode": "NULLABLE"},
]


# ── AuditSubscriber ──────────────────────────────────────────


class AuditSubscriber:
    """Dual-sink audit subscriber for BigQuery + Cloud Logging.

    Subscribes to 8 EventBus event types, enriches each into an audit
    record dict, and flushes batched records to both sinks independently.

    Args:
        settings: Application settings (provides ``audit`` config + GCP project).
        credentials: GCP credentials for BigQuery / Cloud Logging clients.
        bq_client: Optional pre-built BigQuery client (for testing).
        logging_client: Optional pre-built Cloud Logging client (for testing).
    """

    def __init__(
        self,
        settings: Settings,
        credentials: Credentials | None = None,
        *,
        bq_client: BQClient | None = None,
        logging_client: LoggingClient | None = None,
    ) -> None:
        _require_audit_deps()

        self._settings = settings
        self._credentials = credentials
        self._bq_client = bq_client
        self._logging_client = logging_client

        # Resolve identity once at init
        os_user, gcp_user, _ = resolve_identity(credentials)
        self._os_user = os_user
        self._gcp_user = gcp_user
        self._app_version = get_app_version()

        # Buffer (guarded by lock for thread-safety)
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()

        # Track current session
        self._current_session_id: str = ""

        # Table creation tracking
        self._table_ensured = False

        # Periodic flush timer
        self._flush_timer: threading.Timer | None = None

        # Subscribe to events
        self._unsubscribers: list[Callable[[], None]] = []
        self._subscribe_all()

    # ── Subscription wiring ──────────────────────────────────

    def _subscribe_all(self) -> None:
        """Register handlers for 8 audit event types."""
        bus = EventBus.get()
        self._unsubscribers = [
            bus.subscribe(ApiCalled, self._on_api_called),
            bus.subscribe(CliCommandTracked, self._on_cli_command_tracked),
            bus.subscribe(ToolExecuted, self._on_tool_executed),
            bus.subscribe(SkillUsed, self._on_skill_used),
            bus.subscribe(SessionStarted, self._on_session_started),
            bus.subscribe(SessionEnded, self._on_session_ended),
            bus.subscribe(ErrorOccurred, self._on_error_occurred),
            bus.subscribe(RemediationExecuted, self._on_remediation_executed),
        ]

    def unsubscribe_all(self) -> None:
        """Detach all handlers and flush remaining buffer."""
        self._cancel_flush_timer()
        self._flush()
        for unsub in self._unsubscribers:
            unsub()
        self._unsubscribers.clear()

    # ── Periodic flush timer ─────────────────────────────────

    def _start_flush_timer(self) -> None:
        """Start (or restart) the periodic flush timer if not already running."""
        if self._flush_timer is not None and self._flush_timer.is_alive():
            return
        interval = self._settings.audit.flush_interval_seconds
        if interval <= 0:
            return
        self._flush_timer = threading.Timer(interval, self._periodic_flush)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _cancel_flush_timer(self) -> None:
        """Cancel the periodic flush timer if running."""
        if self._flush_timer is not None:
            self._flush_timer.cancel()
            self._flush_timer = None

    def _periodic_flush(self) -> None:
        """Called by the timer — flush and reschedule if buffer is non-empty."""
        self._flush()
        # Reschedule only if there are still records buffered
        with self._lock:
            has_records = bool(self._buffer)
        if has_records:
            self._start_flush_timer()

    # ── Audit record construction ────────────────────────────

    def _make_record(
        self,
        event_type: str,
        *,
        timestamp: str,
        result: str = "success",
        command: str = "",
        skill: str = "",
        model: str = "",
        tokens_in: int = 0,
        tokens_out: int = 0,
        tokens_thinking: int = 0,
        duration_ms: float = 0.0,
        error_message: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        """Build an enriched audit record dict."""
        return {
            "timestamp": timestamp,
            "event_type": event_type,
            "os_user": self._os_user,
            "gcp_user": self._gcp_user,
            "command": command or None,
            "skill": skill or None,
            "model": model or None,
            "tokens_in": tokens_in or None,
            "tokens_out": tokens_out or None,
            "tokens_thinking": tokens_thinking or None,
            "duration_ms": duration_ms or None,
            "result": result,
            "error_message": error_message or None,
            "app_version": self._app_version,
            "session_id": session_id or self._current_session_id or None,
        }

    def _append_record(self, record: dict[str, Any]) -> None:
        """Append a record to the buffer and flush if threshold reached."""
        with self._lock:
            if len(self._buffer) >= _MAX_BUFFER_CAP:
                logger.warning(
                    "AuditSubscriber: buffer at capacity (%d) — triggering emergency flush",
                    _MAX_BUFFER_CAP,
                )
                at_cap = True
            else:
                self._buffer.append(record)
                at_cap = False
            should_flush = at_cap or len(self._buffer) >= self._settings.audit.buffer_size

        if should_flush:
            self._flush()

        # Start periodic flush timer on first record
        self._start_flush_timer()

    # ── Event handlers ───────────────────────────────────────

    def _on_api_called(self, event: ApiCalled) -> None:
        """ApiCalled → audit record with model, tokens, duration."""
        try:
            metadata = dict(event.metadata) if event.metadata else {}
            thinking = int(metadata.get("thinking_tokens", 0))
            record = self._make_record(
                event.event_type,
                timestamp=event.timestamp,
                model=event.model,
                tokens_in=event.tokens_in,
                tokens_out=event.tokens_out,
                tokens_thinking=thinking,
                duration_ms=event.duration_ms,
                result="success",
            )
            self._append_record(record)
        except Exception:  # noqa: BLE001
            logger.debug("AuditSubscriber: failed to handle ApiCalled", exc_info=True)

    def _on_cli_command_tracked(self, event: CliCommandTracked) -> None:
        """CliCommandTracked → audit record with command name."""
        try:
            record = self._make_record(
                event.event_type,
                timestamp=event.timestamp,
                command=event.command_name,
                duration_ms=event.duration_ms,
                result="success",
            )
            self._append_record(record)
        except Exception:  # noqa: BLE001
            logger.debug("AuditSubscriber: failed to handle CliCommandTracked", exc_info=True)

    def _on_tool_executed(self, event: ToolExecuted) -> None:
        """ToolExecuted → audit record with tool name and error status."""
        try:
            record = self._make_record(
                event.event_type,
                timestamp=event.timestamp,
                command=event.tool_name,
                duration_ms=event.duration_ms,
                result="fail" if event.error else "success",
                error_message=event.error_message if event.error else "",
            )
            self._append_record(record)
        except Exception:  # noqa: BLE001
            logger.debug("AuditSubscriber: failed to handle ToolExecuted", exc_info=True)

    def _on_skill_used(self, event: SkillUsed) -> None:
        """SkillUsed → audit record with skill name."""
        try:
            record = self._make_record(
                event.event_type,
                timestamp=event.timestamp,
                skill=event.skill_name,
                duration_ms=event.duration_ms,
                result="success",
            )
            self._append_record(record)
        except Exception:  # noqa: BLE001
            logger.debug("AuditSubscriber: failed to handle SkillUsed", exc_info=True)

    def _on_session_started(self, event: SessionStarted) -> None:
        """SessionStarted → audit record + track session_id."""
        try:
            self._current_session_id = event.session_id
            record = self._make_record(
                event.event_type,
                timestamp=event.timestamp,
                session_id=event.session_id,
                model=event.model,
                skill=event.skill,
                result="success",
            )
            self._append_record(record)
        except Exception:  # noqa: BLE001
            logger.debug("AuditSubscriber: failed to handle SessionStarted", exc_info=True)

    def _on_session_ended(self, event: SessionEnded) -> None:
        """SessionEnded → audit record + immediate flush."""
        try:
            record = self._make_record(
                event.event_type,
                timestamp=event.timestamp,
                session_id=event.session_id,
                duration_ms=event.duration_ms,
                result="success",
            )
            with self._lock:
                if len(self._buffer) < _MAX_BUFFER_CAP:
                    self._buffer.append(record)
            # Immediate flush on session end
            self._flush()
        except Exception:  # noqa: BLE001
            logger.debug("AuditSubscriber: failed to handle SessionEnded", exc_info=True)

    def _on_error_occurred(self, event: ErrorOccurred) -> None:
        """ErrorOccurred → audit record with error details."""
        try:
            record = self._make_record(
                event.event_type,
                timestamp=event.timestamp,
                command=event.source,
                result="fail",
                error_message=event.error_message,
            )
            self._append_record(record)
        except Exception:  # noqa: BLE001
            logger.debug("AuditSubscriber: failed to handle ErrorOccurred", exc_info=True)

    def _on_remediation_executed(self, event: RemediationExecuted) -> None:
        """RemediationExecuted → audit record with command, tier, and outcome."""
        try:
            result = "blocked" if event.tier == "blocked" else ("fail" if event.error else "success")
            record = self._make_record(
                event.event_type,
                timestamp=event.timestamp,
                command=event.command,
                result=result,
                error_message=event.error or "",
            )
            self._append_record(record)
        except Exception:  # noqa: BLE001
            logger.debug("AuditSubscriber: failed to handle RemediationExecuted", exc_info=True)

    # ── Flush logic ──────────────────────────────────────────

    def _flush(self) -> None:
        """Flush buffered records to both sinks independently."""
        with self._lock:
            if not self._buffer:
                return
            records = list(self._buffer)
            self._buffer.clear()

        self._write_to_bigquery(records)
        self._write_to_cloud_logging(records)

    def _write_to_bigquery(self, records: list[dict[str, Any]]) -> None:
        """Write audit records to BigQuery via streaming insert."""
        try:
            import google.cloud.bigquery as bq  # lazy import

            client = self._bq_client
            if client is None:
                client = bq.Client(credentials=self._credentials)
                self._bq_client = client

            dataset = self._settings.audit.bigquery_dataset
            table_name = self._settings.audit.bigquery_table

            # Get project ID from settings (GCP config)
            project_id = getattr(self._settings.gcp, "project_id", None) or client.project
            table_ref = f"{project_id}.{dataset}.{table_name}"

            # Auto-create table if needed
            if not self._table_ensured:
                try:
                    self._ensure_bq_table(client, project_id, dataset, table_name)
                    self._table_ensured = True
                except Exception:  # noqa: BLE001
                    logger.warning("AuditSubscriber: BQ table creation failed — will retry next flush", exc_info=True)

            errors = client.insert_rows_json(table_ref, records)
            if errors:
                logger.warning("BigQuery insert errors: %s", errors)
        except Exception:  # noqa: BLE001
            logger.warning("AuditSubscriber: BigQuery write failed", exc_info=True)

    def _ensure_bq_table(
        self,
        client: BQClient,
        project_id: str,
        dataset_name: str,
        table_name: str,
    ) -> None:
        """Create BigQuery dataset and table if they don't exist."""
        try:
            import google.cloud.bigquery as bq  # lazy import

            dataset_ref = f"{project_id}.{dataset_name}"
            dataset = bq.Dataset(dataset_ref)

            try:
                client.get_dataset(dataset_ref)
            except Exception:
                client.create_dataset(dataset, exists_ok=True)

            table_ref = f"{dataset_ref}.{table_name}"
            schema = [
                bq.SchemaField(col["name"], col["type"], mode=col["mode"])
                for col in _BQ_SCHEMA
            ]
            table = bq.Table(table_ref, schema=schema)
            table.time_partitioning = bq.TimePartitioning(
                type_=bq.TimePartitioningType.DAY,
                field="timestamp",
            )
            table.clustering_fields = ["gcp_user", "event_type"]
            client.create_table(table, exists_ok=True)
        except Exception:  # noqa: BLE001
            logger.warning("AuditSubscriber: Failed to ensure BQ table", exc_info=True)

    def _write_to_cloud_logging(self, records: list[dict[str, Any]]) -> None:
        """Write audit records to Cloud Logging as structured JSON."""
        try:
            import google.cloud.logging as cloud_logging  # lazy import

            client = self._logging_client
            if client is None:
                client = cloud_logging.Client(credentials=self._credentials)
                self._logging_client = client

            log_name = self._settings.audit.cloud_logging_log_name
            gcp_logger = client.logger(log_name)

            for record in records:
                severity = "ERROR" if record.get("event_type") == "error.occurred" else "INFO"
                gcp_logger.log_struct(record, severity=severity)
        except Exception:  # noqa: BLE001
            logger.warning("AuditSubscriber: Cloud Logging write failed", exc_info=True)

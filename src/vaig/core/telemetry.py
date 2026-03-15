"""Local usage telemetry — fire-and-forget event collection with SQLite persistence."""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS telemetry_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT NOT NULL,
    event_name  TEXT NOT NULL,
    session_id  TEXT DEFAULT '',
    timestamp   TEXT NOT NULL,
    duration_ms REAL DEFAULT 0.0,
    metadata    TEXT DEFAULT '',
    tokens_in   INTEGER DEFAULT 0,
    tokens_out  INTEGER DEFAULT 0,
    cost_usd    REAL DEFAULT 0.0,
    model       TEXT DEFAULT '',
    tool_name   TEXT DEFAULT '',
    error_type  TEXT DEFAULT '',
    error_msg   TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_te_type      ON telemetry_events(event_type);
CREATE INDEX IF NOT EXISTS idx_te_timestamp ON telemetry_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_te_session   ON telemetry_events(session_id);
"""


# ── TelemetryEvent dataclass ─────────────────────────────────


@dataclass(slots=True)
class TelemetryEvent:
    """A single telemetry event record.

    All fields have sensible defaults so callers only need to supply
    ``event_type`` and ``event_name``.
    """

    event_type: str  # tool_call | api_call | cli_command | skill_use | error | session | orchestrator
    event_name: str  # e.g. "get_pods", "generate", "ask", "gke_diagnostics", "ValueError"
    session_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_ms: float = 0.0
    metadata_json: str = ""  # JSON string for extensible data
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    model: str = ""
    tool_name: str = ""  # duplicate of event_name for tool_call, kept for query convenience
    error_type: str = ""
    error_message: str = ""

    def to_row(self) -> tuple[str, str, str, str, float, str, int, int, float, str, str, str, str]:
        """Return a tuple matching the INSERT column order."""
        return (
            self.event_type,
            self.event_name,
            self.session_id,
            self.timestamp,
            self.duration_ms,
            self.metadata_json,
            self.tokens_in,
            self.tokens_out,
            self.cost_usd,
            self.model,
            self.tool_name,
            self.error_type,
            self.error_message,
        )


# ── TelemetryCollector ───────────────────────────────────────

_INSERT_SQL = (
    "INSERT INTO telemetry_events "
    "(event_type, event_name, session_id, timestamp, duration_ms, metadata, "
    "tokens_in, tokens_out, cost_usd, model, tool_name, error_type, error_msg) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


class TelemetryCollector:
    """Process-wide telemetry collector with buffered SQLite persistence.

    Thread-safe, fire-and-forget — emit methods never raise.
    Follows the same patterns as :class:`~vaig.session.store.SessionStore`
    (lazy SQLite connection, WAL mode) and :class:`~vaig.core.cost_tracker.CostTracker`
    (``threading.Lock``, dataclass records).
    """

    def __init__(
        self,
        db_path: str | Path = "~/.vaig/telemetry.db",
        *,
        enabled: bool = True,
        buffer_size: int = 50,
    ) -> None:
        self._db_path = Path(db_path).expanduser()
        self._enabled = enabled
        self._buffer_size = buffer_size

        self._lock = threading.Lock()
        self._buffer: list[TelemetryEvent] = []
        self._conn: sqlite3.Connection | None = None
        self._aconn: aiosqlite.Connection | None = None
        self._async_lock = asyncio.Lock()
        self._session_id: str = ""
        self._closed = False

        # Register atexit handler to flush remaining events on shutdown
        atexit.register(self._flush_at_exit)

    # ── Session ID ────────────────────────────────────────────

    def set_session_id(self, session_id: str) -> None:
        """Set the current session ID. Future events inherit it automatically."""
        self._session_id = session_id

    # ── Typed emit methods ────────────────────────────────────

    def emit_tool_call(
        self,
        tool_name: str,
        *,
        duration_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
        error_type: str = "",
        error_message: str = "",
    ) -> None:
        """Record a tool invocation."""
        self._emit_safe(
            event_type="tool_call",
            event_name=tool_name,
            duration_ms=duration_ms,
            metadata=metadata,
            tool_name=tool_name,
            error_type=error_type,
            error_message=error_message,
        )

    def emit_api_call(
        self,
        model: str,
        *,
        duration_ms: float = 0.0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
        metadata: dict[str, Any] | None = None,
        error_type: str = "",
        error_message: str = "",
    ) -> None:
        """Record a Gemini API call."""
        self._emit_safe(
            event_type="api_call",
            event_name=model,
            duration_ms=duration_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            model=model,
            metadata=metadata,
            error_type=error_type,
            error_message=error_message,
        )

    def emit_cli_command(
        self,
        command_name: str,
        *,
        duration_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a CLI command invocation."""
        self._emit_safe(
            event_type="cli_command",
            event_name=command_name,
            duration_ms=duration_ms,
            metadata=metadata,
        )

    def emit_skill_use(
        self,
        skill_name: str,
        *,
        duration_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a skill activation."""
        self._emit_safe(
            event_type="skill_use",
            event_name=skill_name,
            duration_ms=duration_ms,
            metadata=metadata,
        )

    def emit_error(
        self,
        error_type: str,
        error_message: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an error event."""
        self._emit_safe(
            event_type="error",
            event_name=error_type,
            error_type=error_type,
            error_message=error_message,
            metadata=metadata,
        )

    def emit(
        self,
        event_type: str,
        event_name: str,
        *,
        duration_ms: float = 0.0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
        model: str = "",
        tool_name: str = "",
        error_type: str = "",
        error_message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Generic emit — use typed methods when possible."""
        self._emit_safe(
            event_type=event_type,
            event_name=event_name,
            duration_ms=duration_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            model=model,
            tool_name=tool_name,
            error_type=error_type,
            error_message=error_message,
            metadata=metadata,
        )

    # ── Internal emit logic ───────────────────────────────────

    def _emit_safe(
        self,
        *,
        event_type: str,
        event_name: str,
        duration_ms: float = 0.0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
        model: str = "",
        tool_name: str = "",
        error_type: str = "",
        error_message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Fire-and-forget wrapper: catches all exceptions."""
        if not self._enabled:
            return
        try:
            event = TelemetryEvent(
                event_type=event_type,
                event_name=event_name,
                session_id=self._session_id,
                duration_ms=duration_ms,
                metadata_json=json.dumps(metadata) if metadata else "",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
                model=model,
                tool_name=tool_name,
                error_type=error_type,
                error_message=error_message,
            )
            self._append(event)
        except Exception:  # noqa: BLE001
            logger.debug("Telemetry emit failed", exc_info=True)

    def _append(self, event: TelemetryEvent) -> None:
        """Append event to buffer, flushing when threshold is reached."""
        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) >= self._buffer_size:
                self._flush_locked()

    # ── Flush ─────────────────────────────────────────────────

    def flush(self) -> None:
        """Flush buffered events to SQLite. Safe to call anytime."""
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        """Write buffered events to DB. MUST be called while holding ``_lock``."""
        if not self._buffer or self._closed:
            return
        try:
            conn = self._get_conn()
            conn.executemany(_INSERT_SQL, [e.to_row() for e in self._buffer])
            conn.commit()
            self._buffer.clear()
        except Exception:  # noqa: BLE001
            logger.debug("Telemetry flush failed", exc_info=True)

    def _flush_at_exit(self) -> None:
        """Atexit hook — flush remaining events and close."""
        try:
            self.flush()
        except Exception:  # noqa: BLE001
            pass

    # ── Lazy SQLite connection ────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create the database connection (lazy, WAL mode)."""
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            logger.debug("Telemetry DB opened: %s", self._db_path)
        return self._conn

    async def _get_aconn(self) -> aiosqlite.Connection:
        """Get or create the async database connection (lazy, WAL mode)."""
        if self._aconn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._aconn = await aiosqlite.connect(str(self._db_path))
            self._aconn.row_factory = aiosqlite.Row
            await self._aconn.execute("PRAGMA journal_mode=WAL")
            await self._aconn.executescript(_SCHEMA)
            logger.debug("Telemetry async DB opened: %s", self._db_path)
        return self._aconn

    # ── Query & Analytics ─────────────────────────────────────

    def query_events(
        self,
        event_type: str | None = None,
        *,
        since: str | None = None,
        until: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query persisted events with optional filters.

        Args:
            event_type: Filter by event type (e.g. ``"tool_call"``).
            since: ISO timestamp lower bound (inclusive).
            until: ISO timestamp upper bound (inclusive).
            limit: Maximum rows to return.

        Returns:
            List of event dicts ordered by timestamp descending.
        """
        # Flush buffer first so queries see latest data
        self.flush()

        conditions: list[str] = []
        params: list[Any] = []

        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        if until:
            conditions.append("timestamp <= ?")
            params.append(until)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM telemetry_events {where} ORDER BY timestamp DESC LIMIT ?"  # noqa: S608
        params.append(limit)

        try:
            conn = self._get_conn()
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]
        except Exception:  # noqa: BLE001
            logger.debug("Telemetry query failed", exc_info=True)
            return []

    def get_summary(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        """Return aggregated usage summary.

        Returns:
            Dict with keys: ``total_events``, ``by_type``, ``top_tools``,
            ``api_calls`` (tokens, cost), ``error_count``.
        """
        # Flush buffer first so summary includes latest data
        self.flush()

        conditions: list[str] = []
        params: list[Any] = []

        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        if until:
            conditions.append("timestamp <= ?")
            params.append(until)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        try:
            conn = self._get_conn()

            # Total events
            total = conn.execute(
                f"SELECT COUNT(*) FROM telemetry_events {where}",  # noqa: S608
                params,
            ).fetchone()[0]

            # By type
            by_type_rows = conn.execute(
                f"SELECT event_type, COUNT(*) as cnt FROM telemetry_events {where} GROUP BY event_type ORDER BY cnt DESC",  # noqa: S608, E501
                params,
            ).fetchall()
            by_type = {row["event_type"]: row["cnt"] for row in by_type_rows}

            # Top tools
            tool_where = f"{where} AND event_type = 'tool_call'" if where else "WHERE event_type = 'tool_call'"
            top_tools_rows = conn.execute(
                f"SELECT tool_name, COUNT(*) as cnt FROM telemetry_events {tool_where} GROUP BY tool_name ORDER BY cnt DESC LIMIT 10",  # noqa: S608, E501
                params,
            ).fetchall()
            top_tools = {row["tool_name"]: row["cnt"] for row in top_tools_rows}

            # API stats
            api_where = f"{where} AND event_type = 'api_call'" if where else "WHERE event_type = 'api_call'"
            api_row = conn.execute(
                f"SELECT COUNT(*) as cnt, COALESCE(SUM(tokens_in), 0) as total_in, "  # noqa: S608
                f"COALESCE(SUM(tokens_out), 0) as total_out, "
                f"COALESCE(SUM(cost_usd), 0.0) as total_cost "
                f"FROM telemetry_events {api_where}",
                params,
            ).fetchone()
            api_calls = {
                "count": api_row["cnt"],
                "total_tokens_in": api_row["total_in"],
                "total_tokens_out": api_row["total_out"],
                "total_cost_usd": api_row["total_cost"],
            }

            # Error count
            error_where = f"{where} AND event_type = 'error'" if where else "WHERE event_type = 'error'"
            error_count = conn.execute(
                f"SELECT COUNT(*) FROM telemetry_events {error_where}",  # noqa: S608
                params,
            ).fetchone()[0]

            return {
                "total_events": total,
                "by_type": by_type,
                "top_tools": top_tools,
                "api_calls": api_calls,
                "error_count": error_count,
            }
        except Exception:  # noqa: BLE001
            logger.debug("Telemetry summary failed", exc_info=True)
            return {
                "total_events": 0,
                "by_type": {},
                "top_tools": {},
                "api_calls": {"count": 0, "total_tokens_in": 0, "total_tokens_out": 0, "total_cost_usd": 0.0},
                "error_count": 0,
            }

    def clear_events(self, older_than_days: int = 30) -> int:
        """Delete events older than the specified number of days.

        Returns:
            Number of rows deleted.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
        try:
            conn = self._get_conn()
            cursor = conn.execute(
                "DELETE FROM telemetry_events WHERE timestamp < ?",
                (cutoff,),
            )
            conn.commit()
            deleted = cursor.rowcount
            logger.debug("Telemetry: cleared %d events older than %d days", deleted, older_than_days)
            return deleted
        except Exception:  # noqa: BLE001
            logger.debug("Telemetry clear failed", exc_info=True)
            return 0

    # ── Async methods ─────────────────────────────────────────

    async def async_flush(self) -> None:
        """Flush buffered events to SQLite asynchronously using aiosqlite.

        Drains the buffer under the threading lock (to avoid races with
        sync ``emit`` / ``_append`` calls) and writes via the async connection.
        """
        async with self._async_lock:
            # Drain buffer under threading lock — emit() is sync
            with self._lock:
                if not self._buffer or self._closed:
                    return
                events = list(self._buffer)
                self._buffer.clear()
            try:
                conn = await self._get_aconn()
                await conn.executemany(_INSERT_SQL, [e.to_row() for e in events])
                await conn.commit()
            except Exception:  # noqa: BLE001
                logger.debug("Telemetry async flush failed", exc_info=True)

    async def async_query_events(
        self,
        event_type: str | None = None,
        *,
        since: str | None = None,
        until: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Async version of :meth:`query_events`.

        Flushes the buffer asynchronously first so queries see latest data.
        """
        await self.async_flush()

        conditions: list[str] = []
        params: list[Any] = []

        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        if until:
            conditions.append("timestamp <= ?")
            params.append(until)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM telemetry_events {where} ORDER BY timestamp DESC LIMIT ?"  # noqa: S608
        params.append(limit)

        try:
            conn = await self._get_aconn()
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception:  # noqa: BLE001
            logger.debug("Telemetry async query failed", exc_info=True)
            return []

    async def async_get_summary(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        """Async version of :meth:`get_summary`."""
        await self.async_flush()

        conditions: list[str] = []
        params: list[Any] = []

        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        if until:
            conditions.append("timestamp <= ?")
            params.append(until)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        try:
            conn = await self._get_aconn()

            # Total events
            cursor = await conn.execute(
                f"SELECT COUNT(*) FROM telemetry_events {where}",  # noqa: S608
                params,
            )
            row = await cursor.fetchone()
            total = row[0] if row else 0

            # By type
            cursor = await conn.execute(
                f"SELECT event_type, COUNT(*) as cnt FROM telemetry_events {where} GROUP BY event_type ORDER BY cnt DESC",  # noqa: S608, E501
                params,
            )
            by_type_rows = await cursor.fetchall()
            by_type = {r["event_type"]: r["cnt"] for r in by_type_rows}

            # Top tools
            tool_where = f"{where} AND event_type = 'tool_call'" if where else "WHERE event_type = 'tool_call'"
            cursor = await conn.execute(
                f"SELECT tool_name, COUNT(*) as cnt FROM telemetry_events {tool_where} GROUP BY tool_name ORDER BY cnt DESC LIMIT 10",  # noqa: S608, E501
                params,
            )
            top_tools_rows = await cursor.fetchall()
            top_tools = {r["tool_name"]: r["cnt"] for r in top_tools_rows}

            # API stats
            api_where = f"{where} AND event_type = 'api_call'" if where else "WHERE event_type = 'api_call'"
            cursor = await conn.execute(
                f"SELECT COUNT(*) as cnt, COALESCE(SUM(tokens_in), 0) as total_in, "  # noqa: S608
                f"COALESCE(SUM(tokens_out), 0) as total_out, "
                f"COALESCE(SUM(cost_usd), 0.0) as total_cost "
                f"FROM telemetry_events {api_where}",
                params,
            )
            api_row = await cursor.fetchone()
            if api_row is not None:
                api_calls = {
                    "count": api_row["cnt"],
                    "total_tokens_in": api_row["total_in"],
                    "total_tokens_out": api_row["total_out"],
                    "total_cost_usd": api_row["total_cost"],
                }
            else:
                api_calls = {"count": 0, "total_tokens_in": 0, "total_tokens_out": 0, "total_cost_usd": 0.0}

            # Error count
            error_where = f"{where} AND event_type = 'error'" if where else "WHERE event_type = 'error'"
            cursor = await conn.execute(
                f"SELECT COUNT(*) FROM telemetry_events {error_where}",  # noqa: S608
                params,
            )
            row = await cursor.fetchone()
            error_count = row[0] if row else 0

            return {
                "total_events": total,
                "by_type": by_type,
                "top_tools": top_tools,
                "api_calls": api_calls,
                "error_count": error_count,
            }
        except Exception:  # noqa: BLE001
            logger.debug("Telemetry async summary failed", exc_info=True)
            return {
                "total_events": 0,
                "by_type": {},
                "top_tools": {},
                "api_calls": {"count": 0, "total_tokens_in": 0, "total_tokens_out": 0, "total_cost_usd": 0.0},
                "error_count": 0,
            }

    async def async_clear_events(self, older_than_days: int = 30) -> int:
        """Async version of :meth:`clear_events`."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
        try:
            conn = await self._get_aconn()
            cursor = await conn.execute(
                "DELETE FROM telemetry_events WHERE timestamp < ?",
                (cutoff,),
            )
            await conn.commit()
            deleted = cursor.rowcount
            logger.debug("Telemetry: async cleared %d events older than %d days", deleted, older_than_days)
            return deleted
        except Exception:  # noqa: BLE001
            logger.debug("Telemetry async clear failed", exc_info=True)
            return 0

    # ── Lifecycle ─────────────────────────────────────────────

    def close(self) -> None:
        """Flush remaining events and close the database connection."""
        with self._lock:
            self._flush_locked()
            self._closed = True
            if self._conn:
                self._conn.close()
                self._conn = None
            # Async connection is closed separately via async_close()

    async def async_close(self) -> None:
        """Flush remaining events asynchronously and close both connections."""
        await self.async_flush()
        self._closed = True
        if self._aconn:
            await self._aconn.close()
            self._aconn = None
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None


# ── Singleton ─────────────────────────────────────────────────

_collector: TelemetryCollector | None = None
_collector_lock = threading.Lock()


def get_telemetry_collector(
    settings: Any | None = None,
) -> TelemetryCollector:
    """Get or create the global TelemetryCollector singleton.

    Respects the ``VAIG_TELEMETRY_ENABLED`` environment variable
    (``"false"`` / ``"0"`` disables collection).

    Args:
        settings: Optional ``Settings`` instance. When ``None``, imports
            and calls ``get_settings()`` lazily to avoid circular imports.

    Returns:
        The process-wide ``TelemetryCollector`` instance.
    """
    global _collector  # noqa: PLW0603
    if _collector is not None:
        return _collector

    with _collector_lock:
        # Double-check after acquiring lock
        if _collector is not None:
            return _collector

        if settings is None:
            from vaig.core.config import get_settings

            logger.debug(
                "get_telemetry_collector() called without settings — "
                "falling back to get_settings() singleton. "
                "Prefer initializing with settings from the CLI layer."
            )
            settings = get_settings()

        # Env var override: VAIG_TELEMETRY_ENABLED=false disables
        env_flag = os.environ.get("VAIG_TELEMETRY_ENABLED", "").lower()
        if env_flag in ("false", "0", "no"):
            enabled = False
        else:
            enabled = settings.telemetry.enabled

        _collector = TelemetryCollector(
            db_path=Path(settings.session.db_path).expanduser().parent / "telemetry.db",
            enabled=enabled,
            buffer_size=settings.telemetry.buffer_size,
        )
        return _collector


def reset_telemetry_collector() -> None:
    """Close and discard the global collector (for testing)."""
    global _collector  # noqa: PLW0603
    with _collector_lock:
        if _collector is not None:
            _collector.close()
            _collector = None

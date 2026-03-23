"""Pure data transformation functions — vaig internal formats → BigQuery row dicts.

These functions are intentionally free of side effects:
- No I/O, no network calls, no GCP dependencies.
- Each function accepts a raw record dict and returns a BQ-compatible row dict.
- All timestamp fields are normalised to ``datetime`` objects in UTC.
- Nested dicts / lists that map to JSON-typed BQ columns are serialised with
  ``json.dumps``.
- Large text fields are truncated to stay within BigQuery row-size limits.

Typical usage::

    from vaig.core.export_transformers import transform_telemetry_record

    bq_row = transform_telemetry_record(sqlite_row_dict)
    bq_client.insert_rows_json(table, [bq_row])
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# ── Truncation limits (BigQuery STRING column max is 10 MB, but we keep
#    values small so entire rows stay well under the 1 MB insert limit.) ──

_TRUNCATE_OUTPUT_SUMMARY = 10_000
_TRUNCATE_HEALTH_SUMMARY = 50_000
_TRUNCATE_HEALTH_MARKDOWN = 100_000


# ── Internal helpers ─────────────────────────────────────────────────────


def _parse_timestamp(value: Any) -> datetime:
    """Coerce *value* to a timezone-aware UTC ``datetime``.

    Accepts:
    - ``datetime`` — returned as-is if already aware, made UTC if naïve.
    - ``str`` — parsed as an ISO-8601 string (with or without timezone info).
    - ``int`` / ``float`` — interpreted as a Unix timestamp (seconds).
    - Anything else — falls back to ``datetime.now(UTC)``.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        except ValueError:
            logger.warning("Could not parse timestamp string %r; using now(UTC)", value)
            return datetime.now(UTC)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=UTC)
    logger.warning("Unexpected timestamp type %s; using now(UTC)", type(value).__name__)
    return datetime.now(UTC)


def _to_json_string(value: Any) -> str:
    """Serialise *value* to a JSON string.

    - If *value* is already a ``str``, it is returned unchanged (assumes the
      caller already serialised it).
    - ``None`` → ``"null"``
    - Everything else → ``json.dumps(value)``
    """
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(value))


def _truncate(text: str | None, max_len: int) -> str | None:
    """Return *text* truncated to *max_len* characters, or ``None`` if falsy."""
    if not text:
        return text
    return text[:max_len]


# ── Public transformers ──────────────────────────────────────────────────


def transform_telemetry_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw SQLite telemetry row to a BigQuery-compatible row dict.

    Args:
        record: A dict representing a single row from the ``telemetry_events``
            SQLite table (e.g. fetched with ``sqlite3.Row`` or ``aiosqlite``).

    Returns:
        A flat dict with types that BigQuery's streaming insert accepts.

    Field mapping:

    +--------------------+--------------------------------------+
    | BQ column          | Source field                         |
    +====================+======================================+
    | timestamp          | timestamp (str → datetime, UTC)      |
    | event_type         | event_type (str)                     |
    | tool_name          | tool_name (str)                      |
    | agent_name         | agent_name / event_name (str)        |
    | duration_ms        | duration_ms (float, default 0.0)     |
    | success            | success (bool)                       |
    | error_message      | error_message / error_msg (str|None) |
    | metadata           | metadata / metadata_json (JSON str)  |
    | session_id         | session_id (str)                     |
    | run_id             | run_id (str)                         |
    +--------------------+--------------------------------------+
    """
    # Prefer 'agent_name'; fall back to 'event_name' (telemetry schema alias).
    agent_name = record.get("agent_name") or record.get("event_name") or ""

    # 'error_message' and 'error_msg' are both used across schema versions.
    error_message = record.get("error_message") or record.get("error_msg") or None

    # 'metadata' may be a pre-serialised JSON string or a raw dict.
    raw_metadata = record.get("metadata") or record.get("metadata_json") or {}
    metadata_str = _to_json_string(raw_metadata)

    # 'success' may come as bool, int (0/1), or string.
    raw_success = record.get("success")
    if isinstance(raw_success, bool):
        success = raw_success
    elif isinstance(raw_success, (int, float)):
        success = bool(raw_success)
    elif isinstance(raw_success, str):
        success = raw_success.lower() not in ("false", "0", "no", "")
    else:
        # Default: infer from absence of error.
        success = error_message is None

    return {
        "timestamp": _parse_timestamp(record.get("timestamp")),
        "event_type": str(record.get("event_type") or ""),
        "tool_name": str(record.get("tool_name") or ""),
        "agent_name": str(agent_name),
        "duration_ms": float(record.get("duration_ms") or 0.0),
        "success": success,
        "error_message": str(error_message) if error_message is not None else None,
        "metadata": metadata_str,
        "session_id": str(record.get("session_id") or ""),
        "run_id": str(record.get("run_id") or ""),
    }


def transform_tool_call_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a JSONL tool-call record to a BigQuery-compatible row dict.

    Args:
        record: A dict parsed from a single line of a ``ToolCallStore`` JSONL
            file (the JSON-decoded ``ToolCallRecord``).

    Returns:
        A flat dict with types that BigQuery's streaming insert accepts.

    Field mapping:

    +--------------------+-------------------------------------------+
    | BQ column          | Source field                              |
    +====================+===========================================+
    | timestamp          | timestamp (str → datetime, UTC)           |
    | tool_name          | tool_name (str)                           |
    | agent_name         | agent_name (str)                          |
    | input_params       | input_params (JSON str)                   |
    | output_summary     | output_summary (str, max 10 000 chars)    |
    | duration_ms        | duration_ms (float)                       |
    | success            | success (bool)                            |
    | error_message      | error_message (str|None)                  |
    | run_id             | run_id (str)                              |
    | session_id         | session_id (str)                          |
    +--------------------+-------------------------------------------+
    """
    error_message = record.get("error_message") or None

    raw_success = record.get("success")
    if isinstance(raw_success, bool):
        success = raw_success
    elif isinstance(raw_success, (int, float)):
        success = bool(raw_success)
    elif isinstance(raw_success, str):
        success = raw_success.lower() not in ("false", "0", "no", "")
    else:
        success = error_message is None

    raw_output = record.get("output_summary") or ""
    output_summary = _truncate(str(raw_output), _TRUNCATE_OUTPUT_SUMMARY)

    return {
        "timestamp": _parse_timestamp(record.get("timestamp")),
        "tool_name": str(record.get("tool_name") or ""),
        "agent_name": str(record.get("agent_name") or ""),
        "input_params": _to_json_string(record.get("input_params") or {}),
        "output_summary": output_summary,
        "duration_ms": float(record.get("duration_ms") or 0.0),
        "success": success,
        "error_message": str(error_message) if error_message is not None else None,
        "run_id": str(record.get("run_id") or ""),
        "session_id": str(record.get("session_id") or ""),
    }


def transform_health_report(
    report: dict[str, Any],
    run_id: str,
    cluster_name: str = "",
    namespace: str = "",
) -> dict[str, Any]:
    """Convert a ``HealthReport.to_dict()`` output to a BigQuery-compatible row.

    The report's ``findings`` list is preserved as a list of flattened dicts
    (BigQuery REPEATED RECORD).  Each finding keeps: ``category``, ``severity``,
    ``title``, ``description``, and ``recommendation``.

    Args:
        report: Output of ``HealthReport.to_dict()`` (or any equivalent dict).
        run_id: Unique pipeline run identifier.
        cluster_name: Kubernetes cluster name (for partitioning / filtering).
        namespace: Kubernetes namespace (for partitioning / filtering).

    Returns:
        A dict suitable for BigQuery streaming insert.

    Field mapping:

    +--------------------+--------------------------------------------------+
    | BQ column          | Source                                           |
    +====================+==================================================+
    | timestamp          | datetime.now(UTC) — report generation time       |
    | run_id             | parameter                                        |
    | cluster_name       | parameter                                        |
    | namespace          | parameter                                        |
    | overall_status     | executive_summary.overall_status or overall_     |
    |                    | status (str)                                     |
    | summary            | executive_summary fields or summary (max 50 000) |
    | findings           | findings list → flattened dicts                  |
    | metadata           | report.metadata → JSON string                    |
    | report_markdown    | report.report_markdown (max 100 000)             |
    +--------------------+--------------------------------------------------+
    """
    # ── overall_status ──
    # HealthReport nests the overall status under executive_summary.
    exec_summary = report.get("executive_summary") or {}
    if isinstance(exec_summary, dict):
        overall_status = exec_summary.get("overall_status") or report.get("overall_status") or ""
    else:
        overall_status = report.get("overall_status") or ""
    # Enum values may be dicts with a 'value' key (Pydantic model_dump output).
    if isinstance(overall_status, dict):
        overall_status = overall_status.get("value", "")
    overall_status = str(overall_status)

    # ── summary ──
    # Build a plain-text summary from executive_summary or a top-level 'summary'.
    if isinstance(exec_summary, dict) and exec_summary:
        summary_parts: list[str] = []
        if exec_summary.get("scope"):
            summary_parts.append(f"Scope: {exec_summary['scope']}")
        if exec_summary.get("issues_found") is not None:
            summary_parts.append(f"Issues: {exec_summary['issues_found']}")
        if exec_summary.get("critical_count") is not None:
            summary_parts.append(f"Critical: {exec_summary['critical_count']}")
        if exec_summary.get("summary"):
            summary_parts.append(str(exec_summary["summary"]))
        summary_raw = " | ".join(summary_parts) if summary_parts else ""
    else:
        summary_raw = str(report.get("summary") or "")
    summary = _truncate(summary_raw, _TRUNCATE_HEALTH_SUMMARY) or ""

    # ── findings ──
    raw_findings: list[Any] = report.get("findings") or []
    findings: list[dict[str, Any]] = []
    for f in raw_findings:
        if not isinstance(f, dict):
            continue
        severity = f.get("severity") or ""
        if isinstance(severity, dict):
            severity = severity.get("value", "")
        findings.append(
            {
                "category": str(f.get("category") or ""),
                "severity": str(severity),
                "title": str(f.get("title") or ""),
                "description": str(f.get("description") or ""),
                "recommendation": str(f.get("recommendation") or ""),
            }
        )

    # ── metadata ──
    raw_meta = report.get("metadata") or {}
    metadata_str = _to_json_string(raw_meta)

    # ── report_markdown ──
    raw_md = report.get("report_markdown") or None
    report_markdown = _truncate(str(raw_md), _TRUNCATE_HEALTH_MARKDOWN) if raw_md else None

    return {
        "timestamp": datetime.now(UTC),
        "run_id": str(run_id),
        "cluster_name": str(cluster_name),
        "namespace": str(namespace),
        "overall_status": overall_status,
        "summary": summary,
        "findings": findings,
        "metadata": metadata_str,
        "report_markdown": report_markdown,
    }


def transform_feedback_record(
    feedback: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Convert a user feedback dict to a BigQuery-compatible row.

    Args:
        feedback: Raw feedback data (e.g. from a CLI prompt or API call).
        run_id: Unique pipeline run identifier.

    Returns:
        A flat dict with types that BigQuery's streaming insert accepts.

    Field mapping:

    +----------------------+--------------------------------------------+
    | BQ column            | Source                                     |
    +======================+============================================+
    | timestamp            | datetime.now(UTC)                          |
    | run_id               | parameter                                  |
    | rating               | rating (int, clamped 1–5)                  |
    | comment              | comment (str)                              |
    | auto_quality_score   | auto_quality_score (float)                 |
    | report_summary       | report_summary (str)                       |
    | metadata             | metadata → JSON string                     |
    +----------------------+--------------------------------------------+
    """
    # ── rating — clamp to valid 1–5 range ──
    raw_rating = feedback.get("rating")
    try:
        rating = int(raw_rating)  # type: ignore[arg-type]
        rating = max(1, min(5, rating))
    except (TypeError, ValueError):
        rating = 0  # sentinel: indicates missing / invalid

    # ── auto_quality_score ──
    raw_score = feedback.get("auto_quality_score")
    try:
        auto_quality_score = float(raw_score)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        auto_quality_score = 0.0

    return {
        "timestamp": datetime.now(UTC),
        "run_id": str(run_id),
        "rating": rating,
        "comment": str(feedback.get("comment") or ""),
        "auto_quality_score": auto_quality_score,
        "report_summary": str(feedback.get("report_summary") or ""),
        "metadata": _to_json_string(feedback.get("metadata") or {}),
    }

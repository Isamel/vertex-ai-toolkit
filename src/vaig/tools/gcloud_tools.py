"""GCP observability tools — Cloud Logging and Cloud Monitoring queries."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from vaig.tools.base import ToolDef, ToolParam, ToolResult

logger = logging.getLogger(__name__)


# ── Lazy SDK imports ─────────────────────────────────────────
# google-cloud-logging and google-cloud-monitoring may not be installed.
# We defer imports to execution time so the module can always be loaded.


def _get_logging_client(project: str | None = None) -> tuple[Any, str | None]:
    """Create a Cloud Logging client with lazy import."""
    try:
        from google.cloud import logging as cloud_logging  # noqa: I001
    except ImportError:
        return None, "google-cloud-logging SDK is not installed. Run: pip install google-cloud-logging"

    try:
        client = cloud_logging.Client(project=project) if project else cloud_logging.Client()
        return client, None
    except Exception as exc:
        return None, f"Failed to create Cloud Logging client: {exc}"


def _get_monitoring_client(project: str | None = None) -> tuple[Any, str | None]:
    """Create a Cloud Monitoring client with lazy import."""
    try:
        from google.cloud import monitoring_v3  # noqa: I001
    except ImportError:
        return None, "google-cloud-monitoring SDK is not installed. Run: pip install google-cloud-monitoring"

    try:
        client = monitoring_v3.MetricServiceClient()
        return client, None
    except Exception as exc:
        return None, f"Failed to create Cloud Monitoring client: {exc}"


# ── Helpers ──────────────────────────────────────────────────


def _format_log_entry(entry) -> str:  # noqa: ANN001
    """Format a single Cloud Logging entry into a readable line."""
    ts = getattr(entry, "timestamp", None)
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "N/A"

    severity = getattr(entry, "severity", "DEFAULT") or "DEFAULT"

    # Resource description
    resource = getattr(entry, "resource", None)
    resource_str = "N/A"
    if resource:
        rtype = getattr(resource, "type", "unknown")
        labels = getattr(resource, "labels", {}) or {}
        label_parts = [f"{k}={v}" for k, v in labels.items()]
        resource_str = f"{rtype}"
        if label_parts:
            resource_str += f"({', '.join(label_parts[:3])})"

    # Payload — prefer textPayload, then jsonPayload, then protoPayload
    payload = ""
    text_payload = getattr(entry, "text_payload", None) or getattr(entry, "payload", None)
    json_payload = getattr(entry, "json_payload", None)
    proto_payload = getattr(entry, "proto_payload", None)

    if text_payload and isinstance(text_payload, str):
        payload = text_payload
    elif json_payload:
        # Flatten JSON payload into key=value pairs for readability
        if isinstance(json_payload, dict):
            msg = json_payload.get("message", "")
            if msg:
                payload = str(msg)
            else:
                parts = [f"{k}={v}" for k, v in list(json_payload.items())[:5]]
                payload = "{" + ", ".join(parts) + "}"
        else:
            payload = str(json_payload)
    elif proto_payload:
        payload = str(proto_payload)[:200]
    else:
        payload = "(empty payload)"

    # Truncate long payloads
    if len(payload) > 500:
        payload = payload[:497] + "..."

    return f"[{ts_str}] {severity:<8} {resource_str}  {payload}"


def _format_time_series(time_series_list, metric_type: str) -> str:
    """Format Cloud Monitoring time series data into a readable table."""
    if not time_series_list:
        return f"No time series data found for metric: {metric_type}"

    lines: list[str] = []
    lines.append(f"Metric: {metric_type}")
    lines.append(f"Series count: {len(time_series_list)}")
    lines.append("")

    for idx, ts in enumerate(time_series_list):
        # Extract labels
        metric_labels = dict(ts.metric.labels) if ts.metric and ts.metric.labels else {}
        resource_labels = dict(ts.resource.labels) if ts.resource and ts.resource.labels else {}

        # Combine labels for display
        all_labels = {**resource_labels, **metric_labels}
        label_str = ", ".join(f"{k}={v}" for k, v in list(all_labels.items())[:5])

        lines.append(f"--- Series {idx + 1} [{label_str}] ---")
        lines.append(f"{'Timestamp':<24} | {'Value':>15}")
        lines.append(f"{'-' * 24}-+-{'-' * 15}")

        points = list(ts.points) if ts.points else []
        # Points come in reverse chronological order from the API
        for point in points[:50]:  # Limit to 50 points per series
            # Timestamp
            ts_val = point.interval.end_time
            if ts_val:
                if hasattr(ts_val, "strftime"):
                    ts_str = ts_val.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    ts_str = str(ts_val)
            else:
                ts_str = "N/A"

            # Value — typed value uses oneof: int64_value, double_value,
            # bool_value, string_value, distribution_value.
            # Use _pb.WhichOneof when available for accurate detection.
            value = point.value
            val_str = "N/A"
            value_pb = getattr(value, "_pb", value)
            kind = None
            if hasattr(value_pb, "WhichOneof"):
                kind = value_pb.WhichOneof("value")

            if kind == "int64_value":
                val_str = str(value.int64_value)
            elif kind == "double_value":
                val_str = f"{value.double_value:.6f}"
            elif kind == "bool_value":
                val_str = str(value.bool_value)
            elif kind == "string_value":
                val_str = value.string_value
            elif kind == "distribution_value":
                dist = value.distribution_value
                val_str = f"mean={dist.mean:.4f}" if hasattr(dist, "mean") else "dist"
            elif kind is None:
                # Fallback for non-protobuf objects (e.g. in tests)
                if getattr(value, "double_value", None) is not None:
                    val_str = f"{value.double_value:.6f}"
                elif getattr(value, "int64_value", None) is not None:
                    val_str = str(value.int64_value)

            lines.append(f"{ts_str:<24} | {val_str:>15}")

        if len(points) > 50:
            lines.append(f"  ... and {len(points) - 50} more points")

        lines.append("")

    return "\n".join(lines)


# ── Tool 3.1 — gcloud_logging_query ─────────────────────────


def gcloud_logging_query(
    filter_expr: str,
    *,
    project: str = "",
    limit: int = 100,
    order_by: str = "timestamp desc",
) -> ToolResult:
    """Query Cloud Logging entries using a filter expression.

    Returns formatted log entries with timestamp, severity, resource,
    and payload content.
    """
    logger.debug(
        "gcloud_logging_query: filter=%r project=%s limit=%d order_by=%s",
        filter_expr, project, limit, order_by,
    )

    if not filter_expr.strip():
        return ToolResult(
            output="Filter expression cannot be empty. Example: 'resource.type=\"k8s_container\" severity>=ERROR'",
            error=True,
        )

    # Clamp limit to reasonable bounds
    if limit < 1:
        limit = 1
    elif limit > 1000:
        limit = 1000

    # Validate order_by
    valid_order_by = {"timestamp desc", "timestamp asc"}
    if order_by not in valid_order_by:
        return ToolResult(
            output=f"Invalid order_by: '{order_by}'. Must be one of: {', '.join(sorted(valid_order_by))}",
            error=True,
        )

    effective_project = project if project else None
    client, err = _get_logging_client(effective_project)
    if err:
        return ToolResult(output=err, error=True)

    try:
        entries = list(
            client.list_entries(
                filter_=filter_expr,
                order_by=order_by,
                max_results=limit,
            )
        )
    except Exception as exc:
        # Attempt typed classification via google.api_core.exceptions
        try:
            from google.api_core.exceptions import (
                Forbidden,
                InvalidArgument,
                NotFound,
                PermissionDenied,
                ResourceExhausted,
            )
        except ImportError:
            # Fallback — SDK not installed; just report the raw error.
            return ToolResult(output=f"Error querying Cloud Logging: {exc}", error=True)

        if isinstance(exc, (PermissionDenied, Forbidden)):
            return ToolResult(
                output=f"Permission denied querying Cloud Logging. Ensure the service account has 'roles/logging.viewer'. Error: {exc}",
                error=True,
            )
        if isinstance(exc, ResourceExhausted):
            return ToolResult(
                output=f"Cloud Logging API quota exceeded. Try reducing the limit or narrowing the filter. Error: {exc}",
                error=True,
            )
        if isinstance(exc, (InvalidArgument, NotFound)):
            return ToolResult(
                output=f"Invalid filter expression: '{filter_expr}'. Check the Cloud Logging filter syntax. Error: {exc}",
                error=True,
            )

        return ToolResult(
            output=f"Error querying Cloud Logging: {exc}",
            error=True,
        )

    if not entries:
        return ToolResult(
            output=f"No log entries found matching filter: {filter_expr}",
        )

    formatted = [_format_log_entry(entry) for entry in entries]
    header = f"Found {len(entries)} log entries (limit: {limit}, order: {order_by}):\n"
    return ToolResult(output=header + "\n".join(formatted))


# ── Tool 3.2 — gcloud_monitoring_query ───────────────────────


def gcloud_monitoring_query(
    metric_type: str,
    *,
    project: str = "",
    interval_minutes: int = 60,
    aggregation: str = "",
    filter_str: str = "",
) -> ToolResult:
    """Query Cloud Monitoring time series data for a given metric.

    Returns formatted time series with timestamps, values, and labels.
    """
    logger.debug(
        "gcloud_monitoring_query: metric=%s project=%s interval=%d",
        metric_type, project, interval_minutes,
    )

    if not metric_type.strip():
        return ToolResult(
            output=(
                "Metric type cannot be empty. Examples:\n"
                "  compute.googleapis.com/instance/cpu/utilization\n"
                "  kubernetes.io/container/memory/used_bytes\n"
                "  kubernetes.io/container/cpu/core_usage_time\n"
                "  kubernetes.io/container/restart_count"
            ),
            error=True,
        )

    # Clamp interval
    if interval_minutes < 1:
        interval_minutes = 1
    elif interval_minutes > 10080:  # max 7 days
        interval_minutes = 10080

    client, err = _get_monitoring_client()
    if err:
        return ToolResult(output=err, error=True)

    # Determine project name for the API
    effective_project = project if project else None
    if not effective_project:
        # Try to detect from environment
        try:
            import google.auth
            _, detected_project = google.auth.default()
            effective_project = detected_project
        except Exception:
            logger.debug("Could not auto-detect GCP project from auth", exc_info=True)

    if not effective_project:
        return ToolResult(
            output="No GCP project specified and could not auto-detect from environment. Set project_id in GKE config or GOOGLE_CLOUD_PROJECT env var.",
            error=True,
        )

    project_name = f"projects/{effective_project}"

    # Build time interval
    try:
        from google.cloud.monitoring_v3 import types as monitoring_types
        from google.protobuf import duration_pb2
    except ImportError:
        return ToolResult(
            output="google-cloud-monitoring SDK is not installed. Run: pip install google-cloud-monitoring",
            error=True,
        )

    now = datetime.now(tz=timezone.utc)
    seconds_ago = interval_minutes * 60

    interval = monitoring_types.TimeInterval()
    interval.end_time.FromDatetime(now)
    interval.start_time.FromDatetime(
        datetime.fromtimestamp(now.timestamp() - seconds_ago, tz=timezone.utc)
    )

    # Build filter — must include metric.type
    monitoring_filter = f'metric.type = "{metric_type}"'
    if filter_str.strip():
        monitoring_filter += f" AND {filter_str}"

    # Build request
    request = monitoring_types.ListTimeSeriesRequest(
        name=project_name,
        filter=monitoring_filter,
        interval=interval,
        view=monitoring_types.ListTimeSeriesRequest.TimeSeriesView.FULL,
    )

    # Apply aggregation if specified
    if aggregation.strip():
        agg_map = {
            "mean": monitoring_types.Aggregation.Reducer.REDUCE_MEAN,
            "max": monitoring_types.Aggregation.Reducer.REDUCE_MAX,
            "min": monitoring_types.Aggregation.Reducer.REDUCE_MIN,
            "sum": monitoring_types.Aggregation.Reducer.REDUCE_SUM,
            "count": monitoring_types.Aggregation.Reducer.REDUCE_COUNT,
        }
        reducer = agg_map.get(aggregation.lower())
        if reducer is None:
            return ToolResult(
                output=f"Invalid aggregation: '{aggregation}'. Must be one of: {', '.join(sorted(agg_map))}",
                error=True,
            )

        agg = monitoring_types.Aggregation(
            alignment_period=duration_pb2.Duration(seconds=60),
            per_series_aligner=monitoring_types.Aggregation.Aligner.ALIGN_MEAN,
            cross_series_reducer=reducer,
        )
        request.aggregation.CopyFrom(agg)

    try:
        results = client.list_time_series(request=request)
        time_series_list = list(results)
    except Exception as exc:
        # Attempt typed classification via google.api_core.exceptions
        try:
            from google.api_core.exceptions import (
                Forbidden,
                InvalidArgument,
                NotFound,
                PermissionDenied,
                ResourceExhausted,
            )
        except ImportError:
            return ToolResult(output=f"Error querying Cloud Monitoring: {exc}", error=True)

        if isinstance(exc, (PermissionDenied, Forbidden)):
            return ToolResult(
                output=f"Permission denied querying Cloud Monitoring. Ensure the service account has 'roles/monitoring.viewer'. Error: {exc}",
                error=True,
            )
        if isinstance(exc, ResourceExhausted):
            return ToolResult(
                output=f"Cloud Monitoring API quota exceeded. Try reducing the interval or narrowing the filter. Error: {exc}",
                error=True,
            )
        if isinstance(exc, (InvalidArgument, NotFound)):
            return ToolResult(
                output=(
                    f"Metric not found or invalid: '{metric_type}'. "
                    "Check the metric name at https://cloud.google.com/monitoring/api/metrics_gcp. "
                    f"Error: {exc}"
                ),
                error=True,
            )

        return ToolResult(
            output=f"Error querying Cloud Monitoring: {exc}",
            error=True,
        )

    formatted = _format_time_series(time_series_list, metric_type)
    return ToolResult(output=formatted)


# ── Task 3.3 — Tool factory ─────────────────────────────────


def create_gcloud_tools(
    project: str = "",
    log_limit: int = 100,
    metrics_interval_minutes: int = 60,
) -> list[ToolDef]:
    """Create all GCP observability tool definitions.

    Args:
        project: GCP project ID (from GKEConfig). Falls back to ADC default if empty.
        log_limit: Default log entry limit (from GKEConfig).
        metrics_interval_minutes: Default monitoring interval (from GKEConfig).

    Returns:
        List of ToolDef instances for Cloud Logging and Cloud Monitoring queries.
    """
    return [
        ToolDef(
            name="gcloud_logging_query",
            description=(
                "Query Google Cloud Logging entries using a filter expression. "
                "Returns formatted log entries with timestamp, severity, resource, and payload. "
                "Common filters: resource.type=\"k8s_container\", severity>=ERROR, "
                "resource.labels.namespace_name=\"production\". "
                "Uses Cloud Logging filter syntax: https://cloud.google.com/logging/docs/view/logging-query-language"
            ),
            parameters=[
                ToolParam(
                    name="filter_expr",
                    type="string",
                    description=(
                        "Cloud Logging filter expression. Examples: "
                        "'resource.type=\"k8s_container\" severity>=ERROR', "
                        "'resource.type=\"k8s_container\" resource.labels.namespace_name=\"production\"', "
                        "'textPayload:\"timeout\" severity>=WARNING'"
                    ),
                ),
                ToolParam(
                    name="project",
                    type="string",
                    description="GCP project ID. Leave empty to use the configured default.",
                    required=False,
                ),
                ToolParam(
                    name="limit",
                    type="integer",
                    description="Maximum number of log entries to return (1-1000, default 100).",
                    required=False,
                ),
                ToolParam(
                    name="order_by",
                    type="string",
                    description="Sort order: 'timestamp desc' (newest first, default) or 'timestamp asc'.",
                    required=False,
                ),
            ],
            execute=lambda filter_expr, project="", limit=0, order_by="timestamp desc", _dp=project, _dl=log_limit: gcloud_logging_query(
                filter_expr,
                project=project or _dp,
                limit=limit or _dl,
                order_by=order_by,
            ),
        ),
        ToolDef(
            name="gcloud_monitoring_query",
            description=(
                "Query Google Cloud Monitoring time series data for a given metric type. "
                "Returns formatted time series with timestamps, values, and resource labels. "
                "Common metrics: compute.googleapis.com/instance/cpu/utilization, "
                "kubernetes.io/container/memory/used_bytes, "
                "kubernetes.io/container/cpu/core_usage_time, "
                "kubernetes.io/container/restart_count."
            ),
            parameters=[
                ToolParam(
                    name="metric_type",
                    type="string",
                    description=(
                        "Fully qualified metric type. Examples: "
                        "'compute.googleapis.com/instance/cpu/utilization', "
                        "'kubernetes.io/container/memory/used_bytes', "
                        "'kubernetes.io/container/restart_count'"
                    ),
                ),
                ToolParam(
                    name="project",
                    type="string",
                    description="GCP project ID. Leave empty to use the configured default.",
                    required=False,
                ),
                ToolParam(
                    name="interval_minutes",
                    type="integer",
                    description="Time window to query, in minutes (1-10080, default 60 = 1 hour).",
                    required=False,
                ),
                ToolParam(
                    name="aggregation",
                    type="string",
                    description="Cross-series aggregation: 'mean', 'max', 'min', 'sum', 'count'. Leave empty for raw data.",
                    required=False,
                ),
                ToolParam(
                    name="filter_str",
                    type="string",
                    description=(
                        "Additional monitoring filter to combine with metric type. "
                        "Example: 'resource.labels.namespace_name=\"production\"'"
                    ),
                    required=False,
                ),
            ],
            execute=lambda metric_type, project="", interval_minutes=0, aggregation="", filter_str="", _dp=project, _di=metrics_interval_minutes: gcloud_monitoring_query(
                metric_type,
                project=project or _dp,
                interval_minutes=interval_minutes or _di,
                aggregation=aggregation,
                filter_str=filter_str,
            ),
        ),
    ]

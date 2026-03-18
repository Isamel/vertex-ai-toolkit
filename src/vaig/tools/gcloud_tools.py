"""GCP observability tools — Cloud Logging and Cloud Monitoring queries."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from vaig.tools.base import ToolDef, ToolParam, ToolResult

if TYPE_CHECKING:
    from google.auth.credentials import Credentials

logger = logging.getLogger(__name__)


# ── Lazy SDK imports ─────────────────────────────────────────
# google-cloud-logging and google-cloud-monitoring may not be installed.
# We defer imports to execution time so the module can always be loaded.


def _get_logging_client(
    project: str | None = None,
    credentials: Credentials | None = None,
) -> tuple[Any, str | None]:
    """Create a Cloud Logging client with lazy import."""
    try:
        from google.cloud import logging as cloud_logging  # noqa: I001
    except ImportError:
        return None, "google-cloud-logging SDK is not installed. Run: pip install google-cloud-logging"

    try:
        kwargs: dict[str, Any] = {}
        if project:
            kwargs["project"] = project
        if credentials is not None:
            kwargs["credentials"] = credentials
        client = cloud_logging.Client(**kwargs)  # type: ignore[no-untyped-call]
        return client, None
    except Exception as exc:
        return None, f"Failed to create Cloud Logging client: {exc}"


def _get_monitoring_client(
    project: str | None = None,
    credentials: Credentials | None = None,
) -> tuple[Any, str | None]:
    """Create a Cloud Monitoring client with lazy import."""
    try:
        from google.cloud import monitoring_v3  # noqa: I001
    except ImportError:
        return None, "google-cloud-monitoring SDK is not installed. Run: pip install google-cloud-monitoring"

    try:
        kwargs: dict[str, Any] = {}
        if credentials is not None:
            kwargs["credentials"] = credentials
        client = monitoring_v3.MetricServiceClient(**kwargs)
        return client, None
    except Exception as exc:
        return None, f"Failed to create Cloud Monitoring client: {exc}"


# ══════════════════════════════════════════════════════════════
# DefaultGCPClientProvider — protocol-satisfying wrapper
# ══════════════════════════════════════════════════════════════


class DefaultGCPClientProvider:
    """Default implementation of ``GCPClientProvider`` protocol.

    Wraps ``_get_logging_client()`` and ``_get_monitoring_client()`` with
    **instance-level caching** keyed on ``(project, credentials_id)`` so that
    repeated calls within the same session reuse the same GCP client objects
    instead of creating new ones each time.
    """

    def __init__(self) -> None:
        self._logging_cache: dict[tuple[str | None, int | None], tuple[Any, str | None]] = {}
        self._monitoring_cache: dict[tuple[str | None, int | None], tuple[Any, str | None]] = {}

    @staticmethod
    def _cred_key(credentials: Any | None) -> int | None:
        """Return a hashable key for credentials (``id()`` or ``None``)."""
        return id(credentials) if credentials is not None else None

    def get_logging_client(
        self,
        project: str | None = None,
        credentials: Any | None = None,
    ) -> tuple[Any, str | None]:
        """Return a cached Cloud Logging client and optional error string."""
        key = (project, self._cred_key(credentials))
        if key in self._logging_cache:
            return self._logging_cache[key]
        result = _get_logging_client(project, credentials)
        client, err = result
        if err is None and client is not None:
            self._logging_cache[key] = result
        return result

    def get_monitoring_client(
        self,
        project: str | None = None,
        credentials: Any | None = None,
    ) -> tuple[Any, str | None]:
        """Return a cached Cloud Monitoring client and optional error string."""
        key = (project, self._cred_key(credentials))
        if key in self._monitoring_cache:
            return self._monitoring_cache[key]
        result = _get_monitoring_client(project, credentials)
        client, err = result
        if err is None and client is not None:
            self._monitoring_cache[key] = result
        return result

    def clear_cache(self) -> None:
        """Clear all cached GCP clients."""
        self._logging_cache.clear()
        self._monitoring_cache.clear()


# ── Helpers ──────────────────────────────────────────────────


def _handle_gcp_api_error(exc: Exception, *, service: str = "GCP API") -> str:
    """Handle GCP API errors with lazy imports.

    Classifies the exception into a user-friendly message with actionable
    guidance.  Uses lazy imports so the module works even when
    ``google.api_core`` is not installed.

    Args:
        exc: The caught exception.
        service: Human-readable service name for error messages
            (e.g. ``"Cloud Logging"``).

    Returns:
        A user-friendly error string.
    """
    try:
        from google.api_core.exceptions import (
            Forbidden,
            InvalidArgument,
            NotFound,
            PermissionDenied,
            ResourceExhausted,
        )

        if isinstance(exc, (PermissionDenied, Forbidden)):
            msg = getattr(exc, "message", str(exc))
            return (
                f"Permission denied querying {service}. "
                f"Check IAM permissions. Error: {msg}"
            )
        if isinstance(exc, ResourceExhausted):
            msg = getattr(exc, "message", str(exc))
            return (
                f"{service} API quota exceeded. "
                f"Try reducing the limit or narrowing the filter. Error: {msg}"
            )
        if isinstance(exc, (InvalidArgument, NotFound)):
            msg = getattr(exc, "message", str(exc))
            return f"Invalid request to {service}: {msg}"
    except ImportError:
        pass  # google.api_core not available — fall through

    return f"Error querying {service}: {str(exc)[:300]}"


def _format_log_entry(entry: Any) -> str:  # noqa: ANN001
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


def _format_time_series(time_series_list: Any, metric_type: str) -> str:
    """Format Cloud Monitoring time series data into a readable table.

    Handles sparse/incomplete data from the Cloud Monitoring API gracefully:
    ``point.interval``, ``point.value``, ``ts.metric``, and ``ts.resource``
    may all be ``None`` for partially-populated time series.
    """
    if not time_series_list:
        return f"No time series data found for metric: {metric_type}"

    lines: list[str] = []
    lines.append(f"Metric: {metric_type}")
    lines.append(f"Series count: {len(time_series_list)}")
    lines.append("")

    for idx, ts in enumerate(time_series_list):
        # Extract labels — guard against None metric/resource objects
        metric_obj = getattr(ts, "metric", None)
        resource_obj = getattr(ts, "resource", None)
        metric_labels = dict(getattr(metric_obj, "labels", None) or {})
        resource_labels = dict(getattr(resource_obj, "labels", None) or {})

        # Combine labels for display
        all_labels = {**resource_labels, **metric_labels}
        label_str = ", ".join(f"{k}={v}" for k, v in list(all_labels.items())[:5])

        lines.append(f"--- Series {idx + 1} [{label_str}] ---")
        lines.append(f"{'Timestamp':<24} | {'Value':>15}")
        lines.append(f"{'-' * 24}-+-{'-' * 15}")

        points = list(ts.points) if getattr(ts, "points", None) else []
        # Points come in reverse chronological order from the API
        for point in points[:50]:  # Limit to 50 points per series
            # Timestamp — guard against None interval
            interval = getattr(point, "interval", None)
            ts_val = getattr(interval, "end_time", None) if interval is not None else None
            if ts_val:
                if hasattr(ts_val, "strftime"):
                    ts_str = ts_val.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    ts_str = str(ts_val)
            else:
                ts_str = "N/A"

            # Value — guard against None value.
            # Typed value uses oneof: int64_value, double_value,
            # bool_value, string_value, distribution_value.
            # Use _pb.WhichOneof when available for accurate detection.
            value = getattr(point, "value", None)
            val_str = "N/A"
            if value is not None:
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
    credentials: Credentials | None = None,
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
    client, err = _get_logging_client(effective_project, credentials=credentials)
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
        return ToolResult(
            output=_handle_gcp_api_error(exc, service="Cloud Logging"),
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
    resource_labels: dict[str, str] | None = None,
    credentials: Credentials | None = None,
) -> ToolResult:
    """Query Cloud Monitoring time series data for a given metric.

    Returns formatted time series with timestamps, values, and labels.

    Args:
        metric_type: Fully qualified metric type (e.g. ``istio.io/service/server/request_count``).
        project: GCP project ID.  Falls back to ADC default if empty.
        interval_minutes: Time window in minutes (1–10080).
        aggregation: Cross-series aggregation (mean/max/min/sum/count).
        filter_str: Raw additional monitoring filter string to AND with the metric type.
        resource_labels: Dict of resource label key→value pairs to filter on.
            Converted to ``resource.labels.<key> = "<value>"`` filter clauses.
            Example: ``{"namespace_name": "production", "cluster_name": "prod-1"}``.
        credentials: Optional GCP credentials for the Monitoring client.
    """
    logger.debug(
        "gcloud_monitoring_query: metric=%s project=%s interval=%d resource_labels=%s",
        metric_type, project, interval_minutes, resource_labels,
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

    client, err = _get_monitoring_client(credentials=credentials)
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
        from google.protobuf import duration_pb2, timestamp_pb2
    except ImportError:
        return ToolResult(
            output="google-cloud-monitoring SDK is not installed. Run: pip install google-cloud-monitoring",
            error=True,
        )

    now = datetime.now(tz=UTC)
    seconds_ago = interval_minutes * 60

    end_ts = timestamp_pb2.Timestamp()
    end_ts.FromDatetime(now)
    start_ts = timestamp_pb2.Timestamp()
    start_ts.FromDatetime(datetime.fromtimestamp(now.timestamp() - seconds_ago, tz=UTC))

    interval = monitoring_types.TimeInterval(
        end_time=end_ts,
        start_time=start_ts,
    )

    # Build filter — must include metric.type
    monitoring_filter = f'metric.type = "{metric_type}"'
    if filter_str.strip():
        monitoring_filter += f" AND {filter_str}"

    # Append resource.labels filters from dict
    if resource_labels:
        import re

        for label_key, label_value in resource_labels.items():
            # Validate label_key: only alphanumeric and underscores
            if not re.fullmatch(r"[A-Za-z_]\w*", label_key):
                return ToolResult(
                    output=f"Invalid resource label key: '{label_key}'. Only alphanumeric characters and underscores are allowed.",
                    error=True,
                )
            # Escape backslashes and double quotes in label_value
            escaped_value = str(label_value).replace("\\", "\\\\").replace('"', '\\"')
            monitoring_filter += f' AND resource.labels.{label_key} = "{escaped_value}"'

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
        return ToolResult(
            output=_handle_gcp_api_error(exc, service="Cloud Monitoring"),
            error=True,
        )

    formatted = _format_time_series(time_series_list, metric_type)
    return ToolResult(output=formatted)


# ── Task 3.3 — Tool factory ─────────────────────────────────


def create_gcloud_tools(
    project: str = "",
    log_limit: int = 100,
    metrics_interval_minutes: int = 60,
    credentials: Credentials | None = None,
) -> list[ToolDef]:
    """Create all GCP observability tool definitions.

    Args:
        project: GCP project ID (from GKEConfig). Falls back to ADC default if empty.
        log_limit: Default log entry limit (from GKEConfig).
        metrics_interval_minutes: Default monitoring interval (from GKEConfig).
        credentials: Optional GCP credentials for Cloud Logging / Monitoring clients.
            When ``None``, clients use Application Default Credentials.

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
            execute=lambda filter_expr, project="", limit=0, order_by="timestamp desc", _dp=project, _dl=log_limit, _dc=credentials: gcloud_logging_query(
                filter_expr,
                project=project or _dp,
                limit=limit or _dl,
                order_by=order_by,
                credentials=_dc,
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
                "kubernetes.io/container/restart_count, "
                "istio.io/service/server/request_count. "
                "IMPORTANT: To filter by resource labels (namespace, cluster, pod, etc.), "
                "use the 'resource_labels' dict parameter — do NOT put resource.labels.* "
                "expressions into 'filter_str'. "
                "Example: resource_labels={\"namespace_name\": \"production\"}."
            ),
            parameters=[
                ToolParam(
                    name="metric_type",
                    type="string",
                    description=(
                        "Fully qualified metric type. Examples: "
                        "'compute.googleapis.com/instance/cpu/utilization', "
                        "'kubernetes.io/container/memory/used_bytes', "
                        "'kubernetes.io/container/restart_count', "
                        "'istio.io/service/server/request_count'"
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
                        "Additional metric-level monitoring filter to AND with the metric type. "
                        "Do NOT use this for resource labels — use the 'resource_labels' parameter instead. "
                        "Example: 'metric.labels.response_code = \"500\"'"
                    ),
                    required=False,
                ),
                ToolParam(
                    name="resource_labels",
                    type="object",
                    description=(
                        "Dict of resource label key-value pairs to filter on. "
                        "This is the PREFERRED way to filter by resource dimensions. "
                        "Each entry becomes a 'resource.labels.<key> = \"<value>\"' filter clause. "
                        "Common keys: namespace_name, cluster_name, container_name, pod_name, "
                        "destination_workload_namespace, location. "
                        "Example: {\"namespace_name\": \"production\", \"cluster_name\": \"prod-1\"}"
                    ),
                    required=False,
                ),
            ],
            execute=lambda metric_type, project="", interval_minutes=0, aggregation="", filter_str="", resource_labels=None, _dp=project, _di=metrics_interval_minutes, _dc=credentials: gcloud_monitoring_query(
                metric_type,
                project=project or _dp,
                interval_minutes=interval_minutes or _di,
                aggregation=aggregation,
                filter_str=filter_str,
                resource_labels=resource_labels,
                credentials=_dc,
            ),
        ),
    ]

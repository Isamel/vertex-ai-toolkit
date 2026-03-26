"""GKE Cloud Monitoring metrics — historical CPU and memory pod metrics.

Wraps the Google Cloud Monitoring ``list_time_series()`` API to fetch
historical pod-level CPU and memory metrics for pods matching a namespace
and pod-name prefix. Designed for the service-health pipeline: produces a
concise LLM-friendly summary table with trend indicators.

Also exposes :func:`get_workload_usage_metrics` which returns structured
numeric metrics (avg CPU vCores, avg memory GiB) keyed by workload_name,
for consumption by the cost estimation pipeline.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from vaig.tools.base import ToolResult

if TYPE_CHECKING:
    from vaig.core.config import GKEConfig

logger = logging.getLogger(__name__)

# ── Lazy import guards ────────────────────────────────────────

_MONITORING_AVAILABLE = True
try:
    from google.api_core import exceptions as gcp_exceptions  # noqa: WPS433
    from google.api_core.exceptions import PermissionDenied  # noqa: WPS433
    from google.cloud import monitoring_v3  # noqa: WPS433
    from google.cloud.monitoring_v3.services.metric_service import (  # noqa: WPS433
        MetricServiceClient,
    )
except ImportError:
    _MONITORING_AVAILABLE = False
    gcp_exceptions = None  # type: ignore[assignment]
    PermissionDenied = None  # type: ignore[assignment,misc]
    monitoring_v3 = None  # type: ignore[assignment]
    MetricServiceClient = None  # type: ignore[assignment,misc]


def _monitoring_unavailable() -> ToolResult:
    return ToolResult(
        output=(
            "Google Cloud Monitoring client library is not available. "
            "Install it with: pip install google-cloud-monitoring"
        ),
        error=True,
    )


# ── Structured metrics dataclasses ───────────────────────────


@dataclass
class ContainerUsageMetrics:
    """Raw usage metrics for a single container from Cloud Monitoring.

    Attributes:
        container_name: Kubernetes container name.
        avg_cpu_cores: Average CPU usage in vCPU cores (not millicores).
        avg_memory_gib: Average memory usage in GiB.
    """

    container_name: str
    avg_cpu_cores: float
    avg_memory_gib: float


@dataclass
class WorkloadUsageMetrics:
    """Aggregated usage metrics for all containers in a workload.

    Attributes:
        namespace: Kubernetes namespace.
        workload_name: Workload identifier (Deployment/StatefulSet name).
        containers: Per-container metrics, keyed by container_name.
    """

    namespace: str
    workload_name: str
    containers: dict[str, ContainerUsageMetrics] = field(default_factory=dict)


# ── Metric type constants ─────────────────────────────────────

_CPU_METRIC = "kubernetes.io/container/cpu/core_usage_time"
_MEMORY_METRIC = "kubernetes.io/container/memory/used_bytes"


# ── Core helpers ──────────────────────────────────────────────


def _build_metric_filter(
    metric_type: str,
    cluster_name: str,
    namespace: str,
    pod_name_prefix: str,
) -> str:
    """Build a Cloud Monitoring filter string for a GKE container metric.

    Args:
        metric_type: Full metric type string (e.g. ``kubernetes.io/container/cpu/core_usage_time``).
        cluster_name: GKE cluster name.
        namespace: Kubernetes namespace.
        pod_name_prefix: Pod name prefix for regex matching.

    Returns:
        A Cloud Monitoring filter string suitable for ``list_time_series()``.
    """
    # Escape any regex metacharacters in prefix so they match literally
    escaped_prefix = re.escape(pod_name_prefix)
    return (
        f'metric.type = "{metric_type}"'
        f' AND resource.type = "k8s_container"'
        f' AND resource.labels.cluster_name = "{cluster_name}"'
        f' AND resource.labels.namespace_name = "{namespace}"'
        f' AND resource.labels.pod_name = monitoring.regex.full_match("^{escaped_prefix}.*")'
    )


def _query_time_series(
    client: Any,
    project_id: str,
    metric_filter: str,
    metric_type: str,
    window_minutes: int,
) -> list[Any]:
    """Query Cloud Monitoring time series for the given filter and window.

    Args:
        client: A ``MetricServiceClient`` instance.
        project_id: GCP project ID.
        metric_filter: The Cloud Monitoring filter string.
        metric_type: One of ``_CPU_METRIC`` or ``_MEMORY_METRIC``.
        window_minutes: How many minutes back from now to query.

    Returns:
        A list of ``TimeSeries`` objects (may be empty).
    """
    now = datetime.now(tz=UTC)
    start = now - timedelta(minutes=window_minutes)

    interval = monitoring_v3.TimeInterval(
        {
            "end_time": {"seconds": int(now.timestamp())},
            "start_time": {"seconds": int(start.timestamp())},
        }
    )

    # Choose aligner per metric type:
    # CPU is a cumulative counter — use ALIGN_RATE to get cores/s then scale.
    # Memory is a gauge — use ALIGN_MEAN.
    is_cpu = metric_type == _CPU_METRIC
    aligner = (
        monitoring_v3.Aggregation.Aligner.ALIGN_RATE
        if is_cpu
        else monitoring_v3.Aggregation.Aligner.ALIGN_MEAN
    )

    aggregation = monitoring_v3.Aggregation(
        {
            "alignment_period": {"seconds": 60},
            "per_series_aligner": aligner,
            "cross_series_reducer": monitoring_v3.Aggregation.Reducer.REDUCE_SUM,
            "group_by_fields": [
                "resource.labels.pod_name",
                "resource.labels.namespace_name",
            ],
        }
    )

    request = monitoring_v3.ListTimeSeriesRequest(
        name=f"projects/{project_id}",
        filter=metric_filter,
        interval=interval,
        view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
        aggregation=aggregation,
    )

    results = list(client.list_time_series(request=request))
    return results


def _extract_points_values(time_series: Any) -> list[float]:
    """Extract float values from a TimeSeries object's data points.

    Handles both double and int64 typed point values.

    Cloud Monitoring returns points in reverse chronological order (newest
    first). This function reverses them so callers receive values in
    chronological order (oldest first), which is required for correct
    trend calculation.
    """
    values: list[float] = []
    for point in time_series.points:
        val = point.value
        if hasattr(val, "double_value"):
            values.append(val.double_value)
        elif hasattr(val, "int64_value"):
            values.append(float(val.int64_value))
    # Reverse: API gives newest-first; callers expect oldest-first
    return list(reversed(values))


def _calculate_trend(values: list[float]) -> str:
    """Calculate trend direction from a list of float data points.

    Compares the last 25% of values to the first 25%. Returns:
    - ``↑`` if the tail average is > 10% higher than the head average
    - ``↓`` if the tail average is > 10% lower than the head average
    - ``→`` otherwise (stable)

    Args:
        values: Chronologically ordered float data points.

    Returns:
        One of ``"↑"``, ``"↓"``, or ``"→"``.
    """
    if len(values) < 4:
        return "→"

    quarter = max(1, len(values) // 4)
    head_avg = sum(values[:quarter]) / quarter
    tail_avg = sum(values[-quarter:]) / quarter

    if head_avg == 0:
        return "→"

    change_pct = round((tail_avg - head_avg) / head_avg, 10)
    if change_pct > 0.10:
        return "↑"
    if change_pct < -0.10:
        return "↓"
    return "→"


def _format_metric_value(value: float, metric_type: str) -> str:
    """Format a float metric value for human-readable display.

    CPU (rate): expressed in millicores (m), e.g. 125m.
    Memory (bytes): expressed in MiB or GiB.
    """
    if metric_type == _CPU_METRIC:
        # ALIGN_RATE gives cores/s — multiply by 1000 for millicores
        millicores = value * 1000
        return f"{millicores:.0f}m"
    else:
        # Memory in bytes → MiB or GiB
        mib = value / (1024 * 1024)
        if mib >= 1024:
            return f"{mib / 1024:.2f}Gi"
        return f"{mib:.1f}Mi"


def _format_metrics_response(
    time_series_list: list[Any],
    metric_type: str,
    namespace: str,
    pod_name_prefix: str,
    window_minutes: int,
) -> str:
    """Format a list of TimeSeries objects into an LLM-friendly Markdown table.

    Args:
        time_series_list: List of TimeSeries from Cloud Monitoring.
        metric_type: ``_CPU_METRIC`` or ``_MEMORY_METRIC``.
        namespace: Kubernetes namespace queried.
        pod_name_prefix: Pod name prefix used for filtering.
        window_minutes: Query window in minutes.

    Returns:
        A Markdown-formatted string with a summary table and totals line.
    """
    metric_label = "CPU" if metric_type == _CPU_METRIC else "Memory"
    unit_label = "(millicores)" if metric_type == _CPU_METRIC else "(MiB/GiB)"

    if not time_series_list:
        return (
            f"**{metric_label} Metrics** — namespace: `{namespace}`, "
            f"prefix: `{pod_name_prefix}`, window: {window_minutes}m\n\n"
            "No data returned. Possible reasons:\n"
            "- No pods match the specified prefix\n"
            "- Cloud Monitoring agent not deployed\n"
            "- Insufficient IAM permissions (roles/monitoring.viewer required)\n"
        )

    rows: list[dict[str, str]] = []

    for ts in time_series_list:
        # Extract pod name from resource labels
        labels = ts.resource.labels if hasattr(ts, "resource") else {}
        pod_name = labels.get("pod_name", "<unknown>") if hasattr(labels, "get") else "<unknown>"

        raw_values = _extract_points_values(ts)
        if not raw_values:
            continue

        avg_val = sum(raw_values) / len(raw_values)
        max_val = max(raw_values)
        latest_val = raw_values[-1] if raw_values else 0.0
        trend = _calculate_trend(raw_values)

        rows.append(
            {
                "pod": pod_name,
                "avg": _format_metric_value(avg_val, metric_type),
                "max": _format_metric_value(max_val, metric_type),
                "latest": _format_metric_value(latest_val, metric_type),
                "trend": trend,
            }
        )

    if not rows:
        return (
            f"**{metric_label} Metrics** — namespace: `{namespace}`, "
            f"prefix: `{pod_name_prefix}`, window: {window_minutes}m\n\n"
            "Data returned but no numeric values could be extracted from time series points.\n"
        )

    # Truncate to top 20 pods to keep output within ~2000 chars
    _MAX_PODS = 20
    total_pods = len(rows)
    truncated = total_pods > _MAX_PODS
    display_rows = rows[:_MAX_PODS]

    # Build Markdown table
    lines: list[str] = [
        f"**{metric_label} Metrics** {unit_label} — namespace: `{namespace}`, "
        f"prefix: `{pod_name_prefix}`, window: {window_minutes}m\n",
        f"| {'Pod':<50} | {'Avg':>10} | {'Max':>10} | {'Latest':>10} | Trend |",
        f"|{'-'*52}|{'-'*12}|{'-'*12}|{'-'*12}|-------|",
    ]

    for row in display_rows:
        lines.append(
            f"| {row['pod']:<50} | {row['avg']:>10} | {row['max']:>10} | {row['latest']:>10} | {row['trend']:^5} |"
        )

    if truncated:
        lines.append(f"| _... and {total_pods - _MAX_PODS} more pods (truncated)_ |")

    lines.append(
        f"\n_Summary: {total_pods} pod(s) matched prefix `{pod_name_prefix}` "
        f"in namespace `{namespace}` over the last {window_minutes} minutes._"
    )

    return "\n".join(lines)


# ── Public tool function ──────────────────────────────────────


def get_pod_metrics(
    namespace: str,
    pod_name_prefix: str,
    *,
    gke_config: GKEConfig,
    window_minutes: int = 60,
    metric_type: str = "all",
) -> ToolResult:
    """Fetch historical CPU and/or memory metrics from Cloud Monitoring for GKE pods.

    Queries ``kubernetes.io/container/cpu/core_usage_time`` and/or
    ``kubernetes.io/container/memory/used_bytes`` for pods matching
    ``pod_name_prefix`` in ``namespace``.

    Args:
        namespace: Kubernetes namespace to query.
        pod_name_prefix: Pod name prefix for matching (e.g. ``"frontend-"``
            matches ``frontend-abc-123``).
        gke_config: GKE cluster configuration (provides project_id, cluster_name).
        window_minutes: Query window in minutes (default: 60).
        metric_type: Which metrics to fetch: ``"cpu"``, ``"memory"``, or
            ``"all"`` (default: ``"all"``).

    Returns:
        A :class:`~vaig.tools.base.ToolResult` with a Markdown-formatted
        summary table.
    """
    if not _MONITORING_AVAILABLE:
        return _monitoring_unavailable()

    # Validate metric_type
    valid_types = {"cpu", "memory", "all"}
    if metric_type not in valid_types:
        return ToolResult(
            output=(
                f"Invalid metric_type '{metric_type}'. "
                f"Must be one of: {', '.join(sorted(valid_types))}"
            ),
            error=True,
        )

    # Validate window_minutes: must be a positive integer, max 1440 (24h)
    if window_minutes <= 0 or window_minutes > 1440:
        return ToolResult(
            output=(
                f"Invalid window_minutes '{window_minutes}'. "
                "Must be a positive integer between 1 and 1440 (24 hours)."
            ),
            error=True,
        )

    project_id = gke_config.project_id
    cluster_name = gke_config.cluster_name

    try:
        client = MetricServiceClient()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to create MetricServiceClient: %s", exc)
        return ToolResult(
            output=f"Failed to initialize Cloud Monitoring client: {exc}",
            error=True,
        )

    sections: list[str] = [
        f"## Pod Metrics — {namespace}/{pod_name_prefix}* (last {window_minutes}m)\n"
    ]

    fetch_cpu = metric_type in {"cpu", "all"}
    fetch_memory = metric_type in {"memory", "all"}

    for do_fetch, mtype in [(fetch_cpu, _CPU_METRIC), (fetch_memory, _MEMORY_METRIC)]:
        if not do_fetch:
            continue

        metric_filter = _build_metric_filter(
            mtype, cluster_name, namespace, pod_name_prefix
        )

        try:
            ts_list = _query_time_series(
                client=client,
                project_id=project_id,
                metric_filter=metric_filter,
                metric_type=mtype,
                window_minutes=window_minutes,
            )
        except Exception as exc:  # noqa: BLE001
            # Check for PermissionDenied — use the imported class when available,
            # fall back to string/code check when the library is unavailable.
            is_permission_denied = (
                (PermissionDenied is not None and isinstance(exc, PermissionDenied))
                or "403" in str(exc)
                or type(exc).__name__ == "PermissionDenied"
            )
            if is_permission_denied:
                sections.append(
                    f"**{'CPU' if mtype == _CPU_METRIC else 'Memory'} Metrics** — "
                    f"Access denied (HTTP 403). "
                    f"Grant the service account ``roles/monitoring.viewer`` on project "
                    f"``{project_id}``.\n"
                )
            else:
                logger.warning(
                    "Cloud Monitoring query failed for %s: %s", mtype, exc
                )
                sections.append(
                    f"**{'CPU' if mtype == _CPU_METRIC else 'Memory'} Metrics** — "
                    f"Query error: {exc}\n"
                )
            continue

        section = _format_metrics_response(
            ts_list,
            metric_type=mtype,
            namespace=namespace,
            pod_name_prefix=pod_name_prefix,
            window_minutes=window_minutes,
        )
        sections.append(section)

    return ToolResult(output="\n\n".join(sections))


# ── Structured metrics for cost estimation ───────────────────


def _build_metric_filter_with_container(
    metric_type: str,
    cluster_name: str,
    namespace: str,
) -> str:
    """Build a Cloud Monitoring filter for container-level metrics in a namespace.

    Unlike :func:`_build_metric_filter`, this does NOT filter by pod prefix —
    it fetches all containers in the namespace so the cost estimation pipeline
    can match metrics by container_name after the fact.

    Args:
        metric_type: Full metric type string.
        cluster_name: GKE cluster name.
        namespace: Kubernetes namespace to query.

    Returns:
        A Cloud Monitoring filter string suitable for ``list_time_series()``.
    """
    return (
        f'metric.type = "{metric_type}"'
        f' AND resource.type = "k8s_container"'
        f' AND resource.labels.cluster_name = "{cluster_name}"'
        f' AND resource.labels.namespace_name = "{namespace}"'
    )


def _query_time_series_with_container(
    client: Any,
    project_id: str,
    metric_filter: str,
    metric_type: str,
    window_minutes: int,
) -> list[Any]:
    """Query Cloud Monitoring time series grouped by pod_name + container_name.

    This variant of :func:`_query_time_series` adds ``metric.labels.container_name``
    to ``group_by_fields`` so each resulting time series represents a single
    (pod, container) pair — enabling per-container cost breakdown.

    Args:
        client: A ``MetricServiceClient`` instance.
        project_id: GCP project ID.
        metric_filter: The Cloud Monitoring filter string.
        metric_type: One of ``_CPU_METRIC`` or ``_MEMORY_METRIC``.
        window_minutes: How many minutes back from now to query.

    Returns:
        A list of ``TimeSeries`` objects (may be empty).
    """
    now = datetime.now(tz=UTC)
    start = now - timedelta(minutes=window_minutes)

    interval = monitoring_v3.TimeInterval(
        {
            "end_time": {"seconds": int(now.timestamp())},
            "start_time": {"seconds": int(start.timestamp())},
        }
    )

    is_cpu = metric_type == _CPU_METRIC
    aligner = (
        monitoring_v3.Aggregation.Aligner.ALIGN_RATE
        if is_cpu
        else monitoring_v3.Aggregation.Aligner.ALIGN_MEAN
    )

    aggregation = monitoring_v3.Aggregation(
        {
            "alignment_period": {"seconds": 60},
            "per_series_aligner": aligner,
            "cross_series_reducer": monitoring_v3.Aggregation.Reducer.REDUCE_MEAN,
            "group_by_fields": [
                "resource.labels.pod_name",
                "resource.labels.namespace_name",
                "metric.labels.container_name",
            ],
        }
    )

    request = monitoring_v3.ListTimeSeriesRequest(
        name=f"projects/{project_id}",
        filter=metric_filter,
        interval=interval,
        view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
        aggregation=aggregation,
    )

    results = list(client.list_time_series(request=request))
    return results


def get_workload_usage_metrics(
    namespace: str,
    workload_pod_names: dict[str, list[str]],
    *,
    gke_config: GKEConfig,
    window_minutes: int = 60,
) -> dict[str, WorkloadUsageMetrics]:
    """Fetch avg CPU and memory usage per container for specified workloads.

    Queries Cloud Monitoring for container-level metrics grouped by pod_name
    and container_name. Matches results back to workloads using the pod names
    provided in ``workload_pod_names``.

    Used by the cost estimation pipeline to populate usage/waste fields that
    v1 always left as ``None``.

    Args:
        namespace: Kubernetes namespace to query.
        workload_pod_names: Mapping of workload_name → list of pod names
            belonging to that workload. Pod names are used to match monitoring
            results back to their workload.
        gke_config: GKE cluster configuration (provides project_id, cluster_name).
        window_minutes: Query window in minutes (default: 60).

    Returns:
        A dict mapping ``workload_name`` → :class:`WorkloadUsageMetrics`.
        Returns an empty dict if monitoring is unavailable or the query fails.
        Workloads without monitoring data are simply absent from the result.
    """
    if not _MONITORING_AVAILABLE:
        logger.debug("Cloud Monitoring not available — skipping usage metrics fetch")
        return {}

    if not workload_pod_names:
        return {}

    project_id = gke_config.project_id
    cluster_name = gke_config.cluster_name

    try:
        client = MetricServiceClient()
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_workload_usage_metrics: failed to create client: %s", exc)
        return {}

    # Build reverse map: pod_name → workload_name for fast lookup
    pod_to_workload: dict[str, str] = {}
    for wl_name, pod_names in workload_pod_names.items():
        for pod_name in pod_names:
            pod_to_workload[pod_name] = wl_name

    # ── Query CPU ───────────────────────────────────────────
    cpu_filter = _build_metric_filter_with_container(_CPU_METRIC, cluster_name, namespace)
    cpu_by_pod_container: dict[tuple[str, str], float] = {}  # (pod, container) → avg cores

    try:
        cpu_ts_list = _query_time_series_with_container(
            client=client,
            project_id=project_id,
            metric_filter=cpu_filter,
            metric_type=_CPU_METRIC,
            window_minutes=window_minutes,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_workload_usage_metrics: CPU query failed for ns=%s: %s", namespace, exc)
        cpu_ts_list = []

    for ts in cpu_ts_list:
        resource_labels = ts.resource.labels if hasattr(ts, "resource") else {}
        metric_labels = ts.metric.labels if hasattr(ts, "metric") else {}
        pod_name = resource_labels.get("pod_name", "") if hasattr(resource_labels, "get") else ""
        container_name = metric_labels.get("container_name", "") if hasattr(metric_labels, "get") else ""

        if not pod_name or not container_name:
            continue

        values = _extract_points_values(ts)
        if not values:
            continue

        # ALIGN_RATE on a cumulative counter (core_usage_time in core-seconds) yields
        # the instantaneous rate in cores — so each aligned point is already in vCPU cores.
        avg_cores = sum(values) / len(values)
        cpu_by_pod_container[(pod_name, container_name)] = avg_cores

    # ── Query Memory ─────────────────────────────────────────
    mem_filter = _build_metric_filter_with_container(_MEMORY_METRIC, cluster_name, namespace)
    mem_by_pod_container: dict[tuple[str, str], float] = {}  # (pod, container) → avg GiB

    try:
        mem_ts_list = _query_time_series_with_container(
            client=client,
            project_id=project_id,
            metric_filter=mem_filter,
            metric_type=_MEMORY_METRIC,
            window_minutes=window_minutes,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_workload_usage_metrics: memory query failed for ns=%s: %s", namespace, exc)
        mem_ts_list = []

    for ts in mem_ts_list:
        resource_labels = ts.resource.labels if hasattr(ts, "resource") else {}
        metric_labels = ts.metric.labels if hasattr(ts, "metric") else {}
        pod_name = resource_labels.get("pod_name", "") if hasattr(resource_labels, "get") else ""
        container_name = metric_labels.get("container_name", "") if hasattr(metric_labels, "get") else ""

        if not pod_name or not container_name:
            continue

        values = _extract_points_values(ts)
        if not values:
            continue

        avg_bytes = sum(values) / len(values)
        avg_gib = avg_bytes / (1024.0 ** 3)
        mem_by_pod_container[(pod_name, container_name)] = avg_gib

    # ── Aggregate per workload ────────────────────────────────
    # Collect all (pod, container) keys observed
    all_keys: set[tuple[str, str]] = set(cpu_by_pod_container.keys()) | set(mem_by_pod_container.keys())

    # Group by workload, summing across pods within the same workload+container.
    # Requests are already summed across pods (total capacity), so usage must
    # also be summed across pods to compare like-for-like and avoid inflating waste.
    # Structure: workload_name → container_name → list of (cpu, memory) per pod
    wl_container_cpu: dict[str, dict[str, list[float]]] = {}
    wl_container_mem: dict[str, dict[str, list[float]]] = {}

    for pod_name, container_name in all_keys:
        workload_name = pod_to_workload.get(pod_name)
        if workload_name is None:
            continue

        wl_container_cpu.setdefault(workload_name, {}).setdefault(container_name, [])
        wl_container_mem.setdefault(workload_name, {}).setdefault(container_name, [])

        if (pod_name, container_name) in cpu_by_pod_container:
            wl_container_cpu[workload_name][container_name].append(
                cpu_by_pod_container[(pod_name, container_name)]
            )
        if (pod_name, container_name) in mem_by_pod_container:
            wl_container_mem[workload_name][container_name].append(
                mem_by_pod_container[(pod_name, container_name)]
            )

    result: dict[str, WorkloadUsageMetrics] = {}

    for workload_name in wl_container_cpu:
        containers: dict[str, ContainerUsageMetrics] = {}

        all_containers = set(wl_container_cpu.get(workload_name, {}).keys()) | set(
            wl_container_mem.get(workload_name, {}).keys()
        )

        for c_name in all_containers:
            cpu_values = wl_container_cpu.get(workload_name, {}).get(c_name, [])
            mem_values = wl_container_mem.get(workload_name, {}).get(c_name, [])

            if not cpu_values or not mem_values:
                continue

            # Sum per-pod values so total usage matches summed requests across replicas
            total_cpu = sum(cpu_values)
            total_mem = sum(mem_values)

            containers[c_name] = ContainerUsageMetrics(
                container_name=c_name,
                avg_cpu_cores=total_cpu,
                avg_memory_gib=total_mem,
            )

        result[workload_name] = WorkloadUsageMetrics(
            namespace=namespace,
            workload_name=workload_name,
            containers=containers,
        )

    return result

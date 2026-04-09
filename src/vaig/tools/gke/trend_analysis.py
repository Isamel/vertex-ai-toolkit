"""Anomaly trend detection for GKE workloads.

Compares current metric values against historical Cloud Monitoring
baselines to detect slowly-degrading services (memory creep, rising
restarts, CPU growth).  Produces a :class:`TrendAnalysis` object
attached to ``ReportMetadata.trends`` via post-pipeline enrichment
(same pattern as :mod:`cost_estimation`).

When Cloud Monitoring is unavailable or the feature is disabled,
``fetch_anomaly_trends()`` returns ``None`` and the health report
proceeds without trend data.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vaig.core.config import GKEConfig, TrendConfig
    from vaig.skills.service_health.schema import MetricTrend, TrendAnalysis

logger = logging.getLogger(__name__)

# ── Lazy import guard: Cloud Monitoring ───────────────────────

_MONITORING_AVAILABLE = True
try:
    from google.api_core import exceptions as gcp_exceptions  # noqa: WPS433
    from google.cloud import monitoring_v3  # noqa: WPS433
    from google.cloud.monitoring_v3.services.metric_service import (  # noqa: WPS433
        MetricServiceClient,
    )
except ImportError:
    _MONITORING_AVAILABLE = False
    gcp_exceptions = None  # type: ignore[assignment]
    monitoring_v3 = None  # type: ignore[assignment]
    MetricServiceClient = None  # type: ignore[assignment,misc]

# ── Metric type constants ─────────────────────────────────────

_CPU_METRIC = "kubernetes.io/container/cpu/core_usage_time"
_MEMORY_METRIC = "kubernetes.io/container/memory/used_bytes"
_RESTART_METRIC = "kubernetes.io/container/restart_count"

_METRIC_NAMES: dict[str, str] = {
    _CPU_METRIC: "cpu_usage",
    _MEMORY_METRIC: "memory_usage",
    _RESTART_METRIC: "restart_count",
}

# ── Internal helpers ──────────────────────────────────────────


def _query_baseline(
    client: Any,
    project_id: str,
    cluster: str,
    namespace: str,
    metric_type: str,
    window_days: int,
) -> float | None:
    """Query Cloud Monitoring for a historical baseline average.

    Uses 3600s (hourly) alignment for multi-day windows to keep
    response sizes manageable.

    Args:
        client: A ``MetricServiceClient`` instance.
        project_id: GCP project ID.
        cluster: GKE cluster name.
        namespace: Kubernetes namespace.
        metric_type: Full Cloud Monitoring metric type string.
        window_days: Number of days for the baseline window.

    Returns:
        Average value across the window, or ``None`` if no data.
    """
    now = datetime.now(tz=UTC)
    end = now - timedelta(days=1)  # baseline ends 24h ago
    start = end - timedelta(days=window_days)

    interval = monitoring_v3.TimeInterval(
        {
            "end_time": {"seconds": int(end.timestamp())},
            "start_time": {"seconds": int(start.timestamp())},
        }
    )

    is_cpu = metric_type == _CPU_METRIC
    is_restart = metric_type == _RESTART_METRIC

    if is_cpu:
        aligner = monitoring_v3.Aggregation.Aligner.ALIGN_RATE
    elif is_restart:
        aligner = monitoring_v3.Aggregation.Aligner.ALIGN_DELTA
    else:
        aligner = monitoring_v3.Aggregation.Aligner.ALIGN_MEAN

    aggregation = monitoring_v3.Aggregation(
        {
            "alignment_period": {"seconds": 3600},
            "per_series_aligner": aligner,
            "cross_series_reducer": monitoring_v3.Aggregation.Reducer.REDUCE_SUM,
            "group_by_fields": [
                "resource.labels.namespace_name",
                "metadata.system_labels.top_level_controller_name",
            ],
        }
    )

    metric_filter = (
        f'metric.type = "{metric_type}"'
        f' AND resource.type = "k8s_container"'
        f' AND resource.labels.cluster_name = "{cluster}"'
        f' AND resource.labels.namespace_name = "{namespace}"'
    )

    request = monitoring_v3.ListTimeSeriesRequest(
        name=f"projects/{project_id}",
        filter=metric_filter,
        interval=interval,
        view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
        aggregation=aggregation,
    )

    results = list(client.list_time_series(request=request))
    if not results:
        return None

    all_values: list[float] = []
    for ts in results:
        for point in ts.points:
            val = point.value.double_value or point.value.int64_value
            if val is not None:
                all_values.append(float(val))

    return sum(all_values) / len(all_values) if all_values else None


def _query_current(
    client: Any,
    project_id: str,
    cluster: str,
    namespace: str,
    metric_type: str,
    window_hours: int = 24,
) -> float | None:
    """Query Cloud Monitoring for the current observation window average.

    Uses 300s (5-minute) alignment for granularity in the recent window.

    Args:
        client: A ``MetricServiceClient`` instance.
        project_id: GCP project ID.
        cluster: GKE cluster name.
        namespace: Kubernetes namespace.
        metric_type: Full Cloud Monitoring metric type string.
        window_hours: Current window size in hours (default 24).

    Returns:
        Average value across the window, or ``None`` if no data.
    """
    now = datetime.now(tz=UTC)
    start = now - timedelta(hours=window_hours)

    interval = monitoring_v3.TimeInterval(
        {
            "end_time": {"seconds": int(now.timestamp())},
            "start_time": {"seconds": int(start.timestamp())},
        }
    )

    is_cpu = metric_type == _CPU_METRIC
    is_restart = metric_type == _RESTART_METRIC

    if is_cpu:
        aligner = monitoring_v3.Aggregation.Aligner.ALIGN_RATE
    elif is_restart:
        aligner = monitoring_v3.Aggregation.Aligner.ALIGN_DELTA
    else:
        aligner = monitoring_v3.Aggregation.Aligner.ALIGN_MEAN

    aggregation = monitoring_v3.Aggregation(
        {
            "alignment_period": {"seconds": 300},
            "per_series_aligner": aligner,
            "cross_series_reducer": monitoring_v3.Aggregation.Reducer.REDUCE_SUM,
            "group_by_fields": [
                "resource.labels.namespace_name",
                "metadata.system_labels.top_level_controller_name",
            ],
        }
    )

    metric_filter = (
        f'metric.type = "{metric_type}"'
        f' AND resource.type = "k8s_container"'
        f' AND resource.labels.cluster_name = "{cluster}"'
        f' AND resource.labels.namespace_name = "{namespace}"'
    )

    request = monitoring_v3.ListTimeSeriesRequest(
        name=f"projects/{project_id}",
        filter=metric_filter,
        interval=interval,
        view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
        aggregation=aggregation,
    )

    results = list(client.list_time_series(request=request))
    if not results:
        return None

    all_values: list[float] = []
    for ts in results:
        for point in ts.points:
            val = point.value.double_value or point.value.int64_value
            if val is not None:
                all_values.append(float(val))

    return sum(all_values) / len(all_values) if all_values else None


def _project_days_to_threshold(
    current: float,
    rate_per_day: float,
    limit: float,
) -> float | None:
    """Project days until a resource limit is reached via linear extrapolation.

    Args:
        current: Current metric value.
        rate_per_day: Daily rate of increase (absolute units).
        limit: Resource limit value.

    Returns:
        Projected days to threshold, or ``None`` if rate is zero/negative
        or current already exceeds limit.
    """
    if rate_per_day <= 0 or current >= limit:
        return None
    remaining = limit - current
    return remaining / rate_per_day


def _compute_trend(
    metric: str,
    service_name: str,
    namespace: str,
    current_avg: float | None,
    baseline_avg: float | None,
    config: TrendConfig,
    window_days: int,
) -> MetricTrend | None:
    """Compute a single MetricTrend from current and baseline averages.

    Handles the zero-baseline edge case by setting direction to ``"new"``
    and ``rate_of_change_percent`` to ``None``.

    Args:
        metric: Full Cloud Monitoring metric type string.
        service_name: Workload identifier (from controller_name label).
        namespace: Kubernetes namespace.
        current_avg: Current window average.
        baseline_avg: Baseline window average.
        config: TrendConfig with severity thresholds.
        window_days: Baseline window size in days.

    Returns:
        A ``MetricTrend`` instance, or ``None`` if both values are missing.
    """
    from vaig.skills.service_health.schema import MetricTrend  # noqa: PLC0415

    metric_name = _METRIC_NAMES.get(metric, metric)

    if current_avg is None and baseline_avg is None:
        return None

    if current_avg is None:
        current_avg = 0.0
    if baseline_avg is None or baseline_avg == 0.0:
        # Zero baseline — new service or no historical data
        return MetricTrend(
            metric=metric_name,
            service_name=service_name,
            namespace=namespace,
            direction="new",
            rate_of_change_percent=None,
            current_value=current_avg,
            baseline_value=baseline_avg,
            baseline_window_days=window_days,
            days_to_threshold=None,
            severity="info",
        )

    rate_pct = ((current_avg - baseline_avg) / baseline_avg) * 100.0

    if rate_pct > 1.0:
        direction = "increasing"
    elif rate_pct < -1.0:
        direction = "decreasing"
    else:
        direction = "stable"

    severity = _classify_severity(metric_name, rate_pct, current_avg, baseline_avg, config)

    # Project days to threshold (only for increasing memory trends)
    days_to_limit: float | None = None
    if direction == "increasing" and metric_name == "memory_usage":
        limit_bytes = config.memory_limit_gib * (1024**3)
        daily_rate = (current_avg - baseline_avg) / window_days
        days_to_limit = _project_days_to_threshold(current_avg, daily_rate, limit_bytes)

    return MetricTrend(
        metric=metric_name,
        service_name=service_name,
        namespace=namespace,
        direction=direction,
        rate_of_change_percent=round(rate_pct, 2),
        current_value=current_avg,
        baseline_value=baseline_avg,
        baseline_window_days=window_days,
        days_to_threshold=round(days_to_limit, 1) if days_to_limit is not None else None,
        severity=severity,
    )


def _classify_severity(
    metric_name: str,
    rate_pct: float,
    current_value: float,
    baseline_value: float,
    config: TrendConfig,
) -> str:
    """Map rate-of-change to severity based on metric type and thresholds.

    For CPU and memory, severity is based on percentage increase thresholds.
    For restarts, severity is based on absolute count delta.

    Args:
        metric_name: Friendly metric name (``cpu_usage``, ``memory_usage``,
            ``restart_count``).
        rate_pct: Percentage change from baseline.
        current_value: Current window average.
        baseline_value: Baseline window average.
        config: TrendConfig with severity thresholds.

    Returns:
        Severity string: ``"critical"``, ``"warning"``, or ``"info"``.
    """
    if metric_name == "memory_usage":
        if rate_pct >= config.memory_critical_pct:
            return "critical"
        if rate_pct >= config.memory_warning_pct:
            return "warning"
    elif metric_name == "cpu_usage":
        if rate_pct >= config.cpu_critical_pct:
            return "critical"
        if rate_pct >= config.cpu_warning_pct:
            return "warning"
    elif metric_name == "restart_count":
        delta = current_value - baseline_value
        if delta >= config.restart_critical_count:
            return "critical"
        if delta >= config.restart_warning_count:
            return "warning"

    return "info"


# ── Public entry point ────────────────────────────────────────


def fetch_anomaly_trends(
    gke_config: GKEConfig,
    namespaces: list[str] | None = None,
) -> TrendAnalysis | None:
    """Analyse GKE metric trends against historical baselines.

    Top-level entry point, mirrors ``fetch_workload_costs()`` signature.
    Iterates namespaces × metrics (CPU, memory, restart_count), queries
    Cloud Monitoring for current and baseline windows, and produces a
    ``TrendAnalysis`` with severity classifications.

    Args:
        gke_config: GKE configuration (cluster, project, etc.).
        namespaces: Namespaces to analyse. ``None`` means the default namespace.

    Returns:
        A ``TrendAnalysis`` with all detected trends, or ``None`` if the
        feature is disabled, Cloud Monitoring is unavailable, or an API
        error occurs.
    """
    from vaig.skills.service_health.schema import TrendAnalysis  # noqa: PLC0415

    trend_config: TrendConfig = gke_config.trends

    if not trend_config.enabled:
        logger.debug("Trend analysis disabled via config")
        return None

    if not _MONITORING_AVAILABLE:
        logger.debug("Cloud Monitoring library not available — skipping trend analysis")
        return None

    project_id = gke_config.project_id
    cluster = gke_config.cluster_name
    if not project_id or not cluster:
        logger.debug("project_id or cluster_name not set — skipping trend analysis")
        return None

    effective_ns = namespaces or [gke_config.default_namespace or "default"]
    metrics = [_CPU_METRIC, _MEMORY_METRIC, _RESTART_METRIC]

    try:
        client = MetricServiceClient()
    except Exception:  # noqa: BLE001
        logger.debug("Failed to create MetricServiceClient — skipping trend analysis")
        return None

    all_trends: list[MetricTrend] = []
    services_seen: set[str] = set()

    try:
        for ns in effective_ns:
            for metric_type in metrics:
                for window_days in trend_config.baseline_days:
                    baseline_avg = _query_baseline(
                        client, project_id, cluster, ns, metric_type, window_days
                    )
                    current_avg = _query_current(
                        client, project_id, cluster, ns, metric_type
                    )

                    # Use namespace as service identifier when controller grouping
                    # returns aggregated results
                    service_name = ns
                    services_seen.add(service_name)

                    trend = _compute_trend(
                        metric_type,
                        service_name,
                        ns,
                        current_avg,
                        baseline_avg,
                        trend_config,
                        window_days,
                    )
                    if trend is not None:
                        all_trends.append(trend)

    except Exception as exc:  # noqa: BLE001
        logger.debug("Trend analysis API error: %s", exc)
        return None

    anomalies = sum(1 for t in all_trends if t.severity in ("warning", "critical"))

    return TrendAnalysis(
        trends=all_trends,
        analyzed_at=datetime.now(tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        baseline_windows=list(trend_config.baseline_days),
        services_analyzed=len(services_seen),
        anomalies_detected=anomalies,
    )

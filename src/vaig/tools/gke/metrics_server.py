"""Kubernetes Metrics Server structured adapter.

Queries the ``metrics.k8s.io/v1beta1`` API (Metrics Server) for live
per-pod CPU and memory snapshots.  Returns :class:`WorkloadUsageMetrics`
dicts keyed by workload name — the same contract as
:func:`vaig.tools.gke.monitoring.get_workload_usage_metrics`.

This is Layer 2 in the multi-source cost measurement pipeline:
  L1 → Cloud Monitoring (historical averages)
  L2 → Metrics Server (live snapshot, this module)
  L3 → Datadog
  L4 → Request-based fallback

Usage is instantaneous (not an average) so freshness is reported as
``"snapshot"``.

Returns ``{}`` silently on:
- 404 ApiException  (Metrics Server not installed)
- 403 ApiException  (RBAC permission denied)
- Any other exception (best-effort; never raises)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from vaig.tools.gke.monitoring import ContainerUsageMetrics, WorkloadUsageMetrics

from . import _clients

if TYPE_CHECKING:
    from vaig.core.config import GKEConfig

logger = logging.getLogger(__name__)

# ── Lazy import guard ─────────────────────────────────────────
_K8S_AVAILABLE = True
try:
    from kubernetes.client import exceptions as k8s_exceptions  # noqa: WPS433
except ImportError:
    _K8S_AVAILABLE = False
    k8s_exceptions = None


def get_metrics_server_usage(
    namespace: str,
    workload_pod_names: dict[str, list[str]],
    *,
    gke_config: GKEConfig,
) -> dict[str, WorkloadUsageMetrics]:
    """Fetch live pod metrics from the Kubernetes Metrics Server.

    Queries ``metrics.k8s.io/v1beta1/pods`` for the given namespace and
    aggregates container-level CPU / memory snapshots per workload.

    Args:
        namespace: Kubernetes namespace to query.
        workload_pod_names: Mapping of workload_name → list[pod_name].
            Used to group pod metrics back to their owning workload.
        gke_config: GKE cluster configuration (for client creation).

    Returns:
        Dict mapping workload_name → :class:`WorkloadUsageMetrics`.
        Returns ``{}`` when Metrics Server is unavailable or any error occurs.
    """
    if not _K8S_AVAILABLE:
        logger.debug("metrics_server: kubernetes client not available")
        return {}

    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, object) and hasattr(result, "error") and getattr(result, "error", False):
        logger.debug("metrics_server: could not create k8s clients")
        return {}

    try:
        # _create_k8s_clients returns (core_v1, apps_v1, custom_objects_api, api_client)
        from vaig.tools.base import ToolResult  # noqa: WPS433
        if isinstance(result, ToolResult):
            return {}
        _, _, custom_api, _ = result
    except (TypeError, ValueError):
        logger.debug("metrics_server: unexpected client result shape")
        return {}

    # Build reverse map: pod_name → workload_name
    pod_to_workload: dict[str, str] = {}
    for wl_name, pod_names in workload_pod_names.items():
        for pod_name in pod_names:
            pod_to_workload[pod_name] = wl_name

    try:
        raw: Any = custom_api.list_namespaced_custom_object(
            group="metrics.k8s.io",
            version="v1beta1",
            namespace=namespace,
            plural="pods",
        )
    except Exception as exc:  # noqa: BLE001
        status = getattr(exc, "status", None)
        if status == 404:
            logger.debug("metrics_server: Metrics Server not installed in cluster (404)")
        elif status == 403:
            logger.warning(
                "metrics_server: RBAC permission denied for metrics.k8s.io/pods in ns=%s: %s",
                namespace,
                exc,
            )
        else:
            logger.debug("metrics_server: query failed for ns=%s: %s", namespace, exc)
        return {}

    # Aggregate containers per workload
    # workload → {container_name → (cpu_sum, mem_sum, count)}
    wl_containers: dict[str, dict[str, list[float | None]]] = {}

    items = raw.get("items") if isinstance(raw, dict) else []
    for pod_item in items or []:
        pod_item_name: str = (
            (pod_item.get("metadata") or {}).get("name") or ""
        )
        pod_wl_name = pod_to_workload.get(pod_item_name)
        if not pod_wl_name:
            continue

        containers = pod_item.get("containers") or []
        for ctr in containers:
            c_name: str = ctr.get("name") or "unknown"
            usage = ctr.get("usage") or {}

            cpu_str: str | None = usage.get("cpu")
            mem_str: str | None = usage.get("memory")

            cpu_cores = _parse_cpu_safe(cpu_str)
            mem_gib = _parse_memory_gib_safe(mem_str)

            wl_containers.setdefault(pod_wl_name, {}).setdefault(c_name, [None, None])
            entry = wl_containers[pod_wl_name][c_name]
            # Accumulate: for multiple pods of the same workload, sum containers
            entry[0] = (entry[0] or 0.0) + (cpu_cores or 0.0) if cpu_cores is not None else entry[0]
            entry[1] = (entry[1] or 0.0) + (mem_gib or 0.0) if mem_gib is not None else entry[1]

    if not wl_containers:
        return {}

    output: dict[str, WorkloadUsageMetrics] = {}
    for wl_name, containers_data in wl_containers.items():
        c_metrics: dict[str, ContainerUsageMetrics] = {}
        for c_name, (cpu, mem) in containers_data.items():
            c_metrics[c_name] = ContainerUsageMetrics(
                container_name=c_name,
                avg_cpu_cores=cpu,
                avg_memory_gib=mem,
            )
        output[wl_name] = WorkloadUsageMetrics(
            namespace=namespace,
            workload_name=wl_name,
            containers=c_metrics,
        )

    return output


# ── CPU / memory parsing helpers ─────────────────────────────


def _parse_cpu_safe(value: str | None) -> float | None:
    """Parse a Kubernetes CPU quantity string to vCPU cores.

    Examples: ``"100m"`` → 0.1,  ``"2"`` → 2.0,  ``"500000000n"`` → 0.5.
    Returns ``None`` on any parse error.
    """
    if not value:
        return None
    try:
        if value.endswith("n"):
            return float(value[:-1]) / 1e9
        if value.endswith("u"):
            return float(value[:-1]) / 1e6
        if value.endswith("m"):
            return float(value[:-1]) / 1000.0
        return float(value)
    except (ValueError, AttributeError):
        return None


def _parse_memory_gib_safe(value: str | None) -> float | None:
    """Parse a Kubernetes memory quantity string to GiB.

    Examples: ``"128Mi"`` → 0.125, ``"1Gi"`` → 1.0, ``"1073741824"`` → 1.0.
    Returns ``None`` on any parse error.
    """
    if not value:
        return None
    try:
        v = value.strip()
        if v.endswith("Ki"):
            return float(v[:-2]) / (1024 * 1024)
        if v.endswith("Mi"):
            return float(v[:-2]) / 1024
        if v.endswith("Gi"):
            return float(v[:-2])
        if v.endswith("Ti"):
            return float(v[:-2]) * 1024
        if v.endswith("Pi"):
            return float(v[:-2]) * 1024 * 1024
        if v.endswith("K"):
            return float(v[:-1]) * 1000 / (1024**3)
        if v.endswith("M"):
            return float(v[:-1]) * 1e6 / (1024**3)
        if v.endswith("G"):
            return float(v[:-1]) * 1e9 / (1024**3)
        # Plain bytes
        return float(v) / (1024**3)
    except (ValueError, AttributeError):
        return None

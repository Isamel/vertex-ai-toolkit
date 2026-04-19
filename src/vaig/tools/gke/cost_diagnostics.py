"""GKE cost metrics diagnostic tool.

Checks the prerequisites and health of each layer in the multi-source cost
measurement pipeline:

- IAM permissions for Cloud Monitoring
- Cluster name match between config and K8s API
- Kubernetes Metrics Server availability
- Cloud Monitoring system metrics availability
- Pod age vs. monitoring window coverage
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from vaig.tools.base import ToolResult

if TYPE_CHECKING:
    from vaig.core.config import GKEConfig

logger = logging.getLogger(__name__)


def diagnose_gke_cost_metrics(*, gke_config: GKEConfig) -> ToolResult:
    """Run diagnostic checks for the GKE cost metrics pipeline.

    Runs five checks to identify why cost data may be missing or unreliable:

    1. ``iam_monitoring_viewer`` — Can the service account list Cloud Monitoring time series?
    2. ``cluster_name_match`` — Does ``gke_config.cluster_name`` match the K8s API server?
    3. ``metrics_server_installed`` — Is the Kubernetes Metrics Server available?
    4. ``system_metrics_enabled`` — Does Cloud Monitoring have GKE system metrics (24h window)?
    5. ``window_coverage`` — Are pods older than 60 minutes (so 60m data should exist)?

    Args:
        gke_config: GKE cluster configuration.

    Returns:
        :class:`~vaig.tools.base.ToolResult` with JSON output containing a
        ``checks`` list and a ``recommendation`` string.
    """
    checks: list[dict[str, Any]] = []
    recommendation_parts: list[str] = []

    # ── Check 1: IAM Monitoring Viewer ───────────────────────
    iam_check = _check_iam_monitoring_viewer(gke_config)
    checks.append(iam_check)
    if iam_check["status"] != "ok":
        recommendation_parts.append(
            "Grant the service account the 'roles/monitoring.viewer' IAM role "
            f"on project '{gke_config.project_id}'."
        )

    # ── Check 2: Cluster name match ───────────────────────────
    name_check = _check_cluster_name_match(gke_config)
    checks.append(name_check)
    if name_check["status"] != "ok":
        recommendation_parts.append(
            "Update gke_config.cluster_name to match the actual cluster name "
            "reported by the K8s API: '{}'.".format(name_check.get("actual", "unknown"))
        )

    # ── Check 3: Metrics Server installed ────────────────────
    ms_check = _check_metrics_server_installed(gke_config)
    checks.append(ms_check)
    if ms_check["status"] != "ok":
        recommendation_parts.append(
            "Install the Kubernetes Metrics Server as a fallback data source "
            "(Layer 2).  See https://github.com/kubernetes-sigs/metrics-server."
        )

    # ── Check 4: System metrics enabled ──────────────────────
    sys_check = _check_system_metrics_enabled(gke_config)
    checks.append(sys_check)
    if sys_check["status"] != "ok":
        recommendation_parts.append(
            "Enable GKE system metrics in Cloud Monitoring "
            f"(project '{gke_config.project_id}').  Ensure 'system' metrics are not filtered out "
            "by a metrics exclusion policy."
        )

    # ── Check 5: Window coverage ──────────────────────────────
    cov_check = _check_window_coverage(gke_config)
    checks.append(cov_check)
    if cov_check["status"] == "warning":
        recommendation_parts.append(
            "Some pods appear to be newer than 60 minutes.  "
            "The 60m Cloud Monitoring window may be empty for these pods; "
            "the pipeline will automatically retry with a 24h window."
        )

    recommendation = (
        " | ".join(recommendation_parts)
        if recommendation_parts
        else "All checks passed — the cost metrics pipeline should work correctly."
    )

    output = json.dumps(
        {"checks": checks, "recommendation": recommendation},
        indent=2,
    )
    all_ok = all(c["status"] in ("ok", "warning") for c in checks)
    return ToolResult(output=output, error=not all_ok)


# ── Internal helpers ──────────────────────────────────────────


def _check_iam_monitoring_viewer(gke_config: GKEConfig) -> dict[str, Any]:
    """Try to list a minimal time-series window; catches PermissionDenied."""
    try:
        from google.api_core.exceptions import PermissionDenied  # noqa: WPS433
        from google.cloud import monitoring_v3  # noqa: WPS433

        client = monitoring_v3.MetricServiceClient()
        project_name = f"projects/{gke_config.project_id}"
        interval = monitoring_v3.TimeInterval()
        import time  # noqa: WPS433

        from google.protobuf import timestamp_pb2  # noqa: WPS433
        now = int(time.time())
        interval.end_time = timestamp_pb2.Timestamp(seconds=now)
        interval.start_time = timestamp_pb2.Timestamp(seconds=now - 60)
        request = monitoring_v3.ListTimeSeriesRequest(
            name=project_name,
            filter='metric.type="kubernetes.io/container/cpu/core_usage_time"',
            interval=interval,
            view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.HEADERS,
            page_size=1,
        )
        list(client.list_time_series(request=request))
        return {"name": "iam_monitoring_viewer", "status": "ok", "detail": "Permission check passed."}
    except PermissionDenied as exc:
        return {"name": "iam_monitoring_viewer", "status": "error", "detail": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"name": "iam_monitoring_viewer", "status": "warning", "detail": f"Could not verify: {exc}"}


def _check_cluster_name_match(gke_config: GKEConfig) -> dict[str, Any]:
    """Compare configured cluster name with the K8s API server version info."""
    try:
        from vaig.tools.gke._clients import _create_k8s_clients  # noqa: WPS433

        clients = _create_k8s_clients(gke_config)
        if isinstance(clients, ToolResult):
            return {"name": "cluster_name_match", "status": "warning", "detail": "Could not create K8s clients."}

        _core_v1, _apps_v1, _custom_api, api_client = clients
        from kubernetes.client import VersionApi  # noqa: WPS433

        version_api = VersionApi(api_client=api_client)
        version_info = version_api.get_code()
        # K8s version info doesn't include cluster name; check via node labels instead
        core_v1 = clients[0]
        nodes = core_v1.list_node(limit=1)
        actual_cluster: str | None = None
        if nodes.items:
            labels = nodes.items[0].metadata.labels or {}
            actual_cluster = (
                labels.get("alpha.eksctl.io/cluster-name")
                or labels.get("cloud.google.com/gke-cluster")
                or labels.get("cluster.x-k8s.io/cluster-name")
            )

        configured = gke_config.cluster_name
        if actual_cluster and actual_cluster != configured:
            return {
                "name": "cluster_name_match",
                "status": "warning",
                "detail": f"Configured '{configured}' but node label shows '{actual_cluster}'.",
                "actual": actual_cluster,
            }
        return {
            "name": "cluster_name_match",
            "status": "ok",
            "detail": (
                f"Cluster name '{configured}' matches (or could not be verified from node labels). "
                f"K8s version: {version_info.git_version}."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {"name": "cluster_name_match", "status": "warning", "detail": f"Could not verify: {exc}"}


def _check_metrics_server_installed(gke_config: GKEConfig) -> dict[str, Any]:
    """Probe metrics.k8s.io/v1beta1/pods to check Metrics Server availability."""
    try:
        from kubernetes.client.exceptions import ApiException  # noqa: WPS433

        from vaig.tools.gke._clients import _create_k8s_clients  # noqa: WPS433

        clients = _create_k8s_clients(gke_config)
        if isinstance(clients, ToolResult):
            return {"name": "metrics_server_installed", "status": "warning", "detail": "Could not create K8s clients."}

        _core_v1, _apps_v1, custom_api, _api_client = clients
        custom_api.list_namespaced_custom_object(
            group="metrics.k8s.io",
            version="v1beta1",
            namespace=gke_config.default_namespace or "default",
            plural="pods",
            limit=1,
        )
        return {"name": "metrics_server_installed", "status": "ok", "detail": "Metrics Server is available."}
    except ApiException as exc:
        if exc.status == 404:
            return {
                "name": "metrics_server_installed",
                "status": "warning",
                "detail": "Metrics Server API not found (404) — not installed.",
            }
        if exc.status == 403:
            return {
                "name": "metrics_server_installed",
                "status": "warning",
                "detail": "RBAC denied (403) — Metrics Server may be installed but service account lacks access.",
            }
        return {"name": "metrics_server_installed", "status": "warning", "detail": f"API error {exc.status}: {exc.reason}"}
    except Exception as exc:  # noqa: BLE001
        return {"name": "metrics_server_installed", "status": "warning", "detail": f"Could not verify: {exc}"}


def _check_system_metrics_enabled(gke_config: GKEConfig) -> dict[str, Any]:
    """Check if kubernetes.io/container/cpu/core_usage_time has data in the last 24h."""
    try:
        import time  # noqa: WPS433

        from google.api_core.exceptions import GoogleAPICallError  # noqa: WPS433
        from google.cloud import monitoring_v3  # noqa: WPS433
        from google.protobuf import timestamp_pb2  # noqa: WPS433

        client = monitoring_v3.MetricServiceClient()
        project_name = f"projects/{gke_config.project_id}"
        now = int(time.time())
        interval = monitoring_v3.TimeInterval(
            end_time=timestamp_pb2.Timestamp(seconds=now),
            start_time=timestamp_pb2.Timestamp(seconds=now - 86400),
        )
        request = monitoring_v3.ListTimeSeriesRequest(
            name=project_name,
            filter=(
                'metric.type="kubernetes.io/container/cpu/core_usage_time" '
                f'AND resource.labels.cluster_name="{gke_config.cluster_name}"'
            ),
            interval=interval,
            view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.HEADERS,
            page_size=1,
        )
        series = list(client.list_time_series(request=request))
        if series:
            return {"name": "system_metrics_enabled", "status": "ok", "detail": "GKE system metrics found in Cloud Monitoring."}
        return {
            "name": "system_metrics_enabled",
            "status": "error",
            "detail": (
                "No kubernetes.io/container/cpu/core_usage_time metrics found "
                f"for cluster '{gke_config.cluster_name}' in the last 24h."
            ),
        }
    except GoogleAPICallError as exc:
        return {"name": "system_metrics_enabled", "status": "error", "detail": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"name": "system_metrics_enabled", "status": "warning", "detail": f"Could not verify: {exc}"}


def _check_window_coverage(gke_config: GKEConfig) -> dict[str, Any]:
    """Check if pods in the default namespace are older than 60 minutes."""
    try:
        import datetime  # noqa: WPS433

        from vaig.tools.gke._clients import _create_k8s_clients  # noqa: WPS433

        clients = _create_k8s_clients(gke_config)
        if isinstance(clients, ToolResult):
            return {"name": "window_coverage", "status": "warning", "detail": "Could not create K8s clients."}

        core_v1, _apps_v1, _custom_api, _api_client = clients
        pods = core_v1.list_namespaced_pod(
            namespace=gke_config.default_namespace or "default",
            limit=50,
        )
        now = datetime.datetime.now(tz=datetime.UTC)
        new_pod_names: list[str] = []
        for pod in pods.items:
            start_time = pod.status.start_time
            if start_time and (now - start_time).total_seconds() < 3600:
                new_pod_names.append(pod.metadata.name or "unknown")

        if new_pod_names:
            return {
                "name": "window_coverage",
                "status": "warning",
                "detail": (
                    f"{len(new_pod_names)} pod(s) are newer than 60 minutes — "
                    "60m monitoring window may be sparse.  "
                    "The pipeline will automatically retry with a 24h window."
                ),
                "new_pods": new_pod_names[:10],
            }
        return {
            "name": "window_coverage",
            "status": "ok",
            "detail": "All sampled pods are older than 60 minutes — 60m window should have data.",
        }
    except Exception as exc:  # noqa: BLE001
        return {"name": "window_coverage", "status": "warning", "detail": f"Could not verify: {exc}"}

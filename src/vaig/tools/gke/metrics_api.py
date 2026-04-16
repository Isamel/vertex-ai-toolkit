"""Kubernetes metrics API health check and custom/external metric queries.

Probes the aggregated API layer (metrics.k8s.io, custom.metrics.k8s.io,
external.metrics.k8s.io) to diagnose broken metrics pipelines that cause
HPA scaling failures, and queries individual custom/external metrics for
deeper root-cause analysis.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vaig.tools.base import ToolResult

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

# ── Metrics API groups to probe ───────────────────────────────
_METRICS_GROUPS: dict[str, dict[str, str]] = {
    "metrics.k8s.io": {"version": "v1beta1", "label": "Metrics Server"},
    "custom.metrics.k8s.io": {"version": "v1beta1", "label": "Custom Metrics"},
    "external.metrics.k8s.io": {"version": "v1beta1", "label": "External Metrics"},
}


# ── Health check ──────────────────────────────────────────────


def check_metrics_api_health(*, gke_config: GKEConfig) -> ToolResult:
    """Probe Kubernetes aggregated metrics API groups and report their health.

    Checks whether ``metrics.k8s.io``, ``custom.metrics.k8s.io``, and
    ``external.metrics.k8s.io`` are registered and available.  For each
    registered group, reads the corresponding ``APIService`` object to
    determine its condition.

    Also detects whether the cluster is running in GKE Autopilot mode
    (which always has Metrics Server pre-installed) and annotates the
    output accordingly.

    Args:
        gke_config: GKE cluster configuration.

    Returns:
        ToolResult with a Markdown health report for all three metrics API
        groups, including availability status and actionable guidance.
    """
    if not _clients._K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    _, _, _, api_client = result

    # ── Discover registered API groups ────────────────────────
    try:
        from kubernetes.client import ApisApi  # noqa: WPS433

        apis_api = ApisApi(api_client)
        api_groups_response = apis_api.get_api_versions()
    except k8s_exceptions.ApiException as exc:
        if exc.status in {401, 403}:
            return ToolResult(
                output=f"Authentication/authorization error querying API groups: {exc.reason}",
                error=True,
            )
        logger.warning("Error querying API groups: %s", exc)
        return ToolResult(
            output=f"Kubernetes API error ({exc.status}): {exc.reason}",
            error=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected error querying API groups: %s", exc)
        return ToolResult(
            output=f"Error querying API groups: {exc}",
            error=True,
        )

    # Build set of registered group names
    registered_groups: set[str] = set()
    for group in api_groups_response.groups or []:
        registered_groups.add(group.name)

    # ── Probe each metrics API group ──────────────────────────
    try:
        from kubernetes.client import ApiregistrationV1Api  # noqa: WPS433

        api_reg = ApiregistrationV1Api(api_client)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to create ApiregistrationV1Api: %s", exc)
        api_reg = None

    lines: list[str] = ["## Metrics API Health Check\n"]
    healthy_count = 0
    total_count = len(_METRICS_GROUPS)

    for group_name, group_info in _METRICS_GROUPS.items():
        version = group_info["version"]
        label = group_info["label"]
        api_service_name = f"{version}.{group_name}"

        if group_name not in registered_groups:
            lines.append(f"### {label} (`{group_name}`)")
            lines.append("- Status: ❌ **Not Registered**")
            lines.append(f"- API group `{group_name}` is not present in the cluster.")
            lines.append("- The adapter/server providing this API is not installed.\n")
            continue

        # Group is registered — check APIService condition
        condition_status = "Unknown"
        condition_message = ""
        if api_reg is not None:
            try:
                api_svc = api_reg.read_api_service(api_service_name)
                for condition in api_svc.status.conditions or []:
                    if condition.type == "Available":
                        condition_status = condition.status  # "True" / "False" / "Unknown"
                        condition_message = condition.message or ""
                        break
            except k8s_exceptions.ApiException as exc:
                if exc.status == 404:
                    condition_status = "NotFound"
                    condition_message = f"APIService {api_service_name} not found"
                else:
                    logger.warning("Error reading APIService %s: %s", api_service_name, exc)
                    condition_message = f"Error reading APIService: {exc.reason}"
            except Exception as exc:  # noqa: BLE001
                logger.warning("Unexpected error reading APIService %s: %s", api_service_name, exc)
                condition_message = f"Error: {exc}"

        lines.append(f"### {label} (`{group_name}`)")

        if condition_status == "True":
            lines.append("- Status: ✅ **Available**")
            healthy_count += 1
        elif condition_status == "False":
            lines.append("- Status: ❌ **Unavailable**")
        elif condition_status == "NotFound":
            lines.append("- Status: ⚠️ **APIService Missing**")
        else:
            lines.append("- Status: ⚠️ **Unknown**")

        lines.append(f"- APIService: `{api_service_name}`")
        if condition_message:
            lines.append(f"- Message: {condition_message}")
        lines.append("")

    # ── Autopilot annotation ──────────────────────────────────
    is_autopilot = _clients.detect_autopilot(gke_config)
    if is_autopilot is True:
        lines.append("### Cluster Mode")
        lines.append(
            "- GKE **Autopilot** cluster detected — Metrics Server (`metrics.k8s.io`) "
            "is always pre-installed and managed by Google.\n"
        )
    elif is_autopilot is False:
        lines.append("### Cluster Mode")
        lines.append(
            "- GKE **Standard** cluster — Metrics Server must be deployed manually "
            "or via `gcloud container clusters update --enable-managed-prometheus`.\n"
        )

    # ── Summary ───────────────────────────────────────────────
    lines.append("### Summary")
    lines.append(f"- {healthy_count}/{total_count} metrics API groups healthy.")

    if healthy_count == total_count:
        lines.append("- All metrics APIs are operational — HPA metric pipeline is healthy.")
    elif healthy_count == 0:
        lines.append(
            "- ❌ No metrics APIs are available — HPA cannot scale on any metric type. "
            "Check that Metrics Server and the custom metrics adapter are installed."
        )
    else:
        missing = [
            f"`{g}` ({_METRICS_GROUPS[g]['label']})"
            for g in _METRICS_GROUPS
            if g not in registered_groups
        ]
        if missing:
            lines.append(f"- Missing: {', '.join(missing)}")
        lines.append(
            "- ⚠️ Some metrics APIs are unavailable — HPAs referencing missing groups will fail to scale."
        )

    return ToolResult(output="\n".join(lines))


# ── Custom metrics query ──────────────────────────────────────


def query_custom_metrics(
    metric_name: str = "",
    *,
    gke_config: GKEConfig,
    namespace: str = "",
) -> ToolResult:
    """Query custom metrics from the ``custom.metrics.k8s.io`` API group.

    When *metric_name* is empty, lists all available custom metrics.
    When provided, fetches the named metric scoped to *namespace* (or
    cluster-wide when namespace is empty).

    Args:
        metric_name: Custom metric name to query (e.g. ``requests_per_second``).
            When empty, lists all available custom metrics.
        gke_config: GKE cluster configuration.
        namespace: Kubernetes namespace scope. When empty, queries cluster-wide.

    Returns:
        ToolResult with metric values or a list of available metrics.
    """
    if not _clients._K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    _, _, custom_api, _ = result

    group = "custom.metrics.k8s.io"
    version = "v1beta1"

    # ── List mode (no metric_name) ────────────────────────────
    if not metric_name:
        try:
            api_resources = custom_api.get_api_resources(group=group, version=version)
            resources = api_resources.get("resources", []) if isinstance(api_resources, dict) else []
            if not resources:
                return ToolResult(
                    output=(
                        "## Custom Metrics — Available Metrics\n\n"
                        "No custom metrics are currently registered.\n"
                        "This may indicate that the custom metrics adapter is not installed "
                        "or no applications are exporting custom metrics."
                    ),
                )
            lines = ["## Custom Metrics — Available Metrics\n"]
            lines.append(f"Found {len(resources)} custom metric(s):\n")
            for res in resources:
                name = res.get("name", "<unknown>") if isinstance(res, dict) else str(res)
                lines.append(f"- `{name}`")
            return ToolResult(output="\n".join(lines))
        except k8s_exceptions.ApiException as exc:
            if exc.status == 404:
                return ToolResult(
                    output=(
                        "## Custom Metrics — Unavailable\n\n"
                        "The `custom.metrics.k8s.io` API group is not registered.\n"
                        "Install a custom metrics adapter (e.g. Prometheus Adapter, "
                        "Stackdriver Adapter) to enable custom metrics."
                    ),
                )
            if exc.status in {401, 403}:
                return ToolResult(
                    output=f"Authentication/authorization error listing custom metrics: {exc.reason}",
                    error=True,
                )
            logger.warning("Error listing custom metrics: %s", exc)
            return ToolResult(
                output=f"Kubernetes API error ({exc.status}): {exc.reason}",
                error=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unexpected error listing custom metrics: %s", exc)
            return ToolResult(output=f"Error listing custom metrics: {exc}", error=True)

    # ── Query specific metric ─────────────────────────────────
    try:
        if namespace:
            metric_data = custom_api.list_namespaced_custom_object(
                group=group,
                version=version,
                namespace=namespace,
                plural=metric_name,
            )
        else:
            metric_data = custom_api.list_cluster_custom_object(
                group=group,
                version=version,
                plural=metric_name,
            )
    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            scope = f"namespace '{namespace}'" if namespace else "cluster-wide"
            return ToolResult(
                output=(
                    f"## Custom Metric: `{metric_name}`\n\n"
                    f"Metric not found ({scope}).\n"
                    "Verify the metric name and that the exporting application is running."
                ),
            )
        if exc.status in {401, 403}:
            return ToolResult(
                output=f"Authentication/authorization error querying metric '{metric_name}': {exc.reason}",
                error=True,
            )
        logger.warning("Error querying custom metric %s: %s", metric_name, exc)
        return ToolResult(
            output=f"Kubernetes API error ({exc.status}): {exc.reason}",
            error=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected error querying custom metric %s: %s", metric_name, exc)
        return ToolResult(output=f"Error querying custom metric '{metric_name}': {exc}", error=True)

    # ── Format results ────────────────────────────────────────
    items = metric_data.get("items", []) if isinstance(metric_data, dict) else []
    scope = f"namespace `{namespace}`" if namespace else "cluster-wide"
    lines = [f"## Custom Metric: `{metric_name}` ({scope})\n"]

    if not items:
        lines.append("No data points returned for this metric.")
        lines.append("The metric may not have any current values, or the exporting application may not be running.")
        return ToolResult(output="\n".join(lines))

    lines.append("| Object | Value | Timestamp |")
    lines.append("|--------|-------|-----------|")
    for item in items:
        described = item.get("describedObject", {})
        obj_kind = described.get("kind", "?")
        obj_name = described.get("name", "?")
        obj_ns = described.get("namespace", "")
        value = item.get("value", item.get("averageValue", "?"))
        timestamp = item.get("timestamp", "?")
        obj_ref = f"{obj_kind}/{obj_name}"
        if obj_ns:
            obj_ref = f"{obj_ns}/{obj_ref}"
        lines.append(f"| `{obj_ref}` | {value} | {timestamp} |")

    return ToolResult(output="\n".join(lines))


# ── External metrics query ────────────────────────────────────


def query_external_metrics(
    metric_name: str,
    *,
    gke_config: GKEConfig,
    namespace: str = "",
) -> ToolResult:
    """Query external metrics from the ``external.metrics.k8s.io`` API group.

    External metrics are provided by cloud monitoring systems (e.g. Cloud
    Monitoring, Datadog) and are not tied to Kubernetes objects.

    Args:
        metric_name: External metric name to query (e.g.
            ``pubsub.googleapis.com|subscription|num_undelivered_messages``).
        gke_config: GKE cluster configuration.
        namespace: Kubernetes namespace scope for the query.

    Returns:
        ToolResult with metric values.
    """
    if not _clients._K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    if not metric_name:
        return ToolResult(
            output=(
                "## External Metrics — Error\n\n"
                "``metric_name`` is required for external metric queries.\n"
                "Example: ``pubsub.googleapis.com|subscription|num_undelivered_messages``"
            ),
            error=True,
        )

    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    _, _, custom_api, _ = result

    group = "external.metrics.k8s.io"
    version = "v1beta1"
    ns = namespace or "default"

    try:
        metric_data = custom_api.list_namespaced_custom_object(
            group=group,
            version=version,
            namespace=ns,
            plural=metric_name,
        )
    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(
                output=(
                    f"## External Metric: `{metric_name}`\n\n"
                    f"Metric not found in namespace `{ns}`.\n"
                    "Verify that the external metrics adapter is installed and the metric name is correct.\n"
                    "Common adapters: Stackdriver Adapter (GKE), Datadog Cluster Agent, KEDA."
                ),
            )
        if exc.status in {401, 403}:
            return ToolResult(
                output=f"Authentication/authorization error querying external metric '{metric_name}': {exc.reason}",
                error=True,
            )
        logger.warning("Error querying external metric %s: %s", metric_name, exc)
        return ToolResult(
            output=f"Kubernetes API error ({exc.status}): {exc.reason}",
            error=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected error querying external metric %s: %s", metric_name, exc)
        return ToolResult(output=f"Error querying external metric '{metric_name}': {exc}", error=True)

    # ── Format results ────────────────────────────────────────
    items = metric_data.get("items", []) if isinstance(metric_data, dict) else []
    lines = [f"## External Metric: `{metric_name}` (namespace: `{ns}`)\n"]

    if not items:
        lines.append("No data points returned for this metric.")
        lines.append("The metric may not have current values, or the external metrics adapter may not be serving it.")
        return ToolResult(output="\n".join(lines))

    lines.append("| Metric Name | Value | Timestamp |")
    lines.append("|-------------|-------|-----------|")
    for item in items:
        name = item.get("metricName", metric_name)
        value = item.get("value", item.get("averageValue", "?"))
        timestamp = item.get("timestamp", "?")
        lines.append(f"| `{name}` | {value} | {timestamp} |")

    return ToolResult(output="\n".join(lines))

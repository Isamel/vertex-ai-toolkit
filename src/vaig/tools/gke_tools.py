"""GKE tools — read-only Kubernetes cluster inspection via the ``kubernetes`` Python client."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from vaig.tools.base import ToolDef, ToolParam, ToolResult

if TYPE_CHECKING:
    from vaig.core.config import GKEConfig

logger = logging.getLogger(__name__)

# ── K8s client cache ──────────────────────────────────────────
# Keyed on (kubeconfig_path, context, proxy_url) so each unique
# GKEConfig combination creates clients only once.
_CLIENT_CACHE: dict[tuple[str, str, str], tuple[Any, ...]] = {}

# ── Autopilot detection cache ─────────────────────────────────
# Keyed on (project_id, location, cluster_name).
# Values: True (Autopilot), False (Standard), None (detection failed).
_AUTOPILOT_CACHE: dict[tuple[str, str, str], bool | None] = {}


# ── Lazy import guard ─────────────────────────────────────────
# The ``kubernetes`` package is an optional dependency (``pip install vertex-ai-toolkit[live]``).
# All public functions fail gracefully with a descriptive ToolResult when it is missing.

_K8S_AVAILABLE = True
_K8S_IMPORT_ERROR: str | None = None

try:
    from kubernetes import client as k8s_client  # noqa: WPS433
    from kubernetes import config as k8s_config  # noqa: WPS433
    from kubernetes.client import exceptions as k8s_exceptions  # noqa: WPS433
except ImportError as _exc:
    _K8S_AVAILABLE = False
    _K8S_IMPORT_ERROR = (
        "The 'kubernetes' package is not installed. "
        "Install it with: pip install vertex-ai-toolkit[live]"
    )


def _k8s_unavailable() -> ToolResult:
    """Return a ToolResult indicating that the kubernetes SDK is not installed."""
    return ToolResult(output=_K8S_IMPORT_ERROR or "kubernetes SDK not available", error=True)


# ── Autopilot detection ──────────────────────────────────────


def _query_autopilot_status(project: str, location: str, cluster: str) -> bool:
    """Query the GKE API for the Autopilot status of a cluster.

    This is an internal helper extracted so tests can mock it without
    fighting Python's import machinery.

    Raises:
        ImportError: If ``google-cloud-container`` is not installed.
        Exception: On any API or network error.
    """
    from google.cloud import container_v1  # noqa: WPS433

    client = container_v1.ClusterManagerClient()
    name = f"projects/{project}/locations/{location}/clusters/{cluster}"
    cluster_obj = client.get_cluster(name=name)
    return bool(cluster_obj.autopilot and cluster_obj.autopilot.enabled)


def detect_autopilot(gke_config: GKEConfig) -> bool | None:
    """Detect whether the GKE cluster is running in Autopilot mode.

    Uses the ``google-cloud-container`` library to query the GKE API for
    ``cluster.autopilot.enabled``.  Results are cached per
    (project_id, location, cluster_name) tuple.

    Args:
        gke_config: GKE configuration with ``project_id``, ``location``,
            and ``cluster_name`` populated.

    Returns:
        ``True`` if Autopilot, ``False`` if Standard, ``None`` if detection
        failed (missing config, missing library, API error).
    """
    project = gke_config.project_id
    location = gke_config.location
    cluster = gke_config.cluster_name

    if not project or not location or not cluster:
        logger.debug(
            "Autopilot detection skipped: missing project_id=%r, location=%r, cluster_name=%r",
            project, location, cluster,
        )
        return None

    cache_key = (project, location, cluster)
    if cache_key in _AUTOPILOT_CACHE:
        return _AUTOPILOT_CACHE[cache_key]

    try:
        is_autopilot = _query_autopilot_status(project, location, cluster)
        _AUTOPILOT_CACHE[cache_key] = is_autopilot
        logger.info("GKE Autopilot detection: cluster=%s autopilot=%s", cluster, is_autopilot)
        return is_autopilot

    except ImportError:
        logger.warning(
            "google-cloud-container not installed — Autopilot detection unavailable. "
            "Install with: pip install vertex-ai-toolkit[live]"
        )
        _AUTOPILOT_CACHE[cache_key] = None
        return None

    except Exception as exc:
        logger.warning("Autopilot detection failed for %s: %s", cluster, exc)
        _AUTOPILOT_CACHE[cache_key] = None
        return None


def clear_autopilot_cache() -> None:
    """Clear the Autopilot detection cache (useful for testing)."""
    _AUTOPILOT_CACHE.clear()


# ── K8s client helper (Task 2.6) ─────────────────────────────


def _extract_proxy_url_from_kubeconfig(
    kubeconfig_path: str | None = None,
    context: str | None = None,
) -> str | None:
    """Extract ``proxy-url`` from the active kubeconfig cluster entry.

    The ``kubernetes`` Python client (v35) ignores the ``proxy-url`` field
    that ``kubectl`` honours.  This helper reads the raw YAML so we can
    apply it manually via ``Configuration.proxy``.
    """
    kube_path = kubeconfig_path or os.environ.get(
        "KUBECONFIG", str(Path.home() / ".kube" / "config"),
    )
    try:
        with open(kube_path) as fh:
            kube_config = yaml.safe_load(fh)
    except (FileNotFoundError, yaml.YAMLError):
        return None

    if not isinstance(kube_config, dict):
        return None

    # Determine the active context
    ctx_name = context or kube_config.get("current-context")
    if not ctx_name:
        return None

    # Locate the context entry
    contexts = kube_config.get("contexts", [])
    ctx_entry = next((c for c in contexts if c.get("name") == ctx_name), None)
    if not ctx_entry:
        return None

    cluster_name = ctx_entry.get("context", {}).get("cluster")
    if not cluster_name:
        return None

    # Locate the cluster entry
    clusters = kube_config.get("clusters", [])
    cluster_entry = next((c for c in clusters if c.get("name") == cluster_name), None)
    if not cluster_entry:
        return None

    return cluster_entry.get("cluster", {}).get("proxy-url")


def _cache_key(gke_config: GKEConfig) -> tuple[str, str, str]:
    """Build a hashable cache key from the GKEConfig fields that affect client creation."""
    return (
        gke_config.kubeconfig_path or "",
        gke_config.context or "",
        gke_config.proxy_url or "",
    )


def clear_k8s_client_cache() -> None:
    """Clear the cached Kubernetes API clients.

    Useful in tests or when kubeconfig/credentials change at runtime.
    """
    _CLIENT_CACHE.clear()


def _create_k8s_clients(
    gke_config: GKEConfig,
) -> tuple[Any, Any, Any, Any] | ToolResult:
    """Create and configure Kubernetes API clients from GKEConfig.

    Returns a tuple of ``(CoreV1Api, AppsV1Api, CustomObjectsApi, ApiClient)``
    on success, or a ``ToolResult`` with ``error=True`` on failure.

    Results are cached per unique ``(kubeconfig_path, context, proxy_url)``
    combination so that repeated tool invocations within the same session
    reuse the same authenticated clients instead of rebuilding them on
    every call.

    Supports:
    - Explicit ``kubeconfig_path`` + optional ``context``
    - Default kubeconfig (``~/.kube/config``)
    - In-cluster config (for workload identity / GKE pods)

    The ``proxy-url`` field in kubeconfig cluster entries is **not** supported
    by the ``kubernetes`` Python client (v35).  This function works around the
    limitation by parsing the raw YAML, extracting ``proxy-url``, and injecting
    it into ``kubernetes.client.Configuration.proxy``.
    """
    if not _K8S_AVAILABLE:
        return _k8s_unavailable()

    key = _cache_key(gke_config)
    if key in _CLIENT_CACHE:
        return _CLIENT_CACHE[key]

    try:
        kubeconfig_path = gke_config.kubeconfig_path or None
        context = gke_config.context or None

        # ── Resolve proxy URL ────────────────────────────────
        proxy_url = _extract_proxy_url_from_kubeconfig(kubeconfig_path, context)

        # Explicit GKEConfig override takes precedence
        if gke_config.proxy_url:
            proxy_url = gke_config.proxy_url

        # ── Build a Configuration object with proxy ──────────
        config = k8s_client.Configuration()
        if proxy_url:
            config.proxy = proxy_url
            logger.info("Using proxy URL for K8s API: %s", proxy_url)

        # ── Load kubeconfig into the Configuration ───────────
        if kubeconfig_path:
            k8s_config.load_kube_config(
                config_file=kubeconfig_path,
                context=context,
                client_configuration=config,
            )
        else:
            try:
                k8s_config.load_kube_config(
                    context=context,
                    client_configuration=config,
                )
            except k8s_config.ConfigException:
                # Fallback to in-cluster config (workload identity)
                k8s_config.load_incluster_config()
                # In-cluster doesn't need proxy — return plain clients
                api_client_ic = k8s_client.ApiClient()
                clients = (
                    k8s_client.CoreV1Api(api_client_ic),
                    k8s_client.AppsV1Api(api_client_ic),
                    k8s_client.CustomObjectsApi(api_client_ic),
                    api_client_ic,
                )
                _CLIENT_CACHE[key] = clients
                return clients
    except Exception as exc:
        return ToolResult(
            output=f"Failed to configure Kubernetes client: {exc}",
            error=True,
        )

    # ── Build API clients with the proxy-aware Configuration ─
    api_client = k8s_client.ApiClient(config)
    clients = (
        k8s_client.CoreV1Api(api_client),
        k8s_client.AppsV1Api(api_client),
        k8s_client.CustomObjectsApi(api_client),
        api_client,
    )
    _CLIENT_CACHE[key] = clients
    return clients


# ── Formatting helpers ────────────────────────────────────────

_RESOURCE_API_MAP: dict[str, str] = {
    "pods": "core",
    "services": "core",
    "configmaps": "core",
    "secrets": "core",
    "serviceaccounts": "core",
    "endpoints": "core",
    "nodes": "core",
    "namespaces": "core",
    "pv": "core",
    "persistentvolumes": "core",
    "pvc": "core",
    "persistentvolumeclaims": "core",
    "deployments": "apps",
    "statefulsets": "apps",
    "daemonsets": "apps",
    "replicasets": "apps",
    "jobs": "batch",
    "cronjobs": "batch",
    "hpa": "autoscaling",
    "horizontalpodautoscalers": "autoscaling",
    "ingress": "networking",
    "ingresses": "networking",
    "networkpolicies": "networking",
    "poddisruptionbudgets": "policy",
    "resourcequotas": "core",
}

# Canonical aliases so users can type short names
_RESOURCE_ALIASES: dict[str, str] = {
    "po": "pods",
    "pod": "pods",
    "svc": "services",
    "service": "services",
    "cm": "configmaps",
    "configmap": "configmaps",
    "secret": "secrets",
    "sa": "serviceaccounts",
    "serviceaccount": "serviceaccounts",
    "ep": "endpoints",
    "endpoint": "endpoints",
    "node": "nodes",
    "ns": "namespaces",
    "namespace": "namespaces",
    "deploy": "deployments",
    "deployment": "deployments",
    "sts": "statefulsets",
    "statefulset": "statefulsets",
    "ds": "daemonsets",
    "daemonset": "daemonsets",
    "rs": "replicasets",
    "replicaset": "replicasets",
    "job": "jobs",
    "cronjob": "cronjobs",
    "cj": "cronjobs",
    "horizontalpodautoscaler": "hpa",
    "ing": "ingress",
    "netpol": "networkpolicies",
    "networkpolicy": "networkpolicies",
    "persistentvolume": "pv",
    "persistentvolumeclaim": "pvc",
    "poddisruptionbudget": "poddisruptionbudgets",
    "pdb": "poddisruptionbudgets",
    "pdbs": "poddisruptionbudgets",
    "resourcequota": "resourcequotas",
    "quota": "resourcequotas",
    "quotas": "resourcequotas",
}


def _normalise_resource(resource: str) -> str:
    """Normalise a resource type string to its canonical plural form."""
    lower = resource.lower().strip()
    return _RESOURCE_ALIASES.get(lower, lower)


def _age(creation_timestamp: datetime | None) -> str:
    """Return a human-readable age string from a creation timestamp."""
    if creation_timestamp is None:
        return "<unknown>"
    now = datetime.now(timezone.utc)
    delta = now - creation_timestamp.replace(tzinfo=timezone.utc) if creation_timestamp.tzinfo is None else now - creation_timestamp
    total_seconds = int(delta.total_seconds())
    if total_seconds < 0:
        return "0s"
    if total_seconds < 60:
        return f"{total_seconds}s"
    if total_seconds < 3600:
        return f"{total_seconds // 60}m"
    if total_seconds < 86400:
        return f"{total_seconds // 3600}h"
    return f"{total_seconds // 86400}d"


def _pod_status(pod: Any) -> str:
    """Derive a human-readable status string for a pod (mirrors kubectl)."""
    if pod.metadata.deletion_timestamp:
        return "Terminating"
    if pod.status is None:
        return "Unknown"
    if pod.status.phase in ("Succeeded", "Failed"):
        return pod.status.phase
    # Check container statuses for waiting reasons
    for cs in pod.status.container_statuses or []:
        if cs.state and cs.state.waiting and cs.state.waiting.reason:
            return cs.state.waiting.reason
        if cs.state and cs.state.terminated and cs.state.terminated.reason:
            return cs.state.terminated.reason
    return pod.status.phase or "Unknown"


def _pod_restarts(pod: Any) -> int:
    """Total restart count across all containers in a pod."""
    total = 0
    for cs in (pod.status.container_statuses or []) if pod.status else []:
        total += cs.restart_count or 0
    return total


def _pod_ready_count(pod: Any) -> str:
    """Return READY column value like '1/1' or '0/2'."""
    containers = pod.spec.containers or []
    total = len(containers)
    ready = 0
    for cs in (pod.status.container_statuses or []) if pod.status else []:
        if cs.ready:
            ready += 1
    return f"{ready}/{total}"


# ── Table formatters per resource type ────────────────────────


def _format_pods_table(items: list[Any], wide: bool = False) -> str:
    """Format pod list as a kubectl-style table."""
    if not items:
        return "No resources found."
    lines: list[str] = []
    header = "NAME                                     READY   STATUS             RESTARTS   AGE"
    if wide:
        header += "   NODE                          IP"
    lines.append(header)
    for pod in items:
        name = pod.metadata.name or ""
        ready = _pod_ready_count(pod)
        status = _pod_status(pod)
        restarts = str(_pod_restarts(pod))
        age = _age(pod.metadata.creation_timestamp)
        line = f"{name:<41}{ready:<8}{status:<19}{restarts:<11}{age}"
        if wide:
            node = (pod.spec.node_name or "") if pod.spec else ""
            ip = (pod.status.pod_ip or "") if pod.status else ""
            line += f"   {node:<30}{ip}"
        lines.append(line)
    return "\n".join(lines)


def _format_deployments_table(items: list[Any], wide: bool = False) -> str:
    if not items:
        return "No resources found."
    lines: list[str] = []
    header = "NAME                                     READY   UP-TO-DATE   AVAILABLE   AGE"
    if wide:
        header += "   CONTAINERS        IMAGES"
    lines.append(header)
    for dep in items:
        name = dep.metadata.name or ""
        desired = dep.spec.replicas if dep.spec and dep.spec.replicas is not None else 0
        ready = dep.status.ready_replicas or 0 if dep.status else 0
        up_to_date = dep.status.updated_replicas or 0 if dep.status else 0
        available = dep.status.available_replicas or 0 if dep.status else 0
        age = _age(dep.metadata.creation_timestamp)
        line = f"{name:<41}{ready}/{desired:<6}{up_to_date:<13}{available:<12}{age}"
        if wide and dep.spec and dep.spec.template and dep.spec.template.spec:
            containers = [c.name for c in dep.spec.template.spec.containers or []]
            images = [c.image for c in dep.spec.template.spec.containers or []]
            line += f"   {','.join(containers):<17}{','.join(images)}"
        lines.append(line)
    return "\n".join(lines)


def _format_services_table(items: list[Any], wide: bool = False) -> str:
    if not items:
        return "No resources found."
    lines: list[str] = []
    header = "NAME                                     TYPE           CLUSTER-IP       EXTERNAL-IP      PORT(S)            AGE"
    if wide:
        header += "   SELECTOR"
    lines.append(header)
    for svc in items:
        name = svc.metadata.name or ""
        svc_type = svc.spec.type or "ClusterIP" if svc.spec else "ClusterIP"
        cluster_ip = svc.spec.cluster_ip or "<none>" if svc.spec else "<none>"
        # External IPs
        ext_ips: list[str] = []
        if svc.status and svc.status.load_balancer and svc.status.load_balancer.ingress:
            for ing in svc.status.load_balancer.ingress:
                ext_ips.append(ing.ip or ing.hostname or "")
        external_ip = ",".join(ext_ips) if ext_ips else "<none>"
        # Ports
        ports: list[str] = []
        for p in (svc.spec.ports or []) if svc.spec else []:
            port_str = f"{p.port}"
            if p.node_port:
                port_str += f":{p.node_port}"
            port_str += f"/{p.protocol or 'TCP'}"
            ports.append(port_str)
        ports_str = ",".join(ports) if ports else "<none>"
        age = _age(svc.metadata.creation_timestamp)
        line = f"{name:<41}{svc_type:<15}{cluster_ip:<17}{external_ip:<17}{ports_str:<19}{age}"
        if wide and svc.spec and svc.spec.selector:
            sel = ",".join(f"{k}={v}" for k, v in svc.spec.selector.items())
            line += f"   {sel}"
        lines.append(line)
    return "\n".join(lines)


def _format_nodes_table(items: list[Any], wide: bool = False) -> str:
    if not items:
        return "No resources found."
    lines: list[str] = []
    header = "NAME                                     STATUS   ROLES    AGE    VERSION"
    if wide:
        header += "   INTERNAL-IP       OS-IMAGE"
    lines.append(header)
    for node in items:
        name = node.metadata.name or ""
        # Status from conditions
        status = "Unknown"
        for cond in (node.status.conditions or []) if node.status else []:
            if cond.type == "Ready":
                status = "Ready" if cond.status == "True" else "NotReady"
                break
        roles_set: list[str] = []
        for label_key in (node.metadata.labels or {}):
            if label_key.startswith("node-role.kubernetes.io/"):
                roles_set.append(label_key.split("/")[-1])
        roles = ",".join(roles_set) if roles_set else "<none>"
        age = _age(node.metadata.creation_timestamp)
        version = node.status.node_info.kubelet_version if node.status and node.status.node_info else ""
        line = f"{name:<41}{status:<9}{roles:<9}{age:<7}{version}"
        if wide and node.status:
            int_ip = ""
            for addr in node.status.addresses or []:
                if addr.type == "InternalIP":
                    int_ip = addr.address
                    break
            os_image = node.status.node_info.os_image if node.status.node_info else ""
            line += f"   {int_ip:<17}{os_image}"
        lines.append(line)
    return "\n".join(lines)


def _format_generic_table(items: list[Any]) -> str:
    """Fallback table formatter for resource types without a custom formatter."""
    if not items:
        return "No resources found."
    lines: list[str] = []
    lines.append("NAME                                     NAMESPACE        AGE")
    for item in items:
        name = item.metadata.name or ""
        ns = item.metadata.namespace or ""
        age = _age(item.metadata.creation_timestamp)
        lines.append(f"{name:<41}{ns:<17}{age}")
    return "\n".join(lines)


def _format_items(resource: str, items: list[Any], output_format: str) -> str:
    """Format a list of K8s items into the requested output_format."""
    import json as _json

    if output_format == "json":
        # Use a single ApiClient for serialisation (not one per item)
        api = k8s_client.ApiClient()
        serialised = [api.sanitize_for_serialization(i) for i in items]
        return _json.dumps(serialised, indent=2, default=str)

    if output_format == "yaml":
        try:
            import yaml as _yaml  # noqa: WPS433
        except ImportError:
            return "PyYAML is not installed. Use output_format='json' instead."
        api = k8s_client.ApiClient()
        serialised = [api.sanitize_for_serialization(i) for i in items]
        return _yaml.dump_all(serialised, default_flow_style=False)

    # Table or wide
    wide = output_format == "wide"
    formatter = {
        "pods": _format_pods_table,
        "deployments": _format_deployments_table,
        "services": _format_services_table,
        "nodes": _format_nodes_table,
    }.get(resource)

    if formatter:
        return formatter(items, wide=wide)
    return _format_generic_table(items)


# ── Core list/get dispatch ────────────────────────────────────

def _list_resource(
    core_v1: Any,
    apps_v1: Any,
    custom_api: Any,
    resource: str,
    namespace: str,
    label_selector: str | None = None,
    field_selector: str | None = None,
    api_client: Any | None = None,
) -> Any:
    """Dispatch a list call to the correct API group and return the item list."""
    kwargs: dict[str, Any] = {}
    if label_selector:
        kwargs["label_selector"] = label_selector
    if field_selector:
        kwargs["field_selector"] = field_selector

    api_group = _RESOURCE_API_MAP.get(resource, "core")
    is_cluster_scoped = resource in ("nodes", "namespaces", "pv", "persistentvolumes")

    # ── Core V1 resources ─────────────────────────────────────
    if api_group == "core":
        method_map: dict[str, tuple[str, str]] = {
            "pods": ("list_namespaced_pod", "list_pod_for_all_namespaces"),
            "services": ("list_namespaced_service", "list_service_for_all_namespaces"),
            "configmaps": ("list_namespaced_config_map", "list_config_map_for_all_namespaces"),
            "secrets": ("list_namespaced_secret", "list_secret_for_all_namespaces"),
            "serviceaccounts": ("list_namespaced_service_account", "list_service_account_for_all_namespaces"),
            "endpoints": ("list_namespaced_endpoints", "list_endpoints_for_all_namespaces"),
            "pvc": ("list_namespaced_persistent_volume_claim", "list_persistent_volume_claim_for_all_namespaces"),
            "persistentvolumeclaims": ("list_namespaced_persistent_volume_claim", "list_persistent_volume_claim_for_all_namespaces"),
            "resourcequotas": ("list_namespaced_resource_quota", "list_resource_quota_for_all_namespaces"),
            "nodes": ("", "list_node"),
            "namespaces": ("", "list_namespace"),
            "pv": ("", "list_persistent_volume"),
            "persistentvolumes": ("", "list_persistent_volume"),
        }
        entry = method_map.get(resource)
        if not entry:
            return ToolResult(output=f"Unsupported core resource type: {resource}", error=True)
        namespaced_method, all_ns_method = entry

        if is_cluster_scoped:
            return getattr(core_v1, all_ns_method)(**kwargs)
        if namespace in ("", "all"):
            return getattr(core_v1, all_ns_method)(**kwargs)
        return getattr(core_v1, namespaced_method)(namespace=namespace, **kwargs)

    # ── Apps V1 resources ─────────────────────────────────────
    if api_group == "apps":
        method_map_apps: dict[str, tuple[str, str]] = {
            "deployments": ("list_namespaced_deployment", "list_deployment_for_all_namespaces"),
            "statefulsets": ("list_namespaced_stateful_set", "list_stateful_set_for_all_namespaces"),
            "daemonsets": ("list_namespaced_daemon_set", "list_daemon_set_for_all_namespaces"),
            "replicasets": ("list_namespaced_replica_set", "list_replica_set_for_all_namespaces"),
        }
        entry_apps = method_map_apps.get(resource)
        if not entry_apps:
            return ToolResult(output=f"Unsupported apps resource type: {resource}", error=True)
        ns_method, all_method = entry_apps
        if namespace in ("", "all"):
            return getattr(apps_v1, all_method)(**kwargs)
        return getattr(apps_v1, ns_method)(namespace=namespace, **kwargs)

    # ── Batch V1 resources ────────────────────────────────────
    if api_group == "batch":
        from kubernetes.client import BatchV1Api  # noqa: WPS433

        batch_v1 = BatchV1Api(api_client=api_client)
        method_map_batch: dict[str, tuple[str, str]] = {
            "jobs": ("list_namespaced_job", "list_job_for_all_namespaces"),
            "cronjobs": ("list_namespaced_cron_job", "list_cron_job_for_all_namespaces"),
        }
        entry_batch = method_map_batch.get(resource)
        if not entry_batch:
            return ToolResult(output=f"Unsupported batch resource type: {resource}", error=True)
        ns_method_b, all_method_b = entry_batch
        if namespace in ("", "all"):
            return getattr(batch_v1, all_method_b)(**kwargs)
        return getattr(batch_v1, ns_method_b)(namespace=namespace, **kwargs)

    # ── Autoscaling V2 resources ──────────────────────────────
    if api_group == "autoscaling":
        from kubernetes.client import AutoscalingV2Api  # noqa: WPS433

        auto_v2 = AutoscalingV2Api(api_client=api_client)
        if namespace in ("", "all"):
            return auto_v2.list_horizontal_pod_autoscaler_for_all_namespaces(**kwargs)
        return auto_v2.list_namespaced_horizontal_pod_autoscaler(namespace=namespace, **kwargs)

    # ── Networking V1 resources ───────────────────────────────
    if api_group == "networking":
        from kubernetes.client import NetworkingV1Api  # noqa: WPS433

        net_v1 = NetworkingV1Api(api_client=api_client)
        method_map_net: dict[str, tuple[str, str]] = {
            "ingress": ("list_namespaced_ingress", "list_ingress_for_all_namespaces"),
            "ingresses": ("list_namespaced_ingress", "list_ingress_for_all_namespaces"),
            "networkpolicies": ("list_namespaced_network_policy", "list_network_policy_for_all_namespaces"),
        }
        entry_net = method_map_net.get(resource)
        if not entry_net:
            return ToolResult(output=f"Unsupported networking resource type: {resource}", error=True)
        ns_method_n, all_method_n = entry_net
        if namespace in ("", "all"):
            return getattr(net_v1, all_method_n)(**kwargs)
        return getattr(net_v1, ns_method_n)(namespace=namespace, **kwargs)

    # ── Policy V1 resources ──────────────────────────────────
    if api_group == "policy":
        from kubernetes.client import PolicyV1Api  # noqa: WPS433

        policy_v1 = PolicyV1Api(api_client=api_client)
        if namespace in ("", "all"):
            return policy_v1.list_pod_disruption_budget_for_all_namespaces(**kwargs)
        return policy_v1.list_namespaced_pod_disruption_budget(namespace=namespace, **kwargs)

    return ToolResult(output=f"Unknown API group for resource: {resource}", error=True)


# ── Task 2.1 — kubectl_get ───────────────────────────────────


def kubectl_get(
    resource: str,
    *,
    gke_config: GKEConfig,
    name: str | None = None,
    namespace: str = "default",
    output_format: str = "table",
    label_selector: str | None = None,
    field_selector: str | None = None,
) -> ToolResult:
    """List or get Kubernetes resources (read-only kubectl get equivalent).

    Supports pods, deployments, services, configmaps, hpa, ingress, nodes,
    namespaces, statefulsets, daemonsets, jobs, cronjobs, pv, pvc, secrets,
    serviceaccounts, endpoints, networkpolicies, and replicasets.
    """
    if not _K8S_AVAILABLE:
        return _k8s_unavailable()

    resource = _normalise_resource(resource)
    if resource not in _RESOURCE_API_MAP:
        supported = sorted(_RESOURCE_API_MAP.keys())
        return ToolResult(
            output=f"Unsupported resource type: '{resource}'. Supported: {', '.join(supported)}",
            error=True,
        )

    if output_format not in ("table", "yaml", "json", "wide"):
        return ToolResult(
            output=f"Invalid output_format: '{output_format}'. Must be one of: table, yaml, json, wide",
            error=True,
        )

    # Use the config's default namespace when caller doesn't specify
    ns = namespace or gke_config.default_namespace

    result = _create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, apps_v1, custom_api, api_client_inst = result

    try:
        api_result = _list_resource(
            core_v1, apps_v1, custom_api, resource,
            namespace=ns,
            label_selector=label_selector,
            field_selector=field_selector,
            api_client=api_client_inst,
        )
        # _list_resource may return a ToolResult on unsupported resource
        if isinstance(api_result, ToolResult):
            return api_result

        items = api_result.items

        # If a specific name was requested, filter to that item
        if name:
            items = [i for i in items if i.metadata.name == name]
            if not items:
                return ToolResult(
                    output=f"{resource}/{name} not found in namespace '{ns}'",
                    error=True,
                )

        return ToolResult(output=_format_items(resource, items, output_format))

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(output=f"Namespace '{ns}' not found or resource type '{resource}' not available", error=True)
        if exc.status == 403:
            return ToolResult(output=f"Access denied: insufficient permissions to list {resource} in namespace '{ns}'", error=True)
        if exc.status == 401:
            return ToolResult(output="Authentication failed: check your kubeconfig or GKE credentials", error=True)
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:
        logger.exception("kubectl_get failed")
        return ToolResult(output=f"Error listing {resource}: {exc}", error=True)


# ── Task 2.2 — kubectl_describe ──────────────────────────────


def _describe_resource(
    core_v1: Any,
    apps_v1: Any,
    resource: str,
    name: str,
    namespace: str,
    api_client: Any | None = None,
) -> Any:
    """Read a single resource by name for describe output."""
    api_group = _RESOURCE_API_MAP.get(resource, "core")
    is_cluster_scoped = resource in ("nodes", "namespaces", "pv", "persistentvolumes")

    # ── Core V1 ───────────────────────────────────────────────
    if api_group == "core":
        read_map: dict[str, tuple[str, str]] = {
            "pods": ("read_namespaced_pod", ""),
            "services": ("read_namespaced_service", ""),
            "configmaps": ("read_namespaced_config_map", ""),
            "secrets": ("read_namespaced_secret", ""),
            "serviceaccounts": ("read_namespaced_service_account", ""),
            "endpoints": ("read_namespaced_endpoints", ""),
            "pvc": ("read_namespaced_persistent_volume_claim", ""),
            "persistentvolumeclaims": ("read_namespaced_persistent_volume_claim", ""),
            "resourcequotas": ("read_namespaced_resource_quota", ""),
            "nodes": ("", "read_node"),
            "namespaces": ("", "read_namespace"),
            "pv": ("", "read_persistent_volume"),
            "persistentvolumes": ("", "read_persistent_volume"),
        }
        entry = read_map.get(resource)
        if not entry:
            return None
        ns_method, cluster_method = entry
        if is_cluster_scoped:
            return getattr(core_v1, cluster_method)(name=name)
        return getattr(core_v1, ns_method)(name=name, namespace=namespace)

    # ── Apps V1 ───────────────────────────────────────────────
    if api_group == "apps":
        read_map_apps: dict[str, str] = {
            "deployments": "read_namespaced_deployment",
            "statefulsets": "read_namespaced_stateful_set",
            "daemonsets": "read_namespaced_daemon_set",
            "replicasets": "read_namespaced_replica_set",
        }
        method_name = read_map_apps.get(resource)
        if not method_name:
            return None
        return getattr(apps_v1, method_name)(name=name, namespace=namespace)

    # ── Batch V1 ──────────────────────────────────────────────
    if api_group == "batch":
        from kubernetes.client import BatchV1Api  # noqa: WPS433

        batch_v1 = BatchV1Api(api_client=api_client)
        read_map_batch: dict[str, str] = {
            "jobs": "read_namespaced_job",
            "cronjobs": "read_namespaced_cron_job",
        }
        method_name_b = read_map_batch.get(resource)
        if not method_name_b:
            return None
        return getattr(batch_v1, method_name_b)(name=name, namespace=namespace)

    # ── Autoscaling ───────────────────────────────────────────
    if api_group == "autoscaling":
        from kubernetes.client import AutoscalingV2Api  # noqa: WPS433

        return AutoscalingV2Api(api_client=api_client).read_namespaced_horizontal_pod_autoscaler(
            name=name, namespace=namespace,
        )

    # ── Networking ────────────────────────────────────────────
    if api_group == "networking":
        from kubernetes.client import NetworkingV1Api  # noqa: WPS433

        net_v1 = NetworkingV1Api(api_client=api_client)
        read_map_net: dict[str, str] = {
            "ingress": "read_namespaced_ingress",
            "ingresses": "read_namespaced_ingress",
            "networkpolicies": "read_namespaced_network_policy",
        }
        method_name_n = read_map_net.get(resource)
        if not method_name_n:
            return None
        return getattr(net_v1, method_name_n)(name=name, namespace=namespace)

    # ── Policy V1 ────────────────────────────────────────────
    if api_group == "policy":
        from kubernetes.client import PolicyV1Api  # noqa: WPS433

        return PolicyV1Api(api_client=api_client).read_namespaced_pod_disruption_budget(
            name=name, namespace=namespace,
        )

    return None


def _format_describe(resource: str, obj: Any, api_client: Any | None = None) -> str:
    """Format a single K8s resource object into a kubectl-describe-style output."""
    lines: list[str] = []
    meta = obj.metadata

    lines.append(f"Name:         {meta.name}")
    if meta.namespace:
        lines.append(f"Namespace:    {meta.namespace}")

    # Labels
    labels = meta.labels or {}
    lines.append("Labels:       " + (", ".join(f"{k}={v}" for k, v in sorted(labels.items())) if labels else "<none>"))

    # Annotations
    annotations = meta.annotations or {}
    lines.append("Annotations:  " + (", ".join(f"{k}={v}" for k, v in sorted(annotations.items())) if annotations else "<none>"))

    lines.append(f"CreationTimestamp: {meta.creation_timestamp}")

    # Resource-specific sections
    if resource == "pods" and obj.spec:
        lines.append(f"Node:         {obj.spec.node_name or '<none>'}")
        lines.append(f"Status:       {_pod_status(obj)}")
        if obj.status and obj.status.pod_ip:
            lines.append(f"IP:           {obj.status.pod_ip}")
        # Containers
        lines.append("Containers:")
        for c in obj.spec.containers or []:
            lines.append(f"  {c.name}:")
            lines.append(f"    Image:   {c.image}")
            if c.ports:
                ports = ", ".join(f"{p.container_port}/{p.protocol or 'TCP'}" for p in c.ports)
                lines.append(f"    Ports:   {ports}")
            if c.resources:
                if c.resources.requests:
                    lines.append(f"    Requests: {c.resources.requests}")
                if c.resources.limits:
                    lines.append(f"    Limits:   {c.resources.limits}")
        # Container statuses
        if obj.status and obj.status.container_statuses:
            lines.append("Container Statuses:")
            for cs in obj.status.container_statuses:
                lines.append(f"  {cs.name}:")
                lines.append(f"    Ready:    {cs.ready}")
                lines.append(f"    Restarts: {cs.restart_count}")
                if cs.state:
                    if cs.state.running:
                        lines.append(f"    State:    Running (since {cs.state.running.started_at})")
                    elif cs.state.waiting:
                        lines.append(f"    State:    Waiting ({cs.state.waiting.reason})")
                    elif cs.state.terminated:
                        lines.append(f"    State:    Terminated ({cs.state.terminated.reason})")

    elif resource == "deployments" and obj.spec:
        lines.append(f"Replicas:     {obj.spec.replicas} desired")
        if obj.status:
            lines.append(f"  Ready:      {obj.status.ready_replicas or 0}")
            lines.append(f"  Available:  {obj.status.available_replicas or 0}")
            lines.append(f"  Updated:    {obj.status.updated_replicas or 0}")
        if obj.spec.strategy:
            lines.append(f"Strategy:     {obj.spec.strategy.type}")

    elif resource == "services" and obj.spec:
        lines.append(f"Type:         {obj.spec.type}")
        lines.append(f"ClusterIP:    {obj.spec.cluster_ip}")
        if obj.spec.ports:
            for p in obj.spec.ports:
                port_info = f"{p.port}"
                if p.target_port:
                    port_info += f" → {p.target_port}"
                if p.node_port:
                    port_info += f" (NodePort: {p.node_port})"
                lines.append(f"  Port:       {p.name or ''} {port_info}/{p.protocol or 'TCP'}")
        if obj.spec.selector:
            lines.append(f"Selector:     {obj.spec.selector}")

    elif resource == "nodes" and obj.status:
        # Conditions
        lines.append("Conditions:")
        for cond in obj.status.conditions or []:
            lines.append(f"  {cond.type}: {cond.status} ({cond.reason or ''}) — {cond.message or ''}")
        # Allocatable
        if obj.status.allocatable:
            lines.append("Allocatable:")
            for k, v in sorted(obj.status.allocatable.items()):
                lines.append(f"  {k}: {v}")

    # Events — try to fetch events for the resource
    try:
        events_v1 = k8s_client.CoreV1Api(api_client=api_client) if api_client else k8s_client.CoreV1Api()
        field_sel = f"involvedObject.name={meta.name}"
        if meta.namespace:
            ev_list = events_v1.list_namespaced_event(
                namespace=meta.namespace, field_selector=field_sel,
            )
        else:
            ev_list = events_v1.list_event_for_all_namespaces(field_selector=field_sel)

        events = ev_list.items
        if events:
            lines.append("Events:")
            lines.append(f"  {'TYPE':<10}{'REASON':<25}{'AGE':<8}{'MESSAGE'}")
            for ev in events[-20:]:  # Last 20 events
                ev_type = ev.type or "Normal"
                reason = ev.reason or ""
                ev_age = _age(ev.last_timestamp or ev.metadata.creation_timestamp)
                message = ev.message or ""
                lines.append(f"  {ev_type:<10}{reason:<25}{ev_age:<8}{message}")
        else:
            lines.append("Events:       <none>")
    except Exception:
        logger.debug("Failed to retrieve events for %s/%s", resource, meta.name, exc_info=True)
        lines.append("Events:       <unable to retrieve>")

    return "\n".join(lines)


def kubectl_describe(
    resource: str,
    name: str,
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
) -> ToolResult:
    """Describe a Kubernetes resource in detail (read-only kubectl describe equivalent).

    Returns detailed info including labels, annotations, spec, status, conditions,
    and recent events.
    """
    if not _K8S_AVAILABLE:
        return _k8s_unavailable()

    resource = _normalise_resource(resource)
    if resource not in _RESOURCE_API_MAP:
        supported = sorted(_RESOURCE_API_MAP.keys())
        return ToolResult(
            output=f"Unsupported resource type: '{resource}'. Supported: {', '.join(supported)}",
            error=True,
        )

    ns = namespace or gke_config.default_namespace

    result = _create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, apps_v1, _, api_client_inst = result

    try:
        obj = _describe_resource(core_v1, apps_v1, resource, name, ns, api_client=api_client_inst)
        if obj is None:
            return ToolResult(output=f"Describe not supported for resource type: {resource}", error=True)
        return ToolResult(output=_format_describe(resource, obj, api_client=api_client_inst))

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(output=f"{resource}/{name} not found in namespace '{ns}'", error=True)
        if exc.status == 403:
            return ToolResult(output=f"Access denied: insufficient permissions to read {resource}/{name}", error=True)
        if exc.status == 401:
            return ToolResult(output="Authentication failed: check your kubeconfig or GKE credentials", error=True)
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:
        logger.exception("kubectl_describe failed")
        return ToolResult(output=f"Error describing {resource}/{name}: {exc}", error=True)


# ── Task 2.3 — kubectl_logs ──────────────────────────────────


def _parse_since(since: str) -> int | None:
    """Parse a duration string like '1h', '30m', '2h30m' into total seconds."""
    pattern = re.compile(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$")
    match = pattern.match(since.strip())
    if not match or not any(match.groups()):
        return None
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def kubectl_logs(
    pod: str,
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
    container: str | None = None,
    tail_lines: int = 100,
    since: str | None = None,
) -> ToolResult:
    """Retrieve logs from a pod (read-only kubectl logs equivalent).

    Handles CrashLoopBackOff by automatically fetching previous container logs
    when current logs are unavailable.
    """
    if not _K8S_AVAILABLE:
        return _k8s_unavailable()

    ns = namespace or gke_config.default_namespace
    tail = min(tail_lines, gke_config.log_limit) if gke_config.log_limit else tail_lines

    result = _create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, _, _, _ = result

    kwargs: dict[str, Any] = {
        "name": pod,
        "namespace": ns,
        "tail_lines": tail,
    }
    if container:
        kwargs["container"] = container
    if since:
        seconds = _parse_since(since)
        if seconds is None:
            return ToolResult(
                output=f"Invalid 'since' format: '{since}'. Use formats like '1h', '30m', '1h30m', '90s'",
                error=True,
            )
        kwargs["since_seconds"] = seconds

    try:
        logs = core_v1.read_namespaced_pod_log(**kwargs)
        if not logs:
            return ToolResult(output=f"(no logs available for pod/{pod})")
        return ToolResult(output=logs)

    except k8s_exceptions.ApiException as exc:
        # If the pod is in CrashLoopBackOff, try previous logs
        if exc.status == 400 and "previous" not in kwargs:
            try:
                kwargs["previous"] = True
                prev_logs = core_v1.read_namespaced_pod_log(**kwargs)
                if prev_logs:
                    return ToolResult(
                        output=f"[Previous container logs — current container may be crashing]\n{prev_logs}",
                    )
                return ToolResult(output=f"No current or previous logs available for pod/{pod}")
            except k8s_exceptions.ApiException:
                logger.debug("Failed to retrieve previous logs for pod/%s", pod, exc_info=True)
                return ToolResult(output=f"No current or previous logs available for pod/{pod}", error=True)

        if exc.status == 404:
            msg = f"Pod '{pod}' not found in namespace '{ns}'"
            if container:
                msg = f"Container '{container}' not found in pod '{pod}' (namespace '{ns}')"
            return ToolResult(output=msg, error=True)
        if exc.status == 403:
            return ToolResult(output=f"Access denied: insufficient permissions to read logs for pod/{pod}", error=True)
        if exc.status == 401:
            return ToolResult(output="Authentication failed: check your kubeconfig or GKE credentials", error=True)
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:
        logger.exception("kubectl_logs failed")
        return ToolResult(output=f"Error reading logs for pod/{pod}: {exc}", error=True)


# ── Task 2.4 — kubectl_top ───────────────────────────────────


def kubectl_top(
    resource_type: str = "pods",
    *,
    gke_config: GKEConfig,
    name: str | None = None,
    namespace: str = "default",
) -> ToolResult:
    """Show CPU and memory usage for pods or nodes (read-only kubectl top equivalent).

    Requires the Metrics Server to be installed in the cluster.
    """
    if not _K8S_AVAILABLE:
        return _k8s_unavailable()

    resource_type = resource_type.lower().strip()
    if resource_type not in ("pods", "pod", "nodes", "node"):
        return ToolResult(
            output=f"Invalid resource_type: '{resource_type}'. Must be 'pods' or 'nodes'",
            error=True,
        )
    is_pods = resource_type in ("pods", "pod")
    ns = namespace or gke_config.default_namespace

    result = _create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    _, _, custom_api, _ = result

    try:
        if is_pods:
            if ns in ("", "all"):
                metrics = custom_api.list_cluster_custom_object(
                    group="metrics.k8s.io", version="v1beta1", plural="pods",
                )
            else:
                metrics = custom_api.list_namespaced_custom_object(
                    group="metrics.k8s.io", version="v1beta1",
                    namespace=ns, plural="pods",
                )
        else:
            metrics = custom_api.list_cluster_custom_object(
                group="metrics.k8s.io", version="v1beta1", plural="nodes",
            )

        items = metrics.get("items", [])
        if name:
            items = [i for i in items if i.get("metadata", {}).get("name") == name]
            if not items:
                resource_label = "pod" if is_pods else "node"
                return ToolResult(
                    output=f"No metrics found for {resource_label}/{name}",
                    error=True,
                )

        if not items:
            return ToolResult(output="No metrics data available. Is metrics-server installed?")

        # Format output
        lines: list[str] = []
        if is_pods:
            lines.append(f"{'NAME':<50}{'CPU(cores)':<15}{'MEMORY(bytes)'}")
            for item in items:
                pod_name = item.get("metadata", {}).get("name", "")
                containers = item.get("containers", [])
                total_cpu = ""
                total_mem = ""
                for c in containers:
                    usage = c.get("usage", {})
                    cpu = usage.get("cpu", "0")
                    mem = usage.get("memory", "0")
                    total_cpu = cpu  # For single-container pods
                    total_mem = mem
                if len(containers) > 1:
                    # Sum across containers — simplistic display
                    total_cpu = f"{len(containers)} containers"
                    total_mem = f"{len(containers)} containers"
                lines.append(f"{pod_name:<50}{total_cpu:<15}{total_mem}")
        else:
            lines.append(f"{'NAME':<50}{'CPU(cores)':<15}{'CPU%':<10}{'MEMORY(bytes)':<17}{'MEMORY%'}")
            for item in items:
                node_name = item.get("metadata", {}).get("name", "")
                usage = item.get("usage", {})
                cpu = usage.get("cpu", "0")
                mem = usage.get("memory", "0")
                lines.append(f"{node_name:<50}{cpu:<15}{'N/A':<10}{mem:<17}{'N/A'}")

        return ToolResult(output="\n".join(lines))

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(
                output="Metrics API not available. Is the metrics-server installed? "
                       "Install with: kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml",
                error=True,
            )
        if exc.status in (401, 403):
            return ToolResult(output=f"Access denied to metrics API ({exc.status}): {exc.reason}.", error=True)
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:
        logger.exception("kubectl_top failed")
        return ToolResult(output=f"Error fetching metrics: {exc}", error=True)


# ══════════════════════════════════════════════════════════════
# WRITE OPERATIONS — safe, guarded K8s mutations
# ══════════════════════════════════════════════════════════════

# Allowed resource types for write operations (intentionally restrictive).
_SCALABLE_RESOURCES = frozenset({"deployments", "statefulsets", "replicasets"})
_RESTARTABLE_RESOURCES = frozenset({"deployments", "statefulsets", "daemonsets"})
_LABELABLE_RESOURCES = frozenset({
    "pods", "deployments", "services", "configmaps", "secrets",
    "statefulsets", "daemonsets", "namespaces", "nodes",
})

# Safety limits
_MAX_REPLICAS = 50
_MIN_REPLICAS = 0


def kubectl_scale(
    resource: str,
    name: str,
    replicas: int,
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
) -> ToolResult:
    """Scale a Kubernetes deployment, statefulset, or replicaset.

    Safety guardrails:
    - Only deployments, statefulsets, and replicasets can be scaled.
    - Replicas are clamped to 0-50 range.
    - The current replica count is reported in the response.
    """
    if not _K8S_AVAILABLE:
        return _k8s_unavailable()

    resource = _normalise_resource(resource)
    if resource not in _SCALABLE_RESOURCES:
        return ToolResult(
            output=f"Cannot scale '{resource}'. Scalable resources: {', '.join(sorted(_SCALABLE_RESOURCES))}",
            error=True,
        )

    if replicas < _MIN_REPLICAS or replicas > _MAX_REPLICAS:
        return ToolResult(
            output=f"Replicas must be between {_MIN_REPLICAS} and {_MAX_REPLICAS}. Got: {replicas}",
            error=True,
        )

    ns = namespace or gke_config.default_namespace
    result = _create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, apps_v1, _custom_api, _api_client = result

    try:
        # Read current state
        if resource == "deployments":
            obj = apps_v1.read_namespaced_deployment(name, ns)
        elif resource == "statefulsets":
            obj = apps_v1.read_namespaced_stateful_set(name, ns)
        else:  # replicasets
            obj = apps_v1.read_namespaced_replica_set(name, ns)

        current = obj.spec.replicas or 0

        # Apply scale
        body = {"spec": {"replicas": replicas}}
        if resource == "deployments":
            apps_v1.patch_namespaced_deployment_scale(name, ns, body)
        elif resource == "statefulsets":
            apps_v1.patch_namespaced_stateful_set_scale(name, ns, body)
        else:
            apps_v1.patch_namespaced_replica_set_scale(name, ns, body)

        return ToolResult(
            output=f"Scaled {resource}/{name} in namespace '{ns}': {current} -> {replicas} replicas",
        )

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(output=f"{resource}/{name} not found in namespace '{ns}'", error=True)
        if exc.status == 403:
            return ToolResult(output=f"Access denied: insufficient permissions to scale {resource}/{name}", error=True)
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:
        logger.exception("kubectl_scale failed")
        return ToolResult(output=f"Error scaling {resource}/{name}: {exc}", error=True)


def kubectl_restart(
    resource: str,
    name: str,
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
) -> ToolResult:
    """Trigger a rolling restart of a deployment, statefulset, or daemonset.

    This is equivalent to ``kubectl rollout restart``. It patches the pod
    template annotation with a timestamp, causing a rolling update.
    """
    if not _K8S_AVAILABLE:
        return _k8s_unavailable()

    resource = _normalise_resource(resource)
    if resource not in _RESTARTABLE_RESOURCES:
        return ToolResult(
            output=f"Cannot restart '{resource}'. Restartable resources: {', '.join(sorted(_RESTARTABLE_RESOURCES))}",
            error=True,
        )

    ns = namespace or gke_config.default_namespace
    result = _create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    _core_v1, apps_v1, _custom_api, _api_client = result

    now = datetime.now(timezone.utc).isoformat()
    patch_body = {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "kubectl.kubernetes.io/restartedAt": now,
                    },
                },
            },
        },
    }

    try:
        if resource == "deployments":
            apps_v1.patch_namespaced_deployment(name, ns, patch_body)
        elif resource == "statefulsets":
            apps_v1.patch_namespaced_stateful_set(name, ns, patch_body)
        else:  # daemonsets
            apps_v1.patch_namespaced_daemon_set(name, ns, patch_body)

        return ToolResult(
            output=f"Rolling restart triggered for {resource}/{name} in namespace '{ns}' at {now}",
        )

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(output=f"{resource}/{name} not found in namespace '{ns}'", error=True)
        if exc.status == 403:
            return ToolResult(output=f"Access denied: insufficient permissions to restart {resource}/{name}", error=True)
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:
        logger.exception("kubectl_restart failed")
        return ToolResult(output=f"Error restarting {resource}/{name}: {exc}", error=True)


def kubectl_label(
    resource: str,
    name: str,
    labels: str,
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
) -> ToolResult:
    """Add or update labels on a Kubernetes resource.

    Labels format: ``key1=value1,key2=value2``.
    To remove a label, use ``key-`` (e.g., ``obsolete-``).

    Safety guardrails:
    - Label keys and values are validated (alphanumeric, dashes, dots, underscores).
    - System labels (``kubernetes.io/``, ``k8s.io/``) cannot be modified.
    """
    if not _K8S_AVAILABLE:
        return _k8s_unavailable()

    resource = _normalise_resource(resource)
    if resource not in _LABELABLE_RESOURCES:
        return ToolResult(
            output=f"Cannot label '{resource}'. Supported: {', '.join(sorted(_LABELABLE_RESOURCES))}",
            error=True,
        )

    # Parse and validate labels
    label_dict: dict[str, str | None] = {}
    for part in labels.split(","):
        part = part.strip()
        if not part:
            continue
        if part.endswith("-"):
            # Remove label
            key = part[:-1]
            label_dict[key] = None
        elif "=" in part:
            key, value = part.split("=", 1)
            label_dict[key] = value
        else:
            return ToolResult(
                output=f"Invalid label format: '{part}'. Use 'key=value' to set or 'key-' to remove.",
                error=True,
            )

    if not label_dict:
        return ToolResult(output="No labels specified.", error=True)

    # Validate keys — block system labels
    _SYSTEM_PREFIXES = ("kubernetes.io/", "k8s.io/")
    for key in label_dict:
        if any(key.startswith(p) or ("/" in key and key.split("/")[0] + "/" in p) for p in _SYSTEM_PREFIXES):
            return ToolResult(
                output=f"Cannot modify system label: '{key}'. System labels (kubernetes.io/, k8s.io/) are managed by Kubernetes.",
                error=True,
            )
        # Validate label key format (simplified)
        label_key_part = key.split("/")[-1] if "/" in key else key
        if not re.match(r"^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?$", label_key_part):
            return ToolResult(
                output=f"Invalid label key: '{key}'. Must be alphanumeric with optional dashes, dots, or underscores.",
                error=True,
            )

    ns = namespace or gke_config.default_namespace
    result = _create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, apps_v1, _custom_api, _api_client = result

    patch_body = {"metadata": {"labels": label_dict}}

    try:
        # Route to the correct API
        _CORE_RESOURCES = {"pods", "services", "configmaps", "secrets", "namespaces", "nodes"}
        _APPS_RESOURCES = {"deployments", "statefulsets", "daemonsets"}

        if resource in _CORE_RESOURCES:
            if resource == "pods":
                core_v1.patch_namespaced_pod(name, ns, patch_body)
            elif resource == "services":
                core_v1.patch_namespaced_service(name, ns, patch_body)
            elif resource == "configmaps":
                core_v1.patch_namespaced_config_map(name, ns, patch_body)
            elif resource == "secrets":
                core_v1.patch_namespaced_secret(name, ns, patch_body)
            elif resource == "namespaces":
                core_v1.patch_namespace(name, patch_body)
            elif resource == "nodes":
                core_v1.patch_node(name, patch_body)
        elif resource in _APPS_RESOURCES:
            if resource == "deployments":
                apps_v1.patch_namespaced_deployment(name, ns, patch_body)
            elif resource == "statefulsets":
                apps_v1.patch_namespaced_stateful_set(name, ns, patch_body)
            elif resource == "daemonsets":
                apps_v1.patch_namespaced_daemon_set(name, ns, patch_body)

        applied = [f"{k}={v}" if v is not None else f"{k}-" for k, v in label_dict.items()]
        return ToolResult(
            output=f"Labels updated on {resource}/{name} in namespace '{ns}': {', '.join(applied)}",
        )

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(output=f"{resource}/{name} not found in namespace '{ns}'", error=True)
        if exc.status == 403:
            return ToolResult(output=f"Access denied: insufficient permissions to label {resource}/{name}", error=True)
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:
        logger.exception("kubectl_label failed")
        return ToolResult(output=f"Error labeling {resource}/{name}: {exc}", error=True)


def kubectl_annotate(
    resource: str,
    name: str,
    annotations: str,
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
) -> ToolResult:
    """Add or update annotations on a Kubernetes resource.

    Annotations format: ``key1=value1,key2=value2``.
    To remove an annotation, use ``key-`` (e.g., ``obsolete-``).

    Safety guardrails:
    - System annotations (``kubernetes.io/``, ``k8s.io/``) cannot be modified.
    """
    if not _K8S_AVAILABLE:
        return _k8s_unavailable()

    resource = _normalise_resource(resource)
    if resource not in _LABELABLE_RESOURCES:
        return ToolResult(
            output=f"Cannot annotate '{resource}'. Supported: {', '.join(sorted(_LABELABLE_RESOURCES))}",
            error=True,
        )

    # Parse annotations
    ann_dict: dict[str, str | None] = {}
    for part in annotations.split(","):
        part = part.strip()
        if not part:
            continue
        if part.endswith("-"):
            key = part[:-1]
            ann_dict[key] = None
        elif "=" in part:
            key, value = part.split("=", 1)
            ann_dict[key] = value
        else:
            return ToolResult(
                output=f"Invalid annotation format: '{part}'. Use 'key=value' to set or 'key-' to remove.",
                error=True,
            )

    if not ann_dict:
        return ToolResult(output="No annotations specified.", error=True)

    # Block system annotations
    _SYSTEM_PREFIXES = ("kubernetes.io/", "k8s.io/")
    for key in ann_dict:
        if any(key.startswith(p) for p in _SYSTEM_PREFIXES):
            return ToolResult(
                output=f"Cannot modify system annotation: '{key}'. System annotations (kubernetes.io/, k8s.io/) are managed by Kubernetes.",
                error=True,
            )

    ns = namespace or gke_config.default_namespace
    result = _create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, apps_v1, _custom_api, _api_client = result

    patch_body = {"metadata": {"annotations": ann_dict}}

    try:
        _CORE_RESOURCES = {"pods", "services", "configmaps", "secrets", "namespaces", "nodes"}
        _APPS_RESOURCES = {"deployments", "statefulsets", "daemonsets"}

        if resource in _CORE_RESOURCES:
            if resource == "pods":
                core_v1.patch_namespaced_pod(name, ns, patch_body)
            elif resource == "services":
                core_v1.patch_namespaced_service(name, ns, patch_body)
            elif resource == "configmaps":
                core_v1.patch_namespaced_config_map(name, ns, patch_body)
            elif resource == "secrets":
                core_v1.patch_namespaced_secret(name, ns, patch_body)
            elif resource == "namespaces":
                core_v1.patch_namespace(name, patch_body)
            elif resource == "nodes":
                core_v1.patch_node(name, patch_body)
        elif resource in _APPS_RESOURCES:
            if resource == "deployments":
                apps_v1.patch_namespaced_deployment(name, ns, patch_body)
            elif resource == "statefulsets":
                apps_v1.patch_namespaced_stateful_set(name, ns, patch_body)
            elif resource == "daemonsets":
                apps_v1.patch_namespaced_daemon_set(name, ns, patch_body)

        applied = [f"{k}={v}" if v is not None else f"{k}-" for k, v in ann_dict.items()]
        return ToolResult(
            output=f"Annotations updated on {resource}/{name} in namespace '{ns}': {', '.join(applied)}",
        )

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(output=f"{resource}/{name} not found in namespace '{ns}'", error=True)
        if exc.status == 403:
            return ToolResult(output=f"Access denied: insufficient permissions to annotate {resource}/{name}", error=True)
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:
        logger.exception("kubectl_annotate failed")
        return ToolResult(output=f"Error annotating {resource}/{name}: {exc}", error=True)


# ── Diagnostic tools — Phase 1 ────────────────────────────────


def get_events(
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
    event_type: str | None = None,
    involved_object_name: str | None = None,
    involved_object_kind: str | None = None,
    limit: int = 50,
) -> ToolResult:
    """List Kubernetes events in a namespace, optionally filtered by type and involved object.

    Events reveal WHY pods fail, WHY nodes have issues, and what the scheduler
    is doing. Critical for SRE triage. Equivalent to
    ``kubectl get events --sort-by=.lastTimestamp``.
    """
    if not _K8S_AVAILABLE:
        return _k8s_unavailable()

    if event_type is not None and event_type not in ("Warning", "Normal"):
        return ToolResult(
            output=f"Invalid event_type: '{event_type}'. Must be 'Warning', 'Normal', or omit for all.",
            error=True,
        )

    if limit < 1 or limit > 500:
        return ToolResult(
            output=f"Limit must be between 1 and 500. Got: {limit}",
            error=True,
        )

    ns = namespace or gke_config.default_namespace

    result = _create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, _, _, _ = result

    try:
        # Build field selector for server-side filtering
        field_parts: list[str] = []
        if event_type:
            field_parts.append(f"type={event_type}")
        if involved_object_name:
            field_parts.append(f"involvedObject.name={involved_object_name}")
        if involved_object_kind:
            field_parts.append(f"involvedObject.kind={involved_object_kind}")
        field_selector = ",".join(field_parts) if field_parts else None

        kwargs: dict[str, Any] = {"namespace": ns}
        if field_selector:
            kwargs["field_selector"] = field_selector

        ev_list = core_v1.list_namespaced_event(**kwargs)
        events = ev_list.items or []

        # Sort by last_timestamp descending (most recent first)
        def _sort_key(ev: Any) -> datetime:
            ts = ev.last_timestamp or ev.metadata.creation_timestamp
            if ts is None:
                return datetime.min.replace(tzinfo=timezone.utc)
            return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts

        events.sort(key=_sort_key, reverse=True)

        # Apply limit
        events = events[:limit]

        if not events:
            filter_desc = ""
            if event_type:
                filter_desc += f" type={event_type}"
            if involved_object_name:
                filter_desc += f" object={involved_object_name}"
            return ToolResult(
                output=f"No events found in namespace '{ns}'.{' Filters:' + filter_desc if filter_desc else ''}",
            )

        # Format as table
        lines: list[str] = []
        lines.append(f"{'LAST SEEN':<12}{'TYPE':<10}{'REASON':<25}{'OBJECT':<40}{'MESSAGE'}")
        for ev in events:
            last_seen = _age(ev.last_timestamp or ev.metadata.creation_timestamp)
            ev_type_str = ev.type or "Normal"
            reason = ev.reason or ""
            # Build OBJECT column: Kind/Name
            obj_kind = ev.involved_object.kind if ev.involved_object else ""
            obj_name = ev.involved_object.name if ev.involved_object else ""
            obj_str = f"{obj_kind}/{obj_name}" if obj_kind else obj_name
            message = ev.message or ""
            lines.append(f"{last_seen:<12}{ev_type_str:<10}{reason:<25}{obj_str:<40}{message}")

        header = f"Events in namespace '{ns}' ({len(events)} shown):"
        return ToolResult(output=f"{header}\n" + "\n".join(lines))

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(output=f"Namespace '{ns}' not found", error=True)
        if exc.status == 403:
            return ToolResult(output=f"Access denied: insufficient permissions to list events in namespace '{ns}'", error=True)
        if exc.status == 401:
            return ToolResult(output="Authentication failed: check your kubeconfig or GKE credentials", error=True)
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:
        logger.exception("get_events failed")
        return ToolResult(output=f"Error listing events: {exc}", error=True)


def get_rollout_status(
    name: str,
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
) -> ToolResult:
    """Check the rollout status of a Kubernetes deployment.

    Shows whether a deployment is progressing, complete, stalled, or failed.
    Reports replica counts, conditions, and rollout strategy.
    Equivalent to ``kubectl rollout status deployment/<name>``.
    """
    if not _K8S_AVAILABLE:
        return _k8s_unavailable()

    ns = namespace or gke_config.default_namespace

    result = _create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    _, apps_v1, _, _ = result

    try:
        dep = apps_v1.read_namespaced_deployment(name=name, namespace=ns)

        lines: list[str] = []
        lines.append(f"Deployment: {name}")
        lines.append(f"Namespace:  {ns}")

        # ── Replica counts ────────────────────────────────────
        desired = dep.spec.replicas if dep.spec and dep.spec.replicas is not None else 0
        status = dep.status
        current = status.replicas or 0 if status else 0
        ready = status.ready_replicas or 0 if status else 0
        updated = status.updated_replicas or 0 if status else 0
        available = status.available_replicas or 0 if status else 0
        unavailable = status.unavailable_replicas or 0 if status else 0

        lines.append("")
        lines.append("Replicas:")
        lines.append(f"  Desired:     {desired}")
        lines.append(f"  Current:     {current}")
        lines.append(f"  Ready:       {ready}")
        lines.append(f"  Updated:     {updated}")
        lines.append(f"  Available:   {available}")
        lines.append(f"  Unavailable: {unavailable}")

        # ── Rollout strategy ──────────────────────────────────
        if dep.spec and dep.spec.strategy:
            strategy = dep.spec.strategy
            lines.append("")
            lines.append(f"Strategy: {strategy.type or 'RollingUpdate'}")
            if strategy.rolling_update:
                ru = strategy.rolling_update
                lines.append(f"  Max Unavailable: {ru.max_unavailable}")
                lines.append(f"  Max Surge:       {ru.max_surge}")

        # ── Conditions ────────────────────────────────────────
        conditions = (status.conditions or []) if status else []
        overall_state = "Unknown"
        condition_details: list[str] = []

        progressing_cond = None
        available_cond = None
        failure_cond = None

        for cond in conditions:
            cond_type = cond.type or ""
            cond_status = cond.status or "Unknown"
            reason = cond.reason or ""
            message = cond.message or ""
            condition_details.append(f"  {cond_type}: {cond_status} — {reason}: {message}")

            if cond_type == "Progressing":
                progressing_cond = cond
            elif cond_type == "Available":
                available_cond = cond
            elif cond_type == "ReplicaFailure":
                failure_cond = cond

        # Determine overall state
        if failure_cond and failure_cond.status == "True":
            overall_state = "Failed"
        elif progressing_cond and progressing_cond.reason == "ProgressDeadlineExceeded":
            overall_state = "Stalled"
        elif (
            available_cond
            and available_cond.status == "True"
            and updated == desired
            and ready == desired
        ):
            overall_state = "Complete"
        elif progressing_cond and progressing_cond.status == "True":
            overall_state = "Progressing"
        elif desired == 0 and ready == 0:
            overall_state = "Scaled to zero"

        lines.append("")
        lines.append(f"Overall Status: {overall_state}")

        if condition_details:
            lines.append("")
            lines.append("Conditions:")
            lines.extend(condition_details)

        return ToolResult(output="\n".join(lines))

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(output=f"Deployment '{name}' not found in namespace '{ns}'", error=True)
        if exc.status == 403:
            return ToolResult(output=f"Access denied: insufficient permissions to read deployment/{name}", error=True)
        if exc.status == 401:
            return ToolResult(output="Authentication failed: check your kubeconfig or GKE credentials", error=True)
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:
        logger.exception("get_rollout_status failed")
        return ToolResult(output=f"Error checking rollout status for deployment/{name}: {exc}", error=True)


# ── Phase 2 diagnostic tools ─────────────────────────────────


def get_node_conditions(
    *,
    gke_config: GKEConfig,
    name: str | None = None,
) -> ToolResult:
    """Show node health conditions, resource pressure, taints, and capacity.

    When called WITHOUT a node name, lists ALL nodes with a summary: name,
    status (Ready/NotReady), roles, age, version, OS, kernel, container
    runtime, CPU capacity/allocatable, and memory capacity/allocatable.

    When called WITH a specific node name, shows a detailed view including
    ALL conditions (Ready, MemoryPressure, DiskPressure, PIDPressure,
    NetworkUnavailable), taints, relevant labels, allocatable vs capacity
    comparison, and the unschedulable (cordon) flag.

    Fills the gap left by ``kubectl_get nodes`` which only shows Ready/NotReady
    but hides pressure conditions that indicate imminent node failures.
    """
    if not _K8S_AVAILABLE:
        return _k8s_unavailable()

    result = _create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, _, _, _ = result

    try:
        if name:
            # ── Single node detail ────────────────────────────
            node = core_v1.read_node(name=name)
            return _format_node_detail(node)

        # ── All nodes summary ─────────────────────────────
        node_list = core_v1.list_node()
        nodes = node_list.items or []

        if not nodes:
            return ToolResult(output="No nodes found in the cluster.")

        return _format_nodes_summary(nodes)

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(output=f"Node '{name}' not found", error=True)
        if exc.status == 403:
            return ToolResult(
                output="Access denied: insufficient permissions to read nodes.",
                error=True,
            )
        if exc.status == 401:
            return ToolResult(
                output="Authentication failed: check your kubeconfig or GKE credentials",
                error=True,
            )
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:
        logger.exception("get_node_conditions failed")
        return ToolResult(output=f"Error reading node conditions: {exc}", error=True)


def _format_nodes_summary(nodes: list[Any]) -> ToolResult:
    """Format a list of nodes into a summary table with capacity info."""
    lines: list[str] = []
    header = (
        f"{'NAME':<40}{'STATUS':<12}{'ROLES':<12}{'AGE':<7}{'VERSION':<15}"
        f"{'OS':<12}{'KERNEL':<25}{'RUNTIME':<20}"
        f"{'CPU(cap/alloc)':<17}{'MEM(cap/alloc)'}"
    )
    lines.append(header)

    for node in nodes:
        nd_name = node.metadata.name or ""

        # Status from conditions
        status = "Unknown"
        for cond in (node.status.conditions or []) if node.status else []:
            if cond.type == "Ready":
                status = "Ready" if cond.status == "True" else "NotReady"
                break

        # Roles from labels
        roles_set: list[str] = []
        for label_key in node.metadata.labels or {}:
            if label_key.startswith("node-role.kubernetes.io/"):
                roles_set.append(label_key.split("/")[-1])
        roles = ",".join(roles_set) if roles_set else "<none>"

        age = _age(node.metadata.creation_timestamp)

        # Node info
        node_info = node.status.node_info if node.status and node.status.node_info else None
        version = node_info.kubelet_version if node_info else ""
        os_image = node_info.os_image if node_info else ""
        kernel = node_info.kernel_version if node_info else ""
        runtime = node_info.container_runtime_version if node_info else ""

        # Capacity / Allocatable
        capacity = node.status.capacity or {} if node.status else {}
        allocatable = node.status.allocatable or {} if node.status else {}
        cpu_cap = capacity.get("cpu", "?")
        cpu_alloc = allocatable.get("cpu", "?")
        mem_cap = _format_memory(capacity.get("memory", "?"))
        mem_alloc = _format_memory(allocatable.get("memory", "?"))

        cpu_str = f"{cpu_cap}/{cpu_alloc}"
        mem_str = f"{mem_cap}/{mem_alloc}"

        lines.append(
            f"{nd_name:<40}{status:<12}{roles:<12}{age:<7}{version:<15}"
            f"{os_image:<12}{kernel:<25}{runtime:<20}"
            f"{cpu_str:<17}{mem_str}"
        )

    return ToolResult(output=f"Nodes ({len(nodes)}):\n" + "\n".join(lines))


def _format_memory(mem_str: str) -> str:
    """Convert Kubernetes memory strings (e.g. '16384Ki') to human-readable."""
    if mem_str == "?" or not mem_str:
        return "?"
    # Handle Ki (kibibytes)
    if mem_str.endswith("Ki"):
        try:
            ki = int(mem_str[:-2])
            gi = ki / (1024 * 1024)
            if gi >= 1:
                return f"{gi:.1f}Gi"
            mi = ki / 1024
            return f"{mi:.0f}Mi"
        except ValueError:
            return mem_str
    # Handle Mi
    if mem_str.endswith("Mi"):
        try:
            mi = int(mem_str[:-2])
            gi = mi / 1024
            if gi >= 1:
                return f"{gi:.1f}Gi"
            return f"{mi}Mi"
        except ValueError:
            return mem_str
    # Handle Gi
    if mem_str.endswith("Gi"):
        return mem_str
    # Handle plain bytes
    try:
        b = int(mem_str)
        gi = b / (1024**3)
        if gi >= 1:
            return f"{gi:.1f}Gi"
        mi = b / (1024**2)
        return f"{mi:.0f}Mi"
    except ValueError:
        return mem_str


def _format_node_detail(node: Any) -> ToolResult:
    """Format a single node with full detail: conditions, taints, labels, capacity."""
    lines: list[str] = []
    nd_name = node.metadata.name or ""
    lines.append(f"Node: {nd_name}")

    # ── Basic info ────────────────────────────────────────
    node_info = node.status.node_info if node.status and node.status.node_info else None
    if node_info:
        lines.append(f"  Kubelet Version:         {node_info.kubelet_version}")
        lines.append(f"  OS Image:                {node_info.os_image}")
        lines.append(f"  Kernel Version:          {node_info.kernel_version}")
        lines.append(f"  Container Runtime:       {node_info.container_runtime_version}")
        lines.append(f"  Architecture:            {node_info.architecture}")
        lines.append(f"  Operating System:        {node_info.operating_system}")

    age = _age(node.metadata.creation_timestamp)
    lines.append(f"  Age:                     {age}")

    # ── Addresses ─────────────────────────────────────────
    addresses = node.status.addresses or [] if node.status else []
    if addresses:
        lines.append("")
        lines.append("Addresses:")
        for addr in addresses:
            lines.append(f"  {addr.type}: {addr.address}")

    # ── Unschedulable (cordon) ────────────────────────────
    unschedulable = node.spec.unschedulable if node.spec else False
    lines.append("")
    lines.append(f"Unschedulable (cordoned): {unschedulable or False}")

    # ── Conditions (ALL of them) ──────────────────────────
    conditions = (node.status.conditions or []) if node.status else []
    lines.append("")
    lines.append("Conditions:")
    if not conditions:
        lines.append("  (none)")
    else:
        lines.append(f"  {'TYPE':<25}{'STATUS':<10}{'REASON':<30}{'LAST TRANSITION':<22}{'MESSAGE'}")
        for cond in conditions:
            cond_type = cond.type or ""
            cond_status = cond.status or "Unknown"
            reason = cond.reason or ""
            message = cond.message or ""
            last_transition = _age(cond.last_transition_time) if cond.last_transition_time else "<unknown>"
            lines.append(
                f"  {cond_type:<25}{cond_status:<10}{reason:<30}{last_transition:<22}{message}"
            )

    # ── Taints ────────────────────────────────────────────
    taints = node.spec.taints or [] if node.spec else []
    lines.append("")
    lines.append("Taints:")
    if not taints:
        lines.append("  (none)")
    else:
        for taint in taints:
            taint_key = taint.key or ""
            taint_value = taint.value or ""
            taint_effect = taint.effect or ""
            val_str = f"={taint_value}" if taint_value else ""
            lines.append(f"  {taint_key}{val_str}:{taint_effect}")

    # ── Labels (relevant subset) ──────────────────────────
    labels = node.metadata.labels or {}
    relevant_prefixes = (
        "node-role.kubernetes.io/",
        "topology.kubernetes.io/",
        "cloud.google.com/",
        "node.kubernetes.io/",
        "beta.kubernetes.io/",
    )
    relevant_keys = (
        "kubernetes.io/arch",
        "kubernetes.io/os",
        "kubernetes.io/hostname",
    )
    lines.append("")
    lines.append("Labels (relevant):")
    found_labels = False
    for k, v in sorted(labels.items()):
        if k.startswith(relevant_prefixes) or k in relevant_keys:
            lines.append(f"  {k}={v}")
            found_labels = True
    if not found_labels:
        lines.append("  (none matching filter)")

    # ── Capacity vs Allocatable ───────────────────────────
    capacity = node.status.capacity or {} if node.status else {}
    allocatable = node.status.allocatable or {} if node.status else {}
    lines.append("")
    lines.append("Capacity vs Allocatable:")
    lines.append(f"  {'RESOURCE':<25}{'CAPACITY':<20}{'ALLOCATABLE'}")
    # Show common resources
    resource_keys = sorted(set(list(capacity.keys()) + list(allocatable.keys())))
    for rk in resource_keys:
        cap_val = capacity.get(rk, "-")
        alloc_val = allocatable.get(rk, "-")
        # Human-readable memory
        if "memory" in rk.lower() or rk == "hugepages-2Mi" or rk == "hugepages-1Gi":
            cap_val = _format_memory(str(cap_val))
            alloc_val = _format_memory(str(alloc_val))
        lines.append(f"  {rk:<25}{cap_val:<20}{alloc_val}")

    return ToolResult(output="\n".join(lines))


def get_container_status(
    name: str,
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
) -> ToolResult:
    """Show detailed container-level status for ALL containers in a pod.

    Covers init containers, regular containers, and ephemeral containers.
    For each container shows: name, image, state (Waiting/Running/Terminated
    with details), ready flag, restart count, last termination state (crucial
    for CrashLoopBackOff debugging), resource requests/limits, volume mounts,
    and environment variable sources (configMapRef/secretRef names only — no
    secret values exposed).

    Essential for debugging multi-container pods where ``kubectl_get pods``
    only shows pod-level status.
    """
    if not _K8S_AVAILABLE:
        return _k8s_unavailable()

    ns = namespace or gke_config.default_namespace

    result = _create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, _, _, _ = result

    try:
        pod = core_v1.read_namespaced_pod(name=name, namespace=ns)

        lines: list[str] = []
        lines.append(f"Pod: {name}")
        lines.append(f"Namespace: {ns}")
        lines.append(f"Node: {pod.spec.node_name or '<unassigned>'}")
        lines.append(f"Phase: {pod.status.phase or 'Unknown' if pod.status else 'Unknown'}")
        lines.append("")

        # Build status maps for quick lookup
        container_statuses = {cs.name: cs for cs in (pod.status.container_statuses or []) if pod.status} if pod.status else {}
        init_statuses = {cs.name: cs for cs in (pod.status.init_container_statuses or []) if pod.status} if pod.status else {}
        ephemeral_statuses = {cs.name: cs for cs in (pod.status.ephemeral_container_statuses or []) if pod.status} if pod.status else {}

        # ── Init Containers ───────────────────────────────
        init_containers = pod.spec.init_containers or [] if pod.spec else []
        if init_containers:
            lines.append("=== Init Containers ===")
            for c in init_containers:
                cs = init_statuses.get(c.name)
                _format_container_section(c, cs, lines)
                lines.append("")

        # ── Regular Containers ────────────────────────────
        containers = pod.spec.containers or [] if pod.spec else []
        if containers:
            lines.append("=== Containers ===")
            for c in containers:
                cs = container_statuses.get(c.name)
                _format_container_section(c, cs, lines)
                lines.append("")

        # ── Ephemeral Containers ──────────────────────────
        ephemeral_containers = pod.spec.ephemeral_containers or [] if pod.spec else []
        if ephemeral_containers:
            lines.append("=== Ephemeral Containers ===")
            for c in ephemeral_containers:
                cs = ephemeral_statuses.get(c.name)
                _format_container_section(c, cs, lines)
                lines.append("")

        if not init_containers and not containers and not ephemeral_containers:
            lines.append("No containers found in pod spec.")

        return ToolResult(output="\n".join(lines))

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(output=f"Pod '{name}' not found in namespace '{ns}'", error=True)
        if exc.status == 403:
            return ToolResult(
                output=f"Access denied: insufficient permissions to read pod/{name}",
                error=True,
            )
        if exc.status == 401:
            return ToolResult(
                output="Authentication failed: check your kubeconfig or GKE credentials",
                error=True,
            )
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:
        logger.exception("get_container_status failed")
        return ToolResult(output=f"Error reading container status for pod/{name}: {exc}", error=True)


def _format_container_section(container: Any, status: Any | None, lines: list[str]) -> None:
    """Format a single container's detail into output lines."""
    c_name = container.name or ""
    lines.append(f"  Container: {c_name}")
    lines.append(f"    Image: {container.image or '<none>'}")

    if status:
        lines.append(f"    Image ID: {status.image_id or '<none>'}")
        lines.append(f"    Ready: {status.ready if status.ready is not None else 'N/A'}")
        lines.append(f"    Restart Count: {status.restart_count or 0}")

        # Current state
        state = status.state
        if state:
            if state.running:
                started = _age(state.running.started_at) if state.running.started_at else "<unknown>"
                lines.append(f"    State: Running (started {started} ago)")
            elif state.waiting:
                reason = state.waiting.reason or "Unknown"
                message = state.waiting.message or ""
                lines.append(f"    State: Waiting — {reason}")
                if message:
                    lines.append(f"      Message: {message}")
            elif state.terminated:
                t = state.terminated
                reason = t.reason or "Unknown"
                exit_code = t.exit_code if t.exit_code is not None else "?"
                lines.append(f"    State: Terminated — {reason} (exit code {exit_code})")
                if t.message:
                    lines.append(f"      Message: {t.message}")
                if t.started_at:
                    lines.append(f"      Started:  {t.started_at}")
                if t.finished_at:
                    lines.append(f"      Finished: {t.finished_at}")
            else:
                lines.append("    State: Unknown")

        # Last termination state (crucial for CrashLoopBackOff)
        last_state = status.last_state
        if last_state and last_state.terminated:
            lt = last_state.terminated
            reason = lt.reason or "Unknown"
            exit_code = lt.exit_code if lt.exit_code is not None else "?"
            lines.append(f"    Last State: Terminated — {reason} (exit code {exit_code})")
            if lt.message:
                lines.append(f"      Message: {lt.message}")
            if lt.started_at:
                lines.append(f"      Started:  {lt.started_at}")
            if lt.finished_at:
                lines.append(f"      Finished: {lt.finished_at}")
    else:
        lines.append("    (no status available)")

    # ── Resource requests/limits ──────────────────────────
    resources = container.resources
    if resources:
        requests = resources.requests or {}
        limits = resources.limits or {}
        if requests or limits:
            lines.append("    Resources:")
            if requests:
                parts = [f"{k}={v}" for k, v in requests.items()]
                lines.append(f"      Requests: {', '.join(parts)}")
            if limits:
                parts = [f"{k}={v}" for k, v in limits.items()]
                lines.append(f"      Limits:   {', '.join(parts)}")

    # ── Volume mounts ─────────────────────────────────────
    mounts = container.volume_mounts or []
    if mounts:
        lines.append("    Volume Mounts:")
        for m in mounts:
            ro = " (ro)" if m.read_only else ""
            lines.append(f"      {m.mount_path} from {m.name}{ro}")

    # ── Env from (configMapRef / secretRef — names only) ──
    env_from = container.env_from or []
    if env_from:
        lines.append("    Env From:")
        for ef in env_from:
            if ef.config_map_ref:
                lines.append(f"      ConfigMap: {ef.config_map_ref.name}")
            if ef.secret_ref:
                lines.append(f"      Secret: {ef.secret_ref.name} (ref only, no values shown)")

    # ── Env vars with valueFrom (names only) ──────────────
    env_vars = container.env or []
    env_refs: list[str] = []
    for ev in env_vars:
        if ev.value_from:
            if ev.value_from.config_map_key_ref:
                ref = ev.value_from.config_map_key_ref
                env_refs.append(f"      {ev.name} ← ConfigMap:{ref.name}/{ref.key}")
            elif ev.value_from.secret_key_ref:
                ref = ev.value_from.secret_key_ref
                env_refs.append(f"      {ev.name} ← Secret:{ref.name}/{ref.key} (ref only)")
    if env_refs:
        lines.append("    Env Var References:")
        lines.extend(env_refs)


# ── Phase 3: exec_command security model ─────────────────────

# Denylist — checked FIRST.  If any pattern matches, the command is rejected.
DENIED_PATTERNS: list[str] = [
    r"[;&|`$]",       # shell metacharacters / chaining / substitution
    r">\s*",           # output redirection (overwrite)
    r">>\s*",          # output redirection (append)
    r"\brm\b",         # file deletion
    r"\bkill\b",       # process killing
    r"\bshutdown\b",   # system shutdown
    r"\breboot\b",     # system reboot
    r"\bdd\b",         # raw disk write
    r"\bmkfs\b",       # filesystem creation
    r"\bfdisk\b",      # partition editing
    r"\bchmod\b",      # permission changes
    r"\bchown\b",      # ownership changes
    r"\bsudo\b",       # privilege escalation
]
_COMPILED_DENY = [re.compile(p) for p in DENIED_PATTERNS]

# Allowlist — checked SECOND.  The command must start with one of these prefixes.
ALLOWED_EXEC_COMMANDS: list[str] = [
    "cat", "head", "tail", "ls", "env", "printenv",
    "whoami", "id", "hostname", "date",
    "ps", "top -bn1",
    "df", "du",
    "mount",
    "ip", "ifconfig", "netstat", "ss", "nslookup", "dig", "ping", "curl", "wget", "nc",
    "java -version", "python --version", "node --version",
    "cat /etc/resolv.conf", "cat /etc/hosts",
]

_MAX_EXEC_OUTPUT = 10_000


def _check_denied(command: str) -> str | None:
    """Return a reason string if *command* matches any deny pattern, else ``None``."""
    for pattern, compiled in zip(DENIED_PATTERNS, _COMPILED_DENY):
        if compiled.search(command):
            return f"Command denied — matches blocked pattern: {pattern}"
    return None


def _check_allowed(command: str) -> bool:
    """Return ``True`` if *command* starts with an allowed prefix."""
    cmd_stripped = command.strip()
    return any(cmd_stripped == prefix or cmd_stripped.startswith(prefix + " ") for prefix in ALLOWED_EXEC_COMMANDS)


def exec_command(
    pod_name: str,
    command: str,
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
    container: str | None = None,
    timeout: int = 30,
) -> ToolResult:
    """Execute a diagnostic command inside a running container.

    Uses ``kubernetes.stream.stream()`` with
    ``CoreV1Api.connect_get_namespaced_pod_exec``.

    Security model:
    1. ``gke_config.exec_enabled`` must be ``True`` (disabled by default).
    2. Command is checked against a **denylist** of dangerous patterns.
    3. Command must start with an **allowed prefix** (read-only diagnostics).
    4. Output is truncated to *_MAX_EXEC_OUTPUT* characters.
    """
    if not _K8S_AVAILABLE:
        return _k8s_unavailable()

    # ── Gate: exec must be explicitly enabled ──────────────
    if not gke_config.exec_enabled:
        return ToolResult(
            output=(
                "exec_command is disabled. Set gke.exec_enabled=true in your config "
                "to enable container command execution."
            ),
            error=True,
        )

    # ── Validate command ──────────────────────────────────
    if not command or not command.strip():
        return ToolResult(output="Command cannot be empty.", error=True)

    denied_reason = _check_denied(command)
    if denied_reason:
        return ToolResult(output=denied_reason, error=True)

    if not _check_allowed(command):
        return ToolResult(
            output=(
                f"Command not in allowlist. Allowed prefixes: {', '.join(ALLOWED_EXEC_COMMANDS)}. "
                f"Got: '{command.split()[0]}'"
            ),
            error=True,
        )

    # ── Validate timeout ──────────────────────────────────
    if timeout < 1 or timeout > 300:
        return ToolResult(
            output=f"Timeout must be between 1 and 300 seconds. Got: {timeout}",
            error=True,
        )

    ns = namespace or gke_config.default_namespace

    result = _create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, _, _, _ = result

    try:
        # Lazy-import stream to keep it optional
        from kubernetes.stream import stream as k8s_stream  # noqa: WPS433

        exec_kwargs: dict[str, Any] = {
            "name": pod_name,
            "namespace": ns,
            "command": ["/bin/sh", "-c", command],
            "stderr": True,
            "stdin": False,
            "stdout": True,
            "tty": False,
            "_request_timeout": timeout,
        }
        if container:
            exec_kwargs["container"] = container

        resp = k8s_stream(
            core_v1.connect_get_namespaced_pod_exec,
            **exec_kwargs,
        )

        # resp is the full combined output as a string
        stdout = resp if isinstance(resp, str) else str(resp)
        stderr = ""  # stream() merges stderr into the response string

        # Truncate if needed
        if len(stdout) > _MAX_EXEC_OUTPUT:
            stdout = stdout[:_MAX_EXEC_OUTPUT] + f"\n... [truncated at {_MAX_EXEC_OUTPUT} chars]"

        lines: list[str] = []
        lines.append(f"=== exec_command: {command} ===")
        lines.append(f"Pod: {pod_name} | Namespace: {ns}")
        if container:
            lines.append(f"Container: {container}")
        lines.append("")
        if stdout.strip():
            lines.append("--- stdout ---")
            lines.append(stdout)
        if stderr.strip():
            lines.append("--- stderr ---")
            lines.append(stderr)
        if not stdout.strip() and not stderr.strip():
            lines.append("(no output)")

        return ToolResult(output="\n".join(lines), error=False)

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            msg = f"Pod '{pod_name}' not found in namespace '{ns}'."
            if container:
                msg += f" (container: {container})"
            return ToolResult(output=msg, error=True)
        if exc.status in (401, 403):
            return ToolResult(
                output=f"Permission denied executing command on pod '{pod_name}': {exc.reason}",
                error=True,
            )
        return ToolResult(output=f"K8s API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:  # noqa: BLE001
        return ToolResult(output=f"Error executing command: {exc}", error=True)


# ── Phase 3: check_rbac ──────────────────────────────────────


def check_rbac(
    verb: str,
    resource: str,
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
    service_account: str | None = None,
    resource_name: str | None = None,
) -> ToolResult:
    """Check whether a service account has permission to perform a specific action.

    Uses ``AuthorizationV1Api.create_namespaced_subject_access_review()``
    for specific service accounts, or
    ``create_self_subject_access_review()`` for the current user.

    Read-only — only checks permissions, does not modify any resources.
    """
    if not _K8S_AVAILABLE:
        return _k8s_unavailable()

    # Normalise resource alias (e.g. "po" → "pods")
    normalised = _normalise_resource(resource)

    ns = namespace or gke_config.default_namespace

    result = _create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    _, _, _, api_client = result

    try:
        auth_v1 = k8s_client.AuthorizationV1Api(api_client)

        if service_account:
            # SubjectAccessReview — check for a specific service account
            sar_spec = k8s_client.V1SubjectAccessReviewSpec(
                user=f"system:serviceaccount:{ns}:{service_account}",
                resource_attributes=k8s_client.V1ResourceAttributes(
                    namespace=ns,
                    verb=verb,
                    resource=normalised,
                    name=resource_name or "",
                ),
            )
            sar = k8s_client.V1SubjectAccessReview(spec=sar_spec)
            review = auth_v1.create_subject_access_review(body=sar)
        else:
            # SelfSubjectAccessReview — check for the current user
            self_spec = k8s_client.V1SelfSubjectAccessReviewSpec(
                resource_attributes=k8s_client.V1ResourceAttributes(
                    namespace=ns,
                    verb=verb,
                    resource=normalised,
                    name=resource_name or "",
                ),
            )
            self_sar = k8s_client.V1SelfSubjectAccessReview(spec=self_spec)
            review = auth_v1.create_self_subject_access_review(body=self_sar)

        status = review.status
        allowed = status.allowed if status else False
        reason = status.reason or "" if status else ""
        denied = getattr(status, "denied", False) if status else False
        evaluation_error = getattr(status, "evaluation_error", "") if status else ""

        lines: list[str] = []
        lines.append("=== RBAC Permission Check ===")
        if service_account:
            lines.append(f"Subject: system:serviceaccount:{ns}:{service_account}")
        else:
            lines.append("Subject: current user (self)")
        lines.append(f"Action: {verb} {normalised}")
        if resource_name:
            lines.append(f"Resource Name: {resource_name}")
        lines.append(f"Namespace: {ns}")
        lines.append("")
        lines.append(f"Allowed: {'YES' if allowed else 'NO'}")
        if denied:
            lines.append("Explicitly Denied: YES")
        if reason:
            lines.append(f"Reason: {reason}")
        if evaluation_error:
            lines.append(f"Evaluation Error: {evaluation_error}")

        return ToolResult(output="\n".join(lines), error=False)

    except k8s_exceptions.ApiException as exc:
        if exc.status in (401, 403):
            return ToolResult(
                output=f"Permission denied performing RBAC check: {exc.reason}",
                error=True,
            )
        return ToolResult(output=f"K8s API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:  # noqa: BLE001
        return ToolResult(output=f"Error checking RBAC: {exc}", error=True)


# ── get_rollout_history ──────────────────────────────────────


def get_rollout_history(
    name: str,
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
    revision: int | None = None,
) -> ToolResult:
    """Show the revision history of a Kubernetes deployment.

    Lists all revisions by examining ReplicaSets owned by the deployment,
    similar to ``kubectl rollout history deployment/<name>``.
    When a specific revision number is provided, shows detailed pod template
    information for that revision (containers, images, ports, env var names,
    volume mounts, resource requests/limits).

    Use BEFORE recommending a rollback so you know what changed in each revision.
    """
    if not _K8S_AVAILABLE:
        return _k8s_unavailable()

    ns = namespace or gke_config.default_namespace

    result = _create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    _, apps_v1, _, _ = result

    try:
        # Verify the deployment exists first
        dep = apps_v1.read_namespaced_deployment(name=name, namespace=ns)

        # List all ReplicaSets in the namespace
        all_rs = apps_v1.list_namespaced_replica_set(namespace=ns)

        # Filter to ReplicaSets owned by this deployment
        owned_rs: list = []
        for rs in all_rs.items:
            for owner in (rs.metadata.owner_references or []):
                if owner.kind == "Deployment" and owner.name == name:
                    rev_str = (rs.metadata.annotations or {}).get(
                        "deployment.kubernetes.io/revision", ""
                    )
                    if rev_str:
                        owned_rs.append((int(rev_str), rs))
                    break

        if not owned_rs:
            return ToolResult(
                output=f"Deployment '{name}' in namespace '{ns}' has no revision history (no ReplicaSets found).",
                error=False,
            )

        # Sort by revision number descending (newest first)
        owned_rs.sort(key=lambda pair: pair[0], reverse=True)

        # ── Specific revision detail ──────────────────────
        if revision is not None:
            match = [rs for rev, rs in owned_rs if rev == revision]
            if not match:
                available_revs = ", ".join(str(rev) for rev, _ in owned_rs)
                return ToolResult(
                    output=(
                        f"Revision {revision} not found for deployment '{name}' "
                        f"in namespace '{ns}'. Available revisions: {available_revs}"
                    ),
                    error=True,
                )
            rs = match[0]
            return _format_revision_detail(name, ns, revision, rs)

        # ── List all revisions ────────────────────────────
        # Determine which revision is active (highest replica count > 0)
        current_rev = _find_current_revision(dep, owned_rs)

        lines: list[str] = []
        lines.append(f"=== Rollout History: {name} (namespace: {ns}) ===")
        lines.append("")
        lines.append(f"{'REVISION':<10} {'IMAGE(S)':<50} {'CREATED':<12} {'REPLICAS':<10} {'STATUS'}")
        lines.append("-" * 100)

        for rev_num, rs in owned_rs:
            # Images
            containers = []
            if rs.spec and rs.spec.template and rs.spec.template.spec:
                containers = rs.spec.template.spec.containers or []
            images = ", ".join(c.image or "<none>" for c in containers) if containers else "<none>"
            if len(images) > 48:
                images = images[:45] + "..."

            # Created
            created = _age(rs.metadata.creation_timestamp) if rs.metadata.creation_timestamp else "<unknown>"

            # Replicas
            replicas = rs.status.replicas if rs.status and rs.status.replicas else 0

            # Status
            if rev_num == current_rev:
                status = "active"
            elif replicas == 0:
                status = "scaled-down"
            else:
                status = f"{replicas} replicas"

            lines.append(f"{rev_num:<10} {images:<50} {created:<12} {replicas:<10} {status}")

        change_cause = (dep.metadata.annotations or {}).get("kubernetes.io/change-cause", "")
        if change_cause:
            lines.append("")
            lines.append(f"Last change cause: {change_cause}")

        lines.append("")
        lines.append(f"Total revisions: {len(owned_rs)}")
        lines.append("Tip: Use get_rollout_history with a specific revision number for detailed info.")

        return ToolResult(output="\n".join(lines), error=False)

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(output=f"Deployment '{name}' not found in namespace '{ns}'", error=True)
        if exc.status == 403:
            return ToolResult(
                output=f"Access denied: insufficient permissions to read deployment/{name} or its ReplicaSets",
                error=True,
            )
        if exc.status == 401:
            return ToolResult(output="Authentication failed: check your kubeconfig or GKE credentials", error=True)
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_rollout_history failed")
        return ToolResult(output=f"Error retrieving rollout history for deployment/{name}: {exc}", error=True)


def _find_current_revision(dep: Any, owned_rs: list[tuple[int, Any]]) -> int | None:
    """Find the revision number of the currently active ReplicaSet."""
    # The deployment's own annotation tracks the current revision
    rev_str = (dep.metadata.annotations or {}).get("deployment.kubernetes.io/revision", "")
    if rev_str:
        try:
            return int(rev_str)
        except (ValueError, TypeError):
            pass
    # Fallback: highest revision with replicas > 0
    for rev_num, rs in owned_rs:
        replicas = rs.status.replicas if rs.status and rs.status.replicas else 0
        if replicas > 0:
            return rev_num
    return None


def _format_revision_detail(name: str, ns: str, revision: int, rs: Any) -> ToolResult:
    """Format detailed information for a specific revision."""
    lines: list[str] = []
    lines.append(f"=== Revision {revision} Detail: {name} (namespace: {ns}) ===")
    lines.append("")

    # ReplicaSet info
    rs_name = rs.metadata.name if rs.metadata else "<unknown>"
    lines.append(f"ReplicaSet: {rs_name}")
    created = _age(rs.metadata.creation_timestamp) if rs.metadata and rs.metadata.creation_timestamp else "<unknown>"
    lines.append(f"Created: {created} ago")

    # Replica counts
    replicas = rs.status.replicas if rs.status and rs.status.replicas else 0
    ready = rs.status.ready_replicas if rs.status and rs.status.ready_replicas else 0
    lines.append(f"Replicas: {replicas} current / {ready} ready")

    # Change cause annotation
    change_cause = (rs.metadata.annotations or {}).get("kubernetes.io/change-cause", "")
    if change_cause:
        lines.append(f"Change Cause: {change_cause}")

    # Pod template containers
    containers = []
    if rs.spec and rs.spec.template and rs.spec.template.spec:
        containers = rs.spec.template.spec.containers or []

    if not containers:
        lines.append("")
        lines.append("No container spec found.")
        return ToolResult(output="\n".join(lines), error=False)

    lines.append("")
    lines.append("Containers:")
    for c in containers:
        c_name = c.name if hasattr(c, "name") else "<unnamed>"
        lines.append(f"  Container: {c_name}")
        lines.append(f"    Image: {c.image or '<none>'}")

        # Ports
        ports = c.ports or [] if hasattr(c, "ports") else []
        if ports:
            port_strs = []
            for p in ports:
                proto = p.protocol or "TCP" if hasattr(p, "protocol") else "TCP"
                port_strs.append(f"{p.container_port}/{proto}")
            lines.append(f"    Ports: {', '.join(port_strs)}")

        # Resource requests/limits
        resources = c.resources if hasattr(c, "resources") and c.resources else None
        if resources:
            requests = resources.requests or {} if hasattr(resources, "requests") else {}
            limits = resources.limits or {} if hasattr(resources, "limits") else {}
            if requests or limits:
                lines.append("    Resources:")
                if requests:
                    parts = [f"{k}={v}" for k, v in requests.items()]
                    lines.append(f"      Requests: {', '.join(parts)}")
                if limits:
                    parts = [f"{k}={v}" for k, v in limits.items()]
                    lines.append(f"      Limits:   {', '.join(parts)}")

        # Env vars — names only, NO secret values
        env_vars = c.env or [] if hasattr(c, "env") else []
        if env_vars:
            lines.append("    Environment Variables:")
            for ev in env_vars:
                ev_name = ev.name if hasattr(ev, "name") else "<unnamed>"
                if hasattr(ev, "value_from") and ev.value_from:
                    if hasattr(ev.value_from, "config_map_key_ref") and ev.value_from.config_map_key_ref:
                        ref = ev.value_from.config_map_key_ref
                        lines.append(f"      {ev_name} <- ConfigMap:{ref.name}/{ref.key}")
                    elif hasattr(ev.value_from, "secret_key_ref") and ev.value_from.secret_key_ref:
                        ref = ev.value_from.secret_key_ref
                        lines.append(f"      {ev_name} <- Secret:{ref.name}/{ref.key} (ref only)")
                    else:
                        lines.append(f"      {ev_name} (valueFrom)")
                else:
                    lines.append(f"      {ev_name} = <value set>")

        # Env from (configMapRef / secretRef — names only)
        env_from = c.env_from or [] if hasattr(c, "env_from") else []
        if env_from:
            lines.append("    Env From:")
            for ef in env_from:
                if hasattr(ef, "config_map_ref") and ef.config_map_ref:
                    lines.append(f"      ConfigMap: {ef.config_map_ref.name}")
                if hasattr(ef, "secret_ref") and ef.secret_ref:
                    lines.append(f"      Secret: {ef.secret_ref.name} (ref only, no values shown)")

        # Volume mounts
        mounts = c.volume_mounts or [] if hasattr(c, "volume_mounts") else []
        if mounts:
            lines.append("    Volume Mounts:")
            for m in mounts:
                ro = " (ro)" if hasattr(m, "read_only") and m.read_only else ""
                lines.append(f"      {m.mount_path} from {m.name}{ro}")

    return ToolResult(output="\n".join(lines), error=False)


# ── Task 2.5 — create_gke_tools factory ──────────────────────


def create_gke_tools(gke_config: GKEConfig) -> list[ToolDef]:
    """Create all GKE tool definitions bound to a GKEConfig.

    Follows the exact same factory pattern as ``create_file_tools`` and
    ``create_shell_tools``: returns a list of ``ToolDef`` objects with
    closures that bind the config.

    When the cluster is detected as GKE Autopilot, node-level tools
    (``kubectl_top`` with ``resource_type="nodes"`` and ``get_node_conditions``)
    return an immediate informational message instead of calling the K8s API.
    """
    is_autopilot = detect_autopilot(gke_config)

    def _autopilot_kubectl_top(
        resource_type: str = "pods",
        name: str | None = None,
        namespace: str = "default",
        _cfg: GKEConfig = gke_config,
    ) -> ToolResult:
        """Autopilot-aware kubectl_top wrapper."""
        if is_autopilot and resource_type == "nodes":
            return ToolResult(
                output=(
                    "GKE Autopilot cluster detected — kubectl top nodes is not available. "
                    "Node infrastructure is managed by Google on Autopilot. "
                    "Use kubectl get nodes and get_node_conditions for node status, "
                    "or kubectl_top(resource_type='pods') for workload-level metrics."
                ),
            )
        return kubectl_top(resource_type, gke_config=_cfg, name=name, namespace=namespace)

    return [
        ToolDef(
            name="kubectl_get",
            description=(
                "List or get Kubernetes resources from the connected GKE cluster. "
                "Supports pods, deployments, services, configmaps, secrets, hpa, "
                "ingress, nodes, namespaces, statefulsets, daemonsets, jobs, cronjobs, "
                "pv, pvc, serviceaccounts, endpoints, networkpolicies, replicasets. "
                "Read-only — does not modify any resources."
            ),
            parameters=[
                ToolParam(
                    name="resource",
                    type="string",
                    description=(
                        "Kubernetes resource type to list (e.g. 'pods', 'deployments', "
                        "'services', 'nodes')"
                    ),
                ),
                ToolParam(
                    name="name",
                    type="string",
                    description="Specific resource name to get. Omit to list all.",
                    required=False,
                ),
                ToolParam(
                    name="namespace",
                    type="string",
                    description="Kubernetes namespace (default: 'default'). Use 'all' for all namespaces.",
                    required=False,
                ),
                ToolParam(
                    name="output_format",
                    type="string",
                    description="Output format: 'table' (default), 'yaml', 'json', or 'wide'.",
                    required=False,
                ),
                ToolParam(
                    name="label_selector",
                    type="string",
                    description="Label selector to filter resources (e.g. 'app=nginx,tier=frontend').",
                    required=False,
                ),
                ToolParam(
                    name="field_selector",
                    type="string",
                    description="Field selector to filter resources (e.g. 'status.phase=Running').",
                    required=False,
                ),
            ],
            execute=lambda resource, name=None, namespace="default", output_format="table",
                    label_selector=None, field_selector=None, _cfg=gke_config: kubectl_get(
                resource,
                gke_config=_cfg,
                name=name,
                namespace=namespace,
                output_format=output_format,
                label_selector=label_selector,
                field_selector=field_selector,
            ),
        ),
        ToolDef(
            name="kubectl_describe",
            description=(
                "Describe a Kubernetes resource in detail, including labels, annotations, "
                "spec, status, conditions, and recent events. "
                "Read-only — does not modify any resources."
            ),
            parameters=[
                ToolParam(
                    name="resource",
                    type="string",
                    description="Kubernetes resource type (e.g. 'pod', 'deployment', 'service')",
                ),
                ToolParam(
                    name="name",
                    type="string",
                    description="Name of the specific resource to describe",
                ),
                ToolParam(
                    name="namespace",
                    type="string",
                    description="Kubernetes namespace (default: 'default')",
                    required=False,
                ),
            ],
            execute=lambda resource, name, namespace="default", _cfg=gke_config: kubectl_describe(
                resource, name, gke_config=_cfg, namespace=namespace,
            ),
        ),
        ToolDef(
            name="kubectl_logs",
            description=(
                "Retrieve logs from a Kubernetes pod. Automatically fetches previous "
                "container logs when current container is in CrashLoopBackOff. "
                "Read-only — does not modify any resources."
            ),
            parameters=[
                ToolParam(
                    name="pod",
                    type="string",
                    description="Name of the pod to get logs from",
                ),
                ToolParam(
                    name="namespace",
                    type="string",
                    description="Kubernetes namespace (default: 'default')",
                    required=False,
                ),
                ToolParam(
                    name="container",
                    type="string",
                    description="Specific container name (for multi-container pods)",
                    required=False,
                ),
                ToolParam(
                    name="tail_lines",
                    type="integer",
                    description="Number of recent log lines to return (default: 100)",
                    required=False,
                ),
                ToolParam(
                    name="since",
                    type="string",
                    description="Only return logs newer than this duration (e.g. '1h', '30m', '1h30m')",
                    required=False,
                ),
            ],
            execute=lambda pod, namespace="default", container=None, tail_lines=100,
                    since=None, _cfg=gke_config: kubectl_logs(
                pod,
                gke_config=_cfg,
                namespace=namespace,
                container=container,
                tail_lines=tail_lines,
                since=since,
            ),
        ),
        ToolDef(
            name="kubectl_top",
            description=(
                "Show CPU and memory usage for pods or nodes. "
                "Requires the metrics-server to be installed in the cluster. "
                "Read-only — does not modify any resources."
            ),
            parameters=[
                ToolParam(
                    name="resource_type",
                    type="string",
                    description="Type of resource to show metrics for: 'pods' (default) or 'nodes'",
                    required=False,
                ),
                ToolParam(
                    name="name",
                    type="string",
                    description="Specific pod or node name to show metrics for. Omit for all.",
                    required=False,
                ),
                ToolParam(
                    name="namespace",
                    type="string",
                    description="Kubernetes namespace for pod metrics (default: 'default'). Use 'all' for all namespaces.",
                    required=False,
                ),
            ],
            execute=lambda resource_type="pods", name=None, namespace="default":
                    _autopilot_kubectl_top(resource_type, name=name, namespace=namespace),
        ),
        # ── Write operations ──────────────────────────────────
        ToolDef(
            name="kubectl_scale",
            description=(
                "Scale a Kubernetes deployment, statefulset, or replicaset to a specified "
                "number of replicas (0-50). Reports the previous and new replica count. "
                "WRITE operation — modifies the cluster."
            ),
            parameters=[
                ToolParam(
                    name="resource",
                    type="string",
                    description="Resource type to scale: 'deployments', 'statefulsets', or 'replicasets'",
                ),
                ToolParam(
                    name="name",
                    type="string",
                    description="Name of the resource to scale",
                ),
                ToolParam(
                    name="replicas",
                    type="integer",
                    description="Target number of replicas (0-50)",
                ),
                ToolParam(
                    name="namespace",
                    type="string",
                    description="Kubernetes namespace (default: 'default')",
                    required=False,
                ),
            ],
            execute=lambda resource, name, replicas, namespace="default",
                    _cfg=gke_config: kubectl_scale(
                resource, name, replicas, gke_config=_cfg, namespace=namespace,
            ),
        ),
        ToolDef(
            name="kubectl_restart",
            description=(
                "Trigger a rolling restart of a deployment, statefulset, or daemonset. "
                "Equivalent to 'kubectl rollout restart'. Causes a zero-downtime rolling update. "
                "WRITE operation — modifies the cluster."
            ),
            parameters=[
                ToolParam(
                    name="resource",
                    type="string",
                    description="Resource type to restart: 'deployments', 'statefulsets', or 'daemonsets'",
                ),
                ToolParam(
                    name="name",
                    type="string",
                    description="Name of the resource to restart",
                ),
                ToolParam(
                    name="namespace",
                    type="string",
                    description="Kubernetes namespace (default: 'default')",
                    required=False,
                ),
            ],
            execute=lambda resource, name, namespace="default",
                    _cfg=gke_config: kubectl_restart(
                resource, name, gke_config=_cfg, namespace=namespace,
            ),
        ),
        ToolDef(
            name="kubectl_label",
            description=(
                "Add or update labels on a Kubernetes resource. "
                "Format: 'key1=value1,key2=value2'. To remove: 'key-'. "
                "System labels (kubernetes.io/, k8s.io/) are protected. "
                "WRITE operation — modifies the cluster."
            ),
            parameters=[
                ToolParam(
                    name="resource",
                    type="string",
                    description=(
                        "Resource type (pods, deployments, services, configmaps, secrets, "
                        "statefulsets, daemonsets, namespaces, nodes)"
                    ),
                ),
                ToolParam(
                    name="name",
                    type="string",
                    description="Name of the resource to label",
                ),
                ToolParam(
                    name="labels",
                    type="string",
                    description="Labels to set: 'key1=value1,key2=value2'. Use 'key-' to remove a label.",
                ),
                ToolParam(
                    name="namespace",
                    type="string",
                    description="Kubernetes namespace (default: 'default')",
                    required=False,
                ),
            ],
            execute=lambda resource, name, labels, namespace="default",
                    _cfg=gke_config: kubectl_label(
                resource, name, labels, gke_config=_cfg, namespace=namespace,
            ),
        ),
        ToolDef(
            name="kubectl_annotate",
            description=(
                "Add or update annotations on a Kubernetes resource. "
                "Format: 'key1=value1,key2=value2'. To remove: 'key-'. "
                "System annotations (kubernetes.io/, k8s.io/) are protected. "
                "WRITE operation — modifies the cluster."
            ),
            parameters=[
                ToolParam(
                    name="resource",
                    type="string",
                    description=(
                        "Resource type (pods, deployments, services, configmaps, secrets, "
                        "statefulsets, daemonsets, namespaces, nodes)"
                    ),
                ),
                ToolParam(
                    name="name",
                    type="string",
                    description="Name of the resource to annotate",
                ),
                ToolParam(
                    name="annotations",
                    type="string",
                    description="Annotations to set: 'key1=value1,key2=value2'. Use 'key-' to remove.",
                ),
                ToolParam(
                    name="namespace",
                    type="string",
                    description="Kubernetes namespace (default: 'default')",
                    required=False,
                ),
            ],
            execute=lambda resource, name, annotations, namespace="default",
                    _cfg=gke_config: kubectl_annotate(
                resource, name, annotations, gke_config=_cfg, namespace=namespace,
            ),
        ),
        # ── Diagnostic tools (Phase 1) ────────────────────────
        ToolDef(
            name="get_events",
            description=(
                "List Kubernetes events in a namespace, filtered by type (Warning/Normal) "
                "and optionally by involved object. Events reveal WHY pods fail, WHY nodes "
                "have issues, and what the scheduler is doing. Critical for SRE triage. "
                "Read-only — does not modify any resources."
            ),
            parameters=[
                ToolParam(
                    name="namespace",
                    type="string",
                    description="Kubernetes namespace (default: 'default')",
                    required=False,
                ),
                ToolParam(
                    name="event_type",
                    type="string",
                    description="Filter by event type: 'Warning', 'Normal', or omit for all events.",
                    required=False,
                ),
                ToolParam(
                    name="involved_object_name",
                    type="string",
                    description="Filter events by involved object name (e.g. a pod name or node name).",
                    required=False,
                ),
                ToolParam(
                    name="involved_object_kind",
                    type="string",
                    description="Filter events by involved object kind (e.g. 'Pod', 'Node', 'Deployment').",
                    required=False,
                ),
                ToolParam(
                    name="limit",
                    type="integer",
                    description="Maximum number of events to return (default: 50, max: 500).",
                    required=False,
                ),
            ],
            execute=lambda namespace="default", event_type=None, involved_object_name=None,
                    involved_object_kind=None, limit=50,
                    _cfg=gke_config: get_events(
                gke_config=_cfg,
                namespace=namespace,
                event_type=event_type,
                involved_object_name=involved_object_name,
                involved_object_kind=involved_object_kind,
                limit=limit,
            ),
        ),
        ToolDef(
            name="get_rollout_status",
            description=(
                "Check the rollout status of a Kubernetes deployment — whether it is "
                "progressing, complete, stalled, or failed. Shows replica counts, "
                "conditions, and rollout strategy. Equivalent to "
                "'kubectl rollout status deployment/<name>'. "
                "Read-only — does not modify any resources."
            ),
            parameters=[
                ToolParam(
                    name="name",
                    type="string",
                    description="Name of the deployment to check rollout status for",
                ),
                ToolParam(
                    name="namespace",
                    type="string",
                    description="Kubernetes namespace (default: 'default')",
                    required=False,
                ),
            ],
            execute=lambda name, namespace="default",
                    _cfg=gke_config: get_rollout_status(
                name, gke_config=_cfg, namespace=namespace,
            ),
        ),
        # ── Diagnostic tools (Phase 2) ────────────────────────
        ToolDef(
            name="get_node_conditions",
            description=(
                "Show node health conditions, resource pressure, taints, and capacity. "
                "Without a node name, lists ALL nodes with status, roles, version, OS, "
                "kernel, container runtime, and CPU/memory capacity. With a node name, "
                "shows detailed conditions (MemoryPressure, DiskPressure, PIDPressure, "
                "NetworkUnavailable), taints, labels, and capacity vs allocatable. "
                "Fills the gap where kubectl_get nodes hides pressure conditions. "
                "Read-only — does not modify any resources."
            ),
            parameters=[
                ToolParam(
                    name="name",
                    type="string",
                    description="Specific node name for detailed view. Omit to list all nodes.",
                    required=False,
                ),
            ],
            execute=lambda name=None, _cfg=gke_config: get_node_conditions(
                gke_config=_cfg, name=name,
            ),
        ),
        ToolDef(
            name="get_container_status",
            description=(
                "Show detailed container-level status for ALL containers in a pod "
                "(init, regular, and ephemeral). For each container: name, image, "
                "state (Waiting/Running/Terminated with reason), ready flag, restart "
                "count, last termination state (crucial for CrashLoopBackOff), resource "
                "requests/limits, volume mounts, and env var sources (names only, no "
                "secret values). Essential for multi-container pod debugging. "
                "Read-only — does not modify any resources."
            ),
            parameters=[
                ToolParam(
                    name="name",
                    type="string",
                    description="Name of the pod to inspect",
                ),
                ToolParam(
                    name="namespace",
                    type="string",
                    description="Kubernetes namespace (default: 'default')",
                    required=False,
                ),
            ],
            execute=lambda name, namespace="default",
                    _cfg=gke_config: get_container_status(
                name, gke_config=_cfg, namespace=namespace,
            ),
        ),
        # ── Diagnostic tools (Phase 3) ────────────────────────
        ToolDef(
            name="exec_command",
            description=(
                "Execute a diagnostic command inside a running container. "
                "SECURITY: Disabled by default (requires gke.exec_enabled=true). "
                "Commands are validated against a denylist (shell injection, "
                "destructive operations) and an allowlist (read-only diagnostic "
                "commands like cat, ls, ps, df, curl, etc.). Output is truncated "
                "to 10000 chars. Use for inspecting config files, checking "
                "processes, network debugging, and runtime diagnostics."
            ),
            parameters=[
                ToolParam(
                    name="pod_name",
                    type="string",
                    description="Name of the pod to execute the command in",
                ),
                ToolParam(
                    name="namespace",
                    type="string",
                    description="Kubernetes namespace (required)",
                ),
                ToolParam(
                    name="command",
                    type="string",
                    description=(
                        "Diagnostic command to execute. Must start with an allowed prefix: "
                        "cat, head, tail, ls, env, printenv, whoami, id, hostname, date, "
                        "ps, top -bn1, df, du, mount, ip, ifconfig, netstat, ss, nslookup, "
                        "dig, ping, curl, wget, java -version, python --version, node --version"
                    ),
                ),
                ToolParam(
                    name="container",
                    type="string",
                    description="Container name (for multi-container pods). Omit for default container.",
                    required=False,
                ),
                ToolParam(
                    name="timeout",
                    type="integer",
                    description="Execution timeout in seconds (default: 30, max: 300)",
                    required=False,
                ),
            ],
            execute=lambda pod_name, namespace, command, container=None, timeout=30,
                    _cfg=gke_config: exec_command(
                pod_name, command, gke_config=_cfg, namespace=namespace,
                container=container, timeout=timeout,
            ),
        ),
        ToolDef(
            name="check_rbac",
            description=(
                "Check if a service account or the current user has permission to "
                "perform a specific action on a Kubernetes resource. Uses "
                "SubjectAccessReview (for service accounts) or SelfSubjectAccessReview "
                "(for current user). Use BEFORE operations that might fail with "
                "permission errors. Read-only — does not modify any resources."
            ),
            parameters=[
                ToolParam(
                    name="verb",
                    type="string",
                    description=(
                        "The action to check: get, list, watch, create, update, patch, delete"
                    ),
                ),
                ToolParam(
                    name="resource",
                    type="string",
                    description=(
                        "Resource type (pods, deployments, services, configmaps, secrets, etc.). "
                        "Aliases accepted (po, svc, deploy, etc.)."
                    ),
                ),
                ToolParam(
                    name="namespace",
                    type="string",
                    description="Kubernetes namespace (required)",
                ),
                ToolParam(
                    name="service_account",
                    type="string",
                    description=(
                        "Service account name to check. Omit to check current user's permissions."
                    ),
                    required=False,
                ),
                ToolParam(
                    name="resource_name",
                    type="string",
                    description="Specific resource name to check access for (optional).",
                    required=False,
                ),
            ],
            execute=lambda verb, resource, namespace, service_account=None, resource_name=None,
                    _cfg=gke_config: check_rbac(
                verb, resource, gke_config=_cfg, namespace=namespace,
                service_account=service_account, resource_name=resource_name,
            ),
        ),
        ToolDef(
            name="get_rollout_history",
            description=(
                "Show the revision history of a Kubernetes deployment. Lists all "
                "revisions with images, creation time, replica count, and status "
                "(active vs scaled-down). Optionally show detailed pod template for "
                "a specific revision including containers, ports, env var names, "
                "volume mounts, and resource requests/limits. Use BEFORE recommending "
                "a rollback to understand what changed in each revision. "
                "Read-only — does not modify any resources."
            ),
            parameters=[
                ToolParam(
                    name="name",
                    type="string",
                    description="Name of the deployment to show rollout history for",
                ),
                ToolParam(
                    name="namespace",
                    type="string",
                    description="Kubernetes namespace (required)",
                ),
                ToolParam(
                    name="revision",
                    type="integer",
                    description=(
                        "Specific revision number to show detailed info for. "
                        "Omit to list all revisions."
                    ),
                    required=False,
                ),
            ],
            execute=lambda name, namespace, revision=None,
                    _cfg=gke_config: get_rollout_history(
                name, gke_config=_cfg, namespace=namespace, revision=revision,
            ),
        ),
    ]

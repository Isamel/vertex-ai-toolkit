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


def _create_k8s_clients(
    gke_config: GKEConfig,
) -> tuple[Any, Any, Any] | ToolResult:
    """Create and configure Kubernetes API clients from GKEConfig.

    Returns a tuple of ``(CoreV1Api, AppsV1Api, CustomObjectsApi)`` on success,
    or a ``ToolResult`` with ``error=True`` on failure.

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
                return (
                    k8s_client.CoreV1Api(),
                    k8s_client.AppsV1Api(),
                    k8s_client.CustomObjectsApi(),
                )
    except Exception as exc:
        return ToolResult(
            output=f"Failed to configure Kubernetes client: {exc}",
            error=True,
        )

    # ── Build API clients with the proxy-aware Configuration ─
    api_client = k8s_client.ApiClient(config)
    return (
        k8s_client.CoreV1Api(api_client),
        k8s_client.AppsV1Api(api_client),
        k8s_client.CustomObjectsApi(api_client),
    )


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
        # Use the kubernetes client's serialisation
        serialised = [k8s_client.ApiClient().sanitize_for_serialization(i) for i in items]
        return _json.dumps(serialised, indent=2, default=str)

    if output_format == "yaml":
        try:
            import yaml as _yaml  # noqa: WPS433
        except ImportError:
            return "PyYAML is not installed. Use output_format='json' instead."
        serialised = [k8s_client.ApiClient().sanitize_for_serialization(i) for i in items]
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

        batch_v1 = BatchV1Api()
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

        auto_v2 = AutoscalingV2Api()
        if namespace in ("", "all"):
            return auto_v2.list_horizontal_pod_autoscaler_for_all_namespaces(**kwargs)
        return auto_v2.list_namespaced_horizontal_pod_autoscaler(namespace=namespace, **kwargs)

    # ── Networking V1 resources ───────────────────────────────
    if api_group == "networking":
        from kubernetes.client import NetworkingV1Api  # noqa: WPS433

        net_v1 = NetworkingV1Api()
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
    core_v1, apps_v1, custom_api = result

    try:
        api_result = _list_resource(
            core_v1, apps_v1, custom_api, resource,
            namespace=ns,
            label_selector=label_selector,
            field_selector=field_selector,
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

        batch_v1 = BatchV1Api()
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

        return AutoscalingV2Api().read_namespaced_horizontal_pod_autoscaler(
            name=name, namespace=namespace,
        )

    # ── Networking ────────────────────────────────────────────
    if api_group == "networking":
        from kubernetes.client import NetworkingV1Api  # noqa: WPS433

        net_v1 = NetworkingV1Api()
        read_map_net: dict[str, str] = {
            "ingress": "read_namespaced_ingress",
            "ingresses": "read_namespaced_ingress",
            "networkpolicies": "read_namespaced_network_policy",
        }
        method_name_n = read_map_net.get(resource)
        if not method_name_n:
            return None
        return getattr(net_v1, method_name_n)(name=name, namespace=namespace)

    return None


def _format_describe(resource: str, obj: Any) -> str:
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
        events_v1 = k8s_client.CoreV1Api()
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
    core_v1, apps_v1, _ = result

    try:
        obj = _describe_resource(core_v1, apps_v1, resource, name, ns)
        if obj is None:
            return ToolResult(output=f"Describe not supported for resource type: {resource}", error=True)
        return ToolResult(output=_format_describe(resource, obj))

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
    core_v1, _, _ = result

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
            except Exception:
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
    _, _, custom_api = result

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
            return ToolResult(output=f"Access denied to metrics API ({exc.status}): {exc.reason}", error=True)
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:
        logger.exception("kubectl_top failed")
        return ToolResult(output=f"Error fetching metrics: {exc}", error=True)


# ── Task 2.5 — create_gke_tools factory ──────────────────────


def create_gke_tools(gke_config: GKEConfig) -> list[ToolDef]:
    """Create all GKE tool definitions bound to a GKEConfig.

    Follows the exact same factory pattern as ``create_file_tools`` and
    ``create_shell_tools``: returns a list of ``ToolDef`` objects with
    closures that bind the config.
    """
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
            execute=lambda resource_type="pods", name=None, namespace="default",
                    _cfg=gke_config: kubectl_top(
                resource_type, gke_config=_cfg, name=name, namespace=namespace,
            ),
        ),
    ]

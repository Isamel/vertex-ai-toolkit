"""Resource maps, aliases, and resolution — maps K8s resource types to API groups and methods."""

from __future__ import annotations

from typing import Any

from vaig.tools.base import ToolResult

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

# Allowed resource types for write operations (intentionally restrictive).
_SCALABLE_RESOURCES = frozenset({"deployments", "statefulsets", "replicasets"})
_RESTARTABLE_RESOURCES = frozenset({"deployments", "statefulsets", "daemonsets"})
_LABELABLE_RESOURCES = frozenset({
    "pods", "deployments", "services", "configmaps", "secrets",
    "statefulsets", "daemonsets", "namespaces", "nodes",
})


def _normalise_resource(resource: str) -> str:
    """Normalise a resource type string to its canonical plural form."""
    lower = resource.lower().strip()
    return _RESOURCE_ALIASES.get(lower, lower)


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

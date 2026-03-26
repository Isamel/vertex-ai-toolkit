"""Resource maps, aliases, and resolution — maps K8s resource types to API groups and methods."""

from __future__ import annotations

from typing import Any

from vaig.tools.base import ToolResult

_K8S_AVAILABLE = True
try:
    from kubernetes.client import (
        DiscoveryV1Api,  # noqa: WPS433
        NodeV1Api,  # noqa: WPS433
        RbacAuthorizationV1Api,  # noqa: WPS433
        SchedulingV1Api,  # noqa: WPS433
        StorageV1Api,  # noqa: WPS433
    )
    from kubernetes.client import exceptions as k8s_exceptions  # noqa: WPS433
except ImportError:
    _K8S_AVAILABLE = False


class _DictMeta:
    """Thin wrapper that exposes dict-based K8s metadata as object attributes."""

    def __init__(self, d: dict[str, Any]) -> None:
        self._d = d

    @property
    def name(self) -> str:
        return str(self._d.get("name", ""))

    @property
    def namespace(self) -> str | None:
        return self._d.get("namespace")

    @property
    def labels(self) -> dict[str, str] | None:
        return self._d.get("labels")

    @property
    def annotations(self) -> dict[str, str] | None:
        return self._d.get("annotations")

    @property
    def creation_timestamp(self) -> Any:
        return self._d.get("creationTimestamp")

    @property
    def deletion_timestamp(self) -> Any:
        return self._d.get("deletionTimestamp")


class _DictItem:
    """Thin wrapper that exposes a plain dict custom resource as a K8s-style object."""

    def __init__(self, d: dict[str, Any]) -> None:
        self._d = d
        self.metadata = _DictMeta(d.get("metadata", {}))

    @property
    def spec(self) -> dict[str, Any] | None:
        return self._d.get("spec")

    @property
    def status(self) -> dict[str, Any] | None:
        return self._d.get("status")

    def to_dict(self) -> dict[str, Any]:
        """Return the raw backing dict for this custom resource."""
        return self._d

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style access for compatibility."""
        return self._d.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._d[key]


class _DictItemList:
    """Wrapper that makes a CustomObjectsApi dict response look like a K8s list object."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.items = [_DictItem(i) for i in raw.get("items", [])]


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
    "resourcequotas": "core",
    # Config/Policy
    "limitranges": "core",
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
    # Networking — EndpointSlices (modern replacement for endpoints)
    "endpointslices": "discovery",
    "poddisruptionbudgets": "policy",
    # Admission registration (webhook configurations)
    "mutatingwebhookconfigurations": "admissionregistration",
    "validatingwebhookconfigurations": "admissionregistration",
    # Custom Resource Definitions
    "customresourcedefinitions": "apiextensions",
    "crds": "apiextensions",
    # RBAC resources
    "roles": "rbac",
    "clusterroles": "rbac",
    "rolebindings": "rbac",
    "clusterrolebindings": "rbac",
    # Storage resources
    "storageclasses": "storage",
    "volumeattachments": "storage",
    "csidrivers": "storage",
    "csinodes": "storage",
    # Scheduling resources
    "priorityclasses": "scheduling",
    # Runtime resources
    "runtimeclasses": "node",
    # External Secrets Operator CRDs
    "externalsecrets": "custom_external_secrets",
    "externalsecret": "custom_external_secrets",
    # ArgoCD CRDs
    "applications.argoproj.io": "custom_argocd",
    # Flux HelmRelease CRDs
    "helmreleases.helm.toolkit.fluxcd.io": "custom_flux_helm",
    # Flux Kustomization CRDs
    "kustomizations.kustomize.toolkit.fluxcd.io": "custom_flux_kustomize",
    # VPA CRDs
    "verticalpodautoscalers": "custom_vpa",
    # Argo Rollouts CRDs
    "rollout": "custom_argo_rollouts",
    "rollouts": "custom_argo_rollouts",
    "analysisrun": "custom_argo_rollouts",
    "analysisruns": "custom_argo_rollouts",
    "analysistemplate": "custom_argo_rollouts",
    "analysistemplates": "custom_argo_rollouts",
    "clusteranalysistemplate": "custom_argo_rollouts_cluster",
    "clusteranalysistemplates": "custom_argo_rollouts_cluster",
    "experiment": "custom_argo_rollouts",
    "experiments": "custom_argo_rollouts",
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
    "mutatingwebhookconfiguration": "mutatingwebhookconfigurations",
    "mwc": "mutatingwebhookconfigurations",
    "validatingwebhookconfiguration": "validatingwebhookconfigurations",
    "vwc": "validatingwebhookconfigurations",
    "customresourcedefinition": "customresourcedefinitions",
    "crd": "customresourcedefinitions",
    # RBAC aliases
    "role": "roles",
    "clusterrole": "clusterroles",
    "cr": "clusterroles",
    "rolebinding": "rolebindings",
    "rb": "rolebindings",
    "clusterrolebinding": "clusterrolebindings",
    "crb": "clusterrolebindings",
    # Storage aliases
    "storageclass": "storageclasses",
    "sc": "storageclasses",
    "volumeattachment": "volumeattachments",
    "va": "volumeattachments",
    "csidriver": "csidrivers",
    "csinode": "csinodes",
    # Config/Policy aliases
    "limitrange": "limitranges",
    "lr": "limitranges",
    # Networking aliases
    "endpointslice": "endpointslices",
    "eps": "endpointslices",
    # Scheduling aliases
    "priorityclass": "priorityclasses",
    "pc": "priorityclasses",
    # Runtime aliases
    "runtimeclass": "runtimeclasses",
    # External Secrets aliases
    "es": "externalsecrets",
    # ArgoCD aliases
    "app": "applications.argoproj.io",
    "application": "applications.argoproj.io",
    "argoapp": "applications.argoproj.io",
    # Flux HelmRelease aliases
    "hr": "helmreleases.helm.toolkit.fluxcd.io",
    "helmrelease": "helmreleases.helm.toolkit.fluxcd.io",
    # Flux Kustomization aliases
    "ks": "kustomizations.kustomize.toolkit.fluxcd.io",
    "kustomization": "kustomizations.kustomize.toolkit.fluxcd.io",
    # VPA aliases
    "vpa": "verticalpodautoscalers",
    "verticalpodautoscaler": "verticalpodautoscalers",
    # Argo Rollouts aliases
    "ro": "rollouts",
    "ar": "analysisruns",
    "at": "analysistemplates",
    "cat": "clusteranalysistemplates",
    "exp": "experiments",
}

# Resource types expanded when ``resource="all"`` is requested.
# Mirrors ``kubectl get all`` which covers the most common workload types.
_ALL_RESOURCE_TYPES: tuple[str, ...] = (
    "pods",
    "services",
    "deployments",
    "replicasets",
    "statefulsets",
    "daemonsets",
    "jobs",
    "cronjobs",
    "hpa",
)

# Allowed resource types for write operations (intentionally restrictive).
_SCALABLE_RESOURCES = frozenset({"deployments", "statefulsets", "replicasets"})
_RESTARTABLE_RESOURCES = frozenset({"deployments", "statefulsets", "daemonsets"})
_LABELABLE_RESOURCES = frozenset(
    {
        "pods",
        "deployments",
        "services",
        "configmaps",
        "secrets",
        "statefulsets",
        "daemonsets",
        "namespaces",
        "nodes",
    }
)

# Cluster-scoped resources (no namespace parameter for list/describe).
_CLUSTER_SCOPED_RESOURCES = frozenset(
    {
        "nodes",
        "namespaces",
        "pv",
        "persistentvolumes",
        "mutatingwebhookconfigurations",
        "validatingwebhookconfigurations",
        "customresourcedefinitions",
        "crds",
        # RBAC cluster-scoped
        "clusterroles",
        "clusterrolebindings",
        # Storage cluster-scoped
        "storageclasses",
        "csidrivers",
        "csinodes",
        "volumeattachments",
        # Scheduling cluster-scoped
        "priorityclasses",
        # Runtime cluster-scoped
        "runtimeclasses",
        # Argo Rollouts cluster-scoped
        "clusteranalysistemplate",
        "clusteranalysistemplates",
    }
)

# Resources where kubectl_describe has an actual handler in _describe_resource().
# New API groups (rbac, storage, discovery, scheduling, node) are supported for
# list/get but do NOT have describe handlers — they return None, which surfaces
# as "Describe not supported".  Keep this set accurate so validation is honest.
_DESCRIBE_SUPPORTED_RESOURCES: frozenset[str] = frozenset(
    {
        # Core V1
        "pods",
        "services",
        "configmaps",
        "secrets",
        "serviceaccounts",
        "endpoints",
        "pvc",
        "persistentvolumeclaims",
        "resourcequotas",
        "nodes",
        "namespaces",
        "pv",
        "persistentvolumes",
        # Apps V1
        "deployments",
        "statefulsets",
        "daemonsets",
        "replicasets",
        # Batch V1
        "jobs",
        "cronjobs",
        # Autoscaling
        "hpa",
        "horizontalpodautoscalers",
        # Networking
        "ingress",
        "ingresses",
        "networkpolicies",
        # Policy
        "poddisruptionbudgets",
        # AdmissionRegistration
        "mutatingwebhookconfigurations",
        "validatingwebhookconfigurations",
        # ApiExtensions
        "customresourcedefinitions",
        "crds",
        # Custom resources
        "externalsecrets",
        "externalsecret",
        "verticalpodautoscalers",
    }
)

# Real K8s resources we haven't implemented yet.
# Used to distinguish "gap in our tools" from "hallucinated resource".
_KNOWN_K8S_RESOURCES = frozenset(
    {
        "events",
        "componentstatuses",
        "replicationcontrollers",
        "podtemplates",
        "controllerrevisions",
        "leases",
    }
)


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
    is_cluster_scoped = resource in _CLUSTER_SCOPED_RESOURCES

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
            "persistentvolumeclaims": (
                "list_namespaced_persistent_volume_claim",
                "list_persistent_volume_claim_for_all_namespaces",
            ),
            "resourcequotas": ("list_namespaced_resource_quota", "list_resource_quota_for_all_namespaces"),
            "limitranges": ("list_namespaced_limit_range", "list_limit_range_for_all_namespaces"),
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

    # ── Discovery V1 resources (EndpointSlices) ───────────────
    if api_group == "discovery":
        disc_v1 = DiscoveryV1Api(api_client=api_client)
        if namespace in ("", "all"):
            return disc_v1.list_endpoint_slice_for_all_namespaces(**kwargs)
        return disc_v1.list_namespaced_endpoint_slice(namespace=namespace, **kwargs)

    # ── Policy V1 resources ──────────────────────────────────
    if api_group == "policy":
        from kubernetes.client import PolicyV1Api  # noqa: WPS433

        policy_v1 = PolicyV1Api(api_client=api_client)
        if namespace in ("", "all"):
            return policy_v1.list_pod_disruption_budget_for_all_namespaces(**kwargs)
        return policy_v1.list_namespaced_pod_disruption_budget(namespace=namespace, **kwargs)

    # ── AdmissionRegistration V1 resources ───────────────────
    if api_group == "admissionregistration":
        from kubernetes.client import AdmissionregistrationV1Api  # noqa: WPS433

        admission_v1 = AdmissionregistrationV1Api(api_client=api_client)
        if resource == "mutatingwebhookconfigurations":
            return admission_v1.list_mutating_webhook_configuration(**kwargs)
        if resource == "validatingwebhookconfigurations":
            return admission_v1.list_validating_webhook_configuration(**kwargs)
        return ToolResult(output=f"Unsupported admissionregistration resource type: {resource}", error=True)

    # ── ApiExtensions V1 resources ───────────────────────────
    if api_group == "apiextensions":
        from kubernetes.client import ApiextensionsV1Api  # noqa: WPS433

        ext_v1 = ApiextensionsV1Api(api_client=api_client)
        return ext_v1.list_custom_resource_definition(**kwargs)

    # ── RBAC Authorization V1 resources ──────────────────────
    if api_group == "rbac":
        rbac_v1 = RbacAuthorizationV1Api(api_client=api_client)
        # Cluster-scoped RBAC resources: map to their single list method
        rbac_cluster_map: dict[str, str] = {
            "clusterroles": "list_cluster_role",
            "clusterrolebindings": "list_cluster_role_binding",
        }
        # Namespaced RBAC resources: (all_namespaces_method, namespaced_method)
        rbac_ns_map: dict[str, tuple[str, str]] = {
            "roles": ("list_role_for_all_namespaces", "list_namespaced_role"),
            "rolebindings": ("list_role_binding_for_all_namespaces", "list_namespaced_role_binding"),
        }
        if resource in rbac_cluster_map:
            return getattr(rbac_v1, rbac_cluster_map[resource])(**kwargs)
        entry_rbac = rbac_ns_map.get(resource)
        if not entry_rbac:
            return ToolResult(output=f"Unsupported RBAC resource type: {resource}", error=True)
        all_ns_method, ns_method = entry_rbac
        if namespace in ("", "all"):
            return getattr(rbac_v1, all_ns_method)(**kwargs)
        return getattr(rbac_v1, ns_method)(namespace=namespace, **kwargs)

    # ── Storage V1 resources ──────────────────────────────────
    if api_group == "storage":
        storage_v1 = StorageV1Api(api_client=api_client)
        # All storage resources are cluster-scoped — keys map to method name strings
        storage_method_map: dict[str, str] = {
            "storageclasses": "list_storage_class",
            "volumeattachments": "list_volume_attachment",
            "csidrivers": "list_csi_driver",
            "csinodes": "list_csi_node",
        }
        method_name_s = storage_method_map.get(resource)
        if not method_name_s:
            return ToolResult(output=f"Unsupported storage resource type: {resource}", error=True)
        return getattr(storage_v1, method_name_s)(**kwargs)

    # ── Scheduling V1 resources ───────────────────────────────
    if api_group == "scheduling":
        sched_v1 = SchedulingV1Api(api_client=api_client)
        return sched_v1.list_priority_class(**kwargs)

    # ── Node V1 resources (RuntimeClasses) ───────────────────
    if api_group == "node":
        node_v1 = NodeV1Api(api_client=api_client)
        return node_v1.list_runtime_class(**kwargs)

    # ── External Secrets Operator (custom) ───────────────────
    if api_group == "custom_external_secrets":
        if namespace in ("", "all"):
            raw = custom_api.list_cluster_custom_object(
                group="external-secrets.io",
                version="v1beta1",
                plural="externalsecrets",
                **kwargs,
            )
        else:
            raw = custom_api.list_namespaced_custom_object(
                group="external-secrets.io",
                version="v1beta1",
                namespace=namespace,
                plural="externalsecrets",
                **kwargs,
            )
        return _DictItemList(raw)

    # ── ArgoCD Applications (custom) ─────────────────────────
    if api_group == "custom_argocd":
        if namespace in ("", "all"):
            raw = custom_api.list_cluster_custom_object(
                group="argoproj.io",
                version="v1alpha1",
                plural="applications",
                **kwargs,
            )
        else:
            raw = custom_api.list_namespaced_custom_object(
                group="argoproj.io",
                version="v1alpha1",
                namespace=namespace,
                plural="applications",
                **kwargs,
            )
        return _DictItemList(raw)

    # ── Flux HelmReleases (custom) ───────────────────────────
    if api_group == "custom_flux_helm":
        if namespace in ("", "all"):
            raw = custom_api.list_cluster_custom_object(
                group="helm.toolkit.fluxcd.io",
                version="v2beta1",
                plural="helmreleases",
                **kwargs,
            )
        else:
            raw = custom_api.list_namespaced_custom_object(
                group="helm.toolkit.fluxcd.io",
                version="v2beta1",
                namespace=namespace,
                plural="helmreleases",
                **kwargs,
            )
        return _DictItemList(raw)

    # ── Flux Kustomizations (custom) ─────────────────────────
    if api_group == "custom_flux_kustomize":
        if namespace in ("", "all"):
            raw = custom_api.list_cluster_custom_object(
                group="kustomize.toolkit.fluxcd.io",
                version="v1",
                plural="kustomizations",
                **kwargs,
            )
        else:
            raw = custom_api.list_namespaced_custom_object(
                group="kustomize.toolkit.fluxcd.io",
                version="v1",
                namespace=namespace,
                plural="kustomizations",
                **kwargs,
            )
        return _DictItemList(raw)

    # ── Vertical Pod Autoscaler (custom) ─────────────────────
    if api_group == "custom_vpa":
        try:
            if namespace in ("", "all", None):
                raw = custom_api.list_cluster_custom_object(
                    group="autoscaling.k8s.io",
                    version="v1",
                    plural="verticalpodautoscalers",
                    **kwargs,
                )
            else:
                raw = custom_api.list_namespaced_custom_object(
                    group="autoscaling.k8s.io",
                    version="v1",
                    namespace=namespace,
                    plural="verticalpodautoscalers",
                    **kwargs,
                )
            return _DictItemList(raw)
        except k8s_exceptions.ApiException as exc:
            # Check for 404 — VPA CRD not installed
            if exc.status == 404:
                return ToolResult(
                    output="VPA CRD not installed in this cluster. Install the Vertical Pod Autoscaler to use this resource.",
                )
            raise

    # ── Argo Rollouts namespace-scoped CRDs (custom) ─────────
    _ARGO_ROLLOUTS_PLURAL_MAP: dict[str, str] = {
        "rollout": "rollouts",
        "rollouts": "rollouts",
        "analysisrun": "analysisruns",
        "analysisruns": "analysisruns",
        "analysistemplate": "analysistemplates",
        "analysistemplates": "analysistemplates",
        "experiment": "experiments",
        "experiments": "experiments",
    }
    if api_group == "custom_argo_rollouts":
        plural = _ARGO_ROLLOUTS_PLURAL_MAP.get(resource, resource)
        try:
            if namespace in ("", "all", None):
                raw = custom_api.list_cluster_custom_object(
                    group="argoproj.io",
                    version="v1alpha1",
                    plural=plural,
                    **kwargs,
                )
            else:
                raw = custom_api.list_namespaced_custom_object(
                    group="argoproj.io",
                    version="v1alpha1",
                    namespace=namespace,
                    plural=plural,
                    **kwargs,
                )
            return _DictItemList(raw)
        except k8s_exceptions.ApiException as exc:
            if exc.status == 404:
                return ToolResult(
                    output=f"Argo Rollouts CRD not installed in this cluster (resource: {resource}).",
                )
            raise

    # ── Argo Rollouts cluster-scoped CRDs (custom) ───────────
    if api_group == "custom_argo_rollouts_cluster":
        try:
            raw = custom_api.list_cluster_custom_object(
                group="argoproj.io",
                version="v1alpha1",
                plural="clusteranalysistemplates",
                **kwargs,
            )
            return _DictItemList(raw)
        except k8s_exceptions.ApiException as exc:
            if exc.status == 404:
                return ToolResult(
                    output=f"Argo Rollouts CRD not installed in this cluster (resource: {resource}).",
                )
            raise

    return ToolResult(output=f"Unknown API group for resource: {resource}", error=True)

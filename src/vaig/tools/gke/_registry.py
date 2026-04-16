"""GKE tool definitions and factory function.

Layer 2 — imports all Layer 1 modules and assembles ToolDef objects
with closures that bind the GKEConfig at creation time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vaig.tools.base import ToolDef, ToolParam, ToolResult
from vaig.tools.categories import (
    ARGO_ROLLOUTS,
    ARGOCD,
    DATADOG,
    HELM,
    KUBERNETES,
    KUBERNETES_WRITE,
    LOGGING,
    MESH,
    MONITORING,
    SCALING,
)

from . import _clients, diagnostics, discovery, kubectl, mesh, mutations, security
from .argo_rollouts import (
    detect_argo_rollouts,
    kubectl_get_analysisrun,
    kubectl_get_analysistemplate,
    kubectl_get_cluster_analysis_template,
    kubectl_get_experiment,
    kubectl_get_rollout,
)
from .argocd import (
    argocd_app_diff,
    argocd_app_history,
    argocd_app_managed_resources,
    argocd_app_status,
    argocd_list_applications,
    detect_argocd,
)
from .datadog import get_datadog_config
from .datadog_api import (
    diagnose_datadog_metrics,
    get_datadog_apm_services,
    get_datadog_monitors,
    get_datadog_service_catalog,
    get_datadog_service_dependencies,
    query_datadog_metrics,
)
from .helm import (
    helm_list_releases,
    helm_release_history,
    helm_release_status,
    helm_release_values,
)
from .kubectl import kubectl_get_labels
from .metrics_api import check_metrics_api_health, query_custom_metrics, query_external_metrics
from .monitoring import get_pod_metrics
from .scaling import get_scaling_status

if TYPE_CHECKING:
    from vaig.core.config import GKEConfig

logger = logging.getLogger(__name__)


def create_gke_tools(gke_config: GKEConfig) -> list[ToolDef]:
    """Create all GKE tool definitions bound to a GKEConfig.

    Follows the exact same factory pattern as ``create_file_tools`` and
    ``create_shell_tools``: returns a list of ``ToolDef`` objects with
    closures that bind the config.

    When the cluster is detected as GKE Autopilot, node-level tools
    (``kubectl_top`` with ``resource_type="nodes"`` and ``get_node_conditions``)
    return an immediate informational message instead of calling the K8s API.
    """
    is_autopilot = _clients.detect_autopilot(gke_config)

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
        return kubectl.kubectl_top(resource_type, gke_config=_cfg, name=name, namespace=namespace)

    tools = [
        ToolDef(
            name="kubectl_get",
            description=(
                "List or get Kubernetes resources from the connected GKE cluster. "
                "Supports pods, deployments, services, configmaps, secrets, hpa, "
                "ingress, nodes, namespaces, statefulsets, daemonsets, jobs, cronjobs, "
                "pv, pvc, serviceaccounts, endpoints, networkpolicies, replicasets, "
                "roles, clusterroles, rolebindings, clusterrolebindings, "
                "storageclasses, volumeattachments, csidrivers, csinodes, "
                "limitranges, endpointslices, priorityclasses, runtimeclasses. "
                "Use resource='all' to list pods, services, deployments, replicasets, "
                "statefulsets, daemonsets, jobs, cronjobs, and hpa at once. "
                "Read-only — does not modify any resources."
            ),
            categories=frozenset({KUBERNETES}),
            parameters=[
                ToolParam(
                    name="resource",
                    type="string",
                    description=(
                        "Kubernetes resource type to list (e.g. 'pods', 'deployments', "
                        "'services', 'nodes', or comma-separated like 'pods,deployments')"
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
                    name="output",
                    type="string",
                    description="Output format: 'table' (default), 'yaml', 'json', 'wide', or 'name'.",
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
            execute=lambda resource, name=None, namespace="default", output="table",
                    label_selector=None, field_selector=None, _cfg=gke_config: kubectl.kubectl_get(
                resource,
                gke_config=_cfg,
                name=name,
                namespace=namespace,
                output=output,
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
            categories=frozenset({KUBERNETES}),
            parameters=[
                ToolParam(
                    name="resource_type",
                    type="string",
                    description="Kubernetes resource type (e.g. 'pod', 'deployment', 'service', 'hpa')",
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
            execute=lambda resource_type=None, name=None, namespace="default", resource=None,
                    _cfg=gke_config: (
                ToolResult(output="Error: resource_type (or resource) and name are required")
                if not (resource_type or resource) or not name
                else kubectl.kubectl_describe(
                    resource_type or resource, name, gke_config=_cfg, namespace=namespace,
                )
            ),
        ),
        ToolDef(
            name="kubectl_logs",
            description=(
                "Retrieve logs from a Kubernetes pod. Automatically fetches previous "
                "container logs when current container is in CrashLoopBackOff. "
                "For multi-container pods, auto-detects the app container by filtering "
                "out known sidecars (istio, datadog, linkerd, envoy, init containers); "
                "retries automatically when a single app container is found. "
                "Read-only — does not modify any resources."
            ),
            categories=frozenset({KUBERNETES, LOGGING}),
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
                    description="Container name. Required for multi-container pods. Auto-detected if omitted.",
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
            cacheable=False,
            execute=lambda pod, namespace="default", container=None, tail_lines=100,
                    since=None, _cfg=gke_config: kubectl.kubectl_logs(
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
                "For pods, returns per-container metrics — one row per container with a "
                "CONTAINER column. Sum container rows within each pod to get pod-level totals. "
                "Requires the metrics-server to be installed in the cluster. "
                "Read-only — does not modify any resources."
            ),
            categories=frozenset({KUBERNETES}),
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
            cacheable=False,
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
            categories=frozenset({KUBERNETES_WRITE}),
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
            cacheable=False,
            execute=lambda resource, name, replicas, namespace="default",
                    _cfg=gke_config: mutations.kubectl_scale(
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
            categories=frozenset({KUBERNETES_WRITE}),
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
            cacheable=False,
            execute=lambda resource, name, namespace="default",
                    _cfg=gke_config: mutations.kubectl_restart(
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
            categories=frozenset({KUBERNETES_WRITE}),
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
            cacheable=False,
            execute=lambda resource, name, labels, namespace="default",
                    _cfg=gke_config: mutations.kubectl_label(
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
            categories=frozenset({KUBERNETES_WRITE}),
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
            cacheable=False,
            execute=lambda resource, name, annotations, namespace="default",
                    _cfg=gke_config: mutations.kubectl_annotate(
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
            categories=frozenset({KUBERNETES}),
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
                    _cfg=gke_config: diagnostics.get_events(
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
            categories=frozenset({KUBERNETES}),
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
                    _cfg=gke_config: diagnostics.get_rollout_status(
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
            categories=frozenset({KUBERNETES}),
            parameters=[
                ToolParam(
                    name="name",
                    type="string",
                    description="Specific node name for detailed view. Omit to list all nodes.",
                    required=False,
                ),
            ],
            execute=lambda name=None, _cfg=gke_config: diagnostics.get_node_conditions(
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
            categories=frozenset({KUBERNETES}),
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
                    _cfg=gke_config: diagnostics.get_container_status(
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
            categories=frozenset({KUBERNETES}),
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
            cacheable=False,
            execute=lambda pod_name, namespace, command, container=None, timeout=30,
                    _cfg=gke_config: security.exec_command(
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
            categories=frozenset({KUBERNETES}),
            execute=lambda verb, resource, namespace, service_account=None, resource_name=None,
                    _cfg=gke_config: security.check_rbac(
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
            categories=frozenset({KUBERNETES}),
            execute=lambda name, namespace, revision=None,
                    _cfg=gke_config: diagnostics.get_rollout_history(
                name, gke_config=_cfg, namespace=namespace, revision=revision,
            ),
        ),
        ToolDef(
            name="discover_workloads",
            description=(
                "Discover all workloads running in the cluster or a specific namespace. "
                "Returns a summary table of deployments, statefulsets, daemonsets, and "
                "optionally jobs/cronjobs and Argo Rollouts. Shows ready/desired replicas, restarts, and "
                "age. Unhealthy workloads are listed first. Use this BEFORE drilling into "
                "specific resources to get a high-level overview of what is running. "
                "Results are cached for 60 seconds. Read-only — does not modify any resources."
            ),
            parameters=[
                ToolParam(
                    name="namespace",
                    type="string",
                    description=(
                        "Namespace to scan. Leave empty or use 'all' for all namespaces."
                    ),
                    required=False,
                ),
                ToolParam(
                    name="include_jobs",
                    type="boolean",
                    description="Include Jobs and CronJobs in the discovery (default: false).",
                    required=False,
                ),
                ToolParam(
                    name="include_rollouts",
                    type="boolean",
                    description=(
                        "Include Argo Rollouts in the discovery (default: false). "
                        "Gracefully skipped if Argo Rollouts CRDs are not installed."
                    ),
                    required=False,
                ),
                ToolParam(
                    name="force_refresh",
                    type="boolean",
                    description="Bypass cache and re-scan the cluster (default: false).",
                    required=False,
                ),
            ],
            categories=frozenset({KUBERNETES}),
            execute=lambda namespace="", include_jobs=False, include_rollouts=False,
                    force_refresh=False,
                    _cfg=gke_config: discovery.discover_workloads(
                gke_config=_cfg, namespace=namespace, include_jobs=include_jobs,
                include_rollouts=include_rollouts, force_refresh=force_refresh,
            ),
        ),
        ToolDef(
            name="discover_service_mesh",
            description=(
                "Discover service mesh installations in the cluster (Istio, Linkerd, "
                "Consul). Detects meshes via control-plane namespace existence, CRD "
                "inspection, and sidecar proxy presence on pods. Reports mesh name, "
                "version, control-plane health, CRD counts, and sidecar injection "
                "coverage. Handles multi-mesh scenarios. Use BEFORE investigating "
                "traffic routing or mTLS issues. Results are cached for 60 seconds. "
                "Read-only — does not modify any resources."
            ),
            parameters=[
                ToolParam(
                    name="namespace",
                    type="string",
                    description=(
                        "Namespace to scope sidecar scanning. Leave empty for all namespaces."
                    ),
                    required=False,
                ),
                ToolParam(
                    name="force_refresh",
                    type="boolean",
                    description="Bypass cache and re-scan the cluster (default: false).",
                    required=False,
                ),
            ],
            categories=frozenset({KUBERNETES}),
            execute=lambda namespace="", force_refresh=False,
                    _cfg=gke_config: discovery.discover_service_mesh(
                gke_config=_cfg, namespace=namespace, force_refresh=force_refresh,
            ),
        ),
        ToolDef(
            name="discover_network_topology",
            description=(
                "Discover network topology: services (grouped by type), endpoints "
                "(ready/not-ready counts), ingresses (hosts, paths, TLS), and "
                "network policies (pod selectors, ingress/egress rules). Provides "
                "a comprehensive view of cluster networking. Use BEFORE debugging "
                "connectivity issues or reviewing network security posture. Results "
                "are cached for 60 seconds. Read-only — does not modify any resources."
            ),
            parameters=[
                ToolParam(
                    name="namespace",
                    type="string",
                    description=(
                        "Namespace to scan. Leave empty for all namespaces."
                    ),
                    required=False,
                ),
                ToolParam(
                    name="force_refresh",
                    type="boolean",
                    description="Bypass cache and re-scan the cluster (default: false).",
                    required=False,
                ),
            ],
            categories=frozenset({KUBERNETES}),
            execute=lambda namespace="", force_refresh=False,
                    _cfg=gke_config: discovery.discover_network_topology(
                gke_config=_cfg, namespace=namespace, force_refresh=force_refresh,
            ),
        ),
        ToolDef(
            name="discover_dependencies",
            description=(
                "Map service-to-service dependencies for a given Kubernetes Service. "
                "Resolves the service to its backing pods, scans container environment "
                "variables for service references (hostnames, URLs, endpoints), and "
                "reads Istio VirtualServices (if installed) to extract upstream/downstream "
                "call topology. Sensitive environment variables (passwords, secrets, tokens, "
                "API keys) are NEVER included — only safe hostnames are reported. "
                "Use for cascading failure analysis, dependency graph construction, and "
                "understanding service call chains. Results are cached for 60 seconds. "
                "Read-only — does not modify any resources."
            ),
            parameters=[
                ToolParam(
                    name="namespace",
                    type="string",
                    description="Kubernetes namespace to scan for service dependencies.",
                ),
                ToolParam(
                    name="service_name",
                    type="string",
                    description="Name of a specific Kubernetes Service to analyse. Omit to scan all pods in the namespace.",
                    required=False,
                ),
                ToolParam(
                    name="force_refresh",
                    type="boolean",
                    description="Bypass cache and re-scan (default: false).",
                    required=False,
                ),
            ],
            categories=frozenset({KUBERNETES}),
            execute=lambda namespace, service_name="", force_refresh=False,
                    _cfg=gke_config: discovery.discover_dependencies(
                namespace, service_name=service_name, gke_config=_cfg, force_refresh=force_refresh,
            ),
        ),
        # ── Mesh introspection tools ─────────────────────────
        ToolDef(
            name="get_mesh_overview",
            description=(
                "Show Istio/ASM mesh overview: presence, version, control-plane health, "
                "and per-namespace sidecar injection status. Detects both open-source "
                "Istio and Google-managed Anthos Service Mesh (ASM). Returns 'No service "
                "mesh detected' when no mesh is found. Use BEFORE investigating mesh "
                "config or security to confirm a mesh is installed. Results are cached "
                "for 30 seconds. Read-only — does not modify any resources."
            ),
            parameters=[
                ToolParam(
                    name="namespace",
                    type="string",
                    description=(
                        "Filter injection status to a specific namespace. "
                        "Leave empty for all namespaces."
                    ),
                    required=False,
                ),
                ToolParam(
                    name="force_refresh",
                    type="boolean",
                    description="Bypass cache and re-scan the cluster (default: false).",
                    required=False,
                ),
            ],
            categories=frozenset({MESH}),
            execute=lambda namespace="", force_refresh=False,
                    _cfg=gke_config: mesh.get_mesh_overview(
                gke_config=_cfg, namespace=namespace, force_refresh=force_refresh,
            ),
        ),
        ToolDef(
            name="get_mesh_config",
            description=(
                "Show Istio/ASM traffic management configuration: VirtualServices, "
                "DestinationRules, and Gateways. Displays routing rules, traffic "
                "splitting weights, load balancer settings, circuit breakers, and "
                "TLS configuration. Auto-detects the CRD API version. Use for "
                "debugging traffic routing issues, canary deployments, and service "
                "connectivity problems. Results are cached for 30 seconds. "
                "Read-only — does not modify any resources."
            ),
            parameters=[
                ToolParam(
                    name="namespace",
                    type="string",
                    description=(
                        "Filter to a specific namespace. Leave empty for all namespaces."
                    ),
                    required=False,
                ),
                ToolParam(
                    name="force_refresh",
                    type="boolean",
                    description="Bypass cache and re-scan the cluster (default: false).",
                    required=False,
                ),
            ],
            categories=frozenset({MESH}),
            execute=lambda namespace="", force_refresh=False,
                    _cfg=gke_config: mesh.get_mesh_config(
                gke_config=_cfg, namespace=namespace, force_refresh=force_refresh,
            ),
        ),
        ToolDef(
            name="get_mesh_security",
            description=(
                "Show Istio/ASM security configuration: PeerAuthentication (mTLS "
                "enforcement), AuthorizationPolicy (RBAC rules), and "
                "RequestAuthentication (JWT validation). Use to verify mTLS is "
                "enforced, check authorization rules, and audit JWT policies. "
                "Auto-detects the CRD API version. Results are cached for 30 seconds. "
                "Read-only — does not modify any resources."
            ),
            parameters=[
                ToolParam(
                    name="namespace",
                    type="string",
                    description=(
                        "Filter to a specific namespace. Leave empty for all namespaces."
                    ),
                    required=False,
                ),
                ToolParam(
                    name="force_refresh",
                    type="boolean",
                    description="Bypass cache and re-scan the cluster (default: false).",
                    required=False,
                ),
            ],
            categories=frozenset({MESH}),
            execute=lambda namespace="", force_refresh=False,
                    _cfg=gke_config: mesh.get_mesh_security(
                gke_config=_cfg, namespace=namespace, force_refresh=force_refresh,
            ),
        ),
        ToolDef(
            name="get_sidecar_status",
            description=(
                "Show sidecar injection status for every pod in the cluster. "
                "Checks for istio-proxy container presence, extracts sidecar version, "
                "identifies pod owners (ReplicaSet, Deployment), and detects injection "
                "anomalies: MISSING (pod in injection-enabled namespace without sidecar) "
                "or UNEXPECTED (pod with sidecar in non-injected namespace). Use to "
                "audit sidecar coverage and find misconfigured workloads. Results are "
                "cached for 30 seconds. Read-only — does not modify any resources."
            ),
            parameters=[
                ToolParam(
                    name="namespace",
                    type="string",
                    description=(
                        "Filter to a specific namespace. Leave empty for all namespaces."
                    ),
                    required=False,
                ),
                ToolParam(
                    name="force_refresh",
                    type="boolean",
                    description="Bypass cache and re-scan the cluster (default: false).",
                    required=False,
                ),
            ],
            categories=frozenset({MESH}),
            execute=lambda namespace="", force_refresh=False,
                    _cfg=gke_config: mesh.get_sidecar_status(
                gke_config=_cfg, namespace=namespace, force_refresh=force_refresh,
            ),
        ),
        # ── Labels tool (always registered) ──────────────────
        ToolDef(
            name="kubectl_get_labels",
            description=(
                "Get labels and annotations for Kubernetes resources. Supports "
                "server-side label filtering and client-side annotation filtering."
            ),
            parameters=[
                ToolParam(
                    name="resource_type",
                    type="string",
                    description="K8s resource type (pods, deployments, services, etc.)",
                ),
                ToolParam(
                    name="namespace",
                    type="string",
                    description="Namespace to query",
                    required=False,
                ),
                ToolParam(
                    name="name",
                    type="string",
                    description="Specific resource name. If empty, lists all matching resources.",
                    required=False,
                ),
                ToolParam(
                    name="label_filter",
                    type="string",
                    description="Label selector for server-side filtering (e.g., 'app=nginx', 'app in (web,api)')",
                    required=False,
                ),
                ToolParam(
                    name="annotation_filter",
                    type="string",
                    description="Annotation key or key=value for client-side filtering",
                    required=False,
                ),
            ],
            categories=frozenset({KUBERNETES}),
            execute=lambda resource_type, namespace="default", name="",
                    label_filter="", annotation_filter="",
                    _cfg=gke_config: kubectl_get_labels(
                resource_type=resource_type,
                gke_config=_cfg,
                namespace=namespace,
                name=name,
                label_filter=label_filter,
                annotation_filter=annotation_filter,
            ),
        ),
        # ── Datadog observability tools ───────────────────────
        ToolDef(
            name="get_datadog_config",
            description=(
                "Detect and summarize Datadog observability configuration in GKE workloads. "
                "Scans deployments for Datadog annotations (ad.datadoghq.com/, "
                "admission.datadoghq.com/), labels (tags.datadoghq.com/), and environment "
                "variables (DD_AGENT_HOST, DD_TRACE_ENABLED, DD_SERVICE, etc.). "
                "Also checks for a Datadog agent DaemonSet in common namespaces. "
                "Detects common misconfigurations such as APM enabled without agent host, "
                "or admission webhook without a service tag. "
                "Read-only — does not modify any resources."
            ),
            parameters=[
                ToolParam(
                    name="namespace",
                    type="string",
                    description="Kubernetes namespace to scan (default: 'default').",
                    required=False,
                ),
                ToolParam(
                    name="deployment",
                    type="string",
                    description="Optional deployment name to filter to a single deployment.",
                    required=False,
                ),
            ],
            categories=frozenset({KUBERNETES}),
            execute=lambda namespace="default", deployment="",
                    _cfg=gke_config: get_datadog_config(
                gke_config=_cfg, namespace=namespace, deployment=deployment,
            ),
        ),
        # ── Scaling tools ─────────────────────────────────────
        ToolDef(
            name="get_scaling_status",
            description=(
                "Fetch HPA and VPA scaling status for a deployment, including "
                "current/min/max replicas, CPU/memory targets, and VPA recommendations. "
                "Useful for diagnosing ceiling-hit scenarios (current == max replicas under "
                "load) and VPA-vs-HPA conflicts. Read-only — does not modify any resources."
            ),
            parameters=[
                ToolParam(
                    name="name",
                    type="string",
                    description="Deployment name to inspect.",
                ),
                ToolParam(
                    name="namespace",
                    type="string",
                    description="Kubernetes namespace (default: 'default').",
                    required=False,
                ),
            ],
            categories=frozenset({SCALING}),
            execute=lambda name, namespace="default",
                    _cfg=gke_config: get_scaling_status(
                name, gke_config=_cfg, namespace=namespace,
            ),
        ),
        ToolDef(
            name="check_metrics_api_health",
            description=(
                "Probe Kubernetes aggregated metrics API groups (metrics.k8s.io, "
                "custom.metrics.k8s.io, external.metrics.k8s.io) and report their "
                "health status. Shows whether Metrics Server, custom metrics adapter, "
                "and external metrics adapter are installed and available. Essential "
                "for diagnosing HPA scaling failures caused by a broken metrics pipeline. "
                "Read-only — does not modify any resources."
            ),
            parameters=None,
            categories=frozenset({SCALING}),
            execute=lambda _cfg=gke_config: check_metrics_api_health(
                gke_config=_cfg,
            ),
        ),
        # ── Custom / external metrics query tools ──────────────
        ToolDef(
            name="query_custom_metrics",
            description=(
                "Query custom metrics from the custom.metrics.k8s.io API group. "
                "When metric_name is empty, lists all available custom metrics. "
                "When provided, fetches the named metric's current value and labels. "
                "Use after check_metrics_api_health confirms the custom metrics API is "
                "available, or to verify an HPA-referenced custom metric exists and has data. "
                "Read-only — does not modify any resources."
            ),
            parameters=[
                ToolParam(
                    name="metric_name",
                    type="string",
                    description=(
                        "Custom metric name to query (e.g. 'requests_per_second'). "
                        "Leave empty to list all available custom metrics."
                    ),
                    required=False,
                ),
                ToolParam(
                    name="namespace",
                    type="string",
                    description="Kubernetes namespace to scope the query (default: cluster-wide).",
                    required=False,
                ),
            ],
            categories=frozenset({MONITORING}),
            execute=lambda metric_name="", namespace="",
                    _cfg=gke_config: query_custom_metrics(
                metric_name, gke_config=_cfg, namespace=namespace,
            ),
        ),
        ToolDef(
            name="query_external_metrics",
            description=(
                "Query external metrics from the external.metrics.k8s.io API group. "
                "External metrics come from cloud monitoring systems (e.g. Cloud "
                "Monitoring, Datadog) and are not tied to Kubernetes objects. "
                "Use to verify an HPA-referenced external metric exists and has data, "
                "e.g. pubsub.googleapis.com|subscription|num_undelivered_messages. "
                "Read-only — does not modify any resources."
            ),
            parameters=[
                ToolParam(
                    name="metric_name",
                    type="string",
                    description=(
                        "External metric name to query (e.g. "
                        "'pubsub.googleapis.com|subscription|num_undelivered_messages')."
                    ),
                ),
                ToolParam(
                    name="namespace",
                    type="string",
                    description=(
                        "Kubernetes namespace to scope the query. If omitted, the "
                        "query uses the default namespace ('default')."
                    ),
                    required=False,
                ),
            ],
            categories=frozenset({MONITORING}),
            execute=lambda metric_name, namespace="",
                    _cfg=gke_config: query_external_metrics(
                metric_name, gke_config=_cfg, namespace=namespace,
            ),
        ),
        # ── Cloud Monitoring metrics tools ────────────────────
        ToolDef(
            name="get_pod_metrics",
            description=(
                "Fetch historical CPU and memory metrics from Google Cloud Monitoring "
                "for GKE pods matching a namespace and pod-name prefix. "
                "Returns a Markdown summary table with per-pod average, max, latest "
                "value, and trend direction (↑ rising, ↓ falling, → stable) over "
                "the requested time window. "
                "Use this for HISTORICAL trends (default: last 60 minutes). "
                "For real-time current usage, use ``kubectl_top`` instead. "
                "Requires ``roles/monitoring.viewer`` on the GCP project. "
                "Read-only — does not modify any resources."
            ),
            parameters=[
                ToolParam(
                    name="namespace",
                    type="string",
                    description="Kubernetes namespace to query (e.g. 'default', 'production').",
                ),
                ToolParam(
                    name="pod_name_prefix",
                    type="string",
                    description=(
                        "Pod name prefix to match (e.g. 'frontend-' matches "
                        "'frontend-abc-123', 'frontend-xyz-456')."
                    ),
                ),
                ToolParam(
                    name="window_minutes",
                    type="integer",
                    description="Time window in minutes to query (default: 60). Max recommended: 1440 (24h).",
                    required=False,
                ),
                ToolParam(
                    name="metric_type",
                    type="string",
                    description=(
                        "Which metrics to fetch: 'cpu', 'memory', or 'all' (default: 'all'). "
                        "Use 'cpu' or 'memory' to reduce API calls when only one metric is needed."
                    ),
                    required=False,
                ),
            ],
            categories=frozenset({MONITORING, KUBERNETES}),
            execute=lambda namespace, pod_name_prefix, window_minutes=60,
                    metric_type="all", _cfg=gke_config: get_pod_metrics(
                namespace=namespace,
                pod_name_prefix=pod_name_prefix,
                gke_config=_cfg,
                window_minutes=window_minutes,
                metric_type=metric_type,
            ),
        ),
    ]

    # ── Helm tools (conditional on helm_enabled) ─────────────
    if gke_config.helm_enabled:
        tools.extend([
            ToolDef(
                name="helm_list_releases",
                description=(
                    "List all Helm releases in a namespace. Queries Kubernetes secrets "
                    "with owner=helm label selector. Shows name, chart, version, status, "
                    "and app version. Only shows the latest revision per release. "
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
                        name="force_refresh",
                        type="boolean",
                        description="Bypass cache and re-scan (default: false).",
                        required=False,
                    ),
                ],
                categories=frozenset({HELM}),
                execute=lambda namespace="default", force_refresh=False,
                        _cfg=gke_config: helm_list_releases(
                    gke_config=_cfg, namespace=namespace, force_refresh=force_refresh,
                ),
            ),
            ToolDef(
                name="helm_release_status",
                description=(
                    "Get detailed status of a specific Helm release. Shows chart, "
                    "version, app version, first/last deployed timestamps, description, "
                    "and notes. Read-only — does not modify any resources."
                ),
                parameters=[
                    ToolParam(
                        name="release_name",
                        type="string",
                        description="Name of the Helm release",
                    ),
                    ToolParam(
                        name="namespace",
                        type="string",
                        description="Kubernetes namespace (default: 'default')",
                        required=False,
                    ),
                    ToolParam(
                        name="force_refresh",
                        type="boolean",
                        description="Bypass cache and re-scan (default: false).",
                        required=False,
                    ),
                ],
                categories=frozenset({HELM}),
                execute=lambda release_name, namespace="default", force_refresh=False,
                        _cfg=gke_config: helm_release_status(
                    gke_config=_cfg, release_name=release_name, namespace=namespace,
                    force_refresh=force_refresh,
                ),
            ),
            ToolDef(
                name="helm_release_history",
                description=(
                    "Get revision history of a Helm release. Shows revision number, "
                    "status, chart version, app version, description, and deployment "
                    "timestamp for each revision (newest first). "
                    "Read-only — does not modify any resources."
                ),
                parameters=[
                    ToolParam(
                        name="release_name",
                        type="string",
                        description="Name of the Helm release",
                    ),
                    ToolParam(
                        name="namespace",
                        type="string",
                        description="Kubernetes namespace (default: 'default')",
                        required=False,
                    ),
                    ToolParam(
                        name="force_refresh",
                        type="boolean",
                        description="Bypass cache and re-scan (default: false).",
                        required=False,
                    ),
                ],
                categories=frozenset({HELM}),
                execute=lambda release_name, namespace="default", force_refresh=False,
                        _cfg=gke_config: helm_release_history(
                    gke_config=_cfg, release_name=release_name, namespace=namespace,
                    force_refresh=force_refresh,
                ),
            ),
            ToolDef(
                name="helm_release_values",
                description=(
                    "Get the values used in a Helm release. By default returns only "
                    "user-supplied overrides. Set all_values=true to include chart "
                    "defaults merged with user overrides. Output is YAML formatted. "
                    "Read-only — does not modify any resources."
                ),
                parameters=[
                    ToolParam(
                        name="release_name",
                        type="string",
                        description="Name of the Helm release",
                    ),
                    ToolParam(
                        name="namespace",
                        type="string",
                        description="Kubernetes namespace (default: 'default')",
                        required=False,
                    ),
                    ToolParam(
                        name="all_values",
                        type="boolean",
                        description="Include chart defaults merged with overrides (default: false).",
                        required=False,
                    ),
                    ToolParam(
                        name="force_refresh",
                        type="boolean",
                        description="Bypass cache and re-scan (default: false).",
                        required=False,
                    ),
                ],
                categories=frozenset({HELM}),
                execute=lambda release_name, namespace="default", all_values=False,
                        force_refresh=False,
                        _cfg=gke_config: helm_release_values(
                    gke_config=_cfg, release_name=release_name, namespace=namespace,
                    all_values=all_values, force_refresh=force_refresh,
                ),
            ),
        ])

    # ── ArgoCD tools (auto-detect or explicit toggle) ─────────
    _acd_setting = gke_config.argocd_enabled
    if _acd_setting is False:
        _argocd_active = False
        logger.debug("ArgoCD tools skipped (argocd_enabled=False).")
    elif _acd_setting is True:
        _argocd_active = True
        logger.info("ArgoCD tools force-enabled (argocd_enabled=True).")
    else:
        # None → auto-detect via CRD + annotation fallback
        _acd_api_client = None
        _acd_clients = _clients._create_k8s_clients(gke_config)
        if not isinstance(_acd_clients, ToolResult):
            _acd_api_client = _acd_clients[3]  # api_client from tuple
        _argocd_active = detect_argocd(
            namespace=gke_config.default_namespace,
            api_client=_acd_api_client,
        )
        if not _argocd_active:
            logger.debug(
                "ArgoCD not detected — skipping ArgoCD tools. "
                "Set argocd_enabled=true to force-enable."
            )

    if _argocd_active:
        tools.extend([
            ToolDef(
                name="argocd_list_applications",
                description=(
                    "List all ArgoCD Applications. Namespace is auto-discovered when not "
                    "specified — probes common namespaces (argocd, argo-cd, argocd-system, "
                    "gitops, argo), then falls back to a cluster-wide scan if none match. "
                    "Shows name, project, sync status, health status, source "
                    "repo, target revision, and destination. Read-only — does not modify any resources."
                ),
                parameters=[
                    ToolParam(
                        name="namespace",
                        type="string",
                        description="Namespace where ArgoCD Applications live (auto-discovered if not specified)",
                        required=False,
                    ),
                ],
                categories=frozenset({ARGOCD}),
                execute=lambda namespace="",
                        _cfg=gke_config: argocd_list_applications(
                    namespace=namespace,
                ),
            ),
            ToolDef(
                name="argocd_app_status",
                description=(
                    "Get detailed status of a specific ArgoCD Application. Namespace is "
                    "auto-discovered when not specified — probes common namespaces then falls "
                    "back to a cluster-wide scan. Shows sync status, health status, "
                    "source info, destination, sync policy, conditions, and last operation "
                    "state. Read-only — does not modify any resources."
                ),
                parameters=[
                    ToolParam(
                        name="app_name",
                        type="string",
                        description="Name of the ArgoCD Application",
                    ),
                    ToolParam(
                        name="namespace",
                        type="string",
                        description="Namespace where ArgoCD Applications live (auto-discovered if not specified)",
                        required=False,
                    ),
                ],
                categories=frozenset({ARGOCD}),
                execute=lambda app_name, namespace="",
                        _cfg=gke_config: argocd_app_status(
                    app_name=app_name, namespace=namespace,
                ),
            ),
            ToolDef(
                name="argocd_app_history",
                description=(
                    "Get deployment history of an ArgoCD Application. Namespace is "
                    "auto-discovered when not specified — probes common namespaces then falls "
                    "back to a cluster-wide scan. Shows past deployments with "
                    "revision, deployment time, and source info (most recent first). "
                    "Read-only — does not modify any resources."
                ),
                parameters=[
                    ToolParam(
                        name="app_name",
                        type="string",
                        description="Name of the ArgoCD Application",
                    ),
                    ToolParam(
                        name="namespace",
                        type="string",
                        description="Namespace where ArgoCD Applications live (auto-discovered if not specified)",
                        required=False,
                    ),
                ],
                categories=frozenset({ARGOCD}),
                execute=lambda app_name, namespace="",
                        _cfg=gke_config: argocd_app_history(
                    app_name=app_name, namespace=namespace,
                ),
            ),
            ToolDef(
                name="argocd_app_diff",
                description=(
                    "Show resources that are out-of-sync for an ArgoCD Application. "
                    "Namespace is auto-discovered when not specified — probes common "
                    "namespaces then falls back to a cluster-wide scan. Returns resources "
                    "where sync status is not Synced or health status is not Healthy. "
                    "Read-only — does not modify any resources."
                ),
                parameters=[
                    ToolParam(
                        name="app_name",
                        type="string",
                        description="Name of the ArgoCD Application",
                    ),
                    ToolParam(
                        name="namespace",
                        type="string",
                        description="Namespace where ArgoCD Applications live (auto-discovered if not specified)",
                        required=False,
                    ),
                ],
                categories=frozenset({ARGOCD}),
                execute=lambda app_name, namespace="",
                        _cfg=gke_config: argocd_app_diff(
                    app_name=app_name, namespace=namespace,
                ),
            ),
            ToolDef(
                name="argocd_app_managed_resources",
                description=(
                    "List all resources managed by an ArgoCD Application. Namespace is "
                    "auto-discovered when not specified — probes common namespaces then falls "
                    "back to a cluster-wide scan. Shows resources grouped by kind "
                    "with group, name, namespace, sync status, health status, and pruning "
                    "requirements. Read-only — does not modify any resources."
                ),
                parameters=[
                    ToolParam(
                        name="app_name",
                        type="string",
                        description="Name of the ArgoCD Application",
                    ),
                    ToolParam(
                        name="namespace",
                        type="string",
                        description="Namespace where ArgoCD Applications live (auto-discovered if not specified)",
                        required=False,
                    ),
                ],
                categories=frozenset({ARGOCD}),
                execute=lambda app_name, namespace="",
                        _cfg=gke_config: argocd_app_managed_resources(
                    app_name=app_name, namespace=namespace,
                ),
            ),
        ])

    # ── Datadog API tools (conditional on datadog.enabled) ────
    from vaig.core.config import get_settings as _get_settings  # noqa: WPS433

    _dd_config = _get_settings().datadog
    if _dd_config.enabled:
        logger.info("Datadog API tools registered (site=%s)", _dd_config.site)
        tools.extend([
            ToolDef(
                name="query_datadog_metrics",
                description=(
                    "Query Datadog metrics for a GKE cluster using the Datadog Metrics v1 API. "
                    "Supports built-in metric templates: cpu, memory, restarts, network_in, "
                    "network_out, disk_read, disk_write. "
                    "Optionally filter by service and env tags (e.g. from DD_SERVICE/DD_ENV labels) "
                    "to scope the query to a specific workload. "
                    "Returns average, maximum, and latest values per series over the requested "
                    "time window. Read-only — does not modify any resources."
                ),
                parameters=[
                    ToolParam(
                        name="cluster_name",
                        type="string",
                        description="GKE cluster name used to scope the metric query",
                    ),
                    ToolParam(
                        name="metric",
                        type="string",
                        description=(
                            "Metric template to query: cpu, memory, restarts, network_in, "
                            "network_out, disk_read, or disk_write (default: cpu)"
                        ),
                        required=False,
                    ),
                    ToolParam(
                        name="from_ts",
                        type="integer",
                        description="Unix timestamp for the start of the query window (defaults to now-3600)",
                        required=False,
                    ),
                    ToolParam(
                        name="to_ts",
                        type="integer",
                        description="Unix timestamp for the end of the query window (defaults to now)",
                        required=False,
                    ),
                    ToolParam(
                        name="service",
                        type="string",
                        description=(
                            "Optional Datadog service tag to narrow the query "
                            "(e.g. value of DD_SERVICE or tags.datadoghq.com/service label)"
                        ),
                        required=False,
                    ),
                    ToolParam(
                        name="env",
                        type="string",
                        description=(
                            "Optional Datadog environment tag to narrow the query "
                            "(e.g. value of DD_ENV or tags.datadoghq.com/env label)"
                        ),
                        required=False,
                    ),
                ],
                categories=frozenset({DATADOG}),
                execute=lambda cluster_name, metric="cpu", from_ts=0, to_ts=0,
                        service=None, env=None,
                        _dd=_dd_config: query_datadog_metrics(
                    cluster_name=cluster_name, metric=metric,
                    from_ts=from_ts, to_ts=to_ts, service=service, env=env, config=_dd,
                ),
            ),
            ToolDef(
                name="get_datadog_monitors",
                description=(
                    "Fetch active Datadog monitors using the Datadog Monitors v1 API. "
                    "Returns monitors filtered by state (default: Alert) and optionally "
                    "by cluster name, service, and environment tags. Shows monitor ID, name, type, "
                    "and current state. Read-only — does not modify any resources."
                ),
                parameters=[
                    ToolParam(
                        name="cluster_name",
                        type="string",
                        description="Optional cluster name to filter monitors by tag cluster_name:<name>",
                        required=False,
                    ),
                    ToolParam(
                        name="state",
                        type="string",
                        description="Monitor state to filter on: Alert, Warn, No Data (default: Alert)",
                        required=False,
                    ),
                    ToolParam(
                        name="service",
                        type="string",
                        description=(
                            "Optional Datadog service tag to filter monitors "
                            "(e.g. value of DD_SERVICE or tags.datadoghq.com/service label)"
                        ),
                        required=False,
                    ),
                    ToolParam(
                        name="env",
                        type="string",
                        description=(
                            "Optional Datadog environment tag to filter monitors "
                            "(e.g. value of DD_ENV or tags.datadoghq.com/env label)"
                        ),
                        required=False,
                    ),
                ],
                categories=frozenset({DATADOG}),
                execute=lambda cluster_name="", state="Alert",
                        service=None, env=None,
                        _dd=_dd_config: get_datadog_monitors(
                    cluster_name=cluster_name, state=state, service=service, env=env, config=_dd,
                ),
            ),
            ToolDef(
                name="get_datadog_service_catalog",
                description=(
                    "Fetch service ownership metadata from the Datadog Service Catalog (Service Definition v2 API). "
                    "Returns service name, team, language, and tier ownership metadata for a specific service. "
                    "Always call this tool — provide service_name when resolved from Kubernetes pod labels, "
                    "but call it even without service_name: the tool handles the empty case gracefully and "
                    "returns guidance on how to resolve service identity. "
                    "Resolve service_name from 'tags.datadoghq.com/service' pod labels first, "
                    "then app.kubernetes.io/name, then app label, then deployment name. "
                    "Results are cached for 60 seconds. Read-only — does not modify any resources."
                ),
                parameters=[
                    ToolParam(
                        name="env",
                        type="string",
                        description="Datadog environment tag (e.g. production, staging) — default: production",
                        required=False,
                    ),
                    ToolParam(
                        name="cluster_name",
                        type="string",
                        description="Optional cluster name to include in the output header",
                        required=False,
                    ),
                    ToolParam(
                        name="service_name",
                        type="string",
                        description=(
                            "Service name to filter catalog results. "
                            "Resolve from Kubernetes pod labels in priority order: "
                            "(1) tags.datadoghq.com/service label, "
                            "(2) app.kubernetes.io/name label, "
                            "(3) app label, "
                            "(4) deployment or service name. "
                            "If service_name cannot be resolved, call without it — "
                            "the tool returns guidance on resolution."
                        ),
                        required=False,
                    ),
                ],
                categories=frozenset({DATADOG}),
                execute=lambda env="production", cluster_name="",
                        service_name=None,
                        _dd=_dd_config: get_datadog_service_catalog(
                    env=env, cluster_name=cluster_name, service_name=service_name, config=_dd,
                ),
            ),
            ToolDef(
                name="get_datadog_apm_services",
                description=(
                    "Fetch live APM trace metrics for a specific service from Datadog using Spans Events Search v2. "
                    "Returns throughput, error rate, and avg latency from actual span data. "
                    "Uses a fallback chain: POST /api/v2/spans/events/search → GET /api/v2/apm/traces/search → empty result with warning. "
                    "Use this for real-time performance data — NOT for ownership metadata (use get_datadog_service_catalog for that). "
                    "Provide service_name when known — resolve it from Kubernetes pod labels before calling. "
                    "If service_name cannot be determined, call the tool anyway — it will return guidance on resolution. "
                    "Results are cached for 60 seconds. Read-only — does not modify any resources."
                ),
                parameters=[
                    ToolParam(
                        name="service_name",
                        type="string",
                        description=(
                            "Service name to query APM trace metrics for. Strongly recommended. "
                            "Must match the 'service' tag in Datadog APM. "
                            "Resolve from Kubernetes pod labels in priority order: "
                            "(1) tags.datadoghq.com/service label, "
                            "(2) app.kubernetes.io/name label, "
                            "(3) app label, "
                            "(4) the deployment or service name from Kubernetes."
                        ),
                        required=False,
                    ),
                    ToolParam(
                        name="env",
                        type="string",
                        description=(
                            "Datadog environment tag used to scope the APM query "
                            "(e.g. production, staging) — default: production"
                        ),
                        required=False,
                    ),
                    ToolParam(
                        name="hours_back",
                        type="number",
                        description=(
                            "Lookback window in hours for the APM query. "
                            "Defaults to 1 hour. Use fractional values for sub-hour windows "
                            "(e.g. 0.5 for 30 minutes, 4 for 4 hours)."
                        ),
                        required=False,
                    ),
                ],
                categories=frozenset({DATADOG}),
                execute=lambda service_name="", env="production", hours_back=1.0,
                        _dd=_dd_config: get_datadog_apm_services(
                    service_name=service_name, env=env, hours_back=hours_back, config=_dd,
                ),
            ),
            ToolDef(
                name="get_datadog_service_dependencies",
                description=(
                    "Fetch upstream and downstream service dependencies from the Datadog Service Dependencies v1 API. "
                    "Returns which services this service calls (downstream) and which services call it (upstream). "
                    "Requires service_name — resolve from Kubernetes pod labels before calling. "
                    "Also emits structured DependencyEdge data for dependency graph rendering. "
                    "Results are cached for 60 seconds. Read-only — does not modify any resources."
                ),
                parameters=[
                    ToolParam(
                        name="service_name",
                        type="string",
                        description=(
                            "Service name to look up dependencies for. Required — must match the "
                            "'service' tag in Datadog APM. Resolve from Kubernetes pod labels: "
                            "(1) tags.datadoghq.com/service, "
                            "(2) app.kubernetes.io/name, "
                            "(3) app label, "
                            "(4) deployment or service name."
                        ),
                    ),
                ],
                categories=frozenset({DATADOG}),
                execute=lambda service_name,
                        _dd=_dd_config: get_datadog_service_dependencies(
                    service_name=service_name, config=_dd,
                ),
            ),
            ToolDef(
                name="diagnose_datadog_metrics",
                description=(
                    "Run a diagnostic probe against the Datadog API to discover available metrics "
                    "and tag keys. Searches for kubernetes.* (infra) and trace.* (APM) metrics, "
                    "discovers host tag keys, and provides suggestions for metric_mode configuration. "
                    "Use this FIRST when Datadog metric queries return empty results — the diagnostic "
                    "output explains what metrics exist and what tags are available. "
                    "Read-only — does not modify any resources."
                ),
                parameters=[],
                categories=frozenset({DATADOG}),
                execute=lambda _dd=_dd_config: diagnose_datadog_metrics(config=_dd),
            ),
        ])
    else:
        logger.debug(
            "Datadog API tools skipped (enabled=False). "
            "Set datadog.enabled=true or provide API keys to enable."
        )

    # ── Argo Rollouts tools (auto-detect or explicit toggle) ──
    _ar_setting = gke_config.argo_rollouts_enabled
    if _ar_setting is False:
        _argo_rollouts_active = False
        logger.debug("Argo Rollouts tools skipped (argo_rollouts_enabled=False).")
    elif _ar_setting is True:
        _argo_rollouts_active = True
        logger.info("Argo Rollouts tools registered (forced via argo_rollouts_enabled=True).")
    else:
        # None → auto-detect via CRD presence; derive api_client from gke_config
        # so the probe targets the same cluster context as the configured tools.
        _ar_api_client = None
        _ar_clients = _clients._create_k8s_clients(gke_config)
        if not isinstance(_ar_clients, ToolResult):
            _ar_api_client = _ar_clients[3]  # ApiClient is the 4th element
        _argo_rollouts_active = detect_argo_rollouts(
            namespace=gke_config.default_namespace,
            api_client=_ar_api_client,
        )
        if _argo_rollouts_active:
            logger.info("Argo Rollouts CRD detected — registering Rollout tools.")
        else:
            logger.debug(
                "Argo Rollouts CRD not found — skipping rollout tools. "
                "Set argo_rollouts_enabled=true to force-enable."
            )

    if _argo_rollouts_active:
        tools.extend([
            ToolDef(
                name="kubectl_get_rollout",
                description=(
                    "List or inspect Argo Rollout resources in the cluster using the "
                    "Argo Rollouts CRD API (argoproj.io/v1alpha1). "
                    "Returns phase, replica counts (desired/ready/available/updated), "
                    "strategy (canary step index and weight, or blueGreen active/preview RS), "
                    "and condition messages. "
                    "Use without 'name' to list all Rollouts in a namespace; provide 'name' "
                    "for a single Rollout. "
                    "Use this BEFORE investigating pod failures in Rollout-managed workloads — "
                    "Rollouts replace Deployments and are the authoritative source of rollout state. "
                    "Read-only — does not modify any resources."
                ),
                parameters=[
                    ToolParam(
                        name="namespace",
                        type="string",
                        description="Kubernetes namespace to query (default: all namespaces)",
                        required=False,
                    ),
                    ToolParam(
                        name="name",
                        type="string",
                        description="Rollout name; omit to list all Rollouts in the namespace",
                        required=False,
                    ),
                ],
                categories=frozenset({ARGO_ROLLOUTS}),
                execute=lambda namespace="", name="",
                        _cfg=gke_config: kubectl_get_rollout(
                    namespace=namespace, name=name,
                ),
            ),
            ToolDef(
                name="kubectl_get_analysisrun",
                description=(
                    "List or inspect Argo Rollouts AnalysisRun resources "
                    "(argoproj.io/v1alpha1). "
                    "Returns phase (Running/Successful/Failed/Error), metric results, "
                    "and error messages for each metric provider. "
                    "Use to diagnose why a canary or blueGreen rollout was paused or "
                    "aborted due to a failing analysis. "
                    "Use without 'name' to list all AnalysisRuns in a namespace. "
                    "Read-only — does not modify any resources."
                ),
                parameters=[
                    ToolParam(
                        name="namespace",
                        type="string",
                        description="Kubernetes namespace to query (default: all namespaces)",
                        required=False,
                    ),
                    ToolParam(
                        name="name",
                        type="string",
                        description="AnalysisRun name; omit to list all AnalysisRuns in the namespace",
                        required=False,
                    ),
                ],
                categories=frozenset({ARGO_ROLLOUTS}),
                execute=lambda namespace="", name="",
                        _cfg=gke_config: kubectl_get_analysisrun(
                    namespace=namespace, name=name,
                ),
            ),
            ToolDef(
                name="kubectl_get_analysistemplate",
                description=(
                    "List or inspect Argo Rollouts AnalysisTemplate resources "
                    "(argoproj.io/v1alpha1). "
                    "Returns template name, namespace, and the list of defined metrics "
                    "with their provider types (Prometheus, Web, Job, Datadog, etc.). "
                    "Use to understand what analysis criteria are used by rollouts in "
                    "a namespace before investigating a failed AnalysisRun. "
                    "Read-only — does not modify any resources."
                ),
                parameters=[
                    ToolParam(
                        name="namespace",
                        type="string",
                        description="Kubernetes namespace to query (default: all namespaces)",
                        required=False,
                    ),
                    ToolParam(
                        name="name",
                        type="string",
                        description=(
                            "AnalysisTemplate name; omit to list all templates in the namespace"
                        ),
                        required=False,
                    ),
                ],
                categories=frozenset({ARGO_ROLLOUTS}),
                execute=lambda namespace="", name="",
                        _cfg=gke_config: kubectl_get_analysistemplate(
                    namespace=namespace, name=name,
                ),
            ),
            ToolDef(
                name="kubectl_get_cluster_analysis_template",
                description=(
                    "List or inspect Argo Rollouts ClusterAnalysisTemplate resources "
                    "(argoproj.io/v1alpha1). "
                    "ClusterAnalysisTemplates are cluster-scoped (not namespace-bound) and define "
                    "reusable analysis metrics (Prometheus, Datadog, Web, Job, etc.) that can be "
                    "referenced by Rollouts across any namespace. "
                    "Returns template name and the list of defined metrics with their provider types. "
                    "Use to understand what cluster-wide analysis criteria are available before "
                    "investigating a failed AnalysisRun. "
                    "Read-only — does not modify any resources."
                ),
                parameters=[
                    ToolParam(
                        name="name",
                        type="string",
                        description=(
                            "ClusterAnalysisTemplate name; omit to list all cluster-scoped templates"
                        ),
                        required=False,
                    ),
                ],
                categories=frozenset({ARGO_ROLLOUTS}),
                execute=lambda name="",
                        _cfg=gke_config: kubectl_get_cluster_analysis_template(
                    name=name,
                ),
            ),
            ToolDef(
                name="kubectl_get_experiment",
                description=(
                    "List or inspect Argo Rollouts Experiment resources "
                    "(argoproj.io/v1alpha1). "
                    "Experiments run multiple ReplicaSets simultaneously for A/B or canary testing, "
                    "each with a defined template and replica count. "
                    "Returns experiment name, namespace, phase (Running/Successful/Failed), "
                    "and the list of templates with their replica counts. "
                    "Use to debug active or failed canary experiments associated with a Rollout. "
                    "Read-only — does not modify any resources."
                ),
                parameters=[
                    ToolParam(
                        name="namespace",
                        type="string",
                        description="Kubernetes namespace to query (default: all namespaces)",
                        required=False,
                    ),
                    ToolParam(
                        name="name",
                        type="string",
                        description=(
                            "Experiment name; omit to list all experiments in the namespace"
                        ),
                        required=False,
                    ),
                ],
                categories=frozenset({ARGO_ROLLOUTS}),
                execute=lambda namespace="", name="",
                        _cfg=gke_config: kubectl_get_experiment(
                    namespace=namespace, name=name,
                ),
            ),
        ])

    return tools

"""GKE tool definitions and factory function.

Layer 2 — imports all Layer 1 modules and assembles ToolDef objects
with closures that bind the GKEConfig at creation time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vaig.tools.base import ToolDef, ToolParam, ToolResult

from . import _clients, diagnostics, discovery, kubectl, mesh, mutations, security
from .argocd import (
    argocd_app_diff,
    argocd_app_history,
    argocd_app_managed_resources,
    argocd_app_status,
    argocd_list_applications,
)
from .datadog import get_datadog_config
from .datadog_api import (
    get_datadog_apm_services,
    get_datadog_monitors,
    get_datadog_service_catalog,
    query_datadog_metrics,
)
from .helm import (
    helm_list_releases,
    helm_release_history,
    helm_release_status,
    helm_release_values,
)
from .kubectl import kubectl_get_labels
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
                "pv, pvc, serviceaccounts, endpoints, networkpolicies, replicasets. "
                "Use resource='all' to list pods, services, deployments, replicasets, "
                "statefulsets, daemonsets, jobs, cronjobs, and hpa at once. "
                "Read-only — does not modify any resources."
            ),
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
            execute=lambda resource, name, namespace="default", _cfg=gke_config: kubectl.kubectl_describe(
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
                "optionally jobs/cronjobs. Shows ready/desired replicas, restarts, and "
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
                    name="force_refresh",
                    type="boolean",
                    description="Bypass cache and re-scan the cluster (default: false).",
                    required=False,
                ),
            ],
            execute=lambda namespace="", include_jobs=False, force_refresh=False,
                    _cfg=gke_config: discovery.discover_workloads(
                gke_config=_cfg, namespace=namespace, include_jobs=include_jobs,
                force_refresh=force_refresh,
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
            execute=lambda namespace="", force_refresh=False,
                    _cfg=gke_config: discovery.discover_network_topology(
                gke_config=_cfg, namespace=namespace, force_refresh=force_refresh,
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
            execute=lambda name, namespace="default",
                    _cfg=gke_config: get_scaling_status(
                name, gke_config=_cfg, namespace=namespace,
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
                execute=lambda release_name, namespace="default", all_values=False,
                        force_refresh=False,
                        _cfg=gke_config: helm_release_values(
                    gke_config=_cfg, release_name=release_name, namespace=namespace,
                    all_values=all_values, force_refresh=force_refresh,
                ),
            ),
        ])

    # ── ArgoCD tools (conditional on argocd_enabled) ─────────
    if gke_config.argocd_enabled:
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
                    "Always provide service_name — calling without it returns all registered services and should "
                    "be avoided (high cost, low signal). Resolve service_name from 'tags.datadoghq.com/service' "
                    "pod labels first, then DD_SERVICE env var; skip the call entirely if neither is available. "
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
                            "Look for 'tags.datadoghq.com/service' label on pods/deployments first, "
                            "then custom Datadog labels from config (e.g. DD_SERVICE env var). "
                            "Do NOT call this tool without a service_name — if unknown, skip the call entirely."
                        ),
                        required=False,
                    ),
                ],
                execute=lambda env="production", cluster_name="",
                        service_name=None,
                        _dd=_dd_config: get_datadog_service_catalog(
                    env=env, cluster_name=cluster_name, service_name=service_name, config=_dd,
                ),
            ),
            ToolDef(
                name="get_datadog_apm_services",
                description=(
                    "Fetch live APM trace metrics for a specific service from Datadog. "
                    "Returns throughput, error rate, and avg latency from actual trace data over the last 15 minutes. "
                    "Use this for real-time performance data — NOT for ownership metadata (use get_datadog_service_catalog for that). "
                    "Always provide service_name — it must match the 'service' tag in Datadog APM. "
                    "Results are cached for 60 seconds. Read-only — does not modify any resources."
                ),
                parameters=[
                    ToolParam(
                        name="service_name",
                        type="string",
                        description=(
                            "Service name to query APM trace metrics for. Required. "
                            "Must match the 'service' tag in Datadog APM — typically from the "
                            "'tags.datadoghq.com/service' label or custom APM instrumentation (DD_SERVICE env var)."
                        ),
                        required=True,
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
                ],
                execute=lambda service_name, env="production",
                        _dd=_dd_config: get_datadog_apm_services(
                    service_name=service_name, env=env, config=_dd,
                ),
            ),
        ])
    else:
        logger.debug(
            "Datadog API tools skipped (enabled=False). "
            "Set datadog.enabled=true or provide API keys to enable."
        )

    return tools

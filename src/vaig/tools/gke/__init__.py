"""GKE tools package — split from gke_tools.py monolith.

Re-exports every public name so that:
    from vaig.tools.gke import X
works for any X that was previously available via:
    from vaig.tools.gke_tools import X
"""

from __future__ import annotations

# ── Layer 2: factory function ────────────────────────────────
from ._registry import create_gke_tools

# ── Layer 1: read operations (kubectl) ───────────────────────
from .kubectl import (
    _describe_resource,
    _format_describe,
    _parse_since,
    async_kubectl_describe,
    async_kubectl_get,
    async_kubectl_logs,
    async_kubectl_top,
    kubectl_describe,
    kubectl_get,
    kubectl_logs,
    kubectl_top,
)

# ── Layer 1: diagnostics ────────────────────────────────────
from .diagnostics import (
    _find_current_revision,
    _format_container_section,
    _format_node_detail,
    _format_nodes_summary,
    _format_revision_detail,
    async_get_container_status,
    async_get_events,
    async_get_node_conditions,
    async_get_rollout_history,
    async_get_rollout_status,
    get_container_status,
    get_events,
    get_node_conditions,
    get_rollout_history,
    get_rollout_status,
)

# ── Layer 1: mutations ──────────────────────────────────────
from .mutations import (
    async_kubectl_annotate,
    async_kubectl_label,
    async_kubectl_restart,
    async_kubectl_scale,
    kubectl_annotate,
    kubectl_label,
    kubectl_restart,
    kubectl_scale,
)

# ── Layer 1: discovery ──────────────────────────────────────
from .discovery import (
    async_discover_network_topology,
    async_discover_service_mesh,
    async_discover_workloads,
    discover_network_topology,
    discover_service_mesh,
    discover_workloads,
)

# ── Layer 1: security ───────────────────────────────────────
from .security import (
    ALLOWED_EXEC_COMMANDS,
    DENIED_PATTERNS,
    _check_allowed,
    _check_denied,
    async_check_rbac,
    async_exec_command,
    check_rbac,
    exec_command,
)

# ── Layer 1: mesh introspection ─────────────────────────────
from .mesh import (
    async_get_mesh_config,
    async_get_mesh_overview,
    async_get_mesh_security,
    async_get_sidecar_status,
    get_mesh_config,
    get_mesh_overview,
    get_mesh_security,
    get_sidecar_status,
)

# ── Layer 0: client infrastructure ──────────────────────────
from ._clients import (
    _AUTOPILOT_CACHE,
    _CLIENT_CACHE,
    _K8S_AVAILABLE,
    _K8S_IMPORT_ERROR,
    _cache_key,
    _create_k8s_clients,
    _extract_proxy_url_from_kubeconfig,
    _k8s_unavailable,
    _query_autopilot_status,
    clear_autopilot_cache,
    clear_k8s_client_cache,
    detect_autopilot,
)

# k8s module aliases — conditionally available (only when kubernetes is installed)
try:
    from ._clients import k8s_client, k8s_config, k8s_exceptions  # noqa: WPS433
except ImportError:
    pass

# ── Layer 0: discovery cache ────────────────────────────────
from ._cache import (
    _DISCOVERY_CACHE,
    _DISCOVERY_TTL,
    _cache_key_discovery,
    _get_cached,
    _set_cache,
    clear_discovery_cache,
)

# ── Layer 0: resources ──────────────────────────────────────
from ._resources import (
    _RESOURCE_ALIASES,
    _RESOURCE_API_MAP,
    _list_resource,
    _normalise_resource,
)

# ── Layer 0: formatters ─────────────────────────────────────
from ._formatters import (
    _age,
    _format_deployments_table,
    _format_generic_table,
    _format_items,
    _format_memory,
    _format_nodes_table,
    _format_pods_table,
    _format_services_table,
    _pod_ready_count,
    _pod_restarts,
    _pod_status,
)

__all__ = [
    # Factory
    "create_gke_tools",
    # kubectl read ops (sync)
    "kubectl_get",
    "kubectl_describe",
    "kubectl_logs",
    "kubectl_top",
    "_describe_resource",
    "_format_describe",
    "_parse_since",
    # kubectl read ops (async)
    "async_kubectl_get",
    "async_kubectl_describe",
    "async_kubectl_logs",
    "async_kubectl_top",
    # Diagnostics (sync)
    "get_events",
    "get_rollout_status",
    "get_node_conditions",
    "get_container_status",
    "get_rollout_history",
    "_format_container_section",
    "_format_nodes_summary",
    "_format_node_detail",
    "_find_current_revision",
    "_format_revision_detail",
    # Diagnostics (async)
    "async_get_events",
    "async_get_rollout_status",
    "async_get_node_conditions",
    "async_get_container_status",
    "async_get_rollout_history",
    # Mutations (sync)
    "kubectl_scale",
    "kubectl_restart",
    "kubectl_label",
    "kubectl_annotate",
    # Mutations (async)
    "async_kubectl_scale",
    "async_kubectl_restart",
    "async_kubectl_label",
    "async_kubectl_annotate",
    # Discovery (sync)
    "discover_workloads",
    "discover_service_mesh",
    "discover_network_topology",
    # Discovery (async)
    "async_discover_workloads",
    "async_discover_service_mesh",
    "async_discover_network_topology",
    # Security (sync)
    "exec_command",
    "check_rbac",
    "_check_denied",
    "_check_allowed",
    "DENIED_PATTERNS",
    "ALLOWED_EXEC_COMMANDS",
    # Security (async)
    "async_exec_command",
    "async_check_rbac",
    # Mesh introspection (sync)
    "get_mesh_overview",
    "get_mesh_config",
    "get_mesh_security",
    "get_sidecar_status",
    # Mesh introspection (async)
    "async_get_mesh_overview",
    "async_get_mesh_config",
    "async_get_mesh_security",
    "async_get_sidecar_status",
    # Client infrastructure
    "_K8S_AVAILABLE",
    "_K8S_IMPORT_ERROR",
    "_CLIENT_CACHE",
    "_AUTOPILOT_CACHE",
    "_k8s_unavailable",
    "_create_k8s_clients",
    "detect_autopilot",
    "clear_autopilot_cache",
    "clear_k8s_client_cache",
    "_cache_key",
    "_extract_proxy_url_from_kubeconfig",
    "_query_autopilot_status",
    "k8s_client",
    "k8s_config",
    "k8s_exceptions",
    # Discovery cache
    "_DISCOVERY_CACHE",
    "_DISCOVERY_TTL",
    "_cache_key_discovery",
    "_get_cached",
    "_set_cache",
    "clear_discovery_cache",
    # Resources
    "_RESOURCE_API_MAP",
    "_RESOURCE_ALIASES",
    "_normalise_resource",
    "_list_resource",
    # Formatters
    "_age",
    "_pod_status",
    "_pod_restarts",
    "_pod_ready_count",
    "_format_pods_table",
    "_format_deployments_table",
    "_format_services_table",
    "_format_nodes_table",
    "_format_generic_table",
    "_format_items",
    "_format_memory",
]

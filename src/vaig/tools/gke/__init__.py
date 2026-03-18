"""GKE tools package — split from gke_tools.py monolith.

Re-exports every public name so that:
    from vaig.tools.gke import X
works for any X that was previously available via:
    from vaig.tools.gke_tools import X
"""

from __future__ import annotations

# ── Layer 0: client infrastructure ──────────────────────────
from ._clients import (
    DefaultK8sClientProvider,
    clear_autopilot_cache,
    clear_k8s_client_cache,
    detect_autopilot,
)
from ._clients import (
    _get_exec_client as get_exec_client,
)
from ._clients import (
    ensure_client_initialized as ensure_client_initialized,
)

# ── Layer 2: factory function ────────────────────────────────
from ._registry import create_gke_tools

# ── Layer 1: ArgoCD introspection ───────────────────────────
from .argocd import (
    argocd_app_diff,
    argocd_app_history,
    argocd_app_managed_resources,
    argocd_app_status,
    argocd_list_applications,
    async_argocd_app_diff,
    async_argocd_app_history,
    async_argocd_app_managed_resources,
    async_argocd_app_status,
    async_argocd_list_applications,
)

# ── Layer 1: diagnostics ────────────────────────────────────
from .diagnostics import (
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

# ── Layer 1: discovery ──────────────────────────────────────
from .discovery import (
    async_discover_network_topology,
    async_discover_service_mesh,
    async_discover_workloads,
    discover_network_topology,
    discover_service_mesh,
    discover_workloads,
)

# ── Layer 1: Helm introspection ─────────────────────────────
from .helm import (
    async_helm_list_releases,
    async_helm_release_history,
    async_helm_release_status,
    async_helm_release_values,
    helm_list_releases,
    helm_release_history,
    helm_release_status,
    helm_release_values,
)

# ── Layer 1: read operations (kubectl) ───────────────────────
# ── Layer 1: labels ─────────────────────────────────────────
from .kubectl import (
    async_kubectl_describe,
    async_kubectl_get,
    async_kubectl_get_labels,
    async_kubectl_logs,
    async_kubectl_top,
    kubectl_describe,
    kubectl_get,
    kubectl_get_labels,
    kubectl_logs,
    kubectl_top,
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

# ── Layer 1: security ───────────────────────────────────────
from .security import (
    ALLOWED_EXEC_COMMANDS,
    DENIED_PATTERNS,
    async_check_rbac,
    async_exec_command,
    check_rbac,
    exec_command,
)

# k8s module aliases — conditionally available (only when kubernetes is installed)
try:
    from ._clients import k8s_client, k8s_config, k8s_exceptions  # type: ignore[attr-defined]  # noqa: WPS433
except ImportError:
    pass

# ── Layer 0: discovery cache ────────────────────────────────
from ._cache import (
    clear_discovery_cache,
)

# ── Layer 0: formatters ─────────────────────────────────────
# Private formatter symbols are available via direct submodule import
# (e.g., from vaig.tools.gke._formatters import _redact_secret_item)

# ── Layer 0: resources ──────────────────────────────────────
# Private resource symbols are available via direct submodule import
# (e.g., from vaig.tools.gke._resources import _RESOURCE_API_MAP)

__all__ = [
    # Factory
    "create_gke_tools",
    # kubectl read ops (sync)
    "kubectl_get",
    "kubectl_describe",
    "kubectl_logs",
    "kubectl_top",
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
    # Labels (sync)
    "kubectl_get_labels",
    # Labels (async)
    "async_kubectl_get_labels",
    # Helm introspection (sync)
    "helm_list_releases",
    "helm_release_status",
    "helm_release_history",
    "helm_release_values",
    # Helm introspection (async)
    "async_helm_list_releases",
    "async_helm_release_status",
    "async_helm_release_history",
    "async_helm_release_values",
    # ArgoCD introspection (sync)
    "argocd_list_applications",
    "argocd_app_status",
    "argocd_app_history",
    "argocd_app_diff",
    "argocd_app_managed_resources",
    # ArgoCD introspection (async)
    "async_argocd_list_applications",
    "async_argocd_app_status",
    "async_argocd_app_history",
    "async_argocd_app_diff",
    "async_argocd_app_managed_resources",
    # Client infrastructure (public)
    "DefaultK8sClientProvider",
    "detect_autopilot",
    "clear_autopilot_cache",
    "clear_k8s_client_cache",
    "get_exec_client",
    "k8s_client",
    "k8s_config",
    "k8s_exceptions",
    # Discovery cache (public)
    "clear_discovery_cache",
]

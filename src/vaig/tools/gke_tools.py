"""Backward-compatibility shim — all functionality moved to vaig.tools.gke package."""

from vaig.tools.gke import *  # noqa: F401,F403
from vaig.tools.gke._cache import _DISCOVERY_CACHE, _cache_key_discovery, _get_cached, _set_cache
from vaig.tools.gke._clients import _AUTOPILOT_CACHE, _create_k8s_clients, _extract_proxy_url_from_kubeconfig
from vaig.tools.gke._formatters import (
    _age,
    _format_crds_table,
    _format_deployments_table,
    _format_generic_table,
    _format_memory,
    _format_nodes_table,
    _format_pods_table,
    _format_services_table,
    _format_webhook_config,
    _format_webhooks_table,
    _pod_ready_count,
    _pod_restarts,
    _pod_status,
)

# Re-export internal utilities used by tests
from vaig.tools.gke._resources import (
    _CLUSTER_SCOPED_RESOURCES,
    _KNOWN_K8S_RESOURCES,
    _RESOURCE_API_MAP,
    _normalise_resource,
)
from vaig.tools.gke.kubectl import _parse_since
from vaig.tools.gke.security import ALLOWED_EXEC_COMMANDS, _check_allowed, _check_denied

__all__ = [
    "_normalise_resource", "_RESOURCE_API_MAP", "_KNOWN_K8S_RESOURCES", "_CLUSTER_SCOPED_RESOURCES",
    "_parse_since", "_age", "_pod_status", "_pod_restarts", "_pod_ready_count", 
    "_format_pods_table", "_format_deployments_table", "_format_services_table", 
    "_format_nodes_table", "_format_generic_table", "_format_memory", 
    "_format_webhook_config", "_format_webhooks_table", "_format_crds_table",
    "_extract_proxy_url_from_kubeconfig", "_create_k8s_clients", "_AUTOPILOT_CACHE",
    "_check_allowed", "_check_denied", "ALLOWED_EXEC_COMMANDS",
    "_cache_key_discovery", "_get_cached", "_set_cache", "_DISCOVERY_CACHE"
]

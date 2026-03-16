"""ArgoCD introspection tools — application status, history, diff, and managed resources."""

from __future__ import annotations

import logging
from typing import Any

from vaig.tools.base import ToolResult

from . import _cache, _clients

logger = logging.getLogger(__name__)

# ── Lazy import guard (mirrors _clients.py) ──────────────────
_K8S_AVAILABLE = True
try:
    from kubernetes.client import exceptions as k8s_exceptions  # noqa: WPS433
except ImportError:
    _K8S_AVAILABLE = False

# ── Constants ────────────────────────────────────────────────
_ARGOCD_CACHE_TTL: int = 30  # seconds
_ARGOCD_GROUP = "argoproj.io"
_ARGOCD_VERSION = "v1alpha1"
_ARGOCD_PLURAL = "applications"


# ── Helpers ──────────────────────────────────────────────────


def _get_custom_objects_api() -> Any | None:
    """Return a CustomObjectsApi instance for same-cluster mode.

    Returns ``None`` when the kubernetes SDK is not available.
    """
    if not _K8S_AVAILABLE:
        return None
    from kubernetes import client as k8s_client  # noqa: WPS433
    from kubernetes import config as k8s_config  # noqa: WPS433

    try:
        k8s_config.load_incluster_config()
    except Exception:
        try:
            k8s_config.load_kube_config()
        except Exception:
            return None
    return k8s_client.CustomObjectsApi()


def _get_argocd_client(
    *,
    server: str = "",
    token: str = "",
    context: str = "",
) -> tuple[str, Any]:
    """Return an ArgoCD client based on connection mode.

    Returns:
        Tuple of (mode, client) where mode is one of:
        - ``"api"`` — REST API mode (stub, raises NotImplementedError)
        - ``"context"`` — separate kubeconfig context (stub, raises NotImplementedError)
        - ``"cluster"`` — same-cluster CustomObjectsApi
    """
    # Mode 1: API server
    if server and token:
        raise NotImplementedError(
            "ArgoCD REST API mode not yet implemented (Phase 3). "
            "Use same-cluster mode instead."
        )

    # Mode 2: Separate kubeconfig context
    if context:
        raise NotImplementedError(
            "ArgoCD separate-context mode not yet implemented (Phase 3). "
            "Use same-cluster mode instead."
        )

    # Mode 3: Same-cluster (default fallback)
    client = _get_custom_objects_api()
    if client is None:
        raise RuntimeError("Cannot create CustomObjectsApi — kubernetes SDK unavailable or unconfigured")
    return ("cluster", client)


def _list_applications_raw(
    custom_api: Any,
    namespace: str,
) -> list[dict[str, Any]]:
    """List ArgoCD Application CRDs in the given namespace via CustomObjectsApi."""
    try:
        result = custom_api.list_namespaced_custom_object(
            group=_ARGOCD_GROUP,
            version=_ARGOCD_VERSION,
            namespace=namespace,
            plural=_ARGOCD_PLURAL,
        )
        return result.get("items", [])  # type: ignore[no-any-return]  # K8s CustomObject API returns Any
    except Exception as exc:
        if _K8S_AVAILABLE and isinstance(exc, k8s_exceptions.ApiException):
            if exc.status == 404:
                logger.debug("ArgoCD CRD not found (404)")
                return []
            if exc.status == 403:
                logger.warning("RBAC: cannot list ArgoCD applications (403 Forbidden)")
                return []
        logger.warning("Error listing ArgoCD applications: %s", exc)
        return []


def _get_application_raw(
    custom_api: Any,
    app_name: str,
    namespace: str,
) -> dict[str, Any] | None:
    """Get a single ArgoCD Application CRD by name."""
    try:
        return custom_api.get_namespaced_custom_object(  # type: ignore[no-any-return]  # K8s API
            group=_ARGOCD_GROUP,
            version=_ARGOCD_VERSION,
            namespace=namespace,
            name=app_name,
            plural=_ARGOCD_PLURAL,
        )
    except Exception as exc:
        if _K8S_AVAILABLE and isinstance(exc, k8s_exceptions.ApiException):
            if exc.status == 404:
                return None
            if exc.status == 403:
                logger.warning("RBAC: cannot get ArgoCD application '%s' (403 Forbidden)", app_name)
                return None
        logger.warning("Error getting ArgoCD application '%s': %s", app_name, exc)
        return None


# ── Formatters ───────────────────────────────────────────────


def _format_app_row(app: dict[str, Any]) -> dict[str, str]:
    """Extract key fields from an ArgoCD Application for table display."""
    metadata = app.get("metadata", {})
    spec = app.get("spec", {})
    status = app.get("status", {})

    source = spec.get("source", {})
    destination = spec.get("destination", {})
    sync_status = status.get("sync", {}).get("status", "Unknown")
    health_status = status.get("health", {}).get("status", "Unknown")

    return {
        "name": metadata.get("name", "<unknown>"),
        "project": spec.get("project", "default"),
        "sync_status": sync_status,
        "health_status": health_status,
        "repo": source.get("repoURL", "-"),
        "target_revision": source.get("targetRevision", "-"),
        "dest_cluster": destination.get("server", destination.get("name", "-")),
        "dest_namespace": destination.get("namespace", "-"),
    }


def _format_app_table(apps: list[dict[str, Any]]) -> str:
    """Format a list of ArgoCD Applications as a table."""
    if not apps:
        return "No ArgoCD applications found."

    rows = [_format_app_row(app) for app in apps]

    lines: list[str] = []
    lines.append(
        f"  {'NAME':<30} {'PROJECT':<15} {'SYNC':<12} {'HEALTH':<12} {'REPO':<40} {'REVISION':<15} {'DEST NS':<15}"
    )
    lines.append("  " + "-" * 139)

    for row in rows:
        repo = row["repo"]
        if len(repo) > 39:
            repo = "..." + repo[-36:]
        lines.append(
            f"  {row['name']:<30} {row['project']:<15} {row['sync_status']:<12} "
            f"{row['health_status']:<12} {repo:<40} {row['target_revision']:<15} {row['dest_namespace']:<15}"
        )

    return "\n".join(lines)


def _format_app_detail(app: dict[str, Any]) -> str:
    """Format detailed status of a single ArgoCD Application."""
    metadata = app.get("metadata", {})
    spec = app.get("spec", {})
    status = app.get("status", {})

    source = spec.get("source", {})
    destination = spec.get("destination", {})
    sync = status.get("sync", {})
    health = status.get("health", {})
    operation_state = status.get("operationState", {})
    sync_policy = spec.get("syncPolicy", {})

    lines: list[str] = []
    lines.append(f"Name: {metadata.get('name', '<unknown>')}")
    lines.append(f"Namespace: {metadata.get('namespace', '-')}")
    lines.append(f"Project: {spec.get('project', 'default')}")
    lines.append("")

    # Source
    lines.append("--- Source ---")
    lines.append(f"  Repo URL: {source.get('repoURL', '-')}")
    lines.append(f"  Path: {source.get('path', '-')}")
    lines.append(f"  Target Revision: {source.get('targetRevision', '-')}")
    lines.append(f"  Chart: {source.get('chart', '-')}")
    lines.append("")

    # Destination
    lines.append("--- Destination ---")
    lines.append(f"  Server: {destination.get('server', destination.get('name', '-'))}")
    lines.append(f"  Namespace: {destination.get('namespace', '-')}")
    lines.append("")

    # Sync status
    lines.append("--- Sync Status ---")
    lines.append(f"  Status: {sync.get('status', 'Unknown')}")
    lines.append(f"  Revision: {sync.get('revision', '-')}")
    lines.append("")

    # Health status
    lines.append("--- Health Status ---")
    lines.append(f"  Status: {health.get('status', 'Unknown')}")
    if health.get("message"):
        lines.append(f"  Message: {health['message']}")
    lines.append("")

    # Sync policy
    lines.append("--- Sync Policy ---")
    automated = sync_policy.get("automated")
    if automated:
        prune = automated.get("prune", False)
        self_heal = automated.get("selfHeal", False)
        lines.append(f"  Mode: Automated (prune={prune}, selfHeal={self_heal})")
    else:
        lines.append("  Mode: Manual")
    lines.append("")

    # Operation state
    if operation_state:
        lines.append("--- Last Operation ---")
        lines.append(f"  Phase: {operation_state.get('phase', '-')}")
        lines.append(f"  Message: {operation_state.get('message', '-')}")
        started = operation_state.get("startedAt", "-")
        finished = operation_state.get("finishedAt", "-")
        lines.append(f"  Started: {started}")
        lines.append(f"  Finished: {finished}")
        lines.append("")

    # Conditions
    conditions = status.get("conditions", [])
    if conditions:
        lines.append("--- Conditions ---")
        for cond in conditions:
            ctype = cond.get("type", "?")
            msg = cond.get("message", "")
            last_transition = cond.get("lastTransitionTime", "-")
            lines.append(f"  [{ctype}] {msg} (at {last_transition})")
        lines.append("")

    return "\n".join(lines)


def _format_history_table(history: list[dict[str, Any]]) -> str:
    """Format deployment history as a table (most recent first)."""
    if not history:
        return "No deployment history found."

    # Sort by ID descending (most recent first)
    sorted_history = sorted(history, key=lambda h: h.get("id", 0), reverse=True)

    lines: list[str] = []
    lines.append(
        f"  {'ID':<6} {'DEPLOYED AT':<25} {'REVISION':<45} {'SOURCE':<40}"
    )
    lines.append("  " + "-" * 116)

    for entry in sorted_history:
        entry_id = str(entry.get("id", "-"))
        deployed_at = entry.get("deployedAt", "-")
        revision = entry.get("revision", "-")
        if len(revision) > 44:
            revision = revision[:41] + "..."
        source = entry.get("source", {})
        repo = source.get("repoURL", "-")
        if len(repo) > 39:
            repo = "..." + repo[-36:]

        lines.append(
            f"  {entry_id:<6} {deployed_at:<25} {revision:<45} {repo:<40}"
        )

    return "\n".join(lines)


def _format_diff_summary(resources: list[dict[str, Any]]) -> str:
    """Format out-of-sync resources as a diff summary."""
    out_of_sync = []
    for res in resources:
        sync_status = res.get("status", "Unknown")
        health = res.get("health", {})
        health_status = health.get("status", "Unknown") if isinstance(health, dict) else "Unknown"

        if sync_status != "Synced" or health_status not in ("Healthy", "Unknown"):
            out_of_sync.append(res)

    if not out_of_sync:
        return "All resources are in sync and healthy."

    lines: list[str] = []
    lines.append(
        f"  {'KIND':<25} {'NAME':<30} {'NAMESPACE':<18} {'SYNC':<12} {'HEALTH':<12} {'HOOK':<10}"
    )
    lines.append("  " + "-" * 107)

    for res in out_of_sync:
        kind = res.get("kind", "?")
        name = res.get("name", "?")
        ns = res.get("namespace", "-")
        sync_status = res.get("status", "Unknown")
        health = res.get("health", {})
        health_status = health.get("status", "-") if isinstance(health, dict) else "-"
        hook = res.get("hook", False)
        hook_str = "Yes" if hook else "-"

        lines.append(
            f"  {kind:<25} {name:<30} {ns:<18} {sync_status:<12} {health_status:<12} {hook_str:<10}"
        )

    lines.append(f"\n  Out-of-sync resources: {len(out_of_sync)}")
    return "\n".join(lines)


def _format_managed_resources_table(resources: list[dict[str, Any]]) -> str:
    """Format managed resources as a table grouped by kind."""
    if not resources:
        return "No managed resources found."

    # Group by kind
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for res in resources:
        kind = res.get("kind", "Unknown")
        by_kind.setdefault(kind, []).append(res)

    lines: list[str] = []

    for kind in sorted(by_kind.keys()):
        group_resources = by_kind[kind]
        lines.append(f"--- {kind} ({len(group_resources)}) ---")

        lines.append(
            f"  {'GROUP':<25} {'NAME':<30} {'NAMESPACE':<18} {'SYNC':<12} {'HEALTH':<12} {'PRUNE':<8}"
        )
        lines.append("  " + "-" * 105)

        for res in group_resources:
            group = res.get("group", "-") or "-"
            name = res.get("name", "?")
            ns = res.get("namespace", "-")
            sync_status = res.get("status", "Unknown")
            health = res.get("health", {})
            health_status = health.get("status", "-") if isinstance(health, dict) else "-"
            requires_pruning = res.get("requiresPruning", False)
            prune_str = "Yes" if requires_pruning else "-"

            lines.append(
                f"  {group:<25} {name:<30} {ns:<18} {sync_status:<12} {health_status:<12} {prune_str:<8}"
            )

        lines.append("")

    lines.append(f"Total managed resources: {len(resources)}")
    return "\n".join(lines)


# ── Public Tool Functions ────────────────────────────────────


def argocd_list_applications(
    *,
    namespace: str = "argocd",
    _custom_api: Any = None,
) -> ToolResult:
    """List all ArgoCD Applications in the given namespace.

    Returns a formatted table with name, project, sync status, health status,
    source repo, target revision, and destination for each application.

    Args:
        namespace: Kubernetes namespace where ArgoCD is deployed.
        _custom_api: Optional pre-configured CustomObjectsApi (for testing).
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    # Cache check
    cache_key = _cache._cache_key_discovery("argocd_list", namespace)
    cached = _cache._get_cached(cache_key)
    if cached is not None:
        return ToolResult(output=cached, error=False)

    # Get client
    custom_api = _custom_api
    if custom_api is None:
        try:
            _, custom_api = _get_argocd_client()
        except Exception as exc:
            return ToolResult(output=f"Failed to connect to ArgoCD: {exc}", error=True)

    apps = _list_applications_raw(custom_api, namespace)

    sections: list[str] = [
        "=== ArgoCD Applications ===",
        f"Namespace: {namespace}",
        "",
    ]

    sections.append(_format_app_table(apps))
    sections.append("")
    sections.append(f"Total applications: {len(apps)}")

    output = "\n".join(sections)
    _cache._set_cache(cache_key, output)
    return ToolResult(output=output, error=False)


def argocd_app_status(
    *,
    app_name: str,
    namespace: str = "argocd",
    _custom_api: Any = None,
) -> ToolResult:
    """Get detailed status of a specific ArgoCD Application.

    Returns sync status, health status, source info, destination,
    conditions, operation state, and sync policy.

    Args:
        app_name: Name of the ArgoCD Application.
        namespace: Kubernetes namespace where ArgoCD is deployed.
        _custom_api: Optional pre-configured CustomObjectsApi (for testing).
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    # Cache check
    cache_key = _cache._cache_key_discovery("argocd_status", namespace, app_name)
    cached = _cache._get_cached(cache_key)
    if cached is not None:
        return ToolResult(output=cached, error=False)

    # Get client
    custom_api = _custom_api
    if custom_api is None:
        try:
            _, custom_api = _get_argocd_client()
        except Exception as exc:
            return ToolResult(output=f"Failed to connect to ArgoCD: {exc}", error=True)

    app = _get_application_raw(custom_api, app_name, namespace)
    if app is None:
        return ToolResult(
            output=f"ArgoCD application '{app_name}' not found in namespace '{namespace}'.",
            error=True,
        )

    sections: list[str] = [
        "=== ArgoCD Application Status ===",
        "",
    ]
    sections.append(_format_app_detail(app))

    output = "\n".join(sections)
    _cache._set_cache(cache_key, output)
    return ToolResult(output=output, error=False)


def argocd_app_history(
    *,
    app_name: str,
    namespace: str = "argocd",
    _custom_api: Any = None,
) -> ToolResult:
    """Get deployment history of an ArgoCD Application.

    Returns a table of past deployments sorted by most recent first,
    including revision, deployment time, and source info.

    Args:
        app_name: Name of the ArgoCD Application.
        namespace: Kubernetes namespace where ArgoCD is deployed.
        _custom_api: Optional pre-configured CustomObjectsApi (for testing).
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    # Cache check
    cache_key = _cache._cache_key_discovery("argocd_history", namespace, app_name)
    cached = _cache._get_cached(cache_key)
    if cached is not None:
        return ToolResult(output=cached, error=False)

    # Get client
    custom_api = _custom_api
    if custom_api is None:
        try:
            _, custom_api = _get_argocd_client()
        except Exception as exc:
            return ToolResult(output=f"Failed to connect to ArgoCD: {exc}", error=True)

    app = _get_application_raw(custom_api, app_name, namespace)
    if app is None:
        return ToolResult(
            output=f"ArgoCD application '{app_name}' not found in namespace '{namespace}'.",
            error=True,
        )

    status = app.get("status", {})
    history = status.get("history", [])

    sections: list[str] = [
        f"=== Deployment History: {app_name} ===",
        "",
    ]
    sections.append(_format_history_table(history))
    sections.append("")
    sections.append(f"Total deployments: {len(history)}")

    output = "\n".join(sections)
    _cache._set_cache(cache_key, output)
    return ToolResult(output=output, error=False)


def argocd_app_diff(
    *,
    app_name: str,
    namespace: str = "argocd",
    _custom_api: Any = None,
) -> ToolResult:
    """Show resources that are out-of-sync for an ArgoCD Application.

    Examines ``.status.resources[]`` and returns resources where
    sync status is not ``Synced`` or health status is not ``Healthy``.

    Args:
        app_name: Name of the ArgoCD Application.
        namespace: Kubernetes namespace where ArgoCD is deployed.
        _custom_api: Optional pre-configured CustomObjectsApi (for testing).
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    # Cache check
    cache_key = _cache._cache_key_discovery("argocd_diff", namespace, app_name)
    cached = _cache._get_cached(cache_key)
    if cached is not None:
        return ToolResult(output=cached, error=False)

    # Get client
    custom_api = _custom_api
    if custom_api is None:
        try:
            _, custom_api = _get_argocd_client()
        except Exception as exc:
            return ToolResult(output=f"Failed to connect to ArgoCD: {exc}", error=True)

    app = _get_application_raw(custom_api, app_name, namespace)
    if app is None:
        return ToolResult(
            output=f"ArgoCD application '{app_name}' not found in namespace '{namespace}'.",
            error=True,
        )

    status = app.get("status", {})
    resources = status.get("resources", [])

    sections: list[str] = [
        f"=== Diff Summary: {app_name} ===",
        "",
    ]
    sections.append(_format_diff_summary(resources))

    output = "\n".join(sections)
    _cache._set_cache(cache_key, output)
    return ToolResult(output=output, error=False)


def argocd_app_managed_resources(
    *,
    app_name: str,
    namespace: str = "argocd",
    _custom_api: Any = None,
) -> ToolResult:
    """List all resources managed by an ArgoCD Application.

    Returns a formatted table grouped by kind, showing group, name,
    namespace, sync status, health status, and pruning requirements.

    Args:
        app_name: Name of the ArgoCD Application.
        namespace: Kubernetes namespace where ArgoCD is deployed.
        _custom_api: Optional pre-configured CustomObjectsApi (for testing).
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    # Cache check
    cache_key = _cache._cache_key_discovery("argocd_managed", namespace, app_name)
    cached = _cache._get_cached(cache_key)
    if cached is not None:
        return ToolResult(output=cached, error=False)

    # Get client
    custom_api = _custom_api
    if custom_api is None:
        try:
            _, custom_api = _get_argocd_client()
        except Exception as exc:
            return ToolResult(output=f"Failed to connect to ArgoCD: {exc}", error=True)

    app = _get_application_raw(custom_api, app_name, namespace)
    if app is None:
        return ToolResult(
            output=f"ArgoCD application '{app_name}' not found in namespace '{namespace}'.",
            error=True,
        )

    status = app.get("status", {})
    resources = status.get("resources", [])

    sections: list[str] = [
        f"=== Managed Resources: {app_name} ===",
        "",
    ]
    sections.append(_format_managed_resources_table(resources))

    output = "\n".join(sections)
    _cache._set_cache(cache_key, output)
    return ToolResult(output=output, error=False)


# ── Async wrappers ───────────────────────────────────────────

from vaig.core.async_utils import to_async  # noqa: E402

async_argocd_list_applications = to_async(argocd_list_applications)
async_argocd_app_status = to_async(argocd_app_status)
async_argocd_app_history = to_async(argocd_app_history)
async_argocd_app_diff = to_async(argocd_app_diff)
async_argocd_app_managed_resources = to_async(argocd_app_managed_resources)

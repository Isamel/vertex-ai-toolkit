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

    class _StubExceptions:
        """Fallback so ``except k8s_exceptions.ApiException`` never matches."""

        ApiException = type(None)

    k8s_exceptions = _StubExceptions()  # type: ignore[assignment,unused-ignore]

# ── Constants ────────────────────────────────────────────────
_ARGOCD_CACHE_TTL: int = 30  # seconds
_ARGOCD_GROUP = "argoproj.io"
_ARGOCD_VERSION = "v1alpha1"
_ARGOCD_PLURAL = "applications"

_ARGOCD_COMMON_NAMESPACES: tuple[str, ...] = (
    "argocd",
    "argo-cd",
    "argocd-system",
    "gitops",
    "argo",
)
_argocd_namespace_cache: dict[str, str | None] = {}
_crd_exists_cache: dict[str, bool] = {}
_argocd_ns_cache: dict[str, bool] = {}


# ── Helpers ──────────────────────────────────────────────────


def _check_crd_exists(crd_name: str, api_client: Any = None) -> bool:
    """Check whether a CustomResourceDefinition exists in the cluster.

    Uses ``ApiextensionsV1Api.read_custom_resource_definition`` which requires
    the ``apiextensions.k8s.io/v1`` endpoint.  Results are cached per-process
    to avoid repeated API round-trips within the same invocation.

    Args:
        crd_name: Fully-qualified CRD name, e.g. ``"applications.argoproj.io"``.
        api_client: Optional pre-configured ``kubernetes.client.ApiClient``.
            When ``None`` the function loads the in-cluster or kube-config
            credentials automatically.

    Returns:
        ``True`` if the CRD exists and is accessible, ``False`` otherwise.
        Returns ``False`` on any error (404, 403, SDK unavailable, network
        failure) to keep the caller logic simple.
    """
    if not _K8S_AVAILABLE:
        return False

    if crd_name in _crd_exists_cache:
        return _crd_exists_cache[crd_name]

    try:
        from kubernetes import client as k8s_client  # noqa: WPS433
        from kubernetes import config as k8s_config  # noqa: WPS433

        if api_client is None:
            try:
                k8s_config.load_incluster_config()
            except k8s_config.ConfigException:
                try:
                    k8s_config.load_kube_config()
                except k8s_config.ConfigException:
                    _crd_exists_cache[crd_name] = False
                    return False
            ext_api = k8s_client.ApiextensionsV1Api()
        else:
            ext_api = k8s_client.ApiextensionsV1Api(api_client)

        ext_api.read_custom_resource_definition(crd_name)
        _crd_exists_cache[crd_name] = True
        return True

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            logger.debug("CRD '%s' not found (404)", crd_name)
        elif exc.status == 403:
            logger.warning(
                "RBAC: cannot check CRD '%s' (403 Forbidden) — assuming absent", crd_name
            )
        else:
            logger.warning("K8s API error checking CRD '%s': %s", crd_name, exc)
        _crd_exists_cache[crd_name] = False
        return False

    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected error checking CRD '%s': %s", crd_name, exc)
        _crd_exists_cache[crd_name] = False
        return False


_ARGOCD_ANNOTATION_MARKERS = (
    "argocd.argoproj.io/tracking-id",
    "argocd.argoproj.io/managed-by",
)


def detect_argocd(namespace: str = "", api_client: Any = None) -> bool:
    """Detect ArgoCD presence for a specific namespace.

    Three-phase detection:
    1. CRD probe: checks if ``applications.argoproj.io`` CRD exists cluster-wide.
       If absent, ArgoCD is definitely not installed → return False.
    2. Namespace annotation scan: checks if deployments in the target namespace
       have ArgoCD management annotations (``argocd.argoproj.io/tracking-id``,
       ``argocd.argoproj.io/managed-by``). CRD existing cluster-wide does NOT
       mean this namespace is managed by ArgoCD.
    3. Results are cached per-namespace to avoid redundant API calls.

    Args:
        namespace: Target namespace to check. Empty = "default".
        api_client: Optional pre-configured kubernetes ApiClient.

    Returns:
        True if ArgoCD manages resources in the target namespace.
    """
    ns = namespace or "default"

    # Check namespace-level cache
    if ns in _argocd_ns_cache:
        return _argocd_ns_cache[ns]

    result = _detect_argocd_for_namespace(ns, api_client)
    _argocd_ns_cache[ns] = result
    return result


def _detect_argocd_for_namespace(namespace: str, api_client: Any) -> bool:
    """Internal: uncached namespace-scoped ArgoCD detection."""
    # Phase 1: CRD must exist cluster-wide (necessary condition)
    crd_present = _check_crd_exists("applications.argoproj.io", api_client=api_client)

    # Phase 2: Check namespace annotations (sufficient condition)
    # Even if CRD exists, we must verify THIS namespace has ArgoCD-managed resources.
    # If CRD check failed due to RBAC (403), annotation scan is the only path.
    ns_has_annotations = _scan_namespace_for_argocd_annotations(namespace, api_client)

    if ns_has_annotations:
        if crd_present:
            logger.info(
                "ArgoCD detected for namespace '%s' (CRD present + annotations found).",
                namespace,
            )
        else:
            logger.info(
                "ArgoCD detected for namespace '%s' via annotations "
                "(CRD probe unavailable/failed).",
                namespace,
            )
        return True

    if crd_present:
        logger.debug(
            "ArgoCD CRD exists cluster-wide but no managed resources found "
            "in namespace '%s' — skipping ArgoCD tools for this namespace.",
            namespace,
        )

    return False


def _scan_namespace_for_argocd_annotations(namespace: str, api_client: Any) -> bool:
    """Scan deployments in a namespace for ArgoCD management annotations."""
    if not _K8S_AVAILABLE:
        return False

    try:
        from kubernetes import client as k8s_client  # noqa: WPS433
        from kubernetes import config as k8s_config  # noqa: WPS433

        if api_client is None:
            try:
                k8s_config.load_incluster_config()
            except k8s_config.ConfigException:
                try:
                    k8s_config.load_kube_config()
                except k8s_config.ConfigException:
                    return False
            apps_api = k8s_client.AppsV1Api()
        else:
            apps_api = k8s_client.AppsV1Api(api_client)

        deployments = apps_api.list_namespaced_deployment(
            namespace=namespace,
            limit=50,
        )
        for dep in deployments.items or []:
            annotations = dep.metadata.annotations or {}
            if any(k in annotations for k in _ARGOCD_ANNOTATION_MARKERS):
                return True

    except Exception:  # noqa: BLE001
        logger.debug(
            "ArgoCD annotation scan failed for namespace '%s'",
            namespace,
            exc_info=True,
        )

    return False


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
    except k8s_config.ConfigException:
        try:
            k8s_config.load_kube_config()
        except k8s_config.ConfigException:
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
        raise NotImplementedError("ArgoCD REST API mode not yet implemented (Phase 3). Use same-cluster mode instead.")

    # Mode 2: Separate kubeconfig context
    if context:
        raise NotImplementedError(
            "ArgoCD separate-context mode not yet implemented (Phase 3). Use same-cluster mode instead."
        )

    # Mode 3: Same-cluster (default fallback)
    client = _get_custom_objects_api()
    if client is None:
        raise RuntimeError("Cannot create CustomObjectsApi — kubernetes SDK unavailable or unconfigured")
    return ("cluster", client)


def _discover_argocd_namespace(custom_api: Any) -> str | None:
    """Auto-discover the ArgoCD namespace by probing common namespace names.

    Strategy:
    0. (NEW) Check whether the ``applications.argoproj.io`` CRD exists in the
       cluster at all.  If it does not, ArgoCD is definitely not installed —
       return ``None`` immediately without probing any namespaces.  This handles
       hub-spoke architectures where the CRD is registered locally even though
       the ArgoCD control-plane lives in a separate cluster.
    1. Probe each well-known namespace for the ``argoproj.io/applications`` CRD.
       The first namespace that responds with at least one Application object is used.
    2. If no known namespace matches, fall back to a cluster-wide scan and extract
       the namespace from the first Application found there.

    Results are cached per-process (one cache per Python interpreter lifetime).
    Different kubeconfig contexts are NOT distinguished — this is intentional for
    CLI/short-lived processes where the context does not change mid-run.

    Returns:
        The namespace string if an ArgoCD Application CRD is found, else ``None``.
    """
    # Cache key is constant because this is a per-process cache.
    # CLI tools are single-context per invocation, so one entry per process is correct.
    cache_key = "default"
    if cache_key in _argocd_namespace_cache:
        return _argocd_namespace_cache[cache_key]

    # Step 0: CRD existence pre-check (fast, avoids O(n) namespace probes when ArgoCD absent)
    # Pass the api_client from the CustomObjectsApi so both checks use the same kubeconfig context.
    client_for_crd = getattr(custom_api, "api_client", None)
    if not _check_crd_exists("applications.argoproj.io", api_client=client_for_crd):
        _argocd_namespace_cache[cache_key] = None
        return None

    # Try each common namespace first (faster than cluster-wide scan)
    for ns in _ARGOCD_COMMON_NAMESPACES:
        try:
            result = custom_api.list_namespaced_custom_object(
                group=_ARGOCD_GROUP,
                version=_ARGOCD_VERSION,
                namespace=ns,
                plural=_ARGOCD_PLURAL,
                limit=1,
            )
            # Verify the response contains actual ArgoCD Application items
            items = result.get("items", [])
            if items and isinstance(items[0], dict) and "spec" in items[0]:
                _argocd_namespace_cache[cache_key] = ns
                return ns
            # Empty list is still a valid "CRD exists here" signal
            if isinstance(result, dict) and "items" in result:
                _argocd_namespace_cache[cache_key] = ns
                return ns
        except k8s_exceptions.ApiException as exc:
            if exc.status == 404:
                continue  # CRD or namespace not found — try next
            if exc.status == 403:
                logger.warning(
                    "RBAC: cannot probe namespace '%s' for ArgoCD (403 Forbidden) — skipping", ns
                )
                continue
            # Unexpected API error (5xx, network, etc.) — log and continue probing
            logger.warning("K8s API error probing namespace '%s' for ArgoCD: %s", ns, exc)
            continue
        except Exception as exc:  # noqa: BLE001
            # Non-API errors (SDK not configured, network unreachable) — log and continue
            logger.warning("Unexpected error probing namespace '%s' for ArgoCD: %s", ns, exc)
            continue

    # Fallback: cluster-wide search for any ArgoCD Application
    try:
        result = custom_api.list_cluster_custom_object(
            group=_ARGOCD_GROUP,
            version=_ARGOCD_VERSION,
            plural=_ARGOCD_PLURAL,
            limit=1,
        )
        items = result.get("items", [])
        if items and isinstance(items[0], dict):
            discovered_ns: str | None = items[0].get("metadata", {}).get("namespace") or None
            if discovered_ns:
                _argocd_namespace_cache[cache_key] = discovered_ns
                return discovered_ns
    except k8s_exceptions.ApiException as exc:
        if exc.status not in (404, 403):
            logger.warning("K8s API error during cluster-wide ArgoCD discovery: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected error during cluster-wide ArgoCD discovery: %s", exc)

    _argocd_namespace_cache[cache_key] = None
    return None


def _resolve_namespace(custom_api: Any, namespace: str) -> tuple[str, bool]:
    """Resolve an ArgoCD namespace — use provided value or auto-discover.

    Args:
        custom_api: Configured ``CustomObjectsApi`` instance.
        namespace: Caller-supplied namespace (may be empty string).

    Returns:
        A ``(namespace, found)`` tuple.  ``found`` is ``False`` when discovery
        returned nothing and the caller should report "ArgoCD not found".
    """
    if namespace:
        return (namespace, True)
    discovered = _discover_argocd_namespace(custom_api)
    if discovered:
        return (discovered, True)
    return ("", False)


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
    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            logger.debug("ArgoCD CRD not found (404)")
            return []
        if exc.status == 403:
            logger.warning("RBAC: cannot list ArgoCD applications (403 Forbidden)")
            return []
        logger.warning("K8s API error listing ArgoCD applications: %s", exc)
        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected error listing ArgoCD applications: %s", exc)
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
    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return None
        if exc.status == 403:
            logger.warning("RBAC: cannot get ArgoCD application '%s' (403 Forbidden)", app_name)
            return None
        logger.warning("K8s API error getting ArgoCD application '%s': %s", app_name, exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected error getting ArgoCD application '%s': %s", app_name, exc)
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
    lines.append(f"  {'ID':<6} {'DEPLOYED AT':<25} {'REVISION':<45} {'SOURCE':<40}")
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

        lines.append(f"  {entry_id:<6} {deployed_at:<25} {revision:<45} {repo:<40}")

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
    lines.append(f"  {'KIND':<25} {'NAME':<30} {'NAMESPACE':<18} {'SYNC':<12} {'HEALTH':<12} {'HOOK':<10}")
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

        lines.append(f"  {kind:<25} {name:<30} {ns:<18} {sync_status:<12} {health_status:<12} {hook_str:<10}")

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

        lines.append(f"  {'GROUP':<25} {'NAME':<30} {'NAMESPACE':<18} {'SYNC':<12} {'HEALTH':<12} {'PRUNE':<8}")
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

            lines.append(f"  {group:<25} {name:<30} {ns:<18} {sync_status:<12} {health_status:<12} {prune_str:<8}")

        lines.append("")

    lines.append(f"Total managed resources: {len(resources)}")
    return "\n".join(lines)


# ── Public Tool Functions ────────────────────────────────────


def argocd_list_applications(
    *,
    namespace: str = "",
    _custom_api: Any = None,
) -> ToolResult:
    """List all ArgoCD Applications in the given namespace.

    Returns a formatted table with name, project, sync status, health status,
    source repo, target revision, and destination for each application.

    Args:
        namespace: Kubernetes namespace where ArgoCD is deployed. When empty,
            the namespace is auto-discovered by probing common namespace names.
        _custom_api: Optional pre-configured CustomObjectsApi (for testing).
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    custom_api = _custom_api
    if custom_api is None:
        try:
            _, custom_api = _get_argocd_client()
        except (RuntimeError, NotImplementedError) as exc:
            return ToolResult(output=f"Failed to connect to ArgoCD: {exc}", error=True)

    namespace, found = _resolve_namespace(custom_api, namespace)
    if not found:
        return ToolResult(
            output=(
                "ArgoCD not found. No Applications CRD detected in any known"
                " namespace. If ArgoCD is installed in a custom namespace,"
                " pass namespace=<your-namespace> explicitly."
            ),
            error=False,
        )

    # Cache check
    cache_key = _cache._cache_key_discovery("argocd_list", namespace)
    cached = _cache._get_cached(cache_key)
    if cached is not None:
        return ToolResult(output=cached, error=False)

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
    namespace: str = "",
    _custom_api: Any = None,
) -> ToolResult:
    """Get detailed status of a specific ArgoCD Application.

    Returns sync status, health status, source info, destination,
    conditions, operation state, and sync policy.

    Args:
        app_name: Name of the ArgoCD Application.
        namespace: Kubernetes namespace where ArgoCD is deployed. When empty,
            the namespace is auto-discovered by probing common namespace names.
        _custom_api: Optional pre-configured CustomObjectsApi (for testing).
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    custom_api = _custom_api
    if custom_api is None:
        try:
            _, custom_api = _get_argocd_client()
        except (RuntimeError, NotImplementedError) as exc:
            return ToolResult(output=f"Failed to connect to ArgoCD: {exc}", error=True)

    namespace, found = _resolve_namespace(custom_api, namespace)
    if not found:
        return ToolResult(
            output=(
                "ArgoCD not found. No Applications CRD detected in any known"
                " namespace. If ArgoCD is installed in a custom namespace,"
                " pass namespace=<your-namespace> explicitly."
            ),
            error=False,
        )

    # Cache check
    cache_key = _cache._cache_key_discovery("argocd_status", namespace, app_name)
    cached = _cache._get_cached(cache_key)
    if cached is not None:
        return ToolResult(output=cached, error=False)

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
    namespace: str = "",
    _custom_api: Any = None,
) -> ToolResult:
    """Get deployment history of an ArgoCD Application.

    Returns a table of past deployments sorted by most recent first,
    including revision, deployment time, and source info.

    Args:
        app_name: Name of the ArgoCD Application.
        namespace: Kubernetes namespace where ArgoCD is deployed. When empty,
            the namespace is auto-discovered by probing common namespace names.
        _custom_api: Optional pre-configured CustomObjectsApi (for testing).
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    custom_api = _custom_api
    if custom_api is None:
        try:
            _, custom_api = _get_argocd_client()
        except (RuntimeError, NotImplementedError) as exc:
            return ToolResult(output=f"Failed to connect to ArgoCD: {exc}", error=True)

    namespace, found = _resolve_namespace(custom_api, namespace)
    if not found:
        return ToolResult(
            output=(
                "ArgoCD not found. No Applications CRD detected in any known"
                " namespace. If ArgoCD is installed in a custom namespace,"
                " pass namespace=<your-namespace> explicitly."
            ),
            error=False,
        )

    # Cache check
    cache_key = _cache._cache_key_discovery("argocd_history", namespace, app_name)
    cached = _cache._get_cached(cache_key)
    if cached is not None:
        return ToolResult(output=cached, error=False)

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
    namespace: str = "",
    _custom_api: Any = None,
) -> ToolResult:
    """Show resources that are out-of-sync for an ArgoCD Application.

    Examines ``.status.resources[]`` and returns resources where
    sync status is not ``Synced`` or health status is not ``Healthy``.

    Args:
        app_name: Name of the ArgoCD Application.
        namespace: Kubernetes namespace where ArgoCD is deployed. When empty,
            the namespace is auto-discovered by probing common namespace names.
        _custom_api: Optional pre-configured CustomObjectsApi (for testing).
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    custom_api = _custom_api
    if custom_api is None:
        try:
            _, custom_api = _get_argocd_client()
        except (RuntimeError, NotImplementedError) as exc:
            return ToolResult(output=f"Failed to connect to ArgoCD: {exc}", error=True)

    namespace, found = _resolve_namespace(custom_api, namespace)
    if not found:
        return ToolResult(
            output=(
                "ArgoCD not found. No Applications CRD detected in any known"
                " namespace. If ArgoCD is installed in a custom namespace,"
                " pass namespace=<your-namespace> explicitly."
            ),
            error=False,
        )

    # Cache check
    cache_key = _cache._cache_key_discovery("argocd_diff", namespace, app_name)
    cached = _cache._get_cached(cache_key)
    if cached is not None:
        return ToolResult(output=cached, error=False)

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
    namespace: str = "",
    _custom_api: Any = None,
) -> ToolResult:
    """List all resources managed by an ArgoCD Application.

    Returns a formatted table grouped by kind, showing group, name,
    namespace, sync status, health status, and pruning requirements.

    Args:
        app_name: Name of the ArgoCD Application.
        namespace: Kubernetes namespace where ArgoCD is deployed. When empty,
            the namespace is auto-discovered by probing common namespace names.
        _custom_api: Optional pre-configured CustomObjectsApi (for testing).
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    custom_api = _custom_api
    if custom_api is None:
        try:
            _, custom_api = _get_argocd_client()
        except (RuntimeError, NotImplementedError) as exc:
            return ToolResult(output=f"Failed to connect to ArgoCD: {exc}", error=True)

    namespace, found = _resolve_namespace(custom_api, namespace)
    if not found:
        return ToolResult(
            output=(
                "ArgoCD not found. No Applications CRD detected in any known"
                " namespace. If ArgoCD is installed in a custom namespace,"
                " pass namespace=<your-namespace> explicitly."
            ),
            error=False,
        )

    # Cache check
    cache_key = _cache._cache_key_discovery("argocd_managed", namespace, app_name)
    cached = _cache._get_cached(cache_key)
    if cached is not None:
        return ToolResult(output=cached, error=False)

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

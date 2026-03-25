"""Argo Rollouts introspection tools — rollouts, analysis runs, and analysis templates."""

from __future__ import annotations

import logging
import time
from typing import Any

from vaig.tools.base import ToolResult

from . import _clients
from .argocd import (
    _RETRY_ATTEMPTS,
    _RETRY_BACKOFF_SECONDS,
    _check_crd_exists,
)

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
_ARGO_ROLLOUTS_GROUP = "argoproj.io"
_ARGO_ROLLOUTS_VERSION = "v1alpha1"
_ARGO_ROLLOUTS_CRD = "rollouts.argoproj.io"

_ROLLOUT_ANNOTATION_MARKERS = (
    "rollout.argoproj.io/desired-replicas",
    "rollout.argoproj.io/revision",
    "rollout.argoproj.io/workload-generation",
    "argo-rollouts.argoproj.io/managed-by-rollouts",
)

# ── Namespace-level detection cache ─────────────────────────
_rollouts_ns_cache: dict[tuple[str, Any], bool] = {}


# ── Detection ────────────────────────────────────────────────


def detect_argo_rollouts(namespace: str = "", api_client: Any = None) -> bool:
    """Detect Argo Rollouts presence for a specific namespace.

    Three-phase detection:
    1. CRD probe (best-effort, fast signal): checks whether the
       ``rollouts.argoproj.io`` CRD appears to exist cluster-wide. If this
       probe can **positively** confirm the CRD is missing (404), Argo Rollouts
       is considered not installed and the function returns ``False`` early.
       If the probe is unavailable, forbidden (403), or otherwise inconclusive,
       detection still continues via the annotation scan.
    2. Namespace annotation scan: first queries Rollout CRDs directly; falls
       back to checking deployment annotations if the CRD query fails.
       A CRD existing cluster-wide does **not** guarantee this namespace has
       Argo Rollouts–managed resources, and the annotation scan is used as a
       namespace-scoped signal even when the CRD probe could not confirm
       presence or absence.
    3. Results are cached per ``(namespace, api_client)`` to avoid redundant
       API calls and to prevent stale cache hits when the process switches
       clusters/contexts. **Transient failures are NOT cached** — the next
       call will retry.

    Args:
        namespace: Target namespace to check. Empty = "default".
        api_client: Optional pre-configured ``kubernetes.client.ApiClient``.
            When ``None`` the function loads in-cluster or kube-config
            credentials automatically.

    Returns:
        ``True`` if Argo Rollouts manages resources in the target namespace.
    """
    ns = namespace or "default"
    cache_key = (ns, api_client)

    # Check namespace-level cache, scoped by api_client/context
    if cache_key in _rollouts_ns_cache:
        return _rollouts_ns_cache[cache_key]

    result = _detect_rollouts_for_namespace(ns, api_client)
    # Only cache definitive outcomes — None means transient failure, do NOT cache.
    if result is not None:
        _rollouts_ns_cache[cache_key] = result
    return result or False


def _detect_rollouts_for_namespace(namespace: str, api_client: Any) -> bool | None:
    """Internal: uncached namespace-scoped Argo Rollouts detection.

    Returns:
        ``True`` — Argo Rollouts detected.
        ``False`` — definitively not detected (safe to cache).
        ``None`` — transient failure; caller must NOT cache this result.
    """
    # Phase 1: CRD must exist cluster-wide (necessary condition)
    crd_present = _check_crd_exists(_ARGO_ROLLOUTS_CRD, api_client=api_client)

    # Phase 2: Check namespace for Rollout resources (sufficient condition)
    # Even if CRD exists, we must verify THIS namespace has Rollouts-managed resources.
    # If CRD check failed due to RBAC (403), annotation scan is the only path.
    # Returns None on transient failure — do NOT cache that result.
    ns_has_rollouts = _scan_namespace_for_rollouts_annotations(namespace, api_client)

    if ns_has_rollouts is None:
        # Transient API failure — signal caller to skip caching.
        return None

    if ns_has_rollouts:
        if crd_present:
            logger.info(
                "Argo Rollouts detected for namespace '%s' (CRD present + resources found).",
                namespace,
            )
        else:
            logger.info(
                "Argo Rollouts detected for namespace '%s' via resource scan "
                "(CRD probe unavailable/failed).",
                namespace,
            )
        return True

    if crd_present:
        logger.debug(
            "Argo Rollouts CRD exists cluster-wide but no managed resources found "
            "in namespace '%s' — skipping Argo Rollouts tools for this namespace.",
            namespace,
        )

    return False


def _scan_namespace_for_rollouts_annotations(namespace: str, api_client: Any) -> bool | None:
    """Scan a namespace for Argo Rollouts-managed resources.

    Detection strategy (Fix C):
    1. **Primary**: Query Rollout CRDs directly via ``CustomObjectsApi``.
       If any Rollout objects exist in the namespace → True (definitive).
    2. **Fallback**: If the Rollout CRD query itself fails transiently, fall back to
       checking Deployment annotations (``rollout.argoproj.io/revision``, etc.).

    Transient errors (5xx, 429, network) are retried up to ``_RETRY_ATTEMPTS`` times
    with ``_RETRY_BACKOFF_SECONDS`` delay.  If all attempts are transient → return
    ``None`` to signal the caller that this result must NOT be cached.

    Definitive failures (403, 404, config errors) → return ``False`` (safe to cache).

    Returns:
        ``True``  — Rollout resources found in this namespace.
        ``False`` — Definitively none found (scan succeeded, nothing there).
        ``None``  — Transient API failure; caller must NOT cache this result.
    """
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
            custom_api = k8s_client.CustomObjectsApi()
            apps_api = k8s_client.AppsV1Api()
        else:
            custom_api = k8s_client.CustomObjectsApi(api_client)
            apps_api = k8s_client.AppsV1Api(api_client)

        # ── Primary path: query Rollout CRDs directly (Fix C) ──────────────
        last_exc: Exception | None = None
        crd_query_transient = False
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                result = custom_api.list_namespaced_custom_object(
                    group=_ARGO_ROLLOUTS_GROUP,
                    version=_ARGO_ROLLOUTS_VERSION,
                    namespace=namespace,
                    plural="rollouts",
                    limit=1,
                )
                items = result.get("items", [])
                if items:
                    return True
                # Successful query, no Rollout objects → definitive False for this path
                # but still check annotation fallback
                crd_query_transient = False
                break
            except k8s_exceptions.ApiException as exc:
                if exc.status in (403, 404):
                    # Definitive: no permission or CRD not installed — fall through to annotation scan.
                    logger.debug(
                        "Rollout CRD query for namespace '%s' returned %d — trying annotation fallback",
                        namespace, exc.status,
                    )
                    crd_query_transient = False
                    break
                # Transient — retry.
                last_exc = exc
                crd_query_transient = True
                if attempt < _RETRY_ATTEMPTS - 1:
                    logger.warning(
                        "Transient K8s API error querying Rollout CRDs for '%s' "
                        "(attempt %d/%d): %s — retrying",
                        namespace, attempt + 1, _RETRY_ATTEMPTS, exc,
                    )
                    time.sleep(_RETRY_BACKOFF_SECONDS)
                else:
                    logger.warning(
                        "Rollout CRD query failed for namespace '%s' after %d attempt(s): "
                        "%s — trying annotation fallback",
                        namespace, _RETRY_ATTEMPTS, exc,
                    )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                crd_query_transient = True
                if attempt < _RETRY_ATTEMPTS - 1:
                    logger.warning(
                        "Unexpected error querying Rollout CRDs for '%s' (attempt %d/%d): %s — retrying",
                        namespace, attempt + 1, _RETRY_ATTEMPTS, exc,
                    )
                    time.sleep(_RETRY_BACKOFF_SECONDS)
                else:
                    logger.warning(
                        "Rollout CRD query failed for namespace '%s' after %d attempt(s): "
                        "%s — trying annotation fallback",
                        namespace, _RETRY_ATTEMPTS, last_exc,
                    )

        # ── Fallback: Deployment annotation scan (Fix C fallback) ───────────
        last_exc = None
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                deployments = apps_api.list_namespaced_deployment(
                    namespace=namespace,
                    limit=50,
                )
                for dep in deployments.items or []:
                    annotations = dep.metadata.annotations or {}
                    if any(k in annotations for k in _ROLLOUT_ANNOTATION_MARKERS):
                        return True
                # Scan succeeded, nothing found — only report transient if CRD path was transient.
                return None if crd_query_transient else False
            except k8s_exceptions.ApiException as exc:
                if exc.status in (403, 404):
                    logger.warning(
                        "Rollout annotation scan for namespace '%s' returned %d — skipping",
                        namespace, exc.status,
                    )
                    return None if crd_query_transient else False
                # Transient — retry.
                last_exc = exc
                if attempt < _RETRY_ATTEMPTS - 1:
                    logger.warning(
                        "Transient K8s API error during Rollouts annotation scan for '%s' "
                        "(attempt %d/%d): %s — retrying",
                        namespace, attempt + 1, _RETRY_ATTEMPTS, exc,
                    )
                    time.sleep(_RETRY_BACKOFF_SECONDS)
                else:
                    logger.warning(
                        "Rollouts annotation scan failed for namespace '%s' after %d attempt(s): "
                        "%s — not caching result",
                        namespace, _RETRY_ATTEMPTS, exc,
                    )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < _RETRY_ATTEMPTS - 1:
                    logger.warning(
                        "Unexpected error during Rollouts annotation scan for '%s' "
                        "(attempt %d/%d): %s — retrying",
                        namespace, attempt + 1, _RETRY_ATTEMPTS, exc,
                    )
                    time.sleep(_RETRY_BACKOFF_SECONDS)
                else:
                    logger.warning(
                        "Rollouts annotation scan failed for namespace '%s' after %d attempt(s): "
                        "%s — not caching result",
                        namespace, _RETRY_ATTEMPTS, last_exc,
                    )

        # Both paths exhausted by transient errors — signal caller to skip caching.
        return None

    except Exception as exc:  # noqa: BLE001
        # Outer catch: config loading errors — treat as definitive failure (return False).
        logger.warning(
            "Rollouts scan failed for namespace '%s': %s",
            namespace,
            exc,
        )
        return False


# ── Helpers ──────────────────────────────────────────────────


def _get_custom_objects_api() -> Any | None:
    """Return a CustomObjectsApi instance.

    Returns ``None`` when the kubernetes SDK is not available or
    credentials cannot be loaded.
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


def _format_rollout(rollout: dict[str, Any]) -> str:
    """Format a Rollout CRD dict into a human-readable string.

    Args:
        rollout: Raw CustomObjectsApi response for an Argo Rollout resource.

    Returns:
        Multi-line string with key fields extracted safely.
    """
    meta = rollout.get("metadata", {})
    spec = rollout.get("spec", {})
    status = rollout.get("status", {})

    name = meta.get("name", "<unknown>")
    namespace = meta.get("namespace", "<unknown>")
    replicas = spec.get("replicas", "?")

    ready = status.get("readyReplicas", 0)
    available = status.get("availableReplicas", 0)
    updated = status.get("updatedReplicas", 0)
    phase = status.get("phase", "Unknown")

    # Strategy: canary or blueGreen
    strategy_spec = spec.get("strategy", {})
    if "canary" in strategy_spec:
        strategy = "canary"
        canary_status = status.get("canary", {})
        step_index = canary_status.get("currentStepIndex", "?")
        weights = canary_status.get("weights", {})
        canary_weight = weights.get("canary", {}).get("weight", "?") if isinstance(weights, dict) else "?"
        strategy_detail = f"step={step_index}, canary_weight={canary_weight}%"
    elif "blueGreen" in strategy_spec:
        strategy = "blueGreen"
        active_rs = status.get("blueGreen", {}).get("activeRS", "?")
        preview_rs = status.get("blueGreen", {}).get("previewRS", "?")
        strategy_detail = f"active={active_rs}, preview={preview_rs}"
    else:
        strategy = "unknown"
        strategy_detail = ""

    conditions = status.get("conditions", [])
    conditions = conditions if isinstance(conditions, list) else []
    cond_lines = []
    for cond in conditions:
        cond_type = cond.get("type", "?")
        cond_status = cond.get("status", "?")
        cond_msg = cond.get("message", "")
        cond_lines.append(f"    {cond_type}={cond_status}" + (f": {cond_msg}" if cond_msg else ""))

    lines = [
        f"Rollout: {name}",
        f"  Namespace:  {namespace}",
        f"  Replicas:   desired={replicas}, ready={ready}, available={available}, updated={updated}",
        f"  Phase:      {phase}",
        f"  Strategy:   {strategy}" + (f" ({strategy_detail})" if strategy_detail else ""),
    ]
    if cond_lines:
        lines.append("  Conditions:")
        lines.extend(cond_lines)
    return "\n".join(lines)


def _format_analysisrun(run: dict[str, Any]) -> str:
    """Format an AnalysisRun CRD dict into a human-readable string.

    Args:
        run: Raw CustomObjectsApi response for an AnalysisRun resource.

    Returns:
        Multi-line string with key fields extracted safely.
    """
    meta = run.get("metadata", {})
    status = run.get("status", {})

    name = meta.get("name", "<unknown>")
    namespace = meta.get("namespace", "<unknown>")
    phase = status.get("phase", "Unknown")
    message = status.get("message", "")

    metric_results = status.get("metricResults", [])
    metric_results = metric_results if isinstance(metric_results, list) else []
    metric_lines = []
    for metric in metric_results:
        metric_name = metric.get("name", "?")
        metric_phase = metric.get("phase", "?")
        metric_lines.append(f"    {metric_name}: {metric_phase}")

    lines = [
        f"AnalysisRun: {name}",
        f"  Namespace: {namespace}",
        f"  Phase:     {phase}" + (f" — {message}" if message else ""),
    ]
    if metric_lines:
        lines.append("  Metrics:")
        lines.extend(metric_lines)
    return "\n".join(lines)


def _format_analysistemplate(template: dict[str, Any]) -> str:
    """Format an AnalysisTemplate CRD dict into a human-readable string.

    Args:
        template: Raw CustomObjectsApi response for an AnalysisTemplate resource.

    Returns:
        Multi-line string with key fields extracted safely.
    """
    meta = template.get("metadata", {})
    spec = template.get("spec", {})

    name = meta.get("name", "<unknown>")
    namespace = meta.get("namespace", "<unknown>")

    metrics = spec.get("metrics", [])
    metrics = metrics if isinstance(metrics, list) else []
    metric_lines = []
    for metric in metrics:
        metric_name = metric.get("name", "?")
        provider = next(iter(metric.get("provider", {})), "unknown")
        metric_lines.append(f"    {metric_name} (provider: {provider})")

    lines = [
        f"AnalysisTemplate: {name}",
        f"  Namespace: {namespace}",
        f"  Metrics:   {len(metrics)} defined",
    ]
    if metric_lines:
        lines.extend(metric_lines)
    return "\n".join(lines)


# ── Tools ─────────────────────────────────────────────────────


def kubectl_get_rollout(namespace: str = "", name: str = "") -> ToolResult:
    """List or get Argo Rollout resources via CustomObjectsApi.

    Args:
        namespace: Kubernetes namespace.  Empty string queries all namespaces.
        name: Rollout name.  When provided, fetches a single resource;
            otherwise lists all Rollouts in the namespace.

    Returns:
        :class:`~vaig.tools.base.ToolResult` with formatted Rollout output,
        or an error result on API failure.
    """
    if not _clients._K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    custom_api = _get_custom_objects_api()
    if custom_api is None:
        return ToolResult(
            output="kubectl_get_rollout: cannot connect to cluster — kubernetes unconfigured",
            error=True,
        )

    try:
        if name:
            ns = namespace or "default"
            resource = custom_api.get_namespaced_custom_object(
                group=_ARGO_ROLLOUTS_GROUP,
                version=_ARGO_ROLLOUTS_VERSION,
                namespace=ns,
                plural="rollouts",
                name=name,
            )
            return ToolResult(output=_format_rollout(resource))

        if namespace:
            result = custom_api.list_namespaced_custom_object(
                group=_ARGO_ROLLOUTS_GROUP,
                version=_ARGO_ROLLOUTS_VERSION,
                namespace=namespace,
                plural="rollouts",
            )
        else:
            result = custom_api.list_cluster_custom_object(
                group=_ARGO_ROLLOUTS_GROUP,
                version=_ARGO_ROLLOUTS_VERSION,
                plural="rollouts",
            )

        items = result.get("items", [])
        if not items:
            scope = f"namespace '{namespace}'" if namespace else "cluster"
            return ToolResult(output=f"No Rollouts found in {scope}.")

        formatted = [_format_rollout(item) for item in items]
        return ToolResult(output="\n\n".join(formatted))

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            if name:
                ns = namespace or "default"
                return ToolResult(
                    output=f"Rollout '{name}' not found in namespace '{ns}'.", error=False
                )
            return ToolResult(output="No Rollouts found.", error=False)
        if exc.status == 403:
            return ToolResult(
                output=(
                    "RBAC: permission denied listing Rollouts — "
                    "ensure the service account has 'get/list' on "
                    "apiGroups: [\"argoproj.io\"], resources: [\"rollouts\"]."
                ),
                error=True,
            )
        return ToolResult(output=f"kubectl_get_rollout error: {exc}", error=True)

    except Exception as exc:  # noqa: BLE001
        logger.warning("kubectl_get_rollout unexpected error: %s", exc)
        return ToolResult(output=f"kubectl_get_rollout unexpected error: {exc}", error=True)


def kubectl_get_analysisrun(namespace: str = "", name: str = "") -> ToolResult:
    """List or get Argo Rollouts AnalysisRun resources via CustomObjectsApi.

    Args:
        namespace: Kubernetes namespace.  Empty string queries all namespaces.
        name: AnalysisRun name.  When provided, fetches a single resource;
            otherwise lists all AnalysisRuns in the namespace.

    Returns:
        :class:`~vaig.tools.base.ToolResult` with formatted AnalysisRun output,
        or an error result on API failure.
    """
    if not _clients._K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    custom_api = _get_custom_objects_api()
    if custom_api is None:
        return ToolResult(
            output="kubectl_get_analysisrun: cannot connect to cluster — kubernetes unconfigured",
            error=True,
        )

    try:
        if name:
            ns = namespace or "default"
            resource = custom_api.get_namespaced_custom_object(
                group=_ARGO_ROLLOUTS_GROUP,
                version=_ARGO_ROLLOUTS_VERSION,
                namespace=ns,
                plural="analysisruns",
                name=name,
            )
            return ToolResult(output=_format_analysisrun(resource))

        if namespace:
            result = custom_api.list_namespaced_custom_object(
                group=_ARGO_ROLLOUTS_GROUP,
                version=_ARGO_ROLLOUTS_VERSION,
                namespace=namespace,
                plural="analysisruns",
            )
        else:
            result = custom_api.list_cluster_custom_object(
                group=_ARGO_ROLLOUTS_GROUP,
                version=_ARGO_ROLLOUTS_VERSION,
                plural="analysisruns",
            )

        items = result.get("items", [])
        if not items:
            scope = f"namespace '{namespace}'" if namespace else "cluster"
            return ToolResult(output=f"No AnalysisRuns found in {scope}.")

        formatted = [_format_analysisrun(item) for item in items]
        return ToolResult(output="\n\n".join(formatted))

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            if name:
                ns = namespace or "default"
                return ToolResult(
                    output=f"AnalysisRun '{name}' not found in namespace '{ns}'.", error=False
                )
            return ToolResult(output="No AnalysisRuns found.", error=False)
        if exc.status == 403:
            return ToolResult(
                output=(
                    "RBAC: permission denied listing AnalysisRuns — "
                    "ensure the service account has 'get/list' on "
                    "apiGroups: [\"argoproj.io\"], resources: [\"analysisruns\"]."
                ),
                error=True,
            )
        return ToolResult(output=f"kubectl_get_analysisrun error: {exc}", error=True)

    except Exception as exc:  # noqa: BLE001
        logger.warning("kubectl_get_analysisrun unexpected error: %s", exc)
        return ToolResult(output=f"kubectl_get_analysisrun unexpected error: {exc}", error=True)


def kubectl_get_analysistemplate(namespace: str = "", name: str = "") -> ToolResult:
    """List or get Argo Rollouts AnalysisTemplate resources via CustomObjectsApi.

    Args:
        namespace: Kubernetes namespace.  Empty string queries all namespaces.
        name: AnalysisTemplate name.  When provided, fetches a single resource;
            otherwise lists all AnalysisTemplates in the namespace.

    Returns:
        :class:`~vaig.tools.base.ToolResult` with formatted AnalysisTemplate output,
        or an error result on API failure.
    """
    if not _clients._K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    custom_api = _get_custom_objects_api()
    if custom_api is None:
        return ToolResult(
            output="kubectl_get_analysistemplate: cannot connect to cluster — kubernetes unconfigured",
            error=True,
        )

    try:
        if name:
            ns = namespace or "default"
            resource = custom_api.get_namespaced_custom_object(
                group=_ARGO_ROLLOUTS_GROUP,
                version=_ARGO_ROLLOUTS_VERSION,
                namespace=ns,
                plural="analysistemplates",
                name=name,
            )
            return ToolResult(output=_format_analysistemplate(resource))

        if namespace:
            result = custom_api.list_namespaced_custom_object(
                group=_ARGO_ROLLOUTS_GROUP,
                version=_ARGO_ROLLOUTS_VERSION,
                namespace=namespace,
                plural="analysistemplates",
            )
        else:
            result = custom_api.list_cluster_custom_object(
                group=_ARGO_ROLLOUTS_GROUP,
                version=_ARGO_ROLLOUTS_VERSION,
                plural="analysistemplates",
            )

        items = result.get("items", [])
        if not items:
            scope = f"namespace '{namespace}'" if namespace else "cluster"
            return ToolResult(output=f"No AnalysisTemplates found in {scope}.")

        formatted = [_format_analysistemplate(item) for item in items]
        return ToolResult(output="\n\n".join(formatted))

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            if name:
                ns = namespace or "default"
                return ToolResult(
                    output=f"AnalysisTemplate '{name}' not found in namespace '{ns}'.", error=False
                )
            return ToolResult(output="No AnalysisTemplates found.", error=False)
        if exc.status == 403:
            return ToolResult(
                output=(
                    "RBAC: permission denied listing AnalysisTemplates — "
                    "ensure the service account has 'get/list' on "
                    "apiGroups: [\"argoproj.io\"], resources: [\"analysistemplates\"]."
                ),
                error=True,
            )
        return ToolResult(output=f"kubectl_get_analysistemplate error: {exc}", error=True)

    except Exception as exc:  # noqa: BLE001
        logger.warning("kubectl_get_analysistemplate unexpected error: %s", exc)
        return ToolResult(output=f"kubectl_get_analysistemplate unexpected error: {exc}", error=True)

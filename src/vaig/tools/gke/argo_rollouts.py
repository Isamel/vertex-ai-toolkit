"""Argo Rollouts introspection tools — rollouts, analysis runs, and analysis templates."""

from __future__ import annotations

import logging
from typing import Any

from vaig.tools.base import ToolResult

from . import _clients
from .argocd import _check_crd_exists

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


# ── Detection ────────────────────────────────────────────────


def detect_argo_rollouts(api_client: Any = None) -> bool:
    """Detect whether Argo Rollouts is installed in the cluster.

    Checks for the ``rollouts.argoproj.io`` CRD.  Results are cached
    per-process via :func:`~vaig.tools.gke.argocd._check_crd_exists`.

    Args:
        api_client: Optional pre-configured ``kubernetes.client.ApiClient``.
            When ``None`` the function loads in-cluster or kube-config
            credentials automatically.

    Returns:
        ``True`` if the CRD exists and is accessible, ``False`` otherwise.
    """
    return _check_crd_exists(_ARGO_ROLLOUTS_CRD, api_client)


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

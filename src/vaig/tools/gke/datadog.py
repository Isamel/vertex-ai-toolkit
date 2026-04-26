"""Datadog observability configuration detection for GKE workloads."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from vaig.core.config import get_settings
from vaig.tools.base import ToolResult

from . import _clients

if TYPE_CHECKING:
    from vaig.core.config import GKEConfig

logger = logging.getLogger(__name__)

# ── Datadog detection constants (module-level defaults / fallbacks) ───────────
# These are kept as fallbacks in case get_settings() is unavailable or raises.
# Detection functions try config first, then fall back to these defaults.

_DD_ANNOTATION_PREFIXES = (
    "ad.datadoghq.com/",
    "admission.datadoghq.com/",
)

_DD_LABEL_PREFIX = "tags.datadoghq.com/"

_DD_ENV_VARS = (
    "DD_AGENT_HOST",
    "DD_TRACE_AGENT_URL",
    "DD_SERVICE",
    "DD_ENV",
    "DD_VERSION",
    "DD_TRACE_ENABLED",
    "DD_PROFILING_ENABLED",
    "DD_LOGS_INJECTION",
    "DD_RUNTIME_METRICS_ENABLED",
)

_DD_AGENT_NAMESPACES = ("datadog", "monitoring", "default", "kube-system")


# ── Helper functions ──────────────────────────────────────────


def _scan_deployment_for_datadog(deploy: Any) -> dict[str, Any]:
    """Scan a single deployment object for Datadog configuration.

    Returns a dict with keys:
    - has_datadog: bool
    - annotations: dict of matching DD annotations
    - labels: dict of matching DD labels
    - env_vars: dict of DD env var names → values (redacted for secrets)
    - issues: list of detected configuration problems
    """
    result: dict[str, Any] = {
        "has_datadog": False,
        "name": "",
        "annotations": {},
        "labels": {},
        "env_vars": {},
        "issues": [],
    }

    meta = deploy.metadata
    if not meta:
        return result

    result["name"] = meta.name or ""

    # ── Deployment-level annotations & labels ─────────────────
    dd_ann, dd_lbl = _extract_dd_metadata(meta.annotations or {}, meta.labels or {})
    result["annotations"].update(dd_ann)
    result["labels"].update(dd_lbl)
    if dd_ann or dd_lbl:
        result["has_datadog"] = True

    # ── Pod template annotations & labels ────────────────────
    spec = deploy.spec
    pod_template = spec.template if spec else None
    pod_meta = pod_template.metadata if pod_template else None

    if pod_meta:
        pod_ann, pod_lbl = _extract_dd_metadata(pod_meta.annotations or {}, pod_meta.labels or {})
        result["annotations"].update(pod_ann)
        result["labels"].update(pod_lbl)
        if pod_ann or pod_lbl:
            result["has_datadog"] = True

    # ── Env vars in all containers ────────────────────────────
    pod_spec = pod_template.spec if pod_template else None
    containers = (pod_spec.containers or []) if pod_spec else []

    try:
        detection = get_settings().datadog.detection
        dd_env_vars: tuple[str, ...] | list[str] = detection.env_vars
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to load datadog detection settings: {e!r}")
        dd_env_vars = _DD_ENV_VARS

    for container in containers:
        env_list = container.env or []
        for env_var in env_list:
            if env_var.name in dd_env_vars:
                # Don't expose values from secrets/configmaps/fieldRefs
                val = env_var.value or "(from valueFrom)"
                result["env_vars"][env_var.name] = val
                result["has_datadog"] = True

    # ── Issue detection ───────────────────────────────────────
    if result["has_datadog"]:
        env_vars = result["env_vars"]
        annotations_found = result["annotations"]

        # APM enabled but no agent host configured
        trace_enabled = env_vars.get("DD_TRACE_ENABLED", "").lower()
        has_agent_host = "DD_AGENT_HOST" in env_vars or "DD_TRACE_AGENT_URL" in env_vars
        if trace_enabled == "true" and not has_agent_host:
            result["issues"].append(
                "APM tracing enabled (DD_TRACE_ENABLED=true) but no agent host configured "
                "(DD_AGENT_HOST / DD_TRACE_AGENT_URL missing). "
                "Traces will be dropped silently."
            )

        # Admission webhook present but no service tag
        # Check both the label AND the DD_SERVICE env var
        has_webhook = any(k.startswith("admission.datadoghq.com/") for k in annotations_found)
        has_service_tag = any(k == "tags.datadoghq.com/service" for k in result["labels"]) or "DD_SERVICE" in env_vars
        if has_webhook and not has_service_tag:
            result["issues"].append(
                "Datadog admission webhook annotation detected but "
                "'tags.datadoghq.com/service' label and DD_SERVICE env var are both missing. "
                "Service will be reported under a default name in Datadog APM."
            )

    return result


def _extract_dd_metadata(annotations: dict[str, str], labels: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """Extract Datadog-relevant annotations and labels from a metadata dict.

    Args:
        annotations: Raw annotations dict from a Kubernetes metadata object.
        labels: Raw labels dict from a Kubernetes metadata object.

    Returns:
        A tuple of (dd_annotations, dd_labels) containing only entries that
        match Datadog annotation prefixes or the Datadog label prefix.
    """
    try:
        detection = get_settings().datadog.detection
        ann_prefixes: tuple[str, ...] | list[str] = detection.annotation_prefixes
        lbl_prefix: str = detection.label_prefix
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to load datadog detection settings: {e!r}")
        ann_prefixes = _DD_ANNOTATION_PREFIXES
        lbl_prefix = _DD_LABEL_PREFIX

    dd_annotations: dict[str, str] = {}
    for key, val in annotations.items():
        for prefix in ann_prefixes:
            if key.startswith(prefix):
                dd_annotations[key] = val
                break

    dd_labels: dict[str, str] = {key: val for key, val in labels.items() if key.startswith(lbl_prefix)}

    return dd_annotations, dd_labels


_DD_CLUSTER_NAME_ENV_VARS: tuple[str, ...] = (
    "DD_CLUSTER_NAME",
    "DD_CLUSTER_AGENT_KUBERNETES_CLUSTER_NAME",
    "DD_KUBERNETES_KUBELET_CLUSTER_NAME",
)

_DD_CLUSTER_NAME_LABEL_KEYS: tuple[str, ...] = (
    "tags.datadoghq.com/cluster-name",
    "cluster-name",
    "datadog.com/cluster-name",
)


def _extract_cluster_name_from_daemonset(ds: Any) -> str:  # noqa: ANN401
    """Extract the Datadog cluster name from a DaemonSet object.

    Checks env vars in all containers first (highest priority), then falls
    back to metadata labels and pod template labels.

    Returns the first non-empty value found, or ``""`` if none.
    """
    # ── Env vars (all containers) ─────────────────────────────
    try:
        containers = ds.spec.template.spec.containers or []
        for container in containers:
            for env_var in container.env or []:
                if env_var.name in _DD_CLUSTER_NAME_ENV_VARS and env_var.value:
                    return str(env_var.value)
    except Exception:  # noqa: BLE001
        pass

    # ── Labels (metadata then pod template) ──────────────────
    for label_source in (
        getattr(getattr(ds, "metadata", None), "labels", None) or {},
        getattr(
            getattr(getattr(ds, "spec", None), "template", None),
            "metadata",
            None,
        )
        and getattr(ds.spec.template.metadata, "labels", None)
        or {},
    ):
        for key in _DD_CLUSTER_NAME_LABEL_KEYS:
            value = label_source.get(key, "")
            if value:
                return str(value)

    return ""


def _check_datadog_agent(apps_v1: Any) -> dict[str, Any]:
    """Check for a Datadog agent DaemonSet across common namespaces.

    Uses the AppsV1 API to look for a DaemonSet named ``datadog-agent`` (or
    labelled ``app=datadog-agent``).  A DaemonSet is the canonical way Datadog
    is deployed on Kubernetes — checking for it is more accurate than listing
    pods.

    Returns a dict with:
    - found: bool
    - namespace: str (where it was found) or ""
    - name: str (DaemonSet name) or ""
    - cluster_name: str (from env vars / labels) or ""
    - error: str or ""
    """
    try:
        from kubernetes.client import exceptions as k8s_exceptions  # noqa: WPS433
    except ImportError:
        return {
            "found": False,
            "namespace": "",
            "name": "",
            "cluster_name": "",
            "error": "kubernetes library not available",
        }

    permission_denied = False

    for ns in _DD_AGENT_NAMESPACES:
        try:
            ds_list = apps_v1.list_namespaced_daemon_set(namespace=ns, label_selector="app=datadog-agent")
            if ds_list.items:
                ds = ds_list.items[0]
                ds_name = ds.metadata.name or "datadog-agent"
                cluster_name = _extract_cluster_name_from_daemonset(ds)
                return {
                    "found": True,
                    "namespace": ns,
                    "name": ds_name,
                    "cluster_name": cluster_name,
                    "error": "",
                }
        except k8s_exceptions.ApiException as exc:
            if exc.status in (401, 403):
                permission_denied = True
                continue
            # Other errors: just continue searching
            logger.debug("Error checking datadog-agent DaemonSet in namespace %s: %s", ns, exc)
            continue
        except Exception as exc:  # noqa: BLE001
            logger.debug("Unexpected error checking datadog-agent DaemonSet in namespace %s: %s", ns, exc)
            continue

    if permission_denied:
        return {
            "found": False,
            "namespace": "",
            "name": "",
            "cluster_name": "",
            "error": "insufficient permissions to list DaemonSets in agent namespaces",
        }

    return {"found": False, "namespace": "", "name": "", "cluster_name": "", "error": ""}


def _format_datadog_report(
    namespace: str,
    deployment_filter: str,
    deployments: list[dict[str, Any]],
    agent_info: dict[str, Any],
) -> str:
    """Format the Datadog configuration findings into a text report."""
    lines: list[str] = []
    lines.append(f"## Datadog Configuration: namespace/{namespace}")
    if deployment_filter:
        lines.append(f"(filtered to deployment: {deployment_filter})")

    # ── Agent status ──────────────────────────────────────────
    lines.append("\n### Datadog Agent Status")
    if agent_info["found"]:
        lines.append(
            f"✅ Datadog agent detected: DaemonSet `{agent_info['name']}` in namespace `{agent_info['namespace']}`"
        )
        if agent_info.get("cluster_name"):
            lines.append(f"  Cluster name tag: {agent_info['cluster_name']}")
    else:
        if agent_info["error"]:
            lines.append(f"⚠️  Could not determine agent status: {agent_info['error']}")
        else:
            lines.append(
                "❌ No Datadog agent DaemonSet found in namespaces: "
                + ", ".join(_DD_AGENT_NAMESPACES)
                + ". APM traces and metrics may not be collected."
            )

    # ── Deployments with Datadog config ───────────────────────
    dd_deployments = [d for d in deployments if d["has_datadog"]]
    no_dd_deployments = [d for d in deployments if not d["has_datadog"]]

    lines.append(f"\n### Datadog-Instrumented Deployments ({len(dd_deployments)} found)")
    if not dd_deployments:
        lines.append("No Datadog configuration detected in any deployment.")
    else:
        for dd in dd_deployments:
            lines.append(f"\n#### deployment/{dd['name']}")

            if dd["annotations"]:
                lines.append("**Annotations:**")
                for key, val in sorted(dd["annotations"].items()):
                    lines.append(f"  - `{key}`: {val}")

            if dd["labels"]:
                lines.append("**Datadog Labels:**")
                for key, val in sorted(dd["labels"].items()):
                    lines.append(f"  - `{key}`: {val}")

            if dd["env_vars"]:
                lines.append("**Datadog Env Vars:**")
                for key, val in sorted(dd["env_vars"].items()):
                    lines.append(f"  - `{key}`: {val}")

            if dd["issues"]:
                lines.append("**⚠️  Configuration Issues:**")
                for issue in dd["issues"]:
                    lines.append(f"  - {issue}")
            else:
                lines.append("✅ No configuration issues detected.")

    # ── Non-Datadog deployments (brief) ───────────────────────
    if no_dd_deployments:
        names = [d["name"] for d in no_dd_deployments]
        lines.append(f"\n### Deployments Without Datadog ({len(no_dd_deployments)}): " + ", ".join(names))

    # ── Summary ───────────────────────────────────────────────
    all_issues = [issue for d in dd_deployments for issue in d["issues"]]
    lines.append("\n### Summary")
    lines.append(f"- Total deployments scanned: {len(deployments)}")
    lines.append(f"- Datadog-instrumented: {len(dd_deployments)}")
    lines.append(f"- Configuration issues found: {len(all_issues)}")

    return "\n".join(lines)


# ── Main tool function ────────────────────────────────────────


def get_datadog_config(
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
    deployment: str = "",
) -> ToolResult:
    """Detect and summarize Datadog observability configuration in GKE workloads.

    Scans deployments for Datadog annotations (ad.datadoghq.com/,
    admission.datadoghq.com/), labels (tags.datadoghq.com/), and environment
    variables (DD_AGENT_HOST, DD_TRACE_ENABLED, etc.).  Also checks for a
    Datadog agent DaemonSet in common namespaces.

    Args:
        gke_config: GKE cluster configuration.
        namespace: Kubernetes namespace to scan (default: "default").
        deployment: Optional deployment name filter.  When provided, only
            that deployment is scanned.

    Returns:
        ToolResult with a formatted Datadog configuration report.
    """
    if not _clients._K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    ns = namespace or gke_config.default_namespace or "default"

    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, apps_v1, _, _ = result

    try:
        from kubernetes.client import exceptions as k8s_exceptions  # noqa: WPS433
    except ImportError:
        return ToolResult(
            output="kubernetes library not available",
            error=True,
        )

    # ── Fetch deployments ─────────────────────────────────────
    deployment_objects: list[Any] = []

    try:
        if deployment:
            deploy_obj = apps_v1.read_namespaced_deployment(name=deployment, namespace=ns)
            deployment_objects = [deploy_obj]
        else:
            deploy_list = apps_v1.list_namespaced_deployment(namespace=ns)
            deployment_objects = deploy_list.items or []
    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(
                output=f"Deployment '{deployment}' not found in namespace '{ns}'.",
                error=True,
            )
        if exc.status == 403:
            return ToolResult(
                output="Access denied: insufficient permissions to list deployments.",
                error=True,
            )
        if exc.status == 401:
            return ToolResult(
                output="Authentication failed: check your kubeconfig or GKE credentials.",
                error=True,
            )
        logger.warning("Error fetching deployments in namespace %s: %s", ns, exc)
        return ToolResult(
            output=f"Kubernetes API error ({exc.status}): {exc.reason}",
            error=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected error fetching deployments: %s", exc)
        return ToolResult(
            output=f"Unexpected error fetching deployments: {exc}",
            error=True,
        )

    # ── Scan each deployment ───────────────────────────────────
    scanned: list[dict[str, Any]] = [_scan_deployment_for_datadog(d) for d in deployment_objects]

    # ── Check for Datadog agent ────────────────────────────────
    agent_info = _check_datadog_agent(apps_v1)

    report = _format_datadog_report(
        namespace=ns,
        deployment_filter=deployment,
        deployments=scanned,
        agent_info=agent_info,
    )

    return ToolResult(output=report)

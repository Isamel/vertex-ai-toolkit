"""Scaling diagnostics — HPA + VPA status for a target deployment."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from vaig.tools.base import ToolResult

from . import _clients

if TYPE_CHECKING:
    from vaig.core.config import GKEConfig

logger = logging.getLogger(__name__)

# ── Lazy import guard ─────────────────────────────────────────
_K8S_AVAILABLE = True
try:
    from kubernetes.client import exceptions as k8s_exceptions  # noqa: WPS433
except ImportError:
    _K8S_AVAILABLE = False

# VPA CRD details
_VPA_GROUP = "autoscaling.k8s.io"
_VPA_VERSION = "v1"
_VPA_PLURAL = "verticalpodautoscalers"


# ── HPA helpers ───────────────────────────────────────────────


def _format_quantity(value: str | None) -> str:
    """Return the quantity string or '?' if None."""
    if value is None:
        return "?"
    return str(value)


def _metric_current_value(current_metric: dict[str, Any]) -> str:
    """Extract current value string from a single currentMetrics entry."""
    mtype = current_metric.get("type", "")
    if mtype == "Resource":
        resource = current_metric.get("resource", {})
        current = resource.get("current", {})
        avg_util = current.get("averageUtilization")
        avg_val = current.get("averageValue")
        if avg_util is not None:
            return f"{avg_util}%"
        if avg_val is not None:
            return str(avg_val)
    elif mtype == "External":
        external = current_metric.get("external", {})
        current = external.get("current", {})
        avg_val = current.get("averageValue")
        val = current.get("value")
        return str(avg_val or val or "?")
    elif mtype == "Pods":
        pods = current_metric.get("pods", {})
        current = pods.get("current", {})
        return str(current.get("averageValue", "?"))
    elif mtype == "ContainerResource":
        container = current_metric.get("containerResource", {})
        current = container.get("current", {})
        avg_util = current.get("averageUtilization")
        avg_val = current.get("averageValue")
        if avg_util is not None:
            return f"{avg_util}%"
        if avg_val is not None:
            return str(avg_val)
    elif mtype == "Object":
        obj = current_metric.get("object", {})
        current = obj.get("current", {})
        val = current.get("value") or current.get("averageValue")
        return str(val) if val is not None else "?"
    return "?"


def _metric_target_value(spec_metric: dict[str, Any]) -> tuple[str, str, str]:
    """Return (type, name, target) from a single spec.metrics entry."""
    mtype = spec_metric.get("type", "")
    if mtype == "Resource":
        resource = spec_metric.get("resource", {})
        name = resource.get("name", "<unknown>")
        target = resource.get("target", {})
        target_type = target.get("type", "")
        if target_type == "Utilization":
            return mtype, name, f"{target.get('averageUtilization', '?')}%"
        if target_type == "AverageValue":
            return mtype, name, str(target.get("averageValue", "?"))
        return mtype, name, str(target.get("value", "?"))
    elif mtype == "External":
        external = spec_metric.get("external", {})
        metric = external.get("metric", {})
        name = metric.get("name", "<unknown>")
        target = external.get("target", {})
        val = target.get("averageValue") or target.get("value")
        return mtype, name, str(val or "?")
    elif mtype == "Pods":
        pods = spec_metric.get("pods", {})
        metric = pods.get("metric", {})
        name = metric.get("name", "<unknown>")
        target = pods.get("target", {})
        return mtype, name, str(target.get("averageValue", "?"))
    elif mtype == "ContainerResource":
        container = spec_metric.get("containerResource", {})
        name = container.get("name", "<unknown>")
        target = container.get("target", {})
        target_type = target.get("type", "")
        if target_type == "Utilization":
            return mtype, name, f"{target.get('averageUtilization', '?')}%"
        return mtype, name, str(target.get("averageValue", "?"))
    elif mtype == "Object":
        obj = spec_metric.get("object", {})
        metric = obj.get("metric", {})
        name = metric.get("name", "<unknown>")
        target = obj.get("target", {})
        val = target.get("value") or target.get("averageValue")
        return mtype, name, str(val) if val is not None else "?"
    return mtype, "<unknown>", "?"


def _build_current_metrics_index(
    current_metrics: list[dict[str, Any]],
) -> dict[tuple[str, str], str]:
    """Build a (type, name) → current_value index from status.currentMetrics."""
    index: dict[tuple[str, str], str] = {}
    for cm in current_metrics:
        mtype = cm.get("type", "")
        name = "<unknown>"
        if mtype == "Resource":
            name = cm.get("resource", {}).get("name", "<unknown>")
        elif mtype == "External":
            name = cm.get("external", {}).get("metric", {}).get("name", "<unknown>")
        elif mtype == "Pods":
            name = cm.get("pods", {}).get("metric", {}).get("name", "<unknown>")
        elif mtype == "ContainerResource":
            name = cm.get("containerResource", {}).get("name", "<unknown>")
        elif mtype == "Object":
            name = cm.get("object", {}).get("metric", {}).get("name", "<unknown>")
        index[(mtype, name)] = _metric_current_value(cm)
    return index


def _format_hpa_section(hpa: Any) -> str:
    """Format a single HPA object into a text section."""
    lines: list[str] = []
    meta = hpa.metadata
    spec = hpa.spec
    status = hpa.status

    hpa_name = meta.name if meta else "<unknown>"
    lines.append(f"\n### Horizontal Pod Autoscaler: {hpa_name}")

    current_replicas_raw = status.current_replicas if status else None
    current_replicas: int | str = current_replicas_raw if current_replicas_raw is not None else "unknown"
    desired_replicas = (status.desired_replicas or 0) if status else 0
    min_replicas = spec.min_replicas if spec else None
    max_replicas = spec.max_replicas if spec else "?"
    min_str = str(min_replicas) if min_replicas is not None else "1"
    lines.append(f"- Replicas: {current_replicas}/{desired_replicas} (min: {min_str}, max: {max_replicas})")

    # Conditions
    conditions = (status.conditions or []) if status else []
    if conditions:
        cond_strs = [f"{c.type}={c.status}" for c in conditions if hasattr(c, "type") and hasattr(c, "status")]
        if cond_strs:
            lines.append(f"- Conditions: {', '.join(cond_strs)}")

    # Metrics table
    spec_metrics: list[dict[str, Any]] = []
    if spec and spec.metrics:
        # spec.metrics items are objects — convert via __dict__ or use to_dict()
        for m in spec.metrics:
            if hasattr(m, "to_dict"):
                spec_metrics.append(m.to_dict())
            elif hasattr(m, "__dict__"):
                spec_metrics.append(m.__dict__)

    current_metrics_raw: list[dict[str, Any]] = []
    if status and status.current_metrics:
        for cm in status.current_metrics:
            if hasattr(cm, "to_dict"):
                current_metrics_raw.append(cm.to_dict())
            elif hasattr(cm, "__dict__"):
                current_metrics_raw.append(cm.__dict__)

    current_index = _build_current_metrics_index(current_metrics_raw)

    if spec_metrics:
        lines.append("\n#### Metrics")
        lines.append(f"{'Type':<18} {'Name':<22} {'Current':<12} {'Target':<12} {'% of Target':<13} Status")
        lines.append("-" * 90)
        for sm in spec_metrics:
            mtype, mname, target = _metric_target_value(sm)
            current = current_index.get((mtype, mname), "?")

            # Compute % of target where possible
            pct_str = "?"
            try:
                if "%" in target and "%" in current:
                    pct = float(current.rstrip("%")) / float(target.rstrip("%")) * 100
                    pct_str = f"{pct:.0f}%"
            except (ValueError, ZeroDivisionError):
                pass

            status_icon = "✅" if current != "?" else "⚠️ "
            lines.append(f"{mtype:<18} {mname:<22} {current:<12} {target:<12} {pct_str:<13} {status_icon}")

    return "\n".join(lines)


# ── VPA helpers ───────────────────────────────────────────────


def _format_vpa_section(vpa: dict[str, Any]) -> str:
    """Format a single VPA dict into a text section."""
    lines: list[str] = []
    meta = vpa.get("metadata", {})
    spec = vpa.get("spec", {})
    status = vpa.get("status", {})

    vpa_name = meta.get("name", "<unknown>")
    lines.append(f"\n### Vertical Pod Autoscaler: {vpa_name}")

    update_policy = spec.get("updatePolicy", {})
    update_mode = update_policy.get("updateMode", "<unknown>")
    lines.append(f"- Update Mode: {update_mode}")

    resource_policy = spec.get("resourcePolicy", {})
    container_policies = resource_policy.get("containerPolicies", [])
    controlled_values = "<default>"
    if container_policies:
        cv = container_policies[0].get("controlledValues", "RequestsAndLimits")
        controlled_values = str(cv)
    lines.append(f"- Controlled Values: {controlled_values}")

    # Recommendations
    recommendation = status.get("recommendation", {})
    container_recommendations = recommendation.get("containerRecommendations", [])

    if container_recommendations:
        lines.append("\n#### Recommendations")
        for cr in container_recommendations:
            container_name = cr.get("containerName", "<unknown>")
            lines.append(f"\n**Container: {container_name}**")
            lines.append(f"{'Resource':<12} {'Target':<14} {'Lower Bound':<16} {'Upper Bound':<16} Uncapped Target")
            lines.append("-" * 75)

            target = cr.get("target", {})
            lower = cr.get("lowerBound", {})
            upper = cr.get("upperBound", {})
            uncapped = cr.get("uncappedTarget", {})

            resources = sorted(
                set(list(target.keys()) + list(lower.keys()) + list(upper.keys()) + list(uncapped.keys()))
            )
            for res in resources:
                t_val = _format_quantity(target.get(res))
                l_val = _format_quantity(lower.get(res))
                u_val = _format_quantity(upper.get(res))
                uc_val = _format_quantity(uncapped.get(res))
                lines.append(f"{res:<12} {t_val:<14} {l_val:<16} {u_val:<16} {uc_val}")
    else:
        lines.append("- Recommendations: Not yet available (VPA may still be learning)")

    return "\n".join(lines)


# ── Scaling assessment ────────────────────────────────────────


def _scaling_assessment(hpa_found: bool, vpa_found: bool) -> str:
    """Return a brief scaling assessment blurb."""
    lines = ["\n### Scaling Assessment"]
    if hpa_found and vpa_found:
        lines.append(
            "⚠️  Both HPA and VPA are configured. HPA scales replicas horizontally while VPA "
            "adjusts resource requests vertically. This combination can cause conflicts — "
            "ensure HPA uses CPU *utilization* targets (not raw requests) and VPA update mode "
            "is set to 'Initial' or 'Off' to avoid eviction loops."
        )
    elif hpa_found:
        lines.append(
            "✅ HPA is active. Horizontal scaling is configured. "
            "Consider adding VPA in 'Off' mode to get resource recommendations without automatic evictions."
        )
    elif vpa_found:
        lines.append(
            "✅ VPA is active. Vertical scaling is configured. "
            "No HPA detected — replicas are static. "
            "Consider HPA if the workload has variable throughput."
        )
    else:
        lines.append(
            "❌ No autoscaler found for this deployment. "
            "Consider configuring HPA for horizontal scaling or VPA for resource right-sizing."
        )
    return "\n".join(lines)


# ── Main tool function ────────────────────────────────────────


def get_scaling_status(
    name: str,
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
) -> ToolResult:
    """Retrieve and format comprehensive scaling metrics for a target deployment.

    Combines HPA (Horizontal Pod Autoscaler) and VPA (Vertical Pod Autoscaler)
    data into a unified scaling status report. Shows current vs target metrics
    for HPA, and resource recommendations for VPA.

    Args:
        name: The deployment name to check scaling status for.
        gke_config: GKE cluster configuration.
        namespace: Kubernetes namespace (default: "default").

    Returns:
        ToolResult with a formatted scaling status report.
    """
    if not _clients._K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    ns = namespace or gke_config.default_namespace or "default"

    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    _, _, custom_api, api_client = result

    # ── Fetch HPA ─────────────────────────────────────────────
    hpa_obj: Any = None
    hpa_error: str | None = None
    target_kind: str = "deployment"

    try:
        from kubernetes.client import AutoscalingV2Api  # noqa: WPS433

        auto_v2 = AutoscalingV2Api(api_client=api_client)
        hpa_list = auto_v2.list_namespaced_horizontal_pod_autoscaler(namespace=ns)
        for hpa in hpa_list.items or []:
            spec = hpa.spec
            scale_target = spec.scale_target_ref if spec else None
            if scale_target and scale_target.kind.lower() in ("deployment", "rollout") and scale_target.name == name:
                hpa_obj = hpa
                target_kind = scale_target.kind.lower()
                break
    except k8s_exceptions.ApiException as exc:
        if exc.status == 403:
            hpa_error = "Access denied: insufficient permissions to list HPAs"
        elif exc.status == 401:
            hpa_error = "Authentication failed: check your kubeconfig or GKE credentials"
        else:
            logger.warning("Error fetching HPA for deployment %s: %s", name, exc)
            hpa_error = f"Kubernetes API error ({exc.status}): {exc.reason}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected error fetching HPA for deployment %s: %s", name, exc)
        hpa_error = f"Error fetching HPA: {exc}"

    # ── Fetch VPA ─────────────────────────────────────────────
    vpa_obj: dict[str, Any] | None = None
    vpa_not_installed = False
    vpa_error: str | None = None

    try:
        vpa_list = custom_api.list_namespaced_custom_object(
            group=_VPA_GROUP,
            version=_VPA_VERSION,
            namespace=ns,
            plural=_VPA_PLURAL,
        )
        for vpa in vpa_list.get("items", []) or []:
            spec = vpa.get("spec", {})
            target_ref = spec.get("targetRef", {})
            if target_ref.get("kind", "").lower() in ("deployment", "rollout") and target_ref.get("name") == name:
                vpa_obj = vpa
                break
    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            vpa_not_installed = True
        elif exc.status == 403:
            vpa_error = "Access denied: insufficient permissions to list VPAs"
        elif exc.status == 401:
            vpa_error = "Authentication failed: check your kubeconfig or GKE credentials"
        else:
            logger.warning("Error fetching VPA for deployment %s: %s", name, exc)
            vpa_error = f"Kubernetes API error ({exc.status}): {exc.reason}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected error fetching VPA for deployment %s: %s", name, exc)
        vpa_error = f"Error fetching VPA: {exc}"

    # ── Build report ──────────────────────────────────────────
    lines: list[str] = []
    lines.append(f"## Scaling Status: {name} (namespace: {ns})")

    # Active strategy summary
    lines.append("\n### Active Scaling Strategy")
    if hpa_error:
        lines.append(f"- HPA: ⚠️  Error — {hpa_error}")
    elif hpa_obj is not None:
        lines.append("- HPA: ✅ Active")
    else:
        lines.append("- HPA: ❌ Not found")

    if vpa_error:
        lines.append(f"- VPA: ⚠️  Error — {vpa_error}")
    elif vpa_not_installed:
        lines.append("- VPA: ❌ Not installed (VPA CRD not present in cluster)")
    elif vpa_obj is not None:
        spec = vpa_obj.get("spec", {})
        update_policy = spec.get("updatePolicy", {})
        update_mode = update_policy.get("updateMode", "Auto")
        lines.append(f"- VPA: ✅ Active (mode: {update_mode})")
    else:
        lines.append("- VPA: ❌ Not found for this deployment")

    # HPA section
    if hpa_obj is not None:
        lines.append(_format_hpa_section(hpa_obj))
    elif hpa_error is None:
        lines.append(f"\n### Horizontal Pod Autoscaler\nNo HPA configured for {target_kind}/{name} in namespace '{ns}'.")

    # VPA section
    if vpa_obj is not None:
        lines.append(_format_vpa_section(vpa_obj))
    elif not vpa_not_installed and vpa_error is None:
        lines.append(f"\n### Vertical Pod Autoscaler\nNo VPA configured for {target_kind}/{name} in namespace '{ns}'.")

    # Assessment
    lines.append(
        _scaling_assessment(
            hpa_found=hpa_obj is not None,
            vpa_found=vpa_obj is not None,
        )
    )

    return ToolResult(output="\n".join(lines))

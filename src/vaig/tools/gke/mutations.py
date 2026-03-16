"""Write operations — scale, restart, label, annotate."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from vaig.tools.base import ToolResult

from . import _clients, _resources

if TYPE_CHECKING:
    from vaig.core.config import GKEConfig

logger = logging.getLogger(__name__)

# ── Lazy import guard (mirrors _clients.py) ──────────────────
# Needed locally for except clauses.
_K8S_AVAILABLE = True
try:
    from kubernetes.client import exceptions as k8s_exceptions  # noqa: WPS433
except ImportError:
    _K8S_AVAILABLE = False


# ── Resource sets ────────────────────────────────────────────
_SCALABLE_RESOURCES = frozenset({"deployments", "statefulsets", "replicasets"})
_RESTARTABLE_RESOURCES = frozenset({"deployments", "statefulsets", "daemonsets"})
_LABELABLE_RESOURCES = frozenset({
    "pods", "deployments", "services", "configmaps", "secrets",
    "statefulsets", "daemonsets", "namespaces", "nodes",
})

# Safety limits
_MAX_REPLICAS = 50
_MIN_REPLICAS = 0


# ── kubectl_scale ────────────────────────────────────────────


def kubectl_scale(
    resource: str,
    name: str,
    replicas: int,
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
) -> ToolResult:
    """Scale a Kubernetes deployment, statefulset, or replicaset.

    Safety guardrails:
    - Only deployments, statefulsets, and replicasets can be scaled.
    - Replicas are clamped to 0-50 range.
    - The current replica count is reported in the response.
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    resource = _resources._normalise_resource(resource)
    if resource not in _SCALABLE_RESOURCES:
        return ToolResult(
            output=f"Cannot scale '{resource}'. Scalable resources: {', '.join(sorted(_SCALABLE_RESOURCES))}",
            error=True,
        )

    if replicas < _MIN_REPLICAS or replicas > _MAX_REPLICAS:
        return ToolResult(
            output=f"Replicas must be between {_MIN_REPLICAS} and {_MAX_REPLICAS}. Got: {replicas}",
            error=True,
        )

    ns = namespace or gke_config.default_namespace
    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, apps_v1, _custom_api, _api_client = result

    try:
        # Read current state
        if resource == "deployments":
            obj = apps_v1.read_namespaced_deployment(name, ns)
        elif resource == "statefulsets":
            obj = apps_v1.read_namespaced_stateful_set(name, ns)
        else:  # replicasets
            obj = apps_v1.read_namespaced_replica_set(name, ns)

        current = obj.spec.replicas or 0

        # Apply scale
        body = {"spec": {"replicas": replicas}}
        if resource == "deployments":
            apps_v1.patch_namespaced_deployment_scale(name, ns, body)
        elif resource == "statefulsets":
            apps_v1.patch_namespaced_stateful_set_scale(name, ns, body)
        else:
            apps_v1.patch_namespaced_replica_set_scale(name, ns, body)

        return ToolResult(
            output=f"Scaled {resource}/{name} in namespace '{ns}': {current} -> {replicas} replicas",
        )

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(output=f"{resource}/{name} not found in namespace '{ns}'", error=True)
        if exc.status == 403:
            return ToolResult(output=f"Access denied: insufficient permissions to scale {resource}/{name}", error=True)
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:
        logger.exception("kubectl_scale failed")
        return ToolResult(output=f"Error scaling {resource}/{name}: {exc}", error=True)


# ── kubectl_restart ──────────────────────────────────────────


def kubectl_restart(
    resource: str,
    name: str,
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
) -> ToolResult:
    """Trigger a rolling restart of a deployment, statefulset, or daemonset.

    This is equivalent to ``kubectl rollout restart``. It patches the pod
    template annotation with a timestamp, causing a rolling update.
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    resource = _resources._normalise_resource(resource)
    if resource not in _RESTARTABLE_RESOURCES:
        return ToolResult(
            output=f"Cannot restart '{resource}'. Restartable resources: {', '.join(sorted(_RESTARTABLE_RESOURCES))}",
            error=True,
        )

    ns = namespace or gke_config.default_namespace
    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    _core_v1, apps_v1, _custom_api, _api_client = result

    now = datetime.now(UTC).isoformat()
    patch_body = {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "kubectl.kubernetes.io/restartedAt": now,
                    },
                },
            },
        },
    }

    try:
        if resource == "deployments":
            apps_v1.patch_namespaced_deployment(name, ns, patch_body)
        elif resource == "statefulsets":
            apps_v1.patch_namespaced_stateful_set(name, ns, patch_body)
        else:  # daemonsets
            apps_v1.patch_namespaced_daemon_set(name, ns, patch_body)

        return ToolResult(
            output=f"Rolling restart triggered for {resource}/{name} in namespace '{ns}' at {now}",
        )

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(output=f"{resource}/{name} not found in namespace '{ns}'", error=True)
        if exc.status == 403:
            return ToolResult(output=f"Access denied: insufficient permissions to restart {resource}/{name}", error=True)
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:
        logger.exception("kubectl_restart failed")
        return ToolResult(output=f"Error restarting {resource}/{name}: {exc}", error=True)


# ── kubectl_label ────────────────────────────────────────────


def kubectl_label(
    resource: str,
    name: str,
    labels: str,
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
) -> ToolResult:
    """Add or update labels on a Kubernetes resource.

    Labels format: ``key1=value1,key2=value2``.
    To remove a label, use ``key-`` (e.g., ``obsolete-``).

    Safety guardrails:
    - Label keys and values are validated (alphanumeric, dashes, dots, underscores).
    - System labels (``kubernetes.io/``, ``k8s.io/``) cannot be modified.
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    resource = _resources._normalise_resource(resource)
    if resource not in _LABELABLE_RESOURCES:
        return ToolResult(
            output=f"Cannot label '{resource}'. Supported: {', '.join(sorted(_LABELABLE_RESOURCES))}",
            error=True,
        )

    # Parse and validate labels
    label_dict: dict[str, str | None] = {}
    for part in labels.split(","):
        part = part.strip()
        if not part:
            continue
        if part.endswith("-"):
            # Remove label
            key = part[:-1]
            label_dict[key] = None
        elif "=" in part:
            key, value = part.split("=", 1)
            label_dict[key] = value
        else:
            return ToolResult(
                output=f"Invalid label format: '{part}'. Use 'key=value' to set or 'key-' to remove.",
                error=True,
            )

    if not label_dict:
        return ToolResult(output="No labels specified.", error=True)

    # Validate keys — block system labels
    _SYSTEM_PREFIXES = ("kubernetes.io/", "k8s.io/")
    for key in label_dict:
        if any(key.startswith(p) or ("/" in key and key.split("/")[0] + "/" in p) for p in _SYSTEM_PREFIXES):
            return ToolResult(
                output=f"Cannot modify system label: '{key}'. System labels (kubernetes.io/, k8s.io/) are managed by Kubernetes.",
                error=True,
            )
        # Validate label key format (simplified)
        label_key_part = key.split("/")[-1] if "/" in key else key
        if not re.match(r"^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?$", label_key_part):
            return ToolResult(
                output=f"Invalid label key: '{key}'. Must be alphanumeric with optional dashes, dots, or underscores.",
                error=True,
            )

    ns = namespace or gke_config.default_namespace
    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, apps_v1, _custom_api, _api_client = result

    patch_body = {"metadata": {"labels": label_dict}}

    try:
        # Route to the correct API
        _CORE_RESOURCES = {"pods", "services", "configmaps", "secrets", "namespaces", "nodes"}
        _APPS_RESOURCES = {"deployments", "statefulsets", "daemonsets"}

        if resource in _CORE_RESOURCES:
            if resource == "pods":
                core_v1.patch_namespaced_pod(name, ns, patch_body)
            elif resource == "services":
                core_v1.patch_namespaced_service(name, ns, patch_body)
            elif resource == "configmaps":
                core_v1.patch_namespaced_config_map(name, ns, patch_body)
            elif resource == "secrets":
                core_v1.patch_namespaced_secret(name, ns, patch_body)
            elif resource == "namespaces":
                core_v1.patch_namespace(name, patch_body)
            elif resource == "nodes":
                core_v1.patch_node(name, patch_body)
        elif resource in _APPS_RESOURCES:
            if resource == "deployments":
                apps_v1.patch_namespaced_deployment(name, ns, patch_body)
            elif resource == "statefulsets":
                apps_v1.patch_namespaced_stateful_set(name, ns, patch_body)
            elif resource == "daemonsets":
                apps_v1.patch_namespaced_daemon_set(name, ns, patch_body)

        applied = [f"{k}={v}" if v is not None else f"{k}-" for k, v in label_dict.items()]
        return ToolResult(
            output=f"Labels updated on {resource}/{name} in namespace '{ns}': {', '.join(applied)}",
        )

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(output=f"{resource}/{name} not found in namespace '{ns}'", error=True)
        if exc.status == 403:
            return ToolResult(output=f"Access denied: insufficient permissions to label {resource}/{name}", error=True)
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:
        logger.exception("kubectl_label failed")
        return ToolResult(output=f"Error labeling {resource}/{name}: {exc}", error=True)


# ── kubectl_annotate ─────────────────────────────────────────


def kubectl_annotate(
    resource: str,
    name: str,
    annotations: str,
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
) -> ToolResult:
    """Add or update annotations on a Kubernetes resource.

    Annotations format: ``key1=value1,key2=value2``.
    To remove an annotation, use ``key-`` (e.g., ``obsolete-``).

    Safety guardrails:
    - System annotations (``kubernetes.io/``, ``k8s.io/``) cannot be modified.
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    resource = _resources._normalise_resource(resource)
    if resource not in _LABELABLE_RESOURCES:
        return ToolResult(
            output=f"Cannot annotate '{resource}'. Supported: {', '.join(sorted(_LABELABLE_RESOURCES))}",
            error=True,
        )

    # Parse annotations
    ann_dict: dict[str, str | None] = {}
    for part in annotations.split(","):
        part = part.strip()
        if not part:
            continue
        if part.endswith("-"):
            key = part[:-1]
            ann_dict[key] = None
        elif "=" in part:
            key, value = part.split("=", 1)
            ann_dict[key] = value
        else:
            return ToolResult(
                output=f"Invalid annotation format: '{part}'. Use 'key=value' to set or 'key-' to remove.",
                error=True,
            )

    if not ann_dict:
        return ToolResult(output="No annotations specified.", error=True)

    # Block system annotations
    _SYSTEM_PREFIXES = ("kubernetes.io/", "k8s.io/")
    for key in ann_dict:
        if any(key.startswith(p) for p in _SYSTEM_PREFIXES):
            return ToolResult(
                output=f"Cannot modify system annotation: '{key}'. System annotations (kubernetes.io/, k8s.io/) are managed by Kubernetes.",
                error=True,
            )

    ns = namespace or gke_config.default_namespace
    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, apps_v1, _custom_api, _api_client = result

    patch_body = {"metadata": {"annotations": ann_dict}}

    try:
        _CORE_RESOURCES = {"pods", "services", "configmaps", "secrets", "namespaces", "nodes"}
        _APPS_RESOURCES = {"deployments", "statefulsets", "daemonsets"}

        if resource in _CORE_RESOURCES:
            if resource == "pods":
                core_v1.patch_namespaced_pod(name, ns, patch_body)
            elif resource == "services":
                core_v1.patch_namespaced_service(name, ns, patch_body)
            elif resource == "configmaps":
                core_v1.patch_namespaced_config_map(name, ns, patch_body)
            elif resource == "secrets":
                core_v1.patch_namespaced_secret(name, ns, patch_body)
            elif resource == "namespaces":
                core_v1.patch_namespace(name, patch_body)
            elif resource == "nodes":
                core_v1.patch_node(name, patch_body)
        elif resource in _APPS_RESOURCES:
            if resource == "deployments":
                apps_v1.patch_namespaced_deployment(name, ns, patch_body)
            elif resource == "statefulsets":
                apps_v1.patch_namespaced_stateful_set(name, ns, patch_body)
            elif resource == "daemonsets":
                apps_v1.patch_namespaced_daemon_set(name, ns, patch_body)

        applied = [f"{k}={v}" if v is not None else f"{k}-" for k, v in ann_dict.items()]
        return ToolResult(
            output=f"Annotations updated on {resource}/{name} in namespace '{ns}': {', '.join(applied)}",
        )

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(output=f"{resource}/{name} not found in namespace '{ns}'", error=True)
        if exc.status == 403:
            return ToolResult(output=f"Access denied: insufficient permissions to annotate {resource}/{name}", error=True)
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:
        logger.exception("kubectl_annotate failed")
        return ToolResult(output=f"Error annotating {resource}/{name}: {exc}", error=True)


# ── Task 3.4 — async wrappers ───────────────────────────────
# Offload blocking kubernetes-client calls to a thread pool via to_async.

from vaig.core.async_utils import to_async  # noqa: E402

async_kubectl_scale = to_async(kubectl_scale)
async_kubectl_restart = to_async(kubectl_restart)
async_kubectl_label = to_async(kubectl_label)
async_kubectl_annotate = to_async(kubectl_annotate)

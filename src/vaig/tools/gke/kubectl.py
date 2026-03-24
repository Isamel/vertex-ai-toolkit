"""Read operations — kubectl get, describe, logs, top equivalents."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from vaig.tools.base import ToolResult

from . import _clients, _formatters, _resources

if TYPE_CHECKING:
    from vaig.core.config import GKEConfig

logger = logging.getLogger(__name__)

# ── Lazy import guard (mirrors _clients.py) ──────────────────
# Needed locally for except clauses and k8s_client usage in _format_describe.
_K8S_AVAILABLE = True
try:
    from kubernetes import client as k8s_client  # noqa: WPS433
    from kubernetes.client import exceptions as k8s_exceptions  # noqa: WPS433
except ImportError:
    _K8S_AVAILABLE = False


# ── _kubectl_get_comma_separated ──────────────────────────────


def _kubectl_get_comma_separated(
    resource: str,
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
    output: str = "table",
    label_selector: str | None = None,
    field_selector: str | None = None,
) -> ToolResult:
    """Handle comma-separated resource types (e.g. 'pods,deployments,hpa').

    Splits the resource string, validates each part individually via
    ``_normalise_resource()``, then calls the single-resource logic for each
    valid type and combines the results — mirroring the ``resource='all'``
    pattern.
    """
    parts = [p.strip() for p in resource.split(",") if p.strip()]

    if not parts:
        return ToolResult(
            output="No resource types provided; expected comma-separated list like 'pods,deployments'.",
            error=True,
        )

    # Validate all parts first — fail fast on any invalid resource
    normalised: list[str] = []
    for part in parts:
        norm = _resources._normalise_resource(part)
        if norm == "all":
            return ToolResult(
                output=(
                    "'all' cannot be used inside a comma-separated resource list. "
                    "Use resource='all' by itself to list all resource types."
                ),
                error=True,
            )
        if norm not in _resources._RESOURCE_API_MAP:
            if norm in _resources._KNOWN_K8S_RESOURCES:
                return ToolResult(
                    output=(
                        f"Resource type '{part}' is a valid Kubernetes resource but is not yet "
                        f"supported by this tool. Consider using kubectl directly for this resource."
                    ),
                    error=True,
                )
            supported = sorted(_resources._RESOURCE_API_MAP.keys())
            return ToolResult(
                output=f"Unsupported resource type: '{part}'. Supported: {', '.join(supported)}",
                error=True,
            )
        normalised.append(norm)

    sections: list[str] = []
    errors: list[str] = []
    any_success = False

    for rtype in normalised:
        sub = kubectl_get(
            rtype,
            gke_config=gke_config,
            namespace=namespace,
            output=output,
            label_selector=label_selector,
            field_selector=field_selector,
        )
        if sub.error:
            errors.append(f"{rtype}: {sub.output}")
            continue
        any_success = True
        body = sub.output.strip() if sub.output else ""
        if body and body != "No resources found." and body != "[]":
            sections.append(f"=== {rtype.upper()} ===\n{body}")
        else:
            logger.debug(
                "kubectl_get comma-separated: no results for resource '%s' in namespace '%s' — skipping section",
                rtype,
                namespace,
            )

    if not any_success and errors:
        return ToolResult(
            output="Failed to list resources:\n" + "\n".join(errors),
            error=True,
        )

    combined = "\n\n".join(sections)
    if errors:
        combined += "\n\n--- Errors ---\n" + "\n".join(errors)
    if not combined:
        ns = namespace or gke_config.default_namespace
        combined = f"No resources found in namespace '{ns}'."
    return ToolResult(output=combined)


# ── kubectl_get ──────────────────────────────────────────────


def kubectl_get(
    resource: str,
    *,
    gke_config: GKEConfig,
    name: str | None = None,
    namespace: str = "default",
    output: str = "table",
    label_selector: str | None = None,
    field_selector: str | None = None,
) -> ToolResult:
    """List or get Kubernetes resources (read-only kubectl get equivalent).

    Supports pods, deployments, services, configmaps, hpa, ingress, nodes,
    namespaces, statefulsets, daemonsets, jobs, cronjobs, pv, pvc, secrets,
    serviceaccounts, endpoints, networkpolicies, and replicasets.

    Use ``resource='all'`` to query pods, services, deployments, replicasets,
    statefulsets, daemonsets, jobs, cronjobs, and hpa at once (mirrors
    ``kubectl get all``).
    """
    if not _clients._K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    # ── Handle comma-separated resource types ────────────────
    # LLMs often call kubectl_get(resource="pods,deployments,replicasets,hpa").
    # Split, validate each part individually, and combine results.
    if "," in resource:
        if name:
            return ToolResult(
                output="Cannot use 'name' filter with comma-separated resources. Specify a single resource type instead.",
                error=True,
            )
        return _kubectl_get_comma_separated(
            resource,
            gke_config=gke_config,
            namespace=namespace,
            output=output,
            label_selector=label_selector,
            field_selector=field_selector,
        )

    resource = _resources._normalise_resource(resource)

    # ── Handle resource="all" ────────────────────────────────
    # Mirrors ``kubectl get all``: expand into multiple resource queries and
    # combine the results.  The ``name`` filter is incompatible with ``all``
    # because different resource types have unrelated names.
    if resource == "all":
        if name:
            return ToolResult(
                output="Cannot use 'name' filter with resource='all'. Specify a concrete resource type instead.",
                error=True,
            )
        return _kubectl_get_all(
            gke_config=gke_config,
            namespace=namespace,
            output=output,
            label_selector=label_selector,
            field_selector=field_selector,
        )

    if resource not in _resources._RESOURCE_API_MAP:
        if resource in _resources._KNOWN_K8S_RESOURCES:
            return ToolResult(
                output=(
                    f"Resource type '{resource}' is a valid Kubernetes resource but is not yet "
                    f"supported by this tool. Consider using kubectl directly for this resource."
                ),
                error=True,
            )
        supported = sorted(_resources._RESOURCE_API_MAP.keys())
        return ToolResult(
            output=f"Unsupported resource type: '{resource}'. Supported: {', '.join(supported)}",
            error=True,
        )

    if output not in ("table", "yaml", "json", "wide", "name"):
        return ToolResult(
            output=f"Invalid output: '{output}'. Must be one of: table, yaml, json, wide, name",
            error=True,
        )

    # Use the config's default namespace when caller doesn't specify
    ns = namespace or gke_config.default_namespace

    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, apps_v1, custom_api, api_client_inst = result

    try:
        api_result = _resources._list_resource(
            core_v1,
            apps_v1,
            custom_api,
            resource,
            namespace=ns,
            label_selector=label_selector,
            field_selector=field_selector,
            api_client=api_client_inst,
        )
        # _list_resource may return a ToolResult on unsupported resource
        if isinstance(api_result, ToolResult):
            return api_result

        items = api_result.items

        # If a specific name was requested, filter to that item
        if name:
            items = [i for i in items if i.metadata.name == name]
            if not items:
                return ToolResult(
                    output=f"{resource}/{name} not found in namespace '{ns}'",
                    error=True,
                )

        return ToolResult(output=_formatters._format_items(resource, items, output))

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(
                output=f"Namespace '{ns}' not found or resource type '{resource}' not available", error=True
            )
        if exc.status == 403:
            return ToolResult(
                output=f"Access denied: insufficient permissions to list {resource} in namespace '{ns}'", error=True
            )
        if exc.status == 401:
            return ToolResult(output="Authentication failed: check your kubeconfig or GKE credentials", error=True)
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:
        logger.exception("kubectl_get failed")
        return ToolResult(output=f"Error listing {resource}: {exc}", error=True)


def _kubectl_get_all(
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
    output: str = "table",
    label_selector: str | None = None,
    field_selector: str | None = None,
) -> ToolResult:
    """Expand ``resource='all'`` into per-type queries and combine results."""
    ns = namespace or gke_config.default_namespace
    sections: list[str] = []
    errors: list[str] = []
    any_success = False

    for rtype in _resources._ALL_RESOURCE_TYPES:
        sub = kubectl_get(
            rtype,
            gke_config=gke_config,
            namespace=ns,
            output=output,
            label_selector=label_selector,
            field_selector=field_selector,
        )
        if sub.error:
            errors.append(f"{rtype}: {sub.output}")
            continue
        any_success = True
        # Only include resource types that actually have items
        body = sub.output.strip() if sub.output else ""
        if body and body != "No resources found." and body != "[]":
            sections.append(f"=== {rtype.upper()} ===\n{body}")

    if not any_success and errors:
        return ToolResult(
            output="Failed to list resources:\n" + "\n".join(errors),
            error=True,
        )

    combined = "\n\n".join(sections)
    if errors:
        combined += "\n\n--- Errors ---\n" + "\n".join(errors)
    if not combined:
        combined = f"No resources found in namespace '{ns}'."
    return ToolResult(output=combined)


# ── _describe_resource ───────────────────────────────────────


def _describe_resource(
    core_v1: Any,
    apps_v1: Any,
    resource: str,
    name: str,
    namespace: str,
    api_client: Any | None = None,
    custom_api: Any | None = None,
) -> Any:
    """Read a single resource by name for describe output."""
    api_group = _resources._RESOURCE_API_MAP.get(resource, "core")
    is_cluster_scoped = resource in _resources._CLUSTER_SCOPED_RESOURCES

    # ── Core V1 ───────────────────────────────────────────────
    if api_group == "core":
        read_map: dict[str, tuple[str, str]] = {
            "pods": ("read_namespaced_pod", ""),
            "services": ("read_namespaced_service", ""),
            "configmaps": ("read_namespaced_config_map", ""),
            "secrets": ("read_namespaced_secret", ""),
            "serviceaccounts": ("read_namespaced_service_account", ""),
            "endpoints": ("read_namespaced_endpoints", ""),
            "pvc": ("read_namespaced_persistent_volume_claim", ""),
            "persistentvolumeclaims": ("read_namespaced_persistent_volume_claim", ""),
            "resourcequotas": ("read_namespaced_resource_quota", ""),
            "nodes": ("", "read_node"),
            "namespaces": ("", "read_namespace"),
            "pv": ("", "read_persistent_volume"),
            "persistentvolumes": ("", "read_persistent_volume"),
        }
        entry = read_map.get(resource)
        if not entry:
            return None
        ns_method, cluster_method = entry
        if is_cluster_scoped:
            return getattr(core_v1, cluster_method)(name=name)
        return getattr(core_v1, ns_method)(name=name, namespace=namespace)

    # ── Apps V1 ───────────────────────────────────────────────
    if api_group == "apps":
        read_map_apps: dict[str, str] = {
            "deployments": "read_namespaced_deployment",
            "statefulsets": "read_namespaced_stateful_set",
            "daemonsets": "read_namespaced_daemon_set",
            "replicasets": "read_namespaced_replica_set",
        }
        method_name = read_map_apps.get(resource)
        if not method_name:
            return None
        return getattr(apps_v1, method_name)(name=name, namespace=namespace)

    # ── Batch V1 ──────────────────────────────────────────────
    if api_group == "batch":
        from kubernetes.client import BatchV1Api  # noqa: WPS433

        batch_v1 = BatchV1Api(api_client=api_client)
        read_map_batch: dict[str, str] = {
            "jobs": "read_namespaced_job",
            "cronjobs": "read_namespaced_cron_job",
        }
        method_name_b = read_map_batch.get(resource)
        if not method_name_b:
            return None
        return getattr(batch_v1, method_name_b)(name=name, namespace=namespace)

    # ── Autoscaling ───────────────────────────────────────────
    if api_group == "autoscaling":
        from kubernetes.client import AutoscalingV2Api  # noqa: WPS433

        return AutoscalingV2Api(api_client=api_client).read_namespaced_horizontal_pod_autoscaler(
            name=name,
            namespace=namespace,
        )

    # ── Networking ────────────────────────────────────────────
    if api_group == "networking":
        from kubernetes.client import NetworkingV1Api  # noqa: WPS433

        net_v1 = NetworkingV1Api(api_client=api_client)
        read_map_net: dict[str, str] = {
            "ingress": "read_namespaced_ingress",
            "ingresses": "read_namespaced_ingress",
            "networkpolicies": "read_namespaced_network_policy",
        }
        method_name_n = read_map_net.get(resource)
        if not method_name_n:
            return None
        return getattr(net_v1, method_name_n)(name=name, namespace=namespace)

    # ── Policy V1 ────────────────────────────────────────────
    if api_group == "policy":
        from kubernetes.client import PolicyV1Api  # noqa: WPS433

        return PolicyV1Api(api_client=api_client).read_namespaced_pod_disruption_budget(
            name=name,
            namespace=namespace,
        )

    # ── AdmissionRegistration V1 ─────────────────────────────
    if api_group == "admissionregistration":
        from kubernetes.client import AdmissionregistrationV1Api  # noqa: WPS433

        admission_v1 = AdmissionregistrationV1Api(api_client=api_client)
        if resource == "mutatingwebhookconfigurations":
            return admission_v1.read_mutating_webhook_configuration(name=name)
        if resource == "validatingwebhookconfigurations":
            return admission_v1.read_validating_webhook_configuration(name=name)
        return None

    # ── ApiExtensions V1 ─────────────────────────────────────
    if api_group == "apiextensions":
        from kubernetes.client import ApiextensionsV1Api  # noqa: WPS433

        return ApiextensionsV1Api(api_client=api_client).read_custom_resource_definition(name=name)

    # ── External Secrets Operator (custom) ───────────────────
    if api_group == "custom_external_secrets":
        if custom_api is None:
            return None
        raw = custom_api.get_namespaced_custom_object(
            group="external-secrets.io",
            version="v1beta1",
            plural="externalsecrets",
            namespace=namespace,
            name=name,
        )
        return _resources._DictItem(raw)

    # ── Vertical Pod Autoscaler (custom) ─────────────────────
    if api_group == "custom_vpa":
        if custom_api is None:
            return None
        raw = custom_api.get_namespaced_custom_object(
            group="autoscaling.k8s.io",
            version="v1",
            plural="verticalpodautoscalers",
            namespace=namespace,
            name=name,
        )
        return _resources._DictItem(raw)

    return None


# ── _format_describe ─────────────────────────────────────────


def _format_describe(resource: str, obj: Any, api_client: Any | None = None) -> str:
    """Format a single K8s resource object into a kubectl-describe-style output."""
    # ── ExternalSecret (dict-based _DictItem wrapper) ─────────
    if isinstance(obj, _resources._DictItem):
        import yaml  # noqa: WPS433

        lines: list[str] = []
        meta = obj.metadata
        lines.append(f"Name:         {meta.name}")
        if meta.namespace:
            lines.append(f"Namespace:    {meta.namespace}")
        labels = meta.labels or {}
        lines.append(
            "Labels:       " + (", ".join(f"{k}={v}" for k, v in sorted(labels.items())) if labels else "<none>")
        )
        annotations = meta.annotations or {}
        lines.append(
            "Annotations:  "
            + (", ".join(f"{k}={v}" for k, v in sorted(annotations.items())) if annotations else "<none>")
        )
        lines.append(f"CreationTimestamp: {meta.creation_timestamp}")
        if obj.spec:
            lines.append("Spec:")
            lines.append("  " + yaml.dump(obj.spec, default_flow_style=False).replace("\n", "\n  ").rstrip())
        if obj.status:
            lines.append("Status:")
            lines.append("  " + yaml.dump(obj.status, default_flow_style=False).replace("\n", "\n  ").rstrip())
        lines.append("Events:       <not available for custom resources>")
        return "\n".join(lines)

    lines: list[str] = []  # type: ignore[no-redef]
    meta = obj.metadata

    lines.append(f"Name:         {meta.name}")
    if meta.namespace:
        lines.append(f"Namespace:    {meta.namespace}")

    # Labels
    labels = meta.labels or {}
    lines.append("Labels:       " + (", ".join(f"{k}={v}" for k, v in sorted(labels.items())) if labels else "<none>"))

    # Annotations
    annotations = meta.annotations or {}
    lines.append(
        "Annotations:  " + (", ".join(f"{k}={v}" for k, v in sorted(annotations.items())) if annotations else "<none>")
    )

    lines.append(f"CreationTimestamp: {meta.creation_timestamp}")

    # Resource-specific sections
    if resource == "pods" and obj.spec:
        lines.append(f"Node:         {obj.spec.node_name or '<none>'}")
        lines.append(f"Status:       {_formatters._pod_status(obj)}")
        if obj.status and obj.status.pod_ip:
            lines.append(f"IP:           {obj.status.pod_ip}")
        # Containers
        lines.append("Containers:")
        for c in obj.spec.containers or []:
            lines.append(f"  {c.name}:")
            lines.append(f"    Image:   {c.image}")
            if c.ports:
                ports = ", ".join(f"{p.container_port}/{p.protocol or 'TCP'}" for p in c.ports)
                lines.append(f"    Ports:   {ports}")
            if c.resources:
                if c.resources.requests:
                    lines.append(f"    Requests: {c.resources.requests}")
                if c.resources.limits:
                    lines.append(f"    Limits:   {c.resources.limits}")
        # Container statuses
        if obj.status and obj.status.container_statuses:
            lines.append("Container Statuses:")
            for cs in obj.status.container_statuses:
                lines.append(f"  {cs.name}:")
                lines.append(f"    Ready:    {cs.ready}")
                lines.append(f"    Restarts: {cs.restart_count}")
                if cs.state:
                    if cs.state.running:
                        lines.append(f"    State:    Running (since {cs.state.running.started_at})")
                    elif cs.state.waiting:
                        lines.append(f"    State:    Waiting ({cs.state.waiting.reason})")
                    elif cs.state.terminated:
                        lines.append(f"    State:    Terminated ({cs.state.terminated.reason})")

    elif resource == "deployments" and obj.spec:
        lines.append(f"Replicas:     {obj.spec.replicas} desired")
        if obj.status:
            lines.append(f"  Ready:      {obj.status.ready_replicas or 0}")
            lines.append(f"  Available:  {obj.status.available_replicas or 0}")
            lines.append(f"  Updated:    {obj.status.updated_replicas or 0}")
        if obj.spec.strategy:
            lines.append(f"Strategy:     {obj.spec.strategy.type}")

    elif resource == "services" and obj.spec:
        lines.append(f"Type:         {obj.spec.type}")
        lines.append(f"ClusterIP:    {obj.spec.cluster_ip}")
        if obj.spec.ports:
            for p in obj.spec.ports:
                port_info = f"{p.port}"
                if p.target_port:
                    port_info += f" → {p.target_port}"
                if p.node_port:
                    port_info += f" (NodePort: {p.node_port})"
                lines.append(f"  Port:       {p.name or ''} {port_info}/{p.protocol or 'TCP'}")
        if obj.spec.selector:
            lines.append(f"Selector:     {obj.spec.selector}")

    elif resource == "nodes" and obj.status:
        # Conditions
        lines.append("Conditions:")
        for cond in obj.status.conditions or []:
            lines.append(f"  {cond.type}: {cond.status} ({cond.reason or ''}) — {cond.message or ''}")
        # Allocatable
        if obj.status.allocatable:
            lines.append("Allocatable:")
            for k, v in sorted(obj.status.allocatable.items()):
                lines.append(f"  {k}: {v}")

    # Events — try to fetch events for the resource
    try:
        events_v1 = k8s_client.CoreV1Api(api_client=api_client) if api_client else k8s_client.CoreV1Api()
        field_sel = f"involvedObject.name={meta.name}"
        if meta.namespace:
            ev_list = events_v1.list_namespaced_event(
                namespace=meta.namespace,
                field_selector=field_sel,
            )
        else:
            ev_list = events_v1.list_event_for_all_namespaces(field_selector=field_sel)

        events = ev_list.items
        if events:
            lines.append("Events:")
            lines.append(f"  {'TYPE':<10}{'REASON':<25}{'AGE':<8}{'MESSAGE'}")
            for ev in events[-20:]:  # Last 20 events
                ev_type = ev.type or "Normal"
                reason = ev.reason or ""
                ev_age = _formatters._age(ev.last_timestamp or ev.metadata.creation_timestamp)
                message = ev.message or ""
                lines.append(f"  {ev_type:<10}{reason:<25}{ev_age:<8}{message}")
        else:
            lines.append("Events:       <none>")
    except Exception:  # noqa: BLE001
        logger.debug("Failed to retrieve events for %s/%s", resource, meta.name, exc_info=True)
        lines.append("Events:       <unable to retrieve>")

    return "\n".join(lines)


# ── kubectl_describe ─────────────────────────────────────────


def kubectl_describe(
    resource: str,
    name: str,
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
) -> ToolResult:
    """Describe a Kubernetes resource in detail (read-only kubectl describe equivalent).

    Returns detailed info including labels, annotations, spec, status, conditions,
    and recent events.
    """
    if not _clients._K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    resource = _resources._normalise_resource(resource)
    if resource not in _resources._DESCRIBE_SUPPORTED_RESOURCES:
        if resource in _resources._RESOURCE_API_MAP:
            return ToolResult(
                output=(
                    f"Resource type '{resource}' can be listed with kubectl_get but "
                    f"describe is not yet supported for this resource type."
                ),
                error=True,
            )
        if resource in _resources._KNOWN_K8S_RESOURCES:
            return ToolResult(
                output=(
                    f"Resource type '{resource}' is a valid Kubernetes resource but is not yet "
                    f"supported by this tool. Consider using kubectl directly for this resource."
                ),
                error=True,
            )
        supported = sorted(_resources._DESCRIBE_SUPPORTED_RESOURCES)
        return ToolResult(
            output=f"Unsupported resource type: '{resource}'. Supported: {', '.join(supported)}",
            error=True,
        )

    ns = namespace or gke_config.default_namespace

    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, apps_v1, custom_api, api_client_inst = result

    try:
        obj = _describe_resource(
            core_v1, apps_v1, resource, name, ns, api_client=api_client_inst, custom_api=custom_api
        )
        if isinstance(obj, ToolResult):
            return obj
        if obj is None:
            return ToolResult(output=f"Describe not supported for resource type: {resource}", error=True)
        return ToolResult(output=_format_describe(resource, obj, api_client=api_client_inst))

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(output=f"{resource}/{name} not found in namespace '{ns}'", error=True)
        if exc.status == 403:
            return ToolResult(output=f"Access denied: insufficient permissions to read {resource}/{name}", error=True)
        if exc.status == 401:
            return ToolResult(output="Authentication failed: check your kubeconfig or GKE credentials", error=True)
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:
        logger.exception("kubectl_describe failed")
        return ToolResult(output=f"Error describing {resource}/{name}: {exc}", error=True)


# ── _parse_since ─────────────────────────────────────────────


def _parse_since(since: str) -> int | None:
    """Parse a duration string like '1h', '30m', '2h30m' into total seconds."""
    pattern = re.compile(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$")
    match = pattern.match(since.strip())
    if not match or not any(match.groups()):
        return None
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


# ── kubectl_logs ─────────────────────────────────────────────


def kubectl_logs(
    pod: str,
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
    container: str | None = None,
    tail_lines: int = 100,
    since: str | None = None,
) -> ToolResult:
    """Retrieve logs from a pod (read-only kubectl logs equivalent).

    Handles CrashLoopBackOff by automatically fetching previous container logs
    when current logs are unavailable.
    """
    if not _clients._K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    ns = namespace or gke_config.default_namespace
    tail = min(tail_lines, gke_config.log_limit) if gke_config.log_limit else tail_lines

    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, _, _, _ = result

    kwargs: dict[str, Any] = {
        "name": pod,
        "namespace": ns,
        "tail_lines": tail,
    }
    if container:
        kwargs["container"] = container
    if since:
        seconds = _parse_since(since)
        if seconds is None:
            return ToolResult(
                output=f"Invalid 'since' format: '{since}'. Use formats like '1h', '30m', '1h30m', '90s'",
                error=True,
            )
        kwargs["since_seconds"] = seconds

    try:
        logs = core_v1.read_namespaced_pod_log(**kwargs)
        if not logs:
            return ToolResult(output=f"(no logs available for pod/{pod})")
        return ToolResult(output=logs)

    except k8s_exceptions.ApiException as exc:
        # If the pod has multiple containers, auto-detect the app container and retry
        if exc.status == 400 and "container name must be specified" in str(exc.body):
            match = re.search(r"\[([^\]]+)\]", str(exc.body))
            if match:
                containers = match.group(1).split()
                sidecar_prefixes = ("istio-", "datadog-", "linkerd-", "envoy-")
                sidecar_exact = frozenset({"envoy", "istio-proxy", "datadog-agent", "linkerd-proxy"})
                app_containers = [
                    c for c in containers
                    if c not in sidecar_exact
                    and not c.startswith(sidecar_prefixes)
                    and "-init-" not in c
                ]
                target = app_containers[0] if len(app_containers) == 1 else None
                if target:
                    kwargs["container"] = target
                    try:
                        logs = core_v1.read_namespaced_pod_log(**kwargs)
                        if not logs:
                            return ToolResult(
                                output=f"(no logs available for container '{target}' in pod/{pod})"
                            )
                        return ToolResult(output=f"[container: {target}]\n{logs}")
                    except k8s_exceptions.ApiException:
                        pass
                return ToolResult(
                    output=(
                        f"Pod {pod} has multiple containers: {', '.join(containers)}. "
                        f"Likely app containers: {', '.join(app_containers) if app_containers else 'unknown'}. "
                        f"Retry with container= parameter."
                    ),
                    error=True,
                )

        # If the pod is in CrashLoopBackOff, try previous logs
        if exc.status == 400 and "previous" not in kwargs:
            try:
                kwargs["previous"] = True
                prev_logs = core_v1.read_namespaced_pod_log(**kwargs)
                if prev_logs:
                    return ToolResult(
                        output=f"[Previous container logs — current container may be crashing]\n{prev_logs}",
                    )
                return ToolResult(output=f"No current or previous logs available for pod/{pod}")
            except k8s_exceptions.ApiException:
                logger.debug("Failed to retrieve previous logs for pod/%s", pod, exc_info=True)
                return ToolResult(output=f"No current or previous logs available for pod/{pod}", error=True)

        if exc.status == 404:
            msg = f"Pod '{pod}' not found in namespace '{ns}'"
            if container:
                msg = f"Container '{container}' not found in pod '{pod}' (namespace '{ns}')"
            return ToolResult(output=msg, error=True)
        if exc.status == 403:
            return ToolResult(output=f"Access denied: insufficient permissions to read logs for pod/{pod}", error=True)
        if exc.status == 401:
            return ToolResult(output="Authentication failed: check your kubeconfig or GKE credentials", error=True)
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:
        logger.exception("kubectl_logs failed")
        return ToolResult(output=f"Error reading logs for pod/{pod}: {exc}", error=True)


# ── kubectl_top ──────────────────────────────────────────────


def kubectl_top(
    resource_type: str = "pods",
    *,
    gke_config: GKEConfig,
    name: str | None = None,
    namespace: str = "default",
) -> ToolResult:
    """Show CPU and memory usage for pods or nodes (read-only kubectl top equivalent).

    For pods, returns per-container metrics — one row per container with a
    CONTAINER column.  To get pod-level totals, sum the container rows for
    each pod.  Requires the Metrics Server to be installed in the cluster.
    """
    if not _clients._K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    resource_type = resource_type.lower().strip()
    if resource_type not in ("pods", "pod", "nodes", "node"):
        return ToolResult(
            output=f"Invalid resource_type: '{resource_type}'. Must be 'pods' or 'nodes'",
            error=True,
        )
    is_pods = resource_type in ("pods", "pod")
    ns = namespace or gke_config.default_namespace

    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    _, _, custom_api, _ = result

    try:
        if is_pods:
            if ns in ("", "all"):
                metrics = custom_api.list_cluster_custom_object(
                    group="metrics.k8s.io",
                    version="v1beta1",
                    plural="pods",
                )
            else:
                metrics = custom_api.list_namespaced_custom_object(
                    group="metrics.k8s.io",
                    version="v1beta1",
                    namespace=ns,
                    plural="pods",
                )
        else:
            metrics = custom_api.list_cluster_custom_object(
                group="metrics.k8s.io",
                version="v1beta1",
                plural="nodes",
            )

        items = metrics.get("items", [])
        if name:
            items = [i for i in items if i.get("metadata", {}).get("name") == name]
            if not items:
                resource_label = "pod" if is_pods else "node"
                return ToolResult(
                    output=f"No metrics found for {resource_label}/{name}",
                    error=True,
                )

        if not items:
            return ToolResult(output="No metrics data available. Is metrics-server installed?")

        # Format output
        lines: list[str] = []
        if is_pods:
            lines.append(f"{'NAME':<50}{'CONTAINER':<30}{'CPU(cores)':<20}{'MEMORY'}")
            for item in items:
                pod_name = item.get("metadata", {}).get("name", "")
                containers = item.get("containers", [])
                for c in containers:
                    usage = c.get("usage", {})
                    cname = c.get("name", "")
                    cpu = _formatters._format_cpu(usage.get("cpu", "0"))
                    mem = _formatters._format_memory(usage.get("memory", "?"))
                    lines.append(
                        f"{pod_name:<50}{cname:<30}{cpu:<20}{mem}"
                    )
        else:
            lines.append(f"{'NAME':<50}{'CPU(cores)':<20}{'CPU%':<10}{'MEMORY':<17}{'MEMORY%'}")
            for item in items:
                node_name = item.get("metadata", {}).get("name", "")
                usage = item.get("usage", {})
                cpu = _formatters._format_cpu(usage.get("cpu", "0"))
                mem = _formatters._format_memory(usage.get("memory", "?"))
                lines.append(f"{node_name:<50}{cpu:<20}{'N/A':<10}{mem:<17}{'N/A'}")

        return ToolResult(output="\n".join(lines))

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(
                output="Metrics API not available. Is the metrics-server installed? "
                "Install with: kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml",
                error=True,
            )
        if exc.status in (401, 403):
            return ToolResult(output=f"Access denied to metrics API ({exc.status}): {exc.reason}.", error=True)
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:
        logger.exception("kubectl_top failed")
        return ToolResult(output=f"Error fetching metrics: {exc}", error=True)


# ── kubectl_get_labels ───────────────────────────────────────


def kubectl_get_labels(
    *,
    resource_type: str,
    gke_config: GKEConfig,
    namespace: str = "default",
    name: str = "",
    label_filter: str = "",
    annotation_filter: str = "",
) -> ToolResult:
    """List labels and annotations for Kubernetes resources.

    Supports server-side label filtering and client-side annotation filtering.
    Returns a formatted view of labels and annotations for matching resources.
    """
    if not _clients._K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    resource = _resources._normalise_resource(resource_type)
    if resource not in _resources._RESOURCE_API_MAP:
        if resource in _resources._KNOWN_K8S_RESOURCES:
            return ToolResult(
                output=(
                    f"Resource type '{resource}' is a valid Kubernetes resource but is not yet "
                    f"supported by this tool. Consider using kubectl directly for this resource."
                ),
                error=True,
            )
        supported = sorted(_resources._RESOURCE_API_MAP.keys())
        return ToolResult(
            output=f"Unsupported resource type: '{resource}'. Supported: {', '.join(supported)}",
            error=True,
        )

    ns = namespace or gke_config.default_namespace

    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, apps_v1, custom_api, api_client_inst = result

    try:
        api_result = _resources._list_resource(
            core_v1,
            apps_v1,
            custom_api,
            resource,
            namespace=ns,
            label_selector=label_filter or None,
            api_client=api_client_inst,
        )
        if isinstance(api_result, ToolResult):
            return api_result

        items = api_result.items

        # If a specific name was requested, filter to that item
        if name:
            items = [i for i in items if i.metadata.name == name]
            if not items:
                return ToolResult(
                    output=f"{resource}/{name} not found in namespace '{ns}'",
                    error=True,
                )

        # Client-side annotation filtering
        if annotation_filter:
            items = _filter_by_annotation(items, annotation_filter)

        if not items:
            qualifier = " matching filters" if label_filter or annotation_filter else ""
            return ToolResult(output=f"No {resource}{qualifier} found in namespace '{ns}'")

        # Format output
        lines: list[str] = []
        for item in items:
            meta = item.metadata
            item_ns = getattr(meta, "namespace", None) or ns
            lines.append(f"{resource}/{meta.name} (namespace: {item_ns})")

            labels = meta.labels or {}
            lines.append("  Labels:")
            if labels:
                for k, v in sorted(labels.items()):
                    lines.append(f"    {k}: {v}")
            else:
                lines.append("    <none>")

            annotations = meta.annotations or {}
            lines.append("  Annotations:")
            if annotations:
                for k, v in sorted(annotations.items()):
                    lines.append(f"    {k}: {v}")
            else:
                lines.append("    <none>")

        if not name:
            lines.append(f"\nTotal: {len(items)} {resource}")

        return ToolResult(output="\n".join(lines))

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(
                output=f"Namespace '{ns}' not found or resource type '{resource}' not available", error=True
            )
        if exc.status == 403:
            return ToolResult(
                output=f"Access denied: insufficient permissions to list {resource} in namespace '{ns}'", error=True
            )
        if exc.status == 401:
            return ToolResult(output="Authentication failed: check your kubeconfig or GKE credentials", error=True)
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:
        logger.exception("kubectl_get_labels failed")
        return ToolResult(output=f"Error listing labels for {resource}: {exc}", error=True)


def _filter_by_annotation(items: list[Any], annotation_filter: str) -> list[Any]:
    """Filter items by annotation key=value or key presence (client-side)."""
    filtered = []
    for item in items:
        annotations = item.metadata.annotations or {}
        if "=" in annotation_filter:
            key, value = annotation_filter.split("=", 1)
            if annotations.get(key) == value:
                filtered.append(item)
        else:
            # Key-only filter: check if the annotation key exists
            if annotation_filter in annotations:
                filtered.append(item)
    return filtered


# ── Task 3.4 — async wrappers ───────────────────────────────
# Offload blocking kubernetes-client calls to a thread pool via to_async.

from vaig.core.async_utils import to_async  # noqa: E402

async_kubectl_get = to_async(kubectl_get)
async_kubectl_describe = to_async(kubectl_describe)
async_kubectl_logs = to_async(kubectl_logs)
async_kubectl_top = to_async(kubectl_top)
async_kubectl_get_labels = to_async(kubectl_get_labels)

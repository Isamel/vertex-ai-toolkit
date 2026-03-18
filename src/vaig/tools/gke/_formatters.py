"""Pure formatting helpers — kubectl-style table formatters for K8s resources."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


def _age(creation_timestamp: datetime | None) -> str:
    """Return a human-readable age string from a creation timestamp."""
    if creation_timestamp is None:
        return "<unknown>"
    now = datetime.now(UTC)
    delta = now - creation_timestamp.replace(tzinfo=UTC) if creation_timestamp.tzinfo is None else now - creation_timestamp
    total_seconds = int(delta.total_seconds())
    if total_seconds < 0:
        return "0s"
    if total_seconds < 60:
        return f"{total_seconds}s"
    if total_seconds < 3600:
        return f"{total_seconds // 60}m"
    if total_seconds < 86400:
        return f"{total_seconds // 3600}h"
    return f"{total_seconds // 86400}d"


def _pod_status(pod: Any) -> str:
    """Derive a human-readable status string for a pod (mirrors kubectl)."""
    if pod.metadata.deletion_timestamp:
        return "Terminating"
    if pod.status is None:
        return "Unknown"
    if pod.status.phase in ("Succeeded", "Failed"):
        return str(pod.status.phase)
    # Check container statuses for waiting reasons
    for cs in pod.status.container_statuses or []:
        if cs.state and cs.state.waiting and cs.state.waiting.reason:
            return str(cs.state.waiting.reason)
        if cs.state and cs.state.terminated and cs.state.terminated.reason:
            return str(cs.state.terminated.reason)
    return str(pod.status.phase or "Unknown")


def _pod_restarts(pod: Any) -> int:
    """Total restart count across all containers in a pod."""
    total = 0
    for cs in (pod.status.container_statuses or []) if pod.status else []:
        total += cs.restart_count or 0
    return total


def _pod_ready_count(pod: Any) -> str:
    """Return READY column value like '1/1' or '0/2'."""
    containers = pod.spec.containers or []
    total = len(containers)
    ready = 0
    for cs in (pod.status.container_statuses or []) if pod.status else []:
        if cs.ready:
            ready += 1
    return f"{ready}/{total}"


# ── Table formatters per resource type ────────────────────────


def _format_pods_table(items: list[Any], wide: bool = False) -> str:
    """Format pod list as a kubectl-style table."""
    if not items:
        return "No resources found."
    lines: list[str] = []
    header = "NAME                                     READY   STATUS             RESTARTS   AGE"
    if wide:
        header += "   NODE                          IP"
    lines.append(header)
    for pod in items:
        name = pod.metadata.name or ""
        ready = _pod_ready_count(pod)
        status = _pod_status(pod)
        restarts = str(_pod_restarts(pod))
        age = _age(pod.metadata.creation_timestamp)
        line = f"{name:<41}{ready:<8}{status:<19}{restarts:<11}{age}"
        if wide:
            node = (pod.spec.node_name or "") if pod.spec else ""
            ip = (pod.status.pod_ip or "") if pod.status else ""
            line += f"   {node:<30}{ip}"
        lines.append(line)
    return "\n".join(lines)


def _format_deployments_table(items: list[Any], wide: bool = False) -> str:
    if not items:
        return "No resources found."
    lines: list[str] = []
    header = "NAME                                     READY   UP-TO-DATE   AVAILABLE   AGE"
    if wide:
        header += "   CONTAINERS        IMAGES"
    lines.append(header)
    for dep in items:
        name = dep.metadata.name or ""
        desired = dep.spec.replicas if dep.spec and dep.spec.replicas is not None else 0
        ready = dep.status.ready_replicas or 0 if dep.status else 0
        up_to_date = dep.status.updated_replicas or 0 if dep.status else 0
        available = dep.status.available_replicas or 0 if dep.status else 0
        age = _age(dep.metadata.creation_timestamp)
        line = f"{name:<41}{ready}/{desired:<6}{up_to_date:<13}{available:<12}{age}"
        if wide and dep.spec and dep.spec.template and dep.spec.template.spec:
            containers = [c.name for c in dep.spec.template.spec.containers or []]
            images = [c.image for c in dep.spec.template.spec.containers or []]
            line += f"   {','.join(containers):<17}{','.join(images)}"
        lines.append(line)
    return "\n".join(lines)


def _format_services_table(items: list[Any], wide: bool = False) -> str:
    if not items:
        return "No resources found."
    lines: list[str] = []
    header = "NAME                                     TYPE           CLUSTER-IP       EXTERNAL-IP      PORT(S)            AGE"
    if wide:
        header += "   SELECTOR"
    lines.append(header)
    for svc in items:
        name = svc.metadata.name or ""
        svc_type = svc.spec.type or "ClusterIP" if svc.spec else "ClusterIP"
        cluster_ip = svc.spec.cluster_ip or "<none>" if svc.spec else "<none>"
        # External IPs
        ext_ips: list[str] = []
        if svc.status and svc.status.load_balancer and svc.status.load_balancer.ingress:
            for ing in svc.status.load_balancer.ingress:
                ext_ips.append(ing.ip or ing.hostname or "")
        external_ip = ",".join(ext_ips) if ext_ips else "<none>"
        # Ports
        ports: list[str] = []
        for p in (svc.spec.ports or []) if svc.spec else []:
            port_str = f"{p.port}"
            if p.node_port:
                port_str += f":{p.node_port}"
            port_str += f"/{p.protocol or 'TCP'}"
            ports.append(port_str)
        ports_str = ",".join(ports) if ports else "<none>"
        age = _age(svc.metadata.creation_timestamp)
        line = f"{name:<41}{svc_type:<15}{cluster_ip:<17}{external_ip:<17}{ports_str:<19}{age}"
        if wide and svc.spec and svc.spec.selector:
            sel = ",".join(f"{k}={v}" for k, v in svc.spec.selector.items())
            line += f"   {sel}"
        lines.append(line)
    return "\n".join(lines)


def _format_nodes_table(items: list[Any], wide: bool = False) -> str:
    if not items:
        return "No resources found."
    lines: list[str] = []
    header = "NAME                                     STATUS   ROLES    AGE    VERSION"
    if wide:
        header += "   INTERNAL-IP       OS-IMAGE"
    lines.append(header)
    for node in items:
        name = node.metadata.name or ""
        # Status from conditions
        status = "Unknown"
        for cond in (node.status.conditions or []) if node.status else []:
            if cond.type == "Ready":
                status = "Ready" if cond.status == "True" else "NotReady"
                break
        roles_set: list[str] = []
        for label_key in (node.metadata.labels or {}):
            if label_key.startswith("node-role.kubernetes.io/"):
                roles_set.append(label_key.split("/")[-1])
        roles = ",".join(roles_set) if roles_set else "<none>"
        age = _age(node.metadata.creation_timestamp)
        version = node.status.node_info.kubelet_version if node.status and node.status.node_info else ""
        line = f"{name:<41}{status:<9}{roles:<9}{age:<7}{version}"
        if wide and node.status:
            int_ip = ""
            for addr in node.status.addresses or []:
                if addr.type == "InternalIP":
                    int_ip = addr.address
                    break
            os_image = node.status.node_info.os_image if node.status.node_info else ""
            line += f"   {int_ip:<17}{os_image}"
        lines.append(line)
    return "\n".join(lines)


def _format_generic_table(items: list[Any]) -> str:
    """Fallback table formatter for resource types without a custom formatter."""
    if not items:
        return "No resources found."
    lines: list[str] = []
    lines.append("NAME                                     NAMESPACE        AGE")
    for item in items:
        name = item.metadata.name or ""
        ns = item.metadata.namespace or ""
        age = _age(item.metadata.creation_timestamp)
        lines.append(f"{name:<41}{ns:<17}{age}")
    return "\n".join(lines)


def _format_webhook_config(item: Any) -> str:
    """Format a MutatingWebhookConfiguration or ValidatingWebhookConfiguration."""
    lines = [f"Name: {item.metadata.name}"]
    for webhook in (item.webhooks or []):
        lines.append(f"  Webhook: {webhook.name}")
        if webhook.namespace_selector:
            lines.append(f"    NamespaceSelector: {webhook.namespace_selector}")
        if webhook.object_selector:
            lines.append(f"    ObjectSelector: {webhook.object_selector}")
        rules = webhook.rules or []
        for rule in rules:
            resources = rule.resources or ["*"]
            operations = rule.operations or ["*"]
            lines.append(f"    Rules: {', '.join(operations)} on {', '.join(resources)}")
        if webhook.failure_policy:
            lines.append(f"    FailurePolicy: {webhook.failure_policy}")
    return "\n".join(lines)


def _format_webhooks_table(items: list[Any], wide: bool = False) -> str:
    """Format webhook configuration list as a kubectl-style table."""
    if not items:
        return "No resources found."
    lines: list[str] = []
    lines.append("NAME                                     WEBHOOKS   AGE")
    for item in items:
        name = item.metadata.name or ""
        webhook_count = len(item.webhooks or [])
        age = _age(item.metadata.creation_timestamp)
        line = f"{name:<41}{webhook_count:<11}{age}"
        if wide:
            line += "\n" + _format_webhook_config(item)
        lines.append(line)
    return "\n".join(lines)


def _format_crds_table(items: list[Any], wide: bool = False) -> str:
    """Format CustomResourceDefinition list as a kubectl-style table."""
    if not items:
        return "No resources found."
    lines: list[str] = []
    header = "NAME                                                         CREATED AT"
    if wide:
        header += "   GROUP                         SCOPE"
    lines.append(header)
    for item in items:
        name = item.metadata.name or ""
        age = _age(item.metadata.creation_timestamp)
        line = f"{name:<61}{age}"
        if wide and item.spec:
            group = item.spec.group or ""
            scope = item.spec.scope or ""
            line += f"   {group:<30}{scope}"
        lines.append(line)
    return "\n".join(lines)


def _format_external_secrets_table(items: list[Any], wide: bool = False) -> str:
    """Format ExternalSecret list as a kubectl-style table."""
    if not items:
        return "No resources found."
    lines: list[str] = []
    header = "NAME                                     STORE              STATUS             AGE"
    if wide:
        header += "   REFRESH-INTERVAL"
    lines.append(header)
    for item in items:
        meta = item.metadata
        name = meta.name or ""
        spec = item.spec or {}
        store_ref = spec.get("secretStoreRef", {})
        store = store_ref.get("name", "<unknown>")
        status_dict = item.status or {}
        conditions = status_dict.get("conditions", [])
        sync_status = "<unknown>"
        for cond in conditions:
            if isinstance(cond, dict) and cond.get("type") == "Ready":
                sync_status = "Ready" if cond.get("status") == "True" else "NotReady"
                break
        age = _age(meta.creation_timestamp) if meta.creation_timestamp else "<unknown>"
        line = f"{name:<41}{store:<19}{sync_status:<19}{age}"
        if wide:
            refresh = spec.get("refreshInterval", "")
            line += f"   {refresh}"
        lines.append(line)
    return "\n".join(lines)


_SECRET_REDACTED = "[REDACTED]"


def _redact_secret_item(item: dict[str, Any]) -> dict[str, Any]:
    """Redact ``data`` and ``stringData`` fields from a single K8s Secret dict.

    Replaces every value inside ``data`` / ``stringData`` with ``[REDACTED]``
    and appends a ``_redacted_note`` field with the count of redacted keys.

    The original dict is **not** mutated — a shallow copy is returned with the
    affected fields replaced.
    """
    redacted = dict(item)  # shallow copy

    for field in ("data", "stringData"):
        section = redacted.get(field)
        if not isinstance(section, dict) or not section:
            continue
        key_count = len(section)
        redacted[field] = dict.fromkeys(section, _SECRET_REDACTED)
        # Add a note so the LLM knows keys were stripped
        note_key = f"_{field}_redacted_note"
        redacted[note_key] = f"{key_count} key(s) redacted for security"

    return redacted


def _redact_k8s_secret_data(
    serialised: list[Any],
) -> list[Any]:
    """Redact secret data from a list of serialised K8s resource dicts.

    Each item that looks like a Secret (has ``kind == 'Secret'`` or contains
    a ``data`` / ``stringData`` dict) gets its values replaced with
    ``[REDACTED]``.  Items that are *not* Secrets pass through unchanged.

    This is intentionally safe — if anything unexpected happens we return
    the input unmodified rather than crashing.
    """
    try:
        return [_redact_secret_item(item) for item in serialised]
    except Exception:
        logger.warning("Failed to redact secret data — returning unredacted output")
        return serialised


def _serialise_item(item: Any, api: Any) -> Any:
    """Serialise a single K8s item to a plain dict.

    For dict-backed custom resource wrappers (``_DictItem``), the underlying
    raw dict is returned directly — ``ApiClient.sanitize_for_serialization``
    does not reliably unwrap these objects.  For all other items the standard
    ``sanitize_for_serialization`` path is used.
    """
    # Import here to avoid a circular dependency; _resources is a sibling module.
    from vaig.tools.gke._resources import _DictItem  # noqa: WPS433

    if isinstance(item, _DictItem):
        # _DictItem._d holds the raw dict returned by CustomObjectsApi — use it directly.
        return dict(item._d)
    return api.sanitize_for_serialization(item)


def _format_items(resource: str, items: list[Any], output_format: str) -> str:
    """Format a list of K8s items into the requested output_format."""
    import json as _json

    from kubernetes import client as k8s_client  # noqa: WPS433

    is_secret = resource == "secrets"

    if output_format == "json":
        # Use a single ApiClient for serialisation (not one per item)
        api = k8s_client.ApiClient()
        serialised = [_serialise_item(i, api) for i in items]
        if is_secret:
            serialised = _redact_k8s_secret_data(serialised)
        return _json.dumps(serialised, indent=2, default=str)

    if output_format == "yaml":
        try:
            import yaml as _yaml  # noqa: WPS433
        except ImportError:
            return "PyYAML is not installed. Use output_format='json' instead."
        api = k8s_client.ApiClient()
        serialised = [_serialise_item(i, api) for i in items]
        if is_secret:
            serialised = _redact_k8s_secret_data(serialised)
        return str(_yaml.dump_all(serialised, default_flow_style=False))

    # Table or wide
    wide = output_format == "wide"
    formatter = {
        "pods": _format_pods_table,
        "deployments": _format_deployments_table,
        "services": _format_services_table,
        "nodes": _format_nodes_table,
        "mutatingwebhookconfigurations": _format_webhooks_table,
        "validatingwebhookconfigurations": _format_webhooks_table,
        "customresourcedefinitions": _format_crds_table,
        "crds": _format_crds_table,
        "externalsecrets": _format_external_secrets_table,
        "externalsecret": _format_external_secrets_table,
    }.get(resource)

    if formatter:
        return formatter(items, wide=wide)
    return _format_generic_table(items)


def _format_memory(mem_str: str) -> str:
    """Convert Kubernetes memory strings (e.g. '16384Ki') to human-readable."""
    if mem_str == "?" or not mem_str:
        return "?"
    # Handle Ki (kibibytes)
    if mem_str.endswith("Ki"):
        try:
            ki = int(mem_str[:-2])
            gi = ki / (1024 * 1024)
            if gi >= 1:
                return f"{gi:.1f}Gi"
            mi = ki / 1024
            return f"{mi:.0f}Mi"
        except ValueError:
            return mem_str
    # Handle Mi
    if mem_str.endswith("Mi"):
        try:
            mi = int(mem_str[:-2])
            gi = mi / 1024
            if gi >= 1:
                return f"{gi:.1f}Gi"
            return f"{mi}Mi"
        except ValueError:
            return mem_str
    # Handle Gi
    if mem_str.endswith("Gi"):
        return mem_str
    # Handle plain bytes
    try:
        b = int(mem_str)
        gi = b / (1024**3)
        if gi >= 1:
            return f"{gi:.1f}Gi"
        mi = b / (1024**2)
        return f"{mi:.0f}Mi"
    except ValueError:
        return mem_str

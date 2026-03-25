"""Diagnostic tools — events, node conditions, rollout status/history."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from vaig.tools.base import ToolResult

from . import _clients, _formatters

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


# ── get_events ───────────────────────────────────────────────


def get_events(
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
    event_type: str | None = None,
    involved_object_name: str | None = None,
    involved_object_kind: str | None = None,
    limit: int = 50,
) -> ToolResult:
    """List Kubernetes events in a namespace, optionally filtered by type and involved object.

    Events reveal WHY pods fail, WHY nodes have issues, and what the scheduler
    is doing. Critical for SRE triage. Equivalent to
    ``kubectl get events --sort-by=.lastTimestamp``.
    """
    if not _clients._K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    if event_type is not None and event_type not in ("Warning", "Normal"):
        return ToolResult(
            output=f"Invalid event_type: '{event_type}'. Must be 'Warning', 'Normal', or omit for all.",
            error=True,
        )

    if limit < 1 or limit > 500:
        return ToolResult(
            output=f"Limit must be between 1 and 500. Got: {limit}",
            error=True,
        )

    ns = namespace or gke_config.default_namespace

    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, _, _, _ = result

    try:
        # Build field selector for server-side filtering
        field_parts: list[str] = []
        if event_type:
            field_parts.append(f"type={event_type}")
        if involved_object_name:
            field_parts.append(f"involvedObject.name={involved_object_name}")
        if involved_object_kind:
            field_parts.append(f"involvedObject.kind={involved_object_kind}")
        field_selector = ",".join(field_parts) if field_parts else None

        kwargs: dict[str, Any] = {"namespace": ns}
        if field_selector:
            kwargs["field_selector"] = field_selector

        ev_list = core_v1.list_namespaced_event(**kwargs)
        events = ev_list.items or []

        # Sort by last_timestamp descending (most recent first)
        def _sort_key(ev: Any) -> datetime:
            ts = ev.last_timestamp or ev.metadata.creation_timestamp
            if ts is None:
                return datetime.min.replace(tzinfo=UTC)
            ts_aware = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts
            return ts_aware  # type: ignore[no-any-return]  # K8s API returns datetime via Any

        events.sort(key=_sort_key, reverse=True)

        # Apply limit
        events = events[:limit]

        if not events:
            filter_desc = ""
            if event_type:
                filter_desc += f" type={event_type}"
            if involved_object_name:
                filter_desc += f" object={involved_object_name}"
            return ToolResult(
                output=f"No events found in namespace '{ns}'.{' Filters:' + filter_desc if filter_desc else ''}",
            )

        # Format as table
        lines: list[str] = []
        lines.append(f"{'LAST SEEN':<12}{'TYPE':<10}{'REASON':<25}{'OBJECT':<40}{'MESSAGE'}")
        for ev in events:
            last_seen = _formatters._age(ev.last_timestamp or ev.metadata.creation_timestamp)
            ev_type_str = ev.type or "Normal"
            reason = ev.reason or ""
            # Build OBJECT column: Kind/Name
            obj_kind = ev.involved_object.kind if ev.involved_object else ""
            obj_name = ev.involved_object.name if ev.involved_object else ""
            obj_str = f"{obj_kind}/{obj_name}" if obj_kind else obj_name
            message = ev.message or ""
            lines.append(f"{last_seen:<12}{ev_type_str:<10}{reason:<25}{obj_str:<40}{message}")

        header = f"Events in namespace '{ns}' ({len(events)} shown):"
        return ToolResult(output=f"{header}\n" + "\n".join(lines))

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(output=f"Namespace '{ns}' not found", error=True)
        if exc.status == 403:
            return ToolResult(output=f"Access denied: insufficient permissions to list events in namespace '{ns}'", error=True)
        if exc.status == 401:
            return ToolResult(output="Authentication failed: check your kubeconfig or GKE credentials", error=True)
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:
        logger.exception("get_events failed")
        return ToolResult(output=f"Error listing events: {exc}", error=True)


# ── get_rollout_status ───────────────────────────────────────


def get_rollout_status(
    name: str,
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
) -> ToolResult:
    """Check the rollout status of a Kubernetes deployment.

    Shows whether a deployment is progressing, complete, stalled, or failed.
    Reports replica counts, conditions, and rollout strategy.
    Equivalent to ``kubectl rollout status deployment/<name>``.
    """
    if not _clients._K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    ns = namespace or gke_config.default_namespace

    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    _, apps_v1, _, _ = result

    try:
        dep = apps_v1.read_namespaced_deployment(name=name, namespace=ns)

        lines: list[str] = []
        lines.append(f"Deployment: {name}")
        lines.append(f"Namespace:  {ns}")

        # ── Replica counts ────────────────────────────────────
        # When HPA manages the deployment, K8s sets spec.replicas=None.
        # In that case use status.replicas as the effective desired count so
        # we do NOT wrongly report the deployment as "Scaled to zero".
        # Only spec.replicas == 0 (explicit) means intentionally scaled down.
        spec_replicas = dep.spec.replicas if dep.spec else None
        status = dep.status
        current = status.replicas or 0 if status else 0
        if spec_replicas is not None:
            desired = spec_replicas
        else:
            # HPA-managed: fall back to current running count
            desired = current
        ready = status.ready_replicas or 0 if status else 0
        updated = status.updated_replicas or 0 if status else 0
        available = status.available_replicas or 0 if status else 0
        unavailable = status.unavailable_replicas or 0 if status else 0

        lines.append("")
        lines.append("Replicas:")
        lines.append(f"  Desired:     {desired}")
        lines.append(f"  Current:     {current}")
        lines.append(f"  Ready:       {ready}")
        lines.append(f"  Updated:     {updated}")
        lines.append(f"  Available:   {available}")
        lines.append(f"  Unavailable: {unavailable}")

        # ── Rollout strategy ──────────────────────────────────
        if dep.spec and dep.spec.strategy:
            strategy = dep.spec.strategy
            lines.append("")
            lines.append(f"Strategy: {strategy.type or 'RollingUpdate'}")
            if strategy.rolling_update:
                ru = strategy.rolling_update
                lines.append(f"  Max Unavailable: {ru.max_unavailable}")
                lines.append(f"  Max Surge:       {ru.max_surge}")

        # ── Conditions ────────────────────────────────────────
        conditions = (status.conditions or []) if status else []
        overall_state = "Unknown"
        condition_details: list[str] = []

        progressing_cond = None
        available_cond = None
        failure_cond = None

        for cond in conditions:
            cond_type = cond.type or ""
            cond_status = cond.status or "Unknown"
            reason = cond.reason or ""
            message = cond.message or ""
            condition_details.append(f"  {cond_type}: {cond_status} — {reason}: {message}")

            if cond_type == "Progressing":
                progressing_cond = cond
            elif cond_type == "Available":
                available_cond = cond
            elif cond_type == "ReplicaFailure":
                failure_cond = cond

        # Determine overall state
        if failure_cond and failure_cond.status == "True":
            overall_state = "Failed"
        elif progressing_cond and progressing_cond.reason == "ProgressDeadlineExceeded":
            overall_state = "Stalled"
        elif (
            available_cond
            and available_cond.status == "True"
            and updated == desired
            and ready == desired
        ):
            overall_state = "Complete"
        elif progressing_cond and progressing_cond.status == "True":
            overall_state = "Progressing"
        elif spec_replicas == 0 and ready == 0:
            overall_state = "Scaled to zero"

        lines.append("")
        lines.append(f"Overall Status: {overall_state}")

        if condition_details:
            lines.append("")
            lines.append("Conditions:")
            lines.extend(condition_details)

        return ToolResult(output="\n".join(lines))

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(output=f"Deployment '{name}' not found in namespace '{ns}'", error=True)
        if exc.status == 403:
            return ToolResult(output=f"Access denied: insufficient permissions to read deployment/{name}", error=True)
        if exc.status == 401:
            return ToolResult(output="Authentication failed: check your kubeconfig or GKE credentials", error=True)
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:
        logger.exception("get_rollout_status failed")
        return ToolResult(output=f"Error checking rollout status for deployment/{name}: {exc}", error=True)


# ── get_node_conditions ──────────────────────────────────────


def get_node_conditions(
    *,
    gke_config: GKEConfig,
    name: str | None = None,
) -> ToolResult:
    """Show node health conditions, resource pressure, taints, and capacity.

    When called WITHOUT a node name, lists ALL nodes with a summary: name,
    status (Ready/NotReady), roles, age, version, OS, kernel, container
    runtime, CPU capacity/allocatable, and memory capacity/allocatable.

    When called WITH a specific node name, shows a detailed view including
    ALL conditions (Ready, MemoryPressure, DiskPressure, PIDPressure,
    NetworkUnavailable), taints, relevant labels, allocatable vs capacity
    comparison, and the unschedulable (cordon) flag.

    Fills the gap left by ``kubectl_get nodes`` which only shows Ready/NotReady
    but hides pressure conditions that indicate imminent node failures.
    """
    if not _clients._K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, _, _, _ = result

    try:
        if name:
            # ── Single node detail ────────────────────────────
            node = core_v1.read_node(name=name)
            return _format_node_detail(node)

        # ── All nodes summary ─────────────────────────────
        node_list = core_v1.list_node()
        nodes = node_list.items or []

        if not nodes:
            return ToolResult(output="No nodes found in the cluster.")

        return _format_nodes_summary(nodes)

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(output=f"Node '{name}' not found", error=True)
        if exc.status == 403:
            return ToolResult(
                output="Access denied: insufficient permissions to read nodes.",
                error=True,
            )
        if exc.status == 401:
            return ToolResult(
                output="Authentication failed: check your kubeconfig or GKE credentials",
                error=True,
            )
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:
        logger.exception("get_node_conditions failed")
        return ToolResult(output=f"Error reading node conditions: {exc}", error=True)


# ── _format_nodes_summary ────────────────────────────────────


def _format_nodes_summary(nodes: list[Any]) -> ToolResult:
    """Format a list of nodes into a summary table with capacity info."""
    lines: list[str] = []
    header = (
        f"{'NAME':<40}{'STATUS':<12}{'ROLES':<12}{'AGE':<7}{'VERSION':<15}"
        f"{'OS':<12}{'KERNEL':<25}{'RUNTIME':<20}"
        f"{'CPU(cap/alloc)':<17}{'MEM(cap/alloc)'}"
    )
    lines.append(header)

    for node in nodes:
        nd_name = node.metadata.name or ""

        # Status from conditions
        status = "Unknown"
        for cond in (node.status.conditions or []) if node.status else []:
            if cond.type == "Ready":
                status = "Ready" if cond.status == "True" else "NotReady"
                break

        # Roles from labels
        roles_set: list[str] = []
        for label_key in node.metadata.labels or {}:
            if label_key.startswith("node-role.kubernetes.io/"):
                roles_set.append(label_key.split("/")[-1])
        roles = ",".join(roles_set) if roles_set else "<none>"

        age = _formatters._age(node.metadata.creation_timestamp)

        # Node info
        node_info = node.status.node_info if node.status and node.status.node_info else None
        version = node_info.kubelet_version if node_info else ""
        os_image = node_info.os_image if node_info else ""
        kernel = node_info.kernel_version if node_info else ""
        runtime = node_info.container_runtime_version if node_info else ""

        # Capacity / Allocatable
        capacity = node.status.capacity or {} if node.status else {}
        allocatable = node.status.allocatable or {} if node.status else {}
        cpu_cap = capacity.get("cpu", "?")
        cpu_alloc = allocatable.get("cpu", "?")
        mem_cap = _formatters._format_memory(capacity.get("memory", "?"))
        mem_alloc = _formatters._format_memory(allocatable.get("memory", "?"))

        cpu_str = f"{cpu_cap}/{cpu_alloc}"
        mem_str = f"{mem_cap}/{mem_alloc}"

        lines.append(
            f"{nd_name:<40}{status:<12}{roles:<12}{age:<7}{version:<15}"
            f"{os_image:<12}{kernel:<25}{runtime:<20}"
            f"{cpu_str:<17}{mem_str}"
        )

    return ToolResult(output=f"Nodes ({len(nodes)}):\n" + "\n".join(lines))


# ── _format_node_detail ──────────────────────────────────────


def _format_node_detail(node: Any) -> ToolResult:
    """Format a single node with full detail: conditions, taints, labels, capacity."""
    lines: list[str] = []
    nd_name = node.metadata.name or ""
    lines.append(f"Node: {nd_name}")

    # ── Basic info ────────────────────────────────────────
    node_info = node.status.node_info if node.status and node.status.node_info else None
    if node_info:
        lines.append(f"  Kubelet Version:         {node_info.kubelet_version}")
        lines.append(f"  OS Image:                {node_info.os_image}")
        lines.append(f"  Kernel Version:          {node_info.kernel_version}")
        lines.append(f"  Container Runtime:       {node_info.container_runtime_version}")
        lines.append(f"  Architecture:            {node_info.architecture}")
        lines.append(f"  Operating System:        {node_info.operating_system}")

    age = _formatters._age(node.metadata.creation_timestamp)
    lines.append(f"  Age:                     {age}")

    # ── Addresses ─────────────────────────────────────────
    addresses = node.status.addresses or [] if node.status else []
    if addresses:
        lines.append("")
        lines.append("Addresses:")
        for addr in addresses:
            lines.append(f"  {addr.type}: {addr.address}")

    # ── Unschedulable (cordon) ────────────────────────────
    unschedulable = node.spec.unschedulable if node.spec else False
    lines.append("")
    lines.append(f"Unschedulable (cordoned): {unschedulable or False}")

    # ── Conditions (ALL of them) ──────────────────────────
    conditions = (node.status.conditions or []) if node.status else []
    lines.append("")
    lines.append("Conditions:")
    if not conditions:
        lines.append("  (none)")
    else:
        lines.append(f"  {'TYPE':<25}{'STATUS':<10}{'REASON':<30}{'LAST TRANSITION':<22}{'MESSAGE'}")
        for cond in conditions:
            cond_type = cond.type or ""
            cond_status = cond.status or "Unknown"
            reason = cond.reason or ""
            message = cond.message or ""
            last_transition = _formatters._age(cond.last_transition_time) if cond.last_transition_time else "<unknown>"
            lines.append(
                f"  {cond_type:<25}{cond_status:<10}{reason:<30}{last_transition:<22}{message}"
            )

    # ── Taints ────────────────────────────────────────────
    taints = node.spec.taints or [] if node.spec else []
    lines.append("")
    lines.append("Taints:")
    if not taints:
        lines.append("  (none)")
    else:
        for taint in taints:
            taint_key = taint.key or ""
            taint_value = taint.value or ""
            taint_effect = taint.effect or ""
            val_str = f"={taint_value}" if taint_value else ""
            lines.append(f"  {taint_key}{val_str}:{taint_effect}")

    # ── Labels (relevant subset) ──────────────────────────
    labels = node.metadata.labels or {}
    relevant_prefixes = (
        "node-role.kubernetes.io/",
        "topology.kubernetes.io/",
        "cloud.google.com/",
        "node.kubernetes.io/",
        "beta.kubernetes.io/",
    )
    relevant_keys = (
        "kubernetes.io/arch",
        "kubernetes.io/os",
        "kubernetes.io/hostname",
    )
    lines.append("")
    lines.append("Labels (relevant):")
    found_labels = False
    for k, v in sorted(labels.items()):
        if k.startswith(relevant_prefixes) or k in relevant_keys:
            lines.append(f"  {k}={v}")
            found_labels = True
    if not found_labels:
        lines.append("  (none matching filter)")

    # ── Capacity vs Allocatable ───────────────────────────
    capacity = node.status.capacity or {} if node.status else {}
    allocatable = node.status.allocatable or {} if node.status else {}
    lines.append("")
    lines.append("Capacity vs Allocatable:")
    lines.append(f"  {'RESOURCE':<25}{'CAPACITY':<20}{'ALLOCATABLE'}")
    # Show common resources
    resource_keys = sorted(set(list(capacity.keys()) + list(allocatable.keys())))
    for rk in resource_keys:
        cap_val = capacity.get(rk, "-")
        alloc_val = allocatable.get(rk, "-")
        # Human-readable memory
        if "memory" in rk.lower() or rk == "hugepages-2Mi" or rk == "hugepages-1Gi":
            cap_val = _formatters._format_memory(str(cap_val))
            alloc_val = _formatters._format_memory(str(alloc_val))
        lines.append(f"  {rk:<25}{cap_val:<20}{alloc_val}")

    return ToolResult(output="\n".join(lines))


# ── get_rollout_history ──────────────────────────────────────


def get_rollout_history(
    name: str,
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
    revision: int | None = None,
) -> ToolResult:
    """Show the revision history of a Kubernetes deployment.

    Lists all revisions by examining ReplicaSets owned by the deployment,
    similar to ``kubectl rollout history deployment/<name>``.
    When a specific revision number is provided, shows detailed pod template
    information for that revision (containers, images, ports, env var names,
    volume mounts, resource requests/limits).

    Use BEFORE recommending a rollback so you know what changed in each revision.
    """
    if not _clients._K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    ns = namespace or gke_config.default_namespace

    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    _, apps_v1, _, _ = result

    try:
        # Verify the deployment exists first
        dep = apps_v1.read_namespaced_deployment(name=name, namespace=ns)

        # List all ReplicaSets in the namespace
        all_rs = apps_v1.list_namespaced_replica_set(namespace=ns)

        # Filter to ReplicaSets owned by this deployment
        owned_rs: list[tuple[int, Any]] = []
        for rs in all_rs.items:
            for owner in (rs.metadata.owner_references or []):
                if owner.kind == "Deployment" and owner.name == name:
                    rev_str = (rs.metadata.annotations or {}).get(
                        "deployment.kubernetes.io/revision", ""
                    )
                    if rev_str:
                        owned_rs.append((int(rev_str), rs))
                    break

        if not owned_rs:
            return ToolResult(
                output=f"Deployment '{name}' in namespace '{ns}' has no revision history (no ReplicaSets found).",
                error=False,
            )

        # Sort by revision number descending (newest first)
        owned_rs.sort(key=lambda pair: pair[0], reverse=True)

        # ── Specific revision detail ──────────────────────
        if revision is not None:
            match = [rs for rev, rs in owned_rs if rev == revision]
            if not match:
                available_revs = ", ".join(str(rev) for rev, _ in owned_rs)
                return ToolResult(
                    output=(
                        f"Revision {revision} not found for deployment '{name}' "
                        f"in namespace '{ns}'. Available revisions: {available_revs}"
                    ),
                    error=True,
                )
            rs = match[0]
            return _format_revision_detail(name, ns, revision, rs)

        # ── List all revisions ────────────────────────────
        # Determine which revision is active (highest replica count > 0)
        current_rev = _find_current_revision(dep, owned_rs)

        lines: list[str] = []
        lines.append(f"=== Rollout History: {name} (namespace: {ns}) ===")
        lines.append("")
        lines.append(f"{'REVISION':<10} {'IMAGE(S)':<50} {'CREATED':<12} {'REPLICAS':<10} {'STATUS'}")
        lines.append("-" * 100)

        for rev_num, rs in owned_rs:
            # Images
            containers: list[Any] = []
            if rs.spec and rs.spec.template and rs.spec.template.spec:
                containers = rs.spec.template.spec.containers or []
            images = ", ".join(c.image or "<none>" for c in containers) if containers else "<none>"
            if len(images) > 48:
                images = images[:45] + "..."

            # Created
            created = _formatters._age(rs.metadata.creation_timestamp) if rs.metadata.creation_timestamp else "<unknown>"

            # Replicas
            replicas = rs.status.replicas if rs.status and rs.status.replicas else 0

            # Status
            if rev_num == current_rev:
                status = "active"
            elif replicas == 0:
                status = "scaled-down"
            else:
                status = f"{replicas} replicas"

            lines.append(f"{rev_num:<10} {images:<50} {created:<12} {replicas:<10} {status}")

        change_cause = (dep.metadata.annotations or {}).get("kubernetes.io/change-cause", "")
        if change_cause:
            lines.append("")
            lines.append(f"Last change cause: {change_cause}")

        lines.append("")
        lines.append(f"Total revisions: {len(owned_rs)}")
        lines.append("Tip: Use get_rollout_history with a specific revision number for detailed info.")

        return ToolResult(output="\n".join(lines), error=False)

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(output=f"Deployment '{name}' not found in namespace '{ns}'", error=True)
        if exc.status == 403:
            return ToolResult(
                output=f"Access denied: insufficient permissions to read deployment/{name} or its ReplicaSets",
                error=True,
            )
        if exc.status == 401:
            return ToolResult(output="Authentication failed: check your kubeconfig or GKE credentials", error=True)
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_rollout_history failed")
        return ToolResult(output=f"Error retrieving rollout history for deployment/{name}: {exc}", error=True)


# ── _find_current_revision ───────────────────────────────────


def _find_current_revision(dep: Any, owned_rs: list[tuple[int, Any]]) -> int | None:
    """Find the revision number of the currently active ReplicaSet."""
    # The deployment's own annotation tracks the current revision
    rev_str = (dep.metadata.annotations or {}).get("deployment.kubernetes.io/revision", "")
    if rev_str:
        try:
            return int(rev_str)
        except (ValueError, TypeError):
            pass
    # Fallback: highest revision with replicas > 0
    for rev_num, rs in owned_rs:
        replicas = rs.status.replicas if rs.status and rs.status.replicas else 0
        if replicas > 0:
            return rev_num
    return None


# ── _format_revision_detail ──────────────────────────────────


def _format_revision_detail(name: str, ns: str, revision: int, rs: Any) -> ToolResult:
    """Format detailed information for a specific revision."""
    lines: list[str] = []
    lines.append(f"=== Revision {revision} Detail: {name} (namespace: {ns}) ===")
    lines.append("")

    # ReplicaSet info
    rs_name = rs.metadata.name if rs.metadata else "<unknown>"
    lines.append(f"ReplicaSet: {rs_name}")
    created = _formatters._age(rs.metadata.creation_timestamp) if rs.metadata and rs.metadata.creation_timestamp else "<unknown>"
    lines.append(f"Created: {created} ago")

    # Replica counts
    replicas = rs.status.replicas if rs.status and rs.status.replicas else 0
    ready = rs.status.ready_replicas if rs.status and rs.status.ready_replicas else 0
    lines.append(f"Replicas: {replicas} current / {ready} ready")

    # Change cause annotation
    change_cause = (rs.metadata.annotations or {}).get("kubernetes.io/change-cause", "")
    if change_cause:
        lines.append(f"Change Cause: {change_cause}")

    # Pod template containers
    containers: list[Any] = []
    if rs.spec and rs.spec.template and rs.spec.template.spec:
        containers = rs.spec.template.spec.containers or []

    if not containers:
        lines.append("")
        lines.append("No container spec found.")
        return ToolResult(output="\n".join(lines), error=False)

    lines.append("")
    lines.append("Containers:")
    for c in containers:
        c_name = c.name if hasattr(c, "name") else "<unnamed>"
        lines.append(f"  Container: {c_name}")
        lines.append(f"    Image: {c.image or '<none>'}")

        # Ports
        ports = c.ports or [] if hasattr(c, "ports") else []
        if ports:
            port_strs = []
            for p in ports:
                proto = p.protocol or "TCP" if hasattr(p, "protocol") else "TCP"
                port_strs.append(f"{p.container_port}/{proto}")
            lines.append(f"    Ports: {', '.join(port_strs)}")

        # Resource requests/limits
        resources = c.resources if hasattr(c, "resources") and c.resources else None
        if resources:
            requests = resources.requests or {} if hasattr(resources, "requests") else {}
            limits = resources.limits or {} if hasattr(resources, "limits") else {}
            if requests or limits:
                lines.append("    Resources:")
                if requests:
                    parts = [f"{k}={v}" for k, v in requests.items()]
                    lines.append(f"      Requests: {', '.join(parts)}")
                if limits:
                    parts = [f"{k}={v}" for k, v in limits.items()]
                    lines.append(f"      Limits:   {', '.join(parts)}")

        # Env vars — names only, NO secret values
        env_vars = c.env or [] if hasattr(c, "env") else []
        if env_vars:
            lines.append("    Environment Variables:")
            for ev in env_vars:
                ev_name = ev.name if hasattr(ev, "name") else "<unnamed>"
                if hasattr(ev, "value_from") and ev.value_from:
                    if hasattr(ev.value_from, "config_map_key_ref") and ev.value_from.config_map_key_ref:
                        ref = ev.value_from.config_map_key_ref
                        lines.append(f"      {ev_name} <- ConfigMap:{ref.name}/{ref.key}")
                    elif hasattr(ev.value_from, "secret_key_ref") and ev.value_from.secret_key_ref:
                        ref = ev.value_from.secret_key_ref
                        lines.append(f"      {ev_name} <- Secret:{ref.name}/{ref.key} (ref only)")
                    else:
                        lines.append(f"      {ev_name} (valueFrom)")
                else:
                    lines.append(f"      {ev_name} = <value set>")

        # Env from (configMapRef / secretRef — names only)
        env_from = c.env_from or [] if hasattr(c, "env_from") else []
        if env_from:
            lines.append("    Env From:")
            for ef in env_from:
                if hasattr(ef, "config_map_ref") and ef.config_map_ref:
                    lines.append(f"      ConfigMap: {ef.config_map_ref.name}")
                if hasattr(ef, "secret_ref") and ef.secret_ref:
                    lines.append(f"      Secret: {ef.secret_ref.name} (ref only, no values shown)")

        # Volume mounts
        mounts = c.volume_mounts or [] if hasattr(c, "volume_mounts") else []
        if mounts:
            lines.append("    Volume Mounts:")
            for m in mounts:
                ro = " (ro)" if hasattr(m, "read_only") and m.read_only else ""
                lines.append(f"      {m.mount_path} from {m.name}{ro}")
    return ToolResult(output="\n".join(lines), error=False)


# ── get_container_status ─────────────────────────────────────


def get_container_status(
    name: str,
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
) -> ToolResult:
    """Show detailed container-level status for ALL containers in a pod.

    Covers init containers, regular containers, and ephemeral containers.
    For each container shows: name, image, state (Waiting/Running/Terminated
    with details), ready flag, restart count, last termination state (crucial
    for CrashLoopBackOff debugging), resource requests/limits, volume mounts,
    and environment variable sources (configMapRef/secretRef names only — no
    secret values exposed).

    Essential for debugging multi-container pods where ``kubectl_get pods``
    only shows pod-level status.
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    ns = namespace or gke_config.default_namespace

    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, _, _, _ = result

    try:
        pod = core_v1.read_namespaced_pod(name=name, namespace=ns)

        lines: list[str] = []
        lines.append(f"Pod: {name}")
        lines.append(f"Namespace: {ns}")
        lines.append(f"Node: {pod.spec.node_name or '<unassigned>'}")
        lines.append(f"Phase: {pod.status.phase or 'Unknown' if pod.status else 'Unknown'}")
        lines.append("")

        # Build status maps for quick lookup
        container_statuses = {cs.name: cs for cs in (pod.status.container_statuses or []) if pod.status} if pod.status else {}
        init_statuses = {cs.name: cs for cs in (pod.status.init_container_statuses or []) if pod.status} if pod.status else {}
        ephemeral_statuses = {cs.name: cs for cs in (pod.status.ephemeral_container_statuses or []) if pod.status} if pod.status else {}

        # ── Init Containers ───────────────────────────────
        init_containers = pod.spec.init_containers or [] if pod.spec else []
        if init_containers:
            lines.append("=== Init Containers ===")
            for c in init_containers:
                cs = init_statuses.get(c.name)
                _format_container_section(c, cs, lines)
                lines.append("")

        # ── Regular Containers ────────────────────────────
        containers = pod.spec.containers or [] if pod.spec else []
        if containers:
            lines.append("=== Containers ===")
            for c in containers:
                cs = container_statuses.get(c.name)
                _format_container_section(c, cs, lines)
                lines.append("")

        # ── Ephemeral Containers ──────────────────────────
        ephemeral_containers = pod.spec.ephemeral_containers or [] if pod.spec else []
        if ephemeral_containers:
            lines.append("=== Ephemeral Containers ===")
            for c in ephemeral_containers:
                cs = ephemeral_statuses.get(c.name)
                _format_container_section(c, cs, lines)
                lines.append("")

        if not init_containers and not containers and not ephemeral_containers:
            lines.append("No containers found in pod spec.")

        return ToolResult(output="\n".join(lines))

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            return ToolResult(output=f"Pod '{name}' not found in namespace '{ns}'", error=True)
        if exc.status == 403:
            return ToolResult(
                output=f"Access denied: insufficient permissions to read pod/{name}",
                error=True,
            )
        if exc.status == 401:
            return ToolResult(
                output="Authentication failed: check your kubeconfig or GKE credentials",
                error=True,
            )
        return ToolResult(output=f"Kubernetes API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:
        logger.exception("get_container_status failed")
        return ToolResult(output=f"Error reading container status for pod/{name}: {exc}", error=True)


def _format_container_section(container: Any, status: Any | None, lines: list[str]) -> None:
    """Format a single container's detail into output lines."""
    c_name = container.name or ""
    lines.append(f"  Container: {c_name}")
    lines.append(f"    Image: {container.image or '<none>'}")

    if status:
        lines.append(f"    Image ID: {status.image_id or '<none>'}")
        lines.append(f"    Ready: {status.ready if status.ready is not None else 'N/A'}")
        lines.append(f"    Restart Count: {status.restart_count or 0}")

        # Current state
        state = status.state
        if state:
            if state.running:
                started = _formatters._age(state.running.started_at) if state.running.started_at else "<unknown>"
                lines.append(f"    State: Running (started {started} ago)")
            elif state.waiting:
                reason = state.waiting.reason or "Unknown"
                message = state.waiting.message or ""
                lines.append(f"    State: Waiting — {reason}")
                if message:
                    lines.append(f"      Message: {message}")
            elif state.terminated:
                t = state.terminated
                reason = t.reason or "Unknown"
                exit_code = t.exit_code if t.exit_code is not None else "?"
                lines.append(f"    State: Terminated — {reason} (exit code {exit_code})")
                if t.message:
                    lines.append(f"      Message: {t.message}")
                if t.started_at:
                    lines.append(f"      Started:  {t.started_at}")
                if t.finished_at:
                    lines.append(f"      Finished: {t.finished_at}")
            else:
                lines.append("    State: Unknown")

        # Last termination state (crucial for CrashLoopBackOff)
        last_state = status.last_state
        if last_state and last_state.terminated:
            lt = last_state.terminated
            reason = lt.reason or "Unknown"
            exit_code = lt.exit_code if lt.exit_code is not None else "?"
            lines.append(f"    Last State: Terminated — {reason} (exit code {exit_code})")
            if lt.message:
                lines.append(f"      Message: {lt.message}")
            if lt.started_at:
                lines.append(f"      Started:  {lt.started_at}")
            if lt.finished_at:
                lines.append(f"      Finished: {lt.finished_at}")
    else:
        lines.append("    (no status available)")

    # ── Resource requests/limits ──────────────────────────
    resources = container.resources
    if resources:
        requests = resources.requests or {}
        limits = resources.limits or {}
        if requests or limits:
            lines.append("    Resources:")
            if requests:
                parts = [f"{k}={v}" for k, v in requests.items()]
                lines.append(f"      Requests: {', '.join(parts)}")
            if limits:
                parts = [f"{k}={v}" for k, v in limits.items()]
                lines.append(f"      Limits:   {', '.join(parts)}")

    # ── Volume mounts ─────────────────────────────────────
    mounts = container.volume_mounts or []
    if mounts:
        lines.append("    Volume Mounts:")
        for m in mounts:
            ro = " (ro)" if m.read_only else ""
            lines.append(f"      {m.mount_path} from {m.name}{ro}")

    # ── Env from (configMapRef / secretRef — names only) ──
    env_from = container.env_from or []
    if env_from:
        lines.append("    Env From:")
        for ef in env_from:
            if ef.config_map_ref:
                lines.append(f"      ConfigMap: {ef.config_map_ref.name}")
            if ef.secret_ref:
                lines.append(f"      Secret: {ef.secret_ref.name} (ref only, no values shown)")

    # ── Env vars with valueFrom (names only) ──────────────
    env_vars = container.env or []
    env_refs: list[str] = []
    for ev in env_vars:
        if ev.value_from:
            if ev.value_from.config_map_key_ref:
                ref = ev.value_from.config_map_key_ref
                env_refs.append(f"      {ev.name} ← ConfigMap:{ref.name}/{ref.key}")
            elif ev.value_from.secret_key_ref:
                ref = ev.value_from.secret_key_ref
                env_refs.append(f"      {ev.name} ← Secret:{ref.name}/{ref.key} (ref only)")
    if env_refs:
        lines.append("    Env Var References:")
        lines.extend(env_refs)


# ── Task 3.4 — async wrappers ───────────────────────────────
# Offload blocking kubernetes-client calls to a thread pool via to_async.

from vaig.core.async_utils import to_async  # noqa: E402

async_get_events = to_async(get_events)
async_get_rollout_status = to_async(get_rollout_status)
async_get_node_conditions = to_async(get_node_conditions)
async_get_container_status = to_async(get_container_status)
async_get_rollout_history = to_async(get_rollout_history)


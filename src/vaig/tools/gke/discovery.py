"""Auto-discovery tools — workloads, service mesh, network topology."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vaig.tools.base import ToolResult

from . import _cache, _clients, _formatters, _resources

if TYPE_CHECKING:
    from vaig.core.config import GKEConfig

logger = logging.getLogger(__name__)

# ── Lazy import guard (mirrors _clients.py) ──────────────────
# Needed locally for except clauses.
_K8S_AVAILABLE = True
try:
    from kubernetes.client import exceptions as k8s_exceptions  # noqa: WPS433, F401
except ImportError:
    _K8S_AVAILABLE = False


# ── discover_workloads ───────────────────────────────────────


def discover_workloads(
    *,
    gke_config: GKEConfig,
    namespace: str = "",
    include_jobs: bool = False,
    force_refresh: bool = False,
) -> ToolResult:
    """Discover all workloads in the cluster (or a namespace).

    Returns a summary table of deployments, statefulsets, daemonsets, and
    optionally jobs/cronjobs.  Unhealthy workloads are listed first.
    Results are cached for ``_DISCOVERY_TTL`` seconds unless *force_refresh*
    is ``True``.
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    # ── Cache check ───────────────────────────────────────────
    cache_key = _cache._cache_key_discovery("workloads", namespace, str(include_jobs))
    if not force_refresh:
        cached = _cache._get_cached(cache_key)
        if cached is not None:
            return ToolResult(output=cached, error=False)

    # ── Create clients ────────────────────────────────────────
    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, apps_v1, custom_api, api_client = result

    ns_label = namespace if namespace else "all namespaces"
    sections: list[str] = [f"=== Workload Discovery ({ns_label}) ===\n"]
    errors: list[str] = []
    all_items: list[tuple[str, str, str, int, int, int, str, bool]] = []
    # Each item: (kind, name, namespace, ready, desired, restarts, age, healthy)

    is_autopilot = _clients.detect_autopilot(gke_config)

    # ── Resource types to scan ────────────────────────────────
    resource_types: list[str] = ["deployments", "statefulsets", "daemonsets"]
    if include_jobs:
        resource_types.extend(["jobs", "cronjobs"])

    for rtype in resource_types:
        try:
            res = _resources._list_resource(core_v1, apps_v1, custom_api, rtype, namespace, api_client=api_client)
            if isinstance(res, ToolResult):
                errors.append(f"{rtype}: {res.output}")
                continue
            items = getattr(res, "items", []) or []
            for item in items:
                meta = item.metadata
                name = meta.name or "<unknown>"
                ns = meta.namespace or ""
                age = _formatters._age(meta.creation_timestamp)

                ready, desired, restarts = 0, 0, 0
                healthy = True

                if rtype == "deployments":
                    spec_replicas = item.spec.replicas if item.spec and item.spec.replicas is not None else 0
                    status_ready = item.status.ready_replicas if item.status and item.status.ready_replicas is not None else 0
                    desired = spec_replicas
                    ready = status_ready
                    healthy = ready >= desired and desired > 0
                    # Restarts from conditions (unavailable replicas)
                    if item.status and item.status.unavailable_replicas:
                        restarts = item.status.unavailable_replicas
                elif rtype == "statefulsets":
                    spec_replicas = item.spec.replicas if item.spec and item.spec.replicas is not None else 0
                    status_ready = item.status.ready_replicas if item.status and item.status.ready_replicas is not None else 0
                    desired = spec_replicas
                    ready = status_ready
                    healthy = ready >= desired and desired > 0
                elif rtype == "daemonsets":
                    desired_scheduled = item.status.desired_number_scheduled if item.status else 0
                    number_ready = item.status.number_ready if item.status else 0
                    desired = desired_scheduled or 0
                    ready = number_ready or 0
                    healthy = ready >= desired and desired > 0
                    # Flag GKE-managed daemonsets on Autopilot
                    if is_autopilot and ns == "kube-system":
                        name = f"{name} [GKE-managed]"
                elif rtype == "jobs":
                    succeeded = item.status.succeeded if item.status and item.status.succeeded else 0
                    failed = item.status.failed if item.status and item.status.failed else 0
                    desired = 1
                    ready = succeeded
                    healthy = succeeded > 0 and failed == 0
                    restarts = failed
                elif rtype == "cronjobs":
                    active = len(item.status.active) if item.status and item.status.active else 0
                    desired = 0  # CronJobs don't have a "desired" concept
                    ready = active
                    healthy = True  # CronJobs are always "healthy" unless suspended
                    if item.spec and item.spec.suspend:
                        name = f"{name} [suspended]"

                all_items.append((rtype, name, ns, ready, desired, restarts, age, healthy))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{rtype}: {exc}")

    # ── Sort: unhealthy first, then by kind + name ────────────
    all_items.sort(key=lambda x: (x[7], x[0], x[1]))

    # ── Truncate at 200 items ─────────────────────────────────
    truncated = False
    if len(all_items) > 200:
        all_items = all_items[:200]
        truncated = True

    # ── Format output ─────────────────────────────────────────
    if not all_items and not errors:
        sections.append("No workloads found.")
    else:
        # Header
        sections.append(f"{'STATUS':<8} {'KIND':<14} {'NAMESPACE':<20} {'NAME':<40} {'READY':<10} {'RESTARTS':<10} {'AGE':<8}")
        sections.append("-" * 110)

        for kind, name, ns, ready, desired, restarts, age, healthy in all_items:
            status = "OK" if healthy else "WARN"
            ready_str = f"{ready}/{desired}" if kind != "cronjobs" else str(ready)
            sections.append(
                f"{status:<8} {kind:<14} {ns:<20} {name:<40} {ready_str:<10} {restarts:<10} {age:<8}"
            )

    if truncated:
        sections.append("\n... truncated to 200 items (total: more than 200)")

    if errors:
        sections.append("\n--- Partial errors ---")
        for err in errors:
            sections.append(f"  {err}")

    # ── Summary line ──────────────────────────────────────────
    total = len(all_items)
    unhealthy = sum(1 for x in all_items if not x[7])
    sections.append(f"\nTotal: {total} workloads, {unhealthy} unhealthy")

    output = "\n".join(sections)
    _cache._set_cache(cache_key, output)
    return ToolResult(output=output, error=bool(errors and not all_items))


# ── discover_service_mesh ────────────────────────────────────


def discover_service_mesh(
    *,
    gke_config: GKEConfig,
    namespace: str = "",
    force_refresh: bool = False,
) -> ToolResult:
    """Discover service mesh installations (Istio, Linkerd, Consul).

    Detects meshes via three strategies: control-plane namespace existence,
    CRD inspection, and sidecar proxy detection on pods.  Results are cached
    for ``_DISCOVERY_TTL`` seconds unless *force_refresh* is ``True``.
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    # ── Cache check ───────────────────────────────────────────
    cache_key = _cache._cache_key_discovery("mesh", namespace)
    if not force_refresh:
        cached = _cache._get_cached(cache_key)
        if cached is not None:
            return ToolResult(output=cached, error=False)

    # ── Create clients ────────────────────────────────────────
    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, apps_v1, custom_api, api_client = result

    ns_label = namespace if namespace else "all"
    sections: list[str] = [
        "=== Service Mesh Discovery ===",
        f"Namespace filter: {ns_label} | Cached: no",
        "",
    ]
    warnings: list[str] = []

    # Known mesh definitions: (display_name, control_namespace, crd_group, sidecar_container)
    _MESHES = [
        ("Istio", "istio-system", "istio.io", "istio-proxy"),
        ("Linkerd", "linkerd", "linkerd.io", "linkerd-proxy"),
        ("Consul", "consul", "consul.hashicorp.com", "consul-dataplane"),
    ]
    # Extra sidecar name not tied to a specific mesh above
    _EXTRA_SIDECARS = {"envoy-sidecar"}

    # ── 1. Detect control-plane namespaces ────────────────────
    existing_namespaces: set[str] = set()
    try:
        ns_list = core_v1.list_namespace()
        existing_namespaces = {ns_obj.metadata.name for ns_obj in (ns_list.items or [])}
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Could not list namespaces: {exc}")

    # ── 2. Detect CRDs ───────────────────────────────────────
    crd_counts: dict[str, int] = {}  # mesh_name → count
    crd_check_skipped = False
    try:
        from kubernetes.client import ApiextensionsV1Api  # noqa: WPS433

        ext_api = ApiextensionsV1Api(api_client=api_client)
        crds = ext_api.list_custom_resource_definition()
        for crd_item in crds.items or []:
            crd_group = getattr(crd_item.spec, "group", "") or ""
            for mesh_name, _, mesh_group, _ in _MESHES:
                if mesh_group in crd_group:
                    crd_counts[mesh_name] = crd_counts.get(mesh_name, 0) + 1
    except Exception as exc:  # noqa: BLE001
        crd_check_skipped = True
        warnings.append(f"CRD check skipped (RBAC or API error): {exc}")

    # ── 3. Sidecar detection on pods ─────────────────────────
    sidecar_names = {sc for _, _, _, sc in _MESHES} | _EXTRA_SIDECARS
    pods_with_sidecar: dict[str, int] = {}  # sidecar_container → count
    total_pods = 0
    try:
        if namespace:
            pod_list = core_v1.list_namespaced_pod(namespace=namespace)
        else:
            pod_list = core_v1.list_pod_for_all_namespaces()
        for pod in pod_list.items or []:
            total_pods += 1
            containers = pod.spec.containers or [] if pod.spec else []
            for ctr in containers:
                ctr_name = ctr.name or ""
                if ctr_name in sidecar_names:
                    pods_with_sidecar[ctr_name] = pods_with_sidecar.get(ctr_name, 0) + 1
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Pod sidecar scan failed: {exc}")

    # ── 4. Per-mesh report ────────────────────────────────────
    detected_any = False
    not_detected: list[str] = []

    for mesh_name, ctrl_ns, _, sidecar_ctr in _MESHES:
        has_ns = ctrl_ns in existing_namespaces
        has_crds = crd_counts.get(mesh_name, 0) > 0
        has_sidecars = pods_with_sidecar.get(sidecar_ctr, 0) > 0

        if not (has_ns or has_crds or has_sidecars):
            not_detected.append(mesh_name)
            continue

        detected_any = True
        sections.append(f"--- DETECTED: {mesh_name} ---")

        # Control plane status
        if has_ns:
            # List pods in control namespace for health
            ctrl_pods: list[str] = []
            version_tag: str | None = None
            try:
                ctrl_pod_list = core_v1.list_namespaced_pod(namespace=ctrl_ns)
                for cpod in ctrl_pod_list.items or []:
                    cpod_name = cpod.metadata.name or "<unknown>"
                    cpod_phase = cpod.status.phase if cpod.status else "Unknown"
                    # Extract ready count
                    c_statuses = cpod.status.container_statuses or [] if cpod.status else []
                    c_ready = sum(1 for cs in c_statuses if cs.ready)
                    c_total = len(c_statuses)
                    ctrl_pods.append(f"    {cpod_name} ({c_ready}/{c_total} ready, {cpod_phase})")

                    # Try to extract version from image tag (first match wins)
                    if version_tag is None:
                        for cs in c_statuses:
                            img = cs.image or ""
                            if ":" in img:
                                tag = img.rsplit(":", 1)[-1]
                                if tag and tag[0].isdigit():
                                    version_tag = tag
                                    break
            except Exception as exc:  # noqa: BLE001
                ctrl_pods.append(f"    (could not list pods: {exc})")

            sections.append(f"  Control plane: {ctrl_ns} ({len(ctrl_pods)} pod(s))")
            if ctrl_pods:
                sections.append("  Components:")
                sections.extend(ctrl_pods)
            if version_tag is not None:
                sections.append(f"  Version: {version_tag} (from control-plane image tag)")
        else:
            sections.append(f"  Control plane: {ctrl_ns} (namespace not found)")

        # CRDs
        if has_crds:
            sections.append(f"  CRDs: {crd_counts[mesh_name]} {mesh_name} CRDs found")
        elif crd_check_skipped:
            sections.append("  CRDs: check skipped (insufficient permissions)")
        else:
            sections.append("  CRDs: none found")

        # Sidecar injection
        sc_count = pods_with_sidecar.get(sidecar_ctr, 0)
        sections.append(
            f"  Sidecar injection: {sc_count}/{total_pods} pods have {sidecar_ctr} sidecar"
        )
        sections.append("")

    # Extra envoy-sidecar detection (not tied to a named mesh)
    envoy_count = pods_with_sidecar.get("envoy-sidecar", 0)
    if envoy_count > 0 and not detected_any:
        detected_any = True
        sections.append("--- DETECTED: Envoy (standalone) ---")
        sections.append(f"  Sidecar injection: {envoy_count}/{total_pods} pods have envoy-sidecar")
        sections.append("")

    if not_detected:
        sections.append(f"--- NO MESH DETECTED: {', '.join(not_detected)} ---")
        sections.append("")

    if not detected_any and not not_detected:
        sections.append("No service mesh detected.")
        sections.append("")

    # Warnings
    if warnings:
        sections.append("Warnings:")
        for w in warnings:
            sections.append(f"  {w}")
    else:
        sections.append("Warnings: none")

    output = "\n".join(sections)
    _cache._set_cache(cache_key, output)
    return ToolResult(output=output, error=False)


# ── discover_network_topology ────────────────────────────────


def discover_network_topology(
    *,
    gke_config: GKEConfig,
    namespace: str = "",
    force_refresh: bool = False,
) -> ToolResult:
    """Discover network topology: services, endpoints, ingresses, network policies.

    Returns a plain-text report grouped by resource type.  Results are cached
    for ``_DISCOVERY_TTL`` seconds unless *force_refresh* is ``True``.
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    # ── Cache check ───────────────────────────────────────────
    cache_key = _cache._cache_key_discovery("network", namespace)
    if not force_refresh:
        cached = _cache._get_cached(cache_key)
        if cached is not None:
            return ToolResult(output=cached, error=False)

    # ── Create clients ────────────────────────────────────────
    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, apps_v1, custom_api, api_client = result

    ns_label = namespace if namespace else "all"
    sections: list[str] = [
        "=== Network Topology Discovery ===",
        f"Namespace: {ns_label} | Cached: no",
        "",
    ]
    errors: list[str] = []
    _MAX_ITEMS = 200

    svc_count = 0
    ep_count = 0
    ing_count = 0
    pol_count = 0

    # ── 1. Services ───────────────────────────────────────────
    try:
        if namespace:
            svc_list = core_v1.list_namespaced_service(namespace=namespace)
        else:
            svc_list = core_v1.list_service_for_all_namespaces()
        services = (svc_list.items or [])[:_MAX_ITEMS]
        svc_count = len(services)

        # Group by type
        by_type: dict[str, list[str]] = {}
        for svc in services:
            meta = svc.metadata
            spec = svc.spec
            svc_name = meta.name or "<unknown>"
            svc_ns = meta.namespace or ""
            svc_type = spec.type if spec else "ClusterIP"
            # Ports
            ports_str = ""
            if spec and spec.ports:
                port_parts = []
                for p in spec.ports:
                    port_val = p.port or 0
                    target = p.target_port if p.target_port else port_val
                    port_parts.append(f"{port_val}\u2192{target}")
                ports_str = ", ".join(port_parts)
            # Selector
            selector_str = ""
            if spec and spec.selector:
                sel_parts = [f"{k}={v}" for k, v in spec.selector.items()]
                selector_str = "selector: " + ",".join(sel_parts)
            # External IP hint
            ext_hint = ""
            if svc_type == "LoadBalancer":
                ingress_list = (svc.status.load_balancer.ingress or []) if svc.status and svc.status.load_balancer else []
                if not ingress_list:
                    ext_hint = "  [pending LB IP]"

            line = f"    {svc_name:<30} {svc_ns:<20} {ports_str:<25} {selector_str}{ext_hint}"
            by_type.setdefault(svc_type, []).append(line)

        sections.append(f"--- SERVICES ({svc_count}) ---")
        for stype in ("LoadBalancer", "NodePort", "ClusterIP", "ExternalName"):
            items = by_type.get(stype, [])
            if items:
                sections.append(f"  {stype}:")
                sections.extend(items)
        # Any other types
        for stype, items in by_type.items():
            if stype not in ("LoadBalancer", "NodePort", "ClusterIP", "ExternalName"):
                sections.append(f"  {stype}:")
                sections.extend(items)
        if not services:
            sections.append("  (none)")
        sections.append("")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Services: {exc}")
        sections.append("--- SERVICES ---")
        sections.append(f"  Error: {exc}")
        sections.append("")

    # ── 2. Endpoints ──────────────────────────────────────────
    try:
        if namespace:
            ep_list = core_v1.list_namespaced_endpoints(namespace=namespace)
        else:
            ep_list = core_v1.list_endpoints_for_all_namespaces()
        endpoints = (ep_list.items or [])[:_MAX_ITEMS]
        ep_count = len(endpoints)

        sections.append(f"--- ENDPOINTS ({ep_count}) ---")
        for ep in endpoints:
            meta = ep.metadata
            ep_name = meta.name or "<unknown>"
            ep_ns = meta.namespace or ""
            ready_count = 0
            not_ready_count = 0
            for subset in ep.subsets or []:
                ready_count += len(subset.addresses or [])
                not_ready_count += len(subset.not_ready_addresses or [])
            sections.append(
                f"  {ep_name:<30} {ep_ns:<20} {ready_count} ready, {not_ready_count} not-ready"
            )
        if not endpoints:
            sections.append("  (none)")
        sections.append("")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Endpoints: {exc}")
        sections.append("--- ENDPOINTS ---")
        sections.append(f"  Error: {exc}")
        sections.append("")

    # ── 3. Ingresses ──────────────────────────────────────────
    try:
        from kubernetes.client import NetworkingV1Api  # noqa: WPS433

        net_v1 = NetworkingV1Api(api_client=api_client)
        if namespace:
            ing_list = net_v1.list_namespaced_ingress(namespace=namespace)
        else:
            ing_list = net_v1.list_ingress_for_all_namespaces()
        ingresses = (ing_list.items or [])[:_MAX_ITEMS]
        ing_count = len(ingresses)

        sections.append(f"--- INGRESSES ({ing_count}) ---")
        for ing in ingresses:
            meta = ing.metadata
            ing_name = meta.name or "<unknown>"
            ing_ns = meta.namespace or ""
            spec = ing.spec
            # Hosts & paths
            hosts: list[str] = []
            paths: list[str] = []
            if spec and spec.rules:
                for rule in spec.rules:
                    if rule.host:
                        hosts.append(rule.host)
                    if rule.http and rule.http.paths:
                        for p in rule.http.paths:
                            backend_str = ""
                            if p.backend and p.backend.service:
                                b_name = p.backend.service.name or "?"
                                b_port = ""
                                if p.backend.service.port:
                                    b_port = str(
                                        p.backend.service.port.number
                                        or p.backend.service.port.name
                                        or ""
                                    )
                                backend_str = f"{b_name}:{b_port}" if b_port else b_name
                            path_val = p.path or "/"
                            paths.append(f"{path_val}\u2192{backend_str}")
            hosts_str = ", ".join(hosts) if hosts else "(no host)"
            paths_str = ", ".join(paths) if paths else "(no paths)"
            has_tls = "yes" if spec and spec.tls else "no"
            sections.append(
                f"  {ing_name:<20} {ing_ns:<16} hosts: {hosts_str}  paths: {paths_str}  TLS: {has_tls}"
            )
        if not ingresses:
            sections.append("  (none)")
        sections.append("")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Ingresses: {exc}")
        sections.append("--- INGRESSES ---")
        sections.append(f"  Error: {exc}")
        sections.append("")

    # ── 4. Network Policies ───────────────────────────────────
    try:
        from kubernetes.client import NetworkingV1Api as _NetV1  # noqa: WPS433

        net_v1_pol = _NetV1(api_client=api_client)
        if namespace:
            pol_list = net_v1_pol.list_namespaced_network_policy(namespace=namespace)
        else:
            pol_list = net_v1_pol.list_network_policy_for_all_namespaces()
        policies = (pol_list.items or [])[:_MAX_ITEMS]
        pol_count = len(policies)

        sections.append(f"--- NETWORK POLICIES ({pol_count}) ---")
        for pol in policies:
            meta = pol.metadata
            pol_name = meta.name or "<unknown>"
            pol_ns = meta.namespace or ""
            spec = pol.spec
            # Pod selector
            sel_str = ""
            if spec and spec.pod_selector:
                labels = spec.pod_selector.match_labels or {}
                if labels:
                    sel_str = ",".join(f"{k}={v}" for k, v in labels.items())
                else:
                    sel_str = "matchLabels={}"

            # Ingress rules summary
            ingress_summary = "allow all"
            if spec and spec.ingress is not None:
                if len(spec.ingress) == 0:
                    ingress_summary = "DENY ALL"
                else:
                    parts: list[str] = []
                    for rule in spec.ingress:
                        from_parts: list[str] = []
                        if rule._from:
                            for frm in rule._from:
                                if frm.pod_selector and frm.pod_selector.match_labels:
                                    lbl = ",".join(
                                        f"{k}={v}" for k, v in frm.pod_selector.match_labels.items()
                                    )
                                    from_parts.append(lbl)
                                elif frm.namespace_selector:
                                    from_parts.append("namespace-selector")
                                elif frm.ip_block:
                                    from_parts.append(f"cidr:{frm.ip_block.cidr}")
                        port_parts: list[str] = []  # type: ignore[no-redef]
                        if rule.ports:
                            for rp in rule.ports:
                                port_parts.append(str(rp.port or rp.protocol or ""))
                        from_str = " from " + ",".join(from_parts) if from_parts else ""
                        port_str = " port " + ",".join(port_parts) if port_parts else ""
                        parts.append(f"{from_str}{port_str}".strip())
                    ingress_summary = "; ".join(parts) if parts else "allow all"
            elif spec and spec.policy_types and "Ingress" in spec.policy_types:
                # policy_types includes Ingress but no ingress rules → deny
                ingress_summary = "DENY ALL"

            # Egress rules summary
            egress_summary = "allow all"
            if spec and spec.egress is not None:
                if len(spec.egress) == 0:
                    egress_summary = "DENY ALL"
                else:
                    egress_summary = f"{len(spec.egress)} rule(s)"
            elif spec and spec.policy_types and "Egress" in spec.policy_types:
                egress_summary = "DENY ALL"

            sections.append(
                f"  {pol_name:<24} {pol_ns:<16} pods: {sel_str}  "
                f"ingress: {ingress_summary}  egress: {egress_summary}"
            )
        if not policies:
            sections.append("  (none)")
        sections.append("")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"NetworkPolicies: {exc}")
        sections.append("--- NETWORK POLICIES ---")
        sections.append(f"  Error: {exc}")
        sections.append("")

    # ── Summary ───────────────────────────────────────────────
    sections.append(
        f"(showing {svc_count} services, {ep_count} endpoints, "
        f"{ing_count} ingresses, {pol_count} policies)"
    )

    if errors:
        sections.append("")
        sections.append("--- Partial errors ---")
        for err in errors:
            sections.append(f"  {err}")

    output = "\n".join(sections)
    _cache._set_cache(cache_key, output)
    return ToolResult(output=output, error=bool(errors and svc_count == 0 and ep_count == 0))


# ── Task 3.4 — async wrappers ───────────────────────────────
# Offload blocking kubernetes-client calls to a thread pool via to_async.

from vaig.core.async_utils import to_async  # noqa: E402

async_discover_workloads = to_async(discover_workloads)
async_discover_service_mesh = to_async(discover_service_mesh)
async_discover_network_topology = to_async(discover_network_topology)

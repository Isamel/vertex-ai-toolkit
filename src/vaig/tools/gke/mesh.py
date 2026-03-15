"""Mesh introspection tools — Istio/ASM configuration, security, and sidecar status."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from vaig.tools.base import ToolResult

from . import _cache, _clients

if TYPE_CHECKING:
    from vaig.core.config import GKEConfig

logger = logging.getLogger(__name__)

# ── Lazy import guard (mirrors _clients.py) ──────────────────
_K8S_AVAILABLE = True
try:
    from kubernetes.client import exceptions as k8s_exceptions  # noqa: WPS433
except ImportError:
    _K8S_AVAILABLE = False

# ── Constants ────────────────────────────────────────────────
_MESH_CACHE_TTL: int = 30  # seconds (spec: 30s, shorter than _DISCOVERY_TTL's 60s)
_MAX_RESOURCES_PER_TYPE: int = 50

_ISTIO_NETWORKING_GROUP = "networking.istio.io"
_ISTIO_SECURITY_GROUP = "security.istio.io"

# CRD kind → preferred API version fallback chain
_ISTIO_CRD_VERSIONS: dict[str, list[str]] = {
    "VirtualService": ["v1", "v1beta1", "v1alpha3"],
    "DestinationRule": ["v1", "v1beta1", "v1alpha3"],
    "Gateway": ["v1", "v1beta1", "v1alpha3"],
    "PeerAuthentication": ["v1", "v1beta1"],
    "AuthorizationPolicy": ["v1", "v1beta1"],
    "RequestAuthentication": ["v1", "v1beta1"],
}

# CRD kind → plural resource name for the K8s API
_KIND_PLURALS: dict[str, str] = {
    "VirtualService": "virtualservices",
    "DestinationRule": "destinationrules",
    "Gateway": "gateways",
    "PeerAuthentication": "peerauthentications",
    "AuthorizationPolicy": "authorizationpolicies",
    "RequestAuthentication": "requestauthentications",
    "ServiceEntry": "serviceentries",
    "Sidecar": "sidecars",
    "EnvoyFilter": "envoyfilters",
    "WorkloadEntry": "workloadentries",
    "WorkloadGroup": "workloadgroups",
    "Telemetry": "telemetries",
}

# Cache for resolved CRD versions (group+kind → version)
_CRD_VERSION_CACHE: dict[str, str] = {}


# ── Helpers ──────────────────────────────────────────────────


def _kind_to_plural(kind: str) -> str:
    """Map an Istio CRD kind to its plural resource name.

    Falls back to a naive lowercasing + 's' if the kind is not in the mapping.
    """
    return _KIND_PLURALS.get(kind, kind.lower() + "s")


def _detect_mesh_presence(
    core_v1: Any,
    apps_v1: Any,
) -> dict[str, Any]:
    """Detect Istio/ASM mesh presence on the cluster.

    Checks for:
    - ``istio-system`` namespace existence
    - ``istiod`` deployment health
    - Managed ASM via ``asm-managed`` revision labels on namespaces

    Returns a dict with keys: ``installed``, ``managed``, ``namespace``,
    ``istiod_found``, ``warnings``.
    """
    result: dict[str, Any] = {
        "installed": False,
        "managed": False,
        "namespace": "istio-system",
        "istiod_found": False,
        "warnings": [],
    }

    # 1. Check for istio-system namespace
    try:
        core_v1.read_namespace("istio-system")
        result["installed"] = True
    except Exception as exc:
        if _K8S_AVAILABLE and isinstance(exc, k8s_exceptions.ApiException):
            if exc.status == 404:
                # Namespace doesn't exist — mesh not installed via standard path
                pass
            elif exc.status == 403:
                result["warnings"].append(
                    "RBAC: cannot read istio-system namespace (403 Forbidden)"
                )
            else:
                result["warnings"].append(f"Error checking istio-system namespace: {exc.reason}")
        else:
            result["warnings"].append(f"Error checking istio-system namespace: {exc}")
        # If we can't read the namespace, still check for managed ASM below

    # 2. Check for istiod deployment (only if namespace exists)
    if result["installed"]:
        try:
            istiod_deploy = apps_v1.read_namespaced_deployment(
                name="istiod", namespace="istio-system",
            )
            result["istiod_found"] = True

            # Check replica health
            ready = istiod_deploy.status.ready_replicas or 0
            desired = istiod_deploy.spec.replicas or 0
            if ready < desired:
                result["warnings"].append(
                    f"istiod: {ready}/{desired} replicas ready"
                )
        except Exception as exc:
            if _K8S_AVAILABLE and isinstance(exc, k8s_exceptions.ApiException):
                if exc.status == 404:
                    # istiod deployment not found — might be managed ASM
                    pass
                elif exc.status == 403:
                    result["warnings"].append(
                        "RBAC: cannot read istiod deployment (403 Forbidden)"
                    )
                else:
                    result["warnings"].append(f"Error checking istiod: {exc.reason}")
            else:
                result["warnings"].append(f"Error checking istiod: {exc}")

    # 3. Check for managed ASM (asm-managed revision labels on namespaces)
    try:
        ns_list = core_v1.list_namespace()
        for ns in ns_list.items or []:
            labels = ns.metadata.labels or {}
            # Managed ASM uses revision labels like 'istio.io/rev: asm-managed'
            # or 'istio.io/rev: asm-managed-rapid', 'asm-managed-stable'
            rev = labels.get("istio.io/rev", "")
            if rev.startswith("asm-managed"):
                result["managed"] = True
                result["installed"] = True
                break
    except Exception as exc:
        if _K8S_AVAILABLE and isinstance(exc, k8s_exceptions.ApiException):
            if exc.status == 403:
                result["warnings"].append(
                    "RBAC: cannot list namespaces for ASM detection (403 Forbidden)"
                )
            else:
                result["warnings"].append(f"Error listing namespaces: {exc.reason}")
        else:
            result["warnings"].append(f"Error listing namespaces: {exc}")

    return result


def _get_istio_version(
    apps_v1: Any,
    custom_api: Any,
) -> str:
    """Extract Istio/ASM version from the cluster.

    Strategy:
    1. Read istiod deployment image tag (most reliable)
    2. Fall back to IstioOperator CR if available

    Returns version string or ``"unknown"``.
    """
    # Strategy 1: istiod deployment image tag
    try:
        istiod = apps_v1.read_namespaced_deployment(
            name="istiod", namespace="istio-system",
        )
        containers = (
            istiod.spec.template.spec.containers
            if istiod.spec and istiod.spec.template and istiod.spec.template.spec
            else []
        )
        for container in containers or []:
            image = container.image or ""
            if ":" in image:
                tag = image.rsplit(":", 1)[-1]
                if tag and (tag[0].isdigit() or tag.startswith("v")):
                    return tag
    except Exception:
        pass

    # Strategy 2: IstioOperator CR
    try:
        operators = custom_api.list_namespaced_custom_object(
            group="install.istio.io",
            version="v1alpha1",
            namespace="istio-system",
            plural="istiooperators",
            limit=1,
        )
        items = operators.get("items", [])
        if items:
            spec = items[0].get("spec", {})
            version = spec.get("tag") or spec.get("version", "")
            if version:
                return str(version)
    except Exception:
        pass

    return "unknown"


def _resolve_crd_version(
    custom_api: Any,
    group: str,
    kind: str,
) -> str | None:
    """Resolve the available API version for an Istio CRD kind.

    Tries versions from ``_ISTIO_CRD_VERSIONS`` in order (v1 → v1beta1 → v1alpha3)
    using a ``limit=1`` list call to check availability. Caches results per group+kind.

    Returns the version string or ``None`` if none work.
    """
    cache_key = f"{group}/{kind}"
    if cache_key in _CRD_VERSION_CACHE:
        return _CRD_VERSION_CACHE[cache_key]

    versions = _ISTIO_CRD_VERSIONS.get(kind, ["v1", "v1beta1", "v1alpha3"])
    plural = _kind_to_plural(kind)

    for version in versions:
        try:
            custom_api.list_cluster_custom_object(
                group=group,
                version=version,
                plural=plural,
                limit=1,
            )
            _CRD_VERSION_CACHE[cache_key] = version
            return version
        except Exception as exc:
            if _K8S_AVAILABLE and isinstance(exc, k8s_exceptions.ApiException):
                if exc.status in (404, 403):
                    continue
                # Other errors (e.g., 500) — skip this version
                continue
            # Non-K8s exception — skip
            continue

    return None


def _read_custom_resources(
    custom_api: Any,
    group: str,
    version: str,
    kind: str,
    namespace: str | None = None,
) -> list[dict[str, Any]]:
    """Read custom resources (CRs) of a given type from the cluster.

    Args:
        custom_api: K8s CustomObjectsApi instance.
        group: API group (e.g. ``networking.istio.io``).
        version: API version (e.g. ``v1``).
        kind: CRD kind (e.g. ``VirtualService``). Converted to plural internally.
        namespace: If set, list resources in this namespace only.
            If ``None``, list across all namespaces.

    Returns a list of resource dicts, truncated at ``_MAX_RESOURCES_PER_TYPE``.
    Returns empty list on 404 or other errors (with logging).
    """
    plural = _kind_to_plural(kind)
    items: list[dict[str, Any]] = []

    try:
        if namespace:
            result = custom_api.list_namespaced_custom_object(
                group=group,
                version=version,
                namespace=namespace,
                plural=plural,
            )
        else:
            result = custom_api.list_cluster_custom_object(
                group=group,
                version=version,
                plural=plural,
            )

        all_items = result.get("items", [])

        if len(all_items) > _MAX_RESOURCES_PER_TYPE:
            items = all_items[:_MAX_RESOURCES_PER_TYPE]
            logger.info(
                "Truncated %s/%s to %d items (total: %d)",
                group, plural, _MAX_RESOURCES_PER_TYPE, len(all_items),
            )
        else:
            items = all_items

    except Exception as exc:
        if _K8S_AVAILABLE and isinstance(exc, k8s_exceptions.ApiException):
            if exc.status == 403:
                logger.warning(
                    "RBAC: cannot list %s/%s (403 Forbidden)", group, plural,
                )
            elif exc.status == 404:
                logger.debug("CRD %s/%s not found (404)", group, plural)
            else:
                logger.warning(
                    "Error listing %s/%s: %s %s",
                    group, plural, exc.status, exc.reason,
                )
        else:
            logger.warning("Error listing %s/%s: %s", group, plural, exc)

    return items


def clear_crd_version_cache() -> None:
    """Clear the CRD version resolution cache (useful for testing)."""
    _CRD_VERSION_CACHE.clear()


# ── Formatters ───────────────────────────────────────────────


def _format_mesh_status(presence: dict[str, Any], version: str) -> str:
    """Format mesh presence and version info into a human-readable block."""
    lines: list[str] = []

    if presence["managed"]:
        lines.append("Mesh type: Anthos Service Mesh (managed)")
    elif presence["installed"]:
        lines.append("Mesh type: Istio")
    else:
        return "No service mesh detected."

    lines.append(f"Namespace: {presence['namespace']}")
    lines.append(f"Version: {version}")

    if presence["istiod_found"]:
        lines.append("Control plane (istiod): running")
    elif presence["managed"]:
        lines.append("Control plane: Google-managed (no in-cluster istiod)")
    else:
        lines.append("Control plane (istiod): not found")

    return "\n".join(lines)


def _format_injection_table(namespaces: list[dict[str, Any]]) -> str:
    """Format namespace injection status as a table.

    Each entry in *namespaces* is a dict with keys: ``name``, ``injection``,
    ``revision``.
    """
    if not namespaces:
        return "  (no namespaces found)"

    lines: list[str] = []
    lines.append(
        f"  {'NAMESPACE':<30} {'INJECTION':<14} {'REVISION':<20}"
    )
    lines.append("  " + "-" * 64)

    for ns in namespaces:
        name = ns["name"]
        injection = ns["injection"]
        revision = ns["revision"] or "-"
        lines.append(f"  {name:<30} {injection:<14} {revision:<20}")

    return "\n".join(lines)


def _format_virtual_service(vs: dict[str, Any]) -> str:
    """Format a VirtualService CR into a concise summary.

    Shows hosts, HTTP route destinations with traffic-split weights,
    and match rules.
    """
    metadata = vs.get("metadata", {})
    name = metadata.get("name", "<unknown>")
    namespace = metadata.get("namespace", "")
    spec = vs.get("spec", {})

    lines: list[str] = [f"  {name} (ns: {namespace})"]

    # Hosts
    hosts = spec.get("hosts", [])
    if hosts:
        lines.append(f"    Hosts: {', '.join(str(h) for h in hosts)}")

    # Gateways
    gateways = spec.get("gateways", [])
    if gateways:
        lines.append(f"    Gateways: {', '.join(str(g) for g in gateways)}")

    # HTTP routes
    http_routes = spec.get("http", [])
    for i, route in enumerate(http_routes):
        route_parts: list[str] = []

        # Match rules
        match_list = route.get("match", [])
        if match_list:
            match_strs: list[str] = []
            for match in match_list:
                uri = match.get("uri", {})
                if uri:
                    for match_type, match_val in uri.items():
                        match_strs.append(f"uri {match_type}={match_val}")
                headers = match.get("headers", {})
                for hdr_name, hdr_match in headers.items():
                    for mt, mv in hdr_match.items():
                        match_strs.append(f"header {hdr_name} {mt}={mv}")
            if match_strs:
                route_parts.append(f"match: {'; '.join(match_strs)}")

        # Destinations with weights
        destinations = route.get("route", [])
        dest_strs: list[str] = []
        for dest in destinations:
            d = dest.get("destination", {})
            host = d.get("host", "?")
            subset = d.get("subset", "")
            port = d.get("port", {}).get("number", "")
            weight = dest.get("weight", "")
            dest_str = host
            if subset:
                dest_str += f"/{subset}"
            if port:
                dest_str += f":{port}"
            if weight:
                dest_str += f" ({weight}%)"
            dest_strs.append(dest_str)

        if dest_strs:
            route_parts.append(f"-> {', '.join(dest_strs)}")

        # Timeout / retries
        timeout = route.get("timeout")
        if timeout:
            route_parts.append(f"timeout={timeout}")
        retries = route.get("retries")
        if retries:
            attempts = retries.get("attempts", "")
            route_parts.append(f"retries={attempts}")

        label = f"    Route {i + 1}: " if len(http_routes) > 1 else "    Route: "
        lines.append(label + " | ".join(route_parts) if route_parts else label + "(empty)")

    # TCP routes (brief)
    tcp_routes = spec.get("tcp", [])
    if tcp_routes:
        lines.append(f"    TCP routes: {len(tcp_routes)}")

    return "\n".join(lines)


def _format_destination_rule(dr: dict[str, Any]) -> str:
    """Format a DestinationRule CR into a concise summary.

    Shows host, subsets, and traffic policy (connection pool, outlier
    detection, load balancer settings).
    """
    metadata = dr.get("metadata", {})
    name = metadata.get("name", "<unknown>")
    namespace = metadata.get("namespace", "")
    spec = dr.get("spec", {})

    lines: list[str] = [f"  {name} (ns: {namespace})"]

    # Host
    host = spec.get("host", "")
    if host:
        lines.append(f"    Host: {host}")

    # Traffic policy
    policy = spec.get("trafficPolicy", {})
    if policy:
        parts: list[str] = []

        # Load balancer
        lb = policy.get("loadBalancer", {})
        if lb:
            simple = lb.get("simple", "")
            if simple:
                parts.append(f"LB: {simple}")

        # Connection pool
        conn_pool = policy.get("connectionPool", {})
        if conn_pool:
            tcp_settings = conn_pool.get("tcp", {})
            http_settings = conn_pool.get("http", {})
            pool_parts: list[str] = []
            if tcp_settings.get("maxConnections"):
                pool_parts.append(f"maxConn={tcp_settings['maxConnections']}")
            if http_settings.get("h2UpgradePolicy"):
                pool_parts.append(f"h2={http_settings['h2UpgradePolicy']}")
            if http_settings.get("maxRequestsPerConnection"):
                pool_parts.append(
                    f"maxReqPerConn={http_settings['maxRequestsPerConnection']}"
                )
            if pool_parts:
                parts.append(f"pool: {', '.join(pool_parts)}")

        # Outlier detection
        outlier = policy.get("outlierDetection", {})
        if outlier:
            od_parts: list[str] = []
            if outlier.get("consecutiveErrors"):
                od_parts.append(f"errors={outlier['consecutiveErrors']}")
            if outlier.get("consecutive5xxErrors"):
                od_parts.append(f"5xx={outlier['consecutive5xxErrors']}")
            if outlier.get("interval"):
                od_parts.append(f"interval={outlier['interval']}")
            if outlier.get("baseEjectionTime"):
                od_parts.append(f"eject={outlier['baseEjectionTime']}")
            if od_parts:
                parts.append(f"outlier: {', '.join(od_parts)}")

        # TLS
        tls = policy.get("tls", {})
        if tls:
            mode = tls.get("mode", "")
            if mode:
                parts.append(f"TLS: {mode}")

        if parts:
            lines.append(f"    Policy: {' | '.join(parts)}")

    # Subsets
    subsets = spec.get("subsets", [])
    if subsets:
        subset_strs: list[str] = []
        for subset in subsets:
            s_name = subset.get("name", "?")
            s_labels = subset.get("labels", {})
            label_str = ",".join(f"{k}={v}" for k, v in s_labels.items()) if s_labels else ""
            subset_strs.append(f"{s_name}({label_str})" if label_str else s_name)
        lines.append(f"    Subsets: {', '.join(subset_strs)}")

    return "\n".join(lines)


def _format_gateway(gw: dict[str, Any]) -> str:
    """Format a Gateway CR into a concise summary.

    Shows servers with ports, hosts, and TLS configuration.
    """
    metadata = gw.get("metadata", {})
    name = metadata.get("name", "<unknown>")
    namespace = metadata.get("namespace", "")
    spec = gw.get("spec", {})

    lines: list[str] = [f"  {name} (ns: {namespace})"]

    # Selector
    selector = spec.get("selector", {})
    if selector:
        sel_str = ", ".join(f"{k}={v}" for k, v in selector.items())
        lines.append(f"    Selector: {sel_str}")

    # Servers
    servers = spec.get("servers", [])
    for server in servers:
        port = server.get("port", {})
        port_num = port.get("number", "?")
        port_name = port.get("name", "")
        protocol = port.get("protocol", "")

        hosts = server.get("hosts", [])
        hosts_str = ", ".join(str(h) for h in hosts) if hosts else "*"

        tls = server.get("tls", {})
        tls_mode = tls.get("mode", "")
        tls_str = f" TLS: {tls_mode}" if tls_mode else ""

        port_label = f"{port_num}/{protocol}" + (f" ({port_name})" if port_name else "")
        lines.append(f"    Server: {port_label} -> {hosts_str}{tls_str}")

    return "\n".join(lines)


def _format_peer_authentication(pa: dict[str, Any]) -> str:
    """Format a PeerAuthentication CR into a concise summary.

    Shows mTLS mode and any port-level overrides.
    """
    metadata = pa.get("metadata", {})
    name = metadata.get("name", "<unknown>")
    namespace = metadata.get("namespace", "")
    spec = pa.get("spec", {})

    lines: list[str] = [f"  {name} (ns: {namespace})"]

    # mTLS mode
    mtls = spec.get("mtls", {})
    mode = mtls.get("mode", "UNSET") if mtls else "UNSET"
    lines.append(f"    mTLS: {mode}")

    # Port-level overrides
    port_mtls = spec.get("portLevelMtls", {})
    if port_mtls:
        overrides = ", ".join(
            f"{port}={cfg.get('mode', '?')}" for port, cfg in port_mtls.items()
        )
        lines.append(f"    Port overrides: {overrides}")

    return "\n".join(lines)


def _format_authorization_policy(ap: dict[str, Any]) -> str:
    """Format an AuthorizationPolicy CR into a concise summary.

    Shows action, rules with source principals/namespaces and
    destination operations (methods, paths, ports).
    """
    metadata = ap.get("metadata", {})
    name = metadata.get("name", "<unknown>")
    namespace = metadata.get("namespace", "")
    spec = ap.get("spec", {})

    lines: list[str] = [f"  {name} (ns: {namespace})"]

    action = spec.get("action", "ALLOW")
    lines.append(f"    Action: {action}")

    rules = spec.get("rules", [])
    if not rules:
        if action == "ALLOW":
            lines.append("    (allow-all)")
        elif action == "DENY":
            lines.append("    (deny-all)")
    else:
        for i, rule in enumerate(rules):
            rule_parts: list[str] = []

            # From — sources
            from_list = rule.get("from", [])
            for from_entry in from_list:
                source = from_entry.get("source", {})
                principals = source.get("principals", [])
                namespaces = source.get("namespaces", [])
                if principals:
                    rule_parts.append(f"principals: {', '.join(principals)}")
                if namespaces:
                    rule_parts.append(f"namespaces: {', '.join(namespaces)}")

            # To — operations
            to_list = rule.get("to", [])
            for to_entry in to_list:
                operation = to_entry.get("operation", {})
                methods = operation.get("methods", [])
                paths = operation.get("paths", [])
                ports = operation.get("ports", [])
                if methods:
                    rule_parts.append(f"methods: {', '.join(methods)}")
                if paths:
                    rule_parts.append(f"paths: {', '.join(paths)}")
                if ports:
                    rule_parts.append(f"ports: {', '.join(str(p) for p in ports)}")

            # When — conditions
            when_list = rule.get("when", [])
            if when_list:
                rule_parts.append(f"conditions: {len(when_list)}")

            label = f"    Rule {i + 1}: " if len(rules) > 1 else "    Rule: "
            lines.append(label + " | ".join(rule_parts) if rule_parts else label + "(empty)")

    return "\n".join(lines)


def _format_request_authentication(ra: dict[str, Any]) -> str:
    """Format a RequestAuthentication CR into a concise summary.

    Shows JWT rules with issuer and audiences.
    """
    metadata = ra.get("metadata", {})
    name = metadata.get("name", "<unknown>")
    namespace = metadata.get("namespace", "")
    spec = ra.get("spec", {})

    lines: list[str] = [f"  {name} (ns: {namespace})"]

    jwt_rules = spec.get("jwtRules", [])
    if not jwt_rules:
        lines.append("    (no JWT rules configured)")
    else:
        for jwt_rule in jwt_rules:
            issuer = jwt_rule.get("issuer", "?")
            audiences = jwt_rule.get("audiences", [])
            aud_str = ", ".join(audiences) if audiences else "(any)"
            lines.append(f"    JWT: issuer={issuer}, audiences={aud_str}")

    return "\n".join(lines)


# ── Public Tool Functions ────────────────────────────────────


def get_mesh_overview(
    *,
    gke_config: GKEConfig,
    namespace: str = "",
    force_refresh: bool = False,
) -> ToolResult:
    """Show Istio/ASM mesh overview: presence, version, injection status.

    Returns mesh type (Istio vs managed ASM), version, istiod health,
    and per-namespace sidecar injection status (``istio-injection`` label
    and ``istio.io/rev`` revision label).

    Returns ``"No service mesh detected"`` when no mesh is found.
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    # ── Cache check ───────────────────────────────────────────
    cache_key = _cache._cache_key_discovery("mesh_overview", namespace)
    if not force_refresh:
        cached = _cache._get_cached(cache_key)
        if cached is not None:
            return ToolResult(output=cached, error=False)

    # ── Create clients ────────────────────────────────────────
    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, apps_v1, custom_api, _api_client = result

    ns_filter = namespace if namespace else "all"
    sections: list[str] = [
        "=== Mesh Overview ===",
        f"Namespace filter: {ns_filter}",
        "",
    ]
    warnings: list[str] = []

    # ── 1. Mesh presence ──────────────────────────────────────
    presence = _detect_mesh_presence(core_v1, apps_v1)
    warnings.extend(presence.get("warnings", []))

    if not presence["installed"]:
        output = "No service mesh detected."
        if warnings:
            output += "\n\nWarnings:\n" + "\n".join(f"  {w}" for w in warnings)
        _cache._set_cache(cache_key, output)
        return ToolResult(output=output, error=False)

    # ── 2. Version ────────────────────────────────────────────
    version = _get_istio_version(apps_v1, custom_api)

    sections.append(_format_mesh_status(presence, version))
    sections.append("")

    # ── 3. Namespace injection status ─────────────────────────
    sections.append("--- Sidecar Injection Status ---")
    ns_injection: list[dict[str, Any]] = []

    try:
        if namespace:
            try:
                ns_obj = core_v1.read_namespace(namespace)
                ns_items = [ns_obj]
            except Exception:
                ns_items = []
                warnings.append(f"Cannot read namespace '{namespace}'")
        else:
            ns_list = core_v1.list_namespace()
            ns_items = ns_list.items or []

        for ns_obj in ns_items:
            labels = ns_obj.metadata.labels or {} if ns_obj.metadata else {}
            ns_name = ns_obj.metadata.name if ns_obj.metadata else "<unknown>"

            injection_label = labels.get("istio-injection", "")
            rev_label = labels.get("istio.io/rev", "")

            if injection_label == "enabled":
                injection_status = "enabled"
            elif injection_label == "disabled":
                injection_status = "disabled"
            elif rev_label:
                injection_status = "enabled"
            else:
                injection_status = "not set"

            ns_injection.append({
                "name": ns_name,
                "injection": injection_status,
                "revision": rev_label,
            })

    except Exception as exc:
        if _K8S_AVAILABLE and isinstance(exc, k8s_exceptions.ApiException):
            if exc.status == 403:
                warnings.append(
                    "RBAC: cannot list namespaces for injection status (403 Forbidden)"
                )
            else:
                warnings.append(f"Error listing namespaces: {exc.reason}")
        else:
            warnings.append(f"Error listing namespaces: {exc}")

    sections.append(_format_injection_table(ns_injection))

    # ── Injection summary ─────────────────────────────────────
    enabled_count = sum(1 for ns in ns_injection if ns["injection"] == "enabled")
    total_count = len(ns_injection)
    sections.append(f"\n  Injection enabled: {enabled_count}/{total_count} namespaces")
    sections.append("")

    # ── Warnings ──────────────────────────────────────────────
    if warnings:
        sections.append("Warnings:")
        for w in warnings:
            sections.append(f"  {w}")
    else:
        sections.append("Warnings: none")

    output = "\n".join(sections)
    _cache._set_cache(cache_key, output)
    return ToolResult(output=output, error=False)


def get_mesh_config(
    *,
    gke_config: GKEConfig,
    namespace: str = "",
    force_refresh: bool = False,
) -> ToolResult:
    """Show Istio/ASM traffic management configuration.

    Lists VirtualServices, DestinationRules, and Gateways with formatted
    details. Auto-detects the CRD API version via ``_resolve_crd_version()``.

    Fail-open: if one CRD type fails (403/404), the tool still returns
    results for the other types with a warning.
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    # ── Cache check ───────────────────────────────────────────
    cache_key = _cache._cache_key_discovery("mesh_config", namespace)
    if not force_refresh:
        cached = _cache._get_cached(cache_key)
        if cached is not None:
            return ToolResult(output=cached, error=False)

    # ── Create clients ────────────────────────────────────────
    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    _core_v1, _apps_v1, custom_api, _api_client = result

    ns_filter = namespace if namespace else "all namespaces"
    sections: list[str] = [
        "=== Mesh Traffic Configuration ===",
        f"Namespace: {ns_filter}",
        "",
    ]
    warnings: list[str] = []
    total_resources = 0

    # CRD types to query: (kind, group, formatter)
    crd_types: list[tuple[str, str, Any]] = [
        ("VirtualService", _ISTIO_NETWORKING_GROUP, _format_virtual_service),
        ("DestinationRule", _ISTIO_NETWORKING_GROUP, _format_destination_rule),
        ("Gateway", _ISTIO_NETWORKING_GROUP, _format_gateway),
    ]

    for kind, group, formatter in crd_types:
        plural = _kind_to_plural(kind)

        # Resolve version
        version = _resolve_crd_version(custom_api, group, kind)
        if version is None:
            warnings.append(f"{kind}: CRD not found (no supported API version)")
            sections.append(f"--- {kind}s ---")
            sections.append("  (CRD not available)")
            sections.append("")
            continue

        # Read resources
        ns_arg = namespace if namespace else None
        resources = _read_custom_resources(
            custom_api, group, version, kind, namespace=ns_arg,
        )

        sections.append(f"--- {kind}s ({len(resources)}) ---")

        if not resources:
            sections.append("  (none)")
        else:
            for resource in resources:
                sections.append(formatter(resource))

        if len(resources) >= _MAX_RESOURCES_PER_TYPE:
            sections.append(f"  ... truncated at {_MAX_RESOURCES_PER_TYPE}")

        total_resources += len(resources)
        sections.append("")

    # ── Summary ───────────────────────────────────────────────
    sections.append(f"Total resources: {total_resources}")

    if warnings:
        sections.append("")
        sections.append("Warnings:")
        for w in warnings:
            sections.append(f"  {w}")

    output = "\n".join(sections)
    _cache._set_cache(cache_key, output)
    return ToolResult(output=output, error=False)


def get_mesh_security(
    *,
    gke_config: GKEConfig,
    namespace: str = "",
    force_refresh: bool = False,
) -> ToolResult:
    """Show Istio/ASM security configuration.

    Lists PeerAuthentication (mTLS), AuthorizationPolicy (RBAC),
    and RequestAuthentication (JWT) resources with formatted details.
    Auto-detects the CRD API version via ``_resolve_crd_version()``.

    Fail-open: if one CRD type fails (403/404), the tool still returns
    results for the other types with a warning.
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    # ── Cache check ───────────────────────────────────────────
    cache_key = _cache._cache_key_discovery("mesh_security", namespace)
    if not force_refresh:
        cached = _cache._get_cached(cache_key)
        if cached is not None:
            return ToolResult(output=cached, error=False)

    # ── Create clients ────────────────────────────────────────
    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    _core_v1, _apps_v1, custom_api, _api_client = result

    ns_filter = namespace if namespace else "all namespaces"
    sections: list[str] = [
        "=== Mesh Security Configuration ===",
        f"Namespace: {ns_filter}",
        "",
    ]
    warnings: list[str] = []
    total_resources = 0

    # CRD types to query: (kind, group, formatter)
    crd_types: list[tuple[str, str, Any]] = [
        ("PeerAuthentication", _ISTIO_SECURITY_GROUP, _format_peer_authentication),
        ("AuthorizationPolicy", _ISTIO_SECURITY_GROUP, _format_authorization_policy),
        ("RequestAuthentication", _ISTIO_SECURITY_GROUP, _format_request_authentication),
    ]

    for kind, group, formatter in crd_types:
        # Resolve version
        version = _resolve_crd_version(custom_api, group, kind)
        if version is None:
            warnings.append(f"{kind}: CRD not found (no supported API version)")
            sections.append(f"--- {kind}s ---")
            sections.append("  (CRD not available)")
            sections.append("")
            continue

        # Read resources
        ns_arg = namespace if namespace else None
        resources = _read_custom_resources(
            custom_api, group, version, kind, namespace=ns_arg,
        )

        sections.append(f"--- {kind}s ({len(resources)}) ---")

        if not resources:
            sections.append("  (none)")
        else:
            for resource in resources:
                sections.append(formatter(resource))

        if len(resources) >= _MAX_RESOURCES_PER_TYPE:
            sections.append(f"  ... truncated at {_MAX_RESOURCES_PER_TYPE}")

        total_resources += len(resources)
        sections.append("")

    # ── Summary ───────────────────────────────────────────────
    sections.append(f"Total resources: {total_resources}")

    if warnings:
        sections.append("")
        sections.append("Warnings:")
        for w in warnings:
            sections.append(f"  {w}")

    output = "\n".join(sections)
    _cache._set_cache(cache_key, output)
    return ToolResult(output=output, error=False)


def _format_sidecar_table(rows: list[dict[str, Any]]) -> str:
    """Format per-pod sidecar status as a table.

    Each entry in *rows* is a dict with keys: ``pod``, ``namespace``,
    ``has_sidecar``, ``sidecar_version``, ``owner``, ``anomaly``.
    """
    if not rows:
        return "  (no pods found)"

    lines: list[str] = []
    lines.append(
        f"  {'POD':<40} {'NAMESPACE':<18} {'SIDECAR':<10} {'VERSION':<18} {'OWNER':<25} {'ANOMALY':<10}"
    )
    lines.append("  " + "-" * 121)

    for row in rows:
        pod = row["pod"][:39]
        ns = row["namespace"][:17]
        sidecar = "yes" if row["has_sidecar"] else "no"
        version = (row.get("sidecar_version") or "-")[:17]
        owner = (row.get("owner") or "-")[:24]
        anomaly = row.get("anomaly", "")
        lines.append(
            f"  {pod:<40} {ns:<18} {sidecar:<10} {version:<18} {owner:<25} {anomaly:<10}"
        )

    return "\n".join(lines)


def get_sidecar_status(
    *,
    gke_config: GKEConfig,
    namespace: str = "",
    force_refresh: bool = False,
) -> ToolResult:
    """Show sidecar injection status for pods in the cluster.

    For each pod, checks whether the ``istio-proxy`` container is present.
    Detects injection anomalies: pods in injection-enabled namespaces
    without a sidecar, or pods with a sidecar in non-injected namespaces.

    Reports per-pod status in a table and provides a summary with
    injection coverage statistics.
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    # ── Cache check ───────────────────────────────────────────
    cache_key = _cache._cache_key_discovery("sidecar_status", namespace)
    if not force_refresh:
        cached = _cache._get_cached(cache_key)
        if cached is not None:
            return ToolResult(output=cached, error=False)

    # ── Create clients ────────────────────────────────────────
    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, _apps_v1, _custom_api, _api_client = result

    ns_filter = namespace if namespace else "all namespaces"
    sections: list[str] = [
        "=== Sidecar Injection Status ===",
        f"Namespace: {ns_filter}",
        "",
    ]
    warnings: list[str] = []

    # ── 1. Build namespace injection map ──────────────────────
    ns_injection_map: dict[str, bool] = {}
    try:
        ns_list = core_v1.list_namespace()
        for ns_obj in ns_list.items or []:
            labels = ns_obj.metadata.labels or {} if ns_obj.metadata else {}
            ns_name = ns_obj.metadata.name if ns_obj.metadata else ""
            if not ns_name:
                continue
            injection_label = labels.get("istio-injection", "")
            rev_label = labels.get("istio.io/rev", "")
            ns_injection_map[ns_name] = (
                injection_label == "enabled" or bool(rev_label)
            )
    except Exception as exc:
        if _K8S_AVAILABLE and isinstance(exc, k8s_exceptions.ApiException):
            if exc.status == 403:
                warnings.append(
                    "RBAC: cannot list namespaces (403 Forbidden)"
                )
            else:
                warnings.append(f"Error listing namespaces: {exc.reason}")
        else:
            warnings.append(f"Error listing namespaces: {exc}")

    # ── 2. List pods and check sidecar ────────────────────────
    rows: list[dict[str, Any]] = []
    try:
        if namespace:
            pod_list = core_v1.list_namespaced_pod(namespace)
        else:
            pod_list = core_v1.list_pod_for_all_namespaces()

        for pod in pod_list.items or []:
            meta = pod.metadata
            if not meta:
                continue
            pod_name = meta.name or "<unknown>"
            pod_ns = meta.namespace or ""
            spec = pod.spec

            # Determine owner (first ownerReference)
            owner = "-"
            owner_refs = meta.owner_references or []
            if owner_refs:
                ref = owner_refs[0]
                owner = f"{ref.kind}/{ref.name}" if ref.kind and ref.name else "-"

            # Check for istio-proxy container
            has_sidecar = False
            sidecar_version = ""
            containers = spec.containers if spec else []
            for container in containers or []:
                if container.name == "istio-proxy":
                    has_sidecar = True
                    image = container.image or ""
                    if ":" in image:
                        sidecar_version = image.rsplit(":", 1)[-1]
                    break

            # Also check init containers (some setups use istio-init)
            init_containers = spec.init_containers if spec else []
            if not has_sidecar and init_containers:
                for container in init_containers:
                    if container.name == "istio-proxy":
                        has_sidecar = True
                        image = container.image or ""
                        if ":" in image:
                            sidecar_version = image.rsplit(":", 1)[-1]
                        break

            # Detect anomalies
            anomaly = ""
            ns_injected = ns_injection_map.get(pod_ns, False)
            if ns_injected and not has_sidecar:
                anomaly = "MISSING"
            elif not ns_injected and has_sidecar:
                anomaly = "UNEXPECTED"

            rows.append({
                "pod": pod_name,
                "namespace": pod_ns,
                "has_sidecar": has_sidecar,
                "sidecar_version": sidecar_version,
                "owner": owner,
                "anomaly": anomaly,
            })

    except Exception as exc:
        if _K8S_AVAILABLE and isinstance(exc, k8s_exceptions.ApiException):
            if exc.status == 403:
                warnings.append("RBAC: cannot list pods (403 Forbidden)")
            else:
                warnings.append(f"Error listing pods: {exc.reason}")
        else:
            warnings.append(f"Error listing pods: {exc}")

    # Truncate if too many pods
    total_pods = len(rows)
    if total_pods > _MAX_RESOURCES_PER_TYPE:
        rows = rows[:_MAX_RESOURCES_PER_TYPE]
        warnings.append(
            f"Showing {_MAX_RESOURCES_PER_TYPE}/{total_pods} pods (truncated)"
        )

    sections.append(_format_sidecar_table(rows))
    sections.append("")

    # ── 3. Summary ────────────────────────────────────────────
    with_sidecar = sum(1 for r in rows if r["has_sidecar"])
    anomaly_count = sum(1 for r in rows if r["anomaly"])
    sections.append(f"Pods with sidecar: {with_sidecar}/{len(rows)}")
    if anomaly_count:
        missing = sum(1 for r in rows if r["anomaly"] == "MISSING")
        unexpected = sum(1 for r in rows if r["anomaly"] == "UNEXPECTED")
        sections.append(f"Anomalies: {anomaly_count} (missing: {missing}, unexpected: {unexpected})")
    else:
        sections.append("Anomalies: none")

    if warnings:
        sections.append("")
        sections.append("Warnings:")
        for w in warnings:
            sections.append(f"  {w}")

    output = "\n".join(sections)
    _cache._set_cache(cache_key, output)
    return ToolResult(output=output, error=False)


# ── Task 3.4 — async wrappers ───────────────────────────────
# Offload blocking kubernetes-client calls to a thread pool via to_async.

from vaig.core.async_utils import to_async  # noqa: E402

async_get_mesh_overview = to_async(get_mesh_overview)
async_get_mesh_config = to_async(get_mesh_config)
async_get_mesh_security = to_async(get_mesh_security)
async_get_sidecar_status = to_async(get_sidecar_status)

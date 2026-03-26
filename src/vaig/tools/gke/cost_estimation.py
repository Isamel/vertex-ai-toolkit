"""GKE Autopilot workload cost estimation.

Computes estimated monthly costs for GKE Autopilot workloads based on
Kubernetes resource requests (CPU, memory, ephemeral storage) and — when
available — actual usage metrics from Cloud Monitoring.

Cost model:
    monthly_cost = resource_quantity * hourly_rate_per_unit * 730 hours/month

Only Autopilot clusters are supported; Standard clusters bill at the node
level (not per-workload), so per-workload estimation would be misleading.

Regional pricing tables are maintained in this module. Only regions present
in the table are supported; unknown regions degrade gracefully.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vaig.core.config import GKEConfig
    from vaig.skills.service_health.schema import GKECostReport

logger = logging.getLogger(__name__)

# ── Module-level imports for patchability ──────────────────────
# Imported at module load so tests can patch them via their dotted path.
# Guarded so the module remains importable even if the optional dep is missing.

try:
    from vaig.tools.gke._clients import _create_k8s_clients, detect_autopilot  # noqa: WPS433
except ImportError:  # pragma: no cover
    _create_k8s_clients = None  # type: ignore[assignment]
    detect_autopilot = None  # type: ignore[assignment]

# ── Hours per month constant ──────────────────────────────────

_HOURS_PER_MONTH: float = 730.0

# ── Lazy import guard: kubernetes ────────────────────────────

_K8S_AVAILABLE = True
try:
    from kubernetes.client import CoreV1Api  # noqa: WPS433, F401
except ImportError:
    _K8S_AVAILABLE = False

# ── Pricing dataclass ─────────────────────────────────────────


@dataclass(frozen=True)
class GKEPricing:
    """Hourly Autopilot pricing rates for a single GCP region.

    All rates are per unit:
      - ``cpu_per_vcpu_hour``: USD per vCPU per hour
      - ``ram_per_gib_hour``:  USD per GiB of RAM per hour
      - ``ephemeral_per_gib_hour``: USD per GiB of ephemeral storage per hour
    """

    cpu_per_vcpu_hour: float
    ram_per_gib_hour: float
    ephemeral_per_gib_hour: float


# ── Regional pricing table ────────────────────────────────────
# Source: https://cloud.google.com/kubernetes-engine/pricing (Autopilot)
# Prices in USD as of 2024. Expand this dict to add more regions.

AUTOPILOT_PRICING: dict[str, GKEPricing] = {
    # Canada / Montréal — northamerica-northeast1
    "northamerica-northeast1": GKEPricing(
        cpu_per_vcpu_hour=0.0511,
        ram_per_gib_hour=0.0056,
        ephemeral_per_gib_hour=0.000063,
    ),
    # US (Iowa) — us-central1
    "us-central1": GKEPricing(
        cpu_per_vcpu_hour=0.0485,
        ram_per_gib_hour=0.0052,
        ephemeral_per_gib_hour=0.000054,
    ),
    # US (Virginia) — us-east4
    "us-east4": GKEPricing(
        cpu_per_vcpu_hour=0.0510,
        ram_per_gib_hour=0.0056,
        ephemeral_per_gib_hour=0.000063,
    ),
    # Europe (Belgium) — europe-west1
    "europe-west1": GKEPricing(
        cpu_per_vcpu_hour=0.0534,
        ram_per_gib_hour=0.0059,
        ephemeral_per_gib_hour=0.000066,
    ),
    # Asia Pacific (Tokyo) — asia-northeast1
    "asia-northeast1": GKEPricing(
        cpu_per_vcpu_hour=0.0613,
        ram_per_gib_hour=0.0068,
        ephemeral_per_gib_hour=0.000076,
    ),
}


# ── Unit parsers ──────────────────────────────────────────────


def parse_cpu(value: str | None) -> float:
    """Parse a Kubernetes CPU quantity string to vCPUs.

    Supports:
      - Millicores: ``"100m"`` → 0.1 vCPU
      - Whole / decimal cores: ``"0.5"`` → 0.5 vCPU, ``"2"`` → 2.0 vCPU

    Args:
        value: CPU quantity string from a Kubernetes resource spec, or None.

    Returns:
        Float vCPU count. Returns 0.0 for None / unparseable input.

    Examples:
        >>> parse_cpu("100m")
        0.1
        >>> parse_cpu("2")
        2.0
        >>> parse_cpu("500m")
        0.5
    """
    if not value:
        return 0.0
    value = value.strip()
    if value.endswith("m"):
        try:
            return float(value[:-1]) / 1000.0
        except ValueError:
            logger.debug("Unparseable CPU millicores value: %r", value)
            return 0.0
    try:
        return float(value)
    except ValueError:
        logger.debug("Unparseable CPU value: %r", value)
        return 0.0


def parse_memory(value: str | None) -> float:
    """Parse a Kubernetes memory quantity string to GiB.

    Supports binary suffixes (Ki, Mi, Gi) and decimal suffixes (K, M, G).

    Args:
        value: Memory quantity string from a Kubernetes resource spec, or None.

    Returns:
        Float GiB value. Returns 0.0 for None / unparseable input.

    Examples:
        >>> parse_memory("128Mi")
        0.125
        >>> parse_memory("1Gi")
        1.0
        >>> parse_memory("24000Mi")
        23.4375
        >>> parse_memory("512")
        0.000000476837158203125
    """
    if not value:
        return 0.0
    value = value.strip()

    _SUFFIXES: dict[str, float] = {
        "Ki": 1.0 / 1024.0,         # KiB → GiB
        "Mi": 1.0 / 1024.0,         # MiB → GiB  (divide by 1024 from MiB)
        "Gi": 1.0,                   # GiB → GiB
        "Ti": 1024.0,                # TiB → GiB
        "K":  1_000 / (1024.0 ** 3),
        "M":  1_000_000 / (1024.0 ** 3),
        "G":  1_000_000_000 / (1024.0 ** 3),
        "T":  1_000_000_000_000 / (1024.0 ** 3),
    }

    # Binary suffixes first (longer match wins)
    for suffix in ("Ti", "Gi", "Mi", "Ki", "T", "G", "M", "K"):
        if value.endswith(suffix):
            numeric_part = value[: -len(suffix)]
            try:
                raw = float(numeric_part)
            except ValueError:
                logger.debug("Unparseable memory value: %r", value)
                return 0.0
            # Special-case MiB: raw is in MiB, convert to GiB
            if suffix == "Mi":
                return raw / 1024.0
            if suffix == "Ki":
                return raw / (1024.0 * 1024.0)
            return raw * _SUFFIXES[suffix]

    # Plain bytes (no suffix)
    try:
        return float(value) / (1024.0 ** 3)
    except ValueError:
        logger.debug("Unparseable memory value (no suffix): %r", value)
        return 0.0


def parse_ephemeral(value: str | None) -> float:
    """Parse a Kubernetes ephemeral-storage quantity to GiB.

    Delegates to :func:`parse_memory` since the same suffix rules apply.
    """
    return parse_memory(value)


# ── Cost calculation ──────────────────────────────────────────


def calculate_resource_cost(
    quantity: float,
    hourly_rate: float,
    hours_per_month: float = _HOURS_PER_MONTH,
) -> float:
    """Compute monthly cost for a single resource dimension.

    Args:
        quantity: Amount of the resource (vCPUs, GiB, etc.).
        hourly_rate: Cost per unit per hour in USD.
        hours_per_month: Hours in a billing month (default 730).

    Returns:
        Monthly cost in USD.
    """
    return quantity * hourly_rate * hours_per_month


def calculate_workload_cost(
    cpu_requests: float,
    memory_requests_gib: float,
    ephemeral_requests_gib: float,
    pricing: GKEPricing,
    cpu_usage: float | None = None,
    memory_usage_gib: float | None = None,
    ephemeral_usage_gib: float | None = None,
) -> dict[str, Any]:
    """Calculate cost estimates for a single workload.

    Args:
        cpu_requests: Total vCPU requests across all containers and replicas.
        memory_requests_gib: Total memory requests in GiB.
        ephemeral_requests_gib: Total ephemeral storage requests in GiB.
        pricing: Regional pricing to apply.
        cpu_usage: Actual avg CPU usage (vCPUs); None if unavailable.
        memory_usage_gib: Actual avg memory usage (GiB); None if unavailable.
        ephemeral_usage_gib: Actual avg ephemeral usage (GiB); None if unavailable.

    Returns:
        Dict with keys:
          ``resource_costs``: list of per-resource cost dicts
          ``total_request_cost_usd``: float
          ``total_usage_cost_usd``: float | None
          ``total_waste_usd``: float | None
    """
    from vaig.skills.service_health.schema import GKEResourceCost  # noqa: WPS433

    # Each tuple: (resource_type, requests, usage, hourly_rate)
    resource_specs: list[tuple[str, float, float | None, float]] = [
        ("cpu", cpu_requests, cpu_usage, pricing.cpu_per_vcpu_hour),
        ("memory", memory_requests_gib, memory_usage_gib, pricing.ram_per_gib_hour),
        ("ephemeral", ephemeral_requests_gib, ephemeral_usage_gib, pricing.ephemeral_per_gib_hour),
    ]

    result_costs: list[GKEResourceCost] = []
    total_request = 0.0
    total_usage: float | None = 0.0

    for res_type, req_qty, use_qty, rate in resource_specs:
        req_cost = calculate_resource_cost(req_qty, rate)
        total_request += req_cost

        if use_qty is not None and total_usage is not None:
            use_cost = calculate_resource_cost(use_qty, rate)
            total_usage += use_cost
            waste: float | None = req_cost - use_cost
        else:
            use_cost = None
            waste = None
            total_usage = None  # any missing usage makes total unavailable

        result_costs.append(
            GKEResourceCost(
                resource_type=res_type,
                requests=req_qty,
                usage=use_qty,
                request_cost_usd=req_cost,
                usage_cost_usd=use_cost,
                waste_cost_usd=waste,
            )
        )

    total_waste = (total_request - total_usage) if total_usage is not None else None

    return {
        "resource_costs": result_costs,
        "total_request_cost_usd": total_request,
        "total_usage_cost_usd": total_usage,
        "total_waste_usd": total_waste,
    }


# ── K8s resource fetcher ──────────────────────────────────────


def _aggregate_container_requests(
    pods: list[Any],
) -> tuple[float, float, float]:
    """Sum resource requests across all containers in a list of pods.

    Args:
        pods: List of kubernetes ``V1Pod`` objects.

    Returns:
        Tuple of (total_cpu_vcpu, total_memory_gib, total_ephemeral_gib).
    """
    total_cpu = 0.0
    total_memory = 0.0
    total_ephemeral = 0.0

    for pod in pods:
        spec = pod.spec
        if not spec:
            continue
        containers = list(spec.containers or [])
        init_containers = list(spec.init_containers or [])
        # Only bill regular containers (init containers are transient)
        for container in containers:
            requests = {}
            if container.resources and container.resources.requests:
                requests = container.resources.requests
            total_cpu += parse_cpu(requests.get("cpu"))
            total_memory += parse_memory(requests.get("memory"))
            total_ephemeral += parse_ephemeral(requests.get("ephemeral-storage"))

        # suppress unused variable warning
        _ = init_containers

    return total_cpu, total_memory, total_ephemeral


def fetch_workload_costs(
    gke_config: GKEConfig,
    namespaces: list[str] | None = None,
) -> GKECostReport:
    """Fetch resource requests from the K8s API and build a GKECostReport.

    This is the top-level entry point for the cost estimation pipeline.
    It:
    1. Detects whether the cluster is Autopilot (required for per-workload billing).
    2. Looks up regional pricing from :data:`AUTOPILOT_PRICING`.
    3. Lists all pods in the target namespaces and sums container requests.
    4. Returns a :class:`GKECostReport` with per-workload breakdowns.

    Args:
        gke_config: GKE configuration (project_id, location, cluster_name).
        namespaces: Optional list of namespaces to include. When None, all
            non-system namespaces are used.

    Returns:
        A :class:`GKECostReport` instance. If the cluster type cannot be
        determined or the cluster is Standard, ``supported=False`` is set
        with an explanatory ``unsupported_reason``.
    """
    from vaig.skills.service_health.schema import (  # noqa: WPS433
        GKECostReport,
        GKEWorkloadCost,
    )

    region = gke_config.location or ""

    # ── 1. Autopilot detection ─────────────────────────────────
    is_autopilot = detect_autopilot(gke_config)

    if is_autopilot is None:
        return GKECostReport(
            cluster_type="unknown",
            region=region,
            supported=False,
            unsupported_reason="Could not determine cluster type (Autopilot detection failed).",
        )

    if not is_autopilot:
        return GKECostReport(
            cluster_type="standard",
            region=region,
            supported=False,
            unsupported_reason="Standard clusters bill at the node level — per-workload estimation is N/A.",
        )

    # ── 2. Pricing lookup ──────────────────────────────────────
    pricing = AUTOPILOT_PRICING.get(region)
    if pricing is None:
        return GKECostReport(
            cluster_type="autopilot",
            region=region,
            supported=False,
            unsupported_reason=f"Region '{region}' is not in the pricing table. Supported: {', '.join(AUTOPILOT_PRICING)}.",
        )

    # ── 3. K8s API — list pods ─────────────────────────────────
    if not _K8S_AVAILABLE:
        return GKECostReport(
            cluster_type="autopilot",
            region=region,
            supported=False,
            unsupported_reason="kubernetes client library not installed (pip install vertex-ai-toolkit[live]).",
        )

    try:
        clients = _create_k8s_clients(gke_config)
    except Exception as exc:  # noqa: BLE001
        logger.warning("GKE cost estimation: failed to create K8s clients: %s", exc)
        return GKECostReport(
            cluster_type="autopilot",
            region=region,
            supported=False,
            unsupported_reason=f"Failed to connect to cluster: {exc}",
        )

    from vaig.tools.base import ToolResult as _ToolResult  # noqa: WPS433

    if isinstance(clients, _ToolResult):
        return GKECostReport(
            cluster_type="autopilot",
            region=region,
            supported=False,
            unsupported_reason=f"Failed to connect to cluster: {clients.output}",
        )

    core_v1, _apps_v1, _custom, _api_client = clients

    # Resolve namespace list
    if namespaces is None:
        try:
            ns_list = core_v1.list_namespace()
            _system_prefixes = ("kube-", "istio-", "gke-", "gmp-", "cert-manager")
            namespaces = [
                ns.metadata.name
                for ns in ns_list.items
                if not any(ns.metadata.name.startswith(p) for p in _system_prefixes)
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("GKE cost estimation: failed to list namespaces: %s", exc)
            namespaces = []

    # ── 4. Per-namespace workload cost aggregation ─────────────
    # Group pods by (namespace, workload_name) — use the owner reference
    # label (app / app.kubernetes.io/name) as workload identifier.
    workload_pods: dict[tuple[str, str], list[Any]] = {}

    for ns in namespaces:
        try:
            pod_list = core_v1.list_namespaced_pod(ns)
        except Exception as exc:  # noqa: BLE001
            logger.warning("GKE cost estimation: failed to list pods in ns=%s: %s", ns, exc)
            continue

        for pod in pod_list.items:
            # Skip completed / failed pods
            phase = pod.status.phase if pod.status else None
            if phase in ("Succeeded", "Failed"):
                continue

            # Determine workload name from owner references
            workload_name = _get_workload_name(pod)
            key = (ns, workload_name)
            workload_pods.setdefault(key, []).append(pod)

    workloads: list[GKEWorkloadCost] = []
    total_request = 0.0
    total_usage: float | None = 0.0  # stays None if any workload is missing

    for (ns, wl_name), pods in sorted(workload_pods.items()):
        cpu_req, mem_req, eph_req = _aggregate_container_requests(pods)

        cost_data = calculate_workload_cost(
            cpu_requests=cpu_req,
            memory_requests_gib=mem_req,
            ephemeral_requests_gib=eph_req,
            pricing=pricing,
        )

        workloads.append(
            GKEWorkloadCost(
                namespace=ns,
                workload_name=wl_name,
                resource_costs=cost_data["resource_costs"],
                total_request_cost_usd=cost_data["total_request_cost_usd"],
                total_usage_cost_usd=cost_data["total_usage_cost_usd"],
                total_waste_usd=cost_data["total_waste_usd"],
            )
        )

        total_request += cost_data["total_request_cost_usd"] or 0.0
        if cost_data["total_usage_cost_usd"] is not None and total_usage is not None:
            total_usage += cost_data["total_usage_cost_usd"]
        else:
            total_usage = None

    total_savings = (total_request - total_usage) if total_usage is not None else None

    return GKECostReport(
        cluster_type="autopilot",
        region=region,
        supported=True,
        workloads=workloads,
        total_request_cost_usd=total_request,
        total_usage_cost_usd=total_usage,
        total_savings_usd=total_savings,
    )


def _get_workload_name(pod: Any) -> str:
    """Extract a human-readable workload name from a pod's owner references.

    Strips ReplicaSet hash suffixes so pods from the same Deployment group
    together (e.g. ``my-app-59967f9ccc`` → ``my-app``).

    Args:
        pod: A kubernetes ``V1Pod`` object.

    Returns:
        Workload name string. Falls back to the pod name if no owner found.
    """
    import re  # noqa: WPS433

    meta = pod.metadata
    if not meta:
        return "unknown"

    # Prefer owner references (Deployment → ReplicaSet → Pod)
    owners = meta.owner_references or []
    for owner in owners:
        if owner.kind in ("Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob"):
            return str(owner.name)
        if owner.kind == "ReplicaSet":
            # Strip the hash suffix appended by the Deployment controller
            return str(re.sub(r"-[a-z0-9]{5,10}$", "", owner.name))

    # Fall back to pod labels
    labels = meta.labels or {}
    for label_key in ("app", "app.kubernetes.io/name", "app.kubernetes.io/instance"):
        if label_key in labels:
            return str(labels[label_key])

    # Last resort: pod name minus the hash suffix
    pod_name = meta.name or "unknown"
    return str(re.sub(r"-[a-z0-9]{5,10}-[a-z0-9]{4,7}$", "", pod_name))

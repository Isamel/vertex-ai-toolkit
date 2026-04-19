"""GKE Autopilot workload cost estimation.

Computes estimated monthly costs for GKE Autopilot workloads based on
Kubernetes resource *requests* (CPU, memory, ephemeral storage).

v2 enhancements:
- Fetches actual usage metrics from Cloud Monitoring when available, populating
  ``total_usage_cost_usd`` and ``total_waste_usd`` fields.
- Computes per-container cost breakdowns stored in ``GKEWorkloadCost.containers``.
- Aggregates per-namespace summaries into ``GKECostReport.namespace_summaries``.

When Cloud Monitoring is unavailable, v1 behavior is preserved:
``total_usage_cost_usd``, ``total_waste_usd`` and per-container usage/waste
fields will be ``None``, and the UI will display "N/A". Containers are still
reported with their request costs derived from K8s resource specs.

Cost model:
    monthly_cost = resource_quantity * hourly_rate_per_unit * 730 hours/month

Only Autopilot clusters are supported; Standard clusters bill at the node
level (not per-workload), so per-workload estimation would be misleading.

Regional pricing tables are maintained in this module. Only regions present
in the table are supported; unknown regions degrade gracefully.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vaig.core.config import DatadogAPIConfig, GKEConfig
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
# Source:      https://cloud.google.com/kubernetes-engine/pricing (Autopilot)
# Last update: 2026-03-26
# Compute class: Autopilot Balanced Pod (default for Autopilot clusters)
#
# NOTE: These are best-effort estimates derived from GCP Cloud Billing SKU data
# and regional pricing multipliers.  Actual prices may differ — GCP updates
# rates without notice.  Always verify against the official pricing page or the
# Cloud Billing API before making financial decisions.
#
# Regions not present in this table will cause fetch_workload_costs() to return
# supported=False with an "unknown region" message.  Add new entries here to
# extend coverage.

AUTOPILOT_PRICING: dict[str, GKEPricing] = {
    # ── North America ─────────────────────────────────────────────────────────

    # Canada / Montréal — northamerica-northeast1
    "northamerica-northeast1": GKEPricing(
        cpu_per_vcpu_hour=0.0511,
        ram_per_gib_hour=0.0056,
        ephemeral_per_gib_hour=0.000063,
    ),
    # Canada / Toronto — northamerica-northeast2
    "northamerica-northeast2": GKEPricing(
        cpu_per_vcpu_hour=0.0520,
        ram_per_gib_hour=0.0058,
        ephemeral_per_gib_hour=0.000066,
    ),
    # US (Iowa) — us-central1
    "us-central1": GKEPricing(
        cpu_per_vcpu_hour=0.0485,
        ram_per_gib_hour=0.0052,
        ephemeral_per_gib_hour=0.000054,
    ),
    # US (South Carolina) — us-east1
    "us-east1": GKEPricing(
        cpu_per_vcpu_hour=0.0508,
        ram_per_gib_hour=0.0056,
        ephemeral_per_gib_hour=0.000063,
    ),
    # US (Northern Virginia) — us-east4
    "us-east4": GKEPricing(
        cpu_per_vcpu_hour=0.0510,
        ram_per_gib_hour=0.0056,
        ephemeral_per_gib_hour=0.000063,
    ),
    # US (Oregon) — us-west1
    "us-west1": GKEPricing(
        cpu_per_vcpu_hour=0.0489,
        ram_per_gib_hour=0.0052,
        ephemeral_per_gib_hour=0.000056,
    ),
    # US (Los Angeles) — us-west2
    "us-west2": GKEPricing(
        cpu_per_vcpu_hour=0.0520,
        ram_per_gib_hour=0.0058,
        ephemeral_per_gib_hour=0.000066,
    ),
    # US (Las Vegas) — us-west4
    "us-west4": GKEPricing(
        cpu_per_vcpu_hour=0.0511,
        ram_per_gib_hour=0.0056,
        ephemeral_per_gib_hour=0.000063,
    ),
    # US (Dallas) — us-south1
    "us-south1": GKEPricing(
        cpu_per_vcpu_hour=0.0506,
        ram_per_gib_hour=0.0055,
        ephemeral_per_gib_hour=0.000062,
    ),

    # ── Europe ────────────────────────────────────────────────────────────────

    # Europe (Belgium) — europe-west1
    "europe-west1": GKEPricing(
        cpu_per_vcpu_hour=0.0534,
        ram_per_gib_hour=0.0059,
        ephemeral_per_gib_hour=0.000066,
    ),
    # Europe (London) — europe-west2
    "europe-west2": GKEPricing(
        cpu_per_vcpu_hour=0.0539,
        ram_per_gib_hour=0.0059,
        ephemeral_per_gib_hour=0.000066,
    ),
    # Europe (Frankfurt) — europe-west3
    "europe-west3": GKEPricing(
        cpu_per_vcpu_hour=0.0540,
        ram_per_gib_hour=0.0060,
        ephemeral_per_gib_hour=0.000066,
    ),
    # Europe (Netherlands) — europe-west4
    "europe-west4": GKEPricing(
        cpu_per_vcpu_hour=0.0534,
        ram_per_gib_hour=0.0059,
        ephemeral_per_gib_hour=0.000066,
    ),
    # Europe (Zurich) — europe-west6
    "europe-west6": GKEPricing(
        cpu_per_vcpu_hour=0.0574,
        ram_per_gib_hour=0.0063,
        ephemeral_per_gib_hour=0.000070,
    ),
    # Europe (Paris) — europe-west9
    "europe-west9": GKEPricing(
        cpu_per_vcpu_hour=0.0540,
        ram_per_gib_hour=0.0060,
        ephemeral_per_gib_hour=0.000066,
    ),
    # Europe (Finland) — europe-north1
    "europe-north1": GKEPricing(
        cpu_per_vcpu_hour=0.0520,
        ram_per_gib_hour=0.0058,
        ephemeral_per_gib_hour=0.000066,
    ),

    # ── Asia-Pacific ──────────────────────────────────────────────────────────

    # Asia (Taiwan) — asia-east1
    "asia-east1": GKEPricing(
        cpu_per_vcpu_hour=0.0586,
        ram_per_gib_hour=0.0068,
        ephemeral_per_gib_hour=0.000075,
    ),
    # Asia (Hong Kong) — asia-east2
    "asia-east2": GKEPricing(
        cpu_per_vcpu_hour=0.0660,
        ram_per_gib_hour=0.0072,
        ephemeral_per_gib_hour=0.000080,
    ),
    # Asia (Tokyo) — asia-northeast1
    "asia-northeast1": GKEPricing(
        cpu_per_vcpu_hour=0.0613,
        ram_per_gib_hour=0.0068,
        ephemeral_per_gib_hour=0.000076,
    ),
    # Asia (Osaka) — asia-northeast2
    "asia-northeast2": GKEPricing(
        cpu_per_vcpu_hour=0.0601,
        ram_per_gib_hour=0.0068,
        ephemeral_per_gib_hour=0.000075,
    ),
    # Asia (Seoul) — asia-northeast3
    "asia-northeast3": GKEPricing(
        cpu_per_vcpu_hour=0.0620,
        ram_per_gib_hour=0.0072,
        ephemeral_per_gib_hour=0.000080,
    ),
    # Asia (Mumbai) — asia-south1
    "asia-south1": GKEPricing(
        cpu_per_vcpu_hour=0.0598,
        ram_per_gib_hour=0.0068,
        ephemeral_per_gib_hour=0.000075,
    ),
    # Asia (Delhi) — asia-south2
    "asia-south2": GKEPricing(
        cpu_per_vcpu_hour=0.0601,
        ram_per_gib_hour=0.0068,
        ephemeral_per_gib_hour=0.000076,
    ),
    # Asia (Singapore) — asia-southeast1
    "asia-southeast1": GKEPricing(
        cpu_per_vcpu_hour=0.0601,
        ram_per_gib_hour=0.0068,
        ephemeral_per_gib_hour=0.000076,
    ),
    # Asia (Jakarta) — asia-southeast2
    "asia-southeast2": GKEPricing(
        cpu_per_vcpu_hour=0.0630,
        ram_per_gib_hour=0.0072,
        ephemeral_per_gib_hour=0.000080,
    ),
    # Australia (Sydney) — australia-southeast1
    "australia-southeast1": GKEPricing(
        cpu_per_vcpu_hour=0.0629,
        ram_per_gib_hour=0.0072,
        ephemeral_per_gib_hour=0.000080,
    ),
    # Australia (Melbourne) — australia-southeast2
    "australia-southeast2": GKEPricing(
        cpu_per_vcpu_hour=0.0629,
        ram_per_gib_hour=0.0072,
        ephemeral_per_gib_hour=0.000080,
    ),

    # ── South America ─────────────────────────────────────────────────────────

    # South America (São Paulo) — southamerica-east1
    "southamerica-east1": GKEPricing(
        cpu_per_vcpu_hour=0.0727,
        ram_per_gib_hour=0.0080,
        ephemeral_per_gib_hour=0.000090,
    ),
    # South America (Santiago) — southamerica-west1
    "southamerica-west1": GKEPricing(
        cpu_per_vcpu_hour=0.0731,
        ram_per_gib_hour=0.0081,
        ephemeral_per_gib_hour=0.000090,
    ),

    # ── Middle East ───────────────────────────────────────────────────────────

    # Middle East (Tel Aviv) — me-west1
    "me-west1": GKEPricing(
        cpu_per_vcpu_hour=0.0650,
        ram_per_gib_hour=0.0072,
        ephemeral_per_gib_hour=0.000080,
    ),
    # Middle East (Doha) — me-central1
    "me-central1": GKEPricing(
        cpu_per_vcpu_hour=0.0681,
        ram_per_gib_hour=0.0074,
        ephemeral_per_gib_hour=0.000082,
    ),
    # Middle East (Dammam) — me-central2
    "me-central2": GKEPricing(
        cpu_per_vcpu_hour=0.0681,
        ram_per_gib_hour=0.0074,
        ephemeral_per_gib_hour=0.000082,
    ),
}


# ── Dynamic pricing lookup ────────────────────────────────────


class PricingLookupResult:
    """Result of a pricing lookup with source tracking."""

    __slots__ = ("pricing", "source")

    def __init__(self, pricing: GKEPricing, source: str) -> None:
        self.pricing = pricing
        self.source = source


def get_autopilot_pricing(
    region: str,
    project_id: str | None = None,
) -> PricingLookupResult | None:
    """Look up Autopilot pricing for *region*, trying dynamic billing API first.

    1. If *project_id* is provided, attempts to fetch live pricing from the
       Cloud Billing Catalog API via :func:`billing.get_dynamic_pricing`.
    2. Falls back to the hardcoded ``AUTOPILOT_PRICING`` table.
    3. Returns ``None`` if the region is not found in either source.

    Args:
        region: GCP region, e.g. ``"us-central1"``.
        project_id: GCP project ID for dynamic pricing lookup. When ``None``,
            skips the billing API and goes straight to hardcoded prices.

    Returns:
        A :class:`PricingLookupResult` with the pricing and its source, or
        ``None`` if the region is unknown.
    """
    # 1. Try dynamic pricing from Cloud Billing API
    if project_id:
        try:
            from vaig.tools.gke.billing import get_dynamic_pricing  # noqa: WPS433

            dynamic = get_dynamic_pricing(project_id=project_id, region=region)
            if dynamic is not None:
                pricing = GKEPricing(
                    cpu_per_vcpu_hour=dynamic.cpu_per_vcpu_hour,
                    ram_per_gib_hour=dynamic.ram_per_gib_hour,
                    ephemeral_per_gib_hour=dynamic.ephemeral_per_gib_hour,
                )
                logger.info("Using dynamic Billing API pricing for region=%s", region)
                return PricingLookupResult(pricing=pricing, source="billing_api")
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("Dynamic pricing lookup failed, falling back to hardcoded: %s", exc)

    # 2. Fallback to hardcoded pricing table
    hardcoded = AUTOPILOT_PRICING.get(region)
    if hardcoded is not None:
        logger.debug("Using hardcoded pricing for region=%s", region)
        return PricingLookupResult(pricing=hardcoded, source="hardcoded_fallback")

    return None


# ── Memory suffix table (module-level constant) ───────────────
# Maps Kubernetes memory suffix → factor to convert raw value to GiB.
# Sorted longest-first so iteration always matches the most specific suffix.

_MEMORY_SUFFIXES: dict[str, float] = {
    "Ti": 1024.0,                             # TiB → GiB
    "Gi": 1.0,                                # GiB → GiB
    "Mi": 1.0 / 1024.0,                       # MiB → GiB
    "Ki": 1.0 / (1024.0 * 1024.0),            # KiB → GiB  (fixed: was 1/1024)
    "T":  1_000_000_000_000 / (1024.0 ** 3),  # TB  → GiB
    "G":  1_000_000_000 / (1024.0 ** 3),      # GB  → GiB
    "M":  1_000_000 / (1024.0 ** 3),          # MB  → GiB
    "K":  1_000 / (1024.0 ** 3),              # KB  → GiB
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

    Supports binary suffixes (Ki, Mi, Gi, Ti) and decimal suffixes (K, M, G, T).

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

    # Iterate suffixes longest-first (Ti/Gi/Mi/Ki before T/G/M/K)
    for suffix, factor in _MEMORY_SUFFIXES.items():
        if value.endswith(suffix):
            numeric_part = value[: -len(suffix)]
            try:
                return float(numeric_part) * factor
            except ValueError:
                logger.debug("Unparseable memory value: %r", value)
                return 0.0

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
    container_requests: dict[str, tuple[float, float, float]] | None = None,
    container_usage: dict[str, Any] | None = None,
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
        container_requests: Optional per-container request breakdown as
            ``{container_name: (cpu_vcpu, memory_gib, ephemeral_gib)}``.
            When provided, per-container ``GKEContainerCost`` objects are included
            in the returned dict under the ``"containers"`` key.
        container_usage: Optional per-container usage metrics as a dict of
            ``ContainerUsageMetrics`` objects keyed by container name.
            When provided, usage and waste costs are computed per container.

    Returns:
        Dict with keys:
          ``resource_costs``: list of per-resource cost dicts (includes ephemeral when provided)
          ``total_request_cost_usd``: float
          ``total_usage_cost_usd``: float | None — derived from tracked dimensions only (CPU +
            memory). Ephemeral storage costs appear in ``resource_costs`` but are excluded from
            totals because ephemeral usage is never available from Cloud Monitoring and therefore
            cannot be validated against request-based costs.
          ``total_waste_usd``: float | None
          ``partial_metrics``: bool — True when some but not all tracked dimensions have usage data
          ``containers``: list of GKEContainerCost (empty if container_requests is None)
    """
    from vaig.skills.service_health.schema import (  # noqa: WPS433
        GKEContainerCost,
        GKEResourceCost,
    )

    # Each tuple: (resource_type, requests, usage, hourly_rate, tracked)
    # ``tracked=False`` means the dimension is excluded from the usage-availability
    # check.  Ephemeral storage is never available from Cloud Monitoring, so it is
    # not tracked — CPU+memory coverage alone is sufficient to declare a workload
    # as having full metrics.  This mirrors the per-container loop below.
    # NOTE: because ephemeral is ``tracked=False``, it is also excluded from
    # ``total_usage_cost_usd``; that field only sums CPU and memory usage costs.
    resource_specs: list[tuple[str, float, float | None, float, bool]] = [
        ("cpu",      cpu_requests,          cpu_usage,          pricing.cpu_per_vcpu_hour,      True),
        ("memory",   memory_requests_gib,   memory_usage_gib,   pricing.ram_per_gib_hour,       True),
        ("ephemeral", ephemeral_requests_gib, ephemeral_usage_gib, pricing.ephemeral_per_gib_hour, False),
    ]

    result_costs: list[GKEResourceCost] = []
    total_request = 0.0
    total_usage = 0.0
    any_usage_available = False   # True if at least one tracked dimension has data
    all_usage_available = True    # False if any tracked dimension is missing data
    partial_metrics = False       # True when some (not all) tracked dimensions have data

    for res_type, req_qty, use_qty, rate, tracked in resource_specs:
        req_cost = calculate_resource_cost(req_qty, rate)
        total_request += req_cost

        if use_qty is not None:
            use_cost_val: float = calculate_resource_cost(use_qty, rate)
            if tracked:
                total_usage += use_cost_val
                any_usage_available = True
            use_cost: float | None = use_cost_val
            waste: float | None = req_cost - use_cost_val
        else:
            use_cost = None
            waste = None
            if tracked:
                all_usage_available = False  # this tracked dimension is missing

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

    # Expose usage total whenever at least one dimension has data (partial is better than None)
    if any_usage_available:
        final_usage: float | None = total_usage
        total_waste: float | None = (total_request - total_usage) if all_usage_available else None
        partial_metrics = not all_usage_available
    else:
        final_usage = None
        total_waste = None
        partial_metrics = False

    # ── Per-container cost breakdown ──────────────────────────
    containers: list[GKEContainerCost] = []

    if container_requests:
        for c_name, (c_cpu, c_mem, c_eph) in container_requests.items():
            c_usage = container_usage.get(c_name) if container_usage else None

            c_cpu_usage: float | None = c_usage.avg_cpu_cores if c_usage is not None else None
            c_mem_usage: float | None = c_usage.avg_memory_gib if c_usage is not None else None

            # Ephemeral usage is not available from Cloud Monitoring; it is excluded from
            # the usage-availability check so that CPU+memory usage can still produce a total.
            c_specs: list[tuple[str, float, float | None, float, bool]] = [
                ("cpu",      c_cpu, c_cpu_usage, pricing.cpu_per_vcpu_hour,      True),
                ("memory",   c_mem, c_mem_usage, pricing.ram_per_gib_hour,       True),
                ("ephemeral", c_eph, None,        pricing.ephemeral_per_gib_hour, False),
            ]

            c_resource_costs: list[GKEResourceCost] = []
            c_total_request = 0.0
            c_total_usage = 0.0
            c_all_usage_available = True

            for c_res_type, c_req_qty, c_use_qty, c_rate, c_tracked in c_specs:
                c_req_cost = calculate_resource_cost(c_req_qty, c_rate)
                c_total_request += c_req_cost

                if c_use_qty is not None:
                    c_use_val: float = calculate_resource_cost(c_use_qty, c_rate)
                    if c_tracked:
                        c_total_usage += c_use_val
                    c_use_cost: float | None = c_use_val
                    c_waste: float | None = c_req_cost - c_use_val
                else:
                    c_use_cost = None
                    c_waste = None
                    if c_tracked:
                        c_all_usage_available = False

                c_resource_costs.append(
                    GKEResourceCost(
                        resource_type=c_res_type,
                        requests=c_req_qty,
                        usage=c_use_qty,
                        request_cost_usd=c_req_cost,
                        usage_cost_usd=c_use_cost,
                        waste_cost_usd=c_waste,
                    )
                )

            c_final_usage: float | None = c_total_usage if c_all_usage_available else None
            c_final_waste: float | None = (
                (c_total_request - c_final_usage) if c_final_usage is not None else None
            )

            containers.append(
                GKEContainerCost(
                    container_name=c_name,
                    resource_costs=c_resource_costs,
                    total_request_cost_usd=c_total_request,
                    total_usage_cost_usd=c_final_usage,
                    total_waste_usd=c_final_waste,
                )
            )

    return {
        "resource_costs": result_costs,
        "total_request_cost_usd": total_request,
        "total_usage_cost_usd": final_usage,
        "total_waste_usd": total_waste,
        "containers": containers,
        "partial_metrics": partial_metrics,
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
        # Only bill regular containers (init containers are transient)
        for container in containers:
            requests = {}
            if container.resources and container.resources.requests:
                requests = container.resources.requests
            total_cpu += parse_cpu(requests.get("cpu"))
            total_memory += parse_memory(requests.get("memory"))
            total_ephemeral += parse_ephemeral(requests.get("ephemeral-storage"))

    return total_cpu, total_memory, total_ephemeral


def _aggregate_container_requests_per_container(
    pods: list[Any],
) -> dict[str, tuple[float, float, float]]:
    """Sum resource requests per container name across a list of pods.

    Pods belonging to the same workload may have different replicas; requests
    from the same container name are summed across all replicas so the result
    represents the total requested capacity (not per-replica).

    Args:
        pods: List of kubernetes ``V1Pod`` objects.

    Returns:
        Dict mapping ``container_name`` →
        ``(total_cpu_vcpu, total_memory_gib, total_ephemeral_gib)``.
        Init containers are excluded (they are transient and not billed
        continuously like regular containers).
    """
    per_container: dict[str, list[float]] = {}  # name → [cpu, mem, eph]

    for pod in pods:
        spec = pod.spec
        if not spec:
            continue
        containers = list(spec.containers or [])
        for container in containers:
            requests = {}
            if container.resources and container.resources.requests:
                requests = container.resources.requests
            c_name_raw = container.name
            c_name: str = c_name_raw if isinstance(c_name_raw, str) and c_name_raw else "unknown"
            cpu = parse_cpu(requests.get("cpu"))
            mem = parse_memory(requests.get("memory"))
            eph = parse_ephemeral(requests.get("ephemeral-storage"))
            if c_name not in per_container:
                per_container[c_name] = [0.0, 0.0, 0.0]
            per_container[c_name][0] += cpu
            per_container[c_name][1] += mem
            per_container[c_name][2] += eph

    return {name: (vals[0], vals[1], vals[2]) for name, vals in per_container.items()}


def _compute_namespace_summaries(
    workloads: list[Any],
) -> dict[str, Any]:
    """Aggregate per-workload costs into per-namespace summaries.

    Iterates over all ``GKEWorkloadCost`` objects and sums request/usage/waste
    costs per namespace. A namespace's ``total_usage_cost_usd`` is ``None`` if
    any workload in that namespace is missing usage data (to avoid misleading
    partial totals).

    Args:
        workloads: List of ``GKEWorkloadCost`` objects from a cost report.

    Returns:
        Dict mapping namespace name → ``GKENamespaceSummary``.
    """
    from vaig.skills.service_health.schema import GKENamespaceSummary  # noqa: WPS433

    # Accumulators: ns → [total_request, total_usage (None if any missing), total_waste]
    ns_request: dict[str, float] = {}
    ns_usage: dict[str, float | None] = {}
    ns_waste: dict[str, float | None] = {}

    for wl in workloads:
        ns = wl.namespace
        req = wl.total_request_cost_usd or 0.0
        use = wl.total_usage_cost_usd  # may be None
        waste = wl.total_waste_usd     # may be None

        ns_request[ns] = ns_request.get(ns, 0.0) + req

        if ns not in ns_usage:
            # First workload for this namespace
            ns_usage[ns] = use
            ns_waste[ns] = waste
        elif ns_usage[ns] is None or use is None:
            # Any workload missing usage → namespace total becomes None
            ns_usage[ns] = None
            ns_waste[ns] = None
        else:
            ns_usage[ns] = (ns_usage[ns] or 0.0) + use
            ns_waste[ns] = (ns_waste[ns] or 0.0) + (waste or 0.0)

    result: dict[str, Any] = {}
    for ns, total_req in ns_request.items():
        result[ns] = GKENamespaceSummary(
            namespace=ns,
            total_request_cost_usd=total_req,
            total_usage_cost_usd=ns_usage.get(ns),
            total_waste_usd=ns_waste.get(ns),
        )
    return result


# ── Multi-source usage resolution (Layers 1-4) ───────────────


@dataclass
class UsageResolutionResult:
    """Result from :func:`_resolve_workload_usage`.

    Attributes:
        usage: workload_name → :class:`WorkloadUsageMetrics` dict.
            Empty dict when all layers returned no data.
        source: Which layer provided the data.
            Values: ``"cloud_monitoring_60m"``, ``"cloud_monitoring_24h"``,
            ``"metrics_server"``, ``"datadog"``, ``"none"``.
        freshness: Human-readable freshness label for the source.
            Values: ``"60m_avg"``, ``"24h_avg"``, ``"snapshot"``, ``"1h_avg"``, ``"none"``.
    """

    usage: dict[str, Any] = field(default_factory=dict)
    source: str = "none"
    freshness: str = "none"


def _resolve_workload_usage(
    namespace: str,
    workload_pod_names: dict[str, list[str]],
    gke_config: GKEConfig,
    datadog_config: DatadogAPIConfig | None = None,
) -> UsageResolutionResult:
    """Resolve workload usage metrics through the 4-layer pipeline.

    Layers are tried in order; the first one returning data wins:

    L1 — Cloud Monitoring 60m (+ 24h retry on empty)
    L2 — Kubernetes Metrics Server (live snapshot)
    L3 — Datadog (only when ``datadog_config.enabled == True``)
    L4 — No real data available (returns empty result; caller uses requests as fallback)

    Args:
        namespace: Kubernetes namespace.
        workload_pod_names: Workload → list[pod_name] mapping.
        gke_config: GKE cluster configuration.
        datadog_config: Optional Datadog configuration.  When ``None`` or
            ``enabled=False``, Layer 3 is skipped.

    Returns:
        :class:`UsageResolutionResult` with the best available data.
    """
    # ── Layer 1: Cloud Monitoring ─────────────────────────────
    try:
        from vaig.tools.gke.monitoring import _get_monitoring_usage_with_retry  # noqa: WPS433
        usage_l1, window = _get_monitoring_usage_with_retry(
            namespace,
            workload_pod_names,
            gke_config=gke_config,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("_resolve_workload_usage: L1 (monitoring) failed for ns=%s: %s", namespace, exc)
        usage_l1, window = {}, "none"

    if usage_l1:
        if window == "60m":
            return UsageResolutionResult(usage=usage_l1, source="cloud_monitoring_60m", freshness="60m_avg")
        return UsageResolutionResult(usage=usage_l1, source="cloud_monitoring_24h", freshness="24h_avg")

    # ── Layer 2: Metrics Server ───────────────────────────────
    try:
        from vaig.tools.gke.metrics_server import get_metrics_server_usage  # noqa: WPS433
        usage_l2 = get_metrics_server_usage(
            namespace,
            workload_pod_names,
            gke_config=gke_config,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("_resolve_workload_usage: L2 (metrics_server) failed for ns=%s: %s", namespace, exc)
        usage_l2 = {}

    if usage_l2:
        return UsageResolutionResult(usage=usage_l2, source="metrics_server", freshness="snapshot")

    # ── Layer 3: Datadog ──────────────────────────────────────
    if datadog_config is not None and datadog_config.enabled:
        try:
            from vaig.tools.gke.datadog_api import get_datadog_workload_usage  # noqa: WPS433
            workload_names = list(workload_pod_names.keys())
            usage_l3 = get_datadog_workload_usage(
                namespace,
                workload_names,
                gke_config=gke_config,
                config=datadog_config,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("_resolve_workload_usage: L3 (datadog) failed for ns=%s: %s", namespace, exc)
            usage_l3 = {}

        if usage_l3:
            return UsageResolutionResult(usage=usage_l3, source="datadog", freshness="1h_avg")

    # ── Layer 4: No real data ─────────────────────────────────
    logger.info(
        "_resolve_workload_usage: all layers returned empty for ns=%s — "
        "will use requests as fallback (waste figures unreliable)",
        namespace,
    )
    return UsageResolutionResult(usage={}, source="none", freshness="none")


def _compute_cost_data_quality(
    workloads: list[Any],
    cross_check_data: dict[str, Any],
) -> Any | None:
    """Compute a :class:`CostDataQuality` summary from resolved workload metrics.

    Inspects the ``metrics_source`` field on each workload to determine the
    dominant source, count fallbacks, and assign a confidence level.

    Args:
        workloads: List of ``GKEWorkloadCost`` objects (after metrics have been
            assigned).  The ``metrics_source`` field drives classification.
        cross_check_data: Reserved for future cross-source discrepancy data.
            Currently unused; pass an empty dict.

    Returns:
        A :class:`CostDataQuality` instance, or ``None`` when the workloads
        list is empty.
    """
    if not workloads:
        return None

    from vaig.skills.service_health.schema import CostDataQuality  # noqa: WPS433

    total = len(workloads)
    source_counts: dict[str, int] = {}
    fallback_count = 0
    for wl in workloads:
        src = getattr(wl, "metrics_source", None) or "none"
        source_counts[src] = source_counts.get(src, 0) + 1
        if src in ("none", "requests_fallback"):
            fallback_count += 1

    coverage_count = total - fallback_count

    # Dominant source = highest count
    primary_source = max(source_counts, key=lambda k: source_counts[k])
    if primary_source == "none":
        primary_source = "requests_fallback"

    # Freshness label derived from primary source
    _freshness_map = {
        "cloud_monitoring_60m": "60m_avg",
        "cloud_monitoring_24h": "24h_avg",
        "metrics_server": "snapshot",
        "datadog": "1h_avg",
        "requests_fallback": "none",
        "none": "none",
    }
    freshness = _freshness_map.get(primary_source, "unknown")

    # Confidence logic:
    # low    → any fallback
    # medium → no fallback but any 24h or datadog
    # high   → all real L1-60m or L2 data
    if fallback_count > 0:
        confidence = "low"
    elif source_counts.get("cloud_monitoring_24h", 0) > 0 or source_counts.get("datadog", 0) > 0:
        confidence = "medium"
    else:
        confidence = "high"

    cross_check_discrepancies: list[str] = list(cross_check_data.get("discrepancies", []))

    return CostDataQuality(
        primary_source=primary_source,
        freshness=freshness,
        coverage_count=coverage_count,
        total_count=total,
        confidence=confidence,
        fallback_count=fallback_count,
        cross_check_discrepancies=cross_check_discrepancies,
    )


def fetch_workload_costs(
    gke_config: GKEConfig,
    namespaces: list[str] | None = None,
) -> GKECostReport:
    """Fetch resource *requests* from the K8s API and build a GKECostReport.

    This is the top-level entry point for the cost estimation pipeline.
    It:
    1. Detects whether the cluster is Autopilot (required for per-workload billing).
    2. Looks up regional pricing from :data:`AUTOPILOT_PRICING`.
    3. Lists all pods in the target namespaces and sums container *requests*.
    4. Optionally fetches actual usage metrics from Cloud Monitoring to populate
       ``total_usage_cost_usd``, ``total_waste_usd``, and per-container breakdowns.
    5. Returns a :class:`GKECostReport` with per-workload cost breakdowns and
       per-namespace aggregated summaries.

    When Cloud Monitoring is unavailable or the query fails, v1 behavior is
    preserved: ``total_usage_cost_usd``, ``total_waste_usd`` and per-container
    fields will be ``None`` / ``[]``, and the UI will display "N/A".

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
    # Provide specific error messages for common failure cases upfront.
    if detect_autopilot is None:
        return GKECostReport(
            cluster_type="unknown",
            region=region,
            supported=False,
            unsupported_reason=(
                "GKE client library not available. "
                "Install the optional dependency: pip install vertex-ai-toolkit[live]."
            ),
        )

    missing_fields = [
        f for f in ("project_id", "location", "cluster_name")
        if not getattr(gke_config, f, None)
    ]
    if missing_fields:
        return GKECostReport(
            cluster_type="unknown",
            region=region,
            supported=False,
            unsupported_reason=(
                f"GKE configuration is incomplete — missing fields: {', '.join(missing_fields)}."
            ),
        )

    try:
        is_autopilot = detect_autopilot(gke_config)
    except Exception as exc:  # noqa: BLE001
        return GKECostReport(
            cluster_type="unknown",
            region=region,
            supported=False,
            unsupported_reason=f"Autopilot detection failed with API error: {exc}",
        )

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
    pricing_result = get_autopilot_pricing(region=region, project_id=gke_config.project_id)
    if pricing_result is None:
        return GKECostReport(
            cluster_type="autopilot",
            region=region,
            supported=False,
            unsupported_reason=f"Could not fetch dynamic pricing from Cloud Billing API and region '{region}' is not in the hardcoded pricing table. Supported: {', '.join(AUTOPILOT_PRICING)}.",
        )
    pricing = pricing_result.pricing
    pricing_source = pricing_result.source

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

    # ── 5. Resolve usage metrics via multi-source pipeline ───────
    # Build workload_pod_names per namespace for all metric queries.
    ns_workload_pod_names: dict[str, dict[str, list[str]]] = {}
    for (ns, wl_name), pods in workload_pods.items():
        pod_names = [
            p.metadata.name
            for p in pods
            if p.metadata and p.metadata.name
        ]
        if pod_names:
            ns_workload_pod_names.setdefault(ns, {}).setdefault(wl_name, []).extend(pod_names)

    # Resolve per-namespace: ns → UsageResolutionResult
    datadog_cfg = getattr(gke_config, "datadog", None)
    ns_resolution: dict[str, UsageResolutionResult] = {}
    for ns, workload_pod_names in ns_workload_pod_names.items():
        ns_resolution[ns] = _resolve_workload_usage(
            ns,
            workload_pod_names,
            gke_config,
            datadog_config=datadog_cfg,
        )

    # Derive overall monitoring_status for backward compat with existing report field
    monitoring_status: str | None = None
    for res in ns_resolution.values():
        if res.source in ("cloud_monitoring_60m", "cloud_monitoring_24h"):
            monitoring_status = "ok"
            break
    if monitoring_status is None and ns_resolution:
        sources = {res.source for res in ns_resolution.values()}
        if "none" in sources and len(sources) == 1:
            monitoring_status = "no_data: all layers returned empty"

    workloads: list[GKEWorkloadCost] = []
    total_request = 0.0
    total_usage: float | None = None
    workloads_with_full_metrics = 0
    workloads_with_partial_metrics = 0
    workloads_without_metrics = 0
    any_estimated = False

    # For cross-check: store L1 + L2 data if both are available
    # ns → {wl_name → (l1_cpu, l2_cpu, l1_mem, l2_mem)}
    # (only populated when we have two independent sources in same ns)
    cross_check_data: dict[str, Any] = {}

    for (ns, wl_name), pods in sorted(workload_pods.items()):
        cpu_req, mem_req, eph_req = _aggregate_container_requests(pods)
        c_requests = _aggregate_container_requests_per_container(pods)

        # Get resolved metrics for this workload
        res = ns_resolution.get(ns, UsageResolutionResult())
        wl_usage_metrics = res.usage.get(wl_name)
        c_usage = wl_usage_metrics.containers if wl_usage_metrics is not None else None

        wl_cpu_usage: float | None = None
        wl_mem_usage: float | None = None
        wl_estimated = False
        wl_source: str | None = None
        wl_freshness: str | None = None

        if c_usage:
            cpu_vals = [c.avg_cpu_cores for c in c_usage.values() if c.avg_cpu_cores is not None]
            mem_vals = [c.avg_memory_gib for c in c_usage.values() if c.avg_memory_gib is not None]
            wl_cpu_usage = sum(cpu_vals) if cpu_vals else None
            wl_mem_usage = sum(mem_vals) if mem_vals else None
            wl_source = res.source if res.source != "none" else None
            wl_freshness = res.freshness if res.freshness != "none" else None
        else:
            # ── Layer 4 fallback: no real data from any source.
            # Estimate usage as equal to requests (100% utilization).
            # ⚠️  This means waste=$0 — figures are UNRELIABLE.
            wl_cpu_usage = cpu_req
            wl_mem_usage = mem_req
            wl_estimated = True
            any_estimated = True
            wl_source = "requests_fallback"
            wl_freshness = None

            if c_requests:
                from vaig.tools.gke.monitoring import ContainerUsageMetrics  # noqa: WPS433
                c_usage = {}
                for c_name, (c_cpu, c_mem, _c_eph) in c_requests.items():
                    c_usage[c_name] = ContainerUsageMetrics(
                        container_name=c_name,
                        avg_cpu_cores=c_cpu,
                        avg_memory_gib=c_mem,
                    )

        cost_data = calculate_workload_cost(
            cpu_requests=cpu_req,
            memory_requests_gib=mem_req,
            ephemeral_requests_gib=eph_req,
            pricing=pricing,
            cpu_usage=wl_cpu_usage,
            memory_usage_gib=wl_mem_usage,
            container_requests=c_requests,
            container_usage=c_usage,
        )

        wl_partial = cost_data["partial_metrics"]

        workloads.append(
            GKEWorkloadCost(
                namespace=ns,
                workload_name=wl_name,
                resource_costs=cost_data["resource_costs"],
                total_request_cost_usd=cost_data["total_request_cost_usd"],
                total_usage_cost_usd=cost_data["total_usage_cost_usd"],
                total_waste_usd=cost_data["total_waste_usd"],
                containers=cost_data["containers"],
                partial_metrics=wl_partial,
                metrics_estimated=wl_estimated,
                metrics_source=wl_source,
                metrics_freshness=wl_freshness,
            )
        )

        total_request += cost_data["total_request_cost_usd"] or 0.0
        wl_usage_cost = cost_data["total_usage_cost_usd"]
        if wl_usage_cost is not None:
            total_usage = (total_usage or 0.0) + wl_usage_cost
            if wl_estimated:
                workloads_without_metrics += 1
            elif wl_partial:
                workloads_with_partial_metrics += 1
            else:
                workloads_with_full_metrics += 1
        else:
            workloads_without_metrics += 1

    total_savings = (total_request - total_usage) if total_usage is not None else None

    # ── 6. Compute CostDataQuality ──────────────────────────────
    cost_data_quality = _compute_cost_data_quality(workloads, cross_check_data)

    # ── 7. Namespace aggregation ───────────────────────────────
    namespace_summaries = _compute_namespace_summaries(workloads)

    return GKECostReport(
        cluster_type="autopilot",
        region=region,
        supported=True,
        workloads=workloads,
        total_request_cost_usd=total_request,
        total_usage_cost_usd=total_usage,
        total_savings_usd=total_savings,
        namespace_summaries=namespace_summaries,
        monitoring_status=monitoring_status,
        workloads_with_full_metrics=workloads_with_full_metrics,
        workloads_with_partial_metrics=workloads_with_partial_metrics,
        workloads_without_metrics=workloads_without_metrics,
        pricing_source=pricing_source,
        metrics_estimated=any_estimated,
        cost_data_quality=cost_data_quality,
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

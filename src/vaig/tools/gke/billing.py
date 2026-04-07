"""Dynamic GKE Autopilot pricing via Google Cloud Billing Catalog API.

Fetches real-time per-region Autopilot pricing from the Cloud Billing API,
with in-memory caching and graceful fallback to the hardcoded
``AUTOPILOT_PRICING`` table.

Architecture
~~~~~~~~~~~~
1. ``get_dynamic_pricing(project_id, region)`` — main entry point.
2. Internally queries the Cloud Billing Catalog API for "Kubernetes Engine"
   SKUs filtered by region and description keywords.
3. Extracts vCPU, RAM, and ephemeral storage hourly rates.
4. Results are cached in-memory per ``(project_id, region)`` tuple for the
   session lifetime (no TTL — billing prices rarely change mid-session).
5. On *any* failure (missing library, API error, auth issue), returns
   ``None`` so the caller can fall back to hardcoded prices.

Dependencies
~~~~~~~~~~~~
Requires ``google-cloud-billing`` (optional — part of ``live`` extras).
"""

from __future__ import annotations

import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)

# ── In-memory pricing cache ──────────────────────────────────
# Key: (project_id, region) → BillingPricingResult
_PRICING_CACHE: dict[tuple[str, str], BillingPricingResult | None] = {}


class BillingPricingResult(NamedTuple):
    """Parsed Autopilot pricing from Cloud Billing API."""

    cpu_per_vcpu_hour: float
    ram_per_gib_hour: float
    ephemeral_per_gib_hour: float


# ── SKU description keywords ────────────────────────────────
# These are the keywords used to identify Autopilot pricing SKUs from the
# Cloud Billing Catalog API.  The service display name is "Kubernetes Engine".
_CPU_KEYWORDS = ("autopilot", "vcpu", "cpu")
_RAM_KEYWORDS = ("autopilot", "ram", "memory")
_EPHEMERAL_KEYWORDS = ("autopilot", "ephemeral", "storage")

# Cloud Billing Catalog service display name for GKE
_GKE_SERVICE_DISPLAY_NAME = "Kubernetes Engine"


def get_dynamic_pricing(
    project_id: str,
    region: str,
) -> BillingPricingResult | None:
    """Fetch Autopilot pricing for *region* using Cloud Billing Catalog API.

    Returns a ``BillingPricingResult`` with hourly rates, or ``None`` if the
    billing API is unavailable or the lookup fails for any reason.

    Results are cached per ``(project_id, region)`` for the session lifetime.

    Args:
        project_id: GCP project ID (used for cache key and future billing
            account lookups).
        region: GCP region, e.g. ``"us-central1"``.

    Returns:
        Pricing tuple or ``None`` on failure.
    """
    cache_key = (project_id, region)

    # Check cache (None is a valid cached "not found" sentinel)
    if cache_key in _PRICING_CACHE:
        cached = _PRICING_CACHE[cache_key]
        if cached is not None:
            logger.debug("Billing pricing cache hit for %s/%s", project_id, region)
        return cached

    result = _fetch_pricing_from_catalog(region)
    _PRICING_CACHE[cache_key] = result
    return result


def _fetch_pricing_from_catalog(region: str) -> BillingPricingResult | None:
    """Query the Cloud Billing Catalog API for Autopilot SKUs in *region*.

    Returns ``None`` on any failure (import, API, parsing).
    """
    try:
        from google.cloud import billing_v1  # noqa: WPS433
    except ImportError:
        logger.debug("google-cloud-billing not installed — skipping dynamic pricing")
        return None

    try:
        client = billing_v1.CloudCatalogClient()

        # 1. Find the GKE service ID
        gke_service_name = _find_gke_service(client)
        if gke_service_name is None:
            logger.warning("Could not find '%s' service in Cloud Billing Catalog", _GKE_SERVICE_DISPLAY_NAME)
            return None

        # 2. List SKUs and filter for Autopilot pricing in this region
        cpu_rate: float | None = None
        ram_rate: float | None = None
        eph_rate: float | None = None

        request = billing_v1.ListSkusRequest(parent=gke_service_name)
        for sku in client.list_skus(request=request):
            # Filter by region
            if not _sku_matches_region(sku, region):
                continue

            desc_lower = sku.description.lower()

            # Match Autopilot pricing SKUs by description keywords
            if cpu_rate is None and _matches_keywords(desc_lower, _CPU_KEYWORDS):
                cpu_rate = _extract_hourly_rate(sku)
            elif ram_rate is None and _matches_keywords(desc_lower, _RAM_KEYWORDS):
                ram_rate = _extract_hourly_rate(sku)
            elif eph_rate is None and _matches_keywords(desc_lower, _EPHEMERAL_KEYWORDS):
                eph_rate = _extract_hourly_rate(sku)

            # Early exit once all three are found
            if cpu_rate is not None and ram_rate is not None and eph_rate is not None:
                break

        if cpu_rate is None or ram_rate is None or eph_rate is None:
            logger.info(
                "Incomplete Autopilot pricing from Billing API for region=%s "
                "(cpu=%s, ram=%s, eph=%s) — falling back to hardcoded",
                region,
                cpu_rate,
                ram_rate,
                eph_rate,
            )
            return None

        logger.info(
            "Dynamic Autopilot pricing for %s: cpu=$%.4f/h, ram=$%.4f/h, eph=$%.6f/h",
            region,
            cpu_rate,
            ram_rate,
            eph_rate,
        )
        return BillingPricingResult(
            cpu_per_vcpu_hour=cpu_rate,
            ram_per_gib_hour=ram_rate,
            ephemeral_per_gib_hour=eph_rate,
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning("Cloud Billing API pricing lookup failed: %s", exc)
        return None


def _find_gke_service(client: object) -> str | None:
    """Find the Cloud Billing service name for Kubernetes Engine.

    Returns the service resource name (e.g.
    ``services/6F81-5844-456A``) or ``None`` if not found.
    """
    from google.cloud import billing_v1  # noqa: WPS433

    assert isinstance(client, billing_v1.CloudCatalogClient)  # noqa: S101
    for svc in client.list_services():
        if svc.display_name == _GKE_SERVICE_DISPLAY_NAME:
            return str(svc.name)
    return None


def _sku_matches_region(sku: object, region: str) -> bool:
    """Check if a SKU applies to the given GCP region."""
    return any(
        sr.lower() == region.lower() or sr.lower() == "global"
        for sr in getattr(sku, "service_regions", [])
    )


def _matches_keywords(description: str, keywords: tuple[str, ...]) -> bool:
    """Return True if *description* contains ALL keywords (case-insensitive)."""
    description_lower = description.lower()
    return all(kw in description_lower for kw in keywords)


def _extract_hourly_rate(sku: object) -> float | None:
    """Extract the hourly USD rate from a SKU's pricing info.

    Cloud Billing stores prices as ``units`` (integer part) + ``nanos``
    (fractional part × 10⁹).  We combine them:
    ``price = units + nanos / 1_000_000_000``

    Returns ``None`` if the pricing structure is unexpected.
    """
    pricing_info = getattr(sku, "pricing_info", [])
    if not pricing_info:
        return None

    pricing_expression = getattr(pricing_info[0], "pricing_expression", None)
    if pricing_expression is None:
        return None

    tiered_rates = getattr(pricing_expression, "tiered_rates", [])
    if not tiered_rates:
        return None

    unit_price = getattr(tiered_rates[0], "unit_price", None)
    if unit_price is None:
        return None

    units = getattr(unit_price, "units", 0) or 0
    nanos = getattr(unit_price, "nanos", 0) or 0

    rate = float(units) + float(nanos) / 1_000_000_000
    return rate if rate > 0 else None


def clear_pricing_cache() -> None:
    """Clear the in-memory pricing cache (useful for testing)."""
    _PRICING_CACHE.clear()

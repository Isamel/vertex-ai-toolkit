"""Tests for the dynamic billing pricing module and pricing integration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from vaig.tools.gke.billing import (
    BillingPricingResult,
    _extract_hourly_rate,
    _matches_keywords,
    _sku_matches_region,
    clear_pricing_cache,
    get_dynamic_pricing,
)
from vaig.tools.gke.cost_estimation import (
    GKEPricing,
    PricingLookupResult,
    get_autopilot_pricing,
)

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_cache() -> None:  # noqa: PT004
    """Clear the billing pricing cache before each test."""
    clear_pricing_cache()


def _make_sku(
    description: str,
    regions: list[str],
    units: int = 0,
    nanos: int = 0,
) -> SimpleNamespace:
    """Build a mock SKU object matching the Cloud Billing API structure."""
    return SimpleNamespace(
        description=description,
        service_regions=regions,
        pricing_info=[
            SimpleNamespace(
                pricing_expression=SimpleNamespace(
                    tiered_rates=[
                        SimpleNamespace(
                            unit_price=SimpleNamespace(units=units, nanos=nanos),
                        ),
                    ],
                ),
            ),
        ],
    )


# ── _matches_keywords ────────────────────────────────────────


class TestMatchesKeywords:
    """Keyword matching logic used to identify Autopilot SKUs."""

    def test_all_keywords_present(self) -> None:
        assert _matches_keywords("autopilot pod vcpu time", ("autopilot", "vcpu"))

    def test_missing_keyword(self) -> None:
        assert not _matches_keywords("autopilot pod ram time", ("autopilot", "vcpu"))

    def test_case_sensitive(self) -> None:
        # Keywords must be lowercase and description should be lowered by caller
        assert not _matches_keywords("Autopilot Pod vCPU", ("autopilot", "vcpu"))


# ── _sku_matches_region ──────────────────────────────────────


class TestSkuMatchesRegion:
    """Region filtering for Cloud Billing SKUs."""

    def test_exact_match(self) -> None:
        sku = SimpleNamespace(service_regions=["us-central1", "us-east1"])
        assert _sku_matches_region(sku, "us-central1")

    def test_case_insensitive(self) -> None:
        sku = SimpleNamespace(service_regions=["US-Central1"])
        assert _sku_matches_region(sku, "us-central1")

    def test_global_matches_any(self) -> None:
        sku = SimpleNamespace(service_regions=["global"])
        assert _sku_matches_region(sku, "us-central1")

    def test_no_match(self) -> None:
        sku = SimpleNamespace(service_regions=["europe-west1"])
        assert not _sku_matches_region(sku, "us-central1")

    def test_empty_regions(self) -> None:
        sku = SimpleNamespace(service_regions=[])
        assert not _sku_matches_region(sku, "us-central1")


# ── _extract_hourly_rate ─────────────────────────────────────


class TestExtractHourlyRate:
    """Price extraction from Cloud Billing SKU pricing info."""

    def test_units_and_nanos(self) -> None:
        sku = _make_sku("test", ["us-central1"], units=0, nanos=35_000_000)
        assert _extract_hourly_rate(sku) == pytest.approx(0.035)

    def test_units_only(self) -> None:
        sku = _make_sku("test", ["us-central1"], units=1, nanos=0)
        assert _extract_hourly_rate(sku) == pytest.approx(1.0)

    def test_combined(self) -> None:
        sku = _make_sku("test", ["us-central1"], units=2, nanos=500_000_000)
        assert _extract_hourly_rate(sku) == pytest.approx(2.5)

    def test_zero_price_returns_none(self) -> None:
        sku = _make_sku("test", ["us-central1"], units=0, nanos=0)
        assert _extract_hourly_rate(sku) is None

    def test_no_pricing_info(self) -> None:
        sku = SimpleNamespace(pricing_info=[])
        assert _extract_hourly_rate(sku) is None

    def test_missing_pricing_expression(self) -> None:
        sku = SimpleNamespace(pricing_info=[SimpleNamespace(pricing_expression=None)])
        assert _extract_hourly_rate(sku) is None


# ── get_dynamic_pricing ──────────────────────────────────────


class TestGetDynamicPricing:
    """Dynamic pricing lookup with API mocking."""

    def test_returns_none_when_billing_not_installed(self) -> None:
        with patch.dict("sys.modules", {"google.cloud.billing_v1": None}):
            # Force re-import failure
            with patch(
                "vaig.tools.gke.billing._fetch_pricing_from_catalog",
                return_value=None,
            ):
                result = get_dynamic_pricing("my-project", "us-central1")
        assert result is None

    def test_cache_hit(self) -> None:
        """Second call with same args returns cached result without re-fetching."""
        expected = BillingPricingResult(0.035, 0.004, 0.00005)
        with patch(
            "vaig.tools.gke.billing._fetch_pricing_from_catalog",
            return_value=expected,
        ) as mock_fetch:
            r1 = get_dynamic_pricing("proj", "us-east1")
            r2 = get_dynamic_pricing("proj", "us-east1")

        assert r1 == expected
        assert r2 == expected
        mock_fetch.assert_called_once()  # Only 1 API call, second is cache hit

    def test_cache_miss_different_region(self) -> None:
        """Different regions trigger separate API calls."""
        expected = BillingPricingResult(0.035, 0.004, 0.00005)
        with patch(
            "vaig.tools.gke.billing._fetch_pricing_from_catalog",
            return_value=expected,
        ) as mock_fetch:
            get_dynamic_pricing("proj", "us-east1")
            get_dynamic_pricing("proj", "us-west1")

        assert mock_fetch.call_count == 2

    def test_none_result_is_cached(self) -> None:
        """Even None results are cached to avoid repeated API failures."""
        with patch(
            "vaig.tools.gke.billing._fetch_pricing_from_catalog",
            return_value=None,
        ) as mock_fetch:
            r1 = get_dynamic_pricing("proj", "bad-region")
            r2 = get_dynamic_pricing("proj", "bad-region")

        assert r1 is None
        assert r2 is None
        mock_fetch.assert_called_once()


# ── get_autopilot_pricing (integration) ──────────────────────


class TestGetAutopilotPricing:
    """Pricing lookup with fallback logic."""

    def test_dynamic_pricing_success(self) -> None:
        """When billing API returns data, use it with 'billing_api' source."""
        dynamic = BillingPricingResult(0.04, 0.005, 0.00006)
        with patch(
            "vaig.tools.gke.billing.get_dynamic_pricing",
            return_value=dynamic,
        ):
            result = get_autopilot_pricing("us-central1", project_id="my-proj")

        assert result is not None
        assert result.source == "billing_api"
        assert result.pricing.cpu_per_vcpu_hour == pytest.approx(0.04)

    def test_fallback_to_hardcoded_on_api_failure(self) -> None:
        """When billing API returns None, fall back to hardcoded prices."""
        with patch(
            "vaig.tools.gke.billing.get_dynamic_pricing",
            return_value=None,
        ):
            result = get_autopilot_pricing("us-central1", project_id="my-proj")

        assert result is not None
        assert result.source == "hardcoded_fallback"
        assert result.pricing.cpu_per_vcpu_hour > 0

    def test_fallback_to_hardcoded_on_import_error(self) -> None:
        """When billing module can't be imported, fall back to hardcoded prices."""
        with patch(
            "builtins.__import__",
            side_effect=_import_blocker("vaig.tools.gke.billing"),
        ):
            result = get_autopilot_pricing("us-central1", project_id="my-proj")

        assert result is not None
        assert result.source == "hardcoded_fallback"

    def test_no_project_id_uses_hardcoded(self) -> None:
        """When project_id is None, skip billing API entirely."""
        result = get_autopilot_pricing("us-central1", project_id=None)

        assert result is not None
        assert result.source == "hardcoded_fallback"

    def test_unknown_region_returns_none(self) -> None:
        """Unknown region returns None even with project_id."""
        with patch(
            "vaig.tools.gke.billing.get_dynamic_pricing",
            return_value=None,
        ):
            result = get_autopilot_pricing("antarctica-south1", project_id="my-proj")

        assert result is None

    def test_pricing_lookup_result_fields(self) -> None:
        """PricingLookupResult exposes pricing and source correctly."""
        pricing = GKEPricing(0.035, 0.004, 0.00005)
        r = PricingLookupResult(pricing=pricing, source="billing_api")
        assert r.pricing is pricing
        assert r.source == "billing_api"


# ── GKECostReport.pricing_source ─────────────────────────────


class TestPricingSourceInReport:
    """Verify pricing_source field is present in GKECostReport."""

    def test_default_pricing_source(self) -> None:
        from vaig.skills.service_health.schema import GKECostReport

        report = GKECostReport()
        assert report.pricing_source == "hardcoded_fallback"

    def test_billing_api_pricing_source(self) -> None:
        from vaig.skills.service_health.schema import GKECostReport

        report = GKECostReport(pricing_source="billing_api")
        assert report.pricing_source == "billing_api"


# ── Helpers ──────────────────────────────────────────────────


def _import_blocker(blocked_module: str) -> Any:
    """Create an __import__ side_effect that blocks a specific module."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__  # type: ignore[union-attr]

    def _blocking_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == blocked_module or name.startswith(blocked_module + "."):
            raise ImportError(f"Mocked: {name} not available")
        return real_import(name, *args, **kwargs)

    return _blocking_import

"""Unit tests for GKE Autopilot workload cost estimation.

Tests cover:
- CPU quantity parser (millicores, decimals, edge cases)
- Memory quantity parser (Mi, Gi, Ki, plain bytes)
- Cost calculation for single resources and workloads
- K8s resource aggregation (mocked pods)
- Autopilot / Standard / unknown cluster handling (mocked API)
- v2: per-container cost breakdown in calculate_workload_cost()
- v2: _aggregate_container_requests_per_container() helper
- v2: _compute_namespace_summaries() aggregation
- v2: GKEContainerCost and GKENamespaceSummary schema models
- v2: fetch_workload_costs() backward-compat (monitoring unavailable)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vaig.tools.gke.cost_estimation import (
    _MEMORY_SUFFIXES,
    AUTOPILOT_PRICING,
    _aggregate_container_requests,
    _get_workload_name,
    calculate_resource_cost,
    calculate_workload_cost,
    parse_cpu,
    parse_ephemeral,
    parse_memory,
)

# ── parse_cpu ─────────────────────────────────────────────────


class TestParseCpu:
    def test_millicores_100(self) -> None:
        assert parse_cpu("100m") == pytest.approx(0.1)

    def test_millicores_500(self) -> None:
        assert parse_cpu("500m") == pytest.approx(0.5)

    def test_millicores_1000(self) -> None:
        assert parse_cpu("1000m") == pytest.approx(1.0)

    def test_whole_vcpu(self) -> None:
        assert parse_cpu("2") == pytest.approx(2.0)

    def test_decimal_vcpu(self) -> None:
        assert parse_cpu("0.5") == pytest.approx(0.5)

    def test_none_returns_zero(self) -> None:
        assert parse_cpu(None) == 0.0

    def test_empty_string_returns_zero(self) -> None:
        assert parse_cpu("") == 0.0

    def test_invalid_returns_zero(self) -> None:
        assert parse_cpu("notanumber") == 0.0

    def test_invalid_millicores_returns_zero(self) -> None:
        assert parse_cpu("xm") == 0.0

    def test_whitespace_stripped(self) -> None:
        assert parse_cpu("  250m  ") == pytest.approx(0.25)


# ── parse_memory ──────────────────────────────────────────────


class TestParseMemory:
    def test_mebibytes_128(self) -> None:
        assert parse_memory("128Mi") == pytest.approx(0.125)

    def test_mebibytes_1024(self) -> None:
        assert parse_memory("1024Mi") == pytest.approx(1.0)

    def test_gibibytes_1(self) -> None:
        assert parse_memory("1Gi") == pytest.approx(1.0)

    def test_gibibytes_4(self) -> None:
        assert parse_memory("4Gi") == pytest.approx(4.0)

    def test_large_mebibytes(self) -> None:
        # 24000Mi → 24000 / 1024 = 23.4375 GiB
        assert parse_memory("24000Mi") == pytest.approx(23.4375)

    def test_kibibytes(self) -> None:
        # 1024Ki = 1 MiB = 0.0009765625 GiB
        assert parse_memory("1024Ki") == pytest.approx(1.0 / 1024.0)

    def test_none_returns_zero(self) -> None:
        assert parse_memory(None) == 0.0

    def test_empty_string_returns_zero(self) -> None:
        assert parse_memory("") == 0.0

    def test_invalid_returns_zero(self) -> None:
        assert parse_memory("badvalue") == 0.0

    def test_plain_bytes(self) -> None:
        # 1073741824 bytes = 1 GiB
        assert parse_memory("1073741824") == pytest.approx(1.0)

    def test_whitespace_stripped(self) -> None:
        assert parse_memory("  512Mi  ") == pytest.approx(0.5)


# ── parse_ephemeral ───────────────────────────────────────────


class TestParseEphemeral:
    """Ephemeral storage uses the same suffix rules as memory."""

    def test_gibibytes(self) -> None:
        assert parse_ephemeral("2Gi") == pytest.approx(2.0)

    def test_mebibytes(self) -> None:
        assert parse_ephemeral("512Mi") == pytest.approx(0.5)

    def test_none_returns_zero(self) -> None:
        assert parse_ephemeral(None) == 0.0


# ── calculate_resource_cost ───────────────────────────────────


class TestCalculateResourceCost:
    def test_basic(self) -> None:
        # 1 vCPU × $0.05/hr × 730 hr = $36.50
        result = calculate_resource_cost(1.0, 0.05)
        assert result == pytest.approx(36.50)

    def test_zero_quantity(self) -> None:
        assert calculate_resource_cost(0.0, 0.05) == 0.0

    def test_zero_rate(self) -> None:
        assert calculate_resource_cost(2.0, 0.0) == 0.0

    def test_custom_hours(self) -> None:
        result = calculate_resource_cost(1.0, 1.0, hours_per_month=10.0)
        assert result == pytest.approx(10.0)


# ── calculate_workload_cost ───────────────────────────────────


_MONTREAL_PRICING = AUTOPILOT_PRICING["northamerica-northeast1"]


class TestCalculateWorkloadCost:
    def test_requests_only(self) -> None:
        result = calculate_workload_cost(
            cpu_requests=1.0,
            memory_requests_gib=1.0,
            ephemeral_requests_gib=0.0,
            pricing=_MONTREAL_PRICING,
        )
        # CPU: 1 × 0.0511 × 730 = 37.303
        # RAM: 1 × 0.0056 × 730 = 4.088
        # Eph: 0 → 0
        expected_total = (0.0511 + 0.0056) * 730.0
        assert result["total_request_cost_usd"] == pytest.approx(expected_total, rel=1e-4)
        assert result["total_usage_cost_usd"] is None  # no usage provided
        assert result["total_waste_usd"] is None

    def test_with_usage(self) -> None:
        result = calculate_workload_cost(
            cpu_requests=2.0,
            memory_requests_gib=4.0,
            ephemeral_requests_gib=1.0,
            pricing=_MONTREAL_PRICING,
            cpu_usage=1.0,
            memory_usage_gib=2.0,
            ephemeral_usage_gib=0.5,
        )
        assert result["total_usage_cost_usd"] is not None
        assert result["total_waste_usd"] is not None
        # waste = request_cost - usage_cost → positive (over-provisioned)
        assert result["total_waste_usd"] > 0

    def test_usage_equals_requests_zero_waste(self) -> None:
        # When usage == requests for all TRACKED dimensions (cpu + memory),
        # per-resource waste for cpu and memory must be 0.
        # total_waste_usd = total_request - tracked_usage = ephemeral_request_cost
        # because ephemeral is intentionally excluded from the usage tracking
        # (Cloud Monitoring never provides ephemeral usage data).
        result = calculate_workload_cost(
            cpu_requests=1.0,
            memory_requests_gib=1.0,
            ephemeral_requests_gib=1.0,
            pricing=_MONTREAL_PRICING,
            cpu_usage=1.0,
            memory_usage_gib=1.0,
            ephemeral_usage_gib=1.0,
        )
        by_type = {rc.resource_type: rc for rc in result["resource_costs"]}
        # Tracked dimensions: cpu and memory waste must be exactly 0
        assert by_type["cpu"].waste_cost_usd == pytest.approx(0.0, abs=1e-9)
        assert by_type["memory"].waste_cost_usd == pytest.approx(0.0, abs=1e-9)
        # total_waste = total_request - tracked_usage = ephemeral request cost only
        expected_ephemeral_cost = calculate_resource_cost(1.0, _MONTREAL_PRICING.ephemeral_per_gib_hour)
        assert result["total_waste_usd"] == pytest.approx(expected_ephemeral_cost, rel=1e-6)

    def test_resource_costs_count(self) -> None:
        result = calculate_workload_cost(
            cpu_requests=1.0,
            memory_requests_gib=1.0,
            ephemeral_requests_gib=1.0,
            pricing=_MONTREAL_PRICING,
        )
        assert len(result["resource_costs"]) == 3
        types = {rc.resource_type for rc in result["resource_costs"]}
        assert types == {"cpu", "memory", "ephemeral"}

    def test_partial_usage_produces_partial_cost(self) -> None:
        # Providing cpu_usage but not memory_usage — partial metrics policy:
        # total_usage_cost_usd is populated with the partial sum (not None),
        # but total_waste_usd remains None because not all tracked dims are available.
        result = calculate_workload_cost(
            cpu_requests=1.0,
            memory_requests_gib=1.0,
            ephemeral_requests_gib=0.0,
            pricing=_MONTREAL_PRICING,
            cpu_usage=0.5,
            memory_usage_gib=None,
        )
        # Partial usage is exposed — not None anymore (new behavior)
        assert result["total_usage_cost_usd"] is not None
        assert result["total_usage_cost_usd"] > 0
        # waste is still None (not all tracked dimensions available)
        assert result["total_waste_usd"] is None
        # partial_metrics flag must be set
        assert result["partial_metrics"] is True

    def test_partial_usage_later_dimension_still_computed(self) -> None:
        # Fix 6: order-dependent bug.
        # cpu_usage=None (first dim missing), memory_usage_gib and
        # ephemeral_usage_gib are present.  Per-resource usage_cost_usd
        # for memory and ephemeral must still be computed.
        # With partial-metrics policy, total_usage_cost_usd is the partial sum
        # (memory only, since cpu is the missing tracked dim).
        # total_waste_usd is still None (not all tracked dimensions available).
        result = calculate_workload_cost(
            cpu_requests=1.0,
            memory_requests_gib=4.0,
            ephemeral_requests_gib=1.0,
            pricing=_MONTREAL_PRICING,
            cpu_usage=None,          # first dimension — missing
            memory_usage_gib=2.0,    # second dimension — present
            ephemeral_usage_gib=0.5, # third dimension — present
        )
        # Partial usage is now exposed — not None (new partial-metrics behavior)
        assert result["total_usage_cost_usd"] is not None
        assert result["total_usage_cost_usd"] > 0
        # waste is still None (not all tracked dimensions available)
        assert result["total_waste_usd"] is None
        # partial_metrics flag must be set
        assert result["partial_metrics"] is True

        # But per-resource usage costs for memory and ephemeral must be filled
        by_type = {rc.resource_type: rc for rc in result["resource_costs"]}
        assert by_type["cpu"].usage_cost_usd is None       # missing
        assert by_type["memory"].usage_cost_usd is not None    # must not be None
        assert by_type["ephemeral"].usage_cost_usd is not None # must not be None
        assert by_type["memory"].usage_cost_usd > 0
        assert by_type["ephemeral"].usage_cost_usd > 0


# ── _aggregate_container_requests ────────────────────────────


def _make_pod(containers: list[dict[str, Any]]) -> MagicMock:
    """Build a minimal mock V1Pod for testing."""
    pod = MagicMock()
    pod.status.phase = "Running"
    mock_containers = []
    for c in containers:
        mc = MagicMock()
        mc.resources.requests = c.get("requests", {})
        mock_containers.append(mc)
    pod.spec.containers = mock_containers
    pod.spec.init_containers = []
    return pod


class TestAggregateContainerRequests:
    def test_single_pod_single_container(self) -> None:
        pod = _make_pod([{"requests": {"cpu": "500m", "memory": "512Mi"}}])
        cpu, mem, eph = _aggregate_container_requests([pod])
        assert cpu == pytest.approx(0.5)
        assert mem == pytest.approx(0.5)
        assert eph == pytest.approx(0.0)

    def test_multiple_containers_summed(self) -> None:
        # 3-container pod: app + datadog + istio pattern
        pod = _make_pod([
            {"requests": {"cpu": "100m", "memory": "128Mi"}},
            {"requests": {"cpu": "50m", "memory": "64Mi"}},
            {"requests": {"cpu": "50m", "memory": "64Mi"}},
        ])
        cpu, mem, eph = _aggregate_container_requests([pod])
        assert cpu == pytest.approx(0.2)
        assert mem == pytest.approx(0.25)

    def test_multiple_pods_summed(self) -> None:
        pods = [
            _make_pod([{"requests": {"cpu": "500m", "memory": "512Mi"}}]),
            _make_pod([{"requests": {"cpu": "500m", "memory": "512Mi"}}]),
        ]
        cpu, mem, eph = _aggregate_container_requests(pods)
        assert cpu == pytest.approx(1.0)
        assert mem == pytest.approx(1.0)

    def test_pod_with_no_requests(self) -> None:
        pod = _make_pod([{"requests": {}}])
        cpu, mem, eph = _aggregate_container_requests([pod])
        assert cpu == 0.0
        assert mem == 0.0

    def test_empty_pod_list(self) -> None:
        cpu, mem, eph = _aggregate_container_requests([])
        assert cpu == 0.0
        assert mem == 0.0
        assert eph == 0.0

    def test_pod_with_no_spec(self) -> None:
        pod = MagicMock()
        pod.spec = None
        cpu, mem, eph = _aggregate_container_requests([pod])
        assert cpu == 0.0

    def test_ephemeral_storage(self) -> None:
        pod = _make_pod([{"requests": {"cpu": "100m", "memory": "64Mi", "ephemeral-storage": "1Gi"}}])
        cpu, mem, eph = _aggregate_container_requests([pod])
        assert eph == pytest.approx(1.0)


# ── _get_workload_name ────────────────────────────────────────


def _make_pod_with_owner(kind: str, name: str) -> MagicMock:
    pod = MagicMock()
    owner = MagicMock()
    owner.kind = kind
    owner.name = name
    pod.metadata.owner_references = [owner]
    pod.metadata.labels = {}
    pod.metadata.name = "fallback-pod-abc12-xyz"
    return pod


class TestGetWorkloadName:
    def test_deployment_owner(self) -> None:
        pod = _make_pod_with_owner("Deployment", "my-app")
        assert _get_workload_name(pod) == "my-app"

    def test_statefulset_owner(self) -> None:
        pod = _make_pod_with_owner("StatefulSet", "postgres")
        assert _get_workload_name(pod) == "postgres"

    def test_replicaset_strips_hash(self) -> None:
        pod = _make_pod_with_owner("ReplicaSet", "my-app-59967f9ccc")
        assert _get_workload_name(pod) == "my-app"

    def test_replicaset_short_hash_not_stripped(self) -> None:
        # Hash must be 5-10 chars to be stripped
        pod = _make_pod_with_owner("ReplicaSet", "my-app-abc")
        # "abc" is 3 chars — too short — name kept as-is
        assert _get_workload_name(pod) == "my-app-abc"

    def test_label_fallback(self) -> None:
        pod = MagicMock()
        pod.metadata.owner_references = []
        pod.metadata.labels = {"app": "frontend"}
        pod.metadata.name = "frontend-pod"
        assert _get_workload_name(pod) == "frontend"

    def test_pod_name_fallback(self) -> None:
        pod = MagicMock()
        pod.metadata.owner_references = []
        pod.metadata.labels = {}
        pod.metadata.name = "my-service-abc12-xyz9"
        result = _get_workload_name(pod)
        # Strip the two-part hash suffix
        assert result == "my-service"

    def test_no_metadata(self) -> None:
        pod = MagicMock()
        pod.metadata = None
        assert _get_workload_name(pod) == "unknown"


# ── fetch_workload_costs — integration (mocked) ───────────────


class TestFetchWorkloadCosts:
    """Integration tests that mock K8s API and detect_autopilot."""

    def _make_gke_config(
        self, project_id: str = "my-project", location: str = "northamerica-northeast1", cluster_name: str = "my-cluster"
    ) -> MagicMock:
        cfg = MagicMock()
        cfg.project_id = project_id
        cfg.location = location
        cfg.cluster_name = cluster_name
        return cfg

    @patch("vaig.tools.gke._clients.detect_autopilot", return_value=None)
    def test_unknown_cluster_type(self, _mock: MagicMock) -> None:
        from vaig.tools.gke.cost_estimation import fetch_workload_costs
        with patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=None):
            report = fetch_workload_costs(self._make_gke_config())
        assert report.supported is False
        assert report.cluster_type == "unknown"
        assert "detection failed" in (report.unsupported_reason or "")

    def test_standard_cluster_not_supported(self) -> None:
        from vaig.tools.gke.cost_estimation import fetch_workload_costs
        with patch("vaig.tools.gke._clients.detect_autopilot", return_value=False), \
             patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=False):
            report = fetch_workload_costs(self._make_gke_config())
        assert report.supported is False
        assert report.cluster_type == "standard"
        assert "N/A" in (report.unsupported_reason or "")

    def test_unknown_region_not_supported(self) -> None:
        from vaig.tools.gke.cost_estimation import fetch_workload_costs
        with patch("vaig.tools.gke._clients.detect_autopilot", return_value=True), \
             patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True):
            report = fetch_workload_costs(self._make_gke_config(location="mars-west1"))
        assert report.supported is False
        assert "pricing table" in (report.unsupported_reason or "")

    def test_missing_k8s_library(self) -> None:
        from vaig.tools.gke.cost_estimation import fetch_workload_costs
        with patch("vaig.tools.gke.cost_estimation._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True):
            report = fetch_workload_costs(self._make_gke_config())
        assert report.supported is False
        assert "kubernetes" in (report.unsupported_reason or "").lower()

    @patch("vaig.tools.gke.cost_estimation._create_k8s_clients")
    @patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True)
    def test_autopilot_with_pods(self, _mock_autopilot: MagicMock, mock_clients: MagicMock) -> None:
        from vaig.tools.gke.cost_estimation import fetch_workload_costs

        # Build a mock namespace list and pod list
        ns_item = MagicMock()
        ns_item.metadata.name = "default"

        pod = _make_pod([
            {"requests": {"cpu": "100m", "memory": "128Mi"}},
            {"requests": {"cpu": "50m", "memory": "64Mi"}},
        ])
        pod.status.phase = "Running"
        # Give the pod a Deployment owner
        owner = MagicMock()
        owner.kind = "Deployment"
        owner.name = "my-app"
        pod.metadata.owner_references = [owner]
        pod.metadata.labels = {}
        pod.metadata.name = "my-app-pod-abc12"

        core_v1 = MagicMock()
        core_v1.list_namespace.return_value.items = [ns_item]
        core_v1.list_namespaced_pod.return_value.items = [pod]

        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        # Patch monitoring so it doesn't attempt a real GCP connection
        with patch(
            "vaig.tools.gke.monitoring.get_workload_usage_metrics",
            return_value={},
        ):
            report = fetch_workload_costs(self._make_gke_config())
        assert report.supported is True
        assert report.cluster_type == "autopilot"
        assert report.region == "northamerica-northeast1"
        assert len(report.workloads) == 1
        assert report.workloads[0].workload_name == "my-app"
        assert report.workloads[0].namespace == "default"
        assert report.total_request_cost_usd is not None
        assert report.total_request_cost_usd > 0


# ── AUTOPILOT_PRICING completeness ───────────────────────────


class TestPricingTable:
    def test_montreal_exists(self) -> None:
        assert "northamerica-northeast1" in AUTOPILOT_PRICING

    def test_all_regions_have_positive_rates(self) -> None:
        for region, pricing in AUTOPILOT_PRICING.items():
            assert pricing.cpu_per_vcpu_hour > 0, f"{region}: cpu rate must be > 0"
            assert pricing.ram_per_gib_hour > 0, f"{region}: ram rate must be > 0"
            assert pricing.ephemeral_per_gib_hour > 0, f"{region}: ephemeral rate must be > 0"

    def test_pricing_is_frozen(self) -> None:
        pricing = AUTOPILOT_PRICING["northamerica-northeast1"]
        with pytest.raises((AttributeError, TypeError)):
            pricing.cpu_per_vcpu_hour = 999.0  # type: ignore[misc]


# ── _MEMORY_SUFFIXES module-level constant (Fix 7) ───────────


class TestMemorySuffixesModuleLevel:
    def test_is_at_module_level(self) -> None:
        """_MEMORY_SUFFIXES must be a module-level dict, not re-created per call."""
        import vaig.tools.gke.cost_estimation as mod  # noqa: PLC0415
        assert hasattr(mod, "_MEMORY_SUFFIXES"), "_MEMORY_SUFFIXES must be module-level"
        assert isinstance(mod._MEMORY_SUFFIXES, dict)

    def test_ki_factor_is_correct(self) -> None:
        """Ki factor must be 1/(1024*1024) for KiB→GiB, not 1/1024 (Fix 5+7)."""
        assert _MEMORY_SUFFIXES["Ki"] == pytest.approx(1.0 / (1024.0 * 1024.0))

    def test_ki_conversion(self) -> None:
        # 1024 KiB = 1 MiB = 1/1024 GiB
        assert parse_memory("1024Ki") == pytest.approx(1.0 / 1024.0)

    def test_mi_conversion(self) -> None:
        # 1024 MiB = 1 GiB
        assert parse_memory("1024Mi") == pytest.approx(1.0)

    def test_gi_conversion(self) -> None:
        assert parse_memory("2Gi") == pytest.approx(2.0)

    def test_ti_conversion(self) -> None:
        assert parse_memory("1Ti") == pytest.approx(1024.0)


# ── Fix 3: missing config fields → specific error message ─────


class TestFetchWorkloadCostsMissingConfig:
    def _make_incomplete_config(self, **overrides: Any) -> MagicMock:
        cfg = MagicMock()
        cfg.project_id = overrides.get("project_id", "my-project")
        cfg.location = overrides.get("location", "northamerica-northeast1")
        cfg.cluster_name = overrides.get("cluster_name", "my-cluster")
        return cfg

    def test_missing_project_id_gives_specific_message(self) -> None:
        from vaig.tools.gke.cost_estimation import fetch_workload_costs  # noqa: PLC0415
        with patch("vaig.tools.gke.cost_estimation.detect_autopilot", new=lambda _: None):
            cfg = self._make_incomplete_config(project_id=None)
            cfg.project_id = None
            report = fetch_workload_costs(cfg)
        assert report.supported is False
        assert "project_id" in (report.unsupported_reason or "")

    def test_missing_location_gives_specific_message(self) -> None:
        from vaig.tools.gke.cost_estimation import fetch_workload_costs  # noqa: PLC0415
        with patch("vaig.tools.gke.cost_estimation.detect_autopilot", new=lambda _: None):
            cfg = self._make_incomplete_config(location=None)
            cfg.location = None
            report = fetch_workload_costs(cfg)
        assert report.supported is False
        assert "location" in (report.unsupported_reason or "")


# ── Multi-region pricing table spot-checks ────────────────────


class TestMultiRegionPricingTable:
    """Spot-check that major regions exist and have sensible rates."""

    def test_total_region_count(self) -> None:
        """We should have at least 32 regions in the table."""
        assert len(AUTOPILOT_PRICING) >= 32, (
            f"Expected >= 32 regions, got {len(AUTOPILOT_PRICING)}"
        )

    # ── North America spot-checks ─────────────────────────────

    def test_us_central1_iowa_exists(self) -> None:
        pricing = AUTOPILOT_PRICING["us-central1"]
        # Iowa is typically the cheapest US region
        assert pricing.cpu_per_vcpu_hour == pytest.approx(0.0485, rel=1e-3)
        assert pricing.ram_per_gib_hour == pytest.approx(0.0052, rel=1e-3)

    def test_us_east1_south_carolina_exists(self) -> None:
        assert "us-east1" in AUTOPILOT_PRICING
        p = AUTOPILOT_PRICING["us-east1"]
        assert p.cpu_per_vcpu_hour > 0
        assert p.ram_per_gib_hour > 0

    def test_northamerica_northeast2_toronto_exists(self) -> None:
        assert "northamerica-northeast2" in AUTOPILOT_PRICING
        p = AUTOPILOT_PRICING["northamerica-northeast2"]
        # Toronto should be close to Montreal pricing
        montreal = AUTOPILOT_PRICING["northamerica-northeast1"]
        assert abs(p.cpu_per_vcpu_hour - montreal.cpu_per_vcpu_hour) < 0.01

    # ── Europe spot-checks ────────────────────────────────────

    def test_europe_west1_belgium_exists(self) -> None:
        pricing = AUTOPILOT_PRICING["europe-west1"]
        assert pricing.cpu_per_vcpu_hour == pytest.approx(0.0534, rel=1e-3)
        assert pricing.ram_per_gib_hour == pytest.approx(0.0059, rel=1e-3)

    def test_europe_west2_london_exists(self) -> None:
        assert "europe-west2" in AUTOPILOT_PRICING
        assert AUTOPILOT_PRICING["europe-west2"].cpu_per_vcpu_hour > 0

    def test_europe_west6_zurich_is_priciest_europe(self) -> None:
        """Zurich carries a Swiss premium — should be the most expensive EU region."""
        zurich = AUTOPILOT_PRICING["europe-west6"]
        other_eu = [
            AUTOPILOT_PRICING[r]
            for r in ("europe-west1", "europe-west2", "europe-west3", "europe-west4", "europe-west9", "europe-north1")
        ]
        for other in other_eu:
            assert zurich.cpu_per_vcpu_hour >= other.cpu_per_vcpu_hour, (
                "Zurich should have >= CPU price vs other EU regions"
            )

    # ── Asia-Pacific spot-checks ──────────────────────────────

    def test_asia_northeast1_tokyo_exists(self) -> None:
        pricing = AUTOPILOT_PRICING["asia-northeast1"]
        assert pricing.cpu_per_vcpu_hour == pytest.approx(0.0613, rel=1e-3)
        assert pricing.ram_per_gib_hour == pytest.approx(0.0068, rel=1e-3)

    def test_asia_east1_taiwan_exists(self) -> None:
        assert "asia-east1" in AUTOPILOT_PRICING
        assert AUTOPILOT_PRICING["asia-east1"].cpu_per_vcpu_hour > 0

    # ── South America spot-checks ─────────────────────────────

    def test_southamerica_east1_sao_paulo_exists(self) -> None:
        assert "southamerica-east1" in AUTOPILOT_PRICING
        sao_paulo = AUTOPILOT_PRICING["southamerica-east1"]
        iowa = AUTOPILOT_PRICING["us-central1"]
        # São Paulo should be ~30-40% more expensive than Iowa
        ratio = sao_paulo.cpu_per_vcpu_hour / iowa.cpu_per_vcpu_hour
        assert ratio >= 1.30, f"São Paulo should be >= 30% more than Iowa, got {ratio:.2f}x"
        assert ratio <= 1.60, f"São Paulo ratio seems too high: {ratio:.2f}x"

    # ── Middle East spot-checks ───────────────────────────────

    def test_me_west1_tel_aviv_exists(self) -> None:
        assert "me-west1" in AUTOPILOT_PRICING
        assert AUTOPILOT_PRICING["me-west1"].cpu_per_vcpu_hour > 0

    # ── Parametrized existence checks (simple membership) ─────

    @pytest.mark.parametrize("region", [
        # North America
        "us-west4",
        "us-south1",
        # Europe
        "europe-west3",
        "europe-west4",
        "europe-west9",
        "europe-north1",
        # Asia-Pacific
        "asia-east2",
        "asia-northeast2",
        "asia-northeast3",
        "asia-south1",
        "asia-south2",
        "asia-southeast1",
        "asia-southeast2",
        "australia-southeast1",
        "australia-southeast2",
        # South America
        "southamerica-west1",
        # Middle East
        "me-central1",
        "me-central2",
    ])
    def test_region_exists_in_pricing_table(self, region: str) -> None:
        """All listed regions must have an entry in AUTOPILOT_PRICING."""
        assert region in AUTOPILOT_PRICING, (
            f"Region '{region}' not found in AUTOPILOT_PRICING"
        )

    # ── Regional ordering sanity checks ──────────────────────

    def test_us_regions_cheaper_than_south_america(self) -> None:
        """All US regions should be cheaper than São Paulo on CPU."""
        sao_paulo_cpu = AUTOPILOT_PRICING["southamerica-east1"].cpu_per_vcpu_hour
        for us_region, pricing in AUTOPILOT_PRICING.items():
            if not us_region.startswith("us-"):
                continue
            us_cpu = pricing.cpu_per_vcpu_hour
            assert us_cpu < sao_paulo_cpu, (
                f"{us_region} ({us_cpu}) should be cheaper than São Paulo ({sao_paulo_cpu})"
            )


# ── Unknown region graceful degradation ───────────────────────


class TestUnknownRegionFallback:
    """Verify fetch_workload_costs() degrades gracefully for unsupported regions."""

    def _make_gke_config(self, location: str) -> MagicMock:
        cfg = MagicMock()
        cfg.project_id = "my-project"
        cfg.location = location
        cfg.cluster_name = "my-cluster"
        return cfg

    def test_unknown_region_returns_unsupported(self) -> None:
        from vaig.tools.gke.cost_estimation import fetch_workload_costs  # noqa: PLC0415
        with patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True):
            report = fetch_workload_costs(self._make_gke_config("mars-west1"))
        assert report.supported is False
        assert report.cluster_type == "autopilot"
        # Error message should name the region and mention the pricing table
        assert "mars-west1" in (report.unsupported_reason or "")
        assert "pricing table" in (report.unsupported_reason or "")

    def test_empty_region_returns_unsupported(self) -> None:
        from vaig.tools.gke.cost_estimation import fetch_workload_costs  # noqa: PLC0415
        with patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True):
            cfg = self._make_gke_config("")
            cfg.location = ""
            report = fetch_workload_costs(cfg)
        assert report.supported is False
        # Empty location triggers the "missing fields" validation guard before reaching
        # the pricing table lookup — so the error message differs from unknown regions.
        reason = report.unsupported_reason or ""
        assert "location" in reason

    def test_unknown_region_message_lists_supported_regions(self) -> None:
        """The error message should list at least some supported regions so users know what to use."""
        from vaig.tools.gke.cost_estimation import fetch_workload_costs  # noqa: PLC0415
        with patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True):
            report = fetch_workload_costs(self._make_gke_config("invalid-region1"))
        reason = report.unsupported_reason or ""
        # Should mention at least one known region
        assert any(r in reason for r in ("us-central1", "northamerica-northeast1", "europe-west1"))


# ── v2: GKEContainerCost / GKENamespaceSummary schema ────────


class TestGKEContainerCostSchema:
    """Verify the new v2 schema models round-trip correctly."""

    def test_container_cost_minimal(self) -> None:
        from vaig.skills.service_health.schema import GKEContainerCost

        cc = GKEContainerCost(container_name="app", resource_costs=[])
        assert cc.container_name == "app"
        assert cc.resource_costs == []
        assert cc.total_request_cost_usd is None
        assert cc.total_usage_cost_usd is None
        assert cc.total_waste_usd is None

    def test_container_cost_with_values(self) -> None:
        from vaig.skills.service_health.schema import GKEContainerCost

        cc = GKEContainerCost(
            container_name="sidecar",
            resource_costs=[],
            total_request_cost_usd=1.5,
            total_usage_cost_usd=0.8,
            total_waste_usd=0.7,
        )
        assert cc.total_request_cost_usd == pytest.approx(1.5)
        assert cc.total_usage_cost_usd == pytest.approx(0.8)
        assert cc.total_waste_usd == pytest.approx(0.7)

    def test_namespace_summary_minimal(self) -> None:
        from vaig.skills.service_health.schema import GKENamespaceSummary

        ns = GKENamespaceSummary(namespace="default")
        assert ns.namespace == "default"
        assert ns.total_request_cost_usd == pytest.approx(0.0)
        assert ns.total_usage_cost_usd is None
        assert ns.total_waste_usd is None

    def test_namespace_summary_with_values(self) -> None:
        from vaig.skills.service_health.schema import GKENamespaceSummary

        ns = GKENamespaceSummary(
            namespace="production",
            total_request_cost_usd=10.0,
            total_usage_cost_usd=6.0,
            total_waste_usd=4.0,
        )
        assert ns.total_request_cost_usd == pytest.approx(10.0)
        assert ns.total_waste_usd == pytest.approx(4.0)

    def test_workload_cost_containers_field_defaults_empty(self) -> None:
        from vaig.skills.service_health.schema import GKEWorkloadCost

        wl = GKEWorkloadCost(
            namespace="default",
            workload_name="my-app",
            resource_costs=[],
        )
        assert wl.containers == []

    def test_cost_report_namespace_summaries_defaults_empty(self) -> None:
        from vaig.skills.service_health.schema import GKECostReport

        report = GKECostReport(supported=True, cluster_type="autopilot")
        assert report.namespace_summaries == {}


# ── v2: _aggregate_container_requests_per_container ─────────


def _make_pod_with_named_container(
    containers: list[dict[str, Any]],
) -> MagicMock:
    """Build a mock V1Pod with named containers."""
    pod = MagicMock()
    mock_containers = []
    for c in containers:
        mc = MagicMock()
        mc.name = c["name"]
        mc.resources.requests = c.get("requests", {})
        mock_containers.append(mc)
    pod.spec.containers = mock_containers
    pod.spec.init_containers = []
    return pod


class TestAggregateContainerRequestsPerContainer:
    """Tests for the v2 per-container request aggregation helper."""

    def test_single_pod_two_containers(self) -> None:
        from vaig.tools.gke.cost_estimation import _aggregate_container_requests_per_container

        pod = _make_pod_with_named_container([
            {"name": "app", "requests": {"cpu": "500m", "memory": "512Mi"}},
            {"name": "sidecar", "requests": {"cpu": "100m", "memory": "128Mi"}},
        ])
        result = _aggregate_container_requests_per_container([pod])

        assert "app" in result
        assert "sidecar" in result
        app_cpu, app_mem, _ = result["app"]
        assert app_cpu == pytest.approx(0.5)
        assert app_mem == pytest.approx(0.5)
        sidecar_cpu, sidecar_mem, _ = result["sidecar"]
        assert sidecar_cpu == pytest.approx(0.1)
        assert sidecar_mem == pytest.approx(0.125)

    def test_two_replicas_same_container_name_summed(self) -> None:
        from vaig.tools.gke.cost_estimation import _aggregate_container_requests_per_container

        pod1 = _make_pod_with_named_container([
            {"name": "app", "requests": {"cpu": "250m", "memory": "256Mi"}},
        ])
        pod2 = _make_pod_with_named_container([
            {"name": "app", "requests": {"cpu": "250m", "memory": "256Mi"}},
        ])
        result = _aggregate_container_requests_per_container([pod1, pod2])

        assert len(result) == 1
        cpu, mem, _ = result["app"]
        assert cpu == pytest.approx(0.5)  # 2 × 250m
        assert mem == pytest.approx(0.5)  # 2 × 256Mi

    def test_empty_pod_list_returns_empty_dict(self) -> None:
        from vaig.tools.gke.cost_estimation import _aggregate_container_requests_per_container

        result = _aggregate_container_requests_per_container([])
        assert result == {}

    def test_pod_with_no_spec_skipped(self) -> None:
        from vaig.tools.gke.cost_estimation import _aggregate_container_requests_per_container

        pod = MagicMock()
        pod.spec = None
        result = _aggregate_container_requests_per_container([pod])
        assert result == {}

    def test_mock_container_without_name_uses_unknown(self) -> None:
        """MagicMock containers that have no string .name fall back to 'unknown'."""
        from vaig.tools.gke.cost_estimation import _aggregate_container_requests_per_container

        pod = MagicMock()
        mc = MagicMock()
        mc.name = MagicMock()  # not a str — simulates the MagicMock gotcha
        mc.resources.requests = {"cpu": "100m", "memory": "64Mi"}
        pod.spec.containers = [mc]
        pod.spec.init_containers = []
        result = _aggregate_container_requests_per_container([pod])
        # Should land in "unknown" bucket, not crash
        assert "unknown" in result

    def test_ephemeral_storage_tracked_per_container(self) -> None:
        from vaig.tools.gke.cost_estimation import _aggregate_container_requests_per_container

        pod = _make_pod_with_named_container([
            {"name": "app", "requests": {"cpu": "100m", "ephemeral-storage": "2Gi"}},
        ])
        result = _aggregate_container_requests_per_container([pod])
        _, _, eph = result["app"]
        assert eph == pytest.approx(2.0)


# ── v2: calculate_workload_cost with container_requests ──────


class TestCalculateWorkloadCostV2Containers:
    """Per-container breakdown in calculate_workload_cost()."""

    def test_no_container_requests_returns_empty_containers_list(self) -> None:
        result = calculate_workload_cost(
            cpu_requests=1.0,
            memory_requests_gib=1.0,
            ephemeral_requests_gib=0.0,
            pricing=_MONTREAL_PRICING,
        )
        assert result["containers"] == []

    def test_two_containers_produces_two_container_cost_objects(self) -> None:
        from vaig.skills.service_health.schema import GKEContainerCost

        container_requests = {
            "app": (0.5, 0.5, 0.0),
            "sidecar": (0.1, 0.125, 0.0),
        }
        result = calculate_workload_cost(
            cpu_requests=0.6,
            memory_requests_gib=0.625,
            ephemeral_requests_gib=0.0,
            pricing=_MONTREAL_PRICING,
            container_requests=container_requests,
        )
        containers = result["containers"]
        assert len(containers) == 2
        names = {c.container_name for c in containers}
        assert names == {"app", "sidecar"}
        for c in containers:
            assert isinstance(c, GKEContainerCost)

    def test_container_with_usage_gets_per_resource_usage_populated(self) -> None:
        """Per-resource usage_cost_usd fields must be populated when usage data is available.

        Ephemeral usage is never available from Cloud Monitoring — only CPU and memory
        are queried. However, the container total is now computed from CPU+memory when
        both are present; ephemeral is excluded from the availability check.
        """
        from vaig.tools.gke.monitoring import ContainerUsageMetrics

        container_requests = {"app": (1.0, 1.0, 0.0)}
        container_usage = {
            "app": ContainerUsageMetrics(
                container_name="app",
                avg_cpu_cores=0.5,
                avg_memory_gib=0.5,
            )
        }
        result = calculate_workload_cost(
            cpu_requests=1.0,
            memory_requests_gib=1.0,
            ephemeral_requests_gib=0.0,
            pricing=_MONTREAL_PRICING,
            container_requests=container_requests,
            container_usage=container_usage,
        )
        (ct,) = result["containers"]
        # Per-resource: cpu and memory must have usage costs populated
        by_type = {rc.resource_type: rc for rc in ct.resource_costs}
        assert by_type["cpu"].usage_cost_usd is not None
        assert by_type["memory"].usage_cost_usd is not None
        # Waste is also per-resource
        assert by_type["cpu"].waste_cost_usd is not None
        assert by_type["memory"].waste_cost_usd is not None
        # total_usage_cost_usd is now computed from CPU+memory (ephemeral excluded from check)
        assert ct.total_usage_cost_usd is not None
        assert ct.total_usage_cost_usd > 0
        assert ct.total_waste_usd is not None

    def test_container_without_usage_has_none_usage_and_waste(self) -> None:
        container_requests = {"app": (1.0, 1.0, 0.0)}
        result = calculate_workload_cost(
            cpu_requests=1.0,
            memory_requests_gib=1.0,
            ephemeral_requests_gib=0.0,
            pricing=_MONTREAL_PRICING,
            container_requests=container_requests,
            container_usage=None,
        )
        (ct,) = result["containers"]
        assert ct.total_usage_cost_usd is None
        assert ct.total_waste_usd is None

    def test_container_request_cost_sums_to_workload_total(self) -> None:
        """Sum of container request costs must equal workload total_request_cost_usd."""
        container_requests = {
            "app": (0.5, 0.5, 0.0),
            "sidecar": (0.5, 0.5, 0.0),
        }
        result = calculate_workload_cost(
            cpu_requests=1.0,
            memory_requests_gib=1.0,
            ephemeral_requests_gib=0.0,
            pricing=_MONTREAL_PRICING,
            container_requests=container_requests,
        )
        container_total = sum(c.total_request_cost_usd or 0.0 for c in result["containers"])
        assert container_total == pytest.approx(result["total_request_cost_usd"])


# ── v2: _compute_namespace_summaries ─────────────────────────


def _make_workload_cost(
    namespace: str,
    workload_name: str,
    request_cost: float,
    usage_cost: float | None = None,
    waste: float | None = None,
) -> Any:
    """Build a minimal mock GKEWorkloadCost for namespace summary tests."""
    wl = MagicMock()
    wl.namespace = namespace
    wl.workload_name = workload_name
    wl.total_request_cost_usd = request_cost
    wl.total_usage_cost_usd = usage_cost
    wl.total_waste_usd = waste
    return wl


class TestComputeNamespaceSummaries:
    """Tests for _compute_namespace_summaries()."""

    def test_empty_workloads_returns_empty_dict(self) -> None:
        from vaig.tools.gke.cost_estimation import _compute_namespace_summaries

        result = _compute_namespace_summaries([])
        assert result == {}

    def test_single_workload_creates_one_namespace_entry(self) -> None:
        from vaig.tools.gke.cost_estimation import _compute_namespace_summaries

        wl = _make_workload_cost("default", "api", 5.0, 3.0, 2.0)
        result = _compute_namespace_summaries([wl])

        assert "default" in result
        ns = result["default"]
        assert ns.total_request_cost_usd == pytest.approx(5.0)
        assert ns.total_usage_cost_usd == pytest.approx(3.0)
        assert ns.total_waste_usd == pytest.approx(2.0)

    def test_two_workloads_same_namespace_costs_are_summed(self) -> None:
        from vaig.tools.gke.cost_estimation import _compute_namespace_summaries

        wl1 = _make_workload_cost("production", "api", 5.0, 3.0, 2.0)
        wl2 = _make_workload_cost("production", "worker", 3.0, 2.0, 1.0)
        result = _compute_namespace_summaries([wl1, wl2])

        ns = result["production"]
        assert ns.total_request_cost_usd == pytest.approx(8.0)
        assert ns.total_usage_cost_usd == pytest.approx(5.0)
        assert ns.total_waste_usd == pytest.approx(3.0)

    def test_two_namespaces_are_independent(self) -> None:
        from vaig.tools.gke.cost_estimation import _compute_namespace_summaries

        wl1 = _make_workload_cost("default", "api", 4.0, 2.0, 2.0)
        wl2 = _make_workload_cost("staging", "worker", 2.0, 1.0, 1.0)
        result = _compute_namespace_summaries([wl1, wl2])

        assert set(result.keys()) == {"default", "staging"}
        assert result["default"].total_request_cost_usd == pytest.approx(4.0)
        assert result["staging"].total_request_cost_usd == pytest.approx(2.0)

    def test_any_workload_missing_usage_makes_namespace_usage_none(self) -> None:
        from vaig.tools.gke.cost_estimation import _compute_namespace_summaries

        wl1 = _make_workload_cost("default", "api", 5.0, 3.0, 2.0)
        wl2 = _make_workload_cost("default", "worker", 3.0, None, None)  # no usage
        result = _compute_namespace_summaries([wl1, wl2])

        ns = result["default"]
        assert ns.total_request_cost_usd == pytest.approx(8.0)  # request always summed
        assert ns.total_usage_cost_usd is None
        assert ns.total_waste_usd is None

    def test_namespace_summary_object_type(self) -> None:
        from vaig.skills.service_health.schema import GKENamespaceSummary
        from vaig.tools.gke.cost_estimation import _compute_namespace_summaries

        wl = _make_workload_cost("default", "api", 1.0)
        result = _compute_namespace_summaries([wl])
        assert isinstance(result["default"], GKENamespaceSummary)


# ── v2: fetch_workload_costs backward-compat ─────────────────


class TestFetchWorkloadCostsV2BackwardCompat:
    """Verify monitoring-unavailable path: containers=[], namespace_summaries preserved."""

    def _make_gke_config(self, location: str = "northamerica-northeast1") -> MagicMock:
        cfg = MagicMock()
        cfg.project_id = "my-project"
        cfg.location = location
        cfg.cluster_name = "my-cluster"
        return cfg

    def _make_mock_clients(self, pod: MagicMock) -> MagicMock:
        """Build a mock (core_v1, ...) tuple with a single namespace + pod."""
        ns_item = MagicMock()
        ns_item.metadata.name = "default"

        core_v1 = MagicMock()
        core_v1.list_namespace.return_value.items = [ns_item]
        core_v1.list_namespaced_pod.return_value.items = [pod]
        return core_v1

    def _make_running_pod(
        self, namespace: str = "default", cpu: str = "100m", memory: str = "128Mi"
    ) -> MagicMock:
        pod = MagicMock()
        pod.status.phase = "Running"
        pod.metadata.namespace = namespace
        pod.metadata.name = "app-abc12"
        pod.metadata.labels = {}
        owner = MagicMock()
        owner.kind = "Deployment"
        owner.name = "my-app"
        pod.metadata.owner_references = [owner]
        mc = MagicMock()
        mc.name = "app"
        mc.resources.requests = {"cpu": cpu, "memory": memory}
        pod.spec.containers = [mc]
        pod.spec.init_containers = []
        return pod

    @patch("vaig.tools.gke.cost_estimation._create_k8s_clients")
    @patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True)
    def test_monitoring_unavailable_workload_has_empty_containers(
        self, _mock_autopilot: MagicMock, mock_clients: MagicMock
    ) -> None:
        """When get_workload_usage_metrics returns {}, containers are populated from K8s requests
        but have no usage/waste data (total_usage_cost_usd and total_waste_usd are None)."""
        from vaig.tools.gke.cost_estimation import fetch_workload_costs

        pod = self._make_running_pod()
        core_v1 = self._make_mock_clients(pod)
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch(
            "vaig.tools.gke.monitoring.get_workload_usage_metrics",
            return_value={},
        ):
            report = fetch_workload_costs(self._make_gke_config())

        assert report.supported is True
        for wl in report.workloads:
            # containers are populated from K8s resource requests
            assert len(wl.containers) >= 1
            for ct in wl.containers:
                # but no usage data → these must be None
                assert ct.total_usage_cost_usd is None
                assert ct.total_waste_usd is None
            # workload-level totals are also None without monitoring
            assert wl.total_usage_cost_usd is None
            assert wl.total_waste_usd is None

    @patch("vaig.tools.gke.cost_estimation._create_k8s_clients")
    @patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True)
    def test_monitoring_unavailable_namespace_summaries_present_without_usage(
        self, _mock_autopilot: MagicMock, mock_clients: MagicMock
    ) -> None:
        """namespace_summaries should contain the namespace but with None usage/waste."""
        from vaig.tools.gke.cost_estimation import fetch_workload_costs

        pod = self._make_running_pod()
        core_v1 = self._make_mock_clients(pod)
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch(
            "vaig.tools.gke.monitoring.get_workload_usage_metrics",
            return_value={},
        ):
            report = fetch_workload_costs(self._make_gke_config())

        # namespace_summaries should contain the namespace
        assert "default" in report.namespace_summaries
        ns = report.namespace_summaries["default"]
        assert ns.total_usage_cost_usd is None
        assert ns.total_waste_usd is None
        # request cost should be non-zero (pod has 100m CPU + 128Mi memory)
        assert ns.total_request_cost_usd > 0

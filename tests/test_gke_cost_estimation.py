"""Unit tests for GKE Autopilot workload cost estimation.

Tests cover:
- CPU quantity parser (millicores, decimals, edge cases)
- Memory quantity parser (Mi, Gi, Ki, plain bytes)
- Cost calculation for single resources and workloads
- K8s resource aggregation (mocked pods)
- Autopilot / Standard / unknown cluster handling (mocked API)
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
        result = calculate_workload_cost(
            cpu_requests=1.0,
            memory_requests_gib=1.0,
            ephemeral_requests_gib=1.0,
            pricing=_MONTREAL_PRICING,
            cpu_usage=1.0,
            memory_usage_gib=1.0,
            ephemeral_usage_gib=1.0,
        )
        assert result["total_waste_usd"] == pytest.approx(0.0, abs=1e-9)

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

    def test_partial_usage_makes_total_none(self) -> None:
        # Providing cpu_usage but not memory_usage — total_usage must be None
        result = calculate_workload_cost(
            cpu_requests=1.0,
            memory_requests_gib=1.0,
            ephemeral_requests_gib=0.0,
            pricing=_MONTREAL_PRICING,
            cpu_usage=0.5,
            memory_usage_gib=None,
        )
        assert result["total_usage_cost_usd"] is None
        assert result["total_waste_usd"] is None

    def test_partial_usage_later_dimension_still_computed(self) -> None:
        # Fix 6: order-dependent bug.
        # cpu_usage=None (first dim missing), memory_usage_gib and
        # ephemeral_usage_gib are present.  Per-resource usage_cost_usd
        # for memory and ephemeral must still be computed; only the TOTAL
        # should be None (because cpu is missing).
        result = calculate_workload_cost(
            cpu_requests=1.0,
            memory_requests_gib=4.0,
            ephemeral_requests_gib=1.0,
            pricing=_MONTREAL_PRICING,
            cpu_usage=None,          # first dimension — missing
            memory_usage_gib=2.0,    # second dimension — present
            ephemeral_usage_gib=0.5, # third dimension — present
        )
        # Total must be None (not all dimensions available)
        assert result["total_usage_cost_usd"] is None
        assert result["total_waste_usd"] is None

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

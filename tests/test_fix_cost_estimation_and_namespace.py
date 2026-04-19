"""Tests for fix/cost-estimation-namespace-and-na.

Covers:
- FIX A1: ContainerUsageMetrics allows partial metrics (cpu-only, mem-only)
- FIX A2: Monitoring errors are logged at ERROR level and captured in GKECostReport
- FIX A3: Partial-data policy — partial usage still contributes to report totals
- FIX B:  Namespace filtering via _inject_report_metadata / fetch_workload_costs
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vaig.core.config import GKEConfig
from vaig.tools.gke.monitoring import ContainerUsageMetrics, WorkloadUsageMetrics

# ── Helpers ──────────────────────────────────────────────────


def _make_gke_config(**kwargs: object) -> GKEConfig:
    defaults = {
        "cluster_name": "test-cluster",
        "project_id": "test-project",
        "location": "us-central1",
        "default_namespace": "default",
        "kubeconfig_path": "",
        "context": "",
        "log_limit": 100,
        "metrics_interval_minutes": 60,
        "proxy_url": "",
    }
    defaults.update(kwargs)
    return GKEConfig(**defaults)


def _make_running_pod(
    namespace: str = "default",
    name: str = "app-abc12",
    cpu: str = "100m",
    memory: str = "128Mi",
) -> MagicMock:
    """Build a minimal mock pod object matching fetch_workload_costs expectations."""
    pod = MagicMock()
    pod.status.phase = "Running"
    pod.metadata.namespace = namespace
    pod.metadata.name = name
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


def _make_mock_clients(
    pods: list[MagicMock],
    namespaces: list[str] | None = None,
) -> MagicMock:
    """Build a mock core_v1 client that returns the given pods.

    Uses list_namespace + list_namespaced_pod to match the real fetch_workload_costs path.
    """
    if namespaces is None:
        namespaces = list({p.metadata.namespace for p in pods})

    ns_items = []
    for ns_name in namespaces:
        ns_item = MagicMock()
        ns_item.metadata.name = ns_name
        ns_items.append(ns_item)

    core_v1 = MagicMock()
    core_v1.list_namespace.return_value.items = ns_items

    def _list_namespaced_pod(ns: str) -> MagicMock:
        pod_list = MagicMock()
        pod_list.items = [p for p in pods if p.metadata.namespace == ns]
        return pod_list

    core_v1.list_namespaced_pod.side_effect = _list_namespaced_pod
    return core_v1


# ── FIX A1: ContainerUsageMetrics partial metrics ─────────────


class TestContainerUsageMetricsPartial:
    """ContainerUsageMetrics now allows None in either dimension."""

    def test_cpu_only_is_valid(self) -> None:
        m = ContainerUsageMetrics(
            container_name="app", avg_cpu_cores=0.25, avg_memory_gib=None
        )
        assert m.avg_cpu_cores == pytest.approx(0.25)
        assert m.avg_memory_gib is None

    def test_mem_only_is_valid(self) -> None:
        m = ContainerUsageMetrics(
            container_name="app", avg_cpu_cores=None, avg_memory_gib=1.0
        )
        assert m.avg_cpu_cores is None
        assert m.avg_memory_gib == pytest.approx(1.0)

    def test_both_none_is_valid(self) -> None:
        m = ContainerUsageMetrics(
            container_name="app", avg_cpu_cores=None, avg_memory_gib=None
        )
        assert m.avg_cpu_cores is None
        assert m.avg_memory_gib is None

    def test_both_present_is_valid(self) -> None:
        m = ContainerUsageMetrics(
            container_name="app", avg_cpu_cores=0.1, avg_memory_gib=0.5
        )
        assert m.avg_cpu_cores == pytest.approx(0.1)
        assert m.avg_memory_gib == pytest.approx(0.5)


# ── FIX A1: get_workload_usage_metrics partial container logic ─


class TestGetWorkloadUsageMetricsPartial:
    """get_workload_usage_metrics should allow containers with only one dimension."""

    def _make_ts(self, pod_name: str, values: list[float], container_name: str = "app") -> MagicMock:
        ts = MagicMock()
        ts.resource.labels = {"pod_name": pod_name, "namespace_name": "default"}
        ts.metric.labels = {"container_name": container_name}
        points = []
        for v in reversed(values):
            point = MagicMock()
            point.value.double_value = v
            point.value.int64_value = 0
            points.append(point)
        ts.points = points
        return ts

    @patch("vaig.tools.gke.monitoring.MetricServiceClient")
    def test_cpu_only_container_included(self, mock_metric_cls: MagicMock) -> None:
        """Container is kept when only CPU data is available."""
        from vaig.tools.gke.monitoring import get_workload_usage_metrics

        cpu_ts = [self._make_ts("workload-pod-0", [0.1, 0.2])]
        mem_ts: list = []  # no mem data

        responses: list[list] = [cpu_ts, mem_ts]
        client = MagicMock()
        client.list_time_series.side_effect = lambda request: responses.pop(0)
        mock_metric_cls.return_value = client

        result = get_workload_usage_metrics(
            namespace="default",
            workload_pod_names={"workload": ["workload-pod-0"]},
            gke_config=_make_gke_config(),
        )

        assert "workload" in result
        containers = result["workload"].containers
        # At least one container should be present
        assert len(containers) >= 1
        all_containers = list(containers.values())
        # CPU should be non-None; mem should be None
        assert any(c.avg_cpu_cores is not None for c in all_containers)
        assert all(c.avg_memory_gib is None for c in all_containers)

    @patch("vaig.tools.gke.monitoring.MetricServiceClient")
    def test_both_missing_container_skipped(self, mock_metric_cls: MagicMock) -> None:
        """Container is skipped when BOTH cpu and mem data are absent."""
        from vaig.tools.gke.monitoring import get_workload_usage_metrics

        client = MagicMock()
        client.list_time_series.return_value = []
        mock_metric_cls.return_value = client

        result = get_workload_usage_metrics(
            namespace="default",
            workload_pod_names={"workload": ["workload-pod-0"]},
            gke_config=_make_gke_config(),
        )

        # Workload may or may not be in result; if present, containers must be empty
        if "workload" in result:
            assert result["workload"].containers == {}


# ── FIX A3: calculate_workload_cost partial-data policy ───────


class TestCalculateWorkloadCostPartialData:
    """calculate_workload_cost uses partial data when only one dimension available."""

    @pytest.fixture()
    def pricing(self):  # noqa: ANN201
        from vaig.tools.gke.cost_estimation import AUTOPILOT_PRICING
        return AUTOPILOT_PRICING["us-central1"]

    def test_cpu_usage_only_yields_partial_total(self, pricing: Any) -> None:
        from vaig.tools.gke.cost_estimation import calculate_workload_cost

        result = calculate_workload_cost(
            cpu_requests=0.25,
            memory_requests_gib=0.5,
            ephemeral_requests_gib=0.0,
            pricing=pricing,
            cpu_usage=0.1,       # only cpu known
            memory_usage_gib=None,  # mem not available
        )

        # There IS a usage total (partial is better than None)
        assert result["total_usage_cost_usd"] is not None
        assert result["total_usage_cost_usd"] > 0.0
        # Waste must be None because we don't have full coverage
        assert result["total_waste_usd"] is None
        # partial_metrics flag must be True
        assert result["partial_metrics"] is True

    def test_mem_usage_only_yields_partial_total(self, pricing: Any) -> None:
        from vaig.tools.gke.cost_estimation import calculate_workload_cost

        result = calculate_workload_cost(
            cpu_requests=0.25,
            memory_requests_gib=0.5,
            ephemeral_requests_gib=0.0,
            pricing=pricing,
            cpu_usage=None,          # cpu not available
            memory_usage_gib=0.3,    # only mem known
        )

        assert result["total_usage_cost_usd"] is not None
        assert result["total_usage_cost_usd"] > 0.0
        assert result["total_waste_usd"] is None
        assert result["partial_metrics"] is True

    def test_both_usage_present_not_partial(self, pricing: Any) -> None:
        from vaig.tools.gke.cost_estimation import calculate_workload_cost

        result = calculate_workload_cost(
            cpu_requests=0.25,
            memory_requests_gib=0.5,
            ephemeral_requests_gib=0.0,
            pricing=pricing,
            cpu_usage=0.1,
            memory_usage_gib=0.2,
        )

        assert result["total_usage_cost_usd"] is not None
        assert result["total_waste_usd"] is not None
        assert result["partial_metrics"] is False

    def test_no_usage_yields_none_total(self, pricing: Any) -> None:
        from vaig.tools.gke.cost_estimation import calculate_workload_cost

        result = calculate_workload_cost(
            cpu_requests=0.25,
            memory_requests_gib=0.5,
            ephemeral_requests_gib=0.0,
            pricing=pricing,
            cpu_usage=None,
            memory_usage_gib=None,
        )

        assert result["total_usage_cost_usd"] is None
        assert result["total_waste_usd"] is None
        assert result["partial_metrics"] is False


# ── FIX A3: fetch_workload_costs coverage counters ────────────


class TestFetchWorkloadCostsCoverageCounters:
    """fetch_workload_costs populates coverage counter fields on GKECostReport."""

    @patch("vaig.tools.gke.cost_estimation._create_k8s_clients")
    @patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True)
    def test_full_metrics_increments_full_counter(
        self, _mock_autopilot: MagicMock, mock_clients: MagicMock
    ) -> None:
        from vaig.tools.gke.cost_estimation import fetch_workload_costs

        pod = _make_running_pod()
        core_v1 = _make_mock_clients([pod])
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        # Provide full metrics (both cpu and mem)
        usage_metrics = {
            "my-app": WorkloadUsageMetrics(
                namespace="default",
                workload_name="my-app",
                containers={
                    "app": ContainerUsageMetrics(
                        container_name="app",
                        avg_cpu_cores=0.05,
                        avg_memory_gib=0.05,
                    )
                },
            )
        }

        with patch(
            "vaig.tools.gke.monitoring.get_workload_usage_metrics",
            return_value=usage_metrics,
        ):
            report = fetch_workload_costs(_make_gke_config())

        assert report.workloads_with_full_metrics == 1
        assert report.workloads_with_partial_metrics == 0
        assert report.workloads_without_metrics == 0
        assert report.total_usage_cost_usd is not None

    @patch("vaig.tools.gke.cost_estimation._create_k8s_clients")
    @patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True)
    def test_partial_metrics_increments_partial_counter(
        self, _mock_autopilot: MagicMock, mock_clients: MagicMock
    ) -> None:
        from vaig.tools.gke.cost_estimation import fetch_workload_costs

        pod = _make_running_pod()
        core_v1 = _make_mock_clients([pod])
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        # Provide partial metrics (cpu only — mem is None)
        usage_metrics = {
            "my-app": WorkloadUsageMetrics(
                namespace="default",
                workload_name="my-app",
                containers={
                    "app": ContainerUsageMetrics(
                        container_name="app",
                        avg_cpu_cores=0.05,
                        avg_memory_gib=None,  # partial
                    )
                },
            )
        }

        with patch(
            "vaig.tools.gke.monitoring.get_workload_usage_metrics",
            return_value=usage_metrics,
        ):
            report = fetch_workload_costs(_make_gke_config())

        assert report.workloads_with_partial_metrics == 1
        assert report.workloads_with_full_metrics == 0
        assert report.workloads_without_metrics == 0
        # Usage is still available (cpu contributes)
        assert report.total_usage_cost_usd is not None

    @patch("vaig.tools.gke.cost_estimation._create_k8s_clients")
    @patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True)
    def test_no_metrics_increments_without_counter(
        self, _mock_autopilot: MagicMock, mock_clients: MagicMock
    ) -> None:
        from vaig.tools.gke.cost_estimation import fetch_workload_costs

        pod = _make_running_pod()
        core_v1 = _make_mock_clients([pod])
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        # No usage metrics available
        with patch(
            "vaig.tools.gke.monitoring.get_workload_usage_metrics",
            return_value={},
        ):
            report = fetch_workload_costs(_make_gke_config())

        assert report.workloads_without_metrics == 1
        assert report.workloads_with_full_metrics == 0
        assert report.workloads_with_partial_metrics == 0
        # Usage is now estimated from requests (fallback), not None
        assert report.total_usage_cost_usd is not None
        assert report.metrics_estimated is True


# ── FIX A2: monitoring errors surfaced in GKECostReport ───────


class TestMonitoringErrorSurfacing:
    """Monitoring failures are logged at ERROR and reflected in monitoring_status."""

    @patch("vaig.tools.gke.cost_estimation._create_k8s_clients")
    @patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True)
    def test_monitoring_exception_sets_monitoring_status(
        self,
        _mock_autopilot: MagicMock,
        mock_clients: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from vaig.tools.gke.cost_estimation import fetch_workload_costs

        pod = _make_running_pod()
        core_v1 = _make_mock_clients([pod])
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        boom = RuntimeError("Cloud Monitoring unavailable")

        # Attach caplog handler to the monitoring logger so we can verify
        # that the exception is logged as a WARNING from the retry helper.
        monitoring_logger = logging.getLogger("vaig.tools.gke.monitoring")
        monitoring_logger.addHandler(caplog.handler)
        try:
            with patch(
                "vaig.tools.gke.monitoring.get_workload_usage_metrics",
                side_effect=boom,
            ), caplog.at_level(logging.WARNING, logger="vaig.tools.gke.monitoring"):
                report = fetch_workload_costs(_make_gke_config())
        finally:
            monitoring_logger.removeHandler(caplog.handler)

        # monitoring_status must be set — multi-source pipeline reports no_data
        # when all layers (L1 monitoring, L2 metrics-server, L3 datadog) fail.
        assert report.monitoring_status is not None
        assert "no_data" in report.monitoring_status

        # Exception must be logged at WARNING level by the retry helper
        warn_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warn_records) >= 1
        combined = " ".join(str(r.getMessage()) for r in warn_records)
        assert "Cloud Monitoring unavailable" in combined

    @patch("vaig.tools.gke.cost_estimation._create_k8s_clients")
    @patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True)
    def test_no_monitoring_error_empty_metrics_sets_no_data_status(
        self, _mock_autopilot: MagicMock, mock_clients: MagicMock
    ) -> None:
        """When monitoring returns empty results, status reflects no_data (not None)."""
        from vaig.tools.gke.cost_estimation import fetch_workload_costs

        pod = _make_running_pod()
        core_v1 = _make_mock_clients([pod])
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch(
            "vaig.tools.gke.monitoring.get_workload_usage_metrics",
            return_value={},
        ):
            report = fetch_workload_costs(_make_gke_config())

        # Empty metrics should now be reflected in monitoring_status rather than None
        assert report.monitoring_status is not None
        assert "no_data" in report.monitoring_status


# ── FIX B: _inject_report_metadata namespace logic ────────────


class TestInjectReportMetadataNamespaceFiltering:
    """_inject_report_metadata passes correct namespaces to fetch_workload_costs."""

    def _make_report_with_metadata(self) -> MagicMock:
        report = MagicMock()
        metadata = MagicMock()
        metadata.gke_cost = None
        report.metadata = metadata
        return report

    def test_default_namespace_passed_when_cost_namespaces_none(self) -> None:
        """When cost_namespaces=None and default_namespace is set, uses default_namespace."""
        from vaig.cli.commands.live import _inject_report_metadata

        report = self._make_report_with_metadata()
        gke_config = _make_gke_config(default_namespace="production")

        with patch(
            "vaig.tools.gke.cost_estimation.fetch_workload_costs"
        ) as mock_fetch:
            mock_fetch.return_value = MagicMock()
            _inject_report_metadata(
                report,
                gke_config=gke_config,
                cost_namespaces=None,  # should fall back to default_namespace
            )

        assert mock_fetch.called
        _, kwargs = mock_fetch.call_args
        assert kwargs.get("namespaces") == ["production"]

    def test_all_namespaces_passes_none_to_fetch(self) -> None:
        """When cost_namespaces=[] (--all-namespaces), effective_namespaces=None."""
        from vaig.cli.commands.live import _inject_report_metadata

        report = self._make_report_with_metadata()
        gke_config = _make_gke_config(default_namespace="production")

        with patch(
            "vaig.tools.gke.cost_estimation.fetch_workload_costs"
        ) as mock_fetch:
            mock_fetch.return_value = MagicMock()
            _inject_report_metadata(
                report,
                gke_config=gke_config,
                cost_namespaces=[],  # --all-namespaces → empty list → None effective
            )

        assert mock_fetch.called
        _, kwargs = mock_fetch.call_args
        # Empty list maps to None (all namespaces)
        assert kwargs.get("namespaces") is None

    def test_explicit_namespaces_passed_through(self) -> None:
        """When cost_namespaces=[...], that exact list is passed."""
        from vaig.cli.commands.live import _inject_report_metadata

        report = self._make_report_with_metadata()
        gke_config = _make_gke_config(default_namespace="default")

        with patch(
            "vaig.tools.gke.cost_estimation.fetch_workload_costs"
        ) as mock_fetch:
            mock_fetch.return_value = MagicMock()
            _inject_report_metadata(
                report,
                gke_config=gke_config,
                cost_namespaces=["ns-a", "ns-b"],
            )

        assert mock_fetch.called
        _, kwargs = mock_fetch.call_args
        assert kwargs.get("namespaces") == ["ns-a", "ns-b"]

    def test_no_default_namespace_and_none_cost_namespaces_passes_none(self) -> None:
        """Legacy: no default_namespace + cost_namespaces=None → all namespaces."""
        from vaig.cli.commands.live import _inject_report_metadata

        report = self._make_report_with_metadata()
        gke_config = _make_gke_config(default_namespace="")  # no default

        with patch(
            "vaig.tools.gke.cost_estimation.fetch_workload_costs"
        ) as mock_fetch:
            mock_fetch.return_value = MagicMock()
            _inject_report_metadata(
                report,
                gke_config=gke_config,
                cost_namespaces=None,
            )

        assert mock_fetch.called
        _, kwargs = mock_fetch.call_args
        assert kwargs.get("namespaces") is None


# ── FIX B: fetch_workload_costs namespace filtering ───────────


class TestFetchWorkloadCostsNamespaceFiltering:
    """fetch_workload_costs respects the namespaces argument."""

    @patch("vaig.tools.gke.cost_estimation._create_k8s_clients")
    @patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True)
    def test_explicit_namespace_restricts_to_that_namespace(
        self, _mock_autopilot: MagicMock, mock_clients: MagicMock
    ) -> None:
        """Passing namespaces=['default'] only processes that namespace."""
        from vaig.tools.gke.cost_estimation import fetch_workload_costs

        pod_default = _make_running_pod(namespace="default", name="app-def-0")

        core_v1 = MagicMock()
        pod_list = MagicMock()
        pod_list.items = [pod_default]
        core_v1.list_namespaced_pod.return_value = pod_list
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch(
            "vaig.tools.gke.monitoring.get_workload_usage_metrics",
            return_value={},
        ):
            report = fetch_workload_costs(
                _make_gke_config(), namespaces=["default"]
            )

        # The call should have been made for "default" namespace only
        core_v1.list_namespaced_pod.assert_called_with("default")
        workload_namespaces = {wl.namespace for wl in report.workloads}
        assert "default" in workload_namespaces

    @patch("vaig.tools.gke.cost_estimation._create_k8s_clients")
    @patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True)
    def test_none_namespaces_queries_list_namespace(
        self, _mock_autopilot: MagicMock, mock_clients: MagicMock
    ) -> None:
        """namespaces=None triggers list_namespace() to discover all namespaces."""
        from vaig.tools.gke.cost_estimation import fetch_workload_costs

        pod_default = _make_running_pod(namespace="default", name="app-def-0")
        pod_staging = _make_running_pod(namespace="staging", name="app-stg-0")

        core_v1 = _make_mock_clients(
            [pod_default, pod_staging],
            namespaces=["default", "staging"],
        )
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch(
            "vaig.tools.gke.monitoring.get_workload_usage_metrics",
            return_value={},
        ):
            report = fetch_workload_costs(
                _make_gke_config(default_namespace=""), namespaces=None
            )

        # list_namespace should have been called to discover namespaces
        core_v1.list_namespace.assert_called_once()
        workload_namespaces = {wl.namespace for wl in report.workloads}
        assert "default" in workload_namespaces
        assert "staging" in workload_namespaces

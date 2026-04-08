"""Tests for cost estimation fallback when Cloud Monitoring is unavailable.

When Cloud Monitoring metrics are unavailable, the cost estimation pipeline
should estimate usage costs from resource requests (100% utilization) instead
of reporting N/A / None.

Covers:
- Fallback estimation: usage = requests when monitoring returns empty
- metrics_estimated flag on GKEWorkloadCost and GKECostReport
- Display output shows "(est.)" label for estimated values
- Mixed scenario: some workloads with real metrics, some estimated
"""

from __future__ import annotations

from io import StringIO
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
    owner_name: str = "my-app",
) -> MagicMock:
    """Build a minimal mock pod object matching fetch_workload_costs expectations."""
    pod = MagicMock()
    pod.status.phase = "Running"
    pod.metadata.namespace = namespace
    pod.metadata.name = name
    pod.metadata.labels = {}
    owner = MagicMock()
    owner.kind = "Deployment"
    owner.name = owner_name
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
    """Build a mock core_v1 client that returns the given pods."""
    if namespaces is None:
        namespaces = sorted({p.metadata.namespace for p in pods})

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


# ── Fallback estimation tests ────────────────────────────────


class TestFallbackEstimationNoMonitoring:
    """When monitoring returns empty metrics, usage should be estimated from requests."""

    @patch("vaig.tools.gke.cost_estimation._create_k8s_clients")
    @patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True)
    def test_no_monitoring_data_estimates_usage_from_requests(
        self, _mock_autopilot: MagicMock, mock_clients: MagicMock
    ) -> None:
        """Usage cost should NOT be None when monitoring returns empty — should be estimated."""
        from vaig.tools.gke.cost_estimation import fetch_workload_costs

        pod = _make_running_pod()
        core_v1 = _make_mock_clients([pod])
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch(
            "vaig.tools.gke.monitoring.get_workload_usage_metrics",
            return_value={},
        ):
            report = fetch_workload_costs(_make_gke_config())

        # Usage cost should be estimated, NOT None
        assert report.total_usage_cost_usd is not None
        assert report.total_usage_cost_usd > 0.0
        # Estimated usage = requests → savings should be 0 (no waste)
        assert report.total_savings_usd is not None
        assert report.total_savings_usd == pytest.approx(0.0, abs=0.01)

    @patch("vaig.tools.gke.cost_estimation._create_k8s_clients")
    @patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True)
    def test_no_monitoring_sets_metrics_estimated_on_workload(
        self, _mock_autopilot: MagicMock, mock_clients: MagicMock
    ) -> None:
        """Individual workload should have metrics_estimated=True."""
        from vaig.tools.gke.cost_estimation import fetch_workload_costs

        pod = _make_running_pod()
        core_v1 = _make_mock_clients([pod])
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch(
            "vaig.tools.gke.monitoring.get_workload_usage_metrics",
            return_value={},
        ):
            report = fetch_workload_costs(_make_gke_config())

        assert len(report.workloads) == 1
        wl = report.workloads[0]
        assert wl.metrics_estimated is True
        assert wl.total_usage_cost_usd is not None

    @patch("vaig.tools.gke.cost_estimation._create_k8s_clients")
    @patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True)
    def test_no_monitoring_sets_metrics_estimated_on_report(
        self, _mock_autopilot: MagicMock, mock_clients: MagicMock
    ) -> None:
        """Report-level metrics_estimated flag should be True."""
        from vaig.tools.gke.cost_estimation import fetch_workload_costs

        pod = _make_running_pod()
        core_v1 = _make_mock_clients([pod])
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch(
            "vaig.tools.gke.monitoring.get_workload_usage_metrics",
            return_value={},
        ):
            report = fetch_workload_costs(_make_gke_config())

        assert report.metrics_estimated is True

    @patch("vaig.tools.gke.cost_estimation._create_k8s_clients")
    @patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True)
    def test_no_monitoring_workloads_without_metrics_still_counted(
        self, _mock_autopilot: MagicMock, mock_clients: MagicMock
    ) -> None:
        """Workloads without real metrics should still be counted in workloads_without_metrics."""
        from vaig.tools.gke.cost_estimation import fetch_workload_costs

        pod = _make_running_pod()
        core_v1 = _make_mock_clients([pod])
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch(
            "vaig.tools.gke.monitoring.get_workload_usage_metrics",
            return_value={},
        ):
            report = fetch_workload_costs(_make_gke_config())

        assert report.workloads_without_metrics == 1
        assert report.workloads_with_full_metrics == 0

    @patch("vaig.tools.gke.cost_estimation._create_k8s_clients")
    @patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True)
    def test_estimated_usage_equals_request_cost(
        self, _mock_autopilot: MagicMock, mock_clients: MagicMock
    ) -> None:
        """When estimated, usage cost should equal request cost (100% utilization)."""
        from vaig.tools.gke.cost_estimation import fetch_workload_costs

        pod = _make_running_pod()
        core_v1 = _make_mock_clients([pod])
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch(
            "vaig.tools.gke.monitoring.get_workload_usage_metrics",
            return_value={},
        ):
            report = fetch_workload_costs(_make_gke_config())

        wl = report.workloads[0]
        # Usage cost should equal request cost (usage = requests → same cost)
        assert wl.total_usage_cost_usd == pytest.approx(wl.total_request_cost_usd, rel=0.01)


class TestFallbackEstimationEmptyContainers:
    """When monitoring returns WorkloadUsageMetrics with empty containers={},
    usage should still be estimated from requests.

    This covers the edge case where Cloud Monitoring is reachable and returns
    a WorkloadUsageMetrics object, but the containers dict is empty (no data
    points).  Previously this fell through both branches (c_usage was falsy,
    but wl_usage_metrics was not None), leaving usage as None / N/A.
    """

    @patch("vaig.tools.gke.cost_estimation._create_k8s_clients")
    @patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True)
    def test_empty_containers_estimates_usage(
        self, _mock_autopilot: MagicMock, mock_clients: MagicMock
    ) -> None:
        """Usage cost should be estimated when containers dict is empty."""
        from vaig.tools.gke.cost_estimation import fetch_workload_costs

        pod = _make_running_pod()
        core_v1 = _make_mock_clients([pod])
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        # Monitoring returns a WorkloadUsageMetrics with empty containers
        usage_metrics = {
            "my-app": WorkloadUsageMetrics(
                namespace="default",
                workload_name="my-app",
                containers={},  # ← empty: monitoring returned no data points
            )
        }

        with patch(
            "vaig.tools.gke.monitoring.get_workload_usage_metrics",
            return_value=usage_metrics,
        ):
            report = fetch_workload_costs(_make_gke_config())

        assert report.total_usage_cost_usd is not None
        assert report.total_usage_cost_usd > 0.0
        assert report.total_savings_usd is not None

    @patch("vaig.tools.gke.cost_estimation._create_k8s_clients")
    @patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True)
    def test_empty_containers_sets_metrics_estimated(
        self, _mock_autopilot: MagicMock, mock_clients: MagicMock
    ) -> None:
        """metrics_estimated should be True when containers dict is empty."""
        from vaig.tools.gke.cost_estimation import fetch_workload_costs

        pod = _make_running_pod()
        core_v1 = _make_mock_clients([pod])
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        usage_metrics = {
            "my-app": WorkloadUsageMetrics(
                namespace="default",
                workload_name="my-app",
                containers={},
            )
        }

        with patch(
            "vaig.tools.gke.monitoring.get_workload_usage_metrics",
            return_value=usage_metrics,
        ):
            report = fetch_workload_costs(_make_gke_config())

        assert report.metrics_estimated is True
        wl = report.workloads[0]
        assert wl.metrics_estimated is True

    @patch("vaig.tools.gke.cost_estimation._create_k8s_clients")
    @patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True)
    def test_empty_containers_usage_equals_requests(
        self, _mock_autopilot: MagicMock, mock_clients: MagicMock
    ) -> None:
        """When estimated from empty containers, usage cost = request cost."""
        from vaig.tools.gke.cost_estimation import fetch_workload_costs

        pod = _make_running_pod()
        core_v1 = _make_mock_clients([pod])
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        usage_metrics = {
            "my-app": WorkloadUsageMetrics(
                namespace="default",
                workload_name="my-app",
                containers={},
            )
        }

        with patch(
            "vaig.tools.gke.monitoring.get_workload_usage_metrics",
            return_value=usage_metrics,
        ):
            report = fetch_workload_costs(_make_gke_config())

        wl = report.workloads[0]
        assert wl.total_usage_cost_usd == pytest.approx(wl.total_request_cost_usd, rel=0.01)

    @patch("vaig.tools.gke.cost_estimation._create_k8s_clients")
    @patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True)
    def test_empty_containers_per_container_costs_populated(
        self, _mock_autopilot: MagicMock, mock_clients: MagicMock
    ) -> None:
        """Per-container cost breakdown should be populated when estimated."""
        from vaig.tools.gke.cost_estimation import fetch_workload_costs

        pod = _make_running_pod()
        core_v1 = _make_mock_clients([pod])
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        usage_metrics = {
            "my-app": WorkloadUsageMetrics(
                namespace="default",
                workload_name="my-app",
                containers={},
            )
        }

        with patch(
            "vaig.tools.gke.monitoring.get_workload_usage_metrics",
            return_value=usage_metrics,
        ):
            report = fetch_workload_costs(_make_gke_config())

        wl = report.workloads[0]
        # Container-level costs should be present (not empty list)
        assert len(wl.containers) > 0, "containers list should not be empty in fallback"

        container = wl.containers[0]
        assert container.container_name == "app"
        # Per-container usage and waste costs should be populated, not None
        assert container.total_request_cost_usd is not None
        assert container.total_request_cost_usd > 0.0
        assert container.total_usage_cost_usd is not None
        assert container.total_usage_cost_usd > 0.0
        assert container.total_waste_usd is not None
        # Since usage = requests (100% utilization), per-container waste should be ~0
        assert container.total_waste_usd == pytest.approx(0.0, abs=0.01)
        # Usage cost should match request cost for each container
        assert container.total_usage_cost_usd == pytest.approx(
            container.total_request_cost_usd, rel=0.01
        )


class TestFallbackEstimationWithRealMetrics:
    """When monitoring returns real data, metrics_estimated should be False."""

    @patch("vaig.tools.gke.cost_estimation._create_k8s_clients")
    @patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True)
    def test_real_metrics_not_marked_as_estimated(
        self, _mock_autopilot: MagicMock, mock_clients: MagicMock
    ) -> None:
        from vaig.tools.gke.cost_estimation import fetch_workload_costs

        pod = _make_running_pod()
        core_v1 = _make_mock_clients([pod])
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

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

        assert report.metrics_estimated is False
        wl = report.workloads[0]
        assert wl.metrics_estimated is False
        assert wl.total_usage_cost_usd is not None


class TestFallbackEstimationMixed:
    """Mixed scenario: some workloads have real metrics, some are estimated."""

    @patch("vaig.tools.gke.cost_estimation._create_k8s_clients")
    @patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True)
    def test_mixed_real_and_estimated(
        self, _mock_autopilot: MagicMock, mock_clients: MagicMock
    ) -> None:
        from vaig.tools.gke.cost_estimation import fetch_workload_costs

        pod1 = _make_running_pod(name="app1-abc12", owner_name="app1")
        pod2 = _make_running_pod(name="app2-def34", owner_name="app2")
        core_v1 = _make_mock_clients([pod1, pod2])
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        # Only app1 has real metrics, app2 does not
        usage_metrics = {
            "app1": WorkloadUsageMetrics(
                namespace="default",
                workload_name="app1",
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

        # Report should be marked as estimated (at least one workload is)
        assert report.metrics_estimated is True
        assert report.total_usage_cost_usd is not None

        # Find the workloads
        wl_by_name = {wl.workload_name: wl for wl in report.workloads}
        assert wl_by_name["app1"].metrics_estimated is False
        assert wl_by_name["app2"].metrics_estimated is True


class TestFallbackEstimationMonitoringException:
    """When monitoring raises an exception, usage should still be estimated."""

    @patch("vaig.tools.gke.cost_estimation._create_k8s_clients")
    @patch("vaig.tools.gke.cost_estimation.detect_autopilot", return_value=True)
    def test_monitoring_exception_still_estimates(
        self, _mock_autopilot: MagicMock, mock_clients: MagicMock
    ) -> None:
        from vaig.tools.gke.cost_estimation import fetch_workload_costs

        pod = _make_running_pod()
        core_v1 = _make_mock_clients([pod])
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch(
            "vaig.tools.gke.monitoring.get_workload_usage_metrics",
            side_effect=RuntimeError("Permission denied"),
        ):
            report = fetch_workload_costs(_make_gke_config())

        # Should still estimate usage from requests
        assert report.total_usage_cost_usd is not None
        assert report.metrics_estimated is True
        assert report.workloads_without_metrics == 1


# ── Display output tests ─────────────────────────────────────


class TestDisplayEstimatedLabel:
    """Cost breakdown table should show (est.) label for estimated values."""

    def _make_health_report_with_cost(
        self,
        metrics_estimated: bool = False,
        wl_metrics_estimated: bool = False,
    ) -> MagicMock:
        """Build a mock HealthReport with GKE cost data."""
        from vaig.skills.service_health.schema import (
            GKECostReport,
            GKEResourceCost,
            GKEWorkloadCost,
        )

        wl = GKEWorkloadCost(
            namespace="default",
            workload_name="my-app",
            resource_costs=[
                GKEResourceCost(
                    resource_type="cpu",
                    requests=0.1,
                    usage=0.1,
                    request_cost_usd=5.0,
                    usage_cost_usd=5.0,
                    waste_cost_usd=0.0,
                ),
            ],
            total_request_cost_usd=5.0,
            total_usage_cost_usd=5.0,
            total_waste_usd=0.0,
            containers=[],
            partial_metrics=False,
            metrics_estimated=wl_metrics_estimated,
        )

        gke_cost = GKECostReport(
            cluster_type="autopilot",
            region="us-central1",
            supported=True,
            workloads=[wl],
            total_request_cost_usd=5.0,
            total_usage_cost_usd=5.0,
            total_savings_usd=0.0,
            metrics_estimated=metrics_estimated,
        )

        report = MagicMock()
        report.metadata.gke_cost = gke_cost
        return report

    def test_estimated_workload_shows_est_label(self) -> None:
        from rich.console import Console

        from vaig.cli.display import print_cost_breakdown_table

        report = self._make_health_report_with_cost(
            metrics_estimated=True, wl_metrics_estimated=True
        )
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120)
        print_cost_breakdown_table(report, console=console)
        output = buf.getvalue()

        assert "(est.)" in output

    def test_real_metrics_no_est_label(self) -> None:
        from rich.console import Console

        from vaig.cli.display import print_cost_breakdown_table

        report = self._make_health_report_with_cost(
            metrics_estimated=False, wl_metrics_estimated=False
        )
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120)
        print_cost_breakdown_table(report, console=console)
        output = buf.getvalue()

        assert "(est.)" not in output

    def test_estimated_report_shows_warning_note(self) -> None:
        from rich.console import Console

        from vaig.cli.display import print_cost_breakdown_table

        report = self._make_health_report_with_cost(
            metrics_estimated=True, wl_metrics_estimated=True
        )
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120)
        print_cost_breakdown_table(report, console=console)
        output = buf.getvalue()

        assert "100% utilization" in output or "estimated" in output.lower()


# ── Schema field tests ───────────────────────────────────────


class TestSchemaMetricsEstimated:
    """GKEWorkloadCost and GKECostReport should have metrics_estimated field."""

    def test_workload_cost_has_metrics_estimated_default_false(self) -> None:
        from vaig.skills.service_health.schema import GKEWorkloadCost

        wl = GKEWorkloadCost()
        assert wl.metrics_estimated is False

    def test_workload_cost_metrics_estimated_true(self) -> None:
        from vaig.skills.service_health.schema import GKEWorkloadCost

        wl = GKEWorkloadCost(metrics_estimated=True)
        assert wl.metrics_estimated is True

    def test_report_has_metrics_estimated_default_false(self) -> None:
        from vaig.skills.service_health.schema import GKECostReport

        report = GKECostReport()
        assert report.metrics_estimated is False

    def test_report_metrics_estimated_true(self) -> None:
        from vaig.skills.service_health.schema import GKECostReport

        report = GKECostReport(metrics_estimated=True)
        assert report.metrics_estimated is True

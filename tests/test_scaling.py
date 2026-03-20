"""Tests for GKE scaling support — _format_vpa_section(), get_scaling_status().

Covers:
- VPA section formatting (_format_vpa_section)
- HPA section formatting (_format_hpa_section)
- get_scaling_status() with all metric types, edge cases, and error scenarios
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vaig.core.config import GKEConfig
from vaig.tools.base import ToolResult

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


def _make_hpa_object(
    name: str = "my-hpa",
    current_replicas: int = 3,
    desired_replicas: int = 3,
    min_replicas: int = 2,
    max_replicas: int = 10,
    spec_metrics: list | None = None,
    current_metrics: list | None = None,
    conditions: list | None = None,
    scale_target_name: str = "my-deploy",
    scale_target_kind: str = "Deployment",
) -> MagicMock:
    """Create a realistic mock HPA object mirroring kubernetes V2 HPA."""
    hpa = MagicMock()
    hpa.metadata.name = name

    hpa.spec.min_replicas = min_replicas
    hpa.spec.max_replicas = max_replicas
    hpa.spec.scale_target_ref.kind = scale_target_kind
    hpa.spec.scale_target_ref.name = scale_target_name
    hpa.spec.metrics = spec_metrics or []

    hpa.status.current_replicas = current_replicas
    hpa.status.desired_replicas = desired_replicas
    hpa.status.conditions = conditions or []
    hpa.status.current_metrics = current_metrics or []

    return hpa


def _make_metric_spec(
    mtype: str = "Resource",
    name: str = "cpu",
    target_type: str = "Utilization",
    average_utilization: int | None = 80,
    average_value: str | None = None,
) -> MagicMock:
    """Build a mock spec.metrics entry with to_dict() support."""
    m = MagicMock()

    if mtype == "Resource":
        d = {
            "type": "Resource",
            "resource": {
                "name": name,
                "target": {
                    "type": target_type,
                    "averageUtilization": average_utilization,
                    "averageValue": average_value,
                },
            },
        }
    elif mtype == "External":
        d = {
            "type": "External",
            "external": {
                "metric": {"name": name},
                "target": {"averageValue": average_value or "100"},
            },
        }
    elif mtype == "Pods":
        d = {
            "type": "Pods",
            "pods": {
                "metric": {"name": name},
                "target": {"averageValue": average_value or "10"},
            },
        }
    elif mtype == "Object":
        d = {
            "type": "Object",
            "object": {
                "metric": {"name": name},
                "target": {"type": "Value", "value": average_value or "50"},
            },
        }
    else:
        d = {"type": mtype}

    m.to_dict.return_value = d
    return m


def _make_current_metric(
    mtype: str = "Resource",
    name: str = "cpu",
    average_utilization: int | None = 65,
    average_value: str | None = None,
) -> MagicMock:
    """Build a mock status.currentMetrics entry with to_dict() support."""
    m = MagicMock()

    if mtype == "Resource":
        d = {
            "type": "Resource",
            "resource": {
                "name": name,
                "current": {
                    "averageUtilization": average_utilization,
                    "averageValue": average_value,
                },
            },
        }
    elif mtype == "External":
        d = {
            "type": "External",
            "external": {
                "metric": {"name": name},
                "current": {"averageValue": average_value or "45"},
            },
        }
    elif mtype == "Pods":
        d = {
            "type": "Pods",
            "pods": {
                "metric": {"name": name},
                "current": {"averageValue": average_value or "7"},
            },
        }
    else:
        d = {"type": mtype}

    m.to_dict.return_value = d
    return m


def _make_vpa_dict(
    name: str = "my-vpa",
    target_name: str = "my-deploy",
    target_kind: str = "Deployment",
    update_mode: str = "Auto",
    container_recommendations: list | None = None,
    controlled_values: str = "RequestsAndLimits",
) -> dict:
    """Build a plain-dict VPA object as returned by CustomObjectsApi."""
    recommendations = container_recommendations
    if recommendations is None:
        recommendations = [
            {
                "containerName": "app",
                "target": {"cpu": "250m", "memory": "256Mi"},
                "lowerBound": {"cpu": "100m", "memory": "128Mi"},
                "upperBound": {"cpu": "500m", "memory": "512Mi"},
                "uncappedTarget": {"cpu": "300m", "memory": "300Mi"},
            }
        ]

    return {
        "metadata": {"name": name, "namespace": "default"},
        "spec": {
            "targetRef": {"kind": target_kind, "name": target_name},
            "updatePolicy": {"updateMode": update_mode},
            "resourcePolicy": {
                "containerPolicies": [{"controlledValues": controlled_values}]
            },
        },
        "status": {
            "recommendation": {
                "containerRecommendations": recommendations,
            }
        },
    }


# ── _format_vpa_section tests ─────────────────────────────────


class TestFormatVpaSection:
    """Tests for _format_vpa_section() helper."""

    def test_typical_vpa_auto_mode_with_recommendations(self) -> None:
        from vaig.tools.gke.scaling import _format_vpa_section

        vpa = _make_vpa_dict(name="web-vpa", update_mode="Auto")
        result = _format_vpa_section(vpa)

        assert "web-vpa" in result
        assert "Auto" in result
        assert "app" in result
        assert "250m" in result
        assert "256Mi" in result
        assert "Recommendations" in result

    def test_vpa_mode_off(self) -> None:
        from vaig.tools.gke.scaling import _format_vpa_section

        vpa = _make_vpa_dict(update_mode="Off")
        result = _format_vpa_section(vpa)

        assert "Off" in result
        assert "Update Mode: Off" in result

    def test_vpa_no_recommendations_empty_status(self) -> None:
        from vaig.tools.gke.scaling import _format_vpa_section

        vpa = {
            "metadata": {"name": "no-recs-vpa"},
            "spec": {
                "targetRef": {"kind": "Deployment", "name": "my-deploy"},
                "updatePolicy": {"updateMode": "Auto"},
            },
            "status": {},  # No recommendation key at all
        }
        result = _format_vpa_section(vpa)

        assert "no-recs-vpa" in result
        # Should say recommendations not yet available
        assert "Not yet available" in result or "learning" in result.lower()

    def test_vpa_no_recommendations_empty_container_list(self) -> None:
        from vaig.tools.gke.scaling import _format_vpa_section

        vpa = _make_vpa_dict(container_recommendations=[])
        result = _format_vpa_section(vpa)

        assert "Not yet available" in result or "learning" in result.lower()

    def test_vpa_multiple_containers(self) -> None:
        from vaig.tools.gke.scaling import _format_vpa_section

        vpa = _make_vpa_dict(
            name="multi-container-vpa",
            container_recommendations=[
                {
                    "containerName": "app",
                    "target": {"cpu": "200m", "memory": "256Mi"},
                    "lowerBound": {"cpu": "100m", "memory": "128Mi"},
                    "upperBound": {"cpu": "400m", "memory": "512Mi"},
                    "uncappedTarget": {"cpu": "200m", "memory": "256Mi"},
                },
                {
                    "containerName": "sidecar",
                    "target": {"cpu": "50m", "memory": "64Mi"},
                    "lowerBound": {"cpu": "25m", "memory": "32Mi"},
                    "upperBound": {"cpu": "100m", "memory": "128Mi"},
                    "uncappedTarget": {"cpu": "50m", "memory": "64Mi"},
                },
            ],
        )
        result = _format_vpa_section(vpa)

        assert "app" in result
        assert "sidecar" in result
        assert "200m" in result
        assert "50m" in result

    def test_vpa_controlled_values_shown(self) -> None:
        from vaig.tools.gke.scaling import _format_vpa_section

        vpa = _make_vpa_dict(controlled_values="RequestsOnly")
        result = _format_vpa_section(vpa)

        assert "RequestsOnly" in result

    def test_vpa_missing_metadata_name(self) -> None:
        from vaig.tools.gke.scaling import _format_vpa_section

        vpa = {
            "metadata": {},
            "spec": {"updatePolicy": {"updateMode": "Auto"}},
            "status": {},
        }
        result = _format_vpa_section(vpa)

        assert "<unknown>" in result


# ── _format_hpa_section tests ─────────────────────────────────


class TestFormatHpaSection:
    """Tests for _format_hpa_section() helper."""

    def test_resource_cpu_metric(self) -> None:
        from vaig.tools.gke.scaling import _format_hpa_section

        spec_m = _make_metric_spec("Resource", "cpu", "Utilization", average_utilization=80)
        curr_m = _make_current_metric("Resource", "cpu", average_utilization=65)

        hpa = _make_hpa_object(
            name="cpu-hpa",
            current_replicas=3,
            desired_replicas=3,
            spec_metrics=[spec_m],
            current_metrics=[curr_m],
        )
        result = _format_hpa_section(hpa)

        assert "cpu-hpa" in result
        assert "cpu" in result
        assert "80%" in result   # target
        assert "65%" in result   # current
        assert "3/3" in result   # replicas

    def test_external_metric(self) -> None:
        from vaig.tools.gke.scaling import _format_hpa_section

        spec_m = _make_metric_spec("External", "pubsub-messages", average_value="100")
        curr_m = _make_current_metric("External", "pubsub-messages", average_value="45")

        hpa = _make_hpa_object(
            name="pubsub-hpa",
            spec_metrics=[spec_m],
            current_metrics=[curr_m],
        )
        result = _format_hpa_section(hpa)

        assert "External" in result
        assert "pubsub-messages" in result

    def test_hpa_at_max_replicas(self) -> None:
        from vaig.tools.gke.scaling import _format_hpa_section

        spec_m = _make_metric_spec("Resource", "cpu", "Utilization", average_utilization=80)
        curr_m = _make_current_metric("Resource", "cpu", average_utilization=95)

        hpa = _make_hpa_object(
            name="maxed-hpa",
            current_replicas=10,
            desired_replicas=10,
            min_replicas=2,
            max_replicas=10,
            spec_metrics=[spec_m],
            current_metrics=[curr_m],
        )
        result = _format_hpa_section(hpa)

        assert "10/10" in result
        assert "max: 10" in result

    def test_pods_metric(self) -> None:
        from vaig.tools.gke.scaling import _format_hpa_section

        spec_m = _make_metric_spec("Pods", "requests-per-second", average_value="10")
        curr_m = _make_current_metric("Pods", "requests-per-second", average_value="7")

        hpa = _make_hpa_object(
            name="pods-hpa",
            spec_metrics=[spec_m],
            current_metrics=[curr_m],
        )
        result = _format_hpa_section(hpa)

        assert "Pods" in result
        assert "requests-per-second" in result

    def test_no_spec_metrics(self) -> None:
        from vaig.tools.gke.scaling import _format_hpa_section

        hpa = _make_hpa_object(name="no-metrics-hpa", spec_metrics=[], current_metrics=[])
        result = _format_hpa_section(hpa)

        # Should not crash — just no metrics table
        assert "no-metrics-hpa" in result

    def test_conditions_shown(self) -> None:
        from vaig.tools.gke.scaling import _format_hpa_section

        cond = MagicMock()
        cond.type = "AbleToScale"
        cond.status = "True"

        hpa = _make_hpa_object(name="cond-hpa", conditions=[cond])
        result = _format_hpa_section(hpa)

        assert "AbleToScale=True" in result


# ── get_scaling_status tests ──────────────────────────────────


class TestGetScalingStatus:
    """Tests for get_scaling_status() — the main public tool function."""

    # ── Helper: build common patch targets ─────────────────────

    @staticmethod
    def _clients_patch(auto_v2: MagicMock, custom_api: MagicMock) -> tuple:
        """Return (result, api_client) tuple that _create_k8s_clients returns."""
        api_client = MagicMock()
        return (MagicMock(), MagicMock(), custom_api, api_client)

    # ── k8s unavailable ─────────────────────────────────────────

    def test_k8s_unavailable(self) -> None:
        from vaig.tools.gke.scaling import get_scaling_status

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.scaling._clients._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = get_scaling_status("my-deploy", gke_config=cfg)

        assert result.error is True
        assert "kubernetes" in result.output.lower()

    # ── HPA with Resource CPU metric ────────────────────────────

    @patch("vaig.tools.gke.scaling._clients._create_k8s_clients")
    @patch("kubernetes.client.AutoscalingV2Api")
    def test_hpa_resource_cpu_metric(
        self, mock_autoscaling_cls: MagicMock, mock_clients: MagicMock
    ) -> None:
        from vaig.tools.gke.scaling import get_scaling_status

        cfg = _make_gke_config()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), custom_api, api_client)

        spec_m = _make_metric_spec("Resource", "cpu", "Utilization", average_utilization=80)
        curr_m = _make_current_metric("Resource", "cpu", average_utilization=65)

        hpa = _make_hpa_object(
            name="web-hpa",
            current_replicas=3,
            desired_replicas=3,
            spec_metrics=[spec_m],
            current_metrics=[curr_m],
        )
        auto_v2_instance = MagicMock()
        auto_v2_instance.list_namespaced_horizontal_pod_autoscaler.return_value = MagicMock(
            items=[hpa]
        )
        mock_autoscaling_cls.return_value = auto_v2_instance

        # No VPA
        custom_api.list_namespaced_custom_object.return_value = {"items": []}

        with patch("vaig.tools.gke.scaling._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.scaling._K8S_AVAILABLE", True):
            result = get_scaling_status("my-deploy", gke_config=cfg)

        assert result.error is not True
        assert "my-deploy" in result.output
        assert "HPA" in result.output
        assert "✅ Active" in result.output
        assert "80%" in result.output
        assert "65%" in result.output

    # ── HPA with External metric ─────────────────────────────────

    @patch("vaig.tools.gke.scaling._clients._create_k8s_clients")
    @patch("kubernetes.client.AutoscalingV2Api")
    def test_hpa_external_metric(
        self, mock_autoscaling_cls: MagicMock, mock_clients: MagicMock
    ) -> None:
        from vaig.tools.gke.scaling import get_scaling_status

        cfg = _make_gke_config()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), custom_api, api_client)

        spec_m = _make_metric_spec("External", "pubsub/messages", average_value="100")
        curr_m = _make_current_metric("External", "pubsub/messages", average_value="45")

        hpa = _make_hpa_object(
            name="queue-hpa",
            spec_metrics=[spec_m],
            current_metrics=[curr_m],
        )
        auto_v2_instance = MagicMock()
        auto_v2_instance.list_namespaced_horizontal_pod_autoscaler.return_value = MagicMock(
            items=[hpa]
        )
        mock_autoscaling_cls.return_value = auto_v2_instance

        custom_api.list_namespaced_custom_object.return_value = {"items": []}

        with patch("vaig.tools.gke.scaling._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.scaling._K8S_AVAILABLE", True):
            result = get_scaling_status("my-deploy", gke_config=cfg)

        assert result.error is not True
        assert "External" in result.output
        assert "pubsub/messages" in result.output

    # ── HPA at maxReplicas (scaling ceiling) ────────────────────

    @patch("vaig.tools.gke.scaling._clients._create_k8s_clients")
    @patch("kubernetes.client.AutoscalingV2Api")
    def test_hpa_at_max_replicas(
        self, mock_autoscaling_cls: MagicMock, mock_clients: MagicMock
    ) -> None:
        from vaig.tools.gke.scaling import get_scaling_status

        cfg = _make_gke_config()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), custom_api, api_client)

        spec_m = _make_metric_spec("Resource", "cpu", "Utilization", average_utilization=80)
        curr_m = _make_current_metric("Resource", "cpu", average_utilization=95)

        hpa = _make_hpa_object(
            name="maxed-hpa",
            current_replicas=10,
            desired_replicas=10,
            min_replicas=2,
            max_replicas=10,
            spec_metrics=[spec_m],
            current_metrics=[curr_m],
        )
        auto_v2_instance = MagicMock()
        auto_v2_instance.list_namespaced_horizontal_pod_autoscaler.return_value = MagicMock(
            items=[hpa]
        )
        mock_autoscaling_cls.return_value = auto_v2_instance

        custom_api.list_namespaced_custom_object.return_value = {"items": []}

        with patch("vaig.tools.gke.scaling._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.scaling._K8S_AVAILABLE", True):
            result = get_scaling_status("my-deploy", gke_config=cfg)

        assert result.error is not True
        # Should show 10/10 — at max capacity
        assert "10/10" in result.output or "max: 10" in result.output

    # ── VPA with recommendations ─────────────────────────────────

    @patch("vaig.tools.gke.scaling._clients._create_k8s_clients")
    @patch("kubernetes.client.AutoscalingV2Api")
    def test_vpa_with_recommendations(
        self, mock_autoscaling_cls: MagicMock, mock_clients: MagicMock
    ) -> None:
        from vaig.tools.gke.scaling import get_scaling_status

        cfg = _make_gke_config()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), custom_api, api_client)

        # No HPA
        auto_v2_instance = MagicMock()
        auto_v2_instance.list_namespaced_horizontal_pod_autoscaler.return_value = MagicMock(
            items=[]
        )
        mock_autoscaling_cls.return_value = auto_v2_instance

        vpa = _make_vpa_dict(
            name="web-vpa",
            target_name="my-deploy",
            update_mode="Auto",
            container_recommendations=[
                {
                    "containerName": "app",
                    "target": {"cpu": "250m", "memory": "256Mi"},
                    "lowerBound": {"cpu": "100m", "memory": "128Mi"},
                    "upperBound": {"cpu": "500m", "memory": "512Mi"},
                    "uncappedTarget": {"cpu": "250m", "memory": "256Mi"},
                }
            ],
        )
        custom_api.list_namespaced_custom_object.return_value = {"items": [vpa]}

        with patch("vaig.tools.gke.scaling._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.scaling._K8S_AVAILABLE", True):
            result = get_scaling_status("my-deploy", gke_config=cfg)

        assert result.error is not True
        assert "VPA" in result.output
        assert "250m" in result.output
        assert "256Mi" in result.output
        assert "app" in result.output

    # ── No HPA, no VPA (no scaling configured) ──────────────────

    @patch("vaig.tools.gke.scaling._clients._create_k8s_clients")
    @patch("kubernetes.client.AutoscalingV2Api")
    def test_no_scaling_configured(
        self, mock_autoscaling_cls: MagicMock, mock_clients: MagicMock
    ) -> None:
        from vaig.tools.gke.scaling import get_scaling_status

        cfg = _make_gke_config()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), custom_api, api_client)

        auto_v2_instance = MagicMock()
        auto_v2_instance.list_namespaced_horizontal_pod_autoscaler.return_value = MagicMock(
            items=[]
        )
        mock_autoscaling_cls.return_value = auto_v2_instance

        custom_api.list_namespaced_custom_object.return_value = {"items": []}

        with patch("vaig.tools.gke.scaling._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.scaling._K8S_AVAILABLE", True):
            result = get_scaling_status("orphan-deploy", gke_config=cfg)

        assert result.error is not True
        assert "No autoscaler found" in result.output or "❌" in result.output
        assert "HPA" in result.output
        assert "VPA" in result.output
        # Both should show not found
        assert "Not found" in result.output or "❌ Not found" in result.output

    # ── VPA CRD not installed (ApiException 404) ─────────────────

    @patch("vaig.tools.gke.scaling._clients._create_k8s_clients")
    @patch("kubernetes.client.AutoscalingV2Api")
    def test_vpa_crd_not_installed(
        self, mock_autoscaling_cls: MagicMock, mock_clients: MagicMock
    ) -> None:
        from vaig.tools.gke.scaling import get_scaling_status

        cfg = _make_gke_config()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), custom_api, api_client)

        # HPA not found for target
        auto_v2_instance = MagicMock()
        auto_v2_instance.list_namespaced_horizontal_pod_autoscaler.return_value = MagicMock(
            items=[]
        )
        mock_autoscaling_cls.return_value = auto_v2_instance

        # Simulate 404 from VPA CRD missing
        exc_class = type("ApiException", (Exception,), {"status": 404, "reason": "Not Found"})
        custom_api.list_namespaced_custom_object.side_effect = exc_class()

        with patch("vaig.tools.gke.scaling._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.scaling._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.scaling.k8s_exceptions") as mock_exc:
            mock_exc.ApiException = exc_class
            result = get_scaling_status("my-deploy", gke_config=cfg)

        assert result.error is not True
        # Should report VPA not installed
        assert "Not installed" in result.output or "not installed" in result.output.lower()

    # ── Both HPA and VPA present ──────────────────────────────────

    @patch("vaig.tools.gke.scaling._clients._create_k8s_clients")
    @patch("kubernetes.client.AutoscalingV2Api")
    def test_both_hpa_and_vpa_present(
        self, mock_autoscaling_cls: MagicMock, mock_clients: MagicMock
    ) -> None:
        from vaig.tools.gke.scaling import get_scaling_status

        cfg = _make_gke_config()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), custom_api, api_client)

        spec_m = _make_metric_spec("Resource", "cpu", "Utilization", average_utilization=70)
        curr_m = _make_current_metric("Resource", "cpu", average_utilization=50)

        hpa = _make_hpa_object(
            name="combined-hpa",
            spec_metrics=[spec_m],
            current_metrics=[curr_m],
        )
        auto_v2_instance = MagicMock()
        auto_v2_instance.list_namespaced_horizontal_pod_autoscaler.return_value = MagicMock(
            items=[hpa]
        )
        mock_autoscaling_cls.return_value = auto_v2_instance

        vpa = _make_vpa_dict(name="combined-vpa", target_name="my-deploy", update_mode="Off")
        custom_api.list_namespaced_custom_object.return_value = {"items": [vpa]}

        with patch("vaig.tools.gke.scaling._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.scaling._K8S_AVAILABLE", True):
            result = get_scaling_status("my-deploy", gke_config=cfg)

        assert result.error is not True
        # Both active
        output = result.output
        assert "HPA: ✅ Active" in output
        assert "VPA: ✅ Active" in output
        # Assessment should warn about conflict
        assert "conflict" in output.lower() or "both" in output.lower()

    # ── HPA with Pods metric type ────────────────────────────────

    @patch("vaig.tools.gke.scaling._clients._create_k8s_clients")
    @patch("kubernetes.client.AutoscalingV2Api")
    def test_hpa_pods_metric(
        self, mock_autoscaling_cls: MagicMock, mock_clients: MagicMock
    ) -> None:
        from vaig.tools.gke.scaling import get_scaling_status

        cfg = _make_gke_config()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), custom_api, api_client)

        spec_m = _make_metric_spec("Pods", "requests-per-second", average_value="100")
        curr_m = _make_current_metric("Pods", "requests-per-second", average_value="70")

        hpa = _make_hpa_object(
            name="pods-hpa",
            spec_metrics=[spec_m],
            current_metrics=[curr_m],
        )
        auto_v2_instance = MagicMock()
        auto_v2_instance.list_namespaced_horizontal_pod_autoscaler.return_value = MagicMock(
            items=[hpa]
        )
        mock_autoscaling_cls.return_value = auto_v2_instance

        custom_api.list_namespaced_custom_object.return_value = {"items": []}

        with patch("vaig.tools.gke.scaling._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.scaling._K8S_AVAILABLE", True):
            result = get_scaling_status("my-deploy", gke_config=cfg)

        assert result.error is not True
        assert "Pods" in result.output
        assert "requests-per-second" in result.output

    # ── HPA with Object metric type ──────────────────────────────

    @patch("vaig.tools.gke.scaling._clients._create_k8s_clients")
    @patch("kubernetes.client.AutoscalingV2Api")
    def test_hpa_object_metric(
        self, mock_autoscaling_cls: MagicMock, mock_clients: MagicMock
    ) -> None:
        from vaig.tools.gke.scaling import get_scaling_status

        cfg = _make_gke_config()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), custom_api, api_client)

        spec_m = _make_metric_spec("Object", "requests-per-second", average_value="50")

        hpa = _make_hpa_object(
            name="object-hpa",
            spec_metrics=[spec_m],
            current_metrics=[],
        )
        auto_v2_instance = MagicMock()
        auto_v2_instance.list_namespaced_horizontal_pod_autoscaler.return_value = MagicMock(
            items=[hpa]
        )
        mock_autoscaling_cls.return_value = auto_v2_instance

        custom_api.list_namespaced_custom_object.return_value = {"items": []}

        with patch("vaig.tools.gke.scaling._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.scaling._K8S_AVAILABLE", True):
            result = get_scaling_status("my-deploy", gke_config=cfg)

        # Object metrics: type is "Object" — falls through to unknown type in _metric_target_value
        assert result.error is not True
        assert "my-deploy" in result.output

    # ── Client creation failure ──────────────────────────────────

    @patch("vaig.tools.gke.scaling._clients._create_k8s_clients")
    def test_client_creation_failure(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke.scaling import get_scaling_status

        cfg = _make_gke_config()
        mock_clients.return_value = ToolResult(output="Failed to configure k8s", error=True)

        with patch("vaig.tools.gke.scaling._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.scaling._K8S_AVAILABLE", True):
            result = get_scaling_status("my-deploy", gke_config=cfg)

        assert result.error is True
        assert "Failed to configure" in result.output

    # ── HPA ApiException (403 forbidden) ────────────────────────

    @patch("vaig.tools.gke.scaling._clients._create_k8s_clients")
    @patch("kubernetes.client.AutoscalingV2Api")
    def test_hpa_api_exception_403(
        self, mock_autoscaling_cls: MagicMock, mock_clients: MagicMock
    ) -> None:
        from vaig.tools.gke.scaling import get_scaling_status

        cfg = _make_gke_config()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), custom_api, api_client)

        exc_class = type("ApiException", (Exception,), {"status": 403, "reason": "Forbidden"})
        auto_v2_instance = MagicMock()
        auto_v2_instance.list_namespaced_horizontal_pod_autoscaler.side_effect = exc_class()
        mock_autoscaling_cls.return_value = auto_v2_instance

        custom_api.list_namespaced_custom_object.return_value = {"items": []}

        with patch("vaig.tools.gke.scaling._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.scaling._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.scaling.k8s_exceptions") as mock_exc:
            mock_exc.ApiException = exc_class
            result = get_scaling_status("my-deploy", gke_config=cfg)

        assert result.error is not True  # Should return a report, not raise
        assert "Access denied" in result.output or "permissions" in result.output.lower()

    # ── HPA ApiException (401 unauthenticated) ───────────────────

    @patch("vaig.tools.gke.scaling._clients._create_k8s_clients")
    @patch("kubernetes.client.AutoscalingV2Api")
    def test_hpa_api_exception_401(
        self, mock_autoscaling_cls: MagicMock, mock_clients: MagicMock
    ) -> None:
        from vaig.tools.gke.scaling import get_scaling_status

        cfg = _make_gke_config()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), custom_api, api_client)

        exc_class = type("ApiException", (Exception,), {"status": 401, "reason": "Unauthorized"})
        auto_v2_instance = MagicMock()
        auto_v2_instance.list_namespaced_horizontal_pod_autoscaler.side_effect = exc_class()
        mock_autoscaling_cls.return_value = auto_v2_instance

        custom_api.list_namespaced_custom_object.return_value = {"items": []}

        with patch("vaig.tools.gke.scaling._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.scaling._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.scaling.k8s_exceptions") as mock_exc:
            mock_exc.ApiException = exc_class
            result = get_scaling_status("my-deploy", gke_config=cfg)

        assert result.error is not True
        assert "Authentication" in result.output or "credentials" in result.output.lower()

    # ── HPA currentMetrics missing / empty ───────────────────────

    @patch("vaig.tools.gke.scaling._clients._create_k8s_clients")
    @patch("kubernetes.client.AutoscalingV2Api")
    def test_hpa_current_metrics_empty(
        self, mock_autoscaling_cls: MagicMock, mock_clients: MagicMock
    ) -> None:
        from vaig.tools.gke.scaling import get_scaling_status

        cfg = _make_gke_config()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), custom_api, api_client)

        spec_m = _make_metric_spec("Resource", "cpu", "Utilization", average_utilization=80)

        # No current metrics → status.current_metrics is None
        hpa = _make_hpa_object(
            name="pending-hpa",
            spec_metrics=[spec_m],
            current_metrics=None,
        )
        hpa.status.current_metrics = None

        auto_v2_instance = MagicMock()
        auto_v2_instance.list_namespaced_horizontal_pod_autoscaler.return_value = MagicMock(
            items=[hpa]
        )
        mock_autoscaling_cls.return_value = auto_v2_instance

        custom_api.list_namespaced_custom_object.return_value = {"items": []}

        with patch("vaig.tools.gke.scaling._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.scaling._K8S_AVAILABLE", True):
            result = get_scaling_status("my-deploy", gke_config=cfg)

        # Should not crash — current shows "?"
        assert result.error is not True
        assert "?" in result.output or "cpu" in result.output

    # ── VPA with no status field ──────────────────────────────────

    @patch("vaig.tools.gke.scaling._clients._create_k8s_clients")
    @patch("kubernetes.client.AutoscalingV2Api")
    def test_vpa_no_status_field(
        self, mock_autoscaling_cls: MagicMock, mock_clients: MagicMock
    ) -> None:
        from vaig.tools.gke.scaling import get_scaling_status

        cfg = _make_gke_config()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), custom_api, api_client)

        auto_v2_instance = MagicMock()
        auto_v2_instance.list_namespaced_horizontal_pod_autoscaler.return_value = MagicMock(
            items=[]
        )
        mock_autoscaling_cls.return_value = auto_v2_instance

        # VPA dict without "status" key at all
        vpa_no_status = {
            "metadata": {"name": "new-vpa", "namespace": "default"},
            "spec": {
                "targetRef": {"kind": "Deployment", "name": "my-deploy"},
                "updatePolicy": {"updateMode": "Auto"},
            },
            # No "status" key
        }
        custom_api.list_namespaced_custom_object.return_value = {"items": [vpa_no_status]}

        with patch("vaig.tools.gke.scaling._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.scaling._K8S_AVAILABLE", True):
            result = get_scaling_status("my-deploy", gke_config=cfg)

        assert result.error is not True
        assert "Not yet available" in result.output or "learning" in result.output.lower()

    # ── Empty namespace uses default_namespace ───────────────────

    @patch("vaig.tools.gke.scaling._clients._create_k8s_clients")
    @patch("kubernetes.client.AutoscalingV2Api")
    def test_empty_namespace_uses_default(
        self, mock_autoscaling_cls: MagicMock, mock_clients: MagicMock
    ) -> None:
        from vaig.tools.gke.scaling import get_scaling_status

        cfg = _make_gke_config(default_namespace="production")
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), custom_api, api_client)

        auto_v2_instance = MagicMock()
        auto_v2_instance.list_namespaced_horizontal_pod_autoscaler.return_value = MagicMock(
            items=[]
        )
        mock_autoscaling_cls.return_value = auto_v2_instance

        custom_api.list_namespaced_custom_object.return_value = {"items": []}

        with patch("vaig.tools.gke.scaling._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.scaling._K8S_AVAILABLE", True):
            # Pass empty namespace — should fall back to gke_config.default_namespace
            result = get_scaling_status("my-deploy", gke_config=cfg, namespace="")

        assert result.error is not True
        # The output header should mention the resolved namespace
        assert "production" in result.output

        # API calls should use "production"
        call_kwargs = auto_v2_instance.list_namespaced_horizontal_pod_autoscaler.call_args
        assert call_kwargs.kwargs.get("namespace") == "production" or \
               (call_kwargs.args and call_kwargs.args[0] == "production")

    # ── VPA ApiException 403 ─────────────────────────────────────

    @patch("vaig.tools.gke.scaling._clients._create_k8s_clients")
    @patch("kubernetes.client.AutoscalingV2Api")
    def test_vpa_api_exception_403(
        self, mock_autoscaling_cls: MagicMock, mock_clients: MagicMock
    ) -> None:
        from vaig.tools.gke.scaling import get_scaling_status

        cfg = _make_gke_config()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), custom_api, api_client)

        auto_v2_instance = MagicMock()
        auto_v2_instance.list_namespaced_horizontal_pod_autoscaler.return_value = MagicMock(
            items=[]
        )
        mock_autoscaling_cls.return_value = auto_v2_instance

        exc_class = type("ApiException", (Exception,), {"status": 403, "reason": "Forbidden"})
        custom_api.list_namespaced_custom_object.side_effect = exc_class()

        with patch("vaig.tools.gke.scaling._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.scaling._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.scaling.k8s_exceptions") as mock_exc:
            mock_exc.ApiException = exc_class
            result = get_scaling_status("my-deploy", gke_config=cfg)

        assert result.error is not True
        assert "Access denied" in result.output or "permissions" in result.output.lower()

    # ── HPA for different deployment not matched ─────────────────

    @patch("vaig.tools.gke.scaling._clients._create_k8s_clients")
    @patch("kubernetes.client.AutoscalingV2Api")
    def test_hpa_for_different_deployment_not_matched(
        self, mock_autoscaling_cls: MagicMock, mock_clients: MagicMock
    ) -> None:
        from vaig.tools.gke.scaling import get_scaling_status

        cfg = _make_gke_config()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), custom_api, api_client)

        # HPA targets "other-deploy" not "my-deploy"
        hpa = _make_hpa_object(name="other-hpa", scale_target_name="other-deploy")
        auto_v2_instance = MagicMock()
        auto_v2_instance.list_namespaced_horizontal_pod_autoscaler.return_value = MagicMock(
            items=[hpa]
        )
        mock_autoscaling_cls.return_value = auto_v2_instance

        custom_api.list_namespaced_custom_object.return_value = {"items": []}

        with patch("vaig.tools.gke.scaling._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.scaling._K8S_AVAILABLE", True):
            result = get_scaling_status("my-deploy", gke_config=cfg)

        assert result.error is not True
        # HPA for my-deploy should NOT be found
        assert "HPA: ❌ Not found" in result.output

    # ── Report header format ─────────────────────────────────────

    @patch("vaig.tools.gke.scaling._clients._create_k8s_clients")
    @patch("kubernetes.client.AutoscalingV2Api")
    def test_report_header_contains_deployment_and_namespace(
        self, mock_autoscaling_cls: MagicMock, mock_clients: MagicMock
    ) -> None:
        from vaig.tools.gke.scaling import get_scaling_status

        cfg = _make_gke_config()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), custom_api, api_client)

        auto_v2_instance = MagicMock()
        auto_v2_instance.list_namespaced_horizontal_pod_autoscaler.return_value = MagicMock(
            items=[]
        )
        mock_autoscaling_cls.return_value = auto_v2_instance
        custom_api.list_namespaced_custom_object.return_value = {"items": []}

        with patch("vaig.tools.gke.scaling._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.scaling._K8S_AVAILABLE", True):
            result = get_scaling_status("webapp", gke_config=cfg, namespace="staging")

        assert "webapp" in result.output
        assert "staging" in result.output


# ── _scaling_assessment tests ─────────────────────────────────


class TestScalingAssessment:
    """Tests for _scaling_assessment() helper."""

    def test_both_hpa_and_vpa_warns_conflict(self) -> None:
        from vaig.tools.gke.scaling import _scaling_assessment

        result = _scaling_assessment(hpa_found=True, vpa_found=True)
        assert "conflict" in result.lower() or "eviction" in result.lower()
        assert "⚠️" in result

    def test_only_hpa_recommends_vpa(self) -> None:
        from vaig.tools.gke.scaling import _scaling_assessment

        result = _scaling_assessment(hpa_found=True, vpa_found=False)
        assert "✅" in result
        assert "HPA" in result

    def test_only_vpa_recommends_hpa(self) -> None:
        from vaig.tools.gke.scaling import _scaling_assessment

        result = _scaling_assessment(hpa_found=False, vpa_found=True)
        assert "✅" in result
        assert "VPA" in result

    def test_no_scaling_recommends_both(self) -> None:
        from vaig.tools.gke.scaling import _scaling_assessment

        result = _scaling_assessment(hpa_found=False, vpa_found=False)
        assert "❌" in result
        assert "No autoscaler" in result


# ── Metric helpers tests ──────────────────────────────────────


class TestMetricHelpers:
    """Tests for internal metric helper functions."""

    def test_format_quantity_none(self) -> None:
        from vaig.tools.gke.scaling import _format_quantity

        assert _format_quantity(None) == "?"

    def test_format_quantity_value(self) -> None:
        from vaig.tools.gke.scaling import _format_quantity

        assert _format_quantity("250m") == "250m"
        assert _format_quantity("512Mi") == "512Mi"

    def test_metric_current_value_resource_utilization(self) -> None:
        from vaig.tools.gke.scaling import _metric_current_value

        cm = {
            "type": "Resource",
            "resource": {"current": {"averageUtilization": 65}},
        }
        assert _metric_current_value(cm) == "65%"

    def test_metric_current_value_resource_average_value(self) -> None:
        from vaig.tools.gke.scaling import _metric_current_value

        cm = {
            "type": "Resource",
            "resource": {"current": {"averageValue": "300m"}},
        }
        assert _metric_current_value(cm) == "300m"

    def test_metric_current_value_external(self) -> None:
        from vaig.tools.gke.scaling import _metric_current_value

        cm = {
            "type": "External",
            "external": {"current": {"averageValue": "45"}},
        }
        assert _metric_current_value(cm) == "45"

    def test_metric_current_value_pods(self) -> None:
        from vaig.tools.gke.scaling import _metric_current_value

        cm = {
            "type": "Pods",
            "pods": {"current": {"averageValue": "7"}},
        }
        assert _metric_current_value(cm) == "7"

    def test_metric_current_value_unknown_type(self) -> None:
        from vaig.tools.gke.scaling import _metric_current_value

        cm = {"type": "Unknown"}
        assert _metric_current_value(cm) == "?"

    def test_metric_target_value_resource_utilization(self) -> None:
        from vaig.tools.gke.scaling import _metric_target_value

        sm = {
            "type": "Resource",
            "resource": {
                "name": "cpu",
                "target": {"type": "Utilization", "averageUtilization": 80},
            },
        }
        mtype, name, target = _metric_target_value(sm)
        assert mtype == "Resource"
        assert name == "cpu"
        assert target == "80%"

    def test_metric_target_value_external(self) -> None:
        from vaig.tools.gke.scaling import _metric_target_value

        sm = {
            "type": "External",
            "external": {
                "metric": {"name": "queue-depth"},
                "target": {"averageValue": "100"},
            },
        }
        mtype, name, target = _metric_target_value(sm)
        assert mtype == "External"
        assert name == "queue-depth"
        assert target == "100"

    def test_metric_target_value_pods(self) -> None:
        from vaig.tools.gke.scaling import _metric_target_value

        sm = {
            "type": "Pods",
            "pods": {
                "metric": {"name": "rps"},
                "target": {"averageValue": "50"},
            },
        }
        mtype, name, target = _metric_target_value(sm)
        assert mtype == "Pods"
        assert name == "rps"
        assert target == "50"

    def test_build_current_metrics_index(self) -> None:
        from vaig.tools.gke.scaling import _build_current_metrics_index

        metrics = [
            {
                "type": "Resource",
                "resource": {"name": "cpu", "current": {"averageUtilization": 65}},
            },
            {
                "type": "Resource",
                "resource": {"name": "memory", "current": {"averageUtilization": 40}},
            },
        ]
        index = _build_current_metrics_index(metrics)

        assert index[("Resource", "cpu")] == "65%"
        assert index[("Resource", "memory")] == "40%"

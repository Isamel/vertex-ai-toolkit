"""Tests for GKE metrics API health check and query tools.

Covers:
- check_metrics_api_health() — all groups healthy, partial, none registered,
  API errors, Autopilot/Standard annotation
- query_custom_metrics() — list mode, specific metric, 404, auth errors
- query_external_metrics() — successful query, 404, missing metric_name
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


def _make_api_group(name: str) -> MagicMock:
    """Create a mock API group entry."""
    group = MagicMock()
    group.name = name
    return group


def _make_api_condition(
    cond_type: str = "Available",
    status: str = "True",
    message: str = "all checks passed",
) -> MagicMock:
    """Create a mock APIService condition."""
    cond = MagicMock()
    cond.type = cond_type
    cond.status = status
    cond.message = message
    return cond


# ── check_metrics_api_health tests ────────────────────────────


class TestCheckMetricsApiHealth:
    """Tests for check_metrics_api_health()."""

    @patch("vaig.tools.gke.metrics_api._clients")
    def test_k8s_unavailable(self, mock_clients: MagicMock) -> None:
        mock_clients._K8S_AVAILABLE = False
        mock_clients._k8s_unavailable.return_value = ToolResult(
            output="kubernetes SDK not available", error=True,
        )
        from vaig.tools.gke.metrics_api import check_metrics_api_health

        result = check_metrics_api_health(gke_config=_make_gke_config())
        assert result.error is True
        assert "kubernetes" in result.output.lower()

    @patch("vaig.tools.gke.metrics_api._clients")
    def test_client_creation_failure(self, mock_clients: MagicMock) -> None:
        mock_clients._K8S_AVAILABLE = True
        mock_clients._create_k8s_clients.return_value = ToolResult(
            output="Failed to configure Kubernetes client", error=True,
        )
        from vaig.tools.gke.metrics_api import check_metrics_api_health

        result = check_metrics_api_health(gke_config=_make_gke_config())
        assert result.error is True
        assert "Failed" in result.output

    @patch("vaig.tools.gke.metrics_api._clients")
    def test_all_groups_healthy(self, mock_clients: MagicMock) -> None:
        mock_clients._K8S_AVAILABLE = True
        api_client = MagicMock()
        mock_clients._create_k8s_clients.return_value = (
            MagicMock(), MagicMock(), MagicMock(), api_client,
        )
        mock_clients.detect_autopilot.return_value = None

        # Mock ApisApi
        mock_apis_api = MagicMock()
        groups_resp = MagicMock()
        groups_resp.groups = [
            _make_api_group("metrics.k8s.io"),
            _make_api_group("custom.metrics.k8s.io"),
            _make_api_group("external.metrics.k8s.io"),
        ]
        mock_apis_api.get_api_versions.return_value = groups_resp

        # Mock ApiregistrationV1Api
        mock_api_reg = MagicMock()
        api_svc = MagicMock()
        api_svc.status.conditions = [_make_api_condition()]
        mock_api_reg.read_api_service.return_value = api_svc

        with (
            patch("kubernetes.client.ApisApi", return_value=mock_apis_api),
            patch("kubernetes.client.ApiregistrationV1Api", return_value=mock_api_reg),
        ):
            from vaig.tools.gke.metrics_api import check_metrics_api_health

            result = check_metrics_api_health(gke_config=_make_gke_config())

        assert result.error is False
        assert "3/3" in result.output
        assert "✅" in result.output
        assert "All metrics APIs are operational" in result.output

    @patch("vaig.tools.gke.metrics_api._clients")
    def test_no_groups_registered(self, mock_clients: MagicMock) -> None:
        mock_clients._K8S_AVAILABLE = True
        api_client = MagicMock()
        mock_clients._create_k8s_clients.return_value = (
            MagicMock(), MagicMock(), MagicMock(), api_client,
        )
        mock_clients.detect_autopilot.return_value = None

        mock_apis_api = MagicMock()
        groups_resp = MagicMock()
        groups_resp.groups = []  # No groups
        mock_apis_api.get_api_versions.return_value = groups_resp

        with patch("kubernetes.client.ApisApi", return_value=mock_apis_api):
            from vaig.tools.gke.metrics_api import check_metrics_api_health

            result = check_metrics_api_health(gke_config=_make_gke_config())

        assert result.error is False
        assert "0/3" in result.output
        assert "Not Registered" in result.output
        assert "No metrics APIs are available" in result.output

    @patch("vaig.tools.gke.metrics_api._clients")
    def test_partial_groups_registered(self, mock_clients: MagicMock) -> None:
        mock_clients._K8S_AVAILABLE = True
        api_client = MagicMock()
        mock_clients._create_k8s_clients.return_value = (
            MagicMock(), MagicMock(), MagicMock(), api_client,
        )
        mock_clients.detect_autopilot.return_value = None

        mock_apis_api = MagicMock()
        groups_resp = MagicMock()
        groups_resp.groups = [
            _make_api_group("metrics.k8s.io"),
        ]
        mock_apis_api.get_api_versions.return_value = groups_resp

        mock_api_reg = MagicMock()
        api_svc = MagicMock()
        api_svc.status.conditions = [_make_api_condition()]
        mock_api_reg.read_api_service.return_value = api_svc

        with (
            patch("kubernetes.client.ApisApi", return_value=mock_apis_api),
            patch("kubernetes.client.ApiregistrationV1Api", return_value=mock_api_reg),
        ):
            from vaig.tools.gke.metrics_api import check_metrics_api_health

            result = check_metrics_api_health(gke_config=_make_gke_config())

        assert result.error is False
        assert "1/3" in result.output
        assert "Some metrics APIs are unavailable" in result.output

    @patch("vaig.tools.gke.metrics_api._clients")
    def test_autopilot_annotation(self, mock_clients: MagicMock) -> None:
        mock_clients._K8S_AVAILABLE = True
        api_client = MagicMock()
        mock_clients._create_k8s_clients.return_value = (
            MagicMock(), MagicMock(), MagicMock(), api_client,
        )
        mock_clients.detect_autopilot.return_value = True

        mock_apis_api = MagicMock()
        groups_resp = MagicMock()
        groups_resp.groups = [_make_api_group("metrics.k8s.io")]
        mock_apis_api.get_api_versions.return_value = groups_resp

        mock_api_reg = MagicMock()
        api_svc = MagicMock()
        api_svc.status.conditions = [_make_api_condition()]
        mock_api_reg.read_api_service.return_value = api_svc

        with (
            patch("kubernetes.client.ApisApi", return_value=mock_apis_api),
            patch("kubernetes.client.ApiregistrationV1Api", return_value=mock_api_reg),
        ):
            from vaig.tools.gke.metrics_api import check_metrics_api_health

            result = check_metrics_api_health(gke_config=_make_gke_config())

        assert result.error is False
        assert "Autopilot" in result.output

    @patch("vaig.tools.gke.metrics_api._clients")
    def test_standard_cluster_annotation(self, mock_clients: MagicMock) -> None:
        mock_clients._K8S_AVAILABLE = True
        api_client = MagicMock()
        mock_clients._create_k8s_clients.return_value = (
            MagicMock(), MagicMock(), MagicMock(), api_client,
        )
        mock_clients.detect_autopilot.return_value = False

        mock_apis_api = MagicMock()
        groups_resp = MagicMock()
        groups_resp.groups = [_make_api_group("metrics.k8s.io")]
        mock_apis_api.get_api_versions.return_value = groups_resp

        mock_api_reg = MagicMock()
        api_svc = MagicMock()
        api_svc.status.conditions = [_make_api_condition()]
        mock_api_reg.read_api_service.return_value = api_svc

        with (
            patch("kubernetes.client.ApisApi", return_value=mock_apis_api),
            patch("kubernetes.client.ApiregistrationV1Api", return_value=mock_api_reg),
        ):
            from vaig.tools.gke.metrics_api import check_metrics_api_health

            result = check_metrics_api_health(gke_config=_make_gke_config())

        assert result.error is False
        assert "Standard" in result.output

    @patch("vaig.tools.gke.metrics_api._clients")
    def test_api_service_unavailable_status(self, mock_clients: MagicMock) -> None:
        mock_clients._K8S_AVAILABLE = True
        api_client = MagicMock()
        mock_clients._create_k8s_clients.return_value = (
            MagicMock(), MagicMock(), MagicMock(), api_client,
        )
        mock_clients.detect_autopilot.return_value = None

        mock_apis_api = MagicMock()
        groups_resp = MagicMock()
        groups_resp.groups = [_make_api_group("metrics.k8s.io")]
        mock_apis_api.get_api_versions.return_value = groups_resp

        mock_api_reg = MagicMock()
        api_svc = MagicMock()
        api_svc.status.conditions = [
            _make_api_condition(status="False", message="failing health check"),
        ]
        mock_api_reg.read_api_service.return_value = api_svc

        with (
            patch("kubernetes.client.ApisApi", return_value=mock_apis_api),
            patch("kubernetes.client.ApiregistrationV1Api", return_value=mock_api_reg),
        ):
            from vaig.tools.gke.metrics_api import check_metrics_api_health

            result = check_metrics_api_health(gke_config=_make_gke_config())

        assert result.error is False
        assert "❌" in result.output
        assert "Unavailable" in result.output
        assert "failing health check" in result.output


# ── query_custom_metrics tests ────────────────────────────────


class TestQueryCustomMetrics:
    """Tests for query_custom_metrics()."""

    @patch("vaig.tools.gke.metrics_api._clients")
    def test_k8s_unavailable(self, mock_clients: MagicMock) -> None:
        mock_clients._K8S_AVAILABLE = False
        mock_clients._k8s_unavailable.return_value = ToolResult(
            output="kubernetes SDK not available", error=True,
        )
        from vaig.tools.gke.metrics_api import query_custom_metrics

        result = query_custom_metrics(gke_config=_make_gke_config())
        assert result.error is True

    @patch("vaig.tools.gke.metrics_api._clients")
    def test_list_mode_with_metrics(self, mock_clients: MagicMock) -> None:
        mock_clients._K8S_AVAILABLE = True
        custom_api = MagicMock()
        mock_clients._create_k8s_clients.return_value = (
            MagicMock(), MagicMock(), custom_api, MagicMock(),
        )
        custom_api.get_api_resources.return_value = {
            "resources": [
                {"name": "requests_per_second"},
                {"name": "queue_depth"},
            ],
        }
        from vaig.tools.gke.metrics_api import query_custom_metrics

        result = query_custom_metrics(gke_config=_make_gke_config())
        assert result.error is False
        assert "requests_per_second" in result.output
        assert "queue_depth" in result.output
        assert "2 custom metric(s)" in result.output

    @patch("vaig.tools.gke.metrics_api._clients")
    def test_list_mode_no_metrics(self, mock_clients: MagicMock) -> None:
        mock_clients._K8S_AVAILABLE = True
        custom_api = MagicMock()
        mock_clients._create_k8s_clients.return_value = (
            MagicMock(), MagicMock(), custom_api, MagicMock(),
        )
        custom_api.get_api_resources.return_value = {"resources": []}

        from vaig.tools.gke.metrics_api import query_custom_metrics

        result = query_custom_metrics(gke_config=_make_gke_config())
        assert result.error is False
        assert "No custom metrics" in result.output

    @patch("vaig.tools.gke.metrics_api._clients")
    def test_query_specific_metric_with_namespace(self, mock_clients: MagicMock) -> None:
        mock_clients._K8S_AVAILABLE = True
        custom_api = MagicMock()
        mock_clients._create_k8s_clients.return_value = (
            MagicMock(), MagicMock(), custom_api, MagicMock(),
        )
        custom_api.list_namespaced_custom_object.return_value = {
            "items": [
                {
                    "describedObject": {
                        "kind": "Pod",
                        "name": "web-abc",
                        "namespace": "production",
                    },
                    "value": "42",
                    "timestamp": "2026-01-01T00:00:00Z",
                },
            ],
        }
        from vaig.tools.gke.metrics_api import query_custom_metrics

        result = query_custom_metrics(
            metric_name="requests_per_second",
            gke_config=_make_gke_config(),
            namespace="production",
        )
        assert result.error is False
        assert "requests_per_second" in result.output
        assert "42" in result.output
        assert "web-abc" in result.output

    @patch("vaig.tools.gke.metrics_api._clients")
    def test_query_metric_cluster_wide(self, mock_clients: MagicMock) -> None:
        mock_clients._K8S_AVAILABLE = True
        custom_api = MagicMock()
        mock_clients._create_k8s_clients.return_value = (
            MagicMock(), MagicMock(), custom_api, MagicMock(),
        )
        custom_api.list_cluster_custom_object.return_value = {"items": []}

        from vaig.tools.gke.metrics_api import query_custom_metrics

        result = query_custom_metrics(
            metric_name="requests_per_second",
            gke_config=_make_gke_config(),
        )
        assert result.error is False
        assert "cluster-wide" in result.output
        assert "No data points" in result.output

    @patch("vaig.tools.gke.metrics_api._clients")
    def test_query_metric_404(self, mock_clients: MagicMock) -> None:
        mock_clients._K8S_AVAILABLE = True
        custom_api = MagicMock()
        mock_clients._create_k8s_clients.return_value = (
            MagicMock(), MagicMock(), custom_api, MagicMock(),
        )
        from kubernetes.client import exceptions as k8s_exc

        custom_api.list_namespaced_custom_object.side_effect = k8s_exc.ApiException(
            status=404, reason="Not Found",
        )
        from vaig.tools.gke.metrics_api import query_custom_metrics

        result = query_custom_metrics(
            metric_name="nonexistent",
            gke_config=_make_gke_config(),
            namespace="default",
        )
        assert result.error is False  # 404 is informational, not an error
        assert "not found" in result.output.lower()

    @patch("vaig.tools.gke.metrics_api._clients")
    def test_list_mode_api_not_registered(self, mock_clients: MagicMock) -> None:
        mock_clients._K8S_AVAILABLE = True
        custom_api = MagicMock()
        mock_clients._create_k8s_clients.return_value = (
            MagicMock(), MagicMock(), custom_api, MagicMock(),
        )
        from kubernetes.client import exceptions as k8s_exc

        custom_api.get_api_resources.side_effect = k8s_exc.ApiException(
            status=404, reason="Not Found",
        )
        from vaig.tools.gke.metrics_api import query_custom_metrics

        result = query_custom_metrics(gke_config=_make_gke_config())
        assert result.error is False
        assert "not registered" in result.output.lower()


# ── query_external_metrics tests ──────────────────────────────


class TestQueryExternalMetrics:
    """Tests for query_external_metrics()."""

    @patch("vaig.tools.gke.metrics_api._clients")
    def test_k8s_unavailable(self, mock_clients: MagicMock) -> None:
        mock_clients._K8S_AVAILABLE = False
        mock_clients._k8s_unavailable.return_value = ToolResult(
            output="kubernetes SDK not available", error=True,
        )
        from vaig.tools.gke.metrics_api import query_external_metrics

        result = query_external_metrics(
            metric_name="some_metric", gke_config=_make_gke_config(),
        )
        assert result.error is True

    @patch("vaig.tools.gke.metrics_api._clients")
    def test_missing_metric_name(self, mock_clients: MagicMock) -> None:
        mock_clients._K8S_AVAILABLE = True
        from vaig.tools.gke.metrics_api import query_external_metrics

        result = query_external_metrics(
            metric_name="", gke_config=_make_gke_config(),
        )
        assert result.error is True
        assert "required" in result.output.lower()

    @patch("vaig.tools.gke.metrics_api._clients")
    def test_successful_query(self, mock_clients: MagicMock) -> None:
        mock_clients._K8S_AVAILABLE = True
        custom_api = MagicMock()
        mock_clients._create_k8s_clients.return_value = (
            MagicMock(), MagicMock(), custom_api, MagicMock(),
        )
        custom_api.list_namespaced_custom_object.return_value = {
            "items": [
                {
                    "metricName": "pubsub.googleapis.com|subscription|num_undelivered_messages",
                    "value": "1500",
                    "timestamp": "2026-01-01T00:00:00Z",
                },
            ],
        }
        from vaig.tools.gke.metrics_api import query_external_metrics

        result = query_external_metrics(
            metric_name="pubsub.googleapis.com|subscription|num_undelivered_messages",
            gke_config=_make_gke_config(),
            namespace="production",
        )
        assert result.error is False
        assert "1500" in result.output
        assert "pubsub" in result.output

    @patch("vaig.tools.gke.metrics_api._clients")
    def test_query_404(self, mock_clients: MagicMock) -> None:
        mock_clients._K8S_AVAILABLE = True
        custom_api = MagicMock()
        mock_clients._create_k8s_clients.return_value = (
            MagicMock(), MagicMock(), custom_api, MagicMock(),
        )
        from kubernetes.client import exceptions as k8s_exc

        custom_api.list_namespaced_custom_object.side_effect = k8s_exc.ApiException(
            status=404, reason="Not Found",
        )
        from vaig.tools.gke.metrics_api import query_external_metrics

        result = query_external_metrics(
            metric_name="nonexistent",
            gke_config=_make_gke_config(),
            namespace="default",
        )
        assert result.error is False
        assert "not found" in result.output.lower()

    @patch("vaig.tools.gke.metrics_api._clients")
    def test_query_no_data_points(self, mock_clients: MagicMock) -> None:
        mock_clients._K8S_AVAILABLE = True
        custom_api = MagicMock()
        mock_clients._create_k8s_clients.return_value = (
            MagicMock(), MagicMock(), custom_api, MagicMock(),
        )
        custom_api.list_namespaced_custom_object.return_value = {"items": []}

        from vaig.tools.gke.metrics_api import query_external_metrics

        result = query_external_metrics(
            metric_name="some_metric",
            gke_config=_make_gke_config(),
            namespace="default",
        )
        assert result.error is False
        assert "No data points" in result.output

    @patch("vaig.tools.gke.metrics_api._clients")
    def test_uses_default_namespace(self, mock_clients: MagicMock) -> None:
        mock_clients._K8S_AVAILABLE = True
        custom_api = MagicMock()
        mock_clients._create_k8s_clients.return_value = (
            MagicMock(), MagicMock(), custom_api, MagicMock(),
        )
        custom_api.list_namespaced_custom_object.return_value = {"items": []}

        from vaig.tools.gke.metrics_api import query_external_metrics

        result = query_external_metrics(
            metric_name="some_metric",
            gke_config=_make_gke_config(),
        )
        assert result.error is False
        # Verify it used default namespace
        custom_api.list_namespaced_custom_object.assert_called_once_with(
            group="external.metrics.k8s.io",
            version="v1beta1",
            namespace="default",
            plural="some_metric",
        )

    @patch("vaig.tools.gke.metrics_api._clients")
    def test_auth_error(self, mock_clients: MagicMock) -> None:
        mock_clients._K8S_AVAILABLE = True
        custom_api = MagicMock()
        mock_clients._create_k8s_clients.return_value = (
            MagicMock(), MagicMock(), custom_api, MagicMock(),
        )
        from kubernetes.client import exceptions as k8s_exc

        custom_api.list_namespaced_custom_object.side_effect = k8s_exc.ApiException(
            status=403, reason="Forbidden",
        )
        from vaig.tools.gke.metrics_api import query_external_metrics

        result = query_external_metrics(
            metric_name="some_metric",
            gke_config=_make_gke_config(),
        )
        assert result.error is True
        assert "authorization" in result.output.lower() or "Forbidden" in result.output

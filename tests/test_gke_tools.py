"""Tests for GKE tools — kubectl_get, kubectl_describe, kubectl_logs, kubectl_top,
kubectl_scale, kubectl_restart, kubectl_label, kubectl_annotate, create_gke_tools."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from vaig.core.config import GKEConfig
from vaig.tools.base import ToolDef, ToolResult


@pytest.fixture(autouse=True)
def _clear_k8s_cache() -> None:
    """Clear the K8s client cache and Autopilot cache before each test."""
    from vaig.tools.gke_tools import clear_autopilot_cache, clear_discovery_cache, clear_k8s_client_cache
    clear_k8s_client_cache()
    clear_autopilot_cache()
    clear_discovery_cache()


# ── Helpers ──────────────────────────────────────────────────

def _make_gke_config(**kwargs) -> GKEConfig:
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


def _mock_pod(
    name: str = "my-pod",
    namespace: str = "default",
    phase: str = "Running",
    ready: bool = True,
    restarts: int = 0,
    node_name: str = "node-1",
    ip: str = "10.0.0.1",
) -> MagicMock:
    """Create a realistic mock pod object mirroring kubernetes client V1Pod."""
    pod = MagicMock()
    pod.metadata.name = name
    pod.metadata.namespace = namespace
    pod.metadata.creation_timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
    pod.metadata.deletion_timestamp = None
    pod.metadata.labels = {"app": "test"}
    pod.metadata.annotations = {}
    pod.spec.node_name = node_name
    pod.spec.containers = [MagicMock(name="main")]
    pod.status.phase = phase
    pod.status.pod_ip = ip

    cs = MagicMock()
    cs.ready = ready
    cs.restart_count = restarts
    cs.state.waiting = None
    cs.state.terminated = None
    cs.state.running.started_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    cs.name = "main"
    pod.status.container_statuses = [cs]
    return pod


def _mock_deployment(
    name: str = "my-deploy",
    namespace: str = "default",
    replicas: int = 3,
    ready: int = 3,
) -> MagicMock:
    dep = MagicMock()
    dep.metadata.name = name
    dep.metadata.namespace = namespace
    dep.metadata.creation_timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
    dep.spec.replicas = replicas
    dep.spec.template.spec.containers = []
    dep.status.ready_replicas = ready
    dep.status.updated_replicas = ready
    dep.status.available_replicas = ready
    return dep


# ── _normalise_resource / _parse_since helpers ───────────────


class TestNormaliseResource:
    def test_canonical_names_unchanged(self) -> None:
        from vaig.tools.gke_tools import _normalise_resource

        assert _normalise_resource("pods") == "pods"
        assert _normalise_resource("deployments") == "deployments"
        assert _normalise_resource("services") == "services"

    def test_aliases_resolved(self) -> None:
        from vaig.tools.gke_tools import _normalise_resource

        assert _normalise_resource("po") == "pods"
        assert _normalise_resource("pod") == "pods"
        assert _normalise_resource("svc") == "services"
        assert _normalise_resource("deploy") == "deployments"
        assert _normalise_resource("sts") == "statefulsets"
        assert _normalise_resource("ds") == "daemonsets"
        assert _normalise_resource("cm") == "configmaps"

    def test_case_insensitive(self) -> None:
        from vaig.tools.gke_tools import _normalise_resource

        assert _normalise_resource("Pods") == "pods"
        assert _normalise_resource("DEPLOY") == "deployments"

    def test_unknown_passes_through(self) -> None:
        from vaig.tools.gke_tools import _normalise_resource

        assert _normalise_resource("customresource") == "customresource"


class TestParseSince:
    def test_hours(self) -> None:
        from vaig.tools.gke_tools import _parse_since

        assert _parse_since("1h") == 3600
        assert _parse_since("2h") == 7200

    def test_minutes(self) -> None:
        from vaig.tools.gke_tools import _parse_since

        assert _parse_since("30m") == 1800

    def test_seconds(self) -> None:
        from vaig.tools.gke_tools import _parse_since

        assert _parse_since("90s") == 90

    def test_combined(self) -> None:
        from vaig.tools.gke_tools import _parse_since

        assert _parse_since("1h30m") == 5400
        assert _parse_since("2h15m30s") == 8130

    def test_invalid_returns_none(self) -> None:
        from vaig.tools.gke_tools import _parse_since

        assert _parse_since("invalid") is None
        assert _parse_since("abc123") is None

    def test_empty_groups_returns_none(self) -> None:
        from vaig.tools.gke_tools import _parse_since

        assert _parse_since("") is None


# ── kubectl_get ──────────────────────────────────────────────


class TestKubectlGet:
    """Tests for kubectl_get function."""

    def test_k8s_unavailable(self) -> None:
        from vaig.tools.gke_tools import kubectl_get

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = kubectl_get("pods", gke_config=cfg)
            assert result.error is True
            assert "kubernetes" in result.output.lower()

    def test_unsupported_resource_type(self) -> None:
        from vaig.tools.gke_tools import kubectl_get

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = kubectl_get("foobar", gke_config=cfg)
            assert result.error is True
            assert "Unsupported resource type" in result.output

    def test_invalid_output_format(self) -> None:
        from vaig.tools.gke_tools import kubectl_get

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = kubectl_get("pods", gke_config=cfg, output_format="xml")
            assert result.error is True
            assert "Invalid output_format" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_successful_pod_list(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_get

        cfg = _make_gke_config()
        # Mock the K8s clients
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        mock_clients.return_value = (core_v1, apps_v1, custom_api, MagicMock())

        # Mock the list response
        pod = _mock_pod()
        api_response = MagicMock()
        api_response.items = [pod]
        core_v1.list_namespaced_pod.return_value = api_response

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = kubectl_get("pods", gke_config=cfg)

        assert result.error is False
        assert "my-pod" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_name_filter_not_found(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_get

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        mock_clients.return_value = (core_v1, apps_v1, custom_api, MagicMock())

        # Return empty list
        api_response = MagicMock()
        api_response.items = [_mock_pod(name="other-pod")]
        core_v1.list_namespaced_pod.return_value = api_response

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = kubectl_get("pods", gke_config=cfg, name="nonexistent")

        assert result.error is True
        assert "not found" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_client_creation_error(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_get

        cfg = _make_gke_config()
        mock_clients.return_value = ToolResult(output="Failed to configure", error=True)

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = kubectl_get("pods", gke_config=cfg)

        assert result.error is True
        assert "Failed to configure" in result.output

    def test_resource_alias_in_get(self) -> None:
        """Using an alias like 'deploy' should normalise to 'deployments'."""
        from vaig.tools.gke_tools import kubectl_get

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._create_k8s_clients") as mock_clients:
            apps_v1 = MagicMock()
            api_response = MagicMock()
            api_response.items = [_mock_deployment()]
            apps_v1.list_namespaced_deployment.return_value = api_response
            mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

            result = kubectl_get("deploy", gke_config=cfg)

        assert result.error is False
        assert "my-deploy" in result.output


# ── kubectl_describe ─────────────────────────────────────────


class TestKubectlDescribe:
    """Tests for kubectl_describe function."""

    def test_k8s_unavailable(self) -> None:
        from vaig.tools.gke_tools import kubectl_describe

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = kubectl_describe("pods", "my-pod", gke_config=cfg)
            assert result.error is True

    def test_unsupported_resource_type(self) -> None:
        from vaig.tools.gke_tools import kubectl_describe

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = kubectl_describe("foobar", "name", gke_config=cfg)
            assert result.error is True
            assert "Unsupported resource type" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_successful_describe_pod(self, mock_clients: MagicMock) -> None:
        import vaig.tools.gke_tools as _mod
        from vaig.tools.gke_tools import kubectl_describe

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        mock_clients.return_value = (core_v1, apps_v1, MagicMock(), MagicMock())

        pod = _mock_pod(name="web-server")
        core_v1.read_namespaced_pod.return_value = pod

        # Mock k8s_client module-level name (may not exist when kubernetes is not installed)
        fake_k8s_client = MagicMock()
        events_v1 = MagicMock()
        events_v1.list_namespaced_event.return_value = MagicMock(items=[])
        fake_k8s_client.CoreV1Api.return_value = events_v1

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch.object(_mod, "k8s_client", fake_k8s_client, create=True):
            result = kubectl_describe("pods", "web-server", gke_config=cfg)

        assert result.error is False
        assert "web-server" in result.output
        assert "Name:" in result.output


# ── kubectl_logs ─────────────────────────────────────────────


class TestKubectlLogs:
    """Tests for kubectl_logs function."""

    def test_k8s_unavailable(self) -> None:
        from vaig.tools.gke_tools import kubectl_logs

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = kubectl_logs("my-pod", gke_config=cfg)
            assert result.error is True

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_successful_logs(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_logs

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        core_v1.read_namespaced_pod_log.return_value = "2025-01-01 INFO Starting server\n2025-01-01 INFO Ready"

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = kubectl_logs("my-pod", gke_config=cfg)

        assert result.error is False
        assert "Starting server" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_empty_logs(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_logs

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        core_v1.read_namespaced_pod_log.return_value = ""

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = kubectl_logs("my-pod", gke_config=cfg)

        assert result.error is False
        assert "no logs available" in result.output.lower()

    def test_invalid_since_format(self) -> None:
        from vaig.tools.gke_tools import kubectl_logs

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._create_k8s_clients") as mock_clients:
            mock_clients.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())

            result = kubectl_logs("my-pod", gke_config=cfg, since="invalid")

        assert result.error is True
        assert "Invalid 'since' format" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_tail_lines_capped_by_config(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_logs

        cfg = _make_gke_config(log_limit=50)
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())
        core_v1.read_namespaced_pod_log.return_value = "some logs"

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            kubectl_logs("my-pod", gke_config=cfg, tail_lines=200)

        # tail_lines should be min(200, 50) = 50
        call_kwargs = core_v1.read_namespaced_pod_log.call_args
        assert call_kwargs[1]["tail_lines"] == 50 or call_kwargs.kwargs.get("tail_lines") == 50


# ── kubectl_top ──────────────────────────────────────────────


class TestKubectlTop:
    """Tests for kubectl_top function."""

    def test_k8s_unavailable(self) -> None:
        from vaig.tools.gke_tools import kubectl_top

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = kubectl_top(gke_config=cfg)
            assert result.error is True

    def test_invalid_resource_type(self) -> None:
        from vaig.tools.gke_tools import kubectl_top

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = kubectl_top("services", gke_config=cfg)
            assert result.error is True
            assert "Invalid resource_type" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_pod_metrics(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_top

        cfg = _make_gke_config()
        custom_api = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), custom_api, MagicMock())

        custom_api.list_namespaced_custom_object.return_value = {
            "items": [
                {
                    "metadata": {"name": "my-pod", "namespace": "default"},
                    "containers": [
                        {"name": "main", "usage": {"cpu": "100m", "memory": "128Mi"}}
                    ],
                }
            ]
        }

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = kubectl_top("pods", gke_config=cfg)

        assert result.error is False
        assert "my-pod" in result.output
        assert "100m" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_node_metrics(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_top

        cfg = _make_gke_config()
        custom_api = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), custom_api, MagicMock())

        custom_api.list_cluster_custom_object.return_value = {
            "items": [
                {
                    "metadata": {"name": "node-1"},
                    "usage": {"cpu": "500m", "memory": "2Gi"},
                }
            ]
        }

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = kubectl_top("nodes", gke_config=cfg)

        assert result.error is False
        assert "node-1" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_no_metrics_available(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_top

        cfg = _make_gke_config()
        custom_api = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), custom_api, MagicMock())

        custom_api.list_namespaced_custom_object.return_value = {"items": []}

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = kubectl_top("pods", gke_config=cfg)

        assert "No metrics data" in result.output or "metrics" in result.output.lower()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_name_filter_no_match(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_top

        cfg = _make_gke_config()
        custom_api = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), custom_api, MagicMock())

        custom_api.list_namespaced_custom_object.return_value = {
            "items": [
                {
                    "metadata": {"name": "other-pod"},
                    "containers": [{"name": "main", "usage": {"cpu": "10m", "memory": "64Mi"}}],
                }
            ]
        }

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = kubectl_top("pods", gke_config=cfg, name="nonexistent")

        assert result.error is True
        assert "No metrics found" in result.output


# ── create_gke_tools factory ─────────────────────────────────


class TestCreateGkeTools:
    """Tests for create_gke_tools factory function."""

    def test_returns_eight_tool_defs(self) -> None:
        from vaig.tools.gke_tools import create_gke_tools

        cfg = _make_gke_config()
        tools = create_gke_tools(cfg)

        assert len(tools) == 22
        assert all(isinstance(t, ToolDef) for t in tools)

    def test_tool_names(self) -> None:
        from vaig.tools.gke_tools import create_gke_tools

        cfg = _make_gke_config()
        tools = create_gke_tools(cfg)
        names = {t.name for t in tools}

        assert names == {
            "kubectl_get", "kubectl_describe", "kubectl_logs", "kubectl_top",
            "kubectl_scale", "kubectl_restart", "kubectl_label", "kubectl_annotate",
            "get_events", "get_rollout_status", "get_node_conditions", "get_container_status",
            "exec_command", "check_rbac", "get_rollout_history", "discover_workloads",
            "discover_service_mesh", "discover_network_topology",
            "get_mesh_overview", "get_mesh_config", "get_mesh_security", "get_sidecar_status",
        }

    def test_all_have_descriptions(self) -> None:
        from vaig.tools.gke_tools import create_gke_tools

        cfg = _make_gke_config()
        tools = create_gke_tools(cfg)

        for t in tools:
            assert t.description, f"Tool {t.name} has no description"
            assert len(t.description) > 20

    def test_all_have_parameters(self) -> None:
        from vaig.tools.gke_tools import create_gke_tools

        cfg = _make_gke_config()
        tools = create_gke_tools(cfg)

        for t in tools:
            assert len(t.parameters) >= 1, f"Tool {t.name} has no parameters"

    def test_kubectl_get_has_required_resource_param(self) -> None:
        from vaig.tools.gke_tools import create_gke_tools

        cfg = _make_gke_config()
        tools = create_gke_tools(cfg)
        get_tool = next(t for t in tools if t.name == "kubectl_get")
        resource_param = next(p for p in get_tool.parameters if p.name == "resource")

        assert resource_param.required is True
        assert resource_param.type == "string"


# ── Formatting helpers ───────────────────────────────────────


class TestFormattingHelpers:
    """Tests for internal formatting functions."""

    def test_age_recent(self) -> None:
        from vaig.tools.gke_tools import _age

        now = datetime.now(timezone.utc)
        assert _age(now) == "0s"

    def test_age_none(self) -> None:
        from vaig.tools.gke_tools import _age

        assert _age(None) == "<unknown>"

    def test_pod_status_running(self) -> None:
        from vaig.tools.gke_tools import _pod_status

        pod = _mock_pod(phase="Running")
        assert _pod_status(pod) == "Running"

    def test_pod_status_terminating(self) -> None:
        from vaig.tools.gke_tools import _pod_status

        pod = _mock_pod()
        pod.metadata.deletion_timestamp = datetime.now(timezone.utc)
        assert _pod_status(pod) == "Terminating"

    def test_pod_status_succeeded(self) -> None:
        from vaig.tools.gke_tools import _pod_status

        pod = _mock_pod(phase="Succeeded")
        pod.metadata.deletion_timestamp = None
        assert _pod_status(pod) == "Succeeded"

    def test_pod_restarts(self) -> None:
        from vaig.tools.gke_tools import _pod_restarts

        pod = _mock_pod(restarts=5)
        assert _pod_restarts(pod) == 5

    def test_pod_ready_count(self) -> None:
        from vaig.tools.gke_tools import _pod_ready_count

        pod = _mock_pod(ready=True)
        assert _pod_ready_count(pod) == "1/1"

    def test_format_pods_table_empty(self) -> None:
        from vaig.tools.gke_tools import _format_pods_table

        assert _format_pods_table([]) == "No resources found."

    def test_format_deployments_table_empty(self) -> None:
        from vaig.tools.gke_tools import _format_deployments_table

        assert _format_deployments_table([]) == "No resources found."

    def test_format_services_table_empty(self) -> None:
        from vaig.tools.gke_tools import _format_services_table

        assert _format_services_table([]) == "No resources found."

    def test_format_nodes_table_empty(self) -> None:
        from vaig.tools.gke_tools import _format_nodes_table

        assert _format_nodes_table([]) == "No resources found."

    def test_format_generic_table_empty(self) -> None:
        from vaig.tools.gke_tools import _format_generic_table

        assert _format_generic_table([]) == "No resources found."


# ── _extract_proxy_url_from_kubeconfig ──────────────────────


class TestExtractProxyUrlFromKubeconfig:
    """Tests for _extract_proxy_url_from_kubeconfig helper."""

    _KUBECONFIG_WITH_PROXY = {
        "current-context": "gke-ctx",
        "contexts": [
            {
                "name": "gke-ctx",
                "context": {"cluster": "gke-cluster", "user": "gke-user"},
            },
        ],
        "clusters": [
            {
                "name": "gke-cluster",
                "cluster": {
                    "server": "https://10.168.57.2",
                    "proxy-url": "https://proxy.example.com:8443",
                },
            },
        ],
    }

    _KUBECONFIG_NO_PROXY = {
        "current-context": "gke-ctx",
        "contexts": [
            {
                "name": "gke-ctx",
                "context": {"cluster": "gke-cluster", "user": "gke-user"},
            },
        ],
        "clusters": [
            {
                "name": "gke-cluster",
                "cluster": {"server": "https://34.1.2.3"},
            },
        ],
    }

    def test_extracts_proxy_url(self, tmp_path) -> None:
        import yaml as _yaml

        from vaig.tools.gke_tools import _extract_proxy_url_from_kubeconfig

        kc = tmp_path / "config"
        kc.write_text(_yaml.dump(self._KUBECONFIG_WITH_PROXY))

        result = _extract_proxy_url_from_kubeconfig(str(kc))
        assert result == "https://proxy.example.com:8443"

    def test_returns_none_when_no_proxy(self, tmp_path) -> None:
        import yaml as _yaml

        from vaig.tools.gke_tools import _extract_proxy_url_from_kubeconfig

        kc = tmp_path / "config"
        kc.write_text(_yaml.dump(self._KUBECONFIG_NO_PROXY))

        result = _extract_proxy_url_from_kubeconfig(str(kc))
        assert result is None

    def test_returns_none_for_missing_file(self) -> None:
        from vaig.tools.gke_tools import _extract_proxy_url_from_kubeconfig

        result = _extract_proxy_url_from_kubeconfig("/nonexistent/path/config")
        assert result is None

    def test_returns_none_for_wrong_context(self, tmp_path) -> None:
        import yaml as _yaml

        from vaig.tools.gke_tools import _extract_proxy_url_from_kubeconfig

        kc = tmp_path / "config"
        kc.write_text(_yaml.dump(self._KUBECONFIG_WITH_PROXY))

        result = _extract_proxy_url_from_kubeconfig(str(kc), context="nonexistent-ctx")
        assert result is None

    def test_explicit_context_overrides_current(self, tmp_path) -> None:
        import yaml as _yaml

        from vaig.tools.gke_tools import _extract_proxy_url_from_kubeconfig

        data = {
            "current-context": "other-ctx",
            "contexts": [
                {
                    "name": "other-ctx",
                    "context": {"cluster": "other-cluster", "user": "u"},
                },
                {
                    "name": "proxy-ctx",
                    "context": {"cluster": "proxy-cluster", "user": "u"},
                },
            ],
            "clusters": [
                {"name": "other-cluster", "cluster": {"server": "https://1.2.3.4"}},
                {
                    "name": "proxy-cluster",
                    "cluster": {
                        "server": "https://10.0.0.1",
                        "proxy-url": "https://my-proxy:443",
                    },
                },
            ],
        }
        kc = tmp_path / "config"
        kc.write_text(_yaml.dump(data))

        result = _extract_proxy_url_from_kubeconfig(str(kc), context="proxy-ctx")
        assert result == "https://my-proxy:443"

    def test_returns_none_for_invalid_yaml(self, tmp_path) -> None:
        from vaig.tools.gke_tools import _extract_proxy_url_from_kubeconfig

        kc = tmp_path / "config"
        kc.write_text("::invalid::yaml::[")

        result = _extract_proxy_url_from_kubeconfig(str(kc))
        assert result is None

    def test_returns_none_when_no_current_context(self, tmp_path) -> None:
        import yaml as _yaml

        from vaig.tools.gke_tools import _extract_proxy_url_from_kubeconfig

        kc = tmp_path / "config"
        kc.write_text(_yaml.dump({"contexts": [], "clusters": []}))

        result = _extract_proxy_url_from_kubeconfig(str(kc))
        assert result is None


# ── _create_k8s_clients with proxy ──────────────────────────


class TestCreateK8sClientsProxy:
    """Tests for proxy-url handling in _create_k8s_clients."""

    @patch("vaig.tools.gke._clients.k8s_config")
    @patch("vaig.tools.gke._clients.k8s_client")
    @patch("vaig.tools.gke._clients._extract_proxy_url_from_kubeconfig")
    def test_proxy_from_kubeconfig_is_set(
        self,
        mock_extract: MagicMock,
        mock_k8s_client: MagicMock,
        mock_k8s_config: MagicMock,
    ) -> None:
        """When kubeconfig has proxy-url, Configuration.proxy must be set."""
        mock_extract.return_value = "https://proxy.example.com:8443"

        # Stub Configuration so we can inspect it
        config_instance = MagicMock()
        mock_k8s_client.Configuration.return_value = config_instance

        cfg = _make_gke_config(kubeconfig_path="/tmp/kube")

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            from vaig.tools.gke_tools import _create_k8s_clients

            result = _create_k8s_clients(cfg)

        # proxy must have been set
        assert config_instance.proxy == "https://proxy.example.com:8443"
        # load_kube_config must receive client_configuration
        mock_k8s_config.load_kube_config.assert_called_once()
        call_kwargs = mock_k8s_config.load_kube_config.call_args
        assert call_kwargs.kwargs.get("client_configuration") is config_instance
        # API clients must be built with the ApiClient wrapping our config
        mock_k8s_client.ApiClient.assert_called_once_with(config_instance)
        assert not isinstance(result, ToolResult)

    @patch("vaig.tools.gke._clients.k8s_config")
    @patch("vaig.tools.gke._clients.k8s_client")
    @patch("vaig.tools.gke._clients._extract_proxy_url_from_kubeconfig")
    def test_gkeconfig_proxy_overrides_kubeconfig(
        self,
        mock_extract: MagicMock,
        mock_k8s_client: MagicMock,
        mock_k8s_config: MagicMock,
    ) -> None:
        """GKEConfig.proxy_url should override kubeconfig proxy-url."""
        mock_extract.return_value = "https://proxy-from-kubeconfig:8443"

        config_instance = MagicMock()
        mock_k8s_client.Configuration.return_value = config_instance

        cfg = _make_gke_config(proxy_url="https://override-proxy:9090")

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            from vaig.tools.gke_tools import _create_k8s_clients

            _create_k8s_clients(cfg)

        assert config_instance.proxy == "https://override-proxy:9090"

    @patch("vaig.tools.gke._clients.k8s_config")
    @patch("vaig.tools.gke._clients.k8s_client")
    @patch("vaig.tools.gke._clients._extract_proxy_url_from_kubeconfig")
    def test_no_proxy_works_normally(
        self,
        mock_extract: MagicMock,
        mock_k8s_client: MagicMock,
        mock_k8s_config: MagicMock,
    ) -> None:
        """When no proxy-url exists, clients are still created without error."""
        mock_extract.return_value = None

        config_instance = MagicMock()
        mock_k8s_client.Configuration.return_value = config_instance

        cfg = _make_gke_config()

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            from vaig.tools.gke_tools import _create_k8s_clients

            result = _create_k8s_clients(cfg)

        # proxy should NOT have been set (MagicMock default attr, not explicitly set)
        # Verify load_kube_config was still called with client_configuration
        mock_k8s_config.load_kube_config.assert_called_once()
        assert not isinstance(result, ToolResult)


# ── kubectl_scale ────────────────────────────────────────────


class TestKubectlScale:
    """Tests for kubectl_scale write operation."""

    def test_k8s_unavailable(self) -> None:
        from vaig.tools.gke_tools import kubectl_scale

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = kubectl_scale("deployments", "nginx", 3, gke_config=cfg)
            assert result.error is True
            assert "kubernetes" in result.output.lower()

    def test_unsupported_resource_type(self) -> None:
        from vaig.tools.gke_tools import kubectl_scale

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_scale("pods", "my-pod", 2, gke_config=cfg)
            assert result.error is True
            assert "Cannot scale" in result.output
            assert "deployments" in result.output

    def test_replicas_too_high(self) -> None:
        from vaig.tools.gke_tools import kubectl_scale

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_scale("deployments", "nginx", 100, gke_config=cfg)
            assert result.error is True
            assert "50" in result.output

    def test_replicas_negative(self) -> None:
        from vaig.tools.gke_tools import kubectl_scale

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_scale("deployments", "nginx", -1, gke_config=cfg)
            assert result.error is True

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_scale_deployment_success(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_scale

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        deploy_obj = MagicMock()
        deploy_obj.spec.replicas = 2
        apps_v1.read_namespaced_deployment.return_value = deploy_obj

        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_scale("deploy", "nginx", 5, gke_config=cfg)
            assert result.error is not True
            assert "2 -> 5" in result.output
            assert "nginx" in result.output
            apps_v1.patch_namespaced_deployment_scale.assert_called_once_with(
                "nginx", "default", {"spec": {"replicas": 5}},
            )

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_scale_statefulset_success(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_scale

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        sts_obj = MagicMock()
        sts_obj.spec.replicas = 3
        apps_v1.read_namespaced_stateful_set.return_value = sts_obj

        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_scale("sts", "redis", 1, gke_config=cfg, namespace="cache")
            assert result.error is not True
            assert "3 -> 1" in result.output
            apps_v1.patch_namespaced_stateful_set_scale.assert_called_once()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_scale_replicaset_success(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_scale

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        rs_obj = MagicMock()
        rs_obj.spec.replicas = 1
        apps_v1.read_namespaced_replica_set.return_value = rs_obj

        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_scale("replicasets", "my-rs", 0, gke_config=cfg)
            assert result.error is not True
            assert "1 -> 0" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_scale_404(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_scale

        cfg = _make_gke_config()
        apps_v1 = MagicMock()

        from unittest.mock import PropertyMock
        exc = MagicMock()
        type(exc).status = PropertyMock(return_value=404)
        exc_class = type("ApiException", (Exception,), {"status": 404, "reason": "Not Found"})
        real_exc = exc_class()

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.mutations.k8s_exceptions") as mock_k8s_exc:
            mock_k8s_exc.ApiException = exc_class
            apps_v1.read_namespaced_deployment.side_effect = real_exc
            mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())
            result = kubectl_scale("deployments", "ghost", 3, gke_config=cfg)
            assert result.error is True
            assert "not found" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_scale_403(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_scale

        cfg = _make_gke_config()
        apps_v1 = MagicMock()

        exc_class = type("ApiException", (Exception,), {"status": 403, "reason": "Forbidden"})
        real_exc = exc_class()

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.mutations.k8s_exceptions") as mock_k8s_exc:
            mock_k8s_exc.ApiException = exc_class
            apps_v1.read_namespaced_deployment.side_effect = real_exc
            mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())
            result = kubectl_scale("deployments", "nginx", 3, gke_config=cfg)
            assert result.error is True
            assert "Access denied" in result.output

    def test_scale_alias_normalisation(self) -> None:
        """Aliases like 'deploy' resolve before validation."""
        from vaig.tools.gke_tools import kubectl_scale

        cfg = _make_gke_config()
        # 'services' is not scalable — just verify it rejects properly
        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_scale("svc", "my-svc", 2, gke_config=cfg)
            assert result.error is True
            assert "Cannot scale" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_scale_to_zero(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_scale

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        deploy_obj = MagicMock()
        deploy_obj.spec.replicas = 3
        apps_v1.read_namespaced_deployment.return_value = deploy_obj

        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_scale("deployments", "nginx", 0, gke_config=cfg)
            assert result.error is not True
            assert "3 -> 0" in result.output


# ── kubectl_restart ──────────────────────────────────────────


class TestKubectlRestart:
    """Tests for kubectl_restart write operation."""

    def test_k8s_unavailable(self) -> None:
        from vaig.tools.gke_tools import kubectl_restart

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = kubectl_restart("deployments", "nginx", gke_config=cfg)
            assert result.error is True

    def test_unsupported_resource_type(self) -> None:
        from vaig.tools.gke_tools import kubectl_restart

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_restart("pods", "my-pod", gke_config=cfg)
            assert result.error is True
            assert "Cannot restart" in result.output
            assert "daemonsets" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_restart_deployment_success(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_restart

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_restart("deploy", "nginx", gke_config=cfg)
            assert result.error is not True
            assert "Rolling restart triggered" in result.output
            assert "nginx" in result.output
            apps_v1.patch_namespaced_deployment.assert_called_once()
            # Verify the annotation was set
            call_args = apps_v1.patch_namespaced_deployment.call_args
            patch_body = call_args[0][2]
            assert "kubectl.kubernetes.io/restartedAt" in \
                patch_body["spec"]["template"]["metadata"]["annotations"]

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_restart_statefulset_success(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_restart

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_restart("sts", "redis", gke_config=cfg, namespace="cache")
            assert result.error is not True
            apps_v1.patch_namespaced_stateful_set.assert_called_once()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_restart_daemonset_success(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_restart

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_restart("ds", "fluentd", gke_config=cfg)
            assert result.error is not True
            assert "fluentd" in result.output
            apps_v1.patch_namespaced_daemon_set.assert_called_once()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_restart_404(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_restart

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        exc_class = type("ApiException", (Exception,), {"status": 404, "reason": "Not Found"})

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.mutations.k8s_exceptions") as mock_k8s_exc:
            mock_k8s_exc.ApiException = exc_class
            apps_v1.patch_namespaced_deployment.side_effect = exc_class()
            mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())
            result = kubectl_restart("deployments", "ghost", gke_config=cfg)
            assert result.error is True
            assert "not found" in result.output

    def test_restart_replicasets_not_allowed(self) -> None:
        """replicasets are scalable but NOT restartable."""
        from vaig.tools.gke_tools import kubectl_restart

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_restart("replicasets", "my-rs", gke_config=cfg)
            assert result.error is True
            assert "Cannot restart" in result.output


# ── kubectl_label ────────────────────────────────────────────


class TestKubectlLabel:
    """Tests for kubectl_label write operation."""

    def test_k8s_unavailable(self) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = kubectl_label("pods", "my-pod", "env=prod", gke_config=cfg)
            assert result.error is True

    def test_unsupported_resource_type(self) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_label("cronjobs", "my-cron", "env=prod", gke_config=cfg)
            assert result.error is True
            assert "Cannot label" in result.output

    def test_invalid_label_format(self) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_label("pods", "my-pod", "invalid_no_equals", gke_config=cfg)
            assert result.error is True
            assert "Invalid label format" in result.output

    def test_empty_labels(self) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_label("pods", "my-pod", "", gke_config=cfg)
            assert result.error is True
            assert "No labels" in result.output

    def test_system_label_blocked(self) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_label("pods", "my-pod", "kubernetes.io/arch=amd64", gke_config=cfg)
            assert result.error is True
            assert "system label" in result.output.lower()

    def test_k8s_io_system_label_blocked(self) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_label("pods", "my-pod", "k8s.io/component=api", gke_config=cfg)
            assert result.error is True
            assert "system label" in result.output.lower()

    def test_invalid_label_key_format(self) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_label("pods", "my-pod", "-bad-key=val", gke_config=cfg)
            assert result.error is True
            assert "Invalid label key" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_label_pod_success(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_label("pods", "my-pod", "env=prod,tier=frontend", gke_config=cfg)
            assert result.error is not True
            assert "Labels updated" in result.output
            assert "env=prod" in result.output
            core_v1.patch_namespaced_pod.assert_called_once()
            patch_body = core_v1.patch_namespaced_pod.call_args[0][2]
            assert patch_body["metadata"]["labels"]["env"] == "prod"
            assert patch_body["metadata"]["labels"]["tier"] == "frontend"

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_label_remove(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_label("pods", "my-pod", "obsolete-", gke_config=cfg)
            assert result.error is not True
            assert "obsolete-" in result.output
            patch_body = core_v1.patch_namespaced_pod.call_args[0][2]
            assert patch_body["metadata"]["labels"]["obsolete"] is None

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_label_deployment(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_label("deploy", "nginx", "version=v2", gke_config=cfg)
            assert result.error is not True
            apps_v1.patch_namespaced_deployment.assert_called_once()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_label_service(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_label("svc", "my-svc", "managed-by=helm", gke_config=cfg)
            assert result.error is not True
            core_v1.patch_namespaced_service.assert_called_once()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_label_namespace(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_label("namespaces", "kube-system", "env=prod", gke_config=cfg)
            assert result.error is not True
            core_v1.patch_namespace.assert_called_once()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_label_node(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_label("nodes", "worker-1", "role=compute", gke_config=cfg)
            assert result.error is not True
            core_v1.patch_node.assert_called_once()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_label_configmap(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_label("cm", "my-config", "app=web", gke_config=cfg)
            assert result.error is not True
            core_v1.patch_namespaced_config_map.assert_called_once()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_label_secret(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_label("secrets", "my-secret", "env=staging", gke_config=cfg)
            assert result.error is not True
            core_v1.patch_namespaced_secret.assert_called_once()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_label_404(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        exc_class = type("ApiException", (Exception,), {"status": 404, "reason": "Not Found"})

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.mutations.k8s_exceptions") as mock_k8s_exc:
            mock_k8s_exc.ApiException = exc_class
            core_v1.patch_namespaced_pod.side_effect = exc_class()
            mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())
            result = kubectl_label("pods", "ghost", "env=prod", gke_config=cfg)
            assert result.error is True
            assert "not found" in result.output

    def test_label_with_prefix_valid(self) -> None:
        """Labels with custom prefixes (not system) should be allowed."""
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        # We only validate up to the system prefix check — should pass validation
        # but fail at k8s client call since _K8S_AVAILABLE blocks it
        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._create_k8s_clients") as mock_clients:
            core_v1 = MagicMock()
            mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())
            result = kubectl_label("pods", "my-pod", "mycompany.io/env=prod", gke_config=cfg)
            assert result.error is not True


# ── kubectl_annotate ─────────────────────────────────────────


class TestKubectlAnnotate:
    """Tests for kubectl_annotate write operation."""

    def test_k8s_unavailable(self) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = kubectl_annotate("pods", "my-pod", "desc=test", gke_config=cfg)
            assert result.error is True

    def test_unsupported_resource_type(self) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_annotate("cronjobs", "my-cron", "desc=test", gke_config=cfg)
            assert result.error is True
            assert "Cannot annotate" in result.output

    def test_invalid_annotation_format(self) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_annotate("pods", "my-pod", "no_equals_sign", gke_config=cfg)
            assert result.error is True
            assert "Invalid annotation format" in result.output

    def test_empty_annotations(self) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_annotate("pods", "my-pod", "", gke_config=cfg)
            assert result.error is True
            assert "No annotations" in result.output

    def test_system_annotation_blocked(self) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_annotate(
                "pods", "my-pod",
                "kubernetes.io/change-cause=test",
                gke_config=cfg,
            )
            assert result.error is True
            assert "system annotation" in result.output.lower()

    def test_k8s_io_system_annotation_blocked(self) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_annotate(
                "pods", "my-pod",
                "k8s.io/something=test",
                gke_config=cfg,
            )
            assert result.error is True
            assert "system annotation" in result.output.lower()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_annotate_pod_success(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_annotate(
                "pods", "my-pod",
                "description=web server,owner=team-a",
                gke_config=cfg,
            )
            assert result.error is not True
            assert "Annotations updated" in result.output
            assert "description=web server" in result.output
            core_v1.patch_namespaced_pod.assert_called_once()
            patch_body = core_v1.patch_namespaced_pod.call_args[0][2]
            assert patch_body["metadata"]["annotations"]["description"] == "web server"
            assert patch_body["metadata"]["annotations"]["owner"] == "team-a"

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_annotate_remove(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_annotate("pods", "my-pod", "old-note-", gke_config=cfg)
            assert result.error is not True
            assert "old-note-" in result.output
            patch_body = core_v1.patch_namespaced_pod.call_args[0][2]
            assert patch_body["metadata"]["annotations"]["old-note"] is None

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_annotate_deployment(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_annotate("deploy", "nginx", "gitsha=abc123", gke_config=cfg)
            assert result.error is not True
            apps_v1.patch_namespaced_deployment.assert_called_once()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_annotate_statefulset(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_annotate("sts", "redis", "backup=daily", gke_config=cfg)
            assert result.error is not True
            apps_v1.patch_namespaced_stateful_set.assert_called_once()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_annotate_daemonset(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_annotate("ds", "fluentd", "log-driver=json", gke_config=cfg)
            assert result.error is not True
            apps_v1.patch_namespaced_daemon_set.assert_called_once()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_annotate_namespace(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_annotate("namespaces", "prod", "team=platform", gke_config=cfg)
            assert result.error is not True
            core_v1.patch_namespace.assert_called_once()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_annotate_node(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True):
            result = kubectl_annotate("nodes", "worker-1", "rack=a3", gke_config=cfg)
            assert result.error is not True
            core_v1.patch_node.assert_called_once()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_annotate_404(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        exc_class = type("ApiException", (Exception,), {"status": 404, "reason": "Not Found"})

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.mutations.k8s_exceptions") as mock_k8s_exc:
            mock_k8s_exc.ApiException = exc_class
            core_v1.patch_namespaced_pod.side_effect = exc_class()
            mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())
            result = kubectl_annotate("pods", "ghost", "note=test", gke_config=cfg)
            assert result.error is True
            assert "not found" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_annotate_403(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        exc_class = type("ApiException", (Exception,), {"status": 403, "reason": "Forbidden"})

        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.mutations.k8s_exceptions") as mock_k8s_exc:
            mock_k8s_exc.ApiException = exc_class
            core_v1.patch_namespaced_pod.side_effect = exc_class()
            mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())
            result = kubectl_annotate("pods", "my-pod", "note=test", gke_config=cfg)
            assert result.error is True
            assert "Access denied" in result.output

    def test_annotate_custom_prefix_allowed(self) -> None:
        """Annotations with custom prefixes (not system) should be allowed."""
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.mutations._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._create_k8s_clients") as mock_clients:
            core_v1 = MagicMock()
            mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())
            result = kubectl_annotate("pods", "my-pod", "myorg.io/team=backend", gke_config=cfg)
            assert result.error is not True


# ── get_events ───────────────────────────────────────────────


def _mock_event(
    name: str = "my-pod.abc123",
    event_type: str = "Warning",
    reason: str = "BackOff",
    message: str = "Back-off restarting failed container",
    involved_name: str = "my-pod",
    involved_kind: str = "Pod",
    last_timestamp: datetime | None = None,
) -> MagicMock:
    """Create a mock Kubernetes event object."""
    ev = MagicMock()
    ev.metadata.name = name
    ev.metadata.creation_timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
    ev.type = event_type
    ev.reason = reason
    ev.message = message
    ev.last_timestamp = last_timestamp or datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    ev.involved_object.name = involved_name
    ev.involved_object.kind = involved_kind
    return ev


class TestGetEvents:
    """Tests for get_events diagnostic tool."""

    def test_k8s_unavailable(self) -> None:
        from vaig.tools.gke_tools import get_events

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = get_events(gke_config=cfg)
            assert result.error is True
            assert "kubernetes" in result.output.lower()

    def test_invalid_event_type(self) -> None:
        from vaig.tools.gke_tools import get_events

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_events(gke_config=cfg, event_type="Error")
            assert result.error is True
            assert "Invalid event_type" in result.output

    def test_invalid_limit_too_low(self) -> None:
        from vaig.tools.gke_tools import get_events

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_events(gke_config=cfg, limit=0)
            assert result.error is True
            assert "Limit must be" in result.output

    def test_invalid_limit_too_high(self) -> None:
        from vaig.tools.gke_tools import get_events

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_events(gke_config=cfg, limit=501)
            assert result.error is True
            assert "Limit must be" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_client_creation_error(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_events

        cfg = _make_gke_config()
        mock_clients.return_value = ToolResult(output="Failed to configure", error=True)

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_events(gke_config=cfg)
            assert result.error is True
            assert "Failed to configure" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_no_events_found(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_events

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        ev_list = MagicMock()
        ev_list.items = []
        core_v1.list_namespaced_event.return_value = ev_list

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_events(gke_config=cfg)

        assert result.error is False
        assert "No events found" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_no_events_with_filter_desc(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_events

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        ev_list = MagicMock()
        ev_list.items = []
        core_v1.list_namespaced_event.return_value = ev_list

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_events(gke_config=cfg, event_type="Warning", involved_object_name="my-pod")

        assert result.error is False
        assert "No events found" in result.output
        assert "Filters:" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_successful_event_list(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_events

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        ev1 = _mock_event(
            name="ev1", event_type="Warning", reason="BackOff",
            message="Back-off restarting", involved_name="my-pod", involved_kind="Pod",
            last_timestamp=datetime(2025, 1, 1, 14, 0, 0, tzinfo=timezone.utc),
        )
        ev2 = _mock_event(
            name="ev2", event_type="Normal", reason="Pulled",
            message="Container image pulled", involved_name="my-pod", involved_kind="Pod",
            last_timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        ev_list = MagicMock()
        ev_list.items = [ev2, ev1]  # out of order to test sorting
        core_v1.list_namespaced_event.return_value = ev_list

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_events(gke_config=cfg)

        assert result.error is False
        assert "LAST SEEN" in result.output
        assert "TYPE" in result.output
        assert "REASON" in result.output
        assert "BackOff" in result.output
        assert "Pod/my-pod" in result.output
        # Verify sort order (most recent first) — BackOff should come before Pulled
        backoff_idx = result.output.index("BackOff")
        pulled_idx = result.output.index("Pulled")
        assert backoff_idx < pulled_idx

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_event_type_filter_passed_as_field_selector(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_events

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        ev_list = MagicMock()
        ev_list.items = [_mock_event()]
        core_v1.list_namespaced_event.return_value = ev_list

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            get_events(gke_config=cfg, event_type="Warning")

        call_kwargs = core_v1.list_namespaced_event.call_args
        assert "type=Warning" in call_kwargs.kwargs.get("field_selector", "")

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_involved_object_filter(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_events

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        ev_list = MagicMock()
        ev_list.items = [_mock_event()]
        core_v1.list_namespaced_event.return_value = ev_list

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            get_events(
                gke_config=cfg,
                involved_object_name="my-pod",
                involved_object_kind="Pod",
            )

        call_kwargs = core_v1.list_namespaced_event.call_args
        field_sel = call_kwargs.kwargs.get("field_selector", "")
        assert "involvedObject.name=my-pod" in field_sel
        assert "involvedObject.kind=Pod" in field_sel

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_limit_applied(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_events

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        # Create 5 events but limit to 2
        events = [
            _mock_event(name=f"ev{i}", last_timestamp=datetime(2025, 1, 1, i, 0, 0, tzinfo=timezone.utc))
            for i in range(5)
        ]
        ev_list = MagicMock()
        ev_list.items = events
        core_v1.list_namespaced_event.return_value = ev_list

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_events(gke_config=cfg, limit=2)

        assert result.error is False
        assert "2 shown" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_events_with_none_last_timestamp(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_events

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        ev = _mock_event()
        ev.last_timestamp = None
        ev_list = MagicMock()
        ev_list.items = [ev]
        core_v1.list_namespaced_event.return_value = ev_list

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_events(gke_config=cfg)

        assert result.error is False
        assert "BackOff" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_namespace_404(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_events

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        exc_class = type("ApiException", (Exception,), {"status": 404, "reason": "Not Found"})

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.diagnostics.k8s_exceptions") as mock_k8s_exc:
            mock_k8s_exc.ApiException = exc_class
            core_v1.list_namespaced_event.side_effect = exc_class()
            mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())
            result = get_events(gke_config=cfg, namespace="nonexistent")

        assert result.error is True
        assert "not found" in result.output.lower()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_events_403(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_events

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        exc_class = type("ApiException", (Exception,), {"status": 403, "reason": "Forbidden"})

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.diagnostics.k8s_exceptions") as mock_k8s_exc:
            mock_k8s_exc.ApiException = exc_class
            core_v1.list_namespaced_event.side_effect = exc_class()
            mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())
            result = get_events(gke_config=cfg)

        assert result.error is True
        assert "Access denied" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_events_401(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_events

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        exc_class = type("ApiException", (Exception,), {"status": 401, "reason": "Unauthorized"})

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.diagnostics.k8s_exceptions") as mock_k8s_exc:
            mock_k8s_exc.ApiException = exc_class
            core_v1.list_namespaced_event.side_effect = exc_class()
            mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())
            result = get_events(gke_config=cfg)

        assert result.error is True
        assert "Authentication failed" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_events_generic_exception(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_events

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())
        core_v1.list_namespaced_event.side_effect = RuntimeError("connection lost")

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_events(gke_config=cfg)

        assert result.error is True
        assert "connection lost" in result.output

    def test_event_type_none_accepted(self) -> None:
        """event_type=None should not trigger validation error."""
        from vaig.tools.gke_tools import get_events

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._create_k8s_clients") as mock_clients:
            core_v1 = MagicMock()
            ev_list = MagicMock()
            ev_list.items = []
            core_v1.list_namespaced_event.return_value = ev_list
            mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())
            result = get_events(gke_config=cfg, event_type=None)

        assert result.error is False

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_uses_config_default_namespace(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_events

        cfg = _make_gke_config(default_namespace="production")
        core_v1 = MagicMock()
        ev_list = MagicMock()
        ev_list.items = []
        core_v1.list_namespaced_event.return_value = ev_list
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            get_events(gke_config=cfg, namespace="")

        call_kwargs = core_v1.list_namespaced_event.call_args
        assert call_kwargs.kwargs.get("namespace") == "production" or call_kwargs[1].get("namespace") == "production"


# ── get_rollout_status ───────────────────────────────────────


def _mock_deployment_status(
    name: str = "my-deploy",
    namespace: str = "default",
    desired: int = 3,
    current: int = 3,
    ready: int = 3,
    updated: int = 3,
    available: int = 3,
    unavailable: int = 0,
    strategy_type: str = "RollingUpdate",
    max_unavailable: str = "25%",
    max_surge: str = "25%",
    conditions: list[tuple[str, str, str, str]] | None = None,
) -> MagicMock:
    """Create a mock deployment with status for rollout status tests.

    conditions: list of (type, status, reason, message) tuples.
    """
    dep = MagicMock()
    dep.metadata.name = name
    dep.metadata.namespace = namespace
    dep.spec.replicas = desired
    dep.spec.strategy.type = strategy_type
    dep.spec.strategy.rolling_update.max_unavailable = max_unavailable
    dep.spec.strategy.rolling_update.max_surge = max_surge

    dep.status.replicas = current
    dep.status.ready_replicas = ready
    dep.status.updated_replicas = updated
    dep.status.available_replicas = available
    dep.status.unavailable_replicas = unavailable

    if conditions is None:
        conditions = [
            ("Available", "True", "MinimumReplicasAvailable", "Deployment has minimum availability."),
            ("Progressing", "True", "NewReplicaSetAvailable", "ReplicaSet has successfully progressed."),
        ]

    mock_conditions = []
    for cond_type, cond_status, reason, message in conditions:
        cond = MagicMock()
        cond.type = cond_type
        cond.status = cond_status
        cond.reason = reason
        cond.message = message
        mock_conditions.append(cond)

    dep.status.conditions = mock_conditions
    return dep


class TestGetRolloutStatus:
    """Tests for get_rollout_status diagnostic tool."""

    def test_k8s_unavailable(self) -> None:
        from vaig.tools.gke_tools import get_rollout_status

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = get_rollout_status("my-deploy", gke_config=cfg)
            assert result.error is True
            assert "kubernetes" in result.output.lower()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_client_creation_error(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_status

        cfg = _make_gke_config()
        mock_clients.return_value = ToolResult(output="Failed to configure", error=True)

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_rollout_status("my-deploy", gke_config=cfg)
            assert result.error is True
            assert "Failed to configure" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_complete_rollout(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_status

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        dep = _mock_deployment_status(
            name="web", desired=3, current=3, ready=3, updated=3, available=3,
        )
        apps_v1.read_namespaced_deployment.return_value = dep

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_rollout_status("web", gke_config=cfg)

        assert result.error is False
        assert "Overall Status: Complete" in result.output
        assert "Deployment: web" in result.output
        assert "Desired:     3" in result.output
        assert "Ready:       3" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_progressing_rollout(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_status

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        dep = _mock_deployment_status(
            name="web", desired=3, current=4, ready=2, updated=2,
            available=2, unavailable=1,
            conditions=[
                ("Available", "True", "MinimumReplicasAvailable", "ok"),
                ("Progressing", "True", "ReplicaSetUpdated", "Updated to 2"),
            ],
        )
        apps_v1.read_namespaced_deployment.return_value = dep

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_rollout_status("web", gke_config=cfg)

        assert result.error is False
        assert "Overall Status: Progressing" in result.output
        assert "Unavailable: 1" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_stalled_rollout(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_status

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        dep = _mock_deployment_status(
            name="web", desired=3, current=4, ready=2, updated=1,
            available=2, unavailable=1,
            conditions=[
                ("Available", "True", "MinimumReplicasAvailable", "ok"),
                ("Progressing", "False", "ProgressDeadlineExceeded", "timed out"),
            ],
        )
        apps_v1.read_namespaced_deployment.return_value = dep

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_rollout_status("web", gke_config=cfg)

        assert result.error is False
        assert "Overall Status: Stalled" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_failed_rollout(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_status

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        dep = _mock_deployment_status(
            name="web", desired=3, current=3, ready=1, updated=1,
            available=1, unavailable=2,
            conditions=[
                ("Available", "False", "MinimumReplicasUnavailable", "not enough"),
                ("Progressing", "True", "ReplicaSetUpdated", "progressing"),
                ("ReplicaFailure", "True", "FailedCreate", "quota exceeded"),
            ],
        )
        apps_v1.read_namespaced_deployment.return_value = dep

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_rollout_status("web", gke_config=cfg)

        assert result.error is False
        assert "Overall Status: Failed" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_scaled_to_zero(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_status

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        dep = _mock_deployment_status(
            name="web", desired=0, current=0, ready=0, updated=0,
            available=0, unavailable=0,
            conditions=[],
        )
        dep.status.conditions = []
        apps_v1.read_namespaced_deployment.return_value = dep

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_rollout_status("web", gke_config=cfg)

        assert result.error is False
        assert "Scaled to zero" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_strategy_displayed(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_status

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        dep = _mock_deployment_status(
            strategy_type="RollingUpdate",
            max_unavailable="1",
            max_surge="2",
        )
        apps_v1.read_namespaced_deployment.return_value = dep

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_rollout_status("my-deploy", gke_config=cfg)

        assert result.error is False
        assert "Strategy: RollingUpdate" in result.output
        assert "Max Unavailable: 1" in result.output
        assert "Max Surge:       2" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_conditions_displayed(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_status

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        dep = _mock_deployment_status()
        apps_v1.read_namespaced_deployment.return_value = dep

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_rollout_status("my-deploy", gke_config=cfg)

        assert result.error is False
        assert "Conditions:" in result.output
        assert "Available: True" in result.output
        assert "Progressing: True" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_deployment_not_found(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_status

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        exc_class = type("ApiException", (Exception,), {"status": 404, "reason": "Not Found"})

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.diagnostics.k8s_exceptions") as mock_k8s_exc:
            mock_k8s_exc.ApiException = exc_class
            apps_v1.read_namespaced_deployment.side_effect = exc_class()
            mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())
            result = get_rollout_status("ghost", gke_config=cfg)

        assert result.error is True
        assert "not found" in result.output.lower()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_deployment_403(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_status

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        exc_class = type("ApiException", (Exception,), {"status": 403, "reason": "Forbidden"})

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.diagnostics.k8s_exceptions") as mock_k8s_exc:
            mock_k8s_exc.ApiException = exc_class
            apps_v1.read_namespaced_deployment.side_effect = exc_class()
            mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())
            result = get_rollout_status("web", gke_config=cfg)

        assert result.error is True
        assert "Access denied" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_deployment_401(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_status

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        exc_class = type("ApiException", (Exception,), {"status": 401, "reason": "Unauthorized"})

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.diagnostics.k8s_exceptions") as mock_k8s_exc:
            mock_k8s_exc.ApiException = exc_class
            apps_v1.read_namespaced_deployment.side_effect = exc_class()
            mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())
            result = get_rollout_status("web", gke_config=cfg)

        assert result.error is True
        assert "Authentication failed" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_generic_exception(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_status

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())
        apps_v1.read_namespaced_deployment.side_effect = RuntimeError("timeout")

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_rollout_status("web", gke_config=cfg)

        assert result.error is True
        assert "timeout" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_uses_config_default_namespace(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_status

        cfg = _make_gke_config(default_namespace="staging")
        apps_v1 = MagicMock()
        dep = _mock_deployment_status()
        apps_v1.read_namespaced_deployment.return_value = dep
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            get_rollout_status("my-deploy", gke_config=cfg, namespace="")

        call_args = apps_v1.read_namespaced_deployment.call_args
        assert call_args.kwargs.get("namespace") == "staging"

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_no_status_conditions(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_status

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        dep = _mock_deployment_status(conditions=[])
        dep.status.conditions = []
        apps_v1.read_namespaced_deployment.return_value = dep

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_rollout_status("my-deploy", gke_config=cfg)

        # Should not crash, should show Unknown or Complete depending on replica state
        assert result.error is False
        assert "Overall Status:" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_null_status_fields(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_status

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        dep = MagicMock()
        dep.metadata.name = "web"
        dep.metadata.namespace = "default"
        dep.spec.replicas = 3
        dep.spec.strategy.type = "RollingUpdate"
        dep.spec.strategy.rolling_update.max_unavailable = "25%"
        dep.spec.strategy.rolling_update.max_surge = "25%"
        dep.status.replicas = None
        dep.status.ready_replicas = None
        dep.status.updated_replicas = None
        dep.status.available_replicas = None
        dep.status.unavailable_replicas = None
        dep.status.conditions = []
        apps_v1.read_namespaced_deployment.return_value = dep

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_rollout_status("web", gke_config=cfg)

        # Should handle None gracefully, defaulting to 0
        assert result.error is False
        assert "Current:     0" in result.output
        assert "Ready:       0" in result.output


# ── Mock factories (Phase 2) ────────────────────────────────


def _mock_node(
    name: str = "gke-node-1",
    ready: bool = True,
    memory_pressure: bool = False,
    disk_pressure: bool = False,
    pid_pressure: bool = False,
    unschedulable: bool = False,
    cpu_capacity: str = "4",
    cpu_allocatable: str = "3920m",
    memory_capacity: str = "16777216Ki",
    memory_allocatable: str = "15728640Ki",
    labels: dict[str, str] | None = None,
    taints: list[dict[str, str]] | None = None,
) -> MagicMock:
    """Create a mock Kubernetes node object."""
    node = MagicMock()
    node.metadata.name = name
    node.metadata.creation_timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
    node.metadata.labels = labels or {
        "node-role.kubernetes.io/worker": "",
        "topology.kubernetes.io/zone": "us-central1-a",
        "cloud.google.com/gke-nodepool": "default-pool",
        "kubernetes.io/arch": "amd64",
        "kubernetes.io/os": "linux",
        "kubernetes.io/hostname": name,
    }

    # Node info
    node.status.node_info.kubelet_version = "v1.28.3-gke.1286000"
    node.status.node_info.os_image = "Ubuntu"
    node.status.node_info.kernel_version = "5.15.0-1049-gke"
    node.status.node_info.container_runtime_version = "containerd://1.7.7"
    node.status.node_info.architecture = "amd64"
    node.status.node_info.operating_system = "linux"

    # Addresses
    addr_internal = MagicMock()
    addr_internal.type = "InternalIP"
    addr_internal.address = "10.128.0.5"
    addr_external = MagicMock()
    addr_external.type = "ExternalIP"
    addr_external.address = "35.192.0.1"
    node.status.addresses = [addr_internal, addr_external]

    # Conditions
    conditions = []
    for ctype, is_problem in [
        ("Ready", ready),
        ("MemoryPressure", memory_pressure),
        ("DiskPressure", disk_pressure),
        ("PIDPressure", pid_pressure),
    ]:
        cond = MagicMock()
        cond.type = ctype
        if ctype == "Ready":
            cond.status = "True" if is_problem else "False"
        else:
            cond.status = "True" if is_problem else "False"
        cond.reason = "KubeletReady" if ctype == "Ready" else f"Kubelet Has No {ctype}"
        cond.message = f"{ctype} condition message"
        cond.last_transition_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
        conditions.append(cond)
    node.status.conditions = conditions

    # Capacity / Allocatable
    node.status.capacity = {
        "cpu": cpu_capacity,
        "memory": memory_capacity,
        "pods": "110",
        "ephemeral-storage": "98831908Ki",
    }
    node.status.allocatable = {
        "cpu": cpu_allocatable,
        "memory": memory_allocatable,
        "pods": "110",
        "ephemeral-storage": "47093746742",
    }

    # Spec
    node.spec.unschedulable = unschedulable

    # Taints
    if taints:
        mock_taints = []
        for t in taints:
            taint = MagicMock()
            taint.key = t["key"]
            taint.value = t.get("value", "")
            taint.effect = t["effect"]
            mock_taints.append(taint)
        node.spec.taints = mock_taints
    else:
        node.spec.taints = []

    return node


def _mock_container_pod(
    name: str = "my-pod",
    namespace: str = "default",
    phase: str = "Running",
    node_name: str = "node-1",
) -> MagicMock:
    """Create a mock pod with detailed container specs for get_container_status tests."""
    pod = MagicMock()
    pod.metadata.name = name
    pod.metadata.namespace = namespace
    pod.spec.node_name = node_name
    pod.status.phase = phase

    # Main container spec
    main_container = MagicMock()
    main_container.name = "app"
    main_container.image = "myapp:v1.2.3"
    main_container.resources.requests = {"cpu": "100m", "memory": "256Mi"}
    main_container.resources.limits = {"cpu": "500m", "memory": "512Mi"}

    # Volume mounts
    vm = MagicMock()
    vm.name = "config-vol"
    vm.mount_path = "/etc/config"
    vm.read_only = True
    main_container.volume_mounts = [vm]

    # Env from
    ef = MagicMock()
    ef.config_map_ref = MagicMock()
    ef.config_map_ref.name = "app-config"
    ef.secret_ref = None
    main_container.env_from = [ef]

    # Env vars with valueFrom
    env_secret = MagicMock()
    env_secret.name = "DB_PASSWORD"
    env_secret.value_from.config_map_key_ref = None
    env_secret.value_from.secret_key_ref.name = "db-secret"
    env_secret.value_from.secret_key_ref.key = "password"
    env_plain = MagicMock()
    env_plain.name = "APP_ENV"
    env_plain.value_from = None
    main_container.env = [env_secret, env_plain]

    pod.spec.containers = [main_container]

    # Init container
    init_container = MagicMock()
    init_container.name = "init-db"
    init_container.image = "db-migrator:latest"
    init_container.resources.requests = {"cpu": "50m", "memory": "64Mi"}
    init_container.resources.limits = {"cpu": "200m", "memory": "128Mi"}
    init_container.volume_mounts = []
    init_container.env_from = []
    init_container.env = []
    pod.spec.init_containers = [init_container]

    # No ephemeral containers
    pod.spec.ephemeral_containers = []

    # Container statuses
    main_cs = MagicMock()
    main_cs.name = "app"
    main_cs.image_id = "docker://sha256:abc123"
    main_cs.ready = True
    main_cs.restart_count = 0
    main_cs.state.running.started_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    main_cs.state.waiting = None
    main_cs.state.terminated = None
    main_cs.last_state.terminated = None
    pod.status.container_statuses = [main_cs]

    # Init container status
    init_cs = MagicMock()
    init_cs.name = "init-db"
    init_cs.image_id = "docker://sha256:def456"
    init_cs.ready = False
    init_cs.restart_count = 0
    init_cs.state.running = None
    init_cs.state.waiting = None
    init_cs.state.terminated.reason = "Completed"
    init_cs.state.terminated.exit_code = 0
    init_cs.state.terminated.message = None
    init_cs.state.terminated.started_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    init_cs.state.terminated.finished_at = datetime(2025, 1, 1, 0, 1, 0, tzinfo=timezone.utc)
    init_cs.last_state.terminated = None
    pod.status.init_container_statuses = [init_cs]

    # No ephemeral statuses
    pod.status.ephemeral_container_statuses = []

    return pod


# ── get_node_conditions tests ────────────────────────────────


class TestGetNodeConditions:
    """Tests for get_node_conditions diagnostic tool."""

    def test_k8s_unavailable(self) -> None:
        from vaig.tools.gke_tools import get_node_conditions

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = get_node_conditions(gke_config=cfg)
            assert result.error is True
            assert "kubernetes" in result.output.lower()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_client_creation_error(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_node_conditions

        cfg = _make_gke_config()
        mock_clients.return_value = ToolResult(output="Failed to configure", error=True)

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_node_conditions(gke_config=cfg)
            assert result.error is True
            assert "Failed to configure" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_no_nodes_found(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_node_conditions

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        core_v1.list_node.return_value.items = []
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_node_conditions(gke_config=cfg)

        assert result.error is False
        assert "No nodes found" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_all_nodes_summary(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_node_conditions

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        nodes = [
            _mock_node(name="node-1"),
            _mock_node(name="node-2", ready=False),
        ]
        core_v1.list_node.return_value.items = nodes
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_node_conditions(gke_config=cfg)

        assert result.error is False
        assert "Nodes (2)" in result.output
        assert "node-1" in result.output
        assert "node-2" in result.output
        assert "Ready" in result.output
        assert "NotReady" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_all_nodes_shows_capacity(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_node_conditions

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        core_v1.list_node.return_value.items = [_mock_node()]
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_node_conditions(gke_config=cfg)

        assert result.error is False
        # Should show CPU capacity/allocatable columns
        assert "CPU" in result.output
        assert "MEM" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_single_node_detail(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_node_conditions

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        node = _mock_node(name="gke-node-detail")
        core_v1.read_node.return_value = node
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_node_conditions(gke_config=cfg, name="gke-node-detail")

        assert result.error is False
        assert "Node: gke-node-detail" in result.output
        assert "Conditions:" in result.output
        assert "Ready" in result.output
        assert "MemoryPressure" in result.output
        assert "DiskPressure" in result.output
        assert "PIDPressure" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_single_node_shows_taints(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_node_conditions

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        node = _mock_node(
            name="tainted-node",
            taints=[
                {"key": "node.kubernetes.io/not-ready", "effect": "NoSchedule"},
                {"key": "dedicated", "value": "gpu", "effect": "NoSchedule"},
            ],
        )
        core_v1.read_node.return_value = node
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_node_conditions(gke_config=cfg, name="tainted-node")

        assert result.error is False
        assert "Taints:" in result.output
        assert "node.kubernetes.io/not-ready" in result.output
        assert "dedicated=gpu:NoSchedule" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_single_node_no_taints(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_node_conditions

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        node = _mock_node(name="clean-node")
        core_v1.read_node.return_value = node
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_node_conditions(gke_config=cfg, name="clean-node")

        assert result.error is False
        assert "(none)" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_single_node_shows_labels(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_node_conditions

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        node = _mock_node(name="labeled-node")
        core_v1.read_node.return_value = node
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_node_conditions(gke_config=cfg, name="labeled-node")

        assert result.error is False
        assert "Labels (relevant):" in result.output
        assert "topology.kubernetes.io/zone=us-central1-a" in result.output
        assert "cloud.google.com/gke-nodepool=default-pool" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_single_node_capacity_vs_allocatable(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_node_conditions

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        node = _mock_node(name="capacity-node")
        core_v1.read_node.return_value = node
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_node_conditions(gke_config=cfg, name="capacity-node")

        assert result.error is False
        assert "Capacity vs Allocatable:" in result.output
        assert "cpu" in result.output
        assert "memory" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_single_node_unschedulable(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_node_conditions

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        node = _mock_node(name="cordoned-node", unschedulable=True)
        core_v1.read_node.return_value = node
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_node_conditions(gke_config=cfg, name="cordoned-node")

        assert result.error is False
        assert "Unschedulable (cordoned): True" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_single_node_addresses(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_node_conditions

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        node = _mock_node(name="addr-node")
        core_v1.read_node.return_value = node
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_node_conditions(gke_config=cfg, name="addr-node")

        assert result.error is False
        assert "Addresses:" in result.output
        assert "InternalIP: 10.128.0.5" in result.output
        assert "ExternalIP: 35.192.0.1" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_node_not_found_404(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_node_conditions

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        exc_class = type("ApiException", (Exception,), {"status": 404, "reason": "Not Found"})

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.diagnostics.k8s_exceptions") as mock_k8s_exc:
            mock_k8s_exc.ApiException = exc_class
            core_v1.read_node.side_effect = exc_class()
            mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())
            result = get_node_conditions(gke_config=cfg, name="ghost-node")

        assert result.error is True
        assert "not found" in result.output.lower()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_node_403(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_node_conditions

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        exc_class = type("ApiException", (Exception,), {"status": 403, "reason": "Forbidden"})

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.diagnostics.k8s_exceptions") as mock_k8s_exc:
            mock_k8s_exc.ApiException = exc_class
            core_v1.read_node.side_effect = exc_class()
            mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())
            result = get_node_conditions(gke_config=cfg, name="node")

        assert result.error is True
        assert "Access denied" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_node_401(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_node_conditions

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        exc_class = type("ApiException", (Exception,), {"status": 401, "reason": "Unauthorized"})

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.diagnostics.k8s_exceptions") as mock_k8s_exc:
            mock_k8s_exc.ApiException = exc_class
            core_v1.read_node.side_effect = exc_class()
            mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())
            result = get_node_conditions(gke_config=cfg, name="node")

        assert result.error is True
        assert "Authentication failed" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_generic_exception(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_node_conditions

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())
        core_v1.list_node.side_effect = RuntimeError("connection timeout")

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_node_conditions(gke_config=cfg)

        assert result.error is True
        assert "connection timeout" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_node_with_pressure_conditions(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_node_conditions

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        node = _mock_node(
            name="pressure-node",
            memory_pressure=True,
            disk_pressure=True,
        )
        core_v1.read_node.return_value = node
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_node_conditions(gke_config=cfg, name="pressure-node")

        assert result.error is False
        # All conditions should be visible
        assert "MemoryPressure" in result.output
        assert "DiskPressure" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_node_roles_displayed(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_node_conditions

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        node = _mock_node(name="worker-node")
        core_v1.list_node.return_value.items = [node]
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_node_conditions(gke_config=cfg)

        assert result.error is False
        assert "worker" in result.output


# ── get_container_status tests ───────────────────────────────


class TestGetContainerStatus:
    """Tests for get_container_status diagnostic tool."""

    def test_k8s_unavailable(self) -> None:
        from vaig.tools.gke_tools import get_container_status

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = get_container_status("my-pod", gke_config=cfg)
            assert result.error is True
            assert "kubernetes" in result.output.lower()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_client_creation_error(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_container_status

        cfg = _make_gke_config()
        mock_clients.return_value = ToolResult(output="Failed to configure", error=True)

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_container_status("my-pod", gke_config=cfg)
            assert result.error is True
            assert "Failed to configure" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_basic_running_pod(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_container_status

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        pod = _mock_container_pod()
        core_v1.read_namespaced_pod.return_value = pod
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_container_status("my-pod", gke_config=cfg)

        assert result.error is False
        assert "Pod: my-pod" in result.output
        assert "Phase: Running" in result.output
        assert "=== Containers ===" in result.output
        assert "Container: app" in result.output
        assert "Image: myapp:v1.2.3" in result.output
        assert "State: Running" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_init_containers_shown(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_container_status

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        pod = _mock_container_pod()
        core_v1.read_namespaced_pod.return_value = pod
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_container_status("my-pod", gke_config=cfg)

        assert "=== Init Containers ===" in result.output
        assert "Container: init-db" in result.output
        assert "Image: db-migrator:latest" in result.output
        assert "Terminated" in result.output
        assert "Completed" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_resource_requests_limits(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_container_status

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        pod = _mock_container_pod()
        core_v1.read_namespaced_pod.return_value = pod
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_container_status("my-pod", gke_config=cfg)

        assert "Resources:" in result.output
        assert "Requests:" in result.output
        assert "Limits:" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_volume_mounts_shown(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_container_status

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        pod = _mock_container_pod()
        core_v1.read_namespaced_pod.return_value = pod
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_container_status("my-pod", gke_config=cfg)

        assert "Volume Mounts:" in result.output
        assert "/etc/config from config-vol (ro)" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_env_from_configmap(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_container_status

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        pod = _mock_container_pod()
        core_v1.read_namespaced_pod.return_value = pod
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_container_status("my-pod", gke_config=cfg)

        assert "Env From:" in result.output
        assert "ConfigMap: app-config" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_env_var_secret_ref_no_values(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_container_status

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        pod = _mock_container_pod()
        core_v1.read_namespaced_pod.return_value = pod
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_container_status("my-pod", gke_config=cfg)

        assert "Env Var References:" in result.output
        assert "DB_PASSWORD" in result.output
        assert "Secret:db-secret/password" in result.output
        # Must NOT contain actual secret values
        assert "ref only" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_crashloopbackoff_last_state(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_container_status

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        pod = _mock_container_pod(phase="Running")

        # Override main container status to CrashLoopBackOff
        cs = pod.status.container_statuses[0]
        cs.restart_count = 5
        cs.state.running = None
        cs.state.terminated = None
        waiting = MagicMock()
        waiting.reason = "CrashLoopBackOff"
        waiting.message = "back-off 5m0s restarting failed container"
        cs.state.waiting = waiting

        # Set last termination state
        last_terminated = MagicMock()
        last_terminated.reason = "OOMKilled"
        last_terminated.exit_code = 137
        last_terminated.message = "Out of memory"
        last_terminated.started_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        last_terminated.finished_at = datetime(2025, 1, 1, 0, 5, 0, tzinfo=timezone.utc)
        cs.last_state.terminated = last_terminated

        core_v1.read_namespaced_pod.return_value = pod
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_container_status("my-pod", gke_config=cfg)

        assert result.error is False
        assert "Restart Count: 5" in result.output
        assert "CrashLoopBackOff" in result.output
        assert "Last State: Terminated" in result.output
        assert "OOMKilled" in result.output
        assert "exit code 137" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_pod_not_found_404(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_container_status

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        exc_class = type("ApiException", (Exception,), {"status": 404, "reason": "Not Found"})

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.diagnostics.k8s_exceptions") as mock_k8s_exc:
            mock_k8s_exc.ApiException = exc_class
            core_v1.read_namespaced_pod.side_effect = exc_class()
            mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())
            result = get_container_status("ghost-pod", gke_config=cfg)

        assert result.error is True
        assert "not found" in result.output.lower()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_pod_403(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_container_status

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        exc_class = type("ApiException", (Exception,), {"status": 403, "reason": "Forbidden"})

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.diagnostics.k8s_exceptions") as mock_k8s_exc:
            mock_k8s_exc.ApiException = exc_class
            core_v1.read_namespaced_pod.side_effect = exc_class()
            mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())
            result = get_container_status("my-pod", gke_config=cfg)

        assert result.error is True
        assert "Access denied" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_pod_401(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_container_status

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        exc_class = type("ApiException", (Exception,), {"status": 401, "reason": "Unauthorized"})

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.diagnostics.k8s_exceptions") as mock_k8s_exc:
            mock_k8s_exc.ApiException = exc_class
            core_v1.read_namespaced_pod.side_effect = exc_class()
            mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())
            result = get_container_status("my-pod", gke_config=cfg)

        assert result.error is True
        assert "Authentication failed" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_generic_exception(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_container_status

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())
        core_v1.read_namespaced_pod.side_effect = RuntimeError("timeout")

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_container_status("my-pod", gke_config=cfg)

        assert result.error is True
        assert "timeout" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_uses_config_default_namespace(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_container_status

        cfg = _make_gke_config(default_namespace="staging")
        core_v1 = MagicMock()
        pod = _mock_container_pod()
        core_v1.read_namespaced_pod.return_value = pod
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            get_container_status("my-pod", gke_config=cfg, namespace="")

        call_args = core_v1.read_namespaced_pod.call_args
        assert call_args.kwargs.get("namespace") == "staging"

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_ephemeral_containers(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_container_status

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        pod = _mock_container_pod()

        # Add ephemeral container
        eph_container = MagicMock()
        eph_container.name = "debug"
        eph_container.image = "busybox:latest"
        eph_container.resources = None
        eph_container.volume_mounts = []
        eph_container.env_from = []
        eph_container.env = []
        pod.spec.ephemeral_containers = [eph_container]

        # Add ephemeral status
        eph_cs = MagicMock()
        eph_cs.name = "debug"
        eph_cs.image_id = "docker://sha256:eph123"
        eph_cs.ready = False
        eph_cs.restart_count = 0
        eph_cs.state.running.started_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        eph_cs.state.waiting = None
        eph_cs.state.terminated = None
        eph_cs.last_state.terminated = None
        pod.status.ephemeral_container_statuses = [eph_cs]

        core_v1.read_namespaced_pod.return_value = pod
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_container_status("my-pod", gke_config=cfg)

        assert result.error is False
        assert "=== Ephemeral Containers ===" in result.output
        assert "Container: debug" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_no_status_available(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_container_status

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        pod = _mock_container_pod()
        # Clear status
        pod.status.container_statuses = []
        core_v1.read_namespaced_pod.return_value = pod
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_container_status("my-pod", gke_config=cfg)

        assert result.error is False
        assert "no status available" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_env_from_secret_ref(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_container_status

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        pod = _mock_container_pod()

        # Add secret ref to env_from
        ef_secret = MagicMock()
        ef_secret.config_map_ref = None
        ef_secret.secret_ref = MagicMock()
        ef_secret.secret_ref.name = "app-secrets"
        pod.spec.containers[0].env_from.append(ef_secret)

        core_v1.read_namespaced_pod.return_value = pod
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_container_status("my-pod", gke_config=cfg)

        assert result.error is False
        assert "Secret: app-secrets (ref only, no values shown)" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_restart_count_displayed(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_container_status

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        pod = _mock_container_pod()
        pod.status.container_statuses[0].restart_count = 42
        core_v1.read_namespaced_pod.return_value = pod
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_container_status("my-pod", gke_config=cfg)

        assert result.error is False
        assert "Restart Count: 42" in result.output


# ── _format_memory helper tests ──────────────────────────────


class TestFormatMemory:
    """Tests for _format_memory helper function."""

    def test_ki_to_gi(self) -> None:
        from vaig.tools.gke_tools import _format_memory

        assert _format_memory("16777216Ki") == "16.0Gi"

    def test_ki_to_mi(self) -> None:
        from vaig.tools.gke_tools import _format_memory

        assert _format_memory("524288Ki") == "512Mi"

    def test_mi_to_gi(self) -> None:
        from vaig.tools.gke_tools import _format_memory

        assert _format_memory("2048Mi") == "2.0Gi"

    def test_mi_small(self) -> None:
        from vaig.tools.gke_tools import _format_memory

        assert _format_memory("256Mi") == "256Mi"

    def test_gi_passthrough(self) -> None:
        from vaig.tools.gke_tools import _format_memory

        assert _format_memory("4Gi") == "4Gi"

    def test_plain_bytes(self) -> None:
        from vaig.tools.gke_tools import _format_memory

        # 2 GiB in bytes
        assert _format_memory("2147483648") == "2.0Gi"

    def test_unknown(self) -> None:
        from vaig.tools.gke_tools import _format_memory

        assert _format_memory("?") == "?"

    def test_empty_string(self) -> None:
        from vaig.tools.gke_tools import _format_memory

        assert _format_memory("") == "?"


# ── exec_command tests ───────────────────────────────────────


class TestExecCommand:
    """Tests for exec_command diagnostic tool (Phase 3)."""

    def test_k8s_unavailable(self) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        with patch("vaig.tools.gke.security._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = exec_command("my-pod", "ls /tmp", gke_config=cfg)
            assert result.error is True
            assert "kubernetes" in result.output.lower()

    def test_exec_disabled_by_default(self) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config()  # exec_enabled defaults to False
        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True):
            result = exec_command("my-pod", "ls /tmp", gke_config=cfg)
            assert result.error is True
            assert "exec_command is disabled" in result.output
            assert "exec_enabled" in result.output

    def test_empty_command_rejected(self) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True):
            result = exec_command("my-pod", "", gke_config=cfg)
            assert result.error is True
            assert "empty" in result.output.lower()

    def test_whitespace_command_rejected(self) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True):
            result = exec_command("my-pod", "   ", gke_config=cfg)
            assert result.error is True
            assert "empty" in result.output.lower()

    # ── Denylist tests ────────────────────────────────────

    def test_deny_semicolon(self) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True):
            result = exec_command("my-pod", "cat /etc/passwd; rm -rf /", gke_config=cfg)
            assert result.error is True
            assert "denied" in result.output.lower()

    def test_deny_pipe(self) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True):
            result = exec_command("my-pod", "cat /etc/passwd | grep root", gke_config=cfg)
            assert result.error is True
            assert "denied" in result.output.lower()

    def test_deny_ampersand(self) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True):
            result = exec_command("my-pod", "ls & rm -rf /", gke_config=cfg)
            assert result.error is True
            assert "denied" in result.output.lower()

    def test_deny_backtick(self) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True):
            result = exec_command("my-pod", "cat `whoami`", gke_config=cfg)
            assert result.error is True
            assert "denied" in result.output.lower()

    def test_deny_dollar(self) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True):
            result = exec_command("my-pod", "cat $(whoami)", gke_config=cfg)
            assert result.error is True
            assert "denied" in result.output.lower()

    def test_deny_redirect(self) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True):
            result = exec_command("my-pod", "echo test > /tmp/x", gke_config=cfg)
            assert result.error is True
            assert "denied" in result.output.lower()

    def test_deny_append_redirect(self) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True):
            result = exec_command("my-pod", "echo test >> /tmp/x", gke_config=cfg)
            assert result.error is True
            assert "denied" in result.output.lower()

    def test_deny_rm(self) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True):
            result = exec_command("my-pod", "rm -rf /", gke_config=cfg)
            assert result.error is True
            assert "denied" in result.output.lower()

    def test_deny_kill(self) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True):
            result = exec_command("my-pod", "kill -9 1", gke_config=cfg)
            assert result.error is True
            assert "denied" in result.output.lower()

    def test_deny_shutdown(self) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True):
            result = exec_command("my-pod", "shutdown -h now", gke_config=cfg)
            assert result.error is True
            assert "denied" in result.output.lower()

    def test_deny_reboot(self) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True):
            result = exec_command("my-pod", "reboot", gke_config=cfg)
            assert result.error is True
            assert "denied" in result.output.lower()

    def test_deny_dd(self) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True):
            result = exec_command("my-pod", "dd if=/dev/zero of=/dev/sda", gke_config=cfg)
            assert result.error is True
            assert "denied" in result.output.lower()

    def test_deny_mkfs(self) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True):
            result = exec_command("my-pod", "mkfs.ext4 /dev/sda1", gke_config=cfg)
            assert result.error is True
            assert "denied" in result.output.lower()

    def test_deny_fdisk(self) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True):
            result = exec_command("my-pod", "fdisk /dev/sda", gke_config=cfg)
            assert result.error is True
            assert "denied" in result.output.lower()

    def test_deny_chmod(self) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True):
            result = exec_command("my-pod", "chmod 777 /tmp/x", gke_config=cfg)
            assert result.error is True
            assert "denied" in result.output.lower()

    def test_deny_chown(self) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True):
            result = exec_command("my-pod", "chown root:root /tmp/x", gke_config=cfg)
            assert result.error is True
            assert "denied" in result.output.lower()

    def test_deny_sudo(self) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True):
            result = exec_command("my-pod", "sudo cat /etc/shadow", gke_config=cfg)
            assert result.error is True
            assert "denied" in result.output.lower()

    # ── Allowlist tests ───────────────────────────────────

    def test_reject_unknown_command(self) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True):
            result = exec_command("my-pod", "python3 -c 'import os; os.system(\"rm -rf /\")'", gke_config=cfg)
            assert result.error is True
            # Should be caught by denylist ($) or by allowlist

    def test_reject_unlisted_command(self) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True):
            result = exec_command("my-pod", "apt-get install malware", gke_config=cfg)
            assert result.error is True
            assert "allowlist" in result.output.lower() or "allowed" in result.output.lower()

    # ── Timeout validation ────────────────────────────────

    def test_invalid_timeout_zero(self) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True):
            result = exec_command("my-pod", "ls /tmp", gke_config=cfg, timeout=0)
            assert result.error is True
            assert "timeout" in result.output.lower()

    def test_invalid_timeout_too_large(self) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True):
            result = exec_command("my-pod", "ls /tmp", gke_config=cfg, timeout=999)
            assert result.error is True
            assert "timeout" in result.output.lower()

    # ── Successful execution ──────────────────────────────

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_successful_ls(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.security.k8s_stream", create=True) as mock_stream:
            mock_stream.return_value = "file1.txt\nfile2.txt\ndir1\n"

            # Patch the import inside the function
            with patch("vaig.tools.gke.security.exec_command.__module__", "vaig.tools.gke.security"):
                with patch("kubernetes.stream.stream", mock_stream):
                    result = exec_command("my-pod", "ls /tmp", gke_config=cfg)

        assert result.error is False
        assert "ls /tmp" in result.output
        assert "my-pod" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_successful_cat(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True), \
             patch("kubernetes.stream.stream") as mock_stream:
            mock_stream.return_value = "nameserver 10.0.0.1\nsearch default.svc.cluster.local\n"
            result = exec_command("my-pod", "cat /etc/resolv.conf", gke_config=cfg)

        assert result.error is False
        assert "cat /etc/resolv.conf" in result.output
        assert "stdout" in result.output.lower()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_successful_with_container(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True), \
             patch("kubernetes.stream.stream") as mock_stream:
            mock_stream.return_value = "root\n"
            result = exec_command(
                "my-pod", "whoami", gke_config=cfg,
                container="sidecar",
            )

        assert result.error is False
        assert "Container: sidecar" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_output_truncation(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True), \
             patch("kubernetes.stream.stream") as mock_stream:
            mock_stream.return_value = "x" * 20000
            result = exec_command("my-pod", "cat /var/log/app.log", gke_config=cfg)

        assert result.error is False
        assert "truncated" in result.output.lower()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_empty_output(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True), \
             patch("kubernetes.stream.stream") as mock_stream:
            mock_stream.return_value = ""
            result = exec_command("my-pod", "ls /empty-dir", gke_config=cfg)

        assert result.error is False
        assert "no output" in result.output.lower()

    # ── Error handling ────────────────────────────────────

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_client_creation_error(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        mock_clients.return_value = ToolResult(output="Failed to configure", error=True)

        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True):
            result = exec_command("my-pod", "ls /tmp", gke_config=cfg)
            assert result.error is True
            assert "Failed to configure" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_pod_not_found(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        exc_class = type("ApiException", (Exception,), {"status": 404, "reason": "Not Found"})

        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.security.k8s_exceptions") as mock_exc, \
             patch("kubernetes.stream.stream") as mock_stream:
            mock_exc.ApiException = exc_class
            mock_stream.side_effect = exc_class()
            result = exec_command("nonexistent-pod", "ls /tmp", gke_config=cfg)

        assert result.error is True
        assert "not found" in result.output.lower()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_permission_denied(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import exec_command

        cfg = _make_gke_config(exec_enabled=True)
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        exc_class = type("ApiException", (Exception,), {"status": 403, "reason": "Forbidden"})

        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.security.k8s_exceptions") as mock_exc, \
             patch("kubernetes.stream.stream") as mock_stream:
            mock_exc.ApiException = exc_class
            mock_stream.side_effect = exc_class()
            result = exec_command("my-pod", "ls /tmp", gke_config=cfg)

        assert result.error is True
        assert "permission denied" in result.output.lower()

    # ── Allowlist edge cases ──────────────────────────────

    def test_allow_top_bn1(self) -> None:
        """Verify 'top -bn1' is allowed (multi-word prefix)."""
        from vaig.tools.gke_tools import _check_allowed

        assert _check_allowed("top -bn1") is True
        assert _check_allowed("top -bn1 -o %MEM") is True

    def test_allow_java_version(self) -> None:
        from vaig.tools.gke_tools import _check_allowed

        assert _check_allowed("java -version") is True

    def test_allow_python_version(self) -> None:
        from vaig.tools.gke_tools import _check_allowed

        assert _check_allowed("python --version") is True

    def test_allow_node_version(self) -> None:
        from vaig.tools.gke_tools import _check_allowed

        assert _check_allowed("node --version") is True

    def test_allow_cat_resolv_conf(self) -> None:
        from vaig.tools.gke_tools import _check_allowed

        assert _check_allowed("cat /etc/resolv.conf") is True

    def test_allow_cat_hosts(self) -> None:
        from vaig.tools.gke_tools import _check_allowed

        assert _check_allowed("cat /etc/hosts") is True

    def test_allow_cat_with_path(self) -> None:
        from vaig.tools.gke_tools import _check_allowed

        assert _check_allowed("cat /var/log/app.log") is True

    def test_allow_ls_with_flags(self) -> None:
        from vaig.tools.gke_tools import _check_allowed

        assert _check_allowed("ls -la /tmp") is True

    def test_allow_ps_aux(self) -> None:
        from vaig.tools.gke_tools import _check_allowed

        assert _check_allowed("ps aux") is True

    def test_allow_df_h(self) -> None:
        from vaig.tools.gke_tools import _check_allowed

        assert _check_allowed("df -h") is True

    def test_allow_curl_with_url(self) -> None:
        from vaig.tools.gke_tools import _check_allowed

        assert _check_allowed("curl http://localhost:8080/health") is True

    def test_reject_top_alone(self) -> None:
        """'top' alone is NOT allowed (only 'top -bn1' is)."""
        from vaig.tools.gke_tools import _check_allowed

        # 'top' without '-bn1' does NOT match 'top -bn1' prefix
        assert _check_allowed("top") is False

    def test_reject_unknown_prefix(self) -> None:
        from vaig.tools.gke_tools import _check_allowed

        assert _check_allowed("apt-get update") is False
        assert _check_allowed("pip install something") is False

    # ── Deny pattern unit tests ───────────────────────────

    def test_check_denied_clean_command(self) -> None:
        from vaig.tools.gke_tools import _check_denied

        assert _check_denied("ls -la /tmp") is None  # clean command

    def test_check_denied_detects_patterns(self) -> None:
        from vaig.tools.gke_tools import _check_denied

        assert _check_denied("ls; rm -rf /") is not None
        assert _check_denied("cat `id`") is not None
        assert _check_denied("echo $HOME") is not None


# ── check_rbac tests ─────────────────────────────────────────


class TestCheckRbac:
    """Tests for check_rbac RBAC permission checking tool (Phase 3)."""

    def test_k8s_unavailable(self) -> None:
        from vaig.tools.gke_tools import check_rbac

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.security._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = check_rbac("get", "pods", gke_config=cfg)
            assert result.error is True
            assert "kubernetes" in result.output.lower()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_client_creation_error(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import check_rbac

        cfg = _make_gke_config()
        mock_clients.return_value = ToolResult(output="Failed to configure", error=True)

        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True):
            result = check_rbac("get", "pods", gke_config=cfg)
            assert result.error is True
            assert "Failed to configure" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_self_access_allowed(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import check_rbac

        cfg = _make_gke_config()
        api_client = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), MagicMock(), api_client)

        mock_review = MagicMock()
        mock_review.status.allowed = True
        mock_review.status.reason = "RBAC: allowed"
        mock_review.status.denied = False
        mock_review.status.evaluation_error = ""

        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.security.k8s_client") as mock_k8s:
            mock_auth = MagicMock()
            mock_k8s.AuthorizationV1Api.return_value = mock_auth
            mock_auth.create_self_subject_access_review.return_value = mock_review
            result = check_rbac("get", "pods", gke_config=cfg)

        assert result.error is False
        assert "Allowed: YES" in result.output
        assert "current user" in result.output.lower()
        assert "get pods" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_self_access_denied(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import check_rbac

        cfg = _make_gke_config()
        api_client = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), MagicMock(), api_client)

        mock_review = MagicMock()
        mock_review.status.allowed = False
        mock_review.status.reason = "RBAC: no bindings found"
        mock_review.status.denied = True
        mock_review.status.evaluation_error = ""

        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.security.k8s_client") as mock_k8s:
            mock_auth = MagicMock()
            mock_k8s.AuthorizationV1Api.return_value = mock_auth
            mock_auth.create_self_subject_access_review.return_value = mock_review
            result = check_rbac("delete", "pods", gke_config=cfg)

        assert result.error is False
        assert "Allowed: NO" in result.output
        assert "Explicitly Denied: YES" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_service_account_check(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import check_rbac

        cfg = _make_gke_config()
        api_client = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), MagicMock(), api_client)

        mock_review = MagicMock()
        mock_review.status.allowed = True
        mock_review.status.reason = ""
        mock_review.status.denied = False
        mock_review.status.evaluation_error = ""

        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.security.k8s_client") as mock_k8s:
            mock_auth = MagicMock()
            mock_k8s.AuthorizationV1Api.return_value = mock_auth
            mock_auth.create_subject_access_review.return_value = mock_review
            result = check_rbac(
                "list", "deployments", gke_config=cfg,
                namespace="production", service_account="my-sa",
            )

        assert result.error is False
        assert "Allowed: YES" in result.output
        assert "system:serviceaccount:production:my-sa" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_resource_name_shown(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import check_rbac

        cfg = _make_gke_config()
        api_client = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), MagicMock(), api_client)

        mock_review = MagicMock()
        mock_review.status.allowed = True
        mock_review.status.reason = ""
        mock_review.status.denied = False
        mock_review.status.evaluation_error = ""

        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.security.k8s_client") as mock_k8s:
            mock_auth = MagicMock()
            mock_k8s.AuthorizationV1Api.return_value = mock_auth
            mock_auth.create_self_subject_access_review.return_value = mock_review
            result = check_rbac(
                "get", "secrets", gke_config=cfg,
                resource_name="my-secret",
            )

        assert result.error is False
        assert "Resource Name: my-secret" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_resource_alias_normalised(self, mock_clients: MagicMock) -> None:
        """Resource aliases like 'po' should be normalised to 'pods'."""
        from vaig.tools.gke_tools import check_rbac

        cfg = _make_gke_config()
        api_client = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), MagicMock(), api_client)

        mock_review = MagicMock()
        mock_review.status.allowed = True
        mock_review.status.reason = ""
        mock_review.status.denied = False
        mock_review.status.evaluation_error = ""

        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.security.k8s_client") as mock_k8s:
            mock_auth = MagicMock()
            mock_k8s.AuthorizationV1Api.return_value = mock_auth
            mock_auth.create_self_subject_access_review.return_value = mock_review
            result = check_rbac("get", "po", gke_config=cfg)

        assert result.error is False
        assert "pods" in result.output  # normalised from "po"

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_api_permission_denied(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import check_rbac

        cfg = _make_gke_config()
        api_client = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), MagicMock(), api_client)

        exc_class = type("ApiException", (Exception,), {"status": 403, "reason": "Forbidden"})

        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.security.k8s_client") as mock_k8s, \
             patch("vaig.tools.gke.security.k8s_exceptions") as mock_exc:
            mock_auth = MagicMock()
            mock_k8s.AuthorizationV1Api.return_value = mock_auth
            mock_exc.ApiException = exc_class
            mock_auth.create_self_subject_access_review.side_effect = exc_class()
            result = check_rbac("get", "pods", gke_config=cfg)

        assert result.error is True
        assert "permission denied" in result.output.lower()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_evaluation_error_shown(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import check_rbac

        cfg = _make_gke_config()
        api_client = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), MagicMock(), api_client)

        mock_review = MagicMock()
        mock_review.status.allowed = False
        mock_review.status.reason = ""
        mock_review.status.denied = False
        mock_review.status.evaluation_error = "webhook timeout"

        with patch("vaig.tools.gke.security._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.security.k8s_client") as mock_k8s:
            mock_auth = MagicMock()
            mock_k8s.AuthorizationV1Api.return_value = mock_auth
            mock_auth.create_self_subject_access_review.return_value = mock_review
            result = check_rbac("get", "pods", gke_config=cfg)

        assert result.error is False
        assert "Evaluation Error: webhook timeout" in result.output


# ── create_gke_tools factory — tool count verification ───────


class TestCreateGkeToolsPhase3:
    """Verify create_gke_tools now returns 18 tools (16 + discover_service_mesh + discover_network_topology)."""

    def test_tool_count(self) -> None:
        from vaig.tools.gke_tools import create_gke_tools

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            tools = create_gke_tools(cfg)
        assert len(tools) == 22

    def test_exec_command_registered(self) -> None:
        from vaig.tools.gke_tools import create_gke_tools

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            tools = create_gke_tools(cfg)
            names = [t.name for t in tools]
            assert "exec_command" in names

    def test_check_rbac_registered(self) -> None:
        from vaig.tools.gke_tools import create_gke_tools

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            tools = create_gke_tools(cfg)
            names = [t.name for t in tools]
            assert "check_rbac" in names

    def test_get_rollout_history_registered(self) -> None:
        from vaig.tools.gke_tools import create_gke_tools

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            tools = create_gke_tools(cfg)
            names = [t.name for t in tools]
            assert "get_rollout_history" in names


# ── Helpers for get_rollout_history tests ────────────────────


def _mock_replica_set(
    name: str = "my-deploy-abc123",
    revision: str = "1",
    owner_name: str = "my-deploy",
    replicas: int = 0,
    ready_replicas: int = 0,
    image: str = "nginx:1.19",
    creation_timestamp: datetime | None = None,
    change_cause: str = "",
    ports: list | None = None,
    env_vars: list | None = None,
    env_from: list | None = None,
    resources: dict | None = None,
    volume_mounts: list | None = None,
) -> MagicMock:
    """Create a mock ReplicaSet for rollout history tests."""
    rs = MagicMock()
    rs.metadata.name = name
    rs.metadata.creation_timestamp = creation_timestamp or datetime(2025, 1, 1, tzinfo=timezone.utc)

    annotations = {"deployment.kubernetes.io/revision": revision}
    if change_cause:
        annotations["kubernetes.io/change-cause"] = change_cause
    rs.metadata.annotations = annotations

    # Owner reference
    owner = MagicMock()
    owner.kind = "Deployment"
    owner.name = owner_name
    rs.metadata.owner_references = [owner]

    # Status
    rs.status.replicas = replicas
    rs.status.ready_replicas = ready_replicas

    # Container spec
    container = MagicMock()
    container.name = "main"
    container.image = image
    container.ports = ports or []
    container.env = env_vars or []
    container.env_from = env_from or []
    container.volume_mounts = volume_mounts or []

    if resources:
        container.resources.requests = resources.get("requests")
        container.resources.limits = resources.get("limits")
    else:
        container.resources = None

    rs.spec.template.spec.containers = [container]
    return rs


class TestGetRolloutHistory:
    """Tests for get_rollout_history diagnostic tool."""

    def test_k8s_unavailable(self) -> None:
        from vaig.tools.gke_tools import get_rollout_history

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = get_rollout_history("my-deploy", gke_config=cfg)
            assert result.error is True
            assert "kubernetes" in result.output.lower()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_client_creation_error(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_history

        cfg = _make_gke_config()
        mock_clients.return_value = ToolResult(output="Failed to configure", error=True)

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_rollout_history("my-deploy", gke_config=cfg)
            assert result.error is True
            assert "Failed to configure" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_deployment_not_found_404(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_history
        from vaig.tools.gke_tools import k8s_exceptions

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        exc_class = type("ApiException", (Exception,), {"status": 404, "reason": "Not Found"})
        k8s_exceptions.ApiException = exc_class
        apps_v1.read_namespaced_deployment.side_effect = exc_class()

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_rollout_history("missing-deploy", gke_config=cfg, namespace="prod")
            assert result.error is True
            assert "not found" in result.output.lower()
            assert "prod" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_deployment_no_replicasets(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_history

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        dep = MagicMock()
        dep.metadata.name = "web"
        dep.metadata.annotations = {}
        apps_v1.read_namespaced_deployment.return_value = dep

        # No ReplicaSets at all
        rs_list = MagicMock()
        rs_list.items = []
        apps_v1.list_namespaced_replica_set.return_value = rs_list

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_rollout_history("web", gke_config=cfg)
            assert result.error is False
            assert "no revision history" in result.output.lower()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_list_multiple_revisions(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_history

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        dep = MagicMock()
        dep.metadata.name = "web"
        dep.metadata.annotations = {"deployment.kubernetes.io/revision": "3"}
        apps_v1.read_namespaced_deployment.return_value = dep

        rs1 = _mock_replica_set(name="web-rs1", revision="1", owner_name="web",
                                replicas=0, image="nginx:1.19")
        rs2 = _mock_replica_set(name="web-rs2", revision="2", owner_name="web",
                                replicas=0, image="nginx:1.20")
        rs3 = _mock_replica_set(name="web-rs3", revision="3", owner_name="web",
                                replicas=3, ready_replicas=3, image="nginx:1.21")

        rs_list = MagicMock()
        rs_list.items = [rs1, rs2, rs3]
        apps_v1.list_namespaced_replica_set.return_value = rs_list

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_rollout_history("web", gke_config=cfg)

        assert result.error is False
        assert "Rollout History: web" in result.output
        # Newest first
        lines = result.output.split("\n")
        rev_lines = [l for l in lines if l.strip().startswith(("1", "2", "3")) and "nginx" in l]
        assert len(rev_lines) == 3
        # First data line should be revision 3 (newest)
        assert rev_lines[0].strip().startswith("3")
        assert "active" in rev_lines[0]
        # Last should be revision 1
        assert rev_lines[2].strip().startswith("1")
        assert "scaled-down" in rev_lines[2]
        assert "Total revisions: 3" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_list_sorted_newest_first(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_history

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        dep = MagicMock()
        dep.metadata.name = "api"
        dep.metadata.annotations = {"deployment.kubernetes.io/revision": "5"}
        apps_v1.read_namespaced_deployment.return_value = dep

        # Create in shuffled order to test sorting
        rs5 = _mock_replica_set(name="api-rs5", revision="5", owner_name="api",
                                replicas=2, image="api:v5")
        rs2 = _mock_replica_set(name="api-rs2", revision="2", owner_name="api",
                                replicas=0, image="api:v2")
        rs4 = _mock_replica_set(name="api-rs4", revision="4", owner_name="api",
                                replicas=0, image="api:v4")

        rs_list = MagicMock()
        rs_list.items = [rs2, rs5, rs4]  # Deliberately not sorted
        apps_v1.list_namespaced_replica_set.return_value = rs_list

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_rollout_history("api", gke_config=cfg)

        assert result.error is False
        lines = result.output.split("\n")
        rev_lines = [l for l in lines if l.strip() and l.strip()[0].isdigit() and ("api:" in l or "active" in l or "scaled-down" in l)]
        # Should be ordered 5, 4, 2
        revs = [int(l.strip().split()[0]) for l in rev_lines]
        assert revs == [5, 4, 2]

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_specific_revision_detail(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_history

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        dep = MagicMock()
        dep.metadata.name = "web"
        dep.metadata.annotations = {"deployment.kubernetes.io/revision": "2"}
        apps_v1.read_namespaced_deployment.return_value = dep

        # Set up a ReplicaSet with detailed spec
        port = MagicMock()
        port.container_port = 8080
        port.protocol = "TCP"

        env_plain = MagicMock()
        env_plain.name = "APP_ENV"
        env_plain.value_from = None

        env_secret = MagicMock()
        env_secret.name = "DB_PASSWORD"
        env_secret.value_from.config_map_key_ref = None
        env_secret.value_from.secret_key_ref.name = "db-creds"
        env_secret.value_from.secret_key_ref.key = "password"

        mount = MagicMock()
        mount.mount_path = "/etc/config"
        mount.name = "config-vol"
        mount.read_only = True

        rs = _mock_replica_set(
            name="web-rs2", revision="2", owner_name="web",
            replicas=3, ready_replicas=3, image="web:v2",
            change_cause="kubectl set image deployment/web main=web:v2",
            ports=[port],
            env_vars=[env_plain, env_secret],
            resources={"requests": {"cpu": "100m", "memory": "128Mi"},
                       "limits": {"cpu": "500m", "memory": "512Mi"}},
            volume_mounts=[mount],
        )

        rs_list = MagicMock()
        rs_list.items = [rs]
        apps_v1.list_namespaced_replica_set.return_value = rs_list

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_rollout_history("web", gke_config=cfg, revision=2)

        assert result.error is False
        assert "Revision 2 Detail" in result.output
        assert "web:v2" in result.output
        assert "8080/TCP" in result.output
        assert "APP_ENV" in result.output
        assert "DB_PASSWORD" in result.output
        assert "Secret:db-creds/password" in result.output
        # Secret values must NOT be shown
        assert "ref only" in result.output
        assert "Change Cause" in result.output
        assert "kubectl set image" in result.output
        assert "Requests: cpu=100m, memory=128Mi" in result.output
        assert "Limits:   cpu=500m, memory=512Mi" in result.output
        assert "/etc/config from config-vol (ro)" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_specific_revision_not_found(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_history

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        dep = MagicMock()
        dep.metadata.name = "web"
        dep.metadata.annotations = {}
        apps_v1.read_namespaced_deployment.return_value = dep

        rs1 = _mock_replica_set(name="web-rs1", revision="1", owner_name="web",
                                replicas=3, image="nginx:1.19")

        rs_list = MagicMock()
        rs_list.items = [rs1]
        apps_v1.list_namespaced_replica_set.return_value = rs_list

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_rollout_history("web", gke_config=cfg, revision=99)

        assert result.error is True
        assert "Revision 99 not found" in result.output
        assert "1" in result.output  # Shows available revisions

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_api_error_403(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_history
        from vaig.tools.gke_tools import k8s_exceptions

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        exc_class = type("ApiException", (Exception,), {"status": 403, "reason": "Forbidden"})
        k8s_exceptions.ApiException = exc_class
        apps_v1.read_namespaced_deployment.side_effect = exc_class()

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_rollout_history("web", gke_config=cfg)
            assert result.error is True
            assert "Access denied" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_api_error_401(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_history
        from vaig.tools.gke_tools import k8s_exceptions

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        exc_class = type("ApiException", (Exception,), {"status": 401, "reason": "Unauthorized"})
        k8s_exceptions.ApiException = exc_class
        apps_v1.read_namespaced_deployment.side_effect = exc_class()

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_rollout_history("web", gke_config=cfg)
            assert result.error is True
            assert "Authentication failed" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_generic_exception(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_history

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        apps_v1.read_namespaced_deployment.side_effect = RuntimeError("boom")

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_rollout_history("web", gke_config=cfg)
            assert result.error is True
            assert "boom" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_replicasets_not_owned_by_deployment_are_ignored(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_history

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        dep = MagicMock()
        dep.metadata.name = "web"
        dep.metadata.annotations = {"deployment.kubernetes.io/revision": "1"}
        apps_v1.read_namespaced_deployment.return_value = dep

        rs_owned = _mock_replica_set(name="web-rs1", revision="1", owner_name="web",
                                     replicas=3, image="nginx:1.19")
        rs_other = _mock_replica_set(name="api-rs1", revision="1", owner_name="api",
                                     replicas=2, image="api:v1")

        rs_list = MagicMock()
        rs_list.items = [rs_owned, rs_other]
        apps_v1.list_namespaced_replica_set.return_value = rs_list

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_rollout_history("web", gke_config=cfg)

        assert result.error is False
        assert "Total revisions: 1" in result.output
        assert "api" not in result.output.lower().replace("rollout history: web", "")

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_env_from_configmap_and_secret_shown(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_history

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        dep = MagicMock()
        dep.metadata.name = "web"
        dep.metadata.annotations = {}
        apps_v1.read_namespaced_deployment.return_value = dep

        env_from_cm = MagicMock()
        env_from_cm.config_map_ref.name = "app-config"
        env_from_cm.secret_ref = None

        env_from_secret = MagicMock()
        env_from_secret.config_map_ref = None
        env_from_secret.secret_ref.name = "app-secrets"

        rs = _mock_replica_set(
            name="web-rs1", revision="1", owner_name="web",
            replicas=3, image="web:v1",
            env_from=[env_from_cm, env_from_secret],
        )

        rs_list = MagicMock()
        rs_list.items = [rs]
        apps_v1.list_namespaced_replica_set.return_value = rs_list

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = get_rollout_history("web", gke_config=cfg, revision=1)

        assert result.error is False
        assert "ConfigMap: app-config" in result.output
        assert "Secret: app-secrets" in result.output
        assert "ref only" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_uses_default_namespace_from_config(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import get_rollout_history

        cfg = _make_gke_config(default_namespace="staging")
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        dep = MagicMock()
        dep.metadata.name = "web"
        dep.metadata.annotations = {}
        apps_v1.read_namespaced_deployment.return_value = dep

        rs_list = MagicMock()
        rs_list.items = []
        apps_v1.list_namespaced_replica_set.return_value = rs_list

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            # Pass empty namespace to trigger config default
            get_rollout_history("web", gke_config=cfg, namespace="")

        apps_v1.read_namespaced_deployment.assert_called_once_with(name="web", namespace="staging")
        apps_v1.list_namespaced_replica_set.assert_called_once_with(namespace="staging")


# ── nc allowlist tests ─────────────────────────────────────


class TestNcInAllowlist:
    """Verify that 'nc' (netcat) is present in the exec_command allowlist."""

    def test_nc_in_allowlist(self) -> None:
        """nc must be listed in ALLOWED_EXEC_COMMANDS."""
        from vaig.tools.gke_tools import ALLOWED_EXEC_COMMANDS

        assert "nc" in ALLOWED_EXEC_COMMANDS

    def test_nc_command_allowed(self) -> None:
        """nc -zv service 8080 must pass _check_allowed validation."""
        from vaig.tools.gke_tools import _check_allowed

        assert _check_allowed("nc -zv service 8080") is True

    def test_nc_bare_allowed(self) -> None:
        """Bare 'nc' command must also pass."""
        from vaig.tools.gke_tools import _check_allowed

        assert _check_allowed("nc") is True


# ── PDB and ResourceQuota resource map tests ───────────────


class TestPdbResourceMap:
    """Verify PodDisruptionBudget entries in _RESOURCE_API_MAP and aliases."""

    def test_pdb_in_resource_map(self) -> None:
        """poddisruptionbudgets must map to 'policy' API group."""
        from vaig.tools.gke_tools import _RESOURCE_API_MAP

        assert "poddisruptionbudgets" in _RESOURCE_API_MAP
        assert _RESOURCE_API_MAP["poddisruptionbudgets"] == "policy"

    def test_pdb_aliases(self) -> None:
        """All PDB aliases must normalise to 'poddisruptionbudgets'."""
        from vaig.tools.gke_tools import _normalise_resource

        aliases = ("pdb", "pdbs", "poddisruptionbudget", "poddisruptionbudgets")
        for alias in aliases:
            assert _normalise_resource(alias) == "poddisruptionbudgets", (
                f"alias '{alias}' did not resolve to 'poddisruptionbudgets'"
            )

    def test_pdb_case_insensitive(self) -> None:
        """PDB aliases must work regardless of case."""
        from vaig.tools.gke_tools import _normalise_resource

        assert _normalise_resource("PDB") == "poddisruptionbudgets"
        assert _normalise_resource("Pdbs") == "poddisruptionbudgets"


class TestResourceQuotaResourceMap:
    """Verify ResourceQuota entries in _RESOURCE_API_MAP and aliases."""

    def test_resourcequota_in_resource_map(self) -> None:
        """resourcequotas must map to 'core' API group."""
        from vaig.tools.gke_tools import _RESOURCE_API_MAP

        assert "resourcequotas" in _RESOURCE_API_MAP
        assert _RESOURCE_API_MAP["resourcequotas"] == "core"

    def test_resourcequota_aliases(self) -> None:
        """All ResourceQuota aliases must normalise to 'resourcequotas'."""
        from vaig.tools.gke_tools import _normalise_resource

        aliases = ("quota", "quotas", "resourcequota", "resourcequotas")
        for alias in aliases:
            assert _normalise_resource(alias) == "resourcequotas", (
                f"alias '{alias}' did not resolve to 'resourcequotas'"
            )

    def test_resourcequota_case_insensitive(self) -> None:
        """ResourceQuota aliases must work regardless of case."""
        from vaig.tools.gke_tools import _normalise_resource

        assert _normalise_resource("QUOTA") == "resourcequotas"
        assert _normalise_resource("ResourceQuotas") == "resourcequotas"


# ══════════════════════════════════════════════════════════════
# Autopilot detection tests
# ══════════════════════════════════════════════════════════════


class TestDetectAutopilot:
    """Tests for detect_autopilot() function."""

    def test_returns_none_when_missing_project_id(self) -> None:
        from vaig.tools.gke_tools import detect_autopilot

        cfg = _make_gke_config(project_id="", location="us-central1", cluster_name="my-cluster")
        assert detect_autopilot(cfg) is None

    def test_returns_none_when_missing_location(self) -> None:
        from vaig.tools.gke_tools import detect_autopilot

        cfg = _make_gke_config(project_id="proj", location="", cluster_name="my-cluster")
        assert detect_autopilot(cfg) is None

    def test_returns_none_when_missing_cluster_name(self) -> None:
        from vaig.tools.gke_tools import detect_autopilot

        cfg = _make_gke_config(project_id="proj", location="us-central1", cluster_name="")
        assert detect_autopilot(cfg) is None

    def test_returns_true_for_autopilot_cluster(self) -> None:
        from vaig.tools.gke_tools import detect_autopilot

        cfg = _make_gke_config()

        with patch("vaig.tools.gke._clients._query_autopilot_status", return_value=True):
            result = detect_autopilot(cfg)

        assert result is True

    def test_returns_false_for_standard_cluster(self) -> None:
        from vaig.tools.gke_tools import detect_autopilot

        cfg = _make_gke_config()

        with patch("vaig.tools.gke._clients._query_autopilot_status", return_value=False):
            result = detect_autopilot(cfg)

        assert result is False

    def test_caches_result(self) -> None:
        from vaig.tools.gke_tools import _AUTOPILOT_CACHE, detect_autopilot

        cfg = _make_gke_config()
        cache_key = (cfg.project_id, cfg.location, cfg.cluster_name)

        # Pre-populate cache
        _AUTOPILOT_CACHE[cache_key] = True

        # Should return cached value without making any API call
        result = detect_autopilot(cfg)
        assert result is True

    def test_returns_none_on_api_error(self) -> None:
        from vaig.tools.gke_tools import detect_autopilot

        with patch(
            "vaig.tools.gke._clients._query_autopilot_status",
            side_effect=Exception("API unavailable"),
        ):
            cfg = _make_gke_config()
            result = detect_autopilot(cfg)

        assert result is None

    def test_clear_autopilot_cache(self) -> None:
        from vaig.tools.gke_tools import _AUTOPILOT_CACHE, clear_autopilot_cache

        _AUTOPILOT_CACHE[("a", "b", "c")] = True
        assert len(_AUTOPILOT_CACHE) > 0

        clear_autopilot_cache()
        assert len(_AUTOPILOT_CACHE) == 0


class TestAutopilotToolBehavior:
    """Tests for Autopilot-aware tool wrappers in create_gke_tools."""

    def _get_tool(self, tools: list[ToolDef], name: str) -> ToolDef:
        for t in tools:
            if t.name == name:
                return t
        raise AssertionError(f"Tool '{name}' not found")

    @patch("vaig.tools.gke._clients.detect_autopilot", return_value=True)
    def test_kubectl_top_nodes_skipped_on_autopilot(self, _mock_detect: MagicMock) -> None:
        from vaig.tools.gke_tools import create_gke_tools

        cfg = _make_gke_config()
        tools = create_gke_tools(cfg)
        top_tool = self._get_tool(tools, "kubectl_top")

        result = top_tool.execute(resource_type="nodes")
        assert "GKE Autopilot cluster detected" in result.output
        assert "kubectl top nodes is not available" in result.output
        assert result.error is False

    @patch("vaig.tools.gke._clients.detect_autopilot", return_value=True)
    @patch("vaig.tools.gke.kubectl.kubectl_top")
    def test_kubectl_top_pods_works_on_autopilot(self, mock_top: MagicMock, _mock_detect: MagicMock) -> None:
        from vaig.tools.gke_tools import create_gke_tools

        mock_top.return_value = ToolResult(output="pod metrics here")

        cfg = _make_gke_config()
        tools = create_gke_tools(cfg)
        top_tool = self._get_tool(tools, "kubectl_top")

        result = top_tool.execute(resource_type="pods")
        assert result.output == "pod metrics here"

    @patch("vaig.tools.gke._clients.detect_autopilot", return_value=True)
    @patch("vaig.tools.gke.diagnostics.get_node_conditions")
    def test_get_node_conditions_works_on_autopilot(self, mock_nodes: MagicMock, _mock_detect: MagicMock) -> None:
        """On Autopilot, get_node_conditions calls the real function (node reads are allowed)."""
        from vaig.tools.gke_tools import create_gke_tools

        mock_nodes.return_value = ToolResult(output="node conditions here")

        cfg = _make_gke_config()
        tools = create_gke_tools(cfg)
        node_tool = self._get_tool(tools, "get_node_conditions")

        result = node_tool.execute()
        assert result.output == "node conditions here"
        mock_nodes.assert_called_once()

    @patch("vaig.tools.gke._clients.detect_autopilot", return_value=False)
    @patch("vaig.tools.gke.kubectl.kubectl_top")
    def test_kubectl_top_nodes_works_on_standard(self, mock_top: MagicMock, _mock_detect: MagicMock) -> None:
        from vaig.tools.gke_tools import create_gke_tools

        mock_top.return_value = ToolResult(output="node metrics here")

        cfg = _make_gke_config()
        tools = create_gke_tools(cfg)
        top_tool = self._get_tool(tools, "kubectl_top")

        result = top_tool.execute(resource_type="nodes")
        assert result.output == "node metrics here"

    @patch("vaig.tools.gke._clients.detect_autopilot", return_value=False)
    @patch("vaig.tools.gke.diagnostics.get_node_conditions")
    def test_get_node_conditions_works_on_standard(self, mock_nodes: MagicMock, _mock_detect: MagicMock) -> None:
        from vaig.tools.gke_tools import create_gke_tools

        mock_nodes.return_value = ToolResult(output="node conditions here")

        cfg = _make_gke_config()
        tools = create_gke_tools(cfg)
        node_tool = self._get_tool(tools, "get_node_conditions")

        result = node_tool.execute()
        assert result.output == "node conditions here"

    @patch("vaig.tools.gke._clients.detect_autopilot", return_value=True)
    def test_tool_count_unchanged_on_autopilot(self, _mock_detect: MagicMock) -> None:
        """Autopilot detection must NOT change the number of tools."""
        from vaig.tools.gke_tools import create_gke_tools

        cfg = _make_gke_config()
        tools = create_gke_tools(cfg)
        assert len(tools) == 22


# ── Discovery cache helpers ──────────────────────────────────


class TestDiscoveryCache:
    """Tests for the module-level discovery cache helpers."""

    def test_cache_key_builds_colon_delimited_key(self) -> None:
        from vaig.tools.gke_tools import _cache_key_discovery

        assert _cache_key_discovery("a", "b", "c") == "a:b:c"

    def test_cache_key_single_part(self) -> None:
        from vaig.tools.gke_tools import _cache_key_discovery

        assert _cache_key_discovery("solo") == "solo"

    def test_set_and_get_cached_round_trip(self) -> None:
        from vaig.tools.gke_tools import _get_cached, _set_cache

        _set_cache("test-key", "test-value")
        assert _get_cached("test-key") == "test-value"

    def test_get_cached_returns_none_for_missing_key(self) -> None:
        from vaig.tools.gke_tools import _get_cached

        assert _get_cached("nonexistent") is None

    def test_get_cached_returns_none_after_ttl_expires(self) -> None:
        from vaig.tools.gke_tools import _DISCOVERY_CACHE, _get_cached, _set_cache

        _set_cache("expired", "old")
        # Manually backdate the timestamp to exceed TTL
        ts, val = _DISCOVERY_CACHE["expired"]
        _DISCOVERY_CACHE["expired"] = (ts - 9999, val)

        assert _get_cached("expired") is None
        # The expired entry should also be removed
        assert "expired" not in _DISCOVERY_CACHE

    def test_clear_discovery_cache_empties_dict(self) -> None:
        from vaig.tools.gke_tools import _DISCOVERY_CACHE, _set_cache, clear_discovery_cache

        _set_cache("k1", "v1")
        _set_cache("k2", "v2")
        assert len(_DISCOVERY_CACHE) >= 2

        clear_discovery_cache()
        assert len(_DISCOVERY_CACHE) == 0


# ── discover_workloads ───────────────────────────────────────


class TestDiscoverWorkloads:
    """Tests for discover_workloads tool."""

    def test_k8s_unavailable(self) -> None:
        from vaig.tools.gke_tools import discover_workloads

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.discovery._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = discover_workloads(gke_config=cfg)
            assert result.error is True
            assert "kubernetes" in result.output.lower()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_client_creation_error(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import discover_workloads

        cfg = _make_gke_config()
        mock_clients.return_value = ToolResult(output="Failed to configure", error=True)

        with patch("vaig.tools.gke.discovery._K8S_AVAILABLE", True):
            result = discover_workloads(gke_config=cfg)
            assert result.error is True
            assert "Failed to configure" in result.output

    @patch("vaig.tools.gke._clients.detect_autopilot", return_value=False)
    @patch("vaig.tools.gke._resources._list_resource")
    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_no_workloads_found(self, mock_clients: MagicMock, mock_list: MagicMock, _mock_ap: MagicMock) -> None:
        from vaig.tools.gke_tools import discover_workloads

        cfg = _make_gke_config()
        mock_clients.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        empty_result = MagicMock()
        empty_result.items = []
        mock_list.return_value = empty_result

        with patch("vaig.tools.gke.discovery._K8S_AVAILABLE", True):
            result = discover_workloads(gke_config=cfg)

        assert result.error is False
        assert "No workloads found" in result.output

    @patch("vaig.tools.gke._clients.detect_autopilot", return_value=False)
    @patch("vaig.tools.gke._resources._list_resource")
    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_deployments_listed(self, mock_clients: MagicMock, mock_list: MagicMock, _mock_ap: MagicMock) -> None:
        from vaig.tools.gke_tools import discover_workloads

        cfg = _make_gke_config()
        mock_clients.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())

        dep = _mock_deployment(name="web-app", namespace="prod", replicas=3, ready=3)
        dep.status.unavailable_replicas = None
        dep_list = MagicMock()
        dep_list.items = [dep]

        empty = MagicMock()
        empty.items = []

        # deployments returns dep_list; statefulsets, daemonsets return empty
        mock_list.side_effect = [dep_list, empty, empty]

        with patch("vaig.tools.gke.discovery._K8S_AVAILABLE", True):
            result = discover_workloads(gke_config=cfg)

        assert result.error is False
        assert "web-app" in result.output
        assert "deployments" in result.output
        assert "Total: 1 workloads, 0 unhealthy" in result.output

    @patch("vaig.tools.gke._clients.detect_autopilot", return_value=False)
    @patch("vaig.tools.gke._resources._list_resource")
    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_unhealthy_deployment_shows_warn(self, mock_clients: MagicMock, mock_list: MagicMock, _mock_ap: MagicMock) -> None:
        from vaig.tools.gke_tools import discover_workloads

        cfg = _make_gke_config()
        mock_clients.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())

        dep = _mock_deployment(name="broken-app", replicas=3, ready=1)
        dep.status.unavailable_replicas = 2
        dep_list = MagicMock()
        dep_list.items = [dep]

        empty = MagicMock()
        empty.items = []
        mock_list.side_effect = [dep_list, empty, empty]

        with patch("vaig.tools.gke.discovery._K8S_AVAILABLE", True):
            result = discover_workloads(gke_config=cfg)

        assert "WARN" in result.output
        assert "broken-app" in result.output
        assert "1 unhealthy" in result.output

    @patch("vaig.tools.gke._clients.detect_autopilot", return_value=False)
    @patch("vaig.tools.gke._resources._list_resource")
    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_include_jobs_adds_job_resources(self, mock_clients: MagicMock, mock_list: MagicMock, _mock_ap: MagicMock) -> None:
        from vaig.tools.gke_tools import discover_workloads

        cfg = _make_gke_config()
        mock_clients.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())

        empty = MagicMock()
        empty.items = []

        # 5 resource types: deployments, statefulsets, daemonsets, jobs, cronjobs
        mock_list.return_value = empty

        with patch("vaig.tools.gke.discovery._K8S_AVAILABLE", True):
            result = discover_workloads(gke_config=cfg, include_jobs=True)

        # Should have been called 5 times (3 base + 2 job types)
        assert mock_list.call_count == 5

    @patch("vaig.tools.gke._clients.detect_autopilot", return_value=False)
    @patch("vaig.tools.gke._resources._list_resource")
    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_cache_returns_cached_result(self, mock_clients: MagicMock, mock_list: MagicMock, _mock_ap: MagicMock) -> None:
        from vaig.tools.gke_tools import discover_workloads

        cfg = _make_gke_config()
        mock_clients.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())

        empty = MagicMock()
        empty.items = []
        mock_list.return_value = empty

        with patch("vaig.tools.gke.discovery._K8S_AVAILABLE", True):
            # First call populates cache
            result1 = discover_workloads(gke_config=cfg)
            # Second call should use cache — clients should not be called again
            mock_clients.reset_mock()
            result2 = discover_workloads(gke_config=cfg)

        assert result2.output == result1.output
        mock_clients.assert_not_called()

    @patch("vaig.tools.gke._clients.detect_autopilot", return_value=False)
    @patch("vaig.tools.gke._resources._list_resource")
    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_force_refresh_bypasses_cache(self, mock_clients: MagicMock, mock_list: MagicMock, _mock_ap: MagicMock) -> None:
        from vaig.tools.gke_tools import discover_workloads

        cfg = _make_gke_config()
        mock_clients.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())

        empty = MagicMock()
        empty.items = []
        mock_list.return_value = empty

        with patch("vaig.tools.gke.discovery._K8S_AVAILABLE", True):
            discover_workloads(gke_config=cfg)
            mock_clients.reset_mock()
            discover_workloads(gke_config=cfg, force_refresh=True)

        # force_refresh should have called clients again
        mock_clients.assert_called_once()

    @patch("vaig.tools.gke._clients.detect_autopilot", return_value=False)
    @patch("vaig.tools.gke._resources._list_resource")
    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_list_resource_error_is_captured(self, mock_clients: MagicMock, mock_list: MagicMock, _mock_ap: MagicMock) -> None:
        from vaig.tools.gke_tools import discover_workloads

        cfg = _make_gke_config()
        mock_clients.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())

        # _list_resource returns ToolResult with error for first type, empty for rest
        mock_list.side_effect = [
            ToolResult(output="RBAC denied", error=True),
            MagicMock(items=[]),
            MagicMock(items=[]),
        ]

        with patch("vaig.tools.gke.discovery._K8S_AVAILABLE", True):
            result = discover_workloads(gke_config=cfg)

        assert "Partial errors" in result.output
        assert "RBAC denied" in result.output


# ── discover_service_mesh ────────────────────────────────────


class TestDiscoverServiceMesh:
    """Tests for discover_service_mesh tool."""

    def test_k8s_unavailable(self) -> None:
        from vaig.tools.gke_tools import discover_service_mesh

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.discovery._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = discover_service_mesh(gke_config=cfg)
            assert result.error is True

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_client_creation_error(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import discover_service_mesh

        cfg = _make_gke_config()
        mock_clients.return_value = ToolResult(output="Failed to configure", error=True)

        with patch("vaig.tools.gke.discovery._K8S_AVAILABLE", True):
            result = discover_service_mesh(gke_config=cfg)
            assert result.error is True
            assert "Failed to configure" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_no_mesh_detected(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import discover_service_mesh

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), api_client)

        # No mesh namespaces
        ns_list = MagicMock()
        ns_default = MagicMock()
        ns_default.metadata.name = "default"
        ns_list.items = [ns_default]
        core_v1.list_namespace.return_value = ns_list

        # No pods with sidecars
        pod_list = MagicMock()
        pod = MagicMock()
        container = MagicMock()
        container.name = "app"
        pod.spec.containers = [container]
        pod_list.items = [pod]
        core_v1.list_pod_for_all_namespaces.return_value = pod_list

        with patch("vaig.tools.gke.discovery._K8S_AVAILABLE", True), \
             patch("kubernetes.client.ApiextensionsV1Api") as mock_ext:
            mock_ext_instance = MagicMock()
            mock_ext.return_value = mock_ext_instance
            crd_list = MagicMock()
            crd_list.items = []
            mock_ext_instance.list_custom_resource_definition.return_value = crd_list

            result = discover_service_mesh(gke_config=cfg)

        assert result.error is False
        assert "NO MESH DETECTED" in result.output
        assert "Istio" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_istio_detected_via_namespace(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import discover_service_mesh

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), api_client)

        # Namespace list includes istio-system
        ns_list = MagicMock()
        ns_default = MagicMock()
        ns_default.metadata.name = "default"
        ns_istio = MagicMock()
        ns_istio.metadata.name = "istio-system"
        ns_list.items = [ns_default, ns_istio]
        core_v1.list_namespace.return_value = ns_list

        # Control-plane pods in istio-system
        ctrl_pod = MagicMock()
        ctrl_pod.metadata.name = "istiod-abc123"
        ctrl_pod.status.phase = "Running"
        cs = MagicMock()
        cs.ready = True
        cs.image = "docker.io/istio/pilot:1.22.0"
        ctrl_pod.status.container_statuses = [cs]
        ctrl_pod_list = MagicMock()
        ctrl_pod_list.items = [ctrl_pod]
        core_v1.list_namespaced_pod.return_value = ctrl_pod_list

        # Pods with istio-proxy sidecar
        pod_list = MagicMock()
        pod = MagicMock()
        sidecar = MagicMock()
        sidecar.name = "istio-proxy"
        app_ctr = MagicMock()
        app_ctr.name = "app"
        pod.spec.containers = [app_ctr, sidecar]
        pod_list.items = [pod]
        core_v1.list_pod_for_all_namespaces.return_value = pod_list

        with patch("vaig.tools.gke.discovery._K8S_AVAILABLE", True), \
             patch("kubernetes.client.ApiextensionsV1Api") as mock_ext:
            mock_ext_instance = MagicMock()
            mock_ext.return_value = mock_ext_instance
            crd = MagicMock()
            crd.spec.group = "networking.istio.io"
            crd_list = MagicMock()
            crd_list.items = [crd]
            mock_ext_instance.list_custom_resource_definition.return_value = crd_list

            result = discover_service_mesh(gke_config=cfg)

        assert result.error is False
        assert "DETECTED: Istio" in result.output
        assert "istiod-abc123" in result.output
        assert "istio-proxy" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_cache_hit_skips_api_calls(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import _set_cache, discover_service_mesh

        cfg = _make_gke_config()
        _set_cache("mesh:", "cached-mesh-output")

        with patch("vaig.tools.gke.discovery._K8S_AVAILABLE", True):
            result = discover_service_mesh(gke_config=cfg)

        assert result.output == "cached-mesh-output"
        mock_clients.assert_not_called()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_namespace_filter_uses_namespaced_pod_list(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import discover_service_mesh

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), api_client)

        # Empty results
        ns_list = MagicMock()
        ns_list.items = []
        core_v1.list_namespace.return_value = ns_list

        pod_list = MagicMock()
        pod_list.items = []
        core_v1.list_namespaced_pod.return_value = pod_list

        with patch("vaig.tools.gke.discovery._K8S_AVAILABLE", True), \
             patch("kubernetes.client.ApiextensionsV1Api") as mock_ext:
            mock_ext_instance = MagicMock()
            mock_ext.return_value = mock_ext_instance
            crd_list = MagicMock()
            crd_list.items = []
            mock_ext_instance.list_custom_resource_definition.return_value = crd_list

            result = discover_service_mesh(gke_config=cfg, namespace="prod")

        # Should have used namespaced call, not all-namespaces
        core_v1.list_namespaced_pod.assert_called_once_with(namespace="prod")
        core_v1.list_pod_for_all_namespaces.assert_not_called()


# ── discover_network_topology ────────────────────────────────


class TestDiscoverNetworkTopology:
    """Tests for discover_network_topology tool."""

    def test_k8s_unavailable(self) -> None:
        from vaig.tools.gke_tools import discover_network_topology

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.discovery._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = discover_network_topology(gke_config=cfg)
            assert result.error is True

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_client_creation_error(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import discover_network_topology

        cfg = _make_gke_config()
        mock_clients.return_value = ToolResult(output="Failed to configure", error=True)

        with patch("vaig.tools.gke.discovery._K8S_AVAILABLE", True):
            result = discover_network_topology(gke_config=cfg)
            assert result.error is True

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_empty_cluster(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import discover_network_topology

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), api_client)

        # Empty services and endpoints
        svc_list = MagicMock()
        svc_list.items = []
        core_v1.list_service_for_all_namespaces.return_value = svc_list

        ep_list = MagicMock()
        ep_list.items = []
        core_v1.list_endpoints_for_all_namespaces.return_value = ep_list

        with patch("vaig.tools.gke.discovery._K8S_AVAILABLE", True), \
             patch("kubernetes.client.NetworkingV1Api") as mock_net:
            mock_net_instance = MagicMock()
            mock_net.return_value = mock_net_instance
            ing_list = MagicMock()
            ing_list.items = []
            mock_net_instance.list_ingress_for_all_namespaces.return_value = ing_list

            pol_list = MagicMock()
            pol_list.items = []
            mock_net_instance.list_network_policy_for_all_namespaces.return_value = pol_list

            result = discover_network_topology(gke_config=cfg)

        assert result.error is False
        assert "SERVICES (0)" in result.output
        assert "ENDPOINTS (0)" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_services_grouped_by_type(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import discover_network_topology

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), api_client)

        # Create a ClusterIP service
        svc = MagicMock()
        svc.metadata.name = "my-service"
        svc.metadata.namespace = "default"
        svc.spec.type = "ClusterIP"
        port = MagicMock()
        port.port = 80
        port.target_port = 8080
        svc.spec.ports = [port]
        svc.spec.selector = {"app": "web"}
        svc_list = MagicMock()
        svc_list.items = [svc]
        core_v1.list_service_for_all_namespaces.return_value = svc_list

        # Endpoint with ready addresses
        ep = MagicMock()
        ep.metadata.name = "my-service"
        ep.metadata.namespace = "default"
        subset = MagicMock()
        subset.addresses = [MagicMock(), MagicMock()]  # 2 ready
        subset.not_ready_addresses = []
        ep.subsets = [subset]
        ep_list = MagicMock()
        ep_list.items = [ep]
        core_v1.list_endpoints_for_all_namespaces.return_value = ep_list

        with patch("vaig.tools.gke.discovery._K8S_AVAILABLE", True), \
             patch("kubernetes.client.NetworkingV1Api") as mock_net:
            mock_net_instance = MagicMock()
            mock_net.return_value = mock_net_instance
            ing_list = MagicMock()
            ing_list.items = []
            mock_net_instance.list_ingress_for_all_namespaces.return_value = ing_list
            pol_list = MagicMock()
            pol_list.items = []
            mock_net_instance.list_network_policy_for_all_namespaces.return_value = pol_list

            result = discover_network_topology(gke_config=cfg)

        assert result.error is False
        assert "SERVICES (1)" in result.output
        assert "my-service" in result.output
        assert "ClusterIP" in result.output
        assert "2 ready" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_namespace_filter(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import discover_network_topology

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), api_client)

        svc_list = MagicMock()
        svc_list.items = []
        core_v1.list_namespaced_service.return_value = svc_list

        ep_list = MagicMock()
        ep_list.items = []
        core_v1.list_namespaced_endpoints.return_value = ep_list

        with patch("vaig.tools.gke.discovery._K8S_AVAILABLE", True), \
             patch("kubernetes.client.NetworkingV1Api") as mock_net:
            mock_net_instance = MagicMock()
            mock_net.return_value = mock_net_instance
            ing_list = MagicMock()
            ing_list.items = []
            mock_net_instance.list_namespaced_ingress.return_value = ing_list
            pol_list = MagicMock()
            pol_list.items = []
            mock_net_instance.list_namespaced_network_policy.return_value = pol_list

            result = discover_network_topology(gke_config=cfg, namespace="prod")

        core_v1.list_namespaced_service.assert_called_once_with(namespace="prod")
        core_v1.list_namespaced_endpoints.assert_called_once_with(namespace="prod")
        core_v1.list_service_for_all_namespaces.assert_not_called()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_cache_hit_skips_api_calls(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import _set_cache, discover_network_topology

        cfg = _make_gke_config()
        _set_cache("network:", "cached-network-output")

        with patch("vaig.tools.gke.discovery._K8S_AVAILABLE", True):
            result = discover_network_topology(gke_config=cfg)

        assert result.output == "cached-network-output"
        mock_clients.assert_not_called()

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_partial_api_errors_reported(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import discover_network_topology

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), api_client)

        # Services fail
        core_v1.list_service_for_all_namespaces.side_effect = Exception("Forbidden")

        # Endpoints succeed but empty
        ep_list = MagicMock()
        ep_list.items = []
        core_v1.list_endpoints_for_all_namespaces.return_value = ep_list

        with patch("vaig.tools.gke.discovery._K8S_AVAILABLE", True), \
             patch("kubernetes.client.NetworkingV1Api") as mock_net:
            mock_net_instance = MagicMock()
            mock_net.return_value = mock_net_instance
            ing_list = MagicMock()
            ing_list.items = []
            mock_net_instance.list_ingress_for_all_namespaces.return_value = ing_list
            pol_list = MagicMock()
            pol_list.items = []
            mock_net_instance.list_network_policy_for_all_namespaces.return_value = pol_list

            result = discover_network_topology(gke_config=cfg)

        assert "Partial errors" in result.output or "Error" in result.output
        assert "Forbidden" in result.output


# ══════════════════════════════════════════════════════════════
# Causa 2 — Admission registration, CRDs, and gap detection
# ══════════════════════════════════════════════════════════════


class TestAdmissionRegistrationResourceMap:
    """Verify webhook configuration entries in _RESOURCE_API_MAP and aliases."""

    def test_mutatingwebhookconfigurations_in_map(self) -> None:
        from vaig.tools.gke_tools import _RESOURCE_API_MAP

        assert "mutatingwebhookconfigurations" in _RESOURCE_API_MAP
        assert _RESOURCE_API_MAP["mutatingwebhookconfigurations"] == "admissionregistration"

    def test_validatingwebhookconfigurations_in_map(self) -> None:
        from vaig.tools.gke_tools import _RESOURCE_API_MAP

        assert "validatingwebhookconfigurations" in _RESOURCE_API_MAP
        assert _RESOURCE_API_MAP["validatingwebhookconfigurations"] == "admissionregistration"

    def test_mutating_webhook_aliases(self) -> None:
        from vaig.tools.gke_tools import _normalise_resource

        aliases = ("mwc", "mutatingwebhookconfiguration", "mutatingwebhookconfigurations")
        for alias in aliases:
            assert _normalise_resource(alias) == "mutatingwebhookconfigurations", (
                f"alias '{alias}' did not resolve to 'mutatingwebhookconfigurations'"
            )

    def test_validating_webhook_aliases(self) -> None:
        from vaig.tools.gke_tools import _normalise_resource

        aliases = ("vwc", "validatingwebhookconfiguration", "validatingwebhookconfigurations")
        for alias in aliases:
            assert _normalise_resource(alias) == "validatingwebhookconfigurations", (
                f"alias '{alias}' did not resolve to 'validatingwebhookconfigurations'"
            )

    def test_webhook_aliases_case_insensitive(self) -> None:
        from vaig.tools.gke_tools import _normalise_resource

        assert _normalise_resource("MWC") == "mutatingwebhookconfigurations"
        assert _normalise_resource("VWC") == "validatingwebhookconfigurations"


class TestCrdResourceMap:
    """Verify CRD entries in _RESOURCE_API_MAP and aliases."""

    def test_crds_in_map(self) -> None:
        from vaig.tools.gke_tools import _RESOURCE_API_MAP

        assert "customresourcedefinitions" in _RESOURCE_API_MAP
        assert _RESOURCE_API_MAP["customresourcedefinitions"] == "apiextensions"
        assert "crds" in _RESOURCE_API_MAP
        assert _RESOURCE_API_MAP["crds"] == "apiextensions"

    def test_crd_aliases(self) -> None:
        from vaig.tools.gke_tools import _normalise_resource

        aliases = ("crd", "customresourcedefinition", "customresourcedefinitions")
        for alias in aliases:
            assert _normalise_resource(alias) == "customresourcedefinitions", (
                f"alias '{alias}' did not resolve to 'customresourcedefinitions'"
            )

    def test_crd_aliases_case_insensitive(self) -> None:
        from vaig.tools.gke_tools import _normalise_resource

        assert _normalise_resource("CRD") == "customresourcedefinitions"
        # "CRDs" lowercases to "crds" which is a direct key in _RESOURCE_API_MAP
        assert _normalise_resource("CRDs") == "crds"


class TestClusterScopedResources:
    """Verify that webhook configs and CRDs are cluster-scoped."""

    def test_webhook_resources_are_cluster_scoped(self) -> None:
        from vaig.tools.gke_tools import _CLUSTER_SCOPED_RESOURCES

        assert "mutatingwebhookconfigurations" in _CLUSTER_SCOPED_RESOURCES
        assert "validatingwebhookconfigurations" in _CLUSTER_SCOPED_RESOURCES

    def test_crd_resources_are_cluster_scoped(self) -> None:
        from vaig.tools.gke_tools import _CLUSTER_SCOPED_RESOURCES

        assert "customresourcedefinitions" in _CLUSTER_SCOPED_RESOURCES
        assert "crds" in _CLUSTER_SCOPED_RESOURCES

    def test_existing_cluster_scoped_preserved(self) -> None:
        from vaig.tools.gke_tools import _CLUSTER_SCOPED_RESOURCES

        assert "nodes" in _CLUSTER_SCOPED_RESOURCES
        assert "namespaces" in _CLUSTER_SCOPED_RESOURCES
        assert "pv" in _CLUSTER_SCOPED_RESOURCES
        assert "persistentvolumes" in _CLUSTER_SCOPED_RESOURCES


class TestListMutatingWebhookConfigurations:
    """Tests for listing MutatingWebhookConfigurations."""

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_list_mutating_webhook_configurations(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_get

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (core_v1, apps_v1, custom_api, api_client)

        mock_item = MagicMock()
        mock_item.metadata.name = "my-webhook"
        mock_item.metadata.namespace = None
        mock_item.metadata.creation_timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        mock_item.webhooks = [MagicMock(name="wh1")]

        mock_list = MagicMock()
        mock_list.items = [mock_item]

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("kubernetes.client.AdmissionregistrationV1Api") as mock_admission:
            mock_admission.return_value.list_mutating_webhook_configuration.return_value = mock_list
            result = kubectl_get("mutatingwebhookconfigurations", gke_config=cfg)

        assert result.error is not True
        assert "my-webhook" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_list_via_mwc_alias(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_get

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (core_v1, apps_v1, custom_api, api_client)

        mock_list = MagicMock()
        mock_list.items = []

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("kubernetes.client.AdmissionregistrationV1Api") as mock_admission:
            mock_admission.return_value.list_mutating_webhook_configuration.return_value = mock_list
            result = kubectl_get("mwc", gke_config=cfg)

        assert result.error is not True
        # Should call the same API as "mutatingwebhookconfigurations"
        mock_admission.return_value.list_mutating_webhook_configuration.assert_called_once()


class TestListValidatingWebhookConfigurations:
    """Tests for listing ValidatingWebhookConfigurations."""

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_list_validating_webhook_configurations(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_get

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (core_v1, apps_v1, custom_api, api_client)

        mock_item = MagicMock()
        mock_item.metadata.name = "validation-hook"
        mock_item.metadata.namespace = None
        mock_item.metadata.creation_timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        mock_item.webhooks = []

        mock_list = MagicMock()
        mock_list.items = [mock_item]

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("kubernetes.client.AdmissionregistrationV1Api") as mock_admission:
            mock_admission.return_value.list_validating_webhook_configuration.return_value = mock_list
            result = kubectl_get("validatingwebhookconfigurations", gke_config=cfg)

        assert result.error is not True
        assert "validation-hook" in result.output


class TestListCustomResourceDefinitions:
    """Tests for listing CustomResourceDefinitions."""

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_list_crds(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_get

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (core_v1, apps_v1, custom_api, api_client)

        mock_item = MagicMock()
        mock_item.metadata.name = "certificates.cert-manager.io"
        mock_item.metadata.namespace = None
        mock_item.metadata.creation_timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        mock_item.spec.group = "cert-manager.io"
        mock_item.spec.scope = "Namespaced"

        mock_list = MagicMock()
        mock_list.items = [mock_item]

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("kubernetes.client.ApiextensionsV1Api") as mock_ext:
            mock_ext.return_value.list_custom_resource_definition.return_value = mock_list
            result = kubectl_get("crds", gke_config=cfg)

        assert result.error is not True
        assert "certificates.cert-manager.io" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_list_via_crd_alias(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_get

        cfg = _make_gke_config()
        mock_clients.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())

        mock_list = MagicMock()
        mock_list.items = []

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("kubernetes.client.ApiextensionsV1Api") as mock_ext:
            mock_ext.return_value.list_custom_resource_definition.return_value = mock_list
            result = kubectl_get("crd", gke_config=cfg)

        assert result.error is not True
        mock_ext.return_value.list_custom_resource_definition.assert_called_once()


class TestDescribeWebhookConfigurations:
    """Tests for describing webhook configurations."""

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_describe_mutating_webhook(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_describe

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (core_v1, apps_v1, MagicMock(), api_client)

        mock_obj = MagicMock()
        mock_obj.metadata.name = "my-webhook"
        mock_obj.metadata.namespace = None
        mock_obj.metadata.labels = {}
        mock_obj.metadata.annotations = {}
        mock_obj.metadata.creation_timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        mock_obj.spec = None
        mock_obj.status = None

        # Mock events
        mock_events = MagicMock()
        mock_events.items = []

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("kubernetes.client.AdmissionregistrationV1Api") as mock_admission, \
             patch("kubernetes.client.CoreV1Api") as mock_events_api:
            mock_admission.return_value.read_mutating_webhook_configuration.return_value = mock_obj
            mock_events_api.return_value.list_event_for_all_namespaces.return_value = mock_events
            result = kubectl_describe("mutatingwebhookconfigurations", "my-webhook", gke_config=cfg)

        assert result.error is not True
        assert "my-webhook" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_describe_validating_webhook(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_describe

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (core_v1, apps_v1, MagicMock(), api_client)

        mock_obj = MagicMock()
        mock_obj.metadata.name = "val-hook"
        mock_obj.metadata.namespace = None
        mock_obj.metadata.labels = {}
        mock_obj.metadata.annotations = {}
        mock_obj.metadata.creation_timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        mock_obj.spec = None
        mock_obj.status = None

        mock_events = MagicMock()
        mock_events.items = []

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("kubernetes.client.AdmissionregistrationV1Api") as mock_admission, \
             patch("kubernetes.client.CoreV1Api") as mock_events_api:
            mock_admission.return_value.read_validating_webhook_configuration.return_value = mock_obj
            mock_events_api.return_value.list_event_for_all_namespaces.return_value = mock_events
            result = kubectl_describe("validatingwebhookconfigurations", "val-hook", gke_config=cfg)

        assert result.error is not True
        assert "val-hook" in result.output


class TestDescribeCustomResourceDefinition:
    """Tests for describing CRDs."""

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_describe_crd(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_describe

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        api_client = MagicMock()
        mock_clients.return_value = (core_v1, apps_v1, MagicMock(), api_client)

        mock_obj = MagicMock()
        mock_obj.metadata.name = "certificates.cert-manager.io"
        mock_obj.metadata.namespace = None
        mock_obj.metadata.labels = {"app": "cert-manager"}
        mock_obj.metadata.annotations = {}
        mock_obj.metadata.creation_timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        mock_obj.spec = None
        mock_obj.status = None

        mock_events = MagicMock()
        mock_events.items = []

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("kubernetes.client.ApiextensionsV1Api") as mock_ext, \
             patch("kubernetes.client.CoreV1Api") as mock_events_api:
            mock_ext.return_value.read_custom_resource_definition.return_value = mock_obj
            mock_events_api.return_value.list_event_for_all_namespaces.return_value = mock_events
            result = kubectl_describe("customresourcedefinitions", "certificates.cert-manager.io", gke_config=cfg)

        assert result.error is not True
        assert "certificates.cert-manager.io" in result.output


class TestWebhookFormatter:
    """Tests for the webhook configuration formatter."""

    def test_format_webhook_config_basic(self) -> None:
        from vaig.tools.gke_tools import _format_webhook_config

        item = MagicMock()
        item.metadata.name = "istio-sidecar-injector"
        webhook = MagicMock()
        webhook.name = "sidecar-injector.istio.io"
        webhook.namespace_selector = MagicMock()
        webhook.object_selector = None
        rule = MagicMock()
        rule.resources = ["pods"]
        rule.operations = ["CREATE"]
        webhook.rules = [rule]
        webhook.failure_policy = "Fail"
        item.webhooks = [webhook]

        result = _format_webhook_config(item)

        assert "Name: istio-sidecar-injector" in result
        assert "Webhook: sidecar-injector.istio.io" in result
        assert "Rules: CREATE on pods" in result
        assert "FailurePolicy: Fail" in result

    def test_format_webhook_config_no_webhooks(self) -> None:
        from vaig.tools.gke_tools import _format_webhook_config

        item = MagicMock()
        item.metadata.name = "empty-config"
        item.webhooks = []

        result = _format_webhook_config(item)

        assert "Name: empty-config" in result

    def test_format_webhook_config_multiple_rules(self) -> None:
        from vaig.tools.gke_tools import _format_webhook_config

        item = MagicMock()
        item.metadata.name = "multi-rule"
        webhook = MagicMock()
        webhook.name = "wh1"
        webhook.namespace_selector = None
        webhook.object_selector = None
        webhook.failure_policy = "Ignore"
        rule1 = MagicMock()
        rule1.resources = ["pods", "deployments"]
        rule1.operations = ["CREATE", "UPDATE"]
        rule2 = MagicMock()
        rule2.resources = ["services"]
        rule2.operations = ["DELETE"]
        webhook.rules = [rule1, rule2]
        item.webhooks = [webhook]

        result = _format_webhook_config(item)

        assert "CREATE, UPDATE on pods, deployments" in result
        assert "DELETE on services" in result

    def test_format_webhooks_table(self) -> None:
        from vaig.tools.gke_tools import _format_webhooks_table

        item = MagicMock()
        item.metadata.name = "my-webhook-config"
        item.metadata.creation_timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        item.webhooks = [MagicMock(), MagicMock()]

        result = _format_webhooks_table([item])

        assert "my-webhook-config" in result
        assert "WEBHOOKS" in result
        assert "2" in result

    def test_format_webhooks_table_empty(self) -> None:
        from vaig.tools.gke_tools import _format_webhooks_table

        result = _format_webhooks_table([])
        assert "No resources found." in result


class TestCrdsFormatter:
    """Tests for the CRD formatter."""

    def test_format_crds_table(self) -> None:
        from vaig.tools.gke_tools import _format_crds_table

        item = MagicMock()
        item.metadata.name = "certificates.cert-manager.io"
        item.metadata.creation_timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        item.spec.group = "cert-manager.io"
        item.spec.scope = "Namespaced"

        result = _format_crds_table([item])

        assert "certificates.cert-manager.io" in result
        assert "CREATED AT" in result

    def test_format_crds_table_wide(self) -> None:
        from vaig.tools.gke_tools import _format_crds_table

        item = MagicMock()
        item.metadata.name = "certificates.cert-manager.io"
        item.metadata.creation_timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        item.spec.group = "cert-manager.io"
        item.spec.scope = "Namespaced"

        result = _format_crds_table([item], wide=True)

        assert "cert-manager.io" in result
        assert "Namespaced" in result
        assert "GROUP" in result
        assert "SCOPE" in result

    def test_format_crds_table_empty(self) -> None:
        from vaig.tools.gke_tools import _format_crds_table

        result = _format_crds_table([])
        assert "No resources found." in result


class TestGapDetection:
    """Tests for gap detection vs hallucination error messages."""

    def test_known_k8s_resource_gives_informative_message(self) -> None:
        """Real K8s resources not yet supported should get a specific message."""
        from vaig.tools.gke_tools import kubectl_get

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = kubectl_get("clusterroles", gke_config=cfg)

        assert result.error is True
        assert "valid Kubernetes resource" in result.output
        assert "not yet supported" in result.output
        assert "Unsupported resource type" not in result.output

    def test_hallucinated_resource_gives_standard_error(self) -> None:
        """Completely invented resource types get the standard error."""
        from vaig.tools.gke_tools import kubectl_get

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = kubectl_get("foobarwidgets", gke_config=cfg)

        assert result.error is True
        assert "Unsupported resource type" in result.output
        assert "valid Kubernetes resource" not in result.output

    def test_known_k8s_resource_describe_gives_informative_message(self) -> None:
        """Real K8s resources not yet supported should get a specific message in describe too."""
        from vaig.tools.gke_tools import kubectl_describe

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = kubectl_describe("storageclasses", "fast-ssd", gke_config=cfg)

        assert result.error is True
        assert "valid Kubernetes resource" in result.output
        assert "not yet supported" in result.output

    def test_hallucinated_resource_describe_gives_standard_error(self) -> None:
        """Invented resource types get the standard error in describe too."""
        from vaig.tools.gke_tools import kubectl_describe

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = kubectl_describe("unicornpods", "my-unicorn", gke_config=cfg)

        assert result.error is True
        assert "Unsupported resource type" in result.output

    def test_all_known_k8s_resources_produce_gap_message(self) -> None:
        """Every resource in _KNOWN_K8S_RESOURCES should trigger the gap message."""
        from vaig.tools.gke_tools import _KNOWN_K8S_RESOURCES, kubectl_get

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            for resource in _KNOWN_K8S_RESOURCES:
                result = kubectl_get(resource, gke_config=cfg)
                assert result.error is True
                assert "valid Kubernetes resource" in result.output, (
                    f"resource '{resource}' should produce gap message"
                )

    def test_supported_resources_are_not_in_known_set(self) -> None:
        """No overlap between supported resources and the 'known but unsupported' set."""
        from vaig.tools.gke_tools import _KNOWN_K8S_RESOURCES, _RESOURCE_API_MAP

        overlap = set(_RESOURCE_API_MAP.keys()) & _KNOWN_K8S_RESOURCES
        assert overlap == set(), f"Resources in both maps: {overlap}"

    def test_cluster_scoped_no_namespace_for_webhooks(self) -> None:
        """Webhook list should NOT pass namespace to the API call."""
        from vaig.tools.gke_tools import kubectl_get

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()

        mock_list = MagicMock()
        mock_list.items = []

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._create_k8s_clients") as mock_clients, \
             patch("kubernetes.client.AdmissionregistrationV1Api") as mock_admission:
            mock_clients.return_value = (core_v1, apps_v1, custom_api, api_client)
            mock_admission.return_value.list_mutating_webhook_configuration.return_value = mock_list
            result = kubectl_get("mutatingwebhookconfigurations", gke_config=cfg, namespace="kube-system")

        # Cluster-scoped: no namespace should be passed to the K8s API
        call_kwargs = mock_admission.return_value.list_mutating_webhook_configuration.call_args
        if call_kwargs:
            assert "namespace" not in (call_kwargs.kwargs or {}), (
                "Cluster-scoped resource should not receive a namespace parameter"
            )

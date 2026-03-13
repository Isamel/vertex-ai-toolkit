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
    """Clear the K8s client cache before each test to avoid cross-test pollution."""
    from vaig.tools.gke_tools import clear_k8s_client_cache
    clear_k8s_client_cache()


# ── Helpers ──────────────────────────────────────────────────

def _make_gke_config(**kwargs) -> GKEConfig:
    defaults = {
        "cluster_name": "test-cluster",
        "project_id": "test-project",
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
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke_tools._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = kubectl_get("pods", gke_config=cfg)
            assert result.error is True
            assert "kubernetes" in result.output.lower()

    def test_unsupported_resource_type(self) -> None:
        from vaig.tools.gke_tools import kubectl_get

        cfg = _make_gke_config()
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_get("foobar", gke_config=cfg)
            assert result.error is True
            assert "Unsupported resource type" in result.output

    def test_invalid_output_format(self) -> None:
        from vaig.tools.gke_tools import kubectl_get

        cfg = _make_gke_config()
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_get("pods", gke_config=cfg, output_format="xml")
            assert result.error is True
            assert "Invalid output_format" in result.output

    @patch("vaig.tools.gke_tools._create_k8s_clients")
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

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_get("pods", gke_config=cfg)

        assert result.error is False
        assert "my-pod" in result.output

    @patch("vaig.tools.gke_tools._create_k8s_clients")
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

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_get("pods", gke_config=cfg, name="nonexistent")

        assert result.error is True
        assert "not found" in result.output

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_client_creation_error(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_get

        cfg = _make_gke_config()
        mock_clients.return_value = ToolResult(output="Failed to configure", error=True)

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_get("pods", gke_config=cfg)

        assert result.error is True
        assert "Failed to configure" in result.output

    def test_resource_alias_in_get(self) -> None:
        """Using an alias like 'deploy' should normalise to 'deployments'."""
        from vaig.tools.gke_tools import kubectl_get

        cfg = _make_gke_config()
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke_tools._create_k8s_clients") as mock_clients:
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
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke_tools._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = kubectl_describe("pods", "my-pod", gke_config=cfg)
            assert result.error is True

    def test_unsupported_resource_type(self) -> None:
        from vaig.tools.gke_tools import kubectl_describe

        cfg = _make_gke_config()
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_describe("foobar", "name", gke_config=cfg)
            assert result.error is True
            assert "Unsupported resource type" in result.output

    @patch("vaig.tools.gke_tools._create_k8s_clients")
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

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True), \
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
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke_tools._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = kubectl_logs("my-pod", gke_config=cfg)
            assert result.error is True

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_successful_logs(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_logs

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        core_v1.read_namespaced_pod_log.return_value = "2025-01-01 INFO Starting server\n2025-01-01 INFO Ready"

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_logs("my-pod", gke_config=cfg)

        assert result.error is False
        assert "Starting server" in result.output

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_empty_logs(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_logs

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        core_v1.read_namespaced_pod_log.return_value = ""

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_logs("my-pod", gke_config=cfg)

        assert result.error is False
        assert "no logs available" in result.output.lower()

    def test_invalid_since_format(self) -> None:
        from vaig.tools.gke_tools import kubectl_logs

        cfg = _make_gke_config()
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke_tools._create_k8s_clients") as mock_clients:
            mock_clients.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())

            result = kubectl_logs("my-pod", gke_config=cfg, since="invalid")

        assert result.error is True
        assert "Invalid 'since' format" in result.output

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_tail_lines_capped_by_config(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_logs

        cfg = _make_gke_config(log_limit=50)
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())
        core_v1.read_namespaced_pod_log.return_value = "some logs"

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
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
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke_tools._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = kubectl_top(gke_config=cfg)
            assert result.error is True

    def test_invalid_resource_type(self) -> None:
        from vaig.tools.gke_tools import kubectl_top

        cfg = _make_gke_config()
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_top("services", gke_config=cfg)
            assert result.error is True
            assert "Invalid resource_type" in result.output

    @patch("vaig.tools.gke_tools._create_k8s_clients")
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

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_top("pods", gke_config=cfg)

        assert result.error is False
        assert "my-pod" in result.output
        assert "100m" in result.output

    @patch("vaig.tools.gke_tools._create_k8s_clients")
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

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_top("nodes", gke_config=cfg)

        assert result.error is False
        assert "node-1" in result.output

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_no_metrics_available(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_top

        cfg = _make_gke_config()
        custom_api = MagicMock()
        mock_clients.return_value = (MagicMock(), MagicMock(), custom_api, MagicMock())

        custom_api.list_namespaced_custom_object.return_value = {"items": []}

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_top("pods", gke_config=cfg)

        assert "No metrics data" in result.output or "metrics" in result.output.lower()

    @patch("vaig.tools.gke_tools._create_k8s_clients")
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

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
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

        assert len(tools) == 8
        assert all(isinstance(t, ToolDef) for t in tools)

    def test_tool_names(self) -> None:
        from vaig.tools.gke_tools import create_gke_tools

        cfg = _make_gke_config()
        tools = create_gke_tools(cfg)
        names = {t.name for t in tools}

        assert names == {
            "kubectl_get", "kubectl_describe", "kubectl_logs", "kubectl_top",
            "kubectl_scale", "kubectl_restart", "kubectl_label", "kubectl_annotate",
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

    @patch("vaig.tools.gke_tools.k8s_config")
    @patch("vaig.tools.gke_tools.k8s_client")
    @patch("vaig.tools.gke_tools._extract_proxy_url_from_kubeconfig")
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

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
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

    @patch("vaig.tools.gke_tools.k8s_config")
    @patch("vaig.tools.gke_tools.k8s_client")
    @patch("vaig.tools.gke_tools._extract_proxy_url_from_kubeconfig")
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

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            from vaig.tools.gke_tools import _create_k8s_clients

            _create_k8s_clients(cfg)

        assert config_instance.proxy == "https://override-proxy:9090"

    @patch("vaig.tools.gke_tools.k8s_config")
    @patch("vaig.tools.gke_tools.k8s_client")
    @patch("vaig.tools.gke_tools._extract_proxy_url_from_kubeconfig")
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

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
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
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke_tools._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = kubectl_scale("deployments", "nginx", 3, gke_config=cfg)
            assert result.error is True
            assert "kubernetes" in result.output.lower()

    def test_unsupported_resource_type(self) -> None:
        from vaig.tools.gke_tools import kubectl_scale

        cfg = _make_gke_config()
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_scale("pods", "my-pod", 2, gke_config=cfg)
            assert result.error is True
            assert "Cannot scale" in result.output
            assert "deployments" in result.output

    def test_replicas_too_high(self) -> None:
        from vaig.tools.gke_tools import kubectl_scale

        cfg = _make_gke_config()
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_scale("deployments", "nginx", 100, gke_config=cfg)
            assert result.error is True
            assert "50" in result.output

    def test_replicas_negative(self) -> None:
        from vaig.tools.gke_tools import kubectl_scale

        cfg = _make_gke_config()
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_scale("deployments", "nginx", -1, gke_config=cfg)
            assert result.error is True

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_scale_deployment_success(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_scale

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        deploy_obj = MagicMock()
        deploy_obj.spec.replicas = 2
        apps_v1.read_namespaced_deployment.return_value = deploy_obj

        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_scale("deploy", "nginx", 5, gke_config=cfg)
            assert result.error is not True
            assert "2 -> 5" in result.output
            assert "nginx" in result.output
            apps_v1.patch_namespaced_deployment_scale.assert_called_once_with(
                "nginx", "default", {"spec": {"replicas": 5}},
            )

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_scale_statefulset_success(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_scale

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        sts_obj = MagicMock()
        sts_obj.spec.replicas = 3
        apps_v1.read_namespaced_stateful_set.return_value = sts_obj

        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_scale("sts", "redis", 1, gke_config=cfg, namespace="cache")
            assert result.error is not True
            assert "3 -> 1" in result.output
            apps_v1.patch_namespaced_stateful_set_scale.assert_called_once()

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_scale_replicaset_success(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_scale

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        rs_obj = MagicMock()
        rs_obj.spec.replicas = 1
        apps_v1.read_namespaced_replica_set.return_value = rs_obj

        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_scale("replicasets", "my-rs", 0, gke_config=cfg)
            assert result.error is not True
            assert "1 -> 0" in result.output

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_scale_404(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_scale

        cfg = _make_gke_config()
        apps_v1 = MagicMock()

        from unittest.mock import PropertyMock
        exc = MagicMock()
        type(exc).status = PropertyMock(return_value=404)
        exc_class = type("ApiException", (Exception,), {"status": 404, "reason": "Not Found"})
        real_exc = exc_class()

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke_tools.k8s_exceptions") as mock_k8s_exc:
            mock_k8s_exc.ApiException = exc_class
            apps_v1.read_namespaced_deployment.side_effect = real_exc
            mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())
            result = kubectl_scale("deployments", "ghost", 3, gke_config=cfg)
            assert result.error is True
            assert "not found" in result.output

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_scale_403(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_scale

        cfg = _make_gke_config()
        apps_v1 = MagicMock()

        exc_class = type("ApiException", (Exception,), {"status": 403, "reason": "Forbidden"})
        real_exc = exc_class()

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke_tools.k8s_exceptions") as mock_k8s_exc:
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
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_scale("svc", "my-svc", 2, gke_config=cfg)
            assert result.error is True
            assert "Cannot scale" in result.output

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_scale_to_zero(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_scale

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        deploy_obj = MagicMock()
        deploy_obj.spec.replicas = 3
        apps_v1.read_namespaced_deployment.return_value = deploy_obj

        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_scale("deployments", "nginx", 0, gke_config=cfg)
            assert result.error is not True
            assert "3 -> 0" in result.output


# ── kubectl_restart ──────────────────────────────────────────


class TestKubectlRestart:
    """Tests for kubectl_restart write operation."""

    def test_k8s_unavailable(self) -> None:
        from vaig.tools.gke_tools import kubectl_restart

        cfg = _make_gke_config()
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke_tools._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = kubectl_restart("deployments", "nginx", gke_config=cfg)
            assert result.error is True

    def test_unsupported_resource_type(self) -> None:
        from vaig.tools.gke_tools import kubectl_restart

        cfg = _make_gke_config()
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_restart("pods", "my-pod", gke_config=cfg)
            assert result.error is True
            assert "Cannot restart" in result.output
            assert "daemonsets" in result.output

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_restart_deployment_success(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_restart

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
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

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_restart_statefulset_success(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_restart

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_restart("sts", "redis", gke_config=cfg, namespace="cache")
            assert result.error is not True
            apps_v1.patch_namespaced_stateful_set.assert_called_once()

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_restart_daemonset_success(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_restart

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_restart("ds", "fluentd", gke_config=cfg)
            assert result.error is not True
            assert "fluentd" in result.output
            apps_v1.patch_namespaced_daemon_set.assert_called_once()

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_restart_404(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_restart

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        exc_class = type("ApiException", (Exception,), {"status": 404, "reason": "Not Found"})

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke_tools.k8s_exceptions") as mock_k8s_exc:
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
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_restart("replicasets", "my-rs", gke_config=cfg)
            assert result.error is True
            assert "Cannot restart" in result.output


# ── kubectl_label ────────────────────────────────────────────


class TestKubectlLabel:
    """Tests for kubectl_label write operation."""

    def test_k8s_unavailable(self) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke_tools._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = kubectl_label("pods", "my-pod", "env=prod", gke_config=cfg)
            assert result.error is True

    def test_unsupported_resource_type(self) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_label("cronjobs", "my-cron", "env=prod", gke_config=cfg)
            assert result.error is True
            assert "Cannot label" in result.output

    def test_invalid_label_format(self) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_label("pods", "my-pod", "invalid_no_equals", gke_config=cfg)
            assert result.error is True
            assert "Invalid label format" in result.output

    def test_empty_labels(self) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_label("pods", "my-pod", "", gke_config=cfg)
            assert result.error is True
            assert "No labels" in result.output

    def test_system_label_blocked(self) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_label("pods", "my-pod", "kubernetes.io/arch=amd64", gke_config=cfg)
            assert result.error is True
            assert "system label" in result.output.lower()

    def test_k8s_io_system_label_blocked(self) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_label("pods", "my-pod", "k8s.io/component=api", gke_config=cfg)
            assert result.error is True
            assert "system label" in result.output.lower()

    def test_invalid_label_key_format(self) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_label("pods", "my-pod", "-bad-key=val", gke_config=cfg)
            assert result.error is True
            assert "Invalid label key" in result.output

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_label_pod_success(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_label("pods", "my-pod", "env=prod,tier=frontend", gke_config=cfg)
            assert result.error is not True
            assert "Labels updated" in result.output
            assert "env=prod" in result.output
            core_v1.patch_namespaced_pod.assert_called_once()
            patch_body = core_v1.patch_namespaced_pod.call_args[0][2]
            assert patch_body["metadata"]["labels"]["env"] == "prod"
            assert patch_body["metadata"]["labels"]["tier"] == "frontend"

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_label_remove(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_label("pods", "my-pod", "obsolete-", gke_config=cfg)
            assert result.error is not True
            assert "obsolete-" in result.output
            patch_body = core_v1.patch_namespaced_pod.call_args[0][2]
            assert patch_body["metadata"]["labels"]["obsolete"] is None

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_label_deployment(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_label("deploy", "nginx", "version=v2", gke_config=cfg)
            assert result.error is not True
            apps_v1.patch_namespaced_deployment.assert_called_once()

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_label_service(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_label("svc", "my-svc", "managed-by=helm", gke_config=cfg)
            assert result.error is not True
            core_v1.patch_namespaced_service.assert_called_once()

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_label_namespace(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_label("namespaces", "kube-system", "env=prod", gke_config=cfg)
            assert result.error is not True
            core_v1.patch_namespace.assert_called_once()

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_label_node(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_label("nodes", "worker-1", "role=compute", gke_config=cfg)
            assert result.error is not True
            core_v1.patch_node.assert_called_once()

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_label_configmap(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_label("cm", "my-config", "app=web", gke_config=cfg)
            assert result.error is not True
            core_v1.patch_namespaced_config_map.assert_called_once()

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_label_secret(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_label("secrets", "my-secret", "env=staging", gke_config=cfg)
            assert result.error is not True
            core_v1.patch_namespaced_secret.assert_called_once()

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_label_404(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_label

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        exc_class = type("ApiException", (Exception,), {"status": 404, "reason": "Not Found"})

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke_tools.k8s_exceptions") as mock_k8s_exc:
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
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke_tools._create_k8s_clients") as mock_clients:
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
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke_tools._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = kubectl_annotate("pods", "my-pod", "desc=test", gke_config=cfg)
            assert result.error is True

    def test_unsupported_resource_type(self) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_annotate("cronjobs", "my-cron", "desc=test", gke_config=cfg)
            assert result.error is True
            assert "Cannot annotate" in result.output

    def test_invalid_annotation_format(self) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_annotate("pods", "my-pod", "no_equals_sign", gke_config=cfg)
            assert result.error is True
            assert "Invalid annotation format" in result.output

    def test_empty_annotations(self) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_annotate("pods", "my-pod", "", gke_config=cfg)
            assert result.error is True
            assert "No annotations" in result.output

    def test_system_annotation_blocked(self) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
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
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_annotate(
                "pods", "my-pod",
                "k8s.io/something=test",
                gke_config=cfg,
            )
            assert result.error is True
            assert "system annotation" in result.output.lower()

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_annotate_pod_success(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
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

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_annotate_remove(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_annotate("pods", "my-pod", "old-note-", gke_config=cfg)
            assert result.error is not True
            assert "old-note-" in result.output
            patch_body = core_v1.patch_namespaced_pod.call_args[0][2]
            assert patch_body["metadata"]["annotations"]["old-note"] is None

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_annotate_deployment(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_annotate("deploy", "nginx", "gitsha=abc123", gke_config=cfg)
            assert result.error is not True
            apps_v1.patch_namespaced_deployment.assert_called_once()

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_annotate_statefulset(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_annotate("sts", "redis", "backup=daily", gke_config=cfg)
            assert result.error is not True
            apps_v1.patch_namespaced_stateful_set.assert_called_once()

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_annotate_daemonset(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        apps_v1 = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_v1, MagicMock(), MagicMock())

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_annotate("ds", "fluentd", "log-driver=json", gke_config=cfg)
            assert result.error is not True
            apps_v1.patch_namespaced_daemon_set.assert_called_once()

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_annotate_namespace(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_annotate("namespaces", "prod", "team=platform", gke_config=cfg)
            assert result.error is not True
            core_v1.patch_namespace.assert_called_once()

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_annotate_node(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True):
            result = kubectl_annotate("nodes", "worker-1", "rack=a3", gke_config=cfg)
            assert result.error is not True
            core_v1.patch_node.assert_called_once()

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_annotate_404(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        exc_class = type("ApiException", (Exception,), {"status": 404, "reason": "Not Found"})

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke_tools.k8s_exceptions") as mock_k8s_exc:
            mock_k8s_exc.ApiException = exc_class
            core_v1.patch_namespaced_pod.side_effect = exc_class()
            mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())
            result = kubectl_annotate("pods", "ghost", "note=test", gke_config=cfg)
            assert result.error is True
            assert "not found" in result.output

    @patch("vaig.tools.gke_tools._create_k8s_clients")
    def test_annotate_403(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke_tools import kubectl_annotate

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        exc_class = type("ApiException", (Exception,), {"status": 403, "reason": "Forbidden"})

        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke_tools.k8s_exceptions") as mock_k8s_exc:
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
        with patch("vaig.tools.gke_tools._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke_tools._create_k8s_clients") as mock_clients:
            core_v1 = MagicMock()
            mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())
            result = kubectl_annotate("pods", "my-pod", "myorg.io/team=backend", gke_config=cfg)
            assert result.error is not True

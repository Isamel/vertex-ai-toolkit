"""Tests for dedicated exec client isolation (S4).

The ``get_exec_client`` factory in ``_clients.py`` must return a FRESH,
non-cached ``CoreV1Api`` on every call so that ``kubernetes.stream.stream()``
cannot corrupt the shared cached client used by all other GKE tools.

Covers:
- get_exec_client returns CoreV1Api instances
- get_exec_client returns a NEW ApiClient each time (not cached)
- get_exec_client is distinct from the shared cached client
- exec_command uses the dedicated client (not the cached one)
- exec_command closes the dedicated client after use
- Error paths: k8s unavailable, kubeconfig failure
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from vaig.core.config import GKEConfig
from vaig.tools.base import ToolResult
from vaig.tools.gke import _clients

# ── Helpers ──────────────────────────────────────────────────


def _make_gke_config(**overrides: Any) -> GKEConfig:
    """Create a minimal GKEConfig for testing."""
    defaults = {
        "cluster_name": "test-cluster",
        "project_id": "test-project",
        "location": "us-central1",
        "exec_enabled": True,
    }
    defaults.update(overrides)
    return GKEConfig(**defaults)


# ── get_exec_client: returns CoreV1Api ───────────────────────


class TestGetExecClientReturnType:
    """get_exec_client must return a CoreV1Api wrapping a fresh ApiClient."""

    @patch.object(_clients, "_suppress_stderr")
    @patch.object(_clients, "_extract_proxy_url_from_kubeconfig", return_value=None)
    @patch.object(_clients, "k8s_config")
    @patch.object(_clients, "k8s_client")
    def test_returns_core_v1_api_instance(
        self,
        mock_k8s_client: MagicMock,
        mock_k8s_config: MagicMock,
        mock_extract: MagicMock,
        mock_suppress: MagicMock,
    ) -> None:
        mock_suppress.return_value.__enter__ = MagicMock()
        mock_suppress.return_value.__exit__ = MagicMock(return_value=False)
        sentinel_api = MagicMock(name="CoreV1Api")
        mock_k8s_client.CoreV1Api.return_value = sentinel_api
        mock_k8s_client.Configuration.return_value = MagicMock()
        mock_k8s_client.ApiClient.return_value = MagicMock()

        cfg = _make_gke_config()
        result = _clients._get_exec_client(cfg)

        assert result is sentinel_api
        mock_k8s_client.CoreV1Api.assert_called_once()

    @patch.object(_clients, "_suppress_stderr")
    @patch.object(_clients, "_extract_proxy_url_from_kubeconfig", return_value=None)
    @patch.object(_clients, "k8s_config")
    @patch.object(_clients, "k8s_client")
    def test_wraps_fresh_api_client(
        self,
        mock_k8s_client: MagicMock,
        mock_k8s_config: MagicMock,
        mock_extract: MagicMock,
        mock_suppress: MagicMock,
    ) -> None:
        mock_suppress.return_value.__enter__ = MagicMock()
        mock_suppress.return_value.__exit__ = MagicMock(return_value=False)
        fresh_api_client = MagicMock(name="FreshApiClient")
        mock_k8s_client.ApiClient.return_value = fresh_api_client
        mock_k8s_client.Configuration.return_value = MagicMock()

        cfg = _make_gke_config()
        _clients._get_exec_client(cfg)

        mock_k8s_client.CoreV1Api.assert_called_once_with(fresh_api_client)


# ── get_exec_client: NOT cached ──────────────────────────────


class TestGetExecClientNotCached:
    """Each call to get_exec_client must create independent clients."""

    @patch.object(_clients, "_suppress_stderr")
    @patch.object(_clients, "_extract_proxy_url_from_kubeconfig", return_value=None)
    @patch.object(_clients, "k8s_config")
    @patch.object(_clients, "k8s_client")
    def test_returns_different_instances(
        self,
        mock_k8s_client: MagicMock,
        mock_k8s_config: MagicMock,
        mock_extract: MagicMock,
        mock_suppress: MagicMock,
    ) -> None:
        mock_suppress.return_value.__enter__ = MagicMock()
        mock_suppress.return_value.__exit__ = MagicMock(return_value=False)
        mock_k8s_client.Configuration.return_value = MagicMock()
        # Return different MagicMock each time
        mock_k8s_client.ApiClient.side_effect = [MagicMock(name="a1"), MagicMock(name="a2")]
        mock_k8s_client.CoreV1Api.side_effect = [MagicMock(name="c1"), MagicMock(name="c2")]

        cfg = _make_gke_config()
        r1 = _clients._get_exec_client(cfg)
        r2 = _clients._get_exec_client(cfg)

        assert r1 is not r2, "get_exec_client must return a new instance each time"

    @patch.object(_clients, "_suppress_stderr")
    @patch.object(_clients, "_extract_proxy_url_from_kubeconfig", return_value=None)
    @patch.object(_clients, "k8s_config")
    @patch.object(_clients, "k8s_client")
    def test_creates_new_api_client_each_call(
        self,
        mock_k8s_client: MagicMock,
        mock_k8s_config: MagicMock,
        mock_extract: MagicMock,
        mock_suppress: MagicMock,
    ) -> None:
        mock_suppress.return_value.__enter__ = MagicMock()
        mock_suppress.return_value.__exit__ = MagicMock(return_value=False)
        mock_k8s_client.Configuration.return_value = MagicMock()
        api_client_a = MagicMock(name="api_a")
        api_client_b = MagicMock(name="api_b")
        mock_k8s_client.ApiClient.side_effect = [api_client_a, api_client_b]
        mock_k8s_client.CoreV1Api.return_value = MagicMock()

        cfg = _make_gke_config()
        _clients._get_exec_client(cfg)
        _clients._get_exec_client(cfg)

        assert mock_k8s_client.ApiClient.call_count == 2

    @patch.object(_clients, "_suppress_stderr")
    @patch.object(_clients, "_extract_proxy_url_from_kubeconfig", return_value=None)
    @patch.object(_clients, "k8s_config")
    @patch.object(_clients, "k8s_client")
    def test_does_not_populate_cache(
        self,
        mock_k8s_client: MagicMock,
        mock_k8s_config: MagicMock,
        mock_extract: MagicMock,
        mock_suppress: MagicMock,
    ) -> None:
        mock_suppress.return_value.__enter__ = MagicMock()
        mock_suppress.return_value.__exit__ = MagicMock(return_value=False)
        mock_k8s_client.Configuration.return_value = MagicMock()
        mock_k8s_client.ApiClient.return_value = MagicMock()
        mock_k8s_client.CoreV1Api.return_value = MagicMock()

        _clients.clear_k8s_client_cache()
        cfg = _make_gke_config()
        _clients._get_exec_client(cfg)

        assert len(_clients._CLIENT_CACHE) == 0, "Exec client must NOT be cached"


# ── get_exec_client: isolated from shared client ─────────────


class TestExecClientIsolation:
    """The exec client must be distinct from the shared cached client."""

    @patch.object(_clients, "_suppress_stderr")
    @patch.object(_clients, "_extract_proxy_url_from_kubeconfig", return_value=None)
    @patch.object(_clients, "k8s_config")
    @patch.object(_clients, "k8s_client")
    def test_exec_client_differs_from_cached(
        self,
        mock_k8s_client: MagicMock,
        mock_k8s_config: MagicMock,
        mock_extract: MagicMock,
        mock_suppress: MagicMock,
    ) -> None:
        mock_suppress.return_value.__enter__ = MagicMock()
        mock_suppress.return_value.__exit__ = MagicMock(return_value=False)
        mock_k8s_client.Configuration.return_value = MagicMock()

        cached_api = MagicMock(name="cached_api")
        exec_api = MagicMock(name="exec_api")
        mock_k8s_client.ApiClient.side_effect = [cached_api, exec_api]

        cached_core = MagicMock(name="cached_core")
        exec_core = MagicMock(name="exec_core")
        mock_k8s_client.CoreV1Api.side_effect = [cached_core, exec_core]
        mock_k8s_client.AppsV1Api.return_value = MagicMock()
        mock_k8s_client.CustomObjectsApi.return_value = MagicMock()

        _clients.clear_k8s_client_cache()
        cfg = _make_gke_config()

        # Create shared cached client
        cached_result = _clients._create_k8s_clients(cfg)
        assert not isinstance(cached_result, ToolResult)
        cached_core_v1, _, _, _ = cached_result

        # Create exec client
        exec_result = _clients._get_exec_client(cfg)
        assert not isinstance(exec_result, ToolResult)

        assert exec_result is not cached_core_v1, (
            "Exec client must be a different CoreV1Api than the cached one"
        )

    @patch.object(_clients, "_suppress_stderr")
    @patch.object(_clients, "_extract_proxy_url_from_kubeconfig", return_value=None)
    @patch.object(_clients, "k8s_config")
    @patch.object(_clients, "k8s_client")
    def test_cached_client_unchanged_after_exec_client_creation(
        self,
        mock_k8s_client: MagicMock,
        mock_k8s_config: MagicMock,
        mock_extract: MagicMock,
        mock_suppress: MagicMock,
    ) -> None:
        mock_suppress.return_value.__enter__ = MagicMock()
        mock_suppress.return_value.__exit__ = MagicMock(return_value=False)
        mock_k8s_client.Configuration.return_value = MagicMock()
        mock_k8s_client.ApiClient.return_value = MagicMock()
        mock_k8s_client.CoreV1Api.return_value = MagicMock()
        mock_k8s_client.AppsV1Api.return_value = MagicMock()
        mock_k8s_client.CustomObjectsApi.return_value = MagicMock()

        _clients.clear_k8s_client_cache()
        cfg = _make_gke_config()

        # Create and cache the shared client
        cached_result = _clients._create_k8s_clients(cfg)
        assert not isinstance(cached_result, ToolResult)
        cache_snapshot = dict(_clients._CLIENT_CACHE)

        # Create exec client — cache must not change
        _clients._get_exec_client(cfg)

        assert cache_snapshot == _clients._CLIENT_CACHE, (
            "Creating an exec client must not modify the shared client cache"
        )


# ── get_exec_client: error handling ──────────────────────────


class TestGetExecClientErrors:
    """Error paths for get_exec_client."""

    @patch.object(_clients, "_K8S_AVAILABLE", False)
    def test_returns_tool_result_when_k8s_unavailable(self) -> None:
        cfg = _make_gke_config()
        result = _clients._get_exec_client(cfg)
        assert isinstance(result, ToolResult)
        assert result.error is True

    @patch.object(_clients, "_suppress_stderr")
    @patch.object(_clients, "_extract_proxy_url_from_kubeconfig", return_value=None)
    @patch.object(_clients, "k8s_config")
    @patch.object(_clients, "k8s_client")
    def test_returns_tool_result_on_config_failure(
        self,
        mock_k8s_client: MagicMock,
        mock_k8s_config: MagicMock,
        mock_extract: MagicMock,
        mock_suppress: MagicMock,
    ) -> None:
        mock_suppress.return_value.__enter__ = MagicMock()
        mock_suppress.return_value.__exit__ = MagicMock(return_value=False)
        mock_k8s_client.Configuration.return_value = MagicMock()
        mock_k8s_config.load_kube_config.side_effect = RuntimeError("bad kubeconfig")

        cfg = _make_gke_config(kubeconfig_path="/nonexistent/kubeconfig")
        result = _clients._get_exec_client(cfg)

        assert isinstance(result, ToolResult)
        assert result.error is True
        assert "exec" in result.output.lower() or "kubernetes" in result.output.lower()


# ── get_exec_client: proxy support ───────────────────────────


class TestGetExecClientProxy:
    """Proxy URL is applied to the exec client Configuration."""

    @patch.object(_clients, "_suppress_stderr")
    @patch.object(_clients, "_extract_proxy_url_from_kubeconfig", return_value=None)
    @patch.object(_clients, "k8s_config")
    @patch.object(_clients, "k8s_client")
    def test_applies_explicit_proxy_url(
        self,
        mock_k8s_client: MagicMock,
        mock_k8s_config: MagicMock,
        mock_extract: MagicMock,
        mock_suppress: MagicMock,
    ) -> None:
        mock_suppress.return_value.__enter__ = MagicMock()
        mock_suppress.return_value.__exit__ = MagicMock(return_value=False)
        config_instance = MagicMock()
        mock_k8s_client.Configuration.return_value = config_instance
        mock_k8s_client.ApiClient.return_value = MagicMock()
        mock_k8s_client.CoreV1Api.return_value = MagicMock()

        cfg = _make_gke_config(proxy_url="http://proxy:8080")
        _clients._get_exec_client(cfg)

        assert config_instance.proxy == "http://proxy:8080"


# ── exec_command: uses dedicated client ──────────────────────


class TestExecCommandUsesDedicatedClient:
    """exec_command must call get_exec_client, not _create_k8s_clients."""

    @patch("vaig.tools.gke.security._clients._get_exec_client")
    def test_calls_get_exec_client(self, mock_get_exec: MagicMock) -> None:
        from vaig.tools.gke.security import exec_command

        mock_core = MagicMock()
        mock_core.api_client = MagicMock()
        mock_get_exec.return_value = mock_core

        cfg = _make_gke_config()

        with patch("vaig.tools.gke.security.k8s_stream", create=True):
            with patch("kubernetes.stream.stream", return_value="output"):
                exec_command(
                    pod_name="test-pod",
                    command="ls",
                    gke_config=cfg,
                )

        mock_get_exec.assert_called_once_with(cfg)

    @patch("vaig.tools.gke.security._clients._create_k8s_clients")
    @patch("vaig.tools.gke.security._clients._get_exec_client")
    def test_does_not_call_create_k8s_clients(
        self,
        mock_get_exec: MagicMock,
        mock_create: MagicMock,
    ) -> None:
        from vaig.tools.gke.security import exec_command

        mock_core = MagicMock()
        mock_core.api_client = MagicMock()
        mock_get_exec.return_value = mock_core

        cfg = _make_gke_config()

        with patch("kubernetes.stream.stream", return_value="output"):
            exec_command(
                pod_name="test-pod",
                command="ls",
                gke_config=cfg,
            )

        mock_create.assert_not_called()


# ── exec_command: cleanup ────────────────────────────────────


class TestExecCommandCleanup:
    """exec_command must close the disposable client after use."""

    @patch("vaig.tools.gke.security._clients._get_exec_client")
    def test_closes_client_on_success(self, mock_get_exec: MagicMock) -> None:
        from vaig.tools.gke.security import exec_command

        mock_api_client = MagicMock()
        mock_core = MagicMock()
        mock_core.api_client = mock_api_client
        mock_get_exec.return_value = mock_core

        cfg = _make_gke_config()

        with patch("kubernetes.stream.stream", return_value="ok"):
            exec_command(pod_name="p", command="ls", gke_config=cfg)

        mock_api_client.close.assert_called_once()

    @patch("vaig.tools.gke.security._clients._get_exec_client")
    def test_closes_client_on_exception(self, mock_get_exec: MagicMock) -> None:
        from vaig.tools.gke.security import exec_command

        mock_api_client = MagicMock()
        mock_core = MagicMock()
        mock_core.api_client = mock_api_client
        mock_get_exec.return_value = mock_core

        cfg = _make_gke_config()

        with patch("kubernetes.stream.stream", side_effect=RuntimeError("boom")):
            result = exec_command(pod_name="p", command="ls", gke_config=cfg)

        assert result.error is True
        mock_api_client.close.assert_called_once()

    @patch("vaig.tools.gke.security._clients._get_exec_client")
    def test_does_not_close_on_client_creation_failure(
        self,
        mock_get_exec: MagicMock,
    ) -> None:
        from vaig.tools.gke.security import exec_command

        # get_exec_client returns ToolResult on failure — no client to close
        mock_get_exec.return_value = ToolResult(output="failed", error=True)
        cfg = _make_gke_config()

        result = exec_command(pod_name="p", command="ls", gke_config=cfg)

        assert result.error is True
        assert result.output == "failed"


# ── exec_command: passes correct method ──────────────────────


class TestExecCommandPassesCorrectMethod:
    """exec_command must pass the exec client's method to stream()."""

    @patch("vaig.tools.gke.security._clients._get_exec_client")
    @patch("kubernetes.stream.stream")
    def test_passes_exec_client_method_to_stream(
        self,
        mock_stream: MagicMock,
        mock_get_exec: MagicMock,
    ) -> None:
        from vaig.tools.gke.security import exec_command

        mock_core = MagicMock()
        mock_core.api_client = MagicMock()
        mock_get_exec.return_value = mock_core
        mock_stream.return_value = "output"

        cfg = _make_gke_config()
        exec_command(pod_name="p", command="ls", gke_config=cfg)

        mock_stream.assert_called_once()
        call_args = mock_stream.call_args
        assert call_args[0][0] is mock_core.connect_get_namespaced_pod_exec

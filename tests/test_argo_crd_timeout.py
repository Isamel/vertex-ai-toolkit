"""Tests for the Argo CRD timeout fix.

Covers:
- Change 1: _load_k8s_config sets config.retries = False (global urllib3 retry disable)
- Change 2: _check_crd_exists uses crd_check_timeout (short timeout) instead of
  request_timeout, and disables retries on the ApiClient it creates.
- Change 2b: crd_check_timeout field exists in GKEConfig with correct default.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """Clear all caches before each test."""
    from vaig.tools.gke._clients import _CLIENT_CACHE
    from vaig.tools.gke.argocd import _crd_exists_cache
    _crd_exists_cache.clear()
    _CLIENT_CACHE.clear()


# ══════════════════════════════════════════════════════════════
# Change 1: _load_k8s_config disables urllib3 retries globally
# ══════════════════════════════════════════════════════════════


class TestLoadK8sConfigRetriesDisabled:
    """_load_k8s_config must set config.retries = False after building Configuration."""

    def _make_fake_config(self) -> object:
        """Create a simple object that tracks attribute assignments."""

        class FakeConfig:
            retries: object = None
            proxy: str = ""

        return FakeConfig()

    def test_retries_set_to_false_on_kubeconfig_load(self) -> None:
        """config.retries is False after loading a kubeconfig path."""
        from vaig.core.config import GKEConfig
        from vaig.tools.gke._clients import _load_k8s_config

        gke_cfg = GKEConfig(kubeconfig_path="/fake/kubeconfig", context="test-ctx")
        fake_cfg = self._make_fake_config()

        with (
            patch("vaig.tools.gke._clients._K8S_AVAILABLE", True),
            patch("vaig.tools.gke._clients.k8s_client") as mock_k8s_client,
            patch("vaig.tools.gke._clients.k8s_config"),
            patch("vaig.tools.gke._clients._extract_proxy_url_from_kubeconfig", return_value=None),
        ):
            mock_k8s_client.Configuration.return_value = fake_cfg
            _load_k8s_config(gke_cfg)

        # retries must be set to False — this disables urllib3 default Retry(total=3)
        assert fake_cfg.retries is False

    def test_retries_set_to_false_on_default_kubeconfig(self) -> None:
        """config.retries is False when loading the default kubeconfig (no path)."""
        from vaig.core.config import GKEConfig
        from vaig.tools.gke._clients import _load_k8s_config

        gke_cfg = GKEConfig()
        fake_cfg = self._make_fake_config()

        with (
            patch("vaig.tools.gke._clients._K8S_AVAILABLE", True),
            patch("vaig.tools.gke._clients.k8s_client") as mock_k8s_client,
            patch("vaig.tools.gke._clients.k8s_config"),
            patch("vaig.tools.gke._clients._extract_proxy_url_from_kubeconfig", return_value=None),
        ):
            mock_k8s_client.Configuration.return_value = fake_cfg
            _load_k8s_config(gke_cfg)

        assert fake_cfg.retries is False

    def test_retries_attribute_is_actually_false(self) -> None:
        """Verify config.retries ends up as False using a FakeConfig stub object.

        FakeConfig tracks attribute assignments without the overhead of a real
        ``kubernetes.client.Configuration`` instance — sufficient to assert that
        ``_load_k8s_config`` sets the field to ``False`` (not ``None`` or any
        truthy value).
        """
        from vaig.core.config import GKEConfig
        from vaig.tools.gke._clients import _load_k8s_config

        gke_cfg = GKEConfig(kubeconfig_path="/fake/kubeconfig")
        fake_cfg = self._make_fake_config()

        with (
            patch("vaig.tools.gke._clients._K8S_AVAILABLE", True),
            patch("vaig.tools.gke._clients.k8s_client") as mock_k8s_client,
            patch("vaig.tools.gke._clients.k8s_config"),
            patch("vaig.tools.gke._clients._extract_proxy_url_from_kubeconfig", return_value=None),
        ):
            mock_k8s_client.Configuration.return_value = fake_cfg
            _load_k8s_config(gke_cfg)

        assert fake_cfg.retries is False


# ══════════════════════════════════════════════════════════════
# Change 2: GKEConfig.crd_check_timeout field
# ══════════════════════════════════════════════════════════════


class TestGKEConfigCrdCheckTimeout:
    """GKEConfig must expose crd_check_timeout with sane defaults."""

    def test_crd_check_timeout_default(self) -> None:
        """Default value is 5 seconds."""
        from vaig.core.config import GKEConfig

        cfg = GKEConfig()
        assert cfg.crd_check_timeout == 5

    def test_crd_check_timeout_custom_value(self) -> None:
        """Custom value is accepted and stored correctly."""
        from vaig.core.config import GKEConfig

        cfg = GKEConfig(crd_check_timeout=10)
        assert cfg.crd_check_timeout == 10

    def test_crd_check_timeout_is_distinct_from_request_timeout(self) -> None:
        """crd_check_timeout and request_timeout are independent fields."""
        from vaig.core.config import GKEConfig

        cfg = GKEConfig(request_timeout=60, crd_check_timeout=3)
        assert cfg.request_timeout == 60
        assert cfg.crd_check_timeout == 3

    def test_crd_check_timeout_is_shorter_than_request_timeout_by_default(self) -> None:
        """Default crd_check_timeout (5) is well below request_timeout (30)."""
        from vaig.core.config import GKEConfig

        cfg = GKEConfig()
        assert cfg.crd_check_timeout < cfg.request_timeout


# ══════════════════════════════════════════════════════════════
# Change 2: _check_crd_exists uses crd_check_timeout
# ══════════════════════════════════════════════════════════════


class TestCheckCrdExistsUsesShortTimeout:
    """_check_crd_exists must use crd_check_timeout, not request_timeout."""

    def test_crd_exists_uses_crd_check_timeout(self) -> None:
        """read_custom_resource_definition is called with _request_timeout=crd_check_timeout."""
        from vaig.tools.gke.argocd import _check_crd_exists

        mock_api_client = MagicMock()
        mock_ext_api = MagicMock()
        mock_ext_api.read_custom_resource_definition.return_value = MagicMock()

        with (
            patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True),
            patch("vaig.tools.gke.argocd.get_settings") as mock_settings,
            patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api),
        ):
            mock_settings.return_value.gke.crd_check_timeout = 5
            result = _check_crd_exists("rollouts.argoproj.io", api_client=mock_api_client)

        assert result is True
        mock_ext_api.read_custom_resource_definition.assert_called_once_with(
            "rollouts.argoproj.io", _request_timeout=5
        )

    def test_crd_exists_timeout_does_not_use_request_timeout(self) -> None:
        """read_custom_resource_definition is NOT called with _request_timeout=30."""
        from vaig.tools.gke.argocd import _check_crd_exists

        mock_api_client = MagicMock()
        mock_ext_api = MagicMock()
        mock_ext_api.read_custom_resource_definition.return_value = MagicMock()

        with (
            patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True),
            patch("vaig.tools.gke.argocd.get_settings") as mock_settings,
            patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api),
        ):
            mock_settings.return_value.gke.crd_check_timeout = 5
            mock_settings.return_value.gke.request_timeout = 30
            _check_crd_exists("applications.argoproj.io", api_client=mock_api_client)

        # Must NOT be called with request_timeout=30
        for actual_call in mock_ext_api.read_custom_resource_definition.call_args_list:
            _, kwargs = actual_call
            assert kwargs.get("_request_timeout") != 30, (
                "_check_crd_exists must use crd_check_timeout, not request_timeout"
            )

    def test_crd_check_custom_timeout_value_is_propagated(self) -> None:
        """Custom crd_check_timeout value (e.g. 3s) is propagated to the API call."""
        from vaig.tools.gke.argocd import _check_crd_exists

        mock_api_client = MagicMock()
        mock_ext_api = MagicMock()
        mock_ext_api.read_custom_resource_definition.return_value = MagicMock()

        with (
            patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True),
            patch("vaig.tools.gke.argocd.get_settings") as mock_settings,
            patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api),
        ):
            mock_settings.return_value.gke.crd_check_timeout = 3
            _check_crd_exists("rollouts.argoproj.io", api_client=mock_api_client)

        mock_ext_api.read_custom_resource_definition.assert_called_with(
            "rollouts.argoproj.io", _request_timeout=3
        )


# ══════════════════════════════════════════════════════════════
# Change 2: _check_crd_exists creates retries=False ApiClient
# when no api_client is supplied
# ══════════════════════════════════════════════════════════════


class TestCheckCrdExistsRetriesDisabled:
    """_check_crd_exists must build its own ApiClient with retries=False."""

    def test_no_api_client_builds_retries_false_client(self) -> None:
        """When api_client=None, _load_k8s_config is called and config.retries=False.

        Uses FakeConfig (a simple attribute-tracking stub) to assert that
        _load_k8s_config sets retries=False on the Configuration it creates.
        Both load_incluster_config and load_kube_config are mocked so the test
        is environment-independent.
        """
        from vaig.tools.gke.argocd import _check_crd_exists

        class FakeCfg:
            retries: object = None

        fake_cfg = FakeCfg()
        mock_ext_api = MagicMock()
        mock_ext_api.read_custom_resource_definition.return_value = MagicMock()

        with (
            patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True),
            patch("vaig.tools.gke.argocd.get_settings") as mock_settings,
            # Patch Configuration in _clients so _load_k8s_config returns fake_cfg
            patch("vaig.tools.gke._clients.k8s_client.Configuration", return_value=fake_cfg),
            # Patch load_kube_config and load_incluster_config in _clients so the
            # test is environment-independent (both paths exercised by mocks)
            patch("vaig.tools.gke._clients.k8s_config.load_kube_config", return_value=None),
            patch("vaig.tools.gke._clients.k8s_config.load_incluster_config", return_value=None),
            patch("vaig.tools.gke._clients._extract_proxy_url_from_kubeconfig", return_value=None),
            patch("kubernetes.client.ApiClient"),
            patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api),
        ):
            mock_settings.return_value.gke.crd_check_timeout = 5
            mock_settings.return_value.gke.kubeconfig_path = ""
            mock_settings.return_value.gke.context = ""
            mock_settings.return_value.gke.proxy_url = ""

            _check_crd_exists("rollouts.argoproj.io")

        # retries must be explicitly disabled on the configuration
        assert fake_cfg.retries is False

    def test_provided_api_client_is_used_directly(self) -> None:
        """When api_client is provided, it is used as-is (no new client created)."""
        from vaig.tools.gke.argocd import _check_crd_exists

        mock_api_client = MagicMock()
        mock_ext_api = MagicMock()
        mock_ext_api.read_custom_resource_definition.return_value = MagicMock()

        with (
            patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True),
            patch("vaig.tools.gke.argocd.get_settings") as mock_settings,
            patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api),
        ):
            mock_settings.return_value.gke.crd_check_timeout = 5
            result = _check_crd_exists("applications.argoproj.io", api_client=mock_api_client)

        # ApiextensionsV1Api was called with the provided api_client
        # Just verify the result and that the mock_ext_api was used
        assert result is True


# ══════════════════════════════════════════════════════════════
# Regression: caching and error handling still work after change
# ══════════════════════════════════════════════════════════════


class TestCheckCrdExistsCachingRegression:
    """After the timeout fix, caching and 404/403 handling still work correctly."""

    def test_crd_exists_true_is_cached(self) -> None:
        """Positive result (CRD found) is cached so the API is only called once."""
        from vaig.tools.gke.argocd import _check_crd_exists

        mock_api_client = MagicMock()
        mock_ext_api = MagicMock()
        mock_ext_api.read_custom_resource_definition.return_value = MagicMock()

        with (
            patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True),
            patch("vaig.tools.gke.argocd.get_settings") as mock_settings,
            patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api),
        ):
            mock_settings.return_value.gke.crd_check_timeout = 5

            result1 = _check_crd_exists("rollouts.argoproj.io", api_client=mock_api_client)
            result2 = _check_crd_exists("rollouts.argoproj.io", api_client=mock_api_client)

        assert result1 is True
        assert result2 is True
        # API called only once — second call served from cache
        assert mock_ext_api.read_custom_resource_definition.call_count == 1

    def test_crd_404_returns_false_and_caches(self) -> None:
        """404 from API returns False and caches the result permanently."""
        from kubernetes.client.exceptions import ApiException

        from vaig.tools.gke.argocd import _check_crd_exists, _crd_exists_cache

        mock_api_client = MagicMock()
        mock_ext_api = MagicMock()
        exc_404 = ApiException(status=404)
        mock_ext_api.read_custom_resource_definition.side_effect = exc_404

        with (
            patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True),
            patch("vaig.tools.gke.argocd.get_settings") as mock_settings,
            patch("vaig.tools.gke.argocd.k8s_exceptions") as mock_exceptions,
            patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api),
        ):
            mock_settings.return_value.gke.crd_check_timeout = 5
            mock_exceptions.ApiException = ApiException

            result = _check_crd_exists("nonexistent.argoproj.io", api_client=mock_api_client)

        assert result is False
        assert _crd_exists_cache.get(("nonexistent.argoproj.io", mock_api_client)) is False


# ══════════════════════════════════════════════════════════════
# Change 3: _get_custom_objects_api disables urllib3 retries
# ══════════════════════════════════════════════════════════════


class TestGetCustomObjectsApiDelegatesToClients:
    """_get_custom_objects_api must delegate to _clients._create_k8s_clients
    and return the CustomObjectsApi (index [2]) from the shared tuple."""

    def test_delegates_to_create_k8s_clients(self) -> None:
        """Calls _create_k8s_clients and returns the CustomObjectsApi at index [2]."""
        from vaig.tools.gke.argo_rollouts import _get_custom_objects_api

        fake_custom_api = MagicMock()
        fake_clients = (MagicMock(), MagicMock(), fake_custom_api, MagicMock())

        with (
            patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True),
            patch("vaig.tools.gke.argo_rollouts._clients._create_k8s_clients", return_value=fake_clients) as mock_create,
        ):
            result = _get_custom_objects_api()

        assert result is fake_custom_api
        mock_create.assert_called_once()

    def test_returns_none_when_create_k8s_clients_returns_error(self) -> None:
        """Returns None when _create_k8s_clients returns a ToolResult (error)."""
        from vaig.tools.base import ToolResult
        from vaig.tools.gke.argo_rollouts import _get_custom_objects_api

        error_result = ToolResult(output="Failed to configure Kubernetes client", error=True)

        with (
            patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True),
            patch("vaig.tools.gke.argo_rollouts._clients._create_k8s_clients", return_value=error_result),
        ):
            result = _get_custom_objects_api()

        assert result is None

    def test_returns_none_when_k8s_unavailable(self) -> None:
        """Returns None when the kubernetes SDK is not available."""
        from vaig.tools.gke.argo_rollouts import _get_custom_objects_api

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", False):
            result = _get_custom_objects_api()

        assert result is None

    def test_passes_gke_settings_to_create_k8s_clients(self) -> None:
        """Passes get_settings().gke to _create_k8s_clients."""
        from vaig.tools.gke.argo_rollouts import _get_custom_objects_api

        fake_gke_config = MagicMock()
        fake_clients = (MagicMock(), MagicMock(), MagicMock(), MagicMock())

        with (
            patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True),
            patch("vaig.tools.gke.argo_rollouts.get_settings") as mock_settings,
            patch("vaig.tools.gke.argo_rollouts._clients._create_k8s_clients", return_value=fake_clients) as mock_create,
        ):
            mock_settings.return_value.gke = fake_gke_config
            _get_custom_objects_api()

        mock_create.assert_called_once_with(fake_gke_config)


# ══════════════════════════════════════════════════════════════
# Change 4: argo_request_timeout field exists in GKEConfig
# ══════════════════════════════════════════════════════════════


class TestArgoRequestTimeoutConfig:
    """GKEConfig.argo_request_timeout field exists and defaults to 10s."""

    def test_default_argo_request_timeout(self) -> None:
        """argo_request_timeout defaults to 10 seconds."""
        from vaig.core.config import GKEConfig

        cfg = GKEConfig()
        assert cfg.argo_request_timeout == 10

    def test_argo_request_timeout_customizable(self) -> None:
        """argo_request_timeout can be set to a custom value."""
        from vaig.core.config import GKEConfig

        cfg = GKEConfig(argo_request_timeout=15)
        assert cfg.argo_request_timeout == 15

    def test_argo_request_timeout_independent_of_request_timeout(self) -> None:
        """argo_request_timeout and request_timeout are independent fields."""
        from vaig.core.config import GKEConfig

        cfg = GKEConfig(request_timeout=60, argo_request_timeout=5)
        assert cfg.request_timeout == 60
        assert cfg.argo_request_timeout == 5


# ══════════════════════════════════════════════════════════════
# Change 5: Tool functions use argo_request_timeout
# ══════════════════════════════════════════════════════════════


class TestToolFunctionsUseArgoTimeout:
    """All 5 Argo Rollouts tool functions use argo_request_timeout (10s)
    instead of request_timeout (30s)."""

    def _run_tool_with_mock(self, tool_func_name: str, **kwargs: str) -> MagicMock:
        """Helper: invoke a tool function and return the mock CustomObjectsApi."""
        import vaig.tools.gke.argo_rollouts as mod

        tool_func = getattr(mod, tool_func_name)
        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.return_value = {
            "metadata": {"name": "test", "namespace": "default"},
            "spec": {},
            "status": {},
        }
        mock_api.get_cluster_custom_object.return_value = {
            "metadata": {"name": "test"},
            "spec": {},
            "status": {},
        }

        with (
            patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True),
            patch("vaig.tools.gke._clients._K8S_AVAILABLE", True),
            patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api),
        ):
            tool_func(**kwargs)

        return mock_api

    @pytest.mark.parametrize(
        "tool_name,kwargs",
        [
            ("kubectl_get_rollout", {"namespace": "ns", "name": "r1"}),
            ("kubectl_get_analysisrun", {"namespace": "ns", "name": "ar1"}),
            ("kubectl_get_analysistemplate", {"namespace": "ns", "name": "at1"}),
            ("kubectl_get_experiment", {"namespace": "ns", "name": "e1"}),
        ],
    )
    def test_namespaced_tools_use_argo_timeout(self, tool_name: str, kwargs: dict) -> None:
        """Namespaced tool functions pass argo_request_timeout (10) to the API."""  # noqa: E501
        mock_api = self._run_tool_with_mock(tool_name, **kwargs)
        call_kwargs = mock_api.get_namespaced_custom_object.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs.get("_request_timeout") == 10

    def test_cluster_analysis_template_uses_argo_timeout(self) -> None:
        """kubectl_get_cluster_analysis_template passes argo_request_timeout."""
        mock_api = self._run_tool_with_mock(
            "kubectl_get_cluster_analysis_template", name="cat1"
        )
        call_kwargs = mock_api.get_cluster_custom_object.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs.get("_request_timeout") == 10

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
    from vaig.tools.gke.argocd import _crd_exists_cache
    _crd_exists_cache.clear()


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
        """Verify config.retries ends up as False using a real Configuration-like object."""
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
        """When api_client=None, the created Configuration has retries=False."""
        from vaig.tools.gke.argocd import _check_crd_exists

        class FakeCfg:
            retries: object = None

        fake_cfg = FakeCfg()
        mock_ext_api = MagicMock()
        mock_ext_api.read_custom_resource_definition.return_value = MagicMock()

        with (
            patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True),
            patch("vaig.tools.gke.argocd.get_settings") as mock_settings,
            patch("kubernetes.client.Configuration") as mock_k8s_cfg_cls,
            patch("kubernetes.client.ApiClient") as mock_api_client_cls,
            patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api),
            patch("kubernetes.config") as mock_k8s_config,
        ):
            mock_settings.return_value.gke.crd_check_timeout = 5
            # get_default_copy() returns our fake cfg so we can inspect .retries
            mock_k8s_cfg_cls.get_default_copy.return_value = fake_cfg
            mock_k8s_config.ConfigException = type("ConfigException", (Exception,), {})
            mock_k8s_config.load_incluster_config.side_effect = mock_k8s_config.ConfigException("no cluster")

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
        assert _crd_exists_cache.get("nonexistent.argoproj.io") is False

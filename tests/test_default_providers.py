"""Tests for DefaultK8sClientProvider and DefaultGCPClientProvider.

Verifies that:
- ``DefaultK8sClientProvider`` satisfies ``K8sClientProvider`` protocol.
- ``DefaultK8sClientProvider`` delegates to module-level functions.
- ``DefaultGCPClientProvider`` satisfies ``GCPClientProvider`` protocol.
- ``DefaultGCPClientProvider`` caches clients at instance level.
- ``build_container()`` now returns non-None providers.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vaig.core.protocols import (
    GCPClientProvider,
    K8sClientProvider,
)

# ══════════════════════════════════════════════════════════════
# DefaultK8sClientProvider
# ══════════════════════════════════════════════════════════════


class TestDefaultK8sClientProvider:
    """Tests for the ``DefaultK8sClientProvider`` class."""

    def test_satisfies_protocol(self) -> None:
        """DefaultK8sClientProvider satisfies K8sClientProvider at runtime."""
        from vaig.tools.gke._clients import DefaultK8sClientProvider

        provider = DefaultK8sClientProvider()
        assert isinstance(provider, K8sClientProvider)

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_get_clients_delegates(self, mock_create: MagicMock) -> None:
        """get_clients() delegates to _create_k8s_clients()."""
        from vaig.tools.gke._clients import DefaultK8sClientProvider

        sentinel = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        mock_create.return_value = sentinel
        gke_config = MagicMock()

        provider = DefaultK8sClientProvider()
        result = provider.get_clients(gke_config)

        mock_create.assert_called_once_with(gke_config)
        assert result is sentinel

    @patch("vaig.tools.gke._clients.get_exec_client")
    def test_get_exec_client_delegates(self, mock_exec: MagicMock) -> None:
        """get_exec_client() delegates to the module-level get_exec_client()."""
        from vaig.tools.gke._clients import DefaultK8sClientProvider

        sentinel = MagicMock()
        mock_exec.return_value = sentinel
        gke_config = MagicMock()

        provider = DefaultK8sClientProvider()
        result = provider.get_exec_client(gke_config)

        mock_exec.assert_called_once_with(gke_config)
        assert result is sentinel

    @patch("vaig.tools.gke._clients.clear_k8s_client_cache")
    def test_clear_cache_delegates(self, mock_clear: MagicMock) -> None:
        """clear_cache() delegates to clear_k8s_client_cache()."""
        from vaig.tools.gke._clients import DefaultK8sClientProvider

        provider = DefaultK8sClientProvider()
        provider.clear_cache()

        mock_clear.assert_called_once()

    def test_has_no_instance_state(self) -> None:
        """DefaultK8sClientProvider uses __slots__ = () — no instance dict."""
        from vaig.tools.gke._clients import DefaultK8sClientProvider

        provider = DefaultK8sClientProvider()
        assert not hasattr(provider, "__dict__")


# ══════════════════════════════════════════════════════════════
# DefaultGCPClientProvider
# ══════════════════════════════════════════════════════════════


class TestDefaultGCPClientProvider:
    """Tests for the ``DefaultGCPClientProvider`` class."""

    def test_satisfies_protocol(self) -> None:
        """DefaultGCPClientProvider satisfies GCPClientProvider at runtime."""
        from vaig.tools.gcloud_tools import DefaultGCPClientProvider

        provider = DefaultGCPClientProvider()
        assert isinstance(provider, GCPClientProvider)

    @patch("vaig.tools.gcloud_tools._get_logging_client")
    def test_get_logging_client_delegates(self, mock_get: MagicMock) -> None:
        """get_logging_client() delegates to _get_logging_client()."""
        from vaig.tools.gcloud_tools import DefaultGCPClientProvider

        sentinel_client = MagicMock()
        mock_get.return_value = (sentinel_client, None)

        provider = DefaultGCPClientProvider()
        client, err = provider.get_logging_client(project="my-project")

        mock_get.assert_called_once_with("my-project", None)
        assert client is sentinel_client
        assert err is None

    @patch("vaig.tools.gcloud_tools._get_monitoring_client")
    def test_get_monitoring_client_delegates(self, mock_get: MagicMock) -> None:
        """get_monitoring_client() delegates to _get_monitoring_client()."""
        from vaig.tools.gcloud_tools import DefaultGCPClientProvider

        sentinel_client = MagicMock()
        mock_get.return_value = (sentinel_client, None)

        provider = DefaultGCPClientProvider()
        client, err = provider.get_monitoring_client(project="my-project")

        mock_get.assert_called_once_with("my-project", None)
        assert client is sentinel_client
        assert err is None

    @patch("vaig.tools.gcloud_tools._get_logging_client")
    def test_logging_client_is_cached(self, mock_get: MagicMock) -> None:
        """Repeated calls with same args return cached client, not re-create."""
        from vaig.tools.gcloud_tools import DefaultGCPClientProvider

        mock_get.return_value = (MagicMock(), None)

        provider = DefaultGCPClientProvider()
        result1 = provider.get_logging_client(project="proj-a")
        result2 = provider.get_logging_client(project="proj-a")

        # Only one call to the underlying factory
        mock_get.assert_called_once()
        assert result1 is result2

    @patch("vaig.tools.gcloud_tools._get_monitoring_client")
    def test_monitoring_client_is_cached(self, mock_get: MagicMock) -> None:
        """Repeated calls with same args return cached client, not re-create."""
        from vaig.tools.gcloud_tools import DefaultGCPClientProvider

        mock_get.return_value = (MagicMock(), None)

        provider = DefaultGCPClientProvider()
        result1 = provider.get_monitoring_client(project="proj-b")
        result2 = provider.get_monitoring_client(project="proj-b")

        mock_get.assert_called_once()
        assert result1 is result2

    @patch("vaig.tools.gcloud_tools._get_logging_client")
    def test_different_projects_create_separate_clients(self, mock_get: MagicMock) -> None:
        """Different project arguments produce separate cache entries."""
        from vaig.tools.gcloud_tools import DefaultGCPClientProvider

        client_a = MagicMock()
        client_b = MagicMock()
        mock_get.side_effect = [(client_a, None), (client_b, None)]

        provider = DefaultGCPClientProvider()
        result_a = provider.get_logging_client(project="proj-a")
        result_b = provider.get_logging_client(project="proj-b")

        assert mock_get.call_count == 2
        assert result_a[0] is client_a
        assert result_b[0] is client_b

    @patch("vaig.tools.gcloud_tools._get_logging_client")
    @patch("vaig.tools.gcloud_tools._get_monitoring_client")
    def test_clear_cache_empties_both_caches(
        self, mock_mon: MagicMock, mock_log: MagicMock,
    ) -> None:
        """clear_cache() empties both logging and monitoring caches."""
        from vaig.tools.gcloud_tools import DefaultGCPClientProvider

        mock_log.return_value = (MagicMock(), None)
        mock_mon.return_value = (MagicMock(), None)

        provider = DefaultGCPClientProvider()
        provider.get_logging_client()
        provider.get_monitoring_client()

        assert len(provider._logging_cache) == 1
        assert len(provider._monitoring_cache) == 1

        provider.clear_cache()

        assert len(provider._logging_cache) == 0
        assert len(provider._monitoring_cache) == 0

    @patch("vaig.tools.gcloud_tools._get_logging_client")
    def test_cache_respects_credentials_identity(self, mock_get: MagicMock) -> None:
        """Different credentials objects produce separate cache entries."""
        from vaig.tools.gcloud_tools import DefaultGCPClientProvider

        cred_a = MagicMock()
        cred_b = MagicMock()
        mock_get.side_effect = [(MagicMock(), None), (MagicMock(), None)]

        provider = DefaultGCPClientProvider()
        provider.get_logging_client(credentials=cred_a)
        provider.get_logging_client(credentials=cred_b)

        assert mock_get.call_count == 2


# ══════════════════════════════════════════════════════════════
# build_container — providers are wired
# ══════════════════════════════════════════════════════════════


class TestBuildContainerProviders:
    """Tests that build_container() creates real providers."""

    def test_k8s_provider_is_default(self) -> None:
        """build_container() creates a DefaultK8sClientProvider."""
        from vaig.core.config import Settings
        from vaig.core.container import build_container
        from vaig.tools.gke._clients import DefaultK8sClientProvider

        container = build_container(Settings())
        assert container.k8s_provider is not None
        assert isinstance(container.k8s_provider, DefaultK8sClientProvider)
        assert isinstance(container.k8s_provider, K8sClientProvider)

    def test_gcp_provider_is_default(self) -> None:
        """build_container() creates a DefaultGCPClientProvider."""
        from vaig.core.config import Settings
        from vaig.core.container import build_container
        from vaig.tools.gcloud_tools import DefaultGCPClientProvider

        container = build_container(Settings())
        assert container.gcp_provider is not None
        assert isinstance(container.gcp_provider, DefaultGCPClientProvider)
        assert isinstance(container.gcp_provider, GCPClientProvider)

    def test_providers_are_distinct_instances(self) -> None:
        """Each build_container() call creates fresh provider instances."""
        from vaig.core.config import Settings
        from vaig.core.container import build_container

        container1 = build_container(Settings())
        container2 = build_container(Settings())

        assert container1.k8s_provider is not container2.k8s_provider
        assert container1.gcp_provider is not container2.gcp_provider

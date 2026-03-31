"""Tests for ServiceContainer and build_container factory.

Verifies that:
- ``build_container`` creates a valid container with all fields populated.
- ``ServiceContainer`` is frozen (immutable).
- Container fields are accessible.
- Mock objects satisfying protocols work in the container.
- ``build_container`` works with minimal (default) settings.
"""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import pytest

from vaig.core.config import Settings
from vaig.core.container import ServiceContainer, build_container
from vaig.core.event_bus import EventBus
from vaig.core.protocols import GCPClientProvider, GeminiClientProtocol, K8sClientProvider

# ══════════════════════════════════════════════════════════════
# build_container
# ══════════════════════════════════════════════════════════════


class TestBuildContainer:
    """Tests for the ``build_container`` factory function."""

    def test_build_creates_valid_container(self) -> None:
        """build_container returns a ServiceContainer with all expected fields."""
        settings = Settings()
        container = build_container(settings)

        assert isinstance(container, ServiceContainer)
        assert container.settings is settings
        assert container.gemini_client is not None
        assert isinstance(container.gemini_client, GeminiClientProtocol)
        assert container.event_bus is not None
        assert isinstance(container.event_bus, EventBus)

    def test_build_with_minimal_settings(self) -> None:
        """build_container works with default Settings (no config file)."""
        settings = Settings()
        container = build_container(settings)

        assert container.settings is settings
        # Providers are real default implementations (not None)
        assert container.k8s_provider is not None
        assert isinstance(container.k8s_provider, K8sClientProvider)
        assert container.gcp_provider is not None
        assert isinstance(container.gcp_provider, GCPClientProvider)

    def test_build_uses_event_bus_singleton(self) -> None:
        """build_container uses the EventBus singleton."""
        settings = Settings()
        container = build_container(settings)

        assert container.event_bus is EventBus.get()

    def test_gemini_client_has_correct_model(self) -> None:
        """The GeminiClient in the container uses the settings' default model."""
        settings = Settings()
        container = build_container(settings)

        assert container.gemini_client.current_model == settings.models.default


# ══════════════════════════════════════════════════════════════
# ServiceContainer — frozen immutability
# ══════════════════════════════════════════════════════════════


class TestServiceContainerFrozen:
    """Tests that ServiceContainer is a frozen dataclass."""

    def test_container_is_frozen(self) -> None:
        """ServiceContainer instances cannot be mutated."""
        settings = Settings()
        container = build_container(settings)

        with pytest.raises(dataclasses.FrozenInstanceError):
            container.settings = Settings()  # type: ignore[misc]

    def test_cannot_set_gemini_client(self) -> None:
        """Cannot reassign gemini_client on a frozen container."""
        settings = Settings()
        container = build_container(settings)

        with pytest.raises(dataclasses.FrozenInstanceError):
            container.gemini_client = MagicMock()  # type: ignore[misc]

    def test_cannot_set_event_bus(self) -> None:
        """Cannot reassign event_bus on a frozen container."""
        settings = Settings()
        container = build_container(settings)

        with pytest.raises(dataclasses.FrozenInstanceError):
            container.event_bus = EventBus()  # type: ignore[misc]

    def test_cannot_set_k8s_provider(self) -> None:
        """Cannot reassign k8s_provider on a frozen container."""
        settings = Settings()
        container = build_container(settings)

        with pytest.raises(dataclasses.FrozenInstanceError):
            container.k8s_provider = MagicMock()  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════
# ServiceContainer — field access
# ══════════════════════════════════════════════════════════════


class TestServiceContainerFields:
    """Tests that ServiceContainer fields are accessible."""

    def test_all_fields_are_accessible(self) -> None:
        """All expected fields can be read from the container."""
        settings = Settings()
        container = build_container(settings)

        # Just accessing each field — should not raise
        _ = container.settings
        _ = container.gemini_client
        _ = container.k8s_provider
        _ = container.gcp_provider
        _ = container.event_bus

    def test_container_has_expected_field_names(self) -> None:
        """ServiceContainer dataclass has exactly the expected fields."""
        field_names = {f.name for f in dataclasses.fields(ServiceContainer)}
        expected = {"settings", "gemini_client", "k8s_provider", "gcp_provider", "event_bus", "quota_checker"}
        assert field_names == expected


# ══════════════════════════════════════════════════════════════
# Mock injection
# ══════════════════════════════════════════════════════════════


class TestMockInjection:
    """Tests that mock objects satisfying protocols work in the container."""

    def _make_mock_gemini_client(self) -> MagicMock:
        """Create a MagicMock that satisfies GeminiClientProtocol."""
        mock = MagicMock(spec=GeminiClientProtocol)
        mock.current_model = "mock-model-v1"
        return mock

    def _make_mock_k8s_provider(self) -> MagicMock:
        """Create a MagicMock that satisfies K8sClientProvider."""
        return MagicMock(spec=K8sClientProvider)

    def _make_mock_gcp_provider(self) -> MagicMock:
        """Create a MagicMock that satisfies GCPClientProvider."""
        return MagicMock(spec=GCPClientProvider)

    def test_container_with_all_mocks(self) -> None:
        """ServiceContainer can be constructed with all mock dependencies."""
        settings = Settings()
        container = ServiceContainer(
            settings=settings,
            gemini_client=self._make_mock_gemini_client(),
            k8s_provider=self._make_mock_k8s_provider(),
            gcp_provider=self._make_mock_gcp_provider(),
            event_bus=EventBus.get(),
        )

        assert container.gemini_client.current_model == "mock-model-v1"
        assert container.k8s_provider is not None
        assert container.gcp_provider is not None

    def test_container_with_none_optional_providers(self) -> None:
        """ServiceContainer accepts None for optional providers."""
        settings = Settings()
        container = ServiceContainer(
            settings=settings,
            gemini_client=self._make_mock_gemini_client(),
            k8s_provider=None,
            gcp_provider=None,
            event_bus=EventBus.get(),
        )

        assert container.k8s_provider is None
        assert container.gcp_provider is None


# ══════════════════════════════════════════════════════════════
# __all__ exports
# ══════════════════════════════════════════════════════════════


class TestContainerExports:
    """Verify __all__ exports are correct."""

    def test_all_exports(self) -> None:
        """ServiceContainer and build_container are listed in __all__."""
        import vaig.core.container as mod

        assert "ServiceContainer" in mod.__all__
        assert "build_container" in mod.__all__
        assert len(mod.__all__) == 2


# ══════════════════════════════════════════════════════════════
# Quota checker wiring
# ══════════════════════════════════════════════════════════════


class TestQuotaCheckerWiring:
    """Tests that QuotaChecker is wired correctly based on settings."""

    def test_quota_checker_none_when_disabled(self) -> None:
        """quota_checker is None when rate_limit.enabled is False (default)."""
        settings = Settings()
        container = build_container(settings)
        assert container.quota_checker is None

    def test_quota_checker_default_in_mock_container(self) -> None:
        """ServiceContainer accepts quota_checker=None by default."""
        settings = Settings()
        container = ServiceContainer(
            settings=settings,
            gemini_client=MagicMock(spec=GeminiClientProtocol),
            k8s_provider=None,
            gcp_provider=None,
            event_bus=EventBus.get(),
        )
        assert container.quota_checker is None

    def test_quota_checker_can_be_injected(self) -> None:
        """ServiceContainer accepts a mock quota_checker."""
        settings = Settings()
        mock_checker = MagicMock()
        container = ServiceContainer(
            settings=settings,
            gemini_client=MagicMock(spec=GeminiClientProtocol),
            k8s_provider=None,
            gcp_provider=None,
            event_bus=EventBus.get(),
            quota_checker=mock_checker,
        )
        assert container.quota_checker is mock_checker

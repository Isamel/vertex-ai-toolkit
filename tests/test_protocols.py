"""Tests for DI protocol interfaces — structural typing and runtime checks.

Verifies that:
- ``GeminiClient`` satisfies ``GeminiClientProtocol`` at runtime.
- Protocol methods match the real class's public interface.
- Structural typing works: a mock class with the same methods satisfies the protocol.
- ``K8sClientProvider`` and ``GCPClientProvider`` are runtime-checkable.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from vaig.core.protocols import GCPClientProvider, GeminiClientProtocol, K8sClientProvider

# ══════════════════════════════════════════════════════════════
# GeminiClientProtocol
# ══════════════════════════════════════════════════════════════


class TestGeminiClientProtocol:
    """Tests for the GeminiClientProtocol."""

    def test_real_client_satisfies_protocol(self) -> None:
        """GeminiClient instance satisfies GeminiClientProtocol at runtime."""
        from vaig.core.client import GeminiClient
        from vaig.core.config import Settings

        settings = Settings()
        client = GeminiClient(settings)

        assert isinstance(client, GeminiClientProtocol)

    def test_protocol_is_runtime_checkable(self) -> None:
        """GeminiClientProtocol is decorated with @runtime_checkable."""
        # A plain object should NOT satisfy the protocol
        assert not isinstance(object(), GeminiClientProtocol)

    def test_protocol_methods_match_real_class(self) -> None:
        """Protocol declares methods that exist on the real GeminiClient."""
        from vaig.core.client import GeminiClient

        # Key public methods that consumers depend on
        expected_methods = [
            "initialize",
            "async_initialize",
            "generate",
            "async_generate",
            "generate_stream",
            "async_generate_stream",
            "generate_with_tools",
            "async_generate_with_tools",
            "count_tokens",
            "switch_model",
            "list_available_models",
            "build_function_response_parts",
        ]

        for method_name in expected_methods:
            assert hasattr(GeminiClient, method_name), (
                f"GeminiClient missing method '{method_name}' declared in protocol"
            )

    def test_structural_typing_with_custom_class(self) -> None:
        """A custom class with the right methods satisfies GeminiClientProtocol."""

        class FakeClient:
            """Minimal fake implementing the protocol surface."""

            @property
            def current_model(self) -> str:
                return "fake-model"

            def initialize(self) -> None:
                pass

            async def async_initialize(self) -> None:
                pass

            def generate(self, prompt: Any, **kwargs: Any) -> Any:
                return None

            async def async_generate(self, prompt: Any, **kwargs: Any) -> Any:
                return None

            def generate_stream(self, prompt: Any, **kwargs: Any) -> Any:
                return None

            async def async_generate_stream(self, prompt: Any, **kwargs: Any) -> Any:
                return None

            def generate_with_tools(self, prompt: Any, **kwargs: Any) -> Any:
                return None

            async def async_generate_with_tools(self, prompt: Any, **kwargs: Any) -> Any:
                return None

            def count_tokens(self, prompt: Any, **kwargs: Any) -> int | None:
                return 42

            def switch_model(self, model_id: str) -> str:
                return model_id

            def list_available_models(self) -> list[dict[str, str]]:
                return []

            @staticmethod
            def build_function_response_parts(results: list[dict[str, Any]]) -> list[Any]:
                return []

        fake = FakeClient()
        assert isinstance(fake, GeminiClientProtocol)

    def test_incomplete_class_does_not_satisfy_protocol(self) -> None:
        """A class missing required methods does NOT satisfy the protocol."""

        class IncompleteClient:
            @property
            def current_model(self) -> str:
                return "nope"

            # Missing all the generate methods, etc.

        assert not isinstance(IncompleteClient(), GeminiClientProtocol)


# ══════════════════════════════════════════════════════════════
# K8sClientProvider
# ══════════════════════════════════════════════════════════════


class TestK8sClientProvider:
    """Tests for the K8sClientProvider protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """K8sClientProvider is decorated with @runtime_checkable."""
        assert not isinstance(object(), K8sClientProvider)

    def test_structural_typing_with_custom_class(self) -> None:
        """A custom class with the right methods satisfies K8sClientProvider."""

        class FakeK8sProvider:
            def get_clients(self, gke_config: Any) -> tuple[Any, Any, Any, Any] | Any:
                return (None, None, None, None)

            def get_exec_client(self, gke_config: Any) -> Any:
                return None

            def clear_cache(self) -> None:
                pass

        assert isinstance(FakeK8sProvider(), K8sClientProvider)

    def test_incomplete_class_does_not_satisfy(self) -> None:
        """A class missing methods does NOT satisfy K8sClientProvider."""

        class PartialK8s:
            def get_clients(self, gke_config: Any) -> Any:
                return None
            # Missing get_exec_client and clear_cache

        assert not isinstance(PartialK8s(), K8sClientProvider)


# ══════════════════════════════════════════════════════════════
# GCPClientProvider
# ══════════════════════════════════════════════════════════════


class TestGCPClientProvider:
    """Tests for the GCPClientProvider protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """GCPClientProvider is decorated with @runtime_checkable."""
        assert not isinstance(object(), GCPClientProvider)

    def test_structural_typing_with_custom_class(self) -> None:
        """A custom class with the right methods satisfies GCPClientProvider."""

        class FakeGCPProvider:
            def get_logging_client(
                self, project: str | None = None, credentials: Any | None = None,
            ) -> tuple[Any, str | None]:
                return (MagicMock(), None)

            def get_monitoring_client(
                self, project: str | None = None, credentials: Any | None = None,
            ) -> tuple[Any, str | None]:
                return (MagicMock(), None)

            def clear_cache(self) -> None:
                pass

        assert isinstance(FakeGCPProvider(), GCPClientProvider)

    def test_incomplete_class_does_not_satisfy(self) -> None:
        """A class missing methods does NOT satisfy GCPClientProvider."""

        class PartialGCP:
            def get_logging_client(self, **kwargs: Any) -> Any:
                return None
            # Missing get_monitoring_client and clear_cache

        assert not isinstance(PartialGCP(), GCPClientProvider)


# ══════════════════════════════════════════════════════════════
# __all__ exports
# ══════════════════════════════════════════════════════════════


class TestProtocolExports:
    """Verify __all__ exports are correct."""

    def test_all_exports(self) -> None:
        """All protocol classes are listed in __all__."""
        import vaig.core.protocols as mod

        assert "GeminiClientProtocol" in mod.__all__
        assert "K8sClientProvider" in mod.__all__
        assert "GCPClientProvider" in mod.__all__
        assert len(mod.__all__) == 3

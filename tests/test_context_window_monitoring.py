"""Tests for context window monitoring feature.

Covers:
- ContextWindowChecked event emission in sync tool loop
- ContextWindowChecked event emission in async tool loop
- peak_context_pct tracked and returned in ToolLoopResult
- ContextWindowExceededError raised on InvalidArgument (sync)
- ContextWindowExceededError raised on InvalidArgument (async)
- ToolAwareAgent.execute catches ContextWindowExceededError → failed AgentResult
- ToolAwareAgent.async_execute catches ContextWindowExceededError → failed AgentResult
- context_window_pct included in metadata for successful execute
- ContextWindowChecked exported from vaig.core
- ContextWindowExceededError exported from vaig.core
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vaig.agents.mixins import ToolLoopMixin, ToolLoopResult
from vaig.agents.tool_aware import ToolAwareAgent
from vaig.core.config import DEFAULT_CONTEXT_WINDOW
from vaig.core.events import ContextWindowChecked
from vaig.core.exceptions import ContextWindowExceededError
from vaig.tools.base import ToolRegistry

# ── Helpers ──────────────────────────────────────────────────


def _make_mock_client() -> MagicMock:
    """Sync mock GeminiClient."""
    client = MagicMock()
    client.current_model = "gemini-2.5-pro"
    client.build_function_response_parts.return_value = []
    return client


def _make_async_mock_client() -> MagicMock:
    """Async mock GeminiClient."""
    client = MagicMock()
    client.current_model = "gemini-2.5-pro"
    client.build_function_response_parts.return_value = []
    client.async_generate_with_tools = AsyncMock()
    return client


def _make_text_result(
    text: str = "All good.",
    prompt_tokens: int = 100,
    model: str = "gemini-2.5-pro",
) -> MagicMock:
    """Mock ToolCallResult with text (no function calls)."""
    r = MagicMock()
    r.function_calls = []
    r.text = text
    r.model = model
    r.finish_reason = "STOP"
    r.usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 50,
        "total_tokens": prompt_tokens + 50,
    }
    return r


def _make_agent() -> tuple[ToolAwareAgent, MagicMock]:
    """Create a ToolAwareAgent with a mock client."""
    client = _make_mock_client()
    registry = ToolRegistry()
    agent = ToolAwareAgent(
        system_instruction="You are a test agent.",
        tool_registry=registry,
        model="gemini-2.5-pro",
        name="test-agent",
        client=client,
    )
    return agent, client


class _ConcreteLoopMixin(ToolLoopMixin):
    """Minimal concrete class so we can call _run_tool_loop directly."""

    pass


# ── TestContextWindowCheckedEvent ────────────────────────────


class TestContextWindowCheckedEvent:
    """ContextWindowChecked event is emitted on each iteration."""

    def test_event_emitted_on_text_response(self) -> None:
        """Sync loop emits ContextWindowChecked when it gets a text response."""
        mixin = _ConcreteLoopMixin()
        client = _make_mock_client()
        client.generate_with_tools.return_value = _make_text_result(prompt_tokens=200_000)
        registry = ToolRegistry()

        emitted: list[ContextWindowChecked] = []

        with patch("vaig.core.event_bus.EventBus.get") as mock_get:
            mock_bus = MagicMock()
            mock_get.return_value = mock_bus
            mock_bus.emit.side_effect = lambda e: emitted.append(e) if isinstance(e, ContextWindowChecked) else None

            mixin._run_tool_loop(
                client=client,
                prompt="hello",
                tool_registry=registry,
                system_instruction="test",
                history=[],
                context_window=1_000_000,
            )

        assert len(emitted) == 1
        evt = emitted[0]
        assert evt.prompt_tokens == 200_000
        assert evt.context_window == 1_000_000
        assert abs(evt.context_pct - 20.0) < 0.01
        assert evt.status == "ok"
        assert evt.iteration == 1

    def test_event_status_warning_at_80_percent(self) -> None:
        """Status is 'warning' when prompt tokens >= 80% of context window."""
        mixin = _ConcreteLoopMixin()
        client = _make_mock_client()
        client.generate_with_tools.return_value = _make_text_result(prompt_tokens=800_000)
        registry = ToolRegistry()

        emitted: list[ContextWindowChecked] = []

        with patch("vaig.core.event_bus.EventBus.get") as mock_get:
            mock_bus = MagicMock()
            mock_get.return_value = mock_bus
            mock_bus.emit.side_effect = lambda e: emitted.append(e) if isinstance(e, ContextWindowChecked) else None

            mixin._run_tool_loop(
                client=client,
                prompt="hello",
                tool_registry=registry,
                system_instruction="test",
                history=[],
                context_window=1_000_000,
            )

        assert emitted[0].status == "warning"

    def test_event_status_error_at_95_percent(self) -> None:
        """Status is 'error' when prompt tokens >= 95% of context window."""
        mixin = _ConcreteLoopMixin()
        client = _make_mock_client()
        client.generate_with_tools.return_value = _make_text_result(prompt_tokens=960_000)
        registry = ToolRegistry()

        emitted: list[ContextWindowChecked] = []

        with patch("vaig.core.event_bus.EventBus.get") as mock_get:
            mock_bus = MagicMock()
            mock_get.return_value = mock_bus
            mock_bus.emit.side_effect = lambda e: emitted.append(e) if isinstance(e, ContextWindowChecked) else None

            mixin._run_tool_loop(
                client=client,
                prompt="hello",
                tool_registry=registry,
                system_instruction="test",
                history=[],
                context_window=1_000_000,
            )

        assert emitted[0].status == "error"

    def test_event_failure_does_not_crash_loop(self) -> None:
        """If EventBus.emit raises, the loop continues normally."""
        mixin = _ConcreteLoopMixin()
        client = _make_mock_client()
        client.generate_with_tools.return_value = _make_text_result()
        registry = ToolRegistry()

        with patch("vaig.core.event_bus.EventBus.get") as mock_get:
            mock_bus = MagicMock()
            mock_get.return_value = mock_bus
            mock_bus.emit.side_effect = RuntimeError("bus broken")

            result = mixin._run_tool_loop(
                client=client,
                prompt="hello",
                tool_registry=registry,
                system_instruction="test",
                history=[],
            )

        assert isinstance(result, ToolLoopResult)


# ── TestPeakContextPct ────────────────────────────────────────


class TestPeakContextPct:
    """peak_context_pct is tracked and returned in ToolLoopResult."""

    def test_peak_context_pct_in_result(self) -> None:
        """ToolLoopResult carries the peak context usage percentage."""
        mixin = _ConcreteLoopMixin()
        client = _make_mock_client()
        client.generate_with_tools.return_value = _make_text_result(prompt_tokens=500_000)
        registry = ToolRegistry()

        result = mixin._run_tool_loop(
            client=client,
            prompt="hello",
            tool_registry=registry,
            system_instruction="test",
            history=[],
            context_window=1_000_000,
        )

        assert abs(result.peak_context_pct - 50.0) < 0.01

    def test_peak_context_pct_default_is_zero(self) -> None:
        """Default peak_context_pct is 0.0 when no prompt_tokens in usage."""
        mixin = _ConcreteLoopMixin()
        client = _make_mock_client()
        r = _make_text_result(prompt_tokens=0)
        r.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        client.generate_with_tools.return_value = r
        registry = ToolRegistry()

        result = mixin._run_tool_loop(
            client=client,
            prompt="hello",
            tool_registry=registry,
            system_instruction="test",
            history=[],
            context_window=1_000_000,
        )

        assert result.peak_context_pct == 0.0


# ── TestContextWindowExceededError ───────────────────────────


class TestContextWindowExceededError:
    """InvalidArgument from the API is converted to ContextWindowExceededError."""

    def test_invalid_argument_raises_context_window_exceeded(self) -> None:
        """Sync loop converts google InvalidArgument → ContextWindowExceededError."""
        from google.api_core.exceptions import InvalidArgument

        mixin = _ConcreteLoopMixin()
        client = _make_mock_client()
        client.generate_with_tools.side_effect = InvalidArgument("context window exceeded")
        registry = ToolRegistry()

        with pytest.raises(ContextWindowExceededError):
            mixin._run_tool_loop(
                client=client,
                prompt="big prompt",
                tool_registry=registry,
                system_instruction="test",
                history=[],
            )

    def test_context_window_exceeded_has_context_pct(self) -> None:
        """ContextWindowExceededError carries context_pct attribute."""
        from google.api_core.exceptions import InvalidArgument

        mixin = _ConcreteLoopMixin()
        client = _make_mock_client()
        client.generate_with_tools.side_effect = InvalidArgument("context window exceeded")
        registry = ToolRegistry()

        with pytest.raises(ContextWindowExceededError) as exc_info:
            mixin._run_tool_loop(
                client=client,
                prompt="big prompt",
                tool_registry=registry,
                system_instruction="test",
                history=[],
            )

        assert isinstance(exc_info.value.context_pct, float)

    def test_other_exceptions_not_wrapped(self) -> None:
        """Non-InvalidArgument exceptions propagate unchanged."""
        mixin = _ConcreteLoopMixin()
        client = _make_mock_client()
        client.generate_with_tools.side_effect = RuntimeError("connection reset")
        registry = ToolRegistry()

        with pytest.raises(RuntimeError, match="connection reset"):
            mixin._run_tool_loop(
                client=client,
                prompt="hello",
                tool_registry=registry,
                system_instruction="test",
                history=[],
            )


# ── TestAsyncContextWindowMonitoring ─────────────────────────


class TestAsyncContextWindowMonitoring:
    """Async loop has the same context window monitoring behaviour."""

    @pytest.mark.asyncio
    async def test_async_event_emitted(self) -> None:
        """Async tool loop emits ContextWindowChecked."""
        mixin = _ConcreteLoopMixin()
        client = _make_async_mock_client()
        client.async_generate_with_tools.return_value = _make_text_result(prompt_tokens=300_000)
        registry = ToolRegistry()

        emitted: list[ContextWindowChecked] = []

        with patch("vaig.core.event_bus.EventBus.get") as mock_get:
            mock_bus = MagicMock()
            mock_get.return_value = mock_bus
            mock_bus.emit.side_effect = lambda e: emitted.append(e) if isinstance(e, ContextWindowChecked) else None

            await mixin._async_run_tool_loop(
                client=client,
                prompt="hello",
                tool_registry=registry,
                system_instruction="test",
                history=[],
                context_window=1_000_000,
            )

        assert len(emitted) == 1
        assert emitted[0].prompt_tokens == 300_000
        assert emitted[0].status == "ok"

    @pytest.mark.asyncio
    async def test_async_invalid_argument_raises_context_window_exceeded(self) -> None:
        """Async loop converts InvalidArgument → ContextWindowExceededError."""
        from google.api_core.exceptions import InvalidArgument

        mixin = _ConcreteLoopMixin()
        client = _make_async_mock_client()
        client.async_generate_with_tools.side_effect = InvalidArgument("context window exceeded")
        registry = ToolRegistry()

        with pytest.raises(ContextWindowExceededError):
            await mixin._async_run_tool_loop(
                client=client,
                prompt="big prompt",
                tool_registry=registry,
                system_instruction="test",
                history=[],
            )

    @pytest.mark.asyncio
    async def test_async_peak_context_pct_in_result(self) -> None:
        """Async ToolLoopResult carries peak_context_pct."""
        mixin = _ConcreteLoopMixin()
        client = _make_async_mock_client()
        client.async_generate_with_tools.return_value = _make_text_result(prompt_tokens=400_000)
        registry = ToolRegistry()

        result = await mixin._async_run_tool_loop(
            client=client,
            prompt="hello",
            tool_registry=registry,
            system_instruction="test",
            history=[],
            context_window=1_000_000,
        )

        assert abs(result.peak_context_pct - 40.0) < 0.01


# ── TestToolAwareAgentIntegration ────────────────────────────


class TestToolAwareAgentIntegration:
    """ToolAwareAgent integrates context window monitoring."""

    def test_execute_includes_context_window_pct_in_metadata(self) -> None:
        """Successful execute adds context_window_pct to metadata."""
        agent, client = _make_agent()
        client.generate_with_tools.return_value = _make_text_result(prompt_tokens=200_000)

        result = agent.execute("Do something")

        assert result.success is True
        assert "context_window_pct" in result.metadata
        assert isinstance(result.metadata["context_window_pct"], float)

    def test_execute_context_window_exceeded_returns_failed_result(self) -> None:
        """ContextWindowExceededError is caught and returns a failed AgentResult."""
        from google.api_core.exceptions import InvalidArgument

        agent, client = _make_agent()
        client.generate_with_tools.side_effect = InvalidArgument("context window exceeded")

        result = agent.execute("huge prompt")

        assert result.success is False
        assert "Context window exceeded" in result.content
        assert result.agent_name == "test-agent"

    def test_execute_context_window_exceeded_metadata_has_pct(self) -> None:
        """Failed result from context window exceeded has context_window_pct in metadata."""
        from google.api_core.exceptions import InvalidArgument

        agent, client = _make_agent()
        client.generate_with_tools.side_effect = InvalidArgument("prompt is too long")

        result = agent.execute("huge prompt")

        assert "context_window_pct" in result.metadata

    @pytest.mark.asyncio
    async def test_async_execute_includes_context_window_pct_in_metadata(self) -> None:
        """async_execute adds context_window_pct to metadata on success."""
        client = _make_async_mock_client()
        client.async_generate_with_tools.return_value = _make_text_result(prompt_tokens=100_000)
        registry = ToolRegistry()
        agent = ToolAwareAgent(
            system_instruction="test",
            tool_registry=registry,
            model="gemini-2.5-pro",
            name="async-agent",
            client=client,
        )

        result = await agent.async_execute("Do something")

        assert result.success is True
        assert "context_window_pct" in result.metadata

    @pytest.mark.asyncio
    async def test_async_execute_context_window_exceeded_returns_failed_result(self) -> None:
        """async_execute catches ContextWindowExceededError and returns failed AgentResult."""
        from google.api_core.exceptions import InvalidArgument

        client = _make_async_mock_client()
        client.async_generate_with_tools.side_effect = InvalidArgument("token limit exceeded")
        registry = ToolRegistry()
        agent = ToolAwareAgent(
            system_instruction="test",
            tool_registry=registry,
            model="gemini-2.5-pro",
            name="async-agent",
            client=client,
        )

        result = await agent.async_execute("huge prompt")

        assert result.success is False
        assert "Context window exceeded" in result.content


# ── TestExports ───────────────────────────────────────────────


class TestExports:
    """New types are properly exported from vaig.core."""

    def test_context_window_checked_exported(self) -> None:
        """ContextWindowChecked is importable from vaig.core."""
        from vaig.core import ContextWindowChecked as CWC  # noqa: N817

        assert CWC is ContextWindowChecked

    def test_context_window_exceeded_error_exported(self) -> None:
        """ContextWindowExceededError is importable from vaig.core."""
        from vaig.core import ContextWindowExceededError as CWE  # noqa: N814

        assert CWE is ContextWindowExceededError

    def test_context_window_checked_is_frozen_dataclass(self) -> None:
        """ContextWindowChecked instances are immutable."""
        evt = ContextWindowChecked(
            model="gemini-2.5-pro",
            prompt_tokens=100,
            context_window=1_000_000,
            context_pct=0.01,
            iteration=1,
            status="ok",
        )
        with pytest.raises((AttributeError, TypeError)):
            evt.status = "warning"  # type: ignore[misc]

    def test_context_window_exceeded_error_inherits_vaig_error(self) -> None:
        """ContextWindowExceededError is a subclass of VAIGError."""
        from vaig.core.exceptions import VAIGError

        exc = ContextWindowExceededError("too big", context_pct=99.0)
        assert isinstance(exc, VAIGError)
        assert exc.context_pct == 99.0

    def test_default_context_window_constant(self) -> None:
        """DEFAULT_CONTEXT_WINDOW is a reasonable positive integer."""
        assert DEFAULT_CONTEXT_WINDOW > 0
        assert isinstance(DEFAULT_CONTEXT_WINDOW, int)


# ── TestContextWindowPropagation ─────────────────────────────


class TestContextWindowPropagation:
    """ToolAwareAgent propagates model-specific context_window to the tool loop."""

    def test_execute_uses_model_specific_context_window(self) -> None:
        """execute() resolves context_window from settings for the agent's model."""
        from unittest.mock import patch

        from vaig.core.config import ModelInfo, Settings

        # Build settings with a model that has a custom context window
        settings = Settings()
        settings.models.available = [
            ModelInfo(id="gemini-custom", context_window=500_000),
        ]

        client = _make_mock_client()
        # Use 400_000 prompt tokens — that's 80% of 500_000
        client.generate_with_tools.return_value = _make_text_result(
            prompt_tokens=400_000,
            model="gemini-custom",
        )
        registry = ToolRegistry()
        agent = ToolAwareAgent(
            system_instruction="test",
            tool_registry=registry,
            model="gemini-custom",
            name="custom-model-agent",
            client=client,
        )

        emitted: list[ContextWindowChecked] = []

        with (
            patch("vaig.core.event_bus.EventBus.get") as mock_get,
            patch("vaig.agents.tool_aware.get_settings", return_value=settings),
        ):
            mock_bus = MagicMock()
            mock_get.return_value = mock_bus
            mock_bus.emit.side_effect = lambda e: emitted.append(e) if isinstance(e, ContextWindowChecked) else None

            result = agent.execute("do something")

        assert result.success is True
        # The event must carry the model-specific context window, not the default 1M
        assert len(emitted) == 1
        assert emitted[0].context_window == 500_000
        # 400_000 / 500_000 * 100 = 80.0%
        assert abs(emitted[0].context_pct - 80.0) < 0.01

    def test_execute_falls_back_to_default_when_model_not_in_settings(self) -> None:
        """execute() falls back to DEFAULT_CONTEXT_WINDOW when model not in settings."""
        from unittest.mock import patch

        from vaig.core.config import Settings

        # Empty available models list — model not registered
        settings = Settings()
        settings.models.available = []

        client = _make_mock_client()
        client.generate_with_tools.return_value = _make_text_result(
            prompt_tokens=100_000,
        )
        registry = ToolRegistry()
        agent = ToolAwareAgent(
            system_instruction="test",
            tool_registry=registry,
            model="gemini-unknown",
            name="fallback-agent",
            client=client,
        )

        emitted: list[ContextWindowChecked] = []

        with (
            patch("vaig.core.event_bus.EventBus.get") as mock_get,
            patch("vaig.agents.tool_aware.get_settings", return_value=settings),
        ):
            mock_bus = MagicMock()
            mock_get.return_value = mock_bus
            mock_bus.emit.side_effect = lambda e: emitted.append(e) if isinstance(e, ContextWindowChecked) else None

            result = agent.execute("do something")

        assert result.success is True
        assert len(emitted) == 1
        # Falls back to the global default context window
        assert emitted[0].context_window == DEFAULT_CONTEXT_WINDOW

    @pytest.mark.asyncio
    async def test_async_execute_uses_model_specific_context_window(self) -> None:
        """async_execute() resolves context_window from settings for the agent's model."""
        from unittest.mock import patch

        from vaig.core.config import ModelInfo, Settings

        settings = Settings()
        settings.models.available = [
            ModelInfo(id="gemini-custom-async", context_window=800_000),
        ]

        client = _make_async_mock_client()
        client.async_generate_with_tools.return_value = _make_text_result(
            prompt_tokens=640_000,
            model="gemini-custom-async",
        )
        registry = ToolRegistry()
        agent = ToolAwareAgent(
            system_instruction="test",
            tool_registry=registry,
            model="gemini-custom-async",
            name="async-custom-agent",
            client=client,
        )

        emitted: list[ContextWindowChecked] = []

        with (
            patch("vaig.core.event_bus.EventBus.get") as mock_get,
            patch("vaig.agents.tool_aware.get_settings", return_value=settings),
        ):
            mock_bus = MagicMock()
            mock_get.return_value = mock_bus
            mock_bus.emit.side_effect = lambda e: emitted.append(e) if isinstance(e, ContextWindowChecked) else None

            result = await agent.async_execute("do something")

        assert result.success is True
        assert len(emitted) == 1
        assert emitted[0].context_window == 800_000
        # 640_000 / 800_000 * 100 = 80.0%
        assert abs(emitted[0].context_pct - 80.0) < 0.01


# ── TestConfigThresholds ──────────────────────────────────────


class TestConfigThresholds:
    """ToolLoopMixin reads warn/error thresholds from ContextWindowConfig, not hardcoded."""

    def test_custom_warn_threshold_changes_event_status(self) -> None:
        """Status reflects custom warn_threshold_pct from settings, not hardcoded 80."""
        from unittest.mock import patch

        from vaig.core.config import ContextWindowConfig, Settings

        # Set a non-default warn threshold
        settings = Settings()
        settings.context_window = ContextWindowConfig(
            warn_threshold_pct=50.0,
            error_threshold_pct=90.0,
        )

        mixin = _ConcreteLoopMixin()
        client = _make_mock_client()
        # 60% usage — above custom 50% warn, below default 80% warn
        client.generate_with_tools.return_value = _make_text_result(prompt_tokens=600_000)
        registry = ToolRegistry()

        emitted: list[ContextWindowChecked] = []

        with (
            patch("vaig.core.event_bus.EventBus.get") as mock_get,
            patch("vaig.agents.mixins.get_settings", return_value=settings),
        ):
            mock_bus = MagicMock()
            mock_get.return_value = mock_bus
            mock_bus.emit.side_effect = lambda e: emitted.append(e) if isinstance(e, ContextWindowChecked) else None

            mixin._run_tool_loop(
                client=client,
                prompt="hello",
                tool_registry=registry,
                system_instruction="test",
                history=[],
                context_window=1_000_000,
            )

        assert len(emitted) == 1
        # With custom 50% threshold, 60% should be "warning"
        assert emitted[0].status == "warning"

    def test_custom_error_threshold_changes_event_status(self) -> None:
        """Status reflects custom error_threshold_pct from settings, not hardcoded 95."""
        from unittest.mock import patch

        from vaig.core.config import ContextWindowConfig, Settings

        settings = Settings()
        settings.context_window = ContextWindowConfig(
            warn_threshold_pct=70.0,
            error_threshold_pct=85.0,
        )

        mixin = _ConcreteLoopMixin()
        client = _make_mock_client()
        # 90% usage — above custom 85% error threshold, below default 95% error threshold
        client.generate_with_tools.return_value = _make_text_result(prompt_tokens=900_000)
        registry = ToolRegistry()

        emitted: list[ContextWindowChecked] = []

        with (
            patch("vaig.core.event_bus.EventBus.get") as mock_get,
            patch("vaig.agents.mixins.get_settings", return_value=settings),
        ):
            mock_bus = MagicMock()
            mock_get.return_value = mock_bus
            mock_bus.emit.side_effect = lambda e: emitted.append(e) if isinstance(e, ContextWindowChecked) else None

            mixin._run_tool_loop(
                client=client,
                prompt="hello",
                tool_registry=registry,
                system_instruction="test",
                history=[],
                context_window=1_000_000,
            )

        assert len(emitted) == 1
        # With custom 85% error threshold, 90% should be "error"
        assert emitted[0].status == "error"

    @pytest.mark.asyncio
    async def test_async_custom_warn_threshold_honored(self) -> None:
        """Async loop also reads thresholds from config, not hardcoded values."""
        from unittest.mock import patch

        from vaig.core.config import ContextWindowConfig, Settings

        settings = Settings()
        settings.context_window = ContextWindowConfig(
            warn_threshold_pct=60.0,
            error_threshold_pct=90.0,
        )

        mixin = _ConcreteLoopMixin()
        client = _make_async_mock_client()
        # 70% usage — above custom 60% warn, below default 80% warn
        client.async_generate_with_tools.return_value = _make_text_result(prompt_tokens=700_000)
        registry = ToolRegistry()

        emitted: list[ContextWindowChecked] = []

        with (
            patch("vaig.core.event_bus.EventBus.get") as mock_get,
            patch("vaig.agents.mixins.get_settings", return_value=settings),
        ):
            mock_bus = MagicMock()
            mock_get.return_value = mock_bus
            mock_bus.emit.side_effect = lambda e: emitted.append(e) if isinstance(e, ContextWindowChecked) else None

            await mixin._async_run_tool_loop(
                client=client,
                prompt="hello",
                tool_registry=registry,
                system_instruction="test",
                history=[],
                context_window=1_000_000,
            )

        assert len(emitted) == 1
        # With custom 60% threshold, 70% should be "warning"
        assert emitted[0].status == "warning"


# ── TestContextWindowConfigValidation ────────────────────────


class TestContextWindowConfigValidation:
    """ContextWindowConfig validates threshold ordering and field ranges (C2)."""

    def test_warn_greater_than_error_raises(self) -> None:
        """warn_threshold_pct > error_threshold_pct must raise ValueError."""
        from pydantic import ValidationError

        from vaig.core.config import ContextWindowConfig

        with pytest.raises(ValidationError, match="warn_threshold_pct"):
            ContextWindowConfig(warn_threshold_pct=90.0, error_threshold_pct=80.0)

    def test_warn_equal_to_error_is_valid(self) -> None:
        """warn_threshold_pct == error_threshold_pct is allowed (boundary is inclusive)."""
        from vaig.core.config import ContextWindowConfig

        cfg = ContextWindowConfig(warn_threshold_pct=85.0, error_threshold_pct=85.0)
        assert cfg.warn_threshold_pct == 85.0
        assert cfg.error_threshold_pct == 85.0

    def test_threshold_below_zero_raises(self) -> None:
        """Threshold values must be >= 0.0."""
        from pydantic import ValidationError

        from vaig.core.config import ContextWindowConfig

        with pytest.raises(ValidationError):
            ContextWindowConfig(warn_threshold_pct=-1.0, error_threshold_pct=95.0)

    def test_threshold_above_100_raises(self) -> None:
        """Threshold values must be <= 100.0."""
        from pydantic import ValidationError

        from vaig.core.config import ContextWindowConfig

        with pytest.raises(ValidationError):
            ContextWindowConfig(warn_threshold_pct=80.0, error_threshold_pct=101.0)


# ── TestNarrowInvalidArgument ─────────────────────────────────


class TestNarrowInvalidArgument:
    """Non-context InvalidArgument errors propagate unchanged (C4)."""

    def test_non_context_invalid_argument_reraises_sync(self) -> None:
        """Sync loop re-raises InvalidArgument with non-context-window message."""
        from google.api_core.exceptions import InvalidArgument

        mixin = _ConcreteLoopMixin()
        client = _make_mock_client()
        # A generic auth/permission error — not a context-window failure.
        client.generate_with_tools.side_effect = InvalidArgument("API key not valid")
        registry = ToolRegistry()

        with pytest.raises(InvalidArgument, match="API key not valid"):
            mixin._run_tool_loop(
                client=client,
                prompt="hello",
                tool_registry=registry,
                system_instruction="test",
                history=[],
            )

    @pytest.mark.asyncio
    async def test_non_context_invalid_argument_reraises_async(self) -> None:
        """Async loop re-raises InvalidArgument with non-context-window message."""
        from google.api_core.exceptions import InvalidArgument

        mixin = _ConcreteLoopMixin()
        client = _make_async_mock_client()
        client.async_generate_with_tools.side_effect = InvalidArgument("API key not valid")
        registry = ToolRegistry()

        with pytest.raises(InvalidArgument, match="API key not valid"):
            await mixin._async_run_tool_loop(
                client=client,
                prompt="hello",
                tool_registry=registry,
                system_instruction="test",
                history=[],
            )

    def test_context_keyword_in_message_wraps_to_context_window_error(self) -> None:
        """All known context-window keyword variants are wrapped correctly."""
        from google.api_core.exceptions import InvalidArgument

        keywords = [
            "context window exceeded",
            "token limit reached",
            "max tokens exceeded",
            "maximum tokens reached",
            "prompt is too long",
            "too many tokens in request",
            "exceeds the maximum allowed size",
        ]
        for kw in keywords:
            mixin = _ConcreteLoopMixin()
            client = _make_mock_client()
            client.generate_with_tools.side_effect = InvalidArgument(kw)
            registry = ToolRegistry()

            with pytest.raises(ContextWindowExceededError):
                mixin._run_tool_loop(
                    client=client,
                    prompt="hello",
                    tool_registry=registry,
                    system_instruction="test",
                    history=[],
                )


# ── TestContextWindowConfigFallbackChain ─────────────────────


class TestContextWindowConfigFallbackChain:
    """_resolve_context_window uses the correct 3-level fallback (C1/C6)."""

    def test_model_specific_context_window_takes_priority(self) -> None:
        """Model's context_window in settings is used when available."""
        from unittest.mock import patch

        from vaig.core.config import ModelInfo, Settings

        settings = Settings()
        settings.models.available = [ModelInfo(id="gemini-big", context_window=2_000_000)]
        settings.context_window.context_window_size = 500_000

        _, client = _make_agent()
        registry = ToolRegistry()
        agent = ToolAwareAgent(
            system_instruction="test",
            tool_registry=registry,
            model="gemini-big",
            name="priority-agent",
            client=client,
        )

        with patch("vaig.agents.tool_aware.get_settings", return_value=settings):
            result = agent._resolve_context_window()

        assert result == 2_000_000

    def test_config_context_window_size_is_intermediate_fallback(self) -> None:
        """When model info has no custom context_window, config.context_window_size is used."""
        from unittest.mock import patch

        from vaig.core.config import ModelInfo, Settings

        settings = Settings()
        # Model entry exists but has no custom context_window (uses DEFAULT_CONTEXT_WINDOW).
        # After fix C1, the default-valued context_window is treated as "not set",
        # so the fallback chain falls through to context_window_size.
        settings.models.available = [ModelInfo(id="gemini-unknown")]
        settings.context_window.context_window_size = 600_000

        _, client = _make_agent()
        registry = ToolRegistry()
        agent = ToolAwareAgent(
            system_instruction="test",
            tool_registry=registry,
            model="gemini-unknown",
            name="fallback-agent",
            client=client,
        )

        with patch("vaig.agents.tool_aware.get_settings", return_value=settings):
            result = agent._resolve_context_window()

        assert result == 600_000

    def test_default_context_window_is_last_resort(self) -> None:
        """When get_settings() raises, DEFAULT_CONTEXT_WINDOW is returned."""
        from unittest.mock import patch

        _, client = _make_agent()
        registry = ToolRegistry()
        agent = ToolAwareAgent(
            system_instruction="test",
            tool_registry=registry,
            model="gemini-broken",
            name="broken-settings-agent",
            client=client,
        )

        with patch("vaig.agents.tool_aware.get_settings", side_effect=RuntimeError("settings broken")):
            result = agent._resolve_context_window()

        assert result == DEFAULT_CONTEXT_WINDOW


# ── TestUsagePropagation ──────────────────────────────────────


class TestUsagePropagation:
    """Accumulated usage is propagated through ContextWindowExceededError (C7)."""

    def test_usage_is_zeroed_when_error_on_first_iteration(self) -> None:
        """On first-iteration failure no tokens are accumulated — usage is zeros."""
        from google.api_core.exceptions import InvalidArgument

        agent, client = _make_agent()
        client.generate_with_tools.side_effect = InvalidArgument("context window exceeded")

        result = agent.execute("huge prompt")

        assert result.success is False
        assert result.usage["total_tokens"] == 0

    def test_usage_carried_through_context_window_exceeded_error(self) -> None:
        """Usage accumulated before a context-window error is preserved in result."""
        agent, _ = _make_agent()
        registry = ToolRegistry()

        exc = ContextWindowExceededError(
            "context window exceeded",
            context_pct=50.0,
            usage={"prompt_tokens": 500_000, "completion_tokens": 10, "total_tokens": 500_010},
        )
        result = agent._handle_context_window_exceeded(exc)

        assert result.success is False
        assert result.usage["prompt_tokens"] == 500_000
        assert result.usage["total_tokens"] == 500_010


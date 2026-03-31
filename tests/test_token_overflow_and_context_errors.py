"""Tests for token overflow prevention (Bug 2) and context error handling (Bug 3).

Covers:
- _merge_parallel_outputs: no truncation under budget, proportional truncation over budget
- _build_previous_agent_summary: no truncation under budget, truncation over budget
- _is_context_window_error in client.py: True for 400/413 with context keywords, False otherwise
- SpecialistAgent.execute catches ContextWindowExceededError → graceful degradation
- SpecialistAgent.async_execute catches ContextWindowExceededError → graceful degradation
- _monitor_context_window circuit breaker: raises at ≥95%, does NOT raise below
- Mixin sync tool loop catches genai_errors.ClientError(400) with context keywords
- Mixin async tool loop catches genai_errors.ClientError(400) with context keywords
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from vaig.agents.base import AgentConfig, AgentResult
from vaig.agents.mixins import CONTEXT_CIRCUIT_BREAKER_PCT, ToolLoopMixin
from vaig.agents.specialist import SpecialistAgent
from vaig.core.client import _CONTEXT_WINDOW_ERROR_KEYWORDS, _is_context_window_error
from vaig.core.exceptions import ContextWindowExceededError
from vaig.tools.base import ToolRegistry

# ── Helpers ──────────────────────────────────────────────────


def _make_mock_client() -> MagicMock:
    """Sync mock GeminiClient (satisfies GeminiClientProtocol)."""
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


def _make_specialist(name: str = "test-specialist") -> tuple[SpecialistAgent, MagicMock]:
    """Create a SpecialistAgent backed by a mock client."""
    client = _make_mock_client()
    config = AgentConfig(
        name=name,
        role="specialist",
        system_instruction="You are a test specialist.",
        model="gemini-2.5-pro",
    )
    agent = SpecialistAgent(config=config, client=client)
    return agent, client


def _make_async_specialist(name: str = "async-specialist") -> tuple[SpecialistAgent, MagicMock]:
    """Create a SpecialistAgent backed by an async mock client."""
    client = _make_async_mock_client()
    # Ensure the non-async generate also exists so the constructor won't fail.
    client.generate = MagicMock()
    config = AgentConfig(
        name=name,
        role="specialist",
        system_instruction="You are an async test specialist.",
        model="gemini-2.5-pro",
    )
    agent = SpecialistAgent(config=config, client=client)
    return agent, client


class _ConcreteLoopMixin(ToolLoopMixin):
    """Minimal concrete class to call _run_tool_loop directly."""

    pass


# ── Orchestrator import helper ──────────────────────────────
# We can't easily instantiate the full Orchestrator (it requires many
# collaborators), so we test the static-like methods by importing them
# and calling on a minimal mock.

def _make_mock_orchestrator() -> MagicMock:
    """Create a mock with the real _merge_parallel_outputs and _build_previous_agent_summary."""
    from vaig.agents.orchestrator import Orchestrator

    mock = MagicMock(spec=Orchestrator)
    # Bind the real methods to our mock.
    mock._merge_parallel_outputs = Orchestrator._merge_parallel_outputs.__get__(mock)
    mock._build_previous_agent_summary = Orchestrator._build_previous_agent_summary.__get__(mock)
    return mock


# ══════════════════════════════════════════════════════════════
# Bug 2 — Token Overflow Prevention
# ══════════════════════════════════════════════════════════════


class TestMergeParallelOutputs:
    """_merge_parallel_outputs truncation (Fix 2a)."""

    def test_no_truncation_under_budget(self) -> None:
        """Short sections are merged without any truncation."""
        orchestrator = _make_mock_orchestrator()
        results = [
            AgentResult(agent_name="alpha", content="Short output.", success=True),
            AgentResult(agent_name="beta", content="Also short.", success=True),
        ]
        merged = orchestrator._merge_parallel_outputs(results, max_chars=10_000)

        assert "--- alpha ---" in merged
        assert "--- beta ---" in merged
        assert "Short output." in merged
        assert "Also short." in merged
        assert "TRUNCATED" not in merged

    def test_proportional_truncation_over_budget(self) -> None:
        """When merged output exceeds max_chars, each section is truncated proportionally."""
        orchestrator = _make_mock_orchestrator()
        # Two agents each produce 500 chars → 1000+ chars total.
        long_content = "x" * 500
        results = [
            AgentResult(agent_name="alpha", content=long_content, success=True),
            AgentResult(agent_name="beta", content=long_content, success=True),
        ]
        merged = orchestrator._merge_parallel_outputs(results, max_chars=300)

        assert "TRUNCATED" in merged
        # Both sections should appear (even if truncated).
        assert "--- alpha ---" in merged
        assert "--- beta ---" in merged
        # The total length should be within a reasonable bound of the budget.
        # (The truncation marker adds some overhead, so we allow some slack.)
        assert len(merged) < 600  # Well below the 1000+ untruncated size.

    def test_empty_results_returns_empty(self) -> None:
        """Empty results list returns an empty string."""
        orchestrator = _make_mock_orchestrator()
        assert orchestrator._merge_parallel_outputs([]) == ""

    def test_failed_agents_included_with_error_note(self) -> None:
        """Failed agents produce an [ERROR] section, not a blank gap."""
        orchestrator = _make_mock_orchestrator()
        results = [
            AgentResult(agent_name="alpha", content="Good data.", success=True),
            AgentResult(agent_name="beta", content="connection timeout", success=False),
        ]
        merged = orchestrator._merge_parallel_outputs(results, max_chars=10_000)

        assert "[ERROR: Agent 'beta' failed" in merged


class TestBuildPreviousAgentSummary:
    """_build_previous_agent_summary truncation (Fix 2b)."""

    def test_no_truncation_under_budget(self) -> None:
        """Short content passes through without truncation."""
        orchestrator = _make_mock_orchestrator()
        prev = AgentResult(
            agent_name="analyzer",
            content="Short analysis.",
            success=True,
            metadata={},
        )
        summary = orchestrator._build_previous_agent_summary("analyzer", prev, max_chars=10_000)

        assert "Short analysis." in summary
        assert "TRUNCATED" not in summary
        assert "Previous Analysis (analyzer)" in summary

    def test_truncation_over_budget(self) -> None:
        """Long content is truncated with the TRUNCATED marker."""
        orchestrator = _make_mock_orchestrator()
        long_content = "y" * 5_000
        prev = AgentResult(
            agent_name="analyzer",
            content=long_content,
            success=True,
            metadata={},
        )
        summary = orchestrator._build_previous_agent_summary("analyzer", prev, max_chars=500)

        assert "TRUNCATED" in summary
        assert "Previous Analysis (analyzer)" in summary
        # The content portion should be approximately max_chars long.
        assert len(summary) < len(long_content) + 500  # Definitely smaller than untruncated.


# ══════════════════════════════════════════════════════════════
# Bug 2 — Circuit Breaker in _monitor_context_window (Fix 2c)
# ══════════════════════════════════════════════════════════════


class TestMonitorContextWindowCircuitBreaker:
    """_monitor_context_window raises ContextWindowExceededError at ≥95%."""

    def test_raises_at_circuit_breaker_threshold(self) -> None:
        """Circuit breaker fires when usage >= CONTEXT_CIRCUIT_BREAKER_PCT."""
        mixin = _ConcreteLoopMixin()
        # 960K / 1M = 96% → above 95% breaker.
        result = _make_text_result(prompt_tokens=960_000)

        with pytest.raises(ContextWindowExceededError, match="circuit breaker"):
            mixin._monitor_context_window(
                result=result,
                context_window=1_000_000,
                peak_context_pct=0.0,
                iteration=1,
                model="gemini-2.5-pro",
                warn_threshold=80.0,
                error_threshold=95.0,
            )

    def test_does_not_raise_below_threshold(self) -> None:
        """No exception below the circuit breaker threshold."""
        mixin = _ConcreteLoopMixin()
        # 800K / 1M = 80% → below 95% breaker.
        result = _make_text_result(prompt_tokens=800_000)

        new_peak = mixin._monitor_context_window(
            result=result,
            context_window=1_000_000,
            peak_context_pct=0.0,
            iteration=1,
            model="gemini-2.5-pro",
            warn_threshold=80.0,
            error_threshold=95.0,
        )
        assert abs(new_peak - 80.0) < 0.01

    def test_circuit_breaker_uses_max_of_error_and_constant(self) -> None:
        """Breaker fires at max(error_threshold, CONTEXT_CIRCUIT_BREAKER_PCT)."""
        mixin = _ConcreteLoopMixin()
        # If error_threshold=98% > 95%, breaker should use 98%.
        # 960K / 1M = 96% → between 95% and 98% → should NOT fire.
        result = _make_text_result(prompt_tokens=960_000)

        new_peak = mixin._monitor_context_window(
            result=result,
            context_window=1_000_000,
            peak_context_pct=0.0,
            iteration=1,
            model="gemini-2.5-pro",
            warn_threshold=80.0,
            error_threshold=98.0,  # Higher than CONTEXT_CIRCUIT_BREAKER_PCT.
        )
        assert abs(new_peak - 96.0) < 0.01

    def test_circuit_breaker_constant_value(self) -> None:
        """CONTEXT_CIRCUIT_BREAKER_PCT is 95.0."""
        assert CONTEXT_CIRCUIT_BREAKER_PCT == 95.0


# ══════════════════════════════════════════════════════════════
# Bug 3 — _is_context_window_error in client.py (Fix 3a)
# ══════════════════════════════════════════════════════════════


class TestIsContextWindowError:
    """_is_context_window_error correctly identifies context-window errors."""

    def test_true_for_400_with_context_keywords(self) -> None:
        """Returns True for code=400 with known context-window keyword."""
        from google.genai import errors as genai_errors

        for kw in _CONTEXT_WINDOW_ERROR_KEYWORDS:
            exc = genai_errors.ClientError(400, f"Request failed: {kw} in this request")
            assert _is_context_window_error(exc) is True, f"Expected True for keyword: {kw}"

    def test_true_for_413_with_context_keywords(self) -> None:
        """Returns True for code=413 (payload too large) with context keywords."""
        from google.genai import errors as genai_errors

        exc = genai_errors.ClientError(413, "request payload size exceeds limit")
        assert _is_context_window_error(exc) is True

    def test_false_for_400_without_context_keywords(self) -> None:
        """Returns False for code=400 with non-context error message."""
        from google.genai import errors as genai_errors

        exc = genai_errors.ClientError(400, "API key not valid")
        assert _is_context_window_error(exc) is False

    def test_false_for_non_400_codes(self) -> None:
        """Returns False for non-400/413 status codes even with context keywords."""
        from google.genai import errors as genai_errors

        for code in (403, 404, 500, 503):
            exc = genai_errors.ClientError(code, "context window exceeded")
            assert _is_context_window_error(exc) is False, f"Expected False for code={code}"

    def test_case_insensitive_matching(self) -> None:
        """Keyword matching is case-insensitive."""
        from google.genai import errors as genai_errors

        exc = genai_errors.ClientError(400, "CONTEXT WINDOW EXCEEDED by request")
        assert _is_context_window_error(exc) is True


# ══════════════════════════════════════════════════════════════
# Bug 3 — SpecialistAgent catches ContextWindowExceededError (Fix 3b)
# ══════════════════════════════════════════════════════════════


class TestSpecialistContextWindowHandling:
    """SpecialistAgent.execute catches ContextWindowExceededError → graceful degradation."""

    def test_execute_returns_failed_result_on_context_exceeded(self) -> None:
        """Sync execute catches ContextWindowExceededError and returns a failed AgentResult."""
        agent, client = _make_specialist()
        client.generate.side_effect = ContextWindowExceededError(
            "context too big",
            context_pct=99.0,
            usage={"prompt_tokens": 990_000, "completion_tokens": 0, "total_tokens": 990_000},
        )

        result = agent.execute("analyze this")

        assert result.success is False
        assert result.agent_name == "test-specialist"
        assert "context" in result.content.lower() or "token limit" in result.content.lower()
        assert result.metadata.get("error_type") == "context_window_exceeded"
        assert result.usage["prompt_tokens"] == 990_000

    @pytest.mark.asyncio
    async def test_async_execute_returns_failed_result_on_context_exceeded(self) -> None:
        """Async execute catches ContextWindowExceededError and returns a failed AgentResult."""
        agent, client = _make_async_specialist()
        client.async_generate = AsyncMock(
            side_effect=ContextWindowExceededError(
                "async context too big",
                context_pct=98.0,
                usage={"prompt_tokens": 980_000, "completion_tokens": 0, "total_tokens": 980_000},
            ),
        )

        result = await agent.async_execute("async analyze this")

        assert result.success is False
        assert result.agent_name == "async-specialist"
        assert "context" in result.content.lower() or "token limit" in result.content.lower()
        assert result.metadata.get("error_type") == "context_window_exceeded"
        assert result.usage["prompt_tokens"] == 980_000


# ══════════════════════════════════════════════════════════════
# Bug 3 — Mixin catches genai_errors.ClientError(400) (Fix 3c)
# ══════════════════════════════════════════════════════════════


class TestMixinClientErrorConversion:
    """Mixin tool loops convert genai ClientError(400) to ContextWindowExceededError."""

    def test_sync_loop_converts_client_error_400_with_context_keywords(self) -> None:
        """Sync tool loop converts ClientError(400) with context keywords to ContextWindowExceededError."""
        from google.genai import errors as genai_errors

        mixin = _ConcreteLoopMixin()
        client = _make_mock_client()
        client.generate_with_tools.side_effect = genai_errors.ClientError(
            400,
            "Request failed: content too large for model context window",
        )
        registry = ToolRegistry()

        with pytest.raises(ContextWindowExceededError, match="genai ClientError 400"):
            mixin._run_tool_loop(
                client=client,
                prompt="big prompt",
                tool_registry=registry,
                system_instruction="test",
                history=[],
            )

    @pytest.mark.asyncio
    async def test_async_loop_converts_client_error_400_with_context_keywords(self) -> None:
        """Async tool loop converts ClientError(400) with context keywords to ContextWindowExceededError."""
        from google.genai import errors as genai_errors

        mixin = _ConcreteLoopMixin()
        client = _make_async_mock_client()
        client.async_generate_with_tools.side_effect = genai_errors.ClientError(
            400,
            "Request failed: content too large for model context window",
        )
        registry = ToolRegistry()

        with pytest.raises(ContextWindowExceededError, match="genai ClientError 400"):
            await mixin._async_run_tool_loop(
                client=client,
                prompt="big prompt",
                tool_registry=registry,
                system_instruction="test",
                history=[],
            )

    def test_sync_loop_does_not_convert_client_error_400_without_context_keywords(self) -> None:
        """ClientError(400) without context keywords propagates unchanged."""
        from google.genai import errors as genai_errors

        mixin = _ConcreteLoopMixin()
        client = _make_mock_client()
        client.generate_with_tools.side_effect = genai_errors.ClientError(
            400,
            "Invalid function calling configuration",
        )
        registry = ToolRegistry()

        with pytest.raises(genai_errors.ClientError, match="Invalid function calling"):
            mixin._run_tool_loop(
                client=client,
                prompt="normal prompt",
                tool_registry=registry,
                system_instruction="test",
                history=[],
            )

    def test_sync_loop_does_not_convert_client_error_non_400(self) -> None:
        """ClientError with non-400 code propagates unchanged regardless of message."""
        from google.genai import errors as genai_errors

        mixin = _ConcreteLoopMixin()
        client = _make_mock_client()
        client.generate_with_tools.side_effect = genai_errors.ClientError(
            403,
            "context window exceeded but wrong code",
        )
        registry = ToolRegistry()

        with pytest.raises(genai_errors.ClientError):
            mixin._run_tool_loop(
                client=client,
                prompt="prompt",
                tool_registry=registry,
                system_instruction="test",
                history=[],
            )


# ══════════════════════════════════════════════════════════════
# Integration: Circuit Breaker Fires During Tool Loop
# ══════════════════════════════════════════════════════════════


class TestCircuitBreakerInToolLoop:
    """Circuit breaker inside the tool loop terminates the loop at ≥95%."""

    def test_sync_loop_raises_on_circuit_breaker(self) -> None:
        """Sync tool loop raises ContextWindowExceededError when usage hits circuit breaker."""
        mixin = _ConcreteLoopMixin()
        client = _make_mock_client()
        # First call returns a result that is 96% usage → trips the breaker.
        client.generate_with_tools.return_value = _make_text_result(prompt_tokens=960_000)
        registry = ToolRegistry()

        with pytest.raises(ContextWindowExceededError, match="circuit breaker"):
            mixin._run_tool_loop(
                client=client,
                prompt="hello",
                tool_registry=registry,
                system_instruction="test",
                history=[],
                context_window=1_000_000,
            )

    @pytest.mark.asyncio
    async def test_async_loop_raises_on_circuit_breaker(self) -> None:
        """Async tool loop raises ContextWindowExceededError when usage hits circuit breaker."""
        mixin = _ConcreteLoopMixin()
        client = _make_async_mock_client()
        client.async_generate_with_tools.return_value = _make_text_result(prompt_tokens=960_000)
        registry = ToolRegistry()

        with pytest.raises(ContextWindowExceededError, match="circuit breaker"):
            await mixin._async_run_tool_loop(
                client=client,
                prompt="hello",
                tool_registry=registry,
                system_instruction="test",
                history=[],
                context_window=1_000_000,
            )

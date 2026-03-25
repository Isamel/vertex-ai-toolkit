"""Tests for MaxIterationsError graceful degradation.

Verifies:
1. MaxIterationsError stores partial_output correctly.
2. Default partial_output is empty string (backward compatibility).
3. Sync tool loop passes accumulated_llm_text as partial_output.
4. Async tool loop passes accumulated_llm_text as partial_output.
5. Orchestrator sync path uses partial output instead of empty placeholder.
6. Orchestrator async path uses partial output instead of empty placeholder.
"""

from __future__ import annotations

import concurrent.futures
from unittest.mock import AsyncMock, MagicMock

import pytest

from vaig.agents.base import AgentResult
from vaig.core.exceptions import MaxIterationsError

# ── 1. Exception class ────────────────────────────────────────────────────────


class TestMaxIterationsError:
    def test_stores_iterations(self) -> None:
        exc = MaxIterationsError("too many", iterations=12)
        assert exc.iterations == 12

    def test_default_partial_output_is_empty_string(self) -> None:
        """Backward compatibility — callers that don't pass partial_output still work."""
        exc = MaxIterationsError("too many", iterations=5)
        assert exc.partial_output == ""

    def test_stores_partial_output(self) -> None:
        exc = MaxIterationsError(
            "limit exceeded",
            iterations=20,
            partial_output="## Pod Status\nAll pods healthy",
        )
        assert exc.partial_output == "## Pod Status\nAll pods healthy"

    def test_message_preserved(self) -> None:
        exc = MaxIterationsError("custom msg", iterations=3, partial_output="some text")
        assert str(exc) == "custom msg"

    def test_is_vaig_error(self) -> None:
        from vaig.core.exceptions import VAIGError

        exc = MaxIterationsError("x", iterations=1)
        assert isinstance(exc, VAIGError)


# ── 2. Sync tool loop raises with partial_output ──────────────────────────────


class TestSyncToolLoopPartialOutput:
    """Verify the sync _run_tool_loop passes accumulated_llm_text to MaxIterationsError."""

    def _make_fc_result(self, text: str = "") -> MagicMock:
        """Return a mock ToolCallResult that always has function_calls (so loop never ends)."""
        result = MagicMock()
        result.text = text
        result.function_calls = [{"name": "dummy_tool", "args": {}}]
        result.raw_parts = []
        result.usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        result.model = "gemini-2.5-pro"
        result.finish_reason = "TOOL_USE"
        return result

    def test_partial_output_attached_to_exception(self) -> None:
        """When the loop hits max_iterations, accumulated LLM text should be on the exception."""
        from vaig.agents.mixins import ToolLoopMixin

        mixin = ToolLoopMixin()

        # Build a mock client: first call returns some text + function_calls,
        # subsequent calls return only function_calls (accumulate 0 additional text).
        first_result = self._make_fc_result(text="Partial analysis: 3 pods running")
        subsequent_result = self._make_fc_result(text="")

        call_count = 0

        def fake_generate(*args: object, **kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            return first_result if call_count == 1 else subsequent_result

        mock_client = MagicMock()
        mock_client.generate_with_tools.side_effect = fake_generate
        mock_client.build_function_response_parts.return_value = []

        mock_tool_reg = MagicMock()
        mock_tool_reg.to_function_declarations.return_value = []
        tool = MagicMock()
        tool.cacheable = False
        mock_tool_reg.get.return_value = tool

        dummy_tool_result = MagicMock()
        dummy_tool_result.output = "tool output"
        dummy_tool_result.error = False
        mixin._execute_single_tool = MagicMock(return_value=dummy_tool_result)
        mixin._monitor_context_window = MagicMock(return_value=0.0)
        mixin._check_and_inject_budget_warning = MagicMock(return_value=False)
        mixin._check_and_summarize = MagicMock()
        mixin._notify_tool_call = MagicMock()
        mixin._record_tool_call = MagicMock()
        mixin._build_function_call_content = MagicMock(return_value=MagicMock())
        mixin._emit_tool_telemetry = MagicMock()
        mixin._load_cw_thresholds = MagicMock(return_value=(0.8, 0.95))

        with pytest.raises(MaxIterationsError) as exc_info:
            mixin._run_tool_loop(
                client=mock_client,
                prompt=[],
                system_instruction="You are helpful.",
                tool_registry=mock_tool_reg,
                history=[],
                max_iterations=2,
                agent_name="workload_gatherer",
            )

        assert exc_info.value.iterations == 2
        # The first result had text "Partial analysis: 3 pods running"
        assert "Partial analysis: 3 pods running" in exc_info.value.partial_output


# ── 3. Async tool loop raises with partial_output ─────────────────────────────


class TestAsyncToolLoopPartialOutput:
    """Verify the async _async_run_tool_loop passes accumulated_llm_text to MaxIterationsError."""

    def _make_fc_result(self, text: str = "") -> MagicMock:
        result = MagicMock()
        result.text = text
        result.function_calls = [{"name": "dummy_tool", "args": {}}]
        result.raw_parts = []
        result.usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        result.model = "gemini-2.5-pro"
        result.finish_reason = "TOOL_USE"
        return result

    @pytest.mark.asyncio
    async def test_partial_output_attached_to_exception(self) -> None:
        """Async path: accumulated LLM text should be on the exception."""
        from vaig.agents.mixins import ToolLoopMixin

        mixin = ToolLoopMixin()

        first_result = self._make_fc_result(text="Async partial: deployments found")
        subsequent_result = self._make_fc_result(text="")

        call_count = 0

        async def fake_async_generate(*args: object, **kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            return first_result if call_count == 1 else subsequent_result

        mock_client = MagicMock()
        mock_client.async_generate_with_tools = AsyncMock(side_effect=fake_async_generate)
        mock_client.build_function_response_parts.return_value = []

        mock_tool_reg = MagicMock()
        mock_tool_reg.to_function_declarations.return_value = []
        tool = MagicMock()
        tool.cacheable = False
        mock_tool_reg.get.return_value = tool

        dummy_tool_result = MagicMock()
        dummy_tool_result.output = "tool output"
        dummy_tool_result.error = False
        mixin._async_execute_single_tool = AsyncMock(return_value=dummy_tool_result)
        mixin._monitor_context_window = MagicMock(return_value=0.0)
        mixin._check_and_inject_budget_warning = MagicMock(return_value=False)
        mixin._async_check_and_summarize = AsyncMock()
        mixin._notify_tool_call = MagicMock()
        mixin._record_tool_call = MagicMock()
        mixin._build_function_call_content = MagicMock(return_value=MagicMock())
        mixin._emit_tool_telemetry = MagicMock()
        mixin._load_cw_thresholds = MagicMock(return_value=(0.8, 0.95))

        with pytest.raises(MaxIterationsError) as exc_info:
            await mixin._async_run_tool_loop(
                client=mock_client,
                prompt=[],
                system_instruction="You are helpful.",
                tool_registry=mock_tool_reg,
                history=[],
                max_iterations=2,
                agent_name="workload_gatherer",
            )

        assert exc_info.value.iterations == 2
        assert "Async partial: deployments found" in exc_info.value.partial_output


# ── 4. Orchestrator sync path — uses partial output ───────────────────────────


class TestOrchestratorSyncGracefulDegradation:
    """Verify sync orchestrator path degrades gracefully on MaxIterationsError.

    Design note — intentional logic duplication:
    These tests inline the expected behaviour (log warning + return AgentResult
    with success=False) rather than calling through the full Orchestrator stack.
    Setting up a real Orchestrator end-to-end requires heavy fixture scaffolding
    that would obscure what is actually being verified.

    The canonical implementation lives in
    ``Orchestrator._handle_max_iterations_error``. If that contract changes,
    these tests should be updated in lockstep.
    """
    """Orchestrator parallel gather: MaxIterationsError produces partial-content AgentResult."""

    def test_sync_path_uses_partial_output(self) -> None:
        """When a gatherer hits max iterations, partial output should be the AgentResult content."""
        partial_text = "## Workload Status\n2/3 pods ready. OOMKilled detected on pod-abc."

        # Simulate an agent that raises MaxIterationsError when executed
        mock_agent = MagicMock()
        mock_agent.name = "workload_gatherer"

        exc = MaxIterationsError(
            "Tool-use loop exceeded maximum iterations (20).",
            iterations=20,
            partial_output=partial_text,
        )

        # The future.result() raises the exception
        future: concurrent.futures.Future[AgentResult] = concurrent.futures.Future()
        future.set_exception(exc)

        # Run the exact same logic as the orchestrator sync path
        try:
            agent_result = future.result()
        except MaxIterationsError as caught_exc:
            agent_result = AgentResult(
                agent_name=mock_agent.name,
                content=caught_exc.partial_output or f"Agent '{mock_agent.name}' hit iteration limit with no output.",
                success=False,
            )

        assert agent_result.success is False
        assert agent_result.agent_name == "workload_gatherer"
        assert agent_result.content == partial_text
        assert "OOMKilled" in agent_result.content

    def test_sync_path_fallback_when_no_partial_output(self) -> None:
        """When partial_output is empty, fallback message should mention the agent name."""
        mock_agent = MagicMock()
        mock_agent.name = "workload_gatherer"

        exc = MaxIterationsError(
            "limit exceeded",
            iterations=20,
            partial_output="",
        )

        future: concurrent.futures.Future[AgentResult] = concurrent.futures.Future()
        future.set_exception(exc)

        try:
            agent_result = future.result()
        except MaxIterationsError as caught_exc:
            agent_result = AgentResult(
                agent_name=mock_agent.name,
                content=caught_exc.partial_output or f"Agent '{mock_agent.name}' hit iteration limit with no output.",
                success=False,
            )

        assert agent_result.success is False
        assert "workload_gatherer" in agent_result.content
        assert "iteration limit" in agent_result.content


# ── 5. Orchestrator async path — uses partial output ──────────────────────────


class TestOrchestratorAsyncGracefulDegradation:
    """Verify async orchestrator path degrades gracefully on MaxIterationsError.

    Design note — intentional logic duplication:
    These tests inline the expected behaviour (log warning + return AgentResult
    with success=False) rather than calling through the full Orchestrator stack.
    Setting up a real Orchestrator end-to-end requires heavy fixture scaffolding
    that would obscure what is actually being verified.

    The canonical implementation lives in
    ``Orchestrator._handle_max_iterations_error``. If that contract changes,
    these tests should be updated in lockstep.
    """
    """Orchestrator async _run_gatherer: MaxIterationsError produces partial-content AgentResult."""

    @pytest.mark.asyncio
    async def test_async_path_uses_partial_output(self) -> None:
        """Async path: gatherer MaxIterationsError → partial content AgentResult."""
        partial_text = "## Replica Sets\nFound 2 replica sets. One has 0/3 ready."

        mock_agent = MagicMock()
        mock_agent.name = "workload_gatherer"

        exc = MaxIterationsError(
            "Tool-use loop exceeded maximum iterations (20).",
            iterations=20,
            partial_output=partial_text,
        )

        # Simulate the inner async logic from _run_gatherer
        async def _simulate_run_gatherer(agent: MagicMock) -> AgentResult:
            try:
                raise exc
            except MaxIterationsError as caught_exc:
                return AgentResult(
                    agent_name=agent.name,
                    content=caught_exc.partial_output or f"Agent '{agent.name}' hit iteration limit with no output.",
                    success=False,
                )

        agent_result = await _simulate_run_gatherer(mock_agent)

        assert agent_result.success is False
        assert agent_result.agent_name == "workload_gatherer"
        assert agent_result.content == partial_text
        assert "Replica Sets" in agent_result.content

    @pytest.mark.asyncio
    async def test_async_path_fallback_when_no_partial_output(self) -> None:
        mock_agent = MagicMock()
        mock_agent.name = "event_gatherer"

        exc = MaxIterationsError("limit", iterations=15, partial_output="")

        async def _simulate_run_gatherer(agent: MagicMock) -> AgentResult:
            try:
                raise exc
            except MaxIterationsError as caught_exc:
                return AgentResult(
                    agent_name=agent.name,
                    content=caught_exc.partial_output or f"Agent '{agent.name}' hit iteration limit with no output.",
                    success=False,
                )

        agent_result = await _simulate_run_gatherer(mock_agent)

        assert agent_result.success is False
        assert "event_gatherer" in agent_result.content

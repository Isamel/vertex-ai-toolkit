"""Tests for ToolLoopMixin async methods — _async_run_tool_loop and _async_execute_single_tool."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vaig.agents.mixins import ToolLoopMixin, ToolLoopResult
from vaig.core.client import ToolCallResult
from vaig.core.exceptions import MaxIterationsError
from vaig.tools.base import ToolDef, ToolParam, ToolRegistry, ToolResult


# ── Helpers ──────────────────────────────────────────────────


def _make_registry(*tools: ToolDef) -> ToolRegistry:
    """Create a ToolRegistry pre-loaded with the given tools."""
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def _make_text_result(text: str, *, model: str = "gemini-2.5-pro") -> ToolCallResult:
    """Create a ToolCallResult that has text (no function calls)."""
    return ToolCallResult(
        text=text,
        model=model,
        function_calls=[],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        finish_reason="STOP",
    )


def _make_fc_result(
    calls: list[dict[str, Any]],
    *,
    model: str = "gemini-2.5-pro",
) -> ToolCallResult:
    """Create a ToolCallResult with function calls (no text)."""
    return ToolCallResult(
        text="",
        model=model,
        function_calls=calls,
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        finish_reason="STOP",
    )


class MixinHost(ToolLoopMixin):
    """Concrete class that inherits ToolLoopMixin for testing."""

    pass


# ══════════════════════════════════════════════════════════════
# _async_execute_single_tool
# ══════════════════════════════════════════════════════════════


class TestAsyncExecuteSingleTool:
    """Tests for the async single-tool execution method."""

    async def test_executes_sync_tool_via_to_thread(self) -> None:
        """Sync tool.execute should be wrapped and run in a thread."""

        def sync_execute(query: str) -> ToolResult:
            return ToolResult(output=f"result: {query}")

        tool = ToolDef(
            name="search",
            description="Search tool",
            parameters=[ToolParam(name="query", type="string", description="q")],
            execute=sync_execute,
        )
        registry = _make_registry(tool)
        host = MixinHost()

        result = await host._async_execute_single_tool(registry, "search", {"query": "test"})

        assert result.output == "result: test"
        assert result.error is False

    async def test_executes_async_tool_directly(self) -> None:
        """Natively async tool.execute should be awaited directly."""

        async def async_execute(query: str) -> ToolResult:
            return ToolResult(output=f"async result: {query}")

        tool = ToolDef(
            name="async_search",
            description="Async search tool",
            parameters=[ToolParam(name="query", type="string", description="q")],
            execute=async_execute,
        )
        registry = _make_registry(tool)
        host = MixinHost()

        result = await host._async_execute_single_tool(registry, "async_search", {"query": "hello"})

        assert result.output == "async result: hello"
        assert result.error is False

    async def test_unknown_tool_returns_error(self) -> None:
        """Requesting a tool not in the registry should return error ToolResult."""
        registry = _make_registry()
        host = MixinHost()

        result = await host._async_execute_single_tool(registry, "nonexistent", {"a": 1})

        assert result.error is True
        assert "Unknown tool: nonexistent" in result.output

    async def test_type_error_returns_error_result(self) -> None:
        """TypeError during execution should be caught and returned as error."""

        def bad_tool(x: int) -> ToolResult:
            msg = "wrong args"
            raise TypeError(msg)

        tool = ToolDef(
            name="bad",
            description="Bad tool",
            parameters=[ToolParam(name="x", type="integer", description="num")],
            execute=bad_tool,
        )
        registry = _make_registry(tool)
        host = MixinHost()

        result = await host._async_execute_single_tool(registry, "bad", {"x": 1})

        assert result.error is True
        assert "Invalid arguments for bad" in result.output

    async def test_unexpected_error_returns_error_result(self) -> None:
        """Generic exceptions should be caught and returned as error."""

        def failing_tool() -> ToolResult:
            msg = "disk full"
            raise OSError(msg)

        tool = ToolDef(
            name="write_file",
            description="Writes a file",
            parameters=[],
            execute=failing_tool,
        )
        registry = _make_registry(tool)
        host = MixinHost()

        result = await host._async_execute_single_tool(registry, "write_file", {})

        assert result.error is True
        assert "Tool execution error (write_file)" in result.output
        assert "disk full" in result.output


# ══════════════════════════════════════════════════════════════
# _async_run_tool_loop
# ══════════════════════════════════════════════════════════════


class TestAsyncRunToolLoop:
    """Tests for the async tool-use loop."""

    async def test_immediate_text_response(self) -> None:
        """When the LLM returns text on the first call, loop exits immediately."""
        client = MagicMock()
        client.async_generate_with_tools = AsyncMock(
            return_value=_make_text_result("Hello, world!"),
        )

        host = MixinHost()
        history: list[Any] = []

        result = await host._async_run_tool_loop(
            client=client,
            prompt="Hi",
            tool_registry=_make_registry(),
            system_instruction="You are helpful.",
            history=history,
        )

        assert isinstance(result, ToolLoopResult)
        assert result.text == "Hello, world!"
        assert result.iterations == 1
        assert result.tools_executed == []
        assert result.model == "gemini-2.5-pro"
        assert result.finish_reason == "STOP"
        client.async_generate_with_tools.assert_awaited_once()

    async def test_single_tool_call_then_text(self) -> None:
        """LLM calls one tool, gets the result, then returns text."""

        def search(query: str) -> ToolResult:
            return ToolResult(output=f"Found: {query}")

        tool = ToolDef(
            name="search",
            description="Search",
            parameters=[ToolParam(name="query", type="string", description="q")],
            execute=search,
        )
        registry = _make_registry(tool)

        client = MagicMock()
        client.async_generate_with_tools = AsyncMock(
            side_effect=[
                # Iteration 1: LLM requests tool call
                _make_fc_result([{"name": "search", "args": {"query": "python"}}]),
                # Iteration 2: LLM returns text
                _make_text_result("Based on the search, Python is great."),
            ],
        )

        host = MixinHost()
        history: list[Any] = []

        result = await host._async_run_tool_loop(
            client=client,
            prompt="Tell me about Python",
            tool_registry=registry,
            system_instruction="You are helpful.",
            history=history,
        )

        assert result.text == "Based on the search, Python is great."
        assert result.iterations == 2
        assert len(result.tools_executed) == 1
        assert result.tools_executed[0]["name"] == "search"
        assert result.tools_executed[0]["error"] is False
        # History should have function call + function response entries
        assert len(history) >= 2

    async def test_multiple_tool_calls_in_one_turn(self) -> None:
        """LLM requests multiple tool calls in a single response."""

        def tool_a(x: str) -> ToolResult:
            return ToolResult(output=f"A:{x}")

        def tool_b(y: str) -> ToolResult:
            return ToolResult(output=f"B:{y}")

        registry = _make_registry(
            ToolDef(name="a", description="A", parameters=[ToolParam(name="x", type="string", description="x")], execute=tool_a),
            ToolDef(name="b", description="B", parameters=[ToolParam(name="y", type="string", description="y")], execute=tool_b),
        )

        client = MagicMock()
        client.async_generate_with_tools = AsyncMock(
            side_effect=[
                _make_fc_result([
                    {"name": "a", "args": {"x": "1"}},
                    {"name": "b", "args": {"y": "2"}},
                ]),
                _make_text_result("Done with both."),
            ],
        )

        host = MixinHost()
        history: list[Any] = []

        result = await host._async_run_tool_loop(
            client=client,
            prompt="Do both",
            tool_registry=registry,
            system_instruction="sys",
            history=history,
        )

        assert result.text == "Done with both."
        assert len(result.tools_executed) == 2
        assert result.tools_executed[0]["name"] == "a"
        assert result.tools_executed[1]["name"] == "b"

    async def test_max_iterations_raises(self) -> None:
        """Exceeding max_iterations raises MaxIterationsError."""
        client = MagicMock()
        # Always return function calls — loop never exits normally
        client.async_generate_with_tools = AsyncMock(
            return_value=_make_fc_result([{"name": "noop", "args": {}}]),
        )

        def noop() -> ToolResult:
            return ToolResult(output="ok")

        registry = _make_registry(
            ToolDef(name="noop", description="No-op", parameters=[], execute=noop),
        )

        host = MixinHost()
        history: list[Any] = []

        with pytest.raises(MaxIterationsError) as exc_info:
            await host._async_run_tool_loop(
                client=client,
                prompt="loop forever",
                tool_registry=registry,
                system_instruction="sys",
                history=history,
                max_iterations=3,
            )

        assert exc_info.value.iterations == 3

    async def test_api_error_propagates(self) -> None:
        """Errors from the LLM client should propagate out of the loop."""
        client = MagicMock()
        client.async_generate_with_tools = AsyncMock(
            side_effect=RuntimeError("API down"),
        )

        host = MixinHost()
        history: list[Any] = []

        with pytest.raises(RuntimeError, match="API down"):
            await host._async_run_tool_loop(
                client=client,
                prompt="fail",
                tool_registry=_make_registry(),
                system_instruction="sys",
                history=history,
            )

    async def test_usage_accumulates_across_iterations(self) -> None:
        """Token usage should sum across all iterations."""

        def echo(msg: str) -> ToolResult:
            return ToolResult(output=msg)

        registry = _make_registry(
            ToolDef(name="echo", description="Echo", parameters=[ToolParam(name="msg", type="string", description="m")], execute=echo),
        )

        client = MagicMock()
        client.async_generate_with_tools = AsyncMock(
            side_effect=[
                _make_fc_result([{"name": "echo", "args": {"msg": "a"}}]),
                _make_fc_result([{"name": "echo", "args": {"msg": "b"}}]),
                _make_text_result("final"),
            ],
        )

        host = MixinHost()
        history: list[Any] = []

        result = await host._async_run_tool_loop(
            client=client,
            prompt="go",
            tool_registry=registry,
            system_instruction="sys",
            history=history,
        )

        # 3 iterations × 10 prompt_tokens each = 30
        assert result.usage["prompt_tokens"] == 30
        assert result.usage["completion_tokens"] == 15
        assert result.usage["total_tokens"] == 45

    async def test_tool_error_fed_back_to_model(self) -> None:
        """When a tool fails, the error is passed back to the LLM so it can self-correct."""

        def failing_tool(query: str) -> ToolResult:
            msg = "connection refused"
            raise OSError(msg)

        registry = _make_registry(
            ToolDef(name="fetch", description="Fetch", parameters=[ToolParam(name="query", type="string", description="q")], execute=failing_tool),
        )

        client = MagicMock()
        client.async_generate_with_tools = AsyncMock(
            side_effect=[
                _make_fc_result([{"name": "fetch", "args": {"query": "url"}}]),
                _make_text_result("Sorry, the fetch failed."),
            ],
        )

        host = MixinHost()
        history: list[Any] = []

        result = await host._async_run_tool_loop(
            client=client,
            prompt="fetch data",
            tool_registry=registry,
            system_instruction="sys",
            history=history,
        )

        assert result.text == "Sorry, the fetch failed."
        assert result.tools_executed[0]["error"] is True
        assert "connection refused" in result.tools_executed[0]["output"]

    async def test_async_tool_in_loop(self) -> None:
        """Natively async tools should be awaited directly inside the loop."""

        async def async_tool(query: str) -> ToolResult:
            return ToolResult(output=f"async:{query}")

        registry = _make_registry(
            ToolDef(name="async_fetch", description="Async fetch", parameters=[ToolParam(name="query", type="string", description="q")], execute=async_tool),
        )

        client = MagicMock()
        client.async_generate_with_tools = AsyncMock(
            side_effect=[
                _make_fc_result([{"name": "async_fetch", "args": {"query": "data"}}]),
                _make_text_result("Got async data."),
            ],
        )

        host = MixinHost()
        history: list[Any] = []

        result = await host._async_run_tool_loop(
            client=client,
            prompt="get data",
            tool_registry=registry,
            system_instruction="sys",
            history=history,
        )

        assert result.text == "Got async data."
        assert result.tools_executed[0]["name"] == "async_fetch"
        assert result.tools_executed[0]["error"] is False

    async def test_frequency_penalty_none_omitted(self) -> None:
        """When frequency_penalty is None, it should not be passed to generate."""
        client = MagicMock()
        client.async_generate_with_tools = AsyncMock(
            return_value=_make_text_result("ok"),
        )

        host = MixinHost()
        history: list[Any] = []

        await host._async_run_tool_loop(
            client=client,
            prompt="test",
            tool_registry=_make_registry(),
            system_instruction="sys",
            history=history,
            frequency_penalty=None,
        )

        call_kwargs = client.async_generate_with_tools.call_args
        assert "frequency_penalty" not in call_kwargs.kwargs

    async def test_prompt_only_sent_on_first_iteration(self) -> None:
        """Prompt is sent on iteration 1; subsequent iterations send empty list."""

        def noop() -> ToolResult:
            return ToolResult(output="ok")

        registry = _make_registry(
            ToolDef(name="noop", description="No-op", parameters=[], execute=noop),
        )

        client = MagicMock()
        client.async_generate_with_tools = AsyncMock(
            side_effect=[
                _make_fc_result([{"name": "noop", "args": {}}]),
                _make_text_result("done"),
            ],
        )

        host = MixinHost()
        history: list[Any] = []

        await host._async_run_tool_loop(
            client=client,
            prompt="initial prompt",
            tool_registry=registry,
            system_instruction="sys",
            history=history,
        )

        calls = client.async_generate_with_tools.call_args_list
        # First call: prompt is the actual prompt
        assert calls[0].args[0] == "initial prompt"
        # Second call: prompt is empty list
        assert calls[1].args[0] == []


# ══════════════════════════════════════════════════════════════
# Sync methods remain unchanged
# ══════════════════════════════════════════════════════════════


class TestSyncMethodsUnchanged:
    """Verify that the original sync methods still work correctly."""

    def test_sync_execute_single_tool_still_works(self) -> None:
        """Sync _execute_single_tool should still function."""

        def add(a: int, b: int) -> ToolResult:
            return ToolResult(output=str(a + b))

        tool = ToolDef(
            name="add",
            description="Add two numbers",
            parameters=[
                ToolParam(name="a", type="integer", description="first"),
                ToolParam(name="b", type="integer", description="second"),
            ],
            execute=add,
        )
        registry = _make_registry(tool)
        host = MixinHost()

        result = host._execute_single_tool(registry, "add", {"a": 3, "b": 4})

        assert result.output == "7"
        assert result.error is False

    def test_sync_run_tool_loop_still_works(self) -> None:
        """Sync _run_tool_loop should still function."""
        client = MagicMock()
        client.generate_with_tools = MagicMock(
            return_value=_make_text_result("sync response"),
        )

        host = MixinHost()
        history: list[Any] = []

        result = host._run_tool_loop(
            client=client,
            prompt="sync test",
            tool_registry=_make_registry(),
            system_instruction="sys",
            history=history,
        )

        assert isinstance(result, ToolLoopResult)
        assert result.text == "sync response"
        client.generate_with_tools.assert_called_once()

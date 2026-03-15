"""Tests for the on_tool_call callback feature and ToolCallLogger.

Covers:
- _truncate_args helper
- ToolCallLogger.__call__ and print_summary
- on_tool_call callback invocation in sync _run_tool_loop
- on_tool_call callback invocation in async _async_run_tool_loop
- Callback errors are swallowed (never break the loop)
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from vaig.agents.mixins import OnToolCall, ToolLoopMixin, ToolLoopResult
from vaig.cli.commands.live import ToolCallLogger, _truncate_args
from vaig.core.client import ToolCallResult
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
# _truncate_args
# ══════════════════════════════════════════════════════════════


class TestTruncateArgs:
    """Tests for the _truncate_args helper."""

    def test_empty_dict(self) -> None:
        assert _truncate_args({}) == ""

    def test_single_short_arg(self) -> None:
        assert _truncate_args({"ns": "default"}) == "ns=default"

    def test_multiple_args(self) -> None:
        result = _truncate_args({"a": "1", "b": "2"})
        assert result == "a=1, b=2"

    def test_long_value_truncated(self) -> None:
        long_val = "x" * 100
        result = _truncate_args({"data": long_val}, max_len=50)
        assert "data=" in result
        assert result.endswith("...")
        # 50 chars of x + "..."
        assert len(result.split("=")[1]) == 53

    def test_custom_max_len(self) -> None:
        result = _truncate_args({"key": "abcdef"}, max_len=3)
        assert result == "key=abc..."

    def test_value_exactly_at_max_len(self) -> None:
        result = _truncate_args({"k": "abc"}, max_len=3)
        # Exactly at max_len — no truncation
        assert result == "k=abc"

    def test_non_string_values_converted(self) -> None:
        result = _truncate_args({"count": 42, "flag": True})
        assert "count=42" in result
        assert "flag=True" in result


# ══════════════════════════════════════════════════════════════
# ToolCallLogger
# ══════════════════════════════════════════════════════════════


class TestToolCallLogger:
    """Tests for the ToolCallLogger class."""

    def test_initial_state(self) -> None:
        logger = ToolCallLogger()
        assert logger.tool_count == 0
        assert logger.total_duration == 0.0
        assert logger.errors == 0

    def test_call_increments_counts(self) -> None:
        logger = ToolCallLogger()
        logger("kubectl_get", {"resource": "pods"}, 1.2, True)
        assert logger.tool_count == 1
        assert logger.total_duration == pytest.approx(1.2)
        assert logger.errors == 0

    def test_failed_call_increments_errors(self) -> None:
        logger = ToolCallLogger()
        logger("kubectl_get", {"resource": "pods"}, 0.5, False)
        assert logger.tool_count == 1
        assert logger.errors == 1

    def test_multiple_calls_accumulate(self) -> None:
        logger = ToolCallLogger()
        logger("tool_a", {}, 1.0, True)
        logger("tool_b", {"x": "1"}, 2.0, True)
        logger("tool_c", {}, 0.5, False)
        assert logger.tool_count == 3
        assert logger.total_duration == pytest.approx(3.5)
        assert logger.errors == 1

    @patch("vaig.cli.commands.live.console")
    def test_call_prints_ok_for_success(self, mock_console: MagicMock) -> None:
        logger = ToolCallLogger()
        logger("kubectl_get", {"resource": "pods"}, 1.2, True)
        mock_console.print.assert_called_once()
        output = mock_console.print.call_args[0][0]
        assert "kubectl_get" in output
        assert "OK" in output
        assert "1.2s" in output

    @patch("vaig.cli.commands.live.console")
    def test_call_prints_fail_for_error(self, mock_console: MagicMock) -> None:
        logger = ToolCallLogger()
        logger("kubectl_get", {}, 0.5, False)
        output = mock_console.print.call_args[0][0]
        assert "FAIL" in output

    @patch("vaig.cli.commands.live.console")
    def test_print_summary_no_errors(self, mock_console: MagicMock) -> None:
        logger = ToolCallLogger()
        logger("t1", {}, 1.0, True)
        logger("t2", {}, 2.0, True)
        # Reset mock to isolate summary call
        mock_console.print.reset_mock()
        logger.print_summary()
        output = mock_console.print.call_args[0][0]
        assert "Pipeline complete" in output
        assert "2 tools executed" in output
        assert "failed" not in output

    @patch("vaig.cli.commands.live.console")
    def test_print_summary_with_errors(self, mock_console: MagicMock) -> None:
        logger = ToolCallLogger()
        logger("t1", {}, 1.0, True)
        logger("t2", {}, 0.5, False)
        mock_console.print.reset_mock()
        logger.print_summary()
        output = mock_console.print.call_args[0][0]
        assert "Pipeline complete with errors" in output
        assert "1 failed" in output

    @patch("vaig.cli.commands.live.console")
    def test_print_summary_single_tool(self, mock_console: MagicMock) -> None:
        logger = ToolCallLogger()
        logger("t1", {}, 1.0, True)
        mock_console.print.reset_mock()
        logger.print_summary()
        output = mock_console.print.call_args[0][0]
        # Should say "1 tool executed" (no 's')
        assert "1 tool executed" in output

    def test_conforms_to_on_tool_call_protocol(self) -> None:
        """ToolCallLogger must satisfy the OnToolCall protocol."""
        logger = ToolCallLogger()
        # This should be type-safe — ToolCallLogger.__call__ matches OnToolCall
        callback: OnToolCall = logger
        callback("test", {}, 0.1, True)
        assert logger.tool_count == 1


# ══════════════════════════════════════════════════════════════
# on_tool_call callback in sync _run_tool_loop
# ══════════════════════════════════════════════════════════════


class TestOnToolCallSync:
    """Tests for on_tool_call callback in the sync _run_tool_loop."""

    def test_callback_invoked_with_correct_args(self) -> None:
        """Callback receives (tool_name, tool_args, duration, success)."""

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
        client.generate_with_tools = MagicMock(
            side_effect=[
                _make_fc_result([{"name": "search", "args": {"query": "python"}}]),
                _make_text_result("done"),
            ],
        )

        callback = MagicMock()
        host = MixinHost()
        history: list[Any] = []

        host._run_tool_loop(
            client=client,
            prompt="test",
            tool_registry=registry,
            system_instruction="sys",
            history=history,
            on_tool_call=callback,
        )

        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == "search"  # tool_name
        assert args[1] == {"query": "python"}  # tool_args
        assert isinstance(args[2], float) and args[2] >= 0  # duration
        assert args[3] is True  # success

    def test_callback_invoked_for_each_tool(self) -> None:
        """Callback is called once per tool call in a multi-tool turn."""

        def tool_a(x: str) -> ToolResult:
            return ToolResult(output=f"A:{x}")

        def tool_b(y: str) -> ToolResult:
            return ToolResult(output=f"B:{y}")

        registry = _make_registry(
            ToolDef(name="a", description="A", parameters=[ToolParam(name="x", type="string", description="x")], execute=tool_a),
            ToolDef(name="b", description="B", parameters=[ToolParam(name="y", type="string", description="y")], execute=tool_b),
        )

        client = MagicMock()
        client.generate_with_tools = MagicMock(
            side_effect=[
                _make_fc_result([
                    {"name": "a", "args": {"x": "1"}},
                    {"name": "b", "args": {"y": "2"}},
                ]),
                _make_text_result("done"),
            ],
        )

        callback = MagicMock()
        host = MixinHost()
        history: list[Any] = []

        host._run_tool_loop(
            client=client,
            prompt="test",
            tool_registry=registry,
            system_instruction="sys",
            history=history,
            on_tool_call=callback,
        )

        assert callback.call_count == 2
        # First call: tool "a"
        assert callback.call_args_list[0][0][0] == "a"
        assert callback.call_args_list[0][0][3] is True
        # Second call: tool "b"
        assert callback.call_args_list[1][0][0] == "b"

    def test_callback_reports_failure(self) -> None:
        """When a tool fails, callback receives success=False."""

        def failing(query: str) -> ToolResult:
            msg = "boom"
            raise RuntimeError(msg)

        registry = _make_registry(
            ToolDef(name="fail", description="Fail", parameters=[ToolParam(name="query", type="string", description="q")], execute=failing),
        )

        client = MagicMock()
        client.generate_with_tools = MagicMock(
            side_effect=[
                _make_fc_result([{"name": "fail", "args": {"query": "x"}}]),
                _make_text_result("error handled"),
            ],
        )

        callback = MagicMock()
        host = MixinHost()
        history: list[Any] = []

        host._run_tool_loop(
            client=client,
            prompt="test",
            tool_registry=registry,
            system_instruction="sys",
            history=history,
            on_tool_call=callback,
        )

        callback.assert_called_once()
        assert callback.call_args[0][3] is False  # success=False

    def test_callback_error_does_not_break_loop(self) -> None:
        """If the callback itself raises, the loop continues normally."""

        def search(query: str) -> ToolResult:
            return ToolResult(output=f"Found: {query}")

        registry = _make_registry(
            ToolDef(name="search", description="S", parameters=[ToolParam(name="query", type="string", description="q")], execute=search),
        )

        client = MagicMock()
        client.generate_with_tools = MagicMock(
            side_effect=[
                _make_fc_result([{"name": "search", "args": {"query": "x"}}]),
                _make_text_result("final"),
            ],
        )

        def exploding_callback(*args: Any) -> None:
            msg = "callback exploded"
            raise ValueError(msg)

        host = MixinHost()
        history: list[Any] = []

        # Should NOT raise — callback error is swallowed
        result = host._run_tool_loop(
            client=client,
            prompt="test",
            tool_registry=registry,
            system_instruction="sys",
            history=history,
            on_tool_call=exploding_callback,
        )

        assert result.text == "final"
        assert len(result.tools_executed) == 1

    def test_none_callback_is_noop(self) -> None:
        """When on_tool_call is None, loop works normally."""
        client = MagicMock()
        client.generate_with_tools = MagicMock(
            return_value=_make_text_result("ok"),
        )

        host = MixinHost()
        history: list[Any] = []

        result = host._run_tool_loop(
            client=client,
            prompt="test",
            tool_registry=_make_registry(),
            system_instruction="sys",
            history=history,
            on_tool_call=None,
        )

        assert result.text == "ok"


# ══════════════════════════════════════════════════════════════
# on_tool_call callback in async _async_run_tool_loop
# ══════════════════════════════════════════════════════════════


class TestOnToolCallAsync:
    """Tests for on_tool_call callback in the async _async_run_tool_loop."""

    async def test_callback_invoked_with_correct_args(self) -> None:
        """Callback receives (tool_name, tool_args, duration, success)."""

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
                _make_fc_result([{"name": "search", "args": {"query": "k8s"}}]),
                _make_text_result("done"),
            ],
        )

        callback = MagicMock()
        host = MixinHost()
        history: list[Any] = []

        await host._async_run_tool_loop(
            client=client,
            prompt="test",
            tool_registry=registry,
            system_instruction="sys",
            history=history,
            on_tool_call=callback,
        )

        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == "search"
        assert args[1] == {"query": "k8s"}
        assert isinstance(args[2], float)
        assert args[3] is True

    async def test_callback_error_does_not_break_async_loop(self) -> None:
        """If the callback raises, the async loop continues normally."""

        def tool_fn(x: str) -> ToolResult:
            return ToolResult(output=x)

        registry = _make_registry(
            ToolDef(name="t", description="T", parameters=[ToolParam(name="x", type="string", description="x")], execute=tool_fn),
        )

        client = MagicMock()
        client.async_generate_with_tools = AsyncMock(
            side_effect=[
                _make_fc_result([{"name": "t", "args": {"x": "1"}}]),
                _make_text_result("final"),
            ],
        )

        def bad_callback(*args: Any) -> None:
            msg = "boom"
            raise RuntimeError(msg)

        host = MixinHost()
        history: list[Any] = []

        result = await host._async_run_tool_loop(
            client=client,
            prompt="test",
            tool_registry=registry,
            system_instruction="sys",
            history=history,
            on_tool_call=bad_callback,
        )

        assert result.text == "final"

    async def test_callback_reports_failure_async(self) -> None:
        """When a tool fails in async loop, callback receives success=False."""

        def failing(query: str) -> ToolResult:
            msg = "fail"
            raise OSError(msg)

        registry = _make_registry(
            ToolDef(name="fail", description="F", parameters=[ToolParam(name="query", type="string", description="q")], execute=failing),
        )

        client = MagicMock()
        client.async_generate_with_tools = AsyncMock(
            side_effect=[
                _make_fc_result([{"name": "fail", "args": {"query": "x"}}]),
                _make_text_result("handled"),
            ],
        )

        callback = MagicMock()
        host = MixinHost()
        history: list[Any] = []

        await host._async_run_tool_loop(
            client=client,
            prompt="test",
            tool_registry=registry,
            system_instruction="sys",
            history=history,
            on_tool_call=callback,
        )

        callback.assert_called_once()
        assert callback.call_args[0][3] is False

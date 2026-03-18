"""Tests for the on_tool_call callback feature and ToolCallLogger.

Covers:
- _truncate_args helper
- ToolCallLogger.__call__ and print_summary
- on_tool_call callback invocation in sync _run_tool_loop
- on_tool_call callback invocation in async _async_run_tool_loop
- Callback errors are swallowed (never break the loop)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vaig.agents.mixins import OnToolCall, ToolLoopMixin
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
        assert "✓" in output
        assert "1.2s" in output
        assert "🔧" in output

    @patch("vaig.cli.commands.live.console")
    def test_call_prints_fail_for_error(self, mock_console: MagicMock) -> None:
        logger = ToolCallLogger()
        logger("kubectl_get", {}, 0.5, False)
        output = mock_console.print.call_args[0][0]
        assert "FAIL" in output
        assert "🔧" in output

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
        logger("t2", {}, 0.5, False, "Connection refused")
        mock_console.print.reset_mock()
        logger.print_summary()
        output = mock_console.print.call_args[0][0]
        assert "Pipeline complete with errors" in output
        assert "1 failed" in output
        assert "Connection refused" in output

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

    def test_backward_compat_four_args(self) -> None:
        """ToolCallLogger works when called with only 4 positional args (no error_message)."""
        logger = ToolCallLogger()
        # Simulates an old caller that doesn't pass error_message
        logger("kubectl_get", {"resource": "pods"}, 1.0, False)
        assert logger.errors == 1
        # Should record "unknown" as reason since no error_message provided
        assert len(logger._error_reasons) == 1
        assert logger._error_reasons[0] == "unknown"

    @patch("vaig.cli.commands.live.console")
    def test_error_message_displayed(self, mock_console: MagicMock) -> None:
        """Error message is shown when tool fails with error_message."""
        logger = ToolCallLogger()
        logger("kubectl_get", {}, 0.5, False, "AuthenticationError: token expired")
        output = mock_console.print.call_args[0][0]
        assert "AuthenticationError: token expired" in output
        assert "FAIL" in output

    @patch("vaig.cli.commands.live.console")
    def test_error_message_truncated(self, mock_console: MagicMock) -> None:
        """Long error messages are truncated to ~80 chars."""
        logger = ToolCallLogger()
        long_error = "x" * 200
        logger("kubectl_get", {}, 0.5, False, long_error)
        output = mock_console.print.call_args[0][0]
        # Should end with "..." after truncation
        assert "..." in output
        # Original 200-char message should NOT appear in full
        assert long_error not in output

    @patch("vaig.cli.commands.live.console")
    def test_error_reasons_grouped_in_summary(self, mock_console: MagicMock) -> None:
        """print_summary groups failures by reason."""
        logger = ToolCallLogger()
        # 3 auth errors, 1 timeout
        for _ in range(3):
            logger("tool_a", {}, 0.1, False, "auth error: token expired")
        logger("tool_b", {}, 0.1, False, "timeout after 30s")
        logger("tool_c", {}, 0.1, True)
        mock_console.print.reset_mock()
        logger.print_summary()
        output = mock_console.print.call_args[0][0]
        assert "4 failed" in output
        assert "×" in output  # Unicode multiplication sign
        assert "auth error" in output
        assert "timeout" in output

    @patch("vaig.cli.commands.live.console")
    def test_extract_reason_truncates_long_messages(self, mock_console: MagicMock) -> None:
        """_extract_reason truncates to 40 chars."""
        reason = ToolCallLogger._extract_reason("a" * 100)
        assert len(reason) <= 44  # 40 + "..."
        assert reason.endswith("...")

    def test_extract_reason_empty(self) -> None:
        """_extract_reason returns 'unknown' for empty string."""
        assert ToolCallLogger._extract_reason("") == "unknown"

    def test_extract_reason_multiline(self) -> None:
        """_extract_reason takes only first line."""
        reason = ToolCallLogger._extract_reason("first line\nsecond line\nthird")
        assert "second" not in reason
        assert "first line" in reason

    def test_tool_name_counts_tracked(self) -> None:
        """Per-tool-name counts are accumulated in tool_name_counts."""
        logger = ToolCallLogger()
        logger("kubectl_get", {"resource": "pods"}, 1.0, True)
        logger("kubectl_get", {"resource": "nodes"}, 0.8, True)
        logger("get_events", {"ns": "default"}, 0.5, True)
        assert logger.tool_name_counts["kubectl_get"] == 2
        assert logger.tool_name_counts["get_events"] == 1

    def test_cache_hits_tracked(self) -> None:
        """Cache hits are counted when cached=True is passed."""
        logger = ToolCallLogger()
        logger("kubectl_get", {}, 0.0, True, cached=True)
        logger("kubectl_get", {}, 1.0, True)
        logger("get_events", {}, 0.0, True, cached=True)
        assert logger.cache_hits == 2
        assert logger.tool_count == 3

    def test_format_tool_counts_empty(self) -> None:
        """format_tool_counts returns empty string when no tools called."""
        logger = ToolCallLogger()
        assert logger.format_tool_counts() == ""

    def test_format_tool_counts_single_tool(self) -> None:
        """format_tool_counts shows single tool with count."""
        logger = ToolCallLogger()
        logger("kubectl_get", {}, 1.0, True)
        logger("kubectl_get", {}, 0.5, True)
        assert logger.format_tool_counts() == "kubectl_get ×2"

    def test_format_tool_counts_multiple_tools(self) -> None:
        """format_tool_counts shows multiple tools separated by pipe."""
        logger = ToolCallLogger()
        for _ in range(4):
            logger("kubectl_get", {}, 1.0, True)
        for _ in range(2):
            logger("get_events", {}, 0.5, True)
        result = logger.format_tool_counts()
        assert "kubectl_get ×4" in result
        assert "get_events ×2" in result
        assert " | " in result

    def test_format_tool_counts_with_cache_hits(self) -> None:
        """format_tool_counts appends cache hit count when present."""
        logger = ToolCallLogger()
        logger("kubectl_get", {}, 0.0, True, cached=True)
        logger("kubectl_get", {}, 1.0, True)
        result = logger.format_tool_counts()
        assert "kubectl_get ×2" in result
        assert "(1 cached)" in result

    def test_reset_clears_per_agent_counters(self) -> None:
        """reset() clears tool_name_counts and cache_hits but keeps totals."""
        logger = ToolCallLogger()
        logger("kubectl_get", {}, 1.0, True)
        logger("get_events", {}, 0.0, True, cached=True)
        assert logger.tool_count == 2
        assert logger.cache_hits == 1

        logger.reset()
        assert len(logger.tool_name_counts) == 0
        assert logger.cache_hits == 0
        # Pipeline-level totals should still be there
        assert logger.tool_count == 2
        assert logger.total_duration == pytest.approx(1.0)

    @patch("vaig.cli.commands.live.console")
    def test_cached_call_shows_cached_tag(self, mock_console: MagicMock) -> None:
        """Cached tool calls display [cached] tag in output."""
        logger = ToolCallLogger()
        logger("kubectl_get", {"resource": "pods"}, 0.0, True, cached=True)
        output = mock_console.print.call_args[0][0]
        assert "cached" in output
        assert "✓" in output

    @patch("vaig.cli.commands.live.console")
    def test_non_cached_call_no_cached_tag(self, mock_console: MagicMock) -> None:
        """Non-cached tool calls do NOT show [cached] tag."""
        logger = ToolCallLogger()
        logger("kubectl_get", {"resource": "pods"}, 1.0, True)
        output = mock_console.print.call_args[0][0]
        assert "cached" not in output

    @patch("vaig.cli.commands.live.console")
    def test_print_summary_includes_tool_breakdown(self, mock_console: MagicMock) -> None:
        """print_summary includes per-tool-name breakdown."""
        logger = ToolCallLogger()
        for _ in range(3):
            logger("kubectl_get", {}, 1.0, True)
        logger("get_events", {}, 0.5, True)
        mock_console.print.reset_mock()
        logger.print_summary()
        # print_summary may produce multiple console.print calls
        all_output = " ".join(call[0][0] for call in mock_console.print.call_args_list)
        assert "kubectl_get ×3" in all_output
        assert "get_events ×1" in all_output
        assert "Tools:" in all_output


# ══════════════════════════════════════════════════════════════
# on_tool_call callback in sync _run_tool_loop
# ══════════════════════════════════════════════════════════════


class TestOnToolCallSync:
    """Tests for on_tool_call callback in the sync _run_tool_loop."""

    def test_callback_invoked_with_correct_args(self) -> None:
        """Callback receives (tool_name, tool_args, duration, success, error_message)."""

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
        assert args[4] == ""  # error_message (empty on success)

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
        """When a tool fails, callback receives success=False and error_message."""

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
        assert "boom" in callback.call_args[0][4]  # error_message contains the error

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

    def test_callback_receives_cached_kwarg(self) -> None:
        """_notify_tool_call passes cached=False to callback on cache miss."""
        host = MixinHost()
        result = ToolResult(output="ok")
        callback = MagicMock()

        host._notify_tool_call(callback, "test_tool", {"a": "1"}, 1.0, result, cached=False)

        callback.assert_called_once()
        kwargs = callback.call_args[1]
        assert "cached" in kwargs
        assert kwargs["cached"] is False

    def test_callback_receives_cached_true(self) -> None:
        """_notify_tool_call passes cached=True on cache hit."""
        host = MixinHost()
        result = ToolResult(output="cached data")
        callback = MagicMock()

        host._notify_tool_call(callback, "test_tool", {}, 0.0, result, cached=True)

        kwargs = callback.call_args[1]
        assert kwargs["cached"] is True

    def test_notify_tool_call_backward_compat_no_cached(self) -> None:
        """Old callbacks that don't accept cached kwarg still work (TypeError caught)."""
        host = MixinHost()
        result = ToolResult(output="ok")

        def old_callback(tool_name: str, tool_args: dict, duration: float, success: bool) -> None:
            pass  # Old-style callback without error_message or cached

        # Should NOT raise
        host._notify_tool_call(old_callback, "test_tool", {}, 1.0, result, cached=True)


# ══════════════════════════════════════════════════════════════
# on_tool_call callback in async _async_run_tool_loop
# ══════════════════════════════════════════════════════════════


class TestOnToolCallAsync:
    """Tests for on_tool_call callback in the async _async_run_tool_loop."""

    async def test_callback_invoked_with_correct_args(self) -> None:
        """Callback receives (tool_name, tool_args, duration, success, error_message)."""

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
        assert args[4] == ""  # error_message empty on success

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
        """When a tool fails in async loop, callback receives success=False and error_message."""

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
        assert "fail" in callback.call_args[0][4]  # error_message contains the error


# ══════════════════════════════════════════════════════════════
# Bug 5: None output guard in _execute_single_tool
# ══════════════════════════════════════════════════════════════


class TestNoneOutputGuard:
    """Tests for the None output guard in tool execution."""

    def test_none_output_replaced_sync(self) -> None:
        """When a tool returns ToolResult with output=None, it is replaced."""

        def bad_tool(x: str) -> ToolResult:
            r = ToolResult(output="ok")
            r.output = None  # type: ignore[assignment]  # simulate buggy tool
            return r

        tool = ToolDef(
            name="bad",
            description="Bad tool",
            parameters=[ToolParam(name="x", type="string", description="x")],
            execute=bad_tool,
        )
        registry = _make_registry(tool)

        client = MagicMock()
        client.generate_with_tools = MagicMock(
            side_effect=[
                _make_fc_result([{"name": "bad", "args": {"x": "1"}}]),
                _make_text_result("done"),
            ],
        )

        host = MixinHost()
        history: list[Any] = []

        # Should NOT raise — None output is guarded
        result = host._run_tool_loop(
            client=client,
            prompt="test",
            tool_registry=registry,
            system_instruction="sys",
            history=history,
        )

        assert result.text == "done"
        # The tool_result should have been replaced with "(no output)"
        assert result.tools_executed[0]["output"] == "(no output)"


# ══════════════════════════════════════════════════════════════
# Bug 5: auth.py stdout None guard
# ══════════════════════════════════════════════════════════════


class TestAuthStdoutGuard:
    """Tests for the None stdout guard in _fetch_gcloud_access_token."""

    @patch("vaig.core.auth.subprocess.run")
    def test_none_stdout_does_not_crash(self, mock_run: MagicMock) -> None:
        """When subprocess returns stdout=None, it should raise RuntimeError not AttributeError."""
        from vaig.core.auth import _fetch_gcloud_access_token

        mock_run.return_value = MagicMock(
            stdout=None,
            stderr="some error",
            returncode=1,
        )

        with pytest.raises(RuntimeError, match="Could not obtain credentials"):
            _fetch_gcloud_access_token()

    @patch("vaig.core.auth.subprocess.run")
    def test_empty_stdout_raises_runtime_error(self, mock_run: MagicMock) -> None:
        """When subprocess returns empty stdout, it should raise RuntimeError."""
        from vaig.core.auth import _fetch_gcloud_access_token

        mock_run.return_value = MagicMock(
            stdout="",
            stderr="",
            returncode=1,
        )

        with pytest.raises(RuntimeError, match="Could not obtain credentials"):
            _fetch_gcloud_access_token()


# ══════════════════════════════════════════════════════════════
# Bug 3: stderr suppression context manager
# ══════════════════════════════════════════════════════════════


class TestSuppressStderr:
    """Tests for the _suppress_stderr context manager."""

    def test_stderr_suppressed(self) -> None:
        """stderr output inside _suppress_stderr should not reach terminal."""
        import os

        from vaig.tools.gke._clients import _suppress_stderr

        # Capture what fd 2 points to after _suppress_stderr exits
        # (it should be restored to original)
        original_fd = os.dup(2)
        try:
            with _suppress_stderr():
                # Writing to fd 2 inside the context should go to /dev/null
                os.write(2, b"this should be suppressed\n")
            # After exiting, fd 2 should be restored
            # Verify by writing to fd 2 — it should work normally
            os.write(2, b"")  # Should not raise
        finally:
            os.close(original_fd)

    def test_stderr_restored_after_exception(self) -> None:
        """stderr is restored even if an exception is raised inside the context."""
        import os

        from vaig.tools.gke._clients import _suppress_stderr

        original_fd = os.dup(2)
        try:
            with pytest.raises(ValueError, match="test"):
                with _suppress_stderr():
                    msg = "test"
                    raise ValueError(msg)
            # fd 2 should still be restored
            os.write(2, b"")  # Should not raise
        finally:
            os.close(original_fd)


# ══════════════════════════════════════════════════════════════
# _NonTTYStream and _suppress_stderr TTY override
# ══════════════════════════════════════════════════════════════


class TestNonTTYStream:
    """Tests for the _NonTTYStream wrapper class."""

    def test_isatty_returns_false(self) -> None:
        """_NonTTYStream.isatty() always returns False, even if the wrapped stream is a TTY."""
        from vaig.tools.gke._clients import _NonTTYStream

        class FakeTTY:
            def isatty(self) -> bool:
                return True

        wrapper = _NonTTYStream(FakeTTY())
        assert wrapper.isatty() is False

    def test_delegates_other_attributes(self) -> None:
        """_NonTTYStream delegates attribute access to the wrapped stream."""
        from vaig.tools.gke._clients import _NonTTYStream

        class FakeStream:
            name = "fake"

            def write(self, data: str) -> int:
                return len(data)

            def isatty(self) -> bool:
                return True

        wrapper = _NonTTYStream(FakeStream())
        assert wrapper.name == "fake"
        assert wrapper.write("hello") == 5

    def test_isatty_false_even_for_real_stdout(self) -> None:
        """Wrapping sys.stdout returns False for isatty()."""
        import sys

        from vaig.tools.gke._clients import _NonTTYStream

        wrapper = _NonTTYStream(sys.stdout)
        assert wrapper.isatty() is False


class TestSuppressStderrTTYOverride:
    """Tests that _suppress_stderr makes sys.stdout non-interactive."""

    def test_stdout_is_non_tty_inside_context(self) -> None:
        """Inside _suppress_stderr, sys.stdout.isatty() returns False."""
        import sys

        from vaig.tools.gke._clients import _suppress_stderr

        with _suppress_stderr():
            assert sys.stdout.isatty() is False

    def test_stdout_restored_after_context(self) -> None:
        """After _suppress_stderr exits, sys.stdout is restored to its original value."""
        import sys

        from vaig.tools.gke._clients import _suppress_stderr

        original = sys.stdout
        with _suppress_stderr():
            # Should be wrapped
            assert sys.stdout is not original
        # Should be restored
        assert sys.stdout is original

    def test_stdout_restored_after_exception(self) -> None:
        """sys.stdout is restored even when an exception is raised inside _suppress_stderr."""
        import sys

        from vaig.tools.gke._clients import _suppress_stderr

        original = sys.stdout
        with pytest.raises(RuntimeError, match="test_exc"):
            with _suppress_stderr():
                assert sys.stdout is not original
                msg = "test_exc"
                raise RuntimeError(msg)
        assert sys.stdout is original

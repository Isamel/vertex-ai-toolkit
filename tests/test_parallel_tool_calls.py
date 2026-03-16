"""Tests for Q4: Parallel async function call execution with semaphore limiter.

Tests cover:
- Config defaults (parallel_tool_calls=True, max_concurrent=5)
- Single function call uses sequential path (no gather overhead)
- Multiple function calls use parallel path when enabled
- Parallel disabled falls back to sequential
- Error handling: one call fails, others succeed (partial failure)
- Error handling: all calls fail
- Semaphore limits concurrency
- Response order matches function_calls order
- Timing: parallel execution is faster than sequential (mock with sleep)
- Sync path always sequential
- on_tool_call callback invoked for parallel calls
- tool_call_store recording works for parallel calls
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from vaig.agents.mixins import ToolLoopMixin
from vaig.core.client import ToolCallResult
from vaig.core.config import AgentsConfig
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


def _simple_tool(name: str) -> ToolDef:
    """Create a simple sync tool for testing."""
    return ToolDef(
        name=name,
        description=f"Tool {name}",
        parameters=[ToolParam(name="query", type="string", description="q")],
        execute=lambda query: ToolResult(output=f"{name}: {query}"),
    )


def _slow_async_tool(name: str, delay: float = 0.1) -> ToolDef:
    """Create a slow async tool for timing tests."""

    async def _execute(query: str) -> ToolResult:
        await asyncio.sleep(delay)
        return ToolResult(output=f"{name}: {query}")

    return ToolDef(
        name=name,
        description=f"Slow tool {name}",
        parameters=[ToolParam(name="query", type="string", description="q")],
        execute=_execute,
    )


def _failing_tool(name: str) -> ToolDef:
    """Create a tool that raises an exception."""

    def _execute(query: str) -> ToolResult:
        raise RuntimeError(f"{name} failed!")

    return ToolDef(
        name=name,
        description=f"Failing tool {name}",
        parameters=[ToolParam(name="query", type="string", description="q")],
        execute=_execute,
    )


# ══════════════════════════════════════════════════════════════
# Config defaults
# ══════════════════════════════════════════════════════════════


class TestConfigDefaults:
    """Tests for parallel tool call configuration defaults."""

    def test_parallel_tool_calls_default_true(self) -> None:
        """parallel_tool_calls should default to True."""
        config = AgentsConfig()
        assert config.parallel_tool_calls is True

    def test_max_concurrent_tool_calls_default_5(self) -> None:
        """max_concurrent_tool_calls should default to 5."""
        config = AgentsConfig()
        assert config.max_concurrent_tool_calls == 5

    def test_parallel_tool_calls_configurable(self) -> None:
        """parallel_tool_calls should be configurable via constructor."""
        config = AgentsConfig(parallel_tool_calls=False)
        assert config.parallel_tool_calls is False

    def test_max_concurrent_configurable(self) -> None:
        """max_concurrent_tool_calls should be configurable."""
        config = AgentsConfig(max_concurrent_tool_calls=10)
        assert config.max_concurrent_tool_calls == 10


# ══════════════════════════════════════════════════════════════
# Sequential path (single call)
# ══════════════════════════════════════════════════════════════


class TestSingleCallSequential:
    """Single function call should always use sequential path."""

    @pytest.mark.asyncio
    async def test_single_fc_does_not_use_gather(self) -> None:
        """A single function call should skip gather, even with parallel=True."""
        tool = _simple_tool("search")
        registry = _make_registry(tool)

        client = MagicMock()
        client.async_generate_with_tools = AsyncMock(
            side_effect=[
                _make_fc_result([{"name": "search", "args": {"query": "hello"}}]),
                _make_text_result("done"),
            ],
        )
        client.build_function_response_parts = MagicMock(return_value=[])

        host = MixinHost()
        result = await host._async_run_tool_loop(
            client=client,
            prompt="test",
            tool_registry=registry,
            system_instruction="sys",
            history=[],
            max_iterations=5,
            parallel_tool_calls=True,
        )
        assert result.text == "done"
        assert len(result.tools_executed) == 1
        assert result.tools_executed[0]["name"] == "search"


# ══════════════════════════════════════════════════════════════
# Parallel execution path
# ══════════════════════════════════════════════════════════════


class TestParallelExecution:
    """Multiple function calls should run in parallel when enabled."""

    @pytest.mark.asyncio
    async def test_multiple_fcs_parallel(self) -> None:
        """Multiple function calls should all be executed in parallel."""
        tool_a = _simple_tool("tool_a")
        tool_b = _simple_tool("tool_b")
        registry = _make_registry(tool_a, tool_b)

        client = MagicMock()
        client.async_generate_with_tools = AsyncMock(
            side_effect=[
                _make_fc_result(
                    [
                        {"name": "tool_a", "args": {"query": "a"}},
                        {"name": "tool_b", "args": {"query": "b"}},
                    ]
                ),
                _make_text_result("both done"),
            ],
        )
        client.build_function_response_parts = MagicMock(return_value=[])

        host = MixinHost()
        result = await host._async_run_tool_loop(
            client=client,
            prompt="test",
            tool_registry=registry,
            system_instruction="sys",
            history=[],
            max_iterations=5,
            parallel_tool_calls=True,
        )
        assert result.text == "both done"
        assert len(result.tools_executed) == 2
        assert result.tools_executed[0]["name"] == "tool_a"
        assert result.tools_executed[1]["name"] == "tool_b"

    @pytest.mark.asyncio
    async def test_parallel_disabled_uses_sequential(self) -> None:
        """When parallel_tool_calls=False, multiple calls run sequentially."""
        tool_a = _simple_tool("tool_a")
        tool_b = _simple_tool("tool_b")
        registry = _make_registry(tool_a, tool_b)

        client = MagicMock()
        client.async_generate_with_tools = AsyncMock(
            side_effect=[
                _make_fc_result(
                    [
                        {"name": "tool_a", "args": {"query": "a"}},
                        {"name": "tool_b", "args": {"query": "b"}},
                    ]
                ),
                _make_text_result("done sequentially"),
            ],
        )
        client.build_function_response_parts = MagicMock(return_value=[])

        host = MixinHost()
        result = await host._async_run_tool_loop(
            client=client,
            prompt="test",
            tool_registry=registry,
            system_instruction="sys",
            history=[],
            max_iterations=5,
            parallel_tool_calls=False,
        )
        assert result.text == "done sequentially"
        assert len(result.tools_executed) == 2

    @pytest.mark.asyncio
    async def test_response_order_matches_function_calls_order(self) -> None:
        """Response parts must be in the same order as function_calls."""
        tools = [_simple_tool(f"tool_{i}") for i in range(4)]
        registry = _make_registry(*tools)

        fc_calls = [{"name": f"tool_{i}", "args": {"query": str(i)}} for i in range(4)]
        client = MagicMock()
        client.async_generate_with_tools = AsyncMock(
            side_effect=[
                _make_fc_result(fc_calls),
                _make_text_result("ordered"),
            ],
        )
        client.build_function_response_parts = MagicMock(return_value=[])

        host = MixinHost()
        result = await host._async_run_tool_loop(
            client=client,
            prompt="test",
            tool_registry=registry,
            system_instruction="sys",
            history=[],
            max_iterations=5,
            parallel_tool_calls=True,
        )
        assert result.text == "ordered"
        # Verify order
        for i in range(4):
            assert result.tools_executed[i]["name"] == f"tool_{i}"
            assert result.tools_executed[i]["output"].startswith(f"tool_{i}:")


# ══════════════════════════════════════════════════════════════
# Error handling
# ══════════════════════════════════════════════════════════════


class TestParallelErrorHandling:
    """Error handling for parallel tool call execution."""

    @pytest.mark.asyncio
    async def test_partial_failure_one_succeeds_one_fails(self) -> None:
        """One tool failing shouldn't affect the other — partial failure."""
        good_tool = _simple_tool("good_tool")
        bad_tool = _failing_tool("bad_tool")
        registry = _make_registry(good_tool, bad_tool)

        client = MagicMock()
        client.async_generate_with_tools = AsyncMock(
            side_effect=[
                _make_fc_result(
                    [
                        {"name": "good_tool", "args": {"query": "a"}},
                        {"name": "bad_tool", "args": {"query": "b"}},
                    ]
                ),
                _make_text_result("handled"),
            ],
        )
        client.build_function_response_parts = MagicMock(return_value=[])

        host = MixinHost()
        result = await host._async_run_tool_loop(
            client=client,
            prompt="test",
            tool_registry=registry,
            system_instruction="sys",
            history=[],
            max_iterations=5,
            parallel_tool_calls=True,
        )
        assert result.text == "handled"
        assert len(result.tools_executed) == 2
        # Good tool should succeed
        assert result.tools_executed[0]["error"] is False
        assert result.tools_executed[0]["output"].startswith("good_tool:")
        # Bad tool should have error
        assert result.tools_executed[1]["error"] is True

    @pytest.mark.asyncio
    async def test_all_calls_fail(self) -> None:
        """When all calls fail, all should have error responses."""
        bad_a = _failing_tool("bad_a")
        bad_b = _failing_tool("bad_b")
        registry = _make_registry(bad_a, bad_b)

        client = MagicMock()
        client.async_generate_with_tools = AsyncMock(
            side_effect=[
                _make_fc_result(
                    [
                        {"name": "bad_a", "args": {"query": "a"}},
                        {"name": "bad_b", "args": {"query": "b"}},
                    ]
                ),
                _make_text_result("recovered"),
            ],
        )
        client.build_function_response_parts = MagicMock(return_value=[])

        host = MixinHost()
        result = await host._async_run_tool_loop(
            client=client,
            prompt="test",
            tool_registry=registry,
            system_instruction="sys",
            history=[],
            max_iterations=5,
            parallel_tool_calls=True,
        )
        assert result.text == "recovered"
        assert len(result.tools_executed) == 2
        assert all(t["error"] for t in result.tools_executed)

    @pytest.mark.asyncio
    async def test_partial_failure_preserves_order(self) -> None:
        """Failed calls should appear at correct index in results."""
        good_tool = _simple_tool("first")
        bad_tool = _failing_tool("second")
        good_tool2 = _simple_tool("third")
        registry = _make_registry(good_tool, bad_tool, good_tool2)

        client = MagicMock()
        client.async_generate_with_tools = AsyncMock(
            side_effect=[
                _make_fc_result(
                    [
                        {"name": "first", "args": {"query": "1"}},
                        {"name": "second", "args": {"query": "2"}},
                        {"name": "third", "args": {"query": "3"}},
                    ]
                ),
                _make_text_result("ordered errors"),
            ],
        )
        client.build_function_response_parts = MagicMock(return_value=[])

        host = MixinHost()
        result = await host._async_run_tool_loop(
            client=client,
            prompt="test",
            tool_registry=registry,
            system_instruction="sys",
            history=[],
            max_iterations=5,
            parallel_tool_calls=True,
        )
        assert result.tools_executed[0]["name"] == "first"
        assert result.tools_executed[0]["error"] is False
        assert result.tools_executed[1]["name"] == "second"
        assert result.tools_executed[1]["error"] is True
        assert result.tools_executed[2]["name"] == "third"
        assert result.tools_executed[2]["error"] is False


# ══════════════════════════════════════════════════════════════
# Semaphore concurrency limit
# ══════════════════════════════════════════════════════════════


class TestSemaphore:
    """Tests that semaphore limits concurrency."""

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self) -> None:
        """With max_concurrent=2 and 4 tools, max 2 should run at once."""
        concurrent_count = 0
        max_concurrent_seen = 0
        lock = asyncio.Lock()

        async def _tracked_execute(query: str) -> ToolResult:
            nonlocal concurrent_count, max_concurrent_seen
            async with lock:
                concurrent_count += 1
                if concurrent_count > max_concurrent_seen:
                    max_concurrent_seen = concurrent_count
            await asyncio.sleep(0.05)
            async with lock:
                concurrent_count -= 1
            return ToolResult(output=f"done: {query}")

        tools = []
        for i in range(4):
            tools.append(
                ToolDef(
                    name=f"tool_{i}",
                    description=f"Tool {i}",
                    parameters=[ToolParam(name="query", type="string", description="q")],
                    execute=_tracked_execute,
                ),
            )
        registry = _make_registry(*tools)

        fc_calls = [{"name": f"tool_{i}", "args": {"query": str(i)}} for i in range(4)]
        client = MagicMock()
        client.async_generate_with_tools = AsyncMock(
            side_effect=[
                _make_fc_result(fc_calls),
                _make_text_result("limited"),
            ],
        )
        client.build_function_response_parts = MagicMock(return_value=[])

        host = MixinHost()
        result = await host._async_run_tool_loop(
            client=client,
            prompt="test",
            tool_registry=registry,
            system_instruction="sys",
            history=[],
            max_iterations=5,
            parallel_tool_calls=True,
            max_concurrent_tool_calls=2,
        )
        assert result.text == "limited"
        assert len(result.tools_executed) == 4
        assert max_concurrent_seen <= 2, f"Expected max 2 concurrent, got {max_concurrent_seen}"


# ══════════════════════════════════════════════════════════════
# Timing — parallel is faster
# ══════════════════════════════════════════════════════════════


class TestParallelTiming:
    """Tests that parallel execution is actually faster than sequential."""

    @pytest.mark.asyncio
    async def test_parallel_faster_than_sequential(self) -> None:
        """3 tools sleeping 0.1s should complete in ~0.1s parallel vs ~0.3s sequential."""
        tools = [_slow_async_tool(f"slow_{i}", delay=0.1) for i in range(3)]
        registry = _make_registry(*tools)

        fc_calls = [{"name": f"slow_{i}", "args": {"query": str(i)}} for i in range(3)]

        # Parallel run
        client_p = MagicMock()
        client_p.async_generate_with_tools = AsyncMock(
            side_effect=[
                _make_fc_result(fc_calls),
                _make_text_result("fast"),
            ],
        )
        client_p.build_function_response_parts = MagicMock(return_value=[])

        host = MixinHost()
        t0 = time.perf_counter()
        result_p = await host._async_run_tool_loop(
            client=client_p,
            prompt="test",
            tool_registry=registry,
            system_instruction="sys",
            history=[],
            max_iterations=5,
            parallel_tool_calls=True,
        )
        parallel_time = time.perf_counter() - t0

        # Sequential run
        client_s = MagicMock()
        client_s.async_generate_with_tools = AsyncMock(
            side_effect=[
                _make_fc_result(fc_calls),
                _make_text_result("slow"),
            ],
        )
        client_s.build_function_response_parts = MagicMock(return_value=[])

        t1 = time.perf_counter()
        result_s = await host._async_run_tool_loop(
            client=client_s,
            prompt="test",
            tool_registry=registry,
            system_instruction="sys",
            history=[],
            max_iterations=5,
            parallel_tool_calls=False,
        )
        sequential_time = time.perf_counter() - t1

        assert result_p.text == "fast"
        assert result_s.text == "slow"
        # Parallel should be meaningfully faster
        # 3 tools * 0.1s = 0.3s sequential vs ~0.1s parallel
        assert parallel_time < sequential_time, (
            f"Parallel ({parallel_time:.3f}s) should be faster than sequential ({sequential_time:.3f}s)"
        )


# ══════════════════════════════════════════════════════════════
# Sync path stays sequential
# ══════════════════════════════════════════════════════════════


class TestSyncPathSequential:
    """Sync _run_tool_loop should always run sequentially."""

    def test_sync_loop_runs_sequentially(self) -> None:
        """Sync path has no parallel option — it always runs tools one by one."""
        tool_a = _simple_tool("tool_a")
        tool_b = _simple_tool("tool_b")
        registry = _make_registry(tool_a, tool_b)

        client = MagicMock()
        client.generate_with_tools = MagicMock(
            side_effect=[
                _make_fc_result(
                    [
                        {"name": "tool_a", "args": {"query": "a"}},
                        {"name": "tool_b", "args": {"query": "b"}},
                    ]
                ),
                _make_text_result("sync done"),
            ],
        )
        client.build_function_response_parts = MagicMock(return_value=[])

        host = MixinHost()
        result = host._run_tool_loop(
            client=client,
            prompt="test",
            tool_registry=registry,
            system_instruction="sys",
            history=[],
            max_iterations=5,
        )
        assert result.text == "sync done"
        assert len(result.tools_executed) == 2
        assert result.tools_executed[0]["name"] == "tool_a"
        assert result.tools_executed[1]["name"] == "tool_b"

    def test_sync_loop_no_parallel_kwarg(self) -> None:
        """Sync _run_tool_loop should NOT accept parallel_tool_calls kwarg."""
        import inspect

        sig = inspect.signature(ToolLoopMixin._run_tool_loop)
        params = list(sig.parameters.keys())
        assert "parallel_tool_calls" not in params
        assert "max_concurrent_tool_calls" not in params


# ══════════════════════════════════════════════════════════════
# Callback and store integration
# ══════════════════════════════════════════════════════════════


class TestParallelCallbacks:
    """on_tool_call and tool_call_store work correctly with parallel execution."""

    @pytest.mark.asyncio
    async def test_on_tool_call_invoked_for_all_parallel_calls(self) -> None:
        """on_tool_call callback should fire for every tool, parallel or not."""
        tool_a = _simple_tool("tool_a")
        tool_b = _simple_tool("tool_b")
        registry = _make_registry(tool_a, tool_b)

        callback_calls: list[str] = []

        def on_tool(name: str, args: dict, dur: float, success: bool, err: str = "") -> None:
            callback_calls.append(name)

        client = MagicMock()
        client.async_generate_with_tools = AsyncMock(
            side_effect=[
                _make_fc_result(
                    [
                        {"name": "tool_a", "args": {"query": "a"}},
                        {"name": "tool_b", "args": {"query": "b"}},
                    ]
                ),
                _make_text_result("done"),
            ],
        )
        client.build_function_response_parts = MagicMock(return_value=[])

        host = MixinHost()
        result = await host._async_run_tool_loop(
            client=client,
            prompt="test",
            tool_registry=registry,
            system_instruction="sys",
            history=[],
            max_iterations=5,
            parallel_tool_calls=True,
            on_tool_call=on_tool,
        )
        assert result.text == "done"
        assert "tool_a" in callback_calls
        assert "tool_b" in callback_calls
        assert len(callback_calls) == 2

    @pytest.mark.asyncio
    async def test_tool_call_store_records_parallel_calls(self) -> None:
        """tool_call_store.record() should be called for each parallel tool."""
        tool_a = _simple_tool("tool_a")
        tool_b = _simple_tool("tool_b")
        registry = _make_registry(tool_a, tool_b)

        store = MagicMock()
        store.run_id = "test-run-123"

        client = MagicMock()
        client.async_generate_with_tools = AsyncMock(
            side_effect=[
                _make_fc_result(
                    [
                        {"name": "tool_a", "args": {"query": "a"}},
                        {"name": "tool_b", "args": {"query": "b"}},
                    ]
                ),
                _make_text_result("stored"),
            ],
        )
        client.build_function_response_parts = MagicMock(return_value=[])

        host = MixinHost()
        result = await host._async_run_tool_loop(
            client=client,
            prompt="test",
            tool_registry=registry,
            system_instruction="sys",
            history=[],
            max_iterations=5,
            parallel_tool_calls=True,
            tool_call_store=store,
            agent_name="test-agent",
        )
        assert result.text == "stored"
        assert store.record.call_count == 2


# ══════════════════════════════════════════════════════════════
# Helper method unit tests
# ══════════════════════════════════════════════════════════════


class TestNotifyToolCall:
    """Unit tests for the _notify_tool_call static method."""

    def test_notify_with_none_callback(self) -> None:
        """Should be a no-op when callback is None."""
        # Should not raise
        ToolLoopMixin._notify_tool_call(
            None,
            "tool",
            {},
            0.1,
            ToolResult(output="ok"),
        )

    def test_notify_invokes_callback(self) -> None:
        """Should invoke the callback with correct args."""
        calls: list[tuple] = []

        def cb(name: str, args: dict, dur: float, ok: bool, err: str = "") -> None:
            calls.append((name, args, dur, ok, err))

        ToolLoopMixin._notify_tool_call(
            cb,
            "my_tool",
            {"x": 1},
            0.5,
            ToolResult(output="result"),
        )
        assert len(calls) == 1
        assert calls[0][0] == "my_tool"
        assert calls[0][3] is True  # success
        assert calls[0][4] == ""  # no error

    def test_notify_with_error_result(self) -> None:
        """Should pass error_message when tool result has error=True."""
        calls: list[tuple] = []

        def cb(name: str, args: dict, dur: float, ok: bool, err: str = "") -> None:
            calls.append((name, ok, err))

        ToolLoopMixin._notify_tool_call(
            cb,
            "fail_tool",
            {},
            0.1,
            ToolResult(output="Something went wrong", error=True),
        )
        assert calls[0][1] is False
        assert "Something went wrong" in calls[0][2]


class TestRecordToolCall:
    """Unit tests for the _record_tool_call static method."""

    def test_record_with_none_store(self) -> None:
        """Should be a no-op when store is None."""
        # Should not raise
        ToolLoopMixin._record_tool_call(
            None,
            "tool",
            {},
            ToolResult(output="ok"),
            0.1,
            "agent",
            1,
        )

    def test_record_invokes_store(self) -> None:
        """Should call store.record() with a ToolCallRecord."""
        store = MagicMock()
        store.run_id = "run-1"

        ToolLoopMixin._record_tool_call(
            store,
            "my_tool",
            {"q": "test"},
            ToolResult(output="result"),
            0.5,
            "agent-1",
            3,
        )
        assert store.record.call_count == 1
        record = store.record.call_args[0][0]
        assert record.tool_name == "my_tool"
        assert record.agent_name == "agent-1"
        assert record.iteration == 3

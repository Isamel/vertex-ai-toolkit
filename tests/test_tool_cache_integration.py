"""Tests for ToolResultCache integration in ToolLoopMixin (sync + async).

Covers:
- Sync: tool call cached on second identical call
- Sync: different args → cache miss
- Sync: cacheable=False tool → always executes
- Sync: error result → not cached, retry executes fresh
- Sync: cache=None → no caching (backward compat)
- Async: same set of tests for async path
- Telemetry: cached flag appears in ToolCallRecord and ToolExecuted event
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from vaig.agents.mixins import ToolLoopMixin
from vaig.core.cache import ToolResultCache
from vaig.core.client import ToolCallResult
from vaig.core.tool_call_store import ToolCallStore
from vaig.tools.base import ToolCallRecord, ToolDef, ToolParam, ToolRegistry, ToolResult

# ── Helpers ──────────────────────────────────────────────────


class MixinHost(ToolLoopMixin):
    """Concrete class that inherits ToolLoopMixin for testing."""

    pass


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


def _echo_tool(**kwargs: Any) -> ToolResult:
    """Tool that echoes its args as output."""
    return ToolResult(output=f"echo: {kwargs}")


def _failing_tool(**kwargs: Any) -> ToolResult:
    """Tool that always returns an error."""
    return ToolResult(output="something went wrong", error=True)


_call_counter = 0


def _counting_tool(**kwargs: Any) -> ToolResult:
    """Tool that counts how many times it has been called."""
    global _call_counter
    _call_counter += 1
    return ToolResult(output=f"call_{_call_counter}")


# ══════════════════════════════════════════════════════════════
# Sync cache integration tests
# ══════════════════════════════════════════════════════════════


class TestSyncCacheIntegration:
    """Tests for tool result caching in sync _run_tool_loop."""

    def _run_sync(
        self,
        registry: ToolRegistry,
        client: MagicMock,
        *,
        cache: ToolResultCache | None = None,
        tool_call_store: ToolCallStore | None = None,
    ) -> Any:
        host = MixinHost()
        return host._run_tool_loop(
            client=client,
            prompt="test prompt",
            tool_registry=registry,
            system_instruction="You are a test.",
            history=[],
            max_iterations=5,
            tool_result_cache=cache,
            tool_call_store=tool_call_store,
        )

    def test_cached_on_second_identical_call(self) -> None:
        """Second identical tool call uses cache, tool only executes once."""
        call_count = 0

        def counting_tool(**kwargs: Any) -> ToolResult:
            nonlocal call_count
            call_count += 1
            return ToolResult(output=f"result_{call_count}")

        tool = ToolDef(
            name="get_pods",
            description="Get pods",
            parameters=[ToolParam("ns", "string", "namespace")],
            execute=counting_tool,
            cacheable=True,
            cache_ttl_seconds=0,
        )
        registry = _make_registry(tool)
        cache = ToolResultCache(default_ttl=0, max_size=100)

        # Two iterations: first calls tool, second calls same tool, third returns text
        fc = [{"name": "get_pods", "args": {"ns": "default"}}]
        client = MagicMock()
        client.generate_with_tools.side_effect = [
            _make_fc_result(fc),
            _make_fc_result(fc),
            _make_text_result("done"),
        ]
        client.build_function_response_parts.return_value = []

        result = self._run_sync(registry, client, cache=cache)

        assert result.text == "done"
        # Tool was only executed once — second call was a cache hit
        assert call_count == 1
        assert cache.stats().hits == 1
        assert cache.stats().misses == 1  # first lookup was a miss

    def test_different_args_cache_miss(self) -> None:
        """Different tool args result in separate cache entries."""
        call_count = 0

        def counting_tool(**kwargs: Any) -> ToolResult:
            nonlocal call_count
            call_count += 1
            return ToolResult(output=f"result_{call_count}")

        tool = ToolDef(
            name="get_pods",
            description="Get pods",
            parameters=[ToolParam("ns", "string", "namespace")],
            execute=counting_tool,
            cacheable=True,
            cache_ttl_seconds=0,
        )
        registry = _make_registry(tool)
        cache = ToolResultCache(default_ttl=0, max_size=100)

        client = MagicMock()
        client.generate_with_tools.side_effect = [
            _make_fc_result([{"name": "get_pods", "args": {"ns": "default"}}]),
            _make_fc_result([{"name": "get_pods", "args": {"ns": "kube-system"}}]),
            _make_text_result("done"),
        ]
        client.build_function_response_parts.return_value = []

        self._run_sync(registry, client, cache=cache)

        # Both calls executed because args differ
        assert call_count == 2
        assert cache.stats().hits == 0
        assert cache.stats().misses == 2

    def test_cacheable_false_always_executes(self) -> None:
        """Tool with cacheable=False always executes, never caches."""
        call_count = 0

        def counting_tool(**kwargs: Any) -> ToolResult:
            nonlocal call_count
            call_count += 1
            return ToolResult(output=f"result_{call_count}")

        tool = ToolDef(
            name="delete_pod",
            description="Delete a pod",
            parameters=[ToolParam("name", "string", "pod name")],
            execute=counting_tool,
            cacheable=False,
            cache_ttl_seconds=0,
        )
        registry = _make_registry(tool)
        cache = ToolResultCache(default_ttl=0, max_size=100)

        fc = [{"name": "delete_pod", "args": {"name": "my-pod"}}]
        client = MagicMock()
        client.generate_with_tools.side_effect = [
            _make_fc_result(fc),
            _make_fc_result(fc),
            _make_text_result("done"),
        ]
        client.build_function_response_parts.return_value = []

        self._run_sync(registry, client, cache=cache)

        # Both calls executed because cacheable=False
        assert call_count == 2
        assert cache.size == 0  # nothing stored

    def test_error_result_not_cached(self) -> None:
        """Error results are not cached — retry executes fresh."""
        call_count = 0

        def flaky_tool(**kwargs: Any) -> ToolResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ToolResult(output="transient error", error=True)
            return ToolResult(output="success")

        tool = ToolDef(
            name="flaky",
            description="Flaky tool",
            parameters=None,
            execute=flaky_tool,
            cacheable=True,
            cache_ttl_seconds=0,
        )
        registry = _make_registry(tool)
        cache = ToolResultCache(default_ttl=0, max_size=100)

        fc = [{"name": "flaky", "args": {}}]
        client = MagicMock()
        client.generate_with_tools.side_effect = [
            _make_fc_result(fc),
            _make_fc_result(fc),
            _make_text_result("done"),
        ]
        client.build_function_response_parts.return_value = []

        self._run_sync(registry, client, cache=cache)

        # First call errored (not cached), second call executed fresh
        assert call_count == 2
        # Second successful result should now be in cache
        assert cache.size == 1

    def test_cache_none_backward_compat(self) -> None:
        """When tool_result_cache=None, tools execute normally."""
        call_count = 0

        def counting_tool(**kwargs: Any) -> ToolResult:
            nonlocal call_count
            call_count += 1
            return ToolResult(output=f"result_{call_count}")

        tool = ToolDef(
            name="get_pods",
            description="Get pods",
            parameters=None,
            execute=counting_tool,
            cacheable=True,
        )
        registry = _make_registry(tool)

        fc = [{"name": "get_pods", "args": {}}]
        client = MagicMock()
        client.generate_with_tools.side_effect = [
            _make_fc_result(fc),
            _make_fc_result(fc),
            _make_text_result("done"),
        ]
        client.build_function_response_parts.return_value = []

        # No cache passed
        self._run_sync(registry, client, cache=None)

        # Both calls executed (no caching)
        assert call_count == 2


# ══════════════════════════════════════════════════════════════
# Async cache integration tests
# ══════════════════════════════════════════════════════════════


class TestAsyncCacheIntegration:
    """Tests for tool result caching in async _async_run_tool_loop."""

    async def _run_async(
        self,
        registry: ToolRegistry,
        client: MagicMock,
        *,
        cache: ToolResultCache | None = None,
        tool_call_store: ToolCallStore | None = None,
        parallel: bool = False,
    ) -> Any:
        host = MixinHost()
        return await host._async_run_tool_loop(
            client=client,
            prompt="test prompt",
            tool_registry=registry,
            system_instruction="You are a test.",
            history=[],
            max_iterations=5,
            tool_result_cache=cache,
            tool_call_store=tool_call_store,
            parallel_tool_calls=parallel,
        )

    @pytest.mark.asyncio
    async def test_cached_on_second_identical_call(self) -> None:
        """Second identical tool call uses cache, tool only executes once."""
        call_count = 0

        def counting_tool(**kwargs: Any) -> ToolResult:
            nonlocal call_count
            call_count += 1
            return ToolResult(output=f"result_{call_count}")

        tool = ToolDef(
            name="get_pods",
            description="Get pods",
            parameters=[ToolParam("ns", "string", "namespace")],
            execute=counting_tool,
            cacheable=True,
            cache_ttl_seconds=0,
        )
        registry = _make_registry(tool)
        cache = ToolResultCache(default_ttl=0, max_size=100)

        fc = [{"name": "get_pods", "args": {"ns": "default"}}]
        client = MagicMock()
        client.async_generate_with_tools = AsyncMock(
            side_effect=[
                _make_fc_result(fc),
                _make_fc_result(fc),
                _make_text_result("done"),
            ]
        )
        client.build_function_response_parts.return_value = []

        result = await self._run_async(registry, client, cache=cache)

        assert result.text == "done"
        assert call_count == 1
        assert cache.stats().hits == 1

    @pytest.mark.asyncio
    async def test_different_args_cache_miss(self) -> None:
        """Different tool args result in separate cache entries."""
        call_count = 0

        def counting_tool(**kwargs: Any) -> ToolResult:
            nonlocal call_count
            call_count += 1
            return ToolResult(output=f"result_{call_count}")

        tool = ToolDef(
            name="get_pods",
            description="Get pods",
            parameters=[ToolParam("ns", "string", "namespace")],
            execute=counting_tool,
            cacheable=True,
            cache_ttl_seconds=0,
        )
        registry = _make_registry(tool)
        cache = ToolResultCache(default_ttl=0, max_size=100)

        client = MagicMock()
        client.async_generate_with_tools = AsyncMock(
            side_effect=[
                _make_fc_result([{"name": "get_pods", "args": {"ns": "default"}}]),
                _make_fc_result([{"name": "get_pods", "args": {"ns": "kube-system"}}]),
                _make_text_result("done"),
            ]
        )
        client.build_function_response_parts.return_value = []

        await self._run_async(registry, client, cache=cache)

        assert call_count == 2
        assert cache.stats().hits == 0

    @pytest.mark.asyncio
    async def test_cacheable_false_always_executes(self) -> None:
        """Tool with cacheable=False always executes."""
        call_count = 0

        def counting_tool(**kwargs: Any) -> ToolResult:
            nonlocal call_count
            call_count += 1
            return ToolResult(output=f"result_{call_count}")

        tool = ToolDef(
            name="delete_pod",
            description="Delete a pod",
            parameters=None,
            execute=counting_tool,
            cacheable=False,
        )
        registry = _make_registry(tool)
        cache = ToolResultCache(default_ttl=0, max_size=100)

        fc = [{"name": "delete_pod", "args": {}}]
        client = MagicMock()
        client.async_generate_with_tools = AsyncMock(
            side_effect=[
                _make_fc_result(fc),
                _make_fc_result(fc),
                _make_text_result("done"),
            ]
        )
        client.build_function_response_parts.return_value = []

        await self._run_async(registry, client, cache=cache)

        assert call_count == 2
        assert cache.size == 0

    @pytest.mark.asyncio
    async def test_error_result_not_cached(self) -> None:
        """Error results are not cached."""
        call_count = 0

        def flaky_tool(**kwargs: Any) -> ToolResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ToolResult(output="transient error", error=True)
            return ToolResult(output="success")

        tool = ToolDef(
            name="flaky",
            description="Flaky",
            parameters=None,
            execute=flaky_tool,
            cacheable=True,
            cache_ttl_seconds=0,
        )
        registry = _make_registry(tool)
        cache = ToolResultCache(default_ttl=0, max_size=100)

        fc = [{"name": "flaky", "args": {}}]
        client = MagicMock()
        client.async_generate_with_tools = AsyncMock(
            side_effect=[
                _make_fc_result(fc),
                _make_fc_result(fc),
                _make_text_result("done"),
            ]
        )
        client.build_function_response_parts.return_value = []

        await self._run_async(registry, client, cache=cache)

        assert call_count == 2
        assert cache.size == 1

    @pytest.mark.asyncio
    async def test_cache_none_backward_compat(self) -> None:
        """When tool_result_cache=None, tools execute normally."""
        call_count = 0

        def counting_tool(**kwargs: Any) -> ToolResult:
            nonlocal call_count
            call_count += 1
            return ToolResult(output=f"result_{call_count}")

        tool = ToolDef(
            name="get_pods",
            description="Get pods",
            parameters=None,
            execute=counting_tool,
            cacheable=True,
        )
        registry = _make_registry(tool)

        fc = [{"name": "get_pods", "args": {}}]
        client = MagicMock()
        client.async_generate_with_tools = AsyncMock(
            side_effect=[
                _make_fc_result(fc),
                _make_fc_result(fc),
                _make_text_result("done"),
            ]
        )
        client.build_function_response_parts.return_value = []

        await self._run_async(registry, client, cache=None)

        assert call_count == 2


# ══════════════════════════════════════════════════════════════
# Telemetry: cached flag
# ══════════════════════════════════════════════════════════════


class TestCachedFlagTelemetry:
    """Tests that the cached flag propagates to records and events."""

    def test_tool_call_record_cached_field(self) -> None:
        """ToolCallRecord.to_dict() includes cached field."""
        record = ToolCallRecord(
            tool_name="test",
            tool_args={"a": 1},
            output="output",
            output_size_bytes=6,
            error=False,
            error_type="",
            error_message="",
            duration_s=0.1,
            timestamp="2026-01-01T00:00:00Z",
            agent_name="test_agent",
            run_id="abc123",
            iteration=1,
            cached=True,
        )
        d = record.to_dict()
        assert d["cached"] is True

    def test_tool_call_record_cached_defaults_false(self) -> None:
        """ToolCallRecord.cached defaults to False."""
        record = ToolCallRecord(
            tool_name="test",
            tool_args={},
            output="",
            output_size_bytes=0,
            error=False,
            error_type="",
            error_message="",
            duration_s=0.0,
            timestamp="",
            agent_name="",
            run_id="",
            iteration=0,
        )
        assert record.cached is False
        assert record.to_dict()["cached"] is False

    def test_tool_executed_event_cached_field(self) -> None:
        """ToolExecuted event has a cached field."""
        from vaig.core.events import ToolExecuted

        event = ToolExecuted(
            tool_name="test_tool",
            duration_ms=1.0,
            cached=True,
        )
        assert event.cached is True

    def test_tool_executed_event_cached_defaults_false(self) -> None:
        """ToolExecuted.cached defaults to False."""
        from vaig.core.events import ToolExecuted

        event = ToolExecuted(tool_name="test_tool")
        assert event.cached is False

    def test_record_tool_call_passes_cached_flag(self) -> None:
        """_record_tool_call creates a ToolCallRecord with cached=True on cache hit."""
        import tempfile

        store = ToolCallStore(base_dir=tempfile.mkdtemp())
        store.start_run("test-run")

        tool_result = ToolResult(output="cached output", error=False)

        # Record with cached=True
        ToolLoopMixin._record_tool_call(
            store,
            "get_pods",
            {"ns": "default"},
            tool_result,
            0.0,
            "test_agent",
            1,
            cached=True,
        )

        # Read the JSONL output and verify
        import json

        run_file = store.get_run_file()
        assert run_file is not None
        with open(run_file) as f:
            line = f.readline()
        data = json.loads(line)
        assert data["cached"] is True

    def test_record_tool_call_cached_false_by_default(self) -> None:
        """_record_tool_call defaults cached to False for backward compat."""
        import tempfile

        store = ToolCallStore(base_dir=tempfile.mkdtemp())
        store.start_run("test-run-2")

        tool_result = ToolResult(output="normal output", error=False)

        # Record without cached= kwarg
        ToolLoopMixin._record_tool_call(
            store,
            "get_pods",
            {"ns": "default"},
            tool_result,
            0.5,
            "test_agent",
            1,
        )

        import json

        run_file = store.get_run_file()
        assert run_file is not None
        with open(run_file) as f:
            line = f.readline()
        data = json.loads(line)
        assert data["cached"] is False

    def test_cached_result_recorded_in_tool_call_store(self) -> None:
        """Cache hit is recorded in ToolCallStore with cached=True."""
        import json
        import tempfile

        call_count = 0

        def counting_tool(**kwargs: Any) -> ToolResult:
            nonlocal call_count
            call_count += 1
            return ToolResult(output=f"result_{call_count}")

        tool = ToolDef(
            name="get_pods",
            description="Get pods",
            parameters=None,
            execute=counting_tool,
            cacheable=True,
            cache_ttl_seconds=0,
        )
        registry = _make_registry(tool)
        cache = ToolResultCache(default_ttl=0, max_size=100)

        store = ToolCallStore(base_dir=tempfile.mkdtemp())
        store.start_run("test-run-3")

        fc = [{"name": "get_pods", "args": {}}]
        client = MagicMock()
        client.generate_with_tools.side_effect = [
            _make_fc_result(fc),
            _make_fc_result(fc),
            _make_text_result("done"),
        ]
        client.build_function_response_parts.return_value = []

        host = MixinHost()
        host._run_tool_loop(
            client=client,
            prompt="test",
            tool_registry=registry,
            system_instruction="test",
            history=[],
            max_iterations=5,
            tool_result_cache=cache,
            tool_call_store=store,
        )

        run_file = store.get_run_file()
        assert run_file is not None
        with open(run_file) as f:
            lines = f.readlines()

        assert len(lines) == 2
        first_record = json.loads(lines[0])
        second_record = json.loads(lines[1])

        assert first_record["cached"] is False
        assert second_record["cached"] is True
        assert call_count == 1

    def test_emit_tool_telemetry_cached_flag(self) -> None:
        """_emit_tool_telemetry passes cached flag to ToolExecuted event."""
        from vaig.core.event_bus import EventBus
        from vaig.core.events import ToolExecuted

        bus = EventBus.get()
        captured_events: list[ToolExecuted] = []

        def listener(event: ToolExecuted) -> None:
            captured_events.append(event)

        unsub = bus.subscribe(ToolExecuted, listener)
        try:
            import time

            t0 = time.perf_counter()
            tool_result = ToolResult(output="cached", error=False)

            ToolLoopMixin._emit_tool_telemetry(
                "get_pods",
                {"ns": "default"},
                tool_result,
                t0,
                cached=True,
            )

            assert len(captured_events) == 1
            assert captured_events[0].cached is True
            assert captured_events[0].tool_name == "get_pods"
        finally:
            unsub()

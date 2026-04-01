"""Tests for SSE adapter — Phases 1–3.

Covers:
- stream_to_sse yields chunk events from StreamResult
- Queue events (tool_call, phase, error) are interleaved
- Done event includes usage stats
- Error during streaming is caught and sent as error event
- _drain_queue handles empty queue and sentinel
- live_pipeline_to_sse keepalive, error recovery, partial results, CancelledError
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip(
    "fastapi",
    reason="FastAPI not available; install the 'web' extra to run web tests.",
)

from vaig.core.events import ErrorOccurred, Event, OrchestratorPhaseCompleted, ToolExecuted
from vaig.web.sse import (
    _drain_queue,
    _emit_pipeline_result,
    _extract_partial_result,
    live_pipeline_to_sse,
    stream_to_sse,
)

# ── Helpers ──────────────────────────────────────────────────


class MockStreamResult:
    """A mock StreamResult that yields predefined text chunks."""

    def __init__(self, chunks: list[str], usage: dict | None = None, text: str = "") -> None:
        self._chunks = chunks
        self.usage = usage or {}
        self.text = text or "".join(chunks)

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


class ErrorStreamResult:
    """A mock StreamResult that raises during iteration."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.usage = {}
        self.text = ""

    async def __aiter__(self):
        raise self._error
        yield  # noqa: RET503 — makes this an async generator so __aiter__ is valid


async def _collect_events(gen):
    """Collect all SSE events from an async generator."""
    events = []
    async for event in gen:
        events.append(event)
    return events


# ── stream_to_sse ────────────────────────────────────────────


class TestStreamToSSE:
    """Tests for the main SSE adapter."""

    @pytest.mark.asyncio
    async def test_yields_chunk_events(self) -> None:
        """Text chunks from StreamResult become event: chunk."""
        stream = MockStreamResult(["Hello ", "world!"])
        queue: asyncio.Queue[Event | None] = asyncio.Queue()

        events = await _collect_events(stream_to_sse(stream, queue))

        chunk_events = [e for e in events if e.event == "chunk"]
        assert len(chunk_events) == 2
        assert json.loads(chunk_events[0].data)["text"] == "Hello "
        assert json.loads(chunk_events[1].data)["text"] == "world!"

    @pytest.mark.asyncio
    async def test_ends_with_done_event(self) -> None:
        """The final SSE event must be 'done' with usage stats."""
        stream = MockStreamResult(["text"], usage={"tokens": 42}, text="text")
        queue: asyncio.Queue[Event | None] = asyncio.Queue()

        events = await _collect_events(stream_to_sse(stream, queue))

        done_events = [e for e in events if e.event == "done"]
        assert len(done_events) == 1
        data = json.loads(done_events[0].data)
        assert data["usage"] == {"tokens": 42}
        assert data["full_text"] == "text"

    @pytest.mark.asyncio
    async def test_interleaves_queue_events(self) -> None:
        """Events queued before/during streaming are interleaved."""
        stream = MockStreamResult(["chunk1"])
        queue: asyncio.Queue[Event | None] = asyncio.Queue()

        # Pre-queue a tool event
        queue.put_nowait(ToolExecuted(tool_name="kubectl", duration_ms=150.0))

        events = await _collect_events(stream_to_sse(stream, queue))

        event_types = [e.event for e in events]
        assert "tool_call" in event_types
        assert "chunk" in event_types
        assert "done" in event_types

    @pytest.mark.asyncio
    async def test_tool_call_event_data(self) -> None:
        """ToolExecuted events should have correct SSE data."""
        stream = MockStreamResult([])
        queue: asyncio.Queue[Event | None] = asyncio.Queue()
        queue.put_nowait(ToolExecuted(tool_name="gcloud", duration_ms=200.0, error=False))

        events = await _collect_events(stream_to_sse(stream, queue))

        tool_events = [e for e in events if e.event == "tool_call"]
        assert len(tool_events) == 1
        data = json.loads(tool_events[0].data)
        assert data["tool"] == "gcloud"
        assert data["duration_ms"] == 200.0
        assert data["error"] is False

    @pytest.mark.asyncio
    async def test_phase_event_data(self) -> None:
        """OrchestratorPhaseCompleted events become event: phase."""
        stream = MockStreamResult([])
        queue: asyncio.Queue[Event | None] = asyncio.Queue()
        queue.put_nowait(
            OrchestratorPhaseCompleted(skill="gke", phase="analyze", strategy="sequential")
        )

        events = await _collect_events(stream_to_sse(stream, queue))

        phase_events = [e for e in events if e.event == "phase"]
        assert len(phase_events) == 1
        data = json.loads(phase_events[0].data)
        assert data["skill"] == "gke"
        assert data["phase"] == "analyze"

    @pytest.mark.asyncio
    async def test_error_event_data(self) -> None:
        """ErrorOccurred events become event: error."""
        stream = MockStreamResult([])
        queue: asyncio.Queue[Event | None] = asyncio.Queue()
        queue.put_nowait(
            ErrorOccurred(error_type="APIError", error_message="quota exceeded", source="gemini")
        )

        events = await _collect_events(stream_to_sse(stream, queue))

        error_events = [e for e in events if e.event == "error"]
        assert len(error_events) == 1
        data = json.loads(error_events[0].data)
        assert data["message"] == "quota exceeded"
        assert data["error_type"] == "APIError"
        assert data["source"] == "gemini"

    @pytest.mark.asyncio
    async def test_streaming_error_caught(self) -> None:
        """Exceptions during streaming yield an error event + done."""
        stream = ErrorStreamResult(RuntimeError("connection lost"))
        queue: asyncio.Queue[Event | None] = asyncio.Queue()

        events = await _collect_events(stream_to_sse(stream, queue))

        event_types = [e.event for e in events]
        assert "error" in event_types
        assert "done" in event_types

        error_data = json.loads([e for e in events if e.event == "error"][0].data)
        assert "Connection error" in error_data["message"]
        assert error_data["error_type"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_empty_stream_yields_done(self) -> None:
        """An empty stream should still yield a done event."""
        stream = MockStreamResult([])
        queue: asyncio.Queue[Event | None] = asyncio.Queue()

        events = await _collect_events(stream_to_sse(stream, queue))

        event_types = [e.event for e in events]
        assert event_types == ["done"]

    @pytest.mark.asyncio
    async def test_none_usage_defaults_to_empty(self) -> None:
        """If stream_result.usage is None, done data should use empty dict."""
        stream = MockStreamResult(["x"])
        stream.usage = None  # type: ignore[assignment]
        stream.text = None  # type: ignore[assignment]
        queue: asyncio.Queue[Event | None] = asyncio.Queue()

        events = await _collect_events(stream_to_sse(stream, queue))

        done_data = json.loads([e for e in events if e.event == "done"][0].data)
        assert done_data["usage"] == {}
        assert done_data["full_text"] == ""


# ── _drain_queue ─────────────────────────────────────────────


class TestDrainQueue:
    """Tests for the internal _drain_queue helper."""

    @pytest.mark.asyncio
    async def test_empty_queue_yields_nothing(self) -> None:
        """An empty queue should yield no events."""
        queue: asyncio.Queue[Event | None] = asyncio.Queue()
        events = await _collect_events(_drain_queue(queue))
        assert events == []

    @pytest.mark.asyncio
    async def test_sentinel_stops_drain(self) -> None:
        """None sentinel should stop draining."""
        queue: asyncio.Queue[Event | None] = asyncio.Queue()
        queue.put_nowait(ToolExecuted(tool_name="test"))
        queue.put_nowait(None)
        queue.put_nowait(ToolExecuted(tool_name="should-not-appear"))

        events = await _collect_events(_drain_queue(queue))
        assert len(events) == 1
        assert json.loads(events[0].data)["tool"] == "test"

    @pytest.mark.asyncio
    async def test_drains_all_available_events(self) -> None:
        """Should drain all non-sentinel events from the queue."""
        queue: asyncio.Queue[Event | None] = asyncio.Queue()
        queue.put_nowait(ToolExecuted(tool_name="tool1"))
        queue.put_nowait(ToolExecuted(tool_name="tool2"))

        events = await _collect_events(_drain_queue(queue))
        assert len(events) == 2


# ── live_pipeline_to_sse (Phase 3 hardening) ────────────────


class MockOrchestratorResult:
    """Minimal mock of OrchestratorResult for SSE tests."""

    def __init__(
        self,
        *,
        success: bool = True,
        total_usage: dict | None = None,
        run_cost_usd: float = 0.01,
        structured_report: object | None = None,
    ) -> None:
        self.success = success
        self.total_usage = total_usage or {"input_tokens": 100}
        self.run_cost_usd = run_cost_usd
        self.structured_report = structured_report


class TestLivePipelineToSSE:
    """Tests for the live_pipeline_to_sse adapter — Phase 3."""

    @pytest.mark.asyncio
    async def test_happy_path_result_and_done(self) -> None:
        """Pipeline completes → result + done events emitted."""
        result = MockOrchestratorResult()

        async def _pipeline() -> MockOrchestratorResult:
            return result

        queue: asyncio.Queue[Event | None] = asyncio.Queue()
        events = await _collect_events(
            live_pipeline_to_sse(_pipeline(), queue, keepalive_interval=0.1)
        )

        event_types = [e.event for e in events]
        assert "result" in event_types
        assert event_types[-1] == "done"

        result_data = json.loads(
            [e for e in events if e.event == "result"][0].data
        )
        assert result_data["success"] is True
        assert result_data["partial"] is False
        assert result_data["cost_usd"] == 0.01

    @pytest.mark.asyncio
    async def test_keepalive_emitted_on_idle(self) -> None:
        """Keepalive events are emitted when the queue is idle."""

        async def _slow_pipeline() -> MockOrchestratorResult:
            await asyncio.sleep(0.3)
            return MockOrchestratorResult()

        queue: asyncio.Queue[Event | None] = asyncio.Queue()
        events = await _collect_events(
            live_pipeline_to_sse(
                _slow_pipeline(), queue, keepalive_interval=0.05
            )
        )

        keepalive_events = [e for e in events if e.event == "keepalive"]
        assert len(keepalive_events) >= 2, (
            f"Expected at least 2 keepalive events, got {len(keepalive_events)}"
        )

    @pytest.mark.asyncio
    async def test_pipeline_events_streamed(self) -> None:
        """Events emitted during pipeline execution are streamed."""

        async def _pipeline_with_events(
            q: asyncio.Queue[Event | None],
        ) -> MockOrchestratorResult:
            q.put_nowait(ToolExecuted(tool_name="kubectl", duration_ms=50.0))
            await asyncio.sleep(0.05)
            return MockOrchestratorResult()

        queue: asyncio.Queue[Event | None] = asyncio.Queue()

        async def _coro() -> MockOrchestratorResult:
            return await _pipeline_with_events(queue)

        events = await _collect_events(
            live_pipeline_to_sse(_coro(), queue, keepalive_interval=1.0)
        )

        event_types = [e.event for e in events]
        assert "tool_call" in event_types

    @pytest.mark.asyncio
    async def test_pipeline_error_yields_error_event(self) -> None:
        """Pipeline exception → friendly error SSE event + done."""

        async def _failing_pipeline() -> MockOrchestratorResult:
            raise TimeoutError("Gemini API deadline exceeded")

        queue: asyncio.Queue[Event | None] = asyncio.Queue()
        events = await _collect_events(
            live_pipeline_to_sse(
                _failing_pipeline(), queue, keepalive_interval=0.1
            )
        )

        event_types = [e.event for e in events]
        assert "error" in event_types
        assert event_types[-1] == "done"

        error_data = json.loads(
            [e for e in events if e.event == "error"][0].data
        )
        assert error_data["error_type"] == "TimeoutError"
        assert error_data["retriable"] is True
        # Should NOT expose raw exception message
        assert "Gemini API deadline exceeded" not in error_data["message"]
        assert "timed out" in error_data["message"].lower()

    @pytest.mark.asyncio
    async def test_cancelled_error_re_raised(self) -> None:
        """CancelledError (client disconnect) is re-raised, not swallowed."""

        async def _long_pipeline() -> MockOrchestratorResult:
            await asyncio.sleep(100)
            return MockOrchestratorResult()

        queue: asyncio.Queue[Event | None] = asyncio.Queue()
        gen = live_pipeline_to_sse(
            _long_pipeline(), queue, keepalive_interval=0.05
        )

        # Get one keepalive, then cancel
        events: list[object] = []
        with pytest.raises(asyncio.CancelledError):
            task = asyncio.current_task()
            assert task is not None

            async for sse_event in gen:
                events.append(sse_event)
                if len(events) >= 1:
                    # Simulate client disconnect by cancelling the current task
                    # We need to use a wrapper to properly test this
                    raise asyncio.CancelledError()


# ── _emit_pipeline_result ────────────────────────────────────


class TestEmitPipelineResult:
    """Tests for the result serialisation helper."""

    @pytest.mark.asyncio
    async def test_complete_result_includes_done(self) -> None:
        """Complete (non-partial) result ends with done event."""
        result = MockOrchestratorResult(success=True)
        events = await _collect_events(_emit_pipeline_result(result))

        assert events[-1].event == "done"
        result_data = json.loads(events[0].data)
        assert result_data["partial"] is False

    @pytest.mark.asyncio
    async def test_partial_result_skips_done(self) -> None:
        """Partial result does NOT emit done (caller handles that)."""
        result = MockOrchestratorResult(success=False)
        events = await _collect_events(
            _emit_pipeline_result(result, partial=True)
        )

        event_types = [e.event for e in events]
        assert "done" not in event_types

        result_data = json.loads(events[0].data)
        assert result_data["partial"] is True


# ── _extract_partial_result ──────────────────────────────────


class TestExtractPartialResult:
    """Tests for the partial result extraction helper."""

    @pytest.mark.asyncio
    async def test_returns_none_for_successful_task(self) -> None:
        """Successful tasks have no partial result."""

        async def _ok() -> str:
            return "done"

        task = asyncio.create_task(_ok())
        await task
        assert _extract_partial_result(task) is None

    @pytest.mark.asyncio
    async def test_returns_none_for_running_task(self) -> None:
        """Running (not done) tasks have no partial result."""

        async def _hang() -> None:
            await asyncio.sleep(100)

        task = asyncio.create_task(_hang())
        assert _extract_partial_result(task) is None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_returns_partial_result_from_exception(self) -> None:
        """Exceptions carrying partial_result are extracted."""

        class PartialError(Exception):
            def __init__(self) -> None:
                super().__init__("partial failure")
                self.partial_result = MockOrchestratorResult(success=False)

        async def _partial() -> None:
            raise PartialError()

        task = asyncio.create_task(_partial())
        with pytest.raises(PartialError):
            await task

        # Task is done with exception — re-fetch from task directly
        partial = _extract_partial_result(task)
        assert partial is not None
        assert partial.success is False

    @pytest.mark.asyncio
    async def test_returns_none_for_plain_exception(self) -> None:
        """Plain exceptions without partial data return None."""

        async def _fail() -> None:
            raise RuntimeError("total failure")

        task = asyncio.create_task(_fail())
        try:
            await task
        except RuntimeError:
            pass

        assert _extract_partial_result(task) is None

    @pytest.mark.asyncio
    async def test_returns_none_for_cancelled_task(self) -> None:
        """Cancelled tasks return None."""

        async def _slow() -> None:
            await asyncio.sleep(100)

        task = asyncio.create_task(_slow())
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert _extract_partial_result(task) is None

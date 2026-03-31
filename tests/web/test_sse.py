"""Tests for SSE adapter — Task 2.3.

Covers:
- stream_to_sse yields chunk events from StreamResult
- Queue events (tool_call, phase, error) are interleaved
- Done event includes usage stats
- Error during streaming is caught and sent as error event
- _drain_queue handles empty queue and sentinel
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
from vaig.web.sse import _drain_queue, stream_to_sse

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

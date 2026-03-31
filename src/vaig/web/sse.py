"""SSE adapter — bridges StreamResult + EventBus queue to Server-Sent Events.

Converts the async iteration of a ``StreamResult`` and an ``asyncio.Queue``
of EventBus events into a stream of ``ServerSentEvent`` objects suitable
for ``sse-starlette``'s ``EventSourceResponse``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from sse_starlette import ServerSentEvent

from vaig.core.events import (
    ErrorOccurred,
    Event,
    OrchestratorPhaseCompleted,
    ToolExecuted,
)

__all__ = [
    "stream_to_sse",
]

logger = logging.getLogger(__name__)

# Mapping from Event subclass to SSE event type string
_EVENT_TYPE_MAP: dict[type[Event], str] = {
    ToolExecuted: "tool_call",
    OrchestratorPhaseCompleted: "phase",
    ErrorOccurred: "error",
}


def _event_to_sse_data(event: Event) -> dict[str, Any]:
    """Convert an EventBus event to SSE-serializable data."""
    if isinstance(event, ToolExecuted):
        return {
            "tool": event.tool_name,
            "duration_ms": event.duration_ms,
            "error": event.error,
            "error_message": event.error_message if event.error else None,
        }
    if isinstance(event, OrchestratorPhaseCompleted):
        return {
            "skill": event.skill,
            "phase": event.phase,
            "strategy": event.strategy,
        }
    if isinstance(event, ErrorOccurred):
        return {
            "message": event.error_message,
            "error_type": event.error_type,
            "source": event.source,
        }
    # Fallback for unknown event types
    return {"event_type": event.event_type}


async def stream_to_sse(
    stream_result: Any,
    event_queue: asyncio.Queue[Event | None],
) -> AsyncGenerator[ServerSentEvent, None]:
    """Yield SSE events from a StreamResult and an EventBus queue.

    Interleaves two sources:

    1. **Text chunks** from ``stream_result.__aiter__()`` → ``event: chunk``
    2. **Domain events** from ``event_queue`` → ``event: tool_call | phase | error``

    After the stream is exhausted, drains any remaining queue events
    and emits a final ``event: done`` with usage statistics.

    Args:
        stream_result: A ``StreamResult`` instance supporting ``async for``.
        event_queue: An ``asyncio.Queue`` fed by EventBus subscriptions.
            A ``None`` sentinel signals no more events.
    """
    # Drain any events already in the queue before streaming
    async for sse_event in _drain_queue(event_queue):
        yield sse_event

    # Stream text chunks
    try:
        async for text in stream_result:
            # Before yielding the chunk, drain any queued events
            async for sse_event in _drain_queue(event_queue):
                yield sse_event

            yield ServerSentEvent(
                data=json.dumps({"text": text}),
                event="chunk",
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error during SSE streaming: %s", exc)
        yield ServerSentEvent(
            data=json.dumps({"message": str(exc), "error_type": type(exc).__name__}),
            event="error",
        )

    # Drain remaining queue events
    async for sse_event in _drain_queue(event_queue):
        yield sse_event

    # Final done event with usage stats
    usage = getattr(stream_result, "usage", {}) or {}
    full_text = getattr(stream_result, "text", "") or ""
    yield ServerSentEvent(
        data=json.dumps({"usage": usage, "full_text": full_text}),
        event="done",
    )


async def _drain_queue(
    queue: asyncio.Queue[Event | None],
) -> AsyncGenerator[ServerSentEvent, None]:
    """Drain all currently available events from the queue without blocking."""
    while True:
        try:
            event = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        if event is None:
            break  # Sentinel — no more events
        sse_type = _EVENT_TYPE_MAP.get(type(event), "status")
        yield ServerSentEvent(
            data=json.dumps(_event_to_sse_data(event)),
            event=sse_type,
        )

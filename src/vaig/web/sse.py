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

    Uses ``asyncio.wait()`` to multiplex two sources so that domain
    events are emitted promptly even when no text chunks are arriving:

    1. **Text chunks** from ``stream_result.__aiter__()`` → ``event: chunk``
    2. **Domain events** from ``event_queue`` → ``event: tool_call | phase | error``

    After the stream is exhausted, drains any remaining queue events
    and emits a final ``event: done`` with usage statistics.

    Args:
        stream_result: A ``StreamResult`` instance supporting ``async for``.
        event_queue: An ``asyncio.Queue`` fed by EventBus subscriptions.
            A ``None`` sentinel signals no more events.
    """
    stream_iter = stream_result.__aiter__()
    stream_done = False

    # Prime the two competing tasks
    next_chunk_task: asyncio.Task[str] | None = asyncio.create_task(
        _anext_or_stop(stream_iter)
    )
    next_event_task: asyncio.Task[Event | None] | None = asyncio.create_task(
        event_queue.get()
    )

    try:
        while not stream_done or next_event_task is not None:
            pending: set[asyncio.Task[Any]] = set()
            if next_chunk_task is not None:
                pending.add(next_chunk_task)
            if next_event_task is not None:
                pending.add(next_event_task)

            if not pending:
                break  # pragma: no cover — safety net

            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                if task is next_chunk_task:
                    try:
                        text = task.result()
                    except _StreamExhaustedError:
                        stream_done = True
                        next_chunk_task = None
                    except Exception as exc:  # noqa: BLE001
                        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                            raise
                        logger.warning("Error during SSE streaming: %s", exc)
                        yield ServerSentEvent(
                            data=json.dumps(
                                {"message": str(exc), "error_type": type(exc).__name__}
                            ),
                            event="error",
                        )
                        stream_done = True
                        next_chunk_task = None
                    else:
                        yield ServerSentEvent(
                            data=json.dumps({"text": text}),
                            event="chunk",
                        )
                        # Schedule the next chunk read
                        next_chunk_task = asyncio.create_task(
                            _anext_or_stop(stream_iter)
                        )

                elif task is next_event_task:
                    event = task.result()
                    if event is None:
                        # Sentinel — no more domain events
                        next_event_task = None
                    else:
                        sse_type = _EVENT_TYPE_MAP.get(type(event), "status")
                        yield ServerSentEvent(
                            data=json.dumps(_event_to_sse_data(event)),
                            event=sse_type,
                        )
                        # Schedule the next event read
                        next_event_task = asyncio.create_task(
                            event_queue.get()
                        )

            # When the stream is exhausted, stop waiting for more queue
            # events — remaining items are drained non-blocking below.
            if stream_done and next_event_task is not None:
                break
    finally:
        # Cancel any outstanding tasks
        for t in (next_chunk_task, next_event_task):
            if t is not None and not t.done():
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass

    # Drain any remaining queue events (non-blocking)
    async for sse_event in _drain_queue(event_queue):
        yield sse_event

    # Final done event with usage stats
    usage = getattr(stream_result, "usage", {}) or {}
    full_text = getattr(stream_result, "text", "") or ""
    yield ServerSentEvent(
        data=json.dumps({"usage": usage, "full_text": full_text}),
        event="done",
    )


class _StreamExhaustedError(Exception):
    """Sentinel exception to signal the async iterator is done."""


async def _anext_or_stop(async_iter: AsyncGenerator[str, None]) -> str:
    """Get next value from an async iterator or raise _StreamExhaustedError."""
    try:
        return await async_iter.__anext__()
    except StopAsyncIteration:
        raise _StreamExhaustedError from None


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

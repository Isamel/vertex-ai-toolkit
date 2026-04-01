"""SSE adapter — bridges StreamResult + EventBus queue to Server-Sent Events.

Converts the async iteration of a ``StreamResult`` and an ``asyncio.Queue``
of EventBus events into a stream of ``ServerSentEvent`` objects suitable
for ``sse-starlette``'s ``EventSourceResponse``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Coroutine
from typing import Any

from sse_starlette import ServerSentEvent

from vaig.core.events import (
    AgentProgressCompleted,
    AgentProgressStarted,
    ErrorOccurred,
    Event,
    OrchestratorPhaseCompleted,
    ToolExecuted,
)

__all__ = [
    "live_pipeline_to_sse",
    "stream_to_sse",
]

logger = logging.getLogger(__name__)

# Mapping from Event subclass to SSE event type string
_EVENT_TYPE_MAP: dict[type[Event], str] = {
    ToolExecuted: "tool_call",
    OrchestratorPhaseCompleted: "phase",
    ErrorOccurred: "error",
    AgentProgressStarted: "agent_start",
    AgentProgressCompleted: "agent_end",
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
    if isinstance(event, AgentProgressStarted):
        return {
            "agent": event.agent_name,
            "index": event.agent_index,
            "total": event.total_agents,
        }
    if isinstance(event, AgentProgressCompleted):
        return {
            "agent": event.agent_name,
            "index": event.agent_index,
            "total": event.total_agents,
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
                        user_message = _friendly_error_message(exc)
                        yield ServerSentEvent(
                            data=json.dumps({
                                "message": user_message,
                                "error_type": type(exc).__name__,
                                "retriable": _is_retriable(exc),
                            }),
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


# ── Error classification helpers ─────────────────────────────


# Error type substrings that indicate a Vertex / Google API issue
_VERTEX_ERROR_MARKERS: tuple[str, ...] = (
    "VertexAI",
    "GoogleAPI",
    "ResourceExhausted",
    "ServiceUnavailable",
    "DeadlineExceeded",
    "InternalServerError",
    "google.api_core",
    "google.auth",
    "grpc",
    "Quota",
    "PermissionDenied",
    "InvalidArgument",
)

# Error types that are safe to retry
_RETRIABLE_ERROR_TYPES: tuple[str, ...] = (
    "ResourceExhausted",
    "ServiceUnavailable",
    "DeadlineExceeded",
    "InternalServerError",
    "Unavailable",
    "ConnectionError",
    "TimeoutError",
)


def _friendly_error_message(exc: Exception) -> str:
    """Convert a raw exception into a user-friendly message.

    Strips internal details (tracebacks, module paths) and returns
    a safe, descriptive message suitable for SSE clients.
    """
    exc_type = type(exc).__name__
    exc_str = str(exc)

    # Check for Vertex / Google API errors
    for marker in _VERTEX_ERROR_MARKERS:
        if marker.lower() in exc_type.lower() or marker.lower() in exc_str.lower():
            if "quota" in exc_str.lower() or "resourceexhausted" in exc_type.lower():
                return "API quota exceeded. Please wait a moment and try again."
            if "permission" in exc_str.lower():
                return "Permission denied. Check your API credentials and project configuration."
            if "deadline" in exc_str.lower() or "timeout" in exc_str.lower():
                return "Request timed out. The model took too long to respond. Please try again."
            return "Model returned an error. Please try again."

    # Connection errors
    if "connection" in exc_str.lower() or "connect" in exc_type.lower():
        return "Connection error. Please check your network and try again."

    # Timeout
    if "timeout" in exc_str.lower() or "timeout" in exc_type.lower():
        return "Request timed out. Please try again."

    # Generic fallback — do NOT expose raw exception messages
    return "An unexpected error occurred. Please try again."


def _is_retriable(exc: Exception) -> bool:
    """Determine whether an error is safe to retry."""
    exc_type = type(exc).__name__
    exc_str = str(exc)

    for marker in _RETRIABLE_ERROR_TYPES:
        if marker.lower() in exc_type.lower() or marker.lower() in exc_str.lower():
            return True

    return False


# ── Live pipeline SSE adapter ────────────────────────────────


async def live_pipeline_to_sse(
    pipeline_coro: Coroutine[Any, Any, Any],
    event_queue: asyncio.Queue[Event | None],
    *,
    keepalive_interval: float = 5.0,
) -> AsyncGenerator[ServerSentEvent, None]:
    """Stream EventBus events during a live pipeline execution, then the final result.

    Unlike :func:`stream_to_sse` which multiplexes text chunks + events,
    this adapter is event-only: the pipeline runs as an ``asyncio.Task``
    and does not produce a streaming text response.  Progress is communicated
    exclusively through EventBus events (tool calls, phase completions,
    agent progress).

    When the event queue is idle for ``keepalive_interval`` seconds, a
    keepalive SSE event is emitted to prevent proxy / load-balancer
    timeouts (e.g. Cloud Run default 60 s).

    After the pipeline task completes, remaining queue events are drained,
    and the final ``OrchestratorResult`` is serialised as ``result`` and
    ``report_html`` SSE events, followed by ``done``.

    Args:
        pipeline_coro: Coroutine that runs the full multi-agent pipeline
            and returns an ``OrchestratorResult``.
        event_queue: Per-request ``asyncio.Queue`` fed by the
            :class:`~vaig.web.events.EventQueueBridge`.
        keepalive_interval: Seconds between keepalive heartbeats when idle.
    """
    pipeline_task: asyncio.Task[Any] = asyncio.create_task(pipeline_coro)

    try:
        # Stream events until the pipeline task completes.
        # Keepalive heartbeats are emitted every ``keepalive_interval``
        # seconds when the queue is idle, preventing Cloud Run / reverse
        # proxy timeouts (typically 60 s default).
        while not pipeline_task.done():
            try:
                event = await asyncio.wait_for(
                    event_queue.get(),
                    timeout=keepalive_interval,
                )
            except TimeoutError:
                # No events for keepalive_interval — emit heartbeat
                yield ServerSentEvent(data=json.dumps({}), event="keepalive")
                continue

            if event is None:
                # Sentinel — event source exhausted
                break

            sse_type = _EVENT_TYPE_MAP.get(type(event), "status")
            yield ServerSentEvent(
                data=json.dumps(_event_to_sse_data(event)),
                event=sse_type,
            )

        # Drain any remaining queued events (non-blocking)
        async for sse_event in _drain_queue(event_queue):
            yield sse_event

        # Await the pipeline result — may raise if the pipeline failed
        result = await pipeline_task

        # Yield result + report events
        async for sse_event in _emit_pipeline_result(result):
            yield sse_event

    except asyncio.CancelledError:
        # Client disconnected — SSE generator was closed by sse-starlette.
        # Cancel the pipeline task to stop wasting API quota on abandoned
        # requests.  This is the primary disconnect handling mechanism.
        logger.info("Live SSE client disconnected — cancelling pipeline")
        raise  # Re-raise so sse-starlette completes cleanup

    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        logger.exception("Live pipeline SSE error")

        # Attempt to extract partial results if some agents completed
        # before the failure.  The pipeline task may have set partial
        # state even though it ultimately raised.
        partial_result = _extract_partial_result(pipeline_task)
        if partial_result is not None:
            logger.info("Partial pipeline result available — sending before error")
            async for sse_event in _emit_pipeline_result(
                partial_result, partial=True
            ):
                yield sse_event

        user_message = _friendly_error_message(exc)
        yield ServerSentEvent(
            data=json.dumps({
                "message": user_message,
                "error_type": type(exc).__name__,
                "retriable": _is_retriable(exc),
            }),
            event="error",
        )
        yield ServerSentEvent(data=json.dumps({}), event="done")
    finally:
        # Ensure the pipeline task is cancelled if still running
        if not pipeline_task.done():
            pipeline_task.cancel()
            try:
                await pipeline_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


async def _emit_pipeline_result(
    result: Any,
    *,
    partial: bool = False,
) -> AsyncGenerator[ServerSentEvent, None]:
    """Serialise an ``OrchestratorResult`` into SSE events.

    Yields:
        ``result`` event with structured report, usage, cost.
        ``report_html`` event if a structured report is available.
        ``done`` event (only for complete results; partial skips done).
    """
    result_data: dict[str, Any] = {}
    structured_report = getattr(result, "structured_report", None)
    if structured_report is not None:
        try:
            result_data["structured_report"] = structured_report.model_dump(
                mode="json"
            )
        except Exception:  # noqa: BLE001
            result_data["structured_report"] = str(structured_report)

    total_usage = getattr(result, "total_usage", {}) or {}
    result_data["usage"] = total_usage
    result_data["cost_usd"] = getattr(result, "run_cost_usd", 0.0)
    result_data["success"] = getattr(result, "success", True)
    result_data["partial"] = partial

    yield ServerSentEvent(
        data=json.dumps(result_data, default=str),
        event="result",
    )

    # Render HTML report if a structured report is available
    if structured_report is not None:
        try:
            from vaig.ui.html_report import render_health_report_html

            html = render_health_report_html(structured_report)
            yield ServerSentEvent(
                data=json.dumps({"html": html}),
                event="report_html",
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to render HTML report for live SSE", exc_info=True
            )

    # Terminal event — only for complete results.  Partial results are
    # followed by the error event and *its* done, so we skip here.
    if not partial:
        yield ServerSentEvent(data=json.dumps({}), event="done")


def _extract_partial_result(pipeline_task: asyncio.Task[Any]) -> Any | None:
    """Try to extract a partial result from a failed pipeline task.

    The orchestrator stores intermediate results on the task's coroutine
    frame.  If the task finished with an exception, the exception object
    may carry partial data (e.g. some agents completed their work).

    Returns the partial result, or ``None`` if nothing usable is available.
    """
    if not pipeline_task.done():
        return None

    try:
        exc = pipeline_task.exception()
    except asyncio.CancelledError:
        return None

    if exc is None:
        # Task succeeded — no partial result scenario
        return None

    # Check if the exception carries a partial result (common pattern:
    # the orchestrator attaches partial_result to the exception).
    partial = getattr(exc, "partial_result", None)
    if partial is not None:
        return partial

    # Check if the exception carries agent_results (fallback)
    agent_results = getattr(exc, "agent_results", None)
    if agent_results is not None:
        return agent_results

    return None

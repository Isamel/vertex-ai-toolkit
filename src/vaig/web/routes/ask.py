"""Ask route — single-shot question with SSE streaming response.

``GET /ask`` renders the ask form.
``POST /ask/stream`` streams the Orchestrator response via Server-Sent Events.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Request
from sse_starlette import EventSourceResponse, ServerSentEvent
from starlette.responses import Response

from vaig.core.events import (
    ErrorOccurred,
    Event,
    OrchestratorPhaseCompleted,
    ToolExecuted,
)
from vaig.web.deps import get_container, get_current_user, get_settings
from vaig.web.sse import stream_to_sse

__all__: list[str] = []

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])

# Event types to forward to the SSE stream
_SUBSCRIBED_EVENTS: tuple[type[Event], ...] = (
    ToolExecuted,
    OrchestratorPhaseCompleted,
    ErrorOccurred,
)


@router.get("/ask")
async def ask_form(request: Request) -> Response:
    """Render the ask form page."""
    templates = request.app.state.templates
    return templates.TemplateResponse(  # type: ignore[no-any-return]
        request=request,
        name="ask.html",
    )


@router.post("/ask/stream")
async def ask_stream(request: Request) -> EventSourceResponse:
    """Stream an Orchestrator response via SSE.

    Constructs per-request dependencies, subscribes to relevant EventBus
    events, runs the Orchestrator in streaming mode, and yields SSE events.
    """
    # Resolve dependencies manually (avoids ruff B008 with Depends defaults)
    user = get_current_user(request)
    settings = await get_settings(request)
    container = get_container(settings)

    # Extract question and service from form data
    form = await request.form()
    question = str(form.get("question", "")).strip()
    service = str(form.get("service", "")).strip()

    if not question:
        return EventSourceResponse(_error_generator("Question is required."))

    # Build the full prompt (service context + question)
    prompt = f"Service: {service}\n\n{question}" if service else question

    # Per-request asyncio.Queue for EventBus → SSE bridge
    event_queue: asyncio.Queue[Event | None] = asyncio.Queue()

    # Subscribe to relevant event types
    bus = container.event_bus
    unsub_fns: list[Any] = []
    for event_type in _SUBSCRIBED_EVENTS:
        unsub = bus.subscribe(event_type, event_queue.put_nowait)
        unsub_fns.append(unsub)

    async def _generate() -> AsyncGenerator[ServerSentEvent, None]:
        """Run orchestrator and stream SSE events."""
        from vaig.agents.orchestrator import Orchestrator

        try:
            orchestrator = Orchestrator(container.gemini_client, settings)
            stream_result = await orchestrator.async_execute_single(
                prompt,
                stream=True,
            )

            async for sse_event in stream_to_sse(stream_result, event_queue):
                yield sse_event
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ask stream error for user=%s", user)
            yield ServerSentEvent(
                data=json.dumps(
                    {"message": str(exc), "error_type": type(exc).__name__}
                ),
                event="error",
            )
            yield ServerSentEvent(
                data=json.dumps({"usage": {}, "full_text": ""}),
                event="done",
            )
        finally:
            # Always unsubscribe from EventBus
            for unsub in unsub_fns:
                unsub()

    return EventSourceResponse(_generate())


async def _error_generator(message: str) -> AsyncGenerator[ServerSentEvent, None]:
    """Yield a single error SSE event and done."""
    yield ServerSentEvent(
        data=json.dumps({"message": message, "error_type": "ValidationError"}),
        event="error",
    )
    yield ServerSentEvent(
        data=json.dumps({"usage": {}, "full_text": ""}),
        event="done",
    )

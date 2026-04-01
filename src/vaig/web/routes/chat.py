"""Chat routes — multi-turn conversation with SSE streaming + session persistence.

``GET /chat``          — start a new chat session
``GET /chat/{id}``     — load an existing session with message history
``POST /chat/{id}/message`` — send a message and stream the response via SSE
``GET /sessions``      — list all sessions for the current user
``DELETE /sessions/{id}`` — delete a session
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Request
from sse_starlette import EventSourceResponse, ServerSentEvent
from starlette.responses import JSONResponse, Response

from vaig.web.deps import get_container, get_current_user, get_settings
from vaig.web.events import EventQueueBridge
from vaig.web.sse import stream_to_sse

__all__: list[str] = []

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _get_session_store(request: Request) -> Any:
    """Retrieve the session store from app state.

    Returns ``None`` if no session store has been configured — routes
    should handle this gracefully.
    """
    return getattr(request.app.state, "session_store", None)


# ── Session List ─────────────────────────────────────────────


@router.get("/sessions")
async def sessions_list(request: Request) -> Response:
    """Render the sessions list page."""
    user = get_current_user(request)
    store = _get_session_store(request)

    sessions: list[dict[str, Any]] = []
    if store is not None:
        sessions = await store.async_list_sessions(user=user)

    templates = request.app.state.templates
    return templates.TemplateResponse(  # type: ignore[no-any-return]
        request=request,
        name="sessions.html",
        context={"sessions": sessions, "user": user},
    )


@router.delete("/sessions/{session_id}")
async def session_delete(request: Request, session_id: str) -> JSONResponse:
    """Delete a session by ID."""
    user = get_current_user(request)  # Auth check
    store = _get_session_store(request)

    if store is None:
        return JSONResponse(
            {"error": "Session store not configured"}, status_code=503
        )

    # Verify session ownership before deleting
    session_data = await store.async_get_session(session_id)
    if session_data is None or session_data.get("user") != user:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    deleted = await store.async_delete_session(session_id)
    if not deleted:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    return JSONResponse({"status": "deleted"})


# ── Chat ─────────────────────────────────────────────────────


@router.get("/chat")
async def chat_new(request: Request) -> Response:
    """Render a new chat session page."""
    settings = await get_settings(request)
    templates = request.app.state.templates
    return templates.TemplateResponse(  # type: ignore[no-any-return]
        request=request,
        name="chat.html",
        context={
            "session": None,
            "messages": [],
            "default_project": settings.gcp.project_id,
        },
    )


@router.get("/chat/{session_id}")
async def chat_resume(request: Request, session_id: str) -> Response:
    """Load an existing chat session with its message history."""
    user = get_current_user(request)  # Auth check
    store = _get_session_store(request)

    session_data: dict[str, Any] | None = None
    messages: list[dict[str, Any]] = []

    if store is not None:
        session_data = await store.async_get_session(session_id)
        # Verify session belongs to the current user
        if session_data is not None and session_data.get("user") != user:
            session_data = None  # Treat as not found
        if session_data is not None:
            messages = await store.async_get_messages(session_id)

    settings = await get_settings(request)
    if session_data is None:
        templates = request.app.state.templates
        return templates.TemplateResponse(  # type: ignore[no-any-return]
            request=request,
            name="chat.html",
            context={
                "session": None,
                "messages": [],
                "error": f"Session {session_id[:12]} not found.",
                "default_project": settings.gcp.project_id,
            },
            status_code=404,
        )

    templates = request.app.state.templates
    return templates.TemplateResponse(  # type: ignore[no-any-return]
        request=request,
        name="chat.html",
        context={
            "session": session_data,
            "messages": messages,
            "default_project": settings.gcp.project_id,
        },
    )


@router.post("/chat/{session_id}/message")
async def chat_message(request: Request, session_id: str) -> EventSourceResponse:
    """Send a message in a chat session and stream the response via SSE.

    Creates a new session if ``session_id`` is ``"new"``. Stores both the
    user message and assistant response in the session store.
    """
    user = get_current_user(request)
    settings = await get_settings(request)
    container = get_container(settings)
    store = _get_session_store(request)

    form = await request.form()
    message = str(form.get("message", "")).strip()

    if not message:
        return EventSourceResponse(_error_generator("Message is required."))

    # Create a new session if requested
    actual_session_id = session_id
    if session_id == "new":
        if store is not None:
            session_name = str(form.get("session_name", "Chat")).strip() or "Chat"
            actual_session_id = await store.async_create_session(
                name=session_name,
                model=settings.models.default,
                user=user,
            )
        else:
            # Generate an ephemeral UUID even without persistence
            actual_session_id = str(uuid4())

    # Validate existing session exists before proceeding
    if session_id != "new" and store is not None:
        existing = await store.async_get_session(actual_session_id)
        if existing is None:
            return EventSourceResponse(
                _error_generator(f"Session {actual_session_id[:12]} not found.")
            )

    # Store the user message
    if store is not None:
        await store.async_add_message(
            actual_session_id, "user", message, model=settings.models.default
        )

    # Build context from conversation history
    context = ""
    if store is not None:
        history = await store.async_get_messages(actual_session_id)
        # Exclude the message we just added (last one)
        prior = history[:-1] if history else []
        if prior:
            context = _format_history(prior)

    async def _generate() -> AsyncGenerator[ServerSentEvent, None]:
        """Run orchestrator and stream SSE events."""
        from vaig.agents.orchestrator import Orchestrator

        try:
            # Send session_id as the first event so the client can update the URL
            yield ServerSentEvent(
                data=json.dumps({"session_id": actual_session_id}),
                event="session",
            )

            async with EventQueueBridge(container.event_bus) as event_queue:
                orchestrator = Orchestrator(container.gemini_client, settings)
                stream_result = await orchestrator.async_execute_single(
                    message,
                    context=context,
                    stream=True,
                )

                full_text_parts: list[str] = []

                async for sse_event in stream_to_sse(stream_result, event_queue):
                    # Capture the text for storing the assistant message
                    if sse_event.event == "chunk":
                        try:
                            chunk_data = json.loads(str(sse_event.data))
                            if chunk_data.get("text"):
                                full_text_parts.append(chunk_data["text"])
                        except (json.JSONDecodeError, TypeError):
                            pass
                    yield sse_event

            # Store the assistant response
            full_text = "".join(full_text_parts)
            if store is not None and full_text:
                await store.async_add_message(
                    actual_session_id,
                    "assistant",
                    full_text,
                    model=settings.models.default,
                )
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            logger.exception("Chat stream error for user=%s session=%s", user, actual_session_id)
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

    return EventSourceResponse(_generate())


# ── Helpers ──────────────────────────────────────────────────


def _format_history(messages: list[dict[str, Any]]) -> str:
    """Format message history as context string for the orchestrator."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


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

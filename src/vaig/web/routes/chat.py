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
import os
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Request
from sse_starlette import EventSourceResponse, ServerSentEvent
from starlette.responses import JSONResponse, Response

from vaig.core.models import SessionRole
from vaig.web.deps import get_container, get_current_user, get_settings
from vaig.web.events import EventQueueBridge
from vaig.web.sse import stream_to_sse

__all__: list[str] = []

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

_FEATURE_FLAG = "VAIG_WEB_SHARED_SESSIONS"


def _sharing_enabled() -> bool:
    """Return ``True`` when the shared-sessions feature flag is active."""
    return os.environ.get(_FEATURE_FLAG, "").lower() in ("true", "1", "yes")


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
        owned = await store.async_list_sessions(user=user)
        # Mark owned sessions with role
        for s in owned:
            s.setdefault("role", "owner")

        # Merge shared sessions if the access service is available
        access_svc = getattr(request.app.state, "session_access", None)
        if access_svc is not None:
            try:
                shared = await access_svc.list_accessible_sessions(user)
            except Exception:  # noqa: BLE001
                logger.warning("Failed to fetch shared sessions for %s", user)
                shared = []

            # Deduplicate: owned sessions take precedence
            owned_ids = {s.get("id") for s in owned}
            for s in shared:
                if s.get("id") not in owned_ids:
                    sessions.append(s)

        sessions = owned + list(sessions)

        # Sort by updated_at descending (most recent first)
        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)

    templates = request.app.state.templates
    return templates.TemplateResponse(  # type: ignore[no-any-return]
        request=request,
        name="sessions.html",
        context={
            "sessions": sessions,
            "user": user,
            "sharing_enabled": _sharing_enabled(),
        },
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

    # Verify session ownership via ACL before deleting
    access_svc = getattr(request.app.state, "session_access", None)
    if access_svc is not None:
        result = await access_svc.check_access(session_id, user, required=SessionRole.OWNER)
        if not result.granted:
            return JSONResponse({"error": "Session not found"}, status_code=404)
    else:
        # Fallback: ownership check when ACL is not configured
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
            "user_role": "owner",
            "is_owner": True,
            "sharing_enabled": _sharing_enabled(),
            "annotations": [],
        },
    )


@router.get("/chat/{session_id}")
async def chat_resume(request: Request, session_id: str) -> Response:
    """Load an existing chat session with its message history."""
    user = get_current_user(request)  # Auth check
    store = _get_session_store(request)

    session_data: dict[str, Any] | None = None
    messages: list[dict[str, Any]] = []
    user_role = "owner"
    is_owner = True

    if store is not None:
        session_data = await store.async_get_session(session_id)
        # Verify user has at least viewer access via ACL
        if session_data is not None:
            access_svc = getattr(request.app.state, "session_access", None)
            if access_svc is not None:
                acl_result = await access_svc.check_access(
                    session_id, user, required=SessionRole.VIEWER
                )
                if not acl_result.granted:
                    session_data = None  # Treat as not found
                else:
                    user_role = acl_result.role.value if acl_result.role else "owner"
                    is_owner = acl_result.role == SessionRole.OWNER if acl_result.role else True
            else:
                # Fallback: ownership check when ACL is not configured
                if session_data.get("user") != user:
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
                "user_role": "owner",
                "is_owner": True,
                "sharing_enabled": _sharing_enabled(),
                "annotations": [],
            },
            status_code=404,
        )

    # Fetch annotations for the session if sharing is enabled
    annotations_list: list[dict[str, Any]] = []
    if _sharing_enabled():
        try:
            access_svc = getattr(request.app.state, "session_access", None)
            if access_svc is not None:
                raw_annotations = await access_svc.list_annotations(session_id, user)
            annotations_list = [
                {
                    "id": a.id,
                    "author": a.author,
                    "content": a.content,
                    "annotation_type": a.annotation_type,
                    "message_ref": a.message_ref,
                    "created_at": a.created_at,
                    "updated_at": a.updated_at,
                }
                for a in raw_annotations
            ]
        except Exception:  # noqa: BLE001
            logger.warning("Failed to fetch annotations for session %s", session_id[:8])

    templates = request.app.state.templates
    return templates.TemplateResponse(  # type: ignore[no-any-return]
        request=request,
        name="chat.html",
        context={
            "session": session_data,
            "messages": messages,
            "default_project": settings.gcp.project_id,
            "user_role": user_role,
            "is_owner": is_owner,
            "sharing_enabled": _sharing_enabled(),
            "annotations": annotations_list,
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

    # Validate existing session exists and user has edit access
    if session_id != "new" and store is not None:
        existing = await store.async_get_session(actual_session_id)
        if existing is None:
            return EventSourceResponse(
                _error_generator(f"Session {actual_session_id[:12]} not found.")
            )
        # Verify the user has editor-level access via ACL
        access_svc = getattr(request.app.state, "session_access", None)
        if access_svc is not None:
            acl_result = await access_svc.check_access(
                actual_session_id, user, required=SessionRole.EDITOR
            )
            if not acl_result.granted:
                return EventSourceResponse(
                    _error_generator("Insufficient permissions")
                )
        else:
            # Fallback: ownership check when ACL is not configured
            if existing.get("user") != user:
                return EventSourceResponse(
                    _error_generator("Insufficient permissions")
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

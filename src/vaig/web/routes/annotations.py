"""Annotation routes — CRUD for session annotations.

``GET    /sessions/{id}/annotations``              — list (viewer+)
``POST   /sessions/{id}/annotations``              — create (editor+)
``PUT    /sessions/{id}/annotations/{ann_id}``     — update (author only)
``DELETE /sessions/{id}/annotations/{ann_id}``     — delete (author only)
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from vaig.web.deps import get_current_user, get_session_access

__all__: list[str] = []

logger = logging.getLogger(__name__)

router = APIRouter(tags=["annotations"])

_FEATURE_FLAG = "VAIG_WEB_SHARED_SESSIONS"


def _sharing_enabled() -> bool:
    """Return ``True`` when the shared-sessions feature flag is active."""
    return os.environ.get(_FEATURE_FLAG, "").lower() in ("true", "1", "yes")


# -- Request Models --------------------------------------------------------


class CreateAnnotationRequest(BaseModel):
    """Body for POST /sessions/{id}/annotations."""

    annotation_type: Literal["observation", "action_item", "question", "root_cause", "resolution"]
    content: str = Field(..., min_length=1, max_length=2000)
    message_ref: str | None = None


class UpdateAnnotationRequest(BaseModel):
    """Body for PUT /sessions/{id}/annotations/{ann_id}."""

    content: str = Field(..., min_length=1, max_length=2000)


# -- Routes ----------------------------------------------------------------


@router.get("/sessions/{session_id}/annotations")
async def list_annotations(
    request: Request,
    session_id: str,
    limit: int = 50,
) -> JSONResponse:
    """List annotations for the session (viewer+ access)."""
    if not _sharing_enabled():
        return JSONResponse({"error": "not found"}, status_code=404)

    user = get_current_user(request)
    access = get_session_access(request)

    annotations = await access.list_annotations(session_id, user, limit=limit)
    return JSONResponse([
        {
            "id": a.id,
            "author": a.author,
            "content": a.content,
            "annotation_type": a.annotation_type,
            "message_ref": a.message_ref,
            "created_at": a.created_at,
            "updated_at": a.updated_at,
        }
        for a in annotations
    ])


@router.post("/sessions/{session_id}/annotations")
async def create_annotation(
    request: Request,
    session_id: str,
    payload: CreateAnnotationRequest,
) -> JSONResponse:
    """Create a new annotation on the session (editor+ access)."""
    if not _sharing_enabled():
        return JSONResponse({"error": "not found"}, status_code=404)

    user = get_current_user(request)
    access = get_session_access(request)

    try:
        annotation = await access.add_annotation(
            session_id,
            user,
            annotation_type=payload.annotation_type,
            content=payload.content,
            message_ref=payload.message_ref,
        )
    except PermissionError as exc:
        return JSONResponse({"error": str(exc)}, status_code=403)

    return JSONResponse(
        {
            "id": annotation.id,
            "author": annotation.author,
            "content": annotation.content,
            "annotation_type": annotation.annotation_type,
            "message_ref": annotation.message_ref,
            "created_at": annotation.created_at,
            "updated_at": annotation.updated_at,
        },
        status_code=200,
    )


@router.put("/sessions/{session_id}/annotations/{annotation_id}")
async def update_annotation(
    request: Request,
    session_id: str,
    annotation_id: str,
    payload: UpdateAnnotationRequest,
) -> JSONResponse:
    """Update an annotation (author only)."""
    if not _sharing_enabled():
        return JSONResponse({"error": "not found"}, status_code=404)

    user = get_current_user(request)
    access = get_session_access(request)

    try:
        annotation = await access.update_annotation(
            session_id,
            annotation_id,
            user,
            payload.content,
        )
    except PermissionError as exc:
        return JSONResponse({"error": str(exc)}, status_code=403)
    except LookupError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)

    return JSONResponse({
        "id": annotation.id,
        "author": annotation.author,
        "content": annotation.content,
        "annotation_type": annotation.annotation_type,
        "message_ref": annotation.message_ref,
        "created_at": annotation.created_at,
        "updated_at": annotation.updated_at,
    })


@router.delete("/sessions/{session_id}/annotations/{annotation_id}")
async def delete_annotation(
    request: Request,
    session_id: str,
    annotation_id: str,
) -> JSONResponse:
    """Delete an annotation (author only)."""
    if not _sharing_enabled():
        return JSONResponse({"error": "not found"}, status_code=404)

    user = get_current_user(request)
    access = get_session_access(request)

    try:
        await access.delete_annotation(session_id, annotation_id, user)
    except PermissionError as exc:
        return JSONResponse({"error": str(exc)}, status_code=403)
    except LookupError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)

    return JSONResponse({"status": "deleted", "id": annotation_id})

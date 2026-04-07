"""Sharing routes — manage session collaborators.

``POST /sessions/{id}/share``         — invite a collaborator (owner only)
``DELETE /sessions/{id}/share/{email}`` — revoke a collaborator (owner only)
``GET /sessions/{id}/collaborators``  — list collaborators (any collaborator)
"""

from __future__ import annotations

import logging
import os
import re

from fastapi import APIRouter, Request
from pydantic import BaseModel, field_validator
from starlette.responses import JSONResponse

from vaig.core.models import SessionRole
from vaig.web.deps import get_current_user, get_session_access

__all__: list[str] = []

logger = logging.getLogger(__name__)

router = APIRouter(tags=["sharing"])

_FEATURE_FLAG = "VAIG_WEB_SHARED_SESSIONS"
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _sharing_enabled() -> bool:
    """Return ``True`` when the shared-sessions feature flag is active."""
    return os.environ.get(_FEATURE_FLAG, "").lower() in ("true", "1", "yes")


# ── Request / Response Models ────────────────────────────────


class ShareRequest(BaseModel):
    """Body for POST /sessions/{id}/share."""

    email: str
    role: str  # "editor" or "viewer"

    @field_validator("email")
    @classmethod
    def _validate_email(cls, v: str) -> str:
        v = v.strip()
        if not _EMAIL_RE.match(v):
            msg = "Invalid email format"
            raise ValueError(msg)
        return v

    @field_validator("role")
    @classmethod
    def _validate_role(cls, v: str) -> str:
        if v not in ("editor", "viewer"):
            msg = "Role must be 'editor' or 'viewer'"
            raise ValueError(msg)
        return v


# ── Routes ───────────────────────────────────────────────────


@router.post("/sessions/{session_id}/share")
async def share_session(request: Request, session_id: str) -> JSONResponse:
    """Invite a collaborator to the session (owner only)."""
    if not _sharing_enabled():
        return JSONResponse({"error": "not found"}, status_code=404)

    user = get_current_user(request)
    access = get_session_access(request)

    try:
        body = await request.json()
        payload = ShareRequest(**body)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)

    try:
        role = SessionRole(payload.role)
    except ValueError:
        return JSONResponse(
            {"error": "Role must be 'editor' or 'viewer'"}, status_code=422
        )

    try:
        collab = await access.share(session_id, user, payload.email, role)
    except PermissionError as exc:
        return JSONResponse({"error": str(exc)}, status_code=403)
    except ValueError as exc:
        error_msg = str(exc)
        if "Cannot share with session owner" in error_msg:
            return JSONResponse({"error": error_msg}, status_code=400)
        return JSONResponse({"error": error_msg}, status_code=400)

    return JSONResponse({
        "status": "shared",
        "email": collab.email,
        "role": collab.role.value,
    })


@router.delete("/sessions/{session_id}/share/{email}")
async def revoke_share(
    request: Request, session_id: str, email: str
) -> JSONResponse:
    """Revoke a collaborator's access (owner only)."""
    if not _sharing_enabled():
        return JSONResponse({"error": "not found"}, status_code=404)

    user = get_current_user(request)
    access = get_session_access(request)

    try:
        revoked = await access.revoke(session_id, user, email)
    except PermissionError as exc:
        return JSONResponse({"error": str(exc)}, status_code=403)

    if not revoked:
        return JSONResponse({"error": "Collaborator not found"}, status_code=404)

    return JSONResponse({"status": "revoked", "email": email})


@router.get("/sessions/{session_id}/collaborators")
async def list_collaborators(
    request: Request, session_id: str
) -> JSONResponse:
    """List collaborators for the session (any collaborator can view)."""
    if not _sharing_enabled():
        return JSONResponse({"error": "not found"}, status_code=404)

    user = get_current_user(request)
    access = get_session_access(request)

    collaborators = await access.list_collaborators(session_id, user)
    return JSONResponse([
        {
            "email": c.email,
            "role": c.role.value,
            "added_at": c.added_at,
            "added_by": c.added_by,
        }
        for c in collaborators
    ])

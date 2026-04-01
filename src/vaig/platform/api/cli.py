"""CLI management API router — ``/api/v1/cli/*`` endpoints.

Handles listing, inspecting, and revoking CLI instances, plus
heartbeat reporting (REQ-API-003).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status

from vaig.platform.core.dependencies import (
    get_current_user,
    get_repository,
    require_admin,
)
from vaig.platform.core.firestore import AbstractRepository
from vaig.platform.models.auth import JWTClaims

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cli", tags=["cli"])


# ── GET /cli/list ─────────────────────────────────────────────


@router.get("/list")
async def list_cli_instances(
    claims: Annotated[JWTClaims, Depends(require_admin)],
    repo: Annotated[AbstractRepository, Depends(get_repository)],
) -> dict[str, Any]:
    """List all CLI instances in the organization (admin only)."""
    instances = await repo.list_cli_instances(claims.org_id)
    return {
        "items": [inst.model_dump() for inst in instances],
        "total": len(instances),
    }


# ── GET /cli/{cli_id} ────────────────────────────────────────


@router.get("/{cli_id}")
async def get_cli_instance(
    cli_id: str,
    claims: Annotated[JWTClaims, Depends(get_current_user)],
    repo: Annotated[AbstractRepository, Depends(get_repository)],
) -> dict[str, Any]:
    """Get details of a specific CLI instance.

    Non-admin users can only view their own CLI instances.
    """
    instance = await repo.get_cli_instance(claims.org_id, cli_id)
    if instance is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="CLI instance not found",
        )

    # Non-admin: can only see own instances
    if claims.role != "admin" and instance.user_email != claims.sub:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    return instance.model_dump()


# ── POST /cli/heartbeat ──────────────────────────────────────


@router.post("/heartbeat")
async def heartbeat(
    body: dict[str, Any],
    claims: Annotated[JWTClaims, Depends(get_current_user)],
    repo: Annotated[AbstractRepository, Depends(get_repository)],
) -> dict[str, Any]:
    """Report a heartbeat from a CLI instance.

    Updates ``last_heartbeat`` and returns current config version
    so the CLI can detect staleness.
    """
    cli_id = claims.cli_id
    if cli_id:
        instance = await repo.get_cli_instance(claims.org_id, cli_id)
        if instance is not None:
            instance.last_heartbeat = datetime.now(UTC)
            if "vaig_version" in body:
                instance.vaig_version = body["vaig_version"]
            if "cluster_name" in body:
                instance.cluster_name = body["cluster_name"]
            await repo.save_cli_instance(claims.org_id, instance)

    return {
        "config_version": "",
        "quota_remaining": -1,  # -1 = unlimited (MVP)
    }


# ── POST /cli/{cli_id}/revoke ────────────────────────────────


@router.post("/{cli_id}/revoke")
async def revoke_cli_instance(
    cli_id: str,
    claims: Annotated[JWTClaims, Depends(require_admin)],
    repo: Annotated[AbstractRepository, Depends(get_repository)],
) -> dict[str, bool]:
    """Revoke a CLI instance (admin only).

    Sets the CLI instance status to ``"revoked"`` so its next
    request will be rejected.
    """
    instance = await repo.get_cli_instance(claims.org_id, cli_id)
    if instance is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="CLI instance not found",
        )

    instance.status = "revoked"
    await repo.save_cli_instance(claims.org_id, instance)

    return {"revoked": True}

"""Config policy API router — ``/api/v1/config/*`` endpoints.

Handles reading, updating, and pushing configuration policies
for organizations (REQ-API-004).
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from vaig.platform.core.dependencies import (
    get_current_user,
    get_repository,
    require_admin,
)
from vaig.platform.core.firestore import AbstractRepository
from vaig.platform.models.auth import JWTClaims
from vaig.platform.models.organization import ConfigHistoryEntry, ConfigPolicy

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/config", tags=["config"])


# ── Request models ────────────────────────────────────────────


class ConfigPolicyUpdate(BaseModel):
    """Typed request body for the ``PUT /config/policy`` endpoint."""

    enforced_fields: dict[str, Any] = Field(default_factory=dict)
    user_configurable_fields: list[str] = Field(default_factory=list)
    blocked_fields: list[str] = Field(default_factory=list)
    changelog: str = ""


# ── GET /config/policy ────────────────────────────────────────


@router.get("/policy")
async def get_policy(
    claims: Annotated[JWTClaims, Depends(get_current_user)],
    repo: Annotated[AbstractRepository, Depends(get_repository)],
) -> dict[str, Any]:
    """Get the current config policy for the organization.

    Available to any authenticated user (admin and non-admin).
    """
    policy = await repo.get_config_policy(claims.org_id)
    if policy is None:
        return {
            "config_version": "",
            "enforced_fields": {},
            "user_configurable_fields": [],
            "blocked_fields": [],
        }
    return {
        "config_version": "",  # MVP: no versioning yet
        "enforced_fields": policy.enforced_fields,
        "user_configurable_fields": policy.user_configurable_fields,
        "blocked_fields": policy.blocked_fields,
    }


# ── PUT /config/policy ────────────────────────────────────────


@router.put("/policy")
async def update_policy(
    body: ConfigPolicyUpdate,
    claims: Annotated[JWTClaims, Depends(require_admin)],
    repo: Annotated[AbstractRepository, Depends(get_repository)],
) -> dict[str, Any]:
    """Update the config policy for the organization (admin only).

    Creates a new config version and records it in config_history.
    """
    policy = ConfigPolicy(
        enforced_fields=body.enforced_fields,
        user_configurable_fields=body.user_configurable_fields,
        blocked_fields=body.blocked_fields,
    )
    await repo.save_config_policy(claims.org_id, policy)

    # Record in history
    version_id = f"v-{uuid.uuid4().hex[:8]}"
    entry = ConfigHistoryEntry(
        version_id=version_id,
        pushed_by=claims.sub,
        pushed_at=datetime.now(UTC),
        config_policy=policy,
        changelog=body.changelog,
    )
    await repo.add_config_history(claims.org_id, entry)

    return {
        "config_version": version_id,
        "updated_at": entry.pushed_at.isoformat() if entry.pushed_at else "",
    }


# ── GET /config/history ───────────────────────────────────────


@router.get("/history")
async def get_history(
    claims: Annotated[JWTClaims, Depends(require_admin)],
    repo: Annotated[AbstractRepository, Depends(get_repository)],
) -> dict[str, Any]:
    """Get config policy change history (admin only)."""
    entries = await repo.list_config_history(claims.org_id)
    return {
        "items": [entry.model_dump() for entry in entries],
        "total": len(entries),
    }


# ── POST /config/push ────────────────────────────────────────


@router.post("/push")
async def push_config(
    claims: Annotated[JWTClaims, Depends(require_admin)],
    repo: Annotated[AbstractRepository, Depends(get_repository)],
) -> dict[str, Any]:
    """Push config to all CLIs (conceptual — sets version for refresh on next heartbeat).

    Admin only. In the MVP this is a no-op that returns the push count.
    In production, this would trigger an update notification to connected CLIs.
    """
    instances = await repo.list_cli_instances(claims.org_id)
    active_count = sum(1 for inst in instances if inst.status == "active")

    return {
        "pushed_to": active_count,
        "config_version": "",
    }

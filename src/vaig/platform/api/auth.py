"""Auth API router — ``/api/v1/auth/*`` endpoints.

Handles CLI registration, token exchange (PKCE), token refresh,
token revocation, and ``whoami`` (REQ-API-002).
"""

from __future__ import annotations

import logging
import secrets
import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from vaig.platform.core.dependencies import (
    get_current_user,
    get_jwt_service,
    get_repository,
)
from vaig.platform.core.firestore import AbstractRepository
from vaig.platform.core.jwt import ACCESS_TOKEN_LIFETIME, JWTService
from vaig.platform.models.auth import (
    JWTClaims,
    RefreshRequest,
    RegisterRequest,
    TokenRequest,
    TokenResponse,
    WhoamiResponse,
)
from vaig.platform.models.organization import CLIInstance

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


# ── POST /auth/register ──────────────────────────────────────


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    repo: Annotated[AbstractRepository, Depends(get_repository)],
    jwt_service: Annotated[JWTService, Depends(get_jwt_service)],
) -> dict[str, str]:
    """Register a new CLI instance.

    In the MVP this is a simplified registration — in production the
    registration would be tied to a completed OAuth flow.
    """
    cli_id = f"cli-{uuid.uuid4().hex[:12]}"
    # For MVP: use a default org_id. In production this comes from the OAuth flow.
    org_id = "default-org"

    instance = CLIInstance(
        cli_id=cli_id,
        user_email="",
        machine_id=body.machine_id,
        hostname=body.hostname,
        os_user=body.os_user,
        vaig_version=body.vaig_version,
        gcp_project=body.gcp_project or "",
        cluster_name=body.cluster_name or "",
        registered_at=datetime.now(UTC),
        status="active",
    )
    await repo.save_cli_instance(org_id, instance)

    return {"cli_id": cli_id, "org_id": org_id, "user_role": "operator"}


# ── POST /auth/token ──────────────────────────────────────────


@router.post("/token")
async def exchange_token(
    body: TokenRequest,
    jwt_service: Annotated[JWTService, Depends(get_jwt_service)],
) -> TokenResponse:
    """Exchange an authorization code for access + refresh tokens (PKCE).

    In the MVP, the code exchange is simplified — production would validate
    against the OAuth provider.
    """
    if not body.code or not body.code_verifier:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authorization code",
        )

    # MVP: issue token directly. Production would validate PKCE code_verifier.
    cli_id = f"cli-{uuid.uuid4().hex[:12]}"
    access_token = jwt_service.issue_token(
        sub="user@example.com",
        org_id="default-org",
        role="operator",
        cli_id=cli_id,
        lifetime=ACCESS_TOKEN_LIFETIME,
    )
    refresh_token = secrets.token_urlsafe(48)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_LIFETIME,
        cli_id=cli_id,
    )


# ── POST /auth/refresh ───────────────────────────────────────


@router.post("/refresh")
async def refresh_token(
    body: RefreshRequest,
    jwt_service: Annotated[JWTService, Depends(get_jwt_service)],
) -> TokenResponse:
    """Refresh an access token using a refresh token.

    In the MVP, any non-empty refresh token is accepted. Production would
    validate against stored refresh tokens in Firestore and enforce
    single-use rotation.
    """
    if not body.refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    access_token = jwt_service.issue_token(
        sub="user@example.com",
        org_id="default-org",
        role="operator",
        lifetime=ACCESS_TOKEN_LIFETIME,
    )
    new_refresh = secrets.token_urlsafe(48)

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_LIFETIME,
    )


# ── POST /auth/revoke ────────────────────────────────────────


@router.post("/revoke")
async def revoke_token(
    claims: Annotated[JWTClaims, Depends(get_current_user)],
) -> dict[str, bool]:
    """Revoke the current token (and optional refresh token).

    In the MVP this is a no-op acknowledgement. Production would
    add the token to a blocklist or delete the refresh token from
    Firestore.
    """
    return {"revoked": True}


# ── GET /auth/whoami ──────────────────────────────────────────


@router.get("/whoami")
async def whoami(
    claims: Annotated[JWTClaims, Depends(get_current_user)],
) -> WhoamiResponse:
    """Return the decoded JWT claims for the authenticated user."""
    return WhoamiResponse(
        email=claims.sub,
        org_id=claims.org_id,
        role=claims.role,
        cli_id=claims.cli_id,
        scopes=claims.scope,
    )

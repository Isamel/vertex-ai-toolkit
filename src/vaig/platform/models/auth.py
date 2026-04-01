"""Authentication and token models for the platform domain.

These Pydantic V2 models define the request/response contracts for
the ``/api/v1/auth/*`` endpoints (REQ-API-002) and the local
credential structures used by the CLI auth manager.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AuthResult(BaseModel):
    """Result of a CLI login attempt."""

    success: bool
    user_email: str = ""
    org_id: str = ""
    role: str = ""
    error: str | None = None


class TokenResponse(BaseModel):
    """Response from ``POST /api/v1/auth/token`` and ``POST /api/v1/auth/refresh``."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    cli_id: str = ""


class TokenRequest(BaseModel):
    """Request body for ``POST /api/v1/auth/token`` (code exchange)."""

    code: str
    code_verifier: str
    redirect_uri: str = ""


class RefreshRequest(BaseModel):
    """Request body for ``POST /api/v1/auth/refresh``."""

    refresh_token: str


class RegisterRequest(BaseModel):
    """Request body for ``POST /api/v1/auth/register``."""

    machine_id: str
    hostname: str = ""
    os_user: str = ""
    vaig_version: str = ""
    gcp_project: str | None = None
    cluster_name: str | None = None


class WhoamiResponse(BaseModel):
    """Response from ``GET /api/v1/auth/whoami``."""

    email: str
    org_id: str
    role: str
    cli_id: str = ""
    scopes: list[str] = Field(default_factory=list)


class JWTClaims(BaseModel):
    """Decoded JWT token claims.

    Maps to the claims included in access tokens issued by the
    platform backend (REQ-API-006).
    """

    sub: str  # email
    org_id: str = ""
    role: str = ""
    cli_id: str = ""
    machine_id: str = ""
    iat: int = 0
    exp: int = 0
    scope: list[str] = Field(default_factory=list)
    config_version: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)

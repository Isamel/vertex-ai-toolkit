"""FastAPI dependency injection for the platform API.

Provides ``Depends()`` callables for:
  - ``get_jwt_service()``: Singleton JWTService
  - ``get_repository()``: Repository instance (abstract — overridable for tests)
  - ``get_current_user()``: Extract and validate JWT from Authorization header
  - ``require_admin()``: Wraps ``get_current_user()`` and checks admin role
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from vaig.platform.core.firestore import AbstractRepository
from vaig.platform.core.jwt import JWTError, JWTService
from vaig.platform.models.auth import JWTClaims

logger = logging.getLogger(__name__)

# ── Security scheme ───────────────────────────────────────────
_bearer_scheme = HTTPBearer(auto_error=False)


# ── Singletons (overridable via app.dependency_overrides) ─────

def get_jwt_service(request: Request) -> JWTService:
    """Return the JWTService stored on ``app.state``."""
    svc: JWTService | None = getattr(request.app.state, "jwt_service", None)
    if svc is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT service not configured",
        )
    return svc


def get_repository(request: Request) -> AbstractRepository:
    """Return the repository stored on ``app.state``."""
    repo: AbstractRepository | None = getattr(request.app.state, "repository", None)
    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Repository not configured",
        )
    return repo


# ── Auth dependencies ─────────────────────────────────────────

async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
    jwt_service: Annotated[JWTService, Depends(get_jwt_service)],
) -> JWTClaims:
    """Extract and validate the JWT from the ``Authorization: Bearer`` header.

    Returns the decoded ``JWTClaims`` on success.

    Raises:
        HTTPException 401: If the token is missing, expired, or invalid.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        claims = jwt_service.validate_token(credentials.credentials)
    except JWTError as exc:
        detail = str(exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    return claims


async def require_admin(
    claims: Annotated[JWTClaims, Depends(get_current_user)],
) -> JWTClaims:
    """Require the authenticated user to have the ``admin`` role.

    Wraps ``get_current_user()`` — passes through if admin, raises 403
    otherwise.
    """
    if claims.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )
    return claims

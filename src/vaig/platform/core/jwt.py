"""JWT service — issue and validate tokens using PyJWT with RS256.

Signing keys are configurable via environment variables for the MVP:
  - ``PLATFORM_JWT_PRIVATE_KEY``: PEM-encoded RSA private key (for signing)
  - ``PLATFORM_JWT_PUBLIC_KEY``: PEM-encoded RSA public key (for validation)

Production deployments should load keys from Secret Manager (future enhancement).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import jwt

from vaig.platform.models.auth import JWTClaims

logger = logging.getLogger(__name__)

# Token lifetimes (seconds)
ACCESS_TOKEN_LIFETIME = 3600  # 1 hour
REFRESH_TOKEN_LIFETIME = 30 * 24 * 3600  # 30 days

_ALGORITHM = "RS256"


class JWTError(Exception):
    """Raised when JWT operations fail."""


class JWTService:
    """Issue and validate JWT tokens using RS256.

    Args:
        private_key: PEM-encoded RSA private key (for signing).
            Falls back to ``PLATFORM_JWT_PRIVATE_KEY`` env var.
        public_key: PEM-encoded RSA public key (for validation).
            Falls back to ``PLATFORM_JWT_PUBLIC_KEY`` env var.
    """

    def __init__(
        self,
        private_key: str | None = None,
        public_key: str | None = None,
    ) -> None:
        self._private_key = private_key or os.environ.get("PLATFORM_JWT_PRIVATE_KEY", "")
        self._public_key = public_key or os.environ.get("PLATFORM_JWT_PUBLIC_KEY", "")

    # ── Issue ─────────────────────────────────────────────────

    def issue_token(
        self,
        sub: str,
        *,
        org_id: str = "",
        role: str = "",
        cli_id: str = "",
        machine_id: str = "",
        scope: list[str] | None = None,
        config_version: str = "",
        extra: dict[str, Any] | None = None,
        lifetime: int = ACCESS_TOKEN_LIFETIME,
    ) -> str:
        """Issue a signed JWT access token.

        Args:
            sub: Subject (user email).
            org_id: Organization ID.
            role: User role (admin, operator, viewer).
            cli_id: CLI instance identifier.
            machine_id: Machine identifier.
            scope: List of allowed command scopes.
            config_version: Current config policy version.
            extra: Additional claims to include.
            lifetime: Token lifetime in seconds (default: 1 hour).

        Returns:
            Encoded JWT string.

        Raises:
            JWTError: If the private key is missing or encoding fails.
        """
        if not self._private_key:
            raise JWTError("Private key not configured — cannot issue tokens")

        now = int(time.time())
        payload: dict[str, Any] = {
            "sub": sub,
            "org_id": org_id,
            "role": role,
            "cli_id": cli_id,
            "machine_id": machine_id,
            "scope": scope or [],
            "config_version": config_version,
            "iat": now,
            "exp": now + lifetime,
        }
        if extra:
            payload.update(extra)

        try:
            token: str = jwt.encode(payload, self._private_key, algorithm=_ALGORITHM)
            return token
        except Exception as exc:
            raise JWTError(f"Failed to encode JWT: {exc}") from exc

    # ── Validate ──────────────────────────────────────────────

    def validate_token(self, token: str) -> JWTClaims:
        """Validate a JWT and return decoded claims.

        Args:
            token: Encoded JWT string.

        Returns:
            Decoded ``JWTClaims`` model.

        Raises:
            JWTError: If validation fails (expired, bad signature, etc.).
        """
        if not self._public_key:
            raise JWTError("Public key not configured — cannot validate tokens")

        try:
            payload = jwt.decode(token, self._public_key, algorithms=[_ALGORITHM])
        except jwt.ExpiredSignatureError as exc:
            raise JWTError("Token expired") from exc
        except jwt.InvalidSignatureError as exc:
            raise JWTError("Invalid token signature") from exc
        except jwt.DecodeError as exc:
            raise JWTError(f"Invalid token: {exc}") from exc
        except jwt.PyJWTError as exc:
            raise JWTError(f"Token validation failed: {exc}") from exc

        return JWTClaims(**payload)

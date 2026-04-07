"""FastAPI dependencies for the VAIG web interface.

Provides ``Depends()``-compatible callables for:

- **Authentication**: ``get_current_user`` — reads IAP header or dev fallback.
- **Settings factory**: ``get_settings`` — per-request ``Settings`` from form overrides.
- **Container factory**: ``get_container`` — per-request ``ServiceContainer``.
"""

from __future__ import annotations

import os
from typing import Any, cast

from fastapi import HTTPException, Request

from vaig.core.config import Settings
from vaig.core.container import ServiceContainer, build_container
from vaig.core.protocols import SessionAccessProtocol

__all__ = [
    "get_container",
    "get_current_user",
    "get_session_access",
    "get_settings",
]

# IAP header set by Cloud Identity-Aware Proxy
_IAP_USER_HEADER = "X-Goog-Authenticated-User-Email"
# Prefix added by IAP (e.g. "accounts.google.com:user@example.com")
_IAP_PREFIX = "accounts.google.com:"
# Fallback env var for local development
_DEV_USER_ENV = "VAIG_WEB_DEV_USER"
_DEV_USER_DEFAULT = "dev@localhost"
# Explicit dev-mode flag — must be set for dev fallback to activate
_DEV_MODE_ENV = "VAIG_WEB_DEV_MODE"


def get_current_user(request: Request) -> str:
    """Extract the authenticated user email from the request.

    In production, Cloud IAP sets ``X-Goog-Authenticated-User-Email``
    with the format ``accounts.google.com:user@example.com``.

    In local development, falls back to the ``VAIG_WEB_DEV_USER``
    environment variable (default: ``dev@localhost``) **only** when
    ``VAIG_WEB_DEV_MODE=true`` is set.  Otherwise raises HTTP 401 to
    prevent silent bypass when deployed without IAP.
    """
    header_value = request.headers.get(_IAP_USER_HEADER, "")
    if header_value:
        # Strip the IAP prefix if present
        if header_value.startswith(_IAP_PREFIX):
            return header_value[len(_IAP_PREFIX):]
        return header_value

    # Only fall back to dev identity when dev mode is explicitly enabled
    if os.environ.get(_DEV_MODE_ENV, "").lower() in ("true", "1", "yes"):
        return os.environ.get(_DEV_USER_ENV, _DEV_USER_DEFAULT)

    raise HTTPException(
        status_code=401,
        detail="Missing IAP authentication header. Set VAIG_WEB_DEV_MODE=true for local development.",
    )


async def get_settings(request: Request) -> Settings:
    """Construct a per-request Settings from form/query parameters.

    Reads well-known override keys from form data or query parameters
    and passes them to ``Settings.from_overrides()``.

    When a ``session_id`` query parameter is present and the app has a
    session store with config support, the stored per-session config is
    loaded first.  Explicit form/query overrides take precedence over
    stored values.
    """
    overrides: dict[str, Any] = {}

    # ── 1. Load session config (lowest priority) ─────────────
    session_id = (
        request.query_params.get("session_id")
        or request.query_params.get("session")
        or request.path_params.get("session_id")
        or request.path_params.get("session")
    )
    if session_id:
        store = getattr(request.app.state, "session_store", None)
        if store is not None and hasattr(store, "async_get_config"):
            user = get_current_user(request)
            session_config = await store.async_get_config(session_id, user)
            if session_config:
                overrides.update(session_config)

    # ── 2. Explicit form/query overrides (highest priority) ──
    # Try form data first (POST), fall back to query params (GET)
    content_type = request.headers.get("content-type", "")
    if "form" in content_type or "multipart" in content_type:
        form = await request.form()
        form_dict: dict[str, Any] = dict(form)
    else:
        form_dict = dict(request.query_params)

    # Map form fields to override keys
    # Note: GKE-specific keys (cluster, namespace, gke_project, gke_location)
    # are intentionally excluded — they are handled by get_gke_config_from_form()
    # and are not mapped by Settings.from_overrides().
    _FORM_KEYS = ("project", "model", "temperature", "max_tokens", "region")
    for key in _FORM_KEYS:
        value = form_dict.get(key)
        if value is not None and str(value).strip():
            if key == "temperature":
                try:
                    overrides[key] = float(value)
                except (ValueError, TypeError):
                    continue
            elif key == "max_tokens":
                try:
                    max_tokens = int(value)
                except (ValueError, TypeError):
                    continue
                if max_tokens > 0:
                    overrides[key] = max_tokens
            else:
                overrides[key] = str(value).strip()

    return Settings.from_overrides(**overrides)


def get_container(settings: Settings) -> ServiceContainer:
    """Build a per-request ServiceContainer from the given settings."""
    return build_container(settings)


def get_session_access(request: Request) -> SessionAccessProtocol:
    """Retrieve the session access control service from app state.

    Raises HTTP 503 when no access control backend has been configured
    (e.g. during startup or when the feature is not wired).
    """
    access = getattr(request.app.state, "session_access", None)
    if access is None:
        raise HTTPException(
            status_code=503,
            detail="Session access control not configured",
        )
    return cast(SessionAccessProtocol, access)

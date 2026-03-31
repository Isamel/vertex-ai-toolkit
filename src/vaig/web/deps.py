"""FastAPI dependencies for the VAIG web interface.

Provides ``Depends()``-compatible callables for:

- **Authentication**: ``get_current_user`` — reads IAP header or dev fallback.
- **Settings factory**: ``get_settings`` — per-request ``Settings`` from form overrides.
- **Container factory**: ``get_container`` — per-request ``ServiceContainer``.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import Request

from vaig.core.config import Settings
from vaig.core.container import ServiceContainer, build_container

__all__ = [
    "get_container",
    "get_current_user",
    "get_settings",
]

# IAP header set by Cloud Identity-Aware Proxy
_IAP_USER_HEADER = "X-Goog-Authenticated-User-Email"
# Prefix added by IAP (e.g. "accounts.google.com:user@example.com")
_IAP_PREFIX = "accounts.google.com:"
# Fallback env var for local development
_DEV_USER_ENV = "VAIG_WEB_DEV_USER"
_DEV_USER_DEFAULT = "dev@localhost"


def get_current_user(request: Request) -> str:
    """Extract the authenticated user email from the request.

    In production, Cloud IAP sets ``X-Goog-Authenticated-User-Email``
    with the format ``accounts.google.com:user@example.com``.

    In local development, falls back to the ``VAIG_WEB_DEV_USER``
    environment variable (default: ``dev@localhost``).
    """
    header_value = request.headers.get(_IAP_USER_HEADER, "")
    if header_value:
        # Strip the IAP prefix if present
        if header_value.startswith(_IAP_PREFIX):
            return header_value[len(_IAP_PREFIX):]
        return header_value

    return os.environ.get(_DEV_USER_ENV, _DEV_USER_DEFAULT)


async def get_settings(request: Request) -> Settings:
    """Construct a per-request Settings from form/query parameters.

    Reads well-known override keys from form data or query parameters
    and passes them to ``Settings.from_overrides()``.
    """
    overrides: dict[str, Any] = {}

    # Try form data first (POST), fall back to query params (GET)
    content_type = request.headers.get("content-type", "")
    if "form" in content_type or "multipart" in content_type:
        form = await request.form()
        form_dict: dict[str, Any] = dict(form)
    else:
        form_dict = dict(request.query_params)

    # Map form fields to override keys
    _FORM_KEYS = ("project", "model", "temperature", "region")
    for key in _FORM_KEYS:
        value = form_dict.get(key)
        if value is not None and str(value).strip():
            # Convert temperature to float
            if key == "temperature":
                try:
                    overrides[key] = float(value)
                except (ValueError, TypeError):
                    continue
            else:
                overrides[key] = str(value).strip()

    return Settings.from_overrides(**overrides)


def get_container(settings: Settings) -> ServiceContainer:
    """Build a per-request ServiceContainer from the given settings."""
    return build_container(settings)

"""Settings routes — per-session configuration editor.

``GET /settings``   — render the settings form (pre-filled from session config)
``POST /settings``  — validate and save config to the session's Firestore doc
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from starlette.responses import RedirectResponse, Response

from vaig.web.deps import get_current_user

__all__: list[str] = []

logger = logging.getLogger(__name__)

router = APIRouter(tags=["settings"])

# ── Validation ───────────────────────────────────────────────

_VALID_CONFIG_KEYS = frozenset(
    {"project", "region", "model", "temperature", "max_tokens", "system_instructions"}
)


def _validate_config(form_data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Validate and normalise form data into a config dict.

    Only keys present in ``_VALID_CONFIG_KEYS`` are considered; unknown
    keys are silently dropped.

    Returns:
        A tuple of ``(config, errors)``.  *config* contains only valid,
        type-coerced values.  *errors* is a list of human-readable
        validation messages (empty on success).
    """
    # Filter to known keys only
    filtered = {k: v for k, v in form_data.items() if k in _VALID_CONFIG_KEYS}
    config: dict[str, Any] = {}
    errors: list[str] = []

    # project — non-empty string
    project = str(filtered.get("project", "")).strip()
    if project:
        config["project"] = project

    # region — non-empty string
    region = str(filtered.get("region", "")).strip()
    if region:
        config["region"] = region

    # model — non-empty string
    model = str(filtered.get("model", "")).strip()
    if model:
        config["model"] = model

    # temperature — float in [0.0, 2.0]
    temp_raw = filtered.get("temperature")
    if temp_raw is not None and str(temp_raw).strip():
        try:
            temp = float(temp_raw)
            if temp < 0.0 or temp > 2.0:
                errors.append("Temperature must be between 0.0 and 2.0.")
            else:
                config["temperature"] = temp
        except (ValueError, TypeError):
            errors.append("Temperature must be a number.")

    # max_tokens — positive integer
    mt_raw = filtered.get("max_tokens")
    if mt_raw is not None and str(mt_raw).strip():
        try:
            mt = int(mt_raw)
            if mt <= 0:
                errors.append("Max tokens must be a positive integer.")
            else:
                config["max_tokens"] = mt
        except (ValueError, TypeError):
            errors.append("Max tokens must be an integer.")

    # system_instructions — free-form text
    si = str(filtered.get("system_instructions", "")).strip()
    if si:
        config["system_instructions"] = si

    return config, errors


# ── Helpers ──────────────────────────────────────────────────


def _get_session_store(request: Request) -> Any:
    """Retrieve the session store from app state."""
    return getattr(request.app.state, "session_store", None)


def _defaults_from_settings() -> dict[str, Any]:
    """Load default values from the base Settings for pre-filling the form."""
    try:
        from vaig.core.config import Settings

        base = Settings.from_overrides()
        return {
            "project": base.gcp.project_id,
            "region": base.gcp.location,
            "model": base.models.default,
            "temperature": base.generation.temperature,
            "max_tokens": base.generation.max_output_tokens,
            "system_instructions": "",
        }
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:  # noqa: BLE001
        logger.warning("Failed to load base settings, using hard-coded defaults", exc_info=True)
        return {
            "project": "",
            "region": "us-central1",
            "model": "gemini-2.5-pro",
            "temperature": 0.7,
            "max_tokens": 16384,
            "system_instructions": "",
        }


# ── Routes ───────────────────────────────────────────────────


@router.get("/settings")
async def settings_form(request: Request) -> Response:
    """Render the settings editor form.

    If a ``session`` query parameter is provided, loads the session's
    stored config to pre-fill the form.  Otherwise falls back to the
    application defaults.
    """
    user = get_current_user(request)
    session_id = request.query_params.get("session", "")
    store = _get_session_store(request)

    # Start with application defaults
    form_values = _defaults_from_settings()
    error: str | None = None

    # Overlay session config if available
    if (
        session_id
        and store is not None
        and hasattr(store, "async_get_session")
        and hasattr(store, "async_get_config")
    ):
        session_data = await store.async_get_session(session_id)
        if session_data is None or session_data.get("user") != user:
            error = f"Session {session_id[:12]} not found."
        else:
            # Extract config directly from session doc (avoids redundant read)
            session_config = session_data.get("config")
            if isinstance(session_config, dict):
                form_values.update(session_config)

    templates = request.app.state.templates
    return templates.TemplateResponse(  # type: ignore[no-any-return]
        request=request,
        name="settings.html",
        context={
            "form_values": form_values,
            "session_id": session_id,
            "user": user,
            "error": error,
            "errors": [],
        },
    )


@router.post("/settings")
async def settings_save(request: Request) -> Response:
    """Validate and persist per-session configuration.

    On success, redirects back to the chat session.
    On validation error, re-renders the form with error messages.
    """
    user = get_current_user(request)
    store = _get_session_store(request)

    form = await request.form()
    form_dict: dict[str, Any] = dict(form)
    session_id = str(form_dict.get("session_id", "")).strip()

    # Validate
    config, errors = _validate_config(form_dict)

    if not errors:
        # Persist
        if not session_id:
            errors.append("No session selected.")
        elif store is None or not hasattr(store, "async_save_config"):
            errors.append("Session store not configured.")
        else:
            saved = await store.async_save_config(session_id, config, user)
            if not saved:
                errors.append("Session not found or access denied.")

    if errors:
        form_values = _defaults_from_settings()
        form_values.update(form_dict)
        templates = request.app.state.templates
        return templates.TemplateResponse(  # type: ignore[no-any-return]
            request=request,
            name="settings.html",
            context={
                "form_values": form_values,
                "session_id": session_id,
                "user": user,
                "error": None,
                "errors": errors,
            },
        )

    # Success — redirect back to the chat session
    redirect_url = f"/chat/{session_id}" if session_id else "/chat"
    return RedirectResponse(url=redirect_url, status_code=303)

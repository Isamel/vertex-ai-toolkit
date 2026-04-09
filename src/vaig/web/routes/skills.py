"""Skills marketplace route — admin-only dashboard for installed skills.

``GET /portal/skills``  — render the skills dashboard (admin-gated)
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import Response

from vaig.core.config import get_settings
from vaig.core.telemetry import get_telemetry_collector
from vaig.skills.registry import SkillRegistry
from vaig.web.deps import get_current_user, is_admin

__all__: list[str] = []

logger = logging.getLogger(__name__)

router = APIRouter(tags=["skills"])


@router.get("/portal/skills")
async def skills_dashboard(request: Request) -> Response:
    """Render the skills marketplace dashboard.

    Admin-only: returns 403 for non-admin users.  Shows installed skills
    with source indicators and aggregated usage stats from telemetry.
    """
    if not is_admin(request):
        raise HTTPException(status_code=403, detail="Admin access required.")

    user = get_current_user(request)

    # ── Fetch skill metadata ────────────────────────────────────
    settings = get_settings()
    registry = SkillRegistry(settings)

    all_skills = registry.list_skills()

    # Build display items with source info
    skills_display: list[dict[str, Any]] = []
    builtin_count = 0
    external_count = 0

    for meta in all_skills:
        source = registry.get_source(meta.name)
        if source == "builtin":
            builtin_count += 1
        else:
            external_count += 1

        skills_display.append({
            "name": meta.name,
            "display_name": meta.display_name,
            "description": meta.description,
            "version": meta.version,
            "author": meta.author,
            "tags": meta.tags,
            "supported_phases": meta.supported_phases,
            "source": source,
            "usage_count": 0,  # filled below if telemetry available
        })

    # ── Fetch usage stats from telemetry ────────────────────────
    telemetry_available = True
    most_used: str | None = None

    try:
        collector = get_telemetry_collector()
        # TODO: Move aggregation to SQL query in TelemetryCollector
        # (e.g. SELECT event_name, COUNT(*) … GROUP BY event_name)
        events = await collector.async_query_events(
            event_type="skill_use", limit=1000,
        )
        usage_counts: Counter[str] = Counter(
            e["event_name"] for e in events if e.get("event_name")
        )

        # Merge usage into display items
        for item in skills_display:
            item["usage_count"] = usage_counts.get(item["name"], 0)

        if usage_counts:
            most_used = usage_counts.most_common(1)[0][0]

    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:  # noqa: BLE001
        logger.warning("Telemetry unavailable for skills dashboard", exc_info=True)
        telemetry_available = False

    # ── Build template context ──────────────────────────────────
    total_count = len(all_skills)

    templates = request.app.state.templates
    return templates.TemplateResponse(  # type: ignore[no-any-return]
        request=request,
        name="skills.html",
        context={
            "user": user,
            "is_admin": True,
            "skills": skills_display,
            "total_count": total_count,
            "builtin_count": builtin_count,
            "external_count": external_count,
            "most_used": most_used,
            "telemetry_available": telemetry_available,
        },
    )

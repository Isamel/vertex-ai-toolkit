"""Schedules routes — CRUD for scheduled health scans + scan history.

``GET  /portal/schedules``                — list all schedules
``POST /portal/schedules``                — create a new schedule
``DELETE /portal/schedules/{schedule_id}`` — remove a schedule
``POST /portal/schedules/{schedule_id}/trigger`` — trigger run-now
``GET  /portal/schedules/{schedule_id}/history`` — scan history
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse, Response

from vaig.web.deps import get_current_user

__all__: list[str] = []

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portal", tags=["schedules"])


# ── Request models ───────────────────────────────────────────


class CreateScheduleRequest(BaseModel):
    """Pydantic model for ``POST /portal/schedules`` body validation."""

    cluster_name: str = Field(..., min_length=1, description="GKE cluster name")
    namespace: str = Field(default="", description="Kubernetes namespace (empty = default)")
    interval_minutes: int | None = Field(
        default=None, ge=1, le=1440,
        description="Interval in minutes between scans (1–1440)",
    )
    cron: str | None = Field(
        default=None,
        description="Cron expression (overrides interval_minutes when set)",
    )
    all_namespaces: bool = Field(default=False, description="Scan all namespaces")
    skip_healthy: bool = Field(default=True, description="Skip healthy services in report")


# ── Engine helpers ───────────────────────────────────────────


def _get_engine(request: Request) -> Any:
    """Retrieve the SchedulerEngine from app state.

    The engine MUST be attached to ``app.state.scheduler_engine``
    by the application factory or startup event.
    """
    engine = getattr(request.app.state, "scheduler_engine", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Scheduler engine not available. Start with 'vaig schedule start'.",
        )
    return engine


# ── Routes ───────────────────────────────────────────────────


@router.get("/schedules")
async def list_schedules(request: Request) -> Response:
    """List all registered schedules.

    Returns HTML (Jinja2 template) if the client accepts HTML,
    otherwise returns JSON.
    """
    _user = get_current_user(request)
    engine = _get_engine(request)
    schedules = await engine.list_schedules()

    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        templates = request.app.state.templates
        resp: Response = templates.TemplateResponse(
            request=request,
            name="schedules.html",
            context={
                "user": _user,
                "schedules": schedules,
            },
        )
        return resp

    return JSONResponse(
        content=[
            {
                "schedule_id": s.schedule_id,
                "cluster_name": s.cluster_name,
                "namespace": s.namespace,
                "interval_minutes": s.interval_minutes,
                "cron_expression": s.cron_expression,
                "paused": s.paused,
                "next_fire_time": (
                    s.next_fire_time.isoformat() if s.next_fire_time else None
                ),
            }
            for s in schedules
        ],
    )


@router.post("/schedules")
async def create_schedule(request: Request, body: CreateScheduleRequest) -> JSONResponse:
    """Create a new scheduled health scan.

    Accepts JSON body::

        {
            "cluster_name": "prod-us",
            "namespace": "",
            "interval_minutes": 30,
            "cron": null,
            "all_namespaces": false,
            "skip_healthy": true
        }
    """
    _user = get_current_user(request)
    engine = _get_engine(request)

    from vaig.core.config import ScheduleTarget

    target = ScheduleTarget(
        cluster_name=body.cluster_name,
        namespace=body.namespace,
        all_namespaces=body.all_namespaces,
        skip_healthy=body.skip_healthy,
    )

    schedule_id = await engine.add_schedule(
        target,
        interval_minutes=body.interval_minutes,
        cron=body.cron,
    )

    logger.info("Schedule created via web: %s by %s", schedule_id, _user)
    return JSONResponse(
        content={"schedule_id": schedule_id, "status": "created"},
        status_code=201,
    )


@router.delete("/schedules/{schedule_id}")
async def delete_schedule(
    request: Request,
    schedule_id: str,
) -> JSONResponse:
    """Remove a schedule by ID."""
    _user = get_current_user(request)
    engine = _get_engine(request)

    removed = await engine.remove_schedule(schedule_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Schedule {schedule_id} not found")

    logger.info("Schedule removed via web: %s by %s", schedule_id, _user)
    return JSONResponse(content={"schedule_id": schedule_id, "status": "removed"})


@router.post("/schedules/{schedule_id}/trigger")
async def trigger_schedule(
    request: Request,
    schedule_id: str,
) -> JSONResponse:
    """Trigger an immediate scan for a schedule."""
    _user = get_current_user(request)
    engine = _get_engine(request)

    try:
        result = await engine.run_now(schedule_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    logger.info("Schedule triggered via web: %s by %s", schedule_id, _user)
    return JSONResponse(
        content={
            "schedule_id": schedule_id,
            "run_id": result.id,
            "status": result.status,
            "alerts_sent": result.alerts_sent,
        },
    )


@router.get("/schedules/{schedule_id}/history")
async def schedule_history(
    request: Request,
    schedule_id: str,
    limit: int = 20,
) -> JSONResponse:
    """Return scan history for a schedule."""
    _user = get_current_user(request)
    engine = _get_engine(request)

    history = await engine.get_history(schedule_id, limit=limit)
    return JSONResponse(
        content=[
            {
                "id": run.id,
                "schedule_id": run.schedule_id,
                "started_at": run.started_at,
                "completed_at": run.completed_at,
                "status": run.status,
                "alerts_sent": run.alerts_sent,
            }
            for run in history
        ],
    )

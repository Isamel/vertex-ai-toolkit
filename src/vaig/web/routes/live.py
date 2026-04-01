"""Live route — multi-agent infrastructure pipeline with SSE streaming.

``GET /live`` renders the live mode form.
``POST /live/stream`` runs the full multi-agent pipeline and streams
progress events via Server-Sent Events.

Production hardening:
- **Concurrency limit**: a global ``asyncio.Semaphore`` restricts the
  number of pipelines running at once.  Configurable via the
  ``VAIG_LIVE_MAX_CONCURRENT`` environment variable (default 5).
- **Client disconnect**: when the SSE client disconnects, the pipeline
  ``asyncio.Task`` is cancelled immediately (handled inside
  ``live_pipeline_to_sse()``), saving API quota.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any, Literal

from fastapi import APIRouter, Request
from sse_starlette import EventSourceResponse, ServerSentEvent
from starlette.responses import Response

from vaig.core.event_bus import EventBus
from vaig.core.events import (
    AgentProgressCompleted,
    AgentProgressStarted,
)
from vaig.core.gke import build_gke_config, register_live_tools
from vaig.web.deps import get_container, get_current_user, get_settings
from vaig.web.events import _LIVE_SUBSCRIBED_EVENTS, EventQueueBridge
from vaig.web.sse import live_pipeline_to_sse

__all__: list[str] = []

logger = logging.getLogger(__name__)

router = APIRouter(tags=["live"])

# Default skill for live mode — only orchestrated skill today
_DEFAULT_SKILL = "service-health"

# ── Concurrent execution limit ───────────────────────────────
# Prevents resource exhaustion from too many parallel multi-agent
# pipelines.  Each pipeline makes dozens of Vertex AI + GKE API calls.
_DEFAULT_MAX_CONCURRENT = 5
try:
    _MAX_CONCURRENT = int(os.environ.get("VAIG_LIVE_MAX_CONCURRENT", str(_DEFAULT_MAX_CONCURRENT)))
except (ValueError, TypeError):
    _MAX_CONCURRENT = _DEFAULT_MAX_CONCURRENT
_pipeline_semaphore = asyncio.Semaphore(_MAX_CONCURRENT)


@router.get("/live")
async def live_form(request: Request) -> Response:
    """Render the live mode page."""
    settings = await get_settings(request)
    templates = request.app.state.templates
    return templates.TemplateResponse(  # type: ignore[no-any-return]
        request=request,
        name="live.html",
        context={"default_project": settings.gcp.project_id},
    )


@router.post("/live/stream")
async def live_stream(request: Request) -> EventSourceResponse:
    """Run the full multi-agent pipeline and stream progress via SSE.

    Extracts GKE config from form fields, resolves the service-health
    skill, subscribes to EventBus events (including agent progress),
    and streams SSE events until the pipeline completes.

    Production guards:
    - Rejects requests when the concurrent pipeline limit is reached
      (``VAIG_LIVE_MAX_CONCURRENT``, default 5).
    - Cancels the pipeline task when the SSE client disconnects,
      preventing wasted API quota on abandoned requests.
    """
    # Resolve dependencies manually (consistent with ask.py pattern)
    user = get_current_user(request)
    settings = await get_settings(request)
    container = get_container(settings)

    # ── Concurrency guard ────────────────────────────────────
    # Fail-fast when all pipeline slots are occupied.  Try to acquire
    # the semaphore immediately; if it's fully held, return 429.
    acquired = False
    try:
        await asyncio.wait_for(_pipeline_semaphore.acquire(), timeout=0)
        acquired = True
    except TimeoutError:
        logger.warning(
            "Live pipeline rejected — concurrency limit reached "
            "(max=%d, user=%s)",
            _MAX_CONCURRENT,
            user,
        )
        return EventSourceResponse(
            _error_generator(
                f"Server busy — {_MAX_CONCURRENT} pipelines are already "
                f"running. Please try again in a few minutes.",
                error_type="TooManyRequests",
            ),
            status_code=429,
        )
    finally:
        # Release immediately — the generator will re-acquire via
        # ``async with _pipeline_semaphore`` for the actual run.
        if acquired:
            _pipeline_semaphore.release()

    # Extract form fields
    form = await request.form()
    question = str(form.get("question", "")).strip()
    service_name = str(form.get("service_name", "")).strip()
    cluster = str(form.get("cluster", "")).strip() or None
    namespace = str(form.get("namespace", "")).strip() or None
    gke_project = str(form.get("gke_project", "")).strip() or None
    gke_location = str(form.get("gke_location", "")).strip() or None

    if not service_name:
        return EventSourceResponse(_error_generator("Service name is required."))

    # Build the prompt
    prompt = f"Service: {service_name}"
    if question:
        prompt = f"{prompt}\n\n{question}"

    # Build GKE config and register tools
    gke_config = build_gke_config(
        settings,
        cluster=cluster,
        namespace=namespace,
        project_id=gke_project,
        location=gke_location,
    )
    tool_registry = await asyncio.to_thread(
        register_live_tools, gke_config, settings=settings
    )

    # Resolve skill
    from vaig.skills.registry import SkillRegistry

    skill_registry = SkillRegistry(settings)
    skill = skill_registry.get(_DEFAULT_SKILL)

    if skill is None:
        return EventSourceResponse(
            _error_generator(f"Skill '{_DEFAULT_SKILL}' not found or not enabled.")
        )

    # Subscribe to EventBus with live event types (includes agent progress)
    bus = container.event_bus
    bridge = EventQueueBridge(bus, event_types=_LIVE_SUBSCRIBED_EVENTS)

    async def _generate() -> AsyncGenerator[ServerSentEvent, None]:
        """Run pipeline and stream SSE events.

        Acquires the concurrency semaphore for the duration of the
        pipeline execution.  On client disconnect (``CancelledError``),
        the semaphore is released and the pipeline task is cancelled
        inside ``live_pipeline_to_sse()``.
        """
        from vaig.agents.orchestrator import Orchestrator

        async with _pipeline_semaphore:
            async with bridge as event_queue:
                orchestrator = Orchestrator(container.gemini_client, settings)

                # Create the agent progress callback that bridges to EventBus
                progress_callback = _progress_to_bus(bus)

                # Build the pipeline coroutine
                pipeline_coro = orchestrator.async_execute_with_tools(
                    prompt,
                    skill,
                    tool_registry,
                    on_agent_progress=progress_callback,
                    gke_namespace=namespace or gke_config.default_namespace,
                    gke_location=gke_location or gke_config.location,
                    gke_cluster_name=cluster or gke_config.cluster_name,
                )

                try:
                    async for sse_event in live_pipeline_to_sse(
                        pipeline_coro, event_queue
                    ):
                        yield sse_event
                except asyncio.CancelledError:
                    # Client disconnected — pipeline task cancellation is
                    # handled by live_pipeline_to_sse's finally block.
                    # Log and re-raise so sse-starlette completes cleanup.
                    logger.info(
                        "Live SSE client disconnected (user=%s) — "
                        "pipeline cancelled",
                        user,
                    )
                    raise

    return EventSourceResponse(_generate())


def _progress_to_bus(
    bus: EventBus,
) -> Any:
    """Create an ``on_agent_progress`` callback that emits EventBus events.

    The callback matches the :class:`~vaig.agents.orchestrator.OnAgentProgress`
    protocol, converting ``"start"`` / ``"end"`` calls into
    :class:`AgentProgressStarted` / :class:`AgentProgressCompleted` events.
    """

    def _callback(
        agent_name: str,
        agent_index: int,
        total_agents: int,
        event: Literal["start", "end"],
        end_agent_index: int | None = None,
    ) -> None:
        if event == "start":
            bus.emit(
                AgentProgressStarted(
                    agent_name=agent_name,
                    agent_index=agent_index,
                    total_agents=total_agents,
                    end_agent_index=end_agent_index,
                )
            )
        elif event == "end":
            bus.emit(
                AgentProgressCompleted(
                    agent_name=agent_name,
                    agent_index=agent_index,
                    total_agents=total_agents,
                    end_agent_index=end_agent_index,
                )
            )

    return _callback


async def _error_generator(
    message: str,
    *,
    error_type: str = "ValidationError",
) -> AsyncGenerator[ServerSentEvent, None]:
    """Yield a single error SSE event and done."""
    yield ServerSentEvent(
        data=json.dumps({"message": message, "error_type": error_type}),
        event="error",
    )
    yield ServerSentEvent(
        data=json.dumps({}),
        event="done",
    )

"""Headless skill runner — programmatic execution without CLI/Rich UI.

Consolidates the orchestration logic duplicated between ``live.py`` and
``webhook_server.py`` into a single reusable function.  The scheduler,
webhook server, and CLI all call this to execute a skill pipeline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vaig.agents.mixins import OnToolCall
    from vaig.agents.orchestrator import OnAgentProgress, OrchestratorResult
    from vaig.core.config import GKEConfig, Settings
    from vaig.core.tool_call_store import ToolCallStore
    from vaig.skills.base import BaseSkill

logger = logging.getLogger(__name__)


def execute_skill_headless(
    settings: Settings,
    skill: BaseSkill,
    query: str,
    gke_config: GKEConfig,
    *,
    timeout: float = 600.0,
    tool_call_store: ToolCallStore | None = None,
    on_tool_call: OnToolCall | None = None,
    on_agent_progress: OnAgentProgress | None = None,
) -> OrchestratorResult:
    """Run a skill pipeline programmatically.  No Rich console, no prompts.

    This is the shared entry point used by:
    - ``live.py`` (CLI — wraps result with Rich UI, passes callbacks)
    - ``webhook_server.py`` (FastAPI — wraps result as JSON response)
    - Scheduler engine (Phase 3 — wraps result with diff + alerting)

    Args:
        settings: Application settings.
        skill: Resolved skill instance (e.g. ``DiscoverySkill``).
        query: Investigation query for the orchestrator.
        gke_config: GKE cluster configuration for tool registration.
        timeout: Maximum wall-clock seconds for the pipeline (unused here,
            the caller is responsible for enforcing this via
            ``asyncio.wait_for`` or similar).
        tool_call_store: Optional store for persisting tool-call metadata.
        on_tool_call: Optional callback for each tool invocation (CLI progress).
        on_agent_progress: Optional callback for agent lifecycle events
            (CLI progress display).

    Returns:
        ``OrchestratorResult`` with ``structured_report``, ``synthesized_output``,
        and cost/usage metadata.

    Raises:
        RuntimeError: If no live tools could be registered (empty registry).
        vaig.core.exceptions.MaxIterationsError: If the pipeline exhausts its
            iteration budget.
    """
    from vaig.agents.orchestrator import Orchestrator
    from vaig.core.gke import register_live_tools

    # Build tool registry
    tool_registry = register_live_tools(gke_config, settings=settings)
    tool_count = len(tool_registry.list_tools())

    if tool_count == 0:
        msg = (
            "No live tools registered — the 'kubernetes' optional dependency "
            "may not be installed.  Install with: pip install 'vertex-ai-toolkit[live]'"
        )
        raise RuntimeError(msg)

    logger.info(
        "Headless execution: skill=%s, tools=%d, cluster=%s",
        skill.get_metadata().name,
        tool_count,
        gke_config.cluster_name,
    )

    # Detect Autopilot mode
    from vaig.core.auth import get_gke_credentials

    gke_credentials = get_gke_credentials(settings)

    is_autopilot: bool | None = None
    try:
        from vaig.tools.gke_tools import detect_autopilot

        is_autopilot = detect_autopilot(gke_config, credentials=gke_credentials)
    except ImportError:
        pass

    # Create client + orchestrator
    from vaig.core.client import GeminiClient

    client = GeminiClient(settings)
    orchestrator = Orchestrator(client, settings)

    result = orchestrator.execute_with_tools(
        query=query,
        skill=skill,
        tool_registry=tool_registry,
        strategy="sequential",
        is_autopilot=is_autopilot,
        on_tool_call=on_tool_call,
        tool_call_store=tool_call_store,
        on_agent_progress=on_agent_progress,
        gke_namespace=gke_config.default_namespace,
        gke_location=gke_config.location,
        gke_cluster_name=gke_config.cluster_name,
    )

    logger.info(
        "Headless execution complete: skill=%s, success=%s, cost=$%.4f",
        skill.get_metadata().name,
        result.success,
        result.run_cost_usd,
    )

    return result

"""Headless skill runner — programmatic execution without CLI/Rich UI.

Consolidates the orchestration logic duplicated between ``live.py`` and
``webhook_server.py`` into a single reusable function.  The scheduler,
webhook server, and CLI all call this to execute a skill pipeline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from vaig.agents.mixins import OnToolCall
    from vaig.agents.orchestrator import OnAgentProgress, OrchestratorResult
    from vaig.core.attachment_adapter import AttachmentAdapter
    from vaig.core.config import GKEConfig, RepoInvestigationConfig, Settings
    from vaig.core.repo_index import RepoIndex
    from vaig.core.tool_call_store import ToolCallStore
    from vaig.core.tool_registry import ToolRegistry
    from vaig.skills.base import BaseSkill

logger = logging.getLogger(__name__)

MAX_ATTACHMENT_CONTEXT_BYTES: Final[int] = 131_072  # 128 KB (was 32 KB)
_TRUNCATION_MARKER = (
    "\n\n[!!! ATTACHMENT CONTEXT TRUNCATED !!!]\n"
    "The attachment context above was cut off because it exceeded the "
    "per-run byte budget. Additional file chunks exist in the source "
    "attachments but were NOT included in this prompt.\n"
    "INSTRUCTION: When emitting your structured HealthReport, you MUST "
    "include an EvidenceGap entry with:\n"
    '  source="attachment_context"\n'
    '  reason="attachment context truncated at run-level byte budget; '
    "investigation ran on a subset of available evidence\"\n"
    "  details=\"see logs for byte counts and chunk counts\"\n"
    "Do not silently omit this gap — it must appear in evidence_gaps so "
    "operators know the analysis was based on partial input.\n"
)


def _stringify_gap(gap: object) -> str:
    """Stringify a ``repo_pipeline.EvidenceGap`` into a single human-readable line.

    Avoids importing ``repo_pipeline`` at module level to prevent circular imports.
    ``gap`` is typed as ``object`` (not ``Any``) since we only read attributes.
    """
    return (
        f"[{gap.level}] {gap.kind}: {gap.path or '<no-path>'} — {gap.details}"  # type: ignore[attr-defined]
    )


def _render_attachment_context(
    index: RepoIndex,
    budget_bytes: int | None = None,
) -> tuple[str, bool]:
    """Render all chunks in *index* as a markdown string capped at *budget_bytes* (UTF-8).

    Each chunk gets a ``### <file_path>`` header followed by a fenced block.
    Triple-backticks within chunk content are escaped by picking a fence
    longer than any run of backticks that appears inside the chunk, so the
    markdown remains well-formed regardless of the content.

    When the accumulated output would exceed the budget,
    rendering stops and :data:`_TRUNCATION_MARKER` is appended.

    Args:
        index: The ``RepoIndex`` whose chunks to render.
        budget_bytes: Maximum byte budget for the rendered output.  When
            ``None``, defaults to :data:`MAX_ATTACHMENT_CONTEXT_BYTES`.

    Returns:
        A ``(rendered_text, truncated)`` tuple.  ``truncated`` is ``True`` if
        and only if at least one chunk was dropped because the budget was hit.
    """
    import re

    budget = budget_bytes if budget_bytes is not None else MAX_ATTACHMENT_CONTEXT_BYTES
    parts: list[str] = []
    cap = budget - len(_TRUNCATION_MARKER.encode("utf-8"))
    accumulated = 0
    truncated = False

    for chunk in index.chunks:
        # Choose a fence longer than any backtick run already in the content
        longest_run = max(
            (len(m.group(0)) for m in re.finditer(r"`+", chunk.content)),
            default=0,
        )
        fence = "`" * max(3, longest_run + 1)
        block = f"### {chunk.file_path}\n{fence}\n{chunk.content}\n{fence}\n"
        block_bytes = len(block.encode("utf-8"))
        if accumulated + block_bytes > cap:
            parts.append(_TRUNCATION_MARKER)
            truncated = True
            break
        parts.append(block)
        accumulated += block_bytes

    return "".join(parts), truncated


def execute_skill_headless(
    settings: Settings,
    skill: BaseSkill,
    query: str,
    gke_config: GKEConfig,
    *,
    tool_registry: ToolRegistry | None = None,
    tool_call_store: ToolCallStore | None = None,
    on_tool_call: OnToolCall | None = None,
    on_agent_progress: OnAgentProgress | None = None,
    attachment_adapters: list[AttachmentAdapter] | None = None,
    repo_config: RepoInvestigationConfig | None = None,
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
        tool_registry: Optional pre-built registry.  When provided the
            function skips its own ``register_live_tools()`` call, avoiding
            duplicate work when the caller already built the registry
            (e.g. to display a tool count header in the CLI).
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

    # Build tool registry (or reuse caller-provided one)
    if tool_registry is None:
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

    # Build attachment context from adapters (if any)
    attachment_context: str | None = None
    attachment_truncated = False
    attachment_gap_strings: list[str] = []

    if attachment_adapters:
        from vaig.core.config import RepoInvestigationConfig
        from vaig.core.repo_index import RepoIndex

        cfg = repo_config if repo_config is not None else RepoInvestigationConfig()
        budget = cfg.attachment_context_budget_bytes

        try:
            index, gaps = RepoIndex.build_from_attachments(
                attachment_adapters,
                settings.attachments,
                cfg,
            )
            attachment_gap_strings = [_stringify_gap(g) for g in gaps]
            attachment_context, attachment_truncated = _render_attachment_context(index, budget)

            if attachment_truncated:
                effective_budget = budget if budget is not None else MAX_ATTACHMENT_CONTEXT_BYTES
                bytes_rendered = len(attachment_context.encode("utf-8"))
                chunks_rendered = attachment_context.count("\n### ") + (
                    1 if attachment_context.startswith("### ") else 0
                )
                total_chunks = len(index.chunks)
                logger.warning(
                    "attachment context truncated: %d/%d bytes used, %d/%d chunks rendered",
                    bytes_rendered,
                    effective_budget,
                    chunks_rendered,
                    total_chunks,
                )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            logger.warning("Failed to build RepoIndex from attachments: %s", exc)
            attachment_context = None

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
        attachment_context=attachment_context,
    )

    result.attachment_truncated = attachment_truncated
    result.attachment_gaps = attachment_gap_strings

    logger.info(
        "Headless execution complete: skill=%s, success=%s, cost=$%.4f",
        skill.get_metadata().name,
        result.success,
        result.run_cost_usd,
    )

    return result

"""Headless skill runner — programmatic execution without CLI/Rich UI.

Consolidates the orchestration logic duplicated between ``live.py`` and
``webhook_server.py`` into a single reusable function.  The scheduler,
webhook server, and CLI all call this to execute a skill pipeline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Final

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
_RENDER_OVERHEAD_PER_CHUNK: Final[int] = 64
"""Per-chunk overhead estimate (bytes) for ``_slice_attachment_windows``.

Accounts for the ``### <file_path>\\n`` header, opening/closing fence lines,
and blank lines added by ``_render_attachment_context`` for each chunk.
"""
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


def _slice_attachment_windows(
    index: RepoIndex,
    budget_bytes: int,
) -> list[RepoIndex]:
    """Partition *index.chunks* into sub-indices each rendering ≤ *budget_bytes*.

    Uses a greedy first-fit algorithm that walks chunks in their existing order.
    Each chunk's estimated rendered size is:

    .. code-block:: text

        len(outline.encode('utf-8'))
        + len(content.encode('utf-8'))
        + _RENDER_OVERHEAD_PER_CHUNK

    When adding the next chunk would push the running total past *budget_bytes*,
    the current window is emitted and a new one is started with that chunk.
    A single chunk that on its own exceeds the budget gets its own window —
    ``_render_attachment_context`` will then truncate it via the existing cap path.

    Args:
        index: Source ``RepoIndex`` to partition.
        budget_bytes: Byte budget per window (matches the per-run rendering budget).

    Returns:
        List of ``RepoIndex`` objects (one per window).  Returns an empty list
        when *index.chunks* is empty.  Preserves the source chunk order across
        and within windows.
    """
    from vaig.core.repo_index import RepoIndex as _RepoIndex

    chunks = index.chunks  # snapshot — property returns a copy
    if not chunks:
        return []

    windows: list[list[Any]] = []
    current: list[Any] = []
    current_bytes = 0

    for chunk in chunks:
        chunk_bytes = (
            len(chunk.outline.encode("utf-8"))
            + len(chunk.content.encode("utf-8"))
            + _RENDER_OVERHEAD_PER_CHUNK
        )
        if current and current_bytes + chunk_bytes > budget_bytes:
            windows.append(current)
            current = [chunk]
            current_bytes = chunk_bytes
        else:
            current.append(chunk)
            current_bytes += chunk_bytes

    if current:
        windows.append(current)

    return [_RepoIndex(w) for w in windows]


def _reduce_window_results(
    window_results: list[OrchestratorResult],
    extra_gaps: list[Any],
    windows_attempted: int,
    skill: BaseSkill,
) -> OrchestratorResult:
    """Merge per-window ``OrchestratorResult`` objects into one consolidated result.

    Strategy:

    1. Pick the first non-empty result as the *carrier* (preserves ``skill_name``,
       ``phase``, ``models_used``, ``final_state``, ``agent_results``).
    2. Call ``merge_health_reports`` on all windows that produced a structured report.
    3. Sum ``run_cost_usd`` and ``total_usage`` across all window results.
    4. Concatenate ``synthesized_output`` with ``\\n\\n---\\n\\n``.
    5. ``success = True`` if at least one window succeeded; ``False`` when all failed.
    6. Append *extra_gaps* to the merged HealthReport's ``evidence_gaps`` AND
       stringify them into ``attachment_gaps``.

    When *window_results* is empty, returns a minimal failure ``OrchestratorResult``
    with ``success=False`` and ``structured_report=None``.

    Args:
        window_results: Successful per-window results (may be empty on full failure).
        extra_gaps: ``EvidenceGap`` objects collected during the MAP loop.
        windows_attempted: Total windows attempted (including failed ones).
        skill: The executing skill (used for ``skill_name``/``phase`` on empty path).
    """
    from vaig.core.report_merge import merge_health_reports
    from vaig.skills.service_health.schema import HealthReport

    # All-windows-fail path
    if not window_results:
        meta = skill.get_metadata()
        from vaig.agents.orchestrator import OrchestratorResult as OrchestratorResultCls
        from vaig.skills.base import SkillPhase

        carrier = OrchestratorResultCls(
            skill_name=meta.name,
            phase=SkillPhase.EXECUTE,
            success=False,
        )
        carrier.structured_report = None
        carrier.attachment_gaps = [
            f"[{g.source}] {g.reason}: {g.details or ''}" for g in extra_gaps
        ]
        return carrier

    carrier = window_results[0]

    # Merge structured reports
    raw_reports = [wr.structured_report for wr in window_results if wr.structured_report is not None]
    valid_reports: list[HealthReport] = [r for r in raw_reports if isinstance(r, HealthReport)]
    merged_report = merge_health_reports(valid_reports)

    # Append extra_gaps into the merged report's evidence_gaps
    if merged_report is not None and extra_gaps:
        merged_report = merged_report.model_copy(update={
            "evidence_gaps": list(merged_report.evidence_gaps) + list(extra_gaps),
        })

    # Sum costs and usage
    total_cost = sum(wr.run_cost_usd for wr in window_results)
    combined_usage: dict[str, int] = {}
    for wr in window_results:
        for k, v in wr.total_usage.items():
            combined_usage[k] = combined_usage.get(k, 0) + v

    # Concatenate synthesized output
    synthesized = "\n\n---\n\n".join(
        wr.synthesized_output for wr in window_results if wr.synthesized_output
    )

    # Stringify extra_gaps for OrchestratorResult.attachment_gaps
    gap_strings = [
        f"[{g.source}] {g.reason}: {g.details or ''}" for g in extra_gaps
    ]

    carrier.structured_report = merged_report
    carrier.run_cost_usd = total_cost
    carrier.total_usage = combined_usage
    carrier.synthesized_output = synthesized
    carrier.success = any(wr.success for wr in window_results)
    carrier.attachment_gaps = gap_strings

    return carrier


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
    index = None
    budget: int | None = None
    cfg = None

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
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            logger.warning("Failed to build RepoIndex from attachments: %s", exc)
            index = None

    # ── Slice into windows (A2 / A3) ─────────────────────────────────────────
    windows: list[Any] = []
    window_cap_hit = False

    if index is not None and cfg is not None:
        effective_budget = budget if budget is not None else MAX_ATTACHMENT_CONTEXT_BYTES
        all_windows = _slice_attachment_windows(index, effective_budget)
        max_windows = cfg.map_reduce_max_windows
        if len(all_windows) > max_windows:
            window_cap_hit = True
            windows = all_windows[:max_windows]
            attachment_gap_strings.append(
                f"map_reduce: window cap hit — processed {max_windows}/"
                f"{len(all_windows)} windows; remaining content not analyzed"
            )
        else:
            windows = all_windows

    # ── Fast path: 0 or 1 window (preserves B1 byte-identical behavior) ──────
    if len(windows) <= 1:
        if windows:
            attachment_context, attachment_truncated = _render_attachment_context(
                windows[0], budget
            )
            if attachment_truncated:
                effective_budget = budget if budget is not None else MAX_ATTACHMENT_CONTEXT_BYTES
                bytes_rendered = len(attachment_context.encode("utf-8"))
                chunks_rendered = attachment_context.count("\n### ") + (
                    1 if attachment_context.startswith("### ") else 0
                )
                total_chunks = len(windows[0].chunks)
                logger.warning(
                    "attachment context truncated: %d/%d bytes used, %d/%d chunks rendered",
                    bytes_rendered,
                    effective_budget,
                    chunks_rendered,
                    total_chunks,
                )

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
        result.map_reduce_windows_used = 1 if windows else 0

    else:
        # ── MAP phase — sequential, one window at a time ──────────────────────
        window_results: list[Any] = []
        map_gaps: list[Any] = []

        for idx, window in enumerate(windows, start=1):
            # A9: skip empty windows defensively
            if not window.chunks:
                logger.debug("map_reduce: skipping empty window %d/%d", idx, len(windows))
                continue

            window_ctx, window_truncated = _render_attachment_context(window, budget)
            try:
                wr = orchestrator.execute_with_tools(
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
                    attachment_context=window_ctx,
                )
                window_results.append(wr)
                if window_truncated:
                    from vaig.skills.service_health.schema import EvidenceGap
                    map_gaps.append(EvidenceGap(
                        source="map_reduce_window",
                        reason="window_internally_truncated",
                        details=f"window {idx}/{len(windows)} exceeded single-window budget",
                    ))
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "map_reduce window %d/%d failed: %s",
                    idx,
                    len(windows),
                    exc,
                )
                from vaig.skills.service_health.schema import EvidenceGap
                map_gaps.append(EvidenceGap(
                    source="map_reduce_window",
                    reason="error",
                    details=f"window {idx}/{len(windows)} failed: {type(exc).__name__}: {exc}",
                ))

        # ── REDUCE phase ──────────────────────────────────────────────────────
        result = _reduce_window_results(
            window_results=window_results,
            extra_gaps=map_gaps,
            windows_attempted=len(windows),
            skill=skill,
        )
        result.attachment_truncated = window_cap_hit
        result.map_reduce_windows_used = len(windows)

    logger.info(
        "Headless execution complete: skill=%s, success=%s, cost=$%.4f",
        skill.get_metadata().name,
        result.success,
        result.run_cost_usd,
    )

    return result

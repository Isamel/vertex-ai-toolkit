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
    'investigation ran on a subset of available evidence"\n'
    '  details="see logs for byte counts and chunk counts"\n'
    "Do not silently omit this gap — it must appear in evidence_gaps so "
    "operators know the analysis was based on partial input.\n"
)


def _stringify_gap(gap: object) -> str:
    """Stringify a ``repo_pipeline.EvidenceGap`` into a single human-readable line.

    Avoids importing ``repo_pipeline`` at module level to prevent circular imports.
    ``gap`` is typed as ``object`` (not ``Any``) since we only read attributes.
    """
    return f"[{gap.level}] {gap.kind}: {gap.path or '<no-path>'} — {gap.details}"  # type: ignore[attr-defined]


def _detect_chunk_references(attachment_context: str, agent_output: str) -> tuple[list[str], list[str]]:
    """Heuristically find which attachment chunks an agent referenced.

    Scans the ``### <file_path>`` headings in *attachment_context* and checks
    whether a meaningful portion of each chunk's content appears in *agent_output*
    using a sliding window of sentence-like fragments.

    Returns:
        (chunk_ids, free_text_quotes) — list of heading labels found and list
        of verbatim excerpt matches (≤ 120 chars each, de-duplicated).
    """
    import re

    chunk_ids: list[str] = []
    free_text_quotes: list[str] = []

    if not attachment_context or not agent_output:
        return chunk_ids, free_text_quotes

    # Split on "### " headings to get (heading, body) pairs
    segments = re.split(r"(### .+)", attachment_context)
    current_heading = ""
    for segment in segments:
        if segment.startswith("### "):
            raw_heading = segment.lstrip("# ").strip()
            # Only treat as a chunk header if it looks like a file path:
            # non-empty, under 200 chars, contains a dot or slash
            if raw_heading and len(raw_heading) < 200 and ("." in raw_heading or "/" in raw_heading):
                current_heading = raw_heading
            else:
                current_heading = ""
            continue
        if not current_heading or len(segment.strip()) < 40:
            continue
        body = segment.strip()
        # Extract 5+ word phrases from the body and check if they appear in agent_output
        # Use sentence fragments split by punctuation/newlines
        phrases = re.split(r"[.\n!?;]", body)
        for phrase in phrases:
            phrase = phrase.strip()
            words = phrase.split()
            if len(words) < 6:
                continue
            # Slide a 6-word window
            for i in range(len(words) - 5):
                probe = " ".join(words[i : i + 6])
                if probe.lower() in agent_output.lower():
                    if current_heading not in chunk_ids:
                        chunk_ids.append(current_heading)
                    quote = phrase[:120]
                    if quote not in free_text_quotes:
                        free_text_quotes.append(quote)
                    break
            if current_heading in chunk_ids:
                break

    return chunk_ids, free_text_quotes


def _compute_attachment_usages(
    attachment_context: str,
    bytes_sent: int,
    bytes_truncated: int,
    agent_results: list[Any],
    attachment_names: list[str] | None = None,
) -> list[Any]:
    """Build :class:`~vaig.skills.service_health.schema.AttachmentUsage` records.

    One record per agent result in *agent_results*.  Heuristically detects
    which attachment chunks each agent referenced in its output.

    Args:
        attachment_context: The rendered attachment context string passed to agents.
        bytes_sent: Byte length of *attachment_context*.
        bytes_truncated: Bytes dropped due to budget (0 when not truncated).
        agent_results: List of ``AgentResult`` objects from the pipeline run.
        attachment_names: Top-level attachment names (from ``adapter.spec.name`` or
            ``adapter.spec.source``).  When provided and contains exactly one entry,
            that name is used for all ``AttachmentUsage`` records.  When ``None`` or
            empty, the name is derived heuristically from the first ``###`` heading.

    Returns a list of ``AttachmentUsage`` instances (imported lazily to avoid
    circular imports at module level).
    """
    import re

    from vaig.skills.service_health.schema import AttachmentUsage  # noqa: PLC0415

    # Resolve the display name for the attachment
    if attachment_names and len(attachment_names) == 1:
        top_level_name = attachment_names[0]
    elif attachment_names and len(attachment_names) > 1:
        top_level_name = "multiple"
    else:
        # Fallback: derive from first ### heading (chunk file_path, not adapter name)
        m = re.search(r"### (.+)", attachment_context)
        top_level_name = m.group(1).strip() if m else "attachment"

    usages: list[AttachmentUsage] = []
    for ar in agent_results:
        agent_output = getattr(ar, "content", "") or ""  # AgentResult uses .content
        agent_name = getattr(ar, "agent_name", "") or ""
        chunk_ids, quotes = _detect_chunk_references(attachment_context, agent_output)
        usages.append(
            AttachmentUsage(
                agent_name=agent_name,
                attachment_name=top_level_name,
                context_bytes_received=bytes_sent,
                context_bytes_truncated=bytes_truncated,
                chunks_referenced=chunk_ids,
                free_text_quotes=quotes,
            )
        )
    return usages


def _aggregate_attachment_summaries(usages: list[Any]) -> list[Any]:
    """Aggregate per-agent usages into per-attachment summaries.

    Returns a list of :class:`~vaig.skills.service_health.schema.AttachmentEvidenceSummary`.
    """
    from vaig.skills.service_health.schema import AttachmentEvidenceSummary  # noqa: PLC0415

    per_attachment: dict[str, dict[str, Any]] = {}
    for u in usages:
        name = u.attachment_name
        if name not in per_attachment:
            per_attachment[name] = {
                "bytes_sent": u.context_bytes_received,
                "bytes_truncated": u.context_bytes_truncated,
                "agents": set(),
                "total_chunks": 0,
            }
        if u.chunks_referenced:
            per_attachment[name]["agents"].add(u.agent_name)
            per_attachment[name]["total_chunks"] += len(u.chunks_referenced)

    summaries = []
    for name, data in per_attachment.items():
        summaries.append(
            AttachmentEvidenceSummary(
                attachment_name=name,
                bytes_sent=data["bytes_sent"],
                bytes_truncated=data["bytes_truncated"],
                agents_that_cited=sorted(data["agents"]),
                total_chunks_cited=data["total_chunks"],
            )
        )
    return summaries


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
            len(chunk.file_path.encode("utf-8")) + len(chunk.content.encode("utf-8")) + _RENDER_OVERHEAD_PER_CHUNK
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
        carrier.attachment_gaps = [f"[{g.source}] {g.reason}: {g.details or ''}" for g in extra_gaps]
        return carrier

    carrier = window_results[0]

    # Merge structured reports
    raw_reports = [wr.structured_report for wr in window_results if wr.structured_report is not None]
    valid_reports: list[HealthReport] = [r for r in raw_reports if isinstance(r, HealthReport)]
    merged_report = merge_health_reports(valid_reports)

    # Append extra_gaps into the merged report's evidence_gaps
    if merged_report is not None and extra_gaps:
        merged_report = merged_report.model_copy(
            update={
                "evidence_gaps": list(merged_report.evidence_gaps) + list(extra_gaps),
            }
        )

    # Sum costs and usage
    total_cost = sum(wr.run_cost_usd for wr in window_results)
    combined_usage: dict[str, int] = {}
    for wr in window_results:
        for k, v in wr.total_usage.items():
            combined_usage[k] = combined_usage.get(k, 0) + v

    # Concatenate synthesized output
    synthesized = "\n\n---\n\n".join(wr.synthesized_output for wr in window_results if wr.synthesized_output)

    # Stringify extra_gaps for OrchestratorResult.attachment_gaps
    gap_strings = [f"[{g.source}] {g.reason}: {g.details or ''}" for g in extra_gaps]

    # Only replace structured_report when merge produced a result;
    # otherwise fall back to the first non-None structured_report from windows.
    if merged_report is not None:
        carrier.structured_report = merged_report
    elif carrier.structured_report is None:
        carrier.structured_report = next(
            (wr.structured_report for wr in window_results if wr.structured_report is not None),
            None,
        )
    carrier.run_cost_usd = total_cost
    carrier.total_usage = combined_usage
    carrier.synthesized_output = synthesized
    carrier.success = any(wr.success for wr in window_results)
    carrier.attachment_gaps = gap_strings

    return carrier


def _run_attachment_gatherer_pass(
    attachment_context: str,
    client: Any,
    *,
    model_id: str | None = None,
) -> Any:
    """Run the ATT-10 §6.5.1 attachment_gatherer pass (post-hoc enrichment).

    Extracts an ``AttachmentPriors`` object from *attachment_context* using a
    single bounded LLM call (or returns a cached result when the composite
    cache key — covering text, system prompt, and model ID — matches a
    previous call).

    **Lifecycle note**: this pass runs *after* ``execute_with_tools`` completes.
    Sub-gatherers in the current sprint therefore do *not* receive the priors
    during their execution; the priors are available for downstream consumers
    (reporters, CLI output) only.  Feeding priors into sub-gatherers is
    deferred to §6.5.2.

    Parameters
    ----------
    attachment_context:
        Rendered attachment context string (already chunked/truncated).
    client:
        ``GeminiClient`` instance used for the extraction call.
    model_id:
        Optional model override.  ``None`` uses the client's current model.

    Returns
    -------
    AttachmentPriors
        The extracted (or cached) priors.  Never raises — returns an empty
        ``AttachmentPriors()`` on any failure.
    """
    from vaig.core.attachment_priors_extractor import extract_priors

    return extract_priors(attachment_context, client, model_id=model_id)


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

    client = GeminiClient(settings, fallback_model=settings.models.fallback)
    orchestrator = Orchestrator(client, settings)

    # Build attachment context from adapters (if any)
    attachment_context: str | None = None
    attachment_truncated = False
    attachment_gap_strings: list[str] = []
    index = None
    budget: int | None = None
    cfg = None
    total_index_bytes: int = 0
    attachment_names: list[str] = []
    attachment_contexts_used: list[str] = []

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
            # ATT-11: compute total available bytes before any slicing/truncation
            total_index_bytes = sum(len(c.content.encode("utf-8")) for c in index.chunks)
            # ATT-11: collect top-level adapter display names
            attachment_names = [a.spec.name or a.spec.source for a in attachment_adapters]
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
            attachment_context, attachment_truncated = _render_attachment_context(windows[0], budget)
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
        # ATT-11: record context for post-run usage tracking
        if attachment_context:
            attachment_contexts_used.append(attachment_context)
        # ATT-10 §6.5.1: extract attachment priors from single-window context
        if attachment_context:
            result.attachment_priors = _run_attachment_gatherer_pass(attachment_context, client)

    else:
        # ── MAP phase — sequential, one window at a time ──────────────────────
        from vaig.skills.service_health.schema import EvidenceGap as _EvidenceGap

        window_results: list[Any] = []
        map_gaps: list[Any] = []

        for idx, window in enumerate(windows, start=1):
            # A9: skip empty windows defensively
            if not window.chunks:
                logger.debug("map_reduce: skipping empty window %d/%d", idx, len(windows))
                continue

            window_ctx, window_truncated = _render_attachment_context(window, budget)
            # ATT-11: always record rendered context, even if execution fails
            attachment_contexts_used.append(window_ctx)
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
                    map_gaps.append(
                        _EvidenceGap(
                            source="map_reduce_window",
                            reason="window_internally_truncated",
                            details=f"window {idx}/{len(windows)} exceeded single-window budget",
                        )
                    )
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "map_reduce window %d/%d failed: %s",
                    idx,
                    len(windows),
                    exc,
                )
                map_gaps.append(
                    _EvidenceGap(
                        source="map_reduce_window",
                        reason="error",
                        details=f"window {idx}/{len(windows)} failed: {type(exc).__name__}: {exc}",
                    )
                )

        # ── REDUCE phase ──────────────────────────────────────────────────────
        result = _reduce_window_results(
            window_results=window_results,
            extra_gaps=map_gaps,
            windows_attempted=len(windows),
            skill=skill,
        )
        result.attachment_truncated = window_cap_hit
        result.map_reduce_windows_used = len(windows)
        # Merge attachment_gap_strings (cap message, adapter gaps) into result
        if attachment_gap_strings:
            existing = list(result.attachment_gaps or [])
            result.attachment_gaps = existing + attachment_gap_strings
        # ATT-10 §6.5.1: extract priors from full combined context (post-hoc)
        if attachment_context:
            result.attachment_priors = _run_attachment_gatherer_pass(attachment_context, client)

    # ── ATT-11: compute and surface attachment usage observability ────────────
    if attachment_contexts_used:
        # Concatenate all window contexts for uniform chunk-reference detection
        combined_context = "\n".join(attachment_contexts_used)
        bytes_sent = len(combined_context.encode("utf-8"))
        # Proper truncation delta: total available - what was actually rendered
        bytes_truncated = max(0, total_index_bytes - bytes_sent) if result.attachment_truncated else 0
        usages = _compute_attachment_usages(
            attachment_context=combined_context,
            bytes_sent=bytes_sent,
            bytes_truncated=bytes_truncated,
            agent_results=result.agent_results,
            attachment_names=attachment_names or None,
        )
        result.attachment_usages = usages
        summaries = _aggregate_attachment_summaries(usages)
        logger.info(
            "ATT-11 attachment usage: %d agent(s) processed, %d attachment(s) summarised",
            len(usages),
            len(summaries),
        )
        if result.structured_report is not None:
            try:
                result.structured_report.attachment_evidence = summaries
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:  # noqa: BLE001
                logger.debug("ATT-11: could not set attachment_evidence on structured_report")

    # ATT-10 §6.5.1: propagate attachment_priors onto structured_report
    _priors = getattr(result, "attachment_priors", None)
    if _priors is not None and result.structured_report is not None:
        try:
            result.structured_report.attachment_priors = _priors
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:  # noqa: BLE001
            logger.debug("ATT-10: could not set attachment_priors on structured_report")

    logger.info(
        "Headless execution complete: skill=%s, success=%s, cost=$%.4f",
        skill.get_metadata().name,
        result.success,
        result.run_cost_usd,
    )

    return result

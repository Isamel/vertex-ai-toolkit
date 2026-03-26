"""Live command — infrastructure investigation (GKE/GCP)."""

from __future__ import annotations

import logging
import tempfile
import threading
import time
import webbrowser
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.panel import Panel
from rich.status import Status
from rich.table import Table

from vaig.cli import _helpers
from vaig.cli._completions import complete_namespace
from vaig.cli._helpers import (
    _apply_subcommand_log_flags,
    _compute_cost_str,
    _handle_export_output,
    _show_coding_summary,
    _show_cost_line,
    console,
    err_console,
    handle_cli_error,
    track_command,
)
from vaig.cli.display import (
    print_colored_report,
    print_executive_summary_panel,
    print_recommendations_table,
)
from vaig.core.cache import ToolResultCache
from vaig.core.tool_call_store import ToolCallStore

if TYPE_CHECKING:
    from vaig.agents.orchestrator import OrchestratorResult
    from vaig.core.config import GKEConfig, Settings
    from vaig.core.protocols import GeminiClientProtocol
    from vaig.skills.base import BaseSkill, SkillMetadata
    from vaig.tools.base import ToolRegistry

logger = logging.getLogger(__name__)

MINIMUM_WATCH_INTERVAL = 10
"""Minimum seconds between watch iterations to prevent abuse."""


def _create_tool_call_store(settings: Settings) -> ToolCallStore | None:
    """Create a :class:`ToolCallStore` if tool-result persistence is enabled.

    Reads ``settings.logging.tool_results`` and
    ``settings.logging.tool_results_dir`` to decide whether to instantiate
    the store.  Returns ``None`` when disabled or on any creation error
    (never crashes the app).
    """
    if not settings.logging.tool_results:
        return None

    try:
        base_dir = Path(settings.logging.tool_results_dir).expanduser()
        store = ToolCallStore(base_dir=base_dir)
        logger.debug("ToolCallStore initialised: %s", base_dir)
        return store
    except Exception:  # noqa: BLE001
        logger.warning("Failed to create ToolCallStore; tool results will not be persisted.", exc_info=True)
        return None


# ── Live tool execution logger ────────────────────────────────


def _truncate_args(args: dict[str, Any], max_len: int = 50) -> str:
    """Format tool args into a compact, truncated string.

    Shows key=value pairs, truncating values longer than *max_len* chars.
    """
    if not args:
        return ""
    parts: list[str] = []
    for k, v in args.items():
        v_str = str(v)
        if len(v_str) > max_len:
            v_str = v_str[:max_len] + "..."
        parts.append(f"{k}={v_str}")
    return ", ".join(parts)


class ToolCallLogger:
    """Callback that prints live tool execution feedback to the console.

    Tracks tool call count and cumulative duration so a final summary
    line can be printed via :meth:`print_summary`.

    Usage::

        tool_logger = ToolCallLogger()
        orchestrator.execute_with_tools(..., on_tool_call=tool_logger)
        tool_logger.print_summary()
    """

    def __init__(self) -> None:
        self.tool_count: int = 0
        self.total_duration: float = 0.0
        self.errors: int = 0
        self._error_reasons: list[str] = []
        self._pipeline_start: float = time.perf_counter()
        self.tool_name_counts: Counter[str] = Counter()
        self.pipeline_tool_name_counts: Counter[str] = Counter()
        self.cache_hits: int = 0

    @staticmethod
    def _extract_reason(error_message: str) -> str:
        """Extract a short reason category from an error message."""
        if not error_message:
            return "unknown"
        # Take the first line, truncated to 40 chars
        first_line = error_message.split("\n", 1)[0].strip()
        if len(first_line) > 40:
            first_line = first_line[:40] + "..."
        return first_line

    def __call__(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        duration: float,
        success: bool,
        error_message: str = "",
        *,
        cached: bool = False,
    ) -> None:
        """Print a single tool execution line."""
        self.tool_count += 1
        self.total_duration += duration
        self.tool_name_counts[tool_name] += 1

        if cached:
            self.cache_hits += 1

        args_str = _truncate_args(tool_args)

        if success:
            cache_tag = r" [yellow]\[cached][/yellow]" if cached else ""
            status = f"[green]✓[/green]{cache_tag}"
        else:
            self.errors += 1
            # Build error detail for display (truncated to 80 chars)
            if error_message:
                err_short = error_message.split("\n", 1)[0].strip()
                if len(err_short) > 80:
                    err_short = err_short[:80] + "..."
                status = f"[red]✗ FAIL[/red] [dim]{err_short}[/dim]"
                self._error_reasons.append(self._extract_reason(error_message))
            else:
                status = "[red]✗ FAIL[/red]"
                self._error_reasons.append("unknown")

        console.print(f"  🔧 [cyan]{tool_name}[/cyan]({args_str}) {status} [dim]({duration:.1f}s)[/dim]")

    def format_tool_counts(self) -> str:
        """Format per-tool-name breakdown for display.

        Returns a string like ``kubectl_get ×4 | get_events ×2 (1 cached)``.
        """
        if not self.tool_name_counts:
            return ""
        parts = [f"{name} ×{count}" for name, count in self.tool_name_counts.most_common()]
        breakdown = " | ".join(parts)
        if self.cache_hits:
            breakdown += f" ({self.cache_hits} cached)"
        return breakdown

    def reset(self) -> None:
        """Reset per-agent counters while keeping pipeline-level totals."""
        self.pipeline_tool_name_counts.update(self.tool_name_counts)
        self.tool_name_counts.clear()
        self.cache_hits = 0

    def print_summary(self) -> None:
        """Print the final pipeline summary line."""
        total_wall = time.perf_counter() - self._pipeline_start
        status = (
            "[green]Pipeline complete[/green]" if self.errors == 0 else "[yellow]Pipeline complete with errors[/yellow]"
        )

        # Build failure detail with grouped reasons
        fail_detail = ""
        if self.errors:
            reason_counts = Counter(self._error_reasons)
            if reason_counts:
                grouped = ", ".join(f"{count}\u00d7 {reason}" for reason, count in reason_counts.most_common())
                fail_detail = f", {self.errors} failed ({grouped})"
            else:
                fail_detail = f", {self.errors} failed"

        # Build tool-name breakdown
        tool_breakdown = self.format_tool_counts()
        breakdown_detail = f"\n  [dim]Tools: {tool_breakdown}[/dim]" if tool_breakdown else ""

        console.print(
            f"\n{status} "
            f"[dim]({total_wall:.1f}s total, "
            f"{self.tool_count} tool{'s' if self.tool_count != 1 else ''} executed"
            f"{fail_detail})[/dim]{breakdown_detail}"
        )


class AgentProgressDisplay:
    """Live progress indicator for multi-agent pipeline execution.

    Shows a Rich :class:`~rich.status.Status` spinner with the current
    agent name, step number, and running tool count::

        [1/4] health_gatherer — running... (12 tools called)

    Implements the :class:`~vaig.agents.orchestrator.OnAgentProgress`
    protocol so it can be passed directly to
    ``Orchestrator.execute_with_tools(on_agent_progress=...)``.

    Thread-safe: a :class:`threading.Lock` serialises all ``start``/``end``
    mutations so parallel gatherers cannot race on ``self._status`` and
    cause a Rich ``LiveError("Only one live display may be active at once")``.

    Args:
        tool_logger: The :class:`ToolCallLogger` for the current pipeline.
            Used to read the running ``tool_count`` for display.
    """

    def __init__(self, tool_logger: ToolCallLogger) -> None:
        self._tool_logger = tool_logger
        self._status: Status | None = None
        self._current_agent_name: str | None = None
        # Per-agent tool counts at start time — keyed by agent_name.
        # Avoids overwriting a single shared counter when parallel gatherers
        # fire "start" events before any "end" events arrive.
        self._agent_start_counts: dict[str, int] = {}
        # Count of agents that have started but not yet ended.
        # Used to defer tool_logger.reset() until ALL active agents finish,
        # so parallel gatherers don't clear shared counters mid-run.
        self._active_agents: int = 0
        self._lock = threading.Lock()

    def _stop_current(self) -> None:
        """Stop and discard the current spinner (if any).

        Swallows any exception from ``Status.stop()`` so a stale or
        already-stopped spinner never breaks the pipeline.  Must be called
        with ``self._lock`` held.
        """
        if self._status is not None:
            try:
                self._status.stop()
            except Exception:  # noqa: BLE001
                logger.debug("AgentProgressDisplay: ignored error stopping previous status", exc_info=True)
            finally:
                self._status = None

    def __call__(
        self,
        agent_name: str,
        agent_index: int,
        total_agents: int,
        event: str,
        end_agent_index: int | None = None,
    ) -> None:
        """Handle agent start/end events (thread-safe).

        Args:
            agent_name: Display name of the agent or agent group.
            agent_index: Zero-based start index in the pipeline.
            total_agents: Total agents in the pipeline.
            event: ``"start"`` or ``"end"``.
            end_agent_index: Optional inclusive end index for a range display.
                When set and different from *agent_index*, the step counter
                renders as ``[start+1-end+1/total]`` (e.g. ``[1-4/7]``)
                instead of the default ``[index+1/total]``.
        """
        # Build step label: "[1-4/7]" for a range, "[3/7]" for a single agent.
        def _step_label() -> str:
            start = agent_index + 1
            if end_agent_index is not None and end_agent_index != agent_index:
                end = end_agent_index + 1
                return f"{start}-{end}/{total_agents}"
            return f"{start}/{total_agents}"

        with self._lock:
            if event == "start":
                # Record tool count at this agent's start for accurate per-agent
                # tool tallies.  Using a dict keyed by agent_name instead of a
                # single shared counter prevents parallel gatherers from
                # overwriting each other's baseline before their "end" fires.
                self._agent_start_counts[agent_name] = self._tool_logger.tool_count
                self._current_agent_name = agent_name
                self._active_agents += 1
                label = (
                    f"[bold cyan]\\[{_step_label()}][/bold cyan] "
                    f"[green]{agent_name}[/green] — running..."
                )
                # Stop any existing Live display before creating a new one.
                # Rich only allows one live display active at a time; without
                # this the second parallel gatherer raises:
                #   LiveError: Only one live display may be active at once
                self._stop_current()
                self._status = console.status(label, spinner="dots")
                self._status.start()
            elif event == "end":
                # In parallel execution, "end" events fire in submission order
                # but the spinner always belongs to the most-recently-started
                # gatherer (i.e. the one whose "start" ran last).  Only stop
                # the spinner when this "end" matches the agent that owns it;
                # otherwise we would kill the wrong gatherer's spinner.
                if agent_name == self._current_agent_name:
                    self._stop_current()
                    self._current_agent_name = None
                start_count = self._agent_start_counts.pop(agent_name, 0)
                tools = self._tool_logger.tool_count - start_count
                breakdown = self._tool_logger.format_tool_counts()
                tools_detail = f" ({tools} tool{'s' if tools != 1 else ''} called)"
                breakdown_detail = f" [dim]{breakdown}[/dim]" if breakdown else ""
                console.print(
                    f"  [bold cyan]\\[{_step_label()}][/bold cyan] "
                    f"[green]{agent_name}[/green] — [green]done[/green]"
                    f"{tools_detail}{breakdown_detail}"
                )
                # Decrement active agent count; reset per-agent counters only
                # when ALL active agents have finished.  Calling reset() on every
                # "end" would clear shared tool_name_counts while other parallel
                # gatherers are still running, corrupting their breakdowns.
                self._active_agents = max(0, self._active_agents - 1)
                if self._active_agents == 0:
                    self._tool_logger.reset()

    def stop(self) -> None:
        """Stop the spinner if it's still running (e.g. on error)."""
        with self._lock:
            self._stop_current()


def _emit_bell(*, no_bell: bool) -> None:
    """Print the terminal bell character unless suppressed by ``--no-bell``."""
    if not no_bell:
        import sys

        sys.stdout.write("\a")
        sys.stdout.flush()


def register(app: typer.Typer) -> None:
    """Register the live command on the given Typer app."""

    @app.command()
    @track_command
    def live(
        question: Annotated[str, typer.Argument(help="Infrastructure question or investigation task")],
        config: Annotated[str | None, typer.Option("--config", "-c", help="Path to config YAML")] = None,
        model: Annotated[str | None, typer.Option("--model", "-m", help="Model to use")] = None,
        output: Annotated[Path | None, typer.Option("--output", "-o", help="Save response to a file")] = None,
        format_: Annotated[str | None, typer.Option("--format", help="Export format: json, md, html")] = None,
        skill: Annotated[str | None, typer.Option("--skill", "-s", help="SRE skill to apply")] = None,
        auto_skill: Annotated[
            bool, typer.Option("--auto-skill", help="Auto-detect the best skill based on query")
        ] = False,
        cluster: Annotated[str | None, typer.Option("--cluster", help="GKE cluster name (overrides config)")] = None,
        namespace: Annotated[
            str | None,
            typer.Option(
                "--namespace", help="Default Kubernetes namespace (overrides config)", autocompletion=complete_namespace
            ),
        ] = None,
        project: Annotated[
            str | None,
            typer.Option("--project", "-p", help="GCP project ID (overrides gcp.project_id only)"),
        ] = None,
        project_id: Annotated[
            str | None, typer.Option("--project-id", help="GCP project ID (overrides config, alias for --project)")
        ] = None,
        location: Annotated[str | None, typer.Option("--location", help="GCP location (overrides config)")] = None,
        gke_project: Annotated[
            str | None,
            typer.Option("--gke-project", help="GKE project ID (overrides gke.project_id; defaults to --project if unset)"),
        ] = None,
        gke_location: Annotated[
            str | None,
            typer.Option("--gke-location", help="GKE cluster location (overrides gke.location)"),
        ] = None,
        watch: Annotated[
            int | None, typer.Option("--watch", "-w", help="Re-execute every N seconds (polling mode, min 10s)")
        ] = None,
        dry_run: Annotated[
            bool,
            typer.Option("--dry-run", "--dry", help="Show execution plan without running"),
        ] = False,
        verbose: Annotated[
            bool,
            typer.Option("--verbose", "-V", help="Enable verbose logging (INFO level)"),
        ] = False,
        debug: Annotated[
            bool,
            typer.Option("--debug", "-d", help="Enable debug logging (DEBUG level, shows paths and full tracebacks)"),
        ] = False,
        summary: Annotated[
            bool,
            typer.Option("--summary", help="Show compact summary instead of full report"),
        ] = False,
        no_bell: Annotated[
            bool,
            typer.Option("--no-bell", help="Suppress terminal bell after pipeline completes"),
        ] = False,
        open_browser: Annotated[
            bool,
            typer.Option("--open", "-O", help="Open HTML report in default browser (requires --format html)"),
        ] = False,
    ) -> None:
        """Investigate live GKE/GCP infrastructure using AI with read-only tools.

        Launches an autonomous SRE agent that can inspect pods, logs, metrics,
        and Cloud Logging/Monitoring to answer infrastructure questions.

        All tools are READ-ONLY — no cluster modifications are possible.

        Examples:
            vaig live "What pods are crashing in the production namespace?"
            vaig live "Check CPU and memory pressure on nodes"
            vaig live "Investigate OOM kills in the last hour" --namespace=production
            vaig live "Show HPA status for frontend deployment" --cluster=prod-gke
            vaig live "Analyze error rate from Cloud Logging" --project-id=my-project
            vaig live "Why is the payment service returning 503s?" -o report.md
            vaig live "Investigate pod crashes" --format json -o report.json
            vaig live "Check pod health in production" --watch 60
            vaig live "Check service health" --dry-run
        """
        _apply_subcommand_log_flags(verbose=verbose, debug=debug)

        # Validate --watch interval
        if watch is not None and watch < MINIMUM_WATCH_INTERVAL:
            err_console.print(f"[red]--watch interval must be >= {MINIMUM_WATCH_INTERVAL}s (got {watch}s)[/red]")
            raise typer.Exit(1)

        # Warn if --open is used without --format html
        normalised_format_flag = format_.strip().lower() if format_ else None
        if open_browser and normalised_format_flag != "html":
            err_console.print(
                "[yellow]⚠ --open requires --format html — ignoring --open flag.[/yellow]"
            )
            open_browser = False

        try:  # ── CLI error boundary ──
            settings = _helpers._get_settings(config)

            # Eagerly initialize the telemetry collector and wire the
            # TelemetrySubscriber so events are forwarded to the SQLite store.
            _helpers._init_telemetry(settings)

            # Apply --project / --project-id: mutate ONLY gcp.project_id
            # The GKE fallback chain (gke.project_id or gcp.project_id) handles single-project setups.
            effective_project = project or project_id
            if effective_project:
                settings.gcp.project_id = effective_project

            # Apply --gke-project: mutate ONLY gke.project_id when explicitly provided
            if gke_project:
                settings.gke.project_id = gke_project

            # Apply --gke-location: mutate ONLY gke.location when explicitly provided
            if gke_location:
                settings.gke.location = gke_location

            # Apply --location: mutate gcp.location before component creation
            if location:
                settings.gcp.location = location

            if model:
                settings.models.default = model

            from vaig.core.container import build_container

            container = build_container(settings)
            client = container.gemini_client
            # Do NOT pass project_id/location here — gke_project/gke_location are
            # already written to settings.gke.* above, and _build_gke_config reads
            # them via its fallback chain (gke.project_id or gcp.project_id).
            # Passing effective_project/location would override gke-specific flags.
            gke_config = _build_gke_config(
                settings,
                cluster=cluster,
                namespace=namespace,
            )

            # Auto-detect skill if requested (or enabled in config) and no explicit skill specified
            effective_auto_skill = auto_skill or settings.skills.auto_routing
            if effective_auto_skill and not skill:
                from vaig.skills.registry import SkillRegistry

                registry = SkillRegistry(settings)
                suggestions = registry.suggest_skill(question)
                if suggestions:
                    best_name, best_score = suggestions[0]
                    threshold = settings.skills.auto_routing_threshold
                    if best_score >= threshold:
                        skill = best_name
                        console.print(
                            f"[dim]🎯 Auto-routing to skill: [cyan]{skill}[/cyan] (score: {best_score:.1f})[/dim]"
                        )
                    else:
                        console.print(
                            f"[dim]Suggested skills: {', '.join(f'{n} ({s:.1f})' for n, s in suggestions)}[/dim]"
                        )

            # If a skill is specified, check whether it needs the full orchestrated
            # pipeline (requires_live_tools=True) or the simple context-prepend approach.
            context_str = ""
            active_skill = None
            if skill:
                from vaig.skills.registry import SkillRegistry

                registry = SkillRegistry(settings)
                active_skill = registry.get(skill)
                if not active_skill:
                    err_console.print(f"[red]Skill not found: {skill}[/red]")
                    err_console.print(f"[dim]Available: {', '.join(registry.list_names())}[/dim]")
                    raise typer.Exit(1)

                skill_meta = active_skill.get_metadata()

                if not skill_meta.requires_live_tools:
                    # ── Legacy context-prepend path (requires_live_tools=False) ──
                    context_str = (
                        f"## Active Skill: {skill_meta.display_name}\n\n"
                        f"{skill_meta.description}\n\n"
                        f"Apply the {skill_meta.name} analysis methodology to the investigation below."
                    )
                    active_skill = None  # Not orchestrated

            # ── Dry-run: show execution plan without running ──────
            if dry_run:
                _display_dry_run_plan(
                    gke_config=gke_config,
                    question=question,
                    settings=settings,
                    skill=active_skill,
                    skill_name=skill,
                    model_id=model or settings.models.default,
                )
                return

            # ── Tool result persistence ────────────────────────
            tool_call_store = _create_tool_call_store(settings)
            # Session-scoped cache: created once, reused across watch
            # iterations and single-shot calls.  TTL=0 (no expiration)
            # is safe because the cache is discarded when the CLI exits,
            # and watch iterations overwrite entries via put() on each run.
            tool_result_cache = ToolResultCache()

            def _run_once() -> None:
                """Execute a single iteration of the live command."""
                if active_skill is not None:
                    _execute_orchestrated_skill(
                        client,
                        settings,
                        gke_config,
                        active_skill,
                        question,
                        output=output if not watch else None,
                        format_=format_ if not watch else None,
                        model_id=model,
                        tool_call_store=tool_call_store,
                        summary=summary,
                        no_bell=no_bell,
                        open_browser=open_browser if not watch else False,
                    )
                else:
                    _execute_live_mode(
                        client,
                        gke_config,
                        question,
                        context_str,
                        settings=settings,
                        output=output if not watch else None,
                        format_=format_ if not watch else None,
                        skill_name=skill,
                        model_id=model,
                        tool_call_store=tool_call_store,
                        tool_result_cache=tool_result_cache,
                        open_browser=open_browser if not watch else False,
                    )

            if not watch:
                # ── Single execution (default) ────────────────────
                _run_once()
                return

            # ── Watch mode: re-execute every N seconds ────────────
            _run_watch_loop(
                run_fn=_run_once,
                interval=watch,
                question=question,
                output=output,
                format_=format_,
            )
        except typer.Exit:
            raise  # Let typer exits pass through
        except Exception as exc:  # noqa: BLE001
            handle_cli_error(exc, debug=debug)


# ── Watch mode loop ──────────────────────────────────────────


def _run_watch_loop(
    *,
    run_fn: Callable[[], None],
    interval: int,
    question: str,
    output: Path | None = None,
    format_: str | None = None,
) -> None:
    """Execute *run_fn* repeatedly every *interval* seconds.

    Handles ``KeyboardInterrupt`` gracefully: stops the loop and prints
    a summary of iterations run and total elapsed time.

    The *output* / *format_* args are only used on the **last** iteration
    (i.e. never — the caller passes ``None`` to *run_fn* for those during
    watch mode, and can export manually after the loop).

    Args:
        run_fn: Zero-argument callable that executes one iteration.
        interval: Seconds between iterations (already validated >= MINIMUM_WATCH_INTERVAL).
        question: Original query (shown in the header).
        output: Export path (informational — not used inside the loop).
        format_: Export format (informational — not used inside the loop).
    """
    iteration = 0
    start_time = time.monotonic()

    console.print(
        Panel.fit(
            f"[bold cyan]Watch mode enabled — refreshing every {interval}s[/bold cyan]\n"
            f"[dim]Query: {question[:80]}{'...' if len(question) > 80 else ''}[/dim]\n"
            "[dim]Press Ctrl+C to stop[/dim]",
            border_style="cyan",
        )
    )

    try:
        while True:
            iteration += 1
            console.print(f"\n{'=' * 60}")
            console.print(
                f"[bold cyan]Watch iteration #{iteration} — {datetime.now().strftime('%H:%M:%S')}[/bold cyan]"
            )
            console.print(f"{'=' * 60}\n")

            try:
                run_fn()
            except (SystemExit, typer.Exit):
                # Agent errors (MaxIterationsError etc.) raise typer.Exit —
                # in watch mode we log and continue to the next iteration.
                console.print(f"[yellow]Iteration #{iteration} exited with error — continuing watch...[/yellow]")

            console.print(f"\n[dim]Next refresh in {interval}s (Ctrl+C to stop)...[/dim]")
            time.sleep(interval)
    except KeyboardInterrupt:
        elapsed = time.monotonic() - start_time
        console.print(
            f"\n[bold yellow]Watch stopped after {iteration} "
            f"iteration{'s' if iteration != 1 else ''} "
            f"({elapsed:.0f}s elapsed)[/bold yellow]"
        )


# ── Live mode helpers ─────────────────────────────────────────


def _build_gke_config(
    settings: Settings,
    *,
    cluster: str | None = None,
    namespace: str | None = None,
    project_id: str | None = None,
    location: str | None = None,
) -> GKEConfig:
    """Build a GKEConfig, applying CLI overrides on top of config file defaults.

    Args:
        settings: Application settings (contains gke section).
        cluster: Optional cluster name override.
        namespace: Optional default namespace override.
        project_id: Optional GCP project ID override.
        location: Optional GKE cluster location/zone/region override.

    Returns:
        GKEConfig with CLI overrides applied.
    """
    from vaig.core.config import GKEConfig

    gke = settings.gke

    return GKEConfig(
        cluster_name=cluster or gke.cluster_name,
        project_id=project_id or gke.project_id or settings.gcp.project_id,
        default_namespace=namespace or gke.default_namespace,
        location=location or gke.location,
        kubeconfig_path=gke.kubeconfig_path,
        context=gke.context,
        log_limit=gke.log_limit,
        metrics_interval_minutes=gke.metrics_interval_minutes,
        proxy_url=gke.proxy_url,
        impersonate_sa=gke.impersonate_sa,
        exec_enabled=gke.exec_enabled,
        # Helm / ArgoCD — merge Settings-level config into GKEConfig flags
        helm_enabled=settings.helm.enabled,
        argocd_enabled=settings.argocd.enabled,
        argocd_server=settings.argocd.server,
        argocd_token=settings.argocd.token,
        argocd_context=settings.argocd.context,
        argocd_namespace=settings.argocd.namespace,
        argocd_verify_ssl=settings.argocd.verify_ssl,
    )


def _display_dry_run_plan(
    *,
    gke_config: GKEConfig,
    question: str,
    settings: Settings,
    skill: BaseSkill | None = None,
    skill_name: str | None = None,
    model_id: str = "",
) -> None:
    """Render the dry-run execution plan without running anything.

    Shows what *would* happen: resolved configuration, agents that
    would be created, available tools, and estimated cost — then exits.

    Args:
        gke_config: Resolved GKE configuration.
        question: The user's infrastructure question.
        settings: Application settings.
        skill: Resolved orchestrated skill instance (``None`` for InfraAgent path).
        skill_name: Skill name string (may be set even when *skill* is ``None``
            for context-prepend skills).
        model_id: Model identifier to display.
    """
    # ── Header ────────────────────────────────────────────────
    console.print(
        Panel.fit(
            f'[bold yellow]Dry Run — vaig live[/bold yellow] [dim]"{question}"[/dim]',
            border_style="yellow",
        )
    )

    # ── Configuration ─────────────────────────────────────────
    config_table = Table(title="Configuration", show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Key", style="bold")
    config_table.add_column("Value")
    config_table.add_row("Cluster", gke_config.cluster_name or "(kubeconfig default)")
    config_table.add_row("Namespace", gke_config.default_namespace or "default")
    config_table.add_row("Project", gke_config.project_id or "(auto-detect)")
    config_table.add_row("Location", gke_config.location or "(auto-detect)")
    config_table.add_row("Model", model_id or settings.models.default)

    if skill_name:
        routing = "explicit (--skill)" if not (settings.skills.auto_routing) else "auto-routed"
        config_table.add_row("Skill", f"{skill_name} ({routing})")

    console.print(config_table)
    console.print()

    # ── Agents ────────────────────────────────────────────────
    if skill is not None:
        # Orchestrated skill — show agent pipeline
        agents_config = skill.get_agents_config()
        agent_table = Table(title="Agents that would be created", show_lines=True)
        agent_table.add_column("#", style="dim", width=3)
        agent_table.add_column("Agent", style="cyan")
        agent_table.add_column("Role", style="dim")
        agent_table.add_column("Tools", style="green")
        agent_table.add_column("Model", style="dim")

        for idx, agent_cfg in enumerate(agents_config, 1):
            has_tools = agent_cfg.get("requires_tools", False)
            tools_label = "[green]yes[/green]" if has_tools else "[dim]no (synthesis)[/dim]"
            agent_table.add_row(
                str(idx),
                agent_cfg.get("name", "unknown"),
                agent_cfg.get("role", ""),
                tools_label,
                agent_cfg.get("model", model_id),
            )
        console.print(agent_table)
    else:
        # InfraAgent path — single agent
        console.print("[bold]Agent:[/bold] InfraAgent (single autonomous agent with tools)")

    console.print()

    # ── Tools ─────────────────────────────────────────────────
    tool_registry = _register_live_tools(gke_config, settings=settings)
    tools = tool_registry.list_tools()
    tool_count = len(tools)

    if tool_count == 0:
        console.print("[bold red]No infrastructure tools available![/bold red]")
        console.print("[yellow]Install: pip install vertex-ai-toolkit[live][/yellow]")
    else:
        tool_names = [t.name for t in tools]
        # Show first 10 tools inline, then "and N more"
        if len(tool_names) <= 10:
            tools_str = ", ".join(tool_names)
        else:
            tools_str = ", ".join(tool_names[:10]) + f", ... and {len(tool_names) - 10} more"
        console.print(f"[bold]Available tools ({tool_count}):[/bold] {tools_str}")

    console.print()

    # ── Cost estimate ─────────────────────────────────────────
    console.print("[bold]Estimated cost:[/bold] [dim]depends on tool usage (typically $0.02-0.10 per run)[/dim]")
    console.print()
    console.print("[dim]Run without --dry-run to execute.[/dim]")


def _register_live_tools(gke_config: GKEConfig, settings: Settings | None = None) -> ToolRegistry:
    """Create a ToolRegistry and register GKE + GCloud + plugin tools.

    Follows the same try/except ImportError pattern as InfraAgent._register_tools()
    so missing optional dependencies degrade gracefully.

    Args:
        gke_config: GKE configuration for tool creation.
        settings: Full application settings (used for plugin tool loading).

    Returns:
        Populated ToolRegistry (may be empty if no optional deps installed).
    """
    from vaig.tools.base import ToolRegistry

    registry = ToolRegistry()

    # Resolve GKE-specific credentials (SA impersonation or ADC)
    gke_credentials = None
    if settings is not None:
        from vaig.core.auth import get_gke_credentials

        gke_credentials = get_gke_credentials(settings)

    # GKE tools — requires 'kubernetes' package
    try:
        from vaig.tools.gke_tools import create_gke_tools  # noqa: WPS433

        for tool in create_gke_tools(gke_config):
            registry.register(tool)
    except ImportError as exc:
        logger.warning("Could not load GKE tools: %s", exc)

    # GCP observability tools — requires google-cloud-logging / google-cloud-monitoring
    try:
        from vaig.tools.gcloud_tools import create_gcloud_tools  # noqa: WPS433

        for tool in create_gcloud_tools(
            project=gke_config.project_id,
            log_limit=gke_config.log_limit,
            metrics_interval_minutes=gke_config.metrics_interval_minutes,
            credentials=gke_credentials,
        ):
            registry.register(tool)
    except ImportError as exc:
        logger.warning("Could not load GCloud observability tools: %s", exc)

    # Plugin tools — MCP auto-registration and Python module plugins
    if settings is not None:
        try:
            from vaig.tools.plugin_loader import load_all_plugin_tools  # noqa: WPS433

            for tool in load_all_plugin_tools(settings):
                registry.register(tool)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to load plugin tools for live mode. Skipping.",
                exc_info=True,
            )

    return registry


def _export_html_report(
    report: Any,
    *,
    console: Any,
    err_console: Any,
    output: Path | None = None,
    open_browser: bool = False,
) -> bool:
    """Try to write a rich HTML report to disk.

    Returns True if the HTML was written successfully, False otherwise.
    ``report`` should be a HealthReport instance (typed as Any to avoid
    a hard import at module level).

    Args:
        report: The HealthReport instance to render.
        console: Rich console for success messages.
        err_console: Rich console for error messages.
        output: Optional explicit output path. When ``None`` and ``open_browser``
            is False, a timestamped filename is generated in the current working
            directory.  When ``None`` and ``open_browser`` is True, a temporary
            file is used instead so the working directory is not cluttered.
        open_browser: When True, open the generated HTML file in the default
            system web browser after writing.
    """
    try:
        from vaig.ui.html_report import render_health_report_html  # noqa: WPS433

        html_content = render_health_report_html(report)
        if output is not None:
            out_path = output
        elif open_browser:
            # Use a temp file so we don't clutter the working directory when
            # the user just wants to view the report quickly.
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                out_path = Path(tmp.name)
        else:
            timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
            out_path = Path(f"vaig-report-{timestamp}.html")
        out_path.write_text(html_content, encoding="utf-8")
        console.print(
            f"[bold green]✓ HTML report written:[/bold green] [cyan]{out_path.resolve()}[/cyan]"
        )
        if open_browser:
            file_url = out_path.resolve().as_uri()
            try:
                opened = webbrowser.open(file_url)
                if not opened:
                    console.print(
                        f"[yellow]⚠ Could not open browser automatically. "
                        f"Open manually:[/yellow] [cyan]{file_url}[/cyan]"
                    )
            except Exception as browser_exc:  # noqa: BLE001
                console.print(
                    f"[yellow]⚠ Browser open failed ({browser_exc}). "
                    f"Open manually:[/yellow] [cyan]{file_url}[/cyan]"
                )
        return True
    except Exception as exc:  # pragma: no cover  # noqa: BLE001
        err_console.print(
            f"[bold red]⚠ Failed to write HTML report:[/bold red] {exc}"
        )
        return False


def _inject_report_metadata(
    report: Any,
    *,
    gke_config: Any = None,
    model_id: str = "",
    orch_result: Any = None,
    tool_logger: Any = None,
) -> None:
    """Fill metadata fields in *report* from runtime context.

    System-authoritative fields (``model_used``, ``cluster_name``,
    ``project_id``, ``generated_at``) are ALWAYS overwritten when the runtime
    has an authoritative value — the LLM may hallucinate these fields so we
    never trust whatever it wrote.  Cost and tool-usage fields are only set
    when not already populated (they are never known to the LLM).

    Args:
        report: A ``HealthReport`` instance (typed as ``Any`` to avoid a hard
            import; we access ``report.metadata`` defensively).
        gke_config: Optional :class:`~vaig.core.config.GKEConfig`.  When
            provided, its ``cluster_name`` and ``project_id`` fields are used
            to overwrite the corresponding metadata slots unconditionally.
        model_id: The model identifier used for the run.  Always overwrites
            ``metadata.model_used`` when a value is available.
        orch_result: Optional :class:`~vaig.agents.orchestrator.OrchestratorResult`.
            When provided, ``run_cost_usd`` and ``total_usage`` are extracted to
            populate ``metadata.cost_metrics``.
        tool_logger: Optional :class:`ToolCallLogger`.  When provided, its
            ``tool_name_counts`` and ``tool_count`` are used to populate
            ``metadata.tool_usage``.
    """
    metadata = getattr(report, "metadata", None)
    if metadata is None:
        return

    def _is_empty(value: Any) -> bool:
        """Return True when *value* is falsy or the sentinel 'N/A'."""
        if not value:
            return True
        if isinstance(value, str) and value.strip().upper() == "N/A":
            return True
        return False

    if gke_config is not None:
        for attr in ["cluster_name", "project_id"]:
            value = getattr(gke_config, attr, None)
            # Overwrite unconditionally — even empty/None clears hallucinated values
            setattr(metadata, attr, value if value is not None else "")

    # ALWAYS overwrite model_used — the LLM may hallucinate this field.
    if orch_result is not None:
        actual_models = getattr(orch_result, "models_used", [])
        effective_model = _format_models_used(actual_models) or model_id
    else:
        effective_model = model_id
    if effective_model:
        metadata.model_used = effective_model
    else:
        metadata.model_used = ""

    # ── Cost metrics ──────────────────────────────────────────
    if orch_result is not None and getattr(metadata, "cost_metrics", None) is None:
        from vaig.skills.service_health.schema import CostMetrics  # noqa: WPS433

        run_cost = getattr(orch_result, "run_cost_usd", None)
        total_usage = getattr(orch_result, "total_usage", None)
        total_tokens: int | None = None
        if isinstance(total_usage, dict):
            total_tokens = total_usage.get("total_tokens") or (
                total_usage.get("prompt_tokens", 0) + total_usage.get("completion_tokens", 0)
            ) or None
        cost_str = f"${run_cost:.6f}" if (run_cost is not None and run_cost > 0) else None

        if run_cost is not None or total_tokens is not None:
            metadata.cost_metrics = CostMetrics(
                run_cost_usd=run_cost,
                total_tokens=total_tokens,
                estimated_cost=cost_str,
            )

    # ── Tool usage ────────────────────────────────────────────
    if tool_logger is not None and getattr(metadata, "tool_usage", None) is None:
        from vaig.skills.service_health.schema import ToolUsageSummary  # noqa: WPS433

        pipeline_counts = dict(getattr(tool_logger, "pipeline_tool_name_counts", {}))
        live_counts = dict(getattr(tool_logger, "tool_name_counts", {}))
        tool_counts = (pipeline_counts or live_counts) or None
        tool_calls = getattr(tool_logger, "tool_count", None)
        if tool_calls == 0:
            tool_calls = None

        if tool_counts is not None or tool_calls is not None:
            metadata.tool_usage = ToolUsageSummary(
                tool_counts=tool_counts,
                tool_calls=tool_calls,
            )

    # ── GKE workload cost estimation ──────────────────────────
    if gke_config is not None and getattr(metadata, "gke_cost", None) is None:
        try:
            from vaig.tools.gke.cost_estimation import fetch_workload_costs  # noqa: WPS433

            metadata.gke_cost = fetch_workload_costs(gke_config)
        except Exception as _gke_cost_exc:  # noqa: BLE001
            logger.debug("GKE cost estimation skipped: %s", _gke_cost_exc)

    # ── Generated-at timestamp — ALWAYS overwrite with actual time ────────────
    metadata.generated_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    # ── Skill version ─────────────────────────────────────────
    if _is_empty(getattr(metadata, "skill_version", None)):
        from vaig import __version__  # noqa: PLC0415

        metadata.skill_version = f"vaig {__version__}"

    # ── Cluster overview: inject Namespace row when missing ───
    if gke_config is not None:
        ns = getattr(gke_config, "default_namespace", None)
        if isinstance(ns, str) and ns and hasattr(report, "cluster_overview") and report.cluster_overview is not None:
            from vaig.skills.service_health.schema import ClusterMetric  # noqa: PLC0415

            already_has_ns = any(
                isinstance(row.metric, str) and row.metric.strip().lower() == "namespace"
                for row in report.cluster_overview
            )
            if not already_has_ns:
                report.cluster_overview.insert(
                    0, ClusterMetric(metric="Namespace", value=ns)
                )


def _dispatch_format_output(
    orch_result: Any,
    *,
    format_: str | None,
    output: Path | None,
    question: str,
    model_id: str,
    skill_name: str,
    console: Any,
    err_console: Any,
    gke_config: Any = None,
    open_browser: bool = False,
    tool_logger: Any = None,
) -> None:
    """Dispatch output based on *format_* for an orchestrated skill result.

    Handles both the HTML rich-export path and the fallback text/json/md export.
    This shared helper is called by both the sync and async orchestrated skill
    functions to avoid duplicating the format-dispatch logic.

    The *format_* value is normalised (stripped and lowercased) once here so
    callers do not need to worry about case or whitespace.

    When ``format_`` is ``"html"`` and ``orch_result.structured_report`` has a
    ``metadata`` attribute, empty / ``"N/A"`` fields are filled in from
    *gke_config* and *model_id* so the SPA dashboard never shows placeholder
    values.

    Args:
        orch_result: The :class:`~vaig.agents.orchestrator.OrchestratorResult`.
        format_: Raw format string from the CLI (may be ``None``).
        output: Optional output path from ``--output``.
        question: Original user question (passed to :func:`_handle_export_output`).
        model_id: Model identifier for metadata.
        skill_name: Skill name for metadata.
        console: Rich console for output.
        err_console: Rich console for errors/warnings.
        gke_config: Optional :class:`~vaig.core.config.GKEConfig` used to inject
            cluster / project metadata into the report before HTML rendering.
        open_browser: When True (and ``format_`` is ``"html"``), open the
            generated file in the default browser.
        tool_logger: Optional :class:`ToolCallLogger`.  When provided, its
            ``tool_name_counts`` and ``tool_count`` are injected into
            ``metadata.tool_usage`` before HTML rendering.
    """
    if not format_ and not output:
        return

    # Normalize once — handles 'HTML', '  html  ', etc.
    normalised_format = format_.strip().lower() if format_ else None

    if normalised_format == "html":
        if orch_result.structured_report is not None:
            # ── Inject runtime metadata into the report before rendering ──────
            _inject_report_metadata(
                orch_result.structured_report,
                gke_config=gke_config,
                model_id=model_id,
                orch_result=orch_result,
                tool_logger=tool_logger,
            )
            _export_html_report(
                orch_result.structured_report,
                console=console,
                err_console=err_console,
                output=output,
                open_browser=open_browser,
            )
        else:
            err_console.print(
                Panel(
                    "[yellow]No structured report available — falling back to basic HTML export.[/yellow]",
                    title="[bold red]HTML Export Warning[/bold red]",
                    border_style="red",
                )
            )
            # Fall back to basic HTML export so the user still gets HTML output.
            # When --open is requested and no explicit output path was given, write
            # to a temp file so the browser has a path to open (otherwise the
            # content would be printed to stdout and there'd be nothing to open).
            effective_output = output
            if open_browser and output is None:
                with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                    effective_output = Path(tmp.name)
            _handle_export_output(
                response_text=orch_result.synthesized_output or "",
                question=question,
                model_id=model_id,
                skill_name=skill_name,
                format_="html",  # basic HTML via ExportPayload.to_html() since structured report unavailable
                output=effective_output,
                tokens=orch_result.total_usage or None,
                cost=_compute_cost_str(orch_result.total_usage, model_id),
            )
            if open_browser and effective_output is not None:
                _open_html_in_browser(effective_output, console)
    else:
        _handle_export_output(
            response_text=orch_result.synthesized_output or "",
            question=question,
            model_id=model_id,
            skill_name=skill_name,
            format_=normalised_format,
            output=output,
            tokens=orch_result.total_usage or None,
            cost=_compute_cost_str(orch_result.total_usage, model_id),
        )


def _get_offline_fallback_context(skill_meta: SkillMetadata) -> str:
    """Return the context string used when no live tools are available.

    Emits a warning to the console and returns a plain-text context block
    that describes the active skill.  Used by both sync and async orchestrated
    skill paths when ``tool_count == 0``.
    """
    console.print(
        f"[bold yellow]⚠️  No live tools available for skill '{skill_meta.name}'. "
        "Running in offline context-prepend mode. Results may be limited.[/bold yellow]"
    )
    return (
        f"## Active Skill: {skill_meta.display_name}\n\n"
        f"{skill_meta.description}\n\n"
        f"Apply the {skill_meta.name} analysis methodology to the investigation below."
    )


def _execute_orchestrated_skill(
    client: GeminiClientProtocol,
    settings: Settings,
    gke_config: GKEConfig,
    skill: BaseSkill,
    question: str,
    *,
    output: Path | None = None,
    format_: str | None = None,
    model_id: str | None = None,
    tool_call_store: ToolCallStore | None = None,
    summary: bool = False,
    no_bell: bool = False,
    open_browser: bool = False,
) -> None:
    """Execute a skill through the Orchestrator's tool-aware pipeline.

    Used when a skill has ``requires_live_tools=True``.  Instead of the
    simple context-prepend approach, this creates a ToolRegistry, populates
    it with GKE/GCloud tools, and delegates to
    ``Orchestrator.execute_with_tools()`` for the full multi-agent pipeline.
    """
    from vaig.agents.orchestrator import Orchestrator
    from vaig.core.exceptions import MaxIterationsError

    skill_meta = skill.get_metadata()

    # Build tool registry with live tools
    tool_registry = _register_live_tools(gke_config, settings=settings)

    # Detect Autopilot mode (result is cached from create_gke_tools)
    # Pass GKE credentials so the Container API call uses the right SA.
    gke_credentials = None
    if settings is not None:
        from vaig.core.auth import get_gke_credentials as _get_gke_creds

        gke_credentials = _get_gke_creds(settings)

    is_autopilot: bool | None = None
    try:
        from vaig.tools.gke_tools import detect_autopilot  # noqa: WPS433

        is_autopilot = detect_autopilot(gke_config, credentials=gke_credentials)
    except ImportError:
        pass

    tool_count = len(tool_registry.list_tools())
    if tool_count == 0:
        context_str = _get_offline_fallback_context(skill_meta)
        prompt = f"{context_str}\n\n## Investigation Question\n\n{question}"
        try:
            result = client.generate(prompt)
        except Exception as exc:  # noqa: BLE001
            handle_cli_error(exc)
            raise typer.Exit(1) from exc
        console.print()
        console.print("[bold]Assistant response (offline mode):[/bold]")
        console.print(result.text)
        return

    console.print(
        Panel.fit(
            f"[bold green]🔍 Orchestrated Skill: {skill_meta.display_name}[/bold green]\n"
            f"[dim]Cluster: {gke_config.cluster_name or '(kubeconfig default)'}[/dim]\n"
            f"[dim]Namespace: {gke_config.default_namespace} | "
            f"Project: {gke_config.project_id or '(auto-detect)'}[/dim]\n"
            f"[dim]{tool_count} tools loaded | Strategy: sequential[/dim]",
            border_style="green",
        )
    )

    orchestrator = Orchestrator(client, settings)

    try:
        console.print(f"[bold cyan]🤖 Running {skill_meta.display_name} pipeline...[/bold cyan]")
        tool_logger = ToolCallLogger()
        progress_display = AgentProgressDisplay(tool_logger)
        try:
            orch_result = orchestrator.execute_with_tools(
                query=question,
                skill=skill,
                tool_registry=tool_registry,
                strategy="sequential",
                is_autopilot=is_autopilot,
                on_tool_call=tool_logger,
                tool_call_store=tool_call_store,
                on_agent_progress=progress_display,
                gke_namespace=gke_config.default_namespace,
                gke_location=gke_config.location,
                gke_cluster_name=gke_config.cluster_name,
            )
        finally:
            progress_display.stop()
        tool_logger.print_summary()

        # Display final response with severity coloring
        console.print()
        if summary and orch_result.structured_report is not None:
            # --summary mode: compact output from the structured report
            console.print(orch_result.structured_report.to_summary())
        else:
            # Rich Panel for executive summary (before the full report)
            if orch_result.structured_report is not None:
                print_executive_summary_panel(
                    orch_result.structured_report, console=console,
                )
            if orch_result.synthesized_output:
                print_colored_report(orch_result.synthesized_output, console=console)
            # Rich Table for recommendations (after the full report)
            if orch_result.structured_report is not None:
                print_recommendations_table(
                    orch_result.structured_report, console=console,
                )
        console.print()

        # Dispatch output format — HTML or text/json/md fallback
        _dispatch_format_output(
            orch_result,
            format_=format_,
            output=output,
            question=question,
            model_id=settings.models.default,
            skill_name=skill_meta.name,
            console=console,
            err_console=err_console,
            gke_config=gke_config,
            open_browser=open_browser,
            tool_logger=tool_logger,
        )

        # Show agent pipeline summary (includes cost line)
        _show_orchestrated_summary(orch_result, model_id=settings.models.default)

        # Auto-export report if configured.
        # ADR-4: auto-export fires here for the health report only, immediately after the live
        # summary.  Telemetry and tool-calls are higher-volume and use the explicit CLI push
        # commands (vaig cloud push telemetry / tool-calls) so they are never auto-exported.
        _auto_export_report(settings, orch_result, gke_config)

        # Notify via terminal bell
        _emit_bell(no_bell=no_bell)

    except MaxIterationsError as exc:
        err_console.print(
            f"\n[bold red]⚠ Max iterations reached ({exc.iterations})[/bold red]\n"
            "[yellow]The orchestrated skill pipeline hit its tool-use iteration limit. "
            "Try narrowing the scope of your question or specifying a namespace/resource.[/yellow]"
        )
        raise typer.Exit(1)  # noqa: B904


def _auto_export_report(
    settings: Settings,
    orch_result: OrchestratorResult,
    gke_config: GKEConfig,
) -> None:
    """Fire-and-forget auto-export of a health report if configured.

    Checks whether auto-export is enabled and a structured report exists, then
    delegates to :func:`vaig.core.export.auto_export_report` on a daemon thread.
    """
    if not (settings.export.enabled and settings.export.auto_export_reports and orch_result.structured_report is not None):
        return
    from vaig.core.export import auto_export_report

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    auto_export_report(
        config=settings.export,
        report=orch_result.structured_report.to_dict(),
        run_id=run_id,
        cluster_name=gke_config.cluster_name or "",
        namespace=gke_config.default_namespace or "",
    )


def _format_models_used(models_used: list[str]) -> str:
    """Format a list of model IDs for display.

    Returns a compact human-readable string:
    - Single unique model → returned as-is (e.g. ``"gemini-2.5-flash"``).
    - All agents use the same model → ``"gemini-2.5-flash ×7"``.
    - Multiple distinct models → comma-separated list.
    - Empty list → empty string.
    """
    if not models_used:
        return ""
    counts = Counter(models_used)
    if len(counts) == 1:
        model, n = next(iter(counts.items()))
        return f"{model} ×{n}" if n > 1 else model
    return ", ".join(sorted(counts))


def _show_orchestrated_summary(orch_result: OrchestratorResult, *, model_id: str = "") -> None:
    """Display a summary table for an orchestrated skill execution."""
    table = Table(title="Pipeline Summary", show_lines=True)
    table.add_column("Agent", style="cyan")
    table.add_column("Role", style="dim")
    table.add_column("Status", style="bold")

    for agent_result in orch_result.agent_results:
        status = "[green]✓ success[/green]" if agent_result.success else "[red]✗ failed[/red]"
        role = agent_result.metadata.get("role") or agent_result.agent_name
        table.add_row(agent_result.agent_name, role, status)

    console.print(table)

    # Show cost summary for the full pipeline.
    # Prefer the pre-accumulated run_cost_usd (correct per-agent model pricing)
    # and fall back to recalculation only when it is zero/unavailable.
    _raw_cost = getattr(orch_result, "run_cost_usd", None)
    run_cost = _raw_cost if isinstance(_raw_cost, (int, float)) else 0.0
    models_used_list: list[str] = getattr(orch_result, "models_used", [])
    # Display label (may be a formatted string like "gemini-2.5-flash ×7")
    effective_model = _format_models_used(models_used_list) or model_id
    # Raw model ID for pricing lookups — must not be a formatted display string.
    # Use the single unique model when all agents share one model, else fall back
    # to the caller-supplied model_id.
    unique_models = list(dict.fromkeys(models_used_list))  # deduplicate, preserve order
    pricing_model_id = unique_models[0] if len(unique_models) == 1 else model_id
    if run_cost > 0.0:
        from vaig.core.pricing import format_cost

        usage = orch_result.total_usage or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        thinking_tokens = usage.get("thinking_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        if total_tokens > 0 or prompt_tokens > 0 or completion_tokens > 0:
            parts = [f"{prompt_tokens:,} in", f"{completion_tokens:,} out"]
            if thinking_tokens:
                parts.append(f"{thinking_tokens:,} thinking")
            cost_str = format_cost(run_cost)
            model_label = f" ({effective_model})" if effective_model else ""
            console.print(
                f"[dim]📊 Tokens: {' / '.join(parts)} "
                f"({total_tokens:,} total) │ Cost: {cost_str}{model_label}[/dim]"
            )
        else:
            cost_str = format_cost(run_cost)
            model_label = f" ({effective_model})" if effective_model else ""
            console.print(f"[dim]📊 Cost: {cost_str}{model_label}[/dim]")
    else:
        # No pre-computed cost — fall back to recalculation via show_cost_line.
        # Pass the raw model ID (pricing_model_id), not the formatted display string.
        _show_cost_line(orch_result.total_usage or None, pricing_model_id or model_id)

    if not orch_result.success:
        err_console.print("[bold red]⚠ Pipeline completed with errors[/bold red]")


def _open_html_in_browser(html_path: Path, console: Any) -> None:
    """Open *html_path* in the system default browser.

    Creates no temporary files — the caller is responsible for providing a
    valid path.  Handles browser-open failures gracefully by printing an
    actionable message instead of raising.

    Args:
        html_path: Absolute or resolvable path to the HTML file to open.
        console: Rich ``Console`` instance for user-facing messages.
    """
    file_url = html_path.resolve().as_uri()
    console.print(
        f"[bold green]✓ Report written:[/bold green] [cyan]{html_path.resolve()}[/cyan]"
    )
    try:
        opened = webbrowser.open(file_url)
        if not opened:
            console.print(
                f"[yellow]⚠ Could not open browser automatically. "
                f"Open manually:[/yellow] [cyan]{file_url}[/cyan]"
            )
    except Exception as browser_exc:  # noqa: BLE001
        console.print(
            f"[yellow]⚠ Browser open failed ({browser_exc}). "
            f"Open manually:[/yellow] [cyan]{file_url}[/cyan]"
        )


def _execute_live_mode(
    client: GeminiClientProtocol,
    gke_config: GKEConfig,
    question: str,
    context: str,
    *,
    settings: Settings | None = None,
    output: Path | None = None,
    format_: str | None = None,
    skill_name: str | None = None,
    model_id: str | None = None,
    tool_call_store: ToolCallStore | None = None,
    tool_result_cache: ToolResultCache | None = None,
    open_browser: bool = False,
) -> None:
    """Execute an infrastructure investigation using the InfraAgent.

    Handles the full lifecycle: banner, agent creation, execution,
    result display, and summary — following the same pattern as _execute_code_mode.
    """
    from vaig.agents.infra_agent import InfraAgent
    from vaig.core.exceptions import MaxIterationsError

    console.print(
        Panel.fit(
            "[bold green]🔍 Live Infrastructure Mode[/bold green]\n"
            f"[dim]Cluster: {gke_config.cluster_name or '(kubeconfig default)'}[/dim]\n"
            f"[dim]Namespace: {gke_config.default_namespace} | "
            f"Project: {gke_config.project_id or '(auto-detect)'}[/dim]\n"
            "[dim]All tools are READ-ONLY[/dim]",
            border_style="green",
        )
    )

    agent = InfraAgent(
        client,
        gke_config,
        settings=settings,
        model_id=model_id or client.current_model,
    )

    # Show registered tool count
    tool_count = len(agent.registry.list_tools())
    if tool_count == 0:
        err_console.print(
            "[bold red]No infrastructure tools available![/bold red]\n"
            "[yellow]Install optional dependencies:[/yellow]\n"
            "  pip install vertex-ai-toolkit[live]       # GKE tools\n"
            "  pip install google-cloud-logging google-cloud-monitoring  # GCP observability"
        )
        raise typer.Exit(1)

    console.print(f"[dim]{tool_count} tools loaded[/dim]")

    try:
        console.print("[bold cyan]🤖 Infrastructure agent investigating...[/bold cyan]")
        tool_logger = ToolCallLogger()
        effective_cache = tool_result_cache or ToolResultCache()
        result = agent.execute(
            question,
            context=context,
            on_tool_call=tool_logger,
            tool_call_store=tool_call_store,
            tool_result_cache=effective_cache,
        )
        tool_logger.print_summary()

        # Display final response with severity coloring
        console.print()
        if result.content:
            print_colored_report(result.content, console=console)
        console.print()

        # When --open is requested, use a temp file so we can pass its path to
        # the browser.  _handle_export_output writes the file when output is set;
        # afterwards we open it in the default browser.
        effective_output = output
        if open_browser and format_ and format_.strip().lower() == "html" and output is None:
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                effective_output = Path(tmp.name)

        _handle_export_output(
            response_text=result.content or "",
            question=question,
            model_id=model_id or client.current_model,
            skill_name=skill_name,
            format_=format_,
            output=effective_output,
            tokens=result.usage or None,
            cost=_compute_cost_str(result.usage, model_id or client.current_model),
        )

        if open_browser and format_ and format_.strip().lower() == "html":
            open_target = effective_output or output
            if open_target is not None:
                _open_html_in_browser(open_target, console)

        # Show summary (reuse coding summary — same metadata shape)
        _show_coding_summary(result)

    except MaxIterationsError as exc:
        err_console.print(
            f"\n[bold red]⚠ Max iterations reached ({exc.iterations})[/bold red]\n"
            "[yellow]The infrastructure agent hit its tool-use iteration limit. "
            "Try narrowing the scope of your question or specifying a namespace/resource.[/yellow]"
        )
        raise typer.Exit(1)  # noqa: B904


# ── Async Live Mode Implementations ──────────────────────────


async def _async_execute_live_mode(
    client: GeminiClientProtocol,
    gke_config: GKEConfig,
    question: str,
    context: str,
    *,
    settings: Settings | None = None,
    output: Path | None = None,
    format_: str | None = None,
    skill_name: str | None = None,
    model_id: str | None = None,
    tool_call_store: ToolCallStore | None = None,
    tool_result_cache: ToolResultCache | None = None,
    open_browser: bool = False,
) -> None:
    """Async version of :func:`_execute_live_mode`.

    Uses ``InfraAgent.async_execute()`` for non-blocking tool loops.
    """
    from vaig.agents.infra_agent import InfraAgent
    from vaig.core.exceptions import MaxIterationsError

    console.print(
        Panel.fit(
            "[bold green]🔍 Live Infrastructure Mode (async)[/bold green]\n"
            f"[dim]Cluster: {gke_config.cluster_name or '(kubeconfig default)'}[/dim]\n"
            f"[dim]Namespace: {gke_config.default_namespace} | "
            f"Project: {gke_config.project_id or '(auto-detect)'}[/dim]\n"
            "[dim]All tools are READ-ONLY[/dim]",
            border_style="green",
        )
    )

    agent = InfraAgent(
        client,
        gke_config,
        settings=settings,
        model_id=model_id or client.current_model,
    )

    tool_count = len(agent.registry.list_tools())
    if tool_count == 0:
        err_console.print(
            "[bold red]No infrastructure tools available![/bold red]\n"
            "[yellow]Install optional dependencies:[/yellow]\n"
            "  pip install vertex-ai-toolkit[live]       # GKE tools\n"
            "  pip install google-cloud-logging google-cloud-monitoring  # GCP observability"
        )
        raise typer.Exit(1)

    console.print(f"[dim]{tool_count} tools loaded[/dim]")

    try:
        console.print("[bold cyan]🤖 Infrastructure agent investigating (async)...[/bold cyan]")
        tool_logger = ToolCallLogger()
        effective_cache = tool_result_cache or ToolResultCache()
        result = await agent.async_execute(
            question,
            context=context,
            on_tool_call=tool_logger,
            tool_call_store=tool_call_store,
            tool_result_cache=effective_cache,
        )
        tool_logger.print_summary()

        console.print()
        if result.content:
            print_colored_report(result.content, console=console)
        console.print()

        # When --open is requested, use a temp file so we can pass its path to
        # the browser.  _handle_export_output writes the file when output is set;
        # afterwards we open it in the default browser.
        effective_output = output
        if open_browser and format_ and format_.strip().lower() == "html" and output is None:
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                effective_output = Path(tmp.name)

        _handle_export_output(
            response_text=result.content or "",
            question=question,
            model_id=model_id or client.current_model,
            skill_name=skill_name,
            format_=format_,
            output=effective_output,
            tokens=result.usage or None,
            cost=_compute_cost_str(result.usage, model_id or client.current_model),
        )

        if open_browser and format_ and format_.strip().lower() == "html":
            open_target = effective_output or output
            if open_target is not None:
                _open_html_in_browser(open_target, console)

        _show_coding_summary(result)

    except MaxIterationsError as exc:
        err_console.print(
            f"\n[bold red]⚠ Max iterations reached ({exc.iterations})[/bold red]\n"
            "[yellow]The infrastructure agent hit its tool-use iteration limit. "
            "Try narrowing the scope of your question or specifying a namespace/resource.[/yellow]"
        )
        raise typer.Exit(1)  # noqa: B904


async def _async_execute_orchestrated_skill(
    client: GeminiClientProtocol,
    settings: Settings,
    gke_config: GKEConfig,
    skill: BaseSkill,
    question: str,
    *,
    output: Path | None = None,
    format_: str | None = None,
    model_id: str | None = None,
    tool_call_store: ToolCallStore | None = None,
    summary: bool = False,
    no_bell: bool = False,
    open_browser: bool = False,
) -> None:
    """Async version of :func:`_execute_orchestrated_skill`.

    Uses ``Orchestrator.async_execute_with_tools()`` for non-blocking execution.
    """
    from vaig.agents.orchestrator import Orchestrator
    from vaig.core.exceptions import MaxIterationsError

    skill_meta = skill.get_metadata()

    tool_registry = _register_live_tools(gke_config, settings=settings)

    gke_credentials = None
    if settings is not None:
        from vaig.core.auth import get_gke_credentials as _get_gke_creds

        gke_credentials = _get_gke_creds(settings)

    is_autopilot: bool | None = None
    try:
        from vaig.tools.gke_tools import detect_autopilot

        is_autopilot = detect_autopilot(gke_config, credentials=gke_credentials)
    except ImportError:
        pass

    tool_count = len(tool_registry.list_tools())
    if tool_count == 0:
        context_str = _get_offline_fallback_context(skill_meta)
        prompt = f"{context_str}\n\n## Investigation Question\n\n{question}"
        try:
            result = await client.async_generate(prompt)
        except Exception as exc:  # noqa: BLE001
            handle_cli_error(exc)
            raise typer.Exit(1) from exc
        console.print()
        console.print("[bold]Assistant response (offline mode):[/bold]")
        console.print(result.text)
        return

    console.print(
        Panel.fit(
            f"[bold green]🔍 Orchestrated Skill: {skill_meta.display_name} (async)[/bold green]\n"
            f"[dim]Cluster: {gke_config.cluster_name or '(kubeconfig default)'}[/dim]\n"
            f"[dim]Namespace: {gke_config.default_namespace} | "
            f"Project: {gke_config.project_id or '(auto-detect)'}[/dim]\n"
            f"[dim]{tool_count} tools loaded | Strategy: sequential[/dim]",
            border_style="green",
        )
    )

    orchestrator = Orchestrator(client, settings)

    try:
        console.print(f"[bold cyan]🤖 Running {skill_meta.display_name} pipeline (async)...[/bold cyan]")
        tool_logger = ToolCallLogger()
        progress_display = AgentProgressDisplay(tool_logger)
        try:
            orch_result = await orchestrator.async_execute_with_tools(
                query=question,
                skill=skill,
                tool_registry=tool_registry,
                strategy="sequential",
                is_autopilot=is_autopilot,
                on_tool_call=tool_logger,
                tool_call_store=tool_call_store,
                on_agent_progress=progress_display,
                gke_namespace=gke_config.default_namespace,
                gke_location=gke_config.location,
                gke_cluster_name=gke_config.cluster_name,
            )
        finally:
            progress_display.stop()
        tool_logger.print_summary()

        # Display final response with severity coloring
        console.print()
        if summary and orch_result.structured_report is not None:
            # --summary mode: compact output from the structured report
            console.print(orch_result.structured_report.to_summary())
        else:
            # Rich Panel for executive summary (before the full report)
            if orch_result.structured_report is not None:
                print_executive_summary_panel(
                    orch_result.structured_report, console=console,
                )
            if orch_result.synthesized_output:
                print_colored_report(orch_result.synthesized_output, console=console)
            # Rich Table for recommendations (after the full report)
            if orch_result.structured_report is not None:
                print_recommendations_table(
                    orch_result.structured_report, console=console,
                )
        console.print()

        # Dispatch output format — HTML or text/json/md fallback
        _dispatch_format_output(
            orch_result,
            format_=format_,
            output=output,
            question=question,
            model_id=settings.models.default,
            skill_name=skill_meta.name,
            console=console,
            err_console=err_console,
            gke_config=gke_config,
            open_browser=open_browser,
            tool_logger=tool_logger,
        )

        _show_orchestrated_summary(orch_result, model_id=settings.models.default)

        # Auto-export report if configured.
        # ADR-4: auto-export fires here for the health report only, immediately after the live
        # summary.  Telemetry and tool-calls are higher-volume and use the explicit CLI push
        # commands (vaig cloud push telemetry / tool-calls) so they are never auto-exported.
        _auto_export_report(settings, orch_result, gke_config)

        # Notify via terminal bell
        _emit_bell(no_bell=no_bell)

    except MaxIterationsError as exc:
        err_console.print(
            f"\n[bold red]⚠ Max iterations reached ({exc.iterations})[/bold red]\n"
            "[yellow]The orchestrated skill pipeline hit its tool-use iteration limit. "
            "Try narrowing the scope of your question or specifying a namespace/resource.[/yellow]"
        )
        raise typer.Exit(1)  # noqa: B904

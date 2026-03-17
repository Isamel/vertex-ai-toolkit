"""Live command — infrastructure investigation (GKE/GCP)."""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import typer
from rich.panel import Panel
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
from vaig.cli.display import print_colored_report
from vaig.core.tool_call_store import ToolCallStore

if TYPE_CHECKING:
    from vaig.agents.orchestrator import OrchestratorResult
    from vaig.core.client import GeminiClient
    from vaig.core.config import GKEConfig, Settings
    from vaig.skills.base import BaseSkill
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
    except Exception:
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
    ) -> None:
        """Print a single tool execution line."""
        self.tool_count += 1
        self.total_duration += duration

        args_str = _truncate_args(tool_args)

        if success:
            status = "[green]✓[/green]"
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

        console.print(
            f"  🔧 [cyan]{tool_name}[/cyan]"
            f"({args_str}) "
            f"{status} [dim]({duration:.1f}s)[/dim]"
        )

    def print_summary(self) -> None:
        """Print the final pipeline summary line."""
        total_wall = time.perf_counter() - self._pipeline_start
        status = "[green]Pipeline complete[/green]" if self.errors == 0 else "[yellow]Pipeline complete with errors[/yellow]"

        # Build failure detail with grouped reasons
        fail_detail = ""
        if self.errors:
            reason_counts = Counter(self._error_reasons)
            if reason_counts:
                grouped = ", ".join(
                    f"{count}\u00d7 {reason}" for reason, count in reason_counts.most_common()
                )
                fail_detail = f", {self.errors} failed ({grouped})"
            else:
                fail_detail = f", {self.errors} failed"

        console.print(
            f"\n{status} "
            f"[dim]({total_wall:.1f}s total, "
            f"{self.tool_count} tool{'s' if self.tool_count != 1 else ''} executed"
            f"{fail_detail})[/dim]"
        )


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
        auto_skill: Annotated[bool, typer.Option("--auto-skill", help="Auto-detect the best skill based on query")] = False,
        cluster: Annotated[str | None, typer.Option("--cluster", help="GKE cluster name (overrides config)")] = None,
        namespace: Annotated[
            str | None, typer.Option("--namespace", help="Default Kubernetes namespace (overrides config)", autocompletion=complete_namespace)
        ] = None,
        project: Annotated[
            str | None, typer.Option("--project", "-p", help="GCP project ID (overrides gcp.project_id and gke.project_id)")
        ] = None,
        project_id: Annotated[
            str | None, typer.Option("--project-id", help="GCP project ID (overrides config, alias for --project)")
        ] = None,
        location: Annotated[
            str | None, typer.Option("--location", help="GCP location (overrides config)")
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
            err_console.print(
                f"[red]--watch interval must be >= {MINIMUM_WATCH_INTERVAL}s "
                f"(got {watch}s)[/red]"
            )
            raise typer.Exit(1)

        try:  # ── CLI error boundary ──
            settings = _helpers._get_settings(config)

            # Eagerly initialize the telemetry collector and wire the
            # TelemetrySubscriber so events are forwarded to the SQLite store.
            _helpers._init_telemetry(settings)

            # Apply --project / --project-id: mutate gcp.project_id AND gke.project_id
            effective_project = project or project_id
            if effective_project:
                settings.gcp.project_id = effective_project
                settings.gke.project_id = effective_project

            # Apply --location: mutate gcp.location before component creation
            if location:
                settings.gcp.location = location

            if model:
                settings.models.default = model

            from vaig.core.client import GeminiClient
            from vaig.core.container import build_container

            container = build_container(settings)
            client = cast(GeminiClient, container.gemini_client)
            gke_config = _build_gke_config(
                settings, cluster=cluster, namespace=namespace, project_id=effective_project, location=location,
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
                            f"[dim]🎯 Auto-routing to skill: [cyan]{skill}[/cyan] "
                            f"(score: {best_score:.1f})[/dim]"
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
        except Exception as exc:
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
                f"[bold cyan]Watch iteration #{iteration} — "
                f"{datetime.now().strftime('%H:%M:%S')}[/bold cyan]"
            )
            console.print(f"{'=' * 60}\n")

            try:
                run_fn()
            except (SystemExit, typer.Exit):
                # Agent errors (MaxIterationsError etc.) raise typer.Exit —
                # in watch mode we log and continue to the next iteration.
                console.print(
                    f"[yellow]Iteration #{iteration} exited with error — "
                    f"continuing watch...[/yellow]"
                )

            console.print(
                f"\n[dim]Next refresh in {interval}s (Ctrl+C to stop)...[/dim]"
            )
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
            f"[bold yellow]Dry Run — vaig live[/bold yellow] [dim]\"{question}\"[/dim]",
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
    console.print(
        "[bold]Estimated cost:[/bold] [dim]depends on tool usage "
        "(typically $0.02-0.10 per run)[/dim]"
    )
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
        except Exception:
            logger.warning(
                "Failed to load plugin tools for live mode. Skipping.",
                exc_info=True,
            )

    return registry


def _execute_orchestrated_skill(
    client: GeminiClient,
    settings: Settings,
    gke_config: GKEConfig,
    skill: BaseSkill,
    question: str,
    *,
    output: Path | None = None,
    format_: str | None = None,
    model_id: str | None = None,
    tool_call_store: ToolCallStore | None = None,
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
        err_console.print(
            "[bold red]No infrastructure tools available![/bold red]\n"
            "[yellow]Install optional dependencies:[/yellow]\n"
            "  pip install vertex-ai-toolkit[live]       # GKE tools\n"
            "  pip install google-cloud-logging google-cloud-monitoring  # GCP observability"
        )
        raise typer.Exit(1)

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
        orch_result = orchestrator.execute_with_tools(
            query=question,
            skill=skill,
            tool_registry=tool_registry,
            strategy="sequential",
            is_autopilot=is_autopilot,
            on_tool_call=tool_logger,
            tool_call_store=tool_call_store,
        )
        tool_logger.print_summary()

        # Display final response with severity coloring
        console.print()
        if orch_result.synthesized_output:
            print_colored_report(orch_result.synthesized_output, console=console)
        console.print()

        _handle_export_output(
            response_text=orch_result.synthesized_output or "",
            question=question,
            model_id=settings.models.default,
            skill_name=skill_meta.name,
            format_=format_,
            output=output,
            tokens=orch_result.total_usage or None,
            cost=_compute_cost_str(orch_result.total_usage, settings.models.default),
        )

        # Show agent pipeline summary (includes cost line)
        _show_orchestrated_summary(orch_result, model_id=settings.models.default)

    except MaxIterationsError as exc:
        err_console.print(
            f"\n[bold red]⚠ Max iterations reached ({exc.iterations})[/bold red]\n"
            "[yellow]The orchestrated skill pipeline hit its tool-use iteration limit. "
            "Try narrowing the scope of your question or specifying a namespace/resource.[/yellow]"
        )
        raise typer.Exit(1)  # noqa: B904


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

    # Show cost summary for the full pipeline
    _show_cost_line(orch_result.total_usage or None, model_id)

    if not orch_result.success:
        err_console.print("[bold red]⚠ Pipeline completed with errors[/bold red]")


def _execute_live_mode(
    client: GeminiClient,
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
        result = agent.execute(question, context=context, on_tool_call=tool_logger, tool_call_store=tool_call_store)
        tool_logger.print_summary()

        # Display final response with severity coloring
        console.print()
        if result.content:
            print_colored_report(result.content, console=console)
        console.print()

        _handle_export_output(
            response_text=result.content or "",
            question=question,
            model_id=model_id or client.current_model,
            skill_name=skill_name,
            format_=format_,
            output=output,
            tokens=result.usage or None,
            cost=_compute_cost_str(result.usage, model_id or client.current_model),
        )

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
    client: GeminiClient,
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
        result = await agent.async_execute(question, context=context, on_tool_call=tool_logger, tool_call_store=tool_call_store)
        tool_logger.print_summary()

        console.print()
        if result.content:
            print_colored_report(result.content, console=console)
        console.print()

        _handle_export_output(
            response_text=result.content or "",
            question=question,
            model_id=model_id or client.current_model,
            skill_name=skill_name,
            format_=format_,
            output=output,
            tokens=result.usage or None,
            cost=_compute_cost_str(result.usage, model_id or client.current_model),
        )

        _show_coding_summary(result)

    except MaxIterationsError as exc:
        err_console.print(
            f"\n[bold red]⚠ Max iterations reached ({exc.iterations})[/bold red]\n"
            "[yellow]The infrastructure agent hit its tool-use iteration limit. "
            "Try narrowing the scope of your question or specifying a namespace/resource.[/yellow]"
        )
        raise typer.Exit(1)  # noqa: B904


async def _async_execute_orchestrated_skill(
    client: GeminiClient,
    settings: Settings,
    gke_config: GKEConfig,
    skill: BaseSkill,
    question: str,
    *,
    output: Path | None = None,
    format_: str | None = None,
    model_id: str | None = None,
    tool_call_store: ToolCallStore | None = None,
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
        err_console.print(
            "[bold red]No infrastructure tools available![/bold red]\n"
            "[yellow]Install optional dependencies:[/yellow]\n"
            "  pip install vertex-ai-toolkit[live]       # GKE tools\n"
            "  pip install google-cloud-logging google-cloud-monitoring  # GCP observability"
        )
        raise typer.Exit(1)

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
        orch_result = await orchestrator.async_execute_with_tools(
            query=question,
            skill=skill,
            tool_registry=tool_registry,
            strategy="sequential",
            is_autopilot=is_autopilot,
            on_tool_call=tool_logger,
            tool_call_store=tool_call_store,
        )
        tool_logger.print_summary()

        console.print()
        if orch_result.synthesized_output:
            print_colored_report(orch_result.synthesized_output, console=console)
        console.print()

        _handle_export_output(
            response_text=orch_result.synthesized_output or "",
            question=question,
            model_id=settings.models.default,
            skill_name=skill_meta.name,
            format_=format_,
            output=output,
            tokens=orch_result.total_usage or None,
            cost=_compute_cost_str(orch_result.total_usage, settings.models.default),
        )

        _show_orchestrated_summary(orch_result, model_id=settings.models.default)

    except MaxIterationsError as exc:
        err_console.print(
            f"\n[bold red]⚠ Max iterations reached ({exc.iterations})[/bold red]\n"
            "[yellow]The orchestrated skill pipeline hit its tool-use iteration limit. "
            "Try narrowing the scope of your question or specifying a namespace/resource.[/yellow]"
        )
        raise typer.Exit(1)  # noqa: B904

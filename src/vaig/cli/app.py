"""CLI app — Typer commands for VAIG toolkit."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from vaig import __version__

if TYPE_CHECKING:
    from vaig.agents.base import AgentResult
    from vaig.agents.orchestrator import OrchestratorResult
    from vaig.cli.export import ExportPayload
    from vaig.core.client import GeminiClient
    from vaig.core.config import GKEConfig, Settings
    from vaig.skills.base import BaseSkill
    from vaig.tools.base import ToolRegistry

console = Console()
err_console = Console(stderr=True)
logger = logging.getLogger(__name__)

# ── Main App ──────────────────────────────────────────────────
app = typer.Typer(
    name="vaig",
    help="🤖 Vertex AI Gemini Toolkit — Multi-agent AI assistant powered by Vertex AI",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# ── Sub-commands ──────────────────────────────────────────────
sessions_app = typer.Typer(help="Manage chat sessions")
models_app = typer.Typer(help="Manage AI models")
skills_app = typer.Typer(help="Manage skills")
mcp_app = typer.Typer(help="Manage MCP (Model Context Protocol) servers")

app.add_typer(sessions_app, name="sessions")
app.add_typer(models_app, name="models")
app.add_typer(skills_app, name="skills")
app.add_typer(mcp_app, name="mcp")


# ── Shared State ──────────────────────────────────────────────
def _get_settings(config: str | None = None) -> "Settings":
    """Load settings (lazy import to avoid heavy imports on --help)."""
    from vaig.core.config import get_settings

    return get_settings(config)


def _banner() -> None:
    """Print the VAIG banner."""
    console.print(
        Panel.fit(
            f"[bold cyan]VAIG[/bold cyan] — Vertex AI Gemini Toolkit v{__version__}\n"
            "[dim]Multi-agent AI assistant powered by Google Gemini[/dim]",
            border_style="bright_blue",
        )
    )


# ── Version ───────────────────────────────────────────────────
def _version_callback(value: bool) -> None:
    if value:
        console.print(f"vaig v{__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option("--version", "-v", help="Show version", callback=_version_callback, is_eager=True),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-V", help="Enable verbose logging (INFO level)"),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option("--debug", "-d", help="Enable debug logging (DEBUG level, shows paths and full tracebacks)"),
    ] = False,
    log_level: Annotated[
        Optional[str],
        typer.Option("--log-level", help="Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"),
    ] = None,
) -> None:
    """VAIG — Vertex AI Gemini Toolkit."""
    from vaig.core.log import setup_logging

    # Priority: --log-level > --debug > --verbose > default (WARNING)
    if log_level:
        level = log_level.upper()
        show_path = level == "DEBUG"
    elif debug:
        level = "DEBUG"
        show_path = True
    elif verbose:
        level = "INFO"
        show_path = False
    else:
        level = "WARNING"
        show_path = False

    setup_logging(level, show_path=show_path)


def _apply_subcommand_log_flags(*, verbose: bool, debug: bool) -> None:
    """Apply --verbose / --debug flags from a subcommand.

    Subcommand-level flags override the global callback flags so that
    ``vaig live "query" -d`` works the same as ``vaig -d live "query"``.
    Only overrides if a flag is actually set (non-default).
    """
    if not verbose and not debug:
        return

    from vaig.core.log import reset_logging, setup_logging

    # Reset the idempotent guard so we can reconfigure with the subcommand flags.
    reset_logging()

    if debug:
        setup_logging("DEBUG", show_path=True)
    elif verbose:
        setup_logging("INFO", show_path=False)


# ══════════════════════════════════════════════════════════════
# CHAT — Interactive REPL
# ══════════════════════════════════════════════════════════════
@app.command()
def chat(
    config: Annotated[Optional[str], typer.Option("--config", "-c", help="Path to config YAML")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Model to use")] = None,
    skill: Annotated[Optional[str], typer.Option("--skill", "-s", help="Activate a skill")] = None,
    session: Annotated[Optional[str], typer.Option("--session", help="Load an existing session by ID")] = None,
    resume: Annotated[bool, typer.Option("--resume", "-r", help="Resume the last session")] = False,
    name: Annotated[str, typer.Option("--name", "-n", help="Name for new session")] = "chat",
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w", help="Workspace root directory for /code mode"),
    ] = None,
) -> None:
    """Start an interactive chat session (REPL mode).

    Examples:
        vaig chat
        vaig chat --model gemini-2.5-flash
        vaig chat --skill rca
        vaig chat --session abc123
        vaig chat --resume
        vaig chat -w ./my-project
    """
    from vaig.cli.repl import start_repl

    _banner()
    settings = _get_settings(config)

    if model:
        settings.models.default = model

    if workspace:
        resolved_ws = workspace.resolve()
        if not resolved_ws.is_dir():
            err_console.print(f"[red]Workspace directory not found: {resolved_ws}[/red]")
            raise typer.Exit(1)
        settings.coding.workspace_root = str(resolved_ws)

    # Resolve --resume to a session ID
    resume_session_id = session
    if resume and not session:
        from vaig.session.manager import SessionManager

        mgr = SessionManager(settings)
        last = mgr.get_last_session()
        mgr.close()
        if last:
            resume_session_id = last["id"]
            console.print(f"[dim]Resuming last session: {last['name']} ({last['id'][:12]})[/dim]")
        else:
            console.print("[yellow]No previous sessions found. Starting new session.[/yellow]")

    start_repl(
        settings=settings,
        skill_name=skill,
        session_id=resume_session_id,
        session_name=name,
    )


# ══════════════════════════════════════════════════════════════
# ASK — Single-shot question
# ══════════════════════════════════════════════════════════════
@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question or prompt to send")],
    config: Annotated[Optional[str], typer.Option("--config", "-c", help="Path to config YAML")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Model to use")] = None,
    files: Annotated[Optional[list[Path]], typer.Option("--file", "-f", help="Files to include as context")] = None,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Save response to a file")] = None,
    format_: Annotated[Optional[str], typer.Option("--format", help="Export format: json, md, html")] = None,
    skill: Annotated[Optional[str], typer.Option("--skill", "-s", help="Use a specific skill")] = None,
    auto_skill: Annotated[bool, typer.Option("--auto-skill", help="Auto-detect the best skill based on query")] = False,
    no_stream: Annotated[bool, typer.Option("--no-stream", help="Disable streaming output")] = False,
    code: Annotated[bool, typer.Option("--code", help="Enable coding agent mode (read/write/edit files)")] = False,
    live: Annotated[bool, typer.Option("--live", help="Enable live infrastructure tools (GKE/GCP)")] = False,
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w", help="Workspace root directory for code mode"),
    ] = None,
    cluster: Annotated[Optional[str], typer.Option("--cluster", help="GKE cluster name (overrides config)")] = None,
    namespace: Annotated[
        Optional[str], typer.Option("--namespace", help="Default Kubernetes namespace (overrides config)")
    ] = None,
    project_id: Annotated[
        Optional[str], typer.Option("--project-id", help="GCP project ID (overrides config)")
    ] = None,
    location: Annotated[
        Optional[str], typer.Option("--location", help="GKE cluster location/zone/region (overrides config)")
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-V", help="Enable verbose logging (INFO level)"),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option("--debug", "-d", help="Enable debug logging (DEBUG level, shows paths and full tracebacks)"),
    ] = False,
) -> None:
    """Ask a single question and get a response.

    Examples:
        vaig ask "What is Kubernetes?"
        vaig ask "Analyze this code" -f main.py
        vaig ask "Analyze this" -f logs.csv -o analysis.md
        vaig ask "Investigate this incident" -s rca -f logs.txt
        vaig ask "Add error handling to app.py" --code
        vaig ask "Fix the bug in utils.py" --code -w ./my-project
        vaig ask "Analyze pod crashes" --live -s log-analysis
        vaig ask "Check OOM kills in prod" --live --namespace=production
        vaig ask "Explain this code" -f main.py --format json -o report.json
    """
    _apply_subcommand_log_flags(verbose=verbose, debug=debug)

    settings = _get_settings(config)

    if model:
        settings.models.default = model

    from vaig.agents.orchestrator import Orchestrator
    from vaig.context.builder import ContextBuilder
    from vaig.core.client import GeminiClient
    from vaig.skills.base import SkillPhase
    from vaig.skills.registry import SkillRegistry

    client = GeminiClient(settings)
    orchestrator = Orchestrator(client, settings)

    # Build context from files
    context_str = ""
    if files:
        builder = ContextBuilder(settings)
        for f in files:
            try:
                builder.add_file(f)
            except FileNotFoundError:
                err_console.print(f"[red]File not found: {f}[/red]")
                raise typer.Exit(1)  # noqa: B904
        builder.show_summary()
        context_str = builder.bundle.to_context_string()

    # Code mode — use CodingAgent (Tasks 5.1, 5.4, 5.5, 5.6, 5.7)
    if code:
        if workspace:
            resolved_ws = workspace.resolve()
            if not resolved_ws.is_dir():
                err_console.print(f"[red]Workspace directory not found: {resolved_ws}[/red]")
                raise typer.Exit(1)
            settings.coding.workspace_root = str(resolved_ws)
        _execute_code_mode(client, settings, question, context_str, output=output)
        return

    # Live infrastructure mode — use InfraAgent
    if live:
        gke_config = _build_gke_config(
            settings, cluster=cluster, namespace=namespace, project_id=project_id, location=location,
        )
        _execute_live_mode(
            client,
            gke_config,
            question,
            context_str,
            settings=settings,
            output=output,
            model_id=model,
        )
        return

    # ── Chunked file analysis ─────────────────────────────────
    # If the context is large enough to exceed the model's context window,
    # use ChunkedProcessor (Map-Reduce) instead of the normal pipeline.
    if context_str and _try_chunked_ask(client, settings, question, context_str, model_id=model, output=output):
        return

    # Execute with or without skill
    context_file_paths = [str(f) for f in files] if files else []

    # Auto-detect skill if requested and no explicit skill specified
    if auto_skill and not skill:
        registry = SkillRegistry(settings)
        suggestions = registry.suggest_skill(question)
        if suggestions:
            best_name, best_score = suggestions[0]
            if best_score >= 1.0:  # Only auto-select if score is meaningful
                skill = best_name
                console.print(f"[dim]Auto-selected skill: [cyan]{skill}[/cyan] (score: {best_score:.1f})[/dim]")
            else:
                console.print(
                    f"[dim]Suggested skills: {', '.join(f'{n} ({s:.1f})' for n, s in suggestions)}[/dim]"
                )

    if skill:
        registry = SkillRegistry(settings)
        active_skill = registry.get(skill)
        if not active_skill:
            err_console.print(f"[red]Skill not found: {skill}[/red]")
            err_console.print(f"[dim]Available: {', '.join(registry.list_names())}[/dim]")
            raise typer.Exit(1)

        with console.status(
            f"[bold cyan]Running {skill} skill on {settings.models.default}...[/bold cyan]"
        ):
            result = orchestrator.execute_skill_phase(
                active_skill,
                SkillPhase.ANALYZE,
                context_str,
                question,
            )
        console.print()
        if result.output:
            console.print(Markdown(result.output))

        # Show cost summary for skill execution
        skill_usage = (result.metadata or {}).get("total_usage")
        _show_cost_line(skill_usage, settings.models.default)

        _handle_export_output(
            response_text=result.output,
            question=question,
            model_id=settings.models.default,
            skill_name=skill,
            context_files=context_file_paths,
            format_=format_,
            output=output,
            tokens=skill_usage,
            cost=_compute_cost_str(skill_usage, settings.models.default),
        )
    else:
        # Direct chat — single agent
        if no_stream:
            with console.status(
                f"[bold cyan]Generating response with {settings.models.default}...[/bold cyan]"
            ):
                result = orchestrator.execute_single(question, context=context_str)
            console.print()
            if hasattr(result, "content") and result.content:
                console.print(Markdown(result.content))  # type: ignore[union-attr]

                # Show cost summary
                result_usage = getattr(result, "usage", None)
                _show_cost_line(result_usage, settings.models.default)

                _handle_export_output(
                    response_text=result.content,  # type: ignore[union-attr]
                    question=question,
                    model_id=settings.models.default,
                    skill_name=skill,
                    context_files=context_file_paths,
                    format_=format_,
                    output=output,
                    tokens=result_usage,
                    cost=_compute_cost_str(result_usage, settings.models.default),
                )
        else:
            stream = orchestrator.execute_single(question, context=context_str, stream=True)
            console.print()
            response_parts: list[str] = []
            for chunk in stream:  # type: ignore[union-attr]
                console.print(chunk, end="")
                response_parts.append(chunk)
            console.print()

            # Show cost summary (usage available after stream exhaustion)
            stream_usage = getattr(stream, "usage", None)
            _show_cost_line(stream_usage, settings.models.default)

            _handle_export_output(
                response_text="".join(response_parts),
                question=question,
                model_id=settings.models.default,
                skill_name=skill,
                context_files=context_file_paths,
                format_=format_,
                output=output,
                tokens=stream_usage,
                cost=_compute_cost_str(stream_usage, settings.models.default),
            )


def _save_output(output: Path, content: str) -> None:
    """Write response content to a file and print confirmation."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    console.print(f"[green]✓ Response saved to {output}[/green]")


def _build_cost_markdown_section(
    tokens: dict[str, int],
    model_id: str,
    cost: str | None = None,
) -> str | None:
    """Build a ``## Cost & Usage Summary`` markdown section for plain file saves.

    Returns ``None`` if no meaningful token data is available.
    """
    prompt_tokens = tokens.get("prompt_tokens", 0)
    completion_tokens = tokens.get("completion_tokens", 0)
    thinking_tokens = tokens.get("thinking_tokens", 0)
    total_tokens = tokens.get("total_tokens", 0)

    if total_tokens == 0 and prompt_tokens == 0 and completion_tokens == 0:
        return None

    lines = [
        "---",
        "## Cost & Usage Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Input tokens | {prompt_tokens:,} |",
        f"| Output tokens | {completion_tokens:,} |",
    ]

    if thinking_tokens:
        lines.append(f"| Thinking tokens | {thinking_tokens:,} |")

    lines.append(f"| Total tokens | {total_tokens:,} |")
    lines.append(f"| Model | {model_id} |")

    if cost:
        lines.append(f"| Estimated cost | {cost} |")

    lines.append("")
    return "\n".join(lines)


def _format_session_date(iso_str: str) -> str:
    """Format an ISO timestamp to a human-friendly short form."""
    if not iso_str:
        return "—"
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return iso_str[:16] if len(iso_str) > 16 else iso_str


def _resolve_session_id(manager: object, prefix: str) -> str:
    """Resolve a session ID prefix to a full session ID.

    Supports both full UUIDs and short prefixes (e.g. first 8 chars).
    """
    from vaig.session.manager import SessionManager

    if not isinstance(manager, SessionManager):
        return prefix

    # Try exact match first
    session = manager._store.get_session(prefix)
    if session:
        return prefix

    # Try prefix match
    sessions = manager.list_sessions(limit=100)
    matches = [s for s in sessions if s["id"].startswith(prefix)]
    if len(matches) == 1:
        return matches[0]["id"]
    if len(matches) > 1:
        err_console.print(f"[yellow]Ambiguous prefix '{prefix}' matches {len(matches)} sessions. Use more characters.[/yellow]")
    return prefix


def _build_export_payload(
    *,
    question: str,
    response_text: str,
    model_id: str,
    skill_name: str | None = None,
    tokens: dict[str, int] | None = None,
    cost: str | None = None,
    context_files: list[str] | None = None,
    agent_results: list[dict] | None = None,
) -> "ExportPayload":
    """Build an ExportPayload from ask/live command results."""
    from datetime import datetime, timezone

    from vaig.cli.export import ExportMetadata, ExportPayload

    meta = ExportMetadata(
        model=model_id,
        skill=skill_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        tokens=tokens or {},
        cost=cost,
        vaig_version=__version__,
    )
    return ExportPayload(
        question=question,
        response=response_text,
        metadata=meta,
        context_files=context_files or [],
        agent_results=agent_results or [],
    )


def _handle_export_output(
    *,
    response_text: str,
    question: str,
    model_id: str,
    skill_name: str | None = None,
    context_files: list[str] | None = None,
    format_: str | None = None,
    output: Path | None = None,
    tokens: dict[str, int] | None = None,
    cost: str | None = None,
) -> None:
    """Handle exporting/saving output based on --format and --output flags.

    When --format is set, builds an ExportPayload and formats it.
    When --output is set (with or without --format), writes to file.
    When --format is set without --output, prints formatted content to stdout.
    When neither is set, does nothing (display already handled by caller).
    """
    if not format_ and not output:
        return

    if format_:
        from vaig.cli.export import format_export

        payload = _build_export_payload(
            question=question,
            response_text=response_text,
            model_id=model_id,
            skill_name=skill_name,
            context_files=context_files,
            tokens=tokens,
            cost=cost,
        )
        content = format_export(payload, format_)
        if output:
            _save_output(output, content)
        else:
            console.print(content)
    elif output:
        # Plain file save — append cost summary section if data is available
        save_content = response_text
        if isinstance(tokens, dict) and tokens:
            cost_section = _build_cost_markdown_section(tokens, model_id, cost)
            if cost_section:
                save_content = f"{response_text}\n\n{cost_section}"
        _save_output(output, save_content)


# ══════════════════════════════════════════════════════════════
# LIVE — Infrastructure investigation (GKE/GCP)
# ══════════════════════════════════════════════════════════════
@app.command()
def live(
    question: Annotated[str, typer.Argument(help="Infrastructure question or investigation task")],
    config: Annotated[Optional[str], typer.Option("--config", "-c", help="Path to config YAML")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Model to use")] = None,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Save response to a file")] = None,
    format_: Annotated[Optional[str], typer.Option("--format", help="Export format: json, md, html")] = None,
    skill: Annotated[Optional[str], typer.Option("--skill", "-s", help="SRE skill to apply")] = None,
    auto_skill: Annotated[bool, typer.Option("--auto-skill", help="Auto-detect the best skill based on query")] = False,
    cluster: Annotated[Optional[str], typer.Option("--cluster", help="GKE cluster name (overrides config)")] = None,
    namespace: Annotated[
        Optional[str], typer.Option("--namespace", help="Default Kubernetes namespace (overrides config)")
    ] = None,
    project_id: Annotated[
        Optional[str], typer.Option("--project-id", help="GCP project ID (overrides config)")
    ] = None,
    location: Annotated[
        Optional[str], typer.Option("--location", help="GKE cluster location/zone/region (overrides config)")
    ] = None,
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
    """
    _apply_subcommand_log_flags(verbose=verbose, debug=debug)

    settings = _get_settings(config)

    if model:
        settings.models.default = model

    from vaig.core.client import GeminiClient

    client = GeminiClient(settings)
    gke_config = _build_gke_config(
        settings, cluster=cluster, namespace=namespace, project_id=project_id, location=location,
    )

    # Auto-detect skill if requested and no explicit skill specified
    if auto_skill and not skill:
        from vaig.skills.registry import SkillRegistry

        registry = SkillRegistry(settings)
        suggestions = registry.suggest_skill(question)
        if suggestions:
            best_name, best_score = suggestions[0]
            if best_score >= 1.0:  # Only auto-select if score is meaningful
                skill = best_name
                console.print(f"[dim]Auto-selected skill: [cyan]{skill}[/cyan] (score: {best_score:.1f})[/dim]")
            else:
                console.print(
                    f"[dim]Suggested skills: {', '.join(f'{n} ({s:.1f})' for n, s in suggestions)}[/dim]"
                )

    # If a skill is specified, check whether it needs the full orchestrated
    # pipeline (requires_live_tools=True) or the simple context-prepend approach.
    context_str = ""
    if skill:
        from vaig.skills.registry import SkillRegistry

        registry = SkillRegistry(settings)
        active_skill = registry.get(skill)
        if not active_skill:
            err_console.print(f"[red]Skill not found: {skill}[/red]")
            err_console.print(f"[dim]Available: {', '.join(registry.list_names())}[/dim]")
            raise typer.Exit(1)

        skill_meta = active_skill.get_metadata()

        if skill_meta.requires_live_tools:
            # ── Orchestrated tool-aware pipeline ──────────────
            _execute_orchestrated_skill(
                client,
                settings,
                gke_config,
                active_skill,
                question,
                output=output,
                format_=format_,
                model_id=model,
            )
            return

        # ── Legacy context-prepend path (requires_live_tools=False) ──
        context_str = (
            f"## Active Skill: {skill_meta.display_name}\n\n"
            f"{skill_meta.description}\n\n"
            f"Apply the {skill_meta.name} analysis methodology to the investigation below."
        )

    _execute_live_mode(
        client,
        gke_config,
        question,
        context_str,
        settings=settings,
        output=output,
        format_=format_,
        skill_name=skill,
        model_id=model,
    )


# ── Live mode helpers ─────────────────────────────────────────


def _build_gke_config(
    settings: "Settings",
    *,
    cluster: str | None = None,
    namespace: str | None = None,
    project_id: str | None = None,
    location: str | None = None,
) -> "GKEConfig":
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
    )


def _register_live_tools(gke_config: "GKEConfig", settings: "Settings | None" = None) -> "ToolRegistry":
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
    client: "GeminiClient",
    settings: "Settings",
    gke_config: "GKEConfig",
    skill: "BaseSkill",
    question: str,
    *,
    output: Path | None = None,
    format_: str | None = None,
    model_id: str | None = None,
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
        orch_result = orchestrator.execute_with_tools(
            query=question,
            skill=skill,
            tool_registry=tool_registry,
            strategy="sequential",
            is_autopilot=is_autopilot,
        )

        # Display final response
        console.print()
        if orch_result.synthesized_output:
            console.print(Markdown(orch_result.synthesized_output))
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


def _show_orchestrated_summary(orch_result: "OrchestratorResult", *, model_id: str = "") -> None:
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
    client: "GeminiClient",
    gke_config: "GKEConfig",
    question: str,
    context: str,
    *,
    settings: "Settings | None" = None,
    output: Path | None = None,
    format_: str | None = None,
    skill_name: str | None = None,
    model_id: str | None = None,
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
        result = agent.execute(question, context=context)

        # Display final response
        console.print()
        if result.content:
            console.print(Markdown(result.content))
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


def _execute_code_mode(
    client: "GeminiClient",
    settings: "Settings",
    question: str,
    context: str,
    *,
    output: Path | None = None,
) -> None:
    """Execute a coding task using the CodingAgent.

    Handles confirmation prompts, tool execution feedback,
    iteration/usage summary, and MaxIterationsError.
    """
    from vaig.agents.coding import CodingAgent
    from vaig.core.exceptions import MaxIterationsError

    coding_config = settings.coding

    console.print(
        Panel.fit(
            "[bold yellow]🔧 Code Mode[/bold yellow]\n"
            f"[dim]Workspace: {Path(coding_config.workspace_root).resolve()}[/dim]\n"
            f"[dim]Model: {settings.models.default} | Max iterations: {coding_config.max_tool_iterations}[/dim]",
            border_style="yellow",
        )
    )

    agent = CodingAgent(
        client,
        coding_config,
        settings=settings,
        confirm_fn=_cli_confirm,
        model_id=settings.models.default,
    )

    try:
        # NOTE: No spinner wrapper here — confirm_fn needs interactive terminal access.
        # A spinner would swallow confirmation prompts and freeze the terminal.
        console.print("[bold cyan]🤖 Coding agent working...[/bold cyan]")
        result = agent.execute(question, context=context)

        # Display final response
        console.print()
        if result.content:
            console.print(Markdown(result.content))
        console.print()

        if output:
            # Append cost summary to saved file
            save_content = result.content
            result_usage = result.usage or None
            if result_usage:
                cost_section = _build_cost_markdown_section(
                    result_usage, settings.models.default,
                    _compute_cost_str(result_usage, settings.models.default),
                )
                if cost_section:
                    save_content = f"{result.content}\n\n{cost_section}"
            _save_output(output, save_content)

        # Task 5.5 + 5.6: Show tool execution feedback and usage summary
        _show_coding_summary(result)

    except MaxIterationsError as exc:
        # Task 5.7: Graceful handling
        err_console.print(
            f"\n[bold red]⚠ Max iterations reached ({exc.iterations})[/bold red]\n"
            "[yellow]The coding agent hit its iteration limit. "
            "This usually means the task is too complex for a single turn.\n"
            "Try breaking it into smaller steps.[/yellow]"
        )
        raise typer.Exit(1)  # noqa: B904


def _cli_confirm(tool_name: str, args: dict) -> bool:
    """Confirm destructive tool operations — delegates to shared display helper."""
    from vaig.cli.display import confirm_tool_operation

    return confirm_tool_operation(tool_name, args, console=console)


def _show_coding_summary(result: "AgentResult") -> None:
    """Display tool execution feedback — delegates to shared display helper."""
    from vaig.cli.display import show_tool_execution_summary

    show_tool_execution_summary(result, console=console)


def _show_cost_line(usage: dict[str, int] | None, model_id: str) -> None:
    """Show a compact cost/token summary line — delegates to display helper."""
    if not isinstance(usage, dict):
        return
    from vaig.cli.display import show_cost_summary

    show_cost_summary(usage, model_id, console=console)


def _compute_cost_str(usage: dict[str, int] | None, model_id: str) -> str | None:
    """Compute a formatted cost string from usage metadata.

    Returns ``None`` if usage is empty or cost cannot be calculated.
    """
    if not isinstance(usage, dict) or not usage:
        return None
    from vaig.core.pricing import calculate_cost, format_cost

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    thinking_tokens = usage.get("thinking_tokens", 0)
    cost = calculate_cost(model_id, prompt_tokens, completion_tokens, thinking_tokens)
    return format_cost(cost)


# ── Chunked file analysis helper ──────────────────────────────


def _try_chunked_ask(
    client: "GeminiClient",
    settings: "Settings",
    question: str,
    context_str: str,
    *,
    model_id: str | None = None,
    output: Path | None = None,
) -> bool:
    """Attempt chunked (Map-Reduce) processing if content exceeds the context window.

    Returns True if chunking was performed (caller should return early),
    False if content fits normally (caller should continue with the standard pipeline).
    """
    from vaig.agents.chunked import ChunkedProcessor
    from vaig.agents.orchestrator import Orchestrator

    processor = ChunkedProcessor(client, settings)
    orchestrator = Orchestrator(client, settings)

    # Use the same system instruction the normal pipeline would use
    system_instruction = orchestrator.default_system_instruction()

    try:
        budget = processor.calculate_budget(
            system_instruction,
            question,
            model_id=model_id or settings.models.default,
        )
    except Exception as exc:
        err_console.print(f"[dim]Chunking budget calculation failed ({exc}), using normal pipeline[/dim]")
        return False

    if not processor.needs_chunking(context_str, budget):
        return False

    # ── Content exceeds context window — use Map-Reduce ───────
    chunks = processor.split_into_chunks(context_str, budget)
    total = len(chunks)

    console.print(
        f"\n[bold yellow]Large file detected[/bold yellow] — "
        f"splitting into [cyan]{total}[/cyan] chunks for analysis"
    )

    with console.status("[bold cyan]Analyzing chunks...[/bold cyan]") as status:

        def _on_progress(current: int, total: int) -> None:
            status.update(f"[bold cyan]Processing chunk {current}/{total}...[/bold cyan]")

        result = processor.process_ask(
            context_str,
            question,
            system_instruction,
            budget,
            model_id=model_id or settings.models.default,
            on_progress=_on_progress,
        )

    console.print()
    if result:
        console.print(Markdown(result))
    if output:
        _save_output(output, result)
    return True


# ══════════════════════════════════════════════════════════════
# EXPORT — Re-export a past session
# ══════════════════════════════════════════════════════════════
@app.command(name="export")
def export_session(
    session_id: Annotated[str, typer.Argument(help="Session ID to export")],
    format_: Annotated[str, typer.Option("--format", "-f", help="Export format: json, md, html")] = "md",
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Save to file (default: stdout)")] = None,
    config: Annotated[Optional[str], typer.Option("--config", "-c", help="Path to config YAML")] = None,
) -> None:
    """Export a past session to JSON, Markdown, or HTML."""
    from vaig.cli.export import ExportMetadata, ExportPayload, format_export
    from vaig.session.manager import SessionManager

    settings = _get_settings(config)
    manager = SessionManager(settings)

    session_data = manager._store.get_session(session_id)  # noqa: SLF001
    if not session_data:
        err_console.print(f"[red]Session not found: {session_id}[/red]")
        manager.close()
        raise typer.Exit(1)

    messages = manager._store.get_messages(session_id)  # noqa: SLF001
    context_files_rows = manager._store.get_context_files(session_id)  # noqa: SLF001
    manager.close()

    # Build question from first user message, response from last assistant message
    user_messages = [m for m in messages if m["role"] == "user"]
    assistant_messages = [m for m in messages if m["role"] in ("assistant", "model")]

    question_text = user_messages[0]["content"] if user_messages else "(no question)"
    response_text = assistant_messages[-1]["content"] if assistant_messages else "(no response)"

    # Sum up tokens from all messages
    total_tokens = sum(m.get("token_count", 0) for m in messages)

    meta = ExportMetadata(
        model=session_data.get("model", "unknown"),
        skill=session_data.get("skill"),
        timestamp=session_data.get("created_at", ""),
        tokens={"total_tokens": total_tokens},
        cost=None,
        vaig_version=__version__,
    )

    payload = ExportPayload(
        question=question_text,
        response=response_text,
        metadata=meta,
        context_files=[cf["file_path"] for cf in context_files_rows],
    )

    content = format_export(payload, format_)
    if output:
        _save_output(output, content)
    else:
        console.print(content)


# ══════════════════════════════════════════════════════════════
# SESSIONS — List, show, delete
# ══════════════════════════════════════════════════════════════
@sessions_app.command("list")
def sessions_list(
    config: Annotated[Optional[str], typer.Option("--config", "-c")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max sessions to show")] = 20,
) -> None:
    """List recent chat sessions."""
    settings = _get_settings(config)

    from vaig.session.manager import SessionManager

    manager = SessionManager(settings)
    sessions = manager.list_sessions(limit=limit)

    if not sessions:
        console.print("[yellow]No sessions found.[/yellow]")
        return

    table = Table(title="Sessions", show_lines=False)
    table.add_column("ID", style="cyan", no_wrap=True, max_width=12)
    table.add_column("Name", style="bold")
    table.add_column("Model", style="magenta")
    table.add_column("Skill", style="green")
    table.add_column("Msgs", style="yellow", justify="right")
    table.add_column("Updated", style="dim")

    for s in sessions:
        if not isinstance(s, dict):
            continue
        table.add_row(
            s.get("id", "?")[:12],
            s.get("name", "—"),
            s.get("model", "—"),
            s.get("skill", "—") or "—",
            str(s.get("message_count", 0)),
            _format_session_date(s.get("updated_at", "")),
        )

    console.print(table)
    manager.close()


@sessions_app.command("delete")
def sessions_delete(
    session_id: Annotated[str, typer.Argument(help="Session ID to delete")],
    config: Annotated[Optional[str], typer.Option("--config", "-c")] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation")] = False,
) -> None:
    """Delete a chat session."""
    settings = _get_settings(config)

    from vaig.session.manager import SessionManager

    if not force:
        confirmed = typer.confirm(f"Delete session {session_id}?")
        if not confirmed:
            raise typer.Exit

    manager = SessionManager(settings)
    if manager.delete_session(session_id):
        console.print(f"[green]✓ Deleted session: {session_id}[/green]")
    else:
        err_console.print(f"[red]Session not found: {session_id}[/red]")
    manager.close()


@sessions_app.command("rename")
def sessions_rename(
    session_id: Annotated[str, typer.Argument(help="Session ID (or prefix) to rename")],
    new_name: Annotated[str, typer.Argument(help="New name for the session")],
    config: Annotated[Optional[str], typer.Option("--config", "-c")] = None,
) -> None:
    """Rename a chat session."""
    settings = _get_settings(config)

    from vaig.session.manager import SessionManager

    manager = SessionManager(settings)
    session_id = _resolve_session_id(manager, session_id)
    if manager.rename_session(session_id, new_name):
        console.print(f"[green]Renamed session {session_id[:12]} to: {new_name}[/green]")
    else:
        err_console.print(f"[red]Session not found: {session_id}[/red]")
    manager.close()


@sessions_app.command("search")
def sessions_search(
    query: Annotated[str, typer.Argument(help="Search term (matches session name and message content)")],
    config: Annotated[Optional[str], typer.Option("--config", "-c")] = None,
) -> None:
    """Search sessions by name or message content."""
    settings = _get_settings(config)

    from vaig.session.manager import SessionManager

    manager = SessionManager(settings)
    sessions = manager.search_sessions(query)

    if not sessions:
        console.print(f"[yellow]No sessions matching '{query}'.[/yellow]")
        manager.close()
        return

    table = Table(title=f"Search: '{query}'", show_lines=False)
    table.add_column("ID", style="cyan", no_wrap=True, max_width=12)
    table.add_column("Name", style="bold")
    table.add_column("Model", style="magenta")
    table.add_column("Msgs", style="yellow", justify="right")
    table.add_column("Updated", style="dim")

    for s in sessions:
        if not isinstance(s, dict):
            continue
        table.add_row(
            s.get("id", "?")[:12],
            s.get("name", "—"),
            s.get("model", "—"),
            str(s.get("message_count", 0)),
            _format_session_date(s.get("updated_at", "")),
        )

    console.print(table)
    manager.close()


@sessions_app.command("show")
def sessions_show(
    session_id: Annotated[str, typer.Argument(help="Session ID (or prefix) to show")],
    config: Annotated[Optional[str], typer.Option("--config", "-c")] = None,
    messages: Annotated[int, typer.Option("--messages", "-m", help="Number of recent messages to show")] = 10,
) -> None:
    """Show detailed information about a session."""
    settings = _get_settings(config)

    from vaig.session.manager import SessionManager

    manager = SessionManager(settings)
    session_id = _resolve_session_id(manager, session_id)

    session = manager._store.get_session(session_id)
    if not session:
        err_console.print(f"[red]Session not found: {session_id}[/red]")
        manager.close()
        return

    msgs = manager._store.get_messages(session_id)
    context_files = manager._store.get_context_files(session_id)

    # Session info panel
    info_lines = [
        f"[cyan]ID:[/cyan]      {session['id']}",
        f"[cyan]Name:[/cyan]    {session['name']}",
        f"[cyan]Model:[/cyan]   {session['model']}",
        f"[cyan]Skill:[/cyan]   {session.get('skill') or '—'}",
        f"[cyan]Created:[/cyan] {_format_session_date(session['created_at'])}",
        f"[cyan]Updated:[/cyan] {_format_session_date(session['updated_at'])}",
        f"[cyan]Messages:[/cyan] {len(msgs)}",
    ]
    if context_files:
        info_lines.append(f"[cyan]Context files:[/cyan] {len(context_files)}")
    console.print(Panel("\n".join(info_lines), title="Session Details", border_style="cyan"))

    # Context files
    if context_files:
        ft = Table(title="Context Files", show_lines=False)
        ft.add_column("Path", style="green")
        ft.add_column("Type", style="dim")
        ft.add_column("Size", style="yellow", justify="right")
        for cf in context_files:
            size = cf.get("size_bytes", 0)
            size_str = f"{size:,}" if size else "—"
            ft.add_row(cf["file_path"], cf["file_type"], size_str)
        console.print(ft)

    # Recent messages
    recent = msgs[-messages:] if len(msgs) > messages else msgs
    if recent:
        mt = Table(title=f"Messages (last {len(recent)} of {len(msgs)})", show_lines=True)
        mt.add_column("Role", style="bold", max_width=8)
        mt.add_column("Content", max_width=100)
        mt.add_column("Time", style="dim", max_width=20)
        for m in recent:
            content = m["content"]
            if len(content) > 200:
                content = content[:200] + "..."
            role_style = "cyan" if m["role"] == "user" else "green"
            mt.add_row(
                f"[{role_style}]{m['role']}[/{role_style}]",
                content,
                _format_session_date(m.get("created_at", "")),
            )
        console.print(mt)

    manager.close()
@models_app.command("list")
def models_list(
    config: Annotated[Optional[str], typer.Option("--config", "-c")] = None,
) -> None:
    """List available models."""
    settings = _get_settings(config)

    from vaig.core.client import GeminiClient

    client = GeminiClient(settings)
    models = client.list_available_models()

    if not models:
        console.print("[yellow]No models configured. Add models to your config YAML.[/yellow]")
        console.print(f"[dim]Default model: {settings.models.default}[/dim]")
        return

    table = Table(title="🤖 Available Models", show_lines=False)
    table.add_column("Model ID", style="cyan")
    table.add_column("Description", style="dim")
    table.add_column("Default", justify="center")

    for m in models:
        if not isinstance(m, dict):
            continue
        is_default = "✓" if m.get("id") == settings.models.default else ""
        table.add_row(m.get("id", "?"), m.get("description", ""), is_default)

    console.print(table)
    console.print(f"\n[dim]Default: {settings.models.default} | Fallback: {settings.models.fallback}[/dim]")


# ══════════════════════════════════════════════════════════════
# SKILLS — List, info
# ══════════════════════════════════════════════════════════════
@skills_app.command("list")
def skills_list(
    config: Annotated[Optional[str], typer.Option("--config", "-c")] = None,
) -> None:
    """List available skills."""
    settings = _get_settings(config)

    from vaig.skills.registry import SkillRegistry

    registry = SkillRegistry(settings)
    skills = registry.list_skills()

    if not skills:
        console.print("[yellow]No skills loaded.[/yellow]")
        return

    table = Table(title="🛠️  Skills", show_lines=False)
    table.add_column("Name", style="cyan")
    table.add_column("Display Name", style="bold")
    table.add_column("Description")
    table.add_column("Phases", style="green")
    table.add_column("Model", style="magenta")

    for meta in skills:
        phases = ", ".join(p.value for p in meta.supported_phases)
        table.add_row(meta.name, meta.display_name, meta.description, phases, meta.recommended_model)

    console.print(table)


@skills_app.command("info")
def skills_info(
    skill_name: Annotated[str, typer.Argument(help="Skill name to inspect")],
    config: Annotated[Optional[str], typer.Option("--config", "-c")] = None,
) -> None:
    """Show detailed info about a skill."""
    settings = _get_settings(config)

    from vaig.skills.registry import SkillRegistry

    registry = SkillRegistry(settings)
    skill = registry.get(skill_name)

    if not skill:
        err_console.print(f"[red]Skill not found: {skill_name}[/red]")
        err_console.print(f"[dim]Available: {', '.join(registry.list_names())}[/dim]")
        raise typer.Exit(1)

    meta = skill.get_metadata()
    agents = skill.get_agents_config()

    console.print(
        Panel(
            f"[bold]{meta.display_name}[/bold] ({meta.name} v{meta.version})\n\n"
            f"{meta.description}\n\n"
            f"[dim]Tags: {', '.join(meta.tags)}[/dim]\n"
            f"[dim]Phases: {', '.join(p.value for p in meta.supported_phases)}[/dim]\n"
            f"[dim]Recommended model: {meta.recommended_model}[/dim]",
            title="🛠️  Skill Info",
            border_style="bright_blue",
        )
    )

    # Show agents
    if agents:
        agent_table = Table(title="Agents", show_lines=False)
        agent_table.add_column("Name", style="cyan")
        agent_table.add_column("Role", style="bold")
        agent_table.add_column("Model", style="magenta")

        for a in agents:
            agent_table.add_row(a["name"], a["role"], a.get("model", "default"))

        console.print(agent_table)


@skills_app.command("create")
def skills_create(
    name: Annotated[str, typer.Argument(help="Skill name (kebab-case, e.g. 'my-analyzer')")],
    description: Annotated[str, typer.Option("--description", "-d")] = "A custom skill",
    tags: Annotated[Optional[str], typer.Option("--tags", "-t", help="Comma-separated tags")] = None,
    output_dir: Annotated[Optional[str], typer.Option("--output", "-o", help="Target directory (default: custom_dir from config)")] = None,
    config: Annotated[Optional[str], typer.Option("--config", "-c")] = None,
) -> None:
    """Scaffold a new custom skill with boilerplate files."""
    from vaig.skills.scaffold import scaffold_skill

    settings = _get_settings(config)

    # Determine target directory
    if output_dir:
        target = Path(output_dir).expanduser().resolve()
    elif settings.skills.custom_dir:
        target = Path(settings.skills.custom_dir).expanduser().resolve()
    else:
        target = Path.cwd() / "skills"
        console.print(
            f"[yellow]No custom_dir configured. Scaffolding to: {target}[/yellow]\n"
            "[dim]Set 'skills.custom_dir' in your config to auto-load custom skills.[/dim]"
        )

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

    try:
        skill_dir = scaffold_skill(name, target, description=description, tags=tag_list)
    except FileExistsError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None

    console.print(f"[green]Skill scaffolded at:[/green] {skill_dir}")
    console.print("\n[bold]Created files:[/bold]")
    console.print(f"  {skill_dir / '__init__.py'}")
    console.print(f"  {skill_dir / 'skill.py'}")
    console.print(f"  {skill_dir / 'prompts.py'}")
    console.print(f"\n[dim]Edit skill.py and prompts.py to implement your skill logic.[/dim]")


# ══════════════════════════════════════════════════════════════
# MCP — Model Context Protocol server management
# ══════════════════════════════════════════════════════════════
@mcp_app.command("list-servers")
def mcp_list_servers(
    config: Annotated[Optional[str], typer.Option("--config", "-c")] = None,
) -> None:
    """List configured MCP servers."""
    from vaig.tools.mcp_bridge import is_mcp_available

    settings = _get_settings(config)

    if not is_mcp_available():
        err_console.print(
            "[bold red]MCP SDK not installed.[/bold red]\n"
            "[yellow]Install with:[/yellow]  pip install mcp"
        )
        raise typer.Exit(1)

    mcp_cfg = settings.mcp

    if not mcp_cfg.enabled:
        console.print(
            "[yellow]MCP integration is disabled.[/yellow]\n"
            "[dim]Set 'mcp.enabled: true' in your config to enable it.[/dim]"
        )
        return

    if not mcp_cfg.servers:
        console.print("[yellow]No MCP servers configured.[/yellow]")
        console.print("[dim]Add servers under 'mcp.servers' in your config YAML.[/dim]")
        return

    table = Table(title="MCP Servers", show_lines=False)
    table.add_column("Name", style="cyan")
    table.add_column("Command", style="bold")
    table.add_column("Args", style="dim")
    table.add_column("Description")

    for srv in mcp_cfg.servers:
        table.add_row(
            srv.name,
            srv.command,
            " ".join(srv.args) if srv.args else "—",
            srv.description or "—",
        )

    console.print(table)


@mcp_app.command("discover")
def mcp_discover(
    server_name: Annotated[str, typer.Argument(help="Name of configured MCP server to discover tools from")],
    config: Annotated[Optional[str], typer.Option("--config", "-c")] = None,
) -> None:
    """Discover available tools from an MCP server.

    Examples:
        vaig mcp discover my-server
    """
    import asyncio

    from vaig.tools.mcp_bridge import discover_tools, is_mcp_available

    if not is_mcp_available():
        err_console.print(
            "[bold red]MCP SDK not installed.[/bold red]\n"
            "[yellow]Install with:[/yellow]  pip install mcp"
        )
        raise typer.Exit(1)

    settings = _get_settings(config)
    mcp_cfg = settings.mcp

    if not mcp_cfg.enabled:
        err_console.print("[red]MCP integration is disabled. Set 'mcp.enabled: true' in your config.[/red]")
        raise typer.Exit(1)

    srv = next((s for s in mcp_cfg.servers if s.name == server_name), None)
    if not srv:
        err_console.print(f"[red]MCP server not found: {server_name}[/red]")
        available = [s.name for s in mcp_cfg.servers]
        if available:
            err_console.print(f"[dim]Available: {', '.join(available)}[/dim]")
        raise typer.Exit(1)

    with console.status(f"[bold cyan]Connecting to {server_name}...[/bold cyan]"):
        try:
            tools = asyncio.run(
                discover_tools(
                    command=srv.command,
                    args=srv.args or None,
                    env=srv.env or None,
                )
            )
        except Exception as exc:
            err_console.print(f"[red]Failed to connect to {server_name}: {exc}[/red]")
            raise typer.Exit(1) from None

    if not tools:
        console.print(f"[yellow]{server_name} exposes no tools.[/yellow]")
        return

    table = Table(title=f"Tools from {server_name}", show_lines=True)
    table.add_column("Tool", style="cyan")
    table.add_column("Description")
    table.add_column("Parameters", style="dim")

    for tool in tools:
        schema = tool.inputSchema or {}
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        param_strs: list[str] = []
        for pname, pschema in props.items():
            req_mark = "*" if pname in required else ""
            ptype = pschema.get("type", "?")
            param_strs.append(f"{pname}{req_mark}: {ptype}")
        table.add_row(
            tool.name,
            tool.description or "—",
            "\n".join(param_strs) if param_strs else "—",
        )

    console.print(table)
    console.print(f"\n[dim]{len(tools)} tools discovered from {server_name}[/dim]")


@mcp_app.command("call")
def mcp_call(
    server_name: Annotated[str, typer.Argument(help="MCP server name")],
    tool_name: Annotated[str, typer.Argument(help="Tool name to call")],
    args_json: Annotated[Optional[str], typer.Argument(help="JSON arguments (e.g. '{\"path\": \"/tmp\"}')")] = None,
    config: Annotated[Optional[str], typer.Option("--config", "-c")] = None,
) -> None:
    """Call a specific tool on an MCP server.

    Examples:
        vaig mcp call my-server read_file '{"path": "/etc/hostname"}'
        vaig mcp call my-server list_files
    """
    import asyncio
    import json

    from vaig.tools.mcp_bridge import call_mcp_tool, is_mcp_available

    if not is_mcp_available():
        err_console.print(
            "[bold red]MCP SDK not installed.[/bold red]\n"
            "[yellow]Install with:[/yellow]  pip install mcp"
        )
        raise typer.Exit(1)

    settings = _get_settings(config)
    mcp_cfg = settings.mcp

    if not mcp_cfg.enabled:
        err_console.print("[red]MCP integration is disabled. Set 'mcp.enabled: true' in your config.[/red]")
        raise typer.Exit(1)

    srv = next((s for s in mcp_cfg.servers if s.name == server_name), None)
    if not srv:
        err_console.print(f"[red]MCP server not found: {server_name}[/red]")
        available = [s.name for s in mcp_cfg.servers]
        if available:
            err_console.print(f"[dim]Available: {', '.join(available)}[/dim]")
        raise typer.Exit(1)

    # Parse arguments
    arguments: dict[str, Any] = {}
    if args_json:
        try:
            arguments = json.loads(args_json)
        except json.JSONDecodeError as exc:
            err_console.print(f"[red]Invalid JSON arguments: {exc}[/red]")
            raise typer.Exit(1) from None

    with console.status(f"[bold cyan]Calling {tool_name} on {server_name}...[/bold cyan]"):
        result = asyncio.run(
            call_mcp_tool(
                command=srv.command,
                tool_name=tool_name,
                arguments=arguments,
                args=srv.args or None,
                env=srv.env or None,
            )
        )

    if result.error:
        err_console.print(f"[red]Tool error:[/red] {result.output}")
        raise typer.Exit(1)

    console.print(result.output)

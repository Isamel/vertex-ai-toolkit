"""CLI app — Typer commands for VAIG toolkit."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from vaig import __version__

if TYPE_CHECKING:
    from vaig.agents.base import AgentResult
    from vaig.agents.orchestrator import OrchestratorResult
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

app.add_typer(sessions_app, name="sessions")
app.add_typer(models_app, name="models")
app.add_typer(skills_app, name="skills")


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
    log_level: Annotated[
        Optional[str],
        typer.Option("--log-level", help="Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"),
    ] = None,
) -> None:
    """VAIG — Vertex AI Gemini Toolkit."""
    from vaig.core.log import setup_logging

    # Determine effective log level: --log-level takes precedence over -V
    if log_level:
        level = log_level.upper()
    elif verbose:
        level = "INFO"
    else:
        level = "WARNING"

    setup_logging(level, show_path=log_level == "DEBUG" if log_level else False)


# ══════════════════════════════════════════════════════════════
# CHAT — Interactive REPL
# ══════════════════════════════════════════════════════════════
@app.command()
def chat(
    config: Annotated[Optional[str], typer.Option("--config", "-c", help="Path to config YAML")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Model to use")] = None,
    skill: Annotated[Optional[str], typer.Option("--skill", "-s", help="Activate a skill")] = None,
    session: Annotated[Optional[str], typer.Option("--session", help="Load an existing session by ID")] = None,
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

    start_repl(
        settings=settings,
        skill_name=skill,
        session_id=session,
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
    skill: Annotated[Optional[str], typer.Option("--skill", "-s", help="Use a specific skill")] = None,
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
    """
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
        gke_config = _build_gke_config(settings, cluster=cluster, namespace=namespace, project_id=project_id)
        _execute_live_mode(
            client,
            gke_config,
            question,
            context_str,
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
        if output:
            _save_output(output, result.output)
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
                if output:
                    _save_output(output, result.content)  # type: ignore[union-attr]
        else:
            stream = orchestrator.execute_single(question, context=context_str, stream=True)
            console.print()
            response_parts: list[str] = []
            for chunk in stream:  # type: ignore[union-attr]
                console.print(chunk, end="")
                response_parts.append(chunk)
            console.print()
            if output:
                _save_output(output, "".join(response_parts))


def _save_output(output: Path, content: str) -> None:
    """Write response content to a file and print confirmation."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)
    console.print(f"[green]✓ Response saved to {output}[/green]")


# ══════════════════════════════════════════════════════════════
# LIVE — Infrastructure investigation (GKE/GCP)
# ══════════════════════════════════════════════════════════════
@app.command()
def live(
    question: Annotated[str, typer.Argument(help="Infrastructure question or investigation task")],
    config: Annotated[Optional[str], typer.Option("--config", "-c", help="Path to config YAML")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Model to use")] = None,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Save response to a file")] = None,
    skill: Annotated[Optional[str], typer.Option("--skill", "-s", help="SRE skill to apply")] = None,
    cluster: Annotated[Optional[str], typer.Option("--cluster", help="GKE cluster name (overrides config)")] = None,
    namespace: Annotated[
        Optional[str], typer.Option("--namespace", help="Default Kubernetes namespace (overrides config)")
    ] = None,
    project_id: Annotated[
        Optional[str], typer.Option("--project-id", help="GCP project ID (overrides config)")
    ] = None,
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
    """
    settings = _get_settings(config)

    if model:
        settings.models.default = model

    from vaig.core.client import GeminiClient

    client = GeminiClient(settings)
    gke_config = _build_gke_config(settings, cluster=cluster, namespace=namespace, project_id=project_id)

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
        output=output,
        model_id=model,
    )


# ── Live mode helpers ─────────────────────────────────────────


def _build_gke_config(
    settings: "Settings",
    *,
    cluster: str | None = None,
    namespace: str | None = None,
    project_id: str | None = None,
) -> "GKEConfig":
    """Build a GKEConfig, applying CLI overrides on top of config file defaults.

    Args:
        settings: Application settings (contains gke section).
        cluster: Optional cluster name override.
        namespace: Optional default namespace override.
        project_id: Optional GCP project ID override.

    Returns:
        GKEConfig with CLI overrides applied.
    """
    from vaig.core.config import GKEConfig

    gke = settings.gke

    return GKEConfig(
        cluster_name=cluster or gke.cluster_name,
        project_id=project_id or gke.project_id or settings.gcp.project_id,
        default_namespace=namespace or gke.default_namespace,
        kubeconfig_path=gke.kubeconfig_path,
        context=gke.context,
        log_limit=gke.log_limit,
        metrics_interval_minutes=gke.metrics_interval_minutes,
        proxy_url=gke.proxy_url,
    )


def _register_live_tools(gke_config: "GKEConfig") -> "ToolRegistry":
    """Create a ToolRegistry and register GKE + GCloud tools.

    Follows the same try/except ImportError pattern as InfraAgent._register_tools()
    so missing optional dependencies degrade gracefully.

    Returns:
        Populated ToolRegistry (may be empty if no optional deps installed).
    """
    from vaig.tools.base import ToolRegistry

    registry = ToolRegistry()

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
        ):
            registry.register(tool)
    except ImportError as exc:
        logger.warning("Could not load GCloud observability tools: %s", exc)

    return registry


def _execute_orchestrated_skill(
    client: "GeminiClient",
    settings: "Settings",
    gke_config: "GKEConfig",
    skill: "BaseSkill",
    question: str,
    *,
    output: Path | None = None,
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
    tool_registry = _register_live_tools(gke_config)

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
        )

        # Display final response
        console.print()
        if orch_result.synthesized_output:
            console.print(Markdown(orch_result.synthesized_output))
        console.print()

        if output and orch_result.synthesized_output:
            _save_output(output, orch_result.synthesized_output)

        # Show agent pipeline summary
        _show_orchestrated_summary(orch_result)

    except MaxIterationsError as exc:
        err_console.print(
            f"\n[bold red]⚠ Max iterations reached ({exc.iterations})[/bold red]\n"
            "[yellow]The orchestrated skill pipeline hit its tool-use iteration limit. "
            "Try narrowing the scope of your question or specifying a namespace/resource.[/yellow]"
        )
        raise typer.Exit(1)  # noqa: B904


def _show_orchestrated_summary(orch_result: "OrchestratorResult") -> None:
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

    if not orch_result.success:
        err_console.print("[bold red]⚠ Pipeline completed with errors[/bold red]")


def _execute_live_mode(
    client: "GeminiClient",
    gke_config: "GKEConfig",
    question: str,
    context: str,
    *,
    output: Path | None = None,
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

        if output and result.content:
            _save_output(output, result.content)

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
            _save_output(output, result.content)

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
    system_instruction = orchestrator._default_system_instruction()  # noqa: SLF001

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

    table = Table(title="📋 Sessions", show_lines=False)
    table.add_column("ID", style="cyan", no_wrap=True, max_width=12)
    table.add_column("Name", style="bold")
    table.add_column("Model", style="magenta")
    table.add_column("Skill", style="green")
    table.add_column("Created", style="dim")

    for s in sessions:
        if not isinstance(s, dict):
            continue
        table.add_row(
            s.get("id", "?")[:12],
            s.get("name", "—"),
            s.get("model", "—"),
            s.get("skill", "—") or "—",
            s.get("created_at", "—"),
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


# ══════════════════════════════════════════════════════════════
# MODELS — List, info
# ══════════════════════════════════════════════════════════════
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

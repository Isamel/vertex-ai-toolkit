"""CLI app — Typer commands for VAIG toolkit."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from vaig import __version__

if TYPE_CHECKING:
    from vaig.agents.base import AgentResult
    from vaig.core.client import GeminiClient
    from vaig.core.config import Settings

console = Console()
err_console = Console(stderr=True)

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
) -> None:
    """Start an interactive chat session (REPL mode).

    Examples:
        vaig chat
        vaig chat --model gemini-2.5-flash
        vaig chat --skill rca
        vaig chat --session abc123
    """
    from vaig.cli.repl import start_repl

    _banner()
    settings = _get_settings(config)

    if model:
        settings.models.default = model

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
    skill: Annotated[Optional[str], typer.Option("--skill", "-s", help="Use a specific skill")] = None,
    no_stream: Annotated[bool, typer.Option("--no-stream", help="Disable streaming output")] = False,
    code: Annotated[bool, typer.Option("--code", help="Enable coding agent mode (read/write/edit files)")] = False,
) -> None:
    """Ask a single question and get a response.

    Examples:
        vaig ask "What is Kubernetes?"
        vaig ask "Analyze this code" -f main.py
        vaig ask "Investigate this incident" -s rca -f logs.txt
        vaig ask "Add error handling to app.py" --code
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
        _execute_code_mode(client, settings, question, context_str)
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
        console.print(result.output)
    else:
        # Direct chat — single agent
        if no_stream:
            with console.status(
                f"[bold cyan]Generating response with {settings.models.default}...[/bold cyan]"
            ):
                result = orchestrator.execute_single(question, context=context_str)
            console.print()
            if hasattr(result, "content"):
                console.print(result.content)  # type: ignore[union-attr]
        else:
            stream = orchestrator.execute_single(question, context=context_str, stream=True)
            console.print()
            for chunk in stream:  # type: ignore[union-attr]
                console.print(chunk, end="")
            console.print()


def _execute_code_mode(
    client: "GeminiClient",
    settings: "Settings",
    question: str,
    context: str,
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
        console.print(result.content)
        console.print()

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
    """Rich-based confirmation prompt for destructive tool operations (Task 5.3)."""
    # Build a readable summary of what the tool will do
    if tool_name == "write_file":
        desc = f"Write file: [cyan]{args.get('path', '?')}[/cyan]"
    elif tool_name == "edit_file":
        desc = f"Edit file: [cyan]{args.get('path', '?')}[/cyan]"
    elif tool_name == "run_command":
        desc = f"Run command: [cyan]{args.get('command', '?')}[/cyan]"
    else:
        desc = f"Execute: [cyan]{tool_name}[/cyan]"

    console.print(f"\n[bold yellow]⚡ {desc}[/bold yellow]")
    return typer.confirm("  Allow this operation?", default=True)


def _show_coding_summary(result: "AgentResult") -> None:
    """Display tool execution feedback and token usage (Tasks 5.5 + 5.6)."""
    metadata = result.metadata or {}
    tools_executed = metadata.get("tools_executed", [])
    iterations = metadata.get("iterations", 0)

    if tools_executed:
        table = Table(title="🔧 Tools Executed", show_lines=False, title_style="bold")
        table.add_column("#", style="dim", width=3)
        table.add_column("Tool", style="cyan")
        table.add_column("Target", style="white")
        table.add_column("Status", justify="center")

        for i, tool in enumerate(tools_executed, 1):
            name = tool.get("name", "?")
            args = tool.get("args", {})
            error = tool.get("error", False)

            # Extract the most relevant arg for display
            target = args.get("path", args.get("command", args.get("pattern", "")))
            if len(str(target)) > 60:
                target = str(target)[:57] + "..."

            status = "[red]✗[/red]" if error else "[green]✓[/green]"
            table.add_row(str(i), name, str(target), status)

        console.print(table)

    # Usage summary
    usage = result.usage or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    console.print(
        f"[dim]Completed in {iterations} iteration{'s' if iterations != 1 else ''} "
        f"| Tokens: {total_tokens:,} total "
        f"({prompt_tokens:,} prompt + {completion_tokens:,} completion)[/dim]"
    )


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

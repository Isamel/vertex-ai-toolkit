"""Interactive REPL — prompt-toolkit based chat with slash commands."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

import typer

from vaig.agents.orchestrator import Orchestrator
from vaig.context.builder import ContextBuilder
from vaig.core.client import GeminiClient
from vaig.core.config import Settings
from vaig.session.manager import SessionManager
from vaig.skills.base import BaseSkill, SkillPhase
from vaig.skills.registry import SkillRegistry

if TYPE_CHECKING:
    from vaig.agents.base import AgentResult

logger = logging.getLogger(__name__)
console = Console()

# ── Prompt Styling ────────────────────────────────────────────
PROMPT_STYLE = Style.from_dict(
    {
        "prompt": "ansicyan bold",
        "": "ansiwhite",
    }
)

# ── Slash Command Completer ──────────────────────────────────
SLASH_COMMANDS = [
    "/add",
    "/code",
    "/model",
    "/skill",
    "/phase",
    "/agents",
    "/sessions",
    "/new",
    "/load",
    "/clear",
    "/context",
    "/help",
    "/quit",
    "/exit",
]

command_completer = WordCompleter(SLASH_COMMANDS, sentence=True)


class REPLState:
    """Mutable state for the REPL session."""

    def __init__(
        self,
        settings: Settings,
        client: GeminiClient,
        orchestrator: Orchestrator,
        session_manager: SessionManager,
        context_builder: ContextBuilder,
        skill_registry: SkillRegistry,
    ) -> None:
        self.settings = settings
        self.client = client
        self.orchestrator = orchestrator
        self.session_manager = session_manager
        self.context_builder = context_builder
        self.skill_registry = skill_registry
        self.active_skill: BaseSkill | None = None
        self.current_phase: SkillPhase = SkillPhase.ANALYZE
        self.stream_enabled: bool = True
        self.code_mode: bool = False

    @property
    def model(self) -> str:
        return self.client.current_model

    @property
    def skill_name(self) -> str | None:
        if self.active_skill:
            return self.active_skill.get_metadata().name
        return None

    def prompt_prefix(self) -> str:
        """Build the prompt prefix showing current state."""
        parts = [f"[{self.model}]"]
        if self.code_mode:
            parts.append("(🔧code)")
        if self.active_skill:
            parts.append(f"({self.skill_name}:{self.current_phase.value})")
        if self.context_builder.bundle.file_count > 0:
            parts.append(f"📁{self.context_builder.bundle.file_count}")
        return " ".join(parts) + " > "


def start_repl(
    settings: Settings,
    *,
    skill_name: str | None = None,
    session_id: str | None = None,
    session_name: str = "chat",
) -> None:
    """Start the interactive REPL loop."""
    # Initialize components
    client = GeminiClient(settings)
    orchestrator = Orchestrator(client, settings)
    session_manager = SessionManager(settings)
    context_builder = ContextBuilder(settings)
    skill_registry = SkillRegistry(settings)

    state = REPLState(
        settings=settings,
        client=client,
        orchestrator=orchestrator,
        session_manager=session_manager,
        context_builder=context_builder,
        skill_registry=skill_registry,
    )

    # Load or create session
    if session_id:
        loaded = session_manager.load_session(session_id)
        if loaded:
            console.print(f"[green]✓ Resumed session: {loaded.name} ({len(loaded.history)} messages)[/green]")
        else:
            console.print(f"[red]Session not found: {session_id}. Starting new session.[/red]")
            session_manager.new_session(session_name, model=settings.models.default)
    else:
        session_manager.new_session(session_name, model=settings.models.default, skill=skill_name)
        console.print(f"[green]✓ New session: {session_name}[/green]")

    # Activate skill if requested
    if skill_name:
        _cmd_skill(state, skill_name)

    # Print help hint
    console.print("[dim]Type /help for commands. Ctrl+D or /quit to exit.[/dim]\n")

    # Start prompt loop
    prompt_session: PromptSession[str] = PromptSession(
        history=InMemoryHistory(),
        completer=command_completer,
        style=PROMPT_STYLE,
    )

    try:
        _repl_loop(prompt_session, state)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
    finally:
        session_manager.close()
        console.print("[dim]Session saved. Goodbye! 👋[/dim]")


def _repl_loop(prompt_session: PromptSession[str], state: REPLState) -> None:
    """Main REPL input loop."""
    while True:
        try:
            user_input = prompt_session.prompt(state.prompt_prefix()).strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            console.print("[yellow]^C[/yellow]")
            continue

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            should_quit = _handle_command(state, user_input)
            if should_quit:
                break
            continue

        # Regular chat message
        _handle_chat(state, user_input)


def _handle_command(state: REPLState, raw_input: str) -> bool:
    """Handle a slash command. Returns True if the REPL should exit."""
    parts = raw_input.split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    handlers: dict[str, object] = {
        "/add": lambda: _cmd_add(state, args),
        "/code": lambda: _cmd_code(state),
        "/model": lambda: _cmd_model(state, args),
        "/skill": lambda: _cmd_skill(state, args),
        "/phase": lambda: _cmd_phase(state, args),
        "/agents": lambda: _cmd_agents(state),
        "/sessions": lambda: _cmd_sessions(state),
        "/new": lambda: _cmd_new_session(state, args),
        "/load": lambda: _cmd_load_session(state, args),
        "/clear": lambda: _cmd_clear(state),
        "/context": lambda: _cmd_context(state),
        "/help": lambda: _cmd_help(),
        "/quit": lambda: True,
        "/exit": lambda: True,
    }

    handler = handlers.get(command)
    if handler is None:
        console.print(f"[red]Unknown command: {command}[/red] — type /help for available commands")
        return False

    result = handler()  # type: ignore[operator]
    return result is True


def _handle_chat(state: REPLState, user_input: str) -> None:
    """Handle a regular chat message."""
    context_str = ""
    if state.context_builder.bundle.file_count > 0:
        context_str = state.context_builder.bundle.to_context_string()

    # Record user message
    state.session_manager.add_message("user", user_input)

    if state.code_mode:
        # Code mode — use CodingAgent (Task 5.4)
        _handle_code_chat(state, user_input, context_str)
    elif state.active_skill:
        # Skill-based execution
        _handle_skill_chat(state, user_input, context_str)
    else:
        # Direct chat
        _handle_direct_chat(state, user_input, context_str)


def _handle_direct_chat(state: REPLState, user_input: str, context: str) -> None:
    """Handle direct chat without a skill."""
    try:
        if state.stream_enabled:
            stream = state.orchestrator.execute_single(user_input, context=context, stream=True)
            response_parts: list[str] = []
            for chunk in stream:  # type: ignore[union-attr]
                console.print(chunk, end="")
                response_parts.append(chunk)
            console.print()
            full_response = "".join(response_parts)
        else:
            with console.status(
                f"[bold cyan]Generating response with {state.model}...[/bold cyan]"
            ):
                result = state.orchestrator.execute_single(user_input, context=context)
            full_response = result.content  # type: ignore[union-attr]
            console.print(Markdown(full_response))

        # Record model response
        state.session_manager.add_message("model", full_response, model=state.model)
        console.print()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Chat error")


def _handle_skill_chat(state: REPLState, user_input: str, context: str) -> None:
    """Handle chat with an active skill."""
    skill = state.active_skill
    if not skill:
        return

    meta = skill.get_metadata()
    console.print(f"[dim]Running {meta.display_name} → {state.current_phase.value} on {state.model}...[/dim]")

    try:
        with console.status(
            f"[bold cyan]{meta.display_name} ({state.current_phase.value}) on {state.model}...[/bold cyan]"
        ):
            result = state.orchestrator.execute_skill_phase(
                skill,
                state.current_phase,
                context,
                user_input,
            )

        console.print()
        console.print(Markdown(result.output))
        console.print()

        # Record model response
        state.session_manager.add_message("model", result.output, model=state.model)

        # Suggest next phase
        if result.next_phase:
            console.print(f"[dim]Suggested next phase: /phase {result.next_phase.value}[/dim]")

    except Exception as e:
        console.print(f"[red]Error during {state.current_phase.value}: {e}[/red]")
        logger.exception("Skill execution error")


def _handle_code_chat(state: REPLState, user_input: str, context: str) -> None:
    """Handle chat in code mode — uses CodingAgent with tool-use loop.

    Implements Tasks 5.4 (routing), 5.5 (tool feedback), 5.6 (usage summary),
    5.7 (MaxIterationsError handling), 5.8 (no streaming).
    """
    from vaig.agents.coding import CodingAgent
    from vaig.core.exceptions import MaxIterationsError

    coding_config = state.settings.coding
    agent = CodingAgent(
        state.client,
        coding_config,
        confirm_fn=_repl_confirm,
        model_id=state.model,
    )

    try:
        # Task 5.8: No streaming in code mode — tool loops are non-streamable
        with console.status(
            "[bold cyan]🤖 Coding agent working...[/bold cyan]",
            spinner="dots",
        ):
            result = agent.execute(user_input, context=context)

        # Display final response
        console.print()
        console.print(Markdown(result.content))
        console.print()

        # Task 5.5 + 5.6: Show tool feedback and usage summary
        _show_repl_coding_summary(result)

        # Record model response
        state.session_manager.add_message("model", result.content, model=state.model)

    except MaxIterationsError as exc:
        # Task 5.7: Graceful handling
        console.print(
            f"\n[bold red]⚠ Max iterations reached ({exc.iterations})[/bold red]\n"
            "[yellow]The coding agent hit its iteration limit. "
            "Try breaking the task into smaller steps.[/yellow]"
        )
    except Exception as e:
        console.print(f"[red]Error in code mode: {e}[/red]")
        logger.exception("Code mode error")


def _repl_confirm(tool_name: str, args: dict[str, Any]) -> bool:
    """Rich-based confirmation prompt for destructive tool operations in REPL (Task 5.3)."""
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


def _show_repl_coding_summary(result: AgentResult) -> None:
    """Display tool execution feedback and token usage in REPL (Tasks 5.5 + 5.6)."""
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
            tool_args = tool.get("args", {})
            error = tool.get("error", False)

            target = tool_args.get("path", tool_args.get("command", tool_args.get("pattern", "")))
            if len(str(target)) > 60:
                target = str(target)[:57] + "..."

            status = "[red]✗[/red]" if error else "[green]✓[/green]"
            table.add_row(str(i), name, str(target), status)

        console.print(table)

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
# Slash Command Handlers
# ══════════════════════════════════════════════════════════════
def _cmd_add(state: REPLState, args: str) -> None:
    """Add files or directories to context."""
    if not args:
        console.print("[yellow]Usage: /add <path> [path2 ...][/yellow]")
        return

    for path_str in args.split():
        path = Path(path_str).expanduser().resolve()

        try:
            if path.is_dir():
                count = state.context_builder.add_directory(path)
                console.print(f"[green]✓ Added {count} files from {path}[/green]")
            elif path.is_file():
                loaded = state.context_builder.add_file(path)
                console.print(f"[green]✓ Added {loaded.path} ({loaded.file_type.value})[/green]")
            else:
                console.print(f"[red]Not found: {path}[/red]")
        except Exception as e:
            console.print(f"[red]Failed to add {path}: {e}[/red]")

    state.context_builder.show_summary()


def _cmd_code(state: REPLState) -> None:
    """Toggle code mode on/off (Task 5.2)."""
    state.code_mode = not state.code_mode

    if state.code_mode:
        workspace = Path(state.settings.coding.workspace_root).resolve()
        console.print(
            Panel.fit(
                "[bold yellow]🔧 Code Mode ON[/bold yellow]\n"
                f"[dim]Workspace: {workspace}[/dim]\n"
                f"[dim]Max iterations: {state.settings.coding.max_tool_iterations}[/dim]\n"
                "[dim]The agent can now read, write, edit files and run commands.[/dim]\n"
                "[dim]Streaming is disabled in code mode.[/dim]",
                border_style="yellow",
            )
        )
    else:
        console.print("[green]✓ Code mode OFF — back to normal chat[/green]")


def _cmd_model(state: REPLState, args: str) -> None:
    """Switch or show the current model."""
    if not args:
        console.print(f"[cyan]Current model: {state.model}[/cyan]")
        models = state.client.list_available_models()
        if models:
            console.print("[dim]Available:[/dim]")
            for m in models:
                marker = " ← current" if m["id"] == state.model else ""
                console.print(f"  [dim]{m['id']}{marker}[/dim]")
        return

    old = state.model
    state.client.switch_model(args.strip())
    console.print(f"[green]✓ Model: {old} → {state.model}[/green]")


def _cmd_skill(state: REPLState, args: str) -> None:
    """Activate or deactivate a skill."""
    if not args:
        if state.active_skill:
            console.print(f"[cyan]Active skill: {state.skill_name} ({state.current_phase.value})[/cyan]")
        else:
            console.print("[dim]No active skill. Use /skill <name> to activate.[/dim]")

        skills = state.skill_registry.list_skills()
        if skills:
            console.print("[dim]Available skills:[/dim]")
            for meta in skills:
                console.print(f"  [dim]{meta.name} — {meta.description}[/dim]")
        return

    name = args.strip()

    if name == "off" or name == "none":
        state.active_skill = None
        state.current_phase = SkillPhase.ANALYZE
        console.print("[green]✓ Skill deactivated[/green]")
        return

    skill = state.skill_registry.get(name)
    if not skill:
        console.print(f"[red]Skill not found: {name}[/red]")
        console.print(f"[dim]Available: {', '.join(state.skill_registry.list_names())}[/dim]")
        return

    state.active_skill = skill
    state.current_phase = SkillPhase.ANALYZE
    meta = skill.get_metadata()
    console.print(f"[green]✓ Activated: {meta.display_name}[/green]")
    console.print(f"[dim]Phases: {', '.join(p.value for p in meta.supported_phases)}[/dim]")


def _cmd_phase(state: REPLState, args: str) -> None:
    """Switch the skill execution phase."""
    if not state.active_skill:
        console.print("[yellow]No active skill. Activate one with /skill <name>[/yellow]")
        return

    if not args:
        meta = state.active_skill.get_metadata()
        console.print(f"[cyan]Current phase: {state.current_phase.value}[/cyan]")
        console.print(f"[dim]Available: {', '.join(p.value for p in meta.supported_phases)}[/dim]")
        return

    try:
        phase = SkillPhase(args.strip())
        state.current_phase = phase
        console.print(f"[green]✓ Phase: {phase.value}[/green]")
    except ValueError:
        console.print(f"[red]Invalid phase: {args}[/red]")
        console.print(f"[dim]Valid: {', '.join(p.value for p in SkillPhase)}[/dim]")


def _cmd_agents(state: REPLState) -> None:
    """Show currently loaded agents."""
    agents = state.orchestrator.list_agents()
    if not agents:
        console.print("[dim]No agents loaded. Agents are created when using a skill.[/dim]")
        return

    table = Table(title="🤖 Active Agents", show_lines=False)
    table.add_column("Name", style="cyan")
    table.add_column("Role", style="bold")

    for name in agents:
        agent = state.orchestrator.get_agent(name)
        if agent:
            table.add_row(agent.name, agent.role)

    console.print(table)


def _cmd_sessions(state: REPLState) -> None:
    """List recent sessions."""
    sessions = state.session_manager.list_sessions(limit=10)

    if not sessions:
        console.print("[dim]No saved sessions.[/dim]")
        return

    table = Table(title="📋 Recent Sessions", show_lines=False)
    table.add_column("ID", style="cyan", max_width=12)
    table.add_column("Name", style="bold")
    table.add_column("Created", style="dim")

    for s in sessions:
        if not isinstance(s, dict):
            continue
        table.add_row(s.get("id", "?")[:12], s.get("name", "—"), s.get("created_at", "—"))

    console.print(table)
    console.print("[dim]Use /load <id> to resume a session[/dim]")


def _cmd_new_session(state: REPLState, args: str) -> None:
    """Start a new session."""
    name = args.strip() or "chat"
    state.session_manager.new_session(name, model=state.model, skill=state.skill_name)
    state.context_builder.clear()
    state.orchestrator.reset_agents()
    console.print(f"[green]✓ New session: {name}[/green]")


def _cmd_load_session(state: REPLState, args: str) -> None:
    """Load an existing session."""
    if not args:
        console.print("[yellow]Usage: /load <session_id>[/yellow]")
        return

    session_id = args.strip()
    loaded = state.session_manager.load_session(session_id)
    if loaded:
        console.print(f"[green]✓ Loaded: {loaded.name} ({len(loaded.history)} messages)[/green]")
    else:
        console.print(f"[red]Session not found: {session_id}[/red]")


def _cmd_clear(state: REPLState) -> None:
    """Clear context and history."""
    state.context_builder.clear()
    state.session_manager.clear_history()
    state.orchestrator.reset_agents()
    state.code_mode = False
    console.print("[green]✓ Cleared context, history, agent states, and code mode[/green]")


def _cmd_context(state: REPLState) -> None:
    """Show current context."""
    state.context_builder.show_summary()


def _cmd_help() -> None:
    """Show help for all slash commands."""
    help_text = """
[bold cyan]Chat Commands[/bold cyan]
  Just type your message to chat with the AI.

[bold cyan]Slash Commands[/bold cyan]
  [cyan]/add <path>[/cyan]      — Add files or directories as context
  [cyan]/code[/cyan]            — Toggle coding agent mode (read/write/edit files)
  [cyan]/model [id][/cyan]      — Show or switch the current model
  [cyan]/skill [name][/cyan]    — Show, activate, or deactivate a skill (use 'off' to deactivate)
  [cyan]/phase [phase][/cyan]   — Show or switch the skill phase (analyze, plan, execute, validate, report)
  [cyan]/agents[/cyan]          — Show currently loaded agents
  [cyan]/sessions[/cyan]        — List recent sessions
  [cyan]/new [name][/cyan]      — Start a new session
  [cyan]/load <id>[/cyan]       — Resume an existing session
  [cyan]/clear[/cyan]           — Clear context, history, and agent states
  [cyan]/context[/cyan]         — Show loaded context files
  [cyan]/help[/cyan]            — Show this help
  [cyan]/quit[/cyan]            — Exit the REPL

[bold cyan]Tips[/bold cyan]
  • Add files before asking questions: [dim]/add src/ logs.txt[/dim]
  • Use skills for specialized tasks: [dim]/skill rca[/dim] then describe the incident
  • Switch models anytime: [dim]/model gemini-2.5-flash[/dim]
  • Enable code mode for file operations: [dim]/code[/dim] then describe the task
"""
    console.print(Panel(help_text.strip(), title="📖 Help", border_style="bright_blue"))

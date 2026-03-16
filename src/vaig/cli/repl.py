"""Interactive REPL — prompt-toolkit based chat with slash commands."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from vaig.agents.orchestrator import Orchestrator
from vaig.context.builder import ContextBuilder
from vaig.core.client import GeminiClient, StreamResult
from vaig.core.config import Settings
from vaig.core.cost_tracker import BudgetStatus, CostTracker
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
    "/cache",
    "/cluster",
    "/code",
    "/config",
    "/cost",
    "/location",
    "/model",
    "/project",
    "/skill",
    "/phase",
    "/agents",
    "/sessions",
    "/new",
    "/load",
    "/resume",
    "/rename",
    "/search",
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
        self.cost_tracker: CostTracker = CostTracker()

    @property
    def model(self) -> str:
        return self.client.current_model

    @property
    def skill_name(self) -> str | None:
        if self.active_skill:
            return self.active_skill.get_metadata().name
        return None

    @property
    def prompt_prefix(self) -> str:
        """Build the prompt prefix showing current state."""
        project = self.settings.gcp.project_id
        parts: list[str] = []
        if project:
            parts.append(f"[{project}]")
        parts.append(f"[{self.model}]")
        if self.code_mode:
            parts.append("(🔧code)")
        if self.active_skill:
            parts.append(f"({self.skill_name}:{self.current_phase.value})")
        if self.context_builder.bundle.file_count > 0:
            parts.append(f"📁{self.context_builder.bundle.file_count}")
        return " ".join(parts) + " > "


def _record_cost(
    cost_tracker: CostTracker,
    model_id: str,
    usage: dict[str, int] | None,
) -> None:
    """Safely record an API call's cost from a usage dict.

    Handles ``None`` or empty usage gracefully — no-op when there's
    nothing to record.
    """
    if not usage:
        return

    prompt_tokens = usage.get("prompt_tokens", 0) or usage.get("prompt_token_count", 0)
    completion_tokens = usage.get("completion_tokens", 0) or usage.get("candidates_token_count", 0)
    thinking_tokens = usage.get("thinking_tokens", 0) or usage.get("thoughts_token_count", 0)

    if prompt_tokens or completion_tokens or thinking_tokens:
        cost_tracker.record(model_id, prompt_tokens, completion_tokens, thinking_tokens)


def _check_budget(state: REPLState) -> bool:
    """Check budget before an API call.

    Returns ``True`` if the call should proceed, ``False`` if it should
    be blocked.  Prints warnings/errors to the console as appropriate.
    """
    budget_config = state.settings.budget
    if not budget_config.enabled:
        return True

    status, message = state.cost_tracker.check_budget(budget_config)

    if status == BudgetStatus.WARNING:
        console.print(f"[bold yellow]⚠ {message}[/bold yellow]")
        return True

    if status == BudgetStatus.EXCEEDED:
        console.print(f"[bold red]🚫 {message}[/bold red]")
        if budget_config.action == "stop":
            console.print("[red]Budget exceeded — request blocked. Use /cost to see details.[/red]")
            return False
        # action == "warn" — let it proceed with a warning
        console.print("[yellow]Proceeding despite budget exceeded (action=warn).[/yellow]")
        return True

    return True


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

    # Eagerly initialize the telemetry collector with settings so that
    # downstream modules (agents, session, cost_tracker) that call
    # get_telemetry_collector() without args get the pre-warmed singleton
    # instead of falling back to get_settings().
    try:
        from vaig.core.telemetry import get_telemetry_collector

        get_telemetry_collector(settings)
    except Exception:  # noqa: BLE001
        pass

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

    # Start prompt loop — use FileHistory so arrow-up recalls commands
    # across sessions.
    history_path = Path(settings.session.repl_history_path).expanduser()
    history_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_path)),
        completer=command_completer,
        style=PROMPT_STYLE,
    )

    # ── Load cost data if resuming a session ────────────────
    if session_id:
        cost_data = session_manager.load_cost_data(session_id)
        if cost_data:
            state.cost_tracker = CostTracker.from_dict(cost_data)
            console.print(f"[dim]Restored cost tracking: {state.cost_tracker.request_count} prior API calls[/dim]")

    try:
        _repl_loop(prompt_session, state)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
    finally:
        _show_session_cost_summary(state)
        _save_cost_data(state)
        session_manager.close()
        console.print("[dim]Session saved. Goodbye! 👋[/dim]")


# ══════════════════════════════════════════════════════════════
# Async REPL
# ══════════════════════════════════════════════════════════════


async def async_start_repl(
    settings: Settings,
    *,
    skill_name: str | None = None,
    session_id: str | None = None,
    session_name: str = "chat",
) -> None:
    """Start the interactive REPL loop using async I/O.

    Async counterpart of :func:`start_repl`.  Uses
    ``PromptSession.prompt_async()`` for non-blocking input and
    async agent / session methods throughout.

    All existing REPL features are preserved: FileHistory, completions,
    toolbar, slash commands, budget checking, skill routing, code mode,
    and cost tracking.
    """
    # Initialize components (same as sync start_repl)
    client = GeminiClient(settings)
    orchestrator = Orchestrator(client, settings)
    session_manager = SessionManager(settings)
    context_builder = ContextBuilder(settings)
    skill_registry = SkillRegistry(settings)

    # Eagerly initialize the telemetry collector
    try:
        from vaig.core.telemetry import get_telemetry_collector

        get_telemetry_collector(settings)
    except Exception:  # noqa: BLE001
        pass

    state = REPLState(
        settings=settings,
        client=client,
        orchestrator=orchestrator,
        session_manager=session_manager,
        context_builder=context_builder,
        skill_registry=skill_registry,
    )

    # Load or create session (async)
    if session_id:
        loaded = await session_manager.async_load_session(session_id)
        if loaded:
            console.print(f"[green]✓ Resumed session: {loaded.name} ({len(loaded.history)} messages)[/green]")
        else:
            console.print(f"[red]Session not found: {session_id}. Starting new session.[/red]")
            await session_manager.async_new_session(session_name, model=settings.models.default)
    else:
        await session_manager.async_new_session(session_name, model=settings.models.default, skill=skill_name)
        console.print(f"[green]✓ New session: {session_name}[/green]")

    # Activate skill if requested
    if skill_name:
        _cmd_skill(state, skill_name)

    # Print help hint
    console.print("[dim]Type /help for commands. Ctrl+D or /quit to exit.[/dim]\n")

    # FileHistory for cross-session arrow-up recall
    history_path = Path(settings.session.repl_history_path).expanduser()
    history_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_path)),
        completer=command_completer,
        style=PROMPT_STYLE,
    )

    # Restore cost data if resuming
    if session_id:
        cost_data = await session_manager.async_load_cost_data(session_id)
        if cost_data:
            state.cost_tracker = CostTracker.from_dict(cost_data)
            console.print(f"[dim]Restored cost tracking: {state.cost_tracker.request_count} prior API calls[/dim]")

    try:
        await _async_repl_loop(prompt_session, state)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
    finally:
        _show_session_cost_summary(state)
        await _async_save_cost_data(state)
        await session_manager.async_close()
        console.print("[dim]Session saved. Goodbye! 👋[/dim]")


async def _async_repl_loop(prompt_session: PromptSession[str], state: REPLState) -> None:
    """Async main REPL input loop.

    Uses ``prompt_async()`` for non-blocking input so the event loop
    stays available for concurrent tasks.
    """
    while True:
        try:
            user_input = (await prompt_session.prompt_async(state.prompt_prefix)).strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            console.print("[yellow]^C[/yellow]")
            continue

        if not user_input:
            continue

        # Handle slash commands (sync — they're lightweight)
        if user_input.startswith("/"):
            should_quit = _handle_command(state, user_input)
            if should_quit:
                break
            continue

        # Regular chat message (async)
        await _async_handle_chat(state, user_input)


async def _async_handle_chat(state: REPLState, user_input: str) -> None:
    """Async version of :func:`_handle_chat`."""
    context_str = ""
    if state.context_builder.bundle.file_count > 0:
        context_str = state.context_builder.bundle.to_context_string()

    # Record user message (async)
    await state.session_manager.async_add_message("user", user_input)

    if state.code_mode:
        await _async_handle_code_chat(state, user_input, context_str)
    elif state.active_skill:
        await _async_handle_skill_chat(state, user_input, context_str)
    else:
        # Auto-route to a skill if confidence is high enough
        auto_skill = _try_auto_route_skill(state, user_input)
        if auto_skill:
            await _async_handle_skill_chat(state, user_input, context_str)
            # Clear auto-routed skill after use — per-message, not sticky
            state.active_skill = None
            state.current_phase = SkillPhase.ANALYZE
        else:
            await _async_handle_direct_chat(state, user_input, context_str)


async def _async_handle_direct_chat(state: REPLState, user_input: str, context: str) -> None:
    """Async version of :func:`_handle_direct_chat`."""
    if not _check_budget(state):
        return

    try:
        # Chunked processing for oversized context (sync — CPU-bound)
        if context and _try_chunked_chat(state, user_input, context):
            return

        if state.stream_enabled:
            status = console.status(
                f"[bold cyan]Thinking ({state.model})...[/bold cyan]"
            )
            status.start()

            stream = await state.orchestrator.async_execute_single(
                user_input, context=context, stream=True,
            )
            response_parts: list[str] = []
            async for chunk in stream:  # type: ignore[union-attr]
                if status is not None:
                    status.stop()
                    status = None  # type: ignore[assignment]
                console.print(chunk, end="")
                response_parts.append(chunk)

            # Edge case: stream yielded nothing — stop the spinner
            if status is not None:
                status.stop()

            console.print()
            full_response = "".join(response_parts)

            # Record cost from stream usage
            if isinstance(stream, StreamResult):
                _record_cost(state.cost_tracker, stream.model, stream.usage)
        else:
            with console.status(
                f"[bold cyan]Generating response with {state.model}...[/bold cyan]"
            ):
                result = await state.orchestrator.async_execute_single(
                    user_input, context=context,
                )
            full_response = result.content  # type: ignore[union-attr]
            console.print(Markdown(full_response))

            # Record cost from non-streaming result
            _record_cost(state.cost_tracker, state.model, result.usage)  # type: ignore[union-attr]

        # Record model response (async)
        await state.session_manager.async_add_message("model", full_response, model=state.model)
        console.print()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Async chat error")
        try:
            from vaig.core.telemetry import get_telemetry_collector

            collector = get_telemetry_collector(state.settings)
            collector.emit_error(type(e).__name__, str(e), metadata={"source": "async_direct_chat"})
        except Exception:  # noqa: BLE001
            pass


async def _async_handle_skill_chat(state: REPLState, user_input: str, context: str) -> None:
    """Async version of :func:`_handle_skill_chat`."""
    if not _check_budget(state):
        return

    skill = state.active_skill
    if not skill:
        return

    meta = skill.get_metadata()
    console.print(f"[dim]Running {meta.display_name} → {state.current_phase.value} on {state.model}...[/dim]")

    try:
        with console.status(
            f"[bold cyan]{meta.display_name} ({state.current_phase.value}) on {state.model}...[/bold cyan]"
        ):
            result = await state.orchestrator.async_execute_skill_phase(
                skill,
                state.current_phase,
                context,
                user_input,
            )

        console.print()
        console.print(Markdown(result.output))
        console.print()

        # Record cost from skill execution
        total_usage = result.metadata.get("total_usage") if result.metadata else None
        if total_usage:
            _record_cost(state.cost_tracker, state.model, total_usage)

        # Record model response (async)
        await state.session_manager.async_add_message("model", result.output, model=state.model)

        # Suggest next phase
        if result.next_phase:
            console.print(f"[dim]Suggested next phase: /phase {result.next_phase.value}[/dim]")

    except Exception as e:
        console.print(f"[red]Error during {state.current_phase.value}: {e}[/red]")
        logger.exception("Async skill execution error")
        try:
            from vaig.core.telemetry import get_telemetry_collector

            collector = get_telemetry_collector(state.settings)
            collector.emit_error(type(e).__name__, str(e), metadata={"source": "async_skill_chat"})
        except Exception:  # noqa: BLE001
            pass


async def _async_handle_code_chat(state: REPLState, user_input: str, context: str) -> None:
    """Async version of :func:`_handle_code_chat`."""
    if not _check_budget(state):
        return

    from vaig.agents.coding import CodingAgent
    from vaig.core.exceptions import MaxIterationsError

    coding_config = state.settings.coding
    agent = CodingAgent(
        state.client,
        coding_config,
        settings=state.settings,
        confirm_fn=_repl_confirm,
        model_id=state.model,
    )

    try:
        console.print("[bold cyan]🤖 Coding agent working (async)...[/bold cyan]")
        result = await agent.async_execute(user_input, context=context)

        # Display final response
        console.print()
        console.print(Markdown(result.content))
        console.print()

        # Show tool feedback and usage summary
        _show_repl_coding_summary(result)

        # Record cost
        _record_cost(state.cost_tracker, state.model, result.usage)

        # Record model response (async)
        await state.session_manager.async_add_message("model", result.content, model=state.model)

    except MaxIterationsError as exc:
        console.print(
            f"\n[bold red]⚠ Max iterations reached ({exc.iterations})[/bold red]\n"
            "[yellow]The coding agent hit its iteration limit. "
            "Try breaking the task into smaller steps.[/yellow]"
        )
    except Exception as e:
        console.print(f"[red]Error in async code mode: {e}[/red]")
        logger.exception("Async code mode error")
        try:
            from vaig.core.telemetry import get_telemetry_collector

            collector = get_telemetry_collector(state.settings)
            collector.emit_error(type(e).__name__, str(e), metadata={"source": "async_code_chat"})
        except Exception:  # noqa: BLE001
            pass


async def _async_save_cost_data(state: REPLState) -> None:
    """Async version of :func:`_save_cost_data`."""
    if state.cost_tracker.request_count == 0:
        return

    try:
        saved = await state.session_manager.async_save_cost_data(state.cost_tracker.to_dict())
        if saved:
            logger.debug("Cost data persisted to session (async).")
        else:
            logger.debug("No active session — cost data not persisted (async).")
    except Exception:
        logger.exception("Failed to persist cost data (async)")


def _repl_loop(prompt_session: PromptSession[str], state: REPLState) -> None:
    """Main REPL input loop."""
    while True:
        try:
            user_input = prompt_session.prompt(state.prompt_prefix).strip()
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
        "/cache": lambda: _cmd_cache(state, args),
        "/cluster": lambda: _cmd_cluster(state, args),
        "/code": lambda: _cmd_code(state),
        "/config": lambda: _cmd_config(state),
        "/cost": lambda: _cmd_cost(state),
        "/location": lambda: _cmd_location(state, args),
        "/model": lambda: _cmd_model(state, args),
        "/project": lambda: _cmd_project(state, args),
        "/skill": lambda: _cmd_skill(state, args),
        "/phase": lambda: _cmd_phase(state, args),
        "/agents": lambda: _cmd_agents(state),
        "/sessions": lambda: _cmd_sessions(state),
        "/new": lambda: _cmd_new_session(state, args),
        "/load": lambda: _cmd_load_session(state, args),
        "/rename": lambda: _cmd_rename_session(state, args),
        "/search": lambda: _cmd_search_sessions(state, args),
        "/resume": lambda: _cmd_resume(state),
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
        # Auto-route to a skill if confidence is high enough
        auto_skill = _try_auto_route_skill(state, user_input)
        if auto_skill:
            _handle_skill_chat(state, user_input, context_str)
            # Clear auto-routed skill after use — it's per-message, not sticky
            state.active_skill = None
            state.current_phase = SkillPhase.ANALYZE
        else:
            # Direct chat
            _handle_direct_chat(state, user_input, context_str)


def _try_auto_route_skill(state: REPLState, user_input: str) -> bool:
    """Attempt to auto-route the query to a skill based on keyword matching.

    Only activates when ``skills.auto_routing`` is enabled in config and the
    best suggestion score exceeds ``skills.auto_routing_threshold``.

    The skill is set as ``active_skill`` for this single message — it's
    cleared after the chat handler completes so subsequent messages go
    through the same routing logic.

    Returns ``True`` if a skill was auto-selected (caller should use
    skill-based execution), ``False`` otherwise.
    """
    skills_config = state.settings.skills
    if not skills_config.auto_routing:
        return False

    suggestions = state.skill_registry.suggest_skill(user_input)
    if not suggestions:
        return False

    best_name, best_score = suggestions[0]
    threshold = skills_config.auto_routing_threshold

    if best_score < threshold:
        logger.debug(
            "Auto-routing: best skill %s scored %.2f (threshold %.2f) — skipping",
            best_name, best_score, threshold,
        )
        return False

    skill = state.skill_registry.get(best_name)
    if not skill:
        return False

    state.active_skill = skill
    state.current_phase = SkillPhase.ANALYZE
    console.print(
        f"[dim]🎯 Auto-routing to skill: [cyan]{best_name}[/cyan] "
        f"(score: {best_score:.1f})[/dim]"
    )
    logger.info("Auto-routed query to skill: %s (score: %.2f)", best_name, best_score)

    return True


def _handle_direct_chat(state: REPLState, user_input: str, context: str) -> None:
    """Handle direct chat without a skill."""
    if not _check_budget(state):
        return

    try:
        # ── Chunked processing for oversized context ──────────
        if context and _try_chunked_chat(state, user_input, context):
            return

        if state.stream_enabled:
            # Show a transient "thinking" indicator while waiting for the
            # first stream chunk.  We use a Live status that auto-clears
            # when we exit the context manager.
            status = console.status(
                f"[bold cyan]Thinking ({state.model})...[/bold cyan]"
            )
            status.start()

            stream = state.orchestrator.execute_single(user_input, context=context, stream=True)
            response_parts: list[str] = []
            for chunk in stream:  # type: ignore[union-attr]
                if status is not None:
                    status.stop()
                    status = None  # type: ignore[assignment]
                console.print(chunk, end="")
                response_parts.append(chunk)

            # Edge case: stream yielded nothing — stop the spinner
            if status is not None:
                status.stop()

            console.print()
            full_response = "".join(response_parts)

            # Record cost from stream usage (StreamResult captures usage on last chunk)
            if isinstance(stream, StreamResult):
                _record_cost(state.cost_tracker, stream.model, stream.usage)
        else:
            with console.status(
                f"[bold cyan]Generating response with {state.model}...[/bold cyan]"
            ):
                result = state.orchestrator.execute_single(user_input, context=context)
            full_response = result.content  # type: ignore[union-attr]
            console.print(Markdown(full_response))

            # Record cost from non-streaming result
            _record_cost(state.cost_tracker, state.model, result.usage)  # type: ignore[union-attr]

        # Record model response
        state.session_manager.add_message("model", full_response, model=state.model)
        console.print()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Chat error")
        try:
            from vaig.core.telemetry import get_telemetry_collector

            collector = get_telemetry_collector(state.settings)
            collector.emit_error(type(e).__name__, str(e), metadata={"source": "direct_chat"})
        except Exception:  # noqa: BLE001
            pass


def _try_chunked_chat(state: REPLState, user_input: str, context: str) -> bool:
    """Attempt chunked (Map-Reduce) processing for chat if context exceeds the window.

    Returns True if chunking was performed (caller should return early),
    False if content fits normally.
    """
    from vaig.agents.chunked import ChunkedProcessor

    processor = ChunkedProcessor(state.client, state.settings)

    system_instruction = state.orchestrator.default_system_instruction()

    try:
        budget = processor.calculate_budget(
            system_instruction,
            user_input,
            model_id=state.model,
        )
    except Exception:
        logger.debug("Chunked budget calculation failed, using normal pipeline", exc_info=True)
        return False

    if not processor.needs_chunking(context, budget):
        return False

    # ── Content exceeds context window — use Map-Reduce ───────
    chunks = processor.split_into_chunks(context, budget)
    total = len(chunks)

    console.print(
        f"\n[bold yellow]Large context detected[/bold yellow] — "
        f"splitting into [cyan]{total}[/cyan] chunks for analysis"
    )

    with console.status("[bold cyan]Analyzing chunks...[/bold cyan]") as status_ctx:

        def _on_progress(current: int, total: int) -> None:
            status_ctx.update(f"[bold cyan]Processing chunk {current}/{total}...[/bold cyan]")

        full_response = processor.process_ask(
            context,
            user_input,
            system_instruction,
            budget,
            model_id=state.model,
            on_progress=_on_progress,
        )

    console.print()
    console.print(Markdown(full_response))
    console.print()

    # Record model response
    state.session_manager.add_message("model", full_response, model=state.model)
    return True


def _handle_skill_chat(state: REPLState, user_input: str, context: str) -> None:
    """Handle chat with an active skill."""
    if not _check_budget(state):
        return

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

        # Record cost from skill execution (total_usage from OrchestratorResult)
        total_usage = result.metadata.get("total_usage") if result.metadata else None
        if total_usage:
            _record_cost(state.cost_tracker, state.model, total_usage)

        # Record model response
        state.session_manager.add_message("model", result.output, model=state.model)

        # Suggest next phase
        if result.next_phase:
            console.print(f"[dim]Suggested next phase: /phase {result.next_phase.value}[/dim]")

    except Exception as e:
        console.print(f"[red]Error during {state.current_phase.value}: {e}[/red]")
        logger.exception("Skill execution error")
        try:
            from vaig.core.telemetry import get_telemetry_collector

            collector = get_telemetry_collector(state.settings)
            collector.emit_error(type(e).__name__, str(e), metadata={"source": "skill_chat"})
        except Exception:  # noqa: BLE001
            pass


def _handle_code_chat(state: REPLState, user_input: str, context: str) -> None:
    """Handle chat in code mode — uses CodingAgent with tool-use loop.

    Implements Tasks 5.4 (routing), 5.5 (tool feedback), 5.6 (usage summary),
    5.7 (MaxIterationsError handling), 5.8 (no streaming).
    """
    if not _check_budget(state):
        return

    from vaig.agents.coding import CodingAgent
    from vaig.core.exceptions import MaxIterationsError

    coding_config = state.settings.coding
    agent = CodingAgent(
        state.client,
        coding_config,
        settings=state.settings,
        confirm_fn=_repl_confirm,
        model_id=state.model,
    )

    try:
        # Task 5.8: No streaming in code mode — tool loops are non-streamable
        # NOTE: No spinner wrapper here — confirm_fn needs interactive terminal access.
        # The agent prints confirmation prompts during execution, so a spinner would
        # swallow that output and freeze the terminal.
        console.print("[bold cyan]🤖 Coding agent working...[/bold cyan]")
        result = agent.execute(user_input, context=context)

        # Display final response
        console.print()
        console.print(Markdown(result.content))
        console.print()

        # Task 5.5 + 5.6: Show tool feedback and usage summary
        _show_repl_coding_summary(result)

        # Record cost from coding agent execution
        _record_cost(state.cost_tracker, state.model, result.usage)

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
        try:
            from vaig.core.telemetry import get_telemetry_collector

            collector = get_telemetry_collector(state.settings)
            collector.emit_error(type(e).__name__, str(e), metadata={"source": "code_chat"})
        except Exception:  # noqa: BLE001
            pass


def _repl_confirm(tool_name: str, args: dict[str, Any]) -> bool:
    """Confirm destructive tool operations — delegates to shared display helper."""
    from vaig.cli.display import confirm_tool_operation

    return confirm_tool_operation(tool_name, args, console=console)


def _show_repl_coding_summary(result: AgentResult) -> None:
    """Display tool execution feedback — delegates to shared display helper."""
    from vaig.cli.display import show_tool_execution_summary

    show_tool_execution_summary(result, console=console)


# ══════════════════════════════════════════════════════════════
# Slash Command Handlers
# ══════════════════════════════════════════════════════════════


def _cmd_project(state: REPLState, args: str) -> None:
    """Switch the active GCP project."""
    if not args:
        console.print(f"[cyan]Current project: {state.settings.gcp.project_id}[/cyan]")
        if state.settings.gcp.available_projects:
            console.print("[dim]Available projects:[/dim]")
            for p in state.settings.gcp.available_projects:
                marker = " ← current" if p.project_id == state.settings.gcp.project_id else ""
                desc = f" ({p.description})" if p.description else ""
                console.print(f"  [dim]{p.project_id}{desc}{marker}[/dim]")
        console.print("[dim]Usage: /project <project-id>[/dim]")
        return

    from vaig.core.config_switcher import switch_project

    result = switch_project(state.settings, args.strip(), client=state.client)

    if result.success:
        console.print(f"[green]✓ {result.message}[/green]")
        if result.reinitialized:
            console.print(f"[dim]Reinitialized: {', '.join(result.reinitialized)}[/dim]")
        # Warn about stale tools/agents
        console.print(
            "[yellow]Note: Tools and agents will use the new project on next creation. "
            "Use /clear to reset agents now.[/yellow]"
        )
    else:
        console.print(f"[red]✗ {result.message}[/red]")


def _cmd_location(state: REPLState, args: str) -> None:
    """Switch the active GCP location."""
    if not args:
        console.print(f"[cyan]Current location: {state.settings.gcp.location}[/cyan]")
        console.print("[dim]Usage: /location <location>  (e.g. us-central1, europe-west1)[/dim]")
        return

    from vaig.core.config_switcher import switch_location

    result = switch_location(state.settings, args.strip(), client=state.client)

    if result.success:
        console.print(f"[green]✓ {result.message}[/green]")
        if result.reinitialized:
            console.print(f"[dim]Reinitialized: {', '.join(result.reinitialized)}[/dim]")
    else:
        console.print(f"[red]✗ {result.message}[/red]")


def _cmd_cluster(state: REPLState, args: str) -> None:
    """Switch the active GKE cluster."""
    if not args:
        console.print(f"[cyan]Current cluster: {state.settings.gke.cluster_name}[/cyan]")
        if state.settings.gke.context:
            console.print(f"[dim]Context: {state.settings.gke.context}[/dim]")
        console.print("[dim]Usage: /cluster <name> [context][/dim]")
        return

    from vaig.core.config_switcher import switch_cluster

    parts = args.strip().split(maxsplit=1)
    cluster_name = parts[0]
    new_context = parts[1] if len(parts) > 1 else None

    result = switch_cluster(state.settings, cluster_name, new_context)

    if result.success:
        console.print(f"[green]✓ {result.message}[/green]")
        if result.reinitialized:
            console.print(f"[dim]Cleared caches: {', '.join(result.reinitialized)}[/dim]")
        console.print(
            "[yellow]Note: Infrastructure tools will use the new cluster on next invocation.[/yellow]"
        )
    else:
        console.print(f"[red]✗ {result.message}[/red]")


def _cmd_config(state: REPLState) -> None:
    """Show current config snapshot."""
    from vaig.core.config_switcher import get_config_snapshot

    snap = get_config_snapshot(state.settings)

    table = Table(title="⚙ Current Configuration", show_lines=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="bold")

    # Display labels that are more user-friendly than the raw dict keys
    labels = {
        "project": "GCP Project",
        "location": "GCP Location",
        "fallback_location": "Fallback Location",
        "model": "Model",
        "cluster": "GKE Cluster",
        "context": "Kube Context",
        "gke_project": "GKE Project",
        "gke_location": "GKE Location",
    }

    for key, value in snap.items():
        label = labels.get(key, key)
        display_value = value if value else "[dim]—[/dim]"
        table.add_row(label, display_value)

    console.print(table)


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

    table = Table(title="Recent Sessions", show_lines=False)
    table.add_column("ID", style="cyan", max_width=12)
    table.add_column("Name", style="bold")
    table.add_column("Model", style="magenta")
    table.add_column("Skill", style="green")
    table.add_column("Msgs", style="yellow", justify="right")
    table.add_column("Updated", style="dim")

    for s in sessions:
        if not isinstance(s, dict):
            continue
        updated = s.get("updated_at", "")
        if updated:
            try:
                from datetime import datetime

                dt = datetime.fromisoformat(updated)
                updated = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                updated = updated[:16]
        table.add_row(
            s.get("id", "?")[:12],
            s.get("name", "—"),
            s.get("model", "—"),
            s.get("skill", "—") or "—",
            str(s.get("message_count", 0)),
            updated or "—",
        )

    console.print(table)
    console.print("[dim]Use /load <id> to resume, /rename <id> <name>, /search <query>[/dim]")


def _cmd_new_session(state: REPLState, args: str) -> None:
    """Start a new session."""
    # Save current session's cost data before switching
    _save_cost_data(state)

    name = args.strip() or "chat"
    state.session_manager.new_session(name, model=state.model, skill=state.skill_name)
    state.context_builder.clear()
    state.orchestrator.reset_agents()
    state.cost_tracker = CostTracker()
    console.print(f"[green]✓ New session: {name}[/green]")


def _cmd_load_session(state: REPLState, args: str) -> None:
    """Load an existing session."""
    if not args:
        console.print("[yellow]Usage: /load <session_id>[/yellow]")
        return

    # Save current session's cost data before switching
    _save_cost_data(state)

    session_id = args.strip()
    loaded = state.session_manager.load_session(session_id)
    if loaded:
        # Restore cost tracker from the loaded session
        cost_data = state.session_manager.load_cost_data(session_id)
        if cost_data:
            state.cost_tracker = CostTracker.from_dict(cost_data)
            console.print(
                f"[green]Loaded: {loaded.name} ({len(loaded.history)} messages, "
                f"{state.cost_tracker.request_count} prior API calls)[/green]"
            )
        else:
            state.cost_tracker = CostTracker()
            console.print(f"[green]Loaded: {loaded.name} ({len(loaded.history)} messages)[/green]")
    else:
        console.print(f"[red]Session not found: {session_id}[/red]")


def _cmd_rename_session(state: REPLState, args: str) -> None:
    """Rename a session (current or by ID)."""
    parts = args.strip().split(maxsplit=1)
    if not parts:
        console.print("[yellow]Usage: /rename <new_name>  or  /rename <session_id> <new_name>[/yellow]")
        return

    if len(parts) == 1:
        # Rename current session
        new_name = parts[0]
        if not state.session_manager.has_active_session:
            console.print("[red]No active session to rename.[/red]")
            return
        active = state.session_manager.active
        assert active is not None
        if state.session_manager.rename_session(active.id, new_name):
            console.print(f"[green]Renamed to: {new_name}[/green]")
        else:
            console.print("[red]Failed to rename session.[/red]")
    else:
        # Rename by ID
        session_id, new_name = parts
        if state.session_manager.rename_session(session_id, new_name):
            console.print(f"[green]Renamed session {session_id[:12]} to: {new_name}[/green]")
        else:
            console.print(f"[red]Session not found: {session_id}[/red]")


def _cmd_search_sessions(state: REPLState, args: str) -> None:
    """Search sessions by name or content."""
    if not args:
        console.print("[yellow]Usage: /search <query>[/yellow]")
        return

    query = args.strip()
    sessions = state.session_manager.search_sessions(query)

    if not sessions:
        console.print(f"[dim]No sessions matching '{query}'.[/dim]")
        return

    table = Table(title=f"Search: '{query}'", show_lines=False)
    table.add_column("ID", style="cyan", max_width=12)
    table.add_column("Name", style="bold")
    table.add_column("Msgs", style="yellow", justify="right")
    table.add_column("Updated", style="dim")

    for s in sessions:
        if not isinstance(s, dict):
            continue
        updated = s.get("updated_at", "")
        if updated:
            try:
                from datetime import datetime

                dt = datetime.fromisoformat(updated)
                updated = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                updated = updated[:16]
        table.add_row(
            s.get("id", "?")[:12],
            s.get("name", "—"),
            str(s.get("message_count", 0)),
            updated or "—",
        )

    console.print(table)
    console.print("[dim]Use /load <id> to resume a session[/dim]")


def _cmd_resume(state: REPLState) -> None:
    """Resume the last active session."""
    # Save current session's cost data before switching
    _save_cost_data(state)

    loaded = state.session_manager.resume_last_session()
    if loaded:
        # Restore cost tracker from the resumed session
        cost_data = state.session_manager.load_cost_data(loaded.id)
        if cost_data:
            state.cost_tracker = CostTracker.from_dict(cost_data)
            console.print(
                f"[green]Resumed: {loaded.name} ({len(loaded.history)} messages, "
                f"{state.cost_tracker.request_count} prior API calls)[/green]"
            )
        else:
            state.cost_tracker = CostTracker()
            console.print(f"[green]Resumed: {loaded.name} ({len(loaded.history)} messages)[/green]")
    else:
        console.print("[yellow]No previous sessions found.[/yellow]")


def _cmd_clear(state: REPLState) -> None:
    """Clear context and history."""
    state.context_builder.clear()
    state.session_manager.clear_history()
    state.orchestrator.reset_agents()
    state.client.clear_cache()
    state.code_mode = False
    console.print("[green]✓ Cleared context, history, agent states, cache, and code mode[/green]")


def _cmd_cache(state: REPLState, args: str) -> None:
    """Show cache status, stats, or clear the cache.

    Usage:
        /cache          — show cache status and stats
        /cache clear    — clear all cached responses
    """
    sub = args.strip().lower()

    if sub == "clear":
        count = state.client.clear_cache()
        if count > 0:
            console.print(f"[green]✓ Cleared {count} cached response(s)[/green]")
        else:
            console.print("[dim]Cache is already empty (or caching is disabled).[/dim]")
        return

    # Default: show status + stats
    if not state.client.cache_enabled:
        console.print(
            "[dim]Response cache is disabled.[/dim]\n"
            "[dim]Enable via config: cache.enabled = true[/dim]"
        )
        return

    stats = state.client.cache_stats()
    if stats is None:
        return

    table = Table(title="Response Cache", show_lines=False, title_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold", justify="right")

    table.add_row("Status", "[green]enabled[/green]")
    table.add_row("Entries", f"{stats.size} / {stats.max_size}")
    table.add_row("Hits", str(stats.hits))
    table.add_row("Misses", str(stats.misses))
    table.add_row("Hit Rate", f"{stats.hit_rate:.1%}")
    table.add_row("Evictions", str(stats.evictions))

    console.print()
    console.print(table)
    console.print()


def _cmd_context(state: REPLState) -> None:
    """Show current context."""
    state.context_builder.show_summary()


def _cmd_cost(state: REPLState) -> None:
    """Show session cost summary with per-model breakdown."""
    from vaig.core.pricing import format_cost

    summary = state.cost_tracker.summary()

    if summary["request_count"] == 0:
        console.print("[dim]No API calls recorded yet in this session.[/dim]")
        return

    # Overall summary
    console.print()
    table = Table(title="💰 Session Cost Summary", show_lines=False, title_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold", justify="right")

    table.add_row("Total Cost", format_cost(summary["total_cost"]))
    table.add_row("API Calls", str(summary["request_count"]))
    table.add_row("Total Tokens", f"{summary['total_tokens']:,}")
    table.add_row("  Prompt", f"{summary['total_prompt_tokens']:,}")
    table.add_row("  Completion", f"{summary['total_completion_tokens']:,}")
    if summary["total_thinking_tokens"] > 0:
        table.add_row("  Thinking", f"{summary['total_thinking_tokens']:,}")

    console.print(table)

    # Per-model breakdown (if multiple models used)
    records = state.cost_tracker._records  # noqa: SLF001
    if records:
        model_costs: dict[str, dict[str, int | float]] = {}
        for rec in records:
            if rec.model_id not in model_costs:
                model_costs[rec.model_id] = {
                    "calls": 0,
                    "prompt": 0,
                    "completion": 0,
                    "thinking": 0,
                    "cost": 0.0,
                }
            m = model_costs[rec.model_id]
            m["calls"] = int(m["calls"]) + 1
            m["prompt"] = int(m["prompt"]) + rec.prompt_tokens
            m["completion"] = int(m["completion"]) + rec.completion_tokens
            m["thinking"] = int(m["thinking"]) + rec.thinking_tokens
            m["cost"] = float(m["cost"]) + rec.cost

        if len(model_costs) > 1:
            console.print()
            model_table = Table(title="Per-Model Breakdown", show_lines=False, title_style="dim bold")
            model_table.add_column("Model", style="magenta")
            model_table.add_column("Calls", justify="right")
            model_table.add_column("Tokens", justify="right")
            model_table.add_column("Cost", justify="right", style="bold")

            for model_id, data in sorted(model_costs.items()):
                total = int(data["prompt"]) + int(data["completion"]) + int(data["thinking"])
                model_table.add_row(
                    model_id,
                    str(int(data["calls"])),
                    f"{total:,}",
                    format_cost(float(data["cost"])),
                )

            console.print(model_table)

    # Budget status
    budget_config = state.settings.budget
    if budget_config.enabled:
        status, message = state.cost_tracker.check_budget(budget_config)
        if status == BudgetStatus.OK:
            remaining = budget_config.max_cost_usd - summary["total_cost"]
            console.print(
                f"\n[green]Budget: {format_cost(summary['total_cost'])} / "
                f"{format_cost(budget_config.max_cost_usd)} "
                f"({format_cost(remaining)} remaining)[/green]"
            )
        elif status == BudgetStatus.WARNING:
            console.print(f"\n[bold yellow]⚠ {message}[/bold yellow]")
        elif status == BudgetStatus.EXCEEDED:
            console.print(f"\n[bold red]🚫 {message}[/bold red]")

    console.print()


def _show_session_cost_summary(state: REPLState) -> None:
    """Display a brief cost summary when the REPL session ends."""
    from vaig.core.pricing import format_cost

    summary = state.cost_tracker.summary()
    if summary["request_count"] == 0:
        return

    console.print(
        f"\n[dim]Session cost: {format_cost(summary['total_cost'])} "
        f"| {summary['request_count']} API call{'s' if summary['request_count'] != 1 else ''} "
        f"| {summary['total_tokens']:,} tokens[/dim]"
    )


def _save_cost_data(state: REPLState) -> None:
    """Persist cost tracker data to the session store on exit."""
    if state.cost_tracker.request_count == 0:
        return

    try:
        saved = state.session_manager.save_cost_data(state.cost_tracker.to_dict())
        if saved:
            logger.debug("Cost data persisted to session.")
        else:
            logger.debug("No active session — cost data not persisted.")
    except Exception:
        logger.exception("Failed to persist cost data")


def _cmd_help() -> None:
    """Show help for all slash commands."""
    help_text = """
[bold cyan]Chat Commands[/bold cyan]
  Just type your message to chat with the AI.

[bold cyan]Slash Commands[/bold cyan]
  [cyan]/add <path>[/cyan]      — Add files or directories as context
  [cyan]/cache [clear][/cyan]   — Show cache stats or clear the response cache
  [cyan]/code[/cyan]            — Toggle coding agent mode (read/write/edit files)
  [cyan]/cost[/cyan]            — Show session cost summary and budget status
  [cyan]/model [id][/cyan]      — Show or switch the current model
  [cyan]/skill [name][/cyan]    — Show, activate, or deactivate a skill (use 'off' to deactivate)
  [cyan]/phase [phase][/cyan]   — Show or switch the skill phase (analyze, plan, execute, validate, report)
  [cyan]/agents[/cyan]          — Show currently loaded agents
  [cyan]/sessions[/cyan]        — List recent sessions
  [cyan]/new [name][/cyan]      — Start a new session
  [cyan]/load <id>[/cyan]       — Resume an existing session
  [cyan]/resume[/cyan]          — Resume the last active session
  [cyan]/rename [id] <name>[/cyan] — Rename current or specified session
  [cyan]/search <query>[/cyan]  — Search sessions by name or content
  [cyan]/clear[/cyan]           — Clear context, history, and agent states
  [cyan]/context[/cyan]         — Show loaded context files
  [cyan]/help[/cyan]            — Show this help
  [cyan]/quit[/cyan]            — Exit the REPL

[bold cyan]Config Commands[/bold cyan]
  [cyan]/project [id][/cyan]    — Show or switch the active GCP project
  [cyan]/location [name][/cyan] — Show or switch the GCP location (e.g. us-central1)
  [cyan]/cluster [name] [ctx][/cyan] — Show or switch the GKE cluster (optional kubeconfig context)
  [cyan]/config[/cyan]          — Show current configuration snapshot

[bold cyan]Tips[/bold cyan]
  • Add files before asking questions: [dim]/add src/ logs.txt[/dim]
  • Use skills for specialized tasks: [dim]/skill rca[/dim] then describe the incident
  • Switch models anytime: [dim]/model gemini-2.5-flash[/dim]
  • Enable code mode for file operations: [dim]/code[/dim] then describe the task
  • Resume your last session: [dim]/resume[/dim] or [dim]vaig chat --resume[/dim]
  • Switch projects at runtime: [dim]/project my-other-project[/dim]
"""
    console.print(Panel(help_text.strip(), title="📖 Help", border_style="bright_blue"))

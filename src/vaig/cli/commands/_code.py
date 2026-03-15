"""Code mode helpers — CodingAgent execution and chunked analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.markdown import Markdown
from rich.panel import Panel

from vaig.cli._helpers import (
    _build_cost_markdown_section,
    _cli_confirm,
    _compute_cost_str,
    _save_output,
    _show_coding_summary,
    console,
    err_console,
)

if TYPE_CHECKING:
    from vaig.core.client import GeminiClient
    from vaig.core.config import Settings

logger = logging.getLogger(__name__)


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


async def _async_execute_code_mode(
    client: "GeminiClient",
    settings: "Settings",
    question: str,
    context: str,
    *,
    output: Path | None = None,
) -> None:
    """Async version of :func:`_execute_code_mode`.

    Uses ``CodingAgent.async_execute()`` for non-blocking tool loops.
    The confirmation callback still runs synchronously (prompt_toolkit
    limitation in non-async REPL context), but all agent I/O is async.
    """
    from vaig.agents.coding import CodingAgent
    from vaig.core.exceptions import MaxIterationsError

    coding_config = settings.coding

    console.print(
        Panel.fit(
            "[bold yellow]🔧 Code Mode (async)[/bold yellow]\n"
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
        console.print("[bold cyan]🤖 Coding agent working (async)...[/bold cyan]")
        result = await agent.async_execute(question, context=context)

        # Display final response
        console.print()
        if result.content:
            console.print(Markdown(result.content))
        console.print()

        if output:
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

        _show_coding_summary(result)

    except MaxIterationsError as exc:
        err_console.print(
            f"\n[bold red]⚠ Max iterations reached ({exc.iterations})[/bold red]\n"
            "[yellow]The coding agent hit its iteration limit. "
            "This usually means the task is too complex for a single turn.\n"
            "Try breaking it into smaller steps.[/yellow]"
        )
        raise typer.Exit(1)  # noqa: B904


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


async def _async_try_chunked_ask(
    client: "GeminiClient",
    settings: "Settings",
    question: str,
    context_str: str,
    *,
    model_id: str | None = None,
    output: Path | None = None,
) -> bool:
    """Async version of :func:`_try_chunked_ask`.

    Currently wraps the sync ChunkedProcessor via ``asyncio.to_thread()``
    since ``ChunkedProcessor`` doesn't have async methods yet (Phase 5).

    Returns True if chunking was performed, False if content fits normally.
    """
    import asyncio

    from vaig.agents.chunked import ChunkedProcessor
    from vaig.agents.orchestrator import Orchestrator

    processor = ChunkedProcessor(client, settings)
    orchestrator = Orchestrator(client, settings)
    system_instruction = orchestrator.default_system_instruction()

    try:
        budget = await asyncio.to_thread(
            processor.calculate_budget,
            system_instruction,
            question,
            model_id=model_id or settings.models.default,
        )
    except Exception as exc:
        err_console.print(f"[dim]Chunking budget calculation failed ({exc}), using normal pipeline[/dim]")
        return False

    if not processor.needs_chunking(context_str, budget):
        return False

    chunks = processor.split_into_chunks(context_str, budget)
    total = len(chunks)

    console.print(
        f"\n[bold yellow]Large file detected[/bold yellow] — "
        f"splitting into [cyan]{total}[/cyan] chunks for analysis (async)"
    )

    def _run_chunked() -> str:
        return processor.process_ask(
            context_str,
            question,
            system_instruction,
            budget,
            model_id=model_id or settings.models.default,
        )

    result = await asyncio.to_thread(_run_chunked)

    console.print()
    if result:
        console.print(Markdown(result))
    if output:
        _save_output(output, result)
    return True

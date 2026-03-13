"""Shared CLI display helpers — confirmation prompts and execution summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from vaig.agents.base import AgentResult

# Shared console instance — callers may pass their own via keyword arg.
_default_console = Console()


def confirm_tool_operation(
    tool_name: str,
    args: dict[str, Any],
    *,
    console: Console | None = None,
) -> bool:
    """Rich-based confirmation prompt for destructive tool operations.

    Args:
        tool_name: Name of the tool about to execute.
        args: Tool arguments dict.
        console: Optional Rich Console; defaults to module-level instance.

    Returns:
        ``True`` if the user approves, ``False`` otherwise.
    """
    con = console or _default_console

    if tool_name == "write_file":
        desc = f"Write file: [cyan]{args.get('path', '?')}[/cyan]"
    elif tool_name == "edit_file":
        desc = f"Edit file: [cyan]{args.get('path', '?')}[/cyan]"
    elif tool_name == "run_command":
        desc = f"Run command: [cyan]{args.get('command', '?')}[/cyan]"
    else:
        desc = f"Execute: [cyan]{tool_name}[/cyan]"

    con.print(f"\n[bold yellow]⚡ {desc}[/bold yellow]")
    return typer.confirm("  Allow this operation?", default=True)


def show_tool_execution_summary(
    result: AgentResult,
    *,
    console: Console | None = None,
) -> None:
    """Display tool execution feedback and token usage.

    Args:
        result: The ``AgentResult`` containing metadata and usage info.
        console: Optional Rich Console; defaults to module-level instance.
    """
    con = console or _default_console
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

        con.print(table)

    # Usage summary
    usage = result.usage or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    # Cost estimation
    from vaig.core.pricing import calculate_cost, format_cost

    model_id = metadata.get("model", "") if metadata else ""
    cost = calculate_cost(model_id, prompt_tokens, completion_tokens)
    cost_str = format_cost(cost)

    con.print(
        f"[dim]Completed in {iterations} iteration{'s' if iterations != 1 else ''} "
        f"| Tokens: {total_tokens:,} total "
        f"({prompt_tokens:,} prompt + {completion_tokens:,} completion)"
        f" | Cost: {cost_str}[/dim]"
    )

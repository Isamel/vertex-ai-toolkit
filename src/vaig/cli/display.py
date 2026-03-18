"""Shared CLI display helpers — confirmation prompts and execution summaries."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from vaig.agents.base import AgentResult
    from vaig.skills.service_health.schema import HealthReport

# Shared console instance — callers may pass their own via keyword arg.
_default_console = Console()

# ── Severity coloring ────────────────────────────────────────

# Mapping: pattern → Rich markup style.  Order matters — first match wins
# for overlapping words (e.g. "OK" must not match inside "TOKEN").
_SEVERITY_RULES: list[tuple[re.Pattern[str], str]] = [
    # Bold colors (highest / positive)
    (re.compile(r"\bCRITICAL\b", re.IGNORECASE), "bold red"),
    (re.compile(r"\bHEALTHY\b", re.IGNORECASE), "bold green"),
    (re.compile(r"\bPASSED\b", re.IGNORECASE), "bold green"),
    # Red
    (re.compile(r"\bHIGH\b", re.IGNORECASE), "red"),
    (re.compile(r"\bERROR\b", re.IGNORECASE), "red"),
    # Yellow
    (re.compile(r"\bWARNING\b", re.IGNORECASE), "yellow"),
    (re.compile(r"\bWARN\b", re.IGNORECASE), "yellow"),
    (re.compile(r"\bMEDIUM\b", re.IGNORECASE), "yellow"),
    # Green
    (re.compile(r"\bLOW\b", re.IGNORECASE), "green"),
    (re.compile(r"\bINFO\b", re.IGNORECASE), "green"),
    # OK must use word-boundary carefully to avoid false positives
    (re.compile(r"\bOK\b"), "bold green"),
]


def _line_has_severity(line: str) -> bool:
    """Return True if the line contains at least one severity keyword."""
    return any(pat.search(line) for pat, _ in _SEVERITY_RULES)


def colorize_severity(text: str) -> str:
    """Post-process *text* to wrap severity keywords in Rich markup.

    Uses word-boundary-aware regexes so keywords inside other words
    (e.g. "TOKEN", "ALLOW") are not touched.  Already-marked Rich
    tags (``[bold red]...[/...]``) are skipped by only matching text
    that is NOT inside square brackets.

    Returns:
        The text with Rich-markup-wrapped severity keywords.
    """
    for pat, style in _SEVERITY_RULES:
        text = pat.sub(lambda m, s=style: f"[{s}]{m.group(0)}[/{s}]", text)  # type: ignore[misc]
    return text


def print_colored_report(
    text: str,
    *,
    console: Console | None = None,
) -> None:
    """Print a report with severity keywords colorized.

    Lines containing severity keywords are printed with Rich markup
    (plain text, not Markdown) so the colors render.  All other lines
    are accumulated and flushed as Markdown blocks to preserve
    headings, tables, code fences, etc.

    **Table & code-fence awareness**: when inside a multi-line Markdown
    structure (table or fenced code block), lines are ALWAYS kept in the
    Markdown buffer — severity coloring is never applied inside these
    structures so that Rich can render them correctly.

    Args:
        text: Raw Markdown report text (e.g. from an agent response).
        console: Optional Rich Console; defaults to module-level instance.
    """
    con = console or _default_console
    md_buffer: list[str] = []

    def _flush_md() -> None:
        """Flush accumulated Markdown lines."""
        if md_buffer:
            block = "\n".join(md_buffer)
            con.print(Markdown(block))
            md_buffer.clear()

    in_table = False
    in_code_fence = False

    for line in text.splitlines():
        stripped = line.strip()

        # Track code fence state (``` or ~~~, optionally with language tag)
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_fence = not in_code_fence
            md_buffer.append(line)
            continue

        # Inside code fence — never apply severity coloring
        if in_code_fence:
            md_buffer.append(line)
            continue

        # Detect table lines (starts with | and has at least one more |)
        is_table_line = stripped.startswith("|") and "|" in stripped[1:]

        if is_table_line:
            if not in_table:
                in_table = True
            md_buffer.append(line)
            continue

        # Exiting table — a non-table line after table rows
        if in_table and not is_table_line:
            in_table = False

        # Normal line processing — severity coloring only outside structures
        if _line_has_severity(line):
            _flush_md()
            con.print(colorize_severity(line))
        else:
            md_buffer.append(line)

    _flush_md()


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
    thinking_tokens = usage.get("thinking_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    # Cost estimation (including thinking tokens)
    from vaig.core.pricing import calculate_cost, format_cost

    model_id = metadata.get("model", "") if metadata else ""
    cost = calculate_cost(model_id, prompt_tokens, completion_tokens, thinking_tokens)
    cost_str = format_cost(cost)

    # Build token breakdown string
    token_parts = [f"{prompt_tokens:,} prompt", f"{completion_tokens:,} completion"]
    if thinking_tokens:
        token_parts.append(f"{thinking_tokens:,} thinking")

    con.print(
        f"[dim]Completed in {iterations} iteration{'s' if iterations != 1 else ''} "
        f"| Tokens: {total_tokens:,} total "
        f"({' + '.join(token_parts)})"
        f" | Cost: {cost_str}[/dim]"
    )


def show_cost_summary(
    usage_metadata: dict[str, int] | None,
    model_id: str,
    *,
    console: Console | None = None,
) -> None:
    """Display a compact cost/token summary line.

    Designed for one-shot CLI commands (``ask``, ``live``) where the full
    tool-execution table from :func:`show_tool_execution_summary` isn't
    needed — just the token counts and estimated cost.

    Args:
        usage_metadata: Dict with keys like ``prompt_tokens``,
            ``completion_tokens``, ``thinking_tokens``, ``total_tokens``.
            If ``None`` or empty the call is a silent no-op.
        model_id: Model identifier used for cost calculation.
        console: Optional Rich Console; defaults to module-level instance.
    """
    if not usage_metadata:
        return

    con = console or _default_console

    prompt_tokens = usage_metadata.get("prompt_tokens", 0)
    completion_tokens = usage_metadata.get("completion_tokens", 0)
    thinking_tokens = usage_metadata.get("thinking_tokens", 0)
    total_tokens = usage_metadata.get("total_tokens", 0)

    if total_tokens == 0 and prompt_tokens == 0 and completion_tokens == 0:
        return

    from vaig.core.pricing import calculate_cost, format_cost

    cost = calculate_cost(model_id, prompt_tokens, completion_tokens, thinking_tokens)
    cost_str = format_cost(cost)

    # Build compact token breakdown
    parts = [f"{prompt_tokens:,} in", f"{completion_tokens:,} out"]
    if thinking_tokens:
        parts.append(f"{thinking_tokens:,} thinking")

    con.print(
        f"[dim]📊 Tokens: {' / '.join(parts)} "
        f"({total_tokens:,} total) │ Cost: {cost_str}[/dim]"
    )


# ── Executive Summary Panel ──────────────────────────────────

# Maps OverallStatus → (Rich border style, emoji)
_STATUS_PANEL_STYLE: dict[str, tuple[str, str]] = {
    "HEALTHY": ("green", "🟢"),
    "DEGRADED": ("yellow", "🟡"),
    "CRITICAL": ("red", "🔴"),
    "UNKNOWN": ("dim", "⚪"),
}


def print_executive_summary_panel(
    report: HealthReport,
    *,
    console: Console | None = None,
) -> None:
    """Render the executive summary as a Rich Panel with severity-colored border.

    Displayed BEFORE the full Markdown report to give an immediate visual
    overview of the cluster health status.

    Args:
        report: A parsed ``HealthReport`` instance.
        console: Optional Rich Console; defaults to module-level instance.
    """
    con = console or _default_console
    es = report.executive_summary
    status_key = es.overall_status.value
    border_style, emoji = _STATUS_PANEL_STYLE.get(status_key, ("dim", "❓"))

    body = Text()
    body.append(f"{emoji} Status: ", style="bold")
    body.append(f"{status_key}\n", style=f"bold {border_style}")
    body.append(f"📋 Scope: {es.scope}\n")
    body.append(
        f"🔍 Issues: {es.issues_found} "
        f"({es.critical_count} critical, {es.warning_count} warning)\n"
    )
    body.append(f"\n{es.summary_text}")

    panel = Panel(
        body,
        title="Executive Summary",
        border_style=border_style,
        padding=(1, 2),
    )
    con.print(panel)


# ── Recommendations Table ────────────────────────────────────

_URGENCY_STYLE: dict[str, str] = {
    "IMMEDIATE": "bold red",
    "SHORT_TERM": "yellow",
    "LONG_TERM": "dim",
}


def print_recommendations_table(
    report: HealthReport,
    *,
    console: Console | None = None,
) -> None:
    """Render recommended actions as a Rich Table.

    Columns: ``#``, ``Action``, ``Command``, ``Risk``.
    Only displayed when the report contains at least one recommendation.

    Args:
        report: A parsed ``HealthReport`` instance.
        console: Optional Rich Console; defaults to module-level instance.
    """
    if not report.recommendations:
        return

    con = console or _default_console
    table = Table(
        title="📋 Recommended Actions",
        show_lines=True,
        title_style="bold",
    )
    table.add_column("#", style="bold", width=3, justify="right")
    table.add_column("Action", style="cyan", max_width=50)
    table.add_column("Command", style="green", max_width=50)
    table.add_column("Risk", style="yellow", max_width=30)

    sorted_recs = sorted(report.recommendations, key=lambda r: r.priority)
    for rec in sorted_recs:
        urgency_style = _URGENCY_STYLE.get(rec.urgency.value, "")
        action_text = Text(rec.title)
        if urgency_style:
            action_text.append(f" [{rec.urgency.value}]", style=urgency_style)

        cmd = rec.command if rec.command else "—"
        risk = rec.risk if rec.risk else "—"

        table.add_row(str(rec.priority), action_text, cmd, risk)

    con.print(table)

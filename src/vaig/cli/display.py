"""Shared CLI display helpers — confirmation prompts and execution summaries."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from vaig.agents.base import AgentResult
    from vaig.core.compare import CompareReport
    from vaig.core.fleet import FleetReport
    from vaig.skills.service_health.diff import ReportDiff
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


_SECTION_RULE_MAP: list[tuple[re.Pattern[str], str, str]] = [
    # (pattern matching the header text after ##, Rule title, Rule style)
    (re.compile(r"critical\s*issues?|critical", re.IGNORECASE), "🔴 Critical Issues", "red"),
    (re.compile(r"warnings?|warning", re.IGNORECASE), "🟡 Warnings", "yellow"),
    (re.compile(r"findings?", re.IGNORECASE), "📋 Findings", "blue"),
    (re.compile(r"recommend", re.IGNORECASE), "📋 Recommendations", "cyan"),
    (re.compile(r"cost", re.IGNORECASE), "💰 Cost Analysis", "green"),
    (re.compile(r"timeline", re.IGNORECASE), "⏱ Timeline", "blue"),
]


def _section_rule_for_header(header_text: str) -> Rule:
    """Return a styled ``Rule`` for a ``##``-level section header.

    Matches *header_text* against known section patterns and returns a
    Rule with the corresponding emoji and style.  Falls back to a
    ``bright_blue`` Rule with the raw header text.
    """
    for pat, title, style in _SECTION_RULE_MAP:
        if pat.search(header_text):
            return Rule(title=title, style=style)
    return Rule(title=header_text.strip(), style="bright_blue")


def print_colored_report(
    text: str,
    *,
    console: Console | None = None,
) -> None:
    """Print a report with severity keywords colorized and section dividers.

    Lines containing severity keywords are printed with Rich markup
    (plain text, not Markdown) so the colors render.  All other lines
    are accumulated and flushed as Markdown blocks to preserve
    headings, tables, code fences, etc.

    ``##``-level Markdown headers are rendered as Rich ``Rule`` dividers
    with descriptive labels and emojis (e.g. "🔴 Critical Issues").

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

        # ── Section dividers for ## headers ───────────────────
        if stripped.startswith("## ") and not stripped.startswith("### "):
            _flush_md()
            header_text = stripped.lstrip("#").strip()
            con.print()
            con.print(_section_rule_for_header(header_text))
            con.print()
            continue

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


# ── Service Status Table ─────────────────────────────────────


def _format_metric(value: str | None) -> str:
    """Return the value or '—' if it's missing or 'N/A'."""
    return value if value and value.strip().upper() != "N/A" else "—"


# Maps ServiceHealthStatus value → (Rich style, emoji)
_SERVICE_STATUS_STYLE: dict[str, tuple[str, str]] = {
    "HEALTHY": ("green", "🟢"),
    "DEGRADED": ("yellow", "🟡"),
    "FAILED": ("red", "🔴"),
    "UNKNOWN": ("dim", "⚪"),
}


def print_service_status_table(
    report: HealthReport,
    console: Console | None = None,
) -> None:
    """Render a Rich table of per-service health status with colored indicators.

    Uses ``report.service_statuses`` when available.  Falls back to
    extracting a minimal service list from ``report.findings`` grouped
    by severity (Critical → Failed, High/Medium → Degraded, else Healthy).

    Missing metrics (pods, CPU, memory) are displayed as ``—``.

    Args:
        report: A parsed ``HealthReport`` instance.
        console: Optional Rich Console; defaults to module-level instance.
    """
    con = console or _default_console

    # ── Build rows from service_statuses or findings ──────────
    rows: list[dict[str, str]] = []

    if report.service_statuses:
        for svc in report.service_statuses:
            status_key = svc.status.value if hasattr(svc.status, "value") else str(svc.status)
            rows.append(
                {
                    "service": svc.service,
                    "status": status_key,
                    "pods": _format_metric(svc.pods_ready),
                    "cpu": _format_metric(svc.cpu_usage),
                    "memory": _format_metric(svc.memory_usage),
                }
            )
    elif report.findings:
        # Fallback: derive a rough service → status mapping from findings.
        service_severity: dict[str, str] = {}
        for f in report.findings:
            svc_name = f.service or f.id
            if not svc_name:
                continue
            sev = f.severity.value if hasattr(f.severity, "value") else str(f.severity)
            if sev in ("CRITICAL",):
                service_severity[svc_name] = "FAILED"
            elif sev in ("HIGH", "MEDIUM") and service_severity.get(svc_name) != "FAILED":
                service_severity[svc_name] = "DEGRADED"
            elif svc_name not in service_severity:
                service_severity[svc_name] = "HEALTHY"

        for svc_name, status_key in service_severity.items():
            rows.append(
                {
                    "service": svc_name,
                    "status": status_key,
                    "pods": "—",
                    "cpu": "—",
                    "memory": "—",
                }
            )

    if not rows:
        return  # Nothing to display

    # ── Build Rich table inside a Panel ───────────────────────
    table = Table(
        show_header=True,
        header_style="bold",
        show_lines=False,
        pad_edge=True,
        expand=True,
    )
    table.add_column("Service", style="bold white", min_width=20)
    table.add_column("Status", min_width=10)
    table.add_column("Pods", justify="center", min_width=6)
    table.add_column("CPU", justify="center", min_width=8)
    table.add_column("Memory", justify="center", min_width=12)

    for row in rows:
        style, emoji = _SERVICE_STATUS_STYLE.get(row["status"], ("dim", "⚪"))
        status_text = Text(f"{row['status']}", style=style)
        service_label = f"{emoji} {row['service']}"
        table.add_row(service_label, status_text, row["pods"], row["cpu"], row["memory"])

    panel = Panel(table, title="Service Status", border_style="bright_blue", padding=(0, 1))
    con.print(panel)


# ── Cost Breakdown Table ─────────────────────────────────────


def print_cost_breakdown_table(
    report: HealthReport,
    console: Console | None = None,
) -> None:
    """Render a cost breakdown table from GKE workload cost data.

    Extracts data from ``report.metadata.gke_cost`` when available.
    If no structured cost data exists the function returns silently
    (never crashes).

    Args:
        report: A parsed ``HealthReport`` instance.
        console: Optional Rich Console; defaults to module-level instance.
    """
    con = console or _default_console

    # Resolve cost data — prefer gke_cost, fall back gracefully
    gke_cost = getattr(report.metadata, "gke_cost", None) if report.metadata else None
    if gke_cost is None or not getattr(gke_cost, "workloads", None):
        return  # No cost data — skip silently

    workloads = gke_cost.workloads
    if not workloads:
        return

    # Guard: verify cost fields are actually numeric (not mocks/stubs)
    _sample = getattr(gke_cost, "total_request_cost_usd", None)
    if _sample is not None and not isinstance(_sample, (int, float)):
        return

    con.print(Rule("💰 Cost Breakdown", style="green"))
    con.print()

    table = Table(
        show_header=True,
        header_style="bold",
        show_lines=False,
        pad_edge=True,
        expand=True,
    )
    table.add_column("Workload", style="cyan", min_width=25)
    table.add_column("Namespace", style="dim", min_width=12)
    table.add_column("Request Cost", justify="right", style="bold green", min_width=14)
    table.add_column("Usage Cost", justify="right", min_width=14)
    table.add_column("Waste", justify="right", style="red", min_width=14)

    for wl in workloads:
        name = wl.workload_name or "—"
        ns = wl.namespace or "—"
        req_cost = f"${wl.total_request_cost_usd:,.2f}/mo" if wl.total_request_cost_usd is not None else "—"
        use_cost = f"${wl.total_usage_cost_usd:,.2f}/mo" if wl.total_usage_cost_usd is not None else "—"
        waste = f"${wl.total_waste_usd:,.2f}/mo" if wl.total_waste_usd is not None else "—"
        table.add_row(name, ns, req_cost, use_cost, waste)

    # Totals row
    if gke_cost.total_request_cost_usd is not None:
        total_req = f"${gke_cost.total_request_cost_usd:,.2f}/mo"
        total_use = f"${gke_cost.total_usage_cost_usd:,.2f}/mo" if gke_cost.total_usage_cost_usd is not None else "—"
        # Compute total waste by summing per-workload waste (not savings)
        waste_values = [
            wl.total_waste_usd
            for wl in workloads
            if getattr(wl, "total_waste_usd", None) is not None
            and isinstance(wl.total_waste_usd, (int, float))
        ]
        if waste_values:
            total_waste = sum(waste_values)
            total_waste_str = f"${total_waste:,.2f}/mo"
        else:
            total_waste_str = "—"
        table.add_section()
        table.add_row(
            Text("TOTAL", style="bold"),
            "",
            Text(total_req, style="bold green"),
            Text(total_use, style="bold"),
            Text(total_waste_str, style="bold red"),
        )

    con.print(table)
    con.print()


# ── Severity Detail Blocks ───────────────────────────────────

# Emoji & label for severity detail rendering (extends the schema map)
_SEVERITY_DETAIL_STYLE: dict[str, tuple[str, str, str]] = {
    # key → (emoji, label, Rich style)
    "CRITICAL": ("🔴", "Critical", "bold red"),
    "HIGH": ("🟠", "High", "bold #ff8c00"),
    "MEDIUM": ("🟡", "Medium", "bold yellow"),
    "LOW": ("🔵", "Low", "bold blue"),
    "INFO": ("🟢", "Info", "bold green"),
}


def print_severity_detail_blocks(
    report: HealthReport,
    console: Console | None = None,
) -> None:
    """Render findings grouped by severity as Rich-formatted detail blocks.

    Groups findings by severity (CRITICAL → INFO), printing each with
    a colored emoji prefix, bold title, and indented description /
    root cause / evidence.

    Args:
        report: A parsed ``HealthReport`` instance.
        console: Optional Rich Console; defaults to module-level instance.
    """
    con = console or _default_console

    if not report.findings:
        return

    severity_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]

    # Group findings by severity
    grouped: dict[str, list[Any]] = {s: [] for s in severity_order}
    for f in report.findings:
        sev = f.severity.value if hasattr(f.severity, "value") else str(f.severity)
        if sev in grouped:
            grouped[sev].append(f)
        else:
            grouped.setdefault("INFO", []).append(f)

    for sev in severity_order:
        findings = grouped.get(sev, [])
        if not findings:
            continue

        emoji, label, style = _SEVERITY_DETAIL_STYLE.get(sev, ("⚪", sev, "dim"))

        for f in findings:
            # Title line: 🔴 Critical: service-name
            title_parts = Text()
            title_parts.append(f"{emoji} {label}: ", style=style)
            service_hint = getattr(f, "service", "") or getattr(f, "id", "")
            title_parts.append(f"{getattr(f, 'title', service_hint)}", style="bold")
            con.print(title_parts)

            # Description (indented)
            description = getattr(f, "description", "")
            if description:
                for desc_line in description.split("\n"):
                    con.print(f"   {desc_line}")

            # Root cause (indented)
            root_cause = getattr(f, "root_cause", "")
            if root_cause:
                con.print(Text(f"   Root cause: {root_cause}", style="dim italic"))

            # Evidence items (indented)
            evidence = getattr(f, "evidence", [])
            if evidence:
                for ev_item in evidence:
                    con.print(f"   • {ev_item}", style="dim")

            # Impact
            impact = getattr(f, "impact", "")
            if impact:
                con.print(Text(f"   Impact: {impact}", style="dim"))

            con.print()  # Blank line between findings


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
    """Render recommended actions as Rich Panels (one per recommendation).

    Each panel displays the action's fields conditionally — empty fields
    are silently omitted.  The panel border colour reflects urgency via
    ``_URGENCY_STYLE``.

    Args:
        report: A parsed ``HealthReport`` instance.
        console: Optional Rich Console; defaults to module-level instance.
    """
    if not report.recommendations:
        return

    con = console or _default_console
    con.print(Text("📋 Recommended Actions", style="bold"))
    con.print()

    sorted_recs = sorted(report.recommendations, key=lambda r: r.priority)
    for rec in sorted_recs:
        urgency_style = _URGENCY_STYLE.get(rec.urgency.value, "")
        title_text = f"#{rec.priority} {rec.title} [{rec.urgency.value}]"

        body_parts: list[Text | Syntax] = []

        if rec.description and rec.description.strip():
            body_parts.append(Text(rec.description))

        if rec.command and rec.command.strip():
            body_parts.append(Text(""))
            body_parts.append(Text("Command:", style="bold"))
            body_parts.append(
                Syntax(rec.command, "bash", theme="monokai", word_wrap=True)
            )

        if rec.expected_output and rec.expected_output.strip():
            body_parts.append(Text(""))
            body_parts.append(Text("Expected output:", style="bold"))
            body_parts.append(Text(rec.expected_output, style="dim"))

        if rec.interpretation and rec.interpretation.strip():
            body_parts.append(Text(""))
            body_parts.append(Text("Interpretation:", style="bold"))
            body_parts.append(Text(rec.interpretation, style="italic"))

        if rec.why and rec.why.strip():
            body_parts.append(Text(""))
            body_parts.append(Text(f"Why: {rec.why}", style="cyan"))

        if rec.risk and rec.risk.strip():
            body_parts.append(Text(""))
            body_parts.append(Text(f"Risk: {rec.risk}", style="yellow"))

        panel = Panel(
            Group(*body_parts),
            title=title_text,
            border_style=urgency_style,
            expand=True,
        )
        con.print(panel)


# ── Fleet Summary Panel ──────────────────────────────────────


def print_fleet_summary_panel(
    report: FleetReport,
    *,
    detailed: bool = False,
    console: Console | None = None,
) -> None:
    """Render a fleet scan summary as a Rich Panel.

    Shows total scanned, success/fail counts, fleet health status,
    top correlations, and total cost.  When *detailed* is ``True``,
    also renders a per-cluster breakdown table.

    Args:
        report: A :class:`FleetReport` instance from :class:`FleetRunner`.
        detailed: Whether to show per-cluster breakdown.
        console: Optional Rich Console; defaults to module-level instance.
    """
    con = console or _default_console

    total = len(report.clusters)
    successes = sum(1 for c in report.clusters if c.status == "success")
    errors = sum(1 for c in report.clusters if c.status == "error")
    skipped = sum(1 for c in report.clusters if c.status == "skipped")

    # Determine fleet health status
    if errors == 0 and skipped == 0:
        fleet_status = "HEALTHY"
        border_style = "green"
        emoji = "🟢"
    elif errors == total:
        fleet_status = "FAILED"
        border_style = "red"
        emoji = "🔴"
    else:
        fleet_status = "DEGRADED"
        border_style = "yellow"
        emoji = "🟡"

    body = Text()
    body.append(f"{emoji} Fleet Status: ", style="bold")
    body.append(f"{fleet_status}\n", style=f"bold {border_style}")
    body.append(f"📊 Clusters: {total} total")
    if successes:
        body.append(f", {successes} succeeded", style="green")
    if errors:
        body.append(f", {errors} failed", style="red")
    if skipped:
        body.append(f", {skipped} skipped", style="yellow")
    body.append("\n")
    body.append(f"⏱  Duration: {report.total_duration_s:.1f}s\n")
    body.append(f"💰 Cost: ${report.total_cost_usd:.4f}\n")

    if report.budget_exceeded:
        body.append("⚠  Budget exceeded — some clusters were skipped\n", style="bold yellow")

    # Top correlations
    if report.correlations:
        body.append("\n")
        body.append("🔗 Fleet-wide patterns:\n", style="bold")
        for corr in report.correlations[:5]:
            body.append(
                f"  • {corr.pattern} ({corr.category}) — "
                f"{len(corr.affected_clusters)}/{total} clusters\n"
            )

    panel = Panel(
        body,
        title="Fleet Scan Summary",
        border_style=border_style,
        padding=(1, 2),
    )
    con.print(panel)

    # Detailed per-cluster table
    if detailed:
        table = Table(title="📋 Per-Cluster Results", show_lines=True)
        table.add_column("Cluster", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Duration", justify="right")
        table.add_column("Cost", justify="right")
        table.add_column("Details")

        for cr in report.clusters:
            if cr.status == "success":
                status_str = "[green]✓ Success[/green]"
                detail = ""
                if cr.result and cr.result.structured_report:
                    findings = getattr(cr.result.structured_report, "findings", [])
                    detail = f"{len(findings)} findings"
            elif cr.status == "error":
                status_str = "[red]✗ Error[/red]"
                detail = cr.error or "Unknown error"
                if len(detail) > 60:
                    detail = detail[:57] + "..."
            else:
                status_str = "[yellow]⊘ Skipped[/yellow]"
                detail = cr.error or "Skipped"

            table.add_row(
                cr.display_name,
                status_str,
                f"{cr.duration_s:.1f}s",
                f"${cr.cost_usd:.4f}",
                detail,
            )

        con.print(table)


# ── Watch Mode Diff Summary ──────────────────────────────────


def print_watch_diff_summary(
    diff: ReportDiff,
    iteration: int,
    *,
    console: Console | None = None,
) -> None:
    """Render a compact diff summary panel for watch mode iterations.

    Shown AFTER the full health report on iteration 2+ to give an
    immediate visual overview of what changed between consecutive
    watch iterations.

    Sections (only shown when non-empty):

    - ``🆕 NEW`` — count and list with severity badge + title
    - ``✅ RESOLVED`` — count and list
    - ``⚠️  SEVERITY CHANGED`` — finding title with old → new severity
    - ``🔄 UNCHANGED`` — count only (no list)

    Args:
        diff: A :class:`ReportDiff` computed from the current and
            previous ``HealthReport`` instances.
        iteration: The current watch iteration number (for the title).
        console: Optional Rich Console; defaults to module-level instance.
    """
    con = console or _default_console

    body = Text()

    # ── New findings ──────────────────────────────────────────
    if diff.new_findings:
        # Header style: red if any finding is CRITICAL/HIGH, yellow otherwise.
        _high_sevs = {"CRITICAL", "HIGH"}
        _new_header_style = (
            "bold red"
            if any(f.severity.value in _high_sevs for f in diff.new_findings)
            else "bold yellow"
        )
        body.append("🆕 NEW", style=_new_header_style)
        body.append(f" ({len(diff.new_findings)})\n")
        for f in diff.new_findings:
            sev = f.severity.value
            emoji, _label, style = _SEVERITY_DETAIL_STYLE.get(sev, ("⚪", sev, "dim"))
            body.append(f"   {emoji} ", style=style)
            body.append(f"{f.title}\n")

    # ── Resolved findings ─────────────────────────────────────
    if diff.resolved_findings:
        body.append("✅ RESOLVED", style="bold green")
        body.append(f" ({len(diff.resolved_findings)})\n")
        for f in diff.resolved_findings:
            body.append(f"   • {f.title}\n", style="green")

    # ── Severity changes ──────────────────────────────────────
    if diff.severity_changes:
        body.append("⚠️  SEVERITY CHANGED", style="bold yellow")
        body.append(f" ({len(diff.severity_changes)})\n")
        for sc in diff.severity_changes:
            body.append(f"   {sc.finding.title}: ")
            body.append(sc.previous_severity, style="bold")
            body.append(" → ")
            body.append(sc.current_severity, style="bold")
            body.append("\n")

    # ── Unchanged ─────────────────────────────────────────────
    if diff.unchanged_findings:
        body.append("🔄 UNCHANGED", style="dim")
        body.append(f" ({len(diff.unchanged_findings)})\n", style="dim")

    # Fallback when diff contains no findings at all
    if not body.plain:
        body.append("No findings in either report.", style="dim")

    # Choose panel style based on changes
    if diff.has_changes:
        title = f"Changes Detected — Iteration #{iteration}"
        border_style = "yellow"
    else:
        title = f"No Changes — Iteration #{iteration}"
        border_style = "green"

    panel = Panel(
        body,
        title=title,
        border_style=border_style,
        padding=(1, 2),
    )
    con.print(panel)


# ── Cross-Cluster Compare Report ─────────────────────────────


_COMPARE_SEVERITY_STYLE: dict[str, str] = {
    "critical": "bold red",
    "warning": "yellow",
    "info": "dim",
}


def print_compare_report(
    report: CompareReport,
    *,
    console: Console | None = None,
) -> None:
    """Render a cross-cluster comparison as a Rich Table (REQ-CMP-06).

    Columns: one per cluster. Rows: each comparable field. Divergent
    cells are color-coded by severity (red/yellow/dim). Matching
    values are shown in green.

    Error clusters are displayed in a separate panel.

    Args:
        report: A :class:`CompareReport` from :class:`CompareRunner`.
        console: Optional Rich Console; defaults to module-level instance.
    """
    con = console or _default_console

    cluster_names = list(report.snapshots.keys())

    # ── Error clusters panel ──────────────────────────────────
    if report.errors:
        err_body = Text()
        for name, msg in report.errors.items():
            err_body.append(f"  ✗ {name}: ", style="bold red")
            err_body.append(f"{msg}\n")
        con.print(Panel(err_body, title="Unreachable Clusters", border_style="red", padding=(0, 2)))

    if not cluster_names:
        con.print("[bold red]No snapshots collected — nothing to compare.[/bold red]")
        return

    # ── Build comparison table ────────────────────────────────
    table = Table(
        title="🔍 Cross-Cluster Comparison",
        show_lines=True,
        title_style="bold cyan",
    )
    table.add_column("Field", style="bold")
    for name in cluster_names:
        table.add_column(name, justify="center")

    # Build a set of divergent fields for quick lookup
    diff_map: dict[str, dict[str, Any]] = {}
    severity_map: dict[str, str] = {}
    for d in report.diffs:
        diff_map[d.field] = d.values
        severity_map[d.field] = d.severity

    # Fields to display (same as _COMPARABLE_FIELDS in compare.py)
    display_fields = [
        "image_tag",
        "replicas_desired",
        "replicas_ready",
        "hpa_min",
        "hpa_max",
        "cpu_usage_cores",
        "memory_usage_gib",
        "error_rate_pct",
        "rollout_generation",
    ]

    for field_name in display_fields:
        row_cells: list[str] = []
        is_divergent = field_name in diff_map

        for cname in cluster_names:
            snap = report.snapshots[cname]
            raw = getattr(snap, field_name, None)
            display_val = str(raw) if raw is not None else "N/A"

            if is_divergent:
                style = _COMPARE_SEVERITY_STYLE.get(severity_map[field_name], "")
                row_cells.append(f"[{style}]{display_val}[/{style}]")
            else:
                row_cells.append(f"[green]{display_val}[/green]")

        table.add_row(field_name, *row_cells)

    con.print(table)

    # ── Summary line ──────────────────────────────────────────
    total_diffs = len(report.diffs)
    if total_diffs == 0:
        con.print("\n[bold green]✓ No divergences found — clusters are in sync.[/bold green]")
    else:
        critical = sum(1 for d in report.diffs if d.severity == "critical")
        warning = sum(1 for d in report.diffs if d.severity == "warning")
        info = total_diffs - critical - warning
        parts = []
        if critical:
            parts.append(f"[bold red]{critical} critical[/bold red]")
        if warning:
            parts.append(f"[yellow]{warning} warning[/yellow]")
        if info:
            parts.append(f"[dim]{info} info[/dim]")
        con.print(f"\n⚠ {total_diffs} divergence{'s' if total_diffs != 1 else ''} found ({', '.join(parts)})")

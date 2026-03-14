"""Stats sub-commands — usage telemetry and analytics."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.panel import Panel
from rich.table import Table

from vaig.cli import _helpers
from vaig.cli._helpers import console, err_console


def register(stats_app: typer.Typer) -> None:
    """Register stats sub-commands on the given Typer sub-app."""

    @stats_app.command("show")
    def stats_show(
        config: Annotated[Optional[str], typer.Option("--config", "-c", help="Path to config YAML")] = None,
        since: Annotated[Optional[str], typer.Option("--since", help="ISO timestamp lower bound (e.g. 2025-01-01)")] = None,
        until: Annotated[Optional[str], typer.Option("--until", help="ISO timestamp upper bound")] = None,
    ) -> None:
        """Show usage telemetry summary."""
        from vaig.core.telemetry import get_telemetry_collector

        settings = _helpers._get_settings(config)
        collector = get_telemetry_collector(settings)
        summary = collector.get_summary(since=since, until=until)

        if summary["total_events"] == 0:
            console.print("[yellow]No telemetry events recorded yet.[/yellow]")
            return

        # Overview panel
        console.print(
            Panel(
                f"[cyan]Total events:[/cyan] {summary['total_events']}\n"
                f"[cyan]Errors:[/cyan]       {summary['error_count']}",
                title="Telemetry Summary",
                border_style="bright_blue",
            )
        )

        # Events by type
        by_type = summary.get("by_type", {})
        if by_type:
            type_table = Table(title="Events by Type", show_lines=False)
            type_table.add_column("Event Type", style="cyan")
            type_table.add_column("Count", style="yellow", justify="right")
            for etype, count in by_type.items():
                type_table.add_row(etype, str(count))
            console.print(type_table)

        # Top tools
        top_tools = summary.get("top_tools", {})
        if top_tools:
            tools_table = Table(title="Top 10 Tools by Usage", show_lines=False)
            tools_table.add_column("Tool", style="green")
            tools_table.add_column("Calls", style="yellow", justify="right")
            for tool, count in top_tools.items():
                tools_table.add_row(tool, str(count))
            console.print(tools_table)

        # API stats
        api = summary.get("api_calls", {})
        if api.get("count", 0) > 0:
            api_table = Table(title="API Usage", show_lines=False)
            api_table.add_column("Metric", style="cyan")
            api_table.add_column("Value", style="yellow", justify="right")
            api_table.add_row("Total API calls", str(api["count"]))
            api_table.add_row("Tokens in", f"{api['total_tokens_in']:,}")
            api_table.add_row("Tokens out", f"{api['total_tokens_out']:,}")
            api_table.add_row("Total cost (USD)", f"${api['total_cost_usd']:.4f}")
            console.print(api_table)

    @stats_app.command("export")
    def stats_export(
        config: Annotated[Optional[str], typer.Option("--config", "-c", help="Path to config YAML")] = None,
        format_: Annotated[str, typer.Option("--format", "-f", help="Export format: jsonl or csv")] = "jsonl",
        output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output file path (default: stdout)")] = None,
        since: Annotated[Optional[str], typer.Option("--since", help="ISO timestamp lower bound")] = None,
        until: Annotated[Optional[str], typer.Option("--until", help="ISO timestamp upper bound")] = None,
        event_type: Annotated[Optional[str], typer.Option("--type", "-t", help="Filter by event type")] = None,
    ) -> None:
        """Export telemetry events as JSONL or CSV."""
        import csv
        import io
        import json

        from vaig.core.telemetry import get_telemetry_collector

        if format_ not in ("jsonl", "csv"):
            err_console.print(f"[red]Unsupported format: {format_}. Use 'jsonl' or 'csv'.[/red]")
            raise typer.Exit(1)

        settings = _helpers._get_settings(config)
        collector = get_telemetry_collector(settings)
        events = collector.query_events(event_type, since=since, until=until, limit=10_000)

        if not events:
            console.print("[yellow]No events found matching the given filters.[/yellow]")
            return

        if format_ == "jsonl":
            lines = [json.dumps(e, default=str) for e in events]
            content = "\n".join(lines) + "\n"
        else:
            # CSV
            buf = io.StringIO()
            fieldnames = list(events[0].keys())
            writer = csv.DictWriter(buf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(events)
            content = buf.getvalue()

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(content, encoding="utf-8")
            console.print(f"[green]Exported {len(events)} events to {output}[/green]")
        else:
            typer.echo(content, nl=False)

    @stats_app.command("clear")
    def stats_clear(
        config: Annotated[Optional[str], typer.Option("--config", "-c", help="Path to config YAML")] = None,
        days: Annotated[int, typer.Option("--days", "-d", help="Delete events older than N days")] = 30,
        confirm: Annotated[bool, typer.Option("--confirm", help="Confirm deletion (required)")] = False,
    ) -> None:
        """Clear old telemetry events."""
        if not confirm:
            console.print(
                f"[yellow]This will delete telemetry events older than {days} days.[/yellow]\n"
                "[dim]Run with --confirm to proceed.[/dim]"
            )
            raise typer.Exit

        from vaig.core.telemetry import get_telemetry_collector

        settings = _helpers._get_settings(config)
        collector = get_telemetry_collector(settings)
        deleted = collector.clear_events(older_than_days=days)
        console.print(f"[green]Deleted {deleted} events older than {days} days.[/green]")

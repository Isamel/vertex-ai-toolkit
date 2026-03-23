"""Cloud data export commands — push vaig data to BigQuery and GCS."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()
err_console = Console(stderr=True)

# ── Typer apps ────────────────────────────────────────────────

cloud_app = typer.Typer(
    name="cloud",
    help="Cloud data export (BigQuery, GCS, Vertex AI)",
    no_args_is_help=True,
)
push_app = typer.Typer(
    name="push",
    help="Push data to cloud storage",
    no_args_is_help=True,
)
cloud_app.add_typer(push_app, name="push")


# ── Helpers ───────────────────────────────────────────────────


def _parse_since(since: str) -> datetime:
    """Parse time shorthand like '7d', '30d', '1h', '2w' into a datetime.

    Args:
        since: A string like '7d', '30d', '1h', '2w'.

    Returns:
        A timezone-aware UTC datetime representing ``now - delta``.

    Raises:
        typer.BadParameter: If the format is not recognised.
    """
    match = re.match(r"^(\d+)([hdwm])$", since)
    if not match:
        raise typer.BadParameter(
            f"Invalid time format: {since!r}. Use e.g. '7d', '30d', '1h', '2w'."
        )
    value, unit = int(match.group(1)), match.group(2)
    deltas: dict[str, timedelta] = {
        "h": timedelta(hours=value),
        "d": timedelta(days=value),
        "w": timedelta(weeks=value),
        "m": timedelta(days=value * 30),
    }
    return datetime.now(UTC) - deltas[unit]


def _validate_destination(to: str) -> None:
    """Raise BadParameter if *to* is not one of bigquery / gcs / both."""
    if to not in ("bigquery", "gcs", "both"):
        raise typer.BadParameter(
            f"Invalid destination: {to!r}. Choose 'bigquery', 'gcs', or 'both'."
        )


def _print_dry_run_summary(
    record_type: str,
    records: Sequence[object],
    destination: str,
    since: datetime,
) -> None:
    """Print a Rich table summarising what a dry-run *would* export."""
    table = Table(title=f"[bold]Dry Run — {record_type}[/bold]", show_header=True, header_style="bold cyan")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="yellow")
    table.add_row("Record type", record_type)
    table.add_row("Record count", str(len(records)))
    table.add_row("Since", since.isoformat())
    table.add_row("Destination", destination)
    table.add_row("Estimated size", f"~{len(records) * 500} bytes")
    console.print(table)
    console.print("[dim]Dry run complete — no data was exported.[/dim]")


def _print_export_results(
    record_type: str,
    bq_count: int | None,
    gcs_uri: str | None,
) -> None:
    """Print a Rich summary table after a real export."""
    table = Table(title=f"[bold]Export Results — {record_type}[/bold]", show_header=True, header_style="bold green")
    table.add_column("Destination", style="cyan")
    table.add_column("Result", style="yellow")
    if bq_count is not None:
        status = f"[green]{bq_count} rows inserted[/green]" if bq_count > 0 else "[red]0 rows (check logs)[/red]"
        table.add_row("BigQuery", status)
    if gcs_uri is not None:
        uri_display = f"[green]{gcs_uri}[/green]" if gcs_uri else "[red]failed (check logs)[/red]"
        table.add_row("GCS", uri_display)
    console.print(table)


# ── push telemetry ────────────────────────────────────────────


@push_app.command("telemetry")
def push_telemetry(
    since: Annotated[
        str,
        typer.Option("--since", help="Time range (e.g. '7d', '30d', '1h')"),
    ] = "7d",
    dest: Annotated[
        str,
        typer.Option("--dest", help="Destination: bigquery, gcs, or both"),
    ] = "both",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be exported without sending"),
    ] = False,
    config: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Path to config YAML"),
    ] = None,
) -> None:
    """Push telemetry events to BigQuery and/or GCS."""
    from vaig.cli._helpers import _get_settings
    from vaig.core.export import DataExporter
    from vaig.core.telemetry import get_telemetry_collector

    _validate_destination(dest)
    since_dt = _parse_since(since)

    settings = _get_settings(config)
    if not settings.export.enabled:
        err_console.print(
            "[red]Export is disabled.[/red] Set [bold]export.enabled = true[/bold] "
            "in your config or [bold]VAIG_EXPORT__ENABLED=true[/bold] to enable it."
        )
        raise typer.Exit(1)

    collector = get_telemetry_collector(settings)
    since_iso = since_dt.isoformat()
    records = collector.query_events(None, since=since_iso, limit=50_000)

    if dry_run:
        _print_dry_run_summary("Telemetry Events", records, dest, since_dt)
        return

    if not records:
        console.print("[yellow]No telemetry events found in the given time range.[/yellow]")
        return

    exporter = DataExporter(settings.export)

    bq_count: int | None = None
    gcs_uri: str | None = None

    with console.status("[bold cyan]Exporting telemetry…[/bold cyan]"):
        if dest in ("bigquery", "both"):
            bq_count = exporter.export_telemetry_to_bigquery(records)
        if dest in ("gcs", "both"):
            gcs_uri = exporter.export_telemetry_to_gcs(records)

    _print_export_results("Telemetry Events", bq_count, gcs_uri)


# ── push tool-calls ───────────────────────────────────────────


@push_app.command("tool-calls")
def push_tool_calls(
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Specific run ID to export"),
    ] = None,
    since: Annotated[
        str,
        typer.Option("--since", help="Time range (e.g. '7d', '30d', '1h')"),
    ] = "7d",
    dest: Annotated[
        str,
        typer.Option("--dest", help="Destination: bigquery, gcs, or both"),
    ] = "both",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be exported without sending"),
    ] = False,
    config: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Path to config YAML"),
    ] = None,
) -> None:
    """Push tool call records to BigQuery and/or GCS."""
    from pathlib import Path

    from vaig.cli._helpers import _get_settings
    from vaig.core.export import DataExporter
    from vaig.core.tool_call_store import ToolCallStore

    _validate_destination(dest)
    since_dt = _parse_since(since)

    settings = _get_settings(config)
    if not settings.export.enabled:
        err_console.print(
            "[red]Export is disabled.[/red] Set [bold]export.enabled = true[/bold] "
            "in your config or [bold]VAIG_EXPORT__ENABLED=true[/bold] to enable it."
        )
        raise typer.Exit(1)

    store = ToolCallStore(base_dir=Path(settings.logging.tool_results_dir).expanduser())
    records = store.read_records(run_id=run_id, since=since_dt if run_id is None else None)

    if dry_run:
        _print_dry_run_summary("Tool Calls", records, dest, since_dt)
        return

    if not records:
        msg = f"No tool call records found for run_id={run_id!r}." if run_id else "No tool call records found in the given time range."
        console.print(f"[yellow]{msg}[/yellow]")
        return

    exporter = DataExporter(settings.export)

    bq_count: int | None = None
    gcs_uri: str | None = None

    with console.status("[bold cyan]Exporting tool calls…[/bold cyan]"):
        effective_run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        if dest in ("bigquery", "both"):
            bq_count = exporter.export_tool_calls_to_bigquery(records)
        if dest in ("gcs", "both"):
            gcs_uri = exporter.export_tool_results_to_gcs(records, effective_run_id)

    _print_export_results("Tool Calls", bq_count, gcs_uri)


# ── push reports ──────────────────────────────────────────────


@push_app.command("reports")
def push_reports(
    since: Annotated[
        str,
        typer.Option("--since", help="Time range (e.g. '30d', '7d', '1h')"),
    ] = "30d",
    dest: Annotated[
        str,
        typer.Option("--dest", help="Destination: bigquery, gcs, or both"),
    ] = "both",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be exported without sending"),
    ] = False,
    config: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Path to config YAML"),
    ] = None,
) -> None:
    """Push health reports to BigQuery and/or GCS."""
    import json as _json
    from pathlib import Path

    from vaig.cli._helpers import _get_settings
    from vaig.core.export import DataExporter

    _validate_destination(dest)
    since_dt = _parse_since(since)

    settings = _get_settings(config)
    if not settings.export.enabled:
        err_console.print(
            "[red]Export is disabled.[/red] Set [bold]export.enabled = true[/bold] "
            "in your config or [bold]VAIG_EXPORT__ENABLED=true[/bold] to enable it."
        )
        raise typer.Exit(1)

    # Discover report files under ~/.vaig/reports/ since the given date
    reports_dir = Path.home() / ".vaig" / "reports"
    report_files: list[Path] = []
    if reports_dir.is_dir():
        for f in sorted(reports_dir.rglob("*.json")):
            try:
                mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=UTC)
                if mtime >= since_dt:
                    report_files.append(f)
            except OSError:
                continue

    if dry_run:
        _print_dry_run_summary("Health Reports", report_files, dest, since_dt)
        return

    if not report_files:
        console.print("[yellow]No health reports found in the given time range.[/yellow]")
        return

    exporter = DataExporter(settings.export)
    exported_bq = 0
    exported_gcs = 0

    with console.status("[bold cyan]Exporting health reports…[/bold cyan]"):
        for report_path in report_files:
            try:
                report = _json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                err_console.print(f"[yellow]Skipping malformed report: {report_path}[/yellow]")
                continue

            run_id = report_path.stem
            if dest in ("bigquery", "both"):
                ok = exporter.export_report_to_bigquery(report, run_id=run_id)
                if ok:
                    exported_bq += 1
            if dest in ("gcs", "both"):
                uri = exporter.export_report_to_gcs(report, run_id)
                if uri:
                    exported_gcs += 1

    table = Table(title="[bold]Export Results — Health Reports[/bold]", show_header=True, header_style="bold green")
    table.add_column("Destination", style="cyan")
    table.add_column("Result", style="yellow")
    if dest in ("bigquery", "both"):
        table.add_row("BigQuery", f"[green]{exported_bq}/{len(report_files)} reports inserted[/green]")
    if dest in ("gcs", "both"):
        table.add_row("GCS", f"[green]{exported_gcs}/{len(report_files)} reports uploaded[/green]")
    console.print(table)


# ── cloud status (Phase 2 stub) ───────────────────────────────


@cloud_app.command("status")
def cloud_status() -> None:
    """Show cloud export status and last sync info. (Coming soon)"""
    typer.echo("Cloud export status: coming in a future release.")

"""CLI sub-app for incident export — ``vaig incident export`` / ``vaig incident list``."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from vaig.core.config import get_settings
from vaig.core.report_store import ReportStore
from vaig.integrations.finding_exporter import ExportResult, FindingExporter
from vaig.integrations.jira import JiraClient
from vaig.integrations.pagerduty import PagerDutyClient

console = Console()
err_console = Console(stderr=True)

incident_app = typer.Typer(
    name="incident",
    help="Export diagnosis findings to Jira or PagerDuty",
    no_args_is_help=True,
)


# ── Commands ─────────────────────────────────────────────────


@incident_app.command()
def export(
    to: Annotated[
        str,
        typer.Option("--to", help="Export target: jira or pagerduty"),
    ],
    finding: Annotated[
        str,
        typer.Option("--finding", help="Finding slug (ID) to export"),
    ],
    cluster: Annotated[
        str,
        typer.Option("--cluster", help="Cluster context for PagerDuty dedup key"),
    ] = "",
) -> None:
    """Export a finding to Jira or PagerDuty."""
    settings = get_settings()

    # Build clients based on config
    jira_client = None
    pd_client = None

    if to == "jira":
        if not settings.jira.enabled:
            err_console.print(
                "[bold red]Jira integration not configured.[/]\n"
                "Set VAIG_JIRA__BASE_URL and VAIG_JIRA__API_TOKEN to enable.",
            )
            raise typer.Exit(code=1)
        jira_client = JiraClient(settings.jira)

    elif to == "pagerduty":
        if not settings.pagerduty.enabled:
            err_console.print(
                "[bold red]PagerDuty integration not configured.[/]\n"
                "Set VAIG_PAGERDUTY__ROUTING_KEY to enable.",
            )
            raise typer.Exit(code=1)
        pd_client = PagerDutyClient(settings.pagerduty)
    else:
        err_console.print(f"[bold red]Unknown target:[/] {to!r}. Use 'jira' or 'pagerduty'.")
        raise typer.Exit(code=1)

    report_store = ReportStore()
    exporter = FindingExporter(
        jira=jira_client,
        pagerduty=pd_client,
        report_store=report_store,
    )

    result: ExportResult = exporter.export(
        finding_slug=finding,
        target=to,
        cluster_context=cluster,
    )

    if not result.success:
        err_console.print(f"[bold red]Export failed:[/] {result.error}")
        raise typer.Exit(code=1)

    if result.already_existed:
        console.print(
            f"[yellow]⚠ Finding already exported.[/] "
            f"Existing {result.target} issue: [bold]{result.key}[/]"
        )
        if result.url:
            console.print(f"  → {result.url}")
    else:
        console.print(
            f"[green]✓ Exported to {result.target}:[/] [bold]{result.key}[/]"
        )
        if result.url:
            console.print(f"  → {result.url}")


@incident_app.command(name="list")
def list_findings(
    last: Annotated[
        int,
        typer.Option("--last", help="Number of recent reports to scan"),
    ] = 20,
) -> None:
    """List recent findings from stored health reports."""
    report_store = ReportStore()
    exporter = FindingExporter(report_store=report_store)
    findings = exporter.list_findings(last=last)

    if not findings:
        console.print("[yellow]No findings found.[/] Run `vaig live` or `vaig discover` first.")
        return

    table = Table(title="Recent Findings", show_lines=False)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Title", style="white")
    table.add_column("Severity", style="bold")
    table.add_column("Service", style="dim")
    table.add_column("Timestamp", style="dim")

    for f in findings:
        severity = str(f.get("severity", "")).upper()
        severity_style = {
            "CRITICAL": "[bold red]",
            "HIGH": "[red]",
            "MEDIUM": "[yellow]",
            "LOW": "[green]",
            "INFO": "[dim]",
        }.get(severity, "")
        severity_display = f"{severity_style}{severity}[/]" if severity_style else severity

        table.add_row(
            f["id"],
            f["title"],
            severity_display,
            f.get("service", ""),
            f.get("timestamp", "")[:19],
        )

    console.print(table)

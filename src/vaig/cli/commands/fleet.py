"""Fleet discover CLI — multi-cluster scanning subcommand.

Registers ``vaig fleet discover`` as a Typer sub-app with flags for
parallel execution, budget, detailed output, and export.
"""

from __future__ import annotations

import logging
from typing import Annotated, Optional

import click
import typer
from rich.console import Console

from vaig.cli._helpers import _apply_subcommand_log_flags, handle_cli_error, track_command

logger = logging.getLogger(__name__)

fleet_app = typer.Typer(help="Multi-cluster fleet scanning")

_console = Console()


def register(app: typer.Typer) -> None:
    """Register the fleet sub-app on the main Typer app."""
    app.add_typer(fleet_app, name="fleet")


@fleet_app.command()
@track_command
def discover(
    parallel: Annotated[
        bool,
        typer.Option("--parallel", help="Scan clusters concurrently"),
    ] = False,
    max_workers: Annotated[
        int,
        typer.Option("--max-workers", help="Max concurrent workers (with --parallel)"),
    ] = 4,
    budget: Annotated[
        float,
        typer.Option("--budget", help="Daily budget in USD (0 = unlimited)"),
    ] = 0.0,
    detailed: Annotated[
        bool,
        typer.Option("--detailed", help="Show per-cluster breakdown"),
    ] = False,
    export: Annotated[
        Optional[str],  # noqa: UP007, UP045
        typer.Option("--export", help="Export format: json"),
    ] = None,
    namespace: Annotated[
        Optional[str],  # noqa: UP007, UP045
        typer.Option("--namespace", "-n", help="Override namespace for all clusters"),
    ] = None,
    all_namespaces: Annotated[
        bool,
        typer.Option("--all-namespaces", "-A", help="Override: scan all namespaces"),
    ] = False,
    skip_healthy: Annotated[
        bool,
        typer.Option("--skip-healthy/--no-skip-healthy", help="Skip healthy workloads"),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-V", help="Enable verbose logging"),
    ] = False,
) -> None:
    """Scan multiple GKE clusters and produce a fleet health report."""
    try:
        if verbose:
            _apply_subcommand_log_flags(verbose=True, debug=False)

        from vaig.core.config import get_settings

        settings = get_settings()
        fleet_config = settings.fleet

        if not fleet_config.clusters:
            _console.print(
                "[bold red]Error:[/bold red] No fleet clusters configured.\n"
                "Add a [cyan]fleet:[/cyan] section with [cyan]clusters:[/cyan] "
                "to your config file (vaig.yaml or config/default.yaml).",
            )
            raise typer.Exit(code=1)

        # Apply CLI overrides — use click parameter source detection
        ctx = click.get_current_context()
        if ctx.get_parameter_source("parallel") == click.core.ParameterSource.COMMANDLINE:
            fleet_config = fleet_config.model_copy(update={"parallel": parallel})
        if ctx.get_parameter_source("max_workers") == click.core.ParameterSource.COMMANDLINE:
            fleet_config = fleet_config.model_copy(update={"max_workers": max_workers})
        if ctx.get_parameter_source("budget") == click.core.ParameterSource.COMMANDLINE:
            fleet_config = fleet_config.model_copy(update={"daily_budget_usd": budget})

        # Apply cluster-level CLI overrides to all clusters
        updated_clusters = []
        for cluster in fleet_config.clusters:
            updates: dict[str, object] = {"skip_healthy": skip_healthy}
            if all_namespaces:
                updates["all_namespaces"] = True
                updates["namespace"] = ""
            elif namespace:
                updates["namespace"] = namespace
                updates["all_namespaces"] = False
            updated_clusters.append(cluster.model_copy(update=updates))
        fleet_config = fleet_config.model_copy(update={"clusters": updated_clusters})

        # Budget pre-check warning
        if fleet_config.daily_budget_usd > 0:
            n_clusters = len(fleet_config.clusters)
            _console.print(
                f"[yellow]Budget:[/yellow] ${fleet_config.daily_budget_usd:.2f} "
                f"for {n_clusters} cluster{'s' if n_clusters != 1 else ''}",
            )

        from vaig.core.fleet import FleetRunner

        runner = FleetRunner()

        _console.print(
            f"\n[bold cyan]🚀 Fleet Scan[/bold cyan] — "
            f"{len(fleet_config.clusters)} cluster{'s' if len(fleet_config.clusters) != 1 else ''}"
            f"{' (parallel)' if fleet_config.parallel else ' (sequential)'}",
        )

        if fleet_config.parallel:
            report = runner.run_parallel(settings, fleet_config)
        else:
            report = runner.run(settings, fleet_config)

        # Display
        from vaig.cli.display import print_fleet_summary_panel

        print_fleet_summary_panel(report, detailed=detailed, console=_console)

        # Export
        if export:
            from vaig.cli.export import export_fleet

            output = export_fleet(report, fmt=export)
            _console.print(output)

        # Exit code: non-zero only if ALL clusters failed
        all_failed = all(cr.status == "error" for cr in report.clusters)
        if all_failed:
            _console.print("[bold red]All clusters failed.[/bold red]")
            raise typer.Exit(code=1)

        if report.budget_exceeded:
            _console.print(
                "[yellow]⚠ Budget exceeded — some clusters were skipped.[/yellow]"
            )

    except typer.Exit:
        raise
    except Exception as exc:
        handle_cli_error(exc)

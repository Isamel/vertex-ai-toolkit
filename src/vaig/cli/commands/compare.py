"""Compare CLI — cross-cluster deployment comparison subcommand.

Registers ``vaig compare`` as a Typer sub-app with flags for
cluster selection, namespace, deployment, and JSON export.
"""

from __future__ import annotations

import logging
import sys
from typing import Annotated, Optional

import typer
from rich.console import Console

logger = logging.getLogger(__name__)

compare_app = typer.Typer(help="Compare deployments across clusters")

_console = Console()


def register(app: typer.Typer) -> None:
    """Register the compare sub-app on the main Typer app."""
    app.add_typer(compare_app, name="compare")


@compare_app.command()
def run(
    clusters: Annotated[
        str,
        typer.Option("--clusters", help="Comma-separated cluster names (must match fleet config)"),
    ],
    namespace: Annotated[
        str,
        typer.Option("--namespace", "-n", help="Kubernetes namespace"),
    ],
    deployment: Annotated[
        str,
        typer.Option("--deployment", "-d", help="Deployment name to compare"),
    ],
    export: Annotated[
        Optional[str],  # noqa: UP007, UP045
        typer.Option("--export", help="Export format: json"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
) -> None:
    """Compare a deployment across multiple GKE clusters."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

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

    # Resolve cluster names → FleetCluster objects (REQ-CMP-07/SC-07)
    cluster_names = [c.strip() for c in clusters.split(",") if c.strip()]
    cluster_map = {c.name: c for c in fleet_config.clusters}

    resolved = []
    not_found = []
    for name in cluster_names:
        if name in cluster_map:
            resolved.append(cluster_map[name])
        else:
            not_found.append(name)

    if not_found:
        available = ", ".join(sorted(cluster_map.keys()))
        _console.print(
            f"[bold red]Error:[/bold red] Cluster(s) not found in fleet config: "
            f"{', '.join(not_found)}\n"
            f"Available clusters: {available}",
        )
        raise typer.Exit(code=1)

    if len(resolved) < 2:
        _console.print(
            "[bold red]Error:[/bold red] At least 2 clusters are required for comparison.",
        )
        raise typer.Exit(code=1)

    from vaig.core.compare import CompareRunner

    runner = CompareRunner(
        clusters=resolved,
        namespace=namespace,
        deployment=deployment,
        max_workers=fleet_config.max_workers,
        settings=settings,
    )

    _console.print(
        f"\n[bold cyan]🔍 Cross-Cluster Compare[/bold cyan] — "
        f"{deployment} in {namespace} across "
        f"{len(resolved)} cluster{'s' if len(resolved) != 1 else ''}",
    )

    report = runner.run_parallel()

    # Display or export
    if export:
        from vaig.cli.export import export_compare

        output = export_compare(report, fmt=export)
        _console.print(output)
    else:
        from vaig.cli.display import print_compare_report

        print_compare_report(report, console=_console)

    # Exit code: non-zero if ALL clusters failed
    if report.snapshots == {} and report.errors:
        _console.print("[bold red]All clusters failed.[/bold red]")
        raise typer.Exit(code=1)

    sys.exit(0)

"""Schedule command — manage scheduled health scans.

Provides ``vaig schedule`` subcommands: ``add``, ``list``, ``remove``,
``run-now``, ``start``, ``stop``, ``status``.

The ``start`` command launches the scheduler engine in the foreground
(similar to ``vaig webhook-server``), while the management commands
(``add``, ``list``, ``remove``, ``run-now``) operate against the
scheduler engine instance.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Annotated, Any, TypeVar

import typer

from vaig.cli._helpers import (
    console,
    err_console,
)

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from vaig.core.config import Settings
    from vaig.core.scheduler import SchedulerEngine

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

schedule_app = typer.Typer(
    name="schedule",
    help="Manage scheduled health scans.",
    no_args_is_help=True,
)


def register(app: typer.Typer) -> None:
    """Register the ``schedule`` sub-app on the root Typer app."""
    app.add_typer(schedule_app, name="schedule")


# ── Shared helpers ────────────────────────────────────────────


def _get_engine() -> tuple[SchedulerEngine, Settings]:
    """Lazily import and build a SchedulerEngine + Settings.

    Returns ``(engine, settings)`` or exits with error.
    """
    try:
        from vaig.core.config import get_settings
        from vaig.core.scheduler import SchedulerEngine as _Eng

        settings = get_settings()
        engine = _Eng(settings)
        return engine, settings
    except Exception as exc:
        err_console.print(f"[red]Failed to initialise scheduler:[/red] {exc}")
        raise typer.Exit(code=1) from None


def _run_async(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


# ── Commands ──────────────────────────────────────────────────


@schedule_app.command("add")
def schedule_add(
    cluster: Annotated[
        str,
        typer.Option("--cluster", "-c", help="GKE cluster name to scan"),
    ],
    namespace: Annotated[
        str,
        typer.Option("--namespace", "-n", help="Namespace to scan (empty = default)"),
    ] = "",
    interval: Annotated[
        int,
        typer.Option("--interval", "-i", help="Scan interval in minutes"),
    ] = 0,
    cron: Annotated[
        str,
        typer.Option("--cron", help="Cron expression (e.g. '*/30 * * * *')"),
    ] = "",
    all_namespaces: Annotated[
        bool,
        typer.Option("--all-namespaces", "-A", help="Scan all non-system namespaces"),
    ] = False,
    skip_healthy: Annotated[
        bool,
        typer.Option("--skip-healthy/--include-healthy", help="Skip healthy services"),
    ] = True,
) -> None:
    """Add a new scheduled health scan for a GKE cluster."""
    from vaig.core.config import ScheduleTarget

    engine, _settings = _get_engine()
    target = ScheduleTarget(
        cluster_name=cluster,
        namespace=namespace,
        all_namespaces=all_namespaces,
        skip_healthy=skip_healthy,
    )

    async def _add() -> str:
        await engine.start(process=False)
        try:
            return await engine.add_schedule(
                target,
                interval_minutes=interval or None,
                cron=cron or None,
            )
        finally:
            await engine.stop()

    schedule_id = _run_async(_add())
    console.print(f"[green]✓[/green] Schedule created: [bold]{schedule_id}[/bold]")
    console.print(f"  Cluster: {cluster}")
    if namespace:
        console.print(f"  Namespace: {namespace}")
    if cron:
        console.print(f"  Cron: {cron}")
    elif interval:
        console.print(f"  Interval: {interval}m")
    else:
        console.print(f"  Interval: {_settings.schedule.default_interval_minutes}m (default)")


@schedule_app.command("list")
def schedule_list() -> None:
    """List all registered schedules."""
    engine, _settings = _get_engine()

    async def _list() -> list[Any]:
        await engine.start(process=False)
        try:
            return await engine.list_schedules()
        finally:
            await engine.stop()

    schedules = _run_async(_list())

    if not schedules:
        console.print("[dim]No schedules configured.[/dim]")
        return

    from rich.table import Table

    table = Table(title="Scheduled Health Scans")
    table.add_column("ID", style="cyan", max_width=12)
    table.add_column("Cluster", style="bold")
    table.add_column("Namespace")
    table.add_column("Interval")
    table.add_column("Cron")
    table.add_column("Status")
    table.add_column("Next Run")

    for sched in schedules:
        status = "[yellow]paused[/yellow]" if sched.paused else "[green]active[/green]"
        interval_str = f"{sched.interval_minutes}m" if sched.interval_minutes else "—"
        cron_str = sched.cron_expression or "—"
        next_fire = (
            sched.next_fire_time.strftime("%Y-%m-%d %H:%M UTC")
            if sched.next_fire_time
            else "—"
        )
        table.add_row(
            sched.schedule_id[:12],
            sched.cluster_name,
            sched.namespace or "(default)",
            interval_str,
            cron_str,
            status,
            next_fire,
        )

    console.print(table)


@schedule_app.command("remove")
def schedule_remove(
    schedule_id: Annotated[
        str,
        typer.Argument(help="Schedule ID to remove"),
    ],
) -> None:
    """Remove an existing schedule."""
    engine, _settings = _get_engine()

    async def _remove() -> bool:
        await engine.start(process=False)
        try:
            return await engine.remove_schedule(schedule_id)
        finally:
            await engine.stop()

    removed = _run_async(_remove())

    if removed:
        console.print(f"[green]✓[/green] Schedule [bold]{schedule_id[:12]}[/bold] removed.")
    else:
        err_console.print(f"[red]✗[/red] Schedule [bold]{schedule_id[:12]}[/bold] not found.")
        raise typer.Exit(code=1)


@schedule_app.command("run-now")
def schedule_run_now(
    schedule_id: Annotated[
        str,
        typer.Argument(help="Schedule ID to trigger immediately"),
    ],
) -> None:
    """Trigger an immediate scan for an existing schedule."""
    engine, _settings = _get_engine()

    async def _run() -> Any:
        await engine.start(process=False)
        try:
            return await engine.run_now(schedule_id)
        finally:
            await engine.stop()

    console.print(f"[cyan]Triggering scan for schedule {schedule_id[:12]}…[/cyan]")

    try:
        result = _run_async(_run())
        console.print(
            f"[green]✓[/green] Scan complete — status: [bold]{result.status}[/bold], "
            f"alerts sent: {result.alerts_sent}"
        )
    except ValueError as exc:
        err_console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from None


@schedule_app.command("start")
def schedule_start(
    host: Annotated[
        str,
        typer.Option("--host", "-h", help="Bind address (unused — scheduler runs in-process)"),
    ] = "0.0.0.0",  # noqa: S104
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Bind port (unused — reserved for future API)"),
    ] = 8081,
) -> None:
    """Start the scheduler engine in the foreground.

    Runs until interrupted (Ctrl+C).  Uses targets from
    ``schedule.targets`` in the configuration file.

    Examples:
        vaig schedule start
    """
    engine, settings = _get_engine()
    cfg = settings.schedule

    if not cfg.enabled:
        err_console.print(
            "[yellow]⚠[/yellow] Scheduler is disabled. "
            "Set [bold]schedule.enabled: true[/bold] or configure targets in vaig.yaml"
        )
        raise typer.Exit(code=1)

    console.print(
        "[bold cyan]vaig schedule start[/bold cyan] — "
        f"scheduler engine running with {len(cfg.targets)} target(s)"
    )
    console.print(f"  Budget: {cfg.daily_max_analyses} analyses/day")
    console.print(f"  Alert threshold: {cfg.alert_severity_threshold}")

    async def _run_engine() -> None:
        await engine.start(process=True)
        try:
            # Register configured targets
            for target in cfg.targets:
                sid = await engine.add_schedule(
                    target,
                    interval_minutes=cfg.default_interval_minutes,
                    cron=cfg.cron_expression,
                )
                console.print(
                    f"  [green]✓[/green] {target.cluster_name}"
                    f"{'/' + target.namespace if target.namespace else ''}"
                    f" → schedule {sid[:12]}"
                )

            console.print("\n[dim]Press Ctrl+C to stop.[/dim]\n")

            # Block until cancelled
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass
        finally:
            await engine.stop()
            console.print("\n[bold]Scheduler stopped.[/bold]")

    try:
        asyncio.run(_run_engine())
    except KeyboardInterrupt:
        console.print("\n[bold]Scheduler stopped.[/bold]")


@schedule_app.command("stop")
def schedule_stop() -> None:
    """Stop the scheduler engine.

    Note: The scheduler runs in the foreground and is stopped via
    Ctrl+C.  This command is a placeholder for future daemon support.
    """
    console.print(
        "[yellow]The scheduler runs in the foreground.[/yellow]\n"
        "Use [bold]Ctrl+C[/bold] in the terminal running "
        "[bold]vaig schedule start[/bold] to stop it."
    )


@schedule_app.command("status")
def schedule_status() -> None:
    """Show scheduler engine status and configuration."""
    try:
        from vaig.core.config import get_settings

        settings = get_settings()
    except Exception as exc:
        err_console.print(f"[red]Failed to load settings:[/red] {exc}")
        raise typer.Exit(code=1) from None

    cfg = settings.schedule

    from rich.panel import Panel

    status_text = (
        f"[bold]Enabled:[/bold]  {'[green]yes[/green]' if cfg.enabled else '[red]no[/red]'}\n"
        f"[bold]Targets:[/bold]  {len(cfg.targets)}\n"
        f"[bold]Interval:[/bold] {cfg.default_interval_minutes}m\n"
        f"[bold]Cron:[/bold]     {cfg.cron_expression or '(none)'}\n"
        f"[bold]Budget:[/bold]   {cfg.daily_max_analyses}/day\n"
        f"[bold]Alert:[/bold]    severity ≥ {cfg.alert_severity_threshold}\n"
        f"[bold]DB path:[/bold]  {cfg.db_path}\n"
        f"[bold]Store:[/bold]    {'yes' if cfg.store_results else 'no'}"
    )

    console.print(Panel(status_text, title="Scheduler Configuration", border_style="cyan"))

    if cfg.targets:
        from rich.table import Table

        table = Table(title="Configured Targets")
        table.add_column("Cluster", style="bold")
        table.add_column("Namespace")
        table.add_column("All NS")
        table.add_column("Skip Healthy")

        for t in cfg.targets:
            table.add_row(
                t.cluster_name,
                t.namespace or "(default)",
                "✓" if t.all_namespaces else "—",
                "✓" if t.skip_healthy else "—",
            )
        console.print(table)

"""Remediate command — execute recommended actions from health reports.

Lists recommended actions from the last health report with safety tier
classification, and allows executing them with appropriate approval
workflows (SAFE / REVIEW / BLOCKED).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Annotated

import typer
from rich.panel import Panel
from rich.table import Table

from vaig.cli._helpers import (
    _apply_subcommand_log_flags,
    _get_settings,
    console,
    handle_cli_error,
    track_command,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vaig.core.config import Settings
    from vaig.core.remediation import ClassifiedCommand
    from vaig.skills.service_health.schema import HealthReport, RecommendedAction

logger = logging.getLogger(__name__)

# ── Tier colour mapping ──────────────────────────────────────

_TIER_COLORS: dict[str, str] = {
    "safe": "green",
    "review": "yellow",
    "blocked": "red",
}


def _tier_label(tier: str) -> str:
    """Return a Rich-formatted tier label with colour."""
    color = _TIER_COLORS.get(tier.lower(), "white")
    return f"[{color}]{tier.upper()}[/{color}]"


# ── Report loading ───────────────────────────────────────────


def _load_last_report() -> tuple[dict[str, object], str] | None:
    """Load the most recent health report from the local ReportStore.

    Returns a ``(report_dict, run_id)`` tuple or ``None`` if no reports
    are available.
    """
    try:
        from vaig.core.report_store import ReportStore

        store = ReportStore()
        records = store.read_reports(last=1)
        if records:
            record = records[-1]
            report = record.get("report")
            run_id = record.get("run_id", "")
            if report is not None:
                return report, str(run_id)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:  # noqa: BLE001
        logger.debug("Failed to load report from ReportStore", exc_info=True)
    return None


# ── Command registration ─────────────────────────────────────


def register(app: typer.Typer) -> None:
    """Register the remediate command on the given Typer app."""

    @app.command()
    @track_command
    def remediate(
        list_actions: Annotated[
            bool,
            typer.Option(
                "--list",
                help="List all recommended actions from the last health report",
            ),
        ] = False,
        finding: Annotated[
            str | None,
            typer.Option(
                "--finding",
                "-f",
                help="Finding ID to remediate (from --list output)",
            ),
        ] = None,
        approve: Annotated[
            bool,
            typer.Option(
                "--approve",
                help="Auto-approve SAFE tier commands for execution",
            ),
        ] = False,
        dry_run: Annotated[
            bool,
            typer.Option(
                "--dry-run",
                help="Show what would happen without executing",
            ),
        ] = False,
        execute: Annotated[
            bool,
            typer.Option(
                "--execute",
                help="Approve and execute REVIEW tier commands after showing plan",
            ),
        ] = False,
        config: Annotated[
            str | None,
            typer.Option("--config", "-c", help="Path to config YAML"),
        ] = None,
        verbose: Annotated[
            bool,
            typer.Option("--verbose", "-V", help="Enable verbose logging (INFO level)"),
        ] = False,
        debug: Annotated[
            bool,
            typer.Option("--debug", "-d", help="Enable debug logging (DEBUG level)"),
        ] = False,
    ) -> None:
        """Execute recommended actions from the last health report.

        Safety tiers control execution:
        - SAFE (green): auto-executable with --approve
        - REVIEW (yellow): requires --execute after plan review
        - BLOCKED (red): never executed — shows why and suggests alternatives

        Examples:
            vaig remediate --list
            vaig remediate --finding crashloop-payment-svc --dry-run
            vaig remediate --finding crashloop-payment-svc --approve
            vaig remediate --finding high-memory-usage --execute
        """
        _apply_subcommand_log_flags(verbose=verbose, debug=debug)

        try:
            settings = _get_settings(config)

            # ── Check remediation feature flag ──
            if not settings.remediation.enabled:
                console.print(
                    Panel(
                        "[yellow]Remediation is disabled.[/yellow]\n\n"
                        "Enable it in your config:\n"
                        "  [dim]remediation.enabled = true[/dim]\n\n"
                        "Or set the environment variable:\n"
                        "  [dim]VAIG_REMEDIATION__ENABLED=true[/dim]",
                        title="[yellow]⚠ Remediation Disabled[/yellow]",
                        border_style="yellow",
                    )
                )
                raise typer.Exit(code=1)

            # ── Validate arguments ──
            if not list_actions and not finding:
                console.print(
                    "[red]Error:[/red] Specify --list to see actions or "
                    "--finding <id> to remediate a specific finding."
                )
                raise typer.Exit(code=1)

            # ── Load the last report ──
            loaded = _load_last_report()
            if loaded is None:
                console.print(
                    Panel(
                        "[yellow]No health report found.[/yellow]\n\n"
                        "Run a health scan first:\n"
                        "  [dim]vaig discover --namespace <ns>[/dim]\n"
                        "  [dim]vaig live \"check cluster health\"[/dim]",
                        title="[yellow]⚠ No Report[/yellow]",
                        border_style="yellow",
                    )
                )
                raise typer.Exit(code=1)

            report_dict, run_id = loaded

            # ── Parse recommendations ──
            from vaig.skills.service_health.schema import HealthReport

            report = HealthReport.model_validate(report_dict)
            recommendations = report.recommendations

            if not recommendations:
                console.print(
                    "[green]✓[/green] No recommended actions in the last report."
                )
                raise typer.Exit(code=0)

            # ── --list: show table of actions ──
            if list_actions:
                _display_actions_table(recommendations, settings)
                return

            # ── --finding <id>: remediate a specific finding ──
            if finding:
                _handle_finding(
                    finding_id=finding,
                    recommendations=recommendations,
                    report=report,
                    settings=settings,
                    approve=approve,
                    dry_run=dry_run,
                    execute=execute,
                    debug=debug,
                    run_id=run_id,
                )
                return

        except typer.Exit:
            raise  # Let typer exits pass through
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001
            handle_cli_error(exc, debug=debug)


def _display_actions_table(
    recommendations: Sequence[RecommendedAction],
    settings: Settings,
) -> None:
    """Render a Rich table of all recommended actions with tier info."""
    from vaig.core.remediation import CommandClassifier

    rem_config = settings.remediation
    classifier = CommandClassifier(rem_config)

    console.print()
    console.print("[bold cyan]vaig remediate --list[/bold cyan] — recommended actions")
    console.print()

    table = Table(show_lines=True)
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Related Findings", style="cyan", no_wrap=True, max_width=30)
    table.add_column("Title", max_width=40)
    table.add_column("Command", style="dim", max_width=50)
    table.add_column("Tier", justify="center", width=10)
    table.add_column("Priority", justify="center", width=10)

    for i, action in enumerate(recommendations, start=1):
        command = action.command or ""
        if command:
            classified = classifier.classify(command)
            tier = classified.tier.value
        else:
            tier = "blocked"

        finding_ids = ", ".join(action.related_findings) if action.related_findings else "—"

        table.add_row(
            str(i),
            finding_ids,
            action.title,
            command or "[dim]no command[/dim]",
            _tier_label(tier),
            str(action.priority),
        )

    console.print(table)
    console.print()
    console.print(
        "[dim]Use [bold]vaig remediate --finding <id>[/bold] "
        "with --approve (SAFE), --execute (REVIEW), or --dry-run[/dim]"
    )
    console.print()


def _handle_finding(
    *,
    finding_id: str,
    recommendations: Sequence[RecommendedAction],
    report: HealthReport,
    settings: Settings,
    approve: bool,
    dry_run: bool,
    execute: bool,
    debug: bool,
    run_id: str = "",
) -> None:
    """Handle remediation of a specific finding."""
    from vaig.core.gke import build_gke_config
    from vaig.core.remediation import CommandClassifier, RemediationExecutor, SafetyTier

    rem_config = settings.remediation
    classifier = CommandClassifier(rem_config)

    # ── Validate report cluster matches current context ──
    report_cluster = report.metadata.cluster_name
    if report_cluster:
        from vaig.core.event_bus import EventBus

        gke_config = build_gke_config(settings)
        executor = RemediationExecutor(rem_config, EventBus.get())
        if not executor.validate_context(report_cluster, gke_config):
            console.print(
                Panel(
                    f"[red]Cluster mismatch![/red]\n\n"
                    f"Report was generated for: [bold]{report_cluster}[/bold]\n"
                    f"Current context points to: [bold]{gke_config.cluster_name}[/bold]\n\n"
                    f"Run against the correct cluster or generate a new report.",
                    title="[red]✗ Context Mismatch[/red]",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1)

    # Find matching recommendations by related_findings or by index
    matching = [
        a for a in recommendations
        if finding_id in a.related_findings
    ]

    # Also try matching by recommendation index (1-based)
    if not matching:
        try:
            idx = int(finding_id)
            if 1 <= idx <= len(recommendations):
                matching = [recommendations[idx - 1]]
        except ValueError:
            pass

    # Also try partial match on title
    if not matching:
        matching = [
            a for a in recommendations
            if finding_id.lower() in a.title.lower()
        ]

    if not matching:
        console.print(
            f"[red]Error:[/red] No recommendation found for finding "
            f"[bold]{finding_id}[/bold].\n"
            f"Use [bold]vaig remediate --list[/bold] to see available actions."
        )
        raise typer.Exit(code=1)

    for action in matching:
        command = action.command or ""
        if not command:
            console.print(
                Panel(
                    f"[yellow]No command associated with this action.[/yellow]\n\n"
                    f"[bold]{action.title}[/bold]\n"
                    f"{action.description}",
                    title="[yellow]⚠ No Command[/yellow]",
                    border_style="yellow",
                )
            )
            continue

        classified = classifier.classify(command)

        # ── Show command preview panel ──
        tier_color = _TIER_COLORS.get(classified.tier.value, "white")
        console.print()
        console.print(
            Panel(
                f"[bold]{action.title}[/bold]\n"
                f"{action.description}\n\n"
                f"[bold]Command:[/bold] {classified.raw_command}\n"
                f"[bold]Tier:[/bold] {_tier_label(classified.tier.value)}\n"
                f"[bold]Risk:[/bold] {action.risk or 'Not specified'}",
                title=f"[{tier_color}]Remediation — {classified.tier.value.upper()}[/{tier_color}]",
                border_style=tier_color,
            )
        )

        # ── BLOCKED ──
        if classified.tier == SafetyTier.BLOCKED:
            reason = classified.block_reason or "Command is blocked by safety policy"
            console.print(
                Panel(
                    f"[red]This command is BLOCKED and cannot be executed.[/red]\n\n"
                    f"[bold]Reason:[/bold] {reason}\n\n"
                    f"[bold]Suggested:[/bold] Execute this command manually after "
                    f"careful review:\n"
                    f"  [dim]{classified.raw_command}[/dim]",
                    title="[red]✗ Blocked[/red]",
                    border_style="red",
                )
            )
            continue

        # ── dry_run ──
        if dry_run:
            console.print(
                Panel(
                    f"[cyan][DRY RUN][/cyan] Would execute:\n\n"
                    f"  [bold]{classified.raw_command}[/bold]\n\n"
                    f"Tier: {_tier_label(classified.tier.value)}\n"
                    f"No changes will be made.",
                    title="[cyan]Dry Run[/cyan]",
                    border_style="cyan",
                )
            )
            continue

        # ── REVIEW without --execute ──
        if classified.tier == SafetyTier.REVIEW and not execute:
            console.print(
                Panel(
                    f"[yellow]This command requires explicit approval.[/yellow]\n\n"
                    f"Re-run with [bold]--execute[/bold] to approve:\n"
                    f"  [dim]vaig remediate --finding {finding_id} --execute[/dim]",
                    title="[yellow]Review Required[/yellow]",
                    border_style="yellow",
                )
            )
            continue

        # ── SAFE without --approve ──
        if classified.tier == SafetyTier.SAFE and not approve:
            console.print(
                Panel(
                    f"[green]This command is SAFE to execute.[/green]\n\n"
                    f"Re-run with [bold]--approve[/bold] to execute:\n"
                    f"  [dim]vaig remediate --finding {finding_id} --approve[/dim]",
                    title="[green]Approval Needed[/green]",
                    border_style="green",
                )
            )
            continue

        # ── Execute ──
        _execute_remediation(action, classified, settings, debug=debug, run_id=run_id)


def _execute_remediation(
    action: RecommendedAction,
    classified: ClassifiedCommand,
    settings: Settings,
    *,
    debug: bool = False,
    run_id: str = "",
) -> None:
    """Execute a remediation command and display the result."""
    from vaig.core.event_bus import EventBus
    from vaig.core.gke import build_gke_config
    from vaig.core.remediation import RemediationExecutor

    rem_config = settings.remediation
    bus = EventBus.get()

    # ── Wire review gate when enabled ──
    review_store = None
    review_config = None
    if settings.review.enabled:
        from vaig.core.review_store import ReviewStore

        # Fail closed: refuse to execute when review is required but no
        # run_id was provided — callers must pass --run-id explicitly.
        if settings.review.require_review_for_remediation and not run_id:
            console.print(
                Panel(
                    "[red]Review is required for remediation but no "
                    "[bold]--run-id[/bold] was provided.[/red]\n\n"
                    "Pass [bold]--run-id <RUN_ID>[/bold] to link this "
                    "execution to a reviewed health report.",
                    title="[red]✗ Review Gate[/red]",
                    border_style="red",
                )
            )
            return

        review_store = ReviewStore()
        review_config = settings.review

    executor = RemediationExecutor(
        rem_config,
        bus,
        review_store=review_store,
        review_config=review_config,
    )
    gke_config = build_gke_config(settings)

    console.print("[dim]Executing...[/dim]")

    try:
        result = asyncio.run(
            executor.execute(
                action,
                classified,
                gke_config,
                approved=True,
                run_id=run_id or None,
            )
        )

        if result.error:
            console.print(
                Panel(
                    f"[red]Execution failed:[/red]\n\n{result.output}",
                    title="[red]✗ Error[/red]",
                    border_style="red",
                )
            )
        else:
            console.print(
                Panel(
                    f"[green]Command executed successfully.[/green]\n\n{result.output}",
                    title="[green]✓ Success[/green]",
                    border_style="green",
                )
            )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001
        console.print(
            Panel(
                f"[red]Execution error:[/red] {exc}",
                title="[red]✗ Error[/red]",
                border_style="red",
            )
        )
        if debug:
            console.print_exception()

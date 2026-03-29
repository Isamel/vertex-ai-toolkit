"""Feedback command — submit quality feedback for a vaig run.

Allows users to rate a completed live analysis (1-5) and optionally
leave a text comment.  Feedback is exported to the configured BigQuery
``feedback`` table so it can be used for quality tracking.
"""

from __future__ import annotations

import logging
from typing import Annotated

import typer

from vaig.cli._helpers import (
    _apply_subcommand_log_flags,
    _get_settings,
    console,
    err_console,
    handle_cli_error,
    track_command,
)

logger = logging.getLogger(__name__)


def register(app: typer.Typer) -> None:
    """Register the feedback command on the given Typer app."""

    @app.command()
    @track_command
    def feedback(
        rating: Annotated[
            int,
            typer.Option(
                "--rating",
                "-r",
                help="Rating from 1 (poor) to 5 (excellent)",
                min=1,
                max=5,
            ),
        ],
        run_id: Annotated[
            str | None,
            typer.Option("--run-id", help="Run ID to attach feedback to"),
        ] = None,
        last: Annotated[
            bool,
            typer.Option("--last", help="Use the most recent run ID"),
        ] = False,
        comment: Annotated[
            str,
            typer.Option("--comment", "-m", help="Free-text feedback comment"),
        ] = "",
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
        """Submit quality feedback for a vaig live analysis run.

        Rate a completed analysis from 1 (poor) to 5 (excellent) and
        optionally include a text comment. Requires either --run-id or
        --last to identify which run the feedback belongs to.

        Examples:
            vaig feedback --rating 5 --last
            vaig feedback -r 4 -m "Great analysis" --last
            vaig feedback -r 3 --run-id 20250601T120000Z
        """
        _apply_subcommand_log_flags(verbose=verbose, debug=debug)

        try:
            # ── Validate run-id / --last mutual exclusivity ──
            if run_id and last:
                err_console.print(
                    "[red]Error:[/red] --run-id and --last are mutually exclusive."
                )
                raise typer.Exit(code=1)

            if not run_id and not last:
                err_console.print(
                    "[red]Error:[/red] Provide either --run-id <id> or --last."
                )
                raise typer.Exit(code=1)

            # ── Resolve run ID ──
            effective_run_id = run_id or ""
            if last:
                from vaig.core.export import get_last_run_id

                stored_id = get_last_run_id()
                if not stored_id:
                    err_console.print(
                        "[red]Error:[/red] No previous run ID found. "
                        "Run [bold]vaig live[/bold] first or specify --run-id explicitly."
                    )
                    raise typer.Exit(code=1)
                effective_run_id = stored_id

            # ── Load settings and export ──
            settings = _get_settings(config)
            if not settings.export.enabled:
                err_console.print(
                    "[yellow]Warning:[/yellow] Export is disabled in your configuration. "
                    "Enable it with [bold]export.enabled=true[/bold] to save feedback."
                )
                raise typer.Exit(code=1)

            from vaig.core.export import DataExporter

            exporter = DataExporter(settings.export)
            success = exporter.export_feedback_to_bigquery(
                rating=rating,
                comment=comment,
                run_id=effective_run_id,
            )

            # ── Display result ──
            if success:
                stars = "\u2605" * rating + "\u2606" * (5 - rating)
                console.print()
                console.print(f"  [green]\u2713[/green] Feedback submitted  {stars}")
                console.print(f"    Run ID : {effective_run_id}")
                if comment:
                    console.print(f"    Comment: {comment}")
                console.print()
            else:
                err_console.print(
                    "[red]\u2717 Failed to submit feedback.[/red] "
                    "Check your export configuration and GCP credentials."
                )
                raise typer.Exit(code=1)

        except typer.Exit:
            raise  # Let typer exits pass through
        except Exception as exc:  # noqa: BLE001
            handle_cli_error(exc, debug=debug)

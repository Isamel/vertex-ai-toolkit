"""Fine-tuning pipeline CLI commands — prepare training data and submit tuning jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()
err_console = Console(stderr=True)

# ── Typer app ─────────────────────────────────────────────────

train_app = typer.Typer(
    name="train",
    help="Fine-tuning pipeline (prepare data, submit tuning jobs)",
    no_args_is_help=True,
)


# ── Commands ──────────────────────────────────────────────────


@train_app.command()
def prepare(
    min_rating: Annotated[
        int | None,
        typer.Option("--min-rating", help="Minimum feedback rating to include (1-5)"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output JSONL file path"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Report statistics without writing a file"),
    ] = False,
    max_examples: Annotated[
        int | None,
        typer.Option("--max-examples", help="Maximum number of examples to include"),
    ] = None,
) -> None:
    """Extract rated examples from BigQuery and generate training JSONL."""
    from vaig.cli._helpers import _get_settings
    from vaig.core.training import TrainingDataPreparer

    settings = _get_settings()

    if not settings.training.enabled:
        err_console.print(
            "[red]Error:[/red] Training is not enabled. "
            "Set [cyan]training.enabled = true[/cyan] in config or "
            "[cyan]VAIG_TRAINING__ENABLED=true[/cyan]."
        )
        raise typer.Exit(code=1)

    # Apply CLI overrides to training config
    tc = settings.training
    if min_rating is not None:
        tc = tc.model_copy(update={"min_rating": min_rating})
    if max_examples is not None:
        tc = tc.model_copy(update={"max_examples": max_examples})

    try:
        preparer = TrainingDataPreparer(settings.export, tc, bq_client=None)
        result = preparer.prepare(output_path=output, dry_run=dry_run)
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from None
    except ImportError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from None
    except Exception as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    # Display results
    table = Table(title="Training Data Preparation")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Examples", str(result.total_examples))
    table.add_row("Avg Rating", str(result.avg_rating))
    table.add_row("Estimated Tokens", str(result.estimated_tokens))
    if not dry_run:
        table.add_row("Output File", str(result.jsonl_path))
    else:
        table.add_row("Mode", "dry-run (no file written)")
    console.print(table)


@train_app.command()
def submit(
    input_file: Annotated[
        Path,
        typer.Option("--input", "-i", help="Path to JSONL training file"),
    ],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate and report without submitting"),
    ] = False,
) -> None:
    """Upload training JSONL and submit a Vertex AI tuning job."""
    from vaig.cli._helpers import _get_settings
    from vaig.core.training import TuningJobSubmitter

    settings = _get_settings()

    if not settings.training.enabled:
        err_console.print(
            "[red]Error:[/red] Training is not enabled. "
            "Set [cyan]training.enabled = true[/cyan] in config or "
            "[cyan]VAIG_TRAINING__ENABLED=true[/cyan]."
        )
        raise typer.Exit(code=1)

    try:
        submitter = TuningJobSubmitter(
            settings.export,
            settings.training,
            gcs_client=None,
            genai_client=None,
        )
        result = submitter.submit(jsonl_path=input_file, dry_run=dry_run)
    except (FileNotFoundError, ValueError) as exc:
        err_console.print(f"[red]Validation error:[/red] {exc}")
        raise typer.Exit(code=1) from None
    except ImportError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from None
    except Exception as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    # Display results
    table = Table(title="Tuning Job Submission")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Job Name", result.job_name)
    table.add_row("Base Model", result.base_model)
    table.add_row("Status", result.status)
    if result.gcs_uri:
        table.add_row("GCS URI", result.gcs_uri)
    console.print(table)

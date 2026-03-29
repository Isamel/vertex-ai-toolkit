"""Optimize command — analyze tool call efficiency and report quality.

Loads historical tool call records, computes per-tool statistics,
detects redundant calls, and displays actionable suggestions.

When invoked with ``--reports``, analyzes HealthReport quality
instead — computing signals such as hallucination rate, evidence
depth, actionability, and completeness.
"""

from __future__ import annotations

import logging
from typing import Annotated

import typer
from rich.table import Table

from vaig.cli._helpers import (
    _apply_subcommand_log_flags,
    _get_settings,
    console,
    handle_cli_error,
    track_command,
)

logger = logging.getLogger(__name__)


def register(app: typer.Typer) -> None:
    """Register the optimize command on the given Typer app."""

    @app.command()
    @track_command
    def optimize(
        last: Annotated[
            int,
            typer.Option("--last", "-n", help="Number of recent runs to analyze", min=1),
        ] = 50,
        reports: Annotated[
            bool,
            typer.Option("--reports", help="Analyze report quality instead of tool calls"),
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
        """Analyze tool call efficiency and suggest optimizations.

        Scans recent run history for per-tool statistics, redundant calls,
        and performance patterns, then prints actionable suggestions.

        Use ``--reports`` to switch to report quality analysis, which
        computes hallucination rate, evidence depth, actionability, and
        other quality signals from past HealthReports.

        Examples:
            vaig optimize
            vaig optimize --last 20
            vaig optimize --reports
            vaig optimize --reports --last 10
        """
        _apply_subcommand_log_flags(verbose=verbose, debug=debug)

        try:
            settings = _get_settings(config)

            if reports:
                _run_report_analysis(last=last)
                return

            console.print()
            console.print("[bold cyan]vaig optimize[/bold cyan] — tool call analysis")
            console.print()

            # ── Build store & optimizer ───────────────────────
            from vaig.core.optimizer import ToolCallOptimizer
            from vaig.core.tool_call_store import ToolCallStore

            base_dir = settings.logging.tool_results_dir
            store = ToolCallStore(base_dir=base_dir)
            optimizer = ToolCallOptimizer(store)

            insights = optimizer.analyze(last_n_runs=last)

            # ── Summary ───────────────────────────────────────
            console.print(
                f"  Analyzed [bold]{insights.total_runs}[/bold] runs, "
                f"[bold]{insights.total_calls}[/bold] total calls, "
                f"[bold]{insights.total_duration_s:.1f}s[/bold] total duration"
            )
            if insights.total_runs:
                console.print(
                    f"  Average [bold]{insights.avg_calls_per_run:.1f}[/bold] "
                    f"calls per run"
                )
            console.print()

            # ── Per-tool stats table ──────────────────────────
            if insights.tools:
                table = Table(title="Per-Tool Statistics", show_lines=False)
                table.add_column("Tool", style="cyan", no_wrap=True)
                table.add_column("Calls", justify="right")
                table.add_column("Failures", justify="right")
                table.add_column("Fail %", justify="right")
                table.add_column("Avg (s)", justify="right")
                table.add_column("Max (s)", justify="right")
                table.add_column("Cache Hits", justify="right")
                table.add_column("Args Combos", justify="right")

                for name, stats in sorted(
                    insights.tools.items(),
                    key=lambda item: item[1].call_count,
                    reverse=True,
                ):
                    fail_pct = f"{stats.failure_rate * 100:.0f}%"
                    fail_style = "red" if stats.failure_rate > 0.5 else ""
                    table.add_row(
                        name,
                        str(stats.call_count),
                        str(stats.failure_count),
                        f"[{fail_style}]{fail_pct}[/{fail_style}]" if fail_style else fail_pct,
                        f"{stats.avg_duration_s:.2f}",
                        f"{stats.max_duration_s:.2f}",
                        str(stats.cache_hit_count),
                        str(stats.unique_arg_combos),
                    )

                console.print(table)
                console.print()

            # ── Redundant calls ───────────────────────────────
            if insights.redundant_calls:
                console.print("[bold yellow]Redundant Calls[/bold yellow]")
                for rc in insights.redundant_calls[:10]:  # top 10
                    console.print(
                        f"  [yellow]⚠[/yellow] {rc.tool_name} called "
                        f"{rc.count}× with same args in run {rc.run_id}"
                    )
                console.print()

            # ── Suggestions ───────────────────────────────────
            if insights.suggestions:
                console.print("[bold green]Suggestions[/bold green]")
                for suggestion in insights.suggestions:
                    console.print(f"  [green]→[/green] {suggestion}")
                console.print()
            elif insights.total_calls > 0:
                console.print("  [green]✓[/green] No optimization issues detected")
                console.print()

        except typer.Exit:
            raise
        except Exception as exc:  # noqa: BLE001
            handle_cli_error(exc, debug=debug)

    # ── Report quality analysis ───────────────────────────────

    def _run_report_analysis(*, last: int) -> None:
        """Analyze report quality and display results with Rich."""
        from vaig.core.prompt_tuner import PromptTuner
        from vaig.core.report_store import ReportStore

        console.print()
        console.print("[bold cyan]vaig optimize --reports[/bold cyan] — report quality analysis")
        console.print()

        store = ReportStore()
        records = store.read_reports(last=last)
        tuner = PromptTuner()
        insights = tuner.analyze_quality(records)

        console.print(
            f"  Analyzed [bold]{insights.total_reports}[/bold] reports"
        )
        console.print()

        if not insights.signals:
            console.print("  No report data found. Run some analyses first.")
            console.print()
            return

        # ── Signals table ─────────────────────────────────
        table = Table(title="Quality Signals", show_lines=False)
        table.add_column("Signal", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right")
        table.add_column("Threshold", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Detail")

        for signal in insights.signals:
            status = "[green]PASS[/green]" if signal.passed else "[red]FAIL[/red]"
            table.add_row(
                signal.name,
                f"{signal.value:.3f}" if isinstance(signal.value, float) else str(signal.value),
                f"{signal.threshold:.2f}" if signal.threshold else "—",
                status,
                signal.detail,
            )

        console.print(table)
        console.print()

        # ── Suggestions ───────────────────────────────────
        if insights.suggestions:
            console.print("[bold green]Prompt Improvement Suggestions[/bold green]")
            for suggestion in insights.suggestions:
                console.print(f"  [green]→[/green] {suggestion}")
            console.print()
        else:
            console.print("  [green]✓[/green] All quality signals healthy")
            console.print()

"""Live command — infrastructure investigation (GKE/GCP)."""

from __future__ import annotations

import logging
import tempfile
import threading
import time
import webbrowser
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import typer
from rich.columns import Columns
from rich.panel import Panel
from rich.rule import Rule
from rich.status import Status
from rich.table import Table
from rich.text import Text

from vaig.cli import _helpers
from vaig.cli._completions import complete_namespace
from vaig.cli._helpers import (
    _apply_subcommand_log_flags,
    _compute_cost_str,
    _handle_export_output,
    _show_coding_summary,
    _show_cost_line,
    console,
    err_console,
    handle_cli_error,
    track_command,
)
from vaig.cli.display import (
    print_colored_report,
    print_cost_breakdown_table,
    print_executive_summary_panel,
    print_recommendations_table,
    print_service_status_table,
    print_severity_detail_blocks,
    print_trend_analysis_table,
    print_watch_diff_summary,
)
from vaig.core.cache import ToolResultCache
from vaig.core.tool_call_store import ToolCallStore
from vaig.skills.service_health.diff import compute_report_diff

if TYPE_CHECKING:
    from vaig.agents.orchestrator import OrchestratorResult
    from vaig.core.attachment_adapter import AttachmentAdapter
    from vaig.core.config import GKEConfig, Settings
    from vaig.core.protocols import GeminiClientProtocol
    from vaig.skills.base import BaseSkill, SkillMetadata
    from vaig.skills.service_health.diff import ReportDiff
    from vaig.skills.service_health.schema import HealthReport

logger = logging.getLogger(__name__)

MINIMUM_WATCH_INTERVAL = 10
"""Minimum seconds between watch iterations to prevent abuse."""

_AGENT_COLORS = [
    "bright_cyan",
    "bright_green",
    "bright_magenta",
    "bright_yellow",
    "bright_blue",
    "bright_red",
    "cyan",
    "green",
]
"""Per-agent colour palette for progress bars and status icons.

Colours are assigned by ``agent_index % len(_AGENT_COLORS)`` so parallel
gatherers in the same pipeline always get a distinct hue."""


def _create_tool_call_store(settings: Settings) -> ToolCallStore | None:
    """Create a :class:`ToolCallStore` if tool-result persistence is enabled.

    Reads ``settings.logging.tool_results`` and
    ``settings.logging.tool_results_dir`` to decide whether to instantiate
    the store.  Returns ``None`` when disabled or on any creation error
    (never crashes the app).
    """
    if not settings.logging.tool_results:
        return None

    try:
        base_dir = Path(settings.logging.tool_results_dir).expanduser()
        store = ToolCallStore(base_dir=base_dir)
        logger.debug("ToolCallStore initialised: %s", base_dir)
        return store
    except Exception:  # noqa: BLE001
        logger.warning("Failed to create ToolCallStore; tool results will not be persisted.", exc_info=True)
        return None


# ── Live tool execution logger ────────────────────────────────


def _truncate_args(args: dict[str, Any], max_len: int = 50) -> str:
    """Format tool args into a compact, truncated string.

    Shows key=value pairs, truncating values longer than *max_len* chars.
    """
    if not args:
        return ""
    parts: list[str] = []
    for k, v in args.items():
        v_str = str(v)
        if len(v_str) > max_len:
            v_str = v_str[:max_len] + "..."
        parts.append(f"{k}={v_str}")
    return ", ".join(parts)


class ToolCallLogger:
    """Callback that prints live tool execution feedback to the console.

    Tracks tool call count and cumulative duration so a final summary
    line can be printed via :meth:`print_summary`.

    Usage::

        tool_logger = ToolCallLogger()
        orchestrator.execute_with_tools(..., on_tool_call=tool_logger)
        tool_logger.print_summary()
    """

    def __init__(self, *, detailed: bool = False) -> None:
        self.tool_count: int = 0
        self.total_duration: float = 0.0
        self.errors: int = 0
        self._error_reasons: list[str] = []
        self._pipeline_start: float = time.perf_counter()
        self.tool_name_counts: Counter[str] = Counter()
        self.pipeline_tool_name_counts: Counter[str] = Counter()
        self.cache_hits: int = 0
        self.detailed = detailed

    @staticmethod
    def _extract_reason(error_message: str) -> str:
        """Extract a short reason category from an error message."""
        if not error_message:
            return "unknown"
        # Take the first line, truncated to 40 chars
        first_line = error_message.split("\n", 1)[0].strip()
        if len(first_line) > 40:
            first_line = first_line[:40] + "..."
        return first_line

    def __call__(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        duration: float,
        success: bool,
        error_message: str = "",
        *,
        cached: bool = False,
    ) -> None:
        """Print a single tool execution line."""
        self.tool_count += 1
        self.total_duration += duration
        self.tool_name_counts[tool_name] += 1

        if cached:
            self.cache_hits += 1

        args_str = _truncate_args(tool_args)

        if success:
            cache_tag = r" [yellow]\[cached][/yellow]" if cached else ""
            status = f"[green]✓[/green]{cache_tag}"
        else:
            self.errors += 1
            # Build error detail for display (truncated to 80 chars)
            if error_message:
                err_short = error_message.split("\n", 1)[0].strip()
                if len(err_short) > 80:
                    err_short = err_short[:80] + "..."
                status = f"[red]✗ FAIL[/red] [dim]{err_short}[/dim]"
                self._error_reasons.append(self._extract_reason(error_message))
            else:
                status = "[red]✗ FAIL[/red]"
                self._error_reasons.append("unknown")

        # Print tool call line only in detailed mode.
        # In non-detailed mode, failed tool calls are logged to the file
        # logger (which always captures DEBUG+) but never printed to the
        # console, keeping the terminal clean.
        if self.detailed:
            console.print(f"  🔧 [cyan]{tool_name}[/cyan]({args_str}) {status} [dim]({duration:.1f}s)[/dim]")
        elif not success:
            logger.warning("Tool call failed: %s(%s) — %s (%.1fs)", tool_name, args_str, error_message, duration)

    def format_tool_counts(self) -> str:
        """Format per-tool-name breakdown for display.

        Returns a string like ``kubectl_get ×4 | get_events ×2 (1 cached)``.
        """
        if not self.tool_name_counts:
            return ""
        parts = [f"{name} ×{count}" for name, count in self.tool_name_counts.most_common()]
        breakdown = " | ".join(parts)
        if self.cache_hits:
            breakdown += f" ({self.cache_hits} cached)"
        return breakdown

    def reset(self) -> None:
        """Reset per-agent counters while keeping pipeline-level totals."""
        self.pipeline_tool_name_counts.update(self.tool_name_counts)
        self.tool_name_counts.clear()
        self.cache_hits = 0

    def print_summary(self) -> None:
        """Print the final pipeline summary line."""
        total_wall = time.perf_counter() - self._pipeline_start
        status = (
            "[green]Pipeline complete[/green]" if self.errors == 0 else "[yellow]Pipeline complete with errors[/yellow]"
        )

        # Build failure detail with grouped reasons
        fail_detail = ""
        if self.errors:
            reason_counts = Counter(self._error_reasons)
            if reason_counts:
                grouped = ", ".join(f"{count}\u00d7 {reason}" for reason, count in reason_counts.most_common())
                fail_detail = f", {self.errors} failed ({grouped})"
            else:
                fail_detail = f", {self.errors} failed"

        # Build tool-name breakdown
        tool_breakdown = self.format_tool_counts()
        breakdown_detail = f"\n  [dim]Tools: {tool_breakdown}[/dim]" if tool_breakdown else ""

        console.print(
            f"\n{status} "
            f"[dim]({total_wall:.1f}s total, "
            f"{self.tool_count} tool{'s' if self.tool_count != 1 else ''} executed"
            f"{fail_detail})[/dim]{breakdown_detail}"
        )


class AgentProgressDisplay:
    """Live progress indicator for multi-agent pipeline execution.

    Shows a Rich :class:`~rich.status.Status` spinner with the current
    agent name, step number, and running tool count::

        [1/4] health_gatherer — running... (12 tools called)

    Implements the :class:`~vaig.agents.orchestrator.OnAgentProgress`
    protocol so it can be passed directly to
    ``Orchestrator.execute_with_tools(on_agent_progress=...)``.

    Thread-safe: a :class:`threading.Lock` serialises all ``start``/``end``
    mutations so parallel gatherers cannot race on ``self._status`` and
    cause a Rich ``LiveError("Only one live display may be active at once")``.

    Tree-style summary lines are rendered in real-time from ``__call__``
    as each agent completes, using emoji icons per role and completion bars.

    Args:
        tool_logger: The :class:`ToolCallLogger` for the current pipeline.
            Used to read the running ``tool_count`` for display.
    """

    # Emoji lookup for agent roles (first match wins)
    _ROLE_EMOJI: list[tuple[str, str]] = [
        ("gather", "🔍"),
        ("triage", "🔍"),
        ("metric", "📊"),
        ("analys", "📊"),
        ("log", "📝"),
        ("report", "📝"),
        ("synth", "📝"),
        ("rollout", "🔄"),
        ("deploy", "🔄"),
        ("cost", "💰"),
    ]

    @staticmethod
    def _emoji_for_role(role: str) -> str:
        """Return an emoji based on the agent's role keyword."""
        role_lower = role.lower()
        for keyword, emoji in AgentProgressDisplay._ROLE_EMOJI:
            if keyword in role_lower:
                return emoji
        return "🤖"

    def __init__(self, tool_logger: ToolCallLogger) -> None:
        self._tool_logger = tool_logger
        self._status: Status | None = None
        self._current_agent_name: str | None = None
        # Per-agent tool counts at start time — keyed by agent_name.
        # Avoids overwriting a single shared counter when parallel gatherers
        # fire "start" events before any "end" events arrive.
        self._agent_start_counts: dict[str, int] = {}
        # Per-agent error counts at start time — for accurate per-agent
        # error detection (avoids marking all agents ✗ after one failure).
        self._agent_start_errors: dict[str, int] = {}
        # Per-agent tool name counts and cache hits at start time — for
        # accurate per-agent breakdown in parallel execution (avoids showing
        # cumulative counts from all running agents).
        self._agent_start_tool_name_counts: dict[str, Counter[str]] = {}
        self._agent_start_cache_hits: dict[str, int] = {}
        # Per-agent start timestamps for timing.
        self._agent_start_times: dict[str, float] = {}
        # Count of agents that have started but not yet ended.
        # Used to defer tool_logger.reset() until ALL active agents finish,
        # so parallel gatherers don't clear shared counters mid-run.
        self._active_agents: int = 0
        self._lock = threading.Lock()
        # Completed agent records for timing and tree-style summary.
        # Each entry: (name, role_hint, tools_used, elapsed_secs)
        self._completed: list[tuple[str, str, int, float]] = []

    def _stop_current(self) -> None:
        """Stop and discard the current spinner (if any).

        Swallows any exception from ``Status.stop()`` so a stale or
        already-stopped spinner never breaks the pipeline.  Must be called
        with ``self._lock`` held.
        """
        if self._status is not None:
            try:
                self._status.stop()
            except Exception:  # noqa: BLE001
                logger.debug("AgentProgressDisplay: ignored error stopping previous status", exc_info=True)
            finally:
                self._status = None

    def __call__(
        self,
        agent_name: str,
        agent_index: int,
        total_agents: int,
        event: str,
        end_agent_index: int | None = None,
    ) -> None:
        """Handle agent start/end events (thread-safe).

        Args:
            agent_name: Display name of the agent or agent group.
            agent_index: Zero-based start index in the pipeline.
            total_agents: Total agents in the pipeline.
            event: ``"start"`` or ``"end"``.
            end_agent_index: Optional inclusive end index for a range display.
                When set and different from *agent_index*, the step counter
                renders as ``[start+1-end+1/total]`` (e.g. ``[1-4/7]``)
                instead of the default ``[index+1/total]``.
        """

        # Build step label: "[1-4/7]" for a range, "[3/7]" for a single agent.
        def _step_label() -> str:
            start = agent_index + 1
            if end_agent_index is not None and end_agent_index != agent_index:
                end = end_agent_index + 1
                return f"{start}-{end}/{total_agents}"
            return f"{start}/{total_agents}"

        with self._lock:
            if event == "start":
                # Record tool count at this agent's start for accurate per-agent
                # tool tallies.  Using a dict keyed by agent_name instead of a
                # single shared counter prevents parallel gatherers from
                # overwriting each other's baseline before their "end" fires.
                self._agent_start_counts[agent_name] = self._tool_logger.tool_count
                self._agent_start_errors[agent_name] = self._tool_logger.errors
                self._agent_start_tool_name_counts[agent_name] = self._tool_logger.tool_name_counts.copy()
                self._agent_start_cache_hits[agent_name] = self._tool_logger.cache_hits
                self._agent_start_times[agent_name] = time.perf_counter()
                self._current_agent_name = agent_name
                self._active_agents += 1
                label = f"[bold cyan]\\[{_step_label()}][/bold cyan] [green]{agent_name}[/green] — running..."
                # Stop any existing Live display before creating a new one.
                # Rich only allows one live display active at a time; without
                # this the second parallel gatherer raises:
                #   LiveError: Only one live display may be active at once
                self._stop_current()
                self._status = console.status(label, spinner="dots")
                self._status.start()
            elif event == "end":
                # In parallel execution, "end" events fire in submission order
                # but the spinner always belongs to the most-recently-started
                # gatherer (i.e. the one whose "start" ran last).  Only stop
                # the spinner when this "end" matches the agent that owns it;
                # otherwise we would kill the wrong gatherer's spinner.
                if agent_name == self._current_agent_name:
                    self._stop_current()
                    self._current_agent_name = None
                start_count = self._agent_start_counts.pop(agent_name, 0)
                start_errors = self._agent_start_errors.pop(agent_name, 0)
                elapsed = time.perf_counter() - self._agent_start_times.pop(agent_name, time.perf_counter())
                tools = self._tool_logger.tool_count - start_count

                # Track for timing summary (role_hint derived from name).
                self._completed.append((agent_name, agent_name, tools, elapsed))

                # ── Real-time tree line ────────────────────────────
                # Render the tree line immediately as each agent completes.
                is_last = False
                try:
                    effective_idx = end_agent_index if end_agent_index is not None else agent_index
                    is_last = effective_idx == total_agents - 1
                    connector = "└──" if is_last else "├──"
                    emoji = self._emoji_for_role(agent_name)
                    color = _AGENT_COLORS[agent_index % len(_AGENT_COLORS)]

                    # Use per-agent error delta for accurate success marker.
                    # The cumulative self._tool_logger.errors would mark ALL
                    # agents ✗ after a single tool failure in an earlier agent.
                    has_errors = self._tool_logger.errors > start_errors
                    if has_errors:
                        status_icon = "[red]✗[/red]"
                    else:
                        status_icon = f"[{color}]✓[/{color}]"

                    bar = f"[{color}]██████████[/{color}]"
                    tool_str = f"{tools} tool{'s' if tools != 1 else ''}"

                    # Per-tool-name breakdown (e.g. "kubectl_get ×4 | get_events ×2")
                    # Only shown in --detailed mode to keep default output clean.
                    # Uses per-agent delta counts for accuracy in parallel execution.
                    # Always pop start snapshots to avoid memory leaks.
                    start_tool_counts = self._agent_start_tool_name_counts.pop(agent_name, Counter())
                    start_cache = self._agent_start_cache_hits.pop(agent_name, 0)
                    breakdown_detail = ""
                    if self._tool_logger.detailed:
                        agent_tool_counts = self._tool_logger.tool_name_counts - start_tool_counts
                        agent_cache = self._tool_logger.cache_hits - start_cache

                        if agent_tool_counts:
                            parts = [f"{name} ×{count}" for name, count in agent_tool_counts.most_common()]
                            breakdown = " | ".join(parts)
                            if agent_cache > 0:
                                breakdown += f" ({agent_cache} cached)"
                            breakdown_detail = f"\n          [dim]{breakdown}[/dim]"

                    console.print(
                        f"   {connector} {emoji} [bold]{agent_name}[/bold]  "
                        f"{bar}  {tool_str} ({elapsed:.1f}s) {status_icon}{breakdown_detail}"
                    )
                except Exception:  # noqa: BLE001
                    # Graceful degradation: never crash the pipeline from
                    # rendering failures.
                    logger.debug("AgentProgressDisplay: ignored tree-line render error", exc_info=True)

                # Decrement active agent count; reset per-agent counters only
                # when ALL active agents have finished.  Calling reset() on every
                # "end" would clear shared tool_name_counts while other parallel
                # gatherers are still running, corrupting their breakdowns.
                self._active_agents = max(0, self._active_agents - 1)
                if self._active_agents == 0:
                    self._tool_logger.reset()

                # If more agents remain, restart spinner for the next one.
                if not is_last and self._status is None:
                    try:
                        next_label = "[dim]waiting for next agent…[/dim]"
                        self._status = console.status(next_label, spinner="dots")
                        self._status.start()
                    except Exception:  # noqa: BLE001
                        logger.debug("AgentProgressDisplay: ignored spinner restart error", exc_info=True)

    def get_agent_timings(self) -> dict[str, float]:
        """Return a mapping of agent_name → elapsed seconds.

        Built from the ``_completed`` records.  Thread-safe.
        """
        with self._lock:
            return {name: elapsed for name, _role, _tools, elapsed in self._completed}

    def stop(self) -> None:
        """Stop the spinner if it's still running (e.g. on error)."""
        with self._lock:
            self._stop_current()


def _emit_bell(*, no_bell: bool) -> None:
    """Print the terminal bell character unless suppressed by ``--no-bell``."""
    if not no_bell:
        import sys

        sys.stdout.write("\a")
        sys.stdout.flush()


def _print_launch_header(
    skill_display_name: str,
    gke_config: GKEConfig,
    settings: Settings,
    tool_count: int,
    *,
    model_id: str | None = None,
    is_async: bool = False,
) -> None:
    """Print the enhanced launch header panel with two-column layout.

    Left column: Cluster / Namespace / Project.
    Right column: Model / Agents / Tools.
    """
    async_tag = " (async)" if is_async else ""
    model_name = model_id or settings.models.default or "—"

    left = Text()
    left.append("☸  Cluster: ", style="dim")
    left.append(f"{gke_config.cluster_name or '(kubeconfig default)'}\n")
    left.append("📂 Namespace: ", style="dim")
    left.append(f"{gke_config.default_namespace or 'default'}\n")
    left.append("☁  Project: ", style="dim")
    left.append(f"{gke_config.project_id or '(auto-detect)'}")

    right = Text()
    right.append("🧠 Model: ", style="dim")
    right.append(f"{model_name}\n")
    right.append("🔧 Tools: ", style="dim")
    right.append(f"{tool_count} loaded\n")
    right.append("📋 Strategy: ", style="dim")
    right.append("sequential")

    console.print(
        Panel(
            Columns([left, right], padding=(0, 4)),
            title=f"🚀 VAIG Live Mode — {skill_display_name}{async_tag}",
            border_style="bright_cyan",
            padding=(1, 2),
        )
    )


def register(app: typer.Typer) -> None:
    """Register the live command on the given Typer app."""

    @app.command()
    @track_command
    def live(
        question: Annotated[str, typer.Argument(help="Infrastructure question or investigation task")],
        config: Annotated[str | None, typer.Option("--config", "-c", help="Path to config YAML")] = None,
        model: Annotated[str | None, typer.Option("--model", "-m", help="Model to use")] = None,
        output: Annotated[Path | None, typer.Option("--output", "-o", help="Save response to a file")] = None,
        format_: Annotated[str | None, typer.Option("--format", help="Export format: json, md, html")] = None,
        skill: Annotated[str | None, typer.Option("--skill", "-s", help="SRE skill to apply")] = None,
        auto_skill: Annotated[
            bool, typer.Option("--auto-skill", help="Auto-detect the best skill based on query")
        ] = False,
        cluster: Annotated[str | None, typer.Option("--cluster", help="GKE cluster name (overrides config)")] = None,
        namespace: Annotated[
            str | None,
            typer.Option(
                "--namespace", help="Default Kubernetes namespace (overrides config)", autocompletion=complete_namespace
            ),
        ] = None,
        project: Annotated[
            str | None,
            typer.Option("--project", "-p", help="GCP project ID (overrides gcp.project_id only)"),
        ] = None,
        project_id: Annotated[
            str | None, typer.Option("--project-id", help="GCP project ID (overrides config, alias for --project)")
        ] = None,
        location: Annotated[str | None, typer.Option("--location", help="GCP location (overrides config)")] = None,
        gke_project: Annotated[
            str | None,
            typer.Option(
                "--gke-project", help="GKE project ID (overrides gke.project_id; defaults to --project if unset)"
            ),
        ] = None,
        gke_location: Annotated[
            str | None,
            typer.Option("--gke-location", help="GKE cluster location (overrides gke.location)"),
        ] = None,
        watch: Annotated[
            int | None, typer.Option("--watch", "-w", help="Re-execute every N seconds (polling mode, min 10s)")
        ] = None,
        dry_run: Annotated[
            bool,
            typer.Option("--dry-run", "--dry", help="Show execution plan without running"),
        ] = False,
        verbose: Annotated[
            bool,
            typer.Option("--verbose", "-V", help="Enable verbose logging (INFO level)"),
        ] = False,
        debug: Annotated[
            bool,
            typer.Option("--debug", "-d", help="Enable debug logging (DEBUG level, shows paths and full tracebacks)"),
        ] = False,
        summary: Annotated[
            bool,
            typer.Option("--summary", help="Show compact summary instead of full report"),
        ] = False,
        no_bell: Annotated[
            bool,
            typer.Option("--no-bell", help="Suppress terminal bell after pipeline completes"),
        ] = False,
        open_browser: Annotated[
            bool,
            typer.Option("--open", "-O", help="Open HTML report in default browser (requires --format html)"),
        ] = False,
        all_namespaces: Annotated[
            bool,
            typer.Option(
                "--all-namespaces",
                help=(
                    "Analyse all non-system namespaces for cost estimation, "
                    "ignoring the --namespace / config default_namespace filter."
                ),
            ),
        ] = False,
        detailed: Annotated[
            bool,
            typer.Option(
                "--detailed",
                help="Show every tool call as it happens (verbose execution output)",
            ),
        ] = False,
        repo: Annotated[
            str | None,
            typer.Option(
                "--repo",
                help="GitHub repo in owner/repo format for commit correlation (e.g. myorg/myservice)",
            ),
        ] = None,
        repo_ref: Annotated[
            str,
            typer.Option(
                "--repo-ref",
                help="Git ref to use for repo correlation (default: HEAD)",
            ),
        ] = "HEAD",
        repo_path: Annotated[
            list[str] | None,
            typer.Option(
                "--repo-path",
                help="Subdirectory path within the repo to investigate (multi-valued).",
            ),
        ] = None,
        repo_include_glob: Annotated[
            list[str] | None,
            typer.Option(
                "--repo-include-glob",
                help="Glob pattern to include from the repo (multi-valued).",
            ),
        ] = None,
        repo_exclude_glob: Annotated[
            list[str] | None,
            typer.Option(
                "--repo-exclude-glob",
                help="Glob pattern to exclude from the repo (multi-valued).",
            ),
        ] = None,
        repo_max_files: Annotated[
            int | None,
            typer.Option(
                "--repo-max-files",
                help="Maximum number of files to process per repo path.",
            ),
        ] = None,
        repo_max_bytes_per_file: Annotated[
            int | None,
            typer.Option(
                "--repo-max-bytes-per-file",
                help="Files larger than this (bytes) are processed via streaming chunker.",
            ),
        ] = None,
        # ── Attachment flags (SPEC-ATT-01..04) ────────────────────────────────
        attach: Annotated[
            list[str] | None,
            typer.Option(
                "--attach",
                help=(
                    "Local path or URL to attach as context (multi-valued). "
                    "Directories (non-git), individual files, and — in future "
                    "sprints — archives and URLs are supported."
                ),
            ),
        ] = None,
        attach_name: Annotated[
            list[str] | None,
            typer.Option(
                "--attach-name",
                help=(
                    "Optional label for each --attach source, positional-matched. "
                    "If count mismatches, remaining sources get no label (None)."
                ),
            ),
        ] = None,
        attach_max_files: Annotated[
            int,
            typer.Option(
                "--attach-max-files",
                help="Maximum number of files to ingest per attachment source (default 10 000).",
            ),
        ] = 10_000,
        attach_unlimited: Annotated[
            bool,
            typer.Option(
                "--attach-unlimited/--no-attach-unlimited",
                help="When set, ignore --attach-max-files limit.",
            ),
        ] = False,
        attach_max_depth: Annotated[
            int,
            typer.Option(
                "--attach-max-depth",
                help="Max directory recursion depth for attachments (-1 = unlimited, default).",
            ),
        ] = -1,
        attach_follow_symlinks: Annotated[
            bool,
            typer.Option(
                "--attach-follow-symlinks/--no-attach-follow-symlinks",
                help="Follow symlinks when walking attachment directories (default: off).",
            ),
        ] = False,
        attach_no_default_excludes: Annotated[
            bool,
            typer.Option(
                "--attach-no-default-excludes",
                help="Disable the built-in exclude globs (node_modules, .git, etc.) for attachments.",
            ),
        ] = False,
        attach_include_everything: Annotated[
            bool,
            typer.Option(
                "--attach-include-everything",
                help=(
                    "Include all files: disables exclude globs, unlimited depth/files, "
                    "and binary-skip.  Does NOT bypass max_bytes_absolute."
                ),
            ),
        ] = False,
        attach_max_bytes_absolute: Annotated[
            int,
            typer.Option(
                "--attach-max-bytes-absolute",
                help="Reject attachment files larger than this many bytes (default 500 MB).",
            ),
        ] = 500_000_000,
        attach_allow_http: Annotated[
            bool,
            typer.Option(
                "--attach-allow-http/--no-attach-allow-http",
                help="Allow plain HTTP URLs for --attach (default: HTTPS-only).",
            ),
        ] = False,
        attach_allow_domain: Annotated[
            list[str] | None,
            typer.Option(
                "--attach-allow-domain",
                help=(
                    "Hostname or domain suffix allowed for URL attachments. "
                    "May be repeated. Empty = allow all (HTTPS only)."
                ),
            ),
        ] = None,
        attach_session: Annotated[
            str | None,
            typer.Option(
                "--attach-session",
                help=(
                    "Session ID for persistent attachment list. Re-running with the same ID merges prior attachments."
                ),
            ),
        ] = None,
        attach_cache: Annotated[
            bool,
            typer.Option(
                "--attach-cache/--no-attach-cache",
                help="Enable/disable attachment processing cache (default: enabled).",
            ),
        ] = True,
        show_attachments: Annotated[
            bool,
            typer.Option(
                "--show-attachments",
                help="Print a table of resolved attachments with cache status and fingerprints, then continue.",
            ),
        ] = False,
        interactive: Annotated[
            bool,
            typer.Option(
                "--interactive",
                "-i",
                help="After the report, open an interactive drill-in REPL for follow-up questions",
            ),
        ] = False,
        show_pipeline: Annotated[
            bool,
            typer.Option(
                "--show-pipeline",
                help=("Print the pipeline stages and exit without making any LLM calls."),
            ),
        ] = False,
        show_pipeline_format: Annotated[
            str,
            typer.Option(
                "--show-pipeline-format",
                help="Output format for --show-pipeline: 'text' (default) or 'json'.",
            ),
        ] = "text",
    ) -> None:
        """Investigate live GKE/GCP infrastructure using AI with infrastructure tools.

        Launches an autonomous SRE agent that can inspect pods, logs, metrics,
        and Cloud Logging/Monitoring to answer infrastructure questions.

        Most tools are read-only. Write tools (scale, restart, label, annotate)
        are available but require explicit invocation by the agent.

        Examples:
            vaig live "What pods are crashing in the production namespace?"
            vaig live "Check CPU and memory pressure on nodes"
            vaig live "Investigate OOM kills in the last hour" --namespace=production
            vaig live "Show HPA status for frontend deployment" --cluster=prod-gke
            vaig live "Analyze error rate from Cloud Logging" --project-id=my-project
            vaig live "Why is the payment service returning 503s?" -o report.md
            vaig live "Investigate pod crashes" --format json -o report.json
            vaig live "Check pod health in production" --watch 60
            vaig live "Check service health" --dry-run
        """
        _apply_subcommand_log_flags(verbose=verbose, debug=debug)

        # Validate --watch interval
        if watch is not None and watch < MINIMUM_WATCH_INTERVAL:
            err_console.print(f"[red]--watch interval must be >= {MINIMUM_WATCH_INTERVAL}s (got {watch}s)[/red]")
            raise typer.Exit(1)

        # Warn if --open is used without --format html
        normalised_format_flag = format_.strip().lower() if format_ else None
        if open_browser and normalised_format_flag != "html":
            err_console.print("[yellow]⚠ --open requires --format html — ignoring --open flag.[/yellow]")
            open_browser = False

        try:  # ── CLI error boundary ──
            settings = _helpers._get_settings(config)

            # Eagerly initialize the telemetry collector and wire the
            # TelemetrySubscriber so events are forwarded to the SQLite store.
            _helpers._init_telemetry(settings)
            _helpers._init_audit(settings)
            _helpers._init_memory(settings)
            _helpers._init_fix_outcome(settings)
            _helpers._check_platform_auth(settings)

            # Apply --project / --project-id: mutate ONLY gcp.project_id
            # The GKE fallback chain (gke.project_id or gcp.project_id) handles single-project setups.
            effective_project = project or project_id
            if effective_project:
                settings.gcp.project_id = effective_project

            # Apply --gke-project: mutate ONLY gke.project_id when explicitly provided
            if gke_project:
                settings.gke.project_id = gke_project

            # Apply --gke-location: mutate ONLY gke.location when explicitly provided
            if gke_location:
                settings.gke.location = gke_location

            # Apply --location: mutate gcp.location before component creation
            if location:
                settings.gcp.location = location

            if model:
                settings.models.default = model

            # Do NOT pass project_id/location here — gke_project/gke_location are
            # already written to settings.gke.* above, and _build_gke_config reads
            # them via its fallback chain (gke.project_id or gcp.project_id).
            # Passing effective_project/location would override gke-specific flags.
            gke_config = _build_gke_config(
                settings,
                cluster=cluster,
                namespace=namespace,
            )

            # ── Repo investigation config (SPEC-V2-REPO-01) ───────
            # Validate repo config early so malformed globs are caught before
            # any LLM call.  Wired into tools in Sprint 3.
            _repo_cfg = _build_repo_investigation_config(
                repo=repo,
                repo_ref=repo_ref,
                repo_paths=repo_path or [],
                include_globs=repo_include_glob or [],
                exclude_globs=repo_exclude_glob,
                max_files=repo_max_files,
                streaming_threshold_bytes=repo_max_bytes_per_file,
            )

            # ── Attachment config + early resolution (SPEC-ATT-01..06) ───────
            _attachment_adapters = _build_and_resolve_attachments(
                attach_sources=attach or [],
                attach_names=attach_name or [],
                max_files=attach_max_files,
                unlimited_files=attach_unlimited,
                max_depth=attach_max_depth,
                follow_symlinks=attach_follow_symlinks,
                use_default_excludes=not attach_no_default_excludes,
                include_everything=attach_include_everything,
                max_bytes_absolute=attach_max_bytes_absolute,
                allow_http=attach_allow_http,
                url_allowlist=attach_allow_domain or [],
                session_id=attach_session,
                cache_enabled=attach_cache,
            )

            # ── Session persistence (SPEC-ATT-08) ─────────────────────────────
            if attach_session and _attachment_adapters:
                _persist_session(
                    session_id=attach_session,
                    adapters=_attachment_adapters,
                    session_dir=None,  # uses AttachmentsConfig.session_dir default (.vaig/sessions)
                )

            # ── Show-attachments table ─────────────────────────────────────────
            if show_attachments and _attachment_adapters:
                _display_attachments_table(_attachment_adapters)

            # Auto-detect skill if requested (or enabled in config) and no explicit skill specified
            effective_auto_skill = auto_skill or settings.skills.auto_routing
            if effective_auto_skill and not skill:
                from vaig.skills.registry import SkillRegistry

                registry = SkillRegistry(settings)
                suggestions = registry.suggest_skill(question)
                if suggestions:
                    best_name, best_score = suggestions[0]
                    threshold = settings.skills.auto_routing_threshold
                    if best_score >= threshold:
                        skill = best_name
                        console.print(
                            f"[dim]🎯 Auto-routing to skill: [cyan]{skill}[/cyan] (score: {best_score:.1f})[/dim]"
                        )
                    else:
                        console.print(
                            f"[dim]Suggested skills: {', '.join(f'{n} ({s:.1f})' for n, s in suggestions)}[/dim]"
                        )

            # If a skill is specified, check whether it needs the full orchestrated
            # pipeline (requires_live_tools=True) or the simple context-prepend approach.
            context_str = ""
            active_skill = None
            if skill:
                from vaig.skills.registry import SkillRegistry

                registry = SkillRegistry(settings)
                active_skill = registry.get(skill)
                if not active_skill:
                    err_console.print(f"[red]Skill not found: {skill}[/red]")
                    err_console.print(f"[dim]Available: {', '.join(registry.list_names())}[/dim]")
                    raise typer.Exit(1)

                skill_meta = active_skill.get_metadata()

                if not skill_meta.requires_live_tools:
                    # ── Legacy context-prepend path (requires_live_tools=False) ──
                    context_str = (
                        f"## Active Skill: {skill_meta.display_name}\n\n"
                        f"{skill_meta.description}\n\n"
                        f"Apply the {skill_meta.name} analysis methodology to the investigation below."
                    )
                    active_skill = None  # Not orchestrated

            # ── Show-pipeline: print pipeline stages and exit (no LLM/API calls) ──
            if show_pipeline:
                _display_show_pipeline(
                    gke_config=gke_config,
                    question=question,
                    settings=settings,
                    skill=active_skill,
                    skill_name=skill,
                    model_id=model or settings.models.default,
                    output_format=show_pipeline_format,
                )
                return

            from vaig.core.container import build_container

            container = build_container(settings)
            client = container.gemini_client

            # ── Dry-run: show execution plan without running ──────
            if dry_run:
                _display_dry_run_plan(
                    gke_config=gke_config,
                    question=question,
                    settings=settings,
                    skill=active_skill,
                    skill_name=skill,
                    model_id=model or settings.models.default,
                    repo=repo,
                    repo_ref=repo_ref,
                )
                return

            # ── Tool result persistence ────────────────────────
            tool_call_store = _create_tool_call_store(settings)
            # Session-scoped cache: created once, reused across watch
            # iterations and single-shot calls.  TTL=0 (no expiration)
            # is safe because the cache is discarded when the CLI exits,
            # and watch iterations overwrite entries via put() on each run.
            tool_result_cache = ToolResultCache()

            def _run_once() -> HealthReport | None:
                """Execute a single iteration of the live command.

                Returns the structured ``HealthReport`` when the
                orchestrated-skill path produces one, otherwise ``None``
                (e.g. the InfraAgent path).
                """
                if active_skill is not None:
                    return _execute_orchestrated_skill(
                        client,
                        settings,
                        gke_config,
                        active_skill,
                        question,
                        output=output if not watch else None,
                        format_=format_ if not watch else None,
                        model_id=model,
                        tool_call_store=tool_call_store,
                        summary=summary,
                        no_bell=no_bell,
                        open_browser=open_browser if not watch else False,
                        all_namespaces=all_namespaces,
                        detailed=detailed,
                        repo=repo,
                        repo_ref=repo_ref,
                        interactive=interactive,
                    )
                _execute_live_mode(
                    client,
                    gke_config,
                    question,
                    context_str,
                    settings=settings,
                    output=output if not watch else None,
                    format_=format_ if not watch else None,
                    skill_name=skill,
                    model_id=model,
                    tool_call_store=tool_call_store,
                    tool_result_cache=tool_result_cache,
                    open_browser=open_browser if not watch else False,
                    detailed=detailed,
                    repo=repo,
                    repo_ref=repo_ref,
                )
                return None

            if not watch:
                # ── Single execution (default) ────────────────────
                _run_once()
                return

            # ── Watch mode: re-execute every N seconds ────────────
            _run_watch_loop(
                run_fn=_run_once,
                interval=watch,
                question=question,
                output=output,
                format_=format_,
                html_output_path=str(output) if output else None,
            )
        except typer.Exit:
            raise  # Let typer exits pass through
        except Exception as exc:  # noqa: BLE001
            handle_cli_error(exc, debug=debug)


# ── Watch mode loop ──────────────────────────────────────────


def _run_watch_loop(
    *,
    run_fn: Callable[[], HealthReport | None],
    interval: int,
    question: str,
    output: Path | None = None,
    format_: str | None = None,
    html_output_path: str | None = None,
) -> None:
    """Execute *run_fn* repeatedly every *interval* seconds.

    Handles ``KeyboardInterrupt`` gracefully: stops the loop, prints a
    summary of iterations run and total elapsed time, and optionally
    exports a watch-session HTML report containing the last
    ``HealthReport`` plus the accumulated diff timeline.

    From iteration 2 onwards, if the run function returns a
    :class:`~vaig.skills.service_health.schema.HealthReport`, a diff
    summary panel is printed showing new, resolved, unchanged, and
    severity-changed findings compared to the previous iteration.

    The *output* / *format_* args are only used on the **last** iteration
    (i.e. never — the caller passes ``None`` to *run_fn* for those during
    watch mode, and can export manually after the loop).

    Args:
        run_fn: Zero-argument callable that executes one iteration and
            returns an optional ``HealthReport``.
        interval: Seconds between iterations (already validated >= MINIMUM_WATCH_INTERVAL).
        question: Original query (shown in the header).
        output: Export path (informational — not used inside the loop).
        format_: Export format (informational — not used inside the loop).
        html_output_path: Optional destination for a watch-session HTML
            report when the loop is interrupted. If ``None``, an
            auto-generated temporary path is used; any other non-empty
            string or ``Path`` is treated as the literal output path.
    """
    from vaig.skills.service_health.diff import DiffTimelineEntry, WatchSessionData

    iteration = 0
    start_time = time.monotonic()
    session_start = datetime.now(tz=UTC)
    previous_report: HealthReport | None = None
    last_report: HealthReport | None = None
    diff_history: list[DiffTimelineEntry] = []

    console.print(
        Panel.fit(
            f"[bold cyan]Watch mode enabled — refreshing every {interval}s[/bold cyan]\n"
            f"[dim]Query: {question[:80]}{'...' if len(question) > 80 else ''}[/dim]\n"
            "[dim]Press Ctrl+C to stop[/dim]",
            border_style="cyan",
        )
    )

    try:
        while True:
            iteration += 1
            iter_timestamp = datetime.now(tz=UTC).isoformat()
            console.print(f"\n{'=' * 60}")
            console.print(
                f"[bold cyan]Watch iteration #{iteration} — {datetime.now().strftime('%H:%M:%S')}[/bold cyan]"
            )
            console.print(f"{'=' * 60}\n")

            current_report: HealthReport | None = None
            try:
                current_report = run_fn()
            except (SystemExit, typer.Exit):
                # Agent errors (MaxIterationsError etc.) raise typer.Exit —
                # in watch mode we log and continue to the next iteration.
                console.print(f"[yellow]Iteration #{iteration} exited with error — continuing watch...[/yellow]")

            # ── Diff summary (iteration 2+) ───────────────────
            diff: ReportDiff | None = None
            if iteration >= 2 and current_report is not None and previous_report is not None:
                diff = compute_report_diff(current_report, previous_report)
                print_watch_diff_summary(diff, iteration, console=console)

            # ── Accumulate diff timeline entry ────────────────
            diff_history.append(
                DiffTimelineEntry(
                    iteration=iteration,
                    timestamp=iter_timestamp,
                    is_baseline=(iteration == 1),
                    diff=None if iteration == 1 else diff,
                )
            )

            if current_report is not None:
                previous_report = current_report
                last_report = current_report

            console.print(f"\n[dim]Next refresh in {interval}s (Ctrl+C to stop)...[/dim]")
            time.sleep(interval)
    except KeyboardInterrupt:
        elapsed = time.monotonic() - start_time
        console.print(
            f"\n[bold yellow]Watch stopped after {iteration} "
            f"iteration{'s' if iteration != 1 else ''} "
            f"({elapsed:.0f}s elapsed)[/bold yellow]"
        )

        # ── Export watch-session HTML ─────────────────────────
        if last_report is not None and iteration >= 1:
            session_end = datetime.now(tz=UTC)
            session_data = WatchSessionData(
                start_time=session_start.isoformat(),
                end_time=session_end.isoformat(),
                total_iterations=iteration,
                interval_seconds=interval,
                diff_timeline=diff_history,
            )
            try:
                from vaig.ui.html_report import render_watch_session_html

                html_content = render_watch_session_html(last_report, session_data)
                if html_output_path:
                    out_path = Path(html_output_path)
                else:
                    ts = session_end.strftime("%Y%m%d_%H%M%S")
                    out_path = Path(f"vaig-watch-session-{ts}.html")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(html_content, encoding="utf-8")
                console.print(
                    f"[bold green]✓ Watch session HTML report written:[/bold green] [cyan]{out_path.resolve()}[/cyan]"
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"[bold red]⚠ Failed to write watch session HTML report:[/bold red] {exc}")


# ── Live mode helpers ─────────────────────────────────────────


# _build_gke_config — delegated to vaig.core.gke.build_gke_config
from vaig.core.gke import build_gke_config as _build_gke_config


def _build_and_resolve_attachments(
    *,
    attach_sources: list[str],
    attach_names: list[str],
    max_files: int,
    unlimited_files: bool,
    max_depth: int,
    follow_symlinks: bool,
    use_default_excludes: bool,
    include_everything: bool,
    max_bytes_absolute: int,
    allow_http: bool = False,
    url_allowlist: list[str] | None = None,
    session_id: str | None = None,
    cache_enabled: bool = True,
) -> list[AttachmentAdapter]:
    """Build :class:`~vaig.core.config.AttachmentsConfig`, resolve adapters, and
    eagerly call ``list_files()`` to surface errors before any LLM call.

    Returns the list of resolved adapters (may be empty).
    """
    import sys

    from vaig.core.attachment_adapter import (
        ArchiveAttachmentAdapter,
        GitCloneAttachmentAdapter,
        LocalPathAdapter,
        SingleFileAdapter,
        URLAdapter,
        resolve_attachment,
    )
    from vaig.core.config import AttachmentsConfig

    # --attach-include-everything cascades
    if include_everything:
        unlimited_files = True
        max_depth = -1
        use_default_excludes = False
        binary_skip = False
    else:
        binary_skip = True

    cfg = AttachmentsConfig(
        max_files_per_attachment=max_files,
        unlimited_files=unlimited_files,
        max_depth=max_depth,
        follow_symlinks=follow_symlinks,
        use_default_excludes=use_default_excludes,
        include_everything=include_everything,
        max_bytes_absolute=max_bytes_absolute,
        binary_skip=binary_skip,
        allow_http=allow_http,
        url_allowlist=url_allowlist or [],
        session_id=session_id,
        cache_enabled=cache_enabled,
    )

    if not attach_sources:
        return []

    # Pad names list to match sources length
    names: list[str | None] = list(attach_names)
    while len(names) < len(attach_sources):
        names.append(None)

    adapters: list[
        LocalPathAdapter | SingleFileAdapter | ArchiveAttachmentAdapter | GitCloneAttachmentAdapter | URLAdapter
    ] = []
    for raw, name in zip(attach_sources, names, strict=False):
        try:
            adapter = resolve_attachment(raw, name=name, cfg=cfg)
        except ValueError as exc:
            print(f"[attachments] {exc}", file=sys.stderr)
            raise typer.Exit(1) from exc

        # Eagerly list files to surface errors (e.g. permission denied, path escapes)
        try:
            list(adapter.list_files(cfg))
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[attachments] failed to list files for {raw!r}: {exc}", file=sys.stderr)
            raise typer.Exit(1) from exc

        adapters.append(adapter)

    # ── Cache wire (SPEC-ATT-08): put after resolution if cache_enabled ──────
    if cache_enabled:
        try:
            import hashlib

            from vaig.core.attachment_cache import AttachmentCache

            cache_dir = Path(".vaig/attachments-cache")
            config_hash = hashlib.sha256(cfg.model_dump_json().encode()).hexdigest()[:16]
            cache = AttachmentCache(cache_dir, config_hash=config_hash)
            for adapter in adapters:
                try:
                    fp = adapter.fingerprint()
                    manifest = [
                        e.__dict__ if hasattr(e, "__dict__") else {"path": str(e)} for e in adapter.list_files(cfg)
                    ]
                    cache.put(
                        fp, manifest, [], adapter_spec={"source": adapter.spec.source, "kind": str(adapter.spec.kind)}
                    )
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.debug("attachment_cache: put failed for %s: %s", adapter.spec.source, exc)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001
            logger.debug("attachment_cache: cache wire failed: %s", exc)

    label_parts = []
    for adapter in adapters:
        label = adapter.spec.name or adapter.spec.source
        label_parts.append(label)
    print(
        f"[attachments] resolved {len(adapters)} attachment(s): {', '.join(label_parts)}",
        file=sys.stderr,
    )

    return cast("list[AttachmentAdapter]", adapters)


def _persist_session(
    *,
    session_id: str,
    adapters: list[Any],
    session_dir: str | Path | None = None,
) -> None:
    """Save resolved adapters to the session file (SPEC-ATT-08)."""
    try:
        from vaig.core.attachment_cache import AttachmentSession

        resolved_dir = Path(session_dir) if session_dir is not None else Path(".vaig/sessions")
        session = AttachmentSession(resolved_dir, session_id)
        session.load()
        for adapter in adapters:
            spec = adapter.spec
            try:
                fp = adapter.fingerprint()
            except Exception:  # noqa: BLE001
                fp = ""
            session.add(
                source=spec.source,
                name=spec.name,
                fingerprint=fp,
                kind=str(spec.kind),
            )
        session.save()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001
        logger.debug("attachment_session: failed to persist session %s: %s", session_id, exc)


def _display_attachments_table(adapters: list[Any], *, cache_hits: dict[str, bool] | None = None) -> None:
    """Print a Rich table of resolved attachments with fingerprints and cache status (SPEC-ATT-08)."""
    table = Table(title="Resolved Attachments", show_lines=True)
    table.add_column("Source", style="cyan")
    table.add_column("Name")
    table.add_column("Kind", style="green")
    table.add_column("Cache", style="yellow")
    table.add_column("Fingerprint", style="dim")

    for adapter in adapters:
        spec = adapter.spec
        try:
            fp = adapter.fingerprint()
            fp_short = fp[:16] + "…" if len(fp) > 16 else fp
        except Exception:  # noqa: BLE001
            fp = ""
            fp_short = "(unavailable)"

        if cache_hits is None:
            cache_status = "disabled"
        elif fp and fp in cache_hits:
            cache_status = "hit" if cache_hits[fp] else "miss"
        else:
            cache_status = "miss"

        table.add_row(
            spec.source,
            spec.name or "",
            str(spec.kind),
            cache_status,
            fp_short,
        )

    console.print(table)


def _build_repo_investigation_config(
    *,
    repo: str | None,
    repo_ref: str,
    repo_paths: list[str],
    include_globs: list[str],
    exclude_globs: list[str] | None,
    max_files: int | None,
    streaming_threshold_bytes: int | None,
) -> Any:  # RepoInvestigationConfig — imported inside function body
    """Build a :class:`~vaig.core.config.RepoInvestigationConfig` from CLI flags.

    Validates glob patterns early (before any LLM call) and raises
    :exc:`typer.BadParameter` with a clear message on malformed patterns.

    Always returns a ``RepoInvestigationConfig`` (even when ``repo=None``,
    for future wiring).
    """
    import fnmatch
    import re

    from vaig.core.config import RepoInvestigationConfig

    # Validate globs early — give the user a clear error before any network call.
    # Python's fnmatch.translate() and glob.glob() silently accept malformed
    # patterns, so we explicitly check for unclosed bracket expressions which
    # are the most common user mistake (e.g. "[invalid").
    def _validate_glob(pattern: str) -> None:
        try:
            # Attempt to compile the translated regex — catches some bad patterns.
            re.compile(fnmatch.translate(pattern))
        except re.error as exc:
            raise typer.BadParameter(
                f"glob pattern {pattern!r} is malformed: {exc}",
                param_hint="--repo-include-glob/--repo-exclude-glob",
            ) from exc
        # Check for unclosed bracket expressions which fnmatch accepts silently.
        depth = 0
        for ch in pattern:
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth = max(0, depth - 1)
        if depth > 0:
            raise typer.BadParameter(
                f"glob pattern {pattern!r} is malformed: unclosed '[' bracket",
                param_hint="--repo-include-glob/--repo-exclude-glob",
            )

    all_globs = include_globs + (exclude_globs or [])
    for pattern in all_globs:
        _validate_glob(pattern)

    kwargs: dict[str, Any] = {
        "repo": repo,
        "ref": repo_ref,
        "paths": repo_paths,
        "include_globs": include_globs,
    }
    if exclude_globs is not None:
        kwargs["exclude_globs"] = exclude_globs
    if max_files is not None:
        kwargs["max_files"] = max_files
    if streaming_threshold_bytes is not None:
        kwargs["streaming_threshold_bytes"] = streaming_threshold_bytes

    return RepoInvestigationConfig(**kwargs)


def _display_dry_run_plan(
    *,
    gke_config: GKEConfig,
    question: str,
    settings: Settings,
    skill: BaseSkill | None = None,
    skill_name: str | None = None,
    model_id: str = "",
    repo: str | None = None,
    repo_ref: str = "HEAD",
) -> None:
    """Render the dry-run execution plan without running anything.

    Shows what *would* happen: resolved configuration, agents that
    would be created, available tools, and estimated cost — then exits.

    Args:
        gke_config: Resolved GKE configuration.
        question: The user's infrastructure question.
        settings: Application settings.
        skill: Resolved orchestrated skill instance (``None`` for InfraAgent path).
        skill_name: Skill name string (may be set even when *skill* is ``None``
            for context-prepend skills).
        model_id: Model identifier to display.
    """
    # ── Header ────────────────────────────────────────────────
    console.print(
        Panel.fit(
            f'[bold yellow]Dry Run — vaig live[/bold yellow] [dim]"{question}"[/dim]',
            border_style="yellow",
        )
    )

    # ── Configuration ─────────────────────────────────────────
    config_table = Table(title="Configuration", show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Key", style="bold")
    config_table.add_column("Value")
    config_table.add_row("Cluster", gke_config.cluster_name or "(kubeconfig default)")
    config_table.add_row("Namespace", gke_config.default_namespace or "default")
    config_table.add_row("Project", gke_config.project_id or "(auto-detect)")
    config_table.add_row("Location", gke_config.location or "(auto-detect)")
    config_table.add_row("Model", model_id or settings.models.default)

    if skill_name:
        routing = "explicit (--skill)" if not (settings.skills.auto_routing) else "auto-routed"
        config_table.add_row("Skill", f"{skill_name} ({routing})")

    console.print(config_table)
    console.print()

    # ── Agents ────────────────────────────────────────────────
    if skill is not None:
        # Orchestrated skill — show agent pipeline
        agents_config = skill.get_agents_config()
        agent_table = Table(title="Agents that would be created", show_lines=True)
        agent_table.add_column("#", style="dim", width=3)
        agent_table.add_column("Agent", style="cyan")
        agent_table.add_column("Role", style="dim")
        agent_table.add_column("Tools", style="green")
        agent_table.add_column("Model", style="dim")

        for idx, agent_cfg in enumerate(agents_config, 1):
            has_tools = agent_cfg.get("requires_tools", False)
            tools_label = "[green]yes[/green]" if has_tools else "[dim]no (synthesis)[/dim]"
            agent_table.add_row(
                str(idx),
                agent_cfg.get("name", "unknown"),
                agent_cfg.get("role", ""),
                tools_label,
                agent_cfg.get("model", model_id),
            )
        console.print(agent_table)
    else:
        # InfraAgent path — single agent
        console.print("[bold]Agent:[/bold] InfraAgent (single autonomous agent with tools)")

    console.print()

    # ── Tools ─────────────────────────────────────────────────
    tool_registry = _register_live_tools(gke_config, settings=settings, repo=repo, repo_ref=repo_ref)
    tools = tool_registry.list_tools()
    tool_count = len(tools)

    if tool_count == 0:
        console.print("[bold red]No infrastructure tools available![/bold red]")
        console.print("[yellow]Install: pip install vertex-ai-toolkit[live][/yellow]")
    else:
        tool_names = [t.name for t in tools]
        # Show first 10 tools inline, then "and N more"
        if len(tool_names) <= 10:
            tools_str = ", ".join(tool_names)
        else:
            tools_str = ", ".join(tool_names[:10]) + f", ... and {len(tool_names) - 10} more"
        console.print(f"[bold]Available tools ({tool_count}):[/bold] {tools_str}")

    console.print()

    # ── Cost estimate ─────────────────────────────────────────
    console.print("[bold]Estimated cost:[/bold] [dim]depends on tool usage (typically $0.02-0.10 per run)[/dim]")
    console.print()
    console.print("[dim]Run without --dry-run to execute.[/dim]")


# _register_live_tools — delegated to vaig.core.gke.register_live_tools
from vaig.core.gke import register_live_tools as _register_live_tools


def _build_pipeline_phases(
    agents_config: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group an agents config list into pipeline phases for display.

    Groups agents by ``parallel_group`` key:
    - Agents sharing the same ``parallel_group`` value form one phase (parallel).
    - Agents without ``parallel_group`` each become their own sequential phase.

    Returns a list of phase dicts::

        [
            {"name": "Gather", "parallel": True, "agents": [...]},
            {"name": "Analyze", "parallel": False, "agents": [...]},
        ]
    """
    phases: list[dict[str, Any]] = []
    parallel_phases: dict[str, dict[str, Any]] = {}

    for agent_cfg in agents_config:
        group = agent_cfg.get("parallel_group")
        if group:
            if group not in parallel_phases:
                phase = {
                    "name": group.capitalize(),
                    "parallel": True,
                    "agents": [agent_cfg],
                }
                parallel_phases[group] = phase
                phases.append(phase)
            else:
                parallel_phases[group]["agents"].append(agent_cfg)
        else:
            phases.append(
                {
                    "name": (agent_cfg.get("name") or "unknown").replace("_", " ").capitalize(),
                    "parallel": False,
                    "agents": [agent_cfg],
                }
            )

    return phases


def _display_show_pipeline(
    *,
    gke_config: GKEConfig,
    question: str,
    settings: Settings,
    skill: BaseSkill | None = None,
    skill_name: str | None = None,
    model_id: str = "",
    output_format: str = "text",
) -> None:
    """Render the pipeline stage summary and exit without making any LLM calls.

    When *output_format* is ``"json"``, emits a machine-readable JSON object.
    Otherwise renders a Rich tree-style summary to the console.

    Args:
        gke_config: Resolved GKE configuration.
        question: The user's infrastructure question.
        settings: Application settings.
        skill: Resolved orchestrated skill instance (``None`` for InfraAgent path).
        skill_name: Skill name string (for display and metadata).
        model_id: Model identifier to display.
        output_format: ``"text"`` (default) or ``"json"``.
    """
    import json as _json

    if output_format == "json":
        # ── Machine-readable JSON output ───────────────────────
        payload: dict[str, Any] = {
            "skill": None,
            "pipeline_mode": "INFRA_AGENT",
            "phases": [],
            "config": {
                "cluster": gke_config.cluster_name or "(kubeconfig default)",
                "namespace": gke_config.default_namespace or "default",
                "project": gke_config.project_id or "(auto-detect)",
                "model": model_id or settings.models.default,
            },
        }
        if skill is not None:
            meta = skill.get_metadata()
            agents_config = skill.get_agents_config()
            phases = _build_pipeline_phases(agents_config)
            payload["skill"] = meta.display_name
            payload["pipeline_mode"] = "ORCHESTRATED"
            for idx, phase in enumerate(phases, 1):
                payload["phases"].append(
                    {
                        "phase": idx,
                        "name": phase["name"],
                        "parallel": phase["parallel"],
                        "agents": [
                            {
                                "name": a.get("name"),
                                "model": a.get("model", model_id or settings.models.default),
                                "max_iterations": a.get("max_iterations"),
                                "requires_tools": a.get("requires_tools", False),
                                "tool_categories": a.get("tool_categories", []),
                            }
                            for a in phase["agents"]
                        ],
                    }
                )
        else:
            payload["phases"].append(
                {
                    "phase": 1,
                    "name": "InfraAgent",
                    "parallel": False,
                    "agents": [{"name": "infra_agent", "model": model_id or settings.models.default}],
                }
            )
        console.print(_json.dumps(payload, indent=2))
        return

    # ── Rich text tree output ──────────────────────────────────
    console.print(
        Panel.fit(
            f'[bold cyan]Pipeline Preview — vaig live[/bold cyan] [dim]"{question}"[/dim]',
            border_style="cyan",
        )
    )

    if skill is not None:
        meta = skill.get_metadata()
        console.print(f"[bold]Skill:[/bold] {meta.display_name}")
        console.print("[bold]Pipeline mode:[/bold] ORCHESTRATED\n")

        agents_config = skill.get_agents_config()
        phases = _build_pipeline_phases(agents_config)

        for idx, phase in enumerate(phases, 1):
            mode_label = "[dim](parallel)[/dim]" if phase["parallel"] else "[dim](sequential)[/dim]"
            console.print(f"[bold yellow]Phase {idx} — {phase['name']}[/bold yellow] {mode_label}")
            agents = phase["agents"]
            for a_idx, agent_cfg in enumerate(agents):
                is_last = a_idx == len(agents) - 1
                connector = "└──" if is_last else "├──"
                name = agent_cfg.get("name", "unknown")
                agent_model = agent_cfg.get("model", model_id or settings.models.default or "—")
                max_iter = agent_cfg.get("max_iterations")
                iter_str = f"{max_iter} iter" if max_iter is not None else "no limit"
                tool_cats = agent_cfg.get("tool_categories", [])
                tools_str = "+".join(tool_cats) if tool_cats else "none"
                console.print(
                    f"  {connector} [cyan]{name:<25}[/cyan]"
                    f"[dim][{agent_model}, {iter_str}][/dim]"
                    f"  tools: [green]{tools_str}[/green]"
                )
            console.print()
    else:
        console.print("[bold]Pipeline mode:[/bold] INFRA_AGENT (single autonomous agent)\n")
        console.print(
            f"  └── [cyan]infra_agent[/cyan]  [dim][{model_id or settings.models.default or '—'}][/dim]  tools: all\n"
        )

    # ── Config sources summary ────────────────────────────────
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  Cluster:    [dim]{gke_config.cluster_name or '(kubeconfig default)'}[/dim]")
    console.print(f"  Namespace:  [dim]{gke_config.default_namespace or 'default'}[/dim]")
    console.print(f"  Project:    [dim]{gke_config.project_id or '(auto-detect)'}[/dim]")
    console.print(f"  Model:      [dim]{model_id or settings.models.default or '—'}[/dim]")
    console.print()
    console.print("[dim]Run without --show-pipeline to execute.[/dim]")


def _export_html_report(
    report: Any,
    *,
    console: Any,
    err_console: Any,
    output: Path | None = None,
    open_browser: bool = False,
) -> bool:
    """Try to write a rich HTML report to disk.

    Returns True if the HTML was written successfully, False otherwise.
    ``report`` should be a HealthReport instance (typed as Any to avoid
    a hard import at module level).

    Args:
        report: The HealthReport instance to render.
        console: Rich console for success messages.
        err_console: Rich console for error messages.
        output: Optional explicit output path. When ``None`` and ``open_browser``
            is False, a timestamped filename is generated in the current working
            directory.  When ``None`` and ``open_browser`` is True, a temporary
            file is used instead so the working directory is not cluttered.
        open_browser: When True, open the generated HTML file in the default
            system web browser after writing.
    """
    try:
        from vaig.ui.html_report import render_health_report_html  # noqa: WPS433

        html_content = render_health_report_html(report)
        if output is not None:
            out_path = output
        elif open_browser:
            # Use a temp file so we don't clutter the working directory when
            # the user just wants to view the report quickly.
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                out_path = Path(tmp.name)
        else:
            timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
            out_path = Path(f"vaig-report-{timestamp}.html")
        out_path.write_text(html_content, encoding="utf-8")
        console.print(f"[bold green]✓ HTML report written:[/bold green] [cyan]{out_path.resolve()}[/cyan]")
        if open_browser:
            file_url = out_path.resolve().as_uri()
            try:
                opened = webbrowser.open(file_url)
                if not opened:
                    console.print(
                        f"[yellow]⚠ Could not open browser automatically. "
                        f"Open manually:[/yellow] [cyan]{file_url}[/cyan]"
                    )
            except Exception as browser_exc:  # noqa: BLE001
                console.print(
                    f"[yellow]⚠ Browser open failed ({browser_exc}). Open manually:[/yellow] [cyan]{file_url}[/cyan]"
                )
        return True
    except Exception as exc:  # pragma: no cover  # noqa: BLE001
        err_console.print(f"[bold red]⚠ Failed to write HTML report:[/bold red] {exc}")
        return False


def _inject_report_metadata(
    report: Any,
    *,
    gke_config: Any = None,
    model_id: str = "",
    orch_result: Any = None,
    tool_logger: Any = None,
    cost_namespaces: list[str] | None = None,
    tool_call_store: ToolCallStore | None = None,
) -> None:
    """Fill metadata fields in *report* from runtime context.

    Thin delegate to :func:`vaig.core.report_metadata.inject_report_metadata`
    — kept as a private function so existing call-sites in this module remain
    unchanged.
    """
    from vaig.core.report_metadata import inject_report_metadata  # noqa: PLC0415

    inject_report_metadata(
        report,
        gke_config=gke_config,
        model_id=model_id,
        orch_result=orch_result,
        tool_logger=tool_logger,
        cost_namespaces=cost_namespaces,
        tool_call_store=tool_call_store,
    )


def _dispatch_format_output(
    orch_result: Any,
    *,
    format_: str | None,
    output: Path | None,
    question: str,
    model_id: str,
    skill_name: str,
    console: Any,
    err_console: Any,
    gke_config: Any = None,
    open_browser: bool = False,
    tool_logger: Any = None,
    all_namespaces: bool = False,
    tool_call_store: ToolCallStore | None = None,
) -> None:
    """Dispatch output based on *format_* for an orchestrated skill result.

    Handles both the HTML rich-export path and the fallback text/json/md export.
    This shared helper is called by both the sync and async orchestrated skill
    functions to avoid duplicating the format-dispatch logic.

    The *format_* value is normalised (stripped and lowercased) once here so
    callers do not need to worry about case or whitespace.

    When ``format_`` is ``"html"`` and ``orch_result.structured_report`` has a
    ``metadata`` attribute, empty / ``"N/A"`` fields are filled in from
    *gke_config* and *model_id* so the SPA dashboard never shows placeholder
    values.

    Args:
        orch_result: The :class:`~vaig.agents.orchestrator.OrchestratorResult`.
        format_: Raw format string from the CLI (may be ``None``).
        output: Optional output path from ``--output``.
        question: Original user question (passed to :func:`_handle_export_output`).
        model_id: Model identifier for metadata.
        skill_name: Skill name for metadata.
        console: Rich console for output.
        err_console: Rich console for errors/warnings.
        gke_config: Optional :class:`~vaig.core.config.GKEConfig` used to inject
            cluster / project metadata into the report before HTML rendering.
        open_browser: When True (and ``format_`` is ``"html"``), open the
            generated file in the default browser.
        tool_logger: Optional :class:`ToolCallLogger`.  When provided, its
            ``tool_name_counts`` and ``tool_count`` are injected into
            ``metadata.tool_usage`` before HTML rendering.
    """
    if not format_ and not output:
        return

    # Normalize once — handles 'HTML', '  html  ', etc.
    normalised_format = format_.strip().lower() if format_ else None

    if normalised_format == "html":
        if orch_result.structured_report is not None:
            # ── Inject runtime metadata into the report before rendering ──────
            _inject_report_metadata(
                orch_result.structured_report,
                gke_config=gke_config,
                model_id=model_id,
                orch_result=orch_result,
                tool_logger=tool_logger,
                cost_namespaces=[] if all_namespaces else None,
                tool_call_store=tool_call_store,
            )
            _export_html_report(
                orch_result.structured_report,
                console=console,
                err_console=err_console,
                output=output,
                open_browser=open_browser,
            )
        else:
            err_console.print(
                Panel(
                    "[yellow]No structured report available — falling back to basic HTML export.[/yellow]",
                    title="[bold red]HTML Export Warning[/bold red]",
                    border_style="red",
                )
            )
            # Fall back to basic HTML export so the user still gets HTML output.
            # When --open is requested and no explicit output path was given, write
            # to a temp file so the browser has a path to open (otherwise the
            # content would be printed to stdout and there'd be nothing to open).
            effective_output = output
            if open_browser and output is None:
                with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                    effective_output = Path(tmp.name)
            _handle_export_output(
                response_text=orch_result.synthesized_output or "",
                question=question,
                model_id=model_id,
                skill_name=skill_name,
                format_="html",  # basic HTML via ExportPayload.to_html() since structured report unavailable
                output=effective_output,
                tokens=orch_result.total_usage or None,
                cost=_compute_cost_str(orch_result.total_usage, model_id),
            )
            if open_browser and effective_output is not None:
                _open_html_in_browser(effective_output, console)
    else:
        _handle_export_output(
            response_text=orch_result.synthesized_output or "",
            question=question,
            model_id=model_id,
            skill_name=skill_name,
            format_=normalised_format,
            output=output,
            tokens=orch_result.total_usage or None,
            cost=_compute_cost_str(orch_result.total_usage, model_id),
        )


def _get_offline_fallback_context(skill_meta: SkillMetadata) -> str:
    """Return the context string used when no live tools are available.

    Emits a warning to the console and returns a plain-text context block
    that describes the active skill.  Used by both sync and async orchestrated
    skill paths when ``tool_count == 0``.
    """
    console.print(
        f"[bold yellow]⚠️  No live tools available for skill '{skill_meta.name}'. "
        "Running in offline context-prepend mode. Results may be limited.[/bold yellow]"
    )
    return (
        f"## Active Skill: {skill_meta.display_name}\n\n"
        f"{skill_meta.description}\n\n"
        f"Apply the {skill_meta.name} analysis methodology to the investigation below."
    )


async def _run_drill_in_loop(
    orchestrator: Any,
    orch_result: Any,
    settings: Settings,
) -> None:
    """Interactive drill-in REPL: ask follow-up questions about the report.

    Builds a system context from the report markdown + evidence ledger summary,
    then loops reading user questions via Rich ``Prompt.ask``.  Each question is
    sent to Gemini (text-only, no tool calls) with the accumulated conversation
    history.  Oldest Q&A turns are trimmed when the accumulated context exceeds
    ``max_chars`` (derived from ``settings.generation.max_output_tokens * 4``).

    Exit conditions: 'exit', 'quit', empty input, ``KeyboardInterrupt``, ``EOFError``.
    """
    import asyncio

    from rich.prompt import Prompt

    report = orch_result.structured_report
    ledger = getattr(getattr(orch_result, "final_state", None), "evidence_ledger", None)

    # ── Build system context (never trimmed) ───────────────────
    try:
        report_md = report.to_markdown() if hasattr(report, "to_markdown") else str(report)
    except Exception:  # noqa: BLE001
        report_md = str(report)

    system_ctx = f"## Investigation Report\n\n{report_md}"
    if ledger is not None:
        try:
            ledger_summary = ledger.to_summary()
            if ledger_summary:
                system_ctx += f"\n\n## Evidence Ledger\n\n{ledger_summary}"
        except Exception:  # noqa: BLE001
            pass

    max_chars: int = (settings.generation.max_output_tokens or 65_536) * 4

    # Q&A turns accumulate here; system_ctx is prepended each time but never trimmed
    qa_turns: list[str] = []

    console.print()
    console.print("[bold cyan]💬 Drill-In Mode — ask follow-up questions about the report[/bold cyan]")
    console.print("[dim]Type 'exit' or press Ctrl+C to return to normal flow[/dim]")
    console.print()

    while True:
        try:
            user_input = await asyncio.to_thread(
                Prompt.ask,
                "[bold green]You[/bold green]",
            )
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]👋 Drill-in session ended.[/dim]")
            return

        user_input = user_input.strip()
        if not user_input or user_input.lower() in {"exit", "quit"}:
            console.print("[dim]👋 Drill-in session ended.[/dim]")
            return

        # Build full context: system_ctx + accumulated Q&A turns
        accumulated = "\n".join(qa_turns)

        # Trim oldest Q&A turns if over cap (system_ctx is never trimmed)
        while qa_turns and len(system_ctx) + len(accumulated) > max_chars:
            qa_turns.pop(0)
            accumulated = "\n".join(qa_turns)

        full_context = system_ctx + ("\n\n" + accumulated if accumulated else "")

        try:
            result = await asyncio.to_thread(
                orchestrator.execute_single,
                user_input,
                context=full_context,
            )
            response_text = result.content if hasattr(result, "content") else str(result)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Error: {exc}[/red]")
            response_text = ""

        if response_text:
            console.print()
            console.print(f"[bold blue]Assistant:[/bold blue] {response_text}")
            console.print()
            qa_turns.append(f"User: {user_input}\nAssistant: {response_text}")


def _execute_orchestrated_skill(
    client: GeminiClientProtocol,
    settings: Settings,
    gke_config: GKEConfig,
    skill: BaseSkill,
    question: str,
    *,
    output: Path | None = None,
    format_: str | None = None,
    model_id: str | None = None,
    tool_call_store: ToolCallStore | None = None,
    summary: bool = False,
    no_bell: bool = False,
    open_browser: bool = False,
    all_namespaces: bool = False,
    detailed: bool = False,
    repo: str | None = None,
    repo_ref: str = "HEAD",
    interactive: bool = False,
) -> HealthReport | None:
    """Execute a skill through the Orchestrator's tool-aware pipeline.

    Used when a skill has ``requires_live_tools=True``.  Instead of the
    simple context-prepend approach, this creates a ToolRegistry, populates
    it with GKE/GCloud tools, and delegates to
    ``Orchestrator.execute_with_tools()`` for the full multi-agent pipeline.

    Returns:
        The structured ``HealthReport`` if one was produced, else ``None``.
    """
    from vaig.core.exceptions import MaxIterationsError
    from vaig.core.headless import execute_skill_headless

    skill_meta = skill.get_metadata()

    # Build tool registry once — pass it to execute_skill_headless to
    # avoid duplicate registry construction (it accepts an optional
    # pre-built registry since the refactor for Copilot review #12).
    tool_registry = _register_live_tools(gke_config, settings=settings, repo=repo, repo_ref=repo_ref)
    tool_count = len(tool_registry.list_tools())

    if tool_count == 0:
        context_str = _get_offline_fallback_context(skill_meta)
        prompt = f"{context_str}\n\n## Investigation Question\n\n{question}"
        try:
            result = client.generate(prompt)
        except Exception as exc:  # noqa: BLE001
            handle_cli_error(exc)
            raise typer.Exit(1) from exc
        console.print()
        console.print("[bold]Assistant response (offline mode):[/bold]")
        console.print(result.text)
        return None

    _print_launch_header(
        skill_meta.display_name,
        gke_config,
        settings,
        tool_count,
        model_id=model_id,
    )

    try:
        console.print(Rule("⏳ Pipeline Execution", style="bright_blue"))
        tool_logger = ToolCallLogger(detailed=detailed)
        progress_display = AgentProgressDisplay(tool_logger)

        # ── Suppress console warnings during pipeline ──
        _suppressed_handlers: list[tuple[logging.Handler, int]] = []
        if not detailed:
            for h in logging.getLogger().handlers + logging.getLogger("vaig").handlers:
                if hasattr(h, "console"):
                    _suppressed_handlers.append((h, h.level))
                    h.setLevel(logging.ERROR)

        try:
            orch_result = execute_skill_headless(
                settings,
                skill,
                question,
                gke_config,
                tool_registry=tool_registry,
                tool_call_store=tool_call_store,
                on_tool_call=tool_logger,
                on_agent_progress=progress_display,
            )
        finally:
            progress_display.stop()
            for handler, original_level in _suppressed_handlers:
                handler.setLevel(original_level)
        tool_logger.print_summary()

        # ── Health Report section ─────────────────────────────
        console.print()
        console.print(Rule("📊 Health Report", style="bright_blue"))
        if summary and orch_result.structured_report is not None:
            # --summary mode: compact output from the structured report
            console.print(orch_result.structured_report.to_summary())
        else:
            # Rich Panel for executive summary (before the full report)
            if orch_result.structured_report is not None:
                print_executive_summary_panel(
                    orch_result.structured_report,
                    console=console,
                )
            if orch_result.structured_report is not None:
                print_service_status_table(orch_result.structured_report, console=console)
                print_severity_detail_blocks(orch_result.structured_report, console=console)
                print_cost_breakdown_table(orch_result.structured_report, console=console)
                print_trend_analysis_table(orch_result.structured_report, console=console)
            if orch_result.synthesized_output:
                print_colored_report(orch_result.synthesized_output, console=console)
            # Rich Table for recommendations (after the full report)
            if orch_result.structured_report is not None:
                print_recommendations_table(
                    orch_result.structured_report,
                    console=console,
                )
        console.print()

        # Dispatch output format — HTML or text/json/md fallback
        _dispatch_format_output(
            orch_result,
            format_=format_,
            output=output,
            question=question,
            model_id=settings.models.default,
            skill_name=skill_meta.name,
            console=console,
            err_console=err_console,
            gke_config=gke_config,
            open_browser=open_browser,
            tool_logger=tool_logger,
            all_namespaces=all_namespaces,
            tool_call_store=tool_call_store,
        )

        # Show agent pipeline summary (includes cost line)
        _show_orchestrated_summary(
            orch_result,
            model_id=settings.models.default,
            agent_timings=progress_display.get_agent_timings(),
        )

        # Generate a single run_id for this execution — used by both
        # auto-export and feedback to ensure consistency (even when
        # auto_export_reports is disabled, we still need a run_id for
        # feedback if export.enabled is True).
        run_id: str | None = None
        if settings.export.enabled:
            from vaig.core.export import save_last_run_id

            run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            save_last_run_id(run_id)

        # Persist report locally for quality analysis (vaig optimize --reports).
        # Uses its own run_id when export is disabled.
        _persist_report_locally(orch_result, run_id=run_id)
        _emit_health_report_completed(settings, orch_result, run_id=run_id)

        # Auto-export report if configured.
        # ADR-4: auto-export fires here for the health report only, immediately after the live
        # summary.  Telemetry and tool-calls are higher-volume and use the explicit CLI push
        # commands (vaig cloud push telemetry / tool-calls) so they are never auto-exported.
        _auto_export_report(settings, orch_result, gke_config, run_id=run_id)

        # Interactive feedback prompt (only when export is enabled)
        _prompt_feedback(settings, run_id=run_id)

        # Interactive drill-in REPL (only when --interactive flag is set)
        if interactive and orch_result.structured_report is not None:
            import asyncio

            from vaig.agents.orchestrator import Orchestrator as _Orchestrator

            _drill_orchestrator = _Orchestrator(client, settings)
            asyncio.run(_run_drill_in_loop(_drill_orchestrator, orch_result, settings))
        elif interactive:
            logger.debug("Drill-in skipped: no structured report available")

        # Notify via terminal bell
        _emit_bell(no_bell=no_bell)

        return cast("HealthReport | None", orch_result.structured_report)

    except MaxIterationsError as exc:
        err_console.print(
            f"\n[bold red]⚠ Max iterations reached ({exc.iterations})[/bold red]\n"
            "[yellow]The orchestrated skill pipeline hit its tool-use iteration limit. "
            "Try narrowing the scope of your question or specifying a namespace/resource.[/yellow]"
        )
        raise typer.Exit(1)  # noqa: B904


def _prompt_feedback(settings: Settings, *, run_id: str | None = None) -> None:
    """Prompt the user for optional post-run feedback.

    Only activates when export is enabled.  Reads a rating (1-5) from stdin;
    pressing Enter (empty input) skips.  If a valid rating is entered, also
    offers an optional comment prompt.  Feedback is exported to BigQuery on a
    daemon thread so it never blocks exit.

    Args:
        settings: Application settings (checked for ``export.enabled``).
        run_id: Pipeline run identifier for this execution.  When ``None``
            a fallback timestamp is generated — but callers should always
            provide the authoritative run_id from the current execution.
    """
    if not settings.export.enabled:
        return

    try:
        raw = console.input("\n[dim]\U0001f4ca Was this analysis helpful? Rate 1-5 (Enter to skip):[/dim] ")
    except (EOFError, KeyboardInterrupt):
        return

    raw = raw.strip()
    if not raw:
        return

    try:
        rating = int(raw)
    except ValueError:
        return

    if not 1 <= rating <= 5:
        return

    # Optional comment
    comment = ""
    try:
        comment = console.input("[dim]   Comment (Enter to skip):[/dim] ").strip()
    except (EOFError, KeyboardInterrupt):
        pass

    # Export in background
    from vaig.core.export import DataExporter

    effective_run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    def _do_export() -> None:
        try:
            exporter = DataExporter(settings.export)
            exporter.export_feedback_to_bigquery(
                rating=rating,
                comment=comment,
                run_id=effective_run_id,
            )
        except Exception:  # noqa: BLE001
            logger.debug("Post-run feedback export failed", exc_info=True)

    thread = threading.Thread(target=_do_export, daemon=True, name="vaig-feedback-export")
    thread.start()
    stars = "\u2605" * rating + "\u2606" * (5 - rating)
    console.print(f"  [green]\u2713[/green] Thanks for your feedback  {stars}")


def _persist_report_locally(
    orch_result: OrchestratorResult,
    *,
    run_id: str | None = None,
) -> None:
    """Save the structured report to the local ReportStore for quality analysis.

    Works independently of the cloud export setting — always persists
    when a structured report is available.  Generates its own ``run_id``
    if one wasn't provided by the export path.

    Errors are logged and swallowed so they never interrupt the CLI flow.
    """
    if orch_result.structured_report is None:
        return
    try:
        from vaig.core.report_store import ReportStore

        effective_run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        store = ReportStore()
        store.save(effective_run_id, orch_result.structured_report.to_dict())
        logger.debug("Persisted report locally for run %s", effective_run_id)
    except Exception:  # noqa: BLE001
        logger.warning("Failed to persist report locally", exc_info=True)


def _emit_health_report_completed(
    settings: Settings,
    orch_result: OrchestratorResult,
    *,
    run_id: str | None = None,
) -> None:
    """Emit a ``HealthReportCompleted`` event on the EventBus.

    The event carries the run_id and the path to the locally-persisted JSONL
    report so that the :class:`~vaig.core.subscribers.memory_subscriber.MemorySubscriber`
    can read findings and record fingerprints without coupling to this function.

    Errors are logged and swallowed so they never interrupt the CLI flow.
    """
    if not settings.memory.enabled or orch_result.structured_report is None:
        return
    try:
        from pathlib import Path

        from vaig.core.event_bus import EventBus
        from vaig.core.events import HealthReportCompleted
        from vaig.core.report_store import _DEFAULT_DIR  # noqa: PLC2701

        effective_run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        report_path = str(Path(_DEFAULT_DIR) / f"{effective_run_id}.jsonl")
        EventBus.get().emit(HealthReportCompleted(run_id=effective_run_id, report_path=report_path))
        logger.debug("Emitted HealthReportCompleted for run %s", effective_run_id)
    except Exception:  # noqa: BLE001
        logger.debug("Failed to emit HealthReportCompleted", exc_info=True)


def _auto_export_report(
    settings: Settings,
    orch_result: OrchestratorResult,
    gke_config: GKEConfig,
    *,
    run_id: str | None = None,
) -> None:
    """Fire-and-forget auto-export of a health report if configured.

    Checks whether auto-export is enabled and a structured report exists, then
    delegates to :func:`vaig.core.export.auto_export_report` on a daemon thread.

    Args:
        settings: Application settings (checked for ``export.enabled`` and
            ``export.auto_export_reports``).
        orch_result: Orchestrator result containing the optional structured report.
        gke_config: GKE configuration for cluster/namespace metadata.
        run_id: Pipeline run identifier.  When provided it is used as-is;
            when ``None`` a fallback timestamp is generated.
    """
    if not (
        settings.export.enabled and settings.export.auto_export_reports and orch_result.structured_report is not None
    ):
        return
    from vaig.core.export import auto_export_report

    effective_run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    auto_export_report(
        config=settings.export,
        report=orch_result.structured_report.to_dict(),
        run_id=effective_run_id,
        cluster_name=gke_config.cluster_name or "",
        namespace=gke_config.default_namespace or "",
    )


def _format_models_used(models_used: list[str]) -> str:
    """Format a list of model IDs for display.

    Thin delegate to :func:`vaig.core.report_metadata.format_models_used`.
    """
    from vaig.core.report_metadata import format_models_used  # noqa: PLC0415

    return format_models_used(models_used)


def _show_orchestrated_summary(
    orch_result: OrchestratorResult,
    *,
    model_id: str = "",
    agent_timings: dict[str, float] | None = None,
) -> None:
    """Display a summary table for an orchestrated skill execution.

    Args:
        orch_result: The orchestrator result.
        model_id: Model identifier for cost display.
        agent_timings: Optional mapping of agent_name → elapsed seconds,
            typically built from :class:`AgentProgressDisplay._completed`.
            When provided, per-agent timing comes from here instead of
            (unreliable) ``agent_result.metadata`` keys.
    """
    console.print(Rule("📈 Pipeline Summary", style="bright_blue"))

    table = Table(title="Pipeline Summary", show_lines=True)
    table.add_column("Agent", style="cyan")
    table.add_column("Role", style="dim")
    table.add_column("Status", style="bold")
    table.add_column("Time", style="dim", justify="right")
    table.add_column("Tokens", style="dim", justify="right")

    total_time = 0.0
    total_tokens_sum = 0
    timings = agent_timings or {}

    for agent_result in orch_result.agent_results:
        status = "[green]✓ success[/green]" if agent_result.success else "[red]✗ failed[/red]"
        role = agent_result.metadata.get("role") or agent_result.agent_name

        # Extract timing: prefer AgentProgressDisplay data, then metadata fallback
        elapsed = timings.get(agent_result.agent_name)
        if elapsed is None:
            elapsed = agent_result.metadata.get("elapsed_time") or agent_result.metadata.get("duration")
        if isinstance(elapsed, (int, float)):
            time_str = f"{elapsed:.1f}s"
            total_time += elapsed
        else:
            time_str = "—"

        # Extract tokens from agent usage
        agent_usage = agent_result.usage if isinstance(agent_result.usage, dict) else {}
        agent_tokens = agent_usage.get("total_tokens", 0)
        if isinstance(agent_tokens, (int, float)) and agent_tokens:
            tokens_str = f"{int(agent_tokens):,}"
            total_tokens_sum += int(agent_tokens)
        else:
            tokens_str = "—"

        table.add_row(agent_result.agent_name, role, status, time_str, tokens_str)

    # TOTAL row
    table.add_section()
    total_time_str = f"{total_time:.1f}s" if total_time > 0 else "—"
    total_tokens_str = f"{total_tokens_sum:,}" if total_tokens_sum > 0 else "—"
    # Count successful agents for the status column
    success_count = sum(1 for ar in orch_result.agent_results if getattr(ar, "success", True))
    total_count = len(orch_result.agent_results)
    status_text = f"{success_count}/{total_count} ✅" if total_count > 0 else "—"
    table.add_row(
        Text("TOTAL", style="bold"),
        "",
        Text(status_text, style="bold"),
        Text(total_time_str, style="bold"),
        Text(total_tokens_str, style="bold"),
    )

    console.print(table)

    # Show cost summary for the full pipeline.
    # Prefer the pre-accumulated run_cost_usd (correct per-agent model pricing)
    # and fall back to recalculation only when it is zero/unavailable.
    _raw_cost = getattr(orch_result, "run_cost_usd", None)
    run_cost = _raw_cost if isinstance(_raw_cost, (int, float)) else 0.0
    models_used_list: list[str] = getattr(orch_result, "models_used", [])
    # Display label (may be a formatted string like "gemini-2.5-flash ×7")
    effective_model = _format_models_used(models_used_list) or model_id
    # Raw model ID for pricing lookups — must not be a formatted display string.
    # Use the single unique model when all agents share one model, else fall back
    # to the caller-supplied model_id.
    unique_models = list(dict.fromkeys(models_used_list))  # deduplicate, preserve order
    pricing_model_id = unique_models[0] if len(unique_models) == 1 else model_id
    if run_cost > 0.0:
        from vaig.core.pricing import format_cost

        usage = orch_result.total_usage or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        thinking_tokens = usage.get("thinking_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        if total_tokens > 0 or prompt_tokens > 0 or completion_tokens > 0:
            parts = [f"{prompt_tokens:,} in", f"{completion_tokens:,} out"]
            if thinking_tokens:
                parts.append(f"{thinking_tokens:,} thinking")
            cost_str = format_cost(run_cost)
            model_label = f" ({effective_model})" if effective_model else ""
            console.print(
                f"[dim]📊 Tokens: {' / '.join(parts)} ({total_tokens:,} total) │ Cost: {cost_str}{model_label}[/dim]"
            )
        else:
            cost_str = format_cost(run_cost)
            model_label = f" ({effective_model})" if effective_model else ""
            console.print(f"[dim]📊 Cost: {cost_str}{model_label}[/dim]")
    else:
        # No pre-computed cost — fall back to recalculation via show_cost_line.
        # Pass the raw model ID (pricing_model_id), not the formatted display string.
        _show_cost_line(orch_result.total_usage or None, pricing_model_id or model_id)

    if not orch_result.success:
        err_console.print("[bold red]⚠ Pipeline completed with errors[/bold red]")


def _open_html_in_browser(html_path: Path, console: Any) -> None:
    """Open *html_path* in the system default browser.

    Creates no temporary files — the caller is responsible for providing a
    valid path.  Handles browser-open failures gracefully by printing an
    actionable message instead of raising.

    Args:
        html_path: Absolute or resolvable path to the HTML file to open.
        console: Rich ``Console`` instance for user-facing messages.
    """
    file_url = html_path.resolve().as_uri()
    console.print(f"[bold green]✓ Report written:[/bold green] [cyan]{html_path.resolve()}[/cyan]")
    try:
        opened = webbrowser.open(file_url)
        if not opened:
            console.print(
                f"[yellow]⚠ Could not open browser automatically. Open manually:[/yellow] [cyan]{file_url}[/cyan]"
            )
    except Exception as browser_exc:  # noqa: BLE001
        console.print(f"[yellow]⚠ Browser open failed ({browser_exc}). Open manually:[/yellow] [cyan]{file_url}[/cyan]")


def _execute_live_mode(
    client: GeminiClientProtocol,
    gke_config: GKEConfig,
    question: str,
    context: str,
    *,
    settings: Settings | None = None,
    output: Path | None = None,
    format_: str | None = None,
    skill_name: str | None = None,
    model_id: str | None = None,
    tool_call_store: ToolCallStore | None = None,
    tool_result_cache: ToolResultCache | None = None,
    open_browser: bool = False,
    detailed: bool = False,
    repo: str | None = None,
    repo_ref: str = "HEAD",
) -> None:
    """Execute an infrastructure investigation using the InfraAgent.

    Handles the full lifecycle: banner, agent creation, execution,
    result display, and summary — following the same pattern as _execute_code_mode.
    """
    from vaig.agents.infra_agent import InfraAgent
    from vaig.core.exceptions import MaxIterationsError

    console.print(
        Panel.fit(
            "[bold green]🔍 Live Infrastructure Mode[/bold green]\n"
            f"[dim]Cluster: {gke_config.cluster_name or '(kubeconfig default)'}[/dim]\n"
            f"[dim]Namespace: {gke_config.default_namespace} | "
            f"Project: {gke_config.project_id or '(auto-detect)'}[/dim]\n"
            "[dim]Most tools are read-only; write tools (scale/restart/label) are available[/dim]",
            border_style="green",
        )
    )

    agent = InfraAgent(
        client,
        gke_config,
        settings=settings,
        model_id=model_id or client.current_model,
    )

    # Show registered tool count
    tool_count = len(agent.registry.list_tools())
    if tool_count == 0:
        err_console.print(
            "[bold red]No infrastructure tools available![/bold red]\n"
            "[yellow]Install optional dependencies:[/yellow]\n"
            "  pip install vertex-ai-toolkit[live]       # GKE tools\n"
            "  pip install google-cloud-logging google-cloud-monitoring  # GCP observability"
        )
        raise typer.Exit(1)

    console.print(f"[dim]{tool_count} tools loaded[/dim]")

    try:
        console.print("[bold cyan]🤖 Infrastructure agent investigating...[/bold cyan]")
        tool_logger = ToolCallLogger(detailed=detailed)
        effective_cache = tool_result_cache or ToolResultCache()
        result = agent.execute(
            question,
            context=context,
            on_tool_call=tool_logger,
            tool_call_store=tool_call_store,
            tool_result_cache=effective_cache,
        )
        tool_logger.print_summary()

        # Display final response with severity coloring
        console.print()
        if result.content:
            print_colored_report(result.content, console=console)
        console.print()

        # When --open is requested, use a temp file so we can pass its path to
        # the browser.  _handle_export_output writes the file when output is set;
        # afterwards we open it in the default browser.
        effective_output = output
        if open_browser and format_ and format_.strip().lower() == "html" and output is None:
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                effective_output = Path(tmp.name)

        _handle_export_output(
            response_text=result.content or "",
            question=question,
            model_id=model_id or client.current_model,
            skill_name=skill_name,
            format_=format_,
            output=effective_output,
            tokens=result.usage or None,
            cost=_compute_cost_str(result.usage, model_id or client.current_model),
        )

        if open_browser and format_ and format_.strip().lower() == "html":
            open_target = effective_output or output
            if open_target is not None:
                _open_html_in_browser(open_target, console)

        # Show summary (reuse coding summary — same metadata shape)
        _show_coding_summary(result)

    except MaxIterationsError as exc:
        err_console.print(
            f"\n[bold red]⚠ Max iterations reached ({exc.iterations})[/bold red]\n"
            "[yellow]The infrastructure agent hit its tool-use iteration limit. "
            "Try narrowing the scope of your question or specifying a namespace/resource.[/yellow]"
        )
        raise typer.Exit(1)  # noqa: B904


# ── Async Live Mode Implementations ──────────────────────────


async def _async_execute_live_mode(
    client: GeminiClientProtocol,
    gke_config: GKEConfig,
    question: str,
    context: str,
    *,
    settings: Settings | None = None,
    output: Path | None = None,
    format_: str | None = None,
    skill_name: str | None = None,
    model_id: str | None = None,
    tool_call_store: ToolCallStore | None = None,
    tool_result_cache: ToolResultCache | None = None,
    open_browser: bool = False,
    detailed: bool = False,
) -> None:
    """Async version of :func:`_execute_live_mode`.

    Uses ``InfraAgent.async_execute()`` for non-blocking tool loops.
    """
    from vaig.agents.infra_agent import InfraAgent
    from vaig.core.exceptions import MaxIterationsError

    console.print(
        Panel.fit(
            "[bold green]🔍 Live Infrastructure Mode (async)[/bold green]\n"
            f"[dim]Cluster: {gke_config.cluster_name or '(kubeconfig default)'}[/dim]\n"
            f"[dim]Namespace: {gke_config.default_namespace} | "
            f"Project: {gke_config.project_id or '(auto-detect)'}[/dim]\n"
            "[dim]Most tools are read-only; write tools (scale/restart/label) are available[/dim]",
            border_style="green",
        )
    )

    agent = InfraAgent(
        client,
        gke_config,
        settings=settings,
        model_id=model_id or client.current_model,
    )

    tool_count = len(agent.registry.list_tools())
    if tool_count == 0:
        err_console.print(
            "[bold red]No infrastructure tools available![/bold red]\n"
            "[yellow]Install optional dependencies:[/yellow]\n"
            "  pip install vertex-ai-toolkit[live]       # GKE tools\n"
            "  pip install google-cloud-logging google-cloud-monitoring  # GCP observability"
        )
        raise typer.Exit(1)

    console.print(f"[dim]{tool_count} tools loaded[/dim]")

    try:
        console.print("[bold cyan]🤖 Infrastructure agent investigating (async)...[/bold cyan]")
        tool_logger = ToolCallLogger(detailed=detailed)
        effective_cache = tool_result_cache or ToolResultCache()
        result = await agent.async_execute(
            question,
            context=context,
            on_tool_call=tool_logger,
            tool_call_store=tool_call_store,
            tool_result_cache=effective_cache,
        )
        tool_logger.print_summary()

        console.print()
        if result.content:
            print_colored_report(result.content, console=console)
        console.print()

        # When --open is requested, use a temp file so we can pass its path to
        # the browser.  _handle_export_output writes the file when output is set;
        # afterwards we open it in the default browser.
        effective_output = output
        if open_browser and format_ and format_.strip().lower() == "html" and output is None:
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                effective_output = Path(tmp.name)

        _handle_export_output(
            response_text=result.content or "",
            question=question,
            model_id=model_id or client.current_model,
            skill_name=skill_name,
            format_=format_,
            output=effective_output,
            tokens=result.usage or None,
            cost=_compute_cost_str(result.usage, model_id or client.current_model),
        )

        if open_browser and format_ and format_.strip().lower() == "html":
            open_target = effective_output or output
            if open_target is not None:
                _open_html_in_browser(open_target, console)

        _show_coding_summary(result)

    except MaxIterationsError as exc:
        err_console.print(
            f"\n[bold red]⚠ Max iterations reached ({exc.iterations})[/bold red]\n"
            "[yellow]The infrastructure agent hit its tool-use iteration limit. "
            "Try narrowing the scope of your question or specifying a namespace/resource.[/yellow]"
        )
        raise typer.Exit(1)  # noqa: B904


async def _async_execute_orchestrated_skill(
    client: GeminiClientProtocol,
    settings: Settings,
    gke_config: GKEConfig,
    skill: BaseSkill,
    question: str,
    *,
    output: Path | None = None,
    format_: str | None = None,
    model_id: str | None = None,
    tool_call_store: ToolCallStore | None = None,
    summary: bool = False,
    no_bell: bool = False,
    open_browser: bool = False,
    all_namespaces: bool = False,
    detailed: bool = False,
    repo: str | None = None,
    repo_ref: str = "HEAD",
) -> None:
    """Async version of :func:`_execute_orchestrated_skill`.

    Uses ``Orchestrator.async_execute_with_tools()`` for non-blocking execution.
    """
    from vaig.agents.orchestrator import Orchestrator
    from vaig.core.exceptions import MaxIterationsError

    skill_meta = skill.get_metadata()

    tool_registry = _register_live_tools(gke_config, settings=settings, repo=repo, repo_ref=repo_ref)

    gke_credentials = None
    if settings is not None:
        from vaig.core.auth import get_gke_credentials as _get_gke_creds

        gke_credentials = _get_gke_creds(settings)

    is_autopilot: bool | None = None
    try:
        from vaig.tools.gke_tools import detect_autopilot

        is_autopilot = detect_autopilot(gke_config, credentials=gke_credentials)
    except ImportError:
        pass

    tool_count = len(tool_registry.list_tools())
    if tool_count == 0:
        context_str = _get_offline_fallback_context(skill_meta)
        prompt = f"{context_str}\n\n## Investigation Question\n\n{question}"
        try:
            result = await client.async_generate(prompt)
        except Exception as exc:  # noqa: BLE001
            handle_cli_error(exc)
            raise typer.Exit(1) from exc
        console.print()
        console.print("[bold]Assistant response (offline mode):[/bold]")
        console.print(result.text)
        return

    _print_launch_header(
        skill_meta.display_name,
        gke_config,
        settings,
        tool_count,
        model_id=model_id,
        is_async=True,
    )

    orchestrator = Orchestrator(client, settings)

    try:
        console.print(Rule("⏳ Pipeline Execution", style="bright_blue"))
        tool_logger = ToolCallLogger(detailed=detailed)
        progress_display = AgentProgressDisplay(tool_logger)

        # ── Suppress console warnings during pipeline (Change 1) ──
        _suppressed_handlers: list[tuple[logging.Handler, int]] = []
        if not detailed:
            for h in logging.getLogger().handlers + logging.getLogger("vaig").handlers:
                if hasattr(h, "console"):
                    _suppressed_handlers.append((h, h.level))
                    h.setLevel(logging.ERROR)

        try:
            orch_result = await orchestrator.async_execute_with_tools(
                query=question,
                skill=skill,
                tool_registry=tool_registry,
                strategy="sequential",
                is_autopilot=is_autopilot,
                on_tool_call=tool_logger,
                tool_call_store=tool_call_store,
                on_agent_progress=progress_display,
                gke_namespace=gke_config.default_namespace,
                gke_location=gke_config.location,
                gke_cluster_name=gke_config.cluster_name,
            )
        finally:
            progress_display.stop()
            # Restore original handler levels (always, even on exception).
            for handler, original_level in _suppressed_handlers:
                handler.setLevel(original_level)
        tool_logger.print_summary()

        # ── Health Report section ─────────────────────────────
        console.print()
        console.print(Rule("📊 Health Report", style="bright_blue"))
        if summary and orch_result.structured_report is not None:
            # --summary mode: compact output from the structured report
            console.print(orch_result.structured_report.to_summary())
        else:
            # Rich Panel for executive summary (before the full report)
            if orch_result.structured_report is not None:
                print_executive_summary_panel(
                    orch_result.structured_report,
                    console=console,
                )
            if orch_result.structured_report is not None:
                print_service_status_table(orch_result.structured_report, console=console)
                print_severity_detail_blocks(orch_result.structured_report, console=console)
                print_cost_breakdown_table(orch_result.structured_report, console=console)
                print_trend_analysis_table(orch_result.structured_report, console=console)
            if orch_result.synthesized_output:
                print_colored_report(orch_result.synthesized_output, console=console)
            # Rich Table for recommendations (after the full report)
            if orch_result.structured_report is not None:
                print_recommendations_table(
                    orch_result.structured_report,
                    console=console,
                )
        console.print()

        # Dispatch output format — HTML or text/json/md fallback
        _dispatch_format_output(
            orch_result,
            format_=format_,
            output=output,
            question=question,
            model_id=settings.models.default,
            skill_name=skill_meta.name,
            console=console,
            err_console=err_console,
            gke_config=gke_config,
            open_browser=open_browser,
            tool_logger=tool_logger,
            all_namespaces=all_namespaces,
            tool_call_store=tool_call_store,
        )

        _show_orchestrated_summary(
            orch_result,
            model_id=settings.models.default,
            agent_timings=progress_display.get_agent_timings(),
        )

        # Generate a single run_id for this execution — used by both
        # auto-export and feedback to ensure consistency (even when
        # auto_export_reports is disabled, we still need a run_id for
        # feedback if export.enabled is True).
        run_id: str | None = None
        if settings.export.enabled:
            from vaig.core.export import save_last_run_id

            run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            save_last_run_id(run_id)

        # Persist report locally for quality analysis (vaig optimize --reports).
        # Uses its own run_id when export is disabled.
        _persist_report_locally(orch_result, run_id=run_id)

        # Auto-export report if configured.
        # ADR-4: auto-export fires here for the health report only, immediately after the live
        # summary.  Telemetry and tool-calls are higher-volume and use the explicit CLI push
        # commands (vaig cloud push telemetry / tool-calls) so they are never auto-exported.
        _auto_export_report(settings, orch_result, gke_config, run_id=run_id)

        # Interactive feedback prompt (only when export is enabled)
        _prompt_feedback(settings, run_id=run_id)

        # Notify via terminal bell
        _emit_bell(no_bell=no_bell)

    except MaxIterationsError as exc:
        err_console.print(
            f"\n[bold red]⚠ Max iterations reached ({exc.iterations})[/bold red]\n"
            "[yellow]The orchestrated skill pipeline hit its tool-use iteration limit. "
            "Try narrowing the scope of your question or specifying a namespace/resource.[/yellow]"
        )
        raise typer.Exit(1)  # noqa: B904

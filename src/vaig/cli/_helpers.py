"""Shared CLI helpers — utilities used across multiple command modules."""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Callable, Coroutine
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel

from vaig import __version__

if TYPE_CHECKING:
    from vaig.cli.export import ExportPayload
    from vaig.core.config import Settings
    from vaig.core.protocols import PlatformAuthProtocol

console = Console()
err_console = Console(stderr=True)
logger = logging.getLogger(__name__)

# Keep a module-level reference so the subscriber isn't garbage-collected.
_telemetry_subscriber: object | None = None
_audit_subscriber: object | None = None


def _init_telemetry(settings: Settings) -> None:
    """Initialize the telemetry collector **and** wire the TelemetrySubscriber.

    Safe to call multiple times — the collector is a singleton and the
    subscriber is created only once (guarded by ``_telemetry_subscriber``).
    This ensures that events emitted via the EventBus (e.g. from
    ``CostTracker.record`` or ``track_command``) are forwarded to the
    collector's SQLite store.
    """
    global _telemetry_subscriber  # noqa: PLW0603
    try:
        from vaig.core.telemetry import get_telemetry_collector

        collector = get_telemetry_collector(settings)

        if _telemetry_subscriber is None:
            from vaig.core.subscribers import TelemetrySubscriber

            _telemetry_subscriber = TelemetrySubscriber(collector)
    except Exception:  # noqa: BLE001
        pass


def _init_audit(settings: Settings) -> None:
    """Initialize the AuditSubscriber if audit logging is enabled.

    Safe to call multiple times — the subscriber is created only once
    (guarded by ``_audit_subscriber``).  Mirrors :func:`_init_telemetry`.
    """
    global _audit_subscriber  # noqa: PLW0603
    if not settings.audit.enabled:
        return
    try:
        if _audit_subscriber is None:
            from vaig.core.auth import get_credentials
            from vaig.core.subscribers.audit_subscriber import AuditSubscriber

            credentials = get_credentials(settings)
            _audit_subscriber = AuditSubscriber(settings, credentials)
            logger.info("AuditSubscriber initialized")
    except Exception:  # noqa: BLE001
        logger.warning("Failed to initialize AuditSubscriber — audit logging disabled")


# ── Telemetry decorator ───────────────────────────────────────
def track_command(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that emits a ``cli_command`` telemetry event with timing.

    Wraps a Typer command function.  Telemetry failures are silenced so
    they never affect the command itself.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            try:
                from vaig.core.event_bus import EventBus
                from vaig.core.events import CliCommandTracked

                duration_ms = (time.perf_counter() - t0) * 1000
                EventBus.get().emit(
                    CliCommandTracked(command_name=fn.__name__, duration_ms=duration_ms)
                )
            except Exception:  # noqa: BLE001
                pass

    return wrapper


def track_command_async(fn: Callable[..., Coroutine[Any, Any, Any]]) -> Callable[..., Coroutine[Any, Any, Any]]:
    """Async version of :func:`track_command`.

    Wraps an ``async def`` command function, emitting a ``cli_command``
    telemetry event with timing.  Telemetry failures are silenced.
    """

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return await fn(*args, **kwargs)
        finally:
            try:
                from vaig.core.event_bus import EventBus
                from vaig.core.events import CliCommandTracked

                duration_ms = (time.perf_counter() - t0) * 1000
                EventBus.get().emit(
                    CliCommandTracked(command_name=fn.__name__, duration_ms=duration_ms)
                )
            except Exception:  # noqa: BLE001
                pass

    return wrapper


def async_run_command(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run an async command implementation from a sync Typer entry point.

    This is the standard bridge for ``sync Typer command → async internals``.
    Uses ``asyncio.run()`` to create a new event loop and execute the
    coroutine.  If an event loop is already running (e.g. Jupyter),
    falls back to the thread-based approach in :func:`run_sync`.

    Usage::

        def ask_command(...):
            async_run_command(_async_ask_impl(...))

    Args:
        coro: The async coroutine to execute.

    Returns:
        The coroutine's return value.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop running — safe to use asyncio.run()
        return asyncio.run(coro)

    # Already inside a running loop — delegate to run_sync which
    # handles the nested-loop case via a background thread.
    from vaig.core.async_utils import run_sync

    return run_sync(coro)


# ── Shared State ──────────────────────────────────────────────
def _get_settings(config: str | None = None) -> Settings:
    """Load settings (lazy import to avoid heavy imports on --help)."""
    from vaig.core.config import get_settings

    return get_settings(config)


def _banner() -> None:
    """Print the VAIG banner."""
    console.print(
        Panel.fit(
            f"[bold cyan]VAIG[/bold cyan] — Vertex AI Gemini Toolkit v{__version__}\n"
            "[dim]Multi-agent AI assistant powered by Google Gemini[/dim]",
            border_style="bright_blue",
        )
    )


def _apply_subcommand_log_flags(*, verbose: bool, debug: bool) -> None:
    """Apply --verbose / --debug flags from a subcommand.

    Subcommand-level flags override the global callback flags so that
    ``vaig live "query" -d`` works the same as ``vaig -d live "query"``.
    Only overrides if a flag is actually set (non-default).

    File logging settings are preserved from the loaded configuration
    so that subcommand-level overrides only change console level/show_path.
    """
    if not verbose and not debug:
        return

    from vaig.core.config import get_settings
    from vaig.core.log import reset_logging, setup_logging

    # Reset the idempotent guard so we can reconfigure with the subcommand flags.
    reset_logging()

    # Preserve file logging settings from the loaded config
    settings = get_settings()
    log_cfg = settings.logging

    if debug:
        setup_logging(
            "DEBUG",
            show_path=True,
            file_enabled=log_cfg.file_enabled,
            file_path=log_cfg.file_path,
            file_level=log_cfg.file_level,
            file_max_bytes=log_cfg.file_max_bytes,
            file_backup_count=log_cfg.file_backup_count,
        )
    elif verbose:
        setup_logging(
            "INFO",
            show_path=False,
            file_enabled=log_cfg.file_enabled,
            file_path=log_cfg.file_path,
            file_level=log_cfg.file_level,
            file_max_bytes=log_cfg.file_max_bytes,
            file_backup_count=log_cfg.file_backup_count,
        )


# ── Output / export helpers ───────────────────────────────────
def _save_output(output: Path, content: str) -> None:
    """Write response content to a file and print confirmation."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    console.print(f"[green]✓ Response saved to {output}[/green]")


def _build_cost_markdown_section(
    tokens: dict[str, int],
    model_id: str,
    cost: str | None = None,
) -> str | None:
    """Build a ``## Cost & Usage Summary`` markdown section for plain file saves.

    Returns ``None`` if no meaningful token data is available.
    """
    prompt_tokens = tokens.get("prompt_tokens", 0)
    completion_tokens = tokens.get("completion_tokens", 0)
    thinking_tokens = tokens.get("thinking_tokens", 0)
    total_tokens = tokens.get("total_tokens", 0)

    if total_tokens == 0 and prompt_tokens == 0 and completion_tokens == 0:
        return None

    lines = [
        "---",
        "## Cost & Usage Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Input tokens | {prompt_tokens:,} |",
        f"| Output tokens | {completion_tokens:,} |",
    ]

    if thinking_tokens:
        lines.append(f"| Thinking tokens | {thinking_tokens:,} |")

    lines.append(f"| Total tokens | {total_tokens:,} |")
    lines.append(f"| Model | {model_id} |")

    if cost:
        lines.append(f"| Estimated cost | {cost} |")

    lines.append("")
    return "\n".join(lines)


def _format_session_date(iso_str: str) -> str:
    """Format an ISO timestamp to a human-friendly short form."""
    if not iso_str:
        return "—"
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return iso_str[:16] if len(iso_str) > 16 else iso_str


def _resolve_session_id(manager: object, prefix: str) -> str:
    """Resolve a session ID prefix to a full session ID.

    Supports both full UUIDs and short prefixes (e.g. first 8 chars).
    """
    from vaig.session.manager import SessionManager

    if not isinstance(manager, SessionManager):
        return prefix

    # Try exact match first
    session = manager._store.get_session(prefix)
    if session:
        return prefix

    # Try prefix match
    sessions = manager.list_sessions(limit=100)
    matches = [s for s in sessions if s["id"].startswith(prefix)]
    if len(matches) == 1:
        return str(matches[0]["id"])
    if len(matches) > 1:
        err_console.print(f"[yellow]Ambiguous prefix '{prefix}' matches {len(matches)} sessions. Use more characters.[/yellow]")
    return prefix


def _build_export_payload(
    *,
    question: str,
    response_text: str,
    model_id: str,
    skill_name: str | None = None,
    tokens: dict[str, int] | None = None,
    cost: str | None = None,
    context_files: list[str] | None = None,
    agent_results: list[dict[str, Any]] | None = None,
) -> ExportPayload:
    """Build an ExportPayload from ask/live command results."""
    from datetime import datetime

    from vaig.cli.export import ExportMetadata, ExportPayload

    meta = ExportMetadata(
        model=model_id,
        skill=skill_name,
        timestamp=datetime.now(UTC).isoformat(),
        tokens=tokens or {},
        cost=cost,
        vaig_version=__version__,
    )
    return ExportPayload(
        question=question,
        response=response_text,
        metadata=meta,
        context_files=context_files or [],
        agent_results=agent_results or [],
    )


def _handle_export_output(
    *,
    response_text: str,
    question: str,
    model_id: str,
    skill_name: str | None = None,
    context_files: list[str] | None = None,
    format_: str | None = None,
    output: Path | None = None,
    tokens: dict[str, int] | None = None,
    cost: str | None = None,
) -> None:
    """Handle exporting/saving output based on --format and --output flags.

    When --format is set, builds an ExportPayload and formats it.
    When --output is set (with or without --format), writes to file.
    When --format is set without --output, prints formatted content to stdout.
    When neither is set, does nothing (display already handled by caller).

    ``--format rich`` is a special passthrough: it means "use the default Rich
    terminal display" (which the caller already rendered), so no file export is
    performed.  This allows users to explicitly request Rich output without
    triggering a ValueError from the export formatter.
    """
    if not format_ and not output:
        return

    # "rich" means terminal display only — already rendered by the caller.
    if format_ and format_.strip().lower() == "rich":
        return

    if format_:
        from vaig.cli.export import format_export

        payload = _build_export_payload(
            question=question,
            response_text=response_text,
            model_id=model_id,
            skill_name=skill_name,
            context_files=context_files,
            tokens=tokens,
            cost=cost,
        )
        content = format_export(payload, format_)
        if output:
            _save_output(output, content)
        else:
            console.print(content)
    elif output:
        # Plain file save — append cost summary section if data is available
        save_content = response_text
        if isinstance(tokens, dict) and tokens:
            cost_section = _build_cost_markdown_section(tokens, model_id, cost)
            if cost_section:
                save_content = f"{response_text}\n\n{cost_section}"
        _save_output(output, save_content)


def _show_cost_line(usage: dict[str, int] | None, model_id: str) -> None:
    """Show a compact cost/token summary line — delegates to display helper."""
    if not isinstance(usage, dict):
        return
    from vaig.cli.display import show_cost_summary

    show_cost_summary(usage, model_id, console=console)


def _compute_cost_str(usage: dict[str, int] | None, model_id: str) -> str | None:
    """Compute a formatted cost string from usage metadata.

    Returns ``None`` if usage is empty or cost cannot be calculated.
    """
    if not isinstance(usage, dict) or not usage:
        return None
    from vaig.core.pricing import calculate_cost, format_cost

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    thinking_tokens = usage.get("thinking_tokens", 0)
    cost = calculate_cost(model_id, prompt_tokens, completion_tokens, thinking_tokens)
    return format_cost(cost)


def _cli_confirm(tool_name: str, args: dict[str, Any]) -> bool:
    """Confirm destructive tool operations — delegates to shared display helper."""
    from vaig.cli.display import confirm_tool_operation

    return confirm_tool_operation(tool_name, args, console=console)


def _show_coding_summary(result: object) -> None:
    """Display tool execution feedback — delegates to shared display helper."""
    from vaig.cli.display import show_tool_execution_summary

    show_tool_execution_summary(result, console=console)  # type: ignore[arg-type]


# ── Platform auth helpers ─────────────────────────────────────

# Module-level reference to the PlatformAuthManager instance used by
# ``_check_platform_auth``.  Created lazily on first call and reused
# for the remainder of the CLI invocation.
_platform_auth_manager: PlatformAuthProtocol | None = None


def _apply_enforced_config(settings: Settings, enforced: dict[str, Any]) -> None:
    """Override ``settings`` fields with backend-enforced values.

    ``enforced`` maps dotted field paths (e.g. ``"budget.max_cost_per_run"``)
    to their enforced values.  This mutates ``settings`` in-place via
    ``setattr`` on the appropriate nested model.
    """
    for dotted_key, value in enforced.items():
        parts = dotted_key.split(".")
        obj: Any = settings
        try:
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
            logger.debug("Enforced config: %s = %r", dotted_key, value)
        except (AttributeError, TypeError):
            logger.warning("Cannot apply enforced field %r — path not found", dotted_key)


def _check_platform_auth(settings: Settings) -> None:
    """Validate platform auth and apply enforced config.

    No-op when ``settings.platform.enabled`` is ``False``.  When enabled:

    1. Creates (or reuses) a ``PlatformAuthManager``.
    2. Checks authentication — exits if not authenticated.
    3. Fetches a valid token — exits if session expired.
    4. Fetches enforced config and applies it to ``settings``.

    Raises ``typer.Exit(1)`` on auth failures so the calling command
    never executes unauthenticated.
    """
    global _platform_auth_manager  # noqa: PLW0603

    if not settings.platform.enabled:
        return

    import typer

    from vaig.core.platform_auth import PlatformAuthManager

    if _platform_auth_manager is None:
        _platform_auth_manager = PlatformAuthManager(
            backend_url=settings.platform.backend_url,
            org_id=settings.platform.org_id,
        )

    manager: PlatformAuthManager = _platform_auth_manager  # type: ignore[assignment]

    if not manager.is_authenticated():
        err_console.print("[red]Not authenticated. Run: vaig login[/red]")
        raise typer.Exit(code=1)

    token = manager.get_token()
    if token is None:
        err_console.print("[red]Session expired. Run: vaig login[/red]")
        raise typer.Exit(code=1)

    # Fetch and apply enforced config (graceful degradation on failure)
    enforced = manager.get_enforced_config()
    if enforced:
        _apply_enforced_config(settings, enforced)


def handle_cli_error(exc: Exception, *, debug: bool = False) -> None:
    """Format and print a VAIG exception, then raise ``typer.Exit(1)``.

    This is the single error-boundary function for all CLI commands.
    It converts raw exceptions into user-friendly Rich-formatted output
    with actionable fix suggestions.

    Args:
        exc: The exception to format and display.
        debug: When True, includes the full traceback.

    Raises:
        typer.Exit: Always — with ``code=1`` to signal failure.
    """
    import typer

    from vaig.core.exceptions import format_error_for_user

    err_console.print(format_error_for_user(exc, debug=debug))
    raise typer.Exit(code=1)

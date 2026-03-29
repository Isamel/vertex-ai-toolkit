"""CLI app — Typer commands for VAIG toolkit.

This module is the **slim entry point** for the CLI. It creates the main
Typer ``app``, registers sub-apps and commands from the ``commands/``
package, and re-exports symbols that tests and other code import via
``from vaig.cli.app import …``.

Actual command implementations live in:
- ``vaig.cli.commands.chat``        — interactive REPL
- ``vaig.cli.commands.ask``         — single-shot question
- ``vaig.cli.commands.live``        — GKE/GCP infrastructure investigation
- ``vaig.cli.commands.discover``    — autonomous cluster health scanning
- ``vaig.cli.commands.sessions``    — session management sub-commands
- ``vaig.cli.commands.models``      — model listing sub-commands
- ``vaig.cli.commands.skills``      — skill management sub-commands
- ``vaig.cli.commands.mcp``         — MCP server management sub-commands
- ``vaig.cli.commands.doctor``      — environment healthcheck
- ``vaig.cli.commands.stats``       — telemetry stats sub-commands
- ``vaig.cli.commands.export_cmd``  — session export command
- ``vaig.cli.commands._code``       — code mode + chunked analysis helpers

Shared helpers (console, track_command, _get_settings, etc.) live in
``vaig.cli._helpers``.
"""

from __future__ import annotations

from typing import Annotated

import typer

from vaig import __version__

# ── Re-export shared helpers so existing imports keep working ─
# Tests and other modules import from vaig.cli.app:
#   app, track_command, _get_settings, _build_gke_config,
#   _register_live_tools, _format_session_date, _resolve_session_id, ...
from vaig.cli._helpers import (  # noqa: F401
    _apply_subcommand_log_flags,
    _banner,
    _build_cost_markdown_section,
    _build_export_payload,
    _cli_confirm,
    _compute_cost_str,
    _format_session_date,
    _get_settings,
    _handle_export_output,
    _resolve_session_id,
    _save_output,
    _show_coding_summary,
    _show_cost_line,
    async_run_command,
    console,
    err_console,
    handle_cli_error,
    track_command,
    track_command_async,
)

# Re-export code mode helpers
from vaig.cli.commands._code import (  # noqa: F401
    _async_execute_code_mode,
    _async_try_chunked_ask,
    _execute_code_mode,
    _try_chunked_ask,
)

# Re-export live mode helpers (tests import _build_gke_config, _register_live_tools)
from vaig.cli.commands.live import (  # noqa: F401
    _async_execute_live_mode,
    _async_execute_orchestrated_skill,
    _build_gke_config,
    _execute_live_mode,
    _execute_orchestrated_skill,
    _register_live_tools,
    _show_orchestrated_summary,
)

# ── Main App ──────────────────────────────────────────────────
app = typer.Typer(
    name="vaig",
    help="🤖 Vertex AI Gemini Toolkit — Multi-agent AI assistant powered by Vertex AI",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# ── Sub-apps ──────────────────────────────────────────────────
sessions_app = typer.Typer(help="Manage chat sessions")
models_app = typer.Typer(help="Manage AI models")
skills_app = typer.Typer(help="Manage skills")
mcp_app = typer.Typer(help="Manage MCP (Model Context Protocol) servers")
stats_app = typer.Typer(help="Usage telemetry and analytics")

app.add_typer(sessions_app, name="sessions")
app.add_typer(models_app, name="models")
app.add_typer(skills_app, name="skills")
app.add_typer(mcp_app, name="mcp")
app.add_typer(stats_app, name="stats")


# ── Version callback ─────────────────────────────────────────
def _version_callback(value: bool) -> None:
    if value:
        console.print(f"vaig v{__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", "-v", help="Show version", callback=_version_callback, is_eager=True),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-V", help="Enable verbose logging (INFO level)"),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option("--debug", "-d", help="Enable debug logging (DEBUG level, shows paths and full tracebacks)"),
    ] = False,
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", help="Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"),
    ] = None,
) -> None:
    """VAIG — Vertex AI Gemini Toolkit."""
    from vaig.core.log import setup_logging

    # Priority: --log-level > --debug > --verbose > default (WARNING)
    if log_level:
        level = log_level.upper()
        show_path = level == "DEBUG"
    elif debug:
        level = "DEBUG"
        show_path = True
    elif verbose:
        level = "INFO"
        show_path = False
    else:
        level = "WARNING"
        show_path = False

    # Load settings for file logging config
    from vaig.core.config import get_settings

    settings = get_settings()
    log_cfg = settings.logging

    setup_logging(
        level,
        show_path=show_path,
        file_enabled=log_cfg.file_enabled,
        file_path=log_cfg.file_path,
        file_level=log_cfg.file_level,
        file_max_bytes=log_cfg.file_max_bytes,
        file_backup_count=log_cfg.file_backup_count,
    )


# ── Register commands from modules ────────────────────────────
from vaig.cli.commands import (  # noqa: E402, I001
    ask as _ask_mod,
    chat as _chat_mod,
    discover as _discover_mod,
    doctor as _doctor_mod,
    export_cmd as _export_mod,
    feedback as _feedback_mod,
    live as _live_mod,
    mcp as _mcp_mod,
    models as _models_mod,
    optimize as _optimize_mod,
    sessions as _sessions_mod,
    skills as _skills_mod,
    stats as _stats_mod,
)
from vaig.cli.commands.cloud_cmd import cloud_app  # noqa: E402

_chat_mod.register(app)
_ask_mod.register(app)
_live_mod.register(app)
_discover_mod.register(app)
_doctor_mod.register(app)
_feedback_mod.register(app)
_optimize_mod.register(app)
_export_mod.register(app)
app.add_typer(cloud_app, name="cloud")

_sessions_mod.register(sessions_app)
_models_mod.register(models_app)
_skills_mod.register(skills_app)
_mcp_mod.register(mcp_app)
_stats_mod.register(stats_app)

"""Chat command — interactive REPL session with async support."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from vaig.cli import _helpers
from vaig.cli._helpers import (
    _banner,
    console,
    err_console,
    handle_cli_error,
    track_command,
)


def register(app: typer.Typer) -> None:
    """Register the chat command on the given Typer app."""

    @app.command()
    @track_command
    def chat(
        config: Annotated[str | None, typer.Option("--config", "-c", help="Path to config YAML")] = None,
        model: Annotated[str | None, typer.Option("--model", "-m", help="Model to use")] = None,
        skill: Annotated[str | None, typer.Option("--skill", "-s", help="Activate a skill")] = None,
        session: Annotated[str | None, typer.Option("--session", help="Load an existing session by ID")] = None,
        resume: Annotated[bool, typer.Option("--resume", "-r", help="Resume the last session")] = False,
        name: Annotated[str, typer.Option("--name", "-n", help="Name for new session")] = "chat",
        workspace: Annotated[
            Path | None,
            typer.Option("--workspace", "-w", help="Workspace root directory for /code mode"),
        ] = None,
        project: Annotated[
            str | None, typer.Option("--project", "-p", help="GCP project ID (overrides gcp.project_id only)")
        ] = None,
        location: Annotated[
            str | None, typer.Option("--location", help="GCP location (overrides config)")
        ] = None,
        gke_project: Annotated[
            str | None,
            typer.Option("--gke-project", help="GKE project ID (overrides gke.project_id; defaults to --project if unset)"),
        ] = None,
        gke_location: Annotated[
            str | None,
            typer.Option("--gke-location", help="GKE cluster location (overrides gke.location)"),
        ] = None,
    ) -> None:
        """Start an interactive chat session (REPL mode).

        Examples:
            vaig chat
            vaig chat --model gemini-2.5-flash
            vaig chat --skill rca
            vaig chat --session abc123
            vaig chat --resume
            vaig chat -w ./my-project
            vaig chat --project my-project --location europe-west1
        """
        from vaig.cli.repl import start_repl

        _banner()

        try:  # ── CLI error boundary ──
            settings = _helpers._get_settings(config)

            # Eagerly initialize the telemetry collector and wire the
            # TelemetrySubscriber so events are forwarded to the SQLite store.
            _helpers._init_telemetry(settings)
            _helpers._init_audit(settings)

            # Apply --project: mutate ONLY gcp.project_id
            # The GKE fallback chain (gke.project_id or gcp.project_id) handles single-project setups.
            if project:
                settings.gcp.project_id = project

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

            if workspace:
                resolved_ws = workspace.resolve()
                if not resolved_ws.is_dir():
                    err_console.print(f"[red]Workspace directory not found: {resolved_ws}[/red]")
                    raise typer.Exit(1)
                settings.coding.workspace_root = str(resolved_ws)

            # Resolve --resume to a session ID
            resume_session_id = session
            if resume and not session:
                from vaig.session.manager import SessionManager

                mgr = SessionManager(settings)
                last = mgr.get_last_session()
                mgr.close()
                if last:
                    resume_session_id = last["id"]
                    console.print(f"[dim]Resuming last session: {last['name']} ({last['id'][:12]})[/dim]")
                else:
                    console.print("[yellow]No previous sessions found. Starting new session.[/yellow]")

            start_repl(
                settings=settings,
                skill_name=skill,
                session_id=resume_session_id,
                session_name=name,
            )
        except typer.Exit:
            raise  # Let typer exits pass through
        except Exception as exc:  # noqa: BLE001
            handle_cli_error(exc, debug=False)


# ── Async Chat Implementation ────────────────────────────────


async def _async_chat_impl(
    *,
    config: str | None = None,
    model: str | None = None,
    skill: str | None = None,
    session: str | None = None,
    resume: bool = False,
    name: str = "chat",
    workspace: Path | None = None,
    project: str | None = None,
    location: str | None = None,
    gke_project: str | None = None,
    gke_location: str | None = None,
) -> None:
    """Async implementation of the chat command.

    Prepares settings and delegates to ``async_start_repl()`` (when
    available) for a fully async REPL loop using ``prompt_async()``.

    Falls back to ``start_repl()`` in a thread if ``async_start_repl``
    is not yet available (parallel task 4.1).
    """
    import asyncio

    _banner()
    settings = _helpers._get_settings(config)

    _helpers._init_telemetry(settings)
    _helpers._init_audit(settings)

    if project:
        settings.gcp.project_id = project
    if gke_project:
        settings.gke.project_id = gke_project
    if gke_location:
        settings.gke.location = gke_location
    if location:
        settings.gcp.location = location
    if model:
        settings.models.default = model

    if workspace:
        resolved_ws = workspace.resolve()
        if not resolved_ws.is_dir():
            err_console.print(f"[red]Workspace directory not found: {resolved_ws}[/red]")
            raise typer.Exit(1)
        settings.coding.workspace_root = str(resolved_ws)

    # Resolve --resume to a session ID
    resume_session_id = session
    if resume and not session:
        from vaig.session.manager import SessionManager

        mgr = SessionManager(settings)
        last = await mgr.async_get_last_session()
        await mgr.async_close()
        if last:
            resume_session_id = last["id"]
            console.print(f"[dim]Resuming last session: {last['name']} ({last['id'][:12]})[/dim]")
        else:
            console.print("[yellow]No previous sessions found. Starting new session.[/yellow]")

    # Try to use async_start_repl if available (Task 4.1),
    # otherwise fall back to sync start_repl in a thread.
    try:
        from vaig.cli.repl import async_start_repl

        await async_start_repl(
            settings=settings,
            skill_name=skill,
            session_id=resume_session_id,
            session_name=name,
        )
    except ImportError:
        from vaig.cli.repl import start_repl

        await asyncio.to_thread(
            start_repl,
            settings=settings,
            skill_name=skill,
            session_id=resume_session_id,
            session_name=name,
        )

"""Chat command — interactive REPL session."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from vaig.cli import _helpers
from vaig.cli._helpers import (
    _banner,
    console,
    err_console,
    track_command,
)


def register(app: typer.Typer) -> None:
    """Register the chat command on the given Typer app."""

    @app.command()
    @track_command
    def chat(
        config: Annotated[Optional[str], typer.Option("--config", "-c", help="Path to config YAML")] = None,
        model: Annotated[Optional[str], typer.Option("--model", "-m", help="Model to use")] = None,
        skill: Annotated[Optional[str], typer.Option("--skill", "-s", help="Activate a skill")] = None,
        session: Annotated[Optional[str], typer.Option("--session", help="Load an existing session by ID")] = None,
        resume: Annotated[bool, typer.Option("--resume", "-r", help="Resume the last session")] = False,
        name: Annotated[str, typer.Option("--name", "-n", help="Name for new session")] = "chat",
        workspace: Annotated[
            Optional[Path],
            typer.Option("--workspace", "-w", help="Workspace root directory for /code mode"),
        ] = None,
        project: Annotated[
            Optional[str], typer.Option("--project", "-p", help="GCP project ID (overrides gcp.project_id and gke.project_id)")
        ] = None,
        location: Annotated[
            Optional[str], typer.Option("--location", help="GCP location (overrides config)")
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
        settings = _helpers._get_settings(config)

        # Eagerly initialize the telemetry collector so downstream code
        # (agents, cost_tracker, session) uses the pre-warmed singleton
        # instead of falling back to get_settings().
        try:
            from vaig.core.telemetry import get_telemetry_collector

            get_telemetry_collector(settings)
        except Exception:  # noqa: BLE001
            pass

        # Apply --project: mutate gcp.project_id AND gke.project_id
        if project:
            settings.gcp.project_id = project
            settings.gke.project_id = project

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

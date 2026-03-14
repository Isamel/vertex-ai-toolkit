"""Export command — re-export a past session."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from vaig import __version__
from vaig.cli import _helpers
from vaig.cli._helpers import _save_output, console, err_console, track_command


def register(app: typer.Typer) -> None:
    """Register the export command on the given Typer app."""

    @app.command(name="export")
    @track_command
    def export_session(
        session_id: Annotated[str, typer.Argument(help="Session ID to export")],
        format_: Annotated[str, typer.Option("--format", "-f", help="Export format: json, md, html")] = "md",
        output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Save to file (default: stdout)")] = None,
        config: Annotated[Optional[str], typer.Option("--config", "-c", help="Path to config YAML")] = None,
    ) -> None:
        """Export a past session to JSON, Markdown, or HTML."""
        from vaig.cli.export import ExportMetadata, ExportPayload, format_export
        from vaig.session.manager import SessionManager

        settings = _helpers._get_settings(config)
        manager = SessionManager(settings)

        session_data = manager._store.get_session(session_id)  # noqa: SLF001
        if not session_data:
            err_console.print(f"[red]Session not found: {session_id}[/red]")
            manager.close()
            raise typer.Exit(1)

        messages = manager._store.get_messages(session_id)  # noqa: SLF001
        context_files_rows = manager._store.get_context_files(session_id)  # noqa: SLF001
        manager.close()

        # Build question from first user message, response from last assistant message
        user_messages = [m for m in messages if m["role"] == "user"]
        assistant_messages = [m for m in messages if m["role"] in ("assistant", "model")]

        question_text = user_messages[0]["content"] if user_messages else "(no question)"
        response_text = assistant_messages[-1]["content"] if assistant_messages else "(no response)"

        # Sum up tokens from all messages
        total_tokens = sum(m.get("token_count", 0) for m in messages)

        meta = ExportMetadata(
            model=session_data.get("model", "unknown"),
            skill=session_data.get("skill"),
            timestamp=session_data.get("created_at", ""),
            tokens={"total_tokens": total_tokens},
            cost=None,
            vaig_version=__version__,
        )

        payload = ExportPayload(
            question=question_text,
            response=response_text,
            metadata=meta,
            context_files=[cf["file_path"] for cf in context_files_rows],
        )

        content = format_export(payload, format_)
        if output:
            _save_output(output, content)
        else:
            console.print(content)

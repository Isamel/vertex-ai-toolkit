"""Sessions sub-commands — list, show, delete, rename, search."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.panel import Panel
from rich.table import Table

from vaig.cli import _helpers
from vaig.cli._helpers import (
    _format_session_date,
    _resolve_session_id,
    console,
    err_console,
)


def register(sessions_app: typer.Typer) -> None:
    """Register session sub-commands on the given Typer sub-app."""

    @sessions_app.command("list")
    def sessions_list(
        config: Annotated[str | None, typer.Option("--config", "-c")] = None,
        limit: Annotated[int, typer.Option("--limit", "-n", help="Max sessions to show")] = 20,
    ) -> None:
        """List recent chat sessions."""
        settings = _helpers._get_settings(config)

        from vaig.session.manager import SessionManager

        manager = SessionManager(settings)
        sessions = manager.list_sessions(limit=limit)

        if not sessions:
            console.print("[yellow]No sessions found.[/yellow]")
            return

        table = Table(title="Sessions", show_lines=False)
        table.add_column("ID", style="cyan", no_wrap=True, max_width=12)
        table.add_column("Name", style="bold")
        table.add_column("Model", style="magenta")
        table.add_column("Skill", style="green")
        table.add_column("Msgs", style="yellow", justify="right")
        table.add_column("Updated", style="dim")

        for s in sessions:
            if not isinstance(s, dict):
                continue
            table.add_row(
                s.get("id", "?")[:12],
                s.get("name", "—"),
                s.get("model", "—"),
                s.get("skill", "—") or "—",
                str(s.get("message_count", 0)),
                _format_session_date(s.get("updated_at", "")),
            )

        console.print(table)
        manager.close()

    @sessions_app.command("delete")
    def sessions_delete(
        session_id: Annotated[str, typer.Argument(help="Session ID to delete")],
        config: Annotated[str | None, typer.Option("--config", "-c")] = None,
        force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation")] = False,
    ) -> None:
        """Delete a chat session."""
        settings = _helpers._get_settings(config)

        from vaig.session.manager import SessionManager

        if not force:
            confirmed = typer.confirm(f"Delete session {session_id}?")
            if not confirmed:
                raise typer.Exit

        manager = SessionManager(settings)
        if manager.delete_session(session_id):
            console.print(f"[green]✓ Deleted session: {session_id}[/green]")
        else:
            err_console.print(f"[red]Session not found: {session_id}[/red]")
        manager.close()

    @sessions_app.command("rename")
    def sessions_rename(
        session_id: Annotated[str, typer.Argument(help="Session ID (or prefix) to rename")],
        new_name: Annotated[str, typer.Argument(help="New name for the session")],
        config: Annotated[str | None, typer.Option("--config", "-c")] = None,
    ) -> None:
        """Rename a chat session."""
        settings = _helpers._get_settings(config)

        from vaig.session.manager import SessionManager

        manager = SessionManager(settings)
        session_id = _resolve_session_id(manager, session_id)
        if manager.rename_session(session_id, new_name):
            console.print(f"[green]Renamed session {session_id[:12]} to: {new_name}[/green]")
        else:
            err_console.print(f"[red]Session not found: {session_id}[/red]")
        manager.close()

    @sessions_app.command("search")
    def sessions_search(
        query: Annotated[str, typer.Argument(help="Search term (matches session name and message content)")],
        config: Annotated[str | None, typer.Option("--config", "-c")] = None,
    ) -> None:
        """Search sessions by name or message content."""
        settings = _helpers._get_settings(config)

        from vaig.session.manager import SessionManager

        manager = SessionManager(settings)
        sessions = manager.search_sessions(query)

        if not sessions:
            console.print(f"[yellow]No sessions matching '{query}'.[/yellow]")
            manager.close()
            return

        table = Table(title=f"Search: '{query}'", show_lines=False)
        table.add_column("ID", style="cyan", no_wrap=True, max_width=12)
        table.add_column("Name", style="bold")
        table.add_column("Model", style="magenta")
        table.add_column("Msgs", style="yellow", justify="right")
        table.add_column("Updated", style="dim")

        for s in sessions:
            if not isinstance(s, dict):
                continue
            table.add_row(
                s.get("id", "?")[:12],
                s.get("name", "—"),
                s.get("model", "—"),
                str(s.get("message_count", 0)),
                _format_session_date(s.get("updated_at", "")),
            )

        console.print(table)
        manager.close()

    @sessions_app.command("show")
    def sessions_show(
        session_id: Annotated[str, typer.Argument(help="Session ID (or prefix) to show")],
        config: Annotated[str | None, typer.Option("--config", "-c")] = None,
        messages: Annotated[int, typer.Option("--messages", "-m", help="Number of recent messages to show")] = 10,
    ) -> None:
        """Show detailed information about a session."""
        settings = _helpers._get_settings(config)

        from vaig.session.manager import SessionManager

        manager = SessionManager(settings)
        session_id = _resolve_session_id(manager, session_id)

        session = manager._store.get_session(session_id)
        if not session:
            err_console.print(f"[red]Session not found: {session_id}[/red]")
            manager.close()
            return

        msgs = manager._store.get_messages(session_id)
        context_files = manager._store.get_context_files(session_id)

        # Session info panel
        info_lines = [
            f"[cyan]ID:[/cyan]      {session['id']}",
            f"[cyan]Name:[/cyan]    {session['name']}",
            f"[cyan]Model:[/cyan]   {session['model']}",
            f"[cyan]Skill:[/cyan]   {session.get('skill') or '—'}",
            f"[cyan]Created:[/cyan] {_format_session_date(session['created_at'])}",
            f"[cyan]Updated:[/cyan] {_format_session_date(session['updated_at'])}",
            f"[cyan]Messages:[/cyan] {len(msgs)}",
        ]
        if context_files:
            info_lines.append(f"[cyan]Context files:[/cyan] {len(context_files)}")
        console.print(Panel("\n".join(info_lines), title="Session Details", border_style="cyan"))

        # Context files
        if context_files:
            ft = Table(title="Context Files", show_lines=False)
            ft.add_column("Path", style="green")
            ft.add_column("Type", style="dim")
            ft.add_column("Size", style="yellow", justify="right")
            for cf in context_files:
                size = cf.get("size_bytes", 0)
                size_str = f"{size:,}" if size else "—"
                ft.add_row(cf["file_path"], cf["file_type"], size_str)
            console.print(ft)

        # Recent messages
        recent = msgs[-messages:] if len(msgs) > messages else msgs
        if recent:
            mt = Table(title=f"Messages (last {len(recent)} of {len(msgs)})", show_lines=True)
            mt.add_column("Role", style="bold", max_width=8)
            mt.add_column("Content", max_width=100)
            mt.add_column("Time", style="dim", max_width=20)
            for m in recent:
                content = m["content"]
                if len(content) > 200:
                    content = content[:200] + "..."
                role_style = "cyan" if m["role"] == "user" else "green"
                mt.add_row(
                    f"[{role_style}]{m['role']}[/{role_style}]",
                    content,
                    _format_session_date(m.get("created_at", "")),
                )
            console.print(mt)

        manager.close()

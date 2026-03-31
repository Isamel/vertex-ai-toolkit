"""Web command — start the VAIG web server.

Launches a Uvicorn server serving the FastAPI web interface.
Requires web extras: ``pip install vertex-ai-toolkit[web]``
"""

from __future__ import annotations

from typing import Annotated

import typer

from vaig.cli._helpers import (
    console,
    err_console,
)


def register(app: typer.Typer) -> None:
    """Register the web command on the given Typer app."""

    @app.command()
    def web(
        host: Annotated[
            str,
            typer.Option("--host", "-h", help="Bind address"),
        ] = "0.0.0.0",  # noqa: S104
        port: Annotated[
            int,
            typer.Option("--port", "-p", help="Bind port", envvar="PORT"),
        ] = 8080,
        reload: Annotated[
            bool,
            typer.Option("--reload", help="Enable auto-reload for development"),
        ] = False,
    ) -> None:
        """Start the VAIG web server.

        Launches a FastAPI web interface for VAIG, serving
        the same toolkit features available in the CLI.

        Requires: pip install vertex-ai-toolkit[web]

        Examples:
            vaig web
            vaig web --port 9090
            vaig web --reload
        """
        try:
            import uvicorn
        except ImportError:
            err_console.print(
                "[red]Web extras not installed.[/red]\n"
                "Run: [bold]pip install vertex-ai-toolkit\\[web][/bold]"
            )
            raise typer.Exit(code=1) from None

        console.print(f"[bold cyan]vaig web[/bold cyan] — starting on {host}:{port}")

        uvicorn.run(
            "vaig.web.app:create_app",
            host=host,
            port=port,
            reload=reload,
            factory=True,
            log_level="info",
        )

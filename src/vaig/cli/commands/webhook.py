"""Webhook command — start the VAIG webhook server.

Launches a Uvicorn server running the FastAPI webhook endpoint that
receives Datadog alert webhooks and triggers vaig health analyses.
Requires webhook extras: ``pip install vertex-ai-toolkit[webhook]``
"""

from __future__ import annotations

from typing import Annotated

import typer

from vaig.cli._helpers import (
    console,
    err_console,
)


def register(app: typer.Typer) -> None:
    """Register the webhook-server command on the given Typer app."""

    @app.command("webhook-server")
    def webhook_server(
        host: Annotated[
            str,
            typer.Option("--host", "-h", help="Bind address"),
        ] = "0.0.0.0",  # noqa: S104
        port: Annotated[
            int,
            typer.Option("--port", "-p", help="Bind port", envvar="PORT"),
        ] = 8080,
        hmac_secret: Annotated[
            str,
            typer.Option(
                "--hmac-secret",
                help="Datadog webhook HMAC secret for signature verification",
                envvar="VAIG_WEBHOOK_SERVER__HMAC_SECRET",
            ),
        ] = "",
        max_analyses: Annotated[
            int,
            typer.Option(
                "--max-analyses",
                help="Maximum analyses per UTC day (cost protection)",
            ),
        ] = 50,
        dedup_cooldown: Annotated[
            int,
            typer.Option(
                "--dedup-cooldown",
                help="Seconds before re-analyzing the same alert",
            ),
        ] = 300,
        reload: Annotated[
            bool,
            typer.Option("--reload", help="Enable auto-reload for development"),
        ] = False,
    ) -> None:
        """Start the VAIG webhook server for Datadog alerts.

        Receives Datadog alert webhooks (POST /webhook/datadog), runs
        vaig health analysis on affected services, and dispatches results
        to PagerDuty + Google Chat.

        Requires: pip install vertex-ai-toolkit[webhook]

        Examples:
            vaig webhook-server
            vaig webhook-server --port 9090
            vaig webhook-server --hmac-secret my-secret
            vaig webhook-server --max-analyses 100 --dedup-cooldown 600
        """
        try:
            import uvicorn
        except ImportError:
            err_console.print(
                "[red]Webhook extras not installed.[/red]\n"
                "Run: [bold]pip install vertex-ai-toolkit\\[webhook][/bold]"
            )
            raise typer.Exit(code=1) from None

        # Load config to merge CLI args with config file values
        from vaig.core.config import get_settings

        settings = get_settings()
        ws_cfg = settings.webhook_server

        # CLI args take priority over config file, but use config defaults
        # when CLI args match the function defaults
        effective_host = host if host != "0.0.0.0" else ws_cfg.host  # noqa: S104
        effective_port = port if port != 8080 else ws_cfg.port
        effective_secret = hmac_secret or ws_cfg.hmac_secret
        effective_max = max_analyses if max_analyses != 50 else ws_cfg.max_analyses_per_day
        effective_cooldown = dedup_cooldown if dedup_cooldown != 300 else ws_cfg.dedup_cooldown_seconds

        console.print(
            f"[bold cyan]vaig webhook-server[/bold cyan] — "
            f"starting on {effective_host}:{effective_port}"
        )
        if effective_secret:
            console.print("  [green]✓[/green] HMAC signature verification enabled")
        else:
            console.print("  [yellow]⚠[/yellow] HMAC verification disabled (no secret configured)")
        console.print(f"  Budget: {effective_max} analyses/day, dedup cooldown: {effective_cooldown}s")

        # Set environment variables so the factory function can pick them up
        import os

        os.environ["_VAIG_WEBHOOK_HMAC_SECRET"] = effective_secret
        os.environ["_VAIG_WEBHOOK_MAX_ANALYSES"] = str(effective_max)
        os.environ["_VAIG_WEBHOOK_DEDUP_COOLDOWN"] = str(effective_cooldown)
        os.environ["_VAIG_WEBHOOK_ANALYSIS_TIMEOUT"] = str(ws_cfg.analysis_timeout_seconds)

        uvicorn.run(
            "vaig.integrations.webhook_server:create_webhook_app",
            host=effective_host,
            port=effective_port,
            reload=reload,
            factory=True,
            log_level="info",
        )

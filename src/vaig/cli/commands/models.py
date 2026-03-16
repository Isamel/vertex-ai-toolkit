"""Models sub-commands — list available models."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.table import Table

from vaig.cli import _helpers
from vaig.cli._helpers import console


def register(models_app: typer.Typer) -> None:
    """Register model sub-commands on the given Typer sub-app."""

    @models_app.command("list")
    def models_list(
        config: Annotated[str | None, typer.Option("--config", "-c")] = None,
    ) -> None:
        """List available models."""
        settings = _helpers._get_settings(config)

        from vaig.core.client import GeminiClient

        client = GeminiClient(settings)
        models = client.list_available_models()

        if not models:
            console.print("[yellow]No models configured. Add models to your config YAML.[/yellow]")
            console.print(f"[dim]Default model: {settings.models.default}[/dim]")
            return

        table = Table(title="🤖 Available Models", show_lines=False)
        table.add_column("Model ID", style="cyan")
        table.add_column("Description", style="dim")
        table.add_column("Default", justify="center")

        for m in models:
            if not isinstance(m, dict):
                continue
            is_default = "✓" if m.get("id") == settings.models.default else ""
            table.add_row(m.get("id", "?"), m.get("description", ""), is_default)

        console.print(table)
        console.print(f"\n[dim]Default: {settings.models.default} | Fallback: {settings.models.fallback}[/dim]")

"""Auth commands — platform login, logout, whoami, status.

Provides the ``vaig login``, ``vaig logout``, ``vaig whoami``, and
``vaig status`` commands.  All commands require ``platform.enabled: true``
in the configuration — when disabled they print an informative message
and exit with code 1.

Uses the ``register(app)`` pattern consistent with all other command
modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import typer

from vaig.cli._helpers import (
    _get_settings,
    console,
    err_console,
    track_command,
)

if TYPE_CHECKING:
    from vaig.core.config import Settings
    from vaig.core.platform_auth import PlatformAuthManager

_PLATFORM_DISABLED_MSG = (
    "[yellow]Platform mode is not enabled. "
    "Set platform.enabled to true in your config.[/yellow]"
)


def _require_platform_enabled(config: str | None = None) -> Settings:
    """Load settings and validate platform is enabled.

    Returns the ``Settings`` object when enabled, otherwise prints
    an error and raises ``typer.Exit(1)``.
    """
    settings = _get_settings(config)
    if not settings.platform.enabled:
        err_console.print(_PLATFORM_DISABLED_MSG)
        raise typer.Exit(code=1)
    return settings


def _get_auth_manager(settings: Settings) -> PlatformAuthManager:
    """Create a ``PlatformAuthManager`` from settings.

    Lazy-imports to avoid heavy imports on ``--help``.
    """
    from vaig.core.platform_auth import PlatformAuthManager as PAM  # noqa: N817

    return PAM(
        backend_url=settings.platform.backend_url,
        org_id=settings.platform.org_id,
    )


def register(app: typer.Typer) -> None:
    """Register platform auth commands on the given Typer app."""

    @app.command()
    @track_command
    def login(
        config: Annotated[
            str | None,
            typer.Option("--config", "-c", help="Path to config YAML"),
        ] = None,
        force: Annotated[
            bool,
            typer.Option("--force", help="Re-authenticate even if already logged in"),
        ] = False,
    ) -> None:
        """Authenticate with the platform backend (OAuth PKCE flow).

        Opens a browser window for Google OAuth consent.  Stores tokens
        locally at ``~/.vaig/credentials.json``.

        Examples:
            vaig login
            vaig login --force
        """
        settings = _require_platform_enabled(config)
        manager = _get_auth_manager(settings)

        console.print("[dim]Starting platform login…[/dim]")
        result = manager.login(force=force)

        if result.success:
            if result.error and "Already authenticated" in result.error:
                console.print(
                    f"[green]✓ Already authenticated as {result.user_email}[/green]\n"
                    "[dim]Use --force to re-authenticate.[/dim]"
                )
            else:
                console.print(
                    f"[green]✓ Authenticated as {result.user_email}[/green]\n"
                    f"  Organization: {result.org_id}\n"
                    f"  Role: {result.role}"
                )
        else:
            err_console.print(f"[red]✗ Login failed: {result.error}[/red]")
            raise typer.Exit(code=1)

    @app.command()
    @track_command
    def logout(
        config: Annotated[
            str | None,
            typer.Option("--config", "-c", help="Path to config YAML"),
        ] = None,
    ) -> None:
        """Log out from the platform and clear local credentials.

        Examples:
            vaig logout
        """
        settings = _require_platform_enabled(config)
        manager = _get_auth_manager(settings)

        if not manager.is_authenticated():
            console.print("[dim]Not currently authenticated.[/dim]")
            return

        manager.logout()
        console.print("[green]✓ Logged out successfully.[/green]")

    @app.command()
    @track_command
    def whoami(
        config: Annotated[
            str | None,
            typer.Option("--config", "-c", help="Path to config YAML"),
        ] = None,
    ) -> None:
        """Show the currently authenticated platform user.

        Examples:
            vaig whoami
        """
        settings = _require_platform_enabled(config)
        manager = _get_auth_manager(settings)

        if not manager.is_authenticated():
            err_console.print("[red]Not authenticated. Run: vaig login[/red]")
            raise typer.Exit(code=1)

        info = manager.get_user_info()
        if info is None:
            err_console.print("[red]Failed to read user info from credentials.[/red]")
            raise typer.Exit(code=1)

        console.print(
            f"[bold]Email:[/bold]  {info.get('email', '—')}\n"
            f"[bold]Org:[/bold]    {info.get('org_id', '—')}\n"
            f"[bold]Role:[/bold]   {info.get('role', '—')}\n"
            f"[bold]CLI ID:[/bold] {info.get('cli_id', '—')}"
        )

    @app.command()
    @track_command
    def status(
        config: Annotated[
            str | None,
            typer.Option("--config", "-c", help="Path to config YAML"),
        ] = None,
    ) -> None:
        """Show platform registration status and config policy version.

        Examples:
            vaig status
        """
        settings = _require_platform_enabled(config)
        manager = _get_auth_manager(settings)

        if not manager.is_authenticated():
            err_console.print("[red]Not authenticated. Run: vaig login[/red]")
            raise typer.Exit(code=1)

        info = manager.get_user_info()
        if info is None:
            err_console.print("[red]Failed to read user info from credentials.[/red]")
            raise typer.Exit(code=1)

        # Fetch enforced config to show policy version
        enforced = manager.get_enforced_config()
        policy_status = "active" if enforced else "no enforced fields"

        console.print(
            f"[bold]Status:[/bold]         authenticated\n"
            f"[bold]Email:[/bold]          {info.get('email', '—')}\n"
            f"[bold]Organization:[/bold]   {info.get('org_id', '—')}\n"
            f"[bold]Role:[/bold]           {info.get('role', '—')}\n"
            f"[bold]CLI ID:[/bold]         {info.get('cli_id', '—')}\n"
            f"[bold]Config policy:[/bold]  {policy_status}"
        )

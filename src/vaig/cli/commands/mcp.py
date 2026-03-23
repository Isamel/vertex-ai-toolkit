"""MCP sub-commands — Model Context Protocol server management."""

from __future__ import annotations

from typing import Annotated, Any

import typer
from rich.table import Table

from vaig.cli import _helpers
from vaig.cli._helpers import console, err_console


def register(mcp_app: typer.Typer) -> None:
    """Register MCP sub-commands on the given Typer sub-app."""

    @mcp_app.command("list-servers")
    def mcp_list_servers(
        config: Annotated[str | None, typer.Option("--config", "-c")] = None,
    ) -> None:
        """List configured MCP servers."""
        from vaig.tools.mcp_bridge import is_mcp_available

        settings = _helpers._get_settings(config)

        if not is_mcp_available():
            err_console.print(
                "[bold red]MCP SDK not installed.[/bold red]\n"
                "[yellow]Install with:[/yellow]  pip install mcp"
            )
            raise typer.Exit(1)

        mcp_cfg = settings.mcp

        if not mcp_cfg.enabled:
            console.print(
                "[yellow]MCP integration is disabled.[/yellow]\n"
                "[dim]Set 'mcp.enabled: true' in your config to enable it.[/dim]"
            )
            return

        if not mcp_cfg.servers:
            console.print("[yellow]No MCP servers configured.[/yellow]")
            console.print("[dim]Add servers under 'mcp.servers' in your config YAML.[/dim]")
            return

        table = Table(title="MCP Servers", show_lines=False)
        table.add_column("Name", style="cyan")
        table.add_column("Command", style="bold")
        table.add_column("Args", style="dim")
        table.add_column("Description")

        for srv in mcp_cfg.servers:
            table.add_row(
                srv.name,
                srv.command,
                " ".join(srv.args) if srv.args else "—",
                srv.description or "—",
            )

        console.print(table)

    @mcp_app.command("discover")
    def mcp_discover(
        server_name: Annotated[str, typer.Argument(help="Name of configured MCP server to discover tools from")],
        config: Annotated[str | None, typer.Option("--config", "-c")] = None,
    ) -> None:
        """Discover available tools from an MCP server.

        Examples:
            vaig mcp discover my-server
        """
        import asyncio

        from vaig.tools.mcp_bridge import discover_tools, is_mcp_available

        if not is_mcp_available():
            err_console.print(
                "[bold red]MCP SDK not installed.[/bold red]\n"
                "[yellow]Install with:[/yellow]  pip install mcp"
            )
            raise typer.Exit(1)

        settings = _helpers._get_settings(config)
        mcp_cfg = settings.mcp

        if not mcp_cfg.enabled:
            err_console.print("[red]MCP integration is disabled. Set 'mcp.enabled: true' in your config.[/red]")
            raise typer.Exit(1)

        srv = next((s for s in mcp_cfg.servers if s.name == server_name), None)
        if not srv:
            err_console.print(f"[red]MCP server not found: {server_name}[/red]")
            available = [s.name for s in mcp_cfg.servers]
            if available:
                err_console.print(f"[dim]Available: {', '.join(available)}[/dim]")
            raise typer.Exit(1)

        with console.status(f"[bold cyan]Connecting to {server_name}...[/bold cyan]"):
            try:
                tools = asyncio.run(
                    discover_tools(
                        command=srv.command,
                        args=srv.args or None,
                        env=srv.env or None,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                err_console.print(f"[red]Failed to connect to {server_name}: {exc}[/red]")
                raise typer.Exit(1) from None

        if not tools:
            console.print(f"[yellow]{server_name} exposes no tools.[/yellow]")
            return

        table = Table(title=f"Tools from {server_name}", show_lines=True)
        table.add_column("Tool", style="cyan")
        table.add_column("Description")
        table.add_column("Parameters", style="dim")

        for tool in tools:
            schema = tool.inputSchema or {}
            props = schema.get("properties", {})
            required = set(schema.get("required", []))
            param_strs: list[str] = []
            for pname, pschema in props.items():
                req_mark = "*" if pname in required else ""
                ptype = pschema.get("type", "?")
                param_strs.append(f"{pname}{req_mark}: {ptype}")
            table.add_row(
                tool.name,
                tool.description or "—",
                "\n".join(param_strs) if param_strs else "—",
            )

        console.print(table)
        console.print(f"\n[dim]{len(tools)} tools discovered from {server_name}[/dim]")

    @mcp_app.command("call")
    def mcp_call(
        server_name: Annotated[str, typer.Argument(help="MCP server name")],
        tool_name: Annotated[str, typer.Argument(help="Tool name to call")],
        args_json: Annotated[str | None, typer.Argument(help="JSON arguments (e.g. '{\"path\": \"/tmp\"}')")] = None,
        config: Annotated[str | None, typer.Option("--config", "-c")] = None,
    ) -> None:
        """Call a specific tool on an MCP server.

        Examples:
            vaig mcp call my-server read_file '{"path": "/etc/hostname"}'
            vaig mcp call my-server list_files
        """
        import asyncio
        import json

        from vaig.tools.mcp_bridge import call_mcp_tool, is_mcp_available

        if not is_mcp_available():
            err_console.print(
                "[bold red]MCP SDK not installed.[/bold red]\n"
                "[yellow]Install with:[/yellow]  pip install mcp"
            )
            raise typer.Exit(1)

        settings = _helpers._get_settings(config)
        mcp_cfg = settings.mcp

        if not mcp_cfg.enabled:
            err_console.print("[red]MCP integration is disabled. Set 'mcp.enabled: true' in your config.[/red]")
            raise typer.Exit(1)

        srv = next((s for s in mcp_cfg.servers if s.name == server_name), None)
        if not srv:
            err_console.print(f"[red]MCP server not found: {server_name}[/red]")
            available = [s.name for s in mcp_cfg.servers]
            if available:
                err_console.print(f"[dim]Available: {', '.join(available)}[/dim]")
            raise typer.Exit(1)

        # Parse arguments
        arguments: dict[str, Any] = {}
        if args_json:
            try:
                arguments = json.loads(args_json)
            except json.JSONDecodeError as exc:
                err_console.print(f"[red]Invalid JSON arguments: {exc}[/red]")
                raise typer.Exit(1) from None

        with console.status(f"[bold cyan]Calling {tool_name} on {server_name}...[/bold cyan]"):
            result = asyncio.run(
                call_mcp_tool(
                    command=srv.command,
                    tool_name=tool_name,
                    arguments=arguments,
                    args=srv.args or None,
                    env=srv.env or None,
                )
            )

        if result.error:
            err_console.print(f"[red]Tool error:[/red] {result.output}")
            raise typer.Exit(1)

        console.print(result.output)

"""Skills sub-commands — list, info, create."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.panel import Panel
from rich.table import Table

from vaig.cli import _helpers
from vaig.cli._helpers import console, err_console


def register(skills_app: typer.Typer) -> None:
    """Register skills sub-commands on the given Typer sub-app."""

    @skills_app.command("list")
    def skills_list(
        config: Annotated[Optional[str], typer.Option("--config", "-c")] = None,
    ) -> None:
        """List available skills."""
        settings = _helpers._get_settings(config)

        from vaig.skills.registry import SkillRegistry

        registry = SkillRegistry(settings)
        skills = registry.list_skills()

        if not skills:
            console.print("[yellow]No skills loaded.[/yellow]")
            return

        table = Table(title="🛠️  Skills", show_lines=False)
        table.add_column("Name", style="cyan")
        table.add_column("Display Name", style="bold")
        table.add_column("Description")
        table.add_column("Phases", style="green")
        table.add_column("Model", style="magenta")

        for meta in skills:
            phases = ", ".join(p.value for p in meta.supported_phases)
            table.add_row(meta.name, meta.display_name, meta.description, phases, meta.recommended_model)

        console.print(table)

    @skills_app.command("info")
    def skills_info(
        skill_name: Annotated[str, typer.Argument(help="Skill name to inspect")],
        config: Annotated[Optional[str], typer.Option("--config", "-c")] = None,
    ) -> None:
        """Show detailed info about a skill."""
        settings = _helpers._get_settings(config)

        from vaig.skills.registry import SkillRegistry

        registry = SkillRegistry(settings)
        skill = registry.get(skill_name)

        if not skill:
            err_console.print(f"[red]Skill not found: {skill_name}[/red]")
            err_console.print(f"[dim]Available: {', '.join(registry.list_names())}[/dim]")
            raise typer.Exit(1)

        meta = skill.get_metadata()
        agents = skill.get_agents_config()

        console.print(
            Panel(
                f"[bold]{meta.display_name}[/bold] ({meta.name} v{meta.version})\n\n"
                f"{meta.description}\n\n"
                f"[dim]Tags: {', '.join(meta.tags)}[/dim]\n"
                f"[dim]Phases: {', '.join(p.value for p in meta.supported_phases)}[/dim]\n"
                f"[dim]Recommended model: {meta.recommended_model}[/dim]",
                title="🛠️  Skill Info",
                border_style="bright_blue",
            )
        )

        # Show agents
        if agents:
            agent_table = Table(title="Agents", show_lines=False)
            agent_table.add_column("Name", style="cyan")
            agent_table.add_column("Role", style="bold")
            agent_table.add_column("Model", style="magenta")

            for a in agents:
                agent_table.add_row(a["name"], a["role"], a.get("model", "default"))

            console.print(agent_table)

    @skills_app.command("create")
    def skills_create(
        name: Annotated[str, typer.Argument(help="Skill name (kebab-case, e.g. 'my-analyzer')")],
        description: Annotated[str, typer.Option("--description", "-d")] = "A custom skill",
        tags: Annotated[Optional[str], typer.Option("--tags", "-t", help="Comma-separated tags")] = None,
        output_dir: Annotated[Optional[str], typer.Option("--output", "-o", help="Target directory (default: custom_dir from config)")] = None,
        config: Annotated[Optional[str], typer.Option("--config", "-c")] = None,
    ) -> None:
        """Scaffold a new custom skill with boilerplate files."""
        from vaig.skills.scaffold import scaffold_skill

        settings = _helpers._get_settings(config)

        # Determine target directory
        if output_dir:
            target = Path(output_dir).expanduser().resolve()
        elif settings.skills.custom_dir:
            target = Path(settings.skills.custom_dir).expanduser().resolve()
        else:
            target = Path.cwd() / "skills"
            console.print(
                f"[yellow]No custom_dir configured. Scaffolding to: {target}[/yellow]\n"
                "[dim]Set 'skills.custom_dir' in your config to auto-load custom skills.[/dim]"
            )

        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

        try:
            skill_dir = scaffold_skill(name, target, description=description, tags=tag_list)
        except FileExistsError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from None

        console.print(f"[green]Skill scaffolded at:[/green] {skill_dir}")
        console.print("\n[bold]Created files:[/bold]")
        console.print(f"  {skill_dir / '__init__.py'}")
        console.print(f"  {skill_dir / 'skill.py'}")
        console.print(f"  {skill_dir / 'prompts.py'}")
        console.print(f"\n[dim]Edit skill.py and prompts.py to implement your skill logic.[/dim]")

"""Skills sub-commands — list, info, create."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.panel import Panel
from rich.table import Table

from vaig.cli import _helpers
from vaig.cli._helpers import console, err_console

if TYPE_CHECKING:
    from vaig.skills._presets import SkillPreset

_VALID_PRESETS = ("analysis", "live-tools", "coding", "custom")


def register(skills_app: typer.Typer) -> None:
    """Register skills sub-commands on the given Typer sub-app."""

    @skills_app.command("list")
    def skills_list(
        config: Annotated[str | None, typer.Option("--config", "-c")] = None,
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
        config: Annotated[str | None, typer.Option("--config", "-c")] = None,
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
        tags: Annotated[str | None, typer.Option("--tags", "-t", help="Comma-separated tags")] = None,
        output_dir: Annotated[str | None, typer.Option("--output", "-o", help="Target directory (default: custom_dir from config)")] = None,
        preset: Annotated[str | None, typer.Option("--preset", "-p", help="Skill preset: analysis, live-tools, coding, custom")] = None,
        interactive: Annotated[bool, typer.Option("--interactive", "-i", help="Interactive mode — prompts for options")] = False,
        config: Annotated[str | None, typer.Option("--config", "-c")] = None,
    ) -> None:
        """Scaffold a new custom skill with boilerplate files."""
        from vaig.skills.scaffold import scaffold_skill

        # Mutual exclusion: --preset + --interactive
        if preset is not None and interactive:
            raise typer.BadParameter("--preset and --interactive are mutually exclusive. Use one or the other.")

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

        # Resolve preset
        resolved_preset = None
        if preset is not None:
            if preset not in _VALID_PRESETS:
                raise typer.BadParameter(f"Invalid preset {preset!r}. Choose from: {', '.join(_VALID_PRESETS)}")
            if preset == "custom":
                raise typer.BadParameter("The 'custom' preset requires --interactive mode.")
            from vaig.skills._presets import get_preset
            resolved_preset = get_preset(preset)
        elif interactive:
            resolved_preset = _build_interactive_preset()

        try:
            skill_dir = scaffold_skill(name, target, description=description, tags=tag_list, preset=resolved_preset)
        except FileExistsError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from None

        console.print(f"[green]Skill scaffolded at:[/green] {skill_dir}")
        console.print("\n[bold]Created files:[/bold]")
        for f in sorted(skill_dir.iterdir()):
            if f.is_file():
                console.print(f"  {f}")
        console.print("\n[dim]Edit skill.py and prompts.py to implement your skill logic.[/dim]")


def _build_interactive_preset() -> SkillPreset:
    """Prompt the user interactively to build a custom SkillPreset."""
    from vaig.skills._presets import SkillPreset
    from vaig.skills.base import SkillPhase

    all_phases = list(SkillPhase)
    console.print("\n[bold]Available phases:[/bold]")
    for i, phase in enumerate(all_phases, 1):
        console.print(f"  {i}. {phase.value}")

    phase_input = typer.prompt(
        "Select phases (comma-separated numbers, e.g. 1,3,5)",
        default="1,3,5",
    )
    selected_indices = [int(x.strip()) for x in phase_input.split(",") if x.strip().isdigit()]
    phases = [all_phases[i - 1] for i in selected_indices if 1 <= i <= len(all_phases)]
    if not phases:
        phases = [SkillPhase.ANALYZE, SkillPhase.EXECUTE, SkillPhase.REPORT]

    agent_count = int(typer.prompt("Number of agents", default="1"))
    agent_roles: list[str] = []
    if agent_count > 1:
        roles_input = typer.prompt(
            "Agent roles (comma-separated)",
            default=",".join(f"agent-{i}" for i in range(1, agent_count + 1)),
        )
        agent_roles = [r.strip() for r in roles_input.split(",") if r.strip()]
    else:
        agent_roles = ["analyst"]

    generate_schema = typer.confirm("Generate schema.py?", default=False)
    requires_live_tools = typer.confirm("Requires live tools?", default=False)

    return SkillPreset(
        name="custom",
        phases=phases,
        agent_count=agent_count,
        agent_roles=agent_roles,
        generate_schema=generate_schema,
        requires_live_tools=requires_live_tools,
    )

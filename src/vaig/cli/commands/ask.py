"""Ask command — single-shot question with async support."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.markdown import Markdown

from vaig.cli import _helpers
from vaig.cli._completions import complete_namespace
from vaig.cli._helpers import (
    _apply_subcommand_log_flags,
    _compute_cost_str,
    _handle_export_output,
    _show_cost_line,
    console,
    err_console,
    handle_cli_error,
    track_command,
)
from vaig.cli.commands._code import _execute_code_mode, _try_chunked_ask
from vaig.cli.commands.live import _build_gke_config, _execute_live_mode


def register(app: typer.Typer) -> None:
    """Register the ask command on the given Typer app."""

    @app.command()
    @track_command
    def ask(
        question: Annotated[str, typer.Argument(help="Question or prompt to send")],
        config: Annotated[str | None, typer.Option("--config", "-c", help="Path to config YAML")] = None,
        model: Annotated[str | None, typer.Option("--model", "-m", help="Model to use")] = None,
        files: Annotated[list[Path] | None, typer.Option("--file", "-f", help="Files to include as context")] = None,
        output: Annotated[Path | None, typer.Option("--output", "-o", help="Save response to a file")] = None,
        format_: Annotated[str | None, typer.Option("--format", help="Export format: json, md, html")] = None,
        skill: Annotated[str | None, typer.Option("--skill", "-s", help="Use a specific skill")] = None,
        auto_skill: Annotated[bool, typer.Option("--auto-skill", help="Auto-detect the best skill based on query")] = False,
        no_stream: Annotated[bool, typer.Option("--no-stream", help="Disable streaming output")] = False,
        code: Annotated[bool, typer.Option("--code", help="Enable coding agent mode (read/write/edit files)")] = False,
        live: Annotated[bool, typer.Option("--live", help="Enable live infrastructure tools (GKE/GCP)")] = False,
        workspace: Annotated[
            Path | None,
            typer.Option("--workspace", "-w", help="Workspace root directory for code mode"),
        ] = None,
        cluster: Annotated[str | None, typer.Option("--cluster", help="GKE cluster name (overrides config)")] = None,
        namespace: Annotated[
            str | None, typer.Option("--namespace", help="Default Kubernetes namespace (overrides config)", autocompletion=complete_namespace)
        ] = None,
        project: Annotated[
            str | None, typer.Option("--project", "-p", help="GCP project ID (overrides gcp.project_id and gke.project_id)")
        ] = None,
        project_id: Annotated[
            str | None, typer.Option("--project-id", help="GCP project ID for GKE (overrides config, alias for --project)")
        ] = None,
        location: Annotated[
            str | None, typer.Option("--location", help="GCP location (overrides config)")
        ] = None,
        verbose: Annotated[
            bool,
            typer.Option("--verbose", "-V", help="Enable verbose logging (INFO level)"),
        ] = False,
        debug: Annotated[
            bool,
            typer.Option("--debug", "-d", help="Enable debug logging (DEBUG level, shows paths and full tracebacks)"),
        ] = False,
    ) -> None:
        """Ask a single question and get a response.

        Examples:
            vaig ask "What is Kubernetes?"
            vaig ask "Analyze this code" -f main.py
            vaig ask "Analyze this" -f logs.csv -o analysis.md
            vaig ask "Investigate this incident" -s rca -f logs.txt
            vaig ask "Add error handling to app.py" --code
            vaig ask "Fix the bug in utils.py" --code -w ./my-project
            vaig ask "Analyze pod crashes" --live -s log-analysis
            vaig ask "Check OOM kills in prod" --live --namespace=production
            vaig ask "Explain this code" -f main.py --format json -o report.json
        """
        _apply_subcommand_log_flags(verbose=verbose, debug=debug)

        try:  # ── CLI error boundary ──
            settings = _helpers._get_settings(config)

            # Eagerly initialize the telemetry collector so downstream code
            # (agents, cost_tracker, session) uses the pre-warmed singleton
            # instead of falling back to get_settings().
            try:
                from vaig.core.telemetry import get_telemetry_collector

                get_telemetry_collector(settings)
            except Exception:  # noqa: BLE001
                pass

            # Apply --project / --project-id: mutate gcp.project_id AND gke.project_id
            effective_project = project or project_id
            if effective_project:
                settings.gcp.project_id = effective_project
                settings.gke.project_id = effective_project

            # Apply --location: mutate gcp.location before component creation
            if location:
                settings.gcp.location = location

            if model:
                settings.models.default = model

            from vaig.agents.orchestrator import Orchestrator
            from vaig.context.builder import ContextBuilder
            from vaig.core.client import GeminiClient
            from vaig.skills.base import SkillPhase
            from vaig.skills.registry import SkillRegistry

            client = GeminiClient(settings)
            orchestrator = Orchestrator(client, settings)

            # Build context from files
            context_str = ""
            if files:
                builder = ContextBuilder(settings)
                for f in files:
                    try:
                        builder.add_file(f)
                    except FileNotFoundError:
                        err_console.print(f"[red]File not found: {f}[/red]")
                        raise typer.Exit(1)  # noqa: B904
                builder.show_summary()
                context_str = builder.bundle.to_context_string()

            # Code mode — use CodingAgent (Tasks 5.1, 5.4, 5.5, 5.6, 5.7)
            if code:
                if workspace:
                    resolved_ws = workspace.resolve()
                    if not resolved_ws.is_dir():
                        err_console.print(f"[red]Workspace directory not found: {resolved_ws}[/red]")
                        raise typer.Exit(1)
                    settings.coding.workspace_root = str(resolved_ws)
                _execute_code_mode(client, settings, question, context_str, output=output)
                return

            # Live infrastructure mode — use InfraAgent
            if live:
                gke_config = _build_gke_config(
                    settings, cluster=cluster, namespace=namespace, project_id=effective_project, location=location,
                )
                _execute_live_mode(
                    client,
                    gke_config,
                    question,
                    context_str,
                    settings=settings,
                    output=output,
                    model_id=model,
                )
                return

            # ── Chunked file analysis ─────────────────────────────────
            # If the context is large enough to exceed the model's context window,
            # use ChunkedProcessor (Map-Reduce) instead of the normal pipeline.
            if context_str and _try_chunked_ask(client, settings, question, context_str, model_id=model, output=output):
                return

            # Execute with or without skill
            context_file_paths = [str(f) for f in files] if files else []

            # Auto-detect skill if requested (or enabled in config) and no explicit skill specified
            effective_auto_skill = auto_skill or settings.skills.auto_routing
            if effective_auto_skill and not skill:
                registry = SkillRegistry(settings)
                suggestions = registry.suggest_skill(question)
                if suggestions:
                    best_name, best_score = suggestions[0]
                    threshold = settings.skills.auto_routing_threshold
                    if best_score >= threshold:
                        skill = best_name
                        console.print(
                            f"[dim]🎯 Auto-routing to skill: [cyan]{skill}[/cyan] "
                            f"(score: {best_score:.1f})[/dim]"
                        )
                    else:
                        console.print(
                            f"[dim]Suggested skills: {', '.join(f'{n} ({s:.1f})' for n, s in suggestions)}[/dim]"
                        )

            if skill:
                registry = SkillRegistry(settings)
                active_skill = registry.get(skill)
                if not active_skill:
                    err_console.print(f"[red]Skill not found: {skill}[/red]")
                    err_console.print(f"[dim]Available: {', '.join(registry.list_names())}[/dim]")
                    raise typer.Exit(1)

                with console.status(
                    f"[bold cyan]Running {skill} skill on {settings.models.default}...[/bold cyan]"
                ):
                    result = orchestrator.execute_skill_phase(
                        active_skill,
                        SkillPhase.ANALYZE,
                        context_str,
                        question,
                    )
                console.print()
                if result.output:
                    console.print(Markdown(result.output))

                # Show cost summary for skill execution
                skill_usage = (result.metadata or {}).get("total_usage")
                _show_cost_line(skill_usage, settings.models.default)

                _handle_export_output(
                    response_text=result.output,
                    question=question,
                    model_id=settings.models.default,
                    skill_name=skill,
                    context_files=context_file_paths,
                    format_=format_,
                    output=output,
                    tokens=skill_usage,
                    cost=_compute_cost_str(skill_usage, settings.models.default),
                )
            else:
                # Direct chat — single agent
                if no_stream:
                    with console.status(
                        f"[bold cyan]Generating response with {settings.models.default}...[/bold cyan]"
                    ):
                        result = orchestrator.execute_single(question, context=context_str)  # type: ignore[assignment]
                    console.print()
                    if hasattr(result, "content") and result.content:
                        console.print(Markdown(result.content))

                        # Show cost summary
                        result_usage = getattr(result, "usage", None)
                        _show_cost_line(result_usage, settings.models.default)

                        _handle_export_output(
                            response_text=result.content,
                            question=question,
                            model_id=settings.models.default,
                            skill_name=skill,
                            context_files=context_file_paths,
                            format_=format_,
                            output=output,
                            tokens=result_usage,
                            cost=_compute_cost_str(result_usage, settings.models.default),
                        )
                else:
                    stream = orchestrator.execute_single(question, context=context_str, stream=True)
                    console.print()
                    response_parts: list[str] = []
                    for chunk in stream:  # type: ignore[union-attr]
                        console.print(chunk, end="")
                        response_parts.append(chunk)
                    console.print()

                    # Show cost summary (usage available after stream exhaustion)
                    stream_usage = getattr(stream, "usage", None)
                    _show_cost_line(stream_usage, settings.models.default)

                    _handle_export_output(
                        response_text="".join(response_parts),
                        question=question,
                        model_id=settings.models.default,
                        skill_name=skill,
                        context_files=context_file_paths,
                        format_=format_,
                        output=output,
                        tokens=stream_usage,
                        cost=_compute_cost_str(stream_usage, settings.models.default),
                    )
        except typer.Exit:
            raise  # Let typer exits pass through
        except Exception as exc:
            handle_cli_error(exc, debug=debug)


# ── Async Ask Implementation ─────────────────────────────────


async def _async_ask_impl(
    question: str,
    *,
    config: str | None = None,
    model: str | None = None,
    files: list[Path] | None = None,
    output: Path | None = None,
    format_: str | None = None,
    skill: str | None = None,
    auto_skill: bool = False,
    no_stream: bool = False,
    code: bool = False,
    live: bool = False,
    workspace: Path | None = None,
    cluster: str | None = None,
    namespace: str | None = None,
    project: str | None = None,
    location: str | None = None,
) -> None:
    """Async implementation of the ask command.

    Uses async agent methods (``async_execute()``, ``async_execute_stream()``)
    for non-blocking I/O.  Called from the sync Typer entry point via
    ``async_run_command()``.

    All parameters mirror the sync ``ask`` command.
    """
    settings = _helpers._get_settings(config)

    # Initialize telemetry eagerly
    try:
        from vaig.core.telemetry import get_telemetry_collector

        get_telemetry_collector(settings)
    except Exception:  # noqa: BLE001
        pass

    if project:
        settings.gcp.project_id = project
        settings.gke.project_id = project
    if location:
        settings.gcp.location = location
    if model:
        settings.models.default = model

    from vaig.agents.orchestrator import Orchestrator
    from vaig.context.builder import ContextBuilder
    from vaig.core.client import GeminiClient
    from vaig.skills.base import SkillPhase
    from vaig.skills.registry import SkillRegistry

    client = GeminiClient(settings)
    orchestrator = Orchestrator(client, settings)

    # Build context from files (async file loading)
    context_str = ""
    if files:
        builder = ContextBuilder(settings)
        for f in files:
            try:
                await builder.async_add_file(f)
            except FileNotFoundError:
                err_console.print(f"[red]File not found: {f}[/red]")
                raise typer.Exit(1)  # noqa: B904
        builder.show_summary()
        context_str = builder.bundle.to_context_string()

    # Code mode — async CodingAgent
    if code:
        if workspace:
            resolved_ws = workspace.resolve()
            if not resolved_ws.is_dir():
                err_console.print(f"[red]Workspace directory not found: {resolved_ws}[/red]")
                raise typer.Exit(1)
            settings.coding.workspace_root = str(resolved_ws)

        from vaig.cli.commands._code import _async_execute_code_mode

        await _async_execute_code_mode(client, settings, question, context_str, output=output)
        return

    # Live infrastructure mode — async InfraAgent
    if live:
        from vaig.cli.commands.live import _async_execute_live_mode

        gke_config = _build_gke_config(
            settings, cluster=cluster, namespace=namespace, project_id=project, location=location,
        )
        await _async_execute_live_mode(
            client,
            gke_config,
            question,
            context_str,
            settings=settings,
            output=output,
            model_id=model,
        )
        return

    # Chunked file analysis (async)
    if context_str:
        from vaig.cli.commands._code import _async_try_chunked_ask

        if await _async_try_chunked_ask(client, settings, question, context_str, model_id=model, output=output):
            return

    # Execute with or without skill
    context_file_paths = [str(f) for f in files] if files else []

    # Auto-detect skill
    effective_auto_skill = auto_skill or settings.skills.auto_routing
    if effective_auto_skill and not skill:
        registry = SkillRegistry(settings)
        suggestions = registry.suggest_skill(question)
        if suggestions:
            best_name, best_score = suggestions[0]
            threshold = settings.skills.auto_routing_threshold
            if best_score >= threshold:
                skill = best_name
                console.print(
                    f"[dim]🎯 Auto-routing to skill: [cyan]{skill}[/cyan] "
                    f"(score: {best_score:.1f})[/dim]"
                )
            else:
                console.print(
                    f"[dim]Suggested skills: {', '.join(f'{n} ({s:.1f})' for n, s in suggestions)}[/dim]"
                )

    if skill:
        registry = SkillRegistry(settings)
        active_skill = registry.get(skill)
        if not active_skill:
            err_console.print(f"[red]Skill not found: {skill}[/red]")
            err_console.print(f"[dim]Available: {', '.join(registry.list_names())}[/dim]")
            raise typer.Exit(1)

        with console.status(
            f"[bold cyan]Running {skill} skill on {settings.models.default} (async)...[/bold cyan]"
        ):
            result = await orchestrator.async_execute_skill_phase(
                active_skill,
                SkillPhase.ANALYZE,
                context_str,
                question,
            )
        console.print()
        if result.output:
            console.print(Markdown(result.output))

        skill_usage = (result.metadata or {}).get("total_usage")
        _show_cost_line(skill_usage, settings.models.default)

        _handle_export_output(
            response_text=result.output,
            question=question,
            model_id=settings.models.default,
            skill_name=skill,
            context_files=context_file_paths,
            format_=format_,
            output=output,
            tokens=skill_usage,
            cost=_compute_cost_str(skill_usage, settings.models.default),
        )
    else:
        # Direct chat — async single agent
        if no_stream:
            with console.status(
                f"[bold cyan]Generating response with {settings.models.default} (async)...[/bold cyan]"
            ):
                result = await orchestrator.async_execute_single(question, context=context_str)  # type: ignore[assignment]
            console.print()
            if hasattr(result, "content") and result.content:
                console.print(Markdown(result.content))

                result_usage = getattr(result, "usage", None)
                _show_cost_line(result_usage, settings.models.default)

                _handle_export_output(
                    response_text=result.content,
                    question=question,
                    model_id=settings.models.default,
                    skill_name=skill,
                    context_files=context_file_paths,
                    format_=format_,
                    output=output,
                    tokens=result_usage,
                    cost=_compute_cost_str(result_usage, settings.models.default),
                )
        else:
            stream = await orchestrator.async_execute_single(question, context=context_str, stream=True)
            console.print()
            response_parts: list[str] = []
            async for chunk in stream:  # type: ignore[union-attr]
                console.print(chunk, end="")
                response_parts.append(chunk)
            console.print()

            stream_usage = getattr(stream, "usage", None)
            _show_cost_line(stream_usage, settings.models.default)

            _handle_export_output(
                response_text="".join(response_parts),
                question=question,
                model_id=settings.models.default,
                skill_name=skill,
                context_files=context_file_paths,
                format_=format_,
                output=output,
                tokens=stream_usage,
                cost=_compute_cost_str(stream_usage, settings.models.default),
            )

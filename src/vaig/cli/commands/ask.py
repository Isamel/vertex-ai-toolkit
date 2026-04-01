"""Ask command — single-shot question with async support."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.markdown import Markdown

from vaig.cli import _helpers
from vaig.cli._completions import complete_namespace

if TYPE_CHECKING:
    from vaig.skills.base import SkillPhase
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
from vaig.cli.commands.live import _execute_live_mode
from vaig.core.gke import build_gke_config as _build_gke_config


def register(app: typer.Typer) -> None:
    """Register the ``ask`` command group with the CLI application."""

    @app.command()
    @track_command
    def ask(
        question: Annotated[str, typer.Argument(help="Question or prompt to send")],
        config: Annotated[str | None, typer.Option("--config", "-c", help="Path to config YAML")] = None,
        model: Annotated[str | None, typer.Option("--model", "-m", help="Model to use")] = None,
        files: Annotated[list[Path] | None, typer.Option("--file", "-f", help="Files to include as context")] = None,
        dirs: Annotated[
            list[Path] | None,
            typer.Option("--dir", "-d", help="Directories to include as context (recursively loaded)"),
        ] = None,
        examples: Annotated[
            list[Path] | None,
            typer.Option("--examples", "-e", help="Reference/example files to guide the LLM (shown in a separate section)"),
        ] = None,
        phases: Annotated[
            str | None,
            typer.Option(
                "--phases",
                help=(
                    "Comma-separated list of skill phases to run sequentially "
                    "(e.g. 'analyze,plan,execute,validate,report'). "
                    "Requires --skill. Default: run only 'analyze'."
                ),
            ),
        ] = None,
        output: Annotated[Path | None, typer.Option("--output", "-o", help="Save response to a file")] = None,
        format_: Annotated[str | None, typer.Option("--format", help="Export format: json, md, html")] = None,
        skill: Annotated[str | None, typer.Option("--skill", "-s", help="Use a specific skill")] = None,
        auto_skill: Annotated[bool, typer.Option("--auto-skill", help="Auto-detect the best skill based on query")] = False,
        no_stream: Annotated[bool, typer.Option("--no-stream", help="Disable streaming output")] = False,
        code: Annotated[bool, typer.Option("--code", help="Enable coding agent mode (read/write/edit files)")] = False,
        pipeline: Annotated[
            bool,
            typer.Option(
                "--pipeline",
                help=(
                    "Use 3-agent pipeline (Planner→Implementer→Verifier) for code mode. "
                    "Requires --code. Pipeline mode does not support interactive "
                    "confirm_actions; use in non-interactive contexts."
                ),
            ),
        ] = False,
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
            str | None, typer.Option("--project", "-p", help="GCP project ID (overrides gcp.project_id only)")
        ] = None,
        project_id: Annotated[
            str | None, typer.Option("--project-id", help="GCP project ID for GKE (overrides config, alias for --project)")
        ] = None,
        location: Annotated[
            str | None, typer.Option("--location", help="GCP location (overrides config)")
        ] = None,
        gke_project: Annotated[
            str | None,
            typer.Option("--gke-project", help="GKE project ID (overrides gke.project_id; defaults to --project if unset)"),
        ] = None,
        gke_location: Annotated[
            str | None,
            typer.Option("--gke-location", help="GKE cluster location (overrides gke.location)"),
        ] = None,
        verbose: Annotated[
            bool,
            typer.Option("--verbose", "-V", help="Enable verbose logging (INFO level)"),
        ] = False,
        debug: Annotated[
            bool,
            typer.Option("--debug", help="Enable debug logging (DEBUG level, shows paths and full tracebacks)"),
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
            vaig ask "Implement retry logic" --code --pipeline
            vaig ask "Analyze pod crashes" --live -s log-analysis
            vaig ask "Check OOM kills in prod" --live --namespace=production
            vaig ask "Explain this code" -f main.py --format json -o report.json
            vaig ask "Migrate to AWS Glue" -d ./pentaho -s migration --phases analyze,plan,execute,validate,report
            vaig ask "Migrate this code" -f src.py -e example_output.py -s code-migration
        """
        _apply_subcommand_log_flags(verbose=verbose, debug=debug)

        try:  # ── CLI error boundary ──
            settings = _helpers._get_settings(config)

            # Validate: --phases requires --skill
            if phases and not skill:
                console.print("[red]Error:[/red] --phases requires --skill to be specified.")
                raise typer.Exit(code=1)

            # Eagerly initialize the telemetry collector and wire the
            # TelemetrySubscriber so events from CostTracker, track_command,
            # etc. are forwarded to the SQLite telemetry store.
            _helpers._init_telemetry(settings)
            _helpers._init_audit(settings)
            _helpers._check_platform_auth(settings)

            # Apply --project / --project-id: mutate ONLY gcp.project_id
            # The GKE fallback chain (gke.project_id or gcp.project_id) handles single-project setups.
            effective_project = project or project_id
            if effective_project:
                settings.gcp.project_id = effective_project

            # Apply --gke-project: mutate ONLY gke.project_id when explicitly provided
            if gke_project:
                settings.gke.project_id = gke_project

            # Apply --gke-location: mutate ONLY gke.location when explicitly provided
            if gke_location:
                settings.gke.location = gke_location

            # Apply --location: mutate gcp.location before component creation
            if location:
                settings.gcp.location = location

            if model:
                settings.models.default = model

            from vaig.agents.orchestrator import Orchestrator
            from vaig.context.builder import ContextBuilder
            from vaig.core.container import build_container
            from vaig.skills.registry import SkillRegistry

            container = build_container(settings)
            client = container.gemini_client
            orchestrator = Orchestrator(client, settings)

            # Build context from files and/or directories
            context_str = ""
            if files or dirs or examples:
                builder = ContextBuilder(settings)

                # Load --examples first into a separate bundle so we can format them distinctly
                examples_context = ""
                if examples:
                    examples_builder = ContextBuilder(settings)
                    for ex in examples:
                        try:
                            examples_builder.add_file(ex)
                        except FileNotFoundError:
                            err_console.print(f"[red]Example file not found: {ex}[/red]")
                            raise typer.Exit(1)  # noqa: B904
                    examples_context = examples_builder.bundle.to_context_string()

                # Load --file entries
                if files:
                    for f in files:
                        try:
                            builder.add_file(f)
                        except FileNotFoundError:
                            err_console.print(f"[red]File not found: {f}[/red]")
                            raise typer.Exit(1)  # noqa: B904

                # Load --dir entries (recursive)
                if dirs:
                    for d in dirs:
                        try:
                            count = builder.add_directory(d)
                            if count == 0:
                                err_console.print(f"[yellow]Warning: no supported files found in directory: {d}[/yellow]")
                        except FileNotFoundError:
                            err_console.print(f"[red]Directory not found: {d}[/red]")
                            raise typer.Exit(1)  # noqa: B904

                builder.show_summary()
                source_context = builder.bundle.to_context_string()

                # Compose final context: examples section first (if any), then source code
                sections = []
                if examples_context:
                    sections.append("## Reference Examples\n\n" + examples_context)
                if source_context and examples_context:
                    sections.append("## Source Code (to migrate)\n\n" + source_context)

                if len(sections) > 1:
                    context_str = "\n\n---\n\n".join(sections)
                elif sections:
                    context_str = sections[0]
                else:
                    context_str = source_context

            # Code mode — use CodingAgent (Tasks 5.1, 5.4, 5.5, 5.6, 5.7)
            if code:
                if workspace:
                    resolved_ws = workspace.resolve()
                    if not resolved_ws.is_dir():
                        err_console.print(f"[red]Workspace directory not found: {resolved_ws}[/red]")
                        raise typer.Exit(1)
                    settings.coding.workspace_root = str(resolved_ws)
                _execute_code_mode(client, settings, question, context_str, output=output, pipeline=pipeline)
                return

            # Live infrastructure mode — use InfraAgent
            if live:
                # Do NOT pass project_id/location here — gke_project/gke_location are
                # already written to settings.gke.* above, and _build_gke_config reads
                # them via its fallback chain (gke.project_id or gcp.project_id).
                # Passing effective_project/location would override gke-specific flags.
                gke_config = _build_gke_config(
                    settings, cluster=cluster, namespace=namespace,
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
            if dirs:
                context_file_paths.extend(str(d) for d in dirs)
            if examples:
                context_file_paths.extend(str(e) for e in examples)

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

                # Parse --phases (comma-separated phase names, e.g. "analyze,plan,execute")
                phase_list = _parse_phases(phases)

                # Run each requested phase sequentially; each phase output feeds into the next
                from vaig.skills.base import SkillPhase, SkillResult  # noqa: PLC0415
                accumulated_context = context_str
                final_result = SkillResult(output="", success=False, phase=SkillPhase.ANALYZE)
                for phase in phase_list:
                    phase_label = f"{skill} [{phase.value}]"
                    with console.status(
                        f"[bold cyan]Running {phase_label} on {settings.models.default}...[/bold cyan]"
                    ):
                        phase_result = orchestrator.execute_skill_phase(
                            active_skill,
                            phase,
                            accumulated_context,
                            question,
                        )
                    if not phase_result.success:
                        console.print(f"[red]Phase '{phase.value}' failed. Aborting multi-phase execution.[/red]")
                        final_result = phase_result
                        break
                    console.print()
                    if phase_result.output:
                        if len(phase_list) > 1:
                            console.print(f"[bold]── Phase: {phase.value} ──[/bold]")
                        console.print(Markdown(phase_result.output))
                    # Feed this phase's output as context for the next phase
                    if phase_result.output:
                        accumulated_context = (
                            f"{accumulated_context}\n\n"
                            f"## Phase Output ({phase.value})\n\n"
                            f"{phase_result.output}"
                        )
                    final_result = phase_result

                result = final_result

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
        except Exception as exc:  # noqa: BLE001
            handle_cli_error(exc, debug=debug)


# ── Phase parsing helper ──────────────────────────────────────


def _parse_phases(phases: str | None) -> list[SkillPhase]:
    """Parse a comma-separated phases string into a list of SkillPhase values.

    Falls back to [SkillPhase.ANALYZE] when phases is None or empty.
    Raises typer.Exit(1) on invalid phase names.
    """
    from vaig.skills.base import SkillPhase

    if not phases:
        return [SkillPhase.ANALYZE]

    valid = {p.value: p for p in SkillPhase}
    result: list[SkillPhase] = []
    for token in phases.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token not in valid:
            err_console.print(
                f"[red]Unknown phase: '{token}'. Valid: {', '.join(valid)}[/red]"
            )
            raise typer.Exit(1)
        result.append(valid[token])
    return result or [SkillPhase.ANALYZE]


# ── Async Ask Implementation ─────────────────────────────────


async def _async_ask_impl(
    question: str,
    *,
    config: str | None = None,
    model: str | None = None,
    files: list[Path] | None = None,
    dirs: list[Path] | None = None,
    examples: list[Path] | None = None,
    phases: str | None = None,
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
    gke_project: str | None = None,
    gke_location: str | None = None,
) -> None:
    """Async implementation of the ask command.

    Uses async agent methods (``async_execute()``, ``async_execute_stream()``)
    for non-blocking I/O.  Called from the sync Typer entry point via
    ``async_run_command()``.

    All parameters mirror the sync ``ask`` command.
    """
    settings = _helpers._get_settings(config)

    # Validate: --phases requires --skill
    if phases and not skill:
        console.print("[red]Error:[/red] --phases requires --skill to be specified.")
        raise typer.Exit(code=1)

    # Initialize telemetry eagerly + wire subscriber
    _helpers._init_telemetry(settings)
    _helpers._init_audit(settings)

    if project:
        settings.gcp.project_id = project
    if gke_project:
        settings.gke.project_id = gke_project
    if gke_location:
        settings.gke.location = gke_location
    if location:
        settings.gcp.location = location
    if model:
        settings.models.default = model

    from vaig.agents.orchestrator import Orchestrator
    from vaig.context.builder import ContextBuilder
    from vaig.core.container import build_container
    from vaig.skills.registry import SkillRegistry

    container = build_container(settings)
    client = container.gemini_client
    orchestrator = Orchestrator(client, settings)

    # Build context from files and/or directories (async)
    context_str = ""
    if files or dirs or examples:
        builder = ContextBuilder(settings)

        # Load --examples into a separate builder for distinct section
        examples_context = ""
        if examples:
            examples_builder = ContextBuilder(settings)
            for ex in examples:
                try:
                    await examples_builder.async_add_file(ex)
                except FileNotFoundError:
                    err_console.print(f"[red]Example file not found: {ex}[/red]")
                    raise typer.Exit(1)  # noqa: B904
            examples_context = examples_builder.bundle.to_context_string()

        # Load --file entries (async)
        if files:
            for f in files:
                try:
                    await builder.async_add_file(f)
                except FileNotFoundError:
                    err_console.print(f"[red]File not found: {f}[/red]")
                    raise typer.Exit(1)  # noqa: B904

        # Load --dir entries (async, recursive)
        if dirs:
            for d in dirs:
                try:
                    count = await builder.async_add_directory(d)
                    if count == 0:
                        err_console.print(
                            f"[yellow]Warning: no supported files found in directory: {d}[/yellow]"
                        )
                except FileNotFoundError:
                    err_console.print(f"[red]Directory not found: {d}[/red]")
                    raise typer.Exit(1)  # noqa: B904

        builder.show_summary()
        source_context = builder.bundle.to_context_string()

        # Compose final context: examples section first (if any), then source code
        sections = []
        if examples_context:
            sections.append("## Reference Examples\n\n" + examples_context)
        if source_context and examples_context:
            sections.append("## Source Code (to migrate)\n\n" + source_context)

        if len(sections) > 1:
            context_str = "\n\n---\n\n".join(sections)
        elif sections:
            context_str = sections[0]
        else:
            context_str = source_context

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

        # Do NOT pass project_id/location here — gke_project/gke_location are
        # already written to settings.gke.* above, and _build_gke_config reads
        # them via its fallback chain (gke.project_id or gcp.project_id).
        # Passing project/location would override gke-specific flags.
        gke_config = _build_gke_config(
            settings, cluster=cluster, namespace=namespace,
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
    if dirs:
        context_file_paths.extend(str(d) for d in dirs)
    if examples:
        context_file_paths.extend(str(e) for e in examples)

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

        # Parse --phases (comma-separated phase names, e.g. "analyze,plan,execute")
        phase_list = _parse_phases(phases)

        # Run each requested phase sequentially; each phase output feeds into the next
        from vaig.skills.base import SkillPhase, SkillResult  # noqa: PLC0415
        accumulated_context = context_str
        final_result = SkillResult(output="", success=False, phase=SkillPhase.ANALYZE)
        for phase in phase_list:
            phase_label = f"{skill} [{phase.value}]"
            with console.status(
                f"[bold cyan]Running {phase_label} on {settings.models.default} (async)...[/bold cyan]"
            ):
                phase_result = await orchestrator.async_execute_skill_phase(
                    active_skill,
                    phase,
                    accumulated_context,
                    question,
                )
            if not phase_result.success:
                console.print(f"[red]Phase '{phase.value}' failed. Aborting multi-phase execution.[/red]")
                final_result = phase_result
                break
            console.print()
            if phase_result.output:
                if len(phase_list) > 1:
                    console.print(f"[bold]── Phase: {phase.value} ──[/bold]")
                console.print(Markdown(phase_result.output))
            # Feed this phase's output as context for the next phase
            if phase_result.output:
                accumulated_context = (
                    f"{accumulated_context}\n\n"
                    f"## Phase Output ({phase.value})\n\n"
                    f"{phase_result.output}"
                )
            final_result = phase_result

        result = final_result

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

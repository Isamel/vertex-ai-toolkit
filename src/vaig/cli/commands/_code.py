"""Code mode helpers — CodingAgent execution and chunked analysis."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.markdown import Markdown
from rich.panel import Panel

from vaig.cli._helpers import (
    _build_cost_markdown_section,
    _cli_confirm,
    _compute_cost_str,
    _save_output,
    _show_coding_summary,
    console,
    err_console,
)
from vaig.tools.repo.models import ProvenanceMetadata

if TYPE_CHECKING:
    from vaig.core.config import Settings
    from vaig.core.protocols import GeminiClientProtocol

logger = logging.getLogger(__name__)


def _parse_from_repo(from_repo: str) -> tuple[str, str]:
    """Parse ``owner/repo[@ref]`` into ``(owner/repo, ref)``.

    Args:
        from_repo: Repository string in ``owner/repo`` or ``owner/repo@ref`` format.

    Returns:
        Tuple of ``(repo_slug, ref)`` where *ref* defaults to ``"HEAD"`` when
        not specified.

    Examples:
        >>> _parse_from_repo("octocat/hello-world")
        ('octocat/hello-world', 'HEAD')
        >>> _parse_from_repo("octocat/hello-world@main")
        ('octocat/hello-world', 'main')
        >>> _parse_from_repo("octocat/hello-world@abc1234")
        ('octocat/hello-world', 'abc1234')
    """
    if "@" in from_repo:
        repo_part, ref = from_repo.rsplit("@", 1)
        return repo_part, ref
    return from_repo, "HEAD"


def _validate_from_repo_allowlist(repo_slug: str, settings: Settings) -> None:
    """Raise ``typer.Exit(1)`` when *repo_slug* is not in the allowed_repos list.

    No-op when the allowlist is empty (all repos allowed).

    Args:
        repo_slug: ``owner/repo`` identifier to check.
        settings: Application settings containing ``github.allowed_repos``.

    Raises:
        typer.Exit: When the repo is not in the non-empty allowlist.
    """
    allowed = settings.github.allowed_repos
    if allowed and repo_slug not in allowed:
        err_console.print(
            f"[red]Error:[/red] Repository '[cyan]{repo_slug}[/cyan]' is not in "
            "the allowed_repos allowlist.\n"
            "[dim]Configure github.allowed_repos in your config or add the repo.[/dim]"
        )
        raise typer.Exit(1)


def _build_remote_context(
    settings: Settings,
    repo_slug: str,
    ref: str,
) -> tuple[str, list[ProvenanceMetadata]]:
    """Shallow-clone *repo_slug* and return wrapped source content + provenance.

    Performs the following steps:
    1. Converts ``owner/repo`` → HTTPS clone URL.
    2. Validates against the allowed_repos allowlist (GH-03-R3).
    3. Runs :class:`~vaig.tools.repo.batch.TreeTriageReport` on the cloned tree
       (GH-03-R9).
    4. Reads all Tier-1 and Tier-2 files from the clone, wrapping each through
       :func:`~vaig.core.prompt_defense.wrap_untrusted_content` (GH-03-R8).
    5. Attaches :class:`ProvenanceMetadata` to each file reference (GH-03-R6).
    6. Cleans up via the :func:`~vaig.tools.repo.shallow_clone.shallow_clone`
       context manager (GH-03-R7).

    Args:
        settings: Application settings (used for GitHub config).
        repo_slug: ``owner/repo`` identifier.
        ref: Branch, tag, or commit SHA to clone.

    Returns:
        Tuple of ``(context_text, provenance_list)`` where *context_text* is
        the combined file contents wrapped in untrusted-data delimiters and
        *provenance_list* contains one :class:`ProvenanceMetadata` per file.
    """
    from vaig.core.prompt_defense import wrap_untrusted_content
    from vaig.tools.repo.batch import Tier, TreeTriageReport, TriagedEntry
    from vaig.tools.repo.shallow_clone import shallow_clone

    github_config = settings.github
    clone_url = f"https://github.com/{repo_slug}.git"

    # Determine effective ref: use config default when caller passed "HEAD"
    effective_ref = ref if ref != "HEAD" else github_config.default_ref

    provenance_list: list[ProvenanceMetadata] = []
    sections: list[str] = []

    with shallow_clone(github_config, clone_url, ref=effective_ref) as clone_path:
        # GH-03-R9: Run TreeTriageReport before pipeline starts
        repo_files = sorted(clone_path.rglob("*"))
        entries: list[TriagedEntry] = []
        for p in repo_files:
            if not p.is_file():
                continue
            rel = p.relative_to(clone_path)
            # Simple tier assignment: source files → tier_1, tests → tier_2, rest → tier_3
            rel_str = str(rel)
            if rel_str.startswith("."):
                continue
            if any(part.startswith(".") for part in rel.parts):
                continue
            suffix = p.suffix.lower()
            if suffix in {".py", ".ts", ".js", ".go", ".java", ".rs", ".rb", ".cs"}:
                tier = Tier.TIER_1
            elif suffix in {".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".md"}:
                tier = Tier.TIER_2
            else:
                tier = Tier.TIER_3
            entries.append(TriagedEntry(path=rel_str, tier=tier))

        triage_report = TreeTriageReport(
            owner=repo_slug.split("/")[0],
            repo=repo_slug.split("/")[1] if "/" in repo_slug else repo_slug,
            ref=effective_ref,
            entries=entries,
            total_files=len(entries),
        )
        logger.info(
            "GH-03 TreeTriageReport: %d files triaged in %s@%s",
            triage_report.total_files,
            repo_slug,
            effective_ref,
        )

        migrated_at = datetime.now(tz=UTC)

        # GH-03-R4 + R6 + R8: Read Tier-1/2 files, wrap, attach provenance
        for entry in triage_report.entries:
            if entry.tier not in (Tier.TIER_1, Tier.TIER_2):
                continue
            file_path = clone_path / entry.path
            if not file_path.is_file():
                continue
            try:
                raw_content = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            # GH-03-R8: Wrap through prompt defense
            wrapped = wrap_untrusted_content(raw_content)
            sections.append(f"### {entry.path}\n\n{wrapped}")

            provenance_list.append(
                ProvenanceMetadata(
                    source_repo=repo_slug,
                    source_ref=effective_ref,
                    source_path=entry.path,
                    migrated_at=migrated_at,
                )
            )

    context_text = (
        f"## Remote Source Repository: {repo_slug}@{effective_ref}\n\n"
        + "\n\n---\n\n".join(sections)
    )
    return context_text, provenance_list


def _check_write_path_flags(to_repo: str | None, push: bool, settings: Settings) -> None:
    """Log a warning when ``--to-repo`` or ``--push`` are used without git enabled.

    When :attr:`~vaig.core.config.GitConfig.enabled` is True the flags are
    handled by the git lifecycle in :func:`_execute_code_pipeline`; no action
    is needed here.  When disabled, we warn the user that these flags have no
    effect rather than raising an error (Phase 8 replaces the old stub).

    Args:
        to_repo: Value of the ``--to-repo`` CLI option (``None`` if omitted).
        push: Value of the ``--push`` CLI option.
        settings: Application settings (for ``coding.git.enabled`` check).
    """
    if settings.coding.git.enabled:
        return
    if to_repo is not None:
        logger.warning(
            "--to-repo '%s' has no effect: coding.git.enabled is False. "
            "Set coding.git.enabled=true in config to activate git integration.",
            to_repo,
        )
    if push:
        logger.warning(
            "--push has no effect: coding.git.enabled is False. "
            "Set coding.git.enabled=true in config to activate git integration.",
        )


def _execute_code_mode(
    client: GeminiClientProtocol,
    settings: Settings,
    question: str,
    context: str,
    *,
    output: Path | None = None,
    pipeline: bool = False,
) -> None:
    """Execute a coding task using the CodingAgent or CodingSkillOrchestrator.

    When ``pipeline=True`` (or ``settings.coding.pipeline_mode`` is set), routes
    the task through the 3-agent Planner → Implementer → Verifier pipeline.
    Otherwise falls back to the single-agent CodingAgent loop.

    Handles confirmation prompts, tool execution feedback,
    iteration/usage summary, and MaxIterationsError.
    """
    coding_config = settings.coding
    use_pipeline = pipeline or coding_config.pipeline_mode

    if use_pipeline:
        _execute_code_pipeline(client, settings, question, context, output=output)
        return

    from vaig.agents.coding import CodingAgent
    from vaig.core.exceptions import MaxIterationsError

    console.print(
        Panel.fit(
            "[bold yellow]🔧 Code Mode[/bold yellow]\n"
            f"[dim]Workspace: {Path(coding_config.workspace_root).resolve()}[/dim]\n"
            f"[dim]Model: {settings.models.default} | Max iterations: {coding_config.max_tool_iterations}[/dim]",
            border_style="yellow",
        )
    )

    agent = CodingAgent(
        client,
        coding_config,
        settings=settings,
        confirm_fn=_cli_confirm,
        model_id=settings.models.default,
    )

    try:
        # NOTE: No spinner wrapper here — confirm_fn needs interactive terminal access.
        # A spinner would swallow confirmation prompts and freeze the terminal.
        console.print("[bold cyan]🤖 Coding agent working...[/bold cyan]")
        result = agent.execute(question, context=context)

        # Display final response
        console.print()
        if result.content:
            console.print(Markdown(result.content))
        elif result.success:
            console.print("[green]Code task completed successfully.[/green]")
        else:
            console.print("[yellow]Agent finished with no output.[/yellow]")
        console.print()

        if output:
            # Append cost summary to saved file
            save_content = result.content
            result_usage = result.usage or None
            if result_usage:
                cost_section = _build_cost_markdown_section(
                    result_usage, settings.models.default,
                    _compute_cost_str(result_usage, settings.models.default),
                )
                if cost_section:
                    save_content = f"{result.content}\n\n{cost_section}"
            _save_output(output, save_content)

        # Task 5.5 + 5.6: Show tool execution feedback and usage summary
        _show_coding_summary(result)

    except MaxIterationsError as exc:
        # Task 5.7: Graceful handling
        err_console.print(
            f"\n[bold red]⚠ Max iterations reached ({exc.iterations})[/bold red]\n"
            "[yellow]The coding agent hit its iteration limit. "
            "This usually means the task is too complex for a single turn.\n"
            "Try breaking it into smaller steps.[/yellow]"
        )
        raise typer.Exit(1)  # noqa: B904


def _execute_code_pipeline(
    client: GeminiClientProtocol,
    settings: Settings,
    question: str,
    context: str,
    *,
    output: Path | None = None,
) -> None:
    """Execute a coding task using the 3-agent CodingSkillOrchestrator pipeline.

    Runs Planner → Implementer → Verifier sequentially and displays the
    verification report.  Exits with code 1 when the Verifier reports failure.

    .. note::
        Pipeline mode uses :class:`~vaig.agents.tool_aware.ToolAwareAgent` which
        does **not** implement interactive ``confirm_actions``.  If
        ``settings.coding.confirm_actions`` is True, a warning is logged and
        pipeline proceeds without confirmation prompts.  For interactive
        confirmation, use single-agent mode (omit ``--pipeline``).
    """
    from vaig.agents.coding_pipeline import CodingSkillOrchestrator
    from vaig.core.exceptions import MaxIterationsError
    from vaig.core.git_integration import GitManager, GitSafetyError, _sanitize_branch_name

    coding_config = settings.coding
    git_manager = GitManager(coding_config.git, workspace=Path(coding_config.workspace_root).resolve())

    if coding_config.confirm_actions:
        logger.warning(
            "Pipeline mode does not support interactive confirm_actions. "
            "Proceeding without confirmation prompts. "
            "Set confirm_actions=false in config or use single-agent mode for interactive confirmation."
        )

    console.print(
        Panel.fit(
            "[bold yellow]🔧 Code Mode (Pipeline)[/bold yellow]\n"
            f"[dim]Workspace: {Path(coding_config.workspace_root).resolve()}[/dim]\n"
            f"[dim]Model: {settings.models.default} | "
            f"Pipeline: Planner → Implementer → Verifier[/dim]",
            border_style="yellow",
        )
    )

    orchestrator = CodingSkillOrchestrator(
        client,
        coding_config,
        settings=settings,
    )

    try:
        # ── Git pre-flight: create feature branch ─────────────
        if git_manager.enabled and coding_config.git.auto_branch:
            branch_name = _sanitize_branch_name(question[:60], prefix=coding_config.git.branch_prefix)
            try:
                git_manager.create_branch(branch_name)
                logger.info("Git lifecycle: created branch '%s'", branch_name)
            except GitSafetyError as exc:
                err_console.print(f"[yellow]Git safety guard: {exc}[/yellow]")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Git lifecycle: create_branch failed (%s); continuing without branch", exc)

        console.print("[bold cyan]🤖 Pipeline running (Planner → Implementer → Verifier)...[/bold cyan]")
        result = orchestrator.run(question, context=context)

        # ── Git post-flight: commit changes ───────────────────
        if git_manager.enabled and coding_config.git.auto_commit and result.success:
            try:
                git_manager.commit_all(f"feat(code): {question[:72]}")
                logger.info("Git lifecycle: committed changes")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Git lifecycle: commit_all failed (%s); continuing", exc)

    except MaxIterationsError as exc:
        err_console.print(
            f"\n[bold red]⚠ Max iterations reached ({exc.iterations})[/bold red]\n"
            "[yellow]The pipeline hit its iteration limit. "
            "Try breaking the task into smaller steps or increasing max_tool_iterations.[/yellow]"
        )
        raise typer.Exit(1)  # noqa: B904

    # Display verification report
    console.print()
    if result.verification_report:
        console.print(Markdown(result.verification_report))
    console.print()

    if output:
        save_content = result.verification_report
        if result.usage:
            cost_section = _build_cost_markdown_section(
                result.usage, settings.models.default,
                _compute_cost_str(result.usage, settings.models.default),
            )
            if cost_section:
                save_content = f"{result.verification_report}\n\n{cost_section}"
        _save_output(output, save_content)

    status = "✅ PASS" if result.success else "❌ FAIL"
    total = result.usage.get("total_tokens", 0)
    console.print(
        f"[dim]Pipeline {status} | Total tokens: {total}[/dim]"
    )

    if not result.success:
        raise typer.Exit(1)


async def _async_execute_code_mode(
    client: GeminiClientProtocol,
    settings: Settings,
    question: str,
    context: str,
    *,
    output: Path | None = None,
    pipeline: bool = False,
) -> None:
    """Async version of :func:`_execute_code_mode`.

    When ``pipeline=True`` (or ``settings.coding.pipeline_mode`` is set), wraps
    the synchronous :func:`_execute_code_pipeline` via ``asyncio.to_thread``.
    Otherwise uses ``CodingAgent.async_execute()`` for non-blocking tool loops.
    The confirmation callback still runs synchronously (prompt_toolkit
    limitation in non-async REPL context), but all agent I/O is async.
    """
    import asyncio

    coding_config = settings.coding
    use_pipeline = pipeline or coding_config.pipeline_mode

    if use_pipeline:
        await asyncio.to_thread(
            _execute_code_pipeline, client, settings, question, context, output=output
        )
        return

    from vaig.agents.coding import CodingAgent
    from vaig.core.exceptions import MaxIterationsError

    console.print(
        Panel.fit(
            "[bold yellow]🔧 Code Mode (async)[/bold yellow]\n"
            f"[dim]Workspace: {Path(coding_config.workspace_root).resolve()}[/dim]\n"
            f"[dim]Model: {settings.models.default} | Max iterations: {coding_config.max_tool_iterations}[/dim]",
            border_style="yellow",
        )
    )

    agent = CodingAgent(
        client,
        coding_config,
        settings=settings,
        confirm_fn=_cli_confirm,
        model_id=settings.models.default,
    )

    try:
        console.print("[bold cyan]🤖 Coding agent working (async)...[/bold cyan]")
        result = await agent.async_execute(question, context=context)

        # Display final response
        console.print()
        if result.content:
            console.print(Markdown(result.content))
        elif result.success:
            console.print("[green]Code task completed successfully.[/green]")
        else:
            console.print("[yellow]Agent finished with no output.[/yellow]")
        console.print()

        if output:
            save_content = result.content
            result_usage = result.usage or None
            if result_usage:
                cost_section = _build_cost_markdown_section(
                    result_usage, settings.models.default,
                    _compute_cost_str(result_usage, settings.models.default),
                )
                if cost_section:
                    save_content = f"{result.content}\n\n{cost_section}"
            _save_output(output, save_content)

        _show_coding_summary(result)

    except MaxIterationsError as exc:
        err_console.print(
            f"\n[bold red]⚠ Max iterations reached ({exc.iterations})[/bold red]\n"
            "[yellow]The coding agent hit its iteration limit. "
            "This usually means the task is too complex for a single turn.\n"
            "Try breaking it into smaller steps.[/yellow]"
        )
        raise typer.Exit(1)  # noqa: B904


def _try_chunked_ask(
    client: GeminiClientProtocol,
    settings: Settings,
    question: str,
    context_str: str,
    *,
    model_id: str | None = None,
    output: Path | None = None,
) -> bool:
    """Attempt chunked (Map-Reduce) processing if content exceeds the context window.

    Returns True if chunking was performed (caller should return early),
    False if content fits normally (caller should continue with the standard pipeline).
    """
    from vaig.agents.chunked import ChunkedProcessor
    from vaig.agents.orchestrator import Orchestrator

    processor = ChunkedProcessor(client, settings)
    orchestrator = Orchestrator(client, settings)

    # Use the same system instruction the normal pipeline would use
    system_instruction = orchestrator.default_system_instruction()

    try:
        budget = processor.calculate_budget(
            system_instruction,
            question,
            model_id=model_id or settings.models.default,
        )
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[dim]Chunking budget calculation failed ({exc}), using normal pipeline[/dim]")
        return False

    if not processor.needs_chunking(context_str, budget):
        return False

    # ── Content exceeds context window — use Map-Reduce ───────
    chunks = processor.split_into_chunks(context_str, budget)
    total = len(chunks)

    console.print(
        f"\n[bold yellow]Large file detected[/bold yellow] — "
        f"splitting into [cyan]{total}[/cyan] chunks for analysis"
    )

    with console.status("[bold cyan]Analyzing chunks...[/bold cyan]") as status:

        def _on_progress(current: int, total: int) -> None:
            status.update(f"[bold cyan]Processing chunk {current}/{total}...[/bold cyan]")

        result = processor.process_ask(
            context_str,
            question,
            system_instruction,
            budget,
            model_id=model_id or settings.models.default,
            on_progress=_on_progress,
        )

    console.print()
    if result:
        console.print(Markdown(result))
    if output:
        _save_output(output, result)
    return True


async def _async_try_chunked_ask(
    client: GeminiClientProtocol,
    settings: Settings,
    question: str,
    context_str: str,
    *,
    model_id: str | None = None,
    output: Path | None = None,
) -> bool:
    """Async version of :func:`_try_chunked_ask`.

    Currently wraps the sync ChunkedProcessor via ``asyncio.to_thread()``
    since ``ChunkedProcessor`` doesn't have async methods yet (Phase 5).

    Returns True if chunking was performed, False if content fits normally.
    """
    import asyncio

    from vaig.agents.chunked import ChunkedProcessor
    from vaig.agents.orchestrator import Orchestrator

    processor = ChunkedProcessor(client, settings)
    orchestrator = Orchestrator(client, settings)
    system_instruction = orchestrator.default_system_instruction()

    try:
        budget = await asyncio.to_thread(
            processor.calculate_budget,
            system_instruction,
            question,
            model_id=model_id or settings.models.default,
        )
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[dim]Chunking budget calculation failed ({exc}), using normal pipeline[/dim]")
        return False

    if not processor.needs_chunking(context_str, budget):
        return False

    chunks = processor.split_into_chunks(context_str, budget)
    total = len(chunks)

    console.print(
        f"\n[bold yellow]Large file detected[/bold yellow] — "
        f"splitting into [cyan]{total}[/cyan] chunks for analysis (async)"
    )

    def _run_chunked() -> str:
        return processor.process_ask(
            context_str,
            question,
            system_instruction,
            budget,
            model_id=model_id or settings.models.default,
        )

    result = await asyncio.to_thread(_run_chunked)

    console.print()
    if result:
        console.print(Markdown(result))
    if output:
        _save_output(output, result)
    return True

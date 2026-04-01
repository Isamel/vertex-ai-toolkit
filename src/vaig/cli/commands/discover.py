"""Discover command — autonomous cluster health scanning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from vaig.cli import _helpers
from vaig.cli._completions import complete_namespace
from vaig.cli._helpers import (
    _apply_subcommand_log_flags,
    err_console,
    handle_cli_error,
    track_command,
)

# NOTE: These private helpers from live.py are shared orchestration utilities
# (GKE config building, tool-call persistence, skill execution).  They are
# imported with their leading-underscore names intentionally — a future
# refactor will move them to a shared module (e.g. _orchestration.py) so
# both live.py and discover.py import from a common location.
from vaig.cli.commands.live import (
    _create_tool_call_store,
    _execute_orchestrated_skill,
)
from vaig.core.gke import build_gke_config as _build_gke_config
from vaig.core.prompt_defense import _sanitize_namespace
from vaig.skills.discovery.prompts import SYSTEM_NAMESPACES_CSV

logger = logging.getLogger(__name__)

# ── Default auto-generated queries ────────────────────────────

_QUERY_SINGLE_NS = (
    "Scan namespace '{namespace}' and discover all workloads. "
    "Enumerate Deployments, StatefulSets, DaemonSets, and Services. "
    "Classify each workload as Healthy, Degraded, or Failing. "
    "Investigate any non-healthy workloads by checking pods, logs, events, "
    "and resource usage. Produce a comprehensive cluster health report."
)

_QUERY_ALL_NS = (
    "Scan ALL non-system namespaces and discover all workloads. "
    "Skip these system namespaces: {system_ns}. "
    "Enumerate Deployments, StatefulSets, DaemonSets, and Services in each namespace. "
    "Classify each workload as Healthy, Degraded, or Failing. "
    "Investigate any non-healthy workloads by checking pods, logs, events, "
    "and resource usage. Produce a comprehensive cluster health report."
)

_SKIP_HEALTHY_SUFFIX = (
    " Focus the report on 🟡 Degraded and 🔴 Failing workloads only. "
    "Do NOT include detailed output for healthy workloads — just a count."
)


def _build_discover_query(
    *,
    namespace: str | None = None,
    all_namespaces: bool = False,
    skip_healthy: bool = False,
) -> str:
    """Build the auto-generated investigation query for the discovery pipeline.

    Args:
        namespace: Target namespace (used when *all_namespaces* is False).
        all_namespaces: When True, scan all non-system namespaces.
        skip_healthy: When True, append instructions to omit healthy workloads.

    Returns:
        A natural-language query string for the agent pipeline.
    """
    if all_namespaces:
        query = _QUERY_ALL_NS.format(system_ns=SYSTEM_NAMESPACES_CSV)
    else:
        ns = namespace or "default"
        safe_ns = _sanitize_namespace(ns)
        if not safe_ns:
            raise typer.BadParameter(
                f"Invalid namespace name: {ns!r}. "
                "Namespace must contain only lowercase alphanumeric characters or hyphens, "
                "start and end with an alphanumeric character, and be at most 63 characters."
            )
        query = _QUERY_SINGLE_NS.format(namespace=safe_ns)

    if skip_healthy:
        query += _SKIP_HEALTHY_SUFFIX

    return query


def register(app: typer.Typer) -> None:
    """Register the discover command on the given Typer app."""

    @app.command()
    @track_command
    def discover(
        namespace: Annotated[
            str | None,
            typer.Option(
                "--namespace",
                "-n",
                help="Kubernetes namespace to scan (default: config default or 'default')",
                autocompletion=complete_namespace,
            ),
        ] = None,
        all_namespaces: Annotated[
            bool,
            typer.Option(
                "--all-namespaces",
                "-A",
                help="Scan all non-system namespaces",
            ),
        ] = False,
        skip_healthy: Annotated[
            bool,
            typer.Option(
                "--skip-healthy",
                help="Omit healthy workloads from the report — focus on issues only",
            ),
        ] = False,
        config: Annotated[
            str | None,
            typer.Option("--config", "-c", help="Path to config YAML"),
        ] = None,
        model: Annotated[
            str | None,
            typer.Option("--model", "-m", help="Model to use"),
        ] = None,
        output: Annotated[
            Path | None,
            typer.Option("--output", "-o", help="Save report to a file"),
        ] = None,
        format_: Annotated[
            str | None,
            typer.Option("--format", help="Export format: json, md, html"),
        ] = None,
        cluster: Annotated[
            str | None,
            typer.Option("--cluster", help="GKE cluster name (overrides config)"),
        ] = None,
        project: Annotated[
            str | None,
            typer.Option("--project", "-p", help="GCP project ID (overrides config)"),
        ] = None,
        location: Annotated[
            str | None,
            typer.Option("--location", help="GCP location (overrides config)"),
        ] = None,
        gke_project: Annotated[
            str | None,
            typer.Option(
                "--gke-project",
                help="GKE project ID (overrides gke.project_id; defaults to --project if unset)",
            ),
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
            typer.Option("--debug", "-d", help="Enable debug logging (DEBUG level)"),
        ] = False,
        summary: Annotated[
            bool,
            typer.Option("--summary", help="Show compact summary instead of full report"),
        ] = False,
        no_bell: Annotated[
            bool,
            typer.Option("--no-bell", help="Suppress terminal bell after pipeline completes"),
        ] = False,
        open_browser: Annotated[
            bool,
            typer.Option(
                "--open",
                "-O",
                help="Open HTML report in default browser (requires --format html)",
            ),
        ] = False,
        detailed: Annotated[
            bool,
            typer.Option(
                "--detailed",
                help="Show every tool call as it happens (verbose execution output)",
            ),
        ] = False,
    ) -> None:
        """Autonomously scan a Kubernetes cluster and discover workload health issues.

        Unlike ``vaig live`` which takes a specific question, ``vaig discover``
        auto-generates its investigation query and scans the cluster to find
        what's wrong.

        The pipeline runs 4 sequential agents:
        1. Inventory Scanner — enumerates all workloads
        2. Triage Classifier — classifies as Healthy / Degraded / Failing
        3. Deep Investigator — checks logs, events, metrics for non-healthy workloads
        4. Cluster Reporter — produces a structured health report

        Examples:
            vaig discover --namespace production
            vaig discover --all-namespaces
            vaig discover --namespace staging --skip-healthy
            vaig discover -A -o report.md
            vaig discover --namespace production --format html --open
        """
        _apply_subcommand_log_flags(verbose=verbose, debug=debug)

        # Validate --open requires --format html
        normalised_format_flag = format_.strip().lower() if format_ else None
        if open_browser and normalised_format_flag != "html":
            err_console.print(
                "[yellow]⚠ --open requires --format html — ignoring --open flag.[/yellow]"
            )
            open_browser = False

        try:  # ── CLI error boundary ──
            settings = _helpers._get_settings(config)

            # Eagerly initialize telemetry
            _helpers._init_telemetry(settings)
            _helpers._init_audit(settings)
            _helpers._check_platform_auth(settings)

            # Apply CLI overrides to settings
            effective_project = project
            if effective_project:
                settings.gcp.project_id = effective_project

            if gke_project:
                settings.gke.project_id = gke_project

            if gke_location:
                settings.gke.location = gke_location

            if location:
                settings.gcp.location = location

            if model:
                settings.models.default = model

            from vaig.core.container import build_container

            container = build_container(settings)
            client = container.gemini_client

            gke_config = _build_gke_config(
                settings,
                cluster=cluster,
                namespace=namespace,
            )

            # Load the discovery skill
            from vaig.skills.discovery.skill import DiscoverySkill

            discovery_skill = DiscoverySkill()

            # Auto-generate the investigation query
            effective_namespace = namespace or gke_config.default_namespace
            question = _build_discover_query(
                namespace=effective_namespace,
                all_namespaces=all_namespaces,
                skip_healthy=skip_healthy,
            )

            logger.info("Auto-generated discover query: %s", question[:120])

            # Tool result persistence
            tool_call_store = _create_tool_call_store(settings)

            # Execute the orchestrated discovery pipeline
            _execute_orchestrated_skill(
                client,
                settings,
                gke_config,
                discovery_skill,
                question,
                output=output,
                format_=format_,
                model_id=model,
                tool_call_store=tool_call_store,
                summary=summary,
                no_bell=no_bell,
                open_browser=open_browser,
                all_namespaces=all_namespaces,
                detailed=detailed,
            )
        except typer.Exit:
            raise  # Let typer exits pass through
        except Exception as exc:  # noqa: BLE001
            handle_cli_error(exc, debug=debug)

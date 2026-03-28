"""Doctor command — environment healthcheck for VAIG toolkit.

Runs a series of diagnostic checks to verify that the user's
environment is properly configured for using vaig.
"""

from __future__ import annotations

import functools
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any

import typer

from vaig.cli._helpers import (
    _apply_subcommand_log_flags,
    _get_settings,
    console,
    handle_cli_error,
    track_command,
)

logger = logging.getLogger(__name__)

# ── Check result model ────────────────────────────────────────


@dataclass(slots=True)
class CheckResult:
    """Result of a single diagnostic check."""

    name: str
    status: str  # "pass", "warn", "fail"
    message: str
    is_critical: bool = field(default=False)

    @property
    def icon(self) -> str:
        """Rich-formatted icon for the check status."""
        if self.status == "pass":
            return "[green]✓[/green]"
        if self.status == "warn":
            return "[yellow]⚠[/yellow]"
        return "[red]✗[/red]"


# ── Individual check functions ────────────────────────────────


def check_gcp_auth(settings: Any) -> CheckResult:
    """Check GCP Application Default Credentials."""
    name = "GCP Authentication"
    try:
        import google.auth  # noqa: I001

        _credentials, project = google.auth.default()
        effective_project = settings.gcp.project_id or project or "unknown"
        return CheckResult(
            name=name,
            status="pass",
            message=f"ADC valid, project: {effective_project}",
            is_critical=True,
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name=name,
            status="fail",
            message=f"ADC not configured ({exc}). Run: gcloud auth application-default login",
            is_critical=True,
        )


def check_vertex_ai(settings: Any) -> CheckResult:
    """Check Vertex AI API accessibility by counting tokens on the configured model."""
    name = "Vertex AI API"
    try:
        from vaig.core.auth import get_credentials

        credentials = get_credentials(settings)
        from google import genai

        client = genai.Client(
            vertexai=True,
            project=settings.gcp.project_id or None,
            location=settings.gcp.location,
            credentials=credentials,
        )
        model_id = settings.models.default
        # Lightweight API call — count_tokens is fast and cheap.
        client.models.count_tokens(
            model=model_id,
            contents="test",
        )
        return CheckResult(
            name=name,
            status="pass",
            message=f"{model_id} accessible",
            is_critical=True,
        )
    except Exception as exc:  # noqa: BLE001
        model_id = settings.models.default
        short_err = str(exc)[:120]
        return CheckResult(
            name=name,
            status="fail",
            message=f"{model_id} not reachable ({short_err})",
            is_critical=True,
        )


def check_gke_connectivity(settings: Any) -> CheckResult:
    """Check GKE / Kubernetes cluster connectivity."""
    name = "GKE Connectivity"
    try:
        from kubernetes import client as k8s_client  # noqa: I001
    except ImportError:
        return CheckResult(
            name=name,
            status="warn",
            message="kubernetes SDK not installed (pip install vertex-ai-toolkit[live])",
        )

    try:
        from vaig.tools.base import ToolResult
        from vaig.tools.gke._clients import _InClusterClient, _load_k8s_config

        result = _load_k8s_config(settings.gke)

        if isinstance(result, ToolResult):
            short_err = str(result.output)[:120]
            return CheckResult(
                name=name,
                status="fail",
                message=f"cannot connect to cluster ({short_err})",
            )

        if isinstance(result, _InClusterClient):
            api_client = result.api_client
        else:
            # result is a Configuration object
            api_client = k8s_client.ApiClient(result)

        v1 = k8s_client.VersionApi(api_client)
        version_info = v1.get_code()
        cluster_name = settings.gke.cluster_name or "connected"

        # Try autopilot detection
        autopilot_label = ""
        try:
            from vaig.tools.gke._clients import detect_autopilot

            is_autopilot = detect_autopilot(settings.gke)
            if is_autopilot is True:
                autopilot_label = " (Autopilot)"
            elif is_autopilot is False:
                autopilot_label = " (Standard)"
        except Exception:  # noqa: BLE001
            pass

        return CheckResult(
            name=name,
            status="pass",
            message=f"cluster: {cluster_name}{autopilot_label}, k8s {version_info.git_version}",
        )
    except Exception as exc:  # noqa: BLE001
        short_err = str(exc)[:120]
        return CheckResult(
            name=name,
            status="fail",
            message=f"cannot connect to cluster ({short_err})",
        )


def check_cloud_logging(settings: Any) -> CheckResult:
    """Check Cloud Logging API availability."""
    name = "Cloud Logging"
    try:
        from vaig.tools.gcloud_tools import _get_logging_client

        project = settings.gcp.project_id or None
        _, err = _get_logging_client(project)
        if err:
            return CheckResult(name=name, status="fail", message=err)
        return CheckResult(name=name, status="pass", message="API enabled, client ready")
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name=name,
            status="fail",
            message=f"check failed ({exc})",
        )


def check_cloud_monitoring(settings: Any) -> CheckResult:
    """Check Cloud Monitoring API availability."""
    name = "Cloud Monitoring"
    try:
        from vaig.tools.gcloud_tools import _get_monitoring_client

        _, err = _get_monitoring_client()
        if err:
            return CheckResult(name=name, status="fail", message=err)
        return CheckResult(name=name, status="pass", message="API enabled, client ready")
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name=name,
            status="fail",
            message=f"check failed ({exc})",
        )


def check_helm(settings: Any) -> CheckResult:
    """Check Helm integration status."""
    name = "Helm Integration"
    if not settings.helm.enabled:
        return CheckResult(name=name, status="warn", message="disabled (set helm.enabled=true)")

    helm_path = shutil.which("helm")
    if not helm_path:
        return CheckResult(
            name=name,
            status="warn",
            message="enabled, but helm binary not found in PATH",
        )
    return CheckResult(name=name, status="pass", message=f"enabled, binary: {helm_path}")


def check_argocd(settings: Any) -> CheckResult:
    """Check ArgoCD integration status."""
    name = "ArgoCD Integration"
    if not settings.argocd.enabled:
        return CheckResult(
            name=name,
            status="warn",
            message="disabled (set argocd.enabled=true)",
        )
    # ArgoCD is enabled — check if server and token are configured
    if not settings.argocd.server:
        return CheckResult(
            name=name,
            status="warn",
            message="enabled, but argocd.server not configured",
        )
    return CheckResult(
        name=name,
        status="pass",
        message=f"enabled, server: {settings.argocd.server}",
    )


def check_datadog(settings: Any) -> CheckResult:
    """Check Datadog API integration status."""
    name = "Datadog Integration"
    if not settings.datadog.enabled:
        has_keys = bool(settings.datadog.api_key and settings.datadog.app_key)
        if has_keys:
            return CheckResult(
                name=name,
                status="warn",
                message="keys present but disabled",
            )
        return CheckResult(
            name=name,
            status="warn",
            message="disabled (set datadog api_key and app_key to enable)",
        )

    # Check SSL verify setting
    ssl_verify = settings.datadog.ssl_verify
    ssl_note = ""
    if ssl_verify is False:
        ssl_note = " (ssl_verify=False)"
    elif isinstance(ssl_verify, str):
        ssl_note = f" (custom CA: {ssl_verify})"

    return CheckResult(
        name=name,
        status="pass",
        message=f"enabled, site: {settings.datadog.site}{ssl_note}",
    )


def check_optional_deps() -> CheckResult:
    """Check importability of key optional packages."""
    name = "Optional deps"
    available: list[str] = []
    missing: list[str] = []

    packages = {
        "kubernetes": "kubernetes",
        "google.cloud.logging": "google-cloud-logging",
        "google.cloud.monitoring_v3": "google-cloud-monitoring",
        "datadog_api_client": "datadog-api-client",
    }

    for import_name, display_name in packages.items():
        try:
            __import__(import_name)
            available.append(display_name)
        except ImportError:
            missing.append(display_name)

    if missing:
        return CheckResult(
            name=name,
            status="warn",
            message=f"installed: {', '.join(available) or 'none'}; missing: {', '.join(missing)}",
        )
    return CheckResult(
        name=name,
        status="pass",
        message=", ".join(available),
    )


def check_mcp(settings: Any) -> CheckResult:
    """Check MCP (Model Context Protocol) server configuration."""
    name = "MCP Servers"
    if not settings.mcp.enabled:
        return CheckResult(
            name=name,
            status="warn",
            message="disabled",
        )
    server_count = len(settings.mcp.servers)
    if server_count == 0:
        return CheckResult(
            name=name,
            status="warn",
            message="enabled, but no servers configured",
        )
    server_names = ", ".join(s.name for s in settings.mcp.servers)
    return CheckResult(
        name=name,
        status="pass",
        message=f"enabled, {server_count} server(s): {server_names}",
    )


# ── Command registration ─────────────────────────────────────


def register(app: typer.Typer) -> None:
    """Register the doctor command on the given Typer app."""

    @app.command()
    @track_command
    def doctor(
        config: Annotated[
            str | None,
            typer.Option("--config", "-c", help="Path to config YAML"),
        ] = None,
        project: Annotated[
            str | None,
            typer.Option("--project", "-p", help="GCP project ID (overrides config)"),
        ] = None,
        cluster: Annotated[
            str | None,
            typer.Option("--cluster", help="GKE cluster name (overrides config)"),
        ] = None,
        location: Annotated[
            str | None,
            typer.Option("--location", help="GCP location (overrides config)"),
        ] = None,
        gke_project: Annotated[
            str | None,
            typer.Option(
                "--gke-project",
                help="GKE project ID (overrides gke.project_id)",
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
    ) -> None:
        """Run diagnostic checks on your VAIG environment.

        Verifies GCP authentication, Vertex AI API access, Kubernetes
        connectivity, observability integrations, and optional dependencies.

        Examples:
            vaig doctor
            vaig doctor --project my-project
            vaig doctor --config ~/custom-config.yaml
        """
        _apply_subcommand_log_flags(verbose=verbose, debug=debug)

        try:
            settings = _get_settings(config)

            # Apply CLI overrides
            if project:
                settings.gcp.project_id = project
            if location:
                settings.gcp.location = location
            if cluster:
                settings.gke.cluster_name = cluster
            if gke_project:
                settings.gke.project_id = gke_project
            if gke_location:
                settings.gke.location = gke_location

            console.print()
            console.print("[bold cyan]vaig doctor[/bold cyan] — environment healthcheck")
            console.print()

            # Run all checks sequentially — each is independent
            checks: list[CheckResult] = []
            check_functions: list[Callable[[], CheckResult]] = [
                functools.partial(check_gcp_auth, settings),
                functools.partial(check_vertex_ai, settings),
                functools.partial(check_gke_connectivity, settings),
                functools.partial(check_cloud_logging, settings),
                functools.partial(check_cloud_monitoring, settings),
                functools.partial(check_helm, settings),
                functools.partial(check_argocd, settings),
                functools.partial(check_datadog, settings),
                check_optional_deps,
                functools.partial(check_mcp, settings),
            ]

            for check_fn in check_functions:
                try:
                    result = check_fn()
                except Exception as exc:  # noqa: BLE001
                    # Absolute safety net — should never reach here, but if it does
                    # we still continue with remaining checks.
                    result = CheckResult(
                        name="Unknown",
                        status="fail",
                        message=f"unexpected error ({exc})",
                    )
                checks.append(result)
                # Print immediately for real-time feedback
                console.print(f"  {result.icon} {result.name:<24s}— {result.message}")

            console.print()

            # Summary
            passes = sum(1 for c in checks if c.status == "pass")
            warnings = sum(1 for c in checks if c.status == "warn")
            failures = sum(1 for c in checks if c.status == "fail")

            summary_parts: list[str] = []
            if passes:
                summary_parts.append(f"[green]{passes} passed[/green]")
            if warnings:
                summary_parts.append(f"[yellow]{warnings} warnings[/yellow]")
            if failures:
                summary_parts.append(f"[red]{failures} failed[/red]")

            console.print(f"  {', '.join(summary_parts)}")
            console.print()

            # Critical checks: GCP Auth and Vertex AI
            critical_checks = [c for c in checks if c.is_critical]
            has_critical_failure = any(c.status == "fail" for c in critical_checks)

            if has_critical_failure:
                raise typer.Exit(code=1)

        except typer.Exit:
            raise  # Let typer exits pass through
        except Exception as exc:  # noqa: BLE001
            handle_cli_error(exc, debug=debug)

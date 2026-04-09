"""Check command — Terraform-compatible health check with JSON output.

Runs the service-health pipeline via ``execute_skill_headless()`` and
writes a flat, string-only JSON object to stdout.  All Rich/ANSI output
goes to stderr.  Exit codes: 0=healthy, 1=unhealthy, 2=error/timeout.

Designed for consumption by Terraform's ``external`` data source and CI
pipelines.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Annotated

import typer

from vaig.cli import _helpers
from vaig.cli._helpers import (
    _apply_subcommand_log_flags,
    track_command,
)
from vaig.cli.check_schema import CheckOutput
from vaig.core.gke import build_gke_config as _build_gke_config

logger = logging.getLogger(__name__)

# ── Cache helpers ─────────────────────────────────────────────

_CACHE_DIR = Path.home() / ".cache" / "vaig" / "check"


def _cache_key(
    namespace: str | None,
    cluster: str | None,
    project: str | None,
    location: str | None = None,
) -> str:
    """SHA-256 hash of the resolved (namespace, cluster, project, location) tuple."""
    raw = f"{namespace or ''}:{cluster or ''}:{project or ''}:{location or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _read_cache(key: str, ttl: int) -> CheckOutput | None:
    """Read cached result if it exists and is within TTL."""
    cache_file = _CACHE_DIR / f"{key}.json"
    if not cache_file.exists():
        return None
    age = time.time() - cache_file.stat().st_mtime
    if age >= ttl:
        return None
    try:
        data = json.loads(cache_file.read_text())
        return CheckOutput(**data)
    except (json.JSONDecodeError, ValueError, KeyError):
        return None


def _write_cache(key: str, output: CheckOutput) -> None:
    """Write result to cache file."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = _CACHE_DIR / f"{key}.json"
        cache_file.write_text(output.model_dump_json())
    except OSError:
        logger.debug("Failed to write cache file", exc_info=True)


# ── Exit code mapping ────────────────────────────────────────


def _exit_code_for_status(status: str) -> int:
    """Map CheckOutput.status to process exit code.

    0 = HEALTHY, 1 = DEGRADED/CRITICAL/UNKNOWN, 2 = ERROR/TIMEOUT.
    """
    if status == "HEALTHY":
        return 0
    if status in {"ERROR", "TIMEOUT"}:
        return 2
    return 1


# ── Command registration ─────────────────────────────────────


def register(app: typer.Typer) -> None:
    """Register the ``check`` command on the given Typer app."""

    @app.command()
    @track_command
    def check(
        namespace: Annotated[
            str | None,
            typer.Option(
                "--namespace",
                "-n",
                help="Kubernetes namespace to check (default: config default or 'default')",
            ),
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
            typer.Option("--location", "-l", help="GKE cluster location/zone/region (overrides config)"),
        ] = None,
        timeout: Annotated[
            int,
            typer.Option("--timeout", help="Timeout in seconds for the health check pipeline"),
        ] = 120,
        cached: Annotated[
            bool,
            typer.Option("--cached", help="Return cached result if within TTL"),
        ] = False,
        cache_ttl: Annotated[
            int,
            typer.Option("--cache-ttl", help="Cache TTL in seconds (default 300)"),
        ] = 300,
        config: Annotated[
            str | None,
            typer.Option("--config", "-c", help="Path to config YAML"),
        ] = None,
        model: Annotated[
            str | None,
            typer.Option("--model", "-m", help="Model to use"),
        ] = None,
        verbose: Annotated[
            bool,
            typer.Option("--verbose", "-V", help="Enable verbose logging (INFO level, stderr only)"),
        ] = False,
        debug: Annotated[
            bool,
            typer.Option("--debug", "-d", help="Enable debug logging (DEBUG level, stderr only)"),
        ] = False,
    ) -> None:
        """Run a health check and output Terraform-compatible JSON.

        Executes the service-health pipeline and writes a flat JSON object
        with string-only values to stdout.  Designed for Terraform's
        ``external`` data source.

        Exit codes: 0=healthy, 1=unhealthy, 2=error/timeout.

        Examples:
            vaig check --namespace production
            vaig check --cached --cache-ttl 600
            vaig check --timeout 60 --project my-gcp-project
            vaig check --location us-central1 --cluster my-cluster
        """
        _apply_subcommand_log_flags(verbose=verbose, debug=debug)

        # ── Resolve settings and GKE config FIRST for cache key ──
        settings = _helpers._get_settings(config)

        if project:
            settings.gcp.project_id = project
        if model:
            settings.models.default = model

        gke_config = _build_gke_config(
            settings,
            cluster=cluster,
            namespace=namespace,
            location=location,
        )

        # Cache key uses RESOLVED values (after settings/GKE defaults applied)
        c_key = _cache_key(
            gke_config.default_namespace,
            gke_config.cluster_name,
            gke_config.project_id,
            gke_config.location,
        )

        # ── Cache hit path ───────────────────────────────────
        if cached:
            cached_result = _read_cache(c_key, cache_ttl)
            if cached_result is not None:
                cached_result.cached = "true"
                sys.stdout.write(cached_result.model_dump_json() + "\n")
                sys.stdout.flush()
                exit_code = _exit_code_for_status(cached_result.status)
                if exit_code != 0:
                    raise SystemExit(exit_code)
                return

        # ── Fresh pipeline execution ─────────────────────────
        try:
            from vaig.skills.service_health.skill import ServiceHealthSkill

            skill = ServiceHealthSkill()
            effective_namespace = namespace or gke_config.default_namespace
            query = f"Perform a comprehensive health check of the '{effective_namespace}' namespace."

            # ── Timeout-wrapped execution ────────────────────
            import asyncio

            from vaig.core.headless import execute_skill_headless

            async def _run_with_timeout() -> object:
                loop = asyncio.get_running_loop()
                return await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: execute_skill_headless(settings, skill, query, gke_config),
                    ),
                    timeout=timeout,
                )

            try:
                orch_result = asyncio.run(_run_with_timeout())
            except TimeoutError:
                output = CheckOutput.from_error("TIMEOUT", f"Health check timed out after {timeout}s")
                sys.stdout.write(output.model_dump_json() + "\n")
                sys.stdout.flush()
                _write_cache(c_key, output)
                raise SystemExit(2)  # noqa: B904, TRY200

            # ── Map result ───────────────────────────────────
            report = getattr(orch_result, "structured_report", None)
            if report is None:
                output = CheckOutput.from_error("ERROR", "Pipeline completed but no structured report was produced")
            else:
                output = CheckOutput.from_health_report(report)

            _write_cache(c_key, output)

            sys.stdout.write(output.model_dump_json() + "\n")
            sys.stdout.flush()

            exit_code = _exit_code_for_status(output.status)
            if exit_code != 0:
                raise SystemExit(exit_code)

        except (SystemExit, KeyboardInterrupt):
            raise  # Let SystemExit and KeyboardInterrupt pass through
        except Exception as exc:  # noqa: BLE001
            err_msg = f"{type(exc).__name__}: {exc}"
            logger.error("Check command failed: %s", err_msg)
            output = CheckOutput.from_error("ERROR", err_msg)
            sys.stdout.write(output.model_dump_json() + "\n")
            sys.stdout.flush()
            _write_cache(c_key, output)
            raise SystemExit(2)  # noqa: B904, TRY200

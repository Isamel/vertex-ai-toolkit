"""Shared GKE helpers — pure functions reusable by CLI and web.

Extracted from ``vaig.cli.commands.live`` so that both the CLI ``live``
command and the web live-mode route share the same logic for building
:class:`~vaig.core.config.GKEConfig` and registering infrastructure
tools.

Both functions are intentionally free of CLI-specific imports (Typer,
Rich) and web-specific imports (FastAPI, Starlette).  They depend only
on ``vaig.core`` and ``vaig.tools``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vaig.core.config import GKEConfig, Settings
    from vaig.tools.base import ToolRegistry

__all__ = [
    "build_gke_config",
    "register_live_tools",
]

logger = logging.getLogger(__name__)


def build_gke_config(
    settings: Settings,
    *,
    cluster: str | None = None,
    namespace: str | None = None,
    project_id: str | None = None,
    location: str | None = None,
) -> GKEConfig:
    """Build a :class:`GKEConfig`, applying overrides on top of config-file defaults.

    Args:
        settings: Application settings (contains ``gke`` section).
        cluster: Optional cluster name override.
        namespace: Optional default namespace override.
        project_id: Optional GCP project ID override.
        location: Optional GKE cluster location/zone/region override.

    Returns:
        :class:`GKEConfig` with overrides applied.
    """
    from vaig.core.config import GKEConfig as _GKEConfig

    gke = settings.gke

    return _GKEConfig(
        cluster_name=cluster or gke.cluster_name,
        project_id=project_id or gke.project_id or settings.gcp.project_id,
        default_namespace=namespace or gke.default_namespace,
        location=location or gke.location,
        kubeconfig_path=gke.kubeconfig_path,
        context=gke.context,
        log_limit=gke.log_limit,
        metrics_interval_minutes=gke.metrics_interval_minutes,
        proxy_url=gke.proxy_url,
        impersonate_sa=gke.impersonate_sa,
        exec_enabled=gke.exec_enabled,
        # Helm / ArgoCD — merge Settings-level config into GKEConfig flags
        helm_enabled=settings.helm.enabled,
        argocd_enabled=settings.argocd.enabled,
        argocd_server=settings.argocd.server,
        argocd_token=settings.argocd.token,
        argocd_context=settings.argocd.context,
        argocd_namespace=settings.argocd.namespace,
        argocd_verify_ssl=settings.argocd.verify_ssl,
    )


def register_live_tools(
    gke_config: GKEConfig,
    settings: Settings | None = None,
    *,
    repo: str | None = None,
    repo_ref: str = "HEAD",
) -> ToolRegistry:
    """Create a :class:`ToolRegistry` and register GKE + GCloud + plugin tools.

    Follows the same ``try/except ImportError`` pattern as
    ``InfraAgent._register_tools()`` so missing optional dependencies
    degrade gracefully.

    Args:
        gke_config: GKE configuration for tool creation.
        settings: Full application settings (used for plugin tool loading).
        repo: Optional GitHub repo in ``owner/repo`` format. When provided
            and ``settings.github.enabled`` is True, repo tools are injected
            into the analyzer/hypothesis loop.
        repo_ref: Git ref to use for repo correlation (default: ``HEAD``).

    Returns:
        Populated :class:`ToolRegistry` (may be empty if no optional deps
        are installed).
    """
    from vaig.tools.base import ToolRegistry as _ToolRegistry

    registry = _ToolRegistry()

    # Resolve GKE-specific credentials (SA impersonation or ADC)
    gke_credentials = None
    if settings is not None:
        from vaig.core.auth import get_gke_credentials

        gke_credentials = get_gke_credentials(settings)

    # GKE tools — requires 'kubernetes' package
    try:
        from vaig.tools.gke_tools import create_gke_tools  # noqa: WPS433

        for tool in create_gke_tools(gke_config):
            registry.register(tool)
    except ImportError as exc:
        logger.warning("Could not load GKE tools: %s", exc)

    # GCP observability tools — requires google-cloud-logging / google-cloud-monitoring
    try:
        from vaig.tools.gcloud_tools import create_gcloud_tools  # noqa: WPS433

        for tool in create_gcloud_tools(
            project=gke_config.project_id,
            log_limit=gke_config.log_limit,
            metrics_interval_minutes=gke_config.metrics_interval_minutes,
            credentials=gke_credentials,
        ):
            registry.register(tool)
    except ImportError as exc:
        logger.warning("Could not load GCloud observability tools: %s", exc)

    # Plugin tools — MCP auto-registration and Python module plugins
    if settings is not None:
        try:
            from vaig.tools.plugin_loader import load_all_plugin_tools  # noqa: WPS433

            for tool in load_all_plugin_tools(settings):
                registry.register(tool)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to load plugin tools for live mode. Skipping.",
                exc_info=True,
            )

    # Alert correlation tools — PagerDuty, OpsGenie, Slack (incident management)
    if settings is not None:
        try:
            from vaig.tools.integrations._registry import create_alert_correlation_tools  # noqa: WPS433

            for tool in create_alert_correlation_tools(settings):
                registry.register(tool)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to load alert correlation tools. Skipping.",
                exc_info=True,
            )

    # GitHub repo correlation tools — injected when --repo is provided and github is enabled.
    # Only active when both conditions hold; otherwise behavior is identical to pre-Phase-6.
    if settings is not None and repo is not None:
        try:
            from vaig.tools.integrations._github_registry import create_github_repo_tools  # noqa: WPS433

            if settings.github.enabled:
                for tool in create_github_repo_tools(settings):
                    registry.register(tool)
                logger.info(
                    "GitHub repo tools registered for %s @ %s (allowed_repos=%s)",
                    repo,
                    repo_ref,
                    settings.github.allowed_repos or "all",
                )
            else:
                logger.debug(
                    "Skipping GitHub repo tools: github.enabled=False (--repo=%s provided but no token configured)",
                    repo,
                )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to load GitHub repo tools. Skipping.",
                exc_info=True,
            )

    return registry

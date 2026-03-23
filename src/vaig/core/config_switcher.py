"""Runtime config switching — stateless functions to mutate settings and reinitialize components.

Each ``switch_*`` function follows the same pattern:
1. Validate the new value
2. Mutate the Settings singleton in place
3. Reinitialize dependent components (if provided)
4. Rollback on failure

All parameters for client/reinit are optional so callers can use these
functions without needing a fully initialized client (e.g. in tests or
before the first ``/ask`` call).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vaig.core.config import Settings
    from vaig.core.protocols import GeminiClientProtocol

logger = logging.getLogger(__name__)


# ── Result type ──────────────────────────────────────────────


@dataclass
class SwitchResult:
    """Result of a config switch operation."""

    success: bool
    field: str  # what changed (e.g. "project", "location")
    old_value: str
    new_value: str
    message: str  # user-facing message
    reinitialized: list[str] = dataclass_field(default_factory=list)  # what was reinitialized


# ── Switch functions ─────────────────────────────────────────


def switch_project(
    settings: Settings,
    new_project: str,
    client: GeminiClientProtocol | None = None,
) -> SwitchResult:
    """Switch GCP project — mutates settings, reinits client, clears caches.

    Args:
        settings: The application settings to mutate.
        new_project: New GCP project ID.
        client: Optional client to reinitialize.

    Returns:
        SwitchResult with details of the switch.
    """
    if not new_project or not new_project.strip():
        return SwitchResult(
            success=False,
            field="project",
            old_value=settings.gcp.project_id,
            new_value=new_project,
            message="Project ID cannot be empty.",
        )

    new_project = new_project.strip()
    old_project = settings.gcp.project_id

    if new_project == old_project:
        return SwitchResult(
            success=True,
            field="project",
            old_value=old_project,
            new_value=new_project,
            message=f"Already using project '{new_project}'.",
        )

    # Validate against catalog (warn, don't block)
    warning = ""
    if settings.gcp.available_projects:
        known_ids = [p.project_id for p in settings.gcp.available_projects]
        if new_project not in known_ids:
            warning = f" (warning: '{new_project}' is not in available_projects catalog)"

    # Phase 1: Mutate settings
    settings.gcp.project_id = new_project
    # Keep GKE project in sync
    old_gke_project = settings.gke.project_id
    settings.gke.project_id = new_project

    # Phase 2: Reinit client if provided
    reinitialized: list[str] = []
    if client is not None:
        try:
            client.reinitialize(project=new_project)
            reinitialized.append("GeminiClient")
        except Exception as exc:  # noqa: BLE001
            # Rollback on failure
            logger.error("Client reinit failed for project '%s': %s", new_project, exc)
            settings.gcp.project_id = old_project
            settings.gke.project_id = old_gke_project
            return SwitchResult(
                success=False,
                field="project",
                old_value=old_project,
                new_value=new_project,
                message=f"Failed to reinitialize client: {exc}",
            )

    logger.info("Project switched: %s → %s", old_project, new_project)
    return SwitchResult(
        success=True,
        field="project",
        old_value=old_project,
        new_value=new_project,
        message=f"Project switched to '{new_project}'.{warning}",
        reinitialized=reinitialized,
    )


def switch_location(
    settings: Settings,
    new_location: str,
    client: GeminiClientProtocol | None = None,
) -> SwitchResult:
    """Switch GCP location — mutates settings, reinits client.

    Args:
        settings: The application settings to mutate.
        new_location: New GCP location (e.g. ``us-central1``).
        client: Optional client to reinitialize.

    Returns:
        SwitchResult with details of the switch.
    """
    if not new_location or not new_location.strip():
        return SwitchResult(
            success=False,
            field="location",
            old_value=settings.gcp.location,
            new_value=new_location,
            message="Location cannot be empty.",
        )

    new_location = new_location.strip()
    old_location = settings.gcp.location

    if new_location == old_location:
        return SwitchResult(
            success=True,
            field="location",
            old_value=old_location,
            new_value=new_location,
            message=f"Already using location '{new_location}'.",
        )

    # Phase 1: Mutate settings
    settings.gcp.location = new_location

    # Phase 2: Reinit client if provided (location changes require new SDK client)
    reinitialized: list[str] = []
    if client is not None:
        try:
            client.reinitialize(location=new_location)
            reinitialized.append("GeminiClient")
        except Exception as exc:  # noqa: BLE001
            # Rollback on failure
            logger.error("Client reinit failed for location '%s': %s", new_location, exc)
            settings.gcp.location = old_location
            return SwitchResult(
                success=False,
                field="location",
                old_value=old_location,
                new_value=new_location,
                message=f"Failed to reinitialize client: {exc}",
            )

    logger.info("Location switched: %s → %s", old_location, new_location)
    return SwitchResult(
        success=True,
        field="location",
        old_value=old_location,
        new_value=new_location,
        message=f"Location switched to '{new_location}'.",
        reinitialized=reinitialized,
    )


def switch_cluster(
    settings: Settings,
    new_cluster: str,
    new_context: str | None = None,
) -> SwitchResult:
    """Switch GKE cluster — mutates settings, clears K8s caches.

    No client reinit needed — GKE tools create fresh clients from the
    updated settings on next use.

    Args:
        settings: The application settings to mutate.
        new_cluster: New GKE cluster name.
        new_context: Optional new kubeconfig context.

    Returns:
        SwitchResult with details of the switch.
    """
    if not new_cluster or not new_cluster.strip():
        return SwitchResult(
            success=False,
            field="cluster",
            old_value=settings.gke.cluster_name,
            new_value=new_cluster,
            message="Cluster name cannot be empty.",
        )

    new_cluster = new_cluster.strip()
    old_cluster = settings.gke.cluster_name

    if new_cluster == old_cluster and (new_context is None or new_context == settings.gke.context):
        return SwitchResult(
            success=True,
            field="cluster",
            old_value=old_cluster,
            new_value=new_cluster,
            message=f"Already using cluster '{new_cluster}'.",
        )

    # Mutate settings
    settings.gke.cluster_name = new_cluster
    old_context = settings.gke.context
    if new_context is not None:
        settings.gke.context = new_context

    # Clear GKE caches so next tool invocation picks up new cluster
    reinitialized: list[str] = []
    try:
        from vaig.tools.gke._cache import clear_discovery_cache
        from vaig.tools.gke._clients import clear_autopilot_cache, clear_k8s_client_cache

        clear_k8s_client_cache()
        reinitialized.append("k8s_client_cache")
        clear_autopilot_cache()
        reinitialized.append("autopilot_cache")
        clear_discovery_cache()
        reinitialized.append("discovery_cache")
    except ImportError:
        logger.debug("GKE tools not available — cache clear skipped")

    context_msg = ""
    if new_context is not None and new_context != old_context:
        context_msg = f" (context: '{new_context}')"

    logger.info("Cluster switched: %s → %s%s", old_cluster, new_cluster, context_msg)
    return SwitchResult(
        success=True,
        field="cluster",
        old_value=old_cluster,
        new_value=new_cluster,
        message=f"Cluster switched to '{new_cluster}'.{context_msg}",
        reinitialized=reinitialized,
    )


def get_config_snapshot(settings: Settings) -> dict[str, str]:
    """Return a snapshot of the current config values.

    Used by the ``/config`` command to display current state.

    Args:
        settings: The application settings.

    Returns:
        Dict with current config values.
    """
    return {
        "project": settings.gcp.project_id,
        "location": settings.gcp.location,
        "fallback_location": settings.gcp.fallback_location,
        "model": settings.models.default,
        "cluster": settings.gke.cluster_name,
        "context": settings.gke.context,
        "gke_project": settings.gke.project_id,
        "gke_location": settings.gke.location,
    }

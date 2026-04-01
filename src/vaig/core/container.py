"""Service container — frozen dataclass holding all DI dependencies.

The ``ServiceContainer`` is a frozen (immutable) dataclass that aggregates
all service dependencies for the application.  A ``build_container()``
factory function constructs the container from a ``Settings`` instance,
creating real service instances.

This is the **Composition Root** — the single place where concrete
implementations are wired together.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vaig.core.event_bus import EventBus
from vaig.core.protocols import GCPClientProvider, GeminiClientProtocol, K8sClientProvider

if TYPE_CHECKING:
    from vaig.core.config import Settings

__all__ = [
    "ServiceContainer",
    "build_container",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServiceContainer:
    """Frozen dataclass holding all application dependencies.

    Fields are typed against protocols, not concrete classes, so that
    tests can inject mocks without touching the container internals.

    Attributes:
        settings: Application configuration.
        gemini_client: AI generation client (protocol-typed).
        k8s_provider: Kubernetes client provider. Can be ``None`` if not required (e.g., in tests).
        gcp_provider: GCP observability client provider. Can be ``None`` if not required (e.g., in tests).
        event_bus: Process-wide event bus for domain events.
        quota_checker: Rate-limit quota enforcer. ``None`` when rate limiting is disabled.
        platform_auth: Platform authentication manager. ``None`` when platform mode is disabled.
    """

    settings: Settings
    gemini_client: GeminiClientProtocol
    k8s_provider: K8sClientProvider | None
    gcp_provider: GCPClientProvider | None
    event_bus: EventBus
    quota_checker: object | None = None
    platform_auth: object | None = None


def build_container(settings: Settings) -> ServiceContainer:
    """Build a ``ServiceContainer`` from application settings.

    Creates real instances of all services:

    - ``GeminiClient`` from ``settings``
    - ``EventBus`` singleton via ``EventBus.get()``
    - ``DefaultK8sClientProvider`` for K8s client creation/caching
    - ``DefaultGCPClientProvider`` for GCP observability client caching

    Args:
        settings: Fully-loaded application configuration.

    Returns:
        A frozen ``ServiceContainer`` with all dependencies wired.
    """
    from vaig.core.client import GeminiClient
    from vaig.tools.gcloud_tools import DefaultGCPClientProvider
    from vaig.tools.gke._clients import DefaultK8sClientProvider

    # ── Optional: rate-limit quota checker ────────────────────
    quota_checker = None
    if settings.rate_limit.enabled:
        try:
            from vaig.core.auth import get_credentials
            from vaig.core.quota import QuotaChecker

            credentials = get_credentials(settings)
            quota_checker = QuotaChecker(settings, credentials)
            logger.info("QuotaChecker enabled — GCS policy from %s/%s",
                        settings.rate_limit.policy_gcs_bucket,
                        settings.rate_limit.policy_gcs_path)
        except Exception as exc:  # noqa: BLE001
            msg = (
                "Rate limiting is enabled but QuotaChecker failed to initialize. "
                "Check GCS connectivity and [audit] extras installation."
            )
            raise RuntimeError(msg) from exc

    gemini_client = GeminiClient(settings, quota_checker=quota_checker)
    event_bus = EventBus.get()
    k8s_provider = DefaultK8sClientProvider()
    gcp_provider = DefaultGCPClientProvider()

    logger.info("ServiceContainer built — gemini_client=%s", type(gemini_client).__name__)

    return ServiceContainer(
        settings=settings,
        gemini_client=gemini_client,
        k8s_provider=k8s_provider,
        gcp_provider=gcp_provider,
        event_bus=event_bus,
        quota_checker=quota_checker,
    )

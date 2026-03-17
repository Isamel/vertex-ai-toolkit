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
        k8s_provider: Kubernetes client provider (``None`` when K8s is unavailable).
        gcp_provider: GCP observability client provider (``None`` when GCP SDKs unavailable).
        event_bus: Process-wide event bus for domain events.
    """

    settings: Settings
    gemini_client: GeminiClientProtocol
    k8s_provider: K8sClientProvider | None
    gcp_provider: GCPClientProvider | None
    event_bus: EventBus


def build_container(settings: Settings) -> ServiceContainer:
    """Build a ``ServiceContainer`` from application settings.

    Creates real instances of all services:

    - ``GeminiClient`` from ``settings``
    - ``EventBus`` singleton via ``EventBus.get()``
    - ``K8sClientProvider`` and ``GCPClientProvider`` are set to ``None``
      for now — concrete provider implementations (``DefaultK8sClientProvider``,
      ``DefaultGCPClientProvider``) will be created in Batch 2 when those
      wrapper classes are implemented.

    Args:
        settings: Fully-loaded application configuration.

    Returns:
        A frozen ``ServiceContainer`` with all dependencies wired.
    """
    from vaig.core.client import GeminiClient

    gemini_client = GeminiClient(settings)
    event_bus = EventBus.get()

    logger.info("ServiceContainer built — gemini_client=%s", type(gemini_client).__name__)

    return ServiceContainer(
        settings=settings,
        gemini_client=gemini_client,
        k8s_provider=None,
        gcp_provider=None,
        event_bus=event_bus,
    )

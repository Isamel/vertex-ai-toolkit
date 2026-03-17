"""Shared test helpers — importable from any test module.

Unlike ``conftest.py`` (which relies on pytest's auto-discovery mechanism
and should NOT be imported directly), this module provides plain helper
functions that tests can import via::

    from tests._helpers import create_test_container

or, when ``tests/`` is on ``sys.path`` (the pytest default)::

    from _helpers import create_test_container
"""

from __future__ import annotations

from unittest.mock import MagicMock

from vaig.core.config import Settings
from vaig.core.event_bus import EventBus


def create_test_container(
    *,
    settings: Settings | None = None,
    gemini_client: MagicMock | None = None,
    k8s_provider: object | None = None,
    gcp_provider: object | None = None,
    event_bus: EventBus | None = None,
) -> MagicMock:
    """Build a mock ``ServiceContainer`` with sensible test defaults.

    Returns a ``MagicMock`` that mimics ``ServiceContainer`` with all
    fields populated.  Pass keyword overrides to customise individual
    fields — unprovided fields get safe defaults (``MagicMock`` gemini
    client, default ``Settings``, ``EventBus`` singleton, ``None``
    providers).

    Example::

        container = create_test_container()
        container = create_test_container(settings=my_settings)
        container = create_test_container(k8s_provider=mock_provider)
    """
    mock_container = MagicMock()
    mock_container.gemini_client = gemini_client or MagicMock()
    mock_container.settings = settings or Settings()
    mock_container.event_bus = event_bus or EventBus.get()
    mock_container.k8s_provider = k8s_provider
    mock_container.gcp_provider = gcp_provider
    return mock_container

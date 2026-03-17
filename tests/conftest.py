"""Shared pytest configuration and fixtures.

This module provides fixtures that are auto-discovered by pytest across all
test files under ``tests/``.  No explicit imports are needed — pytest injects
conftest fixtures by name.

Fixtures extracted here were previously duplicated in multiple test modules.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from vaig.core.client import GenerationResult
from vaig.core.config import Settings, reset_settings
from vaig.core.event_bus import EventBus
from vaig.core.telemetry import TelemetryCollector, reset_telemetry_collector
from vaig.session.store import SessionStore

# ── Settings singleton reset ────────────────────────────────
# Previously duplicated in 9 test files as ``_reset()``.
# Safe to run even when the singleton was never created — it
# simply sets the module-level ``_settings`` to ``None``.


@pytest.fixture(autouse=True)
def _reset_settings() -> Generator[None, None, None]:
    """Reset the Settings, Telemetry, and EventBus singletons between tests.

    This prevents cross-test contamination when any test constructs
    or mutates a ``Settings`` instance, and ensures orphaned aiosqlite
    connections from the telemetry collector are cleaned up.

    The EventBus singleton is also reset so that subscribers registered
    in one test do not leak into another.
    """
    reset_settings()
    EventBus._reset_singleton()
    yield
    EventBus._reset_singleton()
    reset_telemetry_collector()


# ── Session store ────────────────────────────────────────────
# Previously duplicated identically in test_session_store.py
# and test_session_improvements.py.


@pytest.fixture()
def store(tmp_path: Path) -> Generator[SessionStore, None, None]:
    """Create a ``SessionStore`` backed by a temporary SQLite database."""
    db = tmp_path / "test_sessions.db"
    s = SessionStore(db)
    yield s
    s.close()


# ── Telemetry collector ─────────────────────────────────────
# ``db_path`` and ``collector`` were duplicated identically in
# test_telemetry.py and test_cli_stats.py.


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Return a fresh temporary database path for telemetry tests."""
    return tmp_path / "test_telemetry.db"


@pytest.fixture()
def collector(db_path: Path) -> Generator[TelemetryCollector, None, None]:
    """Create an enabled ``TelemetryCollector`` with a small buffer for fast testing."""
    c = TelemetryCollector(db_path=db_path, enabled=True, buffer_size=5)
    yield c
    c.close()


# ── Mock GeminiClient (with reinitialize) ────────────────────
# Previously duplicated in test_config_switcher.py and
# test_repl_config_commands.py.  The version here is the superset
# (includes ``current_model``).


@pytest.fixture()
def mock_client() -> MagicMock:
    """Mock ``GeminiClient`` with ``reinitialize`` and ``current_model``."""
    client = MagicMock()
    client.reinitialize = MagicMock()
    client.current_model = "gemini-2.5-pro"
    return client


# ── Async fixtures ───────────────────────────────────────────
# These fixtures support pytest-asyncio (mode=auto, configured in
# pyproject.toml).  They provide async-aware store and collector
# instances with proper async cleanup.


@pytest.fixture()
async def async_store(tmp_path: Path) -> AsyncGenerator[SessionStore, None]:
    """Create a ``SessionStore`` with async cleanup for async tests.

    The store supports both sync and async methods.  This fixture
    ensures ``async_close()`` is called on teardown.
    """
    db = tmp_path / "test_async_sessions.db"
    s = SessionStore(db)
    yield s
    await s.async_close()


@pytest.fixture()
async def async_collector(tmp_path: Path) -> AsyncGenerator[TelemetryCollector, None]:
    """Create an enabled ``TelemetryCollector`` with async cleanup.

    Uses a small buffer (5) for fast flushing in tests.
    """
    c = TelemetryCollector(
        db_path=tmp_path / "test_async_telemetry.db",
        enabled=True,
        buffer_size=5,
    )
    yield c
    await c.async_close()


@pytest.fixture()
def mock_async_client() -> MagicMock:
    """Mock ``GeminiClient`` with both sync and async methods.

    Provides:
    - ``generate`` (sync) and ``async_generate`` (async) returning a ``GenerationResult``
    - ``generate_stream`` (sync) and ``async_generate_stream`` (async)
    - ``current_model`` property
    - ``reinitialize`` method
    """
    client = MagicMock()
    gen_result = GenerationResult(
        text="mocked response",
        model="gemini-2.5-pro",
        finish_reason="STOP",
        usage={"prompt_tokens": 10, "candidates_tokens": 20, "total_tokens": 30},
    )
    client.generate.return_value = gen_result
    client.async_generate = AsyncMock(return_value=gen_result)
    client.current_model = "gemini-2.5-pro"
    client.reinitialize = MagicMock()
    return client


# ── Test container helper ────────────────────────────────────
# Previously duplicated as ``_make_mock_container()`` in
# test_cli.py and test_cli_async.py.


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

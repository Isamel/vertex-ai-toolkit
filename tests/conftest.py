"""Shared pytest configuration and fixtures.

This module provides fixtures that are auto-discovered by pytest across all
test files under ``tests/``.  No explicit imports are needed — pytest injects
conftest fixtures by name.

Fixtures extracted here were previously duplicated in multiple test modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import pytest

from vaig.core.config import Settings, reset_settings
from vaig.session.store import SessionStore
from vaig.core.telemetry import TelemetryCollector


# ── Settings singleton reset ────────────────────────────────
# Previously duplicated in 9 test files as ``_reset()``.
# Safe to run even when the singleton was never created — it
# simply sets the module-level ``_settings`` to ``None``.


@pytest.fixture(autouse=True)
def _reset_settings() -> None:
    """Reset the Settings singleton between tests.

    This prevents cross-test contamination when any test constructs
    or mutates a ``Settings`` instance.
    """
    reset_settings()


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

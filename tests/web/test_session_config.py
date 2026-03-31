"""Tests for per-session config persistence — Task 4.1.

Covers:
- async_save_config writes config to Firestore doc with ownership check
- async_save_config rejects non-owner
- async_save_config returns False for missing session
- async_get_config reads config from session doc with ownership check
- async_get_config returns None for non-owner
- async_get_config returns None for missing session
- async_get_config returns None when no config stored
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip(
    "fastapi",
    reason="FastAPI not available; install the 'web' extra to run web tests.",
)
pytest.importorskip(
    "google.cloud.firestore",
    reason="google-cloud-firestore not available; install the 'web' extra to run web tests.",
)

from vaig.web.session.firestore import FirestoreSessionStore

# ── Mock Helpers ─────────────────────────────────────────────


class _MockDocSnapshot:
    """Mock Firestore document snapshot."""

    def __init__(self, doc_id: str, data: dict[str, Any], *, exists: bool = True) -> None:
        self.id = doc_id
        self._data = data
        self.exists = exists

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


def _make_mock_client() -> MagicMock:
    """Create a mock Firestore AsyncClient."""
    client = MagicMock()
    doc_ref = AsyncMock()
    doc_ref.set = AsyncMock()
    doc_ref.update = AsyncMock()
    doc_ref.get = AsyncMock()
    doc_ref.delete = AsyncMock()
    doc_ref.collection = MagicMock()

    collection_ref = MagicMock()
    collection_ref.document = MagicMock(return_value=doc_ref)
    client.collection = MagicMock(return_value=collection_ref)
    return client


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def mock_client() -> MagicMock:
    return _make_mock_client()


@pytest.fixture()
def store(mock_client: MagicMock) -> FirestoreSessionStore:
    return FirestoreSessionStore(mock_client)


# ── save_config ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_save_config_success(
    store: FirestoreSessionStore, mock_client: MagicMock
) -> None:
    """async_save_config writes config to the session doc."""
    doc = _MockDocSnapshot("s1", {"user": "owner@test.com", "name": "test"})
    doc_ref = mock_client.collection.return_value.document.return_value
    doc_ref.get = AsyncMock(return_value=doc)

    config = {"project": "my-proj", "temperature": 0.5}
    result = await store.async_save_config("s1", config, "owner@test.com")

    assert result is True
    doc_ref.update.assert_called_once()
    written = doc_ref.update.call_args[0][0]
    assert written["config"] == config
    assert "updated_at" in written


@pytest.mark.asyncio
async def test_save_config_wrong_user(
    store: FirestoreSessionStore, mock_client: MagicMock
) -> None:
    """async_save_config rejects when caller is not the owner."""
    doc = _MockDocSnapshot("s1", {"user": "owner@test.com", "name": "test"})
    doc_ref = mock_client.collection.return_value.document.return_value
    doc_ref.get = AsyncMock(return_value=doc)

    result = await store.async_save_config("s1", {"model": "gemini"}, "intruder@test.com")

    assert result is False
    doc_ref.update.assert_not_called()


@pytest.mark.asyncio
async def test_save_config_missing_session(
    store: FirestoreSessionStore, mock_client: MagicMock
) -> None:
    """async_save_config returns False for non-existent session."""
    doc = _MockDocSnapshot("s1", {}, exists=False)
    doc_ref = mock_client.collection.return_value.document.return_value
    doc_ref.get = AsyncMock(return_value=doc)

    result = await store.async_save_config("s1", {"model": "gemini"}, "user@test.com")

    assert result is False
    doc_ref.update.assert_not_called()


# ── get_config ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_config_success(
    store: FirestoreSessionStore, mock_client: MagicMock
) -> None:
    """async_get_config returns the stored config dict."""
    config = {"project": "my-proj", "temperature": 0.5, "model": "gemini-2.5-flash"}
    doc = _MockDocSnapshot("s1", {"user": "owner@test.com", "config": config})
    doc_ref = mock_client.collection.return_value.document.return_value
    doc_ref.get = AsyncMock(return_value=doc)

    result = await store.async_get_config("s1", "owner@test.com")

    assert result == config


@pytest.mark.asyncio
async def test_get_config_wrong_user(
    store: FirestoreSessionStore, mock_client: MagicMock
) -> None:
    """async_get_config returns None when caller is not the owner."""
    doc = _MockDocSnapshot("s1", {"user": "owner@test.com", "config": {"model": "gemini"}})
    doc_ref = mock_client.collection.return_value.document.return_value
    doc_ref.get = AsyncMock(return_value=doc)

    result = await store.async_get_config("s1", "intruder@test.com")

    assert result is None


@pytest.mark.asyncio
async def test_get_config_missing_session(
    store: FirestoreSessionStore, mock_client: MagicMock
) -> None:
    """async_get_config returns None for non-existent session."""
    doc = _MockDocSnapshot("s1", {}, exists=False)
    doc_ref = mock_client.collection.return_value.document.return_value
    doc_ref.get = AsyncMock(return_value=doc)

    result = await store.async_get_config("nonexistent", "user@test.com")

    assert result is None


@pytest.mark.asyncio
async def test_get_config_no_config_stored(
    store: FirestoreSessionStore, mock_client: MagicMock
) -> None:
    """async_get_config returns None when session has no config field."""
    doc = _MockDocSnapshot("s1", {"user": "owner@test.com", "name": "test"})
    doc_ref = mock_client.collection.return_value.document.return_value
    doc_ref.get = AsyncMock(return_value=doc)

    result = await store.async_get_config("s1", "owner@test.com")

    assert result is None


@pytest.mark.asyncio
async def test_get_config_non_dict_config(
    store: FirestoreSessionStore, mock_client: MagicMock
) -> None:
    """async_get_config returns None when config field is not a dict."""
    doc = _MockDocSnapshot("s1", {"user": "owner@test.com", "config": "not-a-dict"})
    doc_ref = mock_client.collection.return_value.document.return_value
    doc_ref.get = AsyncMock(return_value=doc)

    result = await store.async_get_config("s1", "owner@test.com")

    assert result is None

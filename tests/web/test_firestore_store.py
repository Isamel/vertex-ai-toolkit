"""Tests for FirestoreSessionStore — Task 3.2.

Covers:
- Session creation with correct Firestore writes
- Message add/get with ordering
- Session listing (with user filter)
- Session deletion with cascade
- Data shapes match SessionStoreProtocol expectations
- Protocol compliance check

All Firestore interactions are mocked — no real Firestore needed.
"""

from __future__ import annotations

import uuid
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

from vaig.core.protocols import SessionStoreProtocol
from vaig.web.session.firestore import FirestoreSessionStore

# ── Mock Helpers ─────────────────────────────────────────────


class _MockDocSnapshot:
    """Mock Firestore document snapshot."""

    def __init__(self, doc_id: str, data: dict[str, Any], *, exists: bool = True) -> None:
        self.id = doc_id
        self._data = data
        self.exists = exists
        self.reference = MagicMock()
        self.reference.collection = MagicMock(return_value=MagicMock())
        self.reference.delete = AsyncMock()

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


async def _async_iter(items: list[Any]):
    """Helper to create an async iterator from a list."""
    for item in items:
        yield item


def _make_mock_client() -> MagicMock:
    """Create a mock Firestore AsyncClient with basic collection/document chains."""
    client = MagicMock()
    # Default: collection().document() returns a mock doc_ref
    doc_ref = AsyncMock()
    doc_ref.set = AsyncMock()
    doc_ref.update = AsyncMock()
    doc_ref.get = AsyncMock()
    doc_ref.delete = AsyncMock()
    doc_ref.collection = MagicMock()

    collection_ref = MagicMock()
    collection_ref.document = MagicMock(return_value=doc_ref)
    collection_ref.order_by = MagicMock(return_value=collection_ref)
    collection_ref.where = MagicMock(return_value=collection_ref)
    collection_ref.limit = MagicMock(return_value=collection_ref)
    collection_ref.stream = MagicMock(return_value=_async_iter([]))

    client.collection = MagicMock(return_value=collection_ref)
    return client


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def mock_client() -> MagicMock:
    return _make_mock_client()


@pytest.fixture()
def store(mock_client: MagicMock) -> FirestoreSessionStore:
    return FirestoreSessionStore(mock_client)


# ── Protocol Compliance ──────────────────────────────────────


def test_firestore_store_satisfies_protocol() -> None:
    """FirestoreSessionStore must satisfy SessionStoreProtocol."""
    store = FirestoreSessionStore(MagicMock())
    assert isinstance(store, SessionStoreProtocol)


# ── Create Session ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_session_returns_uuid(store: FirestoreSessionStore) -> None:
    """async_create_session must return a valid UUID string."""
    session_id = await store.async_create_session(name="test", model="gemini-2.5-pro")
    # Should be a valid UUID4
    uuid.UUID(session_id, version=4)


@pytest.mark.asyncio
async def test_create_session_writes_to_firestore(
    store: FirestoreSessionStore, mock_client: MagicMock
) -> None:
    """async_create_session must write session doc with correct fields."""
    session_id = await store.async_create_session(
        name="my-session",
        model="gemini-2.5-flash",
        skill="rca",
        metadata={"key": "value"},
        user="test@example.com",
    )

    # Verify collection/document chain
    mock_client.collection.assert_called_with("vaig_sessions")
    col = mock_client.collection.return_value
    col.document.assert_called_with(session_id)

    # Verify the set() call
    doc_ref = col.document.return_value
    doc_ref.set.assert_called_once()
    written_data = doc_ref.set.call_args[0][0]

    assert written_data["name"] == "my-session"
    assert written_data["model"] == "gemini-2.5-flash"
    assert written_data["skill"] == "rca"
    assert written_data["user"] == "test@example.com"
    assert written_data["metadata"] == {"key": "value"}
    assert "created_at" in written_data
    assert "updated_at" in written_data


# ── Add Message ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_message_writes_to_subcollection(
    store: FirestoreSessionStore, mock_client: MagicMock
) -> None:
    """async_add_message must write to the messages subcollection."""
    # Set up the chain:
    # client.collection("vaig_sessions").document(session_id) → session_ref
    # session_ref.collection("messages").document(msg_id) → msg_ref
    session_ref = MagicMock()
    msg_collection = MagicMock()
    msg_ref = AsyncMock()
    msg_ref.set = AsyncMock()
    msg_collection.document = MagicMock(return_value=msg_ref)
    session_ref.collection = MagicMock(return_value=msg_collection)
    session_ref.update = AsyncMock()
    mock_client.collection.return_value.document.return_value = session_ref

    await store.async_add_message(
        session_id="sess-123",
        role="user",
        content="What is going on?",
        model="gemini-2.5-pro",
        token_count=42,
    )

    # Verify message subcollection write
    session_ref.collection.assert_called_with("messages")
    msg_ref.set.assert_called_once()
    written = msg_ref.set.call_args[0][0]
    assert written["role"] == "user"
    assert written["content"] == "What is going on?"
    assert written["model"] == "gemini-2.5-pro"
    assert written["token_count"] == 42
    assert "created_at" in written

    # Verify session updated_at is touched
    session_ref.update.assert_called_once()


# ── Get Messages ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_messages_returns_ordered_list(
    store: FirestoreSessionStore, mock_client: MagicMock
) -> None:
    """async_get_messages must return messages ordered by created_at."""
    msg1 = _MockDocSnapshot("m1", {"role": "user", "content": "hello", "created_at": "2026-01-01T00:00:00"})
    msg2 = _MockDocSnapshot("m2", {"role": "assistant", "content": "hi", "created_at": "2026-01-01T00:00:01"})

    # Chain: client.collection().document(sid).collection("messages").order_by().stream()
    session_ref = MagicMock()
    msg_col = MagicMock()
    msg_query = MagicMock()
    msg_query.stream = MagicMock(return_value=_async_iter([msg1, msg2]))
    msg_query.limit = MagicMock(return_value=msg_query)
    msg_col.order_by = MagicMock(return_value=msg_query)
    session_ref.collection = MagicMock(return_value=msg_col)
    mock_client.collection.return_value.document.return_value = session_ref

    messages = await store.async_get_messages("sess-123")
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_get_messages_with_limit(
    store: FirestoreSessionStore, mock_client: MagicMock
) -> None:
    """async_get_messages with limit should apply .limit() to the query."""
    session_ref = MagicMock()
    msg_col = MagicMock()
    msg_query = MagicMock()
    limited_query = MagicMock()
    limited_query.stream = MagicMock(return_value=_async_iter([]))
    msg_query.limit = MagicMock(return_value=limited_query)
    msg_col.order_by = MagicMock(return_value=msg_query)
    session_ref.collection = MagicMock(return_value=msg_col)
    mock_client.collection.return_value.document.return_value = session_ref

    await store.async_get_messages("sess-123", limit=5)
    msg_query.limit.assert_called_once_with(5)


# ── List Sessions ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_sessions_returns_sessions_with_count(
    store: FirestoreSessionStore, mock_client: MagicMock
) -> None:
    """async_list_sessions must return session dicts with message_count."""
    # Session doc mock
    sess_doc = _MockDocSnapshot("s1", {
        "user": "test@test.com",
        "name": "my-session",
        "model": "gemini-2.5-pro",
        "updated_at": "2026-01-01",
    })
    # Make the subcollection stream return 2 messages
    msg_sub_col = MagicMock()
    msg1 = _MockDocSnapshot("m1", {})
    msg2 = _MockDocSnapshot("m2", {})
    msg_sub_col.stream = MagicMock(return_value=_async_iter([msg1, msg2]))
    sess_doc.reference.collection = MagicMock(return_value=msg_sub_col)

    # Wire up the query chain
    collection_ref = mock_client.collection.return_value
    query = MagicMock()
    query.where = MagicMock(return_value=query)
    query.limit = MagicMock(return_value=query)
    query.stream = MagicMock(return_value=_async_iter([sess_doc]))
    collection_ref.order_by = MagicMock(return_value=query)

    sessions = await store.async_list_sessions(user="test@test.com")
    assert len(sessions) == 1
    assert sessions[0]["id"] == "s1"
    assert sessions[0]["name"] == "my-session"
    assert sessions[0]["message_count"] == 2


# ── Get Session ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_session_exists(
    store: FirestoreSessionStore, mock_client: MagicMock
) -> None:
    """async_get_session returns session dict when it exists."""
    doc = _MockDocSnapshot("s1", {"name": "test", "model": "gemini"})
    mock_client.collection.return_value.document.return_value.get = AsyncMock(return_value=doc)

    result = await store.async_get_session("s1")
    assert result is not None
    assert result["name"] == "test"
    assert result["id"] == "s1"


@pytest.mark.asyncio
async def test_get_session_not_found(
    store: FirestoreSessionStore, mock_client: MagicMock
) -> None:
    """async_get_session returns None when session doesn't exist."""
    doc = _MockDocSnapshot("s1", {}, exists=False)
    mock_client.collection.return_value.document.return_value.get = AsyncMock(return_value=doc)

    result = await store.async_get_session("nonexistent")
    assert result is None


# ── Delete Session ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_session_cascades_messages(
    store: FirestoreSessionStore, mock_client: MagicMock
) -> None:
    """async_delete_session must delete all messages before the session."""
    session_doc = _MockDocSnapshot("s1", {"name": "test"})
    msg_doc = _MockDocSnapshot("m1", {"role": "user"})
    msg_doc.reference.delete = AsyncMock()

    doc_ref = AsyncMock()
    doc_ref.get = AsyncMock(return_value=session_doc)
    doc_ref.delete = AsyncMock()

    # subcollection with one message
    msgs_col = MagicMock()
    msgs_col.stream = MagicMock(return_value=_async_iter([msg_doc]))
    doc_ref.collection = MagicMock(return_value=msgs_col)

    mock_client.collection.return_value.document.return_value = doc_ref

    result = await store.async_delete_session("s1")
    assert result is True

    # Message should be deleted first
    msg_doc.reference.delete.assert_called_once()
    # Then the session document
    doc_ref.delete.assert_called_once()


@pytest.mark.asyncio
async def test_delete_session_not_found(
    store: FirestoreSessionStore, mock_client: MagicMock
) -> None:
    """async_delete_session returns False when session doesn't exist."""
    not_found = _MockDocSnapshot("s1", {}, exists=False)
    doc_ref = AsyncMock()
    doc_ref.get = AsyncMock(return_value=not_found)
    mock_client.collection.return_value.document.return_value = doc_ref

    result = await store.async_delete_session("nonexistent")
    assert result is False

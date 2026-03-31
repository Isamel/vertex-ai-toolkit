"""Tests for SessionStoreProtocol — Task 3.1.

Validates that:
- SessionStoreProtocol is importable and @runtime_checkable
- The existing SQLite SessionStore satisfies the protocol
- A minimal mock also satisfies the protocol
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vaig.core.protocols import SessionStoreProtocol


def test_protocol_is_runtime_checkable() -> None:
    """SessionStoreProtocol must be usable with isinstance()."""
    assert hasattr(SessionStoreProtocol, "__protocol_attrs__") or hasattr(
        SessionStoreProtocol, "_is_runtime_protocol"
    )


def test_sqlite_store_satisfies_protocol() -> None:
    """The existing SessionStore must satisfy SessionStoreProtocol."""
    pytest.importorskip(
        "aiosqlite",
        reason="aiosqlite not available; needed for SessionStore.",
    )
    from vaig.session.store import SessionStore

    store = SessionStore(":memory:")
    assert isinstance(store, SessionStoreProtocol)
    store.close()


def test_protocol_in_all() -> None:
    """SessionStoreProtocol must be in __all__."""
    from vaig.core import protocols

    assert "SessionStoreProtocol" in protocols.__all__


def test_mock_satisfies_protocol() -> None:
    """A concrete class with the right methods satisfies the protocol."""

    class FakeStore:
        async def async_create_session(self, *a: object, **kw: object) -> str:
            return ""

        async def async_add_message(self, *a: object, **kw: object) -> None:
            pass

        async def async_get_messages(self, *a: object, **kw: object) -> list[dict[str, object]]:
            return []

        async def async_list_sessions(self, *a: object, **kw: object) -> list[dict[str, object]]:
            return []

        async def async_get_session(self, *a: object, **kw: object) -> dict[str, object] | None:
            return None

        async def async_delete_session(self, *a: object, **kw: object) -> bool:
            return True

    assert isinstance(FakeStore(), SessionStoreProtocol)


def test_empty_object_does_not_satisfy_protocol() -> None:
    """A plain object without the methods must NOT satisfy the protocol."""

    class Empty:
        pass

    assert not isinstance(Empty(), SessionStoreProtocol)

"""Firestore-backed session store for the VAIG web interface.

Implements :class:`~vaig.core.protocols.SessionStoreProtocol` using
Google Cloud Firestore as the persistence backend.

Collection layout::

    vaig_sessions/{session_id}          ← session document
        messages/{msg_id}               ← subcollection of messages

Each session document stores: ``user``, ``name``, ``model``, ``skill``,
``created_at``, ``updated_at``, and ``metadata``.

Each message document stores: ``role``, ``content``, ``model``,
``token_count``, and ``created_at``.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.cloud.firestore import AsyncClient

__all__ = [
    "FirestoreSessionStore",
]

logger = logging.getLogger(__name__)

_COLLECTION = "vaig_sessions"
_MESSAGES_SUBCOLLECTION = "messages"


class FirestoreSessionStore:
    """Firestore-backed session store implementing ``SessionStoreProtocol``.

    Args:
        client: An ``AsyncClient`` instance for Firestore operations.
                Shared across requests — Firestore async client is safe for
                concurrent use.
    """

    def __init__(self, client: AsyncClient) -> None:
        import importlib.util

        if importlib.util.find_spec("google.cloud.firestore") is None:
            raise ImportError(
                "google-cloud-firestore is required for FirestoreSessionStore. "
                "Install it with: pip install google-cloud-firestore"
            )
        self._client = client

    # ── Create ───────────────────────────────────────────────

    async def async_create_session(
        self,
        name: str,
        model: str,
        skill: str | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        user: str = "",
    ) -> str:
        """Create a new session document in Firestore.

        Returns:
            The generated session ID (UUID4).
        """
        session_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()

        doc_ref = self._client.collection(_COLLECTION).document(session_id)
        await doc_ref.set(
            {
                "user": user,
                "name": name,
                "model": model,
                "skill": skill or "",
                "created_at": now,
                "updated_at": now,
                "message_count": 0,
                "metadata": metadata or {},
            }
        )
        logger.info("Created Firestore session: %s (%s)", name, session_id[:8])
        return session_id

    # ── Messages ─────────────────────────────────────────────

    async def async_add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        model: str | None = None,
        token_count: int = 0,
    ) -> None:
        """Add a message to the session's ``messages`` subcollection."""
        now = datetime.now(UTC).isoformat()
        msg_id = str(uuid.uuid4())

        session_ref = self._client.collection(_COLLECTION).document(session_id)
        msg_ref = session_ref.collection(_MESSAGES_SUBCOLLECTION).document(msg_id)
        await msg_ref.set(
            {
                "role": role,
                "content": content,
                "model": model or "",
                "token_count": token_count,
                "created_at": now,
            }
        )
        # Touch the session's updated_at and increment message_count
        try:
            from google.cloud.firestore_v1 import Increment

            await session_ref.update(
                {"updated_at": now, "message_count": Increment(1)}
            )
        except ImportError:
            # Fallback: just update updated_at without atomic increment
            await session_ref.update({"updated_at": now})

    async def async_get_messages(
        self,
        session_id: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get messages for a session, ordered by timestamp ascending."""
        session_ref = self._client.collection(_COLLECTION).document(session_id)
        query = session_ref.collection(_MESSAGES_SUBCOLLECTION).order_by("created_at")

        if limit:
            query = query.limit(limit)

        docs = query.stream()
        messages: list[dict[str, Any]] = []
        async for doc in docs:
            messages.append(doc.to_dict())
        return messages

    # ── Session CRUD ─────────────────────────────────────────

    async def async_list_sessions(
        self,
        *,
        limit: int = 20,
        user: str | None = None,
    ) -> list[dict[str, Any]]:
        """List sessions, optionally filtered by user, newest first."""
        query = self._client.collection(_COLLECTION).order_by(
            "updated_at", direction="DESCENDING"
        )
        if user is not None:
            try:
                query = query.where(filter=_field_filter("user", "==", user))
            except TypeError:
                # Fallback for older SDK versions that don't support filter= kwarg
                query = query.where("user", "==", user)
        query = query.limit(limit)

        sessions: list[dict[str, Any]] = []
        async for doc in query.stream():
            data: dict[str, Any] = doc.to_dict() or {}
            data["id"] = doc.id

            # Read message_count from session doc (maintained by async_add_message)
            if "message_count" not in data:
                data["message_count"] = 0
            sessions.append(data)
        return sessions

    async def async_get_session(
        self, session_id: str
    ) -> dict[str, Any] | None:
        """Get a single session document by ID."""
        doc_ref = self._client.collection(_COLLECTION).document(session_id)
        doc = await doc_ref.get()
        if not doc.exists:
            return None
        data: dict[str, Any] = doc.to_dict() or {}
        data["id"] = doc.id
        return data

    async def async_delete_session(self, session_id: str) -> bool:
        """Delete a session and cascade-delete all its messages.

        Returns:
            True if the session existed and was deleted, False otherwise.
        """
        doc_ref = self._client.collection(_COLLECTION).document(session_id)
        doc = await doc_ref.get()
        if not doc.exists:
            return False

        # Delete all messages in the subcollection first (in parallel)
        msgs_ref = doc_ref.collection(_MESSAGES_SUBCOLLECTION)
        delete_tasks: list[Any] = []
        async for msg_doc in msgs_ref.stream():
            delete_tasks.append(msg_doc.reference.delete())
        if delete_tasks:
            await asyncio.gather(*delete_tasks)

        # Delete the session document
        await doc_ref.delete()
        logger.info("Deleted Firestore session: %s", session_id[:8])
        return True


def _field_filter(field: str, op: str, value: Any) -> Any:
    """Create a Firestore FieldFilter (import-safe helper).

    Uses ``google.cloud.firestore_v1.base_query.FieldFilter`` when
    available, which is the non-deprecated API in recent SDK versions.
    """
    try:
        from google.cloud.firestore_v1.base_query import FieldFilter

        return FieldFilter(field, op, value)
    except ImportError:  # pragma: no cover — fallback for older SDK
        # Older SDK versions accept the 3-arg tuple directly
        return (field, op, value)

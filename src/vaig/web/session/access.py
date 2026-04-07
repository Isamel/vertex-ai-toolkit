"""Firestore-backed session access control (ACL) service.

Implements :class:`~vaig.core.protocols.SessionAccessProtocol` using
a ``collaborators`` subcollection under each session document.

Firestore layout::

    vaig_sessions/{session_id}
        collaborators/{normalized_email}   ← ACL documents
        annotations/{annotation_id}        ← annotation documents
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from vaig.core.models import AccessResult, Annotation, AnnotationType, SessionCollaborator, SessionRole

if TYPE_CHECKING:
    from google.cloud.firestore import AsyncClient

    from vaig.core.protocols import SessionStoreProtocol

__all__ = [
    "SessionAccessControl",
]

logger = logging.getLogger(__name__)

_COLLECTION = "vaig_sessions"
_COLLABORATORS_SUB = "collaborators"
_ANNOTATIONS_SUB = "annotations"
_FEATURE_FLAG = "VAIG_WEB_SHARED_SESSIONS"
_IAP_PREFIX = "accounts.google.com:"


def _normalize_email(email: str) -> str:
    """Normalize an email for storage and comparison.

    Strips the IAP ``accounts.google.com:`` prefix and lowercases.
    """
    if email.startswith(_IAP_PREFIX):
        email = email[len(_IAP_PREFIX) :]
    return email.strip().lower()


def _sharing_enabled() -> bool:
    """Return ``True`` when the shared-sessions feature flag is active."""
    return os.environ.get(_FEATURE_FLAG, "").lower() in ("true", "1", "yes")


def _extract_domain(email: str) -> str:
    """Extract the domain part from an email address."""
    parts = email.rsplit("@", 1)
    return parts[1] if len(parts) == 2 else ""


class SessionAccessControl:
    """Firestore-backed session access control.

    Args:
        client: Firestore ``AsyncClient``.
        store: A ``SessionStoreProtocol`` implementation used to fetch
               session documents (for ownership checks).
    """

    def __init__(self, client: AsyncClient, store: SessionStoreProtocol) -> None:
        self._client = client
        self._store = store

    # ── check_access (Task 2.2) ──────────────────────────────

    async def check_access(
        self,
        session_id: str,
        user: str,
        *,
        required: SessionRole = SessionRole.VIEWER,
    ) -> AccessResult:
        """Check whether *user* has at least *required* role on *session_id*.

        When the feature flag is disabled, only the session owner is
        granted access (original behaviour).
        """
        user = _normalize_email(user)

        session = await self._store.async_get_session(session_id)
        if session is None:
            logger.warning(
                "Access denied: session %s not found (user=%s, required=%s)",
                session_id[:8],
                user,
                required.value,
            )
            return AccessResult(granted=False)

        owner_email = _normalize_email(session.get("user", ""))

        # Owner always has full access
        if user == owner_email:
            return AccessResult(granted=True, role=SessionRole.OWNER)

        # Feature flag off → owner-only fallback
        if not _sharing_enabled():
            logger.warning(
                "Access denied: sharing disabled, non-owner %s on session %s",
                user,
                session_id[:8],
            )
            return AccessResult(granted=False)

        # Look up collaborator document
        collab_ref = (
            self._client.collection(_COLLECTION)
            .document(session_id)
            .collection(_COLLABORATORS_SUB)
            .document(user)
        )
        doc = await collab_ref.get()
        if not doc.exists:
            logger.warning(
                "Access denied: %s is not a collaborator on session %s",
                user,
                session_id[:8],
            )
            return AccessResult(granted=False)

        data: dict[str, Any] = doc.to_dict() or {}
        try:
            role = SessionRole(data.get("role", ""))
        except ValueError:
            logger.warning(
                "Access denied: invalid role for %s on session %s",
                user,
                session_id[:8],
            )
            return AccessResult(granted=False)

        if role >= required:
            return AccessResult(granted=True, role=role)
        logger.warning(
            "Access denied: %s has role %s but %s required on session %s",
            user,
            role.value,
            required.value,
            session_id[:8],
        )
        return AccessResult(granted=False, role=role)

    # ── share (Task 2.3) ─────────────────────────────────────

    async def share(
        self,
        session_id: str,
        owner: str,
        target_email: str,
        role: SessionRole,
    ) -> SessionCollaborator:
        """Grant *target_email* the given *role* on *session_id*.

        Only the session owner may share.  Validates same-domain and
        rejects self-sharing.

        Raises:
            PermissionError: If the caller is not the owner.
            ValueError: If target is self or cross-domain.
        """
        owner = _normalize_email(owner)
        target_email = _normalize_email(target_email)

        # Fetch session to verify ownership
        session = await self._store.async_get_session(session_id)
        if session is None:
            msg = "Session not found"
            raise PermissionError(msg)

        session_owner = _normalize_email(session.get("user", ""))
        if owner != session_owner:
            msg = "Only the session owner can share"
            raise PermissionError(msg)

        # Cannot share with self
        if target_email == owner:
            msg = "Cannot share with session owner"
            raise ValueError(msg)

        # Domain validation
        owner_domain = _extract_domain(owner)
        target_domain = _extract_domain(target_email)
        if owner_domain and target_domain and owner_domain != target_domain:
            msg = "Cannot share outside organization domain"
            raise ValueError(msg)

        now = datetime.now(UTC).isoformat()
        collab = SessionCollaborator(
            email=target_email,
            role=role,
            added_at=now,
            added_by=owner,
        )

        # Write collaborator document
        collab_ref = (
            self._client.collection(_COLLECTION)
            .document(session_id)
            .collection(_COLLABORATORS_SUB)
            .document(target_email)
        )
        await collab_ref.set(
            {
                "role": role.value,
                "added_at": now,
                "added_by": owner,
            }
        )

        logger.info(
            "Shared session %s with %s as %s (by %s)",
            session_id[:8],
            target_email,
            role.value,
            owner,
        )
        return collab

    # ── revoke (Task 2.3) ────────────────────────────────────

    async def revoke(
        self,
        session_id: str,
        owner: str,
        target_email: str,
    ) -> bool:
        """Revoke *target_email*'s access to *session_id*.

        Only the session owner may revoke.

        Raises:
            PermissionError: If the caller is not the owner.
        """
        owner = _normalize_email(owner)
        target_email = _normalize_email(target_email)

        session = await self._store.async_get_session(session_id)
        if session is None:
            msg = "Session not found"
            raise PermissionError(msg)

        session_owner = _normalize_email(session.get("user", ""))
        if owner != session_owner:
            msg = "Only the session owner can revoke access"
            raise PermissionError(msg)

        collab_ref = (
            self._client.collection(_COLLECTION)
            .document(session_id)
            .collection(_COLLABORATORS_SUB)
            .document(target_email)
        )

        doc = await collab_ref.get()
        if not doc.exists:
            return False

        await collab_ref.delete()

        logger.info(
            "Revoked access for %s on session %s (by %s)",
            target_email,
            session_id[:8],
            owner,
        )
        return True

    # ── list_collaborators (Task 2.4) ────────────────────────

    async def list_collaborators(
        self,
        session_id: str,
        user: str,
    ) -> list[SessionCollaborator]:
        """List all collaborators for *session_id*.

        Any collaborator (viewer+) may call this.
        """
        user = _normalize_email(user)

        # Verify the caller has at least viewer access
        result = await self.check_access(session_id, user, required=SessionRole.VIEWER)
        if not result.granted:
            return []

        collab_col = (
            self._client.collection(_COLLECTION)
            .document(session_id)
            .collection(_COLLABORATORS_SUB)
        )

        collaborators: list[SessionCollaborator] = []
        async for doc in collab_col.stream():
            data: dict[str, Any] = doc.to_dict() or {}
            try:
                collaborators.append(
                    SessionCollaborator(
                        email=doc.id,
                        role=SessionRole(data.get("role", "viewer")),
                        added_at=data.get("added_at", ""),
                        added_by=data.get("added_by", ""),
                    )
                )
            except ValueError:
                logger.warning("Invalid collaborator doc %s in session %s", doc.id, session_id[:8])
                continue

        return collaborators

    # ── list_accessible_sessions (Task 2.5) ──────────────────

    async def list_accessible_sessions(
        self,
        user: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List sessions shared with *user* (not owned).

        Uses a Firestore collection group query on ``collaborators``
        where the document ID matches the normalized user email.
        """
        user = _normalize_email(user)

        # Collection group query: find all collaborator docs for this user
        collab_group = self._client.collection_group(_COLLABORATORS_SUB)

        sessions: list[dict[str, Any]] = []
        count = 0
        async for doc in collab_group.stream():
            if doc.id != user:
                continue
            if count >= limit:
                break

            data: dict[str, Any] = doc.to_dict() or {}
            try:
                role = SessionRole(data.get("role", "viewer"))
            except ValueError:
                continue

            # Navigate to the parent session document
            session_ref = doc.reference.parent.parent
            if session_ref is None:
                continue

            session_doc = await session_ref.get()
            if not session_doc.exists:
                continue

            session_data: dict[str, Any] = session_doc.to_dict() or {}
            session_data["id"] = session_doc.id
            session_data["role"] = role.value
            sessions.append(session_data)
            count += 1

        return sessions

    # ── Annotation CRUD (Task 4.1) ───────────────────────────

    async def add_annotation(
        self,
        session_id: str,
        user: str,
        *,
        annotation_type: str,
        content: str,
        message_ref: str | None = None,
    ) -> Annotation:
        """Create a new annotation on *session_id*.

        Requires editor+ access.

        Raises:
            PermissionError: If the caller lacks editor access.
        """
        user = _normalize_email(user)

        result = await self.check_access(session_id, user, required=SessionRole.EDITOR)
        if not result.granted:
            msg = "Insufficient permissions"
            raise PermissionError(msg)

        now = datetime.now(UTC).isoformat()
        ann_id = str(uuid4())

        annotation = Annotation(
            id=ann_id,
            author=user,
            content=content,
            annotation_type=cast(AnnotationType, annotation_type),
            message_ref=message_ref,
            created_at=now,
            updated_at=now,
        )

        ann_ref = (
            self._client.collection(_COLLECTION)
            .document(session_id)
            .collection(_ANNOTATIONS_SUB)
            .document(ann_id)
        )
        await ann_ref.set({
            "author": user,
            "content": annotation.content,
            "annotation_type": annotation_type,
            "message_ref": message_ref,
            "created_at": now,
            "updated_at": now,
        })

        logger.info(
            "Annotation %s added to session %s by %s",
            ann_id[:8],
            session_id[:8],
            user,
        )
        return annotation

    async def update_annotation(
        self,
        session_id: str,
        annotation_id: str,
        user: str,
        content: str,
    ) -> Annotation:
        """Update an existing annotation.  Author-only.

        Raises:
            PermissionError: If the caller is not the annotation author.
            LookupError: If the annotation does not exist.
        """
        user = _normalize_email(user)

        # Must at least be a viewer to access the session
        result = await self.check_access(session_id, user, required=SessionRole.VIEWER)
        if not result.granted:
            msg = "Insufficient permissions"
            raise PermissionError(msg)

        ann_ref = (
            self._client.collection(_COLLECTION)
            .document(session_id)
            .collection(_ANNOTATIONS_SUB)
            .document(annotation_id)
        )
        doc = await ann_ref.get()
        if not doc.exists:
            msg = "Annotation not found"
            raise LookupError(msg)

        data: dict[str, Any] = doc.to_dict() or {}
        author = _normalize_email(data.get("author", ""))

        if author != user:
            msg = "Only the annotation author can edit"
            raise PermissionError(msg)

        now = datetime.now(UTC).isoformat()
        await ann_ref.update({"content": content, "updated_at": now})

        return Annotation(
            id=annotation_id,
            author=author,
            content=content,
            annotation_type=cast(AnnotationType, data.get("annotation_type", "note")),
            message_ref=data.get("message_ref"),
            created_at=data.get("created_at", ""),
            updated_at=now,
        )

    async def delete_annotation(
        self,
        session_id: str,
        annotation_id: str,
        user: str,
    ) -> bool:
        """Delete an annotation.  Author-only.

        Raises:
            PermissionError: If the caller is not the annotation author.
            LookupError: If the annotation does not exist.
        """
        user = _normalize_email(user)

        # Must at least be a viewer to access the session
        result = await self.check_access(session_id, user, required=SessionRole.VIEWER)
        if not result.granted:
            msg = "Insufficient permissions"
            raise PermissionError(msg)

        ann_ref = (
            self._client.collection(_COLLECTION)
            .document(session_id)
            .collection(_ANNOTATIONS_SUB)
            .document(annotation_id)
        )
        doc = await ann_ref.get()
        if not doc.exists:
            msg = "Annotation not found"
            raise LookupError(msg)

        data: dict[str, Any] = doc.to_dict() or {}
        author = _normalize_email(data.get("author", ""))

        if author != user:
            msg = "Only the annotation author can delete"
            raise PermissionError(msg)

        await ann_ref.delete()

        logger.info(
            "Annotation %s deleted from session %s by %s",
            annotation_id[:8],
            session_id[:8],
            user,
        )
        return True

    async def list_annotations(
        self,
        session_id: str,
        user: str,
        *,
        limit: int = 50,
    ) -> list[Annotation]:
        """List annotations for *session_id*.

        Requires viewer+ access.  Returns up to *limit* annotations
        ordered by ``created_at``.
        """
        user = _normalize_email(user)

        result = await self.check_access(session_id, user, required=SessionRole.VIEWER)
        if not result.granted:
            return []

        ann_col = (
            self._client.collection(_COLLECTION)
            .document(session_id)
            .collection(_ANNOTATIONS_SUB)
        )

        annotations: list[Annotation] = []
        count = 0
        async for doc in ann_col.order_by("created_at").limit(limit).stream():
            if count >= limit:
                break
            data: dict[str, Any] = doc.to_dict() or {}
            try:
                annotations.append(
                    Annotation(
                        id=doc.id,
                        author=data.get("author", ""),
                        content=data.get("content", ""),
                        annotation_type=cast(AnnotationType, data.get("annotation_type", "note")),
                        message_ref=data.get("message_ref"),
                        created_at=data.get("created_at", ""),
                        updated_at=data.get("updated_at", ""),
                    )
                )
                count += 1
            except (ValueError, KeyError):
                logger.warning(
                    "Invalid annotation doc %s in session %s",
                    doc.id,
                    session_id[:8],
                )
                continue

        return annotations

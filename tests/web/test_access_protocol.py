"""Tests for session sharing models and SessionAccessProtocol — SPEC-6.1 Phase 1.

Validates that:
- SessionRole enum has correct values and ordering
- SessionCollaborator, AccessResult, Annotation are frozen Pydantic models
- SessionAccessProtocol is importable, runtime-checkable, and in __all__
- A FakeAccess class satisfies the protocol
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from vaig.core.models import (
    AccessResult,
    Annotation,
    SessionCollaborator,
    SessionRole,
)
from vaig.core.protocols import SessionAccessProtocol

# ── SessionRole enum ────────────────────────────────────────


class TestSessionRole:
    """Tests for the SessionRole enum."""

    def test_values(self) -> None:
        assert SessionRole.OWNER.value == "owner"
        assert SessionRole.EDITOR.value == "editor"
        assert SessionRole.VIEWER.value == "viewer"

    def test_is_str_enum(self) -> None:
        assert isinstance(SessionRole.OWNER, str)
        assert SessionRole.EDITOR == "editor"

    def test_ordering_owner_gt_editor(self) -> None:
        assert SessionRole.OWNER > SessionRole.EDITOR

    def test_ordering_editor_gt_viewer(self) -> None:
        assert SessionRole.EDITOR > SessionRole.VIEWER

    def test_ordering_owner_ge_owner(self) -> None:
        assert SessionRole.OWNER >= SessionRole.OWNER

    def test_ordering_viewer_lt_editor(self) -> None:
        assert SessionRole.VIEWER < SessionRole.EDITOR

    def test_ordering_viewer_le_viewer(self) -> None:
        assert SessionRole.VIEWER <= SessionRole.VIEWER

    def test_level_property(self) -> None:
        assert SessionRole.OWNER.level == 2
        assert SessionRole.EDITOR.level == 1
        assert SessionRole.VIEWER.level == 0


# ── SessionCollaborator ─────────────────────────────────────


class TestSessionCollaborator:
    """Tests for the SessionCollaborator Pydantic model."""

    def test_construction(self) -> None:
        collab = SessionCollaborator(
            email="bob@example.com",
            role=SessionRole.EDITOR,
            added_at="2026-01-01T00:00:00Z",
            added_by="alice@example.com",
        )
        assert collab.email == "bob@example.com"
        assert collab.role == SessionRole.EDITOR
        assert collab.added_by == "alice@example.com"

    def test_frozen(self) -> None:
        collab = SessionCollaborator(
            email="bob@example.com",
            role=SessionRole.VIEWER,
            added_at="2026-01-01T00:00:00Z",
            added_by="alice@example.com",
        )
        with pytest.raises(ValidationError):
            collab.email = "new@example.com"  # type: ignore[misc]

    def test_serialization(self) -> None:
        collab = SessionCollaborator(
            email="bob@example.com",
            role=SessionRole.EDITOR,
            added_at="2026-01-01T00:00:00Z",
            added_by="alice@example.com",
        )
        data = collab.model_dump()
        assert data["role"] == "editor"
        assert data["email"] == "bob@example.com"


# ── AccessResult ─────────────────────────────────────────────


class TestAccessResult:
    """Tests for the AccessResult Pydantic model."""

    def test_granted_with_role(self) -> None:
        result = AccessResult(granted=True, role=SessionRole.OWNER)
        assert result.granted is True
        assert result.role == SessionRole.OWNER

    def test_denied_no_role(self) -> None:
        result = AccessResult(granted=False)
        assert result.granted is False
        assert result.role is None

    def test_frozen(self) -> None:
        result = AccessResult(granted=True, role=SessionRole.VIEWER)
        with pytest.raises(ValidationError):
            result.granted = False  # type: ignore[misc]

    def test_serialization(self) -> None:
        result = AccessResult(granted=True, role=SessionRole.EDITOR)
        data = result.model_dump()
        assert data["granted"] is True
        assert data["role"] == "editor"


# ── Annotation ───────────────────────────────────────────────


class TestAnnotation:
    """Tests for the Annotation Pydantic model."""

    def test_construction(self) -> None:
        ann = Annotation(
            id="ann-1",
            author="bob@example.com",
            content="Root cause: OOM",
            annotation_type="root_cause",
            message_ref="msg-5",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        )
        assert ann.annotation_type == "root_cause"
        assert ann.message_ref == "msg-5"

    def test_optional_message_ref(self) -> None:
        ann = Annotation(
            id="ann-2",
            author="bob@example.com",
            content="General note",
            annotation_type="observation",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        )
        assert ann.message_ref is None

    def test_frozen(self) -> None:
        ann = Annotation(
            id="ann-3",
            author="bob@example.com",
            content="A note",
            annotation_type="observation",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        )
        with pytest.raises(ValidationError):
            ann.content = "changed"  # type: ignore[misc]

    def test_content_max_length(self) -> None:
        with pytest.raises(ValidationError, match="string_too_long"):
            Annotation(
                id="ann-4",
                author="bob@example.com",
                content="x" * 2001,
                annotation_type="observation",
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-01-01T00:00:00Z",
            )

    def test_content_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Annotation(
                id="ann-5",
                author="bob@example.com",
                content="",
                annotation_type="observation",
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-01-01T00:00:00Z",
            )

    def test_invalid_annotation_type(self) -> None:
        with pytest.raises(ValidationError):
            Annotation(
                id="ann-6",
                author="bob@example.com",
                content="Something",
                annotation_type="invalid_type",  # type: ignore[arg-type]
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-01-01T00:00:00Z",
            )

    def test_serialization(self) -> None:
        ann = Annotation(
            id="ann-7",
            author="bob@example.com",
            content="Action needed",
            annotation_type="action_item",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        )
        data = ann.model_dump()
        assert data["annotation_type"] == "action_item"
        assert data["message_ref"] is None

    def test_content_whitespace_stripped(self) -> None:
        ann = Annotation(
            id="ann-8",
            author="bob@example.com",
            content="  hello  ",
            annotation_type="observation",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        )
        assert ann.content == "hello"


# ── SessionAccessProtocol ────────────────────────────────────


class TestSessionAccessProtocol:
    """Tests for the SessionAccessProtocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        assert hasattr(SessionAccessProtocol, "__protocol_attrs__") or hasattr(
            SessionAccessProtocol, "_is_runtime_protocol"
        )

    def test_protocol_in_all(self) -> None:
        from vaig.core import protocols

        assert "SessionAccessProtocol" in protocols.__all__

    def test_fake_access_satisfies_protocol(self) -> None:
        class FakeAccess:
            async def check_access(
                self, session_id: str, user: str, *, required: SessionRole = SessionRole.VIEWER
            ) -> AccessResult:
                return AccessResult(granted=True, role=SessionRole.OWNER)

            async def share(
                self, session_id: str, owner: str, target_email: str, role: SessionRole
            ) -> SessionCollaborator:
                return SessionCollaborator(
                    email=target_email, role=role, added_at="", added_by=owner
                )

            async def revoke(self, session_id: str, owner: str, target_email: str) -> bool:
                return True

            async def list_collaborators(
                self, session_id: str, user: str
            ) -> list[SessionCollaborator]:
                return []

            async def list_accessible_sessions(
                self, user: str, *, limit: int = 20
            ) -> list[dict[str, Any]]:
                return []

            async def add_annotation(
                self,
                session_id: str,
                user: str,
                *,
                annotation_type: str,
                content: str,
                message_ref: str | None = None,
            ) -> Annotation:
                return Annotation(
                    id="ann-1",
                    author=user,
                    content=content,
                    annotation_type="observation",
                    created_at="",
                    updated_at="",
                )

            async def update_annotation(
                self, session_id: str, annotation_id: str, user: str, content: str
            ) -> Annotation:
                return Annotation(
                    id=annotation_id,
                    author=user,
                    content=content,
                    annotation_type="observation",
                    created_at="",
                    updated_at="",
                )

            async def delete_annotation(
                self, session_id: str, annotation_id: str, user: str
            ) -> bool:
                return True

            async def list_annotations(
                self, session_id: str, user: str, *, limit: int = 50
            ) -> list[Annotation]:
                return []

        assert isinstance(FakeAccess(), SessionAccessProtocol)

    def test_empty_object_does_not_satisfy_protocol(self) -> None:
        class Empty:
            pass

        assert not isinstance(Empty(), SessionAccessProtocol)

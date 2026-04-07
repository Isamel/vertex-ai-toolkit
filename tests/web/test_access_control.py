"""Tests for SessionAccessControl — SPEC-6.1 Phase 2.

Covers:
- Owner access always granted
- Editor access granted/denied based on role level
- Viewer blocked from edit operations
- No-access user denied
- Revocation removes access
- Email normalization (IAP prefix, case)
- Domain validation (cross-domain blocked)
- Feature flag disabled → owner-only fallback
- Self-share rejection
- list_collaborators returns correct data
- list_accessible_sessions with collection group query
- Protocol compliance

All Firestore interactions are mocked — no real Firestore needed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip(
    "google.cloud.firestore",
    reason="google-cloud-firestore not available; install the 'web' extra.",
)

from vaig.core.models import SessionRole
from vaig.core.protocols import SessionAccessProtocol
from vaig.web.session.access import SessionAccessControl, _normalize_email

# ── Mock Helpers ─────────────────────────────────────────────


class _MockDocSnapshot:
    """Mock Firestore document snapshot."""

    def __init__(
        self,
        doc_id: str,
        data: dict[str, Any],
        *,
        exists: bool = True,
        parent: Any = None,
    ) -> None:
        self.id = doc_id
        self._data = data
        self.exists = exists
        self.reference = MagicMock()
        self.reference.delete = AsyncMock()
        if parent is not None:
            self.reference.parent = parent
        else:
            self.reference.parent = MagicMock()
            self.reference.parent.parent = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


async def _async_iter(items: list[Any]):  # noqa: ANN204
    """Helper to create an async iterator from a list."""
    for item in items:
        yield item


def _make_mock_store(
    session_data: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a mock SessionStoreProtocol."""
    store = MagicMock()
    store.async_get_session = AsyncMock(return_value=session_data)
    return store


def _make_mock_client() -> MagicMock:
    """Create a mock Firestore AsyncClient."""
    client = MagicMock()
    doc_ref = AsyncMock()
    doc_ref.set = AsyncMock()
    doc_ref.get = AsyncMock()
    doc_ref.delete = AsyncMock()

    sub_col = MagicMock()
    sub_col.document = MagicMock(return_value=doc_ref)
    sub_col.stream = MagicMock(return_value=_async_iter([]))

    session_doc_ref = MagicMock()
    session_doc_ref.collection = MagicMock(return_value=sub_col)

    col_ref = MagicMock()
    col_ref.document = MagicMock(return_value=session_doc_ref)

    client.collection = MagicMock(return_value=col_ref)
    client.collection_group = MagicMock(return_value=MagicMock(stream=MagicMock(return_value=_async_iter([]))))
    return client


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def _enable_sharing():
    """Enable the shared sessions feature flag."""
    with patch.dict("os.environ", {"VAIG_WEB_SHARED_SESSIONS": "true"}):
        yield


@pytest.fixture()
def _disable_sharing():
    """Disable the shared sessions feature flag."""
    with patch.dict("os.environ", {"VAIG_WEB_SHARED_SESSIONS": "false"}):
        yield


# ── Protocol Compliance ──────────────────────────────────────


def test_access_control_satisfies_protocol() -> None:
    """SessionAccessControl must satisfy SessionAccessProtocol."""
    acl = SessionAccessControl(MagicMock(), MagicMock())
    assert isinstance(acl, SessionAccessProtocol)


# ── Email Normalization ──────────────────────────────────────


class TestEmailNormalization:
    def test_strips_iap_prefix(self) -> None:
        assert _normalize_email("accounts.google.com:alice@co.com") == "alice@co.com"

    def test_lowercases(self) -> None:
        assert _normalize_email("Alice@CO.COM") == "alice@co.com"

    def test_strips_whitespace(self) -> None:
        assert _normalize_email("  alice@co.com  ") == "alice@co.com"

    def test_combined(self) -> None:
        assert _normalize_email("accounts.google.com:Alice@Company.COM") == "alice@company.com"


# ── check_access ─────────────────────────────────────────────


class TestCheckAccess:
    """Tests for the check_access method."""

    async def test_owner_always_granted(self) -> None:
        """Owner gets full access regardless of feature flag."""
        store = _make_mock_store({"id": "s1", "user": "alice@co.com"})
        client = _make_mock_client()
        acl = SessionAccessControl(client, store)

        result = await acl.check_access("s1", "alice@co.com", required=SessionRole.OWNER)
        assert result.granted is True
        assert result.role == SessionRole.OWNER

    async def test_owner_with_iap_prefix(self) -> None:
        """Owner email with IAP prefix still matches."""
        store = _make_mock_store({"id": "s1", "user": "alice@co.com"})
        client = _make_mock_client()
        acl = SessionAccessControl(client, store)

        result = await acl.check_access(
            "s1", "accounts.google.com:alice@co.com", required=SessionRole.VIEWER
        )
        assert result.granted is True
        assert result.role == SessionRole.OWNER

    @pytest.mark.usefixtures("_enable_sharing")
    async def test_editor_can_view(self) -> None:
        """Editor has enough privilege for viewer access."""
        store = _make_mock_store({"id": "s1", "user": "alice@co.com"})
        client = _make_mock_client()

        # Mock the collaborator doc lookup
        collab_doc = _MockDocSnapshot("bob@co.com", {"role": "editor"})
        collab_ref = AsyncMock()
        collab_ref.get = AsyncMock(return_value=collab_doc)

        sub_col = MagicMock()
        sub_col.document = MagicMock(return_value=collab_ref)
        session_doc_ref = MagicMock()
        session_doc_ref.collection = MagicMock(return_value=sub_col)
        col_ref = MagicMock()
        col_ref.document = MagicMock(return_value=session_doc_ref)
        client.collection = MagicMock(return_value=col_ref)

        acl = SessionAccessControl(client, store)
        result = await acl.check_access("s1", "bob@co.com", required=SessionRole.VIEWER)
        assert result.granted is True
        assert result.role == SessionRole.EDITOR

    @pytest.mark.usefixtures("_enable_sharing")
    async def test_viewer_blocked_from_edit(self) -> None:
        """Viewer cannot access editor-level operations."""
        store = _make_mock_store({"id": "s1", "user": "alice@co.com"})
        client = _make_mock_client()

        collab_doc = _MockDocSnapshot("carol@co.com", {"role": "viewer"})
        collab_ref = AsyncMock()
        collab_ref.get = AsyncMock(return_value=collab_doc)

        sub_col = MagicMock()
        sub_col.document = MagicMock(return_value=collab_ref)
        session_doc_ref = MagicMock()
        session_doc_ref.collection = MagicMock(return_value=sub_col)
        col_ref = MagicMock()
        col_ref.document = MagicMock(return_value=session_doc_ref)
        client.collection = MagicMock(return_value=col_ref)

        acl = SessionAccessControl(client, store)
        result = await acl.check_access("s1", "carol@co.com", required=SessionRole.EDITOR)
        assert result.granted is False
        assert result.role == SessionRole.VIEWER

    @pytest.mark.usefixtures("_enable_sharing")
    async def test_no_access_user(self) -> None:
        """User with no collaborator doc is denied."""
        store = _make_mock_store({"id": "s1", "user": "alice@co.com"})
        client = _make_mock_client()

        missing_doc = _MockDocSnapshot("eve@co.com", {}, exists=False)
        collab_ref = AsyncMock()
        collab_ref.get = AsyncMock(return_value=missing_doc)

        sub_col = MagicMock()
        sub_col.document = MagicMock(return_value=collab_ref)
        session_doc_ref = MagicMock()
        session_doc_ref.collection = MagicMock(return_value=sub_col)
        col_ref = MagicMock()
        col_ref.document = MagicMock(return_value=session_doc_ref)
        client.collection = MagicMock(return_value=col_ref)

        acl = SessionAccessControl(client, store)
        result = await acl.check_access("s1", "eve@co.com", required=SessionRole.VIEWER)
        assert result.granted is False

    async def test_nonexistent_session(self) -> None:
        """Access check on non-existent session returns denied."""
        store = _make_mock_store(None)
        client = _make_mock_client()
        acl = SessionAccessControl(client, store)

        result = await acl.check_access("nonexistent", "alice@co.com")
        assert result.granted is False

    @pytest.mark.usefixtures("_disable_sharing")
    async def test_feature_flag_disabled_non_owner(self) -> None:
        """When feature flag is off, non-owner is denied even if collaborator doc exists."""
        store = _make_mock_store({"id": "s1", "user": "alice@co.com"})
        client = _make_mock_client()
        acl = SessionAccessControl(client, store)

        result = await acl.check_access("s1", "bob@co.com", required=SessionRole.VIEWER)
        assert result.granted is False

    @pytest.mark.usefixtures("_disable_sharing")
    async def test_feature_flag_disabled_owner_still_works(self) -> None:
        """Owner access works even when feature flag is off."""
        store = _make_mock_store({"id": "s1", "user": "alice@co.com"})
        client = _make_mock_client()
        acl = SessionAccessControl(client, store)

        result = await acl.check_access("s1", "alice@co.com", required=SessionRole.OWNER)
        assert result.granted is True
        assert result.role == SessionRole.OWNER


# ── share ────────────────────────────────────────────────────


class TestShare:
    """Tests for the share method."""

    async def test_share_success(self) -> None:
        store = _make_mock_store({"id": "s1", "user": "alice@co.com"})
        client = _make_mock_client()
        acl = SessionAccessControl(client, store)

        collab = await acl.share("s1", "alice@co.com", "bob@co.com", SessionRole.EDITOR)
        assert collab.email == "bob@co.com"
        assert collab.role == SessionRole.EDITOR
        assert collab.added_by == "alice@co.com"

    async def test_share_non_owner_raises(self) -> None:
        store = _make_mock_store({"id": "s1", "user": "alice@co.com"})
        client = _make_mock_client()
        acl = SessionAccessControl(client, store)

        with pytest.raises(PermissionError, match="owner"):
            await acl.share("s1", "bob@co.com", "carol@co.com", SessionRole.VIEWER)

    async def test_self_share_raises(self) -> None:
        store = _make_mock_store({"id": "s1", "user": "alice@co.com"})
        client = _make_mock_client()
        acl = SessionAccessControl(client, store)

        with pytest.raises(ValueError, match="Cannot share with session owner"):
            await acl.share("s1", "alice@co.com", "alice@co.com", SessionRole.EDITOR)

    async def test_cross_domain_raises(self) -> None:
        store = _make_mock_store({"id": "s1", "user": "alice@company.com"})
        client = _make_mock_client()
        acl = SessionAccessControl(client, store)

        with pytest.raises(ValueError, match="Cannot share outside organization domain"):
            await acl.share("s1", "alice@company.com", "mallory@external.com", SessionRole.EDITOR)

    async def test_share_session_not_found(self) -> None:
        store = _make_mock_store(None)
        client = _make_mock_client()
        acl = SessionAccessControl(client, store)

        with pytest.raises(PermissionError, match="Session not found"):
            await acl.share("nonexistent", "alice@co.com", "bob@co.com", SessionRole.EDITOR)

    async def test_share_normalizes_emails(self) -> None:
        store = _make_mock_store({"id": "s1", "user": "alice@co.com"})
        client = _make_mock_client()
        acl = SessionAccessControl(client, store)

        collab = await acl.share(
            "s1",
            "accounts.google.com:Alice@CO.COM",
            "accounts.google.com:Bob@CO.COM",
            SessionRole.VIEWER,
        )
        assert collab.email == "bob@co.com"
        assert collab.added_by == "alice@co.com"


# ── revoke ───────────────────────────────────────────────────


class TestRevoke:
    """Tests for the revoke method."""

    async def test_revoke_success(self) -> None:
        store = _make_mock_store({"id": "s1", "user": "alice@co.com"})
        client = _make_mock_client()

        # Mock the collaborator doc as existing
        existing_doc = _MockDocSnapshot("bob@co.com", {"role": "editor"})
        collab_ref = AsyncMock()
        collab_ref.get = AsyncMock(return_value=existing_doc)
        collab_ref.delete = AsyncMock()

        sub_col = MagicMock()
        sub_col.document = MagicMock(return_value=collab_ref)
        session_doc_ref = MagicMock()
        session_doc_ref.collection = MagicMock(return_value=sub_col)
        col_ref = MagicMock()
        col_ref.document = MagicMock(return_value=session_doc_ref)
        client.collection = MagicMock(return_value=col_ref)

        acl = SessionAccessControl(client, store)
        result = await acl.revoke("s1", "alice@co.com", "bob@co.com")
        assert result is True
        collab_ref.delete.assert_called_once()

    async def test_revoke_non_owner_raises(self) -> None:
        store = _make_mock_store({"id": "s1", "user": "alice@co.com"})
        client = _make_mock_client()
        acl = SessionAccessControl(client, store)

        with pytest.raises(PermissionError, match="owner"):
            await acl.revoke("s1", "bob@co.com", "carol@co.com")

    async def test_revoke_nonexistent_collaborator(self) -> None:
        store = _make_mock_store({"id": "s1", "user": "alice@co.com"})
        client = _make_mock_client()

        missing_doc = _MockDocSnapshot("eve@co.com", {}, exists=False)
        collab_ref = AsyncMock()
        collab_ref.get = AsyncMock(return_value=missing_doc)

        sub_col = MagicMock()
        sub_col.document = MagicMock(return_value=collab_ref)
        session_doc_ref = MagicMock()
        session_doc_ref.collection = MagicMock(return_value=sub_col)
        col_ref = MagicMock()
        col_ref.document = MagicMock(return_value=session_doc_ref)
        client.collection = MagicMock(return_value=col_ref)

        acl = SessionAccessControl(client, store)
        result = await acl.revoke("s1", "alice@co.com", "eve@co.com")
        assert result is False


# ── list_collaborators ───────────────────────────────────────


class TestListCollaborators:
    """Tests for the list_collaborators method."""

    async def test_list_returns_collaborators(self) -> None:
        store = _make_mock_store({"id": "s1", "user": "alice@co.com"})
        client = _make_mock_client()

        collab_doc_1 = _MockDocSnapshot("bob@co.com", {
            "role": "editor",
            "added_at": "2026-01-01T00:00:00Z",
            "added_by": "alice@co.com",
        })
        collab_doc_2 = _MockDocSnapshot("carol@co.com", {
            "role": "viewer",
            "added_at": "2026-01-02T00:00:00Z",
            "added_by": "alice@co.com",
        })

        # For check_access (owner lookup): need collab_ref.get
        # For list iteration: need sub_col.stream
        # Since check_access is called first (owner = alice), it returns granted.
        # Then list_collaborators iterates the subcollection.

        # We need two different collection chains:
        # 1) check_access uses: client.collection(COL).document(sid).collection(SUB).document(user).get()
        # 2) list_collaborators uses: client.collection(COL).document(sid).collection(SUB).stream()

        sub_col = MagicMock()
        sub_col.stream = MagicMock(return_value=_async_iter([collab_doc_1, collab_doc_2]))
        # check_access calls .document(user).get() — since user is alice (owner), it short-circuits
        sub_col.document = MagicMock(return_value=AsyncMock(get=AsyncMock()))

        session_doc_ref = MagicMock()
        session_doc_ref.collection = MagicMock(return_value=sub_col)

        col_ref = MagicMock()
        col_ref.document = MagicMock(return_value=session_doc_ref)
        client.collection = MagicMock(return_value=col_ref)

        acl = SessionAccessControl(client, store)
        collaborators = await acl.list_collaborators("s1", "alice@co.com")
        assert len(collaborators) == 2
        assert collaborators[0].email == "bob@co.com"
        assert collaborators[0].role == SessionRole.EDITOR
        assert collaborators[1].email == "carol@co.com"
        assert collaborators[1].role == SessionRole.VIEWER

    async def test_list_denied_for_non_collaborator(self) -> None:
        """User with no access gets an empty list."""
        store = _make_mock_store({"id": "s1", "user": "alice@co.com"})
        client = _make_mock_client()

        # check_access for eve: not owner, feature flag off by default → denied
        acl = SessionAccessControl(client, store)
        collaborators = await acl.list_collaborators("s1", "eve@co.com")
        assert collaborators == []


# ── list_accessible_sessions ─────────────────────────────────


class TestListAccessibleSessions:
    """Tests for the list_accessible_sessions method."""

    async def test_returns_shared_sessions(self) -> None:
        store = MagicMock()
        client = _make_mock_client()

        # Mock collection_group query: returns collab docs
        session_ref_mock = AsyncMock()
        session_doc = _MockDocSnapshot("s1", {
            "user": "alice@co.com",
            "name": "Shared Session",
            "updated_at": "2026-01-01",
        })
        session_ref_mock.get = AsyncMock(return_value=session_doc)

        parent_col = MagicMock()
        parent_col.parent = session_ref_mock

        collab_doc = _MockDocSnapshot("bob@co.com", {"role": "editor"}, parent=parent_col)

        group_query = MagicMock()
        group_query.stream = MagicMock(return_value=_async_iter([collab_doc]))
        client.collection_group = MagicMock(return_value=group_query)

        acl = SessionAccessControl(client, store)
        sessions = await acl.list_accessible_sessions("bob@co.com", limit=10)
        assert len(sessions) == 1
        assert sessions[0]["id"] == "s1"
        assert sessions[0]["role"] == "editor"

    async def test_respects_limit(self) -> None:
        store = MagicMock()
        client = _make_mock_client()

        # Create 3 collab docs but set limit=2
        collab_docs = []
        for i in range(3):
            session_ref = AsyncMock()
            session_doc = _MockDocSnapshot(
                f"s{i}",
                {"user": "alice@co.com", "name": f"Session {i}", "updated_at": "2026-01-01"},
            )
            session_ref.get = AsyncMock(return_value=session_doc)
            parent_col = MagicMock()
            parent_col.parent = session_ref
            collab_doc = _MockDocSnapshot("bob@co.com", {"role": "viewer"}, parent=parent_col)
            collab_docs.append(collab_doc)

        group_query = MagicMock()
        group_query.stream = MagicMock(return_value=_async_iter(collab_docs))
        client.collection_group = MagicMock(return_value=group_query)

        acl = SessionAccessControl(client, store)
        sessions = await acl.list_accessible_sessions("bob@co.com", limit=2)
        assert len(sessions) == 2

    async def test_skips_non_matching_users(self) -> None:
        store = MagicMock()
        client = _make_mock_client()

        # Doc for a different user
        other_collab = _MockDocSnapshot("carol@co.com", {"role": "editor"})

        group_query = MagicMock()
        group_query.stream = MagicMock(return_value=_async_iter([other_collab]))
        client.collection_group = MagicMock(return_value=group_query)

        acl = SessionAccessControl(client, store)
        sessions = await acl.list_accessible_sessions("bob@co.com")
        assert sessions == []

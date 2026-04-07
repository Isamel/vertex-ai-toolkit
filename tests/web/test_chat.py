"""Tests for Chat routes — Task 3.4.

Covers:
- Route registration (chat + sessions)
- GET /chat returns new chat page
- GET /chat/{session_id} with valid/invalid sessions
- GET /sessions renders session list
- DELETE /sessions/{id} deletes a session
- POST /chat/{id}/message returns SSE content-type
- Navigation bar contains Chat and Sessions links
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip(
    "fastapi",
    reason="FastAPI not available; install the 'web' extra to run web tests.",
)

from httpx import ASGITransport, AsyncClient

from vaig.web.app import create_app

# ── Helpers ──────────────────────────────────────────────────


class _MockStream:
    """Async-iterable mock that behaves like StreamResult."""

    def __init__(self, chunks: list[str], usage: dict | None = None) -> None:
        self._chunks = chunks
        self.usage = usage or {}
        self.text = "".join(chunks)

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for chunk in self._chunks:
            yield chunk


class _FakeSessionStore:
    """In-memory session store for testing."""

    def __init__(self) -> None:
        self.sessions: dict[str, dict] = {}
        self.messages: dict[str, list[dict]] = {}

    async def async_create_session(
        self,
        name: str,
        model: str,
        skill: str | None = None,
        metadata: dict | None = None,
        *,
        user: str = "",
    ) -> str:
        sid = "test-session-001"
        self.sessions[sid] = {
            "id": sid,
            "name": name,
            "model": model,
            "skill": skill,
            "user": user,
        }
        self.messages[sid] = []
        return sid

    async def async_add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        model: str | None = None,
        token_count: int = 0,
    ) -> None:
        if session_id not in self.messages:
            self.messages[session_id] = []
        self.messages[session_id].append(
            {"role": role, "content": content, "model": model}
        )

    async def async_get_messages(self, session_id: str, **kwargs) -> list[dict]:
        return self.messages.get(session_id, [])

    async def async_list_sessions(self, **kwargs) -> list[dict]:
        return list(self.sessions.values())

    async def async_get_session(self, session_id: str) -> dict | None:
        return self.sessions.get(session_id)

    async def async_delete_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.messages.pop(session_id, None)
            return True
        return False


class _FakeSessionAccess:
    """In-memory session access control for testing.

    Grants OWNER to the user stored in ``session["user"]`` and looks up
    other users in an internal collaborator map.
    """

    def __init__(self, store: _FakeSessionStore) -> None:
        self._store = store
        # Mapping: (session_id, email) -> SessionRole
        self.collaborators: dict[tuple[str, str], str] = {}
        # Accessible sessions returned by list_accessible_sessions
        self.shared_sessions: list[dict] = []

    async def check_access(
        self,
        session_id: str,
        user: str,
        *,
        required: str | None = None,
    ) -> MagicMock:
        """Return a mock AccessResult based on store ownership and collaborators."""
        from vaig.core.models import AccessResult, SessionRole

        if required is None:
            required = SessionRole.VIEWER  # type: ignore[assignment]

        session = await self._store.async_get_session(session_id)
        if session is None:
            return AccessResult(granted=False)

        owner = session.get("user", "")
        if user == owner:
            return AccessResult(granted=True, role=SessionRole.OWNER)

        # Check collaborators
        key = (session_id, user)
        if key in self.collaborators:
            role = SessionRole(self.collaborators[key])
            if role >= required:
                return AccessResult(granted=True, role=role)
            return AccessResult(granted=False, role=role)

        return AccessResult(granted=False)

    async def share(self, session_id, owner, target_email, role):
        from vaig.core.models import SessionCollaborator

        session = await self._store.async_get_session(session_id)
        if session is None:
            msg = "Session not found"
            raise PermissionError(msg)
        if owner != session.get("user", ""):
            msg = "Only the session owner can share"
            raise PermissionError(msg)
        if target_email == owner:
            msg = "Cannot share with session owner"
            raise ValueError(msg)

        self.collaborators[(session_id, target_email)] = role.value
        return SessionCollaborator(
            email=target_email, role=role, added_at="now", added_by=owner,
        )

    async def revoke(self, session_id, owner, target_email):
        session = await self._store.async_get_session(session_id)
        if session is None:
            msg = "Session not found"
            raise PermissionError(msg)
        if owner != session.get("user", ""):
            msg = "Only the session owner can revoke access"
            raise PermissionError(msg)
        key = (session_id, target_email)
        if key in self.collaborators:
            del self.collaborators[key]
            return True
        return False

    async def list_collaborators(self, session_id, user):
        return []

    async def list_accessible_sessions(self, user, *, limit=20):
        return self.shared_sessions


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def app():
    """Create a fresh app instance for each test."""
    return create_app()


@pytest.fixture()
def app_with_store():
    """Create an app instance with a fake session store and access control."""
    application = create_app()
    store = _FakeSessionStore()
    application.state.session_store = store
    application.state.session_access = _FakeSessionAccess(store)
    return application


@pytest.fixture()
async def client(app):
    """Async HTTP client wired to the test app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture()
async def client_with_store(app_with_store):
    """Async HTTP client wired to the test app with session store."""
    transport = ASGITransport(app=app_with_store)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── Route Registration ───────────────────────────────────────


def test_chat_routes_registered(app) -> None:
    """App must have chat and session routes."""
    route_paths = [r.path for r in app.routes if hasattr(r, "path")]
    assert "/chat" in route_paths
    assert "/chat/{session_id}" in route_paths
    assert "/chat/{session_id}/message" in route_paths
    assert "/sessions" in route_paths
    assert "/sessions/{session_id}" in route_paths


def test_session_store_initialised_as_none(app) -> None:
    """App state should have session_store set to None by default."""
    assert hasattr(app.state, "session_store")
    assert app.state.session_store is None


# ── GET /chat ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_new_returns_200(client) -> None:
    """GET /chat returns 200."""
    resp = await client.get("/chat")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_chat_new_is_html(client) -> None:
    """GET /chat returns HTML content."""
    resp = await client.get("/chat")
    assert "text/html" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_chat_new_contains_form(client) -> None:
    """GET /chat should contain the message input form."""
    resp = await client.get("/chat")
    body = resp.text
    assert 'id="chat-form"' in body
    assert 'id="message-input"' in body
    assert "New conversation" in body


@pytest.mark.asyncio
async def test_chat_new_contains_settings_panel(client) -> None:
    """GET /chat should contain the collapsible settings panel with project and model fields."""
    resp = await client.get("/chat")
    body = resp.text
    assert 'id="settings-panel"' in body
    assert 'id="settings-toggle"' in body
    assert 'id="chat-project"' in body
    assert 'id="chat-model"' in body
    assert "gemini-2.5-pro" in body
    assert "gemini-2.5-flash" in body


# ── GET /chat/{session_id} ───────────────────────────────────


@pytest.mark.asyncio
async def test_chat_resume_missing_session_returns_404() -> None:
    """GET /chat/{id} returns 404 when session not found."""
    app = create_app()
    app.state.session_store = _FakeSessionStore()

    with patch(
        "vaig.web.routes.chat.get_current_user", return_value="test@test.com"
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/chat/nonexistent-id")
            assert resp.status_code == 404
            assert "not found" in resp.text.lower()


@pytest.mark.asyncio
async def test_chat_resume_existing_session_returns_200() -> None:
    """GET /chat/{id} returns 200 with message history for valid session."""
    app = create_app()
    store = _FakeSessionStore()
    store.sessions["sess-123"] = {
        "id": "sess-123",
        "name": "Test Chat",
        "model": "gemini-2.0-flash",
        "user": "test@test.com",
    }
    store.messages["sess-123"] = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    app.state.session_store = store
    app.state.session_access = _FakeSessionAccess(store)

    with patch(
        "vaig.web.routes.chat.get_current_user", return_value="test@test.com"
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/chat/sess-123")
            assert resp.status_code == 200
            assert "Hello" in resp.text
            assert "Hi there" in resp.text


# ── GET /sessions ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sessions_list_returns_200() -> None:
    """GET /sessions returns 200."""
    app = create_app()
    app.state.session_store = _FakeSessionStore()

    with patch(
        "vaig.web.routes.chat.get_current_user", return_value="test@test.com"
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/sessions")
            assert resp.status_code == 200
            assert "text/html" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_sessions_list_shows_sessions() -> None:
    """GET /sessions displays existing session names."""
    app = create_app()
    store = _FakeSessionStore()
    store.sessions["s1"] = {
        "id": "s1",
        "name": "Debug Session",
        "model": "gemini-2.0-flash",
    }
    app.state.session_store = store

    with patch(
        "vaig.web.routes.chat.get_current_user", return_value="test@test.com"
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/sessions")
            assert "Debug Session" in resp.text


@pytest.mark.asyncio
async def test_sessions_list_empty_state() -> None:
    """GET /sessions shows empty state when no sessions exist."""
    app = create_app()
    app.state.session_store = _FakeSessionStore()

    with patch(
        "vaig.web.routes.chat.get_current_user", return_value="test@test.com"
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/sessions")
            assert "No sessions yet" in resp.text


# ── DELETE /sessions/{id} ────────────────────────────────────


@pytest.mark.asyncio
async def test_session_delete_returns_200() -> None:
    """DELETE /sessions/{id} returns 200 on success."""
    app = create_app()
    store = _FakeSessionStore()
    store.sessions["del-me"] = {
        "id": "del-me",
        "name": "To Delete",
        "model": "gemini-2.0-flash",
        "user": "test@test.com",
    }
    app.state.session_store = store
    app.state.session_access = _FakeSessionAccess(store)

    with patch(
        "vaig.web.routes.chat.get_current_user", return_value="test@test.com"
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.delete("/sessions/del-me")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "deleted"
            assert "del-me" not in store.sessions


@pytest.mark.asyncio
async def test_session_delete_returns_404_for_missing() -> None:
    """DELETE /sessions/{id} returns 404 when session does not exist."""
    app = create_app()
    store = _FakeSessionStore()
    app.state.session_store = store
    app.state.session_access = _FakeSessionAccess(store)

    with patch(
        "vaig.web.routes.chat.get_current_user", return_value="test@test.com"
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.delete("/sessions/nonexistent")
            assert resp.status_code == 404


@pytest.mark.asyncio
async def test_session_delete_returns_503_without_store() -> None:
    """DELETE /sessions/{id} returns 503 when no session store configured."""
    app = create_app()
    # session_store is None by default

    with patch(
        "vaig.web.routes.chat.get_current_user", return_value="test@test.com"
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.delete("/sessions/anything")
            assert resp.status_code == 503


# ── POST /chat/{id}/message ─────────────────────────────────


@pytest.mark.asyncio
async def test_chat_message_returns_sse_content_type() -> None:
    """POST /chat/{id}/message should return text/event-stream."""
    app = create_app()
    store = _FakeSessionStore()
    store.sessions["s1"] = {"id": "s1", "name": "Test", "model": "gemini-2.0-flash", "user": "test@test.com"}
    store.messages["s1"] = []
    app.state.session_store = store
    app.state.session_access = _FakeSessionAccess(store)

    mock_stream = _MockStream(["Hello"])
    mock_orchestrator = MagicMock()
    mock_orchestrator.async_execute_single = AsyncMock(return_value=mock_stream)

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="test@test.com"),
        patch(
            "vaig.web.deps.Settings.from_overrides",
            return_value=MagicMock(models=MagicMock(default="gemini-2.0-flash")),
        ),
        patch(
            "vaig.web.deps.build_container",
            return_value=MagicMock(
                event_bus=MagicMock(subscribe=MagicMock(return_value=lambda: None)),
                gemini_client=MagicMock(),
            ),
        ),
        patch(
            "vaig.agents.orchestrator.Orchestrator",
            return_value=mock_orchestrator,
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/chat/s1/message",
                data={"message": "What is going on?"},
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_chat_message_contains_session_and_chunk_events() -> None:
    """POST /chat/{id}/message should contain session and chunk SSE events."""
    app = create_app()
    store = _FakeSessionStore()
    store.sessions["s2"] = {"id": "s2", "name": "Test", "model": "gemini-2.0-flash", "user": "test@test.com"}
    store.messages["s2"] = []
    app.state.session_store = store
    app.state.session_access = _FakeSessionAccess(store)

    mock_stream = _MockStream(["Hello ", "world"], usage={"tokens": 10})
    mock_orchestrator = MagicMock()
    mock_orchestrator.async_execute_single = AsyncMock(return_value=mock_stream)

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="test@test.com"),
        patch(
            "vaig.web.deps.Settings.from_overrides",
            return_value=MagicMock(models=MagicMock(default="gemini-2.0-flash")),
        ),
        patch(
            "vaig.web.deps.build_container",
            return_value=MagicMock(
                event_bus=MagicMock(subscribe=MagicMock(return_value=lambda: None)),
                gemini_client=MagicMock(),
            ),
        ),
        patch(
            "vaig.agents.orchestrator.Orchestrator",
            return_value=mock_orchestrator,
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/chat/s2/message",
                data={"message": "Hello"},
            )
            body = resp.text
            assert "event: session" in body
            assert "event: chunk" in body
            assert "event: done" in body


@pytest.mark.asyncio
async def test_chat_message_new_session_creates_session() -> None:
    """POST /chat/new/message should create a new session."""
    app = create_app()
    store = _FakeSessionStore()
    app.state.session_store = store

    mock_stream = _MockStream(["OK"])
    mock_orchestrator = MagicMock()
    mock_orchestrator.async_execute_single = AsyncMock(return_value=mock_stream)

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="test@test.com"),
        patch(
            "vaig.web.deps.Settings.from_overrides",
            return_value=MagicMock(models=MagicMock(default="gemini-2.0-flash")),
        ),
        patch(
            "vaig.web.deps.build_container",
            return_value=MagicMock(
                event_bus=MagicMock(subscribe=MagicMock(return_value=lambda: None)),
                gemini_client=MagicMock(),
            ),
        ),
        patch(
            "vaig.agents.orchestrator.Orchestrator",
            return_value=mock_orchestrator,
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/chat/new/message",
                data={"message": "First message"},
            )
            assert resp.status_code == 200
            # Session should have been created
            assert "test-session-001" in store.sessions


@pytest.mark.asyncio
async def test_chat_message_empty_returns_error_event() -> None:
    """POST /chat/{id}/message with empty message returns error SSE."""
    app = create_app()

    with patch(
        "vaig.web.routes.chat.get_current_user", return_value="test@test.com"
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/chat/s1/message",
                data={"message": ""},
            )
            assert resp.status_code == 200
            body = resp.text
            assert "event: error" in body
            assert "Message is required" in body


# ── Navigation ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_nav_contains_chat_and_sessions_links(client) -> None:
    """The nav bar should include Chat and Sessions links."""
    resp = await client.get("/")
    body = resp.text
    assert 'href="/chat"' in body
    assert ">Chat<" in body
    assert 'href="/sessions"' in body
    assert ">Sessions<" in body


# ── Ownership Checks ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_resume_wrong_user_returns_404() -> None:
    """GET /chat/{id} returns 404 when session belongs to a different user."""
    app = create_app()
    store = _FakeSessionStore()
    store.sessions["sess-123"] = {
        "id": "sess-123",
        "name": "Private Chat",
        "model": "gemini-2.0-flash",
        "user": "owner@test.com",
    }
    app.state.session_store = store
    app.state.session_access = _FakeSessionAccess(store)

    with patch(
        "vaig.web.routes.chat.get_current_user", return_value="intruder@test.com"
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/chat/sess-123")
            assert resp.status_code == 404


@pytest.mark.asyncio
async def test_session_delete_wrong_user_returns_404() -> None:
    """DELETE /sessions/{id} returns 404 when session belongs to a different user."""
    app = create_app()
    store = _FakeSessionStore()
    store.sessions["del-me"] = {
        "id": "del-me",
        "name": "Protected",
        "model": "gemini-2.0-flash",
        "user": "owner@test.com",
    }
    app.state.session_store = store
    app.state.session_access = _FakeSessionAccess(store)

    with patch(
        "vaig.web.routes.chat.get_current_user", return_value="intruder@test.com"
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.delete("/sessions/del-me")
            assert resp.status_code == 404
            # Session should NOT be deleted
            assert "del-me" in store.sessions


# ── Ephemeral UUID ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_message_new_without_store_generates_uuid() -> None:
    """POST /chat/new/message without store should generate an ephemeral UUID, not 'new'."""
    app = create_app()
    # session_store is None — no persistence

    mock_stream = _MockStream(["OK"])
    mock_orchestrator = MagicMock()
    mock_orchestrator.async_execute_single = AsyncMock(return_value=mock_stream)

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="test@test.com"),
        patch(
            "vaig.web.deps.Settings.from_overrides",
            return_value=MagicMock(models=MagicMock(default="gemini-2.0-flash")),
        ),
        patch(
            "vaig.web.deps.build_container",
            return_value=MagicMock(
                event_bus=MagicMock(subscribe=MagicMock(return_value=lambda: None)),
                gemini_client=MagicMock(),
            ),
        ),
        patch(
            "vaig.agents.orchestrator.Orchestrator",
            return_value=mock_orchestrator,
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/chat/new/message",
                data={"message": "Hello ephemeral"},
            )
            assert resp.status_code == 200
            body = resp.text
            # The session event should contain a UUID, not "new"
            assert "event: session" in body
            assert '"new"' not in body.split("event: session")[1].split("event:")[0]


# ── Session Validation on Message ────────────────────────────


@pytest.mark.asyncio
async def test_chat_message_nonexistent_session_returns_error() -> None:
    """POST /chat/{id}/message with nonexistent session returns error SSE."""
    app = create_app()
    store = _FakeSessionStore()
    app.state.session_store = store
    app.state.session_access = _FakeSessionAccess(store)

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="test@test.com"),
        patch(
            "vaig.web.deps.Settings.from_overrides",
            return_value=MagicMock(models=MagicMock(default="gemini-2.0-flash")),
        ),
        patch(
            "vaig.web.deps.build_container",
            return_value=MagicMock(
                event_bus=MagicMock(subscribe=MagicMock(return_value=lambda: None)),
                gemini_client=MagicMock(),
            ),
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/chat/nonexistent-session/message",
                data={"message": "Hello?"},
            )
            assert resp.status_code == 200
            body = resp.text
            assert "event: error" in body
            assert "not found" in body.lower()


# ── ACL Integration (Task 3.6) ──────────────────────────────


def _make_acl_app() -> tuple:
    """Create app with store, access control, and a pre-populated session."""
    app = create_app()
    store = _FakeSessionStore()
    store.sessions["shared-sess"] = {
        "id": "shared-sess",
        "name": "Shared Session",
        "model": "gemini-2.0-flash",
        "user": "owner@test.com",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    store.messages["shared-sess"] = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    access = _FakeSessionAccess(store)
    app.state.session_store = store
    app.state.session_access = access
    return app, store, access


@pytest.mark.asyncio
async def test_acl_viewer_can_read_session() -> None:
    """GET /chat/{id} returns 200 when the user has viewer access."""
    app, _store, access = _make_acl_app()
    access.collaborators[("shared-sess", "viewer@test.com")] = "viewer"

    with patch(
        "vaig.web.routes.chat.get_current_user", return_value="viewer@test.com"
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/chat/shared-sess")
            assert resp.status_code == 200
            assert "Hello" in resp.text


@pytest.mark.asyncio
async def test_acl_viewer_cannot_send_message() -> None:
    """POST /chat/{id}/message returns permission error for viewer."""
    app, _store, access = _make_acl_app()
    access.collaborators[("shared-sess", "viewer@test.com")] = "viewer"

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="viewer@test.com"),
        patch(
            "vaig.web.deps.Settings.from_overrides",
            return_value=MagicMock(models=MagicMock(default="gemini-2.0-flash")),
        ),
        patch(
            "vaig.web.deps.build_container",
            return_value=MagicMock(
                event_bus=MagicMock(subscribe=MagicMock(return_value=lambda: None)),
                gemini_client=MagicMock(),
            ),
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/chat/shared-sess/message",
                data={"message": "I am a viewer"},
            )
            assert resp.status_code == 200
            body = resp.text
            assert "event: error" in body
            assert "permission" in body.lower() or "Insufficient" in body


@pytest.mark.asyncio
async def test_acl_editor_can_send_message() -> None:
    """POST /chat/{id}/message succeeds for editor."""
    app, _store, access = _make_acl_app()
    access.collaborators[("shared-sess", "editor@test.com")] = "editor"

    mock_stream = _MockStream(["Response"])
    mock_orchestrator = MagicMock()
    mock_orchestrator.async_execute_single = AsyncMock(return_value=mock_stream)

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="editor@test.com"),
        patch(
            "vaig.web.deps.Settings.from_overrides",
            return_value=MagicMock(models=MagicMock(default="gemini-2.0-flash")),
        ),
        patch(
            "vaig.web.deps.build_container",
            return_value=MagicMock(
                event_bus=MagicMock(subscribe=MagicMock(return_value=lambda: None)),
                gemini_client=MagicMock(),
            ),
        ),
        patch(
            "vaig.agents.orchestrator.Orchestrator",
            return_value=mock_orchestrator,
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/chat/shared-sess/message",
                data={"message": "I am an editor"},
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            assert "event: session" in resp.text


@pytest.mark.asyncio
async def test_acl_non_collaborator_denied_read() -> None:
    """GET /chat/{id} returns 404 for non-collaborator."""
    app, _store, _access = _make_acl_app()

    with patch(
        "vaig.web.routes.chat.get_current_user", return_value="stranger@test.com"
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/chat/shared-sess")
            assert resp.status_code == 404


@pytest.mark.asyncio
async def test_acl_non_collaborator_denied_message() -> None:
    """POST /chat/{id}/message returns permission error for non-collaborator."""
    app, _store, _access = _make_acl_app()

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="stranger@test.com"),
        patch(
            "vaig.web.deps.Settings.from_overrides",
            return_value=MagicMock(models=MagicMock(default="gemini-2.0-flash")),
        ),
        patch(
            "vaig.web.deps.build_container",
            return_value=MagicMock(
                event_bus=MagicMock(subscribe=MagicMock(return_value=lambda: None)),
                gemini_client=MagicMock(),
            ),
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/chat/shared-sess/message",
                data={"message": "Let me in"},
            )
            assert resp.status_code == 200
            body = resp.text
            assert "event: error" in body
            assert "permission" in body.lower() or "Insufficient" in body


@pytest.mark.asyncio
async def test_acl_shared_sessions_appear_in_list() -> None:
    """GET /sessions shows shared sessions alongside owned sessions."""
    app, store, access = _make_acl_app()
    # Add an owned session for the listing user
    store.sessions["my-sess"] = {
        "id": "my-sess",
        "name": "My Session",
        "model": "gemini-2.0-flash",
        "user": "viewer@test.com",
        "updated_at": "2026-01-02T00:00:00Z",
    }
    # The shared session should appear from list_accessible_sessions
    access.shared_sessions = [
        {
            "id": "shared-sess",
            "name": "Shared Session",
            "model": "gemini-2.0-flash",
            "user": "owner@test.com",
            "role": "viewer",
            "updated_at": "2026-01-01T00:00:00Z",
        }
    ]

    with patch(
        "vaig.web.routes.chat.get_current_user", return_value="viewer@test.com"
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/sessions")
            assert resp.status_code == 200
            body = resp.text
            assert "My Session" in body
            assert "Shared Session" in body


@pytest.mark.asyncio
async def test_acl_owner_can_delete() -> None:
    """DELETE /sessions/{id} succeeds for session owner."""
    app, store, _access = _make_acl_app()

    with patch(
        "vaig.web.routes.chat.get_current_user", return_value="owner@test.com"
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.delete("/sessions/shared-sess")
            assert resp.status_code == 200
            assert resp.json()["status"] == "deleted"
            assert "shared-sess" not in store.sessions


@pytest.mark.asyncio
async def test_acl_non_owner_cannot_delete() -> None:
    """DELETE /sessions/{id} returns 404 for non-owner (even with editor access)."""
    app, store, access = _make_acl_app()
    access.collaborators[("shared-sess", "editor@test.com")] = "editor"

    with patch(
        "vaig.web.routes.chat.get_current_user", return_value="editor@test.com"
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.delete("/sessions/shared-sess")
            assert resp.status_code == 404
            # Session must still exist
            assert "shared-sess" in store.sessions

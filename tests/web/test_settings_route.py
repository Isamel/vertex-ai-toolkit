"""Tests for Settings routes — Task 4.2.

Covers:
- Route registration (/settings GET + POST)
- GET /settings returns 200 with form
- GET /settings?session={id} pre-fills from session config
- GET /settings?session={id} shows error for unknown session
- POST /settings validates temperature range
- POST /settings validates max_tokens is positive int
- POST /settings saves config and redirects on success
- POST /settings re-renders with errors on validation failure
- POST /settings shows error when no session store configured
- Navigation bar contains Settings link
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip(
    "fastapi",
    reason="FastAPI not available; install the 'web' extra to run web tests.",
)

from httpx import ASGITransport, AsyncClient

from vaig.web.app import create_app

# ── Helpers ──────────────────────────────────────────────────


class _FakeSessionStore:
    """In-memory session store with config support for testing."""

    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}
        self.messages: dict[str, list[dict[str, Any]]] = {}
        self.configs: dict[str, dict[str, Any]] = {}

    async def async_create_session(
        self,
        name: str,
        model: str,
        skill: str | None = None,
        metadata: dict[str, Any] | None = None,
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

    async def async_get_messages(self, session_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        return self.messages.get(session_id, [])

    async def async_list_sessions(self, **kwargs: Any) -> list[dict[str, Any]]:
        return list(self.sessions.values())

    async def async_get_session(self, session_id: str) -> dict[str, Any] | None:
        return self.sessions.get(session_id)

    async def async_delete_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.messages.pop(session_id, None)
            return True
        return False

    async def async_save_config(
        self, session_id: str, config: dict[str, Any], user: str
    ) -> bool:
        session = self.sessions.get(session_id)
        if session is None or session.get("user") != user:
            return False
        # Mirror Firestore behaviour: store config inside the session doc
        session["config"] = config
        self.configs[session_id] = config
        return True

    async def async_get_config(
        self, session_id: str, user: str
    ) -> dict[str, Any] | None:
        session = self.sessions.get(session_id)
        if session is None or session.get("user") != user:
            return None
        return self.configs.get(session_id)


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def app():
    """Create a fresh app instance."""
    return create_app()


@pytest.fixture()
def app_with_store():
    """Create app with fake session store pre-populated."""
    application = create_app()
    store = _FakeSessionStore()
    config_data = {
        "project": "stored-project",
        "model": "gemini-2.5-flash",
        "temperature": 0.3,
    }
    store.sessions["sess-1"] = {
        "id": "sess-1",
        "name": "My Chat",
        "model": "gemini-2.5-pro",
        "user": "test@test.com",
        "config": config_data,
    }
    store.configs["sess-1"] = config_data
    application.state.session_store = store
    return application


# ── Route Registration ───────────────────────────────────────


def test_settings_routes_registered(app: Any) -> None:
    """App must have settings GET and POST routes."""
    route_paths = [r.path for r in app.routes if hasattr(r, "path")]
    assert "/settings" in route_paths


# ── GET /settings ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_settings_get_returns_200() -> None:
    """GET /settings returns 200."""
    app = create_app()
    with patch(
        "vaig.web.routes.settings.get_current_user", return_value="test@test.com"
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/settings")
            assert resp.status_code == 200
            assert "text/html" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_settings_get_contains_form_elements() -> None:
    """GET /settings contains all expected form fields."""
    app = create_app()
    with patch(
        "vaig.web.routes.settings.get_current_user", return_value="test@test.com"
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/settings")
            body = resp.text
            assert 'name="project"' in body
            assert 'name="region"' in body
            assert 'name="model"' in body
            assert 'name="temperature"' in body
            assert 'name="max_tokens"' in body
            assert 'name="system_instructions"' in body
            assert "Save Settings" in body


@pytest.mark.asyncio
async def test_settings_get_prefills_from_session_config(app_with_store: Any) -> None:
    """GET /settings?session={id} pre-fills form from stored config."""
    with patch(
        "vaig.web.routes.settings.get_current_user", return_value="test@test.com"
    ):
        transport = ASGITransport(app=app_with_store)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/settings?session=sess-1")
            body = resp.text
            assert "stored-project" in body
            assert "gemini-2.5-flash" in body


@pytest.mark.asyncio
async def test_settings_get_unknown_session_shows_error(app_with_store: Any) -> None:
    """GET /settings?session=nonexistent shows error."""
    with patch(
        "vaig.web.routes.settings.get_current_user", return_value="test@test.com"
    ):
        transport = ASGITransport(app=app_with_store)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/settings?session=nonexistent")
            body = resp.text
            assert "not found" in body.lower()


@pytest.mark.asyncio
async def test_settings_get_wrong_user_shows_error(app_with_store: Any) -> None:
    """GET /settings?session={id} with wrong user shows error."""
    with patch(
        "vaig.web.routes.settings.get_current_user", return_value="intruder@test.com"
    ):
        transport = ASGITransport(app=app_with_store)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/settings?session=sess-1")
            body = resp.text
            assert "not found" in body.lower()


# ── POST /settings — validation ──────────────────────────────


@pytest.mark.asyncio
async def test_settings_post_invalid_temperature(app_with_store: Any) -> None:
    """POST /settings with temperature > 2.0 shows validation error."""
    with patch(
        "vaig.web.routes.settings.get_current_user", return_value="test@test.com"
    ):
        transport = ASGITransport(app=app_with_store)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/settings",
                data={
                    "session_id": "sess-1",
                    "temperature": "3.0",
                    "project": "test",
                },
            )
            assert resp.status_code == 200
            body = resp.text
            assert "Temperature must be between 0.0 and 2.0" in body


@pytest.mark.asyncio
async def test_settings_post_invalid_temperature_negative(app_with_store: Any) -> None:
    """POST /settings with temperature < 0 shows validation error."""
    with patch(
        "vaig.web.routes.settings.get_current_user", return_value="test@test.com"
    ):
        transport = ASGITransport(app=app_with_store)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/settings",
                data={
                    "session_id": "sess-1",
                    "temperature": "-0.5",
                },
            )
            assert resp.status_code == 200
            assert "Temperature must be between 0.0 and 2.0" in resp.text


@pytest.mark.asyncio
async def test_settings_post_invalid_max_tokens(app_with_store: Any) -> None:
    """POST /settings with max_tokens <= 0 shows validation error."""
    with patch(
        "vaig.web.routes.settings.get_current_user", return_value="test@test.com"
    ):
        transport = ASGITransport(app=app_with_store)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/settings",
                data={
                    "session_id": "sess-1",
                    "max_tokens": "0",
                },
            )
            assert resp.status_code == 200
            assert "Max tokens must be a positive integer" in resp.text


@pytest.mark.asyncio
async def test_settings_post_non_numeric_max_tokens(app_with_store: Any) -> None:
    """POST /settings with non-numeric max_tokens shows validation error."""
    with patch(
        "vaig.web.routes.settings.get_current_user", return_value="test@test.com"
    ):
        transport = ASGITransport(app=app_with_store)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/settings",
                data={
                    "session_id": "sess-1",
                    "max_tokens": "not-a-number",
                },
            )
            assert resp.status_code == 200
            assert "Max tokens must be an integer" in resp.text


# ── POST /settings — success ────────────────────────────────


@pytest.mark.asyncio
async def test_settings_post_saves_and_redirects(app_with_store: Any) -> None:
    """POST /settings with valid data saves config and redirects to chat."""
    with patch(
        "vaig.web.routes.settings.get_current_user", return_value="test@test.com"
    ):
        transport = ASGITransport(app=app_with_store)
        async with AsyncClient(
            transport=transport, base_url="http://test", follow_redirects=False
        ) as ac:
            resp = await ac.post(
                "/settings",
                data={
                    "session_id": "sess-1",
                    "project": "new-project",
                    "model": "gemini-2.5-flash",
                    "temperature": "0.5",
                    "max_tokens": "8192",
                },
            )
            assert resp.status_code == 303
            assert resp.headers["location"] == "/chat/sess-1"

            # Verify config was actually saved
            store = app_with_store.state.session_store
            saved = await store.async_get_config("sess-1", "test@test.com")
            assert saved is not None
            assert saved["project"] == "new-project"
            assert saved["model"] == "gemini-2.5-flash"
            assert saved["temperature"] == 0.5
            assert saved["max_tokens"] == 8192


@pytest.mark.asyncio
async def test_settings_post_no_session_shows_error() -> None:
    """POST /settings without session_id shows error."""
    app = create_app()
    app.state.session_store = _FakeSessionStore()

    with patch(
        "vaig.web.routes.settings.get_current_user", return_value="test@test.com"
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/settings",
                data={"project": "test"},
            )
            assert resp.status_code == 200
            assert "No session selected" in resp.text


@pytest.mark.asyncio
async def test_settings_post_no_store_shows_error() -> None:
    """POST /settings without session store shows error."""
    app = create_app()
    # session_store is None by default

    with patch(
        "vaig.web.routes.settings.get_current_user", return_value="test@test.com"
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/settings",
                data={"session_id": "sess-1", "project": "test"},
            )
            assert resp.status_code == 200
            assert "Session store not configured" in resp.text


@pytest.mark.asyncio
async def test_settings_post_wrong_user_shows_error(app_with_store: Any) -> None:
    """POST /settings with wrong user shows access denied error."""
    with patch(
        "vaig.web.routes.settings.get_current_user", return_value="intruder@test.com"
    ):
        transport = ASGITransport(app=app_with_store)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/settings",
                data={"session_id": "sess-1", "project": "hack"},
            )
            assert resp.status_code == 200
            assert "access denied" in resp.text.lower() or "not found" in resp.text.lower()


# ── Navigation ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_nav_contains_settings_link() -> None:
    """The nav bar should include a Settings link."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/")
        body = resp.text
        assert 'href="/settings"' in body
        assert ">Settings<" in body

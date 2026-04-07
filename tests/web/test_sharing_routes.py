"""Tests for Sharing routes — Task 3.5.

Covers:
- POST /sessions/{id}/share → 200 (owner shares session)
- DELETE /sessions/{id}/share/{email} → 200 (owner revokes)
- GET /sessions/{id}/collaborators → 200 (lists collaborators)
- Non-owner share → 403
- Invalid email → 422
- Self-share → 400
- Feature flag off → 404

All access control is mocked via a fake ``SessionAccessControl``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip(
    "fastapi",
    reason="FastAPI not available; install the 'web' extra to run web tests.",
)

from httpx import ASGITransport, AsyncClient

from vaig.core.models import SessionCollaborator, SessionRole
from vaig.web.app import create_app

# ── Helpers ──────────────────────────────────────────────────


def _make_mock_access(
    *,
    share_result: SessionCollaborator | None = None,
    share_side_effect: Exception | None = None,
    revoke_result: bool = True,
    revoke_side_effect: Exception | None = None,
    collaborators: list[SessionCollaborator] | None = None,
) -> MagicMock:
    """Build a mock ``SessionAccessProtocol``."""
    access = MagicMock()

    if share_side_effect is not None:
        access.share = AsyncMock(side_effect=share_side_effect)
    elif share_result is not None:
        access.share = AsyncMock(return_value=share_result)
    else:
        access.share = AsyncMock(
            return_value=SessionCollaborator(
                email="bob@test.com",
                role=SessionRole.EDITOR,
                added_at="2026-01-01T00:00:00Z",
                added_by="owner@test.com",
            )
        )

    if revoke_side_effect is not None:
        access.revoke = AsyncMock(side_effect=revoke_side_effect)
    else:
        access.revoke = AsyncMock(return_value=revoke_result)

    access.list_collaborators = AsyncMock(return_value=collaborators or [])
    return access


def _make_app(
    access: MagicMock | None = None,
    *,
    flag_on: bool = True,
) -> tuple:
    """Create app with session_access wired."""
    app = create_app()
    if access is None:
        access = _make_mock_access()
    app.state.session_access = access

    env = {"VAIG_WEB_SHARED_SESSIONS": "true" if flag_on else "false"}
    return app, access, env


# ── POST /sessions/{id}/share ────────────────────────────────


@pytest.mark.asyncio
async def test_share_session_returns_200() -> None:
    """POST /sessions/{id}/share returns 200 on success."""
    app, _access, env = _make_app()

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="owner@test.com"),
        patch("vaig.web.routes.sharing.get_current_user", return_value="owner@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/sessions/s1/share",
                json={"email": "bob@test.com", "role": "editor"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "shared"
            assert data["email"] == "bob@test.com"
            assert data["role"] == "editor"


@pytest.mark.asyncio
async def test_share_non_owner_returns_403() -> None:
    """POST /sessions/{id}/share returns 403 for non-owner."""
    access = _make_mock_access(
        share_side_effect=PermissionError("Only the session owner can share"),
    )
    app, _access, env = _make_app(access)

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="intruder@test.com"),
        patch("vaig.web.routes.sharing.get_current_user", return_value="intruder@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/sessions/s1/share",
                json={"email": "bob@test.com", "role": "editor"},
            )
            assert resp.status_code == 403
            assert "owner" in resp.json()["error"].lower()


@pytest.mark.asyncio
async def test_share_invalid_email_returns_422() -> None:
    """POST /sessions/{id}/share returns 422 for invalid email format."""
    app, _access, env = _make_app()

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="owner@test.com"),
        patch("vaig.web.routes.sharing.get_current_user", return_value="owner@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/sessions/s1/share",
                json={"email": "not-an-email", "role": "editor"},
            )
            assert resp.status_code == 422


@pytest.mark.asyncio
async def test_share_invalid_role_returns_422() -> None:
    """POST /sessions/{id}/share returns 422 for invalid role."""
    app, _access, env = _make_app()

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="owner@test.com"),
        patch("vaig.web.routes.sharing.get_current_user", return_value="owner@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/sessions/s1/share",
                json={"email": "bob@test.com", "role": "admin"},
            )
            assert resp.status_code == 422


@pytest.mark.asyncio
async def test_share_self_returns_400() -> None:
    """POST /sessions/{id}/share returns 400 for self-share."""
    access = _make_mock_access(
        share_side_effect=ValueError("Cannot share with session owner"),
    )
    app, _access, env = _make_app(access)

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="owner@test.com"),
        patch("vaig.web.routes.sharing.get_current_user", return_value="owner@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/sessions/s1/share",
                json={"email": "owner@test.com", "role": "editor"},
            )
            assert resp.status_code == 400
            assert "Cannot share with session owner" in resp.json()["error"]


@pytest.mark.asyncio
async def test_share_flag_off_returns_404() -> None:
    """POST /sessions/{id}/share returns 404 when feature flag is disabled."""
    app, _access, _env = _make_app(flag_on=False)
    env = {"VAIG_WEB_SHARED_SESSIONS": "false"}

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="owner@test.com"),
        patch("vaig.web.routes.sharing.get_current_user", return_value="owner@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/sessions/s1/share",
                json={"email": "bob@test.com", "role": "editor"},
            )
            assert resp.status_code == 404


# ── DELETE /sessions/{id}/share/{email} ──────────────────────


@pytest.mark.asyncio
async def test_revoke_session_returns_200() -> None:
    """DELETE /sessions/{id}/share/{email} returns 200 on success."""
    app, _access, env = _make_app()

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="owner@test.com"),
        patch("vaig.web.routes.sharing.get_current_user", return_value="owner@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.delete("/sessions/s1/share/bob@test.com")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "revoked"
            assert data["email"] == "bob@test.com"


@pytest.mark.asyncio
async def test_revoke_non_owner_returns_403() -> None:
    """DELETE /sessions/{id}/share/{email} returns 403 for non-owner."""
    access = _make_mock_access(
        revoke_side_effect=PermissionError("Only the session owner can revoke access"),
    )
    app, _access, env = _make_app(access)

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="intruder@test.com"),
        patch("vaig.web.routes.sharing.get_current_user", return_value="intruder@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.delete("/sessions/s1/share/bob@test.com")
            assert resp.status_code == 403


@pytest.mark.asyncio
async def test_revoke_nonexistent_returns_404() -> None:
    """DELETE /sessions/{id}/share/{email} returns 404 when collaborator not found."""
    access = _make_mock_access(revoke_result=False)
    app, _access, env = _make_app(access)

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="owner@test.com"),
        patch("vaig.web.routes.sharing.get_current_user", return_value="owner@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.delete("/sessions/s1/share/nobody@test.com")
            assert resp.status_code == 404


@pytest.mark.asyncio
async def test_revoke_flag_off_returns_404() -> None:
    """DELETE /sessions/{id}/share/{email} returns 404 when flag off."""
    app, _access, _env = _make_app(flag_on=False)
    env = {"VAIG_WEB_SHARED_SESSIONS": "false"}

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="owner@test.com"),
        patch("vaig.web.routes.sharing.get_current_user", return_value="owner@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.delete("/sessions/s1/share/bob@test.com")
            assert resp.status_code == 404


# ── GET /sessions/{id}/collaborators ─────────────────────────


@pytest.mark.asyncio
async def test_list_collaborators_returns_200() -> None:
    """GET /sessions/{id}/collaborators returns 200 with collaborator list."""
    collaborators = [
        SessionCollaborator(
            email="bob@test.com",
            role=SessionRole.EDITOR,
            added_at="2026-01-01T00:00:00Z",
            added_by="owner@test.com",
        ),
        SessionCollaborator(
            email="carol@test.com",
            role=SessionRole.VIEWER,
            added_at="2026-01-02T00:00:00Z",
            added_by="owner@test.com",
        ),
    ]
    access = _make_mock_access(collaborators=collaborators)
    app, _access, env = _make_app(access)

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="owner@test.com"),
        patch("vaig.web.routes.sharing.get_current_user", return_value="owner@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/sessions/s1/collaborators")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 2
            assert data[0]["email"] == "bob@test.com"
            assert data[0]["role"] == "editor"
            assert data[1]["email"] == "carol@test.com"
            assert data[1]["role"] == "viewer"


@pytest.mark.asyncio
async def test_list_collaborators_flag_off_returns_404() -> None:
    """GET /sessions/{id}/collaborators returns 404 when flag off."""
    app, _access, _env = _make_app(flag_on=False)
    env = {"VAIG_WEB_SHARED_SESSIONS": "false"}

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="owner@test.com"),
        patch("vaig.web.routes.sharing.get_current_user", return_value="owner@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/sessions/s1/collaborators")
            assert resp.status_code == 404


# ── Route Registration ───────────────────────────────────────


def test_sharing_routes_registered() -> None:
    """App must have sharing routes registered."""
    app = create_app()
    route_paths = [r.path for r in app.routes if hasattr(r, "path")]
    assert "/sessions/{session_id}/share" in route_paths
    assert "/sessions/{session_id}/share/{email}" in route_paths
    assert "/sessions/{session_id}/collaborators" in route_paths


def test_session_access_initialised_as_none() -> None:
    """App state should have session_access set to None by default."""
    app = create_app()
    assert hasattr(app.state, "session_access")
    assert app.state.session_access is None

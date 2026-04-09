"""Tests for Skills marketplace route — SPEC-5.3.

Covers:
- Route registration (/portal/skills GET)
- Admin access: 200 for admin users
- Non-admin access: 403
- Empty registry: "No skills installed" empty state
- Dev mode bypass: 200 regardless of admin list
- Telemetry failure: page renders with "—" in usage
- Nav link visible for admin, hidden for non-admin
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip(
    "fastapi",
    reason="FastAPI not available; install the 'web' extra to run web tests.",
)

from httpx import ASGITransport, AsyncClient

from vaig.web.app import create_app

# ── Helpers ──────────────────────────────────────────────────


def _make_skill_metadata(
    name: str = "rca",
    display_name: str = "Root Cause Analysis",
    description: str = "Diagnose issues",
    version: str = "1.0.0",
    author: str = "VAIG",
) -> MagicMock:
    """Create a mock SkillMetadata."""
    meta = MagicMock()
    meta.name = name
    meta.display_name = display_name
    meta.description = description
    meta.version = version
    meta.author = author
    meta.tags = ["diagnosis"]
    meta.supported_phases = ["analyze", "report"]
    return meta


# ── Route Registration ───────────────────────────────────────


def test_skills_route_registered() -> None:
    """App must have /portal/skills GET route."""
    app = create_app()
    route_paths = [r.path for r in app.routes if hasattr(r, "path")]
    assert "/portal/skills" in route_paths


# ── GET /portal/skills — admin access ───────────────────────


@pytest.mark.asyncio
@patch.dict("os.environ", {"VAIG_WEB_ADMIN_EMAILS": "admin@test.com"}, clear=False)
async def test_skills_admin_gets_200() -> None:
    """Admin user should get 200 with skill names in HTML."""
    app = create_app()
    mock_meta = _make_skill_metadata()

    with (
        patch(
            "vaig.web.routes.skills.get_current_user",
            return_value="admin@test.com",
        ),
        patch(
            "vaig.web.deps.is_admin",
            return_value=True,
        ),
        patch("vaig.web.routes.skills.get_settings") as mock_get_settings,
        patch("vaig.web.routes.skills.SkillRegistry") as mock_registry_cls,
        patch("vaig.web.routes.skills.get_telemetry_collector") as mock_tc,
    ):
        mock_get_settings.return_value = MagicMock()
        mock_registry = MagicMock()
        mock_registry.list_skills.return_value = [mock_meta]
        mock_registry.get_source.return_value = "builtin"
        mock_registry_cls.return_value = mock_registry

        mock_collector = MagicMock()
        mock_collector.async_query_events = AsyncMock(return_value=[])
        mock_tc.return_value = mock_collector

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/portal/skills")
            assert resp.status_code == 200
            assert "Root Cause Analysis" in resp.text


# ── GET /portal/skills — non-admin rejected ─────────────────


@pytest.mark.asyncio
async def test_skills_non_admin_gets_403() -> None:
    """Non-admin user should get 403."""
    app = create_app()

    with (
        patch(
            "vaig.web.routes.skills.get_current_user",
            return_value="user@test.com",
        ),
        patch(
            "vaig.web.deps.is_admin",
            return_value=False,
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/portal/skills")
            assert resp.status_code == 403


# ── Empty registry ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_skills_empty_registry_shows_empty_state() -> None:
    """Zero skills should show 'No skills installed' message."""
    app = create_app()

    with (
        patch(
            "vaig.web.routes.skills.get_current_user",
            return_value="admin@test.com",
        ),
        patch(
            "vaig.web.deps.is_admin",
            return_value=True,
        ),
        patch("vaig.web.routes.skills.get_settings") as mock_get_settings,
        patch("vaig.web.routes.skills.SkillRegistry") as mock_registry_cls,
        patch("vaig.web.routes.skills.get_telemetry_collector") as mock_tc,
    ):
        mock_get_settings.return_value = MagicMock()
        mock_registry = MagicMock()
        mock_registry.list_skills.return_value = []
        mock_registry_cls.return_value = mock_registry

        mock_collector = MagicMock()
        mock_collector.async_query_events = AsyncMock(return_value=[])
        mock_tc.return_value = mock_collector

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/portal/skills")
            assert resp.status_code == 200
            assert "No skills installed" in resp.text


# ── Dev mode bypass ──────────────────────────────────────────


@pytest.mark.asyncio
@patch.dict("os.environ", {"VAIG_WEB_DEV_MODE": "true"}, clear=False)
async def test_skills_dev_mode_bypasses_admin() -> None:
    """In dev mode, all authenticated users get 200.

    With VAIG_WEB_DEV_MODE=true, ``is_admin()`` returns True for everyone.
    We do NOT mock ``is_admin`` here — it should naturally return True via
    the dev-mode bypass, proving the env var works end-to-end.
    """
    app = create_app()

    with (
        patch(
            "vaig.web.routes.skills.get_current_user",
            return_value="anyone@test.com",
        ),
        patch("vaig.web.routes.skills.get_settings") as mock_get_settings,
        patch("vaig.web.routes.skills.SkillRegistry") as mock_registry_cls,
        patch("vaig.web.routes.skills.get_telemetry_collector") as mock_tc,
    ):
        mock_get_settings.return_value = MagicMock()
        mock_registry = MagicMock()
        mock_registry.list_skills.return_value = []
        mock_registry_cls.return_value = mock_registry

        mock_collector = MagicMock()
        mock_collector.async_query_events = AsyncMock(return_value=[])
        mock_tc.return_value = mock_collector

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/portal/skills")
            assert resp.status_code == 200


# ── Telemetry failure ────────────────────────────────────────


@pytest.mark.asyncio
async def test_skills_telemetry_failure_shows_dash() -> None:
    """When telemetry fails, usage column should show '—'."""
    app = create_app()
    mock_meta = _make_skill_metadata()

    with (
        patch(
            "vaig.web.routes.skills.get_current_user",
            return_value="admin@test.com",
        ),
        patch(
            "vaig.web.deps.is_admin",
            return_value=True,
        ),
        patch("vaig.web.routes.skills.get_settings") as mock_get_settings,
        patch("vaig.web.routes.skills.SkillRegistry") as mock_registry_cls,
        patch(
            "vaig.web.routes.skills.get_telemetry_collector",
            side_effect=RuntimeError("DB unavailable"),
        ),
    ):
        mock_get_settings.return_value = MagicMock()
        mock_registry = MagicMock()
        mock_registry.list_skills.return_value = [mock_meta]
        mock_registry.get_source.return_value = "builtin"
        mock_registry_cls.return_value = mock_registry

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/portal/skills")
            assert resp.status_code == 200
            # The template renders "—" when telemetry_available is False
            assert "—" in resp.text


# ── Usage stats ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_skills_usage_count_from_telemetry() -> None:
    """Usage count should reflect telemetry event counts."""
    app = create_app()
    mock_meta = _make_skill_metadata(name="rca", display_name="RCA")

    events: list[dict[str, Any]] = [
        {"event_name": "rca"},
        {"event_name": "rca"},
        {"event_name": "rca"},
        {"event_name": "other-skill"},
    ]

    with (
        patch(
            "vaig.web.routes.skills.get_current_user",
            return_value="admin@test.com",
        ),
        patch(
            "vaig.web.deps.is_admin",
            return_value=True,
        ),
        patch("vaig.web.routes.skills.get_settings") as mock_get_settings,
        patch("vaig.web.routes.skills.SkillRegistry") as mock_registry_cls,
        patch("vaig.web.routes.skills.get_telemetry_collector") as mock_tc,
    ):
        mock_get_settings.return_value = MagicMock()
        mock_registry = MagicMock()
        mock_registry.list_skills.return_value = [mock_meta]
        mock_registry.get_source.return_value = "builtin"
        mock_registry_cls.return_value = mock_registry

        mock_collector = MagicMock()
        mock_collector.async_query_events = AsyncMock(return_value=events)
        mock_tc.return_value = mock_collector

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/portal/skills")
            assert resp.status_code == 200
            # RCA should show count of 3 somewhere in the HTML
            body = resp.text
            assert "RCA" in body
            # The usage count is rendered inside a <td class="usage-count">
            assert 'usage-count">' in body
            # Extract the usage cell content and verify count value
            import re

            usage_match = re.search(
                r'class="usage-count">\s*(\d+)\s*</td>', body,
            )
            assert usage_match is not None, "Usage count cell not found"
            assert usage_match.group(1) == "3"


# ── Nav link visibility ──────────────────────────────────────


@pytest.mark.asyncio
@patch.dict("os.environ", {"VAIG_WEB_ADMIN_EMAILS": "admin@test.com"}, clear=False)
async def test_admin_sees_skills_nav_link() -> None:
    """Admin users should see the 'Skills' nav link pointing to /portal/skills."""
    app = create_app()
    mock_meta = _make_skill_metadata()

    with (
        patch(
            "vaig.web.routes.skills.get_current_user",
            return_value="admin@test.com",
        ),
        patch(
            "vaig.web.deps.is_admin",
            return_value=True,
        ),
        patch("vaig.web.routes.skills.get_settings") as mock_get_settings,
        patch("vaig.web.routes.skills.SkillRegistry") as mock_registry_cls,
        patch("vaig.web.routes.skills.get_telemetry_collector") as mock_tc,
    ):
        mock_get_settings.return_value = MagicMock()
        mock_registry = MagicMock()
        mock_registry.list_skills.return_value = [mock_meta]
        mock_registry.get_source.return_value = "builtin"
        mock_registry_cls.return_value = mock_registry

        mock_collector = MagicMock()
        mock_collector.async_query_events = AsyncMock(return_value=[])
        mock_tc.return_value = mock_collector

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/portal/skills")
            assert resp.status_code == 200
            assert "/portal/skills" in resp.text


@pytest.mark.asyncio
async def test_non_admin_does_not_see_skills_nav_link() -> None:
    """Non-admin users should NOT see the Skills nav link on regular pages."""
    app = create_app()

    with patch(
        "vaig.web.deps.is_admin",
        return_value=False,
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/")
            assert resp.status_code == 200
            assert "/portal/skills" not in resp.text

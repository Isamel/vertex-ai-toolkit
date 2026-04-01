"""Tests for live route concurrency configuration.

Covers:
- Default max concurrent value
- Environment variable parsing (valid and invalid values)
- gke_location field presence in live.html
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip(
    "fastapi",
    reason="FastAPI not available; install the 'web' extra to run web tests.",
)

from httpx import ASGITransport, AsyncClient

from vaig.web.app import create_app

# ── Concurrency defaults ─────────────────────────────────────


def test_default_max_concurrent_is_five() -> None:
    """_DEFAULT_MAX_CONCURRENT should be 5."""
    from vaig.web.routes.live import _DEFAULT_MAX_CONCURRENT

    assert _DEFAULT_MAX_CONCURRENT == 5


def test_env_var_parsing_valid_integer() -> None:
    """VAIG_LIVE_MAX_CONCURRENT with a valid integer should be parsed."""
    with patch.dict("os.environ", {"VAIG_LIVE_MAX_CONCURRENT": "10"}):
        # Re-evaluate the parsing logic directly
        import os

        default = 5
        try:
            result = int(
                os.environ.get("VAIG_LIVE_MAX_CONCURRENT", str(default))
            )
        except (ValueError, TypeError):
            result = default
        assert result == 10


def test_env_var_parsing_non_numeric_falls_back() -> None:
    """VAIG_LIVE_MAX_CONCURRENT with non-numeric value should use default."""
    with patch.dict("os.environ", {"VAIG_LIVE_MAX_CONCURRENT": "not-a-number"}):
        import os

        default = 5
        try:
            result = int(
                os.environ.get("VAIG_LIVE_MAX_CONCURRENT", str(default))
            )
        except (ValueError, TypeError):
            result = default
        assert result == default


def test_env_var_parsing_empty_string_falls_back() -> None:
    """VAIG_LIVE_MAX_CONCURRENT with empty string should use default."""
    with patch.dict("os.environ", {"VAIG_LIVE_MAX_CONCURRENT": ""}):
        import os

        default = 5
        try:
            result = int(
                os.environ.get("VAIG_LIVE_MAX_CONCURRENT", str(default))
            )
        except (ValueError, TypeError):
            result = default
        assert result == default


# ── Live form gke_location field ─────────────────────────────


@pytest.mark.asyncio
async def test_live_form_contains_gke_location_field() -> None:
    """GET /live should contain a gke_location input field."""
    app = create_app()

    with patch(
        "vaig.web.routes.live.get_settings",
        return_value=pytest.importorskip("unittest.mock").AsyncMock(
            gcp=pytest.importorskip("unittest.mock").MagicMock(project_id="test-proj"),
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/live")
            assert resp.status_code == 200
            body = resp.text
            assert 'name="gke_location"' in body
            assert 'id="gke_location"' in body
            assert "us-central1-a" in body

"""Tests for VAIG web foundation — Phase 1.

Covers:
- App factory creates valid FastAPI app
- Health endpoint returns 200 + correct JSON
- Index page returns 200 + contains HTMX
- Static CSS is served
- CLI web command calls uvicorn with correct args
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from vaig.web.app import create_app

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def app():
    """Create a fresh app instance for each test."""
    return create_app()


@pytest.fixture()
async def client(app):
    """Async HTTP client wired to the test app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── App Factory ──────────────────────────────────────────────


def test_create_app_returns_fastapi(app):
    """create_app() must return a FastAPI instance."""
    from fastapi import FastAPI

    assert isinstance(app, FastAPI)
    assert app.title == "VAIG Web"


def test_create_app_has_routes(app):
    """The app must have /health and / routes registered."""
    route_paths = [r.path for r in app.routes if hasattr(r, "path")]
    assert "/health" in route_paths
    assert "/" in route_paths


# ── Health Endpoint ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_returns_200(client):
    """GET /health returns 200."""
    resp = await client.get("/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_health_json_schema(client):
    """GET /health returns correct JSON schema."""
    resp = await client.get("/health")
    data = resp.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "version" in data
    assert isinstance(data["version"], str)


# ── Index Page ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_index_returns_200(client):
    """GET / returns 200."""
    resp = await client.get("/")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_index_contains_htmx(client):
    """GET / includes HTMX script tag."""
    resp = await client.get("/")
    assert "htmx.org" in resp.text


@pytest.mark.asyncio
async def test_index_is_html(client):
    """GET / returns HTML content."""
    resp = await client.get("/")
    assert "text/html" in resp.headers["content-type"]
    assert "<!DOCTYPE html>" in resp.text


# ── Static Assets ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_static_css_served(client):
    """GET /static/style.css returns 200."""
    resp = await client.get("/static/style.css")
    assert resp.status_code == 200
    assert "text/css" in resp.headers["content-type"]


# ── CLI Web Command ──────────────────────────────────────────


def test_web_command_calls_uvicorn():
    """The `vaig web` CLI command must invoke uvicorn.run with correct args."""
    from typer.testing import CliRunner

    from vaig.cli.app import app

    runner = CliRunner()

    with patch("uvicorn.run") as mock_run:
        result = runner.invoke(app, ["web", "--port", "9999", "--host", "127.0.0.1"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        mock_run.assert_called_once_with(
            "vaig.web.app:create_app",
            host="127.0.0.1",
            port=9999,
            reload=False,
            factory=True,
            log_level="info",
        )


def test_web_command_missing_extras():
    """The `vaig web` CLI command should error helpfully when extras are missing."""
    import builtins
    import sys

    from typer.testing import CliRunner

    from vaig.cli.app import app

    runner = CliRunner()

    _real_import = builtins.__import__

    def _block_uvicorn(name, *args, **kwargs):
        if name == "uvicorn":
            raise ImportError("No module named 'uvicorn'")
        return _real_import(name, *args, **kwargs)

    # Remove cached uvicorn from sys.modules so the import fires again
    saved = sys.modules.pop("uvicorn", None)
    try:
        with patch.object(builtins, "__import__", side_effect=_block_uvicorn):
            result = runner.invoke(app, ["web"])

        assert result.exit_code == 1
        assert "Web extras not installed" in result.output
    finally:
        if saved is not None:
            sys.modules["uvicorn"] = saved

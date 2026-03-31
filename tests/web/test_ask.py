"""Tests for the Ask route — Task 2.4.

Covers:
- GET /ask renders the ask form page
- POST /ask/stream returns SSE content-type
- Ask route is registered in the app
- Nav bar contains "Ask" link
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


# ── Route Registration ───────────────────────────────────────


def test_ask_routes_registered(app) -> None:
    """App must have /ask and /ask/stream routes."""
    route_paths = [r.path for r in app.routes if hasattr(r, "path")]
    assert "/ask" in route_paths
    assert "/ask/stream" in route_paths


# ── GET /ask ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ask_form_returns_200(client) -> None:
    """GET /ask returns 200."""
    resp = await client.get("/ask")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_ask_form_is_html(client) -> None:
    """GET /ask returns HTML content."""
    resp = await client.get("/ask")
    assert "text/html" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_ask_form_contains_form_elements(client) -> None:
    """GET /ask should contain form inputs for question and service."""
    resp = await client.get("/ask")
    body = resp.text
    assert 'name="question"' in body
    assert 'name="service"' in body
    assert 'name="project"' in body
    assert 'name="model"' in body


@pytest.mark.asyncio
async def test_ask_form_has_submit_button(client) -> None:
    """GET /ask should contain a submit button."""
    resp = await client.get("/ask")
    assert "btn-submit" in resp.text


# ── Navigation ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_nav_contains_ask_link(client) -> None:
    """The nav bar should include an 'Ask' link."""
    resp = await client.get("/")
    assert 'href="/ask"' in resp.text
    assert ">Ask<" in resp.text


# ── POST /ask/stream ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_ask_stream_returns_sse_content_type() -> None:
    """POST /ask/stream should return text/event-stream content-type."""
    app = create_app()

    mock_stream = _MockStream(["Hello"])
    mock_orchestrator = MagicMock()
    mock_orchestrator.async_execute_single = AsyncMock(return_value=mock_stream)

    with (
        patch("vaig.web.routes.ask.get_current_user", return_value="test@test.com"),
        patch(
            "vaig.web.deps.Settings.from_overrides",
            return_value=MagicMock(),
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
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/ask/stream",
                data={"question": "Why are pods crashing?", "service": "my-cluster"},
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_ask_stream_contains_events() -> None:
    """POST /ask/stream should contain SSE chunk and done events."""
    app = create_app()

    mock_stream = _MockStream(["Hello ", "world"], usage={"tokens": 10})
    mock_orchestrator = MagicMock()
    mock_orchestrator.async_execute_single = AsyncMock(return_value=mock_stream)

    with (
        patch("vaig.web.routes.ask.get_current_user", return_value="test@test.com"),
        patch(
            "vaig.web.deps.Settings.from_overrides",
            return_value=MagicMock(),
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
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/ask/stream",
                data={"question": "Test question"},
            )
            body = resp.text
            assert "event: chunk" in body
            assert "event: done" in body

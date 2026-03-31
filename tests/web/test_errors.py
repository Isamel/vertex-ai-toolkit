"""Tests for error handling — Task 5.1.

Covers:
- 404 page renders styled HTML (not JSON) with Go Home link
- 500 page renders friendly error (no tracebacks in production)
- SSE errors emit event: error with user-friendly messages
- SSE errors include retriable flag
- No raw tracebacks leak in error responses
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip(
    "fastapi",
    reason="FastAPI not available; install the 'web' extra to run web tests.",
)

from httpx import ASGITransport, AsyncClient

from vaig.web.app import create_app
from vaig.web.sse import _friendly_error_message, _is_retriable, stream_to_sse

# ── Helpers ──────────────────────────────────────────────────


class _ErrorStreamResult:
    """A mock StreamResult that raises during iteration."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.usage: dict[str, object] = {}
        self.text = ""

    async def __aiter__(self):
        raise self._error
        yield  # noqa: RET503 — makes this an async generator


async def _collect_events(gen):  # type: ignore[no-untyped-def]
    """Collect all SSE events from an async generator."""
    events = []
    async for event in gen:
        events.append(event)
    return events


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


# ── 404 Page Tests ───────────────────────────────────────────


class TestNotFoundPage:
    """Tests for the custom 404 error page."""

    @pytest.mark.asyncio
    async def test_404_returns_html(self, client: AsyncClient) -> None:
        """A non-existent URL should return HTML, not JSON."""
        resp = await client.get("/this-page-does-not-exist")
        assert resp.status_code == 404
        assert "text/html" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_404_contains_go_home_link(self, client: AsyncClient) -> None:
        """404 page must have a Go Home link."""
        resp = await client.get("/nonexistent-route")
        assert 'href="/"' in resp.text

    @pytest.mark.asyncio
    async def test_404_says_not_found(self, client: AsyncClient) -> None:
        """404 page should say 'Not Found'."""
        resp = await client.get("/nope")
        assert "Not Found" in resp.text
        assert "404" in resp.text

    @pytest.mark.asyncio
    async def test_404_no_traceback(self, client: AsyncClient) -> None:
        """404 page must not contain a traceback."""
        resp = await client.get("/nope")
        assert "Traceback" not in resp.text


# ── 500 Page Tests ───────────────────────────────────────────


class TestServerErrorPage:
    """Tests for the custom 500 error page."""

    @pytest.mark.asyncio
    async def test_500_returns_friendly_html(self) -> None:
        """500 errors should return a styled HTML page."""
        app = create_app()

        # Add a route that always raises
        from fastapi import APIRouter

        error_router = APIRouter()

        @error_router.get("/boom")
        async def boom() -> None:
            msg = "secret database password is hunter2"
            raise RuntimeError(msg)

        app.include_router(error_router)

        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/boom")
            assert resp.status_code == 500
            assert "text/html" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_500_no_traceback_in_production(self) -> None:
        """500 page must NOT contain tracebacks when debug=False."""
        app = create_app()
        app.debug = False

        from fastapi import APIRouter

        error_router = APIRouter()

        @error_router.get("/boom")
        async def boom() -> None:
            msg = "secret internal error"
            raise RuntimeError(msg)

        app.include_router(error_router)

        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/boom")
            body = resp.text
            assert "Traceback" not in body
            assert "secret internal error" not in body

    @pytest.mark.asyncio
    async def test_500_has_go_home_link(self) -> None:
        """500 page must have a Go Home link."""
        app = create_app()

        from fastapi import APIRouter

        error_router = APIRouter()

        @error_router.get("/boom")
        async def boom() -> None:
            raise RuntimeError("fail")

        app.include_router(error_router)

        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/boom")
            assert 'href="/"' in resp.text or "Go Home" in resp.text


# ── SSE Error Tests ──────────────────────────────────────────


class TestSSEErrors:
    """Tests for SSE error events from stream_to_sse."""

    @pytest.mark.asyncio
    async def test_vertex_error_emits_friendly_message(self) -> None:
        """Vertex API errors should yield a user-friendly error event."""
        exc = RuntimeError("google.api_core.exceptions.ResourceExhausted: Quota exceeded")
        stream = _ErrorStreamResult(exc)
        queue: asyncio.Queue[object] = asyncio.Queue()

        events = await _collect_events(stream_to_sse(stream, queue))

        error_events = [e for e in events if e.event == "error"]
        assert len(error_events) >= 1
        data = json.loads(error_events[0].data)
        # Must be user-friendly — no raw exception
        assert "quota" in data["message"].lower()
        assert "google.api_core" not in data["message"]

    @pytest.mark.asyncio
    async def test_error_event_includes_retriable_flag(self) -> None:
        """SSE error events should include a 'retriable' boolean."""
        exc = RuntimeError("ServiceUnavailable: backend not ready")
        stream = _ErrorStreamResult(exc)
        queue: asyncio.Queue[object] = asyncio.Queue()

        events = await _collect_events(stream_to_sse(stream, queue))

        error_events = [e for e in events if e.event == "error"]
        assert len(error_events) >= 1
        data = json.loads(error_events[0].data)
        assert "retriable" in data
        assert data["retriable"] is True

    @pytest.mark.asyncio
    async def test_error_followed_by_done(self) -> None:
        """After an error event, a done event must still be sent."""
        exc = RuntimeError("something broke")
        stream = _ErrorStreamResult(exc)
        queue: asyncio.Queue[object] = asyncio.Queue()

        events = await _collect_events(stream_to_sse(stream, queue))

        event_types = [e.event for e in events]
        assert "error" in event_types
        assert "done" in event_types
        # done must come after error
        error_idx = event_types.index("error")
        done_idx = event_types.index("done")
        assert done_idx > error_idx

    @pytest.mark.asyncio
    async def test_no_raw_traceback_in_sse_error(self) -> None:
        """SSE error events must not contain raw tracebacks."""
        exc = RuntimeError("File '/internal/path/module.py', line 42")
        stream = _ErrorStreamResult(exc)
        queue: asyncio.Queue[object] = asyncio.Queue()

        events = await _collect_events(stream_to_sse(stream, queue))

        error_events = [e for e in events if e.event == "error"]
        assert len(error_events) >= 1
        data = json.loads(error_events[0].data)
        assert "/internal/path/" not in data["message"]
        assert "line 42" not in data["message"]


# ── _friendly_error_message Tests ────────────────────────────


class TestFriendlyErrorMessage:
    """Tests for the _friendly_error_message helper."""

    def test_quota_error(self) -> None:
        """Quota errors should mention quota."""
        exc = RuntimeError("ResourceExhausted: quota limit")
        msg = _friendly_error_message(exc)
        assert "quota" in msg.lower()

    def test_permission_error(self) -> None:
        """Permission errors should mention credentials."""
        exc = RuntimeError("PermissionDenied: caller lacks access")
        msg = _friendly_error_message(exc)
        assert "permission" in msg.lower()

    def test_timeout_error(self) -> None:
        """Timeout errors should mention timeout."""
        exc = TimeoutError("request timed out")
        msg = _friendly_error_message(exc)
        assert "timed out" in msg.lower()

    def test_connection_error(self) -> None:
        """Connection errors should mention connection."""
        exc = ConnectionError("refused")
        msg = _friendly_error_message(exc)
        assert "connection" in msg.lower()

    def test_generic_error_no_raw_message(self) -> None:
        """Unknown errors should NOT expose the raw exception message."""
        exc = ValueError("secret internal detail xyz123")
        msg = _friendly_error_message(exc)
        assert "secret internal detail" not in msg
        assert "unexpected error" in msg.lower()

    def test_vertex_api_error(self) -> None:
        """Vertex AI errors should return generic model message."""
        exc = RuntimeError("google.api_core.exceptions.InternalServerError")
        msg = _friendly_error_message(exc)
        assert "model" in msg.lower() or "try again" in msg.lower()


# ── _is_retriable Tests ──────────────────────────────────────


class TestIsRetriable:
    """Tests for the _is_retriable helper."""

    def test_resource_exhausted_is_retriable(self) -> None:
        exc = RuntimeError("ResourceExhausted")
        assert _is_retriable(exc) is True

    def test_timeout_is_retriable(self) -> None:
        exc = TimeoutError("request timed out")
        assert _is_retriable(exc) is True

    def test_connection_error_is_retriable(self) -> None:
        exc = ConnectionError("refused")
        assert _is_retriable(exc) is True

    def test_value_error_not_retriable(self) -> None:
        exc = ValueError("bad input")
        assert _is_retriable(exc) is False

    def test_permission_denied_not_retriable(self) -> None:
        exc = RuntimeError("PermissionDenied: no access")
        assert _is_retriable(exc) is False

"""Tests for the live pipeline semaphore — TOCTOU race fix.

Verifies the single-acquire pattern introduced to fix the race condition
where the semaphore was acquired (for checking), released, then re-acquired
in the generator — allowing other requests to steal the slot in between.

Covers:
- 429 response when all pipeline slots are occupied
- Semaphore release after the generator completes normally (no leak)
- Semaphore release on ``CancelledError`` (client disconnect)
- Semaphore release even when the generator raises an unexpected error
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip(
    "fastapi",
    reason="FastAPI not available; install the 'web' extra to run web tests.",
)

from httpx import ASGITransport, AsyncClient
from sse_starlette import ServerSentEvent

from vaig.web.app import create_app

# ── Helpers ──────────────────────────────────────────────────


def _mock_settings() -> AsyncMock:
    """Return a minimal mock Settings that satisfies live route deps."""
    settings = AsyncMock()
    settings.gcp = MagicMock(project_id="test-proj")
    return settings


def _mock_request(
    service_name: str = "my-svc",
    question: str = "",
) -> AsyncMock:
    """Return a mock Request with form data."""
    mock = AsyncMock()
    mock.form = AsyncMock(
        return_value={
            "service_name": service_name,
            "question": question,
            "cluster": "",
            "namespace": "",
            "gke_project": "",
            "gke_location": "",
        }
    )
    return mock


def _pipeline_mocks():
    """Return (mock_container, mock_skill_registry) for pipeline patching."""
    mock_skill = MagicMock()
    mock_skill_registry = MagicMock()
    mock_skill_registry.get.return_value = mock_skill

    mock_container = MagicMock()
    mock_container.event_bus = MagicMock()
    return mock_container, mock_skill_registry


def _pipeline_patches(test_sem, mock_container, mock_skill_registry, sse_side_effect):
    """Return a context manager that patches all pipeline dependencies."""
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        with (
            patch("vaig.web.routes.live._pipeline_semaphore", test_sem),
            patch("vaig.web.routes.live.get_current_user", return_value="test-user"),
            patch("vaig.web.routes.live.get_settings", return_value=_mock_settings()),
            patch("vaig.web.routes.live.get_container", return_value=mock_container),
            patch(
                "vaig.web.routes.live.build_gke_config",
                return_value=MagicMock(
                    default_namespace="default",
                    location="us-c1-a",
                    cluster_name="c1",
                ),
            ),
            patch("vaig.web.routes.live.register_live_tools", return_value=MagicMock()),
            patch("vaig.skills.registry.SkillRegistry", return_value=mock_skill_registry),
            patch("vaig.web.routes.live.live_pipeline_to_sse", side_effect=sse_side_effect),
        ):
            yield

    return _ctx()


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def app():
    """Create a fresh app instance for each test."""
    return create_app()


# ── 429 when semaphore full ──────────────────────────────────


@pytest.mark.asyncio
async def test_live_stream_returns_429_when_semaphore_full(app) -> None:
    """POST /live/stream should return 429 when all pipeline slots are taken.

    Uses a Semaphore(0) so that ``locked()`` returns ``True`` immediately.
    """
    locked_sem = asyncio.Semaphore(0)

    with (
        patch("vaig.web.routes.live._pipeline_semaphore", locked_sem),
        patch("vaig.web.routes.live.get_current_user", return_value="test-user"),
        patch("vaig.web.routes.live.get_settings", return_value=_mock_settings()),
        patch("vaig.web.routes.live.get_container", return_value=MagicMock()),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/live/stream",
                data={"service_name": "my-svc", "question": "what is going on"},
            )
            assert resp.status_code == 429


@pytest.mark.asyncio
async def test_429_response_contains_error_sse_event(app) -> None:
    """The 429 SSE body should contain a TooManyRequests error event."""
    locked_sem = asyncio.Semaphore(0)

    with (
        patch("vaig.web.routes.live._pipeline_semaphore", locked_sem),
        patch("vaig.web.routes.live.get_current_user", return_value="test-user"),
        patch("vaig.web.routes.live.get_settings", return_value=_mock_settings()),
        patch("vaig.web.routes.live.get_container", return_value=MagicMock()),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/live/stream", data={"service_name": "my-svc"})
            body = resp.text
            assert "TooManyRequests" in body
            assert "pipelines are already running" in body


# ── Semaphore release after normal completion ────────────────


@pytest.mark.asyncio
async def test_semaphore_released_after_generator_completes() -> None:
    """Semaphore count should be restored after the SSE generator finishes.

    This verifies there is no slot leak — the finally block in ``_generate()``
    releases the semaphore even when the pipeline runs to completion.
    """
    test_sem = asyncio.Semaphore(1)
    mock_container, mock_skill_registry = _pipeline_mocks()

    async def _fake_sse(*args, **kwargs):  # noqa: ANN002, ANN003
        yield ServerSentEvent(data=json.dumps({}), event="done")

    with _pipeline_patches(test_sem, mock_container, mock_skill_registry, _fake_sse):
        import vaig.web.routes.live as live_mod

        sse_response = await live_mod.live_stream(_mock_request())

        # Semaphore acquired by the route
        assert test_sem._value == 0  # noqa: SLF001

        # Drain the generator — simulates normal SSE streaming to completion
        async for _ in sse_response.body_iterator:
            pass

        # The finally block should have released the semaphore
        assert test_sem._value == 1, (  # noqa: SLF001
            "Semaphore was NOT released after generator completed — slot leak!"
        )


# ── Semaphore release on CancelledError ──────────────────────


@pytest.mark.asyncio
async def test_semaphore_released_on_cancelled_error() -> None:
    """Semaphore must be released even when the streaming task is cancelled.

    In production, a client disconnect causes Starlette to cancel the
    asyncio Task that drives the SSE generator.  Task cancellation injects
    ``CancelledError`` through the entire coroutine chain, including the
    ``_generate()`` inner generator whose ``finally`` block releases the
    semaphore.  Using ``athrow`` on the outer SSE wrapper does NOT reach
    the inner generator — task cancellation is the correct simulation.
    """
    test_sem = asyncio.Semaphore(1)
    mock_container, mock_skill_registry = _pipeline_mocks()

    async def _slow_sse(*args, **kwargs):  # noqa: ANN002, ANN003
        await asyncio.sleep(100)  # Block until cancelled
        yield ServerSentEvent(data=json.dumps({}), event="done")  # pragma: no cover

    with _pipeline_patches(test_sem, mock_container, mock_skill_registry, _slow_sse):
        import vaig.web.routes.live as live_mod

        sse_response = await live_mod.live_stream(_mock_request())

        # Semaphore acquired by the route
        assert test_sem._value == 0  # noqa: SLF001

        # Simulate client disconnect: wrap generator consumption in a task
        # and cancel it — this is what Starlette does on disconnect.
        async def _consume(iterator):  # noqa: ANN001
            async for _ in iterator:
                pass  # pragma: no cover

        task = asyncio.create_task(_consume(sse_response.body_iterator))
        await asyncio.sleep(0)  # Let the task start and hit the sleep
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The finally block should have released the semaphore
        assert test_sem._value == 1, (  # noqa: SLF001
            "Semaphore NOT released after CancelledError — "
            "slot leak on client disconnect!"
        )


# ── Semaphore release on unexpected error ────────────────────


@pytest.mark.asyncio
async def test_semaphore_released_on_unexpected_error() -> None:
    """Semaphore must be released even on unexpected exceptions.

    The finally block should catch everything — not just CancelledError.
    """
    test_sem = asyncio.Semaphore(1)
    mock_container, mock_skill_registry = _pipeline_mocks()

    async def _exploding_sse(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("Boom!")  # noqa: TRY003, EM101
        yield  # type: ignore[misc]  # Make it a generator

    with _pipeline_patches(
        test_sem, mock_container, mock_skill_registry, _exploding_sse
    ):
        import vaig.web.routes.live as live_mod

        sse_response = await live_mod.live_stream(_mock_request())

        # Semaphore acquired
        assert test_sem._value == 0  # noqa: SLF001

        # Iterate the generator — should hit RuntimeError
        with pytest.raises(RuntimeError, match="Boom"):
            async for _ in sse_response.body_iterator:
                pass  # pragma: no cover

        # The finally block should still release the semaphore
        assert test_sem._value == 1, (  # noqa: SLF001
            "Semaphore NOT released after RuntimeError — "
            "slot leak on unexpected errors!"
        )

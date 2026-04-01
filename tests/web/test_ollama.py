"""Tests for the Ollama-compatible proxy endpoints.

Covers:
- Helper functions: _clean_model_name, _map_generation_kwargs
- POST /api/generate (non-streaming + streaming)
- POST /api/chat (non-streaming + streaming, role mapping)
- GET /api/tags
- Error translation (rate limit, client error)
- Feature flag gating (disabled → 404)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip(
    "fastapi",
    reason="FastAPI not available; install the 'web' extra to run web tests.",
)

from httpx import ASGITransport, AsyncClient

from vaig.core.config import ModelInfo, Settings, reset_settings
from vaig.core.exceptions import GeminiClientError, GeminiRateLimitError
from vaig.web.ollama_models import OllamaOptions
from vaig.web.routes.ollama import _clean_model_name, _map_generation_kwargs

# ── Helpers / Fakes ──────────────────────────────────────────


@dataclass
class _FakeGenerationResult:
    """Minimal stand-in for ``GenerationResult``."""

    text: str
    model: str = "gemini-2.5-pro"
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""
    thinking_text: str | None = None


class _FakeStreamResult:
    """Async-iterable fake that behaves like ``StreamResult``."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.usage: dict[str, int] = {}
        self.text = "".join(chunks)

    def __aiter__(self) -> _FakeStreamResult:
        self._index = 0
        return self

    async def __anext__(self) -> str:
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


class _FakeGeminiClient:
    """Minimal mock of ``GeminiClient`` for testing the Ollama proxy.

    Supports ``async_generate``, ``async_generate_stream``, and
    ``list_available_models``.  Tests can override specific methods
    with ``AsyncMock`` as needed.
    """

    def __init__(self, text: str = "Hello!", chunks: list[str] | None = None) -> None:
        self.async_generate = AsyncMock(
            return_value=_FakeGenerationResult(text=text),
        )
        self.async_generate_stream = AsyncMock(
            return_value=_FakeStreamResult(chunks or ["Hel", "lo!"]),
        )

    def initialize(self) -> None:
        pass

    async def async_initialize(self) -> None:
        pass

    def list_available_models(self) -> list[dict[str, str]]:
        return []


def _build_app_with_ollama(
    *,
    available_models: list[ModelInfo] | None = None,
) -> Any:
    """Build a FastAPI app with the Ollama router force-enabled.

    Instead of relying on the global ``get_settings()`` inside
    ``create_app()``, we directly include the router after creating
    a minimal app.
    """
    from fastapi import FastAPI

    from vaig.web.routes.ollama import router as ollama_router

    app = FastAPI()
    app.include_router(ollama_router)

    # Stash available_models so tests can access them
    if available_models:
        app.state.available_models = available_models

    return app


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def fake_client() -> _FakeGeminiClient:
    """Default fake GeminiClient."""
    return _FakeGeminiClient()


@pytest.fixture()
def app(fake_client: _FakeGeminiClient) -> Any:
    """FastAPI test app with Ollama router and mocked dependencies."""
    return _build_app_with_ollama(
        available_models=[
            ModelInfo(id="gemini-2.5-pro"),
            ModelInfo(id="gemini-2.5-flash"),
        ],
    )


@pytest.fixture()
def _patch_deps(fake_client: _FakeGeminiClient) -> Any:
    """Patch ``get_settings`` and ``get_container`` for route handlers."""
    settings = Settings(
        models={"default": "gemini-2.5-pro", "available": [{"id": "gemini-2.5-pro"}, {"id": "gemini-2.5-flash"}]},  # type: ignore[arg-type]
    )

    mock_container = type("C", (), {
        "gemini_client": fake_client,
        "settings": settings,
    })()

    with (
        patch("vaig.web.routes.ollama.get_settings", new_callable=AsyncMock, return_value=settings),
        patch("vaig.web.routes.ollama.get_container", return_value=mock_container),
    ):
        yield


@pytest.fixture()
async def client(app: Any, _patch_deps: Any) -> Any:
    """Async HTTP client wired to the test app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── Unit Tests: _clean_model_name ────────────────────────────


class TestCleanModelName:
    """Test model name normalization (task 3.2)."""

    def test_strip_latest_tag(self) -> None:
        assert _clean_model_name("gemini-2.5-pro:latest") == "gemini-2.5-pro"

    def test_strip_other_tag(self) -> None:
        assert _clean_model_name("gemini-2.5-pro:v2") == "gemini-2.5-pro"

    def test_preserve_bare_name(self) -> None:
        assert _clean_model_name("gemini-2.5-pro") == "gemini-2.5-pro"

    def test_empty_string(self) -> None:
        assert _clean_model_name("") == ""


# ── Unit Tests: _map_generation_kwargs ───────────────────────


class TestMapGenerationKwargs:
    """Test Ollama→Gemini options mapping (task 3.3)."""

    def test_maps_num_predict_to_max_output_tokens(self) -> None:
        opts = OllamaOptions(num_predict=1024)
        result = _map_generation_kwargs(opts)
        assert result == {"max_output_tokens": 1024}

    def test_passes_temperature_directly(self) -> None:
        opts = OllamaOptions(temperature=0.7)
        result = _map_generation_kwargs(opts)
        assert result == {"temperature": 0.7}

    def test_passes_top_p_and_top_k(self) -> None:
        opts = OllamaOptions(top_p=0.9, top_k=50)
        result = _map_generation_kwargs(opts)
        assert result == {"top_p": 0.9, "top_k": 50}

    def test_handles_none_options(self) -> None:
        result = _map_generation_kwargs(None)
        assert result == {}

    def test_skips_none_fields(self) -> None:
        opts = OllamaOptions()
        result = _map_generation_kwargs(opts)
        assert result == {}

    def test_all_options_together(self) -> None:
        opts = OllamaOptions(temperature=0.5, top_p=0.8, top_k=40, num_predict=2048)
        result = _map_generation_kwargs(opts)
        assert result == {
            "temperature": 0.5,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }


# ── Integration Tests: POST /api/generate ────────────────────


@pytest.mark.asyncio
async def test_generate_non_streaming(client: AsyncClient, fake_client: _FakeGeminiClient) -> None:
    """Non-streaming generate returns a single JSON with done=true (task 3.4)."""
    resp = await client.post(
        "/api/generate",
        json={"model": "gemini-2.5-pro", "prompt": "Hello", "stream": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["done"] is True
    assert data["model"] == "gemini-2.5-pro"
    assert data["response"] == "Hello!"
    assert data["created_at"] != ""

    # Verify GeminiClient was called correctly
    fake_client.async_generate.assert_awaited_once()
    call_kwargs = fake_client.async_generate.call_args
    assert call_kwargs[0][0] == "Hello"
    assert call_kwargs[1]["model_id"] == "gemini-2.5-pro"


@pytest.mark.asyncio
async def test_generate_strips_model_tag(client: AsyncClient, fake_client: _FakeGeminiClient) -> None:
    """Model name with :latest suffix is stripped before calling GeminiClient."""
    resp = await client.post(
        "/api/generate",
        json={"model": "gemini-2.5-pro:latest", "prompt": "Hi", "stream": False},
    )
    assert resp.status_code == 200
    call_kwargs = fake_client.async_generate.call_args
    assert call_kwargs[1]["model_id"] == "gemini-2.5-pro"


@pytest.mark.asyncio
async def test_generate_streaming(client: AsyncClient, fake_client: _FakeGeminiClient) -> None:
    """Streaming generate returns ndjson lines with done=false then done=true (task 3.5)."""
    import json

    resp = await client.post(
        "/api/generate",
        json={"model": "gemini-2.5-pro", "prompt": "Hello", "stream": True},
    )
    assert resp.status_code == 200
    assert "application/x-ndjson" in resp.headers.get("content-type", "")

    lines = [json.loads(line) for line in resp.text.strip().split("\n") if line.strip()]
    # 2 content chunks + 1 final
    assert len(lines) == 3

    # Intermediate chunks
    assert lines[0]["done"] is False
    assert lines[0]["response"] == "Hel"
    assert lines[1]["done"] is False
    assert lines[1]["response"] == "lo!"

    # Final chunk
    assert lines[2]["done"] is True
    assert lines[2]["response"] == ""
    assert lines[2]["created_at"] != ""


@pytest.mark.asyncio
async def test_generate_with_system_instruction(client: AsyncClient, fake_client: _FakeGeminiClient) -> None:
    """System instruction is forwarded to GeminiClient."""
    resp = await client.post(
        "/api/generate",
        json={
            "model": "gemini-2.5-pro",
            "prompt": "Hi",
            "system": "Be brief",
            "stream": False,
        },
    )
    assert resp.status_code == 200
    call_kwargs = fake_client.async_generate.call_args
    assert call_kwargs[1]["system_instruction"] == "Be brief"


@pytest.mark.asyncio
async def test_generate_with_options(client: AsyncClient, fake_client: _FakeGeminiClient) -> None:
    """Generation options are mapped and forwarded."""
    resp = await client.post(
        "/api/generate",
        json={
            "model": "gemini-2.5-pro",
            "prompt": "Hi",
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 512},
        },
    )
    assert resp.status_code == 200
    call_kwargs = fake_client.async_generate.call_args
    assert call_kwargs[1]["temperature"] == 0.3
    assert call_kwargs[1]["max_output_tokens"] == 512


# ── Integration Tests: POST /api/chat ────────────────────────


@pytest.mark.asyncio
async def test_chat_non_streaming(client: AsyncClient, fake_client: _FakeGeminiClient) -> None:
    """Non-streaming chat with multi-turn messages (task 3.6)."""
    resp = await client.post(
        "/api/chat",
        json={
            "model": "gemini-2.5-pro",
            "messages": [
                {"role": "system", "content": "Be brief"},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "How are you?"},
            ],
            "stream": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["done"] is True
    assert data["message"]["role"] == "assistant"
    assert data["message"]["content"] == "Hello!"
    assert data["model"] == "gemini-2.5-pro"

    # Verify role mapping: system → system_instruction, assistant → model
    call_kwargs = fake_client.async_generate.call_args
    assert call_kwargs[1]["system_instruction"] == "Be brief"
    history = call_kwargs[1]["history"]
    # History should contain: user("Hi"), model("Hello") — last user becomes prompt
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "Hi"
    assert history[1].role == "model"
    assert history[1].content == "Hello"
    # Prompt is the last user message
    assert call_kwargs[0][0] == "How are you?"


@pytest.mark.asyncio
async def test_chat_streaming(client: AsyncClient, fake_client: _FakeGeminiClient) -> None:
    """Streaming chat returns ndjson with message.content chunks (task 3.7)."""
    import json

    resp = await client.post(
        "/api/chat",
        json={
            "model": "gemini-2.5-pro",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert "application/x-ndjson" in resp.headers.get("content-type", "")

    lines = [json.loads(line) for line in resp.text.strip().split("\n") if line.strip()]
    assert len(lines) == 3

    # Intermediate chunks have message with content
    assert lines[0]["done"] is False
    assert lines[0]["message"]["content"] == "Hel"
    assert lines[0]["message"]["role"] == "assistant"

    assert lines[1]["done"] is False
    assert lines[1]["message"]["content"] == "lo!"

    # Final
    assert lines[2]["done"] is True
    assert lines[2]["message"]["content"] == ""


# ── Integration Tests: GET /api/tags ─────────────────────────


@pytest.mark.asyncio
async def test_tags_returns_models(client: AsyncClient) -> None:
    """GET /api/tags returns available models in Ollama format (task 3.8)."""
    resp = await client.get("/api/tags")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    names = [m["name"] for m in data["models"]]
    assert "gemini-2.5-pro" in names
    assert "gemini-2.5-flash" in names


@pytest.mark.asyncio
async def test_tags_empty_models() -> None:
    """GET /api/tags with no models configured returns empty list (task 3.8)."""
    empty_settings = Settings(
        models={"default": "gemini-2.5-pro", "available": []},  # type: ignore[arg-type]
    )
    mock_container = type("C", (), {"gemini_client": _FakeGeminiClient(), "settings": empty_settings})()

    test_app = _build_app_with_ollama()

    with (
        patch("vaig.web.routes.ollama.get_settings", new_callable=AsyncMock, return_value=empty_settings),
        patch("vaig.web.routes.ollama.get_container", return_value=mock_container),
    ):
        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/tags")

    assert resp.status_code == 200
    assert resp.json()["models"] == []


# ── Integration Tests: Error Translation ─────────────────────


@pytest.mark.asyncio
async def test_error_rate_limit(client: AsyncClient, fake_client: _FakeGeminiClient) -> None:
    """GeminiRateLimitError → 429 JSON error (task 3.9)."""
    fake_client.async_generate.side_effect = GeminiRateLimitError("quota exhausted")

    resp = await client.post(
        "/api/generate",
        json={"model": "gemini-2.5-pro", "prompt": "Hi", "stream": False},
    )
    assert resp.status_code == 429
    assert resp.json()["error"] == "rate limit exceeded"


@pytest.mark.asyncio
async def test_error_client_error(client: AsyncClient, fake_client: _FakeGeminiClient) -> None:
    """GeminiClientError → 500 JSON error (task 3.9)."""
    fake_client.async_generate.side_effect = GeminiClientError("something broke")

    resp = await client.post(
        "/api/generate",
        json={"model": "gemini-2.5-pro", "prompt": "Hi", "stream": False},
    )
    assert resp.status_code == 500
    assert "something broke" in resp.json()["error"]


@pytest.mark.asyncio
async def test_error_on_chat_endpoint(client: AsyncClient, fake_client: _FakeGeminiClient) -> None:
    """Error handling also works on the chat endpoint."""
    fake_client.async_generate.side_effect = GeminiRateLimitError("quota exhausted")

    resp = await client.post(
        "/api/chat",
        json={
            "model": "gemini-2.5-pro",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        },
    )
    assert resp.status_code == 429
    assert resp.json()["error"] == "rate limit exceeded"


# ── Integration Tests: Feature Flag ──────────────────────────


@pytest.mark.asyncio
async def test_feature_flag_off() -> None:
    """When ollama.enabled is False, /api/generate is not registered → 404 (task 3.10)."""
    reset_settings()
    try:
        with patch("vaig.core.config.get_settings", return_value=Settings()):
            from vaig.web.app import create_app

            test_app = create_app()

        # The ollama routes should NOT be registered
        route_paths = [r.path for r in test_app.routes if hasattr(r, "path")]
        assert "/api/generate" not in route_paths
        assert "/api/chat" not in route_paths
        assert "/api/tags" not in route_paths
    finally:
        reset_settings()

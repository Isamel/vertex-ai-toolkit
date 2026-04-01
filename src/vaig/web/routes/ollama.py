"""Ollama-compatible proxy endpoints.

Translates Ollama API requests into Gemini API calls via
``GeminiClient`` and streams responses in Ollama's ndjson format.

Endpoints:
- ``POST /api/generate`` — text completion (streaming / non-streaming)
- ``POST /api/chat``     — multi-turn chat   (streaming / non-streaming)
- ``GET  /api/tags``     — list available models
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vaig.core.client import ChatMessage
from vaig.core.exceptions import (
    ContextWindowExceededError,
    GeminiClientError,
    GeminiConnectionError,
    GeminiRateLimitError,
)
from vaig.web.deps import get_container, get_current_user, get_settings
from vaig.web.ollama_models import (
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaChatResponseMessage,
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaOptions,
    OllamaTagModel,
    OllamaTagsResponse,
)

__all__: list[str] = []

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ollama"])


# ── Helpers ──────────────────────────────────────────────────


def _clean_model_name(model: str) -> str:
    """Strip ``:tag`` suffix from an Ollama model name.

    Ollama clients always append ``:latest`` (or other tags) to model
    names.  We strip everything after the first ``:`` to get the bare
    Gemini model ID.

    Examples::

        >>> _clean_model_name("gemini-2.5-pro:latest")
        'gemini-2.5-pro'
        >>> _clean_model_name("gemini-2.5-pro")
        'gemini-2.5-pro'
    """
    return model.split(":")[0]


def _map_generation_kwargs(options: OllamaOptions | None) -> dict[str, Any]:
    """Map Ollama ``options`` to ``GeminiClient`` generation kwargs.

    Ollama uses ``num_predict`` for max output tokens; Gemini expects
    ``max_output_tokens``.  ``temperature``, ``top_p``, and ``top_k``
    pass through directly.
    """
    if options is None:
        return {}
    kwargs: dict[str, Any] = {}
    if options.temperature is not None:
        kwargs["temperature"] = options.temperature
    if options.top_p is not None:
        kwargs["top_p"] = options.top_p
    if options.top_k is not None:
        kwargs["top_k"] = options.top_k
    if options.num_predict is not None:
        kwargs["max_output_tokens"] = options.num_predict
    return kwargs


def _now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


def _handle_gemini_error(exc: Exception) -> JSONResponse:
    """Translate a Gemini exception to an Ollama-format JSON error response."""
    if isinstance(exc, GeminiRateLimitError):
        return JSONResponse({"error": "rate limit exceeded"}, status_code=429)
    if isinstance(exc, GeminiConnectionError):
        return JSONResponse({"error": "model backend unavailable"}, status_code=503)
    if isinstance(exc, ContextWindowExceededError):
        return JSONResponse(
            {"error": f"context window exceeded: {exc}"},
            status_code=400,
        )
    if isinstance(exc, GeminiClientError):
        return JSONResponse({"error": str(exc)}, status_code=500)
    # Fallback — should not normally be reached
    return JSONResponse({"error": str(exc)}, status_code=500)


# ── Streaming Generators ─────────────────────────────────────


async def _stream_ndjson_generate(
    stream_result: Any,
    model: str,
) -> AsyncGenerator[bytes, None]:
    """Yield ndjson lines from a ``StreamResult`` for ``/api/generate``."""
    async for chunk in stream_result:
        line = OllamaGenerateResponse(
            model=model, response=chunk, done=False, created_at=_now_iso(),
        )
        yield line.model_dump_json().encode() + b"\n"
    # Final line with done=true
    final = OllamaGenerateResponse(
        model=model,
        response="",
        done=True,
        created_at=_now_iso(),
    )
    yield final.model_dump_json().encode() + b"\n"


async def _stream_ndjson_chat(
    stream_result: Any,
    model: str,
) -> AsyncGenerator[bytes, None]:
    """Yield ndjson lines from a ``StreamResult`` for ``/api/chat``."""
    async for chunk in stream_result:
        line = OllamaChatResponse(
            model=model,
            message=OllamaChatResponseMessage(content=chunk),
            done=False,
            created_at=_now_iso(),
        )
        yield line.model_dump_json().encode() + b"\n"
    # Final line with done=true
    final = OllamaChatResponse(
        model=model,
        message=OllamaChatResponseMessage(content=""),
        done=True,
        created_at=_now_iso(),
    )
    yield final.model_dump_json().encode() + b"\n"


# ── Endpoints ────────────────────────────────────────────────


@router.post("/api/generate")
async def ollama_generate(body: OllamaGenerateRequest, request: Request) -> Any:
    """Ollama-compatible text completion endpoint."""
    settings = await get_settings(request)
    if not settings.ollama.enabled:
        return JSONResponse({"error": "Ollama API is disabled"}, status_code=404)

    get_current_user(request)  # Auth check — raises 401 if unauthenticated
    container = get_container(settings)
    client = container.gemini_client

    model = _clean_model_name(body.model)
    gen_kwargs = _map_generation_kwargs(body.options)

    try:
        if body.stream:
            stream_result = await client.async_generate_stream(
                body.prompt,
                system_instruction=body.system,
                model_id=model,
                **gen_kwargs,
            )
            return StreamingResponse(
                _stream_ndjson_generate(stream_result, model),
                media_type="application/x-ndjson",
            )

        # Non-streaming
        result = await client.async_generate(
            body.prompt,
            system_instruction=body.system,
            model_id=model,
            **gen_kwargs,
        )
        return OllamaGenerateResponse(
            model=model,
            response=result.text,
            done=True,
            created_at=_now_iso(),
        )
    except (
        GeminiRateLimitError,
        GeminiConnectionError,
        ContextWindowExceededError,
        GeminiClientError,
    ) as exc:
        return _handle_gemini_error(exc)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        return _handle_gemini_error(exc)


@router.post("/api/chat")
async def ollama_chat(body: OllamaChatRequest, request: Request) -> Any:
    """Ollama-compatible multi-turn chat endpoint."""
    settings = await get_settings(request)
    if not settings.ollama.enabled:
        return JSONResponse({"error": "Ollama API is disabled"}, status_code=404)

    get_current_user(request)  # Auth check — raises 401 if unauthenticated
    container = get_container(settings)
    client = container.gemini_client

    model = _clean_model_name(body.model)
    gen_kwargs = _map_generation_kwargs(body.options)

    # ── Extract system messages and build history ────────────
    system_parts: list[str] = []
    history: list[ChatMessage] = []

    for msg in body.messages:
        if msg.role == "system":
            system_parts.append(msg.content)
        elif msg.role == "assistant":
            # Ollama "assistant" → Gemini "model"
            history.append(ChatMessage(role="model", content=msg.content))
        else:
            history.append(ChatMessage(role=msg.role, content=msg.content))

    system_instruction = "\n".join(system_parts) if system_parts else None

    # The last user message is the prompt; everything before is history.
    if history and history[-1].role == "user":
        prompt = history.pop().content
    else:
        return JSONResponse(
            status_code=400,
            content={"error": "The last message must be from the 'user' role."},
        )

    try:
        if body.stream:
            stream_result = await client.async_generate_stream(
                prompt,
                system_instruction=system_instruction,
                history=history if history else None,
                model_id=model,
                **gen_kwargs,
            )
            return StreamingResponse(
                _stream_ndjson_chat(stream_result, model),
                media_type="application/x-ndjson",
            )

        # Non-streaming
        result = await client.async_generate(
            prompt,
            system_instruction=system_instruction,
            history=history if history else None,
            model_id=model,
            **gen_kwargs,
        )
        return OllamaChatResponse(
            model=model,
            message=OllamaChatResponseMessage(content=result.text),
            done=True,
            created_at=_now_iso(),
        )
    except (
        GeminiRateLimitError,
        GeminiConnectionError,
        ContextWindowExceededError,
        GeminiClientError,
    ) as exc:
        return _handle_gemini_error(exc)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        return _handle_gemini_error(exc)


@router.get("/api/tags")
async def ollama_tags(request: Request) -> Any:
    """List available models in Ollama tag format."""
    settings = await get_settings(request)
    if not settings.ollama.enabled:
        return JSONResponse({"error": "Ollama API is disabled"}, status_code=404)

    get_current_user(request)  # Auth check — raises 401 if unauthenticated
    models = [
        OllamaTagModel(name=m.id, model=m.id)
        for m in settings.models.available
    ]
    return OllamaTagsResponse(models=models)

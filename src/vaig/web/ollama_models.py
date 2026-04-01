"""Pydantic request/response schemas for the Ollama-compatible API.

These models mirror the Ollama wire protocol so that clients speaking the
Ollama API (VS Code Continue, Cody, CLI tools) can send requests to VAIG's
Vertex AI backend without modification.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# ── Request Models ───────────────────────────────────────────


class OllamaOptions(BaseModel):
    """Generation options forwarded by Ollama clients."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    num_predict: int | None = None  # Maps to max_output_tokens


class OllamaGenerateRequest(BaseModel):
    """``POST /api/generate`` request body."""

    model: str
    prompt: str
    system: str | None = None
    stream: bool = True
    options: OllamaOptions | None = None


class OllamaChatMessage(BaseModel):
    """A single message in an Ollama chat conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


class OllamaChatRequest(BaseModel):
    """``POST /api/chat`` request body."""

    model: str
    messages: list[OllamaChatMessage]
    stream: bool = True
    options: OllamaOptions | None = None


# ── Response Models ──────────────────────────────────────────


class OllamaGenerateResponse(BaseModel):
    """``POST /api/generate`` response body (single JSON or ndjson line)."""

    model: str
    response: str = ""
    done: bool = False
    created_at: str = ""  # ISO-8601 timestamp


class OllamaChatResponseMessage(BaseModel):
    """Message payload inside an ``/api/chat`` response."""

    role: Literal["assistant"] = "assistant"
    content: str = ""


class OllamaChatResponse(BaseModel):
    """``POST /api/chat`` response body (single JSON or ndjson line)."""

    model: str
    message: OllamaChatResponseMessage
    done: bool = False
    created_at: str = ""  # ISO-8601 timestamp


class OllamaTagModel(BaseModel):
    """A single model entry in the ``/api/tags`` response."""

    name: str
    model: str = ""
    modified_at: str = ""
    size: int = 0


class OllamaTagsResponse(BaseModel):
    """``GET /api/tags`` response body."""

    models: list[OllamaTagModel]

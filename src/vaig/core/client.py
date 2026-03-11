"""Gemini client — Vertex AI wrapper with streaming, multimodal, and model switching."""

from __future__ import annotations

import logging
import random
import ssl
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

import vertexai
from google.api_core import exceptions as google_exceptions
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    GenerativeModel,
    Part,
)

from vaig.core.auth import get_credentials
from vaig.core.exceptions import (
    GeminiClientError,
    GeminiConnectionError,
    GeminiRateLimitError,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from vaig.core.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Exceptions that are safe to retry — all transient / server-side.
_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    google_exceptions.ResourceExhausted,  # 429
    google_exceptions.ServiceUnavailable,  # 503
    google_exceptions.InternalServerError,  # 500
    google_exceptions.DeadlineExceeded,  # 504
    google_exceptions.Aborted,  # transient
)


def _is_ssl_or_connection_error(exc: BaseException) -> bool:
    """Check if an exception is an SSL or connection error (possibly nested).

    SSL errors from VPN/proxy SSL inspection often appear as:
    - ``ssl.SSLError`` / ``ssl.SSLEOFError`` directly
    - ``OSError`` wrapping an SSL error
    - ``google.api_core.exceptions.ServiceUnavailable`` wrapping an SSL error
    - ``requests.exceptions.SSLError`` wrapping an SSL error
    - ``ConnectionError`` / ``ConnectionResetError``

    We check the exception chain (``__cause__`` / ``__context__``) up to 5 levels deep.
    """
    checked: set[int] = set()

    def _walk(err: BaseException | None, depth: int = 0) -> bool:
        if err is None or depth > 5 or id(err) in checked:
            return False
        checked.add(id(err))

        if isinstance(err, (ssl.SSLError, ConnectionResetError, ConnectionAbortedError)):
            return True

        # Check string representation for SSL-related messages
        err_str = str(err).lower()
        if any(kw in err_str for kw in ("ssl", "eof occurred", "certificate", "handshake")):
            return True

        return _walk(err.__cause__, depth + 1) or _walk(err.__context__, depth + 1)

    return _walk(exc)


@dataclass
class ChatMessage:
    """A single message in a conversation."""

    role: str  # "user" | "model"
    content: str
    parts: list[Any] = field(default_factory=list)  # For multimodal parts


@dataclass
class GenerationResult:
    """Result from a generation call."""

    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""


class GeminiClient:
    """Vertex AI Gemini client with multi-model support and streaming."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._initialized = False
        self._models: dict[str, GenerativeModel] = {}
        self._current_model_id: str = settings.models.default
        self._active_location: str = settings.gcp.location
        self._using_fallback: bool = False

    def initialize(self) -> None:
        """Initialize the Vertex AI SDK with credentials."""
        if self._initialized:
            return

        credentials = get_credentials(self._settings)

        vertexai.init(
            project=self._settings.gcp.project_id,
            location=self._active_location,
            credentials=credentials,
        )

        self._initialized = True
        logger.info(
            "Vertex AI initialized — project=%s, location=%s",
            self._settings.gcp.project_id,
            self._active_location,
        )

    def _reinitialize_with_fallback(self) -> None:
        """Re-initialize Vertex AI with the fallback location.

        Clears the model cache (since models are bound to the previous
        location) and marks the client as using the fallback.
        """
        fallback = self._settings.gcp.fallback_location
        if self._using_fallback or not fallback or fallback == self._active_location:
            return  # already on fallback or no fallback configured

        logger.warning(
            "Primary location '%s' failed — falling back to '%s'",
            self._active_location,
            fallback,
        )

        self._active_location = fallback
        self._using_fallback = True
        self._initialized = False
        self._models.clear()  # models are bound to the old location
        self.initialize()

    def _ensure_initialized(self) -> None:
        """Ensure the SDK is initialized before making calls."""
        if not self._initialized:
            self.initialize()

    def _get_model(self, model_id: str | None = None) -> GenerativeModel:
        """Get or create a GenerativeModel instance."""
        mid = model_id or self._current_model_id

        if mid not in self._models:
            self._models[mid] = GenerativeModel(mid)
            logger.info("Created model instance: %s", mid)

        return self._models[mid]

    @property
    def current_model(self) -> str:
        """Get the current model ID."""
        return self._current_model_id

    def switch_model(self, model_id: str) -> str:
        """Switch the active model. Returns the new model ID."""
        old = self._current_model_id
        self._current_model_id = model_id
        logger.info("Model switched: %s → %s", old, model_id)
        return model_id

    def _build_generation_config(self, **overrides: Any) -> GenerationConfig:
        """Build generation config from settings + overrides."""
        cfg = self._settings.generation
        return GenerationConfig(
            temperature=overrides.get("temperature", cfg.temperature),
            max_output_tokens=overrides.get("max_output_tokens", cfg.max_output_tokens),
            top_p=overrides.get("top_p", cfg.top_p),
            top_k=overrides.get("top_k", cfg.top_k),
        )

    # ── Retry logic ───────────────────────────────────────────

    def _retry_with_backoff(self, fn: Callable[[], T], *, timeout: float | None = None) -> T:
        """Execute *fn* with exponential backoff on retryable Google API errors.

        If an SSL or connection error is detected and a fallback location is
        configured, the client re-initializes with the fallback location and
        retries the call once before giving up.

        Args:
            fn: Zero-argument callable that performs the API call.
            timeout: Optional wall-clock timeout in seconds for the entire
                     retry loop (not per-attempt).

        Returns:
            The return value of *fn* on success.

        Raises:
            GeminiRateLimitError: If all retries are exhausted on a 429.
            GeminiConnectionError: If all retries are exhausted on 500/502/503/504.
            GeminiClientError: If an unexpected non-retryable error occurs.
        """
        retry_cfg = self._settings.retry
        delay = retry_cfg.initial_delay
        last_exception: Exception | None = None
        start_time = time.monotonic() if timeout is not None else None

        for attempt in range(retry_cfg.max_retries + 1):
            # Check wall-clock timeout before each attempt (except the first).
            if start_time is not None and attempt > 0:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:  # type: ignore[operator]
                    break  # fall through to raise

            try:
                return fn()
            except _RETRYABLE_EXCEPTIONS as exc:
                last_exception = exc
                if attempt < retry_cfg.max_retries:
                    jitter = random.uniform(0, 0.5)  # noqa: S311
                    sleep_time = min(delay, retry_cfg.max_delay) + jitter
                    logger.warning(
                        "Retryable error on attempt %d/%d (%s: %s) — retrying in %.2fs",
                        attempt + 1,
                        retry_cfg.max_retries + 1,
                        type(exc).__name__,
                        exc,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                    delay *= retry_cfg.backoff_multiplier
                # else: last attempt failed, fall through to raise below.
            except Exception as exc:
                # Check for SSL/connection errors that warrant a location fallback.
                if _is_ssl_or_connection_error(exc) and not self._using_fallback:
                    logger.warning(
                        "SSL/connection error detected (%s: %s) — attempting location fallback",
                        type(exc).__name__,
                        exc,
                    )
                    self._reinitialize_with_fallback()
                    # Retry the call once with the new location.
                    try:
                        return fn()
                    except Exception as fallback_exc:
                        last_exception = fallback_exc
                        break
                raise  # Non-SSL, non-retryable — propagate immediately.

        # All retries exhausted — raise the appropriate custom exception.
        assert last_exception is not None  # noqa: S101
        retries = retry_cfg.max_retries
        msg = f"All {retries} retries exhausted. Last error: {last_exception}"

        if isinstance(last_exception, google_exceptions.ResourceExhausted):
            raise GeminiRateLimitError(
                msg,
                original_error=last_exception,
                retries_attempted=retries,
            ) from last_exception

        raise GeminiConnectionError(
            msg,
            original_error=last_exception,
            retries_attempted=retries,
        ) from last_exception

    # ── Public API ────────────────────────────────────────────

    def generate(
        self,
        prompt: str | list[Part],
        *,
        system_instruction: str | None = None,
        history: list[ChatMessage] | None = None,
        model_id: str | None = None,
        timeout: float | None = None,
        **gen_kwargs: Any,
    ) -> GenerationResult:
        """Generate a response (non-streaming).

        Args:
            prompt: Text prompt or list of multimodal Parts.
            system_instruction: System-level instruction for the model.
            history: Conversation history for multi-turn chat.
            model_id: Override the current model for this call.
            timeout: Optional wall-clock timeout (seconds) for the retry loop.
            **gen_kwargs: Override generation config params.
        """
        self._ensure_initialized()

        mid = model_id or self._current_model_id
        gen_config = self._build_generation_config(**gen_kwargs)

        def _call() -> GenerationResult:
            # Model is created inside the closure so that after a location
            # fallback (which clears the model cache) a fresh model bound
            # to the new location is used on the retry attempt.
            model = self._get_or_create_model(mid, system_instruction)
            if history:
                chat_history = self._build_history(history)
                chat = model.start_chat(history=chat_history)
                response = chat.send_message(prompt, generation_config=gen_config)
            else:
                response = model.generate_content(prompt, generation_config=gen_config)

            usage = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                }

            return GenerationResult(
                text=response.text,
                model=mid,
                usage=usage,
                finish_reason=(
                    str(response.candidates[0].finish_reason) if response.candidates else ""
                ),
            )

        return self._retry_with_backoff(_call, timeout=timeout)

    def generate_stream(
        self,
        prompt: str | list[Part],
        *,
        system_instruction: str | None = None,
        history: list[ChatMessage] | None = None,
        model_id: str | None = None,
        timeout: float | None = None,
        **gen_kwargs: Any,
    ) -> Iterator[str]:
        """Generate a streaming response, yielding text chunks.

        Args:
            prompt: Text prompt or list of multimodal Parts.
            system_instruction: System-level instruction for the model.
            history: Conversation history for multi-turn chat.
            model_id: Override the current model for this call.
            timeout: Optional wall-clock timeout (seconds) for the retry loop.
            **gen_kwargs: Override generation config params.
        """
        self._ensure_initialized()

        mid = model_id or self._current_model_id
        gen_config = self._build_generation_config(**gen_kwargs)

        def _call() -> list[Any]:
            # Model is created inside the closure so that after a location
            # fallback (which clears the model cache) a fresh model bound
            # to the new location is used on the retry attempt.
            model = self._get_or_create_model(mid, system_instruction)
            if history:
                chat_history = self._build_history(history)
                chat = model.start_chat(history=chat_history)
                response_stream = chat.send_message(
                    prompt, generation_config=gen_config, stream=True,
                )
            else:
                response_stream = model.generate_content(
                    prompt, generation_config=gen_config, stream=True,
                )
            # Materialise the stream inside the retryable boundary so that
            # transient errors during iteration are also retried.
            return list(response_stream)

        chunks = self._retry_with_backoff(_call, timeout=timeout)
        for chunk in chunks:
            if chunk.text:
                yield chunk.text

    # ── Internal helpers ──────────────────────────────────────

    def _get_or_create_model(
        self,
        model_id: str,
        system_instruction: str | None = None,
    ) -> GenerativeModel:
        """Get or create a model, optionally with system instruction."""
        # Cache key includes system instruction to avoid stale instances
        cache_key = f"{model_id}::{hash(system_instruction)}"

        if cache_key not in self._models:
            kwargs: dict[str, Any] = {}
            if system_instruction:
                kwargs["system_instruction"] = system_instruction

            self._models[cache_key] = GenerativeModel(model_id, **kwargs)
            logger.info("Created model: %s (system_instruction=%s)", model_id, bool(system_instruction))

        return self._models[cache_key]

    @staticmethod
    def _build_history(messages: list[ChatMessage]) -> list[Content]:
        """Convert ChatMessage list to Vertex AI Content list."""
        history: list[Content] = []
        for msg in messages:
            if msg.parts:
                parts = msg.parts
            else:
                parts = [Part.from_text(msg.content)]
            history.append(Content(role=msg.role, parts=parts))
        return history

    def count_tokens(self, prompt: str | list[Part], *, model_id: str | None = None) -> int:
        """Count tokens in a prompt."""
        self._ensure_initialized()
        model = self._get_model(model_id)
        response = model.count_tokens(prompt)
        return response.total_tokens

    def list_available_models(self) -> list[dict[str, str]]:
        """List configured available models."""
        return [
            {"id": m.id, "description": m.description}
            for m in self._settings.models.available
        ]

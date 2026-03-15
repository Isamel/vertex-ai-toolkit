"""Gemini client — google-genai SDK wrapper with streaming, multimodal, and model switching.

Provides both **async** and **sync** APIs.  Async methods use the ``async_``
prefix (e.g. ``async_generate``).  The original sync methods (``generate``,
``generate_stream``, etc.) delegate to the async versions via ``run_sync()``
so that existing callers work without changes.
"""

from __future__ import annotations

import asyncio
import logging
import random
import ssl
import threading
import time
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from google import genai
from google.api_core import exceptions as google_exceptions
from google.genai import types

from vaig.core.async_utils import run_sync
from vaig.core.auth import get_credentials
from vaig.core.cache import CacheStats, ResponseCache, _make_cache_key
from vaig.core.exceptions import (
    GeminiClientError,
    GeminiConnectionError,
    GeminiRateLimitError,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterator

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


def _safe_int(value: Any) -> int:
    """Coerce *value* to ``int``, returning ``0`` for non-integer types.

    The google-genai SDK may expose usage fields that are ``None``, missing,
    or (under test mocks) ``MagicMock`` objects.  This helper ensures we
    always produce a real ``int``.
    """
    if isinstance(value, int):
        return value
    return 0


def _is_ssl_or_connection_error(exc: BaseException) -> bool:
    """Check if an exception is an SSL/connection/proxy error (possibly nested).

    SSL errors from VPN/proxy SSL inspection often appear as:
    - ``ssl.SSLError`` / ``ssl.SSLEOFError`` directly
    - ``OSError`` wrapping an SSL error
    - ``google.api_core.exceptions.ServiceUnavailable`` wrapping an SSL error
    - ``requests.exceptions.SSLError`` wrapping an SSL error
    - ``ConnectionError`` / ``ConnectionResetError``
    - ``AttributeError`` from ``format_http_response_error`` when a VPN/proxy
      returns a malformed response body (JSON array instead of object), causing
      ``payload.get(...)`` to crash inside ``google-api-core``.

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

        # Detect google-api-core crash when VPN/proxy returns a malformed
        # error response (e.g. a JSON list instead of a dict).  The SDK's
        # ``format_http_response_error`` calls ``payload.get(...)`` which
        # fails with ``AttributeError: 'list' object has no attribute 'get'``.
        if isinstance(err, AttributeError) and "has no attribute 'get'" in err_str:
            return True

        return _walk(err.__cause__, depth + 1) or _walk(err.__context__, depth + 1)

    return _walk(exc)


@dataclass
class ChatMessage:
    """A single message in a conversation."""

    role: str  # "user" | "model"
    content: str
    parts: list[Any] = field(default_factory=list)  # For multimodal parts


class StreamResult:
    """Wraps a materialized streaming response to capture usage metadata.

    Supports **both** sync and async iteration:

    - Sync:  ``for chunk in stream_result``
    - Async: ``async for chunk in stream_result``

    After iteration, ``.usage`` contains the token-usage dict extracted from
    the **last** chunk (where the Gemini SDK reports totals), and ``.text``
    holds the full accumulated response.
    """

    __slots__ = ("_chunks", "_usage", "_text_parts", "_model", "_exhausted", "_iter_index")

    def __init__(self, raw_chunks: Sequence[Any], model: str) -> None:
        self._chunks = raw_chunks
        self._model = model
        self._usage: dict[str, int] = {}
        self._text_parts: list[str] = []
        self._exhausted = False
        self._iter_index = 0

    def _process_chunk(self, chunk: Any) -> str | None:
        """Extract text and usage from a single chunk. Returns text or None."""
        if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
            um = chunk.usage_metadata
            self._usage = {
                "prompt_tokens": um.prompt_token_count or 0,
                "completion_tokens": um.candidates_token_count or 0,
                "total_tokens": um.total_token_count or 0,
                "thinking_tokens": _safe_int(getattr(um, "thoughts_token_count", None)),
            }
        if chunk.text:
            self._text_parts.append(chunk.text)
            return chunk.text
        return None

    def __iter__(self) -> Iterator[str]:
        """Yield text strings from each chunk (sync iteration)."""
        for chunk in self._chunks:
            text = self._process_chunk(chunk)
            if text is not None:
                yield text
        self._exhausted = True

    def __aiter__(self) -> AsyncIterator[str]:
        """Return self for ``async for`` iteration."""
        self._iter_index = 0
        return self

    async def __anext__(self) -> str:
        """Yield next text string (async iteration)."""
        while self._iter_index < len(self._chunks):
            chunk = self._chunks[self._iter_index]
            self._iter_index += 1
            text = self._process_chunk(chunk)
            if text is not None:
                return text
        self._exhausted = True
        raise StopAsyncIteration

    @property
    def usage(self) -> dict[str, int]:
        """Token usage metadata — available after iteration completes."""
        return self._usage

    @property
    def text(self) -> str:
        """Full accumulated text — available after iteration completes."""
        return "".join(self._text_parts)

    @property
    def model(self) -> str:
        """Model ID used for this generation."""
        return self._model


@dataclass
class GenerationResult:
    """Result from a generation call."""

    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""


@dataclass
class ToolCallResult:
    """Result from a generation call that may include function calls."""

    text: str  # Empty if model returned function calls instead of text
    model: str
    function_calls: list[dict[str, Any]] = field(default_factory=list)
    # Each dict: {"name": str, "args": dict[str, Any]}
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""


class GeminiClient:
    """Vertex AI Gemini client with multi-model support and streaming.

    Uses the ``google-genai`` SDK (``google.genai.Client``) with ``vertexai=True``.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._initialized = False
        self._client: genai.Client | None = None
        self._current_model_id: str = settings.models.default
        self._active_location: str = settings.gcp.location
        self._using_fallback: bool = False
        self._fallback_lock = threading.Lock()

        # Response cache — disabled by default (opt-in via config).
        try:
            cache_cfg = settings.cache
            if cache_cfg.enabled is True:
                self._cache: ResponseCache | None = ResponseCache(
                    max_size=int(cache_cfg.max_size),
                    ttl_seconds=int(cache_cfg.ttl_seconds),
                )
                logger.info(
                    "Response cache enabled — max_size=%d, ttl=%ds",
                    cache_cfg.max_size,
                    cache_cfg.ttl_seconds,
                )
            else:
                self._cache = None
        except (TypeError, ValueError, AttributeError):
            # Defensive: if settings is a mock or cache config is invalid,
            # silently disable caching rather than crashing.
            self._cache = None

    def initialize(self) -> None:
        """Initialize the google-genai Client with Vertex AI credentials (sync)."""
        if self._initialized:
            return

        credentials = get_credentials(self._settings)

        self._client = genai.Client(
            vertexai=True,
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

    async def async_initialize(self) -> None:
        """Initialize the google-genai Client with Vertex AI credentials (async).

        ``genai.Client()`` construction is CPU-bound (no I/O), so we call it
        directly.  ``get_credentials()`` may do I/O (gcloud subprocess), so
        we wrap it in ``asyncio.to_thread()``.
        """
        if self._initialized:
            return

        credentials = await asyncio.to_thread(get_credentials, self._settings)

        self._client = genai.Client(
            vertexai=True,
            project=self._settings.gcp.project_id,
            location=self._active_location,
            credentials=credentials,
        )

        self._initialized = True
        logger.info(
            "Vertex AI initialized (async) — project=%s, location=%s",
            self._settings.gcp.project_id,
            self._active_location,
        )

    def _reinitialize_with_fallback(self) -> None:
        """Re-initialize with the fallback location (sync).

        Creates a new Client bound to the fallback location and marks
        the client as using the fallback.  Thread-safe: uses double-checked
        locking so concurrent callers don't race on the fallback switch.
        """
        fallback = self._settings.gcp.fallback_location
        if self._using_fallback or not fallback or fallback == self._active_location:
            return  # already on fallback or no fallback configured

        with self._fallback_lock:
            # Re-check after acquiring the lock (another thread may have switched)
            if self._using_fallback:
                return

            logger.warning(
                "Primary location '%s' failed — falling back to '%s'",
                self._active_location,
                fallback,
            )

            self._active_location = fallback
            self._using_fallback = True
            self._initialized = False
            self._client = None
            self.initialize()

    async def _async_reinitialize_with_fallback(self) -> None:
        """Re-initialize with the fallback location (async)."""
        fallback = self._settings.gcp.fallback_location
        if self._using_fallback or not fallback or fallback == self._active_location:
            return

        # No lock needed in async — single-threaded event loop.
        logger.warning(
            "Primary location '%s' failed — falling back to '%s'",
            self._active_location,
            fallback,
        )

        self._active_location = fallback
        self._using_fallback = True
        self._initialized = False
        self._client = None
        await self.async_initialize()

    def _ensure_initialized(self) -> None:
        """Ensure the SDK is initialized before making calls (sync)."""
        if not self._initialized:
            self.initialize()

    async def _async_ensure_initialized(self) -> None:
        """Ensure the SDK is initialized before making calls (async)."""
        if not self._initialized:
            await self.async_initialize()

    def _get_client(self) -> genai.Client:
        """Get the initialized genai Client, initializing if needed."""
        self._ensure_initialized()
        assert self._client is not None  # noqa: S101
        return self._client

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

    def reinitialize(
        self,
        project: str | None = None,
        location: str | None = None,
    ) -> None:
        """Reinitialize the genai Client with updated project/location.

        Creates a new ``genai.Client`` bound to the updated project and/or
        location.  Thread-safe: uses the existing ``_fallback_lock``.

        Args:
            project: New GCP project ID. If ``None``, keeps current.
            location: New GCP location. If ``None``, keeps current.

        Raises:
            GeminiClientError: If reinitialization fails.
        """
        with self._fallback_lock:
            old_project = self._settings.gcp.project_id
            old_location = self._active_location

            if project is not None:
                self._settings.gcp.project_id = project
            if location is not None:
                self._active_location = location
                self._settings.gcp.location = location

            try:
                credentials = get_credentials(self._settings)
                new_client = genai.Client(
                    vertexai=True,
                    project=self._settings.gcp.project_id,
                    location=self._active_location,
                    credentials=credentials,
                )
                self._client = new_client
                self._initialized = True
                # Reset fallback state since this is an explicit switch
                self._using_fallback = False
                logger.info(
                    "Client reinitialized — project=%s, location=%s",
                    self._settings.gcp.project_id,
                    self._active_location,
                )
            except Exception as exc:
                # Rollback internal state on failure
                self._settings.gcp.project_id = old_project
                self._active_location = old_location
                self._settings.gcp.location = old_location
                raise GeminiClientError(
                    f"Failed to reinitialize client: {exc}",
                    original_error=exc,
                ) from exc

    def _build_generation_config(
        self,
        *,
        system_instruction: str | None = None,
        tools: list[types.Tool] | None = None,
        **overrides: Any,
    ) -> types.GenerateContentConfig:
        """Build generation config from settings + overrides.

        Supports all standard Gemini generation parameters including
        ``frequency_penalty`` and ``presence_penalty`` for controlling
        repetition in model output.
        """
        cfg = self._settings.generation
        kwargs: dict[str, Any] = {
            "temperature": overrides.get("temperature", cfg.temperature),
            "max_output_tokens": overrides.get("max_output_tokens", cfg.max_output_tokens),
            "top_p": overrides.get("top_p", cfg.top_p),
            "top_k": overrides.get("top_k", cfg.top_k),
        }
        # Optional penalty params — only include when explicitly set to avoid
        # sending defaults that could change model behaviour unexpectedly.
        if "frequency_penalty" in overrides:
            kwargs["frequency_penalty"] = overrides["frequency_penalty"]
        if "presence_penalty" in overrides:
            kwargs["presence_penalty"] = overrides["presence_penalty"]
        if system_instruction is not None:
            kwargs["system_instruction"] = system_instruction
        if tools is not None:
            kwargs["tools"] = tools

        # ── Safety settings ───────────────────────────────────
        safety_cfg = self._settings.safety
        if safety_cfg.enabled and safety_cfg.settings:
            kwargs["safety_settings"] = [
                types.SafetySetting(
                    category=s.category,
                    threshold=s.threshold,
                )
                for s in safety_cfg.settings
            ]

        return types.GenerateContentConfig(**kwargs)

    # ── Retry logic ───────────────────────────────────────────

    def _retry_with_backoff(self, fn: Callable[[], T], *, timeout: float | None = None) -> T:
        """Execute *fn* with exponential backoff on retryable Google API errors (sync).

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
                if _is_ssl_or_connection_error(exc) and not self._using_fallback:
                    logger.warning(
                        "Retryable error wraps SSL/proxy error (%s: %s) "
                        "— attempting location fallback instead of retry",
                        type(exc).__name__,
                        exc,
                    )
                    self._reinitialize_with_fallback()
                    try:
                        return fn()
                    except Exception as fallback_exc:
                        last_exception = fallback_exc
                        break
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
            except Exception as exc:
                if _is_ssl_or_connection_error(exc) and not self._using_fallback:
                    logger.warning(
                        "SSL/connection error detected (%s: %s) — attempting location fallback",
                        type(exc).__name__,
                        exc,
                    )
                    self._reinitialize_with_fallback()
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

    async def _async_retry_with_backoff(
        self,
        fn: Callable[[], Awaitable[T]],
        *,
        timeout: float | None = None,
    ) -> T:
        """Execute async *fn* with exponential backoff on retryable errors.

        Async counterpart of ``_retry_with_backoff``.  Uses
        ``asyncio.sleep`` instead of ``time.sleep`` so the event loop is
        never blocked.

        Args:
            fn: Zero-argument async callable that performs the API call.
            timeout: Optional wall-clock timeout in seconds for the entire
                     retry loop (not per-attempt).
        """
        retry_cfg = self._settings.retry
        delay = retry_cfg.initial_delay
        last_exception: Exception | None = None
        start_time = time.monotonic() if timeout is not None else None

        for attempt in range(retry_cfg.max_retries + 1):
            if start_time is not None and attempt > 0:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:  # type: ignore[operator]
                    break

            try:
                return await fn()
            except _RETRYABLE_EXCEPTIONS as exc:
                last_exception = exc
                if _is_ssl_or_connection_error(exc) and not self._using_fallback:
                    logger.warning(
                        "Retryable error wraps SSL/proxy error (%s: %s) "
                        "— attempting location fallback instead of retry",
                        type(exc).__name__,
                        exc,
                    )
                    await self._async_reinitialize_with_fallback()
                    try:
                        return await fn()
                    except Exception as fallback_exc:
                        last_exception = fallback_exc
                        break
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
                    await asyncio.sleep(sleep_time)
                    delay *= retry_cfg.backoff_multiplier
            except Exception as exc:
                if _is_ssl_or_connection_error(exc) and not self._using_fallback:
                    logger.warning(
                        "SSL/connection error detected (%s: %s) — attempting location fallback",
                        type(exc).__name__,
                        exc,
                    )
                    await self._async_reinitialize_with_fallback()
                    try:
                        return await fn()
                    except Exception as fallback_exc:
                        last_exception = fallback_exc
                        break
                raise

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

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        """Extract usage metadata from a google-genai response.

        Fields in ``usage_metadata`` can be ``None`` (e.g. when the model
        returns function calls only), so we default each to 0.
        """
        usage: dict[str, int] = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = {
                "prompt_tokens": um.prompt_token_count or 0,
                "completion_tokens": um.candidates_token_count or 0,
                "total_tokens": um.total_token_count or 0,
                "thinking_tokens": _safe_int(getattr(um, "thoughts_token_count", None)),
            }
        return usage

    def generate(
        self,
        prompt: str | list[types.Part],
        *,
        system_instruction: str | None = None,
        history: list[ChatMessage] | None = None,
        model_id: str | None = None,
        timeout: float | None = None,
        **gen_kwargs: Any,
    ) -> GenerationResult:
        """Generate a response (non-streaming).

        When the response cache is enabled and the call is cacheable
        (string prompt, no conversation history), cached results are
        returned without making an API call.

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
        logger.debug("generate() → model=%s, has_history=%s", mid, bool(history))

        # ── Cache lookup (only for stateless, text-only prompts) ─────
        cache_key: str | None = None
        if (
            self._cache is not None
            and isinstance(prompt, str)
            and not history
        ):
            cache_key = _make_cache_key(prompt, mid, system_instruction)
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.info("Cache hit — skipping API call (model=%s)", mid)
                return cached

        def _call() -> GenerationResult:
            client = self._get_client()
            config = self._build_generation_config(
                system_instruction=system_instruction, **gen_kwargs,
            )

            if history:
                chat_history = self._build_history(history)
                chat = client.chats.create(
                    model=mid, history=chat_history, config=config,  # type: ignore[arg-type]
                )
                response = chat.send_message(prompt)
            else:
                response = client.models.generate_content(
                    model=mid, contents=prompt, config=config,  # type: ignore[arg-type]
                )

            usage = GeminiClient._extract_usage(response)

            return GenerationResult(
                text=response.text or "",
                model=mid,
                usage=usage,
                finish_reason=(
                    str(response.candidates[0].finish_reason) if response.candidates else ""
                ),
            )

        result = self._retry_with_backoff(_call, timeout=timeout)

        # ── Store in cache ───────────────────────────────────────────
        if cache_key is not None and self._cache is not None:
            self._cache.put(cache_key, result)

        return result

    def generate_stream(
        self,
        prompt: str | list[types.Part],
        *,
        system_instruction: str | None = None,
        history: list[ChatMessage] | None = None,
        model_id: str | None = None,
        timeout: float | None = None,
        **gen_kwargs: Any,
    ) -> StreamResult:
        """Generate a streaming response, returning a ``StreamResult``.

        The returned ``StreamResult`` is **iterable** — ``for chunk in result``
        yields text strings exactly like the old generator did, preserving
        backward compatibility.  After iteration completes, ``result.usage``
        contains the token-usage dict captured from the last SDK chunk and
        ``result.text`` holds the full accumulated response.

        The stream is materialised into a list inside ``_call()`` so that the
        **entire iteration** falls within the retryable boundary.  Without
        this, a transient network error mid-stream would surface as an
        unrecoverable ``google.api_core`` exception.  The tradeoff is that peak
        memory equals the full response size rather than a single chunk, but
        for typical LLM responses (< 100 KB) this is negligible.

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
        logger.debug("generate_stream() → model=%s, has_history=%s", mid, bool(history))

        def _call() -> list[Any]:
            client = self._get_client()
            config = self._build_generation_config(
                system_instruction=system_instruction, **gen_kwargs,
            )

            if history:
                chat_history = self._build_history(history)
                chat = client.chats.create(
                    model=mid, history=chat_history, config=config,  # type: ignore[arg-type]
                )
                response_stream = chat.send_message_stream(prompt)
            else:
                response_stream = client.models.generate_content_stream(
                    model=mid, contents=prompt, config=config,  # type: ignore[arg-type]
                )
            # Materialise the stream inside the retryable boundary so that
            # transient errors during iteration are also retried.
            return list(response_stream)

        raw_chunks = self._retry_with_backoff(_call, timeout=timeout)
        return StreamResult(raw_chunks, model=mid)

    def generate_with_tools(
        self,
        prompt: str | list[types.Part],
        *,
        tool_declarations: list[Any],
        system_instruction: str | None = None,
        history: list[ChatMessage] | None = None,
        model_id: str | None = None,
        timeout: float | None = None,
        **gen_kwargs: Any,
    ) -> ToolCallResult:
        """Generate a response that may include function calls.

        Makes a SINGLE API call with tool declarations. The response may
        contain text, function calls, or both. Does NOT loop — the caller
        (e.g. CodingAgent) is responsible for the tool-use loop.

        Args:
            prompt: Text prompt or list of multimodal Parts.
            tool_declarations: List of FunctionDeclaration objects.
            system_instruction: System-level instruction for the model.
            history: Conversation history for multi-turn chat.
            model_id: Override the current model for this call.
            timeout: Optional wall-clock timeout (seconds) for the retry loop.
            **gen_kwargs: Override generation config params.
        """
        self._ensure_initialized()

        mid = model_id or self._current_model_id
        logger.debug(
            "generate_with_tools() → model=%s, has_history=%s, num_tools=%d",
            mid,
            bool(history),
            len(tool_declarations),
        )

        def _call() -> ToolCallResult:
            client = self._get_client()
            tools = [types.Tool(function_declarations=tool_declarations)]
            config = self._build_generation_config(
                system_instruction=system_instruction,
                tools=tools,
                **gen_kwargs,
            )

            try:
                if history:
                    chat_history = self._build_history(history)

                    # On iteration 2+, the CodingAgent sends an empty prompt
                    # because the full context lives in the history (including
                    # function call responses).  The SDK rejects empty prompts
                    # with "value must not be empty", so we pop the last
                    # history entry and send its parts as the new message.
                    actual_prompt: Any = prompt
                    if not prompt and chat_history:
                        last_entry = chat_history.pop()
                        actual_prompt = last_entry.parts

                    chat = client.chats.create(
                        model=mid, history=chat_history, config=config,
                    )
                    response = chat.send_message(actual_prompt)
                else:
                    response = client.models.generate_content(
                        model=mid, contents=prompt, config=config,
                    )
            except Exception as exc:
                # In the old SDK we caught ResponseValidationError for
                # function-call-only responses.  The new google-genai SDK
                # may raise errors differently — handle gracefully.
                exc_name = type(exc).__name__
                if "validation" in exc_name.lower() or "blocked" in str(exc).lower():
                    logger.warning(
                        "Response validation/blocked error caught — %s: %s",
                        exc_name,
                        exc,
                    )
                    return ToolCallResult(
                        text=f"Model response was blocked: {exc}",
                        model=mid,
                        function_calls=[],
                        usage={},
                        finish_reason="BLOCKED",
                    )
                raise

            # Extract usage metadata
            usage = GeminiClient._extract_usage(response)

            # Handle empty response
            candidate = response.candidates[0] if response.candidates else None
            content = candidate.content if candidate and candidate.content else None
            if not content or not content.parts:
                logger.debug("generate_with_tools() — empty response (no candidates/parts)")
                return ToolCallResult(
                    text="",
                    model=mid,
                    function_calls=[],
                    usage=usage,
                    finish_reason=(
                        str(candidate.finish_reason) if candidate else ""
                    ),
                )

            # Parse parts — may contain function calls, text, or both
            function_calls: list[dict[str, Any]] = []
            text_parts: list[str] = []

            for part in content.parts:
                fc = part.function_call
                if fc and fc.name:  # It's a function call
                    function_calls.append({
                        "name": fc.name,
                        "args": dict(fc.args) if fc.args else {},
                    })
                elif part.text:
                    text_parts.append(part.text)

            logger.debug(
                "generate_with_tools() — %d function call(s), %d text part(s)",
                len(function_calls),
                len(text_parts),
            )

            return ToolCallResult(
                text="".join(text_parts),
                model=mid,
                function_calls=function_calls,
                usage=usage,
                finish_reason=str(response.candidates[0].finish_reason),
            )

        return self._retry_with_backoff(_call, timeout=timeout)

    @staticmethod
    def build_function_response_parts(results: list[dict[str, Any]]) -> list[types.Part]:
        """Build Part objects containing function responses.

        The CodingAgent uses this to send function execution results back to
        the model in the conversation history for the next turn.

        Args:
            results: List of dicts with ``"name"`` (str) and ``"response"`` (dict).
                     Each response dict should have ``{"output": str, "error": bool}``.

        Returns:
            List of Part objects to include in history for the next call.
        """
        parts: list[types.Part] = []
        for result in results:
            parts.append(
                types.Part.from_function_response(
                    name=result["name"],
                    response=result["response"],
                )
            )
        return parts

    # ── Internal helpers ──────────────────────────────────────

    @staticmethod
    def _build_history(messages: list[ChatMessage]) -> list[types.Content]:
        """Convert ChatMessage list to google-genai Content list."""
        history: list[types.Content] = []
        for msg in messages:
            if msg.parts:
                parts = msg.parts
            else:
                parts = [types.Part.from_text(text=msg.content)]
            history.append(types.Content(role=msg.role, parts=parts))
        return history

    def count_tokens(self, prompt: str | list[types.Part], *, model_id: str | None = None) -> int:
        """Count tokens in a prompt (sync)."""
        client = self._get_client()
        mid = model_id or self._current_model_id
        response = client.models.count_tokens(model=mid, contents=prompt)  # type: ignore[arg-type]
        total: int = response.total_tokens if response.total_tokens is not None else 0
        return total

    def list_available_models(self) -> list[dict[str, str]]:
        """List configured available models."""
        return [
            {"id": m.id, "description": m.description}
            for m in self._settings.models.available
        ]

    # ── Async Public API ─────────────────────────────────────

    async def async_generate(
        self,
        prompt: str | list[types.Part],
        *,
        system_instruction: str | None = None,
        history: list[ChatMessage] | None = None,
        model_id: str | None = None,
        timeout: float | None = None,
        **gen_kwargs: Any,
    ) -> GenerationResult:
        """Generate a response (non-streaming, async).

        Uses the native ``client.aio.models`` async API.  Same semantics as
        the sync ``generate()`` method, including cache support.
        """
        await self._async_ensure_initialized()

        mid = model_id or self._current_model_id
        logger.debug("async_generate() → model=%s, has_history=%s", mid, bool(history))

        # ── Cache lookup ────────────────────────────────────────
        cache_key: str | None = None
        if (
            self._cache is not None
            and isinstance(prompt, str)
            and not history
        ):
            cache_key = _make_cache_key(prompt, mid, system_instruction)
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.info("Cache hit — skipping API call (model=%s)", mid)
                return cached

        async def _call() -> GenerationResult:
            client = self._get_client()
            config = self._build_generation_config(
                system_instruction=system_instruction, **gen_kwargs,
            )

            if history:
                chat_history = self._build_history(history)
                chat = client.aio.chats.create(
                    model=mid, history=chat_history, config=config,  # type: ignore[arg-type]
                )
                response = await chat.send_message(prompt)
            else:
                response = await client.aio.models.generate_content(
                    model=mid, contents=prompt, config=config,  # type: ignore[arg-type]
                )

            usage = GeminiClient._extract_usage(response)

            return GenerationResult(
                text=response.text or "",
                model=mid,
                usage=usage,
                finish_reason=(
                    str(response.candidates[0].finish_reason) if response.candidates else ""
                ),
            )

        result = await self._async_retry_with_backoff(_call, timeout=timeout)

        if cache_key is not None and self._cache is not None:
            self._cache.put(cache_key, result)

        return result

    async def async_generate_stream(
        self,
        prompt: str | list[types.Part],
        *,
        system_instruction: str | None = None,
        history: list[ChatMessage] | None = None,
        model_id: str | None = None,
        timeout: float | None = None,
        **gen_kwargs: Any,
    ) -> StreamResult:
        """Generate a streaming response (async), returning a ``StreamResult``.

        Uses the native ``client.aio.models.generate_content_stream`` API.
        The async stream is materialised into a list inside the retry boundary,
        matching the sync ``generate_stream()`` behaviour.
        """
        await self._async_ensure_initialized()

        mid = model_id or self._current_model_id
        logger.debug("async_generate_stream() → model=%s, has_history=%s", mid, bool(history))

        async def _call() -> list[Any]:
            client = self._get_client()
            config = self._build_generation_config(
                system_instruction=system_instruction, **gen_kwargs,
            )

            if history:
                chat_history = self._build_history(history)
                chat = client.aio.chats.create(
                    model=mid, history=chat_history, config=config,  # type: ignore[arg-type]
                )
                response_stream = await chat.send_message_stream(prompt)
            else:
                response_stream = await client.aio.models.generate_content_stream(
                    model=mid, contents=prompt, config=config,  # type: ignore[arg-type]
                )
            # Materialise the async stream inside the retryable boundary.
            return [chunk async for chunk in response_stream]

        raw_chunks = await self._async_retry_with_backoff(_call, timeout=timeout)
        return StreamResult(raw_chunks, model=mid)

    async def async_generate_with_tools(
        self,
        prompt: str | list[types.Part],
        *,
        tool_declarations: list[Any],
        system_instruction: str | None = None,
        history: list[ChatMessage] | None = None,
        model_id: str | None = None,
        timeout: float | None = None,
        **gen_kwargs: Any,
    ) -> ToolCallResult:
        """Generate a response that may include function calls (async).

        Uses the native ``client.aio.models`` async API.  Same semantics as
        the sync ``generate_with_tools()`` method.
        """
        await self._async_ensure_initialized()

        mid = model_id or self._current_model_id
        logger.debug(
            "async_generate_with_tools() → model=%s, has_history=%s, num_tools=%d",
            mid,
            bool(history),
            len(tool_declarations),
        )

        async def _call() -> ToolCallResult:
            client = self._get_client()
            tools = [types.Tool(function_declarations=tool_declarations)]
            config = self._build_generation_config(
                system_instruction=system_instruction,
                tools=tools,
                **gen_kwargs,
            )

            try:
                if history:
                    chat_history = self._build_history(history)

                    actual_prompt: Any = prompt
                    if not prompt and chat_history:
                        last_entry = chat_history.pop()
                        actual_prompt = last_entry.parts

                    chat = client.aio.chats.create(
                        model=mid, history=chat_history, config=config,
                    )
                    response = await chat.send_message(actual_prompt)
                else:
                    response = await client.aio.models.generate_content(
                        model=mid, contents=prompt, config=config,
                    )
            except Exception as exc:
                exc_name = type(exc).__name__
                if "validation" in exc_name.lower() or "blocked" in str(exc).lower():
                    logger.warning(
                        "Response validation/blocked error caught — %s: %s",
                        exc_name,
                        exc,
                    )
                    return ToolCallResult(
                        text=f"Model response was blocked: {exc}",
                        model=mid,
                        function_calls=[],
                        usage={},
                        finish_reason="BLOCKED",
                    )
                raise

            usage = GeminiClient._extract_usage(response)

            candidate = response.candidates[0] if response.candidates else None
            content = candidate.content if candidate and candidate.content else None
            if not content or not content.parts:
                logger.debug("async_generate_with_tools() — empty response")
                return ToolCallResult(
                    text="",
                    model=mid,
                    function_calls=[],
                    usage=usage,
                    finish_reason=(
                        str(candidate.finish_reason) if candidate else ""
                    ),
                )

            function_calls: list[dict[str, Any]] = []
            text_parts: list[str] = []

            for part in content.parts:
                fc = part.function_call
                if fc and fc.name:
                    function_calls.append({
                        "name": fc.name,
                        "args": dict(fc.args) if fc.args else {},
                    })
                elif part.text:
                    text_parts.append(part.text)

            logger.debug(
                "async_generate_with_tools() — %d function call(s), %d text part(s)",
                len(function_calls),
                len(text_parts),
            )

            return ToolCallResult(
                text="".join(text_parts),
                model=mid,
                function_calls=function_calls,
                usage=usage,
                finish_reason=str(response.candidates[0].finish_reason),
            )

        return await self._async_retry_with_backoff(_call, timeout=timeout)

    async def async_count_tokens(
        self,
        prompt: str | list[types.Part],
        *,
        model_id: str | None = None,
    ) -> int:
        """Count tokens in a prompt (async)."""
        await self._async_ensure_initialized()
        client = self._get_client()
        mid = model_id or self._current_model_id
        response = await client.aio.models.count_tokens(
            model=mid, contents=prompt,  # type: ignore[arg-type]
        )
        total: int = response.total_tokens if response.total_tokens is not None else 0
        return total

    # ── Cache management ─────────────────────────────────────

    @property
    def cache_enabled(self) -> bool:
        """Whether the response cache is active."""
        return self._cache is not None

    def clear_cache(self) -> int:
        """Clear the response cache. Returns the number of entries removed."""
        if self._cache is None:
            return 0
        return self._cache.clear()

    def cache_stats(self) -> CacheStats | None:
        """Return cache statistics, or ``None`` if caching is disabled."""
        if self._cache is None:
            return None
        return self._cache.stats()

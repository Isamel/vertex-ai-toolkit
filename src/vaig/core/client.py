"""Gemini client — google-genai SDK wrapper with streaming, multimodal, and model switching.

Provides both **async** and **sync** APIs.  Async methods use the ``async_``
prefix (e.g. ``async_generate``).  The original sync methods (``generate``,
``generate_stream``, etc.) delegate to the async versions via ``run_sync()``
so that existing callers work without changes.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import random
import ssl
import threading
import time
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from google import genai
from google.api_core import exceptions as google_exceptions
from google.auth import exceptions as auth_exceptions
from google.genai import errors as genai_errors
from google.genai import types

from vaig.core.auth import get_credentials
from vaig.core.cache import CacheStats, ResponseCache, _make_cache_key
from vaig.core.exceptions import (
    CONTEXT_WINDOW_ERROR_KEYWORDS,
    ContextWindowExceededError,
    GCPAuthError,
    GCPPermissionError,
    GeminiClientError,
    GeminiConnectionError,
    GeminiRateLimitError,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterator

    from google.auth.credentials import Credentials

    from vaig.core.config import RetryConfig, Settings
    from vaig.core.quota import QuotaChecker

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ── Parallel fan-out context (SPEC-RATE-01 invariant RL-3) ────────────────────
#
# Orchestrator sub-agent runners set this ContextVar to True for the duration
# of a fan-out so that :meth:`GeminiClient._retry_with_backoff` can apply the
# tighter ``rate_limit_max_total_wait_s_parallel`` cap.  Outside of a fan-out
# the default (False) keeps the longer serial cap in effect.
IN_PARALLEL_FANOUT: ContextVar[bool] = ContextVar(
    "IN_PARALLEL_FANOUT",
    default=False,
)


def _rate_limit_budget_s(retry_cfg: RetryConfig) -> float:
    """Return the wall-clock cap (seconds) for cumulative 429 waits.

    Uses the tighter parallel cap when called from inside a parallel
    fan-out (see :data:`IN_PARALLEL_FANOUT`), otherwise the longer
    serial cap.  Configured in :class:`~vaig.core.config.RetryConfig`.
    """
    if IN_PARALLEL_FANOUT.get():
        return retry_cfg.rate_limit_max_total_wait_s_parallel
    return retry_cfg.rate_limit_max_total_wait_s


# Exceptions that are safe to retry — all transient / server-side.
_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    google_exceptions.ResourceExhausted,  # 429
    google_exceptions.ServiceUnavailable,  # 503
    google_exceptions.InternalServerError,  # 500
    google_exceptions.DeadlineExceeded,  # 504
    google_exceptions.Aborted,  # transient
    google_exceptions.Cancelled,  # 400 CANCELLED — Vertex AI server-side cancellation under load
)

# HTTP status codes that the SDK already retries via HttpRetryOptions.
# Used in the genai_errors.APIError handler to distinguish retryable (SDK
# already handled) from non-retryable (propagate immediately).
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# Re-export for backward compatibility (tests import from here).
_CONTEXT_WINDOW_ERROR_KEYWORDS = CONTEXT_WINDOW_ERROR_KEYWORDS


def _is_context_window_error(exc: genai_errors.APIError) -> bool:
    """Return ``True`` if *exc* is a 400-family error caused by context overflow.

    Checks the error code and message keywords so we can convert the generic
    ``genai_errors.ClientError`` into :class:`ContextWindowExceededError`.
    """
    if exc.code not in (400, 413):
        return False
    msg_lower = str(exc).lower()
    return any(kw in msg_lower for kw in CONTEXT_WINDOW_ERROR_KEYWORDS)


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
            return str(chunk.text)
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
    thinking_text: str | None = None
    """Thinking content from the model, if thinking mode was enabled."""


@dataclass
class ToolCallResult:
    """Result from a generation call that may include function calls."""

    text: str  # Empty if model returned function calls instead of text
    model: str
    function_calls: list[dict[str, Any]] = field(default_factory=list)
    # Each dict: {"name": str, "args": dict[str, Any]}
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""
    thinking_text: str | None = None
    """Thinking content from the model, if thinking mode was enabled."""
    raw_parts: list[types.Part] | None = None
    """Raw ``types.Part`` objects from the API response (non-thought parts only).

    Preserved for Gemini 2.5+ ``thought_signature`` replay — the raw Part
    objects carry ``thought_signature`` bytes that are lost if we reconstruct
    Parts from the extracted function_calls dicts.  ``None`` when not captured
    (e.g. backward-compat paths).
    """


@dataclass(frozen=True)
class EndpointFlip:
    """Record of a runtime global → regional endpoint switch (SPEC-GEP-03).

    Created whenever ``GeminiClient`` detects persistent 429 errors on the
    global endpoint and re-initializes against the regional fallback.
    """

    flipped_at: datetime.datetime
    """UTC timestamp of the flip."""
    from_location: str
    """Endpoint location before the flip (typically ``"global"``)."""
    to_location: str
    """Endpoint location after the flip (e.g. ``"us-central1"``)."""
    reason: str
    """Human-readable reason (e.g. ``"persistent_429"``)."""


class GeminiClient:
    """Vertex AI Gemini client with multi-model support and streaming.

    Uses the ``google-genai`` SDK (``google.genai.Client``) with ``vertexai=True``.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        quota_checker: QuotaChecker | None = None,
    ) -> None:
        self._settings = settings
        self._initialized = False
        self._client: genai.Client | None = None
        self._current_model_id: str = settings.models.default
        self._active_location: str = settings.gcp.location
        self._using_fallback: bool = False
        self._fallback_lock = threading.Lock()
        # SPEC-GEP-03: track consecutive 429s on the global endpoint to trigger
        # a regional fallback after sustained quota pressure.
        self._consecutive_429_count: int = 0
        self._endpoint_flips: list[EndpointFlip] = []

        # Rate-limit quota enforcement — None when disabled.
        self._quota_checker = quota_checker
        # Store credentials for identity resolution in _check_quota.
        self._credentials: Credentials | None = None

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

    def _build_http_options(self) -> types.HttpOptions:
        """Build HTTP options with retry configuration for the genai SDK.

        Maps vaig's ``RetryConfig`` to the SDK's ``HttpRetryOptions`` so that
        the underlying HTTP transport retries transient errors (429, 5xx)
        automatically — before vaig's own application-level retry even kicks in.
        """
        retry_cfg = self._settings.retry
        return types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                attempts=retry_cfg.max_retries + 1,  # SDK counts initial call as attempt
                initial_delay=retry_cfg.initial_delay,
                max_delay=retry_cfg.max_delay,
                exp_base=retry_cfg.backoff_multiplier,
                jitter=0.5,
                http_status_codes=retry_cfg.retryable_status_codes,
            ),
        )

    def _resolve_initial_location(self, credentials: Credentials | None) -> str:
        """Pick the endpoint location at startup (SPEC-GEP-02).

        Delegates to :func:`vaig.core.endpoint_probe.resolve_endpoint_location`
        which honours ``gcp.endpoint_mode`` and a persistent probe cache.
        On any unexpected failure this method falls back to the value
        currently in ``self._active_location`` so initialisation never
        blocks on the probe.
        """
        from vaig.core.endpoint_probe import resolve_endpoint_location

        try:
            return resolve_endpoint_location(self._settings.gcp, credentials)
        except RuntimeError:
            # endpoint_mode="global" with unreachable endpoint — propagate.
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Endpoint probe failed (%s: %s) — using %s",
                type(exc).__name__,
                exc,
                self._active_location,
            )
            return self._active_location

    def initialize(self) -> None:
        """Initialize the google-genai Client with Vertex AI credentials (sync)."""
        if self._initialized:
            return

        try:
            credentials = get_credentials(self._settings)
            self._credentials = credentials
            # SPEC-GEP-02: resolve global vs regional before building the Client.
            # Runs a short probe when endpoint_mode="auto" and location="global".
            # Skip the probe when we are re-initialising to honour an explicit
            # fallback — ``_reinitialize_with_fallback`` has already set
            # ``_active_location`` to the fallback value.
            if not self._using_fallback:
                self._active_location = self._resolve_initial_location(credentials)
            self._client = genai.Client(
                vertexai=True,
                project=self._settings.gcp.project_id,
                location=self._active_location,
                credentials=credentials,
                http_options=self._build_http_options(),
            )
        except (GCPAuthError, GCPPermissionError):
            raise  # Already our custom type — propagate as-is
        except auth_exceptions.DefaultCredentialsError as exc:
            raise GCPAuthError(
                "GCP credentials not configured. Cannot connect to Vertex AI.",
                fix_suggestion="Run: gcloud auth application-default login",
            ) from exc
        except auth_exceptions.RefreshError as exc:
            raise GCPAuthError(
                "GCP credentials expired or invalid. Please re-authenticate.",
                fix_suggestion="Run: gcloud auth application-default login",
            ) from exc
        except Exception as exc:
            exc_str = str(exc).lower()
            if "permission" in exc_str or "403" in exc_str:
                raise GCPPermissionError(
                    f"Insufficient permissions for Vertex AI: {exc}",
                    required_permissions=["roles/aiplatform.user"],
                    fix_suggestion="Grant the Vertex AI User role to your account",
                ) from exc
            raise GeminiClientError(
                f"Failed to initialize Gemini client: {exc}",
                original_error=exc,
            ) from exc

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

        try:
            credentials = await asyncio.to_thread(get_credentials, self._settings)
            self._credentials = credentials

            # SPEC-GEP-02: resolve global vs regional before building the Client.
            # The probe performs a short blocking SDK call, so wrap in to_thread
            # to keep the event loop unblocked.
            # Skip the probe when we are re-initialising to honour an explicit
            # fallback (see sync counterpart for rationale).
            if not self._using_fallback:
                self._active_location = await asyncio.to_thread(
                    self._resolve_initial_location,
                    credentials,
                )

            self._client = genai.Client(
                vertexai=True,
                project=self._settings.gcp.project_id,
                location=self._active_location,
                credentials=credentials,
                http_options=self._build_http_options(),
            )
        except (GCPAuthError, GCPPermissionError):
            raise  # Already our custom type — propagate as-is
        except auth_exceptions.DefaultCredentialsError as exc:
            raise GCPAuthError(
                "GCP credentials not configured. Cannot connect to Vertex AI.",
                fix_suggestion="Run: gcloud auth application-default login",
            ) from exc
        except auth_exceptions.RefreshError as exc:
            raise GCPAuthError(
                "GCP credentials expired or invalid. Please re-authenticate.",
                fix_suggestion="Run: gcloud auth application-default login",
            ) from exc
        except Exception as exc:
            exc_str = str(exc).lower()
            if "permission" in exc_str or "403" in exc_str:
                raise GCPPermissionError(
                    f"Insufficient permissions for Vertex AI: {exc}",
                    required_permissions=["roles/aiplatform.user"],
                    fix_suggestion="Grant the Vertex AI User role to your account",
                ) from exc
            raise GeminiClientError(
                f"Failed to initialize Gemini client: {exc}",
                original_error=exc,
            ) from exc

        self._initialized = True
        logger.info(
            "Vertex AI initialized (async) — project=%s, location=%s",
            self._settings.gcp.project_id,
            self._active_location,
        )

    def _flip_to_fallback_on_429(self, *, reason: str = "persistent_429") -> bool:
        """Switch from global to regional when persistent 429s are detected (SPEC-GEP-03).

        Records an :class:`EndpointFlip`, resets ``_consecutive_429_count``,
        and delegates to :meth:`_reinitialize_with_fallback` for the actual
        client swap.  The flip record and log are written *inside*
        ``_fallback_lock`` to avoid a race where two threads both observe the
        pre-flip state and append duplicate records.

        Returns:
            ``True`` if a flip occurred, or the client was already on the
            fallback (no action needed).  ``False`` when no fallback is
            configured or the fallback location equals the active location.
        """
        if self._using_fallback:
            return True  # already on regional

        fallback = self._settings.gcp.fallback_location
        if not fallback or fallback == self._active_location:
            return False

        with self._fallback_lock:
            # Re-check after acquiring lock — another thread may have flipped already.
            if self._using_fallback:
                return True

            from_loc = self._active_location
            flip = EndpointFlip(
                flipped_at=datetime.datetime.now(datetime.UTC),
                from_location=from_loc,
                to_location=fallback,
                reason=reason,
            )
            self._endpoint_flips.append(flip)
            self._consecutive_429_count = 0
            logger.warning(
                "Endpoint flipped: %s → %s (reason: %s)",
                from_loc,
                fallback,
                reason,
            )
            # _reinitialize_with_fallback also acquires _fallback_lock but its
            # double-checked guard will short-circuit immediately since we just
            # set _using_fallback = True below before releasing.
            self._active_location = fallback
            self._using_fallback = True
            self._initialized = False
            self._client = None
            self.initialize()

        return True

    async def _async_flip_to_fallback_on_429(self, *, reason: str = "persistent_429") -> bool:
        """Async variant of :meth:`_flip_to_fallback_on_429` (SPEC-GEP-03).

        Calls :meth:`_async_reinitialize_with_fallback` so the event loop is
        never blocked by a sync SDK initialization.

        Returns:
            ``True`` if a flip occurred, or the client was already on the
            fallback (no action needed).  ``False`` when no fallback is
            configured or the fallback location equals the active location.
        """
        if self._using_fallback:
            return True  # already on regional

        fallback = self._settings.gcp.fallback_location
        if not fallback or fallback == self._active_location:
            return False

        # Single-threaded async event loop — no lock needed, but guard anyway.
        if self._using_fallback:
            return True

        from_loc = self._active_location
        flip = EndpointFlip(
            flipped_at=datetime.datetime.now(datetime.UTC),
            from_location=from_loc,
            to_location=fallback,
            reason=reason,
        )
        self._endpoint_flips.append(flip)
        self._consecutive_429_count = 0
        logger.warning(
            "Endpoint flipped: %s → %s (reason: %s)",
            from_loc,
            fallback,
            reason,
        )
        await self._async_reinitialize_with_fallback()
        return True

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

    @property
    def endpoint_flips(self) -> list[EndpointFlip]:
        """Return a copy of all runtime endpoint flips recorded (SPEC-GEP-03).

        Each :class:`EndpointFlip` captures a global → regional switch driven
        by persistent 429 errors.  Non-empty only when the client started on
        the global endpoint and had to fall back during the run.
        """
        return list(self._endpoint_flips)

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
                    http_options=self._build_http_options(),
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
        repetition in model output, and ``thinking_config`` for enabling
        thinking mode on supported models.
        """
        from vaig.core.config import supports_thinking

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
        # Structured output params — only include when explicitly set so that
        # callers not using structured output are completely unaffected.
        if "response_schema" in overrides:
            kwargs["response_schema"] = overrides["response_schema"]
        if "response_mime_type" in overrides:
            kwargs["response_mime_type"] = overrides["response_mime_type"]
        if system_instruction is not None:
            kwargs["system_instruction"] = system_instruction
        if tools is not None:
            kwargs["tools"] = tools

        # ── Safety settings ───────────────────────────────────
        safety_cfg = self._settings.safety
        if safety_cfg.enabled and safety_cfg.settings:
            kwargs["safety_settings"] = [
                types.SafetySetting(
                    category=s.category,  # type: ignore[arg-type]
                    threshold=s.threshold,  # type: ignore[arg-type]
                )
                for s in safety_cfg.settings
            ]

        # ── Thinking config ──────────────────────────────────
        thinking_cfg = cfg.thinking
        if thinking_cfg.enabled:
            model_id = self._current_model_id
            if not supports_thinking(model_id):
                logger.warning(
                    "Thinking mode enabled but model '%s' may not support it — "
                    "sending thinking_config anyway (API may ignore it)",
                    model_id,
                )
            thinking_kwargs: dict[str, Any] = {
                "include_thoughts": thinking_cfg.include_thoughts,
            }
            if thinking_cfg.budget_tokens is not None:
                thinking_kwargs["thinking_budget"] = thinking_cfg.budget_tokens
            kwargs["thinking_config"] = types.ThinkingConfig(**thinking_kwargs)

        return types.GenerateContentConfig(**kwargs)

    # ── Retry logic ───────────────────────────────────────────

    @staticmethod
    def _check_rate_limit_budget(
        sleep_time: float,
        elapsed: float,
        budget: float,
    ) -> bool:
        """Return ``True`` if the next 429 sleep fits in the cumulative sleep budget.

        Enforces a cumulative 429 wall-clock budget across a single retry
        loop: ``elapsed`` is the sum of all prior 429 backoff sleeps in
        this call, and ``sleep_time`` is the next planned 429 sleep.

        Logs a warning and returns ``False`` when the cumulative 429
        wall-clock budget would be exceeded by the next sleep (i.e.
        ``elapsed + sleep_time > budget``). Callers should break out of
        the retry loop on ``False`` and set their local
        ``budget_exceeded`` flag so the exhaustion handler can distinguish
        a budget abort from plain retry exhaustion.

        See SPEC-RATE-01 (invariant RL-2) in
        ``docs/specs/rate-limit-resilience-v1.md``.
        """
        if sleep_time + elapsed > budget:
            logger.warning(
                "Rate-limit wall-clock budget exhausted — elapsed=%.1fs, next=%.1fs, cap=%.1fs (aborting retries)",
                elapsed,
                sleep_time,
                budget,
            )
            return False
        return True

    @staticmethod
    def _compute_backoff_delay(
        delay: float,
        retry_cfg: RetryConfig,
        *,
        is_rate_limit: bool,
    ) -> tuple[float, float]:
        """Compute the sleep duration and updated delay for the next retry.

        Args:
            delay: Current backoff delay in seconds.
            retry_cfg: Retry configuration.
            is_rate_limit: If ``True``, enforce the longer
                ``rate_limit_initial_delay`` floor.

        Returns:
            ``(sleep_time, new_delay)`` — the time to sleep now and the delay
            value to carry into the next iteration.
        """
        if is_rate_limit:
            delay = max(delay, retry_cfg.rate_limit_initial_delay)
        jitter = random.uniform(0, 0.5)  # noqa: S311
        sleep_time = min(delay, retry_cfg.max_delay) + jitter
        new_delay = delay * retry_cfg.backoff_multiplier
        return sleep_time, new_delay

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

        # SPEC-RATE-01: track cumulative 429 sleep time against a wall-clock
        # cap (tighter when inside a parallel fan-out, see IN_PARALLEL_FANOUT).
        rate_limit_elapsed = 0.0
        rate_limit_budget = _rate_limit_budget_s(retry_cfg)
        budget_exceeded = False

        for attempt in range(retry_cfg.max_retries + 1):
            # Check wall-clock timeout before each attempt (except the first).
            if start_time is not None and attempt > 0:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:  # type: ignore[operator]
                    break  # fall through to raise

            try:
                result = fn()
                self._consecutive_429_count = 0  # SPEC-GEP-03: reset on success
                return result
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
                    except Exception as fallback_exc:  # noqa: BLE001
                        last_exception = fallback_exc
                        break
                is_rate_limit = isinstance(exc, google_exceptions.ResourceExhausted)
                # SPEC-GEP-03: ResourceExhausted is the google-api-core form of 429;
                # track consecutive 429s on the global endpoint and flip if needed.
                if is_rate_limit and self._active_location == "global" and not self._using_fallback:
                    self._consecutive_429_count += 1
                    if self._consecutive_429_count >= 2 and self._flip_to_fallback_on_429():
                        continue
                else:
                    self._consecutive_429_count = 0  # reset on any non-429 retryable
                if attempt < retry_cfg.max_retries:
                    sleep_time, delay = self._compute_backoff_delay(
                        delay,
                        retry_cfg,
                        is_rate_limit=is_rate_limit,
                    )
                    # SPEC-RATE-01: enforce cumulative 429 wall-clock cap.
                    if is_rate_limit and not self._check_rate_limit_budget(
                        sleep_time,
                        rate_limit_elapsed,
                        rate_limit_budget,
                    ):
                        budget_exceeded = True
                        break
                    logger.warning(
                        "Retryable error on attempt %d/%d (%s: %s) — retrying in %.2fs",
                        attempt + 1,
                        retry_cfg.max_retries + 1,
                        type(exc).__name__,
                        exc,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                    if is_rate_limit:
                        rate_limit_elapsed += sleep_time
            except genai_errors.APIError as exc:
                # google-genai SDK errors (ClientError / ServerError).
                # The SDK already retried with backoff via HttpRetryOptions.
                # For 429 (rate-limit) we DO retry at app level with a longer
                # backoff — the SDK's retry budget may be too small for
                # sustained quota pressure.  For other retryable codes the SDK
                # already exhausted its budget so we break immediately.
                if exc.code in _RETRYABLE_STATUS_CODES:
                    last_exception = exc
                    # Check for SSL/connection errors that need location fallback.
                    if _is_ssl_or_connection_error(exc) and not self._using_fallback:
                        logger.warning(
                            "Retryable genai error wraps SSL/proxy error (%s: %s) — attempting location fallback",
                            type(exc).__name__,
                            exc,
                        )
                        self._reinitialize_with_fallback()
                        try:
                            return fn()
                        except Exception as fallback_exc:  # noqa: BLE001
                            last_exception = fallback_exc
                        break  # fall through to exhaustion handler
                    # 429 rate-limit: retry at app level with longer backoff
                    if exc.code == 429:
                        # SPEC-GEP-03: track consecutive 429s on the global endpoint;
                        # flip to regional after 2 consecutive 429s.
                        if self._active_location == "global" and not self._using_fallback:
                            self._consecutive_429_count += 1
                            if self._consecutive_429_count >= 2:
                                if self._flip_to_fallback_on_429():
                                    continue  # retry against new regional client
                    else:
                        self._consecutive_429_count = 0  # reset on non-429 retryable
                    if exc.code == 429 and attempt < retry_cfg.max_retries:
                        sleep_time, delay = self._compute_backoff_delay(
                            delay,
                            retry_cfg,
                            is_rate_limit=True,
                        )
                        # SPEC-RATE-01: enforce cumulative 429 wall-clock cap.
                        if not self._check_rate_limit_budget(
                            sleep_time,
                            rate_limit_elapsed,
                            rate_limit_budget,
                        ):
                            budget_exceeded = True
                            break
                        logger.warning(
                            "Rate-limited (genai 429) on attempt %d/%d — retrying in %.2fs",
                            attempt + 1,
                            retry_cfg.max_retries + 1,
                            sleep_time,
                        )
                        time.sleep(sleep_time)
                        rate_limit_elapsed += sleep_time
                        continue
                    break  # non-429 retryable — fall through to exhaustion handler
                # Convert context-window 400 errors to ContextWindowExceededError.
                if _is_context_window_error(exc):
                    raise ContextWindowExceededError(
                        f"Context window exceeded (API {exc.code}): {exc}",
                    ) from exc
                # 400 CANCELLED — Vertex AI server-side cancellation under load.
                # Treat as transient and retry with backoff (same as 429).
                if exc.code == 400 and "cancelled" in str(exc).lower():
                    if attempt < retry_cfg.max_retries:
                        sleep_time, delay = self._compute_backoff_delay(delay, retry_cfg, is_rate_limit=False)
                        logger.warning(
                            "Vertex AI 400 CANCELLED on attempt %d/%d — retrying in %.2fs",
                            attempt + 1,
                            retry_cfg.max_retries + 1,
                            sleep_time,
                        )
                        time.sleep(sleep_time)
                        continue
                    last_exception = exc
                    break
                raise  # Non-retryable (400, 403, etc.) — propagate immediately.
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
                    except Exception as fallback_exc:  # noqa: BLE001
                        last_exception = fallback_exc
                        break
                raise  # Non-SSL, non-retryable — propagate immediately.

        # All retries exhausted — raise the appropriate custom exception.
        assert last_exception is not None  # noqa: S101
        retries = retry_cfg.max_retries
        if budget_exceeded:
            msg = (
                f"Rate-limit budget exceeded ({rate_limit_elapsed:.1f}s of "
                f"cumulative 429 backoff). Last error: {last_exception}"
            )
        else:
            msg = f"All {retries} retries exhausted. Last error: {last_exception}"

        if isinstance(last_exception, google_exceptions.ResourceExhausted):
            raise GeminiRateLimitError(
                msg,
                original_error=last_exception,
                retries_attempted=retries,
            ) from last_exception

        if isinstance(last_exception, genai_errors.APIError) and last_exception.code == 429:
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

        # SPEC-RATE-01: track cumulative 429 sleep time against a wall-clock
        # cap (tighter when inside a parallel fan-out, see IN_PARALLEL_FANOUT).
        rate_limit_elapsed = 0.0
        rate_limit_budget = _rate_limit_budget_s(retry_cfg)
        budget_exceeded = False

        for attempt in range(retry_cfg.max_retries + 1):
            if start_time is not None and attempt > 0:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:  # type: ignore[operator]
                    break

            try:
                result = await fn()
                self._consecutive_429_count = 0  # SPEC-GEP-03: reset on success
                return result
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
                    except Exception as fallback_exc:  # noqa: BLE001
                        last_exception = fallback_exc
                        break
                is_rate_limit = isinstance(exc, google_exceptions.ResourceExhausted)
                # SPEC-GEP-03: ResourceExhausted is the google-api-core form of 429;
                # track consecutive 429s on the global endpoint and flip if needed.
                if is_rate_limit and self._active_location == "global" and not self._using_fallback:
                    self._consecutive_429_count += 1
                    if self._consecutive_429_count >= 2 and await self._async_flip_to_fallback_on_429():
                        continue
                else:
                    self._consecutive_429_count = 0  # reset on any non-429 retryable
                if attempt < retry_cfg.max_retries:
                    sleep_time, delay = self._compute_backoff_delay(
                        delay,
                        retry_cfg,
                        is_rate_limit=is_rate_limit,
                    )
                    # SPEC-RATE-01: enforce cumulative 429 wall-clock cap.
                    if is_rate_limit and not self._check_rate_limit_budget(
                        sleep_time,
                        rate_limit_elapsed,
                        rate_limit_budget,
                    ):
                        budget_exceeded = True
                        break
                    logger.warning(
                        "Retryable error on attempt %d/%d (%s: %s) — retrying in %.2fs",
                        attempt + 1,
                        retry_cfg.max_retries + 1,
                        type(exc).__name__,
                        exc,
                        sleep_time,
                    )
                    await asyncio.sleep(sleep_time)
                    if is_rate_limit:
                        rate_limit_elapsed += sleep_time
            except genai_errors.APIError as exc:
                # google-genai SDK errors (ClientError / ServerError).
                # The SDK already retried with backoff via HttpRetryOptions.
                # For 429 (rate-limit) we DO retry at app level with a longer
                # backoff — the SDK's retry budget may be too small for
                # sustained quota pressure.  For other retryable codes the SDK
                # already exhausted its budget so we break immediately.
                if exc.code in _RETRYABLE_STATUS_CODES:
                    last_exception = exc
                    # Check for SSL/connection errors that need location fallback.
                    if _is_ssl_or_connection_error(exc) and not self._using_fallback:
                        logger.warning(
                            "Retryable genai error wraps SSL/proxy error (%s: %s) — attempting location fallback",
                            type(exc).__name__,
                            exc,
                        )
                        await self._async_reinitialize_with_fallback()
                        try:
                            return await fn()
                        except Exception as fallback_exc:  # noqa: BLE001
                            last_exception = fallback_exc
                        break  # fall through to exhaustion handler
                    # 429 rate-limit: retry at app level with longer backoff
                    if exc.code == 429:
                        # SPEC-GEP-03: track consecutive 429s on the global endpoint;
                        # flip to regional after 2 consecutive 429s.
                        if self._active_location == "global" and not self._using_fallback:
                            self._consecutive_429_count += 1
                            if self._consecutive_429_count >= 2:
                                if await self._async_flip_to_fallback_on_429():
                                    continue  # retry against new regional client
                    else:
                        self._consecutive_429_count = 0  # reset on non-429 retryable
                    if exc.code == 429 and attempt < retry_cfg.max_retries:
                        sleep_time, delay = self._compute_backoff_delay(
                            delay,
                            retry_cfg,
                            is_rate_limit=True,
                        )
                        # SPEC-RATE-01: enforce cumulative 429 wall-clock cap.
                        if not self._check_rate_limit_budget(
                            sleep_time,
                            rate_limit_elapsed,
                            rate_limit_budget,
                        ):
                            budget_exceeded = True
                            break
                        logger.warning(
                            "Rate-limited (genai 429) on attempt %d/%d — retrying in %.2fs",
                            attempt + 1,
                            retry_cfg.max_retries + 1,
                            sleep_time,
                        )
                        await asyncio.sleep(sleep_time)
                        rate_limit_elapsed += sleep_time
                        continue
                    break  # non-429 retryable — fall through to exhaustion handler
                # Convert context-window 400 errors to ContextWindowExceededError.
                if _is_context_window_error(exc):
                    raise ContextWindowExceededError(
                        f"Context window exceeded (API {exc.code}): {exc}",
                    ) from exc
                # 400 CANCELLED — Vertex AI server-side cancellation under load.
                # Treat as transient and retry with backoff (same as 429).
                if exc.code == 400 and "cancelled" in str(exc).lower():
                    if attempt < retry_cfg.max_retries:
                        sleep_time, delay = self._compute_backoff_delay(delay, retry_cfg, is_rate_limit=False)
                        logger.warning(
                            "Vertex AI 400 CANCELLED on attempt %d/%d — retrying in %.2fs",
                            attempt + 1,
                            retry_cfg.max_retries + 1,
                            sleep_time,
                        )
                        await asyncio.sleep(sleep_time)
                        continue
                    last_exception = exc
                    break
                raise  # Non-retryable (400, 403, etc.) — propagate immediately.
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
                    except Exception as fallback_exc:  # noqa: BLE001
                        last_exception = fallback_exc
                        break
                raise

        assert last_exception is not None  # noqa: S101
        retries = retry_cfg.max_retries
        if budget_exceeded:
            msg = (
                f"Rate-limit budget exceeded ({rate_limit_elapsed:.1f}s of "
                f"cumulative 429 backoff). Last error: {last_exception}"
            )
        else:
            msg = f"All {retries} retries exhausted. Last error: {last_exception}"

        if isinstance(last_exception, google_exceptions.ResourceExhausted):
            raise GeminiRateLimitError(
                msg,
                original_error=last_exception,
                retries_attempted=retries,
            ) from last_exception

        if isinstance(last_exception, genai_errors.APIError) and last_exception.code == 429:
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

    @staticmethod
    def _extract_thinking_text(response: Any) -> str | None:
        """Extract thinking content from a google-genai response.

        Gemini thinking models return ``Part`` objects with ``thought=True``
        for internal reasoning.  This method collects all such parts and
        returns their concatenated text, or ``None`` if no thinking content
        is present.
        """
        candidate = response.candidates[0] if getattr(response, "candidates", None) else None
        content = candidate.content if candidate and getattr(candidate, "content", None) else None
        if not content or not getattr(content, "parts", None):
            return None

        thinking_parts: list[str] = []
        for part in content.parts:
            if getattr(part, "thought", None) is True and getattr(part, "text", None):
                thinking_parts.append(part.text)

        return "".join(thinking_parts) if thinking_parts else None

    @staticmethod
    def _extract_output_text(response: Any) -> str:
        """Extract non-thought text from a google-genai response.

        When thinking mode is active, ``response.text`` may include thinking
        content.  This method filters parts, returning only those where
        ``thought`` is not ``True``.
        """
        candidate = response.candidates[0] if getattr(response, "candidates", None) else None
        content = candidate.content if candidate and getattr(candidate, "content", None) else None
        if not content or not getattr(content, "parts", None):
            return response.text or ""

        output_parts: list[str] = []
        for part in content.parts:
            if getattr(part, "thought", None) is not True and getattr(part, "text", None):
                output_parts.append(part.text)

        return "".join(output_parts) if output_parts else (response.text or "")

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
        self._check_quota(prompt)

        mid = model_id or self._current_model_id
        logger.debug("generate() → model=%s, has_history=%s", mid, bool(history))

        # Warn if history token estimate is high
        if history:
            self._warn_if_history_large(history)

        # ── Cache lookup (only for stateless, text-only prompts) ─────
        cache_key: str | None = None
        if self._cache is not None and isinstance(prompt, str) and not history:
            cache_key = _make_cache_key(prompt, mid, system_instruction)
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.info("Cache hit — skipping API call (model=%s)", mid)
                return cached  # type: ignore[no-any-return]  # ResponseCache.get returns Any

        def _call() -> GenerationResult:
            client = self._get_client()
            config = self._build_generation_config(
                system_instruction=system_instruction,
                **gen_kwargs,
            )

            if history:
                chat_history = self._build_history(history)
                chat = client.chats.create(
                    model=mid,
                    history=chat_history,  # type: ignore[arg-type]
                    config=config,
                )
                response = chat.send_message(prompt)  # type: ignore[arg-type]
            else:
                response = client.models.generate_content(
                    model=mid,
                    contents=prompt,  # type: ignore[arg-type]
                    config=config,
                )

            usage = GeminiClient._extract_usage(response)
            thinking_text = GeminiClient._extract_thinking_text(response)

            # Extract non-thought text from parts when thinking is present,
            # otherwise fall back to response.text for backward compat.
            text = GeminiClient._extract_output_text(response) if thinking_text is not None else (response.text or "")

            return GenerationResult(
                text=text,
                model=mid,
                usage=usage,
                finish_reason=(str(response.candidates[0].finish_reason) if response.candidates else ""),
                thinking_text=thinking_text,
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
        self._check_quota(prompt)

        mid = model_id or self._current_model_id
        logger.debug("generate_stream() → model=%s, has_history=%s", mid, bool(history))

        def _call() -> list[Any]:
            client = self._get_client()
            config = self._build_generation_config(
                system_instruction=system_instruction,
                **gen_kwargs,
            )

            if history:
                chat_history = self._build_history(history)
                chat = client.chats.create(
                    model=mid,
                    history=chat_history,  # type: ignore[arg-type]
                    config=config,
                )
                response_stream = chat.send_message_stream(prompt)  # type: ignore[arg-type]
            else:
                response_stream = client.models.generate_content_stream(
                    model=mid,
                    contents=prompt,  # type: ignore[arg-type]
                    config=config,
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
        self._check_quota(prompt)

        mid = model_id or self._current_model_id
        logger.debug(
            "generate_with_tools() → model=%s, has_history=%s, num_tools=%d",
            mid,
            bool(history),
            len(tool_declarations),
        )

        # Warn if history token estimate is high
        if history:
            self._warn_if_history_large(history)

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
                        model=mid,
                        history=chat_history,  # type: ignore[arg-type]
                        config=config,
                    )
                    response = chat.send_message(actual_prompt)
                else:
                    response = client.models.generate_content(
                        model=mid,
                        contents=prompt,  # type: ignore[arg-type]
                        config=config,
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
                    finish_reason=(str(candidate.finish_reason) if candidate else ""),
                )

            # Parse parts — may contain function calls, text, thinking, or a mix
            function_calls: list[dict[str, Any]] = []
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            raw_fc_parts: list[types.Part] = []

            for part in content.parts:
                # Skip thinking parts — extract them separately.
                # Use ``is True`` to avoid MagicMock / truthy-None surprises.
                if getattr(part, "thought", None) is True:
                    if getattr(part, "text", None):
                        thinking_parts.append(part.text)  # type: ignore[arg-type]
                    continue
                fc = part.function_call
                if fc and fc.name:  # It's a function call
                    function_calls.append(
                        {
                            "name": fc.name,
                            "args": dict(fc.args) if fc.args else {},
                        }
                    )
                    raw_fc_parts.append(part)
                elif part.text:
                    text_parts.append(part.text)

            thinking_text = "".join(thinking_parts) if thinking_parts else None

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
                finish_reason=str(response.candidates[0].finish_reason),  # type: ignore[index]
                thinking_text=thinking_text,
                raw_parts=raw_fc_parts if raw_fc_parts else None,
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

    def _check_quota(self, prompt: str | list[types.Part], *, is_execution: bool = False) -> None:
        """Enforce rate-limit quota if a QuotaChecker is configured.

        Estimates token count from the prompt (``len(str) // 4`` heuristic)
        and delegates to :meth:`QuotaChecker.check_and_consume`.  If the
        checker is ``None`` (rate-limiting disabled), this is a no-op.

        Raises:
            QuotaExceededError: If any quota dimension would be exceeded.
        """
        if self._quota_checker is None:
            return

        # Rough token estimate — same 4-chars-per-token heuristic used elsewhere.
        if isinstance(prompt, str):
            estimated_tokens = max(len(prompt) // 4, 1)
        else:
            estimated_tokens = max(sum(len(str(p)) for p in prompt) // 4, 1)

        from vaig.core.identity import build_composite_key, resolve_identity

        os_user, gcp_email, _ = resolve_identity(self._credentials)
        user_key = build_composite_key(os_user, gcp_email)
        self._quota_checker.check_and_consume(user_key, estimated_tokens, is_execution=is_execution)

    @staticmethod
    def _warn_if_history_large(
        history: list[ChatMessage],
        max_tokens: int = 28_000,
    ) -> None:
        """Log a warning if the rough token estimate for *history* exceeds *max_tokens*.

        Uses the fast ``len(text) / 4`` heuristic — no API call.  This is
        advisory only; the request is still sent (the API may handle it, or
        will return its own error).
        """
        from vaig.session.summarizer import estimate_history_tokens

        estimate = estimate_history_tokens(history)
        if estimate > max_tokens:
            logger.warning(
                "History token estimate (%d) exceeds max_history_tokens (%d) — "
                "the API call may fail if the real token count is too large. "
                "Consider enabling history summarization.",
                estimate,
                max_tokens,
            )

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

    def count_tokens(self, prompt: str | list[types.Part], *, model_id: str | None = None) -> int | None:
        """Count tokens in a prompt (sync).

        Token counting is non-critical — returns ``None`` on failure instead
        of propagating API errors.
        """
        try:
            client = self._get_client()
            mid = model_id or self._current_model_id
            response = client.models.count_tokens(model=mid, contents=prompt)  # type: ignore[arg-type]
            total: int = response.total_tokens if response.total_tokens is not None else 0
            return total
        except Exception as exc:  # noqa: BLE001
            logger.warning("Token counting failed: %s", exc)
            return None

    def list_available_models(self) -> list[dict[str, str]]:
        """List configured available models."""
        return [{"id": m.id, "description": m.description} for m in self._settings.models.available]

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
        await asyncio.to_thread(self._check_quota, prompt)

        mid = model_id or self._current_model_id
        logger.debug("async_generate() → model=%s, has_history=%s", mid, bool(history))

        # ── Cache lookup ────────────────────────────────────────
        cache_key: str | None = None
        if self._cache is not None and isinstance(prompt, str) and not history:
            cache_key = _make_cache_key(prompt, mid, system_instruction)
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.info("Cache hit — skipping API call (model=%s)", mid)
                return cached  # type: ignore[no-any-return]  # ResponseCache.get returns Any

        async def _call() -> GenerationResult:
            client = self._get_client()
            config = self._build_generation_config(
                system_instruction=system_instruction,
                **gen_kwargs,
            )

            if history:
                chat_history = self._build_history(history)
                chat = client.aio.chats.create(
                    model=mid,
                    history=chat_history,  # type: ignore[arg-type]
                    config=config,
                )
                response = await chat.send_message(prompt)  # type: ignore[arg-type]
            else:
                response = await client.aio.models.generate_content(
                    model=mid,
                    contents=prompt,  # type: ignore[arg-type]
                    config=config,
                )

            usage = GeminiClient._extract_usage(response)
            thinking_text = GeminiClient._extract_thinking_text(response)
            text = GeminiClient._extract_output_text(response) if thinking_text is not None else (response.text or "")

            return GenerationResult(
                text=text,
                model=mid,
                usage=usage,
                finish_reason=(str(response.candidates[0].finish_reason) if response.candidates else ""),
                thinking_text=thinking_text,
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
        await asyncio.to_thread(self._check_quota, prompt)

        mid = model_id or self._current_model_id
        logger.debug("async_generate_stream() → model=%s, has_history=%s", mid, bool(history))

        async def _call() -> list[Any]:
            client = self._get_client()
            config = self._build_generation_config(
                system_instruction=system_instruction,
                **gen_kwargs,
            )

            if history:
                chat_history = self._build_history(history)
                chat = client.aio.chats.create(
                    model=mid,
                    history=chat_history,  # type: ignore[arg-type]
                    config=config,
                )
                response_stream = await chat.send_message_stream(prompt)  # type: ignore[arg-type]
            else:
                response_stream = await client.aio.models.generate_content_stream(
                    model=mid,
                    contents=prompt,  # type: ignore[arg-type]
                    config=config,
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
        await asyncio.to_thread(self._check_quota, prompt)

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
                        model=mid,
                        history=chat_history,  # type: ignore[arg-type]
                        config=config,
                    )
                    response = await chat.send_message(actual_prompt)
                else:
                    response = await client.aio.models.generate_content(
                        model=mid,
                        contents=prompt,  # type: ignore[arg-type]
                        config=config,
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
                    finish_reason=(str(candidate.finish_reason) if candidate else ""),
                )

            # Parse parts — may contain function calls, text, thinking, or a mix
            function_calls: list[dict[str, Any]] = []
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            raw_fc_parts: list[types.Part] = []

            for part in content.parts:
                # Skip thinking parts — extract them separately.
                # Use ``is True`` to avoid MagicMock / truthy-None surprises.
                if getattr(part, "thought", None) is True:
                    if getattr(part, "text", None):
                        thinking_parts.append(part.text)  # type: ignore[arg-type]
                    continue
                fc = part.function_call
                if fc and fc.name:
                    function_calls.append(
                        {
                            "name": fc.name,
                            "args": dict(fc.args) if fc.args else {},
                        }
                    )
                    raw_fc_parts.append(part)
                elif part.text:
                    text_parts.append(part.text)

            thinking_text = "".join(thinking_parts) if thinking_parts else None

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
                finish_reason=str(response.candidates[0].finish_reason),  # type: ignore[index]
                thinking_text=thinking_text,
                raw_parts=raw_fc_parts if raw_fc_parts else None,
            )

        return await self._async_retry_with_backoff(_call, timeout=timeout)

    async def async_count_tokens(
        self,
        prompt: str | list[types.Part],
        *,
        model_id: str | None = None,
    ) -> int | None:
        """Count tokens in a prompt (async).

        Token counting is non-critical — returns ``None`` on failure instead
        of propagating API errors.
        """
        try:
            await self._async_ensure_initialized()
            client = self._get_client()
            mid = model_id or self._current_model_id
            response = await client.aio.models.count_tokens(
                model=mid,
                contents=prompt,  # type: ignore[arg-type]
            )
            total: int = response.total_tokens if response.total_tokens is not None else 0
            return total
        except Exception as exc:  # noqa: BLE001
            logger.warning("Token counting failed: %s", exc)
            return None

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

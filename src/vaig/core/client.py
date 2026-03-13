"""Gemini client — google-genai SDK wrapper with streaming, multimodal, and model switching."""

from __future__ import annotations

import logging
import random
import ssl
import time
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from google import genai
from google.api_core import exceptions as google_exceptions
from google.genai import types

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

    def initialize(self) -> None:
        """Initialize the google-genai Client with Vertex AI credentials."""
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

    def _reinitialize_with_fallback(self) -> None:
        """Re-initialize with the fallback location.

        Creates a new Client bound to the fallback location and marks
        the client as using the fallback.
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
        self._client = None
        self.initialize()

    def _ensure_initialized(self) -> None:
        """Ensure the SDK is initialized before making calls."""
        if not self._initialized:
            self.initialize()

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
        return types.GenerateContentConfig(**kwargs)

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
                # A retryable exception (e.g. ServiceUnavailable) may wrap
                # the real cause — an SSL/proxy error.  If so, fallback to
                # the alternate location instead of burning retries against
                # the same broken endpoint.
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
                # else: last attempt failed, fall through to raise below.
            except Exception as exc:
                # Check for SSL/connection/proxy errors that warrant a location fallback.
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

        return self._retry_with_backoff(_call, timeout=timeout)

    def generate_stream(
        self,
        prompt: str | list[types.Part],
        *,
        system_instruction: str | None = None,
        history: list[ChatMessage] | None = None,
        model_id: str | None = None,
        timeout: float | None = None,
        **gen_kwargs: Any,
    ) -> Iterator[str]:
        """Generate a streaming response, yielding text chunks.

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

        chunks = self._retry_with_backoff(_call, timeout=timeout)
        for chunk in chunks:
            if chunk.text:
                yield chunk.text

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
        """Count tokens in a prompt."""
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

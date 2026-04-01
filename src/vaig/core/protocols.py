"""Protocol interfaces for Dependency Injection.

Defines structural (duck-typed) protocols for the key service dependencies
in the VAIG codebase.  All protocols use ``typing.Protocol`` with
``@runtime_checkable`` so they can be verified at runtime via ``isinstance()``.

These protocols decouple consumers from concrete implementations, enabling
easier testing (mock injection) and future swappability.

This module has **zero runtime imports** from concrete implementations — only
stdlib and typing are used.  Type-checking-only imports (under
``TYPE_CHECKING``) are used for IDE support and do not create runtime
dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vaig.core.cache import CacheStats
    from vaig.core.client import ChatMessage, GenerationResult, StreamResult, ToolCallResult

__all__ = [
    "GCPClientProvider",
    "GeminiClientProtocol",
    "K8sClientProvider",
    "PlatformAuthProtocol",
    "SessionStoreProtocol",
]


# ══════════════════════════════════════════════════════════════
# A1. GeminiClient Protocol
# ══════════════════════════════════════════════════════════════


@runtime_checkable
class GeminiClientProtocol(Protocol):
    """Structural protocol matching the public interface of ``GeminiClient``.

    Only methods used by downstream consumers (agents, tools) are declared.
    Internal helpers (``_retry_with_backoff``, ``_build_generation_config``,
    ``_ensure_initialized``, etc.) are excluded.
    """

    @property
    def current_model(self) -> str:
        """Get the current model ID."""
        ...

    def initialize(self) -> None:
        """Initialize the client (sync)."""
        ...

    async def async_initialize(self) -> None:
        """Initialize the client (async)."""
        ...

    def generate(
        self,
        prompt: str | list[Any],
        *,
        system_instruction: str | None = None,
        history: list[ChatMessage] | None = None,
        model_id: str | None = None,
        timeout: float | None = None,
        **gen_kwargs: Any,
    ) -> GenerationResult:
        """Generate a response (non-streaming, sync)."""
        ...

    async def async_generate(
        self,
        prompt: str | list[Any],
        *,
        system_instruction: str | None = None,
        history: list[ChatMessage] | None = None,
        model_id: str | None = None,
        timeout: float | None = None,
        **gen_kwargs: Any,
    ) -> GenerationResult:
        """Generate a response (non-streaming, async)."""
        ...

    def generate_stream(
        self,
        prompt: str | list[Any],
        *,
        system_instruction: str | None = None,
        history: list[ChatMessage] | None = None,
        model_id: str | None = None,
        timeout: float | None = None,
        **gen_kwargs: Any,
    ) -> StreamResult:
        """Generate a streaming response (sync)."""
        ...

    async def async_generate_stream(
        self,
        prompt: str | list[Any],
        *,
        system_instruction: str | None = None,
        history: list[ChatMessage] | None = None,
        model_id: str | None = None,
        timeout: float | None = None,
        **gen_kwargs: Any,
    ) -> StreamResult:
        """Generate a streaming response (async)."""
        ...

    def generate_with_tools(
        self,
        prompt: str | list[Any],
        *,
        tool_declarations: list[Any],
        system_instruction: str | None = None,
        history: list[ChatMessage] | None = None,
        model_id: str | None = None,
        timeout: float | None = None,
        **gen_kwargs: Any,
    ) -> ToolCallResult:
        """Generate a response that may include function calls (sync)."""
        ...

    async def async_generate_with_tools(
        self,
        prompt: str | list[Any],
        *,
        tool_declarations: list[Any],
        system_instruction: str | None = None,
        history: list[ChatMessage] | None = None,
        model_id: str | None = None,
        timeout: float | None = None,
        **gen_kwargs: Any,
    ) -> ToolCallResult:
        """Generate a response that may include function calls (async)."""
        ...

    def count_tokens(
        self,
        prompt: str | list[Any],
        *,
        model_id: str | None = None,
    ) -> int | None:
        """Count tokens in a prompt (sync)."""
        ...

    def switch_model(self, model_id: str) -> str:
        """Switch the active model. Returns the new model ID."""
        ...

    def list_available_models(self) -> list[dict[str, str]]:
        """List configured available models."""
        ...

    @staticmethod
    def build_function_response_parts(results: list[dict[str, Any]]) -> list[Any]:
        """Build Part objects containing function responses."""
        ...

    # ── Cache management ─────────────────────────────────────

    @property
    def cache_enabled(self) -> bool:
        """Whether the response cache is active."""
        ...

    def clear_cache(self) -> int:
        """Clear the response cache. Returns the number of entries removed."""
        ...

    def cache_stats(self) -> CacheStats | None:
        """Return cache statistics, or ``None`` if caching is disabled."""
        ...

    def reinitialize(
        self,
        project: str | None = None,
        location: str | None = None,
    ) -> None:
        """Reinitialize the client with updated project/location."""
        ...


# ══════════════════════════════════════════════════════════════
# A2. K8sClientProvider Protocol
# ══════════════════════════════════════════════════════════════


@runtime_checkable
class K8sClientProvider(Protocol):
    """Protocol for obtaining cached Kubernetes API clients.

    Abstracts the ``_create_k8s_clients()`` / ``_get_exec_client()`` /
    ``clear_k8s_client_cache()`` functions from ``tools/gke/_clients.py``
    into an injectable interface.

    Return types use ``Any`` to avoid hard dependency on the ``kubernetes``
    SDK in the protocol module.
    """

    def get_clients(self, gke_config: Any) -> tuple[Any, Any, Any, Any] | Any:
        """Return cached K8s API clients ``(CoreV1Api, AppsV1Api, CustomObjectsApi, ApiClient)`` or a ``ToolResult`` on failure."""
        ...

    def get_exec_client(self, gke_config: Any) -> Any:
        """Return a fresh, disposable ``CoreV1Api`` for exec operations, or a ``ToolResult`` on failure."""
        ...

    def clear_cache(self) -> None:
        """Clear cached Kubernetes clients."""
        ...


# ══════════════════════════════════════════════════════════════
# A3. GCPClientProvider Protocol
# ══════════════════════════════════════════════════════════════


@runtime_checkable
class GCPClientProvider(Protocol):
    """Protocol for obtaining cached GCP observability clients.

    Abstracts the ``_get_logging_client()`` / ``_get_monitoring_client()``
    functions from ``tools/gcloud_tools.py`` into an injectable interface
    with instance-level caching.
    """

    def get_logging_client(
        self,
        project: str | None = None,
        credentials: Any | None = None,
    ) -> tuple[Any, str | None]:
        """Return a Cloud Logging client and optional error string."""
        ...

    def get_monitoring_client(
        self,
        project: str | None = None,
        credentials: Any | None = None,
    ) -> tuple[Any, str | None]:
        """Return a Cloud Monitoring client and optional error string."""
        ...

    def clear_cache(self) -> None:
        """Clear cached GCP clients."""
        ...


# ══════════════════════════════════════════════════════════════
# A4. SessionStore Protocol
# ══════════════════════════════════════════════════════════════


@runtime_checkable
class SessionStoreProtocol(Protocol):
    """Structural protocol for session persistence backends.

    Matches the async public methods of
    :class:`~vaig.session.store.SessionStore` (SQLite) so that alternative
    implementations (e.g. Firestore) can be swapped in via the same
    interface.

    Only the *async* subset used by the web layer is included.
    """

    async def async_create_session(
        self,
        name: str,
        model: str,
        skill: str | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        user: str = "",
    ) -> str:
        """Create a new session. Returns the session ID."""
        ...

    async def async_add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        model: str | None = None,
        token_count: int = 0,
    ) -> None:
        """Add a message to a session."""
        ...

    async def async_get_messages(
        self, session_id: str, *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Get all messages for a session, ordered by creation time."""
        ...

    async def async_list_sessions(
        self, *, limit: int = 20, user: str | None = None
    ) -> list[dict[str, Any]]:
        """List recent sessions."""
        ...

    async def async_get_session(
        self, session_id: str
    ) -> dict[str, Any] | None:
        """Get session details by ID."""
        ...

    async def async_delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages."""
        ...


# ══════════════════════════════════════════════════════════════
# A5. PlatformAuth Protocol
# ══════════════════════════════════════════════════════════════


@runtime_checkable
class PlatformAuthProtocol(Protocol):
    """Structural protocol for the platform authentication manager.

    Defines the interface that ``PlatformAuthManager`` (or any mock)
    must satisfy for CLI auth, token management, and enforced config
    retrieval.  Used by the service container and auth middleware to
    remain decoupled from the concrete implementation.
    """

    def login(self) -> Any:
        """Run the OAuth PKCE login flow and return an ``AuthResult``."""
        ...

    def logout(self) -> None:
        """Revoke the current token and delete local credentials."""
        ...

    def get_token(self) -> str | None:
        """Return a valid access token, auto-refreshing if needed.

        Returns ``None`` when no valid (or refreshable) credentials exist.
        """
        ...

    def is_authenticated(self) -> bool:
        """Return ``True`` if valid (non-expired or refreshable) credentials exist."""
        ...

    def get_user_info(self) -> dict[str, Any] | None:
        """Decode JWT claims and return user info (email, org_id, role).

        Returns ``None`` when not authenticated.
        """
        ...

    def get_enforced_config(self) -> dict[str, Any]:
        """Fetch the enforced config policy from the backend.

        Returns a dict of field paths to enforced values.  Returns an
        empty dict on network failure (graceful degradation).
        """
        ...

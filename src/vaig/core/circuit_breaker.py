"""Per-run circuit breaker for transient failure protection.

Implements a standard three-state circuit breaker (CLOSED → OPEN → HALF-OPEN)
to prevent cascading failures when a downstream dependency is consistently
failing.

States:
    CLOSED  — normal operation; failures are counted.
    OPEN    — all calls rejected; waits for ``recovery_timeout`` seconds.
    HALF_OPEN — one probe call is allowed; success → CLOSED, failure → OPEN.

Usage::

    from vaig.core.config import CircuitBreakerConfig
    from vaig.core.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30.0))

    await cb.allow_request()    # raises CircuitBreakerOpenError when OPEN
    try:
        result = await do_work()
        await cb.record_success()
    except Exception:
        await cb.record_failure()
        raise
"""

from __future__ import annotations

import threading
import time
from enum import StrEnum
from typing import TYPE_CHECKING

from vaig.core.exceptions import CircuitBreakerOpenError

if TYPE_CHECKING:
    from vaig.core.config import CircuitBreakerConfig

__all__ = ["CircuitBreaker", "CircuitBreakerState", "ModelCircuitBreaker"]


class CircuitBreakerState(StrEnum):
    """Possible states of a :class:`CircuitBreaker`."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Thread-safe circuit breaker for runtime failure protection.

    Uses a :class:`threading.Lock` so it is safe to call from both sync and
    async paths (including multiple event-loop threads and ThreadPoolExecutor
    workers).

    Args:
        config: :class:`~vaig.core.config.CircuitBreakerConfig` with thresholds.

    Raises:
        :class:`~vaig.core.exceptions.CircuitBreakerOpenError`: When
            :meth:`allow_request` is called while the breaker is OPEN.
    """

    def __init__(self, config: CircuitBreakerConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._state = CircuitBreakerState.CLOSED
        self._failure_count: int = 0
        self._opened_at: float = 0.0
        # Sliding window of recent outcomes (True = success, False = failure)
        self._window: list[bool] = []

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def state(self) -> CircuitBreakerState:
        """Current state of the circuit breaker (read-only snapshot)."""
        return self._state

    @property
    def is_open(self) -> bool:
        """``True`` when the breaker is in the OPEN state."""
        return self._state == CircuitBreakerState.OPEN

    async def allow_request(self) -> None:
        """Check whether a request is allowed to proceed.

        Raises:
            :class:`~vaig.core.exceptions.CircuitBreakerOpenError`: If the
                breaker is in the OPEN state and the recovery timeout has not
                elapsed yet.
        """
        with self._lock:
            if self._state == CircuitBreakerState.CLOSED:
                return
            if self._state == CircuitBreakerState.HALF_OPEN:
                return
            # OPEN — check if recovery timeout has elapsed
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._config.recovery_timeout:
                self._state = CircuitBreakerState.HALF_OPEN
                return
            raise CircuitBreakerOpenError(
                failure_count=self._failure_count,
                recovery_timeout=self._config.recovery_timeout,
            )

    async def record_success(self) -> None:
        """Record a successful outcome.

        Resets failure count and transitions HALF_OPEN → CLOSED.
        """
        with self._lock:
            self._failure_count = 0
            self._window = self._window[-(self._config.window_size - 1) :] + [True]
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.CLOSED

    async def record_failure(self) -> None:
        """Record a failed outcome.

        Increments failure count and transitions to OPEN when
        ``failure_threshold`` is reached.
        """
        with self._lock:
            self._failure_count += 1
            self._window = self._window[-(self._config.window_size - 1) :] + [False]
            if self._failure_count >= self._config.failure_threshold:
                self._state = CircuitBreakerState.OPEN
                self._opened_at = time.monotonic()

    async def snapshot(self) -> dict[str, object]:
        """Return a point-in-time snapshot of the breaker state.

        Returns:
            Dict with ``state``, ``failure_count``, ``window``.
        """
        with self._lock:
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "window": list(self._window),
            }


class ModelCircuitBreaker:
    """Registry of per-(location, model_id) circuit breakers.

    Keyed by ``(location, model_id)`` tuple so that a tripped breaker for
    ``"gemini-2.5-pro"`` in ``"us-central1"`` does **not** block
    ``"gemini-2.5-flash"`` in the same location (or any model in a different
    location).

    Uses a :class:`threading.Lock` for the registry so the ``_breakers`` dict
    is safe to access from concurrent sync and async callers (including
    ThreadPoolExecutor workers).

    Usage::

        mcb = ModelCircuitBreaker(CircuitBreakerConfig())
        if mcb.is_open(location, model_id):
            ...  # skip request
        try:
            result = do_work()
            await mcb.record_success(location, model_id)
        except Exception:
            await mcb.record_failure(location, model_id)
            raise
    """

    def __init__(self, config: CircuitBreakerConfig) -> None:
        self._config = config
        self._breakers: dict[tuple[str, str], CircuitBreaker] = {}
        self._lock = threading.Lock()

    def _get_or_create(self, location: str, model_id: str) -> CircuitBreaker:
        """Return the breaker for ``(location, model_id)``, creating if absent.

        Must be called with ``self._lock`` held.
        """
        key = (location, model_id)
        if key not in self._breakers:
            self._breakers[key] = CircuitBreaker(self._config)
        return self._breakers[key]

    def is_open(self, location: str, model_id: str) -> bool:
        """Return ``True`` if the breaker for ``(location, model_id)`` is OPEN.

        Thread-safe sync check — safe to call from non-async code.
        Does not transition state (no side effects).
        """
        with self._lock:
            key = (location, model_id)
            if key not in self._breakers:
                return False
            breaker = self._breakers[key]
            if not breaker.is_open:
                return False
            # Check if recovery timeout has elapsed (same logic as allow_request).
            elapsed = time.monotonic() - breaker._opened_at
            return elapsed < self._config.recovery_timeout

    async def allow_request(self, location: str, model_id: str) -> None:
        """Proxy to the underlying breaker's :meth:`CircuitBreaker.allow_request`."""
        with self._lock:
            breaker = self._get_or_create(location, model_id)
        await breaker.allow_request()

    async def record_success(self, location: str, model_id: str) -> None:
        """Record a successful call for ``(location, model_id)``."""
        with self._lock:
            breaker = self._get_or_create(location, model_id)
        await breaker.record_success()

    async def record_failure(self, location: str, model_id: str) -> None:
        """Record a failed call for ``(location, model_id)``."""
        with self._lock:
            breaker = self._get_or_create(location, model_id)
        await breaker.record_failure()

    async def snapshot(self, location: str, model_id: str) -> dict[str, object]:
        """Return a snapshot of the breaker for ``(location, model_id)``."""
        with self._lock:
            breaker = self._get_or_create(location, model_id)
        return await breaker.snapshot()

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

import asyncio
import time
from enum import StrEnum
from typing import TYPE_CHECKING

from vaig.core.exceptions import CircuitBreakerOpenError

if TYPE_CHECKING:
    from vaig.core.config import CircuitBreakerConfig

__all__ = ["CircuitBreaker", "CircuitBreakerState"]


class CircuitBreakerState(StrEnum):
    """Possible states of a :class:`CircuitBreaker`."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Async-safe circuit breaker for runtime failure protection.

    Args:
        config: :class:`~vaig.core.config.CircuitBreakerConfig` with thresholds.

    Raises:
        :class:`~vaig.core.exceptions.CircuitBreakerOpenError`: When
            :meth:`allow_request` is called while the breaker is OPEN.
    """

    def __init__(self, config: CircuitBreakerConfig) -> None:
        self._config = config
        self._lock = asyncio.Lock()
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

    async def allow_request(self) -> None:
        """Check whether a request is allowed to proceed.

        Raises:
            :class:`~vaig.core.exceptions.CircuitBreakerOpenError`: If the
                breaker is in the OPEN state and the recovery timeout has not
                elapsed yet.
        """
        async with self._lock:
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
        async with self._lock:
            self._failure_count = 0
            self._window = self._window[-(self._config.window_size - 1) :] + [True]
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.CLOSED

    async def record_failure(self) -> None:
        """Record a failed outcome.

        Increments failure count and transitions to OPEN when
        ``failure_threshold`` is reached.
        """
        async with self._lock:
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
        async with self._lock:
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "window": list(self._window),
            }

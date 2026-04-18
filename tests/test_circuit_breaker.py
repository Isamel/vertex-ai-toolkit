"""Tests for CircuitBreaker."""

from __future__ import annotations

import asyncio
import time

import pytest

from vaig.core.circuit_breaker import CircuitBreaker, CircuitBreakerState
from vaig.core.config import CircuitBreakerConfig
from vaig.core.exceptions import CircuitBreakerOpenError

# ── Helpers ───────────────────────────────────────────────────────────────────


def _run(coro):
    return asyncio.run(coro)


def make_cb(failure_threshold: int = 3, recovery_timeout: float = 30.0, window_size: int = 10) -> CircuitBreaker:
    return CircuitBreaker(
        CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            window_size=window_size,
        )
    )


# ── Initial state ─────────────────────────────────────────────────────────────


class TestCircuitBreakerInit:
    def test_starts_in_closed_state(self) -> None:
        cb = make_cb()
        assert cb.state == CircuitBreakerState.CLOSED

    def test_allows_request_when_closed(self) -> None:
        cb = make_cb()
        _run(cb.allow_request())  # must not raise


# ── Failure recording → OPEN transition ──────────────────────────────────────


class TestFailureRecording:
    def test_does_not_trip_below_threshold(self) -> None:
        cb = make_cb(failure_threshold=3)
        _run(cb.record_failure())
        _run(cb.record_failure())
        assert cb.state == CircuitBreakerState.CLOSED

    def test_trips_at_threshold(self) -> None:
        cb = make_cb(failure_threshold=3)
        for _ in range(3):
            _run(cb.record_failure())
        assert cb.state == CircuitBreakerState.OPEN

    def test_allow_request_raises_when_open(self) -> None:
        cb = make_cb(failure_threshold=1)
        _run(cb.record_failure())
        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            _run(cb.allow_request())
        assert exc_info.value.failure_count == 1
        assert exc_info.value.recovery_timeout == 30.0


# ── Success recording ─────────────────────────────────────────────────────────


class TestSuccessRecording:
    def test_success_resets_failure_count(self) -> None:
        cb = make_cb(failure_threshold=3)
        _run(cb.record_failure())
        _run(cb.record_failure())
        _run(cb.record_success())
        snap = _run(cb.snapshot())
        assert snap["failure_count"] == 0

    def test_success_in_half_open_transitions_to_closed(self) -> None:
        cb = make_cb(failure_threshold=1, recovery_timeout=0.0)
        _run(cb.record_failure())
        assert cb.state == CircuitBreakerState.OPEN
        # Simulate recovery timeout elapsed
        cb._opened_at = time.monotonic() - 1
        _run(cb.allow_request())  # should transition to HALF_OPEN
        assert cb.state == CircuitBreakerState.HALF_OPEN
        _run(cb.record_success())
        assert cb.state == CircuitBreakerState.CLOSED


# ── Recovery (OPEN → HALF_OPEN) ───────────────────────────────────────────────


class TestRecovery:
    def test_half_open_after_recovery_timeout_elapsed(self) -> None:
        cb = make_cb(failure_threshold=1, recovery_timeout=1.0)
        _run(cb.record_failure())
        assert cb.state == CircuitBreakerState.OPEN
        # Simulate elapsed time beyond recovery_timeout
        cb._opened_at = time.monotonic() - 2
        _run(cb.allow_request())  # must not raise
        assert cb.state == CircuitBreakerState.HALF_OPEN

    def test_still_open_before_recovery_timeout(self) -> None:
        cb = make_cb(failure_threshold=1, recovery_timeout=3600.0)
        _run(cb.record_failure())
        with pytest.raises(CircuitBreakerOpenError):
            _run(cb.allow_request())
        assert cb.state == CircuitBreakerState.OPEN

    def test_failure_in_half_open_reopens(self) -> None:
        cb = make_cb(failure_threshold=1, recovery_timeout=0.0)
        _run(cb.record_failure())
        cb._opened_at = time.monotonic() - 1
        _run(cb.allow_request())  # → HALF_OPEN
        _run(cb.record_failure())  # failure count hits threshold again → OPEN
        assert cb.state == CircuitBreakerState.OPEN


# ── Sliding window ────────────────────────────────────────────────────────────


class TestSlidingWindow:
    def test_window_bounded_by_window_size(self) -> None:
        cb = make_cb(window_size=3)
        for _ in range(10):
            _run(cb.record_success())
        snap = _run(cb.snapshot())
        assert len(snap["window"]) <= 3

    def test_window_records_outcomes(self) -> None:
        cb = make_cb(failure_threshold=10, window_size=5)
        _run(cb.record_success())
        _run(cb.record_failure())
        snap = _run(cb.snapshot())
        assert True in snap["window"]
        assert False in snap["window"]


# ── Snapshot ──────────────────────────────────────────────────────────────────


class TestSnapshot:
    def test_snapshot_initial(self) -> None:
        cb = make_cb()
        snap = _run(cb.snapshot())
        assert snap["state"] == "closed"
        assert snap["failure_count"] == 0
        assert snap["window"] == []

    def test_snapshot_after_failures(self) -> None:
        cb = make_cb(failure_threshold=5)
        _run(cb.record_failure())
        _run(cb.record_failure())
        snap = _run(cb.snapshot())
        assert snap["failure_count"] == 2
        assert snap["state"] == "closed"

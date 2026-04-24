"""Tests for SPEC-RATE-01 / SPEC-RATE-02 (rate-limit resilience).

Covers:
- ``GeminiClient._check_rate_limit_budget`` wall-clock cap.
- ``_rate_limit_budget_s`` switch between serial and parallel caps.
- ``IN_PARALLEL_FANOUT`` ContextVar semantics.
- New default values on :class:`~vaig.core.config.RetryConfig`.
- Orchestrator jitter-stagger config knobs.
"""

from __future__ import annotations

import logging

import pytest

from vaig.core.client import (
    IN_PARALLEL_FANOUT,
    GeminiClient,
    _rate_limit_budget_s,
)
from vaig.core.config import RetryConfig


class TestRateLimitBudget:
    """``_rate_limit_budget_s`` honours the IN_PARALLEL_FANOUT ContextVar."""

    def test_serial_budget_is_longer(self) -> None:
        cfg = RetryConfig()
        # Default: not in fan-out ⇒ longer cap.
        assert IN_PARALLEL_FANOUT.get() is False
        assert _rate_limit_budget_s(cfg) == cfg.rate_limit_max_total_wait_s
        assert cfg.rate_limit_max_total_wait_s == 300.0

    def test_parallel_budget_is_shorter(self) -> None:
        cfg = RetryConfig()
        tok = IN_PARALLEL_FANOUT.set(True)
        try:
            budget = _rate_limit_budget_s(cfg)
        finally:
            IN_PARALLEL_FANOUT.reset(tok)
        assert budget == cfg.rate_limit_max_total_wait_s_parallel
        assert cfg.rate_limit_max_total_wait_s_parallel == 180.0

    def test_parallel_is_strictly_smaller_than_serial(self) -> None:
        """The parallel cap must always be ≤ the serial cap (invariant)."""
        cfg = RetryConfig()
        assert cfg.rate_limit_max_total_wait_s_parallel <= cfg.rate_limit_max_total_wait_s


class TestCheckRateLimitBudget:
    """``GeminiClient._check_rate_limit_budget`` returns False when over budget."""

    def test_within_budget_returns_true(self) -> None:
        assert (
            GeminiClient._check_rate_limit_budget(
                sleep_time=10.0,
                elapsed=20.0,
                budget=60.0,
            )
            is True
        )

    def test_exact_budget_boundary_passes(self) -> None:
        # Boundary: elapsed + sleep == budget must still fit.
        assert (
            GeminiClient._check_rate_limit_budget(
                sleep_time=30.0,
                elapsed=30.0,
                budget=60.0,
            )
            is True
        )

    def test_over_budget_returns_false(self) -> None:
        assert (
            GeminiClient._check_rate_limit_budget(
                sleep_time=15.0,
                elapsed=50.0,
                budget=60.0,
            )
            is False
        )

    def test_logs_warning_on_exhaustion(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING, logger="vaig.core.client")
        result = GeminiClient._check_rate_limit_budget(
            sleep_time=30.0,
            elapsed=45.0,
            budget=60.0,
        )
        assert result is False
        assert any("wall-clock budget exhausted" in rec.message for rec in caplog.records)

    def test_no_log_when_within_budget(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING, logger="vaig.core.client")
        GeminiClient._check_rate_limit_budget(
            sleep_time=5.0,
            elapsed=5.0,
            budget=60.0,
        )
        # Nothing should have been logged at WARNING.
        assert not any("wall-clock budget" in rec.message for rec in caplog.records)


class TestRetryConfigNewFields:
    """New SPEC-RATE-01 fields exist with documented defaults and validation."""

    def test_rate_limit_max_total_wait_bounds(self) -> None:
        # Negative values must be rejected (ge=0.0).
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            RetryConfig(rate_limit_max_total_wait_s=-1.0)
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            RetryConfig(rate_limit_max_total_wait_s_parallel=-1.0)

    def test_jitter_defaults(self) -> None:
        cfg = RetryConfig()
        assert cfg.parallel_launch_jitter_min_s == 0.5
        assert cfg.parallel_launch_jitter_max_s == 2.0
        assert cfg.parallel_launch_jitter_min_s <= cfg.parallel_launch_jitter_max_s

    def test_jitter_zero_disables(self) -> None:
        # Both set to 0 means "no jitter" — orchestrator skips the sleep.
        cfg = RetryConfig(
            parallel_launch_jitter_min_s=0.0,
            parallel_launch_jitter_max_s=0.0,
        )
        assert cfg.parallel_launch_jitter_min_s == 0.0
        assert cfg.parallel_launch_jitter_max_s == 0.0

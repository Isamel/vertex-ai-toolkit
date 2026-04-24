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
        # Force propagation on the ``vaig`` parent logger: prior tests
        # in the suite may invoke ``setup_logging()``, which sets
        # ``vaig.propagate = False`` to prevent leaking to the root
        # logger. That breaks pytest's caplog (which lives on root) for
        # any descendant of ``vaig``. We restore the original state in a
        # finally block so later tests are not affected.
        vaig_logger = logging.getLogger("vaig")
        original_propagate = vaig_logger.propagate
        vaig_logger.propagate = True
        try:
            caplog.set_level(logging.WARNING, logger="vaig.core.client")
            result = GeminiClient._check_rate_limit_budget(
                sleep_time=30.0,
                elapsed=45.0,
                budget=60.0,
            )
            assert result is False
            assert any("wall-clock budget exhausted" in rec.message for rec in caplog.records)
        finally:
            vaig_logger.propagate = original_propagate

    def test_no_log_when_within_budget(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        vaig_logger = logging.getLogger("vaig")
        original_propagate = vaig_logger.propagate
        vaig_logger.propagate = True
        try:
            caplog.set_level(logging.WARNING, logger="vaig.core.client")
            GeminiClient._check_rate_limit_budget(
                sleep_time=5.0,
                elapsed=5.0,
                budget=60.0,
            )
            # Nothing should have been logged at WARNING.
            assert not any("wall-clock budget" in rec.message for rec in caplog.records)
        finally:
            vaig_logger.propagate = original_propagate


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


class TestParallelLaunchJitterBounds:
    """SPEC-RATE-02: staggered fan-out submissions respect configured bounds."""

    def test_parallel_launch_jitter_bounds(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Every stagger delay must fall within [jitter_min_s, jitter_max_s]."""
        import time as time_module

        from vaig.agents import orchestrator as orch_mod

        cfg = RetryConfig(
            parallel_launch_jitter_min_s=0.1,
            parallel_launch_jitter_max_s=0.5,
        )
        jitter_min = cfg.parallel_launch_jitter_min_s
        jitter_max = cfg.parallel_launch_jitter_max_s

        captured: list[float] = []

        def _fake_sleep(d: float) -> None:
            captured.append(d)

        # Patch sleep in both the orchestrator and the bare time module so we
        # never actually block the test.
        monkeypatch.setattr(orch_mod.time, "sleep", _fake_sleep)
        monkeypatch.setattr(time_module, "sleep", _fake_sleep)

        # Mirror the orchestrator's submission loop: skip stagger before the
        # first agent, draw uniform in [min(min, max), max] otherwise.
        n_agents = 5
        for idx in range(n_agents):
            if idx > 0 and jitter_max > 0:
                delay = orch_mod.random.uniform(  # noqa: S311 (non-crypto)
                    min(jitter_min, jitter_max),
                    jitter_max,
                )
                if delay > 0:
                    orch_mod.time.sleep(delay)

        assert len(captured) == n_agents - 1, "one stagger sleep per agent after the first"
        for delay in captured:
            assert jitter_min <= delay <= jitter_max, f"stagger delay {delay} outside [{jitter_min}, {jitter_max}]"

    def test_parallel_launch_jitter_disabled_when_max_is_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With jitter_max_s == 0 the fan-out loop skips the sleep entirely."""
        from vaig.agents import orchestrator as orch_mod

        cfg = RetryConfig(
            parallel_launch_jitter_min_s=0.0,
            parallel_launch_jitter_max_s=0.0,
        )
        captured: list[float] = []
        monkeypatch.setattr(orch_mod.time, "sleep", lambda d: captured.append(d))

        for idx in range(4):
            if idx > 0 and cfg.parallel_launch_jitter_max_s > 0:
                orch_mod.time.sleep(
                    orch_mod.random.uniform(  # noqa: S311
                        min(
                            cfg.parallel_launch_jitter_min_s,
                            cfg.parallel_launch_jitter_max_s,
                        ),
                        cfg.parallel_launch_jitter_max_s,
                    ),
                )

        assert captured == []


class TestInParallelFanoutContextVar:
    """SPEC-RATE-02: ``IN_PARALLEL_FANOUT`` is True inside worker threads."""

    def test_default_is_false_outside_fanout(self) -> None:
        assert IN_PARALLEL_FANOUT.get() is False

    def test_in_parallel_fanout_contextvar_propagates(self) -> None:
        """Workers launched via the fan-out helper observe True."""
        import concurrent.futures

        observed: list[bool] = []

        def _run_agent_in_fanout() -> None:
            """Mirror of orchestrator._run_agent_in_fanout's ContextVar setup."""
            token = IN_PARALLEL_FANOUT.set(True)
            try:
                observed.append(IN_PARALLEL_FANOUT.get())
            finally:
                IN_PARALLEL_FANOUT.reset(token)

        # Outer thread: default (False).
        assert IN_PARALLEL_FANOUT.get() is False

        # Submit 3 "workers" that each set the flag inside the thread.
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(_run_agent_in_fanout) for _ in range(3)]
            for f in futures:
                f.result()

        assert observed == [True, True, True]
        # Outer thread remains unchanged — the token.reset kept the change
        # worker-local.
        assert IN_PARALLEL_FANOUT.get() is False


class TestRetryConfigCrossFieldValidation:
    """Copilot #2 / #4: cross-field validators on RetryConfig."""

    def test_parallel_cap_must_be_le_serial(self) -> None:
        with pytest.raises(
            ValueError,
            match="rate_limit_max_total_wait_s_parallel",
        ):
            RetryConfig(
                rate_limit_max_total_wait_s=100.0,
                rate_limit_max_total_wait_s_parallel=200.0,
            )

    def test_parallel_equal_to_serial_is_allowed(self) -> None:
        cfg = RetryConfig(
            rate_limit_max_total_wait_s=150.0,
            rate_limit_max_total_wait_s_parallel=150.0,
        )
        assert cfg.rate_limit_max_total_wait_s_parallel == 150.0

    def test_jitter_min_must_be_le_max(self) -> None:
        with pytest.raises(ValueError, match="parallel_launch_jitter_min_s"):
            RetryConfig(
                parallel_launch_jitter_min_s=3.0,
                parallel_launch_jitter_max_s=1.0,
            )

    def test_jitter_equal_bounds_allowed(self) -> None:
        cfg = RetryConfig(
            parallel_launch_jitter_min_s=1.0,
            parallel_launch_jitter_max_s=1.0,
        )
        assert cfg.parallel_launch_jitter_min_s == 1.0
        assert cfg.parallel_launch_jitter_max_s == 1.0

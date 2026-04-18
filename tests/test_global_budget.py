"""Tests for GlobalBudgetManager."""

from __future__ import annotations

import asyncio
import time

import pytest

from vaig.core.config import GlobalBudgetConfig
from vaig.core.exceptions import BudgetExhaustedError
from vaig.core.global_budget import GlobalBudgetManager

# ── Helpers ───────────────────────────────────────────────────────────────────


def _run(coro):
    return asyncio.run(coro)


def make_mgr(**kwargs) -> GlobalBudgetManager:
    return GlobalBudgetManager(GlobalBudgetConfig(**kwargs))


# ── Initialization ────────────────────────────────────────────────────────────


class TestGlobalBudgetManagerInit:
    def test_creates_with_defaults(self) -> None:
        mgr = make_mgr()
        snap = _run(mgr.snapshot())
        assert snap["tokens"] == 0
        assert snap["cost_usd"] == 0.0
        assert snap["tool_calls"] == 0
        assert snap["wall_seconds"] >= 0.0


# ── record_tokens ─────────────────────────────────────────────────────────────


class TestRecordTokens:
    def test_accumulates_tokens(self) -> None:
        mgr = make_mgr()
        _run(mgr.record_tokens(1000))
        _run(mgr.record_tokens(500))
        snap = _run(mgr.snapshot())
        assert snap["tokens"] == 1500

    def test_zero_tokens_no_op(self) -> None:
        mgr = make_mgr()
        _run(mgr.record_tokens(0))
        snap = _run(mgr.snapshot())
        assert snap["tokens"] == 0


# ── record_cost ───────────────────────────────────────────────────────────────


class TestRecordCost:
    def test_accumulates_cost(self) -> None:
        mgr = make_mgr()
        _run(mgr.record_cost(1.50))
        _run(mgr.record_cost(0.25))
        snap = _run(mgr.snapshot())
        assert abs(snap["cost_usd"] - 1.75) < 1e-9


# ── record_tool_call ──────────────────────────────────────────────────────────


class TestRecordToolCall:
    def test_increments_counter(self) -> None:
        mgr = make_mgr()
        _run(mgr.record_tool_call())
        _run(mgr.record_tool_call())
        snap = _run(mgr.snapshot())
        assert snap["tool_calls"] == 2


# ── check: unlimited (0 = no limit) ──────────────────────────────────────────


class TestCheckUnlimited:
    def test_no_limits_never_raises(self) -> None:
        mgr = make_mgr()
        _run(mgr.record_tokens(10_000_000))
        _run(mgr.record_cost(9999.0))
        for _ in range(100):
            _run(mgr.record_tool_call())
        _run(mgr.check())  # must not raise


# ── check: token limit ────────────────────────────────────────────────────────


class TestCheckTokenLimit:
    def test_raises_when_tokens_exceeded(self) -> None:
        mgr = make_mgr(max_tokens=100)
        _run(mgr.record_tokens(101))
        with pytest.raises(BudgetExhaustedError) as exc_info:
            _run(mgr.check())
        assert exc_info.value.dimension == "tokens"
        assert exc_info.value.used == 101
        assert exc_info.value.limit == 100

    def test_no_raise_at_exact_limit(self) -> None:
        mgr = make_mgr(max_tokens=100)
        _run(mgr.record_tokens(100))
        _run(mgr.check())  # must not raise


# ── check: cost limit ─────────────────────────────────────────────────────────


class TestCheckCostLimit:
    def test_raises_when_cost_exceeded(self) -> None:
        mgr = make_mgr(max_cost_usd=1.0)
        _run(mgr.record_cost(1.01))
        with pytest.raises(BudgetExhaustedError) as exc_info:
            _run(mgr.check())
        assert exc_info.value.dimension == "cost_usd"

    def test_no_raise_at_exact_limit(self) -> None:
        mgr = make_mgr(max_cost_usd=1.0)
        _run(mgr.record_cost(1.0))
        _run(mgr.check())


# ── check: tool_calls limit ───────────────────────────────────────────────────


class TestCheckToolCallLimit:
    def test_raises_when_tool_calls_exceeded(self) -> None:
        mgr = make_mgr(max_tool_calls=3)
        for _ in range(4):
            _run(mgr.record_tool_call())
        with pytest.raises(BudgetExhaustedError) as exc_info:
            _run(mgr.check())
        assert exc_info.value.dimension == "tool_calls"

    def test_no_raise_at_exact_limit(self) -> None:
        mgr = make_mgr(max_tool_calls=3)
        for _ in range(3):
            _run(mgr.record_tool_call())
        _run(mgr.check())


# ── check: wall_seconds limit ─────────────────────────────────────────────────


class TestCheckWallSecondsLimit:
    def test_raises_when_wall_time_exceeded(self) -> None:
        # Create a manager and manually monkey-patch start time to simulate elapsed time
        mgr = make_mgr(max_wall_seconds=1)
        mgr._start_time = time.monotonic() - 2  # simulate 2 seconds elapsed
        with pytest.raises(BudgetExhaustedError) as exc_info:
            _run(mgr.check())
        assert exc_info.value.dimension == "wall_seconds"

    def test_no_raise_within_limit(self) -> None:
        mgr = make_mgr(max_wall_seconds=3600)
        _run(mgr.check())  # must not raise immediately


# ── check: priority order ─────────────────────────────────────────────────────


class TestCheckPriorityOrder:
    def test_tokens_checked_before_cost(self) -> None:
        mgr = make_mgr(max_tokens=1, max_cost_usd=1.0)
        _run(mgr.record_tokens(2))
        _run(mgr.record_cost(2.0))
        with pytest.raises(BudgetExhaustedError) as exc_info:
            _run(mgr.check())
        # tokens is checked first
        assert exc_info.value.dimension == "tokens"

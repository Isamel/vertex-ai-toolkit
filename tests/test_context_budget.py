"""Tests for ContextBudgetManager."""

from __future__ import annotations

import pytest

from vaig.core.context_budget import ContextBudgetManager


class TestContextBudgetManagerInit:
    """Constructor and validation tests."""

    def test_valid_phases_accepted(self) -> None:
        manager = ContextBudgetManager(
            total_budget=100_000,
            phases={"tool_loop": 0.7, "summariser": 0.2},
        )
        assert manager is not None

    def test_fractions_summing_to_exactly_one_accepted(self) -> None:
        manager = ContextBudgetManager(
            total_budget=10_000,
            phases={"a": 0.5, "b": 0.5},
        )
        assert manager.remaining("a") == 5_000

    def test_fractions_exceeding_one_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="sum"):
            ContextBudgetManager(
                total_budget=10_000,
                phases={"a": 0.7, "b": 0.5},
            )

    def test_empty_phases_accepted(self) -> None:
        manager = ContextBudgetManager(total_budget=10_000, phases={})
        assert manager.summary() == {}


class TestRecordUsage:
    """record_usage accumulates tokens per phase."""

    def test_record_usage_accumulates(self) -> None:
        manager = ContextBudgetManager(10_000, {"tool_loop": 1.0})
        manager.record_usage("tool_loop", 1000)
        manager.record_usage("tool_loop", 500)
        assert manager.summary()["tool_loop"] == 1500

    def test_record_usage_unregistered_phase_tracked(self) -> None:
        manager = ContextBudgetManager(10_000, {})
        manager.record_usage("unknown_phase", 999)
        assert manager.summary()["unknown_phase"] == 999


class TestIsOverBudget:
    """is_over_budget returns True only when usage exceeds allocated budget."""

    def test_under_budget_returns_false(self) -> None:
        manager = ContextBudgetManager(10_000, {"tool_loop": 0.5})
        manager.record_usage("tool_loop", 100)
        assert manager.is_over_budget("tool_loop") is False

    def test_exactly_at_budget_returns_false(self) -> None:
        manager = ContextBudgetManager(10_000, {"tool_loop": 0.5})
        manager.record_usage("tool_loop", 5_000)
        assert manager.is_over_budget("tool_loop") is False

    def test_over_budget_returns_true(self) -> None:
        manager = ContextBudgetManager(10_000, {"tool_loop": 0.5})
        manager.record_usage("tool_loop", 5_001)
        assert manager.is_over_budget("tool_loop") is True

    def test_unregistered_phase_over_budget_when_tokens_recorded(self) -> None:
        manager = ContextBudgetManager(10_000, {})
        manager.record_usage("ghost", 1)
        assert manager.is_over_budget("ghost") is True

    def test_unknown_phase_without_usage_not_over_budget(self) -> None:
        manager = ContextBudgetManager(10_000, {})
        assert manager.is_over_budget("nonexistent") is False


class TestRemaining:
    """remaining returns budget minus usage (may be negative)."""

    def test_remaining_positive_when_under_budget(self) -> None:
        manager = ContextBudgetManager(10_000, {"phase": 0.4})
        manager.record_usage("phase", 1_000)
        assert manager.remaining("phase") == 3_000

    def test_remaining_zero_at_exact_budget(self) -> None:
        manager = ContextBudgetManager(10_000, {"phase": 0.4})
        manager.record_usage("phase", 4_000)
        assert manager.remaining("phase") == 0

    def test_remaining_negative_when_over_budget(self) -> None:
        manager = ContextBudgetManager(10_000, {"phase": 0.1})
        manager.record_usage("phase", 2_000)
        assert manager.remaining("phase") < 0

    def test_remaining_unknown_phase_returns_zero(self) -> None:
        manager = ContextBudgetManager(10_000, {})
        assert manager.remaining("ghost") == 0


class TestSummary:
    """summary returns a snapshot of all usage."""

    def test_summary_empty_at_start(self) -> None:
        manager = ContextBudgetManager(10_000, {"a": 0.3, "b": 0.5})
        assert manager.summary() == {"a": 0, "b": 0}

    def test_summary_reflects_recorded_usage(self) -> None:
        manager = ContextBudgetManager(10_000, {"a": 0.3, "b": 0.5})
        manager.record_usage("a", 100)
        manager.record_usage("b", 200)
        assert manager.summary() == {"a": 100, "b": 200}

    def test_summary_returns_copy(self) -> None:
        manager = ContextBudgetManager(10_000, {"a": 0.5})
        s = manager.summary()
        s["a"] = 99999
        assert manager.summary()["a"] == 0

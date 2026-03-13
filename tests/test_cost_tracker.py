"""Tests for CostTracker — per-session cost accumulation and budget enforcement."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from vaig.core.config import BudgetConfig
from vaig.core.cost_tracker import BudgetStatus, CostRecord, CostTracker


# ══════════════════════════════════════════════════════════════
# CostRecord
# ══════════════════════════════════════════════════════════════


class TestCostRecord:
    """Tests for CostRecord dataclass."""

    def test_basic_construction(self) -> None:
        rec = CostRecord(
            model_id="gemini-2.5-pro",
            prompt_tokens=100,
            completion_tokens=50,
            thinking_tokens=0,
            cost=0.001,
        )
        assert rec.model_id == "gemini-2.5-pro"
        assert rec.prompt_tokens == 100
        assert rec.completion_tokens == 50
        assert rec.thinking_tokens == 0
        assert rec.cost == 0.001

    def test_timestamp_defaults_to_utc_now(self) -> None:
        before = datetime.now(timezone.utc)
        rec = CostRecord(
            model_id="gemini-2.5-pro",
            prompt_tokens=0,
            completion_tokens=0,
            thinking_tokens=0,
            cost=0.0,
        )
        after = datetime.now(timezone.utc)
        assert before <= rec.timestamp <= after

    def test_custom_timestamp(self) -> None:
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        rec = CostRecord(
            model_id="gemini-2.5-pro",
            prompt_tokens=0,
            completion_tokens=0,
            thinking_tokens=0,
            cost=0.0,
            timestamp=ts,
        )
        assert rec.timestamp == ts


# ══════════════════════════════════════════════════════════════
# CostTracker — Recording & Accumulation
# ══════════════════════════════════════════════════════════════


class TestCostTrackerRecording:
    """Tests for recording API calls and accumulating totals."""

    def test_initial_state_is_zero(self) -> None:
        tracker = CostTracker()
        assert tracker.total_cost == 0.0
        assert tracker.total_tokens == 0
        assert tracker.request_count == 0

    def test_record_single_call(self) -> None:
        tracker = CostTracker()
        rec = tracker.record("gemini-2.5-pro", prompt_tokens=1000, completion_tokens=500)

        assert rec.model_id == "gemini-2.5-pro"
        assert rec.prompt_tokens == 1000
        assert rec.completion_tokens == 500
        assert rec.thinking_tokens == 0
        assert rec.cost > 0
        assert tracker.request_count == 1

    def test_accumulate_multiple_calls(self) -> None:
        tracker = CostTracker()
        tracker.record("gemini-2.5-pro", prompt_tokens=1000, completion_tokens=500)
        tracker.record("gemini-2.5-pro", prompt_tokens=2000, completion_tokens=1000)

        summary = tracker.summary()
        assert summary["total_prompt_tokens"] == 3000
        assert summary["total_completion_tokens"] == 1500
        assert summary["request_count"] == 2
        assert summary["total_cost"] > 0

    def test_record_with_thinking_tokens(self) -> None:
        tracker = CostTracker()
        tracker.record(
            "gemini-2.5-pro",
            prompt_tokens=500,
            completion_tokens=200,
            thinking_tokens=3000,
        )

        summary = tracker.summary()
        assert summary["total_thinking_tokens"] == 3000
        assert summary["total_cost"] > 0

    def test_record_zero_tokens(self) -> None:
        tracker = CostTracker()
        tracker.record("gemini-2.5-pro", prompt_tokens=0, completion_tokens=0)

        assert tracker.request_count == 1
        assert tracker.total_cost == 0.0
        assert tracker.total_tokens == 0

    def test_unknown_model_records_zero_cost(self) -> None:
        tracker = CostTracker()
        rec = tracker.record("unknown-model-xyz", prompt_tokens=1000, completion_tokens=500)

        assert rec.cost == 0.0
        assert tracker.total_cost == 0.0
        assert tracker.request_count == 1
        # Tokens still accumulated even if cost is zero
        assert tracker.total_tokens == 1500

    def test_total_tokens_includes_all_categories(self) -> None:
        tracker = CostTracker()
        tracker.record(
            "gemini-2.5-pro",
            prompt_tokens=100,
            completion_tokens=50,
            thinking_tokens=200,
        )

        assert tracker.total_tokens == 350

    def test_mixed_models_accumulate(self) -> None:
        tracker = CostTracker()
        tracker.record("gemini-2.5-pro", prompt_tokens=1000, completion_tokens=500)
        tracker.record("gemini-2.5-flash", prompt_tokens=1000, completion_tokens=500)

        summary = tracker.summary()
        assert summary["request_count"] == 2
        assert summary["total_prompt_tokens"] == 2000
        assert summary["total_completion_tokens"] == 1000


# ══════════════════════════════════════════════════════════════
# CostTracker — Summary
# ══════════════════════════════════════════════════════════════


class TestCostTrackerSummary:
    """Tests for the summary() method."""

    def test_empty_summary(self) -> None:
        tracker = CostTracker()
        summary = tracker.summary()

        assert summary["total_cost"] == 0.0
        assert summary["total_prompt_tokens"] == 0
        assert summary["total_completion_tokens"] == 0
        assert summary["total_thinking_tokens"] == 0
        assert summary["total_tokens"] == 0
        assert summary["request_count"] == 0

    def test_summary_matches_individual_totals(self) -> None:
        tracker = CostTracker()
        tracker.record("gemini-2.5-pro", prompt_tokens=500, completion_tokens=200, thinking_tokens=100)

        summary = tracker.summary()
        assert summary["total_prompt_tokens"] == 500
        assert summary["total_completion_tokens"] == 200
        assert summary["total_thinking_tokens"] == 100
        assert summary["total_tokens"] == 800
        assert summary["total_cost"] == tracker.total_cost


# ══════════════════════════════════════════════════════════════
# CostTracker — Budget Checking
# ══════════════════════════════════════════════════════════════


class TestCostTrackerBudget:
    """Tests for budget checking."""

    def test_budget_disabled_returns_ok(self) -> None:
        tracker = CostTracker()
        tracker.record("gemini-2.5-pro", prompt_tokens=1_000_000, completion_tokens=1_000_000)

        config = BudgetConfig(enabled=False)
        status, message = tracker.check_budget(config)

        assert status == BudgetStatus.OK
        assert message is None

    def test_budget_ok_when_under_threshold(self) -> None:
        tracker = CostTracker()
        tracker.record("gemini-2.5-pro", prompt_tokens=100, completion_tokens=50)

        config = BudgetConfig(enabled=True, max_cost_usd=10.0, warn_threshold=0.8)
        status, message = tracker.check_budget(config)

        assert status == BudgetStatus.OK
        assert message is None

    def test_budget_warning_at_threshold(self) -> None:
        tracker = CostTracker()
        # Make the cost exceed warn_threshold (80%) but under max_cost_usd
        # gemini-2.5-pro: 1M output tokens = $10.00
        # With max_cost_usd=10.0, warn at $8.00
        tracker.record("gemini-2.5-pro", prompt_tokens=0, completion_tokens=900_000)
        # 900k * $10/1M = $9.00 — above 80% warning threshold

        config = BudgetConfig(enabled=True, max_cost_usd=10.0, warn_threshold=0.8)
        status, message = tracker.check_budget(config)

        assert status == BudgetStatus.WARNING
        assert message is not None
        assert "warning" in message.lower()

    def test_budget_exceeded(self) -> None:
        tracker = CostTracker()
        # gemini-2.5-pro: 1M output tokens = $10.00
        tracker.record("gemini-2.5-pro", prompt_tokens=0, completion_tokens=1_000_000)

        config = BudgetConfig(enabled=True, max_cost_usd=5.0)
        status, message = tracker.check_budget(config)

        assert status == BudgetStatus.EXCEEDED
        assert message is not None
        assert "exceeded" in message.lower()

    def test_budget_exactly_at_limit_is_exceeded(self) -> None:
        """Cost equal to max_cost_usd should be EXCEEDED, not WARNING."""
        tracker = CostTracker()
        tracker.record("gemini-2.5-pro", prompt_tokens=0, completion_tokens=500_000)
        # 500k * $10/1M = $5.00

        config = BudgetConfig(enabled=True, max_cost_usd=5.0)
        status, _ = tracker.check_budget(config)

        assert status == BudgetStatus.EXCEEDED


# ══════════════════════════════════════════════════════════════
# CostTracker — Reset
# ══════════════════════════════════════════════════════════════


class TestCostTrackerReset:
    """Tests for reset()."""

    def test_reset_clears_all_data(self) -> None:
        tracker = CostTracker()
        tracker.record("gemini-2.5-pro", prompt_tokens=1000, completion_tokens=500)
        tracker.record("gemini-2.5-flash", prompt_tokens=500, completion_tokens=200)

        tracker.reset()

        assert tracker.total_cost == 0.0
        assert tracker.total_tokens == 0
        assert tracker.request_count == 0

    def test_reset_allows_fresh_recording(self) -> None:
        tracker = CostTracker()
        tracker.record("gemini-2.5-pro", prompt_tokens=1000, completion_tokens=500)
        tracker.reset()
        tracker.record("gemini-2.5-flash", prompt_tokens=100, completion_tokens=50)

        assert tracker.request_count == 1
        summary = tracker.summary()
        assert summary["total_prompt_tokens"] == 100


# ══════════════════════════════════════════════════════════════
# CostTracker — Serialization (to_dict / from_dict)
# ══════════════════════════════════════════════════════════════


class TestCostTrackerSerialization:
    """Tests for to_dict() and from_dict() round-tripping."""

    def test_to_dict_empty_tracker(self) -> None:
        tracker = CostTracker()
        data = tracker.to_dict()

        assert data["total_cost"] == 0.0
        assert data["total_prompt_tokens"] == 0
        assert data["total_completion_tokens"] == 0
        assert data["total_thinking_tokens"] == 0
        assert data["request_count"] == 0
        assert data["records"] == []

    def test_to_dict_with_records(self) -> None:
        tracker = CostTracker()
        tracker.record("gemini-2.5-pro", prompt_tokens=1000, completion_tokens=500, thinking_tokens=200)

        data = tracker.to_dict()

        assert len(data["records"]) == 1
        rec = data["records"][0]
        assert rec["model_id"] == "gemini-2.5-pro"
        assert rec["prompt_tokens"] == 1000
        assert rec["completion_tokens"] == 500
        assert rec["thinking_tokens"] == 200
        assert rec["cost"] > 0
        assert "timestamp" in rec

    def test_from_dict_restores_state(self) -> None:
        original = CostTracker()
        original.record("gemini-2.5-pro", prompt_tokens=1000, completion_tokens=500)
        original.record("gemini-2.5-flash", prompt_tokens=2000, completion_tokens=1000, thinking_tokens=500)

        data = original.to_dict()
        restored = CostTracker.from_dict(data)

        assert restored.total_cost == original.total_cost
        assert restored.total_tokens == original.total_tokens
        assert restored.request_count == original.request_count

        restored_summary = restored.summary()
        original_summary = original.summary()
        assert restored_summary["total_prompt_tokens"] == original_summary["total_prompt_tokens"]
        assert restored_summary["total_completion_tokens"] == original_summary["total_completion_tokens"]
        assert restored_summary["total_thinking_tokens"] == original_summary["total_thinking_tokens"]

    def test_round_trip_preserves_timestamps(self) -> None:
        tracker = CostTracker()
        rec = tracker.record("gemini-2.5-pro", prompt_tokens=100, completion_tokens=50)

        data = tracker.to_dict()
        restored = CostTracker.from_dict(data)

        assert restored._records[0].timestamp == rec.timestamp

    def test_from_dict_with_empty_records(self) -> None:
        data = {
            "total_cost": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_thinking_tokens": 0,
            "request_count": 0,
            "records": [],
        }
        tracker = CostTracker.from_dict(data)

        assert tracker.request_count == 0
        assert tracker.total_cost == 0.0

    def test_from_dict_with_missing_records_key(self) -> None:
        """from_dict should handle data without a 'records' key gracefully."""
        data = {"total_cost": 0.0}
        tracker = CostTracker.from_dict(data)

        assert tracker.request_count == 0

    def test_new_records_added_after_restore(self) -> None:
        """After restoring, new records should accumulate on top of restored state."""
        original = CostTracker()
        original.record("gemini-2.5-pro", prompt_tokens=1000, completion_tokens=500)

        data = original.to_dict()
        restored = CostTracker.from_dict(data)

        # Add more records
        restored.record("gemini-2.5-flash", prompt_tokens=2000, completion_tokens=1000)

        assert restored.request_count == 2
        assert restored.total_tokens > original.total_tokens


# ══════════════════════════════════════════════════════════════
# CostTracker — Thread Safety
# ══════════════════════════════════════════════════════════════


class TestCostTrackerThreadSafety:
    """Tests for concurrent recording."""

    def test_concurrent_recording(self) -> None:
        """Multiple threads recording simultaneously should not lose data."""
        tracker = CostTracker()
        num_threads = 10
        records_per_thread = 100

        def record_many() -> None:
            for _ in range(records_per_thread):
                tracker.record("gemini-2.5-pro", prompt_tokens=100, completion_tokens=50)

        threads = [threading.Thread(target=record_many) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected_count = num_threads * records_per_thread
        assert tracker.request_count == expected_count
        assert tracker.summary()["total_prompt_tokens"] == 100 * expected_count
        assert tracker.summary()["total_completion_tokens"] == 50 * expected_count

    def test_concurrent_read_and_write(self) -> None:
        """Reading totals while recording should not crash."""
        tracker = CostTracker()
        errors: list[Exception] = []

        def writer() -> None:
            for _ in range(200):
                tracker.record("gemini-2.5-pro", prompt_tokens=10, completion_tokens=5)

        def reader() -> None:
            try:
                for _ in range(200):
                    _ = tracker.total_cost
                    _ = tracker.total_tokens
                    _ = tracker.summary()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == []


# ══════════════════════════════════════════════════════════════
# BudgetStatus enum
# ══════════════════════════════════════════════════════════════


class TestBudgetStatus:
    """Tests for BudgetStatus enum values."""

    def test_ok_value(self) -> None:
        assert BudgetStatus.OK == "ok"

    def test_warning_value(self) -> None:
        assert BudgetStatus.WARNING == "warning"

    def test_exceeded_value(self) -> None:
        assert BudgetStatus.EXCEEDED == "exceeded"

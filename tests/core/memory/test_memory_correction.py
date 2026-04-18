"""Tests for memory_correction module (SPEC-MEM-05)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from vaig.core.memory.memory_correction import (
    MemoryWarning,
    check_memory_before_action,
    compute_action_fingerprint,
)
from vaig.core.memory.models import FixOutcome, PatternEntry

# MemoryWarning uses TYPE_CHECKING imports for its field types; rebuild so
# Pydantic can validate instances at runtime inside tests.
MemoryWarning.model_rebuild()


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_pattern(fingerprint: str) -> PatternEntry:
    now = datetime.now(UTC)
    return PatternEntry(
        fingerprint=fingerprint,
        first_seen=now,
        last_seen=now,
        occurrences=1,
        title="test-pattern",
    )


def _make_outcome(
    fix_id: str,
    fingerprint: str,
    outcome: str = "worsened",
) -> FixOutcome:
    return FixOutcome(
        fix_id=fix_id,
        fingerprint=fingerprint,
        strategy="restart-pod",
        applied_at=datetime.now(UTC),
        outcome=outcome,  # type: ignore[arg-type]
    )


def _make_pattern_store(fingerprint: str | None) -> MagicMock:
    """Return a mock PatternMemoryStore.lookup that returns a pattern or None."""
    store = MagicMock()
    if fingerprint is not None:
        store.lookup.return_value = _make_pattern(fingerprint)
    else:
        store.lookup.return_value = None
    return store


def _make_fix_store(outcomes: list[FixOutcome]) -> MagicMock:
    """Return a mock FixOutcomeStore._ensure_index that returns {fix_id: outcome}."""
    store = MagicMock()
    store._ensure_index.return_value = {o.fix_id: o for o in outcomes}
    return store


# ── compute_action_fingerprint ────────────────────────────────────────────


class TestComputeActionFingerprint:
    __test__ = True

    def test_returns_16_hex_chars(self) -> None:
        fp = compute_action_fingerprint("kubectl_describe", "pod/web-abc", "oom-kill")
        assert len(fp) == 16
        assert all(c in "0123456789abcdef" for c in fp)

    def test_deterministic(self) -> None:
        fp1 = compute_action_fingerprint("kubectl_logs", "pod/api-xyz", "high-latency")
        fp2 = compute_action_fingerprint("kubectl_logs", "pod/api-xyz", "high-latency")
        assert fp1 == fp2

    def test_different_hypothesis_slug_gives_different_fingerprint(self) -> None:
        """Same tool + target but different hypothesis → different fingerprint."""
        fp_oom = compute_action_fingerprint("kubectl_describe", "pod/web-abc", "oom-kill")
        fp_latency = compute_action_fingerprint("kubectl_describe", "pod/web-abc", "high-latency")
        assert fp_oom != fp_latency

    def test_different_tool_gives_different_fingerprint(self) -> None:
        fp1 = compute_action_fingerprint("kubectl_describe", "pod/web", "slug")
        fp2 = compute_action_fingerprint("kubectl_logs", "pod/web", "slug")
        assert fp1 != fp2

    def test_different_target_gives_different_fingerprint(self) -> None:
        fp1 = compute_action_fingerprint("kubectl_describe", "pod/web-aaa", "slug")
        fp2 = compute_action_fingerprint("kubectl_describe", "pod/web-bbb", "slug")
        assert fp1 != fp2


# ── check_memory_before_action ────────────────────────────────────────────


class TestCheckMemoryBeforeAction:
    __test__ = True

    def test_returns_none_when_no_pattern(self) -> None:
        """No entry in pattern_store → no warning."""
        pattern_store = _make_pattern_store(fingerprint=None)
        fix_store = _make_fix_store([])

        result = check_memory_before_action(
            fingerprint="abcd1234abcd1234",
            proposed_tool="kubectl_describe",
            proposed_args={"target": "pod/web"},
            pattern_store=pattern_store,
            fix_store=fix_store,
        )
        assert result is None

    def test_returns_none_when_pattern_found_but_no_failure_outcome(self) -> None:
        """Pattern found, but the correlated outcome is 'resolved' → no warning."""
        fp = "abcd1234abcd1234"
        pattern_store = _make_pattern_store(fingerprint=fp)
        outcome = _make_outcome(fix_id="fix-1", fingerprint=fp, outcome="resolved")
        fix_store = _make_fix_store([outcome])

        result = check_memory_before_action(
            fingerprint=fp,
            proposed_tool="kubectl_describe",
            proposed_args={"target": "pod/web"},
            pattern_store=pattern_store,
            fix_store=fix_store,
        )
        assert result is None

    def test_returns_warning_for_worsened_outcome(self) -> None:
        """Pattern + 'worsened' outcome → MemoryWarning returned."""
        fp = "deadbeefdeadbeef"
        pattern_store = _make_pattern_store(fingerprint=fp)
        outcome = _make_outcome(fix_id="fix-2", fingerprint=fp, outcome="worsened")
        fix_store = _make_fix_store([outcome])

        result = check_memory_before_action(
            fingerprint=fp,
            proposed_tool="kubectl_top",
            proposed_args={"target": "pod/web"},
            pattern_store=pattern_store,
            fix_store=fix_store,
        )

        assert isinstance(result, MemoryWarning)
        assert result.past_outcome.outcome == "worsened"
        assert result.past_pattern.fingerprint == fp
        assert "kubectl_top" in result.suggestion
        assert fp in result.suggestion

    def test_returns_warning_for_persisted_outcome(self) -> None:
        """Pattern + 'persisted' outcome → MemoryWarning returned."""
        fp = "cafe0000cafe0000"
        pattern_store = _make_pattern_store(fingerprint=fp)
        outcome = _make_outcome(fix_id="fix-3", fingerprint=fp, outcome="persisted")
        fix_store = _make_fix_store([outcome])

        result = check_memory_before_action(
            fingerprint=fp,
            proposed_tool="kubectl_logs",
            proposed_args={"target": "pod/api"},
            pattern_store=pattern_store,
            fix_store=fix_store,
        )

        assert isinstance(result, MemoryWarning)
        assert result.past_outcome.outcome == "persisted"

    def test_returns_none_when_pattern_found_but_outcome_fingerprint_mismatch(self) -> None:
        """Outcome in fix_store has a different fingerprint → no warning."""
        fp_pattern = "aaaa0000aaaa0000"
        fp_outcome = "bbbb1111bbbb1111"
        pattern_store = _make_pattern_store(fingerprint=fp_pattern)
        outcome = _make_outcome(fix_id="fix-4", fingerprint=fp_outcome, outcome="worsened")
        fix_store = _make_fix_store([outcome])

        result = check_memory_before_action(
            fingerprint=fp_pattern,
            proposed_tool="kubectl_describe",
            proposed_args={},
            pattern_store=pattern_store,
            fix_store=fix_store,
        )
        assert result is None

    def test_returns_none_when_fix_store_raises(self) -> None:
        """Any exception from the stores → error-silent, returns None."""
        pattern_store = MagicMock()
        pattern_store.lookup.side_effect = RuntimeError("I/O error")
        fix_store = _make_fix_store([])

        result = check_memory_before_action(
            fingerprint="abcd1234abcd1234",
            proposed_tool="kubectl_describe",
            proposed_args={},
            pattern_store=pattern_store,
            fix_store=fix_store,
        )
        assert result is None

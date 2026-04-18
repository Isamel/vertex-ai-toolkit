"""Tests for SelfCorrectionController (SPEC-SH-06)."""

from __future__ import annotations

from vaig.core.config import SelfCorrectionConfig
from vaig.core.evidence_ledger import EvidenceEntry, EvidenceLedger, new_ledger
from vaig.core.self_correction import SelfCorrectionAction, SelfCorrectionController

# ── Fixtures ──────────────────────────────────────────────────────────────

def _make_controller(**kwargs) -> SelfCorrectionController:
    """Create a controller with default config, optionally overriding fields."""
    config = SelfCorrectionConfig(enabled=True, **kwargs)
    return SelfCorrectionController(config)


def _ledger_with_repeated_calls(tool_name: str, args_hash: str, count: int) -> EvidenceLedger:
    """Build a ledger with *count* identical (tool_name, args_hash) entries."""
    ledger = new_ledger()
    for _ in range(count):
        entry = EvidenceEntry(
            tool_name=tool_name,
            tool_args_hash=args_hash,
            question="test",
            answer_summary="result",
        )
        ledger = ledger.append(entry)
    return ledger


def _ledger_with_contradiction(claim: str) -> EvidenceLedger:
    """Build a ledger with one entry supporting *claim* and one contradicting it."""
    ledger = new_ledger()
    supporter = EvidenceEntry(
        tool_name="tool_a",
        tool_args_hash="aabbccdd",
        question="q1",
        answer_summary="supports claim",
        supports=(claim,),
    )
    contradictor = EvidenceEntry(
        tool_name="tool_b",
        tool_args_hash="eeff0011",
        question="q2",
        answer_summary="contradicts claim",
        contradicts=(claim,),
    )
    ledger = ledger.append(supporter)
    ledger = ledger.append(contradictor)
    return ledger


# ── check_circles ──────────────────────────────────────────────────────────

class TestCheckCircles:
    def test_no_circles_empty_ledger(self):
        ctrl = _make_controller(max_repeated_calls=3)
        assert ctrl.check_circles(new_ledger()) == []

    def test_no_circles_below_threshold(self):
        ctrl = _make_controller(max_repeated_calls=3)
        ledger = _ledger_with_repeated_calls("kubectl_describe", "aaaa1111", 2)
        assert ctrl.check_circles(ledger) == []

    def test_circle_at_threshold(self):
        ctrl = _make_controller(max_repeated_calls=3)
        ledger = _ledger_with_repeated_calls("kubectl_describe", "aaaa1111", 3)
        circles = ctrl.check_circles(ledger)
        assert len(circles) == 1
        assert "kubectl_describe" in circles[0]
        assert "aaaa1111" in circles[0]

    def test_circle_above_threshold(self):
        ctrl = _make_controller(max_repeated_calls=2)
        ledger = _ledger_with_repeated_calls("kubectl_logs", "bbbb2222", 5)
        circles = ctrl.check_circles(ledger)
        assert len(circles) == 1

    def test_multiple_distinct_tools_no_circle(self):
        ctrl = _make_controller(max_repeated_calls=3)
        ledger = new_ledger()
        for i in range(3):
            entry = EvidenceEntry(
                tool_name=f"tool_{i}",
                tool_args_hash="cccc3333",
                question="q",
                answer_summary="a",
            )
            ledger = ledger.append(entry)
        # Each tool called once — no circles
        assert ctrl.check_circles(ledger) == []


# ── check_contradictions ──────────────────────────────────────────────────

class TestCheckContradictions:
    def test_no_contradictions_empty_ledger(self):
        ctrl = _make_controller()
        assert ctrl.check_contradictions(new_ledger()) == []

    def test_no_contradictions_supports_only(self):
        ctrl = _make_controller()
        ledger = new_ledger().append(
            EvidenceEntry(
                tool_name="t",
                tool_args_hash="0000",
                question="q",
                answer_summary="a",
                supports=("claim_x",),
            )
        )
        assert ctrl.check_contradictions(ledger) == []

    def test_contradiction_detected(self):
        ctrl = _make_controller(contradiction_sensitivity=0.8)
        ledger = _ledger_with_contradiction("oom-kill-root-cause")
        pairs = ctrl.check_contradictions(ledger)
        assert len(pairs) == 1
        assert isinstance(pairs[0], tuple)
        assert len(pairs[0]) == 2

    def test_zero_sensitivity_disables_detection(self):
        ctrl = _make_controller(contradiction_sensitivity=0.0)
        ledger = _ledger_with_contradiction("claim_y")
        assert ctrl.check_contradictions(ledger) == []


# ── check_stale ────────────────────────────────────────────────────────────

class TestCheckStale:
    def test_not_stale_below_threshold(self):
        ctrl = _make_controller(max_stale_iterations=5)
        assert ctrl.check_stale(4) is False

    def test_stale_at_threshold(self):
        ctrl = _make_controller(max_stale_iterations=5)
        assert ctrl.check_stale(5) is True

    def test_stale_above_threshold(self):
        ctrl = _make_controller(max_stale_iterations=3)
        assert ctrl.check_stale(10) is True

    def test_zero_iterations_not_stale(self):
        ctrl = _make_controller(max_stale_iterations=5)
        assert ctrl.check_stale(0) is False


# ── decide ─────────────────────────────────────────────────────────────────

class TestDecide:
    """SH-06 scenario tests — priority order: circle > contradiction > stale > clean."""

    def test_circle_returns_backtrack(self):
        """Scenario: Circle detection triggers BACKTRACK."""
        ctrl = _make_controller(max_repeated_calls=3)
        ledger = _ledger_with_repeated_calls("kubectl_describe", "aaaa1111", 3)
        assert ctrl.decide(ledger, 0) == SelfCorrectionAction.backtrack

    def test_contradiction_returns_escalate(self):
        """Scenario: Contradiction triggers ESCALATE."""
        ctrl = _make_controller(max_repeated_calls=10)  # Prevent circle from firing
        ledger = _ledger_with_contradiction("service-down-claim")
        assert ctrl.decide(ledger, 0) == SelfCorrectionAction.escalate

    def test_stale_returns_force_different(self):
        """Scenario: Stale iterations trigger FORCE_DIFFERENT."""
        ctrl = _make_controller(max_repeated_calls=10, max_stale_iterations=5)
        # Clean ledger (no circles, no contradictions), but stale
        assert ctrl.decide(new_ledger(), 6) == SelfCorrectionAction.force_different

    def test_clean_returns_continue(self):
        """Scenario: Clean ledger returns CONTINUE."""
        ctrl = _make_controller(max_repeated_calls=3, max_stale_iterations=5)
        ledger = new_ledger()
        ledger = ledger.append(EvidenceEntry(tool_name="t1", tool_args_hash="0001", question="q1", answer_summary="a1"))
        ledger = ledger.append(EvidenceEntry(tool_name="t2", tool_args_hash="0002", question="q2", answer_summary="a2"))
        assert ctrl.decide(ledger, 0) == SelfCorrectionAction.continue_

    def test_circle_wins_over_stale(self):
        """Circle detection has higher priority than stale."""
        ctrl = _make_controller(max_repeated_calls=2, max_stale_iterations=1)
        ledger = _ledger_with_repeated_calls("tool_x", "ffff", 3)
        # Both circle AND stale should be true — circle wins
        assert ctrl.decide(ledger, 10) == SelfCorrectionAction.backtrack

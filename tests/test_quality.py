"""Tests for SPEC-RATE-05: QualityIssue and RunQualityCollector.

Scenarios:
  - QualityIssue is frozen (dataclass frozen=True): mutating raises
  - QualityIssue rejects extra fields (it's a dataclass, not pydantic — no __dict__ tricks)
  - RunQualityCollector.record() deduplicates by (kind, where)
  - RunQualityCollector.issues returns a copy (not internal list)
  - RunQualityCollector __bool__/__len__ / has_kind()
"""

from __future__ import annotations

import pytest

from vaig.core.quality import QualityIssue, QualityIssueKind, RunQualityCollector

# ── QualityIssue ──────────────────────────────────────────────


class TestQualityIssue:
    """Unit tests for the QualityIssue frozen dataclass."""

    def test_creation_with_required_fields(self) -> None:
        """QualityIssue can be created with kind and where."""
        issue = QualityIssue(kind=QualityIssueKind.MODEL_DEGRADED, where="agent-alpha")
        assert issue.kind is QualityIssueKind.MODEL_DEGRADED
        assert issue.where == "agent-alpha"
        assert issue.detail == ""

    def test_creation_with_detail(self) -> None:
        """QualityIssue stores optional detail field."""
        issue = QualityIssue(
            kind=QualityIssueKind.AGENT_FAILED,
            where="agent-beta",
            detail="timeout after 60s",
        )
        assert issue.detail == "timeout after 60s"

    def test_is_frozen_raises_on_attribute_mutation(self) -> None:
        """Mutating a frozen QualityIssue raises FrozenInstanceError."""
        issue = QualityIssue(kind=QualityIssueKind.TOOL_ERROR, where="tool-x")
        with pytest.raises((AttributeError, TypeError)):
            issue.where = "something-else"  # type: ignore[misc]

    def test_is_frozen_raises_on_kind_mutation(self) -> None:
        """Mutating kind raises FrozenInstanceError."""
        issue = QualityIssue(kind=QualityIssueKind.CONTEXT_TRUNCATED, where="gather-phase")
        with pytest.raises((AttributeError, TypeError)):
            issue.kind = QualityIssueKind.AGENT_FAILED  # type: ignore[misc]

    def test_is_hashable(self) -> None:
        """Frozen dataclass instances must be hashable (usable in sets)."""
        issue = QualityIssue(kind=QualityIssueKind.BUDGET_EXCEEDED, where="global")
        assert hash(issue) is not None
        s = {issue}
        assert len(s) == 1

    def test_equality_by_value(self) -> None:
        """Two QualityIssue instances with same values are equal."""
        a = QualityIssue(kind=QualityIssueKind.MODEL_DEGRADED, where="x")
        b = QualityIssue(kind=QualityIssueKind.MODEL_DEGRADED, where="x")
        assert a == b

    def test_inequality_different_where(self) -> None:
        """QualityIssue instances differ when where differs."""
        a = QualityIssue(kind=QualityIssueKind.MODEL_DEGRADED, where="x")
        b = QualityIssue(kind=QualityIssueKind.MODEL_DEGRADED, where="y")
        assert a != b

    def test_all_kind_values_constructible(self) -> None:
        """All QualityIssueKind values can be used to construct a QualityIssue."""
        for kind in QualityIssueKind:
            issue = QualityIssue(kind=kind, where="test")
            assert issue.kind is kind


# ── QualityIssueKind ──────────────────────────────────────────


class TestQualityIssueKind:
    """Tests for the QualityIssueKind StrEnum."""

    def test_is_str_enum(self) -> None:
        """QualityIssueKind values behave as strings."""
        assert QualityIssueKind.MODEL_DEGRADED == "model_degraded"
        assert QualityIssueKind.AGENT_FAILED == "agent_failed"

    def test_expected_members_present(self) -> None:
        """All expected kind values exist in the enum."""
        expected = {"model_degraded", "budget_exceeded", "agent_failed", "tool_error", "context_truncated"}
        actual = {k.value for k in QualityIssueKind}
        assert expected == actual


# ── RunQualityCollector ───────────────────────────────────────


class TestRunQualityCollector:
    """Unit tests for RunQualityCollector."""

    def test_empty_on_construction(self) -> None:
        """New collector has no issues."""
        collector = RunQualityCollector()
        assert collector.issues == []
        assert len(collector) == 0
        assert not collector

    def test_has_kind_false_on_empty(self) -> None:
        """has_kind() returns False for any kind on empty collector."""
        collector = RunQualityCollector()
        assert collector.has_kind(QualityIssueKind.MODEL_DEGRADED) is False

    def test_record_adds_issue(self) -> None:
        """record() adds a QualityIssue to the collector."""
        collector = RunQualityCollector()
        issue = QualityIssue(kind=QualityIssueKind.MODEL_DEGRADED, where="agent-a")
        collector.record(issue)
        assert len(collector) == 1
        assert collector.issues[0] == issue

    def test_record_deduplicates_by_kind_and_where(self) -> None:
        """Adding the same (kind, where) twice → only one issue stored."""
        collector = RunQualityCollector()
        issue1 = QualityIssue(kind=QualityIssueKind.AGENT_FAILED, where="agent-x")
        issue2 = QualityIssue(kind=QualityIssueKind.AGENT_FAILED, where="agent-x", detail="different detail")
        collector.record(issue1)
        collector.record(issue2)
        assert len(collector) == 1
        # First one wins
        assert collector.issues[0].detail == ""

    def test_record_different_where_are_distinct(self) -> None:
        """Same kind but different where → two separate issues."""
        collector = RunQualityCollector()
        collector.record(QualityIssue(kind=QualityIssueKind.TOOL_ERROR, where="tool-a"))
        collector.record(QualityIssue(kind=QualityIssueKind.TOOL_ERROR, where="tool-b"))
        assert len(collector) == 2

    def test_record_different_kind_same_where_are_distinct(self) -> None:
        """Different kinds at same location → two separate issues."""
        collector = RunQualityCollector()
        collector.record(QualityIssue(kind=QualityIssueKind.MODEL_DEGRADED, where="agent-a"))
        collector.record(QualityIssue(kind=QualityIssueKind.AGENT_FAILED, where="agent-a"))
        assert len(collector) == 2

    def test_issues_returns_copy(self) -> None:
        """issues property returns a copy — mutating it does not affect collector."""
        collector = RunQualityCollector()
        collector.record(QualityIssue(kind=QualityIssueKind.MODEL_DEGRADED, where="x"))
        copy = collector.issues
        copy.clear()
        # Original should still have 1 issue
        assert len(collector) == 1

    def test_bool_false_empty(self) -> None:
        """Empty collector is falsy."""
        assert not RunQualityCollector()

    def test_bool_true_after_record(self) -> None:
        """Collector is truthy after at least one issue."""
        collector = RunQualityCollector()
        collector.record(QualityIssue(kind=QualityIssueKind.CONTEXT_TRUNCATED, where="attach"))
        assert bool(collector) is True

    def test_has_kind_true_after_adding(self) -> None:
        """has_kind() returns True for a kind that was recorded."""
        collector = RunQualityCollector()
        collector.record(QualityIssue(kind=QualityIssueKind.BUDGET_EXCEEDED, where="run"))
        assert collector.has_kind(QualityIssueKind.BUDGET_EXCEEDED) is True
        assert collector.has_kind(QualityIssueKind.AGENT_FAILED) is False

    def test_record_kind_convenience_wrapper(self) -> None:
        """record_kind() constructs a QualityIssue and records it."""
        collector = RunQualityCollector()
        collector.record_kind(QualityIssueKind.TOOL_ERROR, "tool-y", detail="timed out")
        assert len(collector) == 1
        assert collector.issues[0].kind is QualityIssueKind.TOOL_ERROR
        assert collector.issues[0].where == "tool-y"
        assert collector.issues[0].detail == "timed out"

    def test_merge_adds_issues_from_other(self) -> None:
        """merge() imports all issues from another collector (dedup preserved)."""
        c1 = RunQualityCollector()
        c1.record(QualityIssue(kind=QualityIssueKind.MODEL_DEGRADED, where="a1"))

        c2 = RunQualityCollector()
        c2.record(QualityIssue(kind=QualityIssueKind.AGENT_FAILED, where="a2"))
        c2.record(QualityIssue(kind=QualityIssueKind.MODEL_DEGRADED, where="a1"))  # dup

        c1.merge(c2)
        # a1 model_degraded is a dup — only 2 unique issues
        assert len(c1) == 2

    def test_insertion_order_preserved(self) -> None:
        """Issues are returned in insertion order."""
        collector = RunQualityCollector()
        collector.record(QualityIssue(kind=QualityIssueKind.AGENT_FAILED, where="first"))
        collector.record(QualityIssue(kind=QualityIssueKind.TOOL_ERROR, where="second"))
        collector.record(QualityIssue(kind=QualityIssueKind.MODEL_DEGRADED, where="third"))
        names = [i.where for i in collector.issues]
        assert names == ["first", "second", "third"]

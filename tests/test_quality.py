"""Tests for SPEC-RATE-05: QualityIssue and RunQualityCollector.

Scenarios:
  - QualityIssue is frozen (Pydantic BaseModel frozen=True): mutating raises
  - QualityIssue rejects extra fields (extra="forbid")
  - RunQualityCollector.record() deduplicates by (kind, where)
  - RunQualityCollector.issues returns a copy (not internal list)
  - RunQualityCollector __bool__/__len__ / has_kind()
"""

from __future__ import annotations

import pytest

from vaig.core.quality import QualityIssue, QualityIssueKind, RunQualityCollector

# ── QualityIssue ──────────────────────────────────────────────


class TestQualityIssue:
    """Unit tests for the QualityIssue frozen Pydantic BaseModel."""

    def test_creation_with_required_fields(self) -> None:
        """QualityIssue can be created with kind and where."""
        issue = QualityIssue(kind=QualityIssueKind.model_degraded, where="agent-alpha")
        assert issue.kind is QualityIssueKind.model_degraded
        assert issue.where == "agent-alpha"
        assert issue.consequence == ""

    def test_creation_with_consequence(self) -> None:
        """QualityIssue stores optional consequence field."""
        issue = QualityIssue(
            kind=QualityIssueKind.agent_failed,
            where="agent-beta",
            consequence="timeout after 60s",
        )
        assert issue.consequence == "timeout after 60s"

    def test_is_frozen_raises_on_attribute_mutation(self) -> None:
        """Mutating a frozen QualityIssue raises ValidationError or TypeError."""
        issue = QualityIssue(kind=QualityIssueKind.circuit_breaker_tripped, where="tool-x")
        with pytest.raises((AttributeError, TypeError, Exception)):
            issue.where = "something-else"  # type: ignore[misc]

    def test_is_frozen_raises_on_kind_mutation(self) -> None:
        """Mutating kind raises on frozen model."""
        issue = QualityIssue(kind=QualityIssueKind.attachment_truncated, where="gather-phase")
        with pytest.raises((AttributeError, TypeError, Exception)):
            issue.kind = QualityIssueKind.agent_failed  # type: ignore[misc]

    def test_is_hashable(self) -> None:
        """Frozen Pydantic BaseModel instances must be hashable (usable in sets)."""
        issue = QualityIssue(kind=QualityIssueKind.incomplete_gather, where="global")
        assert hash(issue) is not None
        s = {issue}
        assert len(s) == 1

    def test_equality_by_value(self) -> None:
        """Two QualityIssue instances with same values are equal."""
        a = QualityIssue(kind=QualityIssueKind.model_degraded, where="x")
        b = QualityIssue(kind=QualityIssueKind.model_degraded, where="x")
        assert a == b

    def test_inequality_different_where(self) -> None:
        """QualityIssue instances differ when where differs."""
        a = QualityIssue(kind=QualityIssueKind.model_degraded, where="x")
        b = QualityIssue(kind=QualityIssueKind.model_degraded, where="y")
        assert a != b

    def test_all_kind_values_constructible(self) -> None:
        """All QualityIssueKind values can be used to construct a QualityIssue."""
        for kind in QualityIssueKind:
            issue = QualityIssue(kind=kind, where="test")
            assert issue.kind is kind

    def test_extra_fields_forbidden(self) -> None:
        """QualityIssue rejects unknown fields (extra='forbid')."""
        with pytest.raises(Exception):
            QualityIssue(kind=QualityIssueKind.agent_failed, where="x", unknown_field="bad")  # type: ignore[call-arg]


# ── QualityIssueKind ──────────────────────────────────────────


class TestQualityIssueKind:
    """Tests for the QualityIssueKind StrEnum."""

    def test_is_str_enum(self) -> None:
        """QualityIssueKind values behave as strings."""
        assert QualityIssueKind.model_degraded == "model_degraded"
        assert QualityIssueKind.agent_failed == "agent_failed"

    def test_expected_members_present(self) -> None:
        """All expected kind values exist in the enum."""
        expected = {
            "agent_failed",
            "model_degraded",
            "circuit_breaker_tripped",
            "incomplete_gather",
            "enrichment_timeout",
            "attachment_truncated",
        }
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
        assert collector.has_kind(QualityIssueKind.model_degraded) is False

    def test_record_adds_issue(self) -> None:
        """record() adds a QualityIssue to the collector."""
        collector = RunQualityCollector()
        issue = QualityIssue(kind=QualityIssueKind.model_degraded, where="agent-a")
        collector.record(issue)
        assert len(collector) == 1
        assert collector.issues[0] == issue

    def test_record_deduplicates_by_kind_and_where(self) -> None:
        """Adding the same (kind, where) twice → only one issue stored."""
        collector = RunQualityCollector()
        issue1 = QualityIssue(kind=QualityIssueKind.agent_failed, where="agent-x")
        issue2 = QualityIssue(kind=QualityIssueKind.agent_failed, where="agent-x", consequence="different")
        collector.record(issue1)
        collector.record(issue2)
        assert len(collector) == 1
        # First one wins
        assert collector.issues[0].consequence == ""

    def test_record_different_where_are_distinct(self) -> None:
        """Same kind but different where → two separate issues."""
        collector = RunQualityCollector()
        collector.record(QualityIssue(kind=QualityIssueKind.circuit_breaker_tripped, where="tool-a"))
        collector.record(QualityIssue(kind=QualityIssueKind.circuit_breaker_tripped, where="tool-b"))
        assert len(collector) == 2

    def test_record_different_kind_same_where_are_distinct(self) -> None:
        """Different kinds at same location → two separate issues."""
        collector = RunQualityCollector()
        collector.record(QualityIssue(kind=QualityIssueKind.model_degraded, where="agent-a"))
        collector.record(QualityIssue(kind=QualityIssueKind.agent_failed, where="agent-a"))
        assert len(collector) == 2

    def test_issues_returns_copy(self) -> None:
        """issues property returns a copy — mutating it does not affect collector."""
        collector = RunQualityCollector()
        collector.record(QualityIssue(kind=QualityIssueKind.model_degraded, where="x"))
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
        collector.record(QualityIssue(kind=QualityIssueKind.attachment_truncated, where="attach"))
        assert bool(collector) is True

    def test_has_kind_true_after_adding(self) -> None:
        """has_kind() returns True for a kind that was recorded."""
        collector = RunQualityCollector()
        collector.record(QualityIssue(kind=QualityIssueKind.incomplete_gather, where="run"))
        assert collector.has_kind(QualityIssueKind.incomplete_gather) is True
        assert collector.has_kind(QualityIssueKind.agent_failed) is False

    def test_record_kind_convenience_wrapper(self) -> None:
        """record_kind() constructs a QualityIssue and records it."""
        collector = RunQualityCollector()
        collector.record_kind(QualityIssueKind.enrichment_timeout, "tool-y", consequence="timed out")
        assert len(collector) == 1
        assert collector.issues[0].kind is QualityIssueKind.enrichment_timeout
        assert collector.issues[0].where == "tool-y"
        assert collector.issues[0].consequence == "timed out"

    def test_merge_adds_issues_from_other(self) -> None:
        """merge() imports all issues from another collector (dedup preserved)."""
        c1 = RunQualityCollector()
        c1.record(QualityIssue(kind=QualityIssueKind.model_degraded, where="a1"))

        c2 = RunQualityCollector()
        c2.record(QualityIssue(kind=QualityIssueKind.agent_failed, where="a2"))
        c2.record(QualityIssue(kind=QualityIssueKind.model_degraded, where="a1"))  # dup

        c1.merge(c2)
        # a1 model_degraded is a dup — only 2 unique issues
        assert len(c1) == 2

    def test_insertion_order_preserved(self) -> None:
        """Issues are returned in insertion order."""
        collector = RunQualityCollector()
        collector.record(QualityIssue(kind=QualityIssueKind.agent_failed, where="first"))
        collector.record(QualityIssue(kind=QualityIssueKind.circuit_breaker_tripped, where="second"))
        collector.record(QualityIssue(kind=QualityIssueKind.model_degraded, where="third"))
        names = [i.where for i in collector.issues]
        assert names == ["first", "second", "third"]

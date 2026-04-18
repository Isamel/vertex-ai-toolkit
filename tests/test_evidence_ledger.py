"""Unit tests for EvidenceLedger and EvidenceEntry (T-09)."""

from __future__ import annotations

import pytest

from vaig.core.evidence_ledger import EvidenceEntry, new_ledger

# ══════════════════════════════════════════════════════════════
# EvidenceEntry Tests
# ══════════════════════════════════════════════════════════════


class TestEvidenceEntryCreation:
    def test_defaults(self) -> None:
        entry = EvidenceEntry()
        assert entry.id != ""
        assert entry.timestamp != ""
        assert entry.source_agent == ""
        assert entry.tool_name == ""
        assert entry.tool_args_hash == ""
        assert entry.question == ""
        assert entry.answer_summary == ""
        assert entry.raw_output_ref == ""
        assert entry.supports == ()
        assert entry.contradicts == ()

    def test_all_fields_set(self) -> None:
        entry = EvidenceEntry(
            source_agent="agent-1",
            tool_name="kubectl_get_pods",
            tool_args_hash="abcd1234abcd1234",
            question="Are pods running?",
            answer_summary="All pods running.",
            raw_output_ref="run-abc.jsonl",
            supports=("pods are healthy",),
            contradicts=("cluster is down",),
        )
        assert entry.source_agent == "agent-1"
        assert entry.tool_name == "kubectl_get_pods"
        assert entry.tool_args_hash == "abcd1234abcd1234"
        assert entry.question == "Are pods running?"
        assert entry.answer_summary == "All pods running."
        assert entry.raw_output_ref == "run-abc.jsonl"
        assert entry.supports == ("pods are healthy",)
        assert entry.contradicts == ("cluster is down",)

    def test_immutable(self) -> None:
        entry = EvidenceEntry(tool_name="my_tool")
        with pytest.raises(Exception):
            entry.tool_name = "other"  # type: ignore[misc]

    def test_auto_id_is_uuid_hex(self) -> None:
        entry = EvidenceEntry()
        assert len(entry.id) == 32
        assert entry.id.isalnum()

    def test_auto_timestamp_is_iso(self) -> None:
        entry = EvidenceEntry()
        assert "T" in entry.timestamp

    def test_answer_summary_truncated_at_500(self) -> None:
        long_text = "x" * 600
        entry = EvidenceEntry(answer_summary=long_text)
        assert len(entry.answer_summary) == 500

    def test_answer_summary_exact_500_not_truncated(self) -> None:
        text = "y" * 500
        entry = EvidenceEntry(answer_summary=text)
        assert len(entry.answer_summary) == 500

    def test_answer_summary_under_500_not_truncated(self) -> None:
        text = "short answer"
        entry = EvidenceEntry(answer_summary=text)
        assert entry.answer_summary == "short answer"

    def test_supports_and_contradicts_are_tuples(self) -> None:
        entry = EvidenceEntry(supports=("a", "b"), contradicts=("c",))
        assert isinstance(entry.supports, tuple)
        assert isinstance(entry.contradicts, tuple)

    def test_unique_ids_per_entry(self) -> None:
        e1 = EvidenceEntry()
        e2 = EvidenceEntry()
        assert e1.id != e2.id


# ══════════════════════════════════════════════════════════════
# EvidenceLedger Tests
# ══════════════════════════════════════════════════════════════


class TestEvidenceLedgerEmpty:
    def test_empty_ledger_has_no_entries(self) -> None:
        ledger = new_ledger()
        assert ledger.entries == ()

    def test_search_empty_returns_empty(self) -> None:
        ledger = new_ledger()
        assert ledger.search("pods") == []

    def test_already_answered_empty_returns_empty(self) -> None:
        ledger = new_ledger()
        assert ledger.already_answered("any question") == []

    def test_search_empty_query_on_empty_ledger(self) -> None:
        ledger = new_ledger()
        assert ledger.search("") == []

    def test_immutable(self) -> None:
        ledger = new_ledger()
        with pytest.raises(Exception):
            ledger.entries = ()  # type: ignore[misc]


class TestEvidenceLedgerAppend:
    def test_append_returns_new_instance(self) -> None:
        ledger = new_ledger()
        entry = EvidenceEntry(tool_name="tool_a")
        new = ledger.append(entry)
        assert new is not ledger

    def test_append_does_not_modify_original(self) -> None:
        ledger = new_ledger()
        entry = EvidenceEntry(tool_name="tool_a")
        ledger.append(entry)
        assert len(ledger.entries) == 0

    def test_append_adds_entry(self) -> None:
        ledger = new_ledger()
        entry = EvidenceEntry(tool_name="tool_a")
        new = ledger.append(entry)
        assert len(new.entries) == 1
        assert new.entries[0] is entry

    def test_append_multiple_entries(self) -> None:
        ledger = new_ledger()
        e1 = EvidenceEntry(tool_name="tool_1")
        e2 = EvidenceEntry(tool_name="tool_2")
        ledger = ledger.append(e1).append(e2)
        assert len(ledger.entries) == 2
        assert ledger.entries[0].tool_name == "tool_1"
        assert ledger.entries[1].tool_name == "tool_2"

    def test_append_preserves_immutability(self) -> None:
        ledger = new_ledger()
        e1 = EvidenceEntry(tool_name="first")
        ledger2 = ledger.append(e1)
        e2 = EvidenceEntry(tool_name="second")
        ledger3 = ledger2.append(e2)
        # ledger2 unchanged
        assert len(ledger2.entries) == 1


class TestEvidenceLedgerSearch:
    def test_search_by_tool_name(self) -> None:
        ledger = new_ledger()
        entry = EvidenceEntry(tool_name="kubectl_get")
        ledger = ledger.append(entry)
        results = ledger.search("kubectl")
        assert len(results) == 1
        assert results[0].tool_name == "kubectl_get"

    def test_search_by_question(self) -> None:
        ledger = new_ledger()
        entry = EvidenceEntry(question="Are pods healthy?")
        ledger = ledger.append(entry)
        results = ledger.search("pods healthy")
        assert len(results) == 1

    def test_search_by_answer_summary(self) -> None:
        ledger = new_ledger()
        entry = EvidenceEntry(answer_summary="All nodes are operational.")
        ledger = ledger.append(entry)
        results = ledger.search("nodes are operational")
        assert len(results) == 1

    def test_search_case_insensitive(self) -> None:
        ledger = new_ledger()
        entry = EvidenceEntry(tool_name="KubectlGetPods")
        ledger = ledger.append(entry)
        assert len(ledger.search("kubectl")) == 1
        assert len(ledger.search("KUBECTL")) == 1
        assert len(ledger.search("kubectlgetpods")) == 1

    def test_search_no_match(self) -> None:
        ledger = new_ledger()
        entry = EvidenceEntry(tool_name="unrelated_tool")
        ledger = ledger.append(entry)
        results = ledger.search("kubernetes")
        assert results == []

    def test_search_empty_query_returns_all(self) -> None:
        ledger = new_ledger()
        e1 = EvidenceEntry(tool_name="a")
        e2 = EvidenceEntry(tool_name="b")
        ledger = ledger.append(e1).append(e2)
        assert len(ledger.search("")) == 2

    def test_search_returns_multiple_matches(self) -> None:
        ledger = new_ledger()
        e1 = EvidenceEntry(tool_name="get_pods", question="pods status")
        e2 = EvidenceEntry(tool_name="get_nodes", answer_summary="pods are 3/3")
        e3 = EvidenceEntry(tool_name="unrelated")
        ledger = ledger.append(e1).append(e2).append(e3)
        results = ledger.search("pods")
        assert len(results) == 2


class TestEvidenceLedgerAlreadyAnswered:
    def test_already_answered_returns_matching_entries(self) -> None:
        ledger = new_ledger()
        entry = EvidenceEntry(question="Is the cluster healthy?")
        ledger = ledger.append(entry)
        result = ledger.already_answered("cluster healthy")
        assert len(result) == 1

    def test_already_answered_returns_empty_when_no_match(self) -> None:
        ledger = new_ledger()
        entry = EvidenceEntry(question="CPU usage?")
        ledger = ledger.append(entry)
        result = ledger.already_answered("memory")
        assert result == []

    def test_already_answered_truthy_when_match_exists(self) -> None:
        ledger = new_ledger()
        entry = EvidenceEntry(question="Are pods running?")
        ledger = ledger.append(entry)
        assert ledger.already_answered("pods running")

    def test_already_answered_falsy_when_no_match(self) -> None:
        ledger = new_ledger()
        entry = EvidenceEntry(question="CPU usage?")
        ledger = ledger.append(entry)
        assert not ledger.already_answered("memory")


# ══════════════════════════════════════════════════════════════
# Performance Test
# ══════════════════════════════════════════════════════════════


class TestEvidenceLedgerPerformance:
    def test_large_ledger_100_entries(self) -> None:
        ledger = new_ledger()
        for i in range(100):
            entry = EvidenceEntry(
                tool_name=f"tool_{i}",
                question=f"question {i}",
                answer_summary=f"answer {i}",
            )
            ledger = ledger.append(entry)
        assert len(ledger.entries) == 100

    def test_search_on_large_ledger(self) -> None:
        ledger = new_ledger()
        for i in range(100):
            entry = EvidenceEntry(
                tool_name=f"tool_{i}",
                question=f"question {i}",
                answer_summary=f"answer {i}",
            )
            ledger = ledger.append(entry)
        # Should find exactly one entry with "tool_42"
        results = ledger.search("tool_42")
        assert len(results) == 1
        assert results[0].tool_name == "tool_42"

    def test_already_answered_on_large_ledger(self) -> None:
        ledger = new_ledger()
        for i in range(100):
            entry = EvidenceEntry(question=f"question {i}")
            ledger = ledger.append(entry)
        assert ledger.already_answered("question 99")
        assert not ledger.already_answered("question 200")


# ══════════════════════════════════════════════════════════════
# T-03: EvidenceLedger.to_summary() (SH-09)
# ══════════════════════════════════════════════════════════════


class TestEvidenceLedgerToSummary:
    """Tests for EvidenceLedger.to_summary() added in T-03 (SH-09)."""

    def test_empty_ledger_returns_empty_string(self) -> None:
        ledger = new_ledger()
        assert ledger.to_summary() == ""

    def test_single_entry_format(self) -> None:
        entry = EvidenceEntry(
            tool_name="kubectl_get_pods",
            question="Are pods running?",
            answer_summary="All pods running.",
        )
        ledger = new_ledger().append(entry)
        summary = ledger.to_summary()
        assert "kubectl_get_pods" in summary
        assert "Are pods running?" in summary
        assert "All pods running." in summary

    def test_entry_line_starts_with_dash_bracket(self) -> None:
        entry = EvidenceEntry(tool_name="my_tool", question="q", answer_summary="ans")
        ledger = new_ledger().append(entry)
        summary = ledger.to_summary()
        assert summary.startswith("- [my_tool]")

    def test_multiple_entries_produce_multiple_lines(self) -> None:
        ledger = new_ledger()
        for i in range(3):
            ledger = ledger.append(
                EvidenceEntry(tool_name=f"tool_{i}", question=f"q{i}", answer_summary=f"ans{i}")
            )
        lines = [ln for ln in ledger.to_summary().splitlines() if ln.strip()]
        assert len(lines) == 3

    def test_max_entries_limits_output(self) -> None:
        ledger = new_ledger()
        for i in range(20):
            ledger = ledger.append(
                EvidenceEntry(tool_name=f"t{i}", question=f"q{i}", answer_summary=f"a{i}")
            )
        summary = ledger.to_summary(max_entries=5)
        lines = [ln for ln in summary.splitlines() if ln.strip()]
        assert len(lines) == 5

    def test_answer_summary_truncated_at_100_chars(self) -> None:
        long_answer = "x" * 200
        entry = EvidenceEntry(tool_name="tool", question="q", answer_summary=long_answer)
        ledger = new_ledger().append(entry)
        summary = ledger.to_summary()
        # The answer in the summary line should not exceed 100 chars of original
        assert "x" * 101 not in summary

    def test_default_max_entries_is_ten(self) -> None:
        ledger = new_ledger()
        for i in range(15):
            ledger = ledger.append(
                EvidenceEntry(tool_name=f"t{i}", question=f"q{i}", answer_summary=f"a{i}")
            )
        lines = [ln for ln in ledger.to_summary().splitlines() if ln.strip()]
        assert len(lines) == 10

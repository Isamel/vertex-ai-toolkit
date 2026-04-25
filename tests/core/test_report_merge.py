"""Unit tests for vaig.core.report_merge (E1–E9).

Pure-Python deterministic merge — no LLM, no network, no fixtures from disk.
"""

from __future__ import annotations

from vaig.core.report_merge import (
    _collide_findings,
    _dedup_first,
    _longer_non_empty,
    _merge_evidence_gaps,
    _merge_findings,
    _merge_hypotheses,
    _merge_recommendations,
    _merge_timeline,
    _ordered_union,
    _prune_causal_edges,
    _rebuild_causal_graph,
    _rebuild_executive_summary,
    _slugify,
    _union,
    merge_health_reports,
)
from vaig.skills.service_health.schema import (
    EvidenceGap,
    ExecutiveSummary,
    Finding,
    HealthReport,
    OverallStatus,
    RecommendedAction,
    RootCauseHypothesis,
    Severity,
    TimelineEvent,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _exec_summary(status: OverallStatus = OverallStatus.HEALTHY) -> ExecutiveSummary:
    return ExecutiveSummary(
        overall_status=status,
        scope="Cluster-wide",
        summary_text="test summary",
    )


def _finding(
    finding_id: str,
    severity: Severity = Severity.MEDIUM,
    title: str = "",
    description: str = "",
    evidence: list[str] | None = None,
    affected_resources: list[str] | None = None,
    caused_by: list[str] | None = None,
    causes: list[str] | None = None,
) -> Finding:
    return Finding(
        id=finding_id,
        title=title or finding_id,
        severity=severity,
        description=description,
        evidence=evidence or [],
        affected_resources=affected_resources or [],
        caused_by=caused_by or [],
        causes=causes or [],
    )


def _bare_report(findings: list[Finding] | None = None) -> HealthReport:
    return HealthReport(
        executive_summary=_exec_summary(),
        findings=findings or [],
    )


# ── E1: _slugify ─────────────────────────────────────────────────────────────


class TestSlugify:
    def test_basic(self):
        assert _slugify("Crash-Loop: payment-svc!") == "crash-loop-payment-svc"

    def test_empty(self):
        assert _slugify("") == ""

    def test_all_special(self):
        assert _slugify("!!!") == ""

    def test_already_slug(self):
        assert _slugify("crash-loop") == "crash-loop"

    def test_whitespace(self):
        assert _slugify("  hello world  ") == "hello-world"


# ── E2: pure helpers ─────────────────────────────────────────────────────────


class TestPureHelpers:
    def test_longer_non_empty_prefers_longer(self):
        assert _longer_non_empty("hello world", "hi") == "hello world"

    def test_longer_non_empty_falls_back_when_a_empty(self):
        assert _longer_non_empty("", "fallback") == "fallback"

    def test_longer_non_empty_falls_back_when_b_empty(self):
        assert _longer_non_empty("keep", "") == "keep"

    def test_longer_non_empty_both_empty(self):
        assert _longer_non_empty("", "") == ""

    def test_longer_non_empty_tie_returns_a(self):
        assert _longer_non_empty("aaa", "bbb") == "aaa"

    def test_ordered_union_deduplicates(self):
        result = _ordered_union(["a", "b", "c"], ["b", "d"])
        assert result == ["a", "b", "c", "d"]

    def test_ordered_union_preserves_first_seen_order(self):
        result = _ordered_union(["x", "y"], ["z", "x"])
        assert result == ["x", "y", "z"]

    def test_union_concatenates(self):
        result = _union([["a", "b"], ["b", "c"]])
        assert result == ["a", "b", "b", "c"]

    def test_dedup_first_by_key(self):
        items = [("a", 1), ("b", 2), ("a", 3)]
        result = _dedup_first([items], key=lambda x: x[0])
        assert result == [("a", 1), ("b", 2)]


# ── E3: finding collision ─────────────────────────────────────────────────────


class TestFindingCollision:
    def test_highest_severity_wins(self):
        a = _finding("f1", severity=Severity.LOW)
        b = _finding("f1", severity=Severity.CRITICAL)
        result = _collide_findings(a, b)
        assert result.severity == Severity.CRITICAL

    def test_longer_description_wins(self):
        a = _finding("f1", description="short")
        b = _finding("f1", description="much longer description here")
        result = _collide_findings(a, b)
        assert result.description == "much longer description here"

    def test_evidence_union(self):
        a = _finding("f1", evidence=["log-line-1"])
        b = _finding("f1", evidence=["log-line-1", "metrics-spike"])
        result = _collide_findings(a, b)
        assert result.evidence == ["log-line-1", "metrics-spike"]


# ── E4: _merge_findings ───────────────────────────────────────────────────────


class TestMergeFindings:
    def test_dedup_across_buckets(self):
        f1a = _finding("crash-loop", severity=Severity.HIGH)
        f1b = _finding("crash-loop", severity=Severity.CRITICAL)
        f2 = _finding("oom-kill", severity=Severity.MEDIUM)

        result = _merge_findings([[f1a, f2], [f1b]])
        ids = [f.id for f in result]
        assert len(result) == 2
        # crash-loop should be CRITICAL after collision
        crash = next(f for f in result if f.id == "crash-loop")
        assert crash.severity == Severity.CRITICAL

    def test_empty_slug_dropped(self):
        bad = _finding("!!!")  # slugifies to ""
        good = _finding("real-finding")
        result = _merge_findings([[bad, good]])
        assert len(result) == 1
        assert result[0].id == "real-finding"

    def test_sorted_by_severity(self):
        low = _finding("low-issue", severity=Severity.LOW)
        crit = _finding("crit-issue", severity=Severity.CRITICAL)
        result = _merge_findings([[low, crit]])
        assert result[0].severity == Severity.CRITICAL


# ── E5: causal edge pruning ───────────────────────────────────────────────────


class TestPruneCausalEdges:
    def test_removes_dangling_caused_by(self):
        f1 = _finding("root", caused_by=["missing-finding", "root"])
        f2 = _finding("downstream", caused_by=["root"])
        by_slug = {"root": f1, "downstream": f2}
        _prune_causal_edges([f1, f2], by_slug)
        assert "missing-finding" not in f1.caused_by
        assert "root" in f1.caused_by  # self-ref kept if present in by_slug

    def test_removes_dangling_causes(self):
        f = _finding("root", causes=["ghost"])
        _prune_causal_edges([f], {"root": f})
        assert f.causes == []


# ── E6: evidence gap merge ────────────────────────────────────────────────────


class TestMergeEvidenceGaps:
    def test_dedup_by_source_reason(self):
        g1 = EvidenceGap(source="metrics", reason="error", details="timeout")
        g2 = EvidenceGap(source="metrics", reason="error", details="network")
        result = _merge_evidence_gaps([[g1], [g2]])
        assert len(result) == 1
        assert "timeout" in result[0].details
        assert "network" in result[0].details

    def test_no_duplicate_details(self):
        g1 = EvidenceGap(source="logs", reason="empty", details="no logs")
        g2 = EvidenceGap(source="logs", reason="empty", details="no logs")
        result = _merge_evidence_gaps([[g1], [g2]])
        # details should not be doubled
        assert result[0].details.count("no logs") == 1

    def test_sorted_by_source_reason(self):
        ga = EvidenceGap(source="z-source", reason="a")
        gb = EvidenceGap(source="a-source", reason="z")
        result = _merge_evidence_gaps([[ga, gb]])
        assert result[0].source == "a-source"


# ── E7: hypotheses merge ─────────────────────────────────────────────────────


class TestMergeHypotheses:
    def _hyp(self, label: str, prob: float) -> RootCauseHypothesis:
        return RootCauseHypothesis(
            label=label,
            probability=prob,
            confirms_if="x is observed",
            refutes_if="y is not observed",
        )

    def test_higher_probability_wins(self):
        h1 = self._hyp("memory-leak", 0.3)
        h2 = self._hyp("memory-leak", 0.8)
        result = _merge_hypotheses([[h1], [h2]])
        assert len(result) == 1
        assert result[0].probability == 0.8

    def test_capped_at_4(self):
        hyps = [self._hyp(f"hyp-{i}", 0.1 * i) for i in range(1, 9)]
        result = _merge_hypotheses([hyps])
        assert len(result) <= 4

    def test_sorted_by_probability_desc(self):
        h1 = self._hyp("low", 0.1)
        h2 = self._hyp("high", 0.9)
        result = _merge_hypotheses([[h1, h2]])
        assert result[0].probability >= result[-1].probability


# ── E8: recommendations + timeline ───────────────────────────────────────────


class TestMergeRecommendations:
    def _rec(self, title: str, priority: int) -> RecommendedAction:
        return RecommendedAction(title=title, priority=priority)

    def test_lowest_priority_int_wins(self):
        r1 = self._rec("fix-it", 5)
        r2 = self._rec("fix-it", 1)
        result = _merge_recommendations([[r1], [r2]])
        assert len(result) == 1
        assert result[0].priority == 1

    def test_sorted_by_priority(self):
        r1 = self._rec("second", 2)
        r2 = self._rec("first", 1)
        result = _merge_recommendations([[r1, r2]])
        assert result[0].priority == 1


class TestMergeTimeline:
    def _ev(self, ts: str, event: str) -> TimelineEvent:
        return TimelineEvent(time=ts, event=event, severity="info")

    def test_dedup_by_time_event(self):
        e1 = self._ev("2024-01-01T00:00:00Z", "deploy started")
        e2 = self._ev("2024-01-01T00:00:00Z", "deploy started")
        result = _merge_timeline([[e1], [e2]])
        assert len(result) == 1

    def test_sorted_by_time(self):
        e1 = self._ev("2024-01-02T00:00:00Z", "later")
        e2 = self._ev("2024-01-01T00:00:00Z", "earlier")
        result = _merge_timeline([[e1, e2]])
        assert result[0].time == "2024-01-01T00:00:00Z"


# ── E8b: rebuild helpers ─────────────────────────────────────────────────────


class TestRebuildHelpers:
    def test_executive_summary_critical_finding(self):
        findings = [_finding("f1", severity=Severity.CRITICAL)]
        template = _exec_summary(OverallStatus.HEALTHY)
        result = _rebuild_executive_summary(findings, template)
        assert result.overall_status == OverallStatus.CRITICAL

    def test_executive_summary_no_findings_unknown(self):
        result = _rebuild_executive_summary([], _exec_summary(OverallStatus.HEALTHY))
        assert result.overall_status == OverallStatus.UNKNOWN

    def test_causal_graph_none_when_no_edges(self):
        findings = [_finding("f1")]
        by_slug = {"f1": findings[0]}
        assert _rebuild_causal_graph(findings, by_slug) is None

    def test_causal_graph_emits_mermaid(self):
        f1 = _finding("root")
        f2 = _finding("downstream", caused_by=["root"])
        by_slug = {"root": f1, "downstream": f2}
        graph = _rebuild_causal_graph([f1, f2], by_slug)
        assert graph is not None
        assert "root --> downstream" in graph
        assert graph.startswith("graph TD")


# ── E9: merge_health_reports ─────────────────────────────────────────────────


class TestMergeHealthReports:
    def test_empty_returns_none(self):
        assert merge_health_reports([]) is None

    def test_single_returns_same(self):
        r = _bare_report()
        assert merge_health_reports([r]) is r

    def test_two_reports_merged(self):
        r1 = HealthReport(
            executive_summary=_exec_summary(OverallStatus.HEALTHY),
            findings=[_finding("f1", severity=Severity.HIGH)],
        )
        r2 = HealthReport(
            executive_summary=_exec_summary(OverallStatus.HEALTHY),
            findings=[_finding("f2", severity=Severity.LOW)],
        )
        merged = merge_health_reports([r1, r2])
        assert merged is not None
        ids = {f.id for f in merged.findings}
        assert "f1" in ids
        assert "f2" in ids
        # overall_status driven by worst finding (HIGH → DEGRADED)
        assert merged.executive_summary.overall_status == OverallStatus.DEGRADED

    def test_duplicate_finding_deduped(self):
        f = _finding("dup", severity=Severity.MEDIUM)
        r1 = _bare_report([f])
        r2 = _bare_report([f])
        merged = merge_health_reports([r1, r2])
        assert merged is not None
        assert len(merged.findings) == 1

    def test_evidence_gaps_merged(self):
        g1 = EvidenceGap(source="metrics", reason="error", details="timeout")
        g2 = EvidenceGap(source="logs", reason="empty")
        r1 = HealthReport(executive_summary=_exec_summary(), evidence_gaps=[g1])
        r2 = HealthReport(executive_summary=_exec_summary(), evidence_gaps=[g2])
        merged = merge_health_reports([r1, r2])
        assert merged is not None
        sources = {g.source for g in merged.evidence_gaps}
        assert {"metrics", "logs"} == sources

    def test_result_is_valid_pydantic(self):
        """Merged report must pass Pydantic validation (no ValidationError)."""
        reports = [_bare_report([_finding(f"f{i}") for i in range(3)]) for _ in range(3)]
        merged = merge_health_reports(reports)
        assert merged is not None
        # Re-validate by round-tripping through model_validate
        revalidated = HealthReport.model_validate(merged.model_dump())
        assert revalidated is not None

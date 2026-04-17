"""Tests for causal graph fields on Finding and HealthReport (SH-03)."""

from __future__ import annotations

from vaig.skills.service_health.schema import (
    Confidence,
    ExecutiveSummary,
    Finding,
    HealthReport,
    Severity,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_finding(**kwargs) -> Finding:
    defaults: dict = {
        "id": "f1",
        "title": "Test finding",
        "severity": Severity.HIGH,
        "description": "desc",
        "root_cause": "unknown",
        "evidence": [],
        "confidence": Confidence.HIGH,
        "impact": "impact",
        "affected_resources": [],
    }
    defaults.update(kwargs)
    return Finding(**defaults)


def _make_report(*findings: Finding, causal_graph_mermaid: str | None = None) -> HealthReport:
    return HealthReport(
        executive_summary=ExecutiveSummary(
            overall_status="HEALTHY",
            scope="test",
            summary_text="test summary",
        ),
        findings=list(findings),
        recommendations=[],
        root_cause_hypotheses=[],
        timeline=[],
        metadata={},
        causal_graph_mermaid=causal_graph_mermaid,
    )


# ── Finding causal field defaults ────────────────────────────────────────────


def test_finding_caused_by_defaults_to_empty_list():
    f = _make_finding()
    assert f.caused_by == []


def test_finding_causes_defaults_to_empty_list():
    f = _make_finding()
    assert f.causes == []


def test_finding_caused_by_accepts_slugs():
    f = _make_finding(caused_by=["f0", "f2"])
    assert f.caused_by == ["f0", "f2"]


def test_finding_causes_accepts_slugs():
    f = _make_finding(causes=["f3"])
    assert f.causes == ["f3"]


# ── HealthReport.root_causes property ────────────────────────────────────────


def test_root_causes_empty_when_no_findings():
    report = _make_report()
    assert report.root_causes == []


def test_root_causes_all_findings_when_none_have_caused_by():
    f1 = _make_finding(id="f1")
    f2 = _make_finding(id="f2")
    report = _make_report(f1, f2)
    assert {f.id for f in report.root_causes} == {"f1", "f2"}


def test_root_causes_excludes_findings_with_caused_by():
    root = _make_finding(id="root", caused_by=[])
    child = _make_finding(id="child", caused_by=["root"])
    report = _make_report(root, child)
    assert [f.id for f in report.root_causes] == ["root"]


def test_root_causes_is_empty_when_all_have_caused_by():
    f1 = _make_finding(id="f1", caused_by=["f2"])
    f2 = _make_finding(id="f2", caused_by=["f1"])
    report = _make_report(f1, f2)
    assert report.root_causes == []


# ── to_markdown() mermaid block ───────────────────────────────────────────────


def test_to_markdown_no_causal_graph_when_none():
    report = _make_report(_make_finding(), causal_graph_mermaid=None)
    md = report.to_markdown()
    assert "mermaid" not in md.lower()


def test_to_markdown_includes_mermaid_block_when_set():
    graph = "graph TD\n  f1 --> f2"
    report = _make_report(_make_finding(), causal_graph_mermaid=graph)
    md = report.to_markdown()
    assert "```mermaid" in md
    assert graph in md
    assert "```" in md

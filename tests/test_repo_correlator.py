"""Tests for SPEC-V2-REPO-07 — RepoCorrelator and Finding.repo_evidence."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import ValidationError

from vaig.core.repo_correlator import RepoCorrelator
from vaig.skills.service_health.schema import (
    ExecutiveSummary,
    Finding,
    HealthReport,
    OverallStatus,
    RepoSnippet,
    Severity,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


@dataclass
class _FakeChunk:
    """Minimal duck-typed chunk returned by a fake index."""

    file_path: str = "charts/values.yaml"
    start_line: int = 1
    end_line: int = 5
    content: str = "replicas: 2"
    outline: str = ""
    relevance_score: float = 0.9
    retrieval_query: str = "test query"


class _FakeIndex:
    """Fake repo index that returns a fixed list of chunks."""

    def __init__(self, chunks: list[Any] | None = None) -> None:
        self._chunks = chunks or []

    def search(self, query: str, k: int = 8) -> list[Any]:
        return self._chunks[:k]


def _minimal_finding(**kwargs: Any) -> Finding:
    defaults: dict[str, Any] = {
        "id": "test-finding",
        "title": "Test finding",
        "severity": Severity.INFO,
        "quick_remediation": "kubectl get pods",
    }
    defaults.update(kwargs)
    return Finding(**defaults)


def _minimal_report(findings: list[Finding] | None = None) -> HealthReport:
    return HealthReport(
        executive_summary=ExecutiveSummary(
            overall_status=OverallStatus.HEALTHY,
            scope="test",
            summary_text="Test report",
        ),
        findings=findings or [],
    )


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_correlate_attaches_repo_evidence() -> None:
    """Finding with affected_resources + matching index chunk → repo_evidence populated."""
    chunk = _FakeChunk(content="some config content")
    index = _FakeIndex([chunk])
    correlator = RepoCorrelator(index)

    finding = _minimal_finding(
        id="crashloop-payment",
        title="CrashLoopBackOff in payment-svc",
        affected_resources=["payment-svc"],
    )
    report = _minimal_report([finding])

    new_report, result = correlator.correlate(report)

    assert len(new_report.findings) >= 1
    enriched = next(f for f in new_report.findings if f.id == "crashloop-payment")
    assert len(enriched.repo_evidence) == 1
    assert enriched.repo_evidence[0].file_path == "charts/values.yaml"
    assert enriched.repo_evidence[0].relevance_score == 0.9
    assert result.report_findings_enriched == 1


def test_correlate_no_repo_returns_unchanged() -> None:
    """Empty index → report unchanged, report_findings_enriched == 0."""
    index = _FakeIndex([])
    correlator = RepoCorrelator(index)

    finding = _minimal_finding(
        id="some-finding",
        title="Pod restarts",
        affected_resources=["my-svc"],
    )
    report = _minimal_report([finding])

    new_report, result = correlator.correlate(report)

    assert result.report_findings_enriched == 0
    assert result.contradiction_findings_added == 0
    enriched = next(f for f in new_report.findings if f.id == "some-finding")
    assert enriched.repo_evidence == []


def test_correlate_contradiction_finding_emitted() -> None:
    """Snippet has 'replicas: 3', finding mentions 2 pods → contradiction finding added."""
    chunk = _FakeChunk(
        file_path="helm/values.yaml",
        start_line=10,
        end_line=10,
        content="replicas: 3",
        relevance_score=0.8,
    )
    index = _FakeIndex([chunk])
    correlator = RepoCorrelator(index)

    finding = _minimal_finding(
        id="replica-issue",
        title="Only 2 pods ready in payment-svc",
        affected_resources=["payment-svc"],
        service="payment-svc",
    )
    report = _minimal_report([finding])

    new_report, result = correlator.correlate(report)

    assert result.contradiction_findings_added == 1
    contradiction = next(
        (f for f in new_report.findings if "repo-drift" in f.id),
        None,
    )
    assert contradiction is not None
    assert contradiction.severity == Severity.HIGH
    assert "3" in contradiction.title  # chart replicas
    assert "2" in contradiction.title  # runtime replicas


def test_correlate_no_contradiction_when_replicas_match() -> None:
    """Same replica count in snippet and title → no contradiction finding."""
    chunk = _FakeChunk(content="replicas: 2", relevance_score=0.7)
    index = _FakeIndex([chunk])
    correlator = RepoCorrelator(index)

    finding = _minimal_finding(
        id="replica-ok",
        title="Only 2 pods ready in payment-svc",
        affected_resources=["payment-svc"],
    )
    report = _minimal_report([finding])

    new_report, result = correlator.correlate(report)

    assert result.contradiction_findings_added == 0
    drift_findings = [f for f in new_report.findings if "repo-drift" in f.id]
    assert drift_findings == []


def test_correlate_result_counts_are_accurate() -> None:
    """N findings with matched chunks → report_findings_enriched == N."""
    chunk = _FakeChunk(content="some-config", relevance_score=0.6)
    index = _FakeIndex([chunk])
    correlator = RepoCorrelator(index)

    findings = [
        _minimal_finding(id=f"finding-{i}", title=f"Issue {i}", affected_resources=[f"svc-{i}"])
        for i in range(3)
    ]
    report = _minimal_report(findings)

    _, result = correlator.correlate(report)

    assert result.report_findings_enriched == 3


def test_repo_snippet_model_validation() -> None:
    """relevance_score outside [0, 1] raises ValidationError."""
    with pytest.raises(ValidationError):
        RepoSnippet(
            file_path="foo.yaml",
            line_start=1,
            line_end=5,
            excerpt="content",
            relevance_score=1.5,  # invalid — > 1.0
            retrieval_query="query",
        )

    with pytest.raises(ValidationError):
        RepoSnippet(
            file_path="foo.yaml",
            line_start=1,
            line_end=5,
            excerpt="content",
            relevance_score=-0.1,  # invalid — < 0.0
            retrieval_query="query",
        )


def test_finding_repo_evidence_field_defaults_empty() -> None:
    """New Finding() has repo_evidence == []."""
    finding = _minimal_finding()
    assert finding.repo_evidence == []

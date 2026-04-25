"""ATT-11 — Attachment usage observability.

Tests cover:
  1. AttachmentCitation model validates correctly
  2. AttachmentUsage model validates correctly
  3. AttachmentEvidenceSummary model validates correctly
  4. Finding.attachment_citations defaults to empty list
  5. HealthReport.attachment_evidence defaults to empty list and is excluded from Gemini schema
  6. _detect_chunk_references correctly identifies referenced chunks
  7. _aggregate_attachment_summaries aggregates per-agent usages correctly
  8. to_markdown renders Attachment Evidence section when populated
"""

from __future__ import annotations

from vaig.core.headless import _aggregate_attachment_summaries, _detect_chunk_references
from vaig.skills.service_health.schema import (
    AttachmentCitation,
    AttachmentEvidenceSummary,
    AttachmentUsage,
    Confidence,
    ExecutiveSummary,
    Finding,
    HealthReport,
    HealthReportGeminiSchema,
    OverallStatus,
    Severity,
)

# ── 1. AttachmentCitation validates correctly ─────────────────────────────────


def test_attachment_citation_required_fields():
    citation = AttachmentCitation(attachment_name="runbook.md")
    assert citation.attachment_name == "runbook.md"
    assert citation.file_path == ""
    assert citation.line_start is None
    assert citation.line_end is None
    assert citation.excerpt == ""


def test_attachment_citation_full():
    citation = AttachmentCitation(
        attachment_name="runbook.md",
        file_path="docs/runbook.md",
        line_start=10,
        line_end=20,
        excerpt="Check pod logs with kubectl logs -n prod",
    )
    assert citation.line_start == 10
    assert citation.line_end == 20
    assert "kubectl" in citation.excerpt


# ── 2. AttachmentUsage validates correctly ────────────────────────────────────


def test_attachment_usage_defaults():
    usage = AttachmentUsage(agent_name="reporter", attachment_name="runbook.md")
    assert usage.context_bytes_received == 0
    assert usage.context_bytes_truncated == 0
    assert usage.chunks_referenced == []
    assert usage.free_text_quotes == []


def test_attachment_usage_full():
    usage = AttachmentUsage(
        agent_name="reporter",
        attachment_name="runbook.md",
        context_bytes_received=4096,
        context_bytes_truncated=0,
        chunks_referenced=["runbook.md#step-1"],
        free_text_quotes=["Check the logs"],
    )
    assert usage.context_bytes_received == 4096
    assert len(usage.chunks_referenced) == 1


# ── 3. AttachmentEvidenceSummary validates correctly ─────────────────────────


def test_attachment_evidence_summary_defaults():
    s = AttachmentEvidenceSummary(attachment_name="runbook.md")
    assert s.kind == "file"
    assert s.bytes_sent == 0
    assert s.agents_that_cited == []
    assert s.total_chunks_cited == 0


# ── 4. Finding.attachment_citations defaults to empty ────────────────────────


def test_finding_attachment_citations_default():
    f = Finding(
        id="F-001",
        title="Test finding",
        severity=Severity.HIGH,
        description="desc",
        root_cause="cause",
        evidence=["event text"],
        confidence=Confidence.HIGH,
        impact="some impact",
        affected_resources=["deployment/app"],
    )
    assert f.attachment_citations == []


def test_finding_attachment_citations_populated():
    citation = AttachmentCitation(
        attachment_name="runbook.md",
        excerpt="Check the error logs",
    )
    f = Finding(
        id="F-002",
        title="Finding with citation",
        severity=Severity.MEDIUM,
        description="desc",
        root_cause="cause",
        evidence=["event text"],
        confidence=Confidence.MEDIUM,
        impact="impact",
        affected_resources=["pod/app-xyz"],
        attachment_citations=[citation],
    )
    assert len(f.attachment_citations) == 1
    assert f.attachment_citations[0].attachment_name == "runbook.md"


# ── 5. HealthReport.attachment_evidence defaults + Gemini exclusion ───────────


def test_health_report_attachment_evidence_default():
    report = HealthReport(
        executive_summary=ExecutiveSummary(
            overall_status=OverallStatus.HEALTHY,
            scope="Namespace: test",
            summary_text="All good",
        )
    )
    assert report.attachment_evidence == []


def test_attachment_evidence_excluded_from_gemini_schema():
    assert "attachment_evidence" in HealthReportGeminiSchema._GEMINI_EXCLUDED_FIELDS


# ── 6. _detect_chunk_references identifies referenced chunks ─────────────────


def test_detect_chunk_references_match():
    attachment_ctx = (
        "### runbook.md\n"
        "When a pod is OOMKilled you should check memory limits and increase them. "
        "This is a standard procedure followed by the SRE team. "
        "Always verify with kubectl describe pod before making changes. "
        "Check resource quotas as well to avoid surprises."
    )
    # Agent output contains a verbatim 6-word phrase from the attachment
    agent_output = (
        "The runbook states: when a pod is OOMKilled you should check memory limits "
        "and increase them as a standard procedure followed by the SRE team."
    )
    chunk_ids, quotes = _detect_chunk_references(attachment_ctx, agent_output)
    assert "runbook.md" in chunk_ids
    assert len(quotes) >= 1


def test_detect_chunk_references_no_match():
    attachment_ctx = (
        "### runbook.md\n"
        "Completely unrelated content about database migrations and schema changes. "
        "Nothing to do with Kubernetes or pods whatsoever."
    )
    agent_output = "The pod is crashing due to a liveness probe failure."
    chunk_ids, quotes = _detect_chunk_references(attachment_ctx, agent_output)
    assert chunk_ids == []
    assert quotes == []


def test_detect_chunk_references_empty_context():
    chunk_ids, quotes = _detect_chunk_references("", "some agent output")
    assert chunk_ids == []
    assert quotes == []


# ── 7. _aggregate_attachment_summaries aggregates correctly ──────────────────


def test_aggregate_attachment_summaries():
    usages = [
        AttachmentUsage(
            agent_name="analyzer",
            attachment_name="runbook.md",
            context_bytes_received=2048,
            context_bytes_truncated=0,
            chunks_referenced=["runbook.md#step-1", "runbook.md#step-2"],
        ),
        AttachmentUsage(
            agent_name="reporter",
            attachment_name="runbook.md",
            context_bytes_received=2048,
            context_bytes_truncated=0,
            chunks_referenced=["runbook.md#step-1"],
        ),
    ]
    summaries = _aggregate_attachment_summaries(usages)
    assert len(summaries) == 1
    s = summaries[0]
    assert s.attachment_name == "runbook.md"
    assert s.bytes_sent == 2048
    assert set(s.agents_that_cited) == {"analyzer", "reporter"}
    assert s.total_chunks_cited == 3  # 2 + 1


def test_aggregate_attachment_summaries_no_citations():
    """Agents that did not cite anything should not appear in agents_that_cited."""
    usages = [
        AttachmentUsage(
            agent_name="gatherer",
            attachment_name="runbook.md",
            context_bytes_received=1024,
            chunks_referenced=[],
        ),
    ]
    summaries = _aggregate_attachment_summaries(usages)
    assert len(summaries) == 1
    assert summaries[0].agents_that_cited == []
    assert summaries[0].total_chunks_cited == 0


# ── 8. to_markdown renders Attachment Evidence section ───────────────────────


def test_to_markdown_attachment_evidence_section():
    report = HealthReport(
        executive_summary=ExecutiveSummary(
            overall_status=OverallStatus.DEGRADED,
            scope="Namespace: prod",
            summary_text="One pod crashing",
        ),
        attachment_evidence=[
            AttachmentEvidenceSummary(
                attachment_name="runbook.md",
                kind="file",
                bytes_sent=4096,
                bytes_truncated=0,
                agents_that_cited=["reporter"],
                total_chunks_cited=2,
            )
        ],
    )
    md = report.to_markdown()
    assert "## Attachment Evidence" in md
    assert "runbook.md" in md
    # Table cell — bytes as comma-separated integer
    assert "4,096" in md
    assert "reporter" in md
    # total_chunks_cited appears in the table row
    assert "| 2 |" in md


def test_to_markdown_no_attachment_evidence_section_when_empty():
    report = HealthReport(
        executive_summary=ExecutiveSummary(
            overall_status=OverallStatus.HEALTHY,
            scope="Namespace: test",
            summary_text="All good",
        ),
    )
    md = report.to_markdown()
    assert "## Attachment Evidence" not in md


def test_to_markdown_attachment_evidence_truncated_warning():
    report = HealthReport(
        executive_summary=ExecutiveSummary(
            overall_status=OverallStatus.DEGRADED,
            scope="Namespace: prod",
            summary_text="Partial analysis",
        ),
        attachment_evidence=[
            AttachmentEvidenceSummary(
                attachment_name="large-runbook.md",
                bytes_sent=131072,
                bytes_truncated=50000,
            )
        ],
    )
    md = report.to_markdown()
    assert "⚠" in md
    # truncated bytes appear as ⚠ 50,000 in the table cell
    assert "50,000" in md


def test_to_markdown_attachment_evidence_no_citations_note():
    """When all summaries have total_chunks_cited == 0, render the VIS-4 note."""
    report = HealthReport(
        executive_summary=ExecutiveSummary(
            overall_status=OverallStatus.HEALTHY,
            scope="Namespace: test",
            summary_text="All good",
        ),
        attachment_evidence=[
            AttachmentEvidenceSummary(
                attachment_name="runbook.md",
                bytes_sent=1024,
                bytes_truncated=0,
                total_chunks_cited=0,
            )
        ],
    )
    md = report.to_markdown()
    assert "No findings were influenced by attachments on this run." in md

"""Tests for SPEC-ATT-10 §6.5.4 — Reporter attachment sections.

Covers:
- render_attachment_sections returns "" when no findings have attachment-influenced source_support
- render_attachment_sections returns "" for empty findings list
- Verified Expectations section present when finding has live_matches_expected_state
- Source Evidence section present when finding has non-empty supporting_evidence
- Contradictions rendered with Expected:/Observed: labels
- Attachment references rendered with attachment_name
- Both sections present when finding qualifies for both
- live_only finding excluded from Verified Expectations table
- HealthReport.attachment_sections_md field default is ""
- attachment_sections_md excluded from HealthReportGeminiSchema properties
"""

from __future__ import annotations

from vaig.skills.service_health.schema import (
    AttachmentRef,
    ContradictionRef,
    EvidenceRef,
    Finding,
    HealthReport,
    HealthReportGeminiSchema,
    render_attachment_sections,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_report(*findings: Finding) -> HealthReport:
    report = HealthReport.model_validate(
        {
            "executive_summary": {
                "overall_status": "HEALTHY",
                "scope": "Cluster-wide",
                "summary_text": "All good",
            },
            "findings": [],
        }
    )
    return report.model_copy(update={"findings": list(findings)})


def _basic_finding(**kwargs: object) -> Finding:
    base = {
        "id": "test-finding",
        "title": "Test Finding",
        "severity": "HIGH",
        "confidence": "HIGH",
        "description": "desc",
        "evidence": [],
        "recommendation": "rec",
        "affected_services": [],
        "namespace": "",
        "caused_by": [],
    }
    base.update(kwargs)
    return Finding.model_validate(base)


# ── TestRenderAttachmentSections ─────────────────────────────────────────────


class TestRenderAttachmentSections:
    def test_returns_empty_for_no_findings(self) -> None:
        report = _make_report()
        assert render_attachment_sections(report) == ""

    def test_returns_empty_for_live_only_findings(self) -> None:
        f = _basic_finding(title="Live Only")
        # source_support defaults to "live_only"
        report = _make_report(f)
        assert render_attachment_sections(report) == ""

    def test_verified_expectations_section_present(self) -> None:
        f = _basic_finding(title="Replica Count Check")
        object.__setattr__(f, "source_support", "live_matches_expected_state")
        report = _make_report(f)
        result = render_attachment_sections(report)
        assert "## Verified Expectations" in result
        assert "Replica Count Check" in result
        assert "live_matches_expected_state" in result

    def test_source_evidence_section_present(self) -> None:
        f = _basic_finding(title="Memory Spike")
        ev = EvidenceRef(source="attachment", excerpt="Memory > 90%", attachment_name="runbook.md", line_ref="L42")
        object.__setattr__(f, "supporting_evidence", [ev])
        report = _make_report(f)
        result = render_attachment_sections(report)
        assert "## Source Evidence" in result
        assert "Memory Spike" in result
        assert "Memory > 90%" in result

    def test_contradictions_rendered(self) -> None:
        f = _basic_finding(title="Version Mismatch")
        c = ContradictionRef(expected="v2.3", observed="v2.1", attachment_name="values.yaml")
        object.__setattr__(f, "contradictions", [c])
        report = _make_report(f)
        result = render_attachment_sections(report)
        assert "Expected:" in result
        assert "Observed:" in result
        assert "v2.3" in result
        assert "v2.1" in result

    def test_attachment_refs_rendered(self) -> None:
        f = _basic_finding(title="Runbook Guidance")
        ref = AttachmentRef(attachment_name="runbook.md", relevance="Contains restart procedure")
        object.__setattr__(f, "attachment_references", [ref])
        report = _make_report(f)
        result = render_attachment_sections(report)
        assert "runbook.md" in result
        assert "Contains restart procedure" in result

    def test_both_sections_present(self) -> None:
        f = _basic_finding(title="Full Evidence Finding")
        object.__setattr__(f, "source_support", "live_and_attachment_corroborated")
        ev = EvidenceRef(source="live", excerpt="CPU high", attachment_name="runbook.md")
        object.__setattr__(f, "supporting_evidence", [ev])
        report = _make_report(f)
        result = render_attachment_sections(report)
        assert "## Verified Expectations" in result
        assert "## Source Evidence" in result

    def test_live_only_finding_excluded_from_verified(self) -> None:
        f_live = _basic_finding(title="Live Only Finding")
        # source_support = "live_only" by default
        f_att = _basic_finding(title="Attachment Corroborated")
        object.__setattr__(f_att, "source_support", "live_and_attachment_corroborated")
        report = _make_report(f_live, f_att)
        result = render_attachment_sections(report)
        assert "Live Only Finding" not in result
        assert "Attachment Corroborated" in result

    def test_all_verified_source_supports_included(self) -> None:
        supported = [
            "live_matches_expected_state",
            "live_and_attachment_corroborated",
            "live_matches_known_incident_pattern",
            "live_with_attachment_enrichment",
        ]
        for ss in supported:
            f = _basic_finding(title=f"Finding {ss}")
            object.__setattr__(f, "source_support", ss)
            report = _make_report(f)
            result = render_attachment_sections(report)
            assert "## Verified Expectations" in result, f"Expected Verified section for {ss}"

    def test_attachment_only_excluded_from_verified(self) -> None:
        f = _basic_finding(title="Attachment Only")
        object.__setattr__(f, "source_support", "attachment_only")
        report = _make_report(f)
        result = render_attachment_sections(report)
        # attachment_only is not in the verified set — and no evidence either
        assert result == ""


# ── HealthReport field tests ──────────────────────────────────────────────────


class TestAttachmentSectionsMdField:
    def test_attachment_sections_md_field_default_empty(self) -> None:
        report = HealthReport.model_validate(
            {
                "executive_summary": {
                    "overall_status": "HEALTHY",
                    "scope": "Cluster-wide",
                    "summary_text": "All good",
                },
            }
        )
        assert report.attachment_sections_md == ""

    def test_attachment_sections_md_excluded_from_model_dump(self) -> None:
        report = HealthReport.model_validate(
            {
                "executive_summary": {
                    "overall_status": "HEALTHY",
                    "scope": "Cluster-wide",
                    "summary_text": "All good",
                },
            }
        )
        dumped = report.model_dump()
        assert "attachment_sections_md" not in dumped

    def test_attachment_sections_md_excluded_from_gemini_schema(self) -> None:
        schema = HealthReportGeminiSchema.model_json_schema()
        props = schema.get("properties", {})
        assert "attachment_sections_md" not in props

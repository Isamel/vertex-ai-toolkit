"""SPEC-ATT-10 §6.5.6 — Acceptance criteria unit tests.

Scenarios covered:
- SH-ATT-H5: No attachments → pipeline unchanged (LIVE_ONLY, empty attachment sections)
- SH-ATT-H3: Healthy cluster + values.yaml match → live_matches_expected_state
- SH-ATT-2:  Runbook version contradiction → Contradictions section rendered
- SH-ATT-3:  Live-corroborated → confidence upgrade via ratification
- SH-ATT-H1: AttachmentPriors built with expected fields
- SH-ATT-H2: Seeded hypothesis from runbook hotspot → Source Evidence section

Scenarios intentionally skipped (already fully covered in existing test files):
- SH-ATT-4: ATTACHMENT_ONLY mode → source_support tagged (test_att10_operating_mode.py)
- SH-ATT-5: offline_mode=True → ATTACHMENT_ONLY (test_att10_operating_mode.py)
"""

from __future__ import annotations

import json

from vaig.skills.service_health.schema import (
    AttachmentPriors,
    AttachmentRef,
    Confidence,
    ContradictionRef,
    EvidenceRef,
    Finding,
    HealthReport,
    Hotspot,
    OperatingMode,
    apply_ratification,
    render_attachment_sections,
)
from vaig.skills.service_health.skill import ServiceHealthSkill

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_report(*findings: Finding) -> HealthReport:
    """Return a minimal HealthReport with the given findings."""
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
    """Return a minimal Finding, with optional field overrides."""
    base: dict = {
        "id": "test-finding",
        "title": "Test Finding",
        "severity": "HIGH",
    }
    base.update(kwargs)
    return Finding.model_validate(base)


# ── SH-ATT-H5: No attachments → pipeline unchanged ───────────────────────────


class TestSHATTH5:
    """Regression guard: no attachments must leave the pipeline untouched."""

    def test_operating_mode_is_live_only_when_no_attachments(self) -> None:
        """_detect_operating_mode(offline_mode=False, attachments_present=False) == LIVE_ONLY."""
        mode = ServiceHealthSkill._detect_operating_mode(
            offline_mode=False,
            attachments_present=False,
        )
        assert mode == OperatingMode.LIVE_ONLY

    def test_render_attachment_sections_returns_empty_for_live_only_report(self) -> None:
        """render_attachment_sections returns '' when all findings are live_only."""
        f = _basic_finding(title="Normal Finding")
        # source_support defaults to "live_only" — no attachment influence
        report = _make_report(f)
        assert report.operating_mode == OperatingMode.LIVE_ONLY
        assert render_attachment_sections(report) == ""

    def test_render_attachment_sections_returns_empty_for_empty_report(self) -> None:
        """render_attachment_sections returns '' for a report with no findings."""
        report = _make_report()
        assert render_attachment_sections(report) == ""

    def test_attachment_sections_md_field_defaults_to_empty(self) -> None:
        """HealthReport.attachment_sections_md is '' when no attachment sections were rendered."""
        report = _make_report()
        assert report.attachment_sections_md == ""


# ── SH-ATT-H3: Healthy cluster + values.yaml match → live_matches_expected_state ─


class TestSHATTH3:
    """Healthy cluster + attachment match → Verified Expectations section rendered."""

    def test_verified_expectations_section_present(self) -> None:
        """render_attachment_sections produces a 'Verified Expectations' section."""
        f = _basic_finding(title="Replica Count Matches Expected")
        object.__setattr__(f, "source_support", "live_matches_expected_state")
        report = _make_report(f)

        result = render_attachment_sections(report)

        assert "## Verified Expectations" in result

    def test_verified_table_contains_checkmark_row(self) -> None:
        """The Verified Expectations table contains a row for the finding."""
        f = _basic_finding(title="API Replica Count OK", severity="INFO")
        object.__setattr__(f, "source_support", "live_matches_expected_state")
        report = _make_report(f)

        result = render_attachment_sections(report)

        # The finding title must appear inside the table row
        assert "API Replica Count OK" in result
        assert "live_matches_expected_state" in result

    def test_finding_appears_in_verified_table(self) -> None:
        """The exact finding title is present in the rendered section."""
        title = "Values.yaml Replica Match"
        f = _basic_finding(title=title, severity="INFO")
        object.__setattr__(f, "source_support", "live_matches_expected_state")
        report = _make_report(f)

        result = render_attachment_sections(report)

        assert title in result

    def test_severity_info_finding_rendered(self) -> None:
        """INFO severity finding with live_matches_expected_state appears in verified table."""
        f = _basic_finding(title="Config Check Passed", severity="INFO")
        object.__setattr__(f, "source_support", "live_matches_expected_state")
        report = _make_report(f)

        result = render_attachment_sections(report)

        assert "Config Check Passed" in result
        assert "## Verified Expectations" in result


# ── SH-ATT-2: Version contradiction → Contradictions section ─────────────────


class TestSHATT2:
    """Runbook says v2.3, live says v2.1 → contradiction rendered in report."""

    def test_contradictions_section_present(self) -> None:
        """render_attachment_sections produces a 'Contradictions' section."""
        f = _basic_finding(title="Image Version Mismatch")
        c = ContradictionRef(
            expected="v2.3",
            observed="v2.1",
            attachment_name="runbook.md",
        )
        object.__setattr__(f, "source_support", "live_vs_attachment_contradicts")
        object.__setattr__(f, "contradictions", [c])
        report = _make_report(f)

        result = render_attachment_sections(report)

        assert "## Source Evidence" in result
        assert "**Contradictions**" in result

    def test_contradictions_contains_expected_value(self) -> None:
        """The rendered section contains 'Expected:' with the expected value."""
        f = _basic_finding(title="Image Version Mismatch")
        c = ContradictionRef(
            expected="v2.3",
            observed="v2.1",
            attachment_name="runbook.md",
        )
        object.__setattr__(f, "contradictions", [c])
        report = _make_report(f)

        result = render_attachment_sections(report)

        assert "Expected:" in result
        assert "v2.3" in result

    def test_contradictions_contains_observed_value(self) -> None:
        """The rendered section contains 'Observed:' with the observed value."""
        f = _basic_finding(title="Image Version Mismatch")
        c = ContradictionRef(
            expected="v2.3",
            observed="v2.1",
            attachment_name="runbook.md",
        )
        object.__setattr__(f, "contradictions", [c])
        report = _make_report(f)

        result = render_attachment_sections(report)

        assert "Observed:" in result
        assert "v2.1" in result

    def test_attachment_name_appears_in_contradiction_row(self) -> None:
        """The source attachment name is cited in the contradiction row."""
        f = _basic_finding(title="Version Mismatch Detail")
        c = ContradictionRef(
            expected="v2.3",
            observed="v2.1",
            attachment_name="runbook.md",
        )
        object.__setattr__(f, "contradictions", [c])
        report = _make_report(f)

        result = render_attachment_sections(report)

        assert "runbook.md" in result

    def test_finding_also_appears_in_verified_table_when_contradicts(self) -> None:
        """live_vs_attachment_contradicts findings also appear in the Verified Expectations table."""
        f = _basic_finding(title="Contradicted Finding")
        c = ContradictionRef(expected="v2.3", observed="v2.1")
        object.__setattr__(f, "source_support", "live_vs_attachment_contradicts")
        object.__setattr__(f, "contradictions", [c])
        report = _make_report(f)

        result = render_attachment_sections(report)

        # Both sections should be present (§6.5.4 includes contradicts in verified set)
        assert "## Verified Expectations" in result
        assert "## Source Evidence" in result


# ── SH-ATT-3: Live-corroborated → confidence upgrade via ratification ─────────


class TestSHATT3:
    """apply_ratification upgrades confidence when the verifier confirms corroboration."""

    def test_confidence_upgraded_to_high_via_ratification(self) -> None:
        """Finding with confidence=MEDIUM is upgraded to HIGH when ratification says so."""
        finding = _basic_finding(title="DB Pool Saturation")
        object.__setattr__(finding, "source_support", "live_vs_attachment_contradicts")
        object.__setattr__(finding, "confidence", Confidence.MEDIUM)

        report = _make_report(finding)

        ratif_json = json.dumps(
            [
                {
                    "finding_title": "DB Pool Saturation",
                    "ratified_source_support": "live_and_attachment_corroborated",
                    "confidence_override": "HIGH",
                    "ratification_note": "Live metrics confirm DB pool oversaturation from runbook",
                }
            ]
        )

        result = apply_ratification(report, ratif_json)

        assert result.findings[0].confidence == Confidence.HIGH

    def test_source_support_updated_when_corroborated(self) -> None:
        """source_support is updated from contradicts to corroborated after ratification."""
        finding = _basic_finding(title="DB Pool Saturation")
        object.__setattr__(finding, "source_support", "live_vs_attachment_contradicts")
        object.__setattr__(finding, "confidence", Confidence.MEDIUM)

        report = _make_report(finding)

        ratif_json = json.dumps(
            [
                {
                    "finding_title": "DB Pool Saturation",
                    "ratified_source_support": "live_and_attachment_corroborated",
                    "confidence_override": "HIGH",
                    "ratification_note": "Live corroborated",
                }
            ]
        )

        result = apply_ratification(report, ratif_json)

        assert result.findings[0].source_support == "live_and_attachment_corroborated"

    def test_confidence_medium_to_confirmed_upgrade(self) -> None:
        """MEDIUM confidence can be promoted all the way to CONFIRMED via ratification."""
        finding = _basic_finding(title="OOM Kill Pattern")
        object.__setattr__(finding, "confidence", Confidence.MEDIUM)
        report = _make_report(finding)

        ratif_json = json.dumps(
            [
                {
                    "finding_title": "OOM Kill Pattern",
                    "ratified_source_support": "live_matches_expected_state",
                    "confidence_override": "CONFIRMED",
                    "ratification_note": "kubectl top confirms memory > limit",
                }
            ]
        )

        result = apply_ratification(report, ratif_json)

        assert result.findings[0].confidence == Confidence.CONFIRMED

    def test_none_confidence_override_leaves_confidence_unchanged(self) -> None:
        """confidence_override=None leaves the original confidence intact."""
        finding = _basic_finding(title="Stable Finding")
        object.__setattr__(finding, "confidence", Confidence.MEDIUM)
        original_confidence = finding.confidence
        report = _make_report(finding)

        ratif_json = json.dumps(
            [
                {
                    "finding_title": "Stable Finding",
                    "ratified_source_support": "live_and_attachment_corroborated",
                    "confidence_override": None,
                    "ratification_note": "corroborated without confidence change",
                }
            ]
        )

        result = apply_ratification(report, ratif_json)

        assert result.findings[0].confidence == original_confidence


# ── SH-ATT-H1: AttachmentPriors built with expected fields ───────────────────


class TestSHATTH1:
    """AttachmentPriors model validates correctly and exposes expected fields."""

    def test_model_validates_with_expected_replica_counts(self) -> None:
        """AttachmentPriors accepts expected_replica_counts dict."""
        priors = AttachmentPriors(
            expected_replica_counts={"api": 3},
        )
        assert priors.expected_replica_counts == {"api": 3}

    def test_runbook_hotspots_accessible(self) -> None:
        """runbook_hotspots field is accessible and holds Hotspot objects."""
        hotspot1 = Hotspot(entity="DB connection pool", concern="oversaturation")
        hotspot2 = Hotspot(entity="JVM heap", concern="OOM killer")
        priors = AttachmentPriors(
            expected_replica_counts={"api": 3},
            runbook_hotspots=[hotspot1, hotspot2],
        )
        assert len(priors.runbook_hotspots) == 2  # noqa: PLR2004

    def test_hotspot_entity_and_concern_accessible(self) -> None:
        """Hotspot fields entity and concern are accessible."""
        hotspot = Hotspot(entity="DB pool", concern="oversaturation", source_ref="runbook.md:L42")
        assert hotspot.entity == "DB pool"
        assert hotspot.concern == "oversaturation"

    def test_model_validates_with_hotspot_list(self) -> None:
        """AttachmentPriors with hotspot list validates without error."""
        priors = AttachmentPriors.model_validate(
            {
                "expected_replica_counts": {"api": 3},
                "runbook_hotspots": [
                    {"entity": "DB pool", "concern": "oversaturation"},
                    {"entity": "JVM heap", "concern": "OOM killer"},
                ],
            }
        )
        assert len(priors.runbook_hotspots) == 2  # noqa: PLR2004
        assert priors.runbook_hotspots[0].entity == "DB pool"
        assert priors.runbook_hotspots[1].entity == "JVM heap"

    def test_default_fields_are_empty(self) -> None:
        """AttachmentPriors initialises with sensible empty defaults."""
        priors = AttachmentPriors()
        assert priors.expected_replica_counts == {}
        assert priors.expected_versions == {}
        assert priors.runbook_hotspots == []
        assert priors.historical_incidents == []

    def test_extra_fields_ignored(self) -> None:
        """extra='ignore' — unknown keys do not raise validation errors."""
        priors = AttachmentPriors.model_validate(
            {
                "expected_replica_counts": {"api": 3},
                "unknown_field": "should be ignored",
            }
        )
        assert priors.expected_replica_counts == {"api": 3}


# ── SH-ATT-H2: Seeded hypothesis from runbook hotspot ─────────────────────────


class TestSHATTH2:
    """Finding seeded by runbook hotspot → Source Evidence section rendered."""

    def test_source_evidence_section_present_for_enrichment_finding(self) -> None:
        """render_attachment_sections produces '## Source Evidence' for enrichment findings."""
        f = _basic_finding(title="DB Pool Oversaturation")
        ref = AttachmentRef(
            attachment_name="runbook.md",
            relevance="DB pool oversaturation — known failure mode",
        )
        object.__setattr__(f, "source_support", "live_with_attachment_enrichment")
        object.__setattr__(f, "attachment_references", [ref])
        report = _make_report(f)

        result = render_attachment_sections(report)

        assert "## Source Evidence" in result

    def test_attachment_name_mentioned_in_evidence_section(self) -> None:
        """The attachment_name appears in the rendered Source Evidence section."""
        f = _basic_finding(title="DB Pool Oversaturation")
        ref = AttachmentRef(
            attachment_name="runbook.md",
            relevance="DB pool oversaturation — known failure mode",
        )
        object.__setattr__(f, "source_support", "live_with_attachment_enrichment")
        object.__setattr__(f, "attachment_references", [ref])
        report = _make_report(f)

        result = render_attachment_sections(report)

        assert "runbook.md" in result

    def test_relevance_text_present_in_evidence_section(self) -> None:
        """The relevance note from AttachmentRef is rendered in the section."""
        f = _basic_finding(title="DB Pool Oversaturation")
        ref = AttachmentRef(
            attachment_name="runbook.md",
            relevance="DB pool oversaturation — known failure mode",
        )
        object.__setattr__(f, "source_support", "live_with_attachment_enrichment")
        object.__setattr__(f, "attachment_references", [ref])
        report = _make_report(f)

        result = render_attachment_sections(report)

        assert "DB pool oversaturation" in result

    def test_live_with_attachment_enrichment_appears_in_verified_table(self) -> None:
        """live_with_attachment_enrichment findings also appear in Verified Expectations table."""
        f = _basic_finding(title="DB Pool Hotspot Hypothesis")
        ref = AttachmentRef(attachment_name="runbook.md", relevance="hotspot from runbook")
        object.__setattr__(f, "source_support", "live_with_attachment_enrichment")
        object.__setattr__(f, "attachment_references", [ref])
        report = _make_report(f)

        result = render_attachment_sections(report)

        assert "## Verified Expectations" in result
        assert "DB Pool Hotspot Hypothesis" in result

    def test_supporting_evidence_ref_rendered_with_attachment_name(self) -> None:
        """EvidenceRef with attachment_name renders the attachment name in Source Evidence."""
        f = _basic_finding(title="Hotspot from Runbook")
        ev = EvidenceRef(
            source="attachment",
            excerpt="DB pool oversaturation",
            attachment_name="runbook.md",
            line_ref="hotspot-1",
        )
        object.__setattr__(f, "supporting_evidence", [ev])
        report = _make_report(f)

        result = render_attachment_sections(report)

        assert "## Source Evidence" in result
        assert "runbook.md" in result
        assert "DB pool oversaturation" in result

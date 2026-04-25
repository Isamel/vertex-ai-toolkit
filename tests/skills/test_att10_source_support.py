"""Tests for SPEC-ATT-10 §6.5.2 — source_support field and ref models.

Covers:
- EvidenceRef, ContradictionRef, EnrichmentRef, AttachmentRef defaults and population
- Finding.source_support default and exclusion from Gemini schema
- All 7 Literal values accepted on Finding
- Supporting ref fields excluded from Gemini schema
- build_attachment_seeded_section: empty priors, hotspot injection, incident injection
"""

from __future__ import annotations

import json

from vaig.skills.service_health.prompts import build_attachment_seeded_section
from vaig.skills.service_health.schema import (
    AttachmentPriors,
    AttachmentRef,
    ContradictionRef,
    EnrichmentRef,
    EvidenceRef,
    Finding,
    HealthReportGeminiSchema,
    HistoricalIncident,
    Hotspot,
)

# ── EvidenceRef ───────────────────────────────────────────────


class TestEvidenceRef:
    def test_defaults(self) -> None:
        ref = EvidenceRef()
        assert ref.source == ""
        assert ref.excerpt == ""
        assert ref.attachment_name is None
        assert ref.line_ref is None

    def test_all_fields_populated(self) -> None:
        ref = EvidenceRef(
            source="attachment",
            excerpt="Pod restarts > 5",
            attachment_name="runbook.md",
            line_ref="L42",
        )
        assert ref.source == "attachment"
        assert ref.excerpt == "Pod restarts > 5"
        assert ref.attachment_name == "runbook.md"
        assert ref.line_ref == "L42"

    def test_extra_fields_ignored(self) -> None:
        ref = EvidenceRef(source="live", unknown_field="ignored")  # type: ignore[call-arg]
        assert ref.source == "live"
        assert not hasattr(ref, "unknown_field")


# ── ContradictionRef ──────────────────────────────────────────


class TestContradictionRef:
    def test_defaults(self) -> None:
        ref = ContradictionRef()
        assert ref.expected == ""
        assert ref.observed == ""
        assert ref.attachment_name is None

    def test_all_fields_populated(self) -> None:
        ref = ContradictionRef(
            expected="replicas=3",
            observed="replicas=1",
            attachment_name="values.yaml",
        )
        assert ref.expected == "replicas=3"
        assert ref.observed == "replicas=1"
        assert ref.attachment_name == "values.yaml"


# ── EnrichmentRef ─────────────────────────────────────────────


class TestEnrichmentRef:
    def test_defaults(self) -> None:
        ref = EnrichmentRef()
        assert ref.detail == ""
        assert ref.attachment_name is None

    def test_all_fields_populated(self) -> None:
        ref = EnrichmentRef(
            detail="Run: kubectl rollout restart deploy/checkout",
            attachment_name="runbook.md",
        )
        assert ref.detail == "Run: kubectl rollout restart deploy/checkout"
        assert ref.attachment_name == "runbook.md"


# ── AttachmentRef ─────────────────────────────────────────────


class TestAttachmentRef:
    def test_defaults(self) -> None:
        ref = AttachmentRef()
        assert ref.attachment_name == ""
        assert ref.relevance == ""

    def test_populated(self) -> None:
        ref = AttachmentRef(attachment_name="postmortem.md", relevance="Root cause matched")
        assert ref.attachment_name == "postmortem.md"
        assert ref.relevance == "Root cause matched"


# ── Finding.source_support ────────────────────────────────────


class TestFindingSourceSupport:
    def test_default_is_live_only(self) -> None:
        f = Finding(id="x", title="t", severity="INFO")
        assert f.source_support == "live_only"

    def test_source_support_excluded_from_gemini_schema(self) -> None:
        schema = HealthReportGeminiSchema.model_json_schema()
        finding_props = schema["$defs"]["Finding"]["properties"]
        assert "source_support" not in finding_props

    def test_supporting_evidence_excluded_from_gemini_schema(self) -> None:
        schema = HealthReportGeminiSchema.model_json_schema()
        finding_props = schema["$defs"]["Finding"]["properties"]
        assert "supporting_evidence" not in finding_props

    def test_ref_fields_excluded_from_gemini_schema(self) -> None:
        schema = HealthReportGeminiSchema.model_json_schema()
        finding_props = schema["$defs"]["Finding"]["properties"]
        for field_name in ("contradictions", "enrichments", "attachment_references"):
            assert field_name not in finding_props, (
                f"Field '{field_name}' should be excluded from Gemini schema but was found"
            )

    def test_finding_accepts_all_source_support_values(self) -> None:
        values = [
            "live_only",
            "attachment_only",
            "live_and_attachment_corroborated",
            "live_matches_expected_state",
            "live_with_attachment_enrichment",
            "live_vs_attachment_contradicts",
            "live_matches_known_incident_pattern",
        ]
        for v in values:
            f = Finding(id="x", title="t", severity="INFO", source_support=v)  # type: ignore[call-arg]
            assert f.source_support == v, f"source_support={v!r} not accepted"

    def test_ref_fields_default_to_empty_lists(self) -> None:
        f = Finding(id="x", title="t", severity="INFO")
        assert f.supporting_evidence == []
        assert f.contradictions == []
        assert f.enrichments == []
        assert f.attachment_references == []


# ── build_attachment_seeded_section ──────────────────────────


class TestBuildAttachmentSeededSection:
    def test_returns_string_for_empty_priors(self) -> None:
        priors = AttachmentPriors()
        result = build_attachment_seeded_section(priors.model_dump_json())
        # empty priors → no hotspots/incidents/etc → returns empty string
        assert isinstance(result, str)

    def test_returns_empty_for_blank_input(self) -> None:
        result = build_attachment_seeded_section("")
        assert result == ""

    def test_returns_empty_for_invalid_json(self) -> None:
        result = build_attachment_seeded_section("not-json")
        assert result == ""

    def test_includes_hotspot_entity(self) -> None:
        priors = AttachmentPriors(
            runbook_hotspots=[
                Hotspot(entity="DB connection pool", concern="oversaturation", source_ref="runbook.md:L42")
            ]
        )
        result = build_attachment_seeded_section(priors.model_dump_json())
        assert "DB connection pool" in result

    def test_includes_historical_incident(self) -> None:
        priors = AttachmentPriors(
            historical_incidents=[
                HistoricalIncident(
                    symptom_pattern="latency spike + OOM kills",
                    root_cause="memory leak in worker pool",
                    fix_applied="redeploy with -Xmx512m",
                )
            ]
        )
        result = build_attachment_seeded_section(priors.model_dump_json())
        assert "latency spike + OOM kills" in result

    def test_section_header_present_when_priors_non_empty(self) -> None:
        priors = AttachmentPriors(runbook_hotspots=[Hotspot(entity="checkout-service", concern="high error rate")])
        result = build_attachment_seeded_section(priors.model_dump_json())
        assert "Attachment-Seeded Investigation Directions" in result

    def test_accepts_raw_json_dict(self) -> None:
        raw = json.dumps(
            {
                "hotspots": [{"entity": "api-gateway", "concern": "timeout", "source_ref": ""}],
                "historical_incidents": [],
                "change_signals": [],
                "narrative_hints": [],
            }
        )
        result = build_attachment_seeded_section(raw)
        assert "api-gateway" in result

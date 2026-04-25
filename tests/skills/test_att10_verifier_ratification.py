"""Tests for SPEC-ATT-10 §6.5.3 — Verifier ratification pass.

Covers:
- FindingRatification and RatificationResult model defaults and validation
- apply_ratification: happy path, invalid source_support fallback,
  invalid confidence_override ignored, no-match finding skipped,
  empty JSON, malformed JSON, non-array JSON
- build_verifier_ratification_section: no-op when priors absent, section present when priors set
- extract_ratification_json: present block, absent block, empty array block
- HealthReport.ratification_json field exists with default ""
- HealthReportGeminiSchema includes ratification_json in its Gemini schema
"""

from __future__ import annotations

import json

from vaig.skills.service_health.prompts import (
    build_verifier_ratification_section,
    extract_ratification_json,
)
from vaig.skills.service_health.schema import (
    Finding,
    FindingRatification,
    HealthReport,
    HealthReportGeminiSchema,
    RatificationResult,
    apply_ratification,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _minimal_finding(title: str = "Pod CrashLoopBackOff") -> Finding:
    return Finding(id="test-id", title=title, severity="HIGH")


def _minimal_report(findings: list[Finding] | None = None) -> HealthReport:
    from vaig.skills.service_health.schema import ExecutiveSummary, OverallStatus

    return HealthReport(
        executive_summary=ExecutiveSummary(
            overall_status=OverallStatus.DEGRADED,
            scope="Namespace: default",
            summary_text="Test report",
        ),
        findings=findings or [],
    )


# ── FindingRatification ───────────────────────────────────────────────────────


class TestFindingRatification:
    def test_defaults(self) -> None:
        fr = FindingRatification()
        assert fr.finding_title == ""
        assert fr.ratified_source_support == "live_only"
        assert fr.confidence_override is None
        assert fr.ratification_note == ""

    def test_extra_fields_ignored(self) -> None:
        fr = FindingRatification.model_validate({"finding_title": "X", "unknown_field": "should be ignored"})
        assert fr.finding_title == "X"
        assert not hasattr(fr, "unknown_field")

    def test_all_fields(self) -> None:
        fr = FindingRatification(
            finding_title="CrashLoop",
            ratified_source_support="live_and_attachment_corroborated",
            confidence_override="CONFIRMED",
            ratification_note="kubectl confirmed image v2.3.1",
        )
        assert fr.ratified_source_support == "live_and_attachment_corroborated"
        assert fr.confidence_override == "CONFIRMED"


# ── RatificationResult ────────────────────────────────────────────────────────


class TestRatificationResult:
    def test_defaults(self) -> None:
        rr = RatificationResult()
        assert rr.items == []

    def test_extra_fields_ignored(self) -> None:
        rr = RatificationResult.model_validate({"items": [], "extra": "ignored"})
        assert rr.items == []

    def test_with_items(self) -> None:
        rr = RatificationResult(items=[FindingRatification(finding_title="T1")])
        assert len(rr.items) == 1
        assert rr.items[0].finding_title == "T1"


# ── apply_ratification ────────────────────────────────────────────────────────


class TestApplyRatification:
    def test_happy_path_source_support(self) -> None:
        finding = _minimal_finding("Pod CrashLoopBackOff")
        report = _minimal_report([finding])
        ratif_json = json.dumps(
            [
                {
                    "finding_title": "Pod CrashLoopBackOff",
                    "ratified_source_support": "live_and_attachment_corroborated",
                    "confidence_override": None,
                    "ratification_note": "kubectl confirmed restart count matches runbook threshold",
                }
            ]
        )
        result = apply_ratification(report, ratif_json)
        assert result.findings[0].source_support == "live_and_attachment_corroborated"

    def test_happy_path_confidence_override(self) -> None:
        finding = _minimal_finding("Image Version Mismatch")
        report = _minimal_report([finding])
        ratif_json = json.dumps(
            [
                {
                    "finding_title": "Image Version Mismatch",
                    "ratified_source_support": "live_vs_attachment_contradicts",
                    "confidence_override": "CONFIRMED",
                    "ratification_note": "kubectl shows v2.1 vs runbook v2.3",
                }
            ]
        )
        result = apply_ratification(report, ratif_json)
        assert result.findings[0].source_support == "live_vs_attachment_contradicts"
        assert result.findings[0].confidence == "CONFIRMED"

    def test_invalid_source_support_falls_back_to_live_only(self) -> None:
        finding = _minimal_finding("Memory Pressure")
        report = _minimal_report([finding])
        ratif_json = json.dumps(
            [
                {
                    "finding_title": "Memory Pressure",
                    "ratified_source_support": "INVALID_VALUE",
                    "confidence_override": None,
                    "ratification_note": "test",
                }
            ]
        )
        result = apply_ratification(report, ratif_json)
        assert result.findings[0].source_support == "live_only"

    def test_invalid_confidence_override_ignored(self) -> None:
        finding = _minimal_finding("DNS Timeout")
        original_confidence = finding.confidence
        report = _minimal_report([finding])
        ratif_json = json.dumps(
            [
                {
                    "finding_title": "DNS Timeout",
                    "ratified_source_support": "live_only",
                    "confidence_override": "BOGUS_LEVEL",
                    "ratification_note": "test",
                }
            ]
        )
        result = apply_ratification(report, ratif_json)
        assert result.findings[0].confidence == original_confidence

    def test_no_matching_finding_skipped(self) -> None:
        finding = _minimal_finding("Real Finding")
        report = _minimal_report([finding])
        ratif_json = json.dumps(
            [
                {
                    "finding_title": "Non Existent Finding",
                    "ratified_source_support": "live_and_attachment_corroborated",
                    "confidence_override": None,
                    "ratification_note": "test",
                }
            ]
        )
        result = apply_ratification(report, ratif_json)
        # Real Finding unchanged
        assert result.findings[0].source_support == "live_only"

    def test_case_insensitive_title_match(self) -> None:
        finding = _minimal_finding("Pod CrashLoopBackOff")
        report = _minimal_report([finding])
        ratif_json = json.dumps(
            [
                {
                    "finding_title": "POD CRASHLOOPBACKOFF",
                    "ratified_source_support": "live_matches_expected_state",
                    "confidence_override": None,
                    "ratification_note": "matched case-insensitively",
                }
            ]
        )
        result = apply_ratification(report, ratif_json)
        assert result.findings[0].source_support == "live_matches_expected_state"

    def test_empty_ratification_json_noop(self) -> None:
        finding = _minimal_finding("Stable Finding")
        report = _minimal_report([finding])
        result = apply_ratification(report, "")
        assert result.findings[0].source_support == "live_only"

    def test_empty_array_noop(self) -> None:
        finding = _minimal_finding("Stable Finding")
        report = _minimal_report([finding])
        result = apply_ratification(report, "[]")
        assert result.findings[0].source_support == "live_only"

    def test_malformed_json_returns_original(self) -> None:
        finding = _minimal_finding("Stable Finding")
        report = _minimal_report([finding])
        result = apply_ratification(report, "this is not json {{{")
        assert result.findings[0].source_support == "live_only"

    def test_non_array_json_returns_original(self) -> None:
        finding = _minimal_finding("Stable Finding")
        report = _minimal_report([finding])
        result = apply_ratification(report, '{"not": "an array"}')
        assert result.findings[0].source_support == "live_only"

    def test_all_valid_source_support_values(self) -> None:
        valid_values = [
            "live_only",
            "attachment_only",
            "live_and_attachment_corroborated",
            "live_matches_expected_state",
            "live_with_attachment_enrichment",
            "live_vs_attachment_contradicts",
            "live_matches_known_incident_pattern",
        ]
        for val in valid_values:
            finding = _minimal_finding("Test Finding")
            report = _minimal_report([finding])
            ratif_json = json.dumps([{"finding_title": "Test Finding", "ratified_source_support": val}])
            result = apply_ratification(report, ratif_json)
            assert result.findings[0].source_support == val, f"Failed for {val!r}"

    def test_multiple_findings_ratified(self) -> None:
        f1 = _minimal_finding("Finding One")
        f2 = _minimal_finding("Finding Two")
        report = _minimal_report([f1, f2])
        ratif_json = json.dumps(
            [
                {
                    "finding_title": "Finding One",
                    "ratified_source_support": "live_and_attachment_corroborated",
                    "confidence_override": "CONFIRMED",
                },
                {
                    "finding_title": "Finding Two",
                    "ratified_source_support": "live_vs_attachment_contradicts",
                    "confidence_override": "MEDIUM",
                },
            ]
        )
        result = apply_ratification(report, ratif_json)
        assert result.findings[0].source_support == "live_and_attachment_corroborated"
        assert result.findings[0].confidence == "CONFIRMED"
        assert result.findings[1].source_support == "live_vs_attachment_contradicts"
        assert result.findings[1].confidence == "MEDIUM"

    def test_no_findings_report_noop(self) -> None:
        report = _minimal_report([])
        ratif_json = json.dumps([{"finding_title": "Anything", "ratified_source_support": "live_only"}])
        result = apply_ratification(report, ratif_json)
        assert result.findings == []


# ── build_verifier_ratification_section ──────────────────────────────────────


class TestBuildVerifierRatificationSection:
    def test_no_priors_returns_empty(self) -> None:
        assert build_verifier_ratification_section(None) == ""
        assert build_verifier_ratification_section("") == ""

    def test_with_priors_returns_section(self) -> None:
        priors_json = '{"hotspots": [], "historical_incidents": [], "change_signals": [], "narrative_hints": []}'
        section = build_verifier_ratification_section(priors_json)
        assert "RATIFICATION_JSON" in section
        assert "END_RATIFICATION_JSON" in section
        assert "ratified_source_support" in section
        assert "confidence_override" in section

    def test_section_contains_allowed_values(self) -> None:
        section = build_verifier_ratification_section("{}")
        assert "live_and_attachment_corroborated" in section
        assert "live_vs_attachment_contradicts" in section
        assert "live_matches_expected_state" in section


# ── extract_ratification_json ─────────────────────────────────────────────────


class TestExtractRatificationJson:
    def test_extracts_present_block(self) -> None:
        verifier_text = """
## Verified Findings
Some verifier text here.

RATIFICATION_JSON
[{"finding_title": "CrashLoop", "ratified_source_support": "live_and_attachment_corroborated"}]
END_RATIFICATION_JSON

## Verification Summary
Done.
"""
        raw = extract_ratification_json(verifier_text)
        assert raw.startswith("[")
        parsed = json.loads(raw)
        assert parsed[0]["finding_title"] == "CrashLoop"

    def test_absent_block_returns_empty(self) -> None:
        verifier_text = "No ratification block here."
        assert extract_ratification_json(verifier_text) == ""

    def test_empty_array_block(self) -> None:
        verifier_text = "RATIFICATION_JSON\n[]\nEND_RATIFICATION_JSON"
        raw = extract_ratification_json(verifier_text)
        assert raw == "[]"

    def test_multiline_json_array(self) -> None:
        payload = json.dumps(
            [
                {"finding_title": "A", "ratified_source_support": "live_only"},
                {"finding_title": "B", "ratified_source_support": "attachment_only"},
            ],
            indent=2,
        )
        verifier_text = f"Some text\nRATIFICATION_JSON\n{payload}\nEND_RATIFICATION_JSON\nMore text"
        raw = extract_ratification_json(verifier_text)
        parsed = json.loads(raw)
        assert len(parsed) == 2
        assert parsed[1]["finding_title"] == "B"


# ── HealthReport.ratification_json field ─────────────────────────────────────


class TestHealthReportRatificationJsonField:
    def test_default_is_empty_string(self) -> None:
        report = _minimal_report()
        assert report.ratification_json == ""

    def test_can_be_set(self) -> None:
        report = _minimal_report()
        report2 = report.model_copy(update={"ratification_json": '[{"x": 1}]'})
        assert report2.ratification_json == '[{"x": 1}]'

    def test_included_in_gemini_schema(self) -> None:
        """ratification_json must be present in the Gemini schema so the reporter can populate it."""
        schema = HealthReportGeminiSchema.model_json_schema()
        props = schema.get("properties", {})
        assert "ratification_json" in props, (
            "ratification_json must be in HealthReportGeminiSchema properties so the reporter can populate it"
        )

    def test_model_validate_json_round_trip(self) -> None:
        """model_validate_json accepts ratification_json and preserves it."""
        payload = '[{"finding_title": "Test", "ratified_source_support": "live_only"}]'
        report = _minimal_report()
        dumped = report.model_dump_json()
        # Inject ratification_json manually via dict round-trip
        import json as _json

        d = _json.loads(dumped)
        d["ratification_json"] = payload
        rebuilt = HealthReport.model_validate(_json.loads(_json.dumps(d)))
        assert rebuilt.ratification_json == payload

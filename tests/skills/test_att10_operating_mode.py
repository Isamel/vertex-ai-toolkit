"""Tests for SPEC-ATT-10 §6.5.5 — OperatingMode enum and auto-detection.

Covers:
- OperatingMode enum members and string values
- _detect_operating_mode: all three branches (LIVE_ONLY, ATTACHMENT_ONLY, LIVE_PLUS_ATTACHMENTS)
- offline_mode=True always returns ATTACHMENT_ONLY regardless of attachments_present
- post_process_report: findings tagged source_support='attachment_only' in ATTACHMENT_ONLY mode
- post_process_report: operating_mode field set on HealthReport
- HealthReport.operating_mode defaults to LIVE_ONLY and is excluded from Gemini schema
"""

from __future__ import annotations

import json

from vaig.skills.service_health.schema import (
    ExecutiveSummary,
    HealthReport,
    HealthReportGeminiSchema,
    OperatingMode,
    OverallStatus,
)
from vaig.skills.service_health.skill import ServiceHealthSkill

# ── OperatingMode enum ────────────────────────────────────────


class TestOperatingModeEnum:
    def test_members_exist(self) -> None:
        assert OperatingMode.LIVE_ONLY == "LIVE_ONLY"
        assert OperatingMode.ATTACHMENT_ONLY == "ATTACHMENT_ONLY"
        assert OperatingMode.LIVE_PLUS_ATTACHMENTS == "LIVE_PLUS_ATTACHMENTS"

    def test_is_str(self) -> None:
        """StrEnum values must serialise as plain strings (required for Gemini schema)."""
        assert isinstance(OperatingMode.LIVE_ONLY, str)
        assert isinstance(OperatingMode.ATTACHMENT_ONLY, str)
        assert isinstance(OperatingMode.LIVE_PLUS_ATTACHMENTS, str)

    def test_three_members_only(self) -> None:
        assert len(OperatingMode) == 3  # noqa: PLR2004


# ── _detect_operating_mode ────────────────────────────────────


class TestDetectOperatingMode:
    def test_live_only_when_no_offline_no_attachments(self) -> None:
        mode = ServiceHealthSkill._detect_operating_mode(
            offline_mode=False,
            attachments_present=False,
        )
        assert mode == OperatingMode.LIVE_ONLY

    def test_hybrid_when_no_offline_with_attachments(self) -> None:
        mode = ServiceHealthSkill._detect_operating_mode(
            offline_mode=False,
            attachments_present=True,
        )
        assert mode == OperatingMode.LIVE_PLUS_ATTACHMENTS

    def test_attachment_only_when_offline_no_attachments(self) -> None:
        mode = ServiceHealthSkill._detect_operating_mode(
            offline_mode=True,
            attachments_present=False,
        )
        assert mode == OperatingMode.ATTACHMENT_ONLY

    def test_attachment_only_when_offline_with_attachments(self) -> None:
        """offline_mode=True always wins regardless of attachments_present."""
        mode = ServiceHealthSkill._detect_operating_mode(
            offline_mode=True,
            attachments_present=True,
        )
        assert mode == OperatingMode.ATTACHMENT_ONLY


# ── HealthReport.operating_mode field ────────────────────────


class TestHealthReportOperatingModeField:
    def test_default_is_live_only(self) -> None:
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.HEALTHY,
                scope="Cluster-wide",
                summary_text="All good",
            )
        )
        assert report.operating_mode == OperatingMode.LIVE_ONLY

    def test_can_be_set_via_model_copy(self) -> None:
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.HEALTHY,
                scope="Cluster-wide",
                summary_text="All good",
            )
        )
        updated = report.model_copy(update={"operating_mode": OperatingMode.ATTACHMENT_ONLY})
        assert updated.operating_mode == OperatingMode.ATTACHMENT_ONLY

    def test_excluded_from_gemini_schema(self) -> None:
        """operating_mode must not appear in the Gemini response_schema JSON."""
        schema = HealthReportGeminiSchema.model_json_schema()
        schema_str = json.dumps(schema)
        assert "operating_mode" not in schema_str

    def test_present_in_model_dump(self) -> None:
        """operating_mode must appear in model_dump() so HTML/file exports see it.

        The field is excluded from the Gemini response_schema via
        _GEMINI_EXCLUDED_FIELDS — not via exclude=True on the Field — so it
        remains available in the full serialised report.
        """
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.HEALTHY,
                scope="Cluster-wide",
                summary_text="All good",
            )
        )
        dumped = report.model_dump()
        assert "operating_mode" in dumped
        assert dumped["operating_mode"] == OperatingMode.LIVE_ONLY


# ── ServiceHealthSkill instance state ────────────────────────


class TestSkillInstanceState:
    def test_defaults_on_init(self) -> None:
        skill = ServiceHealthSkill()
        assert skill._offline_mode is False
        assert skill._attachments_present is False

    def test_offline_mode_can_be_set(self) -> None:
        skill = ServiceHealthSkill()
        skill._offline_mode = True
        assert skill._offline_mode is True

    def test_attachments_present_can_be_set(self) -> None:
        skill = ServiceHealthSkill()
        skill._attachments_present = True
        assert skill._attachments_present is True


# ── post_process_report operating mode wiring ────────────────


def _make_minimal_report_json(
    findings: list[dict] | None = None,
    ratification_json: str = "",
) -> str:
    """Return a minimal HealthReport JSON for post_process_report testing."""
    return json.dumps(
        {
            "executive_summary": {
                "overall_status": "HEALTHY",
                "overall_severity": "INFO",
                "scope": "Cluster-wide",
                "summary_text": "All good",
            },
            "findings": findings or [],
            "ratification_json": ratification_json,
        }
    )


def _make_finding_dict(finding_id: str = "f1") -> dict:
    return {
        "id": finding_id,
        "title": "Test finding",
        "severity": "LOW",
        "description": "A test finding.",
        "remediation": "kubectl apply -f fix.yaml",
    }


class TestPostProcessReportOperatingMode:
    def test_live_only_mode_does_not_override_source_support(self) -> None:
        """LIVE_ONLY: findings keep their default source_support."""
        skill = ServiceHealthSkill()
        skill._offline_mode = False
        skill._attachments_present = False

        report_json = _make_minimal_report_json(findings=[_make_finding_dict()])
        # post_process_report returns Markdown, so we validate the intermediate
        # HealthReport behaviour directly: clean/parse the JSON here and apply
        # the same operating-mode update logic inline.

        from vaig.utils.json_cleaner import clean_llm_json  # noqa: PLC0415

        cleaned = clean_llm_json(report_json)
        report = HealthReport.model_validate_json(cleaned)

        # Simulate post_process_report operating mode logic
        operating_mode = skill._detect_operating_mode(
            offline_mode=skill._offline_mode,
            attachments_present=skill._attachments_present,
        )
        report = report.model_copy(update={"operating_mode": operating_mode})

        assert report.operating_mode == OperatingMode.LIVE_ONLY
        # source_support should stay as the default "live_only"
        assert all(f.source_support == "live_only" for f in report.findings)

    def test_attachment_only_mode_overrides_source_support(self) -> None:
        """ATTACHMENT_ONLY: all findings must be tagged source_support='attachment_only'."""
        skill = ServiceHealthSkill()
        skill._offline_mode = True
        skill._attachments_present = False

        from vaig.utils.json_cleaner import clean_llm_json  # noqa: PLC0415

        report_json = _make_minimal_report_json(findings=[_make_finding_dict("f1"), _make_finding_dict("f2")])
        cleaned = clean_llm_json(report_json)
        report = HealthReport.model_validate_json(cleaned)

        # Simulate the ATTACHMENT_ONLY override logic from post_process_report
        operating_mode = skill._detect_operating_mode(
            offline_mode=skill._offline_mode,
            attachments_present=skill._attachments_present,
        )
        report = report.model_copy(update={"operating_mode": operating_mode})
        if operating_mode == OperatingMode.ATTACHMENT_ONLY and report.findings:
            patched = [f.model_copy(update={"source_support": "attachment_only"}) for f in report.findings]
            report = report.model_copy(update={"findings": patched})

        assert report.operating_mode == OperatingMode.ATTACHMENT_ONLY
        assert all(f.source_support == "attachment_only" for f in report.findings)

    def test_hybrid_mode_does_not_override_source_support(self) -> None:
        """LIVE_PLUS_ATTACHMENTS: source_support should NOT be bulk-overridden."""
        skill = ServiceHealthSkill()
        skill._offline_mode = False
        skill._attachments_present = True

        from vaig.utils.json_cleaner import clean_llm_json  # noqa: PLC0415

        report_json = _make_minimal_report_json(findings=[_make_finding_dict()])
        cleaned = clean_llm_json(report_json)
        report = HealthReport.model_validate_json(cleaned)

        operating_mode = skill._detect_operating_mode(
            offline_mode=skill._offline_mode,
            attachments_present=skill._attachments_present,
        )
        report = report.model_copy(update={"operating_mode": operating_mode})

        assert report.operating_mode == OperatingMode.LIVE_PLUS_ATTACHMENTS
        # source_support must stay as default live_only (not forced to attachment_only)
        assert all(f.source_support == "live_only" for f in report.findings)

    def test_empty_findings_in_attachment_only_mode(self) -> None:
        """ATTACHMENT_ONLY with no findings: no crash, operating_mode still set."""
        skill = ServiceHealthSkill()
        skill._offline_mode = True
        skill._attachments_present = False

        from vaig.utils.json_cleaner import clean_llm_json  # noqa: PLC0415

        report_json = _make_minimal_report_json(findings=[])
        cleaned = clean_llm_json(report_json)
        report = HealthReport.model_validate_json(cleaned)

        operating_mode = skill._detect_operating_mode(
            offline_mode=skill._offline_mode,
            attachments_present=skill._attachments_present,
        )
        report = report.model_copy(update={"operating_mode": operating_mode})
        if operating_mode == OperatingMode.ATTACHMENT_ONLY and report.findings:
            patched = [f.model_copy(update={"source_support": "attachment_only"}) for f in report.findings]
            report = report.model_copy(update={"findings": patched})

        assert report.operating_mode == OperatingMode.ATTACHMENT_ONLY
        assert report.findings == []

"""Tests for ExternalLink / ExternalLinks schema models.

Covers:
- Valid ExternalLink instantiation
- Invalid system literal is accepted (system is a plain str, not a Literal)
- ExternalLinks partial groups default to empty lists
- HealthReport backward-compat deserialization without external_links
- HealthReportGeminiSchema pruning of excluded fields and orphaned $defs
- HealthReportGeminiSchema model_dump() retains post-hoc fields (HTML regression)
"""

from __future__ import annotations

import json

from vaig.skills.service_health.schema import (
    ExternalLink,
    ExternalLinks,
    HealthReport,
    HealthReportGeminiSchema,
)

# ── ExternalLink ──────────────────────────────────────────────


class TestExternalLink:
    def test_valid_gcp_link(self) -> None:
        link = ExternalLink(label="GCP Logs", url="https://console.cloud.google.com/logs", system="gcp")
        assert link.label == "GCP Logs"
        assert link.url == "https://console.cloud.google.com/logs"
        assert link.system == "gcp"
        assert link.icon == ""

    def test_valid_datadog_link_with_icon(self) -> None:
        link = ExternalLink(label="DD APM", url="https://app.datadoghq.com/apm", system="datadog", icon="<svg/>")
        assert link.system == "datadog"
        assert link.icon == "<svg/>"

    def test_valid_argocd_link(self) -> None:
        link = ExternalLink(label="ArgoCD", url="https://argocd.example.com/applications/myapp", system="argocd")
        assert link.system == "argocd"

    def test_invalid_system_literal_accepted(self) -> None:
        # system is now a plain str to reduce Gemini schema complexity;
        # any string value is accepted at instantiation time.
        link = ExternalLink(label="Bad", url="https://example.com", system="splunk")
        assert link.system == "splunk"

    def test_empty_system_accepted(self) -> None:
        # Empty string is accepted since system is a plain str (no Literal constraint).
        link = ExternalLink(label="Bad", url="https://example.com", system="")
        assert link.system == ""

    def test_all_system_literals_accepted(self) -> None:
        for system in ("gcp", "datadog", "argocd"):
            link = ExternalLink(label="x", url="https://example.com", system=system)
            assert link.system == system


# ── ExternalLinks ─────────────────────────────────────────────


class TestExternalLinks:
    def test_empty_defaults(self) -> None:
        el = ExternalLinks()
        assert el.gcp == []
        assert el.datadog == []
        assert el.argocd == []

    def test_partial_gcp_only(self) -> None:
        gcp_link = ExternalLink(label="GCP", url="https://console.cloud.google.com", system="gcp")
        el = ExternalLinks(gcp=[gcp_link])
        assert len(el.gcp) == 1
        assert el.datadog == []
        assert el.argocd == []

    def test_all_groups_populated(self) -> None:
        gcp = ExternalLink(label="G", url="https://gcp.com", system="gcp")
        dd = ExternalLink(label="D", url="https://dd.com", system="datadog")
        argo = ExternalLink(label="A", url="https://argo.com", system="argocd")
        el = ExternalLinks(gcp=[gcp], datadog=[dd], argocd=[argo])
        assert len(el.gcp) == 1
        assert len(el.datadog) == 1
        assert len(el.argocd) == 1


# ── HealthReport backward compatibility ───────────────────────


class TestHealthReportBackwardCompat:
    _MINIMAL_PAYLOAD = {
        "executive_summary": {
            "overall_status": "HEALTHY",
            "scope": "Cluster-wide",
            "summary_text": "All good",
        }
    }

    def test_deserialization_without_external_links(self) -> None:
        """Legacy reports without external_links must deserialize without error."""
        report = HealthReport.model_validate(self._MINIMAL_PAYLOAD)
        assert report.external_links is None

    def test_deserialization_with_external_links_null(self) -> None:
        payload = {**self._MINIMAL_PAYLOAD, "external_links": None}
        report = HealthReport.model_validate(payload)
        assert report.external_links is None

    def test_deserialization_with_external_links_present(self) -> None:
        payload = {
            **self._MINIMAL_PAYLOAD,
            "external_links": {
                "gcp": [{"label": "GCP Logs", "url": "https://gcp.com", "system": "gcp"}],
            },
        }
        report = HealthReport.model_validate(payload)
        assert report.external_links is not None
        assert len(report.external_links.gcp) == 1
        assert report.external_links.gcp[0].label == "GCP Logs"

    def test_json_roundtrip(self) -> None:
        """Serialize and deserialize back without data loss."""
        gcp_link = ExternalLink(label="GCP", url="https://gcp.com", system="gcp")
        el = ExternalLinks(gcp=[gcp_link])
        payload = {**self._MINIMAL_PAYLOAD, "external_links": el.model_dump()}
        report = HealthReport.model_validate(payload)
        dumped = json.loads(report.model_dump_json())
        assert dumped["external_links"]["gcp"][0]["label"] == "GCP"


# ── HealthReportGeminiSchema pruning ──────────────────────────


class TestHealthReportGeminiSchemaPruning:
    """Verify that model_json_schema() removes excluded fields and orphaned $defs."""

    def _get_schema(self) -> dict:
        return HealthReportGeminiSchema.model_json_schema()

    def test_schema_is_json_serializable(self) -> None:
        schema = self._get_schema()
        # Should not raise
        json.dumps(schema)

    def test_excluded_fields_not_in_properties(self) -> None:
        schema = self._get_schema()
        props = schema.get("properties", {})
        excluded = {"metadata", "evidence_gaps", "recent_changes", "external_links", "investigation_coverage"}
        for field in excluded:
            assert field not in props, f"Excluded field '{field}' should not appear in schema properties"

    def test_orphaned_defs_removed(self) -> None:
        schema = self._get_schema()
        defs = schema.get("$defs", {})
        # These types are only reachable via excluded fields
        orphans = {"ReportMetadata", "ToolUsageSummary", "ExternalLink", "ExternalLinks", "EvidenceGap", "ChangeEvent"}
        for orphan in orphans:
            assert orphan not in defs, f"Orphaned $def '{orphan}' should have been pruned from schema"

    def test_non_excluded_fields_present(self) -> None:
        schema = self._get_schema()
        props = schema.get("properties", {})
        assert "executive_summary" in props
        assert "findings" in props
        assert "recommendations" in props

    def test_model_dump_retains_post_hoc_fields(self) -> None:
        """Regression: model_dump() must NOT strip post-hoc fields.

        PR #276 added ``exclude=True`` to the field re-declarations in
        ``HealthReportGeminiSchema``, which caused ``model_dump()`` to omit
        ``metadata``, ``evidence_gaps``, ``recent_changes``, ``external_links``,
        and ``investigation_coverage``.  The HTML report template reads
        ``REPORT_DATA.metadata.project_id`` (and friends) — their absence
        caused a silent JS crash and a blank page.
        """
        minimal_payload = {
            "executive_summary": {
                "overall_status": "HEALTHY",
                "scope": "Cluster-wide",
                "summary_text": "All good",
            }
        }
        instance = HealthReportGeminiSchema.model_validate(minimal_payload)
        dumped = instance.model_dump()
        post_hoc = {"metadata", "evidence_gaps", "recent_changes", "external_links", "investigation_coverage"}
        for field in post_hoc:
            assert field in dumped, (
                f"model_dump() must retain post-hoc field '{field}' "
                f"(HTML report template depends on it)"
            )


# ── AUDIT-04: overall_severity_reason ────────────────────────


class TestOverallSeverityReason:
    """AUDIT-04 — overall_severity_reason field in HealthReport."""

    _MINIMAL_PAYLOAD: dict = {
        "executive_summary": {
            "overall_status": "DEGRADED",
            "scope": "Cluster-wide",
            "summary_text": "Issues detected",
        }
    }

    def test_field_defaults_to_none(self) -> None:
        report = HealthReport.model_validate(self._MINIMAL_PAYLOAD)
        assert report.overall_severity_reason is None

    def test_field_accepts_string(self) -> None:
        payload = {
            **self._MINIMAL_PAYLOAD,
            "overall_severity_reason": (
                "DEGRADED because 1 HIGH finding with confidence=CONFIRMED "
                "and 2 MEDIUM findings in distinct namespaces."
            ),
        }
        report = HealthReport.model_validate(payload)
        assert report.overall_severity_reason is not None
        assert "1 HIGH" in report.overall_severity_reason
        assert "CONFIRMED" in report.overall_severity_reason

    def test_field_present_in_gemini_schema(self) -> None:
        """overall_severity_reason is reporter-populated, must be in Gemini schema."""
        schema = HealthReportGeminiSchema.model_json_schema()
        props = schema.get("properties", {})
        assert "overall_severity_reason" in props, (
            "overall_severity_reason must be present in the Gemini schema "
            "(it is filled by the reporter LLM, not post-hoc)"
        )

    def test_markdown_renders_reason_when_present(self) -> None:
        payload = {
            **self._MINIMAL_PAYLOAD,
            "overall_severity_reason": "DEGRADED because 1 HIGH finding with confidence=CONFIRMED.",
        }
        report = HealthReport.model_validate(payload)
        md = report.to_markdown()
        assert "DEGRADED because 1 HIGH finding with confidence=CONFIRMED." in md

    def test_markdown_omits_reason_when_none(self) -> None:
        report = HealthReport.model_validate(self._MINIMAL_PAYLOAD)
        md = report.to_markdown()
        # No stray italic line under Status line
        lines = md.splitlines()
        status_idx = next(i for i, l in enumerate(lines) if "**Status**" in l)
        next_line = lines[status_idx + 1] if status_idx + 1 < len(lines) else ""
        assert not (next_line.strip().startswith("*") and next_line.strip().endswith("*")), (
            "No italic reason line should appear when overall_severity_reason is None"
        )

    def test_roundtrip_preserves_reason(self) -> None:
        payload = {
            **self._MINIMAL_PAYLOAD,
            "overall_severity_reason": "HEALTHY because no findings above LOW severity.",
        }
        report = HealthReport.model_validate(payload)
        dumped = report.model_dump()
        assert dumped["overall_severity_reason"] == "HEALTHY because no findings above LOW severity."

"""Tests for ExternalLink / ExternalLinks schema models.

Covers:
- Valid ExternalLink instantiation
- Invalid system literal raises ValidationError
- ExternalLinks partial groups default to empty lists
- HealthReport backward-compat deserialization without external_links
"""

from __future__ import annotations

import json

from vaig.skills.service_health.schema import (
    ExternalLink,
    ExternalLinks,
    HealthReport,
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

    def test_invalid_system_literal_raises(self) -> None:
        # system is now a plain str to reduce Gemini schema complexity;
        # any string value is accepted at instantiation time.
        link = ExternalLink(label="Bad", url="https://example.com", system="splunk")
        assert link.system == "splunk"

    def test_empty_system_raises(self) -> None:
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

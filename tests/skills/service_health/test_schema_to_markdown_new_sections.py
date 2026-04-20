"""Tests for to_markdown() new sections: evidence_gaps, recent_changes, external_links, investigation_coverage."""

from __future__ import annotations

from vaig.skills.service_health.schema import (
    ChangeEvent,
    EvidenceGap,
    ExecutiveSummary,
    ExternalLink,
    ExternalLinks,
    HealthReport,
    OverallStatus,
)


def _make_report(**kwargs) -> HealthReport:
    return HealthReport(
        executive_summary=ExecutiveSummary(
            overall_status=OverallStatus.HEALTHY,
            scope="Namespace: default",
            summary_text="All good",
        ),
        **kwargs,
    )


class TestToMarkdownEvidenceGaps:
    """to_markdown() renders Evidence Gaps section when evidence_gaps is populated."""

    def test_evidence_gaps_section_present(self) -> None:
        report = _make_report(
            evidence_gaps=[
                EvidenceGap(source="deployment_metrics", reason="error", details="timeout after 30s"),
                EvidenceGap(source="pod_logs", reason="not_called"),
            ]
        )
        md = report.to_markdown()
        assert "## Evidence Gaps" in md

    def test_evidence_gaps_source_and_reason_rendered(self) -> None:
        report = _make_report(
            evidence_gaps=[
                EvidenceGap(source="deployment_metrics", reason="error", details="timeout"),
            ]
        )
        md = report.to_markdown()
        assert "deployment_metrics" in md
        assert "error" in md

    def test_evidence_gaps_details_rendered(self) -> None:
        report = _make_report(
            evidence_gaps=[
                EvidenceGap(source="metrics_server", reason="error", details="connection refused"),
            ]
        )
        md = report.to_markdown()
        assert "connection refused" in md

    def test_evidence_gaps_section_absent_when_empty(self) -> None:
        report = _make_report(evidence_gaps=[])
        md = report.to_markdown()
        assert "## Evidence Gaps" not in md

    def test_evidence_gaps_no_details_renders_cleanly(self) -> None:
        report = _make_report(
            evidence_gaps=[EvidenceGap(source="events", reason="empty_result")]
        )
        md = report.to_markdown()
        assert "events" in md
        assert "empty_result" in md


class TestToMarkdownRecentChanges:
    """to_markdown() renders What Changed Recently section when recent_changes is populated."""

    def test_recent_changes_section_present(self) -> None:
        report = _make_report(
            recent_changes=[
                ChangeEvent(
                    timestamp="2024-01-15T10:00:00Z",
                    type="deployment",
                    description="Deployed payment-svc v1.2.3",
                    correlation_to_issue="Image tag bump coincides with crash start",
                )
            ]
        )
        md = report.to_markdown()
        assert "## What Changed Recently" in md

    def test_recent_changes_timestamp_and_type_rendered(self) -> None:
        report = _make_report(
            recent_changes=[
                ChangeEvent(
                    timestamp="2024-01-15T10:00:00Z",
                    type="deployment",
                    description="Deployed v1.2.3",
                    correlation_to_issue="correlates with errors",
                )
            ]
        )
        md = report.to_markdown()
        assert "2024-01-15T10:00:00Z" in md
        assert "deployment" in md

    def test_recent_changes_description_rendered(self) -> None:
        report = _make_report(
            recent_changes=[
                ChangeEvent(
                    timestamp="2024-01-15T10:00:00Z",
                    type="config_change",
                    description="Updated resource limits for payment-svc",
                    correlation_to_issue="OOM started after this change",
                )
            ]
        )
        md = report.to_markdown()
        assert "Updated resource limits for payment-svc" in md

    def test_recent_changes_correlation_rendered(self) -> None:
        report = _make_report(
            recent_changes=[
                ChangeEvent(
                    timestamp="2024-01-15T10:00:00Z",
                    type="hpa_scaling",
                    description="HPA scaled down to 1 replica",
                    correlation_to_issue="Traffic spike followed scale-down",
                )
            ]
        )
        md = report.to_markdown()
        assert "Traffic spike followed scale-down" in md

    def test_recent_changes_absent_when_empty(self) -> None:
        report = _make_report(recent_changes=[])
        md = report.to_markdown()
        assert "## What Changed Recently" not in md


class TestToMarkdownExternalLinks:
    """to_markdown() renders Quick Links section when external_links is populated."""

    def test_quick_links_section_present(self) -> None:
        report = _make_report(
            external_links=ExternalLinks(
                gcp=[ExternalLink(label="GCP Console", url="https://console.cloud.google.com", system="gcp")]
            )
        )
        md = report.to_markdown()
        assert "## Quick Links" in md

    def test_quick_links_label_and_url_rendered(self) -> None:
        report = _make_report(
            external_links=ExternalLinks(
                gcp=[ExternalLink(label="GKE Workloads", url="https://console.cloud.google.com/gke", system="gcp")]
            )
        )
        md = report.to_markdown()
        assert "GKE Workloads" in md
        assert "https://console.cloud.google.com/gke" in md

    def test_quick_links_multiple_systems_rendered(self) -> None:
        report = _make_report(
            external_links=ExternalLinks(
                gcp=[ExternalLink(label="GCP", url="https://gcp.example.com", system="gcp")],
                datadog=[ExternalLink(label="Datadog", url="https://dd.example.com", system="datadog")],
                argocd=[ExternalLink(label="ArgoCD", url="https://argo.example.com", system="argocd")],
            )
        )
        md = report.to_markdown()
        assert "GCP" in md
        assert "Datadog" in md
        assert "ArgoCD" in md

    def test_quick_links_absent_when_none(self) -> None:
        report = _make_report(external_links=None)
        md = report.to_markdown()
        assert "## Quick Links" not in md

    def test_quick_links_absent_when_all_lists_empty(self) -> None:
        report = _make_report(external_links=ExternalLinks(gcp=[], datadog=[], argocd=[]))
        md = report.to_markdown()
        assert "## Quick Links" not in md


class TestToMarkdownInvestigationCoverage:
    """to_markdown() renders Investigation Coverage section when set."""

    def test_investigation_coverage_section_present(self) -> None:
        report = _make_report(investigation_coverage="9/12 signal sources checked")
        md = report.to_markdown()
        assert "## Investigation Coverage" in md

    def test_investigation_coverage_value_rendered(self) -> None:
        report = _make_report(investigation_coverage="9/12 signal sources checked")
        md = report.to_markdown()
        assert "9/12 signal sources checked" in md

    def test_investigation_coverage_absent_when_none(self) -> None:
        report = _make_report(investigation_coverage=None)
        md = report.to_markdown()
        assert "## Investigation Coverage" not in md

    def test_investigation_coverage_absent_when_empty_string(self) -> None:
        report = _make_report(investigation_coverage="")
        md = report.to_markdown()
        assert "## Investigation Coverage" not in md

"""Tests for FindingExporter — lookup, Jira mapping, dedup, not-found."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vaig.integrations.finding_exporter import FindingExporter

# ── Fixtures ─────────────────────────────────────────────────

_SAMPLE_FINDING = {
    "id": "crashloop-payment",
    "title": "CrashLoop in payment-svc",
    "severity": "HIGH",
    "category": "pod-health",
    "service": "payment-svc",
    "description": "Pod is crash-looping due to OOM.",
    "root_cause": "Memory limit too low.",
    "impact": "Payment processing unavailable.",
    "remediation": "Increase memory limit to 512Mi.",
    "evidence": ["OOMKilled events in pod logs"],
    "affected_resources": ["pod/payment-svc-abc123"],
}

_SAMPLE_REPORT_RECORD = {
    "timestamp": "2026-04-08T12:00:00+00:00",
    "run_id": "run-001",
    "report": {
        "findings": [_SAMPLE_FINDING],
    },
}


@pytest.fixture()
def mock_report_store() -> MagicMock:
    """Return a mock ReportStore."""
    store = MagicMock()
    store.read_reports.return_value = [_SAMPLE_REPORT_RECORD]
    return store


@pytest.fixture()
def mock_jira() -> MagicMock:
    """Return a mock JiraClient."""
    jira = MagicMock()
    jira.severity_field_mapping = {
        "CRITICAL": "Highest",
        "HIGH": "High",
        "MEDIUM": "Medium",
        "LOW": "Low",
        "INFO": "Lowest",
    }
    jira.issue_url.return_value = "https://myorg.atlassian.net/browse/OPS-42"
    return jira


# ── Finding lookup tests ────────────────────────────────────


class TestFindingLookup:
    """Tests for FindingExporter.find_finding."""

    def test_exact_match(self, mock_report_store: MagicMock) -> None:
        exporter = FindingExporter(report_store=mock_report_store)
        result = exporter.find_finding("crashloop-payment")
        assert result is not None
        finding, meta = result
        assert finding["id"] == "crashloop-payment"

    def test_substring_match(self, mock_report_store: MagicMock) -> None:
        exporter = FindingExporter(report_store=mock_report_store)
        result = exporter.find_finding("crashloop")
        assert result is not None
        finding, _ = result
        assert finding["id"] == "crashloop-payment"

    def test_not_found(self, mock_report_store: MagicMock) -> None:
        exporter = FindingExporter(report_store=mock_report_store)
        result = exporter.find_finding("nonexistent-slug")
        assert result is None

    def test_no_report_store(self) -> None:
        exporter = FindingExporter()
        result = exporter.find_finding("any-slug")
        assert result is None


# ── Export to Jira tests ────────────────────────────────────


class TestExportToJira:
    """Tests for FindingExporter.export to Jira target."""

    def test_export_creates_issue(
        self, mock_jira: MagicMock, mock_report_store: MagicMock
    ) -> None:
        mock_jira.search_existing.return_value = None
        mock_jira.create_issue.return_value = {"key": "OPS-42", "id": "10001"}

        exporter = FindingExporter(jira=mock_jira, report_store=mock_report_store)
        result = exporter.export("crashloop-payment", "jira")

        assert result.success is True
        assert result.target == "jira"
        assert result.key == "OPS-42"
        assert result.already_existed is False
        mock_jira.create_issue.assert_called_once()

    def test_export_dedup_detected(
        self, mock_jira: MagicMock, mock_report_store: MagicMock
    ) -> None:
        mock_jira.search_existing.return_value = "OPS-42"

        exporter = FindingExporter(jira=mock_jira, report_store=mock_report_store)
        result = exporter.export("crashloop-payment", "jira")

        assert result.success is True
        assert result.already_existed is True
        assert result.key == "OPS-42"
        mock_jira.create_issue.assert_not_called()

    def test_export_finding_not_found(
        self, mock_jira: MagicMock, mock_report_store: MagicMock
    ) -> None:
        exporter = FindingExporter(jira=mock_jira, report_store=mock_report_store)
        result = exporter.export("nonexistent", "jira")

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_export_jira_not_configured(self, mock_report_store: MagicMock) -> None:
        exporter = FindingExporter(jira=None, report_store=mock_report_store)
        result = exporter.export("crashloop-payment", "jira")

        assert result.success is False
        assert "not configured" in result.error.lower()

    def test_export_adds_evidence_comment(
        self, mock_jira: MagicMock, mock_report_store: MagicMock
    ) -> None:
        mock_jira.search_existing.return_value = None
        mock_jira.create_issue.return_value = {"key": "OPS-42", "id": "10001"}

        exporter = FindingExporter(jira=mock_jira, report_store=mock_report_store)
        exporter.export("crashloop-payment", "jira")

        mock_jira.add_comment.assert_called_once()
        comment_body = mock_jira.add_comment.call_args[0][1]
        assert "OOMKilled" in comment_body


# ── List findings tests ─────────────────────────────────────


class TestListFindings:
    """Tests for FindingExporter.list_findings."""

    def test_list_returns_findings(self, mock_report_store: MagicMock) -> None:
        exporter = FindingExporter(report_store=mock_report_store)
        findings = exporter.list_findings()

        assert len(findings) == 1
        assert findings[0]["id"] == "crashloop-payment"
        assert findings[0]["severity"] == "HIGH"

    def test_list_no_store(self) -> None:
        exporter = FindingExporter()
        findings = exporter.list_findings()
        assert findings == []

    def test_list_deduplicates_by_id(self, mock_report_store: MagicMock) -> None:
        # Two records with same finding
        mock_report_store.read_reports.return_value = [
            _SAMPLE_REPORT_RECORD,
            _SAMPLE_REPORT_RECORD,
        ]
        exporter = FindingExporter(report_store=mock_report_store)
        findings = exporter.list_findings()
        assert len(findings) == 1


# ── Unknown target test ─────────────────────────────────────


class TestExportUnknownTarget:
    """Test unknown export target."""

    def test_unknown_target(self, mock_report_store: MagicMock) -> None:
        exporter = FindingExporter(report_store=mock_report_store)
        # Need a finding that exists for the target check to be reached
        result = exporter.export("crashloop-payment", "unknown_target")
        assert result.success is False
        assert "unknown" in result.error.lower()

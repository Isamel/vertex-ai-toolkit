"""Tests for PagerDutyClient finding-level export methods."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vaig.core.config import PagerDutyConfig
from vaig.integrations.pagerduty import PagerDutyClient

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def pd_config() -> PagerDutyConfig:
    """Return a fully configured PagerDutyConfig."""
    return PagerDutyConfig(
        enabled=True,
        routing_key="test-routing-key",
        api_token="test-api-token",
        service_id="PSERVICE1",
    )


@pytest.fixture()
def client(pd_config: PagerDutyConfig) -> PagerDutyClient:
    """Return a PagerDutyClient with full config."""
    return PagerDutyClient(pd_config)


def _make_finding(**overrides: object) -> MagicMock:
    """Build a mock Finding with defaults."""
    from enum import Enum

    class MockSeverity(Enum):
        HIGH = "high"

    finding = MagicMock()
    finding.id = overrides.get("id", "crashloop-payment")
    finding.title = overrides.get("title", "CrashLoop in payment-svc")
    finding.severity = overrides.get("severity", MockSeverity.HIGH)
    finding.category = overrides.get("category", "pod-health")
    finding.service = overrides.get("service", "payment-svc")
    finding.description = overrides.get("description", "Pod crash-looping")
    finding.root_cause = overrides.get("root_cause", "OOM")
    finding.impact = overrides.get("impact", "Payments down")
    finding.remediation = overrides.get("remediation", "Increase memory")
    finding.evidence = overrides.get("evidence", ["OOMKilled"])
    finding.affected_resources = overrides.get("affected_resources", ["pod/abc"])
    return finding


# ── create_incident_from_finding tests ──────────────────────


class TestCreateIncidentFromFinding:
    """Tests for PagerDutyClient.create_incident_from_finding."""

    @patch("vaig.integrations.pagerduty.requests.get")
    @patch("vaig.integrations.pagerduty.requests.post")
    def test_creates_incident_with_dedup_key(
        self, mock_post: MagicMock, mock_get: MagicMock, client: PagerDutyClient
    ) -> None:
        # trigger_event succeeds
        mock_trigger_resp = MagicMock()
        mock_trigger_resp.raise_for_status = MagicMock()

        # find_incident_by_dedup_key returns an incident
        mock_search_resp = MagicMock()
        mock_search_resp.json.return_value = {
            "incidents": [{"id": "PD-INC-001"}]
        }
        mock_search_resp.raise_for_status = MagicMock()

        # add_incident_note succeeds
        mock_note_resp = MagicMock()
        mock_note_resp.raise_for_status = MagicMock()

        mock_post.side_effect = [mock_trigger_resp, mock_note_resp]
        mock_get.return_value = mock_search_resp

        finding = _make_finding()
        result = client.create_incident_from_finding(
            finding=finding, cluster_context="prod-us"
        )

        assert result.success is True
        assert result.target == "pagerduty"
        assert result.key == "PD-INC-001"

        # Verify dedup_key format includes cluster prefix
        trigger_call = mock_post.call_args_list[0]
        payload = trigger_call.kwargs.get("json") or trigger_call[1].get("json")
        assert payload["dedup_key"] == "prod-us:crashloop-payment"

    @patch("vaig.integrations.pagerduty.requests.get")
    @patch("vaig.integrations.pagerduty.requests.post")
    def test_dedup_key_without_cluster(
        self, mock_post: MagicMock, mock_get: MagicMock, client: PagerDutyClient
    ) -> None:
        mock_trigger_resp = MagicMock()
        mock_trigger_resp.raise_for_status = MagicMock()

        mock_search_resp = MagicMock()
        mock_search_resp.json.return_value = {"incidents": []}
        mock_search_resp.raise_for_status = MagicMock()

        mock_post.return_value = mock_trigger_resp
        mock_get.return_value = mock_search_resp

        finding = _make_finding()
        result = client.create_incident_from_finding(finding=finding)

        assert result.success is True
        # Without cluster_context, dedup_key is just finding.id
        trigger_call = mock_post.call_args_list[0]
        payload = trigger_call.kwargs.get("json") or trigger_call[1].get("json")
        assert payload["dedup_key"] == "crashloop-payment"

    @patch("vaig.integrations.pagerduty.requests.post")
    def test_trigger_event_failure_returns_error(
        self, mock_post: MagicMock, client: PagerDutyClient
    ) -> None:
        import requests as req

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = req.exceptions.HTTPError("fail")
        mock_post.return_value = mock_resp

        finding = _make_finding()
        result = client.create_incident_from_finding(finding=finding)

        assert result.success is False
        assert result.error

    @patch("vaig.integrations.pagerduty.requests.get")
    @patch("vaig.integrations.pagerduty.requests.post")
    def test_adds_formatted_note(
        self, mock_post: MagicMock, mock_get: MagicMock, client: PagerDutyClient
    ) -> None:
        mock_trigger_resp = MagicMock()
        mock_trigger_resp.raise_for_status = MagicMock()

        mock_search_resp = MagicMock()
        mock_search_resp.json.return_value = {
            "incidents": [{"id": "PD-INC-002"}]
        }
        mock_search_resp.raise_for_status = MagicMock()

        mock_note_resp = MagicMock()
        mock_note_resp.raise_for_status = MagicMock()

        mock_post.side_effect = [mock_trigger_resp, mock_note_resp]
        mock_get.return_value = mock_search_resp

        finding = _make_finding()
        client.create_incident_from_finding(finding=finding, cluster_context="prod")

        # Verify note was posted
        assert mock_post.call_count == 2
        note_call = mock_post.call_args_list[1]
        note_payload = note_call.kwargs.get("json") or note_call[1].get("json")
        assert "CrashLoop in payment-svc" in note_payload["note"]["content"]


# ── _format_finding_note tests ──────────────────────────────


class TestFormatFindingNote:
    """Tests for PagerDutyClient._format_finding_note."""

    def test_formats_all_fields(self) -> None:
        finding = _make_finding()
        note = PagerDutyClient._format_finding_note(finding)

        assert "CrashLoop in payment-svc" in note
        assert "pod-health" in note
        assert "payment-svc" in note
        assert "OOM" in note
        assert "OOMKilled" in note
        assert "pod/abc" in note

    def test_handles_empty_fields(self) -> None:
        finding = _make_finding(
            service="",
            description="",
            root_cause="",
            impact="",
            remediation=None,
            evidence=[],
            affected_resources=[],
        )
        note = PagerDutyClient._format_finding_note(finding)

        assert "CrashLoop in payment-svc" in note
        # Should not contain empty sections
        assert "Service:" not in note

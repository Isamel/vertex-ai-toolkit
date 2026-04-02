"""Tests for PagerDuty Events API v2 + REST API v2 client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from vaig.core.config import PagerDutyConfig
from vaig.integrations.pagerduty import _EVENTS_API_URL, PagerDutyClient

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def pd_config() -> PagerDutyConfig:
    """Return a fully configured PagerDutyConfig for tests."""
    return PagerDutyConfig(
        enabled=True,
        routing_key="test-routing-key",
        api_token="test-api-token",
        service_id="PSERVICE1",
        base_url="https://api.pagerduty.com",
    )


@pytest.fixture()
def pd_config_no_token() -> PagerDutyConfig:
    """Return a PagerDutyConfig without api_token (Events API only)."""
    return PagerDutyConfig(
        enabled=True,
        routing_key="test-routing-key",
        api_token="",
    )


@pytest.fixture()
def client(pd_config: PagerDutyConfig) -> PagerDutyClient:
    """Return a PagerDutyClient with full config."""
    return PagerDutyClient(pd_config)


@pytest.fixture()
def client_no_token(pd_config_no_token: PagerDutyConfig) -> PagerDutyClient:
    """Return a PagerDutyClient without api_token."""
    return PagerDutyClient(pd_config_no_token)


# ── Config auto-enable tests ────────────────────────────────


class TestPagerDutyConfigAutoEnable:
    """Tests for PagerDutyConfig auto-enable validator."""

    def test_auto_enable_when_routing_key_set(self) -> None:
        config = PagerDutyConfig(routing_key="some-key")
        assert config.enabled is True

    def test_stays_disabled_without_routing_key(self) -> None:
        config = PagerDutyConfig()
        assert config.enabled is False

    def test_explicit_enabled_true_stays(self) -> None:
        config = PagerDutyConfig(enabled=True, routing_key="key")
        assert config.enabled is True

    def test_severity_mapping_defaults(self) -> None:
        config = PagerDutyConfig()
        assert config.severity_mapping == {
            "critical": "critical",
            "high": "error",
            "medium": "warning",
            "low": "info",
        }

    def test_api_token_not_in_repr(self) -> None:
        config = PagerDutyConfig(api_token="secret-token", routing_key="key")
        assert "secret-token" not in repr(config)

    def test_routing_key_not_in_repr(self) -> None:
        config = PagerDutyConfig(routing_key="secret-key")
        assert "secret-key" not in repr(config)


# ── Events API v2 tests ─────────────────────────────────────


class TestTriggerEvent:
    """Tests for PagerDutyClient.trigger_event."""

    @patch("vaig.integrations.pagerduty.requests.post")
    def test_trigger_event_returns_dedup_key(
        self, mock_post: MagicMock, client: PagerDutyClient
    ) -> None:
        mock_post.return_value = MagicMock(status_code=202)
        mock_post.return_value.raise_for_status = MagicMock()

        dedup = client.trigger_event(
            summary="Test alert",
            severity="critical",
            source="test-service",
            dedup_key="my-dedup-key",
        )

        assert dedup == "my-dedup-key"
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs["json"]
        assert payload["routing_key"] == "test-routing-key"
        assert payload["event_action"] == "trigger"
        assert payload["dedup_key"] == "my-dedup-key"
        assert payload["payload"]["summary"] == "Test alert"
        assert payload["payload"]["severity"] == "critical"

    @patch("vaig.integrations.pagerduty.requests.post")
    def test_trigger_event_auto_generates_dedup_key(
        self, mock_post: MagicMock, client: PagerDutyClient
    ) -> None:
        mock_post.return_value = MagicMock(status_code=202)
        mock_post.return_value.raise_for_status = MagicMock()

        dedup = client.trigger_event(
            summary="Test alert",
            severity="warning",
            source="test-service",
        )

        assert dedup is not None
        assert len(dedup) == 36  # UUID format

    @patch("vaig.integrations.pagerduty.requests.post")
    def test_trigger_event_with_custom_details(
        self, mock_post: MagicMock, client: PagerDutyClient
    ) -> None:
        mock_post.return_value = MagicMock(status_code=202)
        mock_post.return_value.raise_for_status = MagicMock()

        details = {"cluster": "prod", "pods": 5}
        client.trigger_event(
            summary="Alert",
            severity="error",
            source="svc",
            dedup_key="k1",
            custom_details=details,
        )

        payload = mock_post.call_args.kwargs["json"]
        assert payload["payload"]["custom_details"] == details

    @patch("vaig.integrations.pagerduty.requests.post")
    def test_trigger_event_truncates_long_summary(
        self, mock_post: MagicMock, client: PagerDutyClient
    ) -> None:
        mock_post.return_value = MagicMock(status_code=202)
        mock_post.return_value.raise_for_status = MagicMock()

        long_summary = "x" * 2000
        client.trigger_event(
            summary=long_summary,
            severity="info",
            source="svc",
            dedup_key="k2",
        )

        payload = mock_post.call_args.kwargs["json"]
        assert len(payload["payload"]["summary"]) == 1024

    @patch("vaig.integrations.pagerduty.requests.post")
    def test_trigger_event_posts_to_events_api(
        self, mock_post: MagicMock, client: PagerDutyClient
    ) -> None:
        mock_post.return_value = MagicMock(status_code=202)
        mock_post.return_value.raise_for_status = MagicMock()

        client.trigger_event(
            summary="Alert", severity="critical", source="svc", dedup_key="k"
        )

        assert mock_post.call_args.args[0] == _EVENTS_API_URL


class TestAcknowledgeResolveEvents:
    """Tests for acknowledge and resolve event actions."""

    @patch("vaig.integrations.pagerduty.requests.post")
    def test_acknowledge_event(
        self, mock_post: MagicMock, client: PagerDutyClient
    ) -> None:
        mock_post.return_value = MagicMock(status_code=202)
        mock_post.return_value.raise_for_status = MagicMock()

        client.acknowledge_event("my-dedup")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["event_action"] == "acknowledge"
        assert payload["dedup_key"] == "my-dedup"

    @patch("vaig.integrations.pagerduty.requests.post")
    def test_resolve_event(
        self, mock_post: MagicMock, client: PagerDutyClient
    ) -> None:
        mock_post.return_value = MagicMock(status_code=202)
        mock_post.return_value.raise_for_status = MagicMock()

        client.resolve_event("my-dedup")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["event_action"] == "resolve"
        assert payload["dedup_key"] == "my-dedup"


# ── REST API v2 tests ────────────────────────────────────────


class TestFindIncidentByDedupKey:
    """Tests for PagerDutyClient.find_incident_by_dedup_key."""

    @patch("vaig.integrations.pagerduty.requests.get")
    def test_find_incident_returns_id(
        self, mock_get: MagicMock, client: PagerDutyClient
    ) -> None:
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.raise_for_status = MagicMock()
        mock_get.return_value.json.return_value = {
            "incidents": [{"id": "INC123"}]
        }

        result = client.find_incident_by_dedup_key("dedup-1")

        assert result == "INC123"
        mock_get.assert_called_once()
        assert mock_get.call_args.kwargs["params"] == {"incident_key": "dedup-1"}

    @patch("vaig.integrations.pagerduty.requests.get")
    def test_find_incident_returns_none_when_not_found(
        self, mock_get: MagicMock, client: PagerDutyClient
    ) -> None:
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.raise_for_status = MagicMock()
        mock_get.return_value.json.return_value = {"incidents": []}

        result = client.find_incident_by_dedup_key("nonexistent")
        assert result is None

    @patch("vaig.integrations.pagerduty.requests.get")
    def test_find_incident_handles_http_error(
        self, mock_get: MagicMock, client: PagerDutyClient
    ) -> None:
        mock_get.side_effect = requests.exceptions.HTTPError("500 Server Error")

        result = client.find_incident_by_dedup_key("dedup-1")
        assert result is None


class TestAddIncidentNote:
    """Tests for PagerDutyClient.add_incident_note."""

    @patch("vaig.integrations.pagerduty.requests.post")
    def test_add_note_sends_content(
        self, mock_post: MagicMock, client: PagerDutyClient
    ) -> None:
        mock_post.return_value = MagicMock(status_code=201)
        mock_post.return_value.raise_for_status = MagicMock()

        client.add_incident_note("INC123", "This is a note")

        call_kwargs = mock_post.call_args
        assert "INC123" in call_kwargs.args[0]
        assert call_kwargs.kwargs["json"]["note"]["content"] == "This is a note"

    @patch("vaig.integrations.pagerduty.requests.post")
    def test_add_note_uses_rest_headers(
        self, mock_post: MagicMock, client: PagerDutyClient
    ) -> None:
        mock_post.return_value = MagicMock(status_code=201)
        mock_post.return_value.raise_for_status = MagicMock()

        client.add_incident_note("INC123", "note")

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Token token=test-api-token"
        assert "pagerduty" in headers["Accept"]


# ── Graceful degradation (no api_token) ──────────────────────


class TestGracefulDegradation:
    """Tests for graceful degradation when api_token is not configured."""

    def test_find_incident_returns_none_without_token(
        self, client_no_token: PagerDutyClient
    ) -> None:
        result = client_no_token.find_incident_by_dedup_key("dedup-1")
        assert result is None

    def test_add_note_skips_without_token(
        self, client_no_token: PagerDutyClient
    ) -> None:
        # Should not raise — just logs a warning and returns
        client_no_token.add_incident_note("INC123", "note content")

    def test_attach_report_skips_without_token(
        self, client_no_token: PagerDutyClient
    ) -> None:
        mock_report = MagicMock()
        # Should not raise
        client_no_token.attach_report_to_incident("INC123", mock_report)
        # to_markdown should NOT be called since we bail early
        mock_report.to_markdown.assert_not_called()


# ── Error handling ───────────────────────────────────────────


class TestErrorHandling:
    """Tests for error handling in PagerDuty client."""

    @patch("vaig.integrations.pagerduty.requests.post")
    def test_trigger_event_raises_on_http_error(
        self, mock_post: MagicMock, client: PagerDutyClient
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "400 Bad Request"
        )
        mock_post.return_value = mock_resp

        with pytest.raises(requests.exceptions.HTTPError):
            client.trigger_event(
                summary="Alert", severity="critical", source="svc", dedup_key="k"
            )

    @patch("vaig.integrations.pagerduty.requests.post")
    def test_trigger_event_raises_on_timeout(
        self, mock_post: MagicMock, client: PagerDutyClient
    ) -> None:
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

        with pytest.raises(requests.exceptions.Timeout):
            client.trigger_event(
                summary="Alert", severity="critical", source="svc", dedup_key="k"
            )

    @patch("vaig.integrations.pagerduty.requests.post")
    def test_trigger_event_reraises_keyboard_interrupt(
        self, mock_post: MagicMock, client: PagerDutyClient
    ) -> None:
        mock_post.side_effect = KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            client.trigger_event(
                summary="Alert", severity="critical", source="svc", dedup_key="k"
            )

    @patch("vaig.integrations.pagerduty.requests.get")
    def test_find_incident_handles_timeout(
        self, mock_get: MagicMock, client: PagerDutyClient
    ) -> None:
        mock_get.side_effect = requests.exceptions.Timeout("Timeout")

        result = client.find_incident_by_dedup_key("dedup-1")
        assert result is None

    @patch("vaig.integrations.pagerduty.requests.post")
    def test_add_note_handles_error_gracefully(
        self, mock_post: MagicMock, client: PagerDutyClient
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
        mock_post.return_value = mock_resp

        # Should not raise — just logs
        client.add_incident_note("INC123", "note")


class TestAttachReport:
    """Tests for attach_report_to_incident."""

    @patch("vaig.integrations.pagerduty.requests.post")
    def test_attach_report_calls_to_markdown(
        self, mock_post: MagicMock, client: PagerDutyClient
    ) -> None:
        mock_post.return_value = MagicMock(status_code=201)
        mock_post.return_value.raise_for_status = MagicMock()

        mock_report = MagicMock()
        mock_report.to_markdown.return_value = "# Report\nContent here"

        client.attach_report_to_incident("INC123", mock_report)

        mock_report.to_markdown.assert_called_once()
        payload = mock_post.call_args.kwargs["json"]
        assert "# Report" in payload["note"]["content"]

    @patch("vaig.integrations.pagerduty.requests.post")
    def test_attach_report_truncates_long_markdown(
        self, mock_post: MagicMock, client: PagerDutyClient
    ) -> None:
        mock_post.return_value = MagicMock(status_code=201)
        mock_post.return_value.raise_for_status = MagicMock()

        mock_report = MagicMock()
        mock_report.to_markdown.return_value = "x" * 70000

        client.attach_report_to_incident("INC123", mock_report)

        payload = mock_post.call_args.kwargs["json"]
        assert len(payload["note"]["content"]) < 65536
        assert "truncated" in payload["note"]["content"]

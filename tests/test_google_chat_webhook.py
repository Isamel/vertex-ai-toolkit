"""Tests for Google Chat incoming webhook integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from vaig.core.config import GoogleChatConfig
from vaig.integrations.google_chat import (
    GoogleChatWebhook,
    _meets_threshold,
    _status_to_severity,
)

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def gc_config() -> GoogleChatConfig:
    """Return an enabled GoogleChatConfig for tests."""
    return GoogleChatConfig(
        enabled=True,
        webhook_url="https://chat.googleapis.com/v1/spaces/XXX/messages?key=YYY&token=ZZZ",
        notify_on=["critical", "high"],
    )


@pytest.fixture()
def gc_config_all_severities() -> GoogleChatConfig:
    """Return a GoogleChatConfig that notifies on all severities."""
    return GoogleChatConfig(
        enabled=True,
        webhook_url="https://chat.googleapis.com/v1/spaces/XXX/messages?key=YYY&token=ZZZ",
        notify_on=["critical", "high", "medium", "low", "info"],
    )


@pytest.fixture()
def webhook(gc_config: GoogleChatConfig) -> GoogleChatWebhook:
    """Return a GoogleChatWebhook instance."""
    return GoogleChatWebhook(gc_config)


@pytest.fixture()
def webhook_all(gc_config_all_severities: GoogleChatConfig) -> GoogleChatWebhook:
    """Return a GoogleChatWebhook that notifies on all severities."""
    return GoogleChatWebhook(gc_config_all_severities)


# ── Config auto-enable tests ────────────────────────────────


class TestGoogleChatConfigAutoEnable:
    """Tests for GoogleChatConfig auto-enable validator."""

    def test_auto_enable_when_webhook_url_set(self) -> None:
        config = GoogleChatConfig(webhook_url="https://chat.example.com/webhook")
        assert config.enabled is True

    def test_stays_disabled_without_webhook_url(self) -> None:
        config = GoogleChatConfig()
        assert config.enabled is False

    def test_default_notify_on(self) -> None:
        config = GoogleChatConfig()
        assert config.notify_on == ["critical", "high"]

    def test_webhook_url_not_in_repr(self) -> None:
        config = GoogleChatConfig(webhook_url="https://secret.url")
        assert "secret.url" not in repr(config)


# ── Severity threshold tests ─────────────────────────────────


class TestSeverityThreshold:
    """Tests for _meets_threshold helper."""

    def test_critical_meets_critical_high(self) -> None:
        assert _meets_threshold("CRITICAL", ["critical", "high"]) is True

    def test_high_meets_critical_high(self) -> None:
        assert _meets_threshold("HIGH", ["critical", "high"]) is True

    def test_medium_does_not_meet_critical_high(self) -> None:
        assert _meets_threshold("MEDIUM", ["critical", "high"]) is False

    def test_low_does_not_meet_critical_high(self) -> None:
        assert _meets_threshold("LOW", ["critical", "high"]) is False

    def test_info_does_not_meet_critical_high(self) -> None:
        assert _meets_threshold("INFO", ["critical", "high"]) is False

    def test_all_severities_meet_all_threshold(self) -> None:
        all_levels = ["critical", "high", "medium", "low", "info"]
        for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
            assert _meets_threshold(sev, all_levels) is True

    def test_empty_notify_on_blocks_all(self) -> None:
        assert _meets_threshold("CRITICAL", []) is False


class TestStatusToSeverity:
    """Tests for _status_to_severity mapping."""

    def test_critical_maps(self) -> None:
        assert _status_to_severity("CRITICAL") == "CRITICAL"

    def test_degraded_maps_to_high(self) -> None:
        assert _status_to_severity("DEGRADED") == "HIGH"

    def test_healthy_maps_to_info(self) -> None:
        assert _status_to_severity("HEALTHY") == "INFO"

    def test_unknown_maps_to_medium(self) -> None:
        assert _status_to_severity("UNKNOWN") == "MEDIUM"

    def test_unrecognised_defaults_to_medium(self) -> None:
        assert _status_to_severity("SOMETHING_ELSE") == "MEDIUM"


# ── send_alert_card tests ────────────────────────────────────


class TestSendAlertCard:
    """Tests for GoogleChatWebhook.send_alert_card."""

    @patch("vaig.integrations.google_chat.requests.post")
    def test_send_alert_card_basic(
        self, mock_post: MagicMock, webhook: GoogleChatWebhook
    ) -> None:
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        webhook.send_alert_card(
            title="Health Alert",
            severity="CRITICAL",
            service_name="api-server",
            summary="Service is down",
        )

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs["json"]

        # Verify Card v2 format
        assert "cardsV2" in payload
        card = payload["cardsV2"][0]["card"]
        assert card["header"]["title"] == "Health Alert"

        # Verify widgets contain severity and service
        widgets = card["sections"][0]["widgets"]
        widget_texts = str(widgets)
        assert "CRITICAL" in widget_texts
        assert "api-server" in widget_texts

    @patch("vaig.integrations.google_chat.requests.post")
    def test_send_alert_card_with_findings(
        self, mock_post: MagicMock, webhook: GoogleChatWebhook
    ) -> None:
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        webhook.send_alert_card(
            title="Alert",
            severity="HIGH",
            service_name="svc",
            summary="Issue",
            findings=["OOMKilled pods", "High latency"],
        )

        payload = mock_post.call_args.kwargs["json"]
        widgets = payload["cardsV2"][0]["card"]["sections"][0]["widgets"]
        widget_str = str(widgets)
        assert "OOMKilled pods" in widget_str
        assert "High latency" in widget_str

    @patch("vaig.integrations.google_chat.requests.post")
    def test_send_alert_card_with_pagerduty_url(
        self, mock_post: MagicMock, webhook: GoogleChatWebhook
    ) -> None:
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        webhook.send_alert_card(
            title="Alert",
            severity="CRITICAL",
            service_name="svc",
            summary="Issue",
            pagerduty_url="https://app.pagerduty.com/incidents/INC123",
        )

        payload = mock_post.call_args.kwargs["json"]
        widget_str = str(payload)
        assert "pagerduty.com" in widget_str

    @patch("vaig.integrations.google_chat.requests.post")
    def test_send_alert_card_skips_below_threshold(
        self, mock_post: MagicMock, webhook: GoogleChatWebhook
    ) -> None:
        """LOW severity should be skipped when notify_on is [critical, high]."""
        webhook.send_alert_card(
            title="Alert",
            severity="LOW",
            service_name="svc",
            summary="Minor issue",
        )

        mock_post.assert_not_called()

    @patch("vaig.integrations.google_chat.requests.post")
    def test_send_alert_card_posts_to_webhook_url(
        self, mock_post: MagicMock, webhook: GoogleChatWebhook
    ) -> None:
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        webhook.send_alert_card(
            title="Alert",
            severity="CRITICAL",
            service_name="svc",
            summary="Down",
        )

        assert "chat.googleapis.com" in mock_post.call_args.args[0]


# ── send_report_summary tests ────────────────────────────────


class TestSendReportSummary:
    """Tests for GoogleChatWebhook.send_report_summary."""

    def _make_report(
        self,
        status: str = "CRITICAL",
        summary: str = "Service degraded",
        issues: int = 3,
        critical: int = 1,
        warning: int = 2,
    ) -> MagicMock:
        """Build a mock HealthReport."""
        report = MagicMock()
        report.executive_summary.overall_status.value = status
        report.executive_summary.scope = "Namespace: prod"
        report.executive_summary.summary_text = summary
        report.executive_summary.issues_found = issues
        report.executive_summary.critical_count = critical
        report.executive_summary.warning_count = warning

        finding = MagicMock()
        finding.title = "OOMKilled pods"
        report.findings = [finding]
        return report

    @patch("vaig.integrations.google_chat.requests.post")
    def test_send_report_summary_critical(
        self, mock_post: MagicMock, webhook: GoogleChatWebhook
    ) -> None:
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        report = self._make_report(status="CRITICAL")
        webhook.send_report_summary(report, execution_time=12.5)

        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs["json"]
        widget_str = str(payload)
        assert "CRITICAL" in widget_str
        assert "12.5s" in widget_str

    @patch("vaig.integrations.google_chat.requests.post")
    def test_send_report_summary_healthy_skipped(
        self, mock_post: MagicMock, webhook: GoogleChatWebhook
    ) -> None:
        """HEALTHY maps to INFO severity, which is below threshold."""
        report = self._make_report(status="HEALTHY")
        webhook.send_report_summary(report)

        mock_post.assert_not_called()

    @patch("vaig.integrations.google_chat.requests.post")
    def test_send_report_summary_degraded_sends(
        self, mock_post: MagicMock, webhook: GoogleChatWebhook
    ) -> None:
        """DEGRADED maps to HIGH, which meets critical+high threshold."""
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        report = self._make_report(status="DEGRADED")
        webhook.send_report_summary(report)

        mock_post.assert_called_once()


# ── Error handling ───────────────────────────────────────────


class TestGoogleChatErrorHandling:
    """Tests for error handling in GoogleChatWebhook."""

    @patch("vaig.integrations.google_chat.requests.post")
    def test_http_error_propagates(
        self, mock_post: MagicMock, webhook: GoogleChatWebhook
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
        mock_post.return_value = mock_resp

        with pytest.raises(requests.exceptions.HTTPError):
            webhook.send_alert_card(
                title="Alert", severity="CRITICAL", service_name="svc", summary="Down"
            )

    @patch("vaig.integrations.google_chat.requests.post")
    def test_timeout_propagates(
        self, mock_post: MagicMock, webhook: GoogleChatWebhook
    ) -> None:
        mock_post.side_effect = requests.exceptions.Timeout("Timeout")

        with pytest.raises(requests.exceptions.Timeout):
            webhook.send_alert_card(
                title="Alert", severity="CRITICAL", service_name="svc", summary="Down"
            )

    @patch("vaig.integrations.google_chat.requests.post")
    def test_keyboard_interrupt_reraises(
        self, mock_post: MagicMock, webhook: GoogleChatWebhook
    ) -> None:
        mock_post.side_effect = KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            webhook.send_alert_card(
                title="Alert", severity="CRITICAL", service_name="svc", summary="Down"
            )

"""Tests for Slack incoming webhook integration (SC-NH-01, SC-NH-02, SC-NH-03)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from vaig.core.config import SlackConfig
from vaig.integrations.slack import SlackWebhook

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def slack_config() -> SlackConfig:
    """Return an enabled SlackConfig for tests."""
    return SlackConfig(
        enabled=True,
        webhook_url="https://hooks.slack.com/services/T00/B00/secret",
        notify_on=["critical", "high"],
    )


@pytest.fixture()
def slack_config_all() -> SlackConfig:
    """Return a SlackConfig that notifies on all severities."""
    return SlackConfig(
        enabled=True,
        webhook_url="https://hooks.slack.com/services/T00/B00/secret",
        notify_on=["critical", "high", "medium", "low", "info"],
    )


@pytest.fixture()
def webhook(slack_config: SlackConfig) -> SlackWebhook:
    """Return a SlackWebhook instance."""
    return SlackWebhook(slack_config)


@pytest.fixture()
def webhook_all(slack_config_all: SlackConfig) -> SlackWebhook:
    """Return a SlackWebhook that notifies on all severities."""
    return SlackWebhook(slack_config_all)


# ── Helper ───────────────────────────────────────────────────


def _make_report(
    status: str = "CRITICAL",
    summary: str = "Service degraded",
    issues: int = 3,
    critical: int = 1,
    warning: int = 2,
    scope: str = "Namespace: prod",
) -> MagicMock:
    """Build a mock HealthReport."""
    report = MagicMock()
    report.executive_summary.overall_status.value = status
    report.executive_summary.scope = scope
    report.executive_summary.summary_text = summary
    report.executive_summary.issues_found = issues
    report.executive_summary.critical_count = critical
    report.executive_summary.warning_count = warning

    finding = MagicMock()
    finding.title = "OOMKilled pods detected"
    report.findings = [finding]
    return report


# ── send_alert_card tests (SC-NH-01) ────────────────────────


class TestSendAlertCard:
    """Tests for SlackWebhook.send_alert_card."""

    @patch("vaig.integrations.slack.requests.post")
    def test_sends_block_kit_payload(
        self, mock_post: MagicMock, webhook: SlackWebhook
    ) -> None:
        """SC-NH-01: Block Kit payload with severity, service, summary, findings."""
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        webhook.send_alert_card(
            title="Service Health Alert",
            severity="CRITICAL",
            service_name="my-api",
            summary="Service is down",
            findings=["OOMKilled pods", "High latency", "Disk full"],
        )

        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs["json"]

        # Must have Block Kit blocks
        assert "blocks" in payload
        blocks = payload["blocks"]

        # Header block
        assert blocks[0]["type"] == "header"
        assert blocks[0]["text"]["text"] == "Service Health Alert"

        # Severity and service in section fields
        block_str = str(blocks)
        assert "CRITICAL" in block_str
        assert "my-api" in block_str
        assert "OOMKilled pods" in block_str

    @patch("vaig.integrations.slack.requests.post")
    def test_sends_with_pagerduty_url(
        self, mock_post: MagicMock, webhook: SlackWebhook
    ) -> None:
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        webhook.send_alert_card(
            title="Alert",
            severity="CRITICAL",
            service_name="svc",
            summary="Down",
            pagerduty_url="https://app.pagerduty.com/incidents/INC123",
        )

        payload = mock_post.call_args.kwargs["json"]
        block_str = str(payload)
        assert "pagerduty.com" in block_str
        assert "View in PagerDuty" in block_str

    @patch("vaig.integrations.slack.requests.post")
    def test_posts_to_webhook_url(
        self, mock_post: MagicMock, webhook: SlackWebhook
    ) -> None:
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        webhook.send_alert_card(
            title="Alert",
            severity="CRITICAL",
            service_name="svc",
            summary="Down",
        )

        assert "hooks.slack.com" in mock_post.call_args.args[0]

    @patch("vaig.integrations.slack.requests.post")
    def test_uses_30s_timeout(
        self, mock_post: MagicMock, webhook: SlackWebhook
    ) -> None:
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        webhook.send_alert_card(
            title="Alert",
            severity="CRITICAL",
            service_name="svc",
            summary="Down",
        )

        assert mock_post.call_args.kwargs["timeout"] == 30

    @patch("vaig.integrations.slack.requests.post")
    def test_limits_findings_to_5(
        self, mock_post: MagicMock, webhook: SlackWebhook
    ) -> None:
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        findings = [f"Finding {i}" for i in range(10)]
        webhook.send_alert_card(
            title="Alert",
            severity="CRITICAL",
            service_name="svc",
            summary="Many issues",
            findings=findings,
        )

        payload = mock_post.call_args.kwargs["json"]
        block_str = str(payload)
        assert "Finding 4" in block_str
        assert "Finding 5" not in block_str


# ── send_alert_card threshold (SC-NH-03) ─────────────────────


class TestSendAlertCardThreshold:
    """Tests for severity threshold filtering."""

    @patch("vaig.integrations.slack.requests.post")
    def test_skips_below_threshold(
        self, mock_post: MagicMock, webhook: SlackWebhook
    ) -> None:
        """SC-NH-03: LOW severity is skipped when notify_on=[critical,high]."""
        webhook.send_alert_card(
            title="Alert",
            severity="LOW",
            service_name="svc",
            summary="Minor issue",
        )

        mock_post.assert_not_called()

    @patch("vaig.integrations.slack.requests.post")
    def test_skips_info_below_threshold(
        self, mock_post: MagicMock, webhook: SlackWebhook
    ) -> None:
        webhook.send_alert_card(
            title="Alert",
            severity="INFO",
            service_name="svc",
            summary="Informational",
        )

        mock_post.assert_not_called()

    @patch("vaig.integrations.slack.requests.post")
    def test_emits_debug_log_on_skip(
        self, mock_post: MagicMock, webhook: SlackWebhook, caplog: pytest.LogCaptureFixture
    ) -> None:
        """SC-NH-03: DEBUG log emitted when severity doesn't meet threshold."""
        import logging

        with caplog.at_level(logging.DEBUG):
            webhook.send_alert_card(
                title="Alert",
                severity="LOW",
                service_name="svc",
                summary="Minor",
            )

        assert "does not meet threshold" in caplog.text


# ── send_report_summary tests (SC-NH-02) ────────────────────


class TestSendReportSummary:
    """Tests for SlackWebhook.send_report_summary."""

    @patch("vaig.integrations.slack.requests.post")
    def test_sends_report_for_critical(
        self, mock_post: MagicMock, webhook: SlackWebhook
    ) -> None:
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        report = _make_report(status="CRITICAL")
        webhook.send_report_summary(report, execution_time=12.5)

        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs["json"]
        block_str = str(payload)
        assert "CRITICAL" in block_str
        assert "12.5s" in block_str

    @patch("vaig.integrations.slack.requests.post")
    def test_sends_report_for_degraded(
        self, mock_post: MagicMock, webhook: SlackWebhook
    ) -> None:
        """SC-NH-02: DEGRADED maps to HIGH → meets critical+high threshold."""
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        report = _make_report(status="DEGRADED")
        webhook.send_report_summary(report)

        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs["json"]
        block_str = str(payload)
        assert "DEGRADED" in block_str

    @patch("vaig.integrations.slack.requests.post")
    def test_skips_healthy_below_threshold(
        self, mock_post: MagicMock, webhook: SlackWebhook
    ) -> None:
        """HEALTHY maps to INFO → below critical+high threshold."""
        report = _make_report(status="HEALTHY")
        webhook.send_report_summary(report)

        mock_post.assert_not_called()

    @patch("vaig.integrations.slack.requests.post")
    def test_report_includes_findings(
        self, mock_post: MagicMock, webhook: SlackWebhook
    ) -> None:
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        report = _make_report(status="CRITICAL")
        webhook.send_report_summary(report, execution_time=5.0)

        payload = mock_post.call_args.kwargs["json"]
        block_str = str(payload)
        assert "OOMKilled pods" in block_str

    @patch("vaig.integrations.slack.requests.post")
    def test_report_includes_issue_counts(
        self, mock_post: MagicMock, webhook: SlackWebhook
    ) -> None:
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        report = _make_report(status="CRITICAL", issues=5, critical=2, warning=3)
        webhook.send_report_summary(report)

        payload = mock_post.call_args.kwargs["json"]
        block_str = str(payload)
        assert "5 total" in block_str
        assert "2 critical" in block_str
        assert "3 warning" in block_str


# ── Error handling ───────────────────────────────────────────


class TestSlackErrorHandling:
    """Tests for error handling in SlackWebhook."""

    @patch("vaig.integrations.slack.requests.post")
    def test_http_error_propagates(
        self, mock_post: MagicMock, webhook: SlackWebhook
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
        mock_post.return_value = mock_resp

        with pytest.raises(requests.exceptions.HTTPError):
            webhook.send_alert_card(
                title="Alert", severity="CRITICAL", service_name="svc", summary="Down"
            )

    @patch("vaig.integrations.slack.requests.post")
    def test_timeout_propagates(
        self, mock_post: MagicMock, webhook: SlackWebhook
    ) -> None:
        mock_post.side_effect = requests.exceptions.Timeout("Timeout")

        with pytest.raises(requests.exceptions.Timeout):
            webhook.send_alert_card(
                title="Alert", severity="CRITICAL", service_name="svc", summary="Down"
            )

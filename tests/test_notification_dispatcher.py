"""Tests for NotificationDispatcher — fan-out to PagerDuty + Google Chat."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vaig.core.config import GoogleChatConfig, PagerDutyConfig
from vaig.integrations.dispatcher import (
    AlertContext,
    DispatchResult,
    NotificationDispatcher,
)
from vaig.integrations.google_chat import GoogleChatWebhook
from vaig.integrations.pagerduty import PagerDutyClient

# ── Fixtures ─────────────────────────────────────────────────


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
    report.to_markdown.return_value = "# Report\nContent"
    return report


@pytest.fixture()
def pd_config() -> PagerDutyConfig:
    return PagerDutyConfig(
        enabled=True,
        routing_key="test-routing-key",
        api_token="test-api-token",
    )


@pytest.fixture()
def gc_config() -> GoogleChatConfig:
    return GoogleChatConfig(
        enabled=True,
        webhook_url="https://chat.googleapis.com/v1/spaces/XXX/messages?key=YYY&token=ZZZ",
        notify_on=["critical", "high"],
    )


@pytest.fixture()
def pd_client(pd_config: PagerDutyConfig) -> PagerDutyClient:
    return PagerDutyClient(pd_config)


@pytest.fixture()
def gc_client(gc_config: GoogleChatConfig) -> GoogleChatWebhook:
    return GoogleChatWebhook(gc_config)


# ── DispatchResult tests ─────────────────────────────────────


class TestDispatchResult:
    """Tests for DispatchResult dataclass."""

    def test_default_values(self) -> None:
        result = DispatchResult()
        assert result.pagerduty_dedup_key is None
        assert result.pagerduty_incident_id is None
        assert result.google_chat_sent is False
        assert result.errors == []
        assert result.has_errors is False

    def test_has_errors_when_errors_present(self) -> None:
        result = DispatchResult(errors=["Something failed"])
        assert result.has_errors is True


class TestAlertContext:
    """Tests for AlertContext dataclass."""

    def test_alert_context_fields(self) -> None:
        ctx = AlertContext(
            alert_id="alert-123",
            source="datadog",
            service_name="api-server",
            cluster_name="prod-cluster",
            namespace="default",
        )
        assert ctx.alert_id == "alert-123"
        assert ctx.source == "datadog"
        assert ctx.service_name == "api-server"

    def test_alert_context_defaults(self) -> None:
        ctx = AlertContext(alert_id="a", source="manual", service_name="svc")
        assert ctx.cluster_name == ""
        assert ctx.namespace == ""


# ── Dispatcher with both channels ────────────────────────────


class TestDispatchBothEnabled:
    """Tests for dispatch with both PagerDuty and Google Chat enabled."""

    @patch("vaig.integrations.google_chat.requests.post")
    @patch("vaig.integrations.pagerduty.requests.post")
    @patch("vaig.integrations.pagerduty.requests.get")
    def test_dispatch_triggers_both_channels(
        self,
        mock_pd_get: MagicMock,
        mock_pd_post: MagicMock,
        mock_gc_post: MagicMock,
        pd_client: PagerDutyClient,
        gc_client: GoogleChatWebhook,
    ) -> None:
        # PD event trigger
        mock_pd_post.return_value = MagicMock(status_code=202)
        mock_pd_post.return_value.raise_for_status = MagicMock()

        # PD incident search
        mock_pd_get.return_value = MagicMock(status_code=200)
        mock_pd_get.return_value.raise_for_status = MagicMock()
        mock_pd_get.return_value.json.return_value = {
            "incidents": [{"id": "INC456"}]
        }

        # Google Chat
        mock_gc_post.return_value = MagicMock(status_code=200)
        mock_gc_post.return_value.raise_for_status = MagicMock()

        dispatcher = NotificationDispatcher(
            pagerduty=pd_client, google_chat=gc_client
        )
        report = _make_report(status="CRITICAL")
        result = dispatcher.dispatch(report)

        assert result.pagerduty_dedup_key is not None
        assert result.pagerduty_incident_id == "INC456"
        assert result.google_chat_sent is True
        assert result.has_errors is False

    @patch("vaig.integrations.google_chat.requests.post")
    @patch("vaig.integrations.pagerduty.requests.post")
    @patch("vaig.integrations.pagerduty.requests.get")
    def test_dispatch_with_alert_context(
        self,
        mock_pd_get: MagicMock,
        mock_pd_post: MagicMock,
        mock_gc_post: MagicMock,
        pd_client: PagerDutyClient,
        gc_client: GoogleChatWebhook,
    ) -> None:
        mock_pd_post.return_value = MagicMock(status_code=202)
        mock_pd_post.return_value.raise_for_status = MagicMock()
        mock_pd_get.return_value = MagicMock(status_code=200)
        mock_pd_get.return_value.raise_for_status = MagicMock()
        mock_pd_get.return_value.json.return_value = {"incidents": []}
        mock_gc_post.return_value = MagicMock(status_code=200)
        mock_gc_post.return_value.raise_for_status = MagicMock()

        dispatcher = NotificationDispatcher(
            pagerduty=pd_client, google_chat=gc_client
        )
        report = _make_report(status="CRITICAL")
        ctx = AlertContext(
            alert_id="dd-alert-789",
            source="datadog",
            service_name="my-api",
        )
        result = dispatcher.dispatch(report, alert_context=ctx)

        # Should use alert_id as dedup_key
        assert result.pagerduty_dedup_key == "dd-alert-789"
        assert result.google_chat_sent is True


# ── Dispatcher with only PagerDuty ───────────────────────────


class TestDispatchPagerDutyOnly:
    """Tests for dispatch with only PagerDuty enabled."""

    @patch("vaig.integrations.pagerduty.requests.post")
    @patch("vaig.integrations.pagerduty.requests.get")
    def test_dispatch_pd_only(
        self,
        mock_pd_get: MagicMock,
        mock_pd_post: MagicMock,
        pd_client: PagerDutyClient,
    ) -> None:
        mock_pd_post.return_value = MagicMock(status_code=202)
        mock_pd_post.return_value.raise_for_status = MagicMock()
        mock_pd_get.return_value = MagicMock(status_code=200)
        mock_pd_get.return_value.raise_for_status = MagicMock()
        mock_pd_get.return_value.json.return_value = {"incidents": []}

        dispatcher = NotificationDispatcher(pagerduty=pd_client, google_chat=None)
        report = _make_report()
        result = dispatcher.dispatch(report)

        assert result.pagerduty_dedup_key is not None
        assert result.google_chat_sent is False
        assert result.has_errors is False


# ── Dispatcher with only Google Chat ─────────────────────────


class TestDispatchGoogleChatOnly:
    """Tests for dispatch with only Google Chat enabled."""

    @patch("vaig.integrations.google_chat.requests.post")
    def test_dispatch_gchat_only(
        self,
        mock_gc_post: MagicMock,
        gc_client: GoogleChatWebhook,
    ) -> None:
        mock_gc_post.return_value = MagicMock(status_code=200)
        mock_gc_post.return_value.raise_for_status = MagicMock()

        dispatcher = NotificationDispatcher(pagerduty=None, google_chat=gc_client)
        report = _make_report()
        result = dispatcher.dispatch(report)

        assert result.pagerduty_dedup_key is None
        assert result.google_chat_sent is True
        assert result.has_errors is False


# ── Dispatcher with both disabled ────────────────────────────


class TestDispatchBothDisabled:
    """Tests for dispatch with both channels disabled."""

    def test_dispatch_noop(self) -> None:
        dispatcher = NotificationDispatcher(pagerduty=None, google_chat=None)
        report = _make_report()
        result = dispatcher.dispatch(report)

        assert result.pagerduty_dedup_key is None
        assert result.pagerduty_incident_id is None
        assert result.google_chat_sent is False
        assert result.has_errors is False


# ── Severity-based filtering ─────────────────────────────────


class TestDispatchSeverityFiltering:
    """Tests for severity-based dispatch filtering."""

    @patch("vaig.integrations.google_chat.requests.post")
    def test_healthy_report_skips_gchat(
        self,
        mock_gc_post: MagicMock,
        gc_client: GoogleChatWebhook,
    ) -> None:
        """HEALTHY → INFO severity → below critical+high threshold."""
        dispatcher = NotificationDispatcher(pagerduty=None, google_chat=gc_client)
        report = _make_report(status="HEALTHY")
        result = dispatcher.dispatch(report)

        # Google Chat should NOT be called (INFO below threshold)
        assert result.google_chat_sent is False
        mock_gc_post.assert_not_called()


# ── Error collection ─────────────────────────────────────────


class TestDispatchErrorCollection:
    """Tests for error collection in DispatchResult."""

    @patch("vaig.integrations.pagerduty.requests.post")
    def test_pd_error_collected(
        self,
        mock_pd_post: MagicMock,
        pd_client: PagerDutyClient,
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("PD is down")
        mock_pd_post.return_value = mock_resp

        dispatcher = NotificationDispatcher(pagerduty=pd_client, google_chat=None)
        report = _make_report()
        result = dispatcher.dispatch(report)

        assert result.has_errors is True
        assert any("PagerDuty trigger failed" in e for e in result.errors)

    @patch("vaig.integrations.google_chat.requests.post")
    def test_gchat_error_collected(
        self,
        mock_gc_post: MagicMock,
        gc_client: GoogleChatWebhook,
    ) -> None:
        mock_gc_post.side_effect = Exception("Chat webhook failed")

        dispatcher = NotificationDispatcher(pagerduty=None, google_chat=gc_client)
        report = _make_report()
        result = dispatcher.dispatch(report)

        assert result.has_errors is True
        assert any("Google Chat alert failed" in e for e in result.errors)

    @patch("vaig.integrations.google_chat.requests.post")
    @patch("vaig.integrations.pagerduty.requests.post")
    def test_both_errors_collected(
        self,
        mock_pd_post: MagicMock,
        mock_gc_post: MagicMock,
        pd_client: PagerDutyClient,
        gc_client: GoogleChatWebhook,
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("PD down")
        mock_pd_post.return_value = mock_resp
        mock_gc_post.side_effect = Exception("GChat down")

        dispatcher = NotificationDispatcher(
            pagerduty=pd_client, google_chat=gc_client
        )
        report = _make_report()
        result = dispatcher.dispatch(report)

        assert result.has_errors is True
        assert len(result.errors) == 2


# ── from_config factory tests ────────────────────────────────


class TestFromConfig:
    """Tests for NotificationDispatcher.from_config factory."""

    def test_from_config_both_enabled(self) -> None:
        config = MagicMock()
        config.pagerduty = PagerDutyConfig(
            enabled=True, routing_key="rk", api_token="at"
        )
        config.google_chat = GoogleChatConfig(
            enabled=True,
            webhook_url="https://chat.example.com/webhook",
        )

        dispatcher = NotificationDispatcher.from_config(config)
        assert dispatcher.pagerduty is not None
        assert dispatcher.google_chat is not None

    def test_from_config_both_disabled(self) -> None:
        config = MagicMock()
        config.pagerduty = PagerDutyConfig(enabled=False)
        config.google_chat = GoogleChatConfig(enabled=False)

        dispatcher = NotificationDispatcher.from_config(config)
        assert dispatcher.pagerduty is None
        assert dispatcher.google_chat is None

    def test_from_config_missing_attributes(self) -> None:
        """Config without pagerduty/google_chat attributes should not crash."""
        config = MagicMock(spec=[])  # No attributes

        dispatcher = NotificationDispatcher.from_config(config)
        assert dispatcher.pagerduty is None
        assert dispatcher.google_chat is None

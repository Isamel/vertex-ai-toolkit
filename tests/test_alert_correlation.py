"""Tests for alert correlation tools — PagerDuty, OpsGenie, Slack."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from vaig.core.config import OpsGenieConfig, PagerDutyConfig, Settings, SlackConfig
from vaig.tools.categories import INCIDENT_MANAGEMENT
from vaig.tools.integrations._cache import _CACHE, clear_cache
from vaig.tools.integrations._registry import create_alert_correlation_tools

# ── Fixtures ─────────────────────────────────────────────────

_FAKE_PD_TOKEN = "fake-pd-token"  # noqa: S105
_FAKE_OG_KEY = "fake-og-key"  # noqa: S105
_FAKE_SLACK_TOKEN = "xoxb-fake-slack-token"  # noqa: S105


@pytest.fixture(autouse=True)  # noqa: PT004 — no return needed
def _clear_cache() -> None:  # noqa: PT004
    """Clear the integration cache before each test."""
    clear_cache()


def _pd_config(
    *,
    enabled: bool = True,
    api_token: str = _FAKE_PD_TOKEN,
    base_url: str = "https://api.pagerduty.com",
    alert_service_ids: list[str] | None = None,
    alert_fetch_limit: int = 25,
) -> PagerDutyConfig:
    return PagerDutyConfig(
        enabled=enabled,
        routing_key="fake-routing-key",
        api_token=api_token,
        base_url=base_url,
        alert_service_ids=alert_service_ids or [],
        alert_fetch_limit=alert_fetch_limit,
    )


def _og_config(
    *,
    enabled: bool = True,
    api_key: str = "fake-og-key",
    base_url: str = "https://api.opsgenie.com",
    team_ids: list[str] | None = None,
    alert_fetch_limit: int = 25,
) -> OpsGenieConfig:
    return OpsGenieConfig(
        enabled=enabled,
        api_key=SecretStr(api_key),
        base_url=base_url,
        team_ids=team_ids or [],
        alert_fetch_limit=alert_fetch_limit,
    )


def _slack_config(
    *,
    bot_token: str = _FAKE_SLACK_TOKEN,
    alert_channel_ids: list[str] | None = None,
) -> SlackConfig:
    return SlackConfig(
        webhook_url="https://hooks.slack.com/fake",
        bot_token=SecretStr(bot_token),
        alert_channel_ids=alert_channel_ids or [],
    )


def _mock_response(
    status_code: int = 200,
    json_data: dict | None = None,
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    return resp


# ══════════════════════════════════════════════════════════════
# PagerDuty Tests
# ══════════════════════════════════════════════════════════════


class TestPagerDutyTool:
    """Tests for list_pagerduty_incidents handler."""

    def test_happy_path(self) -> None:
        """S1: PD returns incidents → formatted Markdown table."""
        from vaig.tools.integrations.pagerduty import list_pagerduty_incidents

        mock_data = {
            "incidents": [
                {
                    "title": "CPU spike on web-01",
                    "status": "triggered",
                    "urgency": "high",
                    "service": {"summary": "web-service"},
                    "html_url": "https://pd.com/incidents/P1",
                },
                {
                    "title": "Memory leak on api-02",
                    "status": "acknowledged",
                    "urgency": "low",
                    "service": {"summary": "api-service"},
                    "html_url": "https://pd.com/incidents/P2",
                },
            ]
        }

        with patch("vaig.tools.integrations._http.requests.request", return_value=_mock_response(200, mock_data)):
            result = list_pagerduty_incidents(config=_pd_config())

        assert not result.error
        assert "CPU spike on web-01" in result.output
        assert "Memory leak on api-02" in result.output
        assert "triggered" in result.output
        assert "web-service" in result.output

    def test_api_error_503(self) -> None:
        """S3: PD API returns 503 → ToolResult with error=True."""
        from vaig.tools.integrations.pagerduty import list_pagerduty_incidents

        with patch("vaig.tools.integrations._http.requests.request", return_value=_mock_response(503)):
            result = list_pagerduty_incidents(config=_pd_config())

        assert result.error
        assert "503" in result.output

    def test_empty_results(self) -> None:
        """S4: PD returns no incidents → friendly message."""
        from vaig.tools.integrations.pagerduty import list_pagerduty_incidents

        with patch("vaig.tools.integrations._http.requests.request", return_value=_mock_response(200, {"incidents": []})):
            result = list_pagerduty_incidents(config=_pd_config())

        assert not result.error
        assert "No active PagerDuty incidents" in result.output

    def test_service_filter(self) -> None:
        """S5: service_name filters incidents by substring match."""
        from vaig.tools.integrations.pagerduty import list_pagerduty_incidents

        mock_data = {
            "incidents": [
                {
                    "title": "Issue A",
                    "status": "triggered",
                    "urgency": "high",
                    "service": {"summary": "payment-service"},
                    "html_url": "https://pd.com/P1",
                },
                {
                    "title": "Issue B",
                    "status": "triggered",
                    "urgency": "low",
                    "service": {"summary": "auth-service"},
                    "html_url": "https://pd.com/P2",
                },
            ]
        }

        with patch("vaig.tools.integrations._http.requests.request", return_value=_mock_response(200, mock_data)):
            result = list_pagerduty_incidents(config=_pd_config(), service_name="payment")

        assert not result.error
        assert "payment-service" in result.output
        assert "auth-service" not in result.output

    def test_service_ids_sent_in_params(self) -> None:
        """S5 config: alert_service_ids are sent as query params."""
        from vaig.tools.integrations.pagerduty import list_pagerduty_incidents

        cfg = _pd_config(alert_service_ids=["SVC123", "SVC456"])

        with patch("vaig.tools.integrations._http.requests.request", return_value=_mock_response(200, {"incidents": []})) as mock_req:
            list_pagerduty_incidents(config=cfg)

        _, kwargs = mock_req.call_args
        assert kwargs["params"]["service_ids[]"] == ["SVC123", "SVC456"]

    def test_cache_hit(self) -> None:
        """S6: Second call within TTL returns cached result without HTTP."""
        from vaig.tools.integrations.pagerduty import list_pagerduty_incidents

        mock_data = {"incidents": [{"title": "Test", "status": "triggered", "urgency": "high", "service": {"summary": "svc"}, "html_url": "https://pd.com/P1"}]}

        with patch("vaig.tools.integrations._http.requests.request", return_value=_mock_response(200, mock_data)) as mock_req:
            result1 = list_pagerduty_incidents(config=_pd_config())
            result2 = list_pagerduty_incidents(config=_pd_config())

        assert result1.output == result2.output
        assert mock_req.call_count == 1  # Second call used cache

    def test_token_not_leaked(self) -> None:
        """S9: API token must NEVER appear in tool output."""
        from vaig.tools.integrations.pagerduty import list_pagerduty_incidents

        token = "super-secret-pd-token-12345"
        cfg = _pd_config(api_token=token)

        # Test on success
        mock_data = {"incidents": [{"title": "Test", "status": "triggered", "urgency": "high", "service": {"summary": "svc"}, "html_url": "https://pd.com/P1"}]}
        with patch("vaig.tools.integrations._http.requests.request", return_value=_mock_response(200, mock_data)):
            result = list_pagerduty_incidents(config=cfg)
        assert token not in result.output

        clear_cache()

        # Test on error
        with patch("vaig.tools.integrations._http.requests.request", return_value=_mock_response(401)):
            result = list_pagerduty_incidents(config=cfg)
        assert token not in result.output

    def test_category_incident_management(self) -> None:
        """PD tool is registered under INCIDENT_MANAGEMENT category."""
        settings = Settings(
            pagerduty=_pd_config(),
            opsgenie=_og_config(enabled=False, api_key=""),
            slack=_slack_config(bot_token=""),
        )
        tools = create_alert_correlation_tools(settings)
        pd_tools = [t for t in tools if t.name == "list_pagerduty_incidents"]
        assert len(pd_tools) == 1
        assert INCIDENT_MANAGEMENT in pd_tools[0].categories

    def test_auth_401_error(self) -> None:
        """401 returns descriptive auth error."""
        from vaig.tools.integrations.pagerduty import list_pagerduty_incidents

        with patch("vaig.tools.integrations._http.requests.request", return_value=_mock_response(401)):
            result = list_pagerduty_incidents(config=_pd_config())

        assert result.error
        assert "authentication failed" in result.output.lower()


# ══════════════════════════════════════════════════════════════
# OpsGenie Tests
# ══════════════════════════════════════════════════════════════


class TestOpsGenieTool:
    """Tests for list_opsgenie_alerts handler."""

    def test_happy_path(self) -> None:
        """OG returns alerts → formatted Markdown table."""
        from vaig.tools.integrations.opsgenie import list_opsgenie_alerts

        mock_data = {
            "data": [
                {
                    "message": "Disk full on db-01",
                    "status": "open",
                    "priority": "P1",
                    "source": "Datadog",
                    "createdAt": "2026-04-08T10:00:00Z",
                },
            ]
        }

        with patch("vaig.tools.integrations._http.requests.request", return_value=_mock_response(200, mock_data)):
            result = list_opsgenie_alerts(config=_og_config())

        assert not result.error
        assert "Disk full on db-01" in result.output
        assert "P1" in result.output

    def test_empty_results(self) -> None:
        """S4: OG returns no alerts → friendly message."""
        from vaig.tools.integrations.opsgenie import list_opsgenie_alerts

        with patch("vaig.tools.integrations._http.requests.request", return_value=_mock_response(200, {"data": []})):
            result = list_opsgenie_alerts(config=_og_config())

        assert not result.error
        assert "No open OpsGenie alerts" in result.output

    def test_api_error(self) -> None:
        """OG API returns 500 → ToolResult with error=True."""
        from vaig.tools.integrations.opsgenie import list_opsgenie_alerts

        with patch("vaig.tools.integrations._http.requests.request", return_value=_mock_response(500)):
            result = list_opsgenie_alerts(config=_og_config())

        assert result.error

    def test_eu_region_url(self) -> None:
        """OG uses EU base_url when configured."""
        from vaig.tools.integrations.opsgenie import list_opsgenie_alerts

        eu_cfg = _og_config(base_url="https://api.eu.opsgenie.com")

        with patch("vaig.tools.integrations._http.requests.request", return_value=_mock_response(200, {"data": []})) as mock_req:
            list_opsgenie_alerts(config=eu_cfg)

        call_args = mock_req.call_args
        assert "api.eu.opsgenie.com" in call_args[0][1]

    def test_token_not_leaked(self) -> None:
        """S9: OG API key must NEVER appear in tool output."""
        from vaig.tools.integrations.opsgenie import list_opsgenie_alerts

        key = "super-secret-og-key-67890"
        cfg = _og_config(api_key=key)

        with patch("vaig.tools.integrations._http.requests.request", return_value=_mock_response(200, {"data": []})):
            result = list_opsgenie_alerts(config=cfg)
        assert key not in result.output

        clear_cache()

        with patch("vaig.tools.integrations._http.requests.request", return_value=_mock_response(403)):
            result = list_opsgenie_alerts(config=cfg)
        assert key not in result.output

    def test_auto_enable_validator(self) -> None:
        """OpsGenieConfig auto-enables when api_key is provided."""
        cfg = OpsGenieConfig(api_key=SecretStr("some-key"))
        assert cfg.enabled is True

    def test_auto_disable_without_key(self) -> None:
        """OpsGenieConfig disables when enabled=True but no api_key."""
        cfg = OpsGenieConfig(enabled=True, api_key=SecretStr(""))
        assert cfg.enabled is False

    def test_priority_filter_sent(self) -> None:
        """Priority filter is included in the query param."""
        from vaig.tools.integrations.opsgenie import list_opsgenie_alerts

        with patch("vaig.tools.integrations._http.requests.request", return_value=_mock_response(200, {"data": []})) as mock_req:
            list_opsgenie_alerts(config=_og_config(), priority="P1")

        _, kwargs = mock_req.call_args
        assert "priority=P1" in kwargs["params"]["query"]


# ══════════════════════════════════════════════════════════════
# Slack Tests
# ══════════════════════════════════════════════════════════════


class TestSlackTool:
    """Tests for search_slack_messages handler."""

    def test_happy_path(self) -> None:
        """S8: Slack returns channel messages → formatted list."""
        from vaig.tools.integrations.slack import search_slack_messages

        mock_data = {
            "ok": True,
            "messages": [
                {
                    "text": "ALERT: CPU spike on prod-web-01",
                    "ts": "1712563200.000100",
                    "user": "U01BOTID",
                },
                {
                    "text": "Resolved: CPU back to normal",
                    "ts": "1712563210.000200",
                    "user": "U02BOTID",
                },
            ],
        }

        with patch("vaig.tools.integrations._http.requests.request", return_value=_mock_response(200, mock_data)):
            result = search_slack_messages(config=_slack_config(), channel_id="C123")

        assert not result.error
        assert "CPU spike" in result.output
        assert "Resolved" in result.output

    def test_empty_channel(self) -> None:
        """No messages → friendly message."""
        from vaig.tools.integrations.slack import search_slack_messages

        with patch("vaig.tools.integrations._http.requests.request", return_value=_mock_response(200, {"ok": True, "messages": []})):
            result = search_slack_messages(config=_slack_config(), channel_id="C123")

        assert not result.error
        assert "No messages found" in result.output

    def test_slack_api_error(self) -> None:
        """Slack ok=false → ToolResult with error."""
        from vaig.tools.integrations.slack import search_slack_messages

        mock_data = {"ok": False, "error": "channel_not_found"}

        with patch("vaig.tools.integrations._http.requests.request", return_value=_mock_response(200, mock_data)):
            result = search_slack_messages(config=_slack_config(), channel_id="C_BAD")

        assert result.error
        assert "channel_not_found" in result.output

    def test_cache_30s_ttl(self) -> None:
        """Slack cache uses 30s TTL (shorter than PD/OG 60s)."""
        from vaig.tools.integrations.slack import search_slack_messages

        mock_data = {"ok": True, "messages": [{"text": "test", "ts": "1712563200.000100", "user": "U01"}]}

        with patch("vaig.tools.integrations._http.requests.request", return_value=_mock_response(200, mock_data)) as mock_req:
            search_slack_messages(config=_slack_config(), channel_id="C123")
            search_slack_messages(config=_slack_config(), channel_id="C123")

        assert mock_req.call_count == 1  # second call used cache

        # Verify cache entry has 30s TTL
        for _key, (_, _, ttl) in _CACHE.items():
            if "slack" in _key:
                assert ttl == 30

    def test_token_not_leaked(self) -> None:
        """S9: Slack bot token must NEVER appear in tool output."""
        from vaig.tools.integrations.slack import search_slack_messages

        token = "xoxb-super-secret-slack-token"
        cfg = _slack_config(bot_token=token)

        with patch("vaig.tools.integrations._http.requests.request", return_value=_mock_response(200, {"ok": True, "messages": []})):
            result = search_slack_messages(config=cfg, channel_id="C123")
        assert token not in result.output

    def test_missing_channel_id(self) -> None:
        """Empty channel_id → error."""
        from vaig.tools.integrations.slack import search_slack_messages

        result = search_slack_messages(config=_slack_config(), channel_id="")
        assert result.error
        assert "channel_id is required" in result.output

    def test_query_filter(self) -> None:
        """Query param filters messages by substring."""
        from vaig.tools.integrations.slack import search_slack_messages

        mock_data = {
            "ok": True,
            "messages": [
                {"text": "ALERT: CPU spike", "ts": "1712563200.000100", "user": "U01"},
                {"text": "Meeting at 3pm", "ts": "1712563210.000200", "user": "U02"},
            ],
        }

        with patch("vaig.tools.integrations._http.requests.request", return_value=_mock_response(200, mock_data)):
            result = search_slack_messages(config=_slack_config(), channel_id="C123", query="CPU")

        assert "CPU spike" in result.output
        assert "Meeting" not in result.output


# ══════════════════════════════════════════════════════════════
# Registry Tests
# ══════════════════════════════════════════════════════════════


class TestAlertCorrelationRegistry:
    """Tests for create_alert_correlation_tools factory."""

    def test_pd_enabled(self) -> None:
        """PD tool registered when enabled + api_token."""
        settings = Settings(
            pagerduty=_pd_config(),
            opsgenie=_og_config(enabled=False, api_key=""),
            slack=_slack_config(bot_token=""),
        )
        tools = create_alert_correlation_tools(settings)
        names = [t.name for t in tools]
        assert "list_pagerduty_incidents" in names
        assert len(tools) == 1

    def test_pd_disabled(self) -> None:
        """S2: PD tool NOT registered when disabled."""
        settings = Settings(
            pagerduty=PagerDutyConfig(enabled=False),
            opsgenie=_og_config(enabled=False, api_key=""),
            slack=_slack_config(bot_token=""),
        )
        tools = create_alert_correlation_tools(settings)
        names = [t.name for t in tools]
        assert "list_pagerduty_incidents" not in names

    def test_og_enabled(self) -> None:
        """OG tool registered when enabled + api_key."""
        settings = Settings(
            pagerduty=PagerDutyConfig(enabled=False),
            opsgenie=_og_config(),
            slack=_slack_config(bot_token=""),
        )
        tools = create_alert_correlation_tools(settings)
        names = [t.name for t in tools]
        assert "list_opsgenie_alerts" in names
        assert len(tools) == 1

    def test_og_disabled(self) -> None:
        """S2: OG tool NOT registered when disabled."""
        settings = Settings(
            pagerduty=PagerDutyConfig(enabled=False),
            opsgenie=OpsGenieConfig(enabled=False),
            slack=_slack_config(bot_token=""),
        )
        tools = create_alert_correlation_tools(settings)
        names = [t.name for t in tools]
        assert "list_opsgenie_alerts" not in names

    def test_slack_enabled(self) -> None:
        """Slack tool registered when bot_token set."""
        settings = Settings(
            pagerduty=PagerDutyConfig(enabled=False),
            opsgenie=_og_config(enabled=False, api_key=""),
            slack=_slack_config(),
        )
        tools = create_alert_correlation_tools(settings)
        names = [t.name for t in tools]
        assert "search_slack_messages" in names

    def test_slack_disabled(self) -> None:
        """S2: Slack tool NOT registered when no bot_token."""
        settings = Settings(
            pagerduty=PagerDutyConfig(enabled=False),
            opsgenie=_og_config(enabled=False, api_key=""),
            slack=SlackConfig(),
        )
        tools = create_alert_correlation_tools(settings)
        names = [t.name for t in tools]
        assert "search_slack_messages" not in names

    def test_all_integrations_active(self) -> None:
        """S7: All 3 tools registered when all integrations enabled."""
        settings = Settings(
            pagerduty=_pd_config(),
            opsgenie=_og_config(),
            slack=_slack_config(),
        )
        tools = create_alert_correlation_tools(settings)
        names = [t.name for t in tools]
        assert len(tools) == 3
        assert "list_pagerduty_incidents" in names
        assert "list_opsgenie_alerts" in names
        assert "search_slack_messages" in names

    def test_all_tools_have_incident_management_category(self) -> None:
        """All alert correlation tools have INCIDENT_MANAGEMENT category."""
        settings = Settings(
            pagerduty=_pd_config(),
            opsgenie=_og_config(),
            slack=_slack_config(),
        )
        tools = create_alert_correlation_tools(settings)
        for tool in tools:
            assert INCIDENT_MANAGEMENT in tool.categories

    def test_no_tools_when_nothing_configured(self) -> None:
        """Empty list when no integrations are configured."""
        settings = Settings()
        tools = create_alert_correlation_tools(settings)
        assert tools == []


# ══════════════════════════════════════════════════════════════
# HTTP Helper Tests
# ══════════════════════════════════════════════════════════════


class TestHTTPHelper:
    """Tests for api_request shared helper."""

    def test_retry_on_5xx(self) -> None:
        """5xx triggers retry, second attempt succeeds."""
        from vaig.tools.integrations._http import api_request

        responses = [_mock_response(503), _mock_response(200, {"ok": True})]

        with patch("vaig.tools.integrations._http.requests.request", side_effect=responses) as mock_req:
            with patch("vaig.tools.integrations._http.time.sleep"):
                data, error = api_request("GET", "https://example.com", headers={}, service_name="Test")

        assert error is None
        assert data == {"ok": True}
        assert mock_req.call_count == 2

    def test_timeout_returns_error(self) -> None:
        """Timeout after retries returns ToolResult error."""
        import requests as req_lib

        from vaig.tools.integrations._http import api_request

        with patch("vaig.tools.integrations._http.requests.request", side_effect=req_lib.Timeout("timed out")):
            with patch("vaig.tools.integrations._http.time.sleep"):
                data, error = api_request("GET", "https://example.com", headers={}, service_name="TestSvc")

        assert data is None
        assert error is not None
        assert error.error
        assert "timed out" in error.output.lower()

    def test_429_returns_rate_limit_error(self) -> None:
        """429 returns rate limit error without retry."""
        from vaig.tools.integrations._http import api_request

        with patch("vaig.tools.integrations._http.requests.request", return_value=_mock_response(429)):
            data, error = api_request("GET", "https://example.com", headers={}, service_name="Test")

        assert data is None
        assert error is not None
        assert "rate limited" in error.output.lower()


# ══════════════════════════════════════════════════════════════
# Cache Tests
# ══════════════════════════════════════════════════════════════


class TestIntegrationCache:
    """Tests for the integrations _cache module."""

    def test_cache_set_and_get(self) -> None:
        """Basic set/get works."""
        from vaig.tools.integrations._cache import _cache_key, _get_cached, _set_cache

        key = _cache_key("test", "key")
        _set_cache(key, "value123")
        assert _get_cached(key) == "value123"

    def test_cache_expiry(self) -> None:
        """Cache entry expires after TTL."""
        from vaig.tools.integrations._cache import _cache_key, _get_cached, _set_cache

        key = _cache_key("test", "expiry")
        _set_cache(key, "value", ttl=1)

        with patch("vaig.tools.integrations._cache.time.monotonic", return_value=time.monotonic() + 2):
            assert _get_cached(key) is None

    def test_clear_cache(self) -> None:
        """clear_cache empties all entries."""
        from vaig.tools.integrations._cache import _CACHE, _cache_key, _set_cache, clear_cache

        _set_cache(_cache_key("a"), "1")
        _set_cache(_cache_key("b"), "2")
        assert len(_CACHE) == 2

        clear_cache()
        assert len(_CACHE) == 0

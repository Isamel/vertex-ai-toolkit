"""Tests for SlackConfig and EmailConfig auto-enable validators (SC-NH-09, SC-NH-10)."""

from __future__ import annotations

from vaig.core.config import EmailConfig, SlackConfig


class TestSlackConfigAutoEnable:
    """Tests for SlackConfig auto-enable validator."""

    def test_auto_enable_when_webhook_url_set(self) -> None:
        """SC-NH-09: webhook_url present → enabled becomes True."""
        config = SlackConfig(webhook_url="https://hooks.slack.com/services/T/B/secret")
        assert config.enabled is True

    def test_stays_disabled_when_no_webhook_url(self) -> None:
        config = SlackConfig()
        assert config.enabled is False

    def test_disable_when_enabled_but_no_url(self) -> None:
        """SC-NH-10 analog: enabled=True but no webhook_url → disabled + warning."""
        config = SlackConfig(enabled=True, webhook_url="")
        assert config.enabled is False

    def test_explicit_enabled_with_url(self) -> None:
        config = SlackConfig(
            enabled=True,
            webhook_url="https://hooks.slack.com/services/T/B/secret",
        )
        assert config.enabled is True

    def test_notify_on_defaults(self) -> None:
        config = SlackConfig()
        assert config.notify_on == ["critical", "high"]


class TestEmailConfigAutoEnable:
    """Tests for EmailConfig auto-enable validator."""

    def test_auto_enable_when_credentials_set(self) -> None:
        """SC-NH-09: smtp_host + from_address + recipients → enabled becomes True."""
        config = EmailConfig(
            smtp_host="smtp.example.com",
            from_address="noreply@example.com",
            recipients=["ops@example.com"],
        )
        assert config.enabled is True

    def test_stays_disabled_when_no_credentials(self) -> None:
        config = EmailConfig()
        assert config.enabled is False

    def test_disable_when_enabled_but_missing_host(self) -> None:
        """SC-NH-10: enabled=True but smtp_host missing → disabled + warning."""
        config = EmailConfig(enabled=True, smtp_host="")
        assert config.enabled is False

    def test_disable_when_enabled_but_missing_recipients(self) -> None:
        config = EmailConfig(
            enabled=True,
            smtp_host="smtp.example.com",
            from_address="noreply@example.com",
            recipients=[],
        )
        assert config.enabled is False

    def test_explicit_enabled_with_all_credentials(self) -> None:
        config = EmailConfig(
            enabled=True,
            smtp_host="smtp.example.com",
            from_address="noreply@example.com",
            recipients=["ops@example.com"],
        )
        assert config.enabled is True

    def test_notify_on_defaults(self) -> None:
        config = EmailConfig()
        assert config.notify_on == ["critical", "high"]

    def test_default_port_and_tls(self) -> None:
        config = EmailConfig()
        assert config.smtp_port == 587
        assert config.use_tls is True

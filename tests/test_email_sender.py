"""Tests for SMTP email notification integration (SC-NH-04, SC-NH-05, SC-NH-06)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vaig.core.config import EmailConfig
from vaig.integrations.email_sender import EmailSender

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def email_config() -> EmailConfig:
    """Return an enabled EmailConfig for tests."""
    return EmailConfig(
        enabled=True,
        smtp_host="smtp.example.com",
        smtp_port=587,
        username="user",
        password="pass",
        from_address="noreply@example.com",
        recipients=["ops@example.com"],
        use_tls=True,
        notify_on=["critical", "high"],
    )


@pytest.fixture()
def email_config_ssl() -> EmailConfig:
    """Return an EmailConfig for implicit SSL (port 465)."""
    return EmailConfig(
        enabled=True,
        smtp_host="smtp.example.com",
        smtp_port=465,
        username="user",
        password="pass",
        from_address="noreply@example.com",
        recipients=["ops@example.com"],
        use_tls=False,
        notify_on=["critical", "high"],
    )


@pytest.fixture()
def email_config_multi() -> EmailConfig:
    """Return an EmailConfig with multiple recipients."""
    return EmailConfig(
        enabled=True,
        smtp_host="smtp.example.com",
        smtp_port=587,
        username="user",
        password="pass",
        from_address="noreply@example.com",
        recipients=["a@example.com", "b@example.com", "c@example.com"],
        use_tls=True,
        notify_on=["critical", "high"],
    )


@pytest.fixture()
def sender(email_config: EmailConfig) -> EmailSender:
    """Return an EmailSender instance."""
    return EmailSender(email_config)


@pytest.fixture()
def sender_ssl(email_config_ssl: EmailConfig) -> EmailSender:
    """Return an EmailSender using implicit SSL."""
    return EmailSender(email_config_ssl)


@pytest.fixture()
def sender_multi(email_config_multi: EmailConfig) -> EmailSender:
    """Return an EmailSender with multiple recipients."""
    return EmailSender(email_config_multi)


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


# ── send_alert_email tests (SC-NH-04) ───────────────────────


class TestSendAlertEmail:
    """Tests for EmailSender.send_alert_email."""

    @patch("vaig.integrations.email_sender.smtplib.SMTP")
    def test_sends_multipart_alternative(
        self, mock_smtp_cls: MagicMock, sender: EmailSender
    ) -> None:
        """SC-NH-04: multipart/alternative with text/plain and text/html."""
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        sender.send_alert_email(
            title="Service Health Alert",
            severity="CRITICAL",
            service_name="my-api",
            summary="Service is down",
            recipients=["ops@example.com"],
        )

        mock_server.sendmail.assert_called_once()
        _, _, msg_str = mock_server.sendmail.call_args.args
        assert "multipart/alternative" in msg_str
        assert "text/plain" in msg_str
        assert "text/html" in msg_str

    @patch("vaig.integrations.email_sender.smtplib.SMTP")
    def test_subject_includes_severity(
        self, mock_smtp_cls: MagicMock, sender: EmailSender
    ) -> None:
        """SC-NH-04: subject line includes the severity level."""
        from email import message_from_string
        from email.header import decode_header

        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        sender.send_alert_email(
            title="Alert",
            severity="CRITICAL",
            service_name="svc",
            summary="Down",
        )

        _, _, msg_str = mock_server.sendmail.call_args.args
        msg = message_from_string(msg_str)
        decoded_parts = decode_header(msg["Subject"])
        subject = "".join(
            part.decode(enc or "utf-8") if isinstance(part, bytes) else part
            for part, enc in decoded_parts
        )
        assert "[CRITICAL]" in subject

    @patch("vaig.integrations.email_sender.smtplib.SMTP")
    def test_html_body_contains_details(
        self, mock_smtp_cls: MagicMock, sender: EmailSender
    ) -> None:
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        sender.send_alert_email(
            title="Alert",
            severity="HIGH",
            service_name="my-api",
            summary="Latency spike detected",
            findings=["OOMKilled pods", "High latency"],
        )

        _, _, msg_str = mock_server.sendmail.call_args.args
        assert "my-api" in msg_str
        assert "OOMKilled pods" in msg_str
        assert "High latency" in msg_str

    @patch("vaig.integrations.email_sender.smtplib.SMTP")
    def test_skips_below_threshold(
        self, mock_smtp_cls: MagicMock, sender: EmailSender
    ) -> None:
        sender.send_alert_email(
            title="Alert",
            severity="LOW",
            service_name="svc",
            summary="Minor issue",
        )

        mock_smtp_cls.assert_not_called()

    @patch("vaig.integrations.email_sender.smtplib.SMTP")
    def test_pagerduty_url_in_body(
        self, mock_smtp_cls: MagicMock, sender: EmailSender
    ) -> None:
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        sender.send_alert_email(
            title="Alert",
            severity="CRITICAL",
            service_name="svc",
            summary="Down",
            pagerduty_url="https://app.pagerduty.com/incidents/INC123",
        )

        _, _, msg_str = mock_server.sendmail.call_args.args
        assert "pagerduty.com" in msg_str


# ── send_report_email tests (SC-NH-05) ──────────────────────


class TestSendReportEmail:
    """Tests for EmailSender.send_report_email."""

    @patch("vaig.integrations.email_sender.smtplib.SMTP")
    def test_sends_report_for_critical(
        self, mock_smtp_cls: MagicMock, sender: EmailSender
    ) -> None:
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        report = _make_report(status="CRITICAL")
        sender.send_report_email(report, execution_time=12.5)

        mock_server.sendmail.assert_called_once()
        _, _, msg_str = mock_server.sendmail.call_args.args
        assert "CRITICAL" in msg_str
        assert "12.5s" in msg_str

    @patch("vaig.integrations.email_sender.smtplib.SMTP")
    def test_multiple_recipients_in_to_header(
        self, mock_smtp_cls: MagicMock, sender_multi: EmailSender
    ) -> None:
        """SC-NH-05: single SMTP message with all 3 addresses in To header."""
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        report = _make_report(status="CRITICAL")
        sender_multi.send_report_email(report)

        mock_server.sendmail.assert_called_once()
        from_addr, to_addrs, msg_str = mock_server.sendmail.call_args.args

        assert to_addrs == ["a@example.com", "b@example.com", "c@example.com"]
        assert "a@example.com" in msg_str
        assert "b@example.com" in msg_str
        assert "c@example.com" in msg_str

    @patch("vaig.integrations.email_sender.smtplib.SMTP")
    def test_skips_healthy_below_threshold(
        self, mock_smtp_cls: MagicMock, sender: EmailSender
    ) -> None:
        report = _make_report(status="HEALTHY")
        sender.send_report_email(report)

        mock_smtp_cls.assert_not_called()

    @patch("vaig.integrations.email_sender.smtplib.SMTP")
    def test_report_includes_findings(
        self, mock_smtp_cls: MagicMock, sender: EmailSender
    ) -> None:
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        report = _make_report(status="CRITICAL")
        sender.send_report_email(report)

        _, _, msg_str = mock_server.sendmail.call_args.args
        assert "OOMKilled pods" in msg_str


# ── TLS/SSL path tests (SC-NH-06) ───────────────────────────


class TestEmailTLSPaths:
    """Tests for TLS/SSL connection paths."""

    @patch("vaig.integrations.email_sender.smtplib.SMTP")
    def test_starttls_on_port_587(
        self, mock_smtp_cls: MagicMock, sender: EmailSender
    ) -> None:
        """SC-NH-06: use_tls=True → starttls() called before login()."""
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        sender.send_alert_email(
            title="Alert",
            severity="CRITICAL",
            service_name="svc",
            summary="Down",
        )

        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("user", "pass")

        # Verify starttls called before login
        starttls_order = mock_server.starttls.call_args_list
        login_order = mock_server.login.call_args_list
        assert len(starttls_order) == 1
        assert len(login_order) == 1

    @patch("vaig.integrations.email_sender.smtplib.SMTP_SSL")
    def test_smtp_ssl_on_port_465(
        self, mock_smtp_ssl_cls: MagicMock, sender_ssl: EmailSender
    ) -> None:
        """SC-NH-06: port 465 → SMTP_SSL used instead of SMTP."""
        mock_server = MagicMock()
        mock_smtp_ssl_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_ssl_cls.return_value.__exit__ = MagicMock(return_value=False)

        sender_ssl.send_alert_email(
            title="Alert",
            severity="CRITICAL",
            service_name="svc",
            summary="Down",
        )

        mock_smtp_ssl_cls.assert_called_once_with(
            "smtp.example.com", 465, timeout=30,
        )
        mock_server.login.assert_called_once_with("user", "pass")
        mock_server.sendmail.assert_called_once()

    @patch("vaig.integrations.email_sender.smtplib.SMTP")
    def test_no_tls_no_starttls(
        self, mock_smtp_cls: MagicMock
    ) -> None:
        """When use_tls=False and port!=465, no starttls()."""
        config = EmailConfig(
            enabled=True,
            smtp_host="smtp.example.com",
            smtp_port=25,
            from_address="noreply@example.com",
            recipients=["ops@example.com"],
            use_tls=False,
            notify_on=["critical", "high"],
        )
        sender = EmailSender(config)

        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        sender.send_alert_email(
            title="Alert",
            severity="CRITICAL",
            service_name="svc",
            summary="Down",
        )

        mock_server.starttls.assert_not_called()
        mock_server.login.assert_not_called()  # No username/password

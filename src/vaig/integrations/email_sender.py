"""SMTP email notification integration — HTML + plain-text alerts and reports."""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape as html_escape
from typing import TYPE_CHECKING

from vaig.core.config import EmailConfig
from vaig.integrations.formatters import (
    FormattedAlert,
    FormattedReport,
    format_report_summary,
    meets_threshold,
    status_to_severity,
)

if TYPE_CHECKING:
    from vaig.skills.service_health.schema import HealthReport

logger = logging.getLogger(__name__)


class EmailSender:
    """SMTP email notification sender.

    Sends structured HTML + plain-text alert and report emails via
    stdlib ``smtplib``.  Supports STARTTLS (port 587) and implicit
    SSL (port 465).
    """

    def __init__(self, config: EmailConfig) -> None:
        self.smtp_host = config.smtp_host
        self.smtp_port = config.smtp_port
        self.username = config.username
        self.password = config.password
        self.from_address = config.from_address
        self.recipients = config.recipients
        self.use_tls = config.use_tls
        self.timeout = config.timeout
        self.notify_on = config.notify_on

    def send_alert_email(
        self,
        title: str,
        severity: str,
        service_name: str,
        summary: str,
        findings: list[str] | None = None,
        pagerduty_url: str | None = None,
        recipients: list[str] | None = None,
    ) -> None:
        """Send an alert email with HTML and plain-text parts.

        Args:
            title: Email subject prefix (e.g. "Service Health Alert").
            severity: Severity level (CRITICAL, HIGH, MEDIUM, LOW, INFO).
            service_name: Name of the affected service.
            summary: Brief description of the issue.
            findings: Optional list of top finding descriptions.
            pagerduty_url: Optional PagerDuty incident URL.
            recipients: Override default recipients.
        """
        severity_upper = severity.upper()

        if not meets_threshold(severity_upper, self.notify_on):
            logger.debug(
                "Skipping email alert — severity %s does not meet threshold %s",
                severity_upper,
                self.notify_on,
            )
            return

        alert = FormattedAlert(
            title=title,
            severity=severity_upper,
            severity_icon="",  # Not used in email
            service_name=service_name,
            summary=summary,
            findings=findings[:5] if findings else [],
            pagerduty_url=pagerduty_url,
        )

        to_addrs = recipients or self.recipients
        subject = f"[{severity_upper}] {title} — {service_name}"

        html_body = self._build_alert_html(alert)
        plain_body = self._build_alert_plain(alert)

        self._send(subject, html_body, plain_body, to_addrs)

    def send_report_email(
        self,
        report: HealthReport,
        execution_time: float = 0.0,
        recipients: list[str] | None = None,
    ) -> None:
        """Send a report summary email after pipeline completion.

        Only sends if the report severity meets the ``notify_on`` threshold.

        Args:
            report: The completed HealthReport.
            execution_time: Pipeline execution time in seconds.
            recipients: Override default recipients.
        """
        severity = status_to_severity(
            report.executive_summary.overall_status.value,
        )

        if not meets_threshold(severity, self.notify_on):
            logger.debug(
                "Skipping email report — severity %s does not meet threshold",
                severity,
            )
            return

        formatted = format_report_summary(report, execution_time)
        to_addrs = recipients or self.recipients
        subject = f"[{formatted.status}] Service Health Report — {formatted.scope}"

        html_body = self._build_report_html(formatted)
        plain_body = self._build_report_plain(formatted)

        self._send(subject, html_body, plain_body, to_addrs)

    # ── HTML builders ────────────────────────────────────────

    def _build_alert_html(self, alert: FormattedAlert) -> str:
        """Build HTML body for an alert email."""
        findings_html = ""
        if alert.findings:
            items = "".join(f"<li>{html_escape(f)}</li>" for f in alert.findings)
            findings_html = f"<h3>Top Findings</h3><ul>{items}</ul>"

        pd_html = ""
        if alert.pagerduty_url:
            pd_html = (
                f'<p><a href="{html_escape(alert.pagerduty_url)}">'
                "View in PagerDuty</a></p>"
            )

        return (
            f"<h2>{html_escape(alert.title)}</h2>"
            f"<p><strong>Severity:</strong> {html_escape(alert.severity)}</p>"
            f"<p><strong>Service:</strong> {html_escape(alert.service_name)}</p>"
            f"<p>{html_escape(alert.summary)}</p>"
            f"{findings_html}"
            f"{pd_html}"
        )

    def _build_alert_plain(self, alert: FormattedAlert) -> str:
        """Build plain-text body for an alert email."""
        lines = [
            alert.title,
            f"Severity: {alert.severity}",
            f"Service: {alert.service_name}",
            "",
            alert.summary,
        ]

        if alert.findings:
            lines.append("")
            lines.append("Top Findings:")
            for f in alert.findings:
                lines.append(f"  - {f}")

        if alert.pagerduty_url:
            lines.append("")
            lines.append(f"PagerDuty: {alert.pagerduty_url}")

        return "\n".join(lines)

    def _build_report_html(self, report: FormattedReport) -> str:
        """Build HTML body for a report summary email."""
        findings_html = ""
        if report.findings:
            items = "".join(f"<li>{html_escape(f)}</li>" for f in report.findings)
            findings_html = f"<h3>Top Findings</h3><ul>{items}</ul>"

        exec_html = ""
        if report.execution_time > 0:
            exec_html = f"<p><em>Execution time: {report.execution_time:.1f}s</em></p>"

        return (
            f"<h2>{html_escape(report.title)}</h2>"
            f"<p><strong>Status:</strong> {html_escape(report.status)}</p>"
            f"<p><strong>Scope:</strong> {html_escape(report.scope)}</p>"
            f"<p><strong>Issues:</strong> {report.issues_found} total "
            f"({report.critical_count} critical, {report.warning_count} warning)</p>"
            f"<p>{html_escape(report.summary)}</p>"
            f"{findings_html}"
            f"{exec_html}"
        )

    def _build_report_plain(self, report: FormattedReport) -> str:
        """Build plain-text body for a report summary email."""
        lines = [
            report.title,
            f"Status: {report.status}",
            f"Scope: {report.scope}",
            f"Issues: {report.issues_found} total "
            f"({report.critical_count} critical, {report.warning_count} warning)",
            "",
            report.summary,
        ]

        if report.findings:
            lines.append("")
            lines.append("Top Findings:")
            for f in report.findings:
                lines.append(f"  - {f}")

        if report.execution_time > 0:
            lines.append("")
            lines.append(f"Execution time: {report.execution_time:.1f}s")

        return "\n".join(lines)

    # ── SMTP transport ───────────────────────────────────────

    def _send(
        self,
        subject: str,
        html_body: str,
        plain_body: str,
        recipients: list[str],
    ) -> None:
        """Send a multipart/alternative email via SMTP.

        Uses ``SMTP_SSL`` for port 465, ``SMTP`` + ``STARTTLS`` for
        port 587 with ``use_tls=True``, or plain ``SMTP`` otherwise.

        Raises:
            smtplib.SMTPException: On any SMTP error.
        """
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.from_address
        msg["To"] = ", ".join(recipients)

        msg.attach(MIMEText(plain_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        if self.smtp_port == 465:
            # Implicit SSL
            with smtplib.SMTP_SSL(
                self.smtp_host, self.smtp_port, timeout=self.timeout
            ) as server:
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.sendmail(self.from_address, recipients, msg.as_string())
        else:
            # Plain or STARTTLS
            with smtplib.SMTP(
                self.smtp_host, self.smtp_port, timeout=self.timeout
            ) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.sendmail(self.from_address, recipients, msg.as_string())

        logger.info("Email sent to %s", recipients)

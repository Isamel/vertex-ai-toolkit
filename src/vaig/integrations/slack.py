"""Slack incoming webhook integration using Block Kit messages."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import requests

from vaig.core.config import SlackConfig
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

# ── Constants ────────────────────────────────────────────────
_DEFAULT_TIMEOUT = 30


class SlackWebhook:
    """Slack incoming webhook integration using Block Kit messages.

    Sends structured alert cards and report summaries to a Slack
    channel via an incoming webhook URL.
    """

    def __init__(self, config: SlackConfig) -> None:
        self.webhook_url = config.webhook_url
        self.notify_on = config.notify_on

    def send_alert_card(
        self,
        title: str,
        severity: str,
        service_name: str,
        summary: str,
        findings: list[str] | None = None,
        pagerduty_url: str | None = None,
    ) -> None:
        """Send a structured Block Kit alert message to Slack.

        Args:
            title: Card title (e.g. "Service Health Alert").
            severity: Severity level (CRITICAL, HIGH, MEDIUM, LOW, INFO).
            service_name: Name of the affected service.
            summary: Brief description of the issue.
            findings: Optional list of top finding descriptions.
            pagerduty_url: Optional PagerDuty incident URL.
        """
        severity_upper = severity.upper()

        if not meets_threshold(severity_upper, self.notify_on):
            logger.debug(
                "Skipping Slack alert — severity %s does not meet threshold %s",
                severity_upper,
                self.notify_on,
            )
            return

        alert = FormattedAlert(
            title=title,
            severity=severity_upper,
            severity_icon=_severity_icon(severity_upper),
            service_name=service_name,
            summary=summary,
            findings=findings[:5] if findings else [],
            pagerduty_url=pagerduty_url,
        )

        blocks = self._build_alert_blocks(alert)
        self._send({"blocks": blocks})

    def send_report_summary(
        self,
        report: HealthReport,
        execution_time: float = 0.0,
    ) -> None:
        """Send a compact report summary card after pipeline completion.

        Only sends if the report severity meets the ``notify_on`` threshold.
        Severity is derived from the report's overall status.

        Args:
            report: The completed HealthReport.
            execution_time: Pipeline execution time in seconds.
        """
        formatted = format_report_summary(report, execution_time)
        severity = status_to_severity(
            report.executive_summary.overall_status.value,
        )

        if not meets_threshold(severity, self.notify_on):
            logger.debug(
                "Skipping Slack report summary — severity %s does not meet threshold",
                severity,
            )
            return

        blocks = self._build_report_blocks(formatted)
        self._send({"blocks": blocks})

    # ── Block Kit builders ───────────────────────────────────

    def _build_alert_blocks(self, alert: FormattedAlert) -> list[dict[str, Any]]:
        """Build Block Kit blocks for an alert card."""
        blocks: list[dict[str, Any]] = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": alert.title},
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Severity:*\n{alert.severity_icon} {alert.severity}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Service:*\n{alert.service_name}",
                    },
                ],
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": alert.summary},
            },
        ]

        if alert.findings:
            findings_text = "\n".join(f"• {f}" for f in alert.findings)
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Top Findings:*\n{findings_text}",
                    },
                }
            )

        if alert.pagerduty_url:
            blocks.append(
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "View in PagerDuty",
                            },
                            "url": alert.pagerduty_url,
                        },
                    ],
                }
            )

        return blocks

    def _build_report_blocks(self, report: FormattedReport) -> list[dict[str, Any]]:
        """Build Block Kit blocks for a report summary card."""
        blocks: list[dict[str, Any]] = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": report.title},
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Status:*\n{report.status_icon} {report.status}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Scope:*\n{report.scope}",
                    },
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Issues:* {report.issues_found} total "
                        f"({report.critical_count} critical, "
                        f"{report.warning_count} warning)"
                    ),
                },
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": report.summary},
            },
        ]

        if report.findings:
            findings_text = "\n".join(f"• {f}" for f in report.findings)
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Top Findings:*\n{findings_text}",
                    },
                }
            )

        if report.execution_time > 0:
            blocks.append(
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"⏱ Execution time: {report.execution_time:.1f}s",
                        },
                    ],
                }
            )

        return blocks

    # ── Private transport ────────────────────────────────────

    def _send(self, payload: dict[str, Any]) -> None:
        """POST a Block Kit message payload to the Slack webhook.

        Raises:
            requests.exceptions.HTTPError: If the webhook returns an HTTP error.
            requests.exceptions.Timeout: If the webhook request times out.
        """
        resp = requests.post(
            self.webhook_url,
            json=payload,
            timeout=_DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        logger.info("Slack message sent successfully")


# ── Module-level helpers ─────────────────────────────────────

_SEVERITY_ICON: dict[str, str] = {
    "CRITICAL": "\U0001f534",
    "HIGH": "\U0001f7e0",
    "MEDIUM": "\U0001f7e1",
    "LOW": "\U0001f7e2",
    "INFO": "\U0001f535",
}


def _severity_icon(severity: str) -> str:
    """Return the emoji icon for a severity level."""
    return _SEVERITY_ICON.get(severity.upper(), "\u2753")

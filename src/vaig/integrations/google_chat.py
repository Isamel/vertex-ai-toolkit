"""Google Chat incoming webhook integration using Card v2 messages."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import requests

from vaig.core.config import GoogleChatConfig
from vaig.integrations.formatters import (
    FormattedAlert,
    FormattedReport,
    format_report_summary,
    meets_threshold,
    severity_icon,
    status_to_severity,
)

if TYPE_CHECKING:
    from vaig.skills.service_health.schema import HealthReport

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────
_DEFAULT_TIMEOUT = 30


class GoogleChatWebhook:
    """Google Chat incoming webhook integration using Card v2 messages.

    Sends structured alert cards and report summaries to a Google Chat
    space via an incoming webhook URL.
    """

    def __init__(self, config: GoogleChatConfig) -> None:
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
        """Send a structured Card v2 alert message to Google Chat.

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
                "Skipping Google Chat alert — severity %s does not meet threshold %s",
                severity_upper,
                self.notify_on,
            )
            return

        alert = FormattedAlert(
            title=title,
            severity=severity_upper,
            severity_icon=severity_icon(severity_upper),
            service_name=service_name,
            summary=summary,
            findings=findings[:5] if findings else [],
            pagerduty_url=pagerduty_url,
        )

        widgets = self._build_alert_widgets(alert)
        card_payload = self._build_card_message(alert.title, widgets)
        self._send(card_payload)

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
                "Skipping Google Chat report summary — severity %s does not meet threshold",
                severity,
            )
            return

        widgets = self._build_report_widgets(formatted)
        card_payload = self._build_card_message(formatted.title, widgets)
        self._send(card_payload)

    # ── Private helpers ──────────────────────────────────────

    def _build_alert_widgets(self, alert: FormattedAlert) -> list[dict[str, Any]]:
        """Build Card v2 widgets for an alert card from FormattedAlert."""
        widgets: list[dict[str, Any]] = [
            {
                "decoratedText": {
                    "topLabel": "Severity",
                    "text": f"{alert.severity_icon} {alert.severity}",
                },
            },
            {
                "decoratedText": {
                    "topLabel": "Service",
                    "text": alert.service_name,
                },
            },
            {
                "textParagraph": {
                    "text": alert.summary,
                },
            },
        ]

        if alert.findings:
            findings_text = "\n".join(f"• {f}" for f in alert.findings)
            widgets.append(
                {
                    "textParagraph": {
                        "text": f"<b>Top Findings:</b>\n{findings_text}",
                    },
                }
            )

        buttons: list[dict[str, Any]] = []
        if alert.pagerduty_url:
            buttons.append(
                {
                    "text": "View in PagerDuty",
                    "onClick": {
                        "openLink": {"url": alert.pagerduty_url},
                    },
                }
            )

        if buttons:
            widgets.append({"buttonList": {"buttons": buttons}})

        return widgets

    def _build_report_widgets(self, formatted: FormattedReport) -> list[dict[str, Any]]:
        """Build Card v2 widgets for a report summary from FormattedReport."""
        widgets: list[dict[str, Any]] = [
            {
                "decoratedText": {
                    "topLabel": "Status",
                    "text": f"{formatted.status_icon} {formatted.status}",
                },
            },
            {
                "decoratedText": {
                    "topLabel": "Scope",
                    "text": formatted.scope,
                },
            },
            {
                "decoratedText": {
                    "topLabel": "Issues",
                    "text": f"{formatted.issues_found} total ({formatted.critical_count} critical, {formatted.warning_count} warning)",
                },
            },
            {
                "textParagraph": {
                    "text": formatted.summary,
                },
            },
        ]

        if formatted.findings:
            findings_text = "\n".join(f"• {f}" for f in formatted.findings)
            widgets.append(
                {
                    "textParagraph": {
                        "text": f"<b>Top Findings:</b>\n{findings_text}",
                    },
                }
            )

        if formatted.execution_time > 0:
            widgets.append(
                {
                    "decoratedText": {
                        "topLabel": "Execution Time",
                        "text": f"{formatted.execution_time:.1f}s",
                    },
                }
            )

        return widgets

    def _build_card_message(
        self,
        title: str,
        widgets: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build a Google Chat Card v2 message payload."""
        return {
            "cardsV2": [
                {
                    "cardId": "health-alert",
                    "card": {
                        "header": {
                            "title": title,
                            "subtitle": "VAIG Service Health",
                        },
                        "sections": [
                            {
                                "widgets": widgets,
                            },
                        ],
                    },
                },
            ],
        }

    def _send(self, payload: dict[str, Any]) -> None:
        """POST a message payload to the Google Chat webhook.

        Raises:
            requests.exceptions.HTTPError: If the webhook returns an HTTP error.
            requests.exceptions.Timeout: If the webhook request times out.
            Exception: Any other unexpected error during the POST.
        """
        resp = requests.post(
            self.webhook_url,
            json=payload,
            timeout=_DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        logger.info("Google Chat message sent successfully")

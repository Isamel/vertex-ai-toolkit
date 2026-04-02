"""Google Chat incoming webhook integration using Card v2 messages."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import requests

from vaig.core.config import GoogleChatConfig

if TYPE_CHECKING:
    from vaig.skills.service_health.schema import HealthReport

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────
_DEFAULT_TIMEOUT = 30

_SEVERITY_ICON: dict[str, str] = {
    "CRITICAL": "🔴",
    "HIGH": "🟠",
    "MEDIUM": "🟡",
    "LOW": "🟢",
    "INFO": "🔵",
}

# Severity ordering for threshold comparison
_SEVERITY_ORDER: dict[str, int] = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
    "info": 0,
}


def _normalize_notify_on(notify_on: list[str]) -> list[str]:
    """Return normalized valid severities from ``notify_on``.

    Unknown values are silently dropped so misconfiguration fails safe
    (no alerts) rather than fail open.
    """
    normalized: list[str] = []
    for value in notify_on:
        severity = value.lower()
        if severity in _SEVERITY_ORDER:
            normalized.append(severity)
    return normalized


def _meets_threshold(severity: str, notify_on: list[str]) -> bool:
    """Check if severity meets the minimum notification threshold.

    The threshold is determined by the *lowest* valid severity in
    ``notify_on``.  Any severity at or above that threshold will pass.
    Invalid severities in ``notify_on`` are ignored so misconfiguration
    fails safe (no alerts) rather than fail open.
    """
    if not notify_on:
        return False

    severity_lower = severity.lower()
    if severity_lower not in _SEVERITY_ORDER:
        return False

    valid_notify_on = _normalize_notify_on(notify_on)
    if not valid_notify_on:
        return False

    min_threshold = min(_SEVERITY_ORDER[s] for s in valid_notify_on)
    return _SEVERITY_ORDER[severity_lower] >= min_threshold


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

        if not _meets_threshold(severity_upper, self.notify_on):
            logger.debug(
                "Skipping Google Chat alert — severity %s does not meet threshold %s",
                severity_upper,
                self.notify_on,
            )
            return

        icon = _SEVERITY_ICON.get(severity_upper, "❓")

        # Build card widgets
        widgets: list[dict[str, Any]] = [
            {
                "decoratedText": {
                    "topLabel": "Severity",
                    "text": f"{icon} {severity_upper}",
                },
            },
            {
                "decoratedText": {
                    "topLabel": "Service",
                    "text": service_name,
                },
            },
            {
                "textParagraph": {
                    "text": summary,
                },
            },
        ]

        # Add top findings
        if findings:
            findings_text = "\n".join(f"• {f}" for f in findings[:5])
            widgets.append(
                {
                    "textParagraph": {
                        "text": f"<b>Top Findings:</b>\n{findings_text}",
                    },
                }
            )

        # Add PagerDuty link button if available
        buttons: list[dict[str, Any]] = []
        if pagerduty_url:
            buttons.append(
                {
                    "text": "View in PagerDuty",
                    "onClick": {
                        "openLink": {"url": pagerduty_url},
                    },
                }
            )

        if buttons:
            widgets.append({"buttonList": {"buttons": buttons}})

        card_payload = self._build_card_message(title, widgets)
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
        es = report.executive_summary
        severity = _status_to_severity(es.overall_status.value)

        if not _meets_threshold(severity, self.notify_on):
            logger.debug(
                "Skipping Google Chat report summary — severity %s does not meet threshold",
                severity,
            )
            return

        icon = _SEVERITY_ICON.get(severity.upper(), "❓")

        # Extract top findings (max 3)
        top_findings = [f.title for f in report.findings[:3]] if report.findings else []
        findings_text = "\n".join(f"• {f}" for f in top_findings) if top_findings else "No findings."

        widgets: list[dict[str, Any]] = [
            {
                "decoratedText": {
                    "topLabel": "Status",
                    "text": f"{icon} {es.overall_status.value}",
                },
            },
            {
                "decoratedText": {
                    "topLabel": "Scope",
                    "text": es.scope,
                },
            },
            {
                "decoratedText": {
                    "topLabel": "Issues",
                    "text": f"{es.issues_found} total ({es.critical_count} critical, {es.warning_count} warning)",
                },
            },
            {
                "textParagraph": {
                    "text": es.summary_text,
                },
            },
        ]

        if top_findings:
            widgets.append(
                {
                    "textParagraph": {
                        "text": f"<b>Top Findings:</b>\n{findings_text}",
                    },
                }
            )

        if execution_time > 0:
            widgets.append(
                {
                    "decoratedText": {
                        "topLabel": "Execution Time",
                        "text": f"{execution_time:.1f}s",
                    },
                }
            )

        card_payload = self._build_card_message("Service Health Report", widgets)
        self._send(card_payload)

    # ── Private helpers ──────────────────────────────────────

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


def _status_to_severity(overall_status: str) -> str:
    """Map OverallStatus to a severity string for threshold comparison."""
    mapping: dict[str, str] = {
        "CRITICAL": "CRITICAL",
        "DEGRADED": "HIGH",
        "HEALTHY": "INFO",
        "UNKNOWN": "MEDIUM",
    }
    return mapping.get(overall_status.upper(), "MEDIUM")

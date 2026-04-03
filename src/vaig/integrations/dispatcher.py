"""Notification dispatcher — fan-out notifications to PagerDuty, Google Chat, Slack, and Email."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vaig.core.config import EmailConfig, GoogleChatConfig, PagerDutyConfig, SlackConfig
from vaig.integrations.email_sender import EmailSender
from vaig.integrations.formatters import meets_threshold, status_to_severity
from vaig.integrations.google_chat import GoogleChatWebhook
from vaig.integrations.pagerduty import PagerDutyClient
from vaig.integrations.slack import SlackWebhook

if TYPE_CHECKING:
    from vaig.skills.service_health.schema import HealthReport

logger = logging.getLogger(__name__)


@dataclass
class AlertContext:
    """Context from a triggering alert (e.g., Datadog webhook, scheduler)."""

    alert_id: str
    source: str  # "datadog", "scheduler", "manual", etc.
    service_name: str
    cluster_name: str = ""
    namespace: str = ""
    schedule_id: str | None = None
    run_number: int | None = None
    is_scheduled: bool = False


@dataclass
class DispatchResult:
    """Result of notification dispatch."""

    pagerduty_dedup_key: str | None = None
    pagerduty_incident_id: str | None = None
    google_chat_sent: bool = False
    slack_sent: bool = False
    email_sent: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Return True if any errors occurred during dispatch."""
        return len(self.errors) > 0


class NotificationDispatcher:
    """Fan-out notifications to PagerDuty, Google Chat, Slack, and Email.

    Dispatches are independent — a failure in one channel does not
    block the others.  All errors are collected in ``DispatchResult.errors``.
    """

    def __init__(
        self,
        pagerduty: PagerDutyClient | None = None,
        google_chat: GoogleChatWebhook | None = None,
        slack: SlackWebhook | None = None,
        email: EmailSender | None = None,
    ) -> None:
        self.pagerduty = pagerduty
        self.google_chat = google_chat
        self.slack = slack
        self.email = email

    @classmethod
    def from_config(cls, config: Any) -> NotificationDispatcher:
        """Factory: build dispatcher from app config.

        Returns ``None``-clients for disabled integrations.

        Args:
            config: The root ``Settings`` object (or anything with
                ``.pagerduty``, ``.google_chat``, ``.slack``, and
                ``.email`` attributes).
        """
        pd_client: PagerDutyClient | None = None
        gc_client: GoogleChatWebhook | None = None
        slack_client: SlackWebhook | None = None
        email_client: EmailSender | None = None

        pd_config: PagerDutyConfig | None = getattr(config, "pagerduty", None)
        if pd_config is not None and pd_config.enabled:
            pd_client = PagerDutyClient(pd_config)
            logger.info("PagerDuty integration enabled")

        gc_config: GoogleChatConfig | None = getattr(config, "google_chat", None)
        if gc_config is not None and gc_config.enabled:
            gc_client = GoogleChatWebhook(gc_config)
            logger.info("Google Chat integration enabled")

        slack_config: SlackConfig | None = getattr(config, "slack", None)
        if slack_config is not None and slack_config.enabled:
            slack_client = SlackWebhook(slack_config)
            logger.info("Slack integration enabled")

        email_config: EmailConfig | None = getattr(config, "email", None)
        if email_config is not None and email_config.enabled:
            email_client = EmailSender(email_config)
            logger.info("Email integration enabled")

        return cls(
            pagerduty=pd_client,
            google_chat=gc_client,
            slack=slack_client,
            email=email_client,
        )

    def dispatch(
        self,
        report: HealthReport,
        alert_context: AlertContext | None = None,
    ) -> DispatchResult:
        """Send notifications based on severity and config rules.

        Dispatch flow:
        1. Determine severity from report's overall status.
        2. If PagerDuty is enabled and severity warrants → trigger event.
        3. If PagerDuty ``api_token`` is available → find incident + attach report.
        4. If Google Chat is enabled and severity meets threshold → send card.
        5. If Slack is enabled and severity meets threshold → send Block Kit card.
        6. If Email is enabled and severity meets threshold → send HTML email.

        Args:
            report: The HealthReport to dispatch notifications for.
            alert_context: Optional context from the triggering alert.

        Returns:
            A ``DispatchResult`` with what was sent and any errors.
        """
        result = DispatchResult()
        es = report.executive_summary
        severity = status_to_severity(es.overall_status.value)

        # Derive context
        source = alert_context.source if alert_context else "vaig"
        service_name = alert_context.service_name if alert_context else es.scope

        # ── PagerDuty ────────────────────────────────────────
        if self.pagerduty is not None and severity != "INFO":
            pd_severity = self.pagerduty.severity_mapping.get(
                severity.lower(), "warning"
            )
            dedup_key = alert_context.alert_id if alert_context else None

            try:
                result.pagerduty_dedup_key = self.pagerduty.trigger_event(
                    summary=f"[{severity}] {es.summary_text}"[:1024],
                    severity=pd_severity,
                    source=source,
                    dedup_key=dedup_key,
                    custom_details={
                        "overall_status": es.overall_status.value,
                        "issues_found": es.issues_found,
                        "critical_count": es.critical_count,
                        "warning_count": es.warning_count,
                        "scope": es.scope,
                    },
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                error_msg = f"PagerDuty trigger failed: {exc}"
                logger.exception(error_msg)
                result.errors.append(error_msg)

            # Try to find incident and attach report
            if result.pagerduty_dedup_key and not result.has_errors:
                try:
                    incident_id = self.pagerduty.find_incident_by_dedup_key(
                        result.pagerduty_dedup_key
                    )
                    if incident_id:
                        result.pagerduty_incident_id = incident_id
                        self.pagerduty.attach_report_to_incident(incident_id, report)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as exc:
                    error_msg = f"PagerDuty enrichment failed: {exc}"
                    logger.exception(error_msg)
                    result.errors.append(error_msg)

        # Build PagerDuty URL for messaging channels
        pd_url: str | None = None
        if result.pagerduty_incident_id and self.pagerduty is not None:
            pd_url = (
                f"{self.pagerduty.base_url.replace('api.', 'app.')}"
                f"/incidents/{result.pagerduty_incident_id}"
            )
        findings_text = [f.title for f in (report.findings or [])[:5]]

        # ── Google Chat ──────────────────────────────────────
        if self.google_chat is not None and meets_threshold(
            severity, self.google_chat.notify_on
        ):
            try:
                self.google_chat.send_alert_card(
                    title="Service Health Alert",
                    severity=severity,
                    service_name=service_name,
                    summary=es.summary_text,
                    findings=findings_text,
                    pagerduty_url=pd_url,
                )
                result.google_chat_sent = True
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                error_msg = f"Google Chat alert failed: {exc}"
                logger.exception(error_msg)
                result.errors.append(error_msg)

        # ── Slack ────────────────────────────────────────────
        if self.slack is not None and meets_threshold(
            severity, self.slack.notify_on
        ):
            try:
                self.slack.send_alert_card(
                    title="Service Health Alert",
                    severity=severity,
                    service_name=service_name,
                    summary=es.summary_text,
                    findings=findings_text,
                    pagerduty_url=pd_url,
                )
                result.slack_sent = True
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                error_msg = f"Slack alert failed: {exc}"
                logger.exception(error_msg)
                result.errors.append(error_msg)

        # ── Email ────────────────────────────────────────────
        if self.email is not None and meets_threshold(
            severity, self.email.notify_on
        ):
            try:
                self.email.send_alert_email(
                    title="Service Health Alert",
                    severity=severity,
                    service_name=service_name,
                    summary=es.summary_text,
                    findings=findings_text,
                    pagerduty_url=pd_url,
                )
                result.email_sent = True
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                error_msg = f"Email alert failed: {exc}"
                logger.exception(error_msg)
                result.errors.append(error_msg)

        return result

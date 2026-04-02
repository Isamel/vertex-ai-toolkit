"""Notification dispatcher — fan-out notifications to PagerDuty + Google Chat."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vaig.core.config import GoogleChatConfig, PagerDutyConfig
from vaig.integrations.google_chat import (
    GoogleChatWebhook,
    _meets_threshold,
    _status_to_severity,
)
from vaig.integrations.pagerduty import PagerDutyClient

if TYPE_CHECKING:
    from vaig.skills.service_health.schema import HealthReport

logger = logging.getLogger(__name__)


@dataclass
class AlertContext:
    """Context from a triggering alert (e.g., Datadog webhook)."""

    alert_id: str
    source: str  # "datadog", "manual", etc.
    service_name: str
    cluster_name: str = ""
    namespace: str = ""


@dataclass
class DispatchResult:
    """Result of notification dispatch."""

    pagerduty_dedup_key: str | None = None
    pagerduty_incident_id: str | None = None
    google_chat_sent: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Return True if any errors occurred during dispatch."""
        return len(self.errors) > 0


class NotificationDispatcher:
    """Fan-out notifications to PagerDuty + Google Chat from a HealthReport.

    Dispatches are independent — a failure in one channel does not
    block the other.  All errors are collected in ``DispatchResult.errors``.
    """

    def __init__(
        self,
        pagerduty: PagerDutyClient | None = None,
        google_chat: GoogleChatWebhook | None = None,
    ) -> None:
        self.pagerduty = pagerduty
        self.google_chat = google_chat

    @classmethod
    def from_config(cls, config: Any) -> NotificationDispatcher:
        """Factory: build dispatcher from app config.

        Returns ``None``-clients for disabled integrations.

        Args:
            config: The root ``Settings`` object (or anything with
                ``.pagerduty`` and ``.google_chat`` attributes).
        """
        pd_client: PagerDutyClient | None = None
        gc_client: GoogleChatWebhook | None = None

        pd_config: PagerDutyConfig | None = getattr(config, "pagerduty", None)
        if pd_config is not None and pd_config.enabled:
            pd_client = PagerDutyClient(pd_config)
            logger.info("PagerDuty integration enabled")

        gc_config: GoogleChatConfig | None = getattr(config, "google_chat", None)
        if gc_config is not None and gc_config.enabled:
            gc_client = GoogleChatWebhook(gc_config)
            logger.info("Google Chat integration enabled")

        return cls(pagerduty=pd_client, google_chat=gc_client)

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

        Args:
            report: The HealthReport to dispatch notifications for.
            alert_context: Optional context from the triggering alert.

        Returns:
            A ``DispatchResult`` with what was sent and any errors.
        """
        result = DispatchResult()
        es = report.executive_summary
        severity = _status_to_severity(es.overall_status.value)

        # Derive context
        source = alert_context.source if alert_context else "vaig"
        service_name = alert_context.service_name if alert_context else es.scope

        # ── PagerDuty ────────────────────────────────────────
        if self.pagerduty is not None:
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

        # ── Google Chat ──────────────────────────────────────
        if self.google_chat is not None and _meets_threshold(
            severity, self.google_chat.notify_on
        ):
            # Build PagerDuty URL for the card if we have an incident
            pd_url: str | None = None
            if result.pagerduty_incident_id and self.pagerduty is not None:
                pd_url = (
                    f"{self.pagerduty.base_url.replace('api.', 'app.')}"
                    f"/incidents/{result.pagerduty_incident_id}"
                )

            findings_text = [f.title for f in report.findings[:5]]

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

        return result

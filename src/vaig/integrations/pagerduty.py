"""PagerDuty Events API v2 + REST API v2 integration."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

import requests

from vaig.core.config import PagerDutyConfig

if TYPE_CHECKING:
    from vaig.skills.service_health.schema import HealthReport

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────
_EVENTS_API_URL = "https://events.pagerduty.com/v2/enqueue"
_DEFAULT_TIMEOUT = 30


class PagerDutyClient:
    """PagerDuty Events API v2 + REST API v2 integration.

    The Events API v2 uses a ``routing_key`` to trigger, acknowledge,
    and resolve incidents.  The REST API v2 (optional) uses an
    ``api_token`` for enrichment operations like adding notes and
    searching incidents by dedup key.
    """

    def __init__(self, config: PagerDutyConfig) -> None:
        self.routing_key = config.routing_key
        self.api_token = config.api_token
        self.base_url = config.base_url.rstrip("/")
        self.severity_mapping = config.severity_mapping

    # ── Events API v2 ────────────────────────────────────────

    def trigger_event(
        self,
        summary: str,
        severity: str,
        source: str,
        dedup_key: str | None = None,
        custom_details: dict[str, Any] | None = None,
    ) -> str:
        """Trigger an incident via Events API v2.

        Args:
            summary: Human-readable summary of the event.
            severity: PagerDuty severity (critical, error, warning, info).
            source: Source of the event (e.g. service name).
            dedup_key: Deduplication key.  Auto-generated if not provided.
            custom_details: Additional structured data for the event.

        Returns:
            The dedup_key used for the event.
        """
        if dedup_key is None:
            dedup_key = str(uuid.uuid4())

        payload: dict[str, Any] = {
            "routing_key": self.routing_key,
            "event_action": "trigger",
            "dedup_key": dedup_key,
            "payload": {
                "summary": summary[:1024],  # PD limit
                "severity": severity,
                "source": source,
            },
        }
        if custom_details:
            payload["payload"]["custom_details"] = custom_details

        self._send_event(payload)
        return dedup_key

    def acknowledge_event(self, dedup_key: str) -> None:
        """Acknowledge an incident via Events API v2."""
        payload = {
            "routing_key": self.routing_key,
            "event_action": "acknowledge",
            "dedup_key": dedup_key,
        }
        self._send_event(payload)

    def resolve_event(self, dedup_key: str) -> None:
        """Resolve an incident via Events API v2."""
        payload = {
            "routing_key": self.routing_key,
            "event_action": "resolve",
            "dedup_key": dedup_key,
        }
        self._send_event(payload)

    # ── REST API v2 (requires api_token) ─────────────────────

    def find_incident_by_dedup_key(self, dedup_key: str) -> str | None:
        """Search REST API v2 for an incident by dedup_key.

        Returns:
            The incident ID if found, or ``None``.
        """
        if not self.api_token:
            logger.warning(
                "PagerDuty api_token not configured — cannot search incidents. "
                "Set VAIG_PAGERDUTY__API_TOKEN to enable incident enrichment."
            )
            return None

        try:
            resp = requests.get(
                f"{self.base_url}/incidents",
                params={"incident_key": dedup_key},
                headers=self._rest_headers(),
                timeout=_DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            incidents = data.get("incidents", [])
            if incidents:
                return str(incidents[0]["id"])
            return None
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            logger.exception("Failed to search PagerDuty incidents for dedup_key=%s", dedup_key)
            return None

    def add_incident_note(self, incident_id: str, content: str) -> None:
        """Add a note to an existing incident via REST API v2.

        Requires ``api_token`` to be configured.
        """
        if not self.api_token:
            logger.warning(
                "PagerDuty api_token not configured — cannot add incident notes."
            )
            return

        try:
            resp = requests.post(
                f"{self.base_url}/incidents/{incident_id}/notes",
                json={
                    "note": {
                        "content": content[:65535],  # PD note content limit
                    },
                },
                headers=self._rest_headers(),
                timeout=_DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            logger.info("Added note to PagerDuty incident %s", incident_id)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            logger.exception("Failed to add note to PagerDuty incident %s", incident_id)

    def attach_report_to_incident(self, incident_id: str, report: HealthReport) -> None:
        """Render a HealthReport as Markdown and attach as an incident note.

        Gracefully skips if ``api_token`` is not configured.
        """
        if not self.api_token:
            logger.warning(
                "PagerDuty api_token not configured — cannot attach report to incident."
            )
            return

        try:
            markdown = report.to_markdown()
            # Truncate if needed — PD notes have a 65535 char limit
            if len(markdown) > 60000:
                markdown = markdown[:60000] + "\n\n... (report truncated)"
            self.add_incident_note(incident_id, markdown)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            logger.exception(
                "Failed to attach report to PagerDuty incident %s", incident_id
            )

    # ── Private helpers ──────────────────────────────────────

    def _send_event(self, payload: dict[str, Any]) -> None:
        """POST an event to the PagerDuty Events API v2."""
        try:
            resp = requests.post(
                _EVENTS_API_URL,
                json=payload,
                timeout=_DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            logger.info(
                "PagerDuty event sent: action=%s dedup_key=%s",
                payload.get("event_action"),
                payload.get("dedup_key"),
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except requests.exceptions.HTTPError:
            logger.exception(
                "PagerDuty Events API returned HTTP error for action=%s",
                payload.get("event_action"),
            )
            raise
        except requests.exceptions.Timeout:
            logger.error(
                "PagerDuty Events API timed out for action=%s",
                payload.get("event_action"),
            )
            raise
        except Exception:
            logger.exception(
                "Unexpected error sending PagerDuty event action=%s",
                payload.get("event_action"),
            )
            raise

    def _rest_headers(self) -> dict[str, str]:
        """Build headers for REST API v2 requests."""
        return {
            "Authorization": f"Token token={self.api_token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.pagerduty+json;version=2",
        }

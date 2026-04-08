"""PagerDuty Events API v2 + REST API v2 integration."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

import requests

from vaig.core.config import PagerDutyConfig

if TYPE_CHECKING:
    from vaig.integrations.finding_exporter import ExportResult
    from vaig.skills.service_health.schema import Finding, HealthReport

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

    # ── Finding-level export ───────────────────────────────────

    def create_incident_from_finding(
        self,
        finding: Finding,
        *,
        timeline_events: list[dict[str, Any]] | None = None,
        cluster_context: str = "",
    ) -> ExportResult:
        """Create a PagerDuty incident from a health-report finding.

        Uses Events API v2 with a dedup_key of ``{cluster}:{finding.id}``
        so re-exporting the same finding is idempotent.  If ``api_token``
        is configured, adds a formatted note with finding details.

        Args:
            finding: A :class:`Finding` model instance.
            timeline_events: Optional timeline events to include as notes.
            cluster_context: Cluster name prefix for the dedup key.

        Returns:
            :class:`ExportResult` with outcome details.
        """
        from vaig.integrations.finding_exporter import ExportResult

        finding_id = finding.id
        severity_str = self._extract_severity(finding.severity)
        dedup_key = f"{cluster_context}:{finding_id}" if cluster_context else finding_id

        try:
            self.trigger_event(
                summary=finding.title,
                severity=self.severity_mapping.get(severity_str, "warning"),
                source=finding.service or "vaig",
                dedup_key=dedup_key,
                custom_details={
                    "finding_id": finding_id,
                    "category": finding.category,
                    "description": finding.description,
                    "root_cause": finding.root_cause,
                    "impact": finding.impact,
                    "remediation": finding.remediation or "",
                },
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            return ExportResult(
                target="pagerduty",
                success=False,
                error=str(exc),
            )

        # Enrich with note if api_token is available
        url = ""
        incident_id = ""
        if self.api_token:
            found_id = self.find_incident_by_dedup_key(dedup_key)
            if found_id:
                incident_id = found_id
                url = f"{self.base_url}/incidents/{incident_id}"
                note = self._format_finding_note(finding)
                self.add_incident_note(incident_id, note)

                # Add filtered timeline notes
                if timeline_events:
                    service_filter = finding.service.lower() if finding.service else ""
                    filtered = [
                        e for e in timeline_events
                        if not service_filter or service_filter in str(e.get("service", "")).lower()
                    ]
                    for event in filtered[:10]:  # Limit to avoid excessive API calls
                        self.add_incident_note(
                            incident_id,
                            f"[Timeline] {event.get('timestamp', '')}: {event.get('description', '')}",
                        )

        return ExportResult(
            target="pagerduty",
            success=True,
            url=url,
            key=incident_id or dedup_key,
        )

    @staticmethod
    def _format_finding_note(finding: Finding) -> str:
        """Format a finding as a Markdown note for PagerDuty."""
        parts = [
            f"## {finding.title}",
            f"**Severity:** {finding.severity}",
            f"**Category:** {finding.category}",
        ]
        if finding.service:
            parts.append(f"**Service:** {finding.service}")
        if finding.description:
            parts.append(f"\n{finding.description}")
        if finding.root_cause:
            parts.append(f"\n**Root Cause:** {finding.root_cause}")
        if finding.impact:
            parts.append(f"\n**Impact:** {finding.impact}")
        if finding.remediation:
            parts.append(f"\n**Remediation:** {finding.remediation}")
        if finding.evidence:
            parts.append("\n**Evidence:**")
            for e in finding.evidence:
                parts.append(f"- {e}")
        if finding.affected_resources:
            parts.append("\n**Affected Resources:**")
            for r in finding.affected_resources:
                parts.append(f"- {r}")
        return "\n".join(parts)

    # ── Private helpers ──────────────────────────────────────

    @staticmethod
    def _extract_severity(severity: object) -> str:
        """Normalise a severity value to a lowercase string.

        Handles both ``Enum``-style (``severity.value``) and plain-string
        severity fields so callers don't need to inspect ``hasattr``
        manually.
        """
        if hasattr(severity, "value"):
            return str(severity.value).lower()
        return str(severity).lower()

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

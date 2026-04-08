"""Target-agnostic finding export orchestrator — Jira + PagerDuty."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vaig.core.report_store import ReportStore
    from vaig.integrations.jira import JiraClient
    from vaig.integrations.pagerduty import PagerDutyClient

logger = logging.getLogger(__name__)


@dataclass
class ExportResult:
    """Unified result for any export target."""

    target: str  # "jira" | "pagerduty"
    success: bool
    url: str = ""
    key: str = ""
    already_existed: bool = False
    error: str = ""


class FindingExporter:
    """Look up findings from :class:`ReportStore` and dispatch to export targets.

    Supports Jira and PagerDuty.  Finding lookup uses exact slug match
    first, then substring match.
    """

    def __init__(
        self,
        jira: JiraClient | None = None,
        pagerduty: PagerDutyClient | None = None,
        *,
        report_store: ReportStore | None = None,
    ) -> None:
        self._jira = jira
        self._pagerduty = pagerduty
        self._report_store = report_store

    # ── Public API ───────────────────────────────────────────

    def export(
        self,
        finding_slug: str,
        target: str,
        *,
        report_id: str | None = None,
        cluster_context: str = "",
    ) -> ExportResult:
        """Export a finding to the specified target.

        Args:
            finding_slug: The finding ID slug to look up.
            target: Export target — ``"jira"`` or ``"pagerduty"``.
            report_id: Optional run_id to narrow search.
            cluster_context: Cluster name for PD dedup_key prefix.

        Returns:
            :class:`ExportResult` with outcome details.
        """
        result = self.find_finding(finding_slug)
        if result is None:
            return ExportResult(
                target=target,
                success=False,
                error=(
                    f"Finding '{finding_slug}' not found. "
                    "Run `vaig incident list` to see available findings."
                ),
            )
        finding_dict, report_meta = result

        if target == "jira":
            return self._export_to_jira(finding_dict, report_meta)
        if target == "pagerduty":
            return self._export_to_pagerduty(finding_dict, report_meta, cluster_context)

        return ExportResult(
            target=target,
            success=False,
            error=f"Unknown export target: {target!r}. Use 'jira' or 'pagerduty'.",
        )

    def find_finding(self, slug: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Look up a finding by slug from the report store.

        Returns:
            Tuple of ``(finding_dict, report_record)`` or ``None``.
        """
        if self._report_store is None:
            return None

        reports = self._report_store.read_reports(last=50)

        # Pass 1: exact match
        for record in reversed(reports):
            report = record.get("report", {})
            for finding in report.get("findings", []):
                if finding.get("id") == slug:
                    return finding, record

        # Pass 2: substring match
        for record in reversed(reports):
            report = record.get("report", {})
            for finding in report.get("findings", []):
                fid = finding.get("id", "")
                if slug in fid:
                    return finding, record

        return None

    def list_findings(self, last: int = 20) -> list[dict[str, Any]]:
        """List recent findings across stored reports.

        Args:
            last: Max number of report records to scan.

        Returns:
            List of dicts with ``id``, ``title``, ``severity``, ``service``,
            ``timestamp``, and ``run_id``.
        """
        if self._report_store is None:
            return []

        reports = self._report_store.read_reports(last=last)
        findings: list[dict[str, Any]] = []
        seen: set[str] = set()

        for record in reversed(reports):
            report = record.get("report", {})
            for finding in report.get("findings", []):
                fid = finding.get("id", "")
                if fid and fid not in seen:
                    seen.add(fid)
                    findings.append(
                        {
                            "id": fid,
                            "title": finding.get("title", ""),
                            "severity": finding.get("severity", ""),
                            "service": finding.get("service", ""),
                            "timestamp": record.get("timestamp", ""),
                            "run_id": record.get("run_id", ""),
                        }
                    )
        return findings

    # ── Private: Jira export ─────────────────────────────────

    def _export_to_jira(
        self,
        finding: dict[str, Any],
        report_meta: dict[str, Any],
    ) -> ExportResult:
        """Export a finding to Jira."""
        if self._jira is None:
            return ExportResult(
                target="jira",
                success=False,
                error=(
                    "Jira integration not configured. "
                    "Set VAIG_JIRA__BASE_URL and VAIG_JIRA__API_TOKEN."
                ),
            )

        finding_id = finding.get("id", "")

        # Dedup check
        existing_key = self._jira._search_existing(finding_id)
        if existing_key:
            return ExportResult(
                target="jira",
                success=True,
                url=self._jira.issue_url(existing_key),
                key=existing_key,
                already_existed=True,
            )

        # Build description
        description = self._build_jira_description(finding)
        severity = str(finding.get("severity", "MEDIUM")).upper()
        priority = self._jira.severity_field_mapping.get(severity, "Medium")

        try:
            data = self._jira.create_issue(
                summary=finding.get("title", finding_id),
                description=description,
                priority=priority,
                labels=[finding.get("category", ""), finding_id],
            )
            issue_key = data.get("key", "")
            url = self._jira.issue_url(issue_key)

            # Add evidence as a comment
            evidence = finding.get("evidence", [])
            if evidence:
                comment = "**Evidence:**\n" + "\n".join(f"- {e}" for e in evidence)
                self._jira.add_comment(issue_key, comment)

            return ExportResult(
                target="jira",
                success=True,
                url=url,
                key=issue_key,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            return ExportResult(
                target="jira",
                success=False,
                error=str(exc),
            )

    # ── Private: PagerDuty export ────────────────────────────

    def _export_to_pagerduty(
        self,
        finding: dict[str, Any],
        report_meta: dict[str, Any],
        cluster_context: str = "",
    ) -> ExportResult:
        """Export a finding to PagerDuty."""
        if self._pagerduty is None:
            return ExportResult(
                target="pagerduty",
                success=False,
                error=(
                    "PagerDuty integration not configured. "
                    "Set VAIG_PAGERDUTY__ROUTING_KEY."
                ),
            )

        try:
            from vaig.skills.service_health.schema import Finding

            finding_obj = Finding.model_validate(finding)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            logger.warning("Could not parse finding into Finding model; using raw dict")
            finding_obj = None

        if finding_obj is not None:
            return self._pagerduty.create_incident_from_finding(
                finding=finding_obj,
                cluster_context=cluster_context,
            )

        # Fallback: use raw dict with trigger_event
        finding_id = finding.get("id", "")
        severity = str(finding.get("severity", "info")).lower()
        dedup_key = f"{cluster_context}:{finding_id}" if cluster_context else finding_id

        try:
            self._pagerduty.trigger_event(
                summary=finding.get("title", finding_id),
                severity=self._pagerduty.severity_mapping.get(severity, "warning"),
                source=finding.get("service", "vaig"),
                dedup_key=dedup_key,
            )
            return ExportResult(
                target="pagerduty",
                success=True,
                key=dedup_key,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            return ExportResult(
                target="pagerduty",
                success=False,
                error=str(exc),
            )

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _build_jira_description(finding: dict[str, Any]) -> str:
        """Build a plain-text description from finding fields."""
        parts: list[str] = []
        if finding.get("description"):
            parts.append(finding["description"])
        if finding.get("root_cause"):
            parts.append(f"\nRoot Cause: {finding['root_cause']}")
        if finding.get("impact"):
            parts.append(f"\nImpact: {finding['impact']}")
        if finding.get("remediation"):
            parts.append(f"\nRemediation: {finding['remediation']}")
        if finding.get("affected_resources"):
            resources = ", ".join(finding["affected_resources"])
            parts.append(f"\nAffected Resources: {resources}")
        return "\n".join(parts) if parts else "No description available."

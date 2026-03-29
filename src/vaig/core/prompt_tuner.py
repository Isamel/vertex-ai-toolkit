"""Quality analysis engine for HealthReport history.

Computes quality signals from serialised HealthReport dicts stored by
:class:`ReportStore` and generates actionable prompt-improvement
suggestions.  Works exclusively with raw ``dict`` payloads so the
module stays decoupled from the HealthReport Pydantic model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Threshold constants ────────────────────────────────────────
HALLUCINATION_THRESHOLD = 0.3
LOW_ACTIONABILITY_THRESHOLD = 0.5
OVER_ESCALATION_THRESHOLD = 0.4
LOW_CONFIDENCE_THRESHOLD = 0.5
INCOMPLETE_THRESHOLD = 0.5

__all__ = ["PromptTuner", "QualityInsights", "QualitySignal"]


# ── Data models ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class QualitySignal:
    """A single quality metric derived from report history."""

    name: str
    value: float  # 0.0–1.0 (or raw average for evidence_depth)
    threshold: float
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class QualityInsights:
    """Full quality analysis result across multiple reports."""

    total_reports: int
    signals: list[QualitySignal] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


# ── Analyzer ───────────────────────────────────────────────────


class PromptTuner:
    """Analyze report quality and suggest prompt improvements."""

    def analyze_quality(
        self,
        report_records: list[dict[str, Any]],
    ) -> QualityInsights:
        """Compute quality signals from a list of report store records.

        Each *record* is the full JSONL record (containing ``report``,
        ``run_id``, ``timestamp``).  The ``report`` value must be a
        serialised HealthReport dict.

        Args:
            report_records: Records from :meth:`ReportStore.read_reports`.

        Returns:
            A :class:`QualityInsights` with per-signal scores and
            prompt-improvement suggestions.
        """
        reports = [
            r["report"]
            for r in report_records
            if isinstance(r.get("report"), dict)
        ]

        if not reports:
            return QualityInsights(total_reports=0)

        signals = [
            self._hallucination_rate(reports),
            self._evidence_depth(reports),
            self._actionability(reports),
            self._over_escalation(reports),
            self._completeness(reports),
            self._low_confidence(reports),
        ]

        suggestions = self._generate_suggestions(signals)

        return QualityInsights(
            total_reports=len(reports),
            signals=signals,
            suggestions=suggestions,
        )

    # ── Signal implementations ─────────────────────────────────

    @staticmethod
    def _hallucination_rate(reports: list[dict[str, Any]]) -> QualitySignal:
        """% of findings with an empty ``evidence`` list."""
        total = 0
        empty_evidence = 0
        for rpt in reports:
            for finding in rpt.get("findings", []):
                total += 1
                evidence = finding.get("evidence", [])
                if not evidence:
                    empty_evidence += 1

        rate = empty_evidence / total if total else 0.0
        return QualitySignal(
            name="hallucination_rate",
            value=round(rate, 3),
            threshold=HALLUCINATION_THRESHOLD,
            passed=rate <= HALLUCINATION_THRESHOLD,
            detail=f"{empty_evidence}/{total} findings lack evidence",
        )

    @staticmethod
    def _evidence_depth(reports: list[dict[str, Any]]) -> QualitySignal:
        """Average number of evidence items per finding."""
        total_findings = 0
        total_evidence = 0
        for rpt in reports:
            for finding in rpt.get("findings", []):
                total_findings += 1
                total_evidence += len(finding.get("evidence", []))

        avg = total_evidence / total_findings if total_findings else 0.0
        # Depth is informational — no pass/fail threshold, always "passes"
        return QualitySignal(
            name="evidence_depth",
            value=round(avg, 2),
            threshold=0.0,
            passed=True,
            detail=f"{total_evidence} evidence items across {total_findings} findings",
        )

    @staticmethod
    def _actionability(reports: list[dict[str, Any]]) -> QualitySignal:
        """% of recommended actions with a non-empty ``command``.

        The HealthReport schema stores ``recommendations`` as a flat list
        of ``RecommendedAction`` dicts, each with a top-level ``command``
        field — there is no nested ``actions`` list.
        """
        total = 0
        actionable = 0
        for rpt in reports:
            for rec in rpt.get("recommendations", []):
                total += 1
                if rec.get("command", "").strip():
                    actionable += 1

        rate = actionable / total if total else 1.0
        return QualitySignal(
            name="actionability",
            value=round(rate, 3),
            threshold=LOW_ACTIONABILITY_THRESHOLD,
            passed=rate >= LOW_ACTIONABILITY_THRESHOLD,
            detail=f"{actionable}/{total} actions have commands",
        )

    @staticmethod
    def _over_escalation(reports: list[dict[str, Any]]) -> QualitySignal:
        """% of CRITICAL findings in reports with resource-level scope."""
        total_critical = 0
        resource_scope_critical = 0
        for rpt in reports:
            scope = (
                rpt.get("executive_summary", {}).get("scope", "") or ""
            )
            is_resource_scope = scope.lower().startswith("resource:")
            for finding in rpt.get("findings", []):
                severity = (finding.get("severity", "") or "").upper()
                if severity == "CRITICAL":
                    total_critical += 1
                    if is_resource_scope:
                        resource_scope_critical += 1

        rate = resource_scope_critical / total_critical if total_critical else 0.0
        return QualitySignal(
            name="over_escalation",
            value=round(rate, 3),
            threshold=OVER_ESCALATION_THRESHOLD,
            passed=rate <= OVER_ESCALATION_THRESHOLD,
            detail=(
                f"{resource_scope_critical}/{total_critical} "
                f"CRITICAL findings at resource scope"
            ),
        )

    @staticmethod
    def _completeness(reports: list[dict[str, Any]]) -> QualitySignal:
        """% of reports with a non-empty ``timeline``."""
        total = len(reports)
        complete = sum(
            1 for rpt in reports if rpt.get("timeline")
        )
        rate = complete / total if total else 0.0
        return QualitySignal(
            name="completeness",
            value=round(rate, 3),
            threshold=INCOMPLETE_THRESHOLD,
            passed=rate >= INCOMPLETE_THRESHOLD,
            detail=f"{complete}/{total} reports have timeline data",
        )

    @staticmethod
    def _low_confidence(reports: list[dict[str, Any]]) -> QualitySignal:
        """% of findings with LOW confidence."""
        total = 0
        low = 0
        for rpt in reports:
            for finding in rpt.get("findings", []):
                total += 1
                confidence = (finding.get("confidence", "") or "").upper()
                if confidence == "LOW":
                    low += 1

        rate = low / total if total else 0.0
        return QualitySignal(
            name="low_confidence",
            value=round(rate, 3),
            threshold=LOW_CONFIDENCE_THRESHOLD,
            passed=rate <= LOW_CONFIDENCE_THRESHOLD,
            detail=f"{low}/{total} findings have LOW confidence",
        )

    # ── Suggestions ────────────────────────────────────────────

    @staticmethod
    def _generate_suggestions(signals: list[QualitySignal]) -> list[str]:
        """Map failing signals to specific prompt improvement hints."""
        _SIGNAL_HINTS: dict[str, str] = {
            "hallucination_rate": (
                "Too many findings lack evidence — add instructions to "
                "always cite specific log lines, metrics, or resource states"
            ),
            "actionability": (
                "Many recommended actions lack concrete commands — instruct "
                "the agent to include kubectl/gcloud commands for every action"
            ),
            "over_escalation": (
                "CRITICAL findings at resource scope suggest over-escalation "
                "— instruct the agent to reserve CRITICAL for cluster-wide "
                "or namespace-wide issues only"
            ),
            "completeness": (
                "Reports are missing timeline data — instruct the agent to "
                "include timestamps for when issues were first observed"
            ),
            "low_confidence": (
                "Too many LOW-confidence findings — instruct the agent to "
                "gather more evidence before reporting, or raise the "
                "confidence threshold for inclusion"
            ),
        }

        suggestions: list[str] = []
        for signal in signals:
            if not signal.passed and signal.name in _SIGNAL_HINTS:
                suggestions.append(_SIGNAL_HINTS[signal.name])
        return suggestions

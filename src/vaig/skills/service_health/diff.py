"""Report diff computation for watch mode.

Compares consecutive ``HealthReport`` iterations and produces a compact
:class:`ReportDiff` showing new, resolved, unchanged findings and
severity changes.  Used by ``_run_watch_loop`` in the ``live`` command
to surface deltas between polling iterations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vaig.skills.service_health.schema import Finding, HealthReport


@dataclass
class SeverityChange:
    """A finding whose severity changed between iterations."""

    finding: Finding
    previous_severity: str
    current_severity: str


@dataclass
class ReportDiff:
    """Diff between two consecutive HealthReport iterations."""

    new_findings: list[Finding] = field(default_factory=list)
    resolved_findings: list[Finding] = field(default_factory=list)
    unchanged_findings: list[Finding] = field(default_factory=list)
    severity_changes: list[SeverityChange] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """Return ``True`` when any meaningful delta exists."""
        return bool(self.new_findings or self.resolved_findings or self.severity_changes)

    @property
    def summary_line(self) -> str:
        """One-line diff summary, e.g. ``+2 NEW | 1 RESOLVED | 3 UNCHANGED``."""
        parts: list[str] = []
        if self.new_findings:
            parts.append(f"+{len(self.new_findings)} NEW")
        if self.resolved_findings:
            parts.append(f"{len(self.resolved_findings)} RESOLVED")
        if self.severity_changes:
            changes = ", ".join(
                f"{sc.previous_severity}\u2192{sc.current_severity}"
                for sc in self.severity_changes
            )
            parts.append(f"{len(self.severity_changes)} SEVERITY CHANGED ({changes})")
        if self.unchanged_findings:
            parts.append(f"{len(self.unchanged_findings)} UNCHANGED")
        return " | ".join(parts) if parts else "No changes"


def compute_report_diff(current: HealthReport, previous: HealthReport) -> ReportDiff:
    """Compare two ``HealthReport`` instances and return the diff.

    Matching is based on ``Finding.id`` (a stable slug like
    ``'crashloop-payment-svc'``).  Severity changes are detected when
    two findings share the same *id* but differ in ``severity``.

    Args:
        current: The latest report from the current watch iteration.
        previous: The report from the preceding iteration.

    Returns:
        A :class:`ReportDiff` summarising deltas.
    """
    prev_by_id = {f.id: f for f in previous.findings}
    curr_by_id = {f.id: f for f in current.findings}

    prev_ids = set(prev_by_id.keys())
    curr_ids = set(curr_by_id.keys())

    new_findings = [curr_by_id[fid] for fid in sorted(curr_ids - prev_ids)]
    resolved_findings = [prev_by_id[fid] for fid in sorted(prev_ids - curr_ids)]

    severity_changes: list[SeverityChange] = []
    unchanged: list[Finding] = []
    for fid in sorted(curr_ids & prev_ids):
        curr_f = curr_by_id[fid]
        prev_f = prev_by_id[fid]
        if curr_f.severity != prev_f.severity:
            severity_changes.append(
                SeverityChange(
                    finding=curr_f,
                    previous_severity=prev_f.severity.value,
                    current_severity=curr_f.severity.value,
                )
            )
        else:
            unchanged.append(curr_f)

    return ReportDiff(
        new_findings=new_findings,
        resolved_findings=resolved_findings,
        unchanged_findings=unchanged,
        severity_changes=severity_changes,
    )

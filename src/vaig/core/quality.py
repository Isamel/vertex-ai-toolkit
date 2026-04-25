"""Run-quality tracking — lightweight issue collector for the agent pipeline.

SPEC-RATE-05: Surface model-degradation events and quality issues to downstream
consumers (reporter, CLI, API callers) without coupling the pipeline internals
to any specific skill schema.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class QualityIssueKind(StrEnum):
    """Taxonomy of quality issues that can arise during a pipeline run."""

    agent_failed = "agent_failed"
    """A gatherer or sequential agent returned ``success=False``."""

    model_degraded = "model_degraded"
    """One or more agents ran on the fallback model instead of the primary."""

    circuit_breaker_tripped = "circuit_breaker_tripped"
    """The circuit breaker opened after repeated failures."""

    incomplete_gather = "incomplete_gather"
    """One or more gather agents did not return results."""

    enrichment_timeout = "enrichment_timeout"
    """Recommendation enrichment was cancelled due to timeout."""

    attachment_truncated = "attachment_truncated"
    """Attachment or context content was truncated to fit the context window."""


class QualityIssue(BaseModel, frozen=True, extra="forbid"):
    """A single quality issue recorded during a pipeline run."""

    kind: QualityIssueKind
    where: str
    """Human-readable location — e.g. agent name, tool name, or phase label."""
    consequence: str = ""
    """Human-readable consequence or extra detail for this issue."""


class RunQualityCollector:
    """Accumulates :class:`QualityIssue` entries for one pipeline run.

    Deduplicates by ``(kind, where)`` so repeated identical events (e.g.
    multiple tool errors from the same agent) are collapsed into a single
    entry.  Thread-safety is intentionally **not** provided — callers in the
    parallel fan-out path should accumulate per-result and merge afterwards.
    """

    def __init__(self) -> None:
        self._seen: set[tuple[QualityIssueKind, str]] = set()
        self._issues: list[QualityIssue] = []

    def record(self, issue: QualityIssue) -> None:
        """Add *issue* if it has not already been recorded for the same location."""
        key = (issue.kind, issue.where)
        if key not in self._seen:
            self._seen.add(key)
            self._issues.append(issue)

    def record_kind(
        self,
        kind: QualityIssueKind,
        where: str,
        consequence: str = "",
    ) -> None:
        """Convenience wrapper — constructs a :class:`QualityIssue` and calls :meth:`record`."""
        self.record(QualityIssue(kind=kind, where=where, consequence=consequence))

    def merge(self, other: RunQualityCollector) -> None:
        """Merge issues from *other* into this collector (dedup preserved)."""
        for issue in other.issues:
            self.record(issue)

    @property
    def issues(self) -> list[QualityIssue]:
        """Ordered list of unique issues recorded so far (insertion order)."""
        return list(self._issues)

    def has_kind(self, kind: QualityIssueKind) -> bool:
        """Return ``True`` if any issue of *kind* was recorded."""
        return any(i.kind == kind for i in self._issues)

    def __len__(self) -> int:
        return len(self._issues)

    def __bool__(self) -> bool:
        return bool(self._issues)

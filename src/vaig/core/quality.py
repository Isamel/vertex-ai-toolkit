"""Run-quality tracking — lightweight issue collector for the agent pipeline.

SPEC-RATE-05: Surface model-degradation events and quality issues to downstream
consumers (reporter, CLI, API callers) without coupling the pipeline internals
to any specific skill schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class QualityIssueKind(StrEnum):
    """Taxonomy of quality issues that can arise during a pipeline run."""

    MODEL_DEGRADED = "model_degraded"
    """One or more agents ran on the fallback model instead of the primary."""

    BUDGET_EXCEEDED = "budget_exceeded"
    """The pipeline was halted because the cost budget was exceeded."""

    AGENT_FAILED = "agent_failed"
    """A gatherer or sequential agent returned ``success=False``."""

    TOOL_ERROR = "tool_error"
    """A tool call returned an error result."""

    CONTEXT_TRUNCATED = "context_truncated"
    """Attachment or context content was truncated to fit the context window."""


@dataclass(frozen=True)
class QualityIssue:
    """A single quality issue recorded during a pipeline run.

    Designed to be lightweight and hashable so it can be stored in a set
    for deduplication.
    """

    kind: QualityIssueKind
    where: str
    """Human-readable location — e.g. agent name, tool name, or phase label."""
    detail: str = ""
    """Optional extra information such as an error message or metric value."""


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
        detail: str = "",
    ) -> None:
        """Convenience wrapper — constructs a :class:`QualityIssue` and calls :meth:`record`."""
        self.record(QualityIssue(kind=kind, where=where, detail=detail))

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

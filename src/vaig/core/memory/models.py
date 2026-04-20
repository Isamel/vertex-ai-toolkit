"""Data models for the pattern memory subsystem.

These are plain Pydantic models — no heavy dependencies — so they can
be safely imported from any layer of the codebase.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "FixOutcome",
    "PatternEntry",
    "RecurrenceSignal",
]


class PatternEntry(BaseModel):
    """A single persisted observation of a finding pattern.

    Each entry captures one occurrence of a fingerprint so that the
    recurrence analyzer can count how many times the same logical
    issue has appeared across different runs.
    """

    fingerprint: str = Field(description="16-hex-char stable fingerprint")
    first_seen: datetime = Field(description="UTC timestamp of the first observation")
    last_seen: datetime = Field(description="UTC timestamp of the most recent observation")
    occurrences: int = Field(default=1, ge=1, description="Total observation count")
    severity: str = Field(default="", description="Severity of the most recent observation")
    title: str = Field(default="", description="Human-readable title of the finding")
    service: str = Field(default="", description="Service / resource the finding affects")
    category: str = Field(default="", description="Finding category (e.g. 'pod-health')")

    def merge(self, seen_at: datetime, severity: str = "", title: str = "") -> PatternEntry:
        """Return a new entry that records one more occurrence.

        The *first_seen* date is preserved; *last_seen*, *occurrences*,
        and (optionally) *severity* / *title* are updated.
        """
        return self.model_copy(
            update={
                "last_seen": seen_at,
                "occurrences": self.occurrences + 1,
                "severity": severity or self.severity,
                "title": title or self.title,
            }
        )


class RecurrenceSignal(BaseModel):
    """Recurrence annotation attached to a ``Finding`` post-analysis.

    This is the read-only signal that consumers (reporter prompt,
    ``query_pattern_history`` tool) use to surface historical context.
    """

    fingerprint: str = Field(description="16-hex-char fingerprint linking to PatternEntry")
    occurrences: int = Field(default=1, ge=1, description="Total historical occurrences")
    first_seen: datetime = Field(description="UTC datetime of the first observation")
    last_seen: datetime = Field(description="UTC datetime of the most recent observation")
    is_recurring: bool = Field(
        default=False,
        description="True when occurrences >= recurrence threshold (default 2)",
    )
    badge: Literal["NEW", "RECURRING", "CHRONIC"] = Field(
        default="NEW",
        description=(
            "Human-readable badge: NEW (1 occurrence), "
            "RECURRING (2–4), CHRONIC (5+)"
        ),
    )

    @classmethod
    def from_entry(cls, entry: PatternEntry) -> RecurrenceSignal:
        """Build a ``RecurrenceSignal`` from a ``PatternEntry``."""
        badge: Literal["NEW", "RECURRING", "CHRONIC"]
        if entry.occurrences >= 5:
            badge = "CHRONIC"
        elif entry.occurrences >= 2:
            badge = "RECURRING"
        else:
            badge = "NEW"

        return cls(
            fingerprint=entry.fingerprint,
            occurrences=entry.occurrences,
            first_seen=entry.first_seen,
            last_seen=entry.last_seen,
            is_recurring=entry.occurrences >= 2,
            badge=badge,
        )


class FixOutcome(BaseModel):
    """Correlates an applied code fix with a subsequent health report.

    Records the outcome of a fix strategy so the agent can learn which
    strategies are effective over time.
    """

    fix_id: str = Field(description="Unique fix identifier (matches FixAppliedEvent.fix_id)")
    fingerprint: str = Field(description="16-hex-char fingerprint of the finding that was fixed")
    strategy: str = Field(default="", description="Fix strategy label (e.g. 'restart-pod')")
    applied_at: datetime = Field(description="UTC timestamp of fix application")
    outcome: Literal["resolved", "persisted", "worsened", "unknown"] = Field(
        default="unknown",
        description="Correlated result: resolved/persisted/worsened/unknown",
    )
    correlated_run_id: str = Field(
        default="",
        description="Run ID of the health report used for correlation (empty if uncorrelated)",
    )
    correlated_at: datetime | None = Field(
        default=None,
        description="UTC timestamp of correlation (None if uncorrelated)",
    )


class RecalledPattern(BaseModel):
    """A prior-run memory recall formatted for injection into agent prompts.

    Constructed from a :class:`PatternEntry` plus optional fix outcome
    by :class:`~vaig.agents.mixins.MemoryRecallMixin`.
    """

    timestamp: datetime = Field(description="UTC timestamp of the most recent observation")
    title: str = Field(description="Human-readable title of the finding")
    cluster: str = Field(default="", description="Cluster or service this pattern was observed on")
    resolution: str = Field(default="", description="Known fix strategy or resolution, if any")
    fix_outcome: str = Field(default="", description="Fix outcome label (CONFIRMED, unknown, etc.)")
    occurrences: int = Field(default=1, ge=1, description="Total historical occurrences")

    @classmethod
    def from_entry(cls, entry: PatternEntry, resolution: str = "", fix_outcome: str = "") -> RecalledPattern:
        """Build a :class:`RecalledPattern` from a :class:`PatternEntry`."""
        return cls(
            timestamp=entry.last_seen,
            title=entry.title or entry.fingerprint,
            cluster=entry.service,
            resolution=resolution,
            fix_outcome=fix_outcome,
            occurrences=entry.occurrences,
        )

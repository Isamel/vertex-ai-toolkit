"""Recurrence analyzer — annotates findings with RecurrenceSignal.

The analyzer computes fingerprints for each finding, looks them up in
the :class:`~vaig.core.memory.pattern_store.PatternMemoryStore`, and
records the observation.  It returns a mapping of fingerprint →
:class:`~vaig.core.memory.models.RecurrenceSignal` ready to be attached
to ``Finding`` objects.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from vaig.core.memory.fingerprint import ObservationFingerprint
from vaig.core.memory.models import PatternEntry, RecurrenceSignal
from vaig.core.memory.pattern_store import PatternMemoryStore

logger = logging.getLogger(__name__)


class RecurrenceAnalyzer:
    """Annotates a batch of findings with historical recurrence data.

    Args:
        store: The :class:`PatternMemoryStore` to read/write from.
        recurrence_threshold: Minimum occurrences to mark RECURRING (default 2).
        chronic_threshold: Minimum occurrences to mark CHRONIC (default 5).
        max_age_days: Entries older than this are treated as new (default 90).
    """

    def __init__(
        self,
        store: PatternMemoryStore,
        recurrence_threshold: int = 2,
        chronic_threshold: int = 5,
        max_age_days: int = 90,
    ) -> None:
        self._store = store
        self._recurrence_threshold = recurrence_threshold
        self._chronic_threshold = chronic_threshold
        self._max_age_days = max_age_days

    def _is_stale(self, entry: PatternEntry) -> bool:
        """Return True if the entry is older than max_age_days."""
        cutoff = datetime.now(UTC) - timedelta(days=self._max_age_days)
        return entry.last_seen < cutoff

    def analyze(
        self,
        run_id: str,
        findings: list[dict[str, str]],
    ) -> dict[str, RecurrenceSignal]:
        """Record findings and return recurrence signals keyed by fingerprint.

        Args:
            run_id: Unique identifier for the current run.
            findings: List of dicts with keys ``category``, ``service``,
                ``title``, ``description``, ``severity``.

        Returns:
            A dict mapping fingerprint → :class:`RecurrenceSignal`.
        """
        signals: dict[str, RecurrenceSignal] = {}

        for finding in findings:
            try:
                fp = ObservationFingerprint.from_finding(
                    category=finding.get("category", ""),
                    service=finding.get("service", ""),
                    title=finding.get("title", ""),
                    description=finding.get("description", ""),
                )

                entry = self._store.record(
                    run_id=run_id,
                    fingerprint=fp,
                    severity=finding.get("severity", ""),
                    title=finding.get("title", ""),
                    service=finding.get("service", ""),
                    category=finding.get("category", ""),
                )

                if self._is_stale(entry):
                    # Treat stale entries as new — reset to single occurrence
                    entry = PatternEntry(
                        fingerprint=fp,
                        first_seen=entry.last_seen,
                        last_seen=entry.last_seen,
                        occurrences=1,
                        severity=entry.severity,
                        title=entry.title,
                        service=entry.service,
                        category=entry.category,
                    )

                signal = self._build_signal(entry)
                signals[fp] = signal
            except Exception:  # noqa: BLE001
                logger.debug("Recurrence analysis failed for finding: %s", finding.get("title"))

        return signals

    def _build_signal(self, entry: PatternEntry) -> RecurrenceSignal:
        """Build a RecurrenceSignal respecting configured thresholds."""
        if entry.occurrences >= self._chronic_threshold:
            badge = "CHRONIC"
        elif entry.occurrences >= self._recurrence_threshold:
            badge = "RECURRING"
        else:
            badge = "NEW"

        return RecurrenceSignal(
            fingerprint=entry.fingerprint,
            occurrences=entry.occurrences,
            first_seen=entry.first_seen,
            last_seen=entry.last_seen,
            is_recurring=entry.occurrences >= self._recurrence_threshold,
            badge=badge,  # type: ignore[arg-type]
        )

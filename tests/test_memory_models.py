"""Tests for PatternEntry and RecurrenceSignal models."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from vaig.core.memory.models import PatternEntry, RecurrenceSignal


def _entry(occurrences: int = 1) -> PatternEntry:
    now = datetime.now(UTC)
    return PatternEntry(
        fingerprint="abc123def456abcd",
        first_seen=now - timedelta(days=occurrences),
        last_seen=now,
        occurrences=occurrences,
        severity="HIGH",
        title="OOMKilled",
        service="payment",
        category="pod-health",
    )


class TestPatternEntry:
    def test_merge_increments_occurrences(self) -> None:
        entry = _entry(occurrences=2)
        merged = entry.merge(seen_at=datetime.now(UTC))
        assert merged.occurrences == 3

    def test_merge_preserves_first_seen(self) -> None:
        entry = _entry(occurrences=1)
        original_first = entry.first_seen
        merged = entry.merge(seen_at=datetime.now(UTC))
        assert merged.first_seen == original_first

    def test_merge_updates_last_seen(self) -> None:
        entry = _entry(occurrences=1)
        new_time = datetime.now(UTC) + timedelta(hours=1)
        merged = entry.merge(seen_at=new_time)
        assert merged.last_seen == new_time

    def test_merge_updates_severity_when_provided(self) -> None:
        entry = _entry()
        merged = entry.merge(seen_at=datetime.now(UTC), severity="CRITICAL")
        assert merged.severity == "CRITICAL"

    def test_merge_keeps_original_severity_when_not_provided(self) -> None:
        entry = _entry()
        merged = entry.merge(seen_at=datetime.now(UTC))
        assert merged.severity == entry.severity


class TestRecurrenceSignal:
    def test_from_entry_new_badge(self) -> None:
        signal = RecurrenceSignal.from_entry(_entry(occurrences=1))
        assert signal.badge == "NEW"
        assert signal.is_recurring is False

    def test_from_entry_recurring_badge(self) -> None:
        signal = RecurrenceSignal.from_entry(_entry(occurrences=3))
        assert signal.badge == "RECURRING"
        assert signal.is_recurring is True

    def test_from_entry_chronic_badge(self) -> None:
        signal = RecurrenceSignal.from_entry(_entry(occurrences=5))
        assert signal.badge == "CHRONIC"
        assert signal.is_recurring is True

    def test_from_entry_boundary_at_2(self) -> None:
        signal = RecurrenceSignal.from_entry(_entry(occurrences=2))
        assert signal.badge == "RECURRING"

    def test_from_entry_boundary_at_4(self) -> None:
        signal = RecurrenceSignal.from_entry(_entry(occurrences=4))
        assert signal.badge == "RECURRING"

    def test_occurrences_are_copied(self) -> None:
        entry = _entry(occurrences=7)
        signal = RecurrenceSignal.from_entry(entry)
        assert signal.occurrences == 7
        assert signal.fingerprint == entry.fingerprint

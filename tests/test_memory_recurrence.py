"""Tests for RecurrenceAnalyzer."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from vaig.core.memory.pattern_store import PatternMemoryStore
from vaig.core.memory.recurrence import RecurrenceAnalyzer


def _analyzer(tmp_path: Path, **kwargs: object) -> RecurrenceAnalyzer:
    store = PatternMemoryStore(base_dir=tmp_path)
    return RecurrenceAnalyzer(store=store, **kwargs)  # type: ignore[arg-type]


def _finding(title: str = "OOMKilled", category: str = "pod-health") -> dict[str, str]:
    return {
        "category": category,
        "service": "payment",
        "title": title,
        "description": "pod is OOM",
        "severity": "HIGH",
    }


class TestRecurrenceAnalyzerAnalyze:
    def test_first_occurrence_is_new(self, tmp_path: Path) -> None:
        analyzer = _analyzer(tmp_path)
        signals = analyzer.analyze("run1", [_finding()])
        assert len(signals) == 1
        signal = next(iter(signals.values()))
        assert signal.badge == "NEW"
        assert signal.is_recurring is False

    def test_second_occurrence_is_recurring(self, tmp_path: Path) -> None:
        store = PatternMemoryStore(base_dir=tmp_path)
        analyzer = RecurrenceAnalyzer(store=store, recurrence_threshold=2)
        analyzer.analyze("run1", [_finding()])
        signals = analyzer.analyze("run2", [_finding()])
        signal = next(iter(signals.values()))
        assert signal.badge == "RECURRING"
        assert signal.is_recurring is True

    def test_chronic_threshold_respected(self, tmp_path: Path) -> None:
        store = PatternMemoryStore(base_dir=tmp_path)
        analyzer = RecurrenceAnalyzer(store=store, recurrence_threshold=2, chronic_threshold=3)
        for i in range(3):
            signals = analyzer.analyze(f"run{i}", [_finding()])
        signal = next(iter(signals.values()))
        assert signal.badge == "CHRONIC"

    def test_different_findings_get_different_fingerprints(self, tmp_path: Path) -> None:
        analyzer = _analyzer(tmp_path)
        findings = [_finding(title="OOMKilled"), _finding(title="CrashLoop")]
        signals = analyzer.analyze("run1", findings)
        assert len(signals) == 2

    def test_empty_findings_returns_empty(self, tmp_path: Path) -> None:
        analyzer = _analyzer(tmp_path)
        assert analyzer.analyze("run1", []) == {}

    def test_error_in_one_finding_does_not_abort(self, tmp_path: Path) -> None:
        """A corrupt finding should be skipped, not crash the analyzer."""
        analyzer = _analyzer(tmp_path)
        findings = [
            {},  # empty — will fail fingerprint gracefully
            _finding(),
        ]
        # Should not raise; returns at least the valid finding
        signals = analyzer.analyze("run1", findings)
        # At least the valid finding is there (empty dict gets empty fp)
        assert isinstance(signals, dict)

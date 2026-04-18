"""Integration test — pattern memory pipeline end-to-end.

Tests the full flow:
  HealthReportCompleted event
  → MemorySubscriber._process_report
  → PatternMemoryStore.record
  → RecurrenceAnalyzer / RecurrenceSignal
  → query_pattern_history tool returns RECURRING badge on second run
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vaig.core.event_bus import EventBus
from vaig.core.events import HealthReportCompleted
from vaig.core.memory.pattern_store import PatternMemoryStore
from vaig.core.subscribers.memory_subscriber import MemorySubscriber
from vaig.tools.knowledge.query_pattern_history import query_pattern_history


@pytest.fixture(autouse=True)
def reset_event_bus() -> None:
    EventBus._instance = None  # noqa: SLF001
    yield
    EventBus._instance = None  # noqa: SLF001


def _settings(store_path: Path) -> MagicMock:
    s = MagicMock()
    s.memory.enabled = True
    s.memory.store_path = str(store_path)
    return s


def _write_report(directory: Path, run_id: str, findings: list[dict]) -> Path:
    path = directory / f"{run_id}.jsonl"
    path.write_text(json.dumps({"findings": findings}) + "\n")
    return path


FINDING = {
    "category": "pod-health",
    "service": "payment-svc",
    "title": "CrashLoopBackOff detected",
    "description": "Pod payment-svc-59967f9ccc-4zdx6 is crash-looping (3rd time)",
    "severity": "CRITICAL",
}


class TestMemoryPipeline:
    def test_first_run_stores_entry(self, tmp_path: Path) -> None:
        store = PatternMemoryStore(base_dir=tmp_path)
        settings = _settings(tmp_path)
        MemorySubscriber(settings, store=store)

        report_path = _write_report(tmp_path, "run1", [FINDING])
        EventBus.get().emit(
            HealthReportCompleted(run_id="run1", report_path=str(report_path))
        )

        entries = store.all_entries()
        assert len(entries) == 1
        assert entries[0].occurrences == 1
        assert entries[0].title == "CrashLoopBackOff detected"

    def test_second_run_increments_occurrences(self, tmp_path: Path) -> None:
        store = PatternMemoryStore(base_dir=tmp_path)
        settings = _settings(tmp_path)
        MemorySubscriber(settings, store=store)

        path1 = _write_report(tmp_path, "run1", [FINDING])
        EventBus.get().emit(
            HealthReportCompleted(run_id="run1", report_path=str(path1))
        )
        path2 = _write_report(tmp_path, "run2", [FINDING])
        EventBus.get().emit(
            HealthReportCompleted(run_id="run2", report_path=str(path2))
        )

        entries = store.all_entries()
        assert len(entries) == 1
        assert entries[0].occurrences == 2

    def test_query_returns_recurring_badge_after_two_runs(self, tmp_path: Path) -> None:
        store = PatternMemoryStore(base_dir=tmp_path)
        settings = _settings(tmp_path)
        MemorySubscriber(settings, store=store)

        path1 = _write_report(tmp_path, "run1", [FINDING])
        EventBus.get().emit(
            HealthReportCompleted(run_id="run1", report_path=str(path1))
        )
        path2 = _write_report(tmp_path, "run2", [FINDING])
        EventBus.get().emit(
            HealthReportCompleted(run_id="run2", report_path=str(path2))
        )

        cfg = MagicMock()
        cfg.store_path = str(tmp_path)
        result = query_pattern_history(
            category=FINDING["category"],
            service=FINDING["service"],
            title=FINDING["title"],
            description=FINDING["description"],
            config=cfg,
        )

        assert not result.error
        assert "RECURRING" in result.output
        assert "2" in result.output  # occurrences

    def test_fingerprint_normalises_ephemeral_tokens(self, tmp_path: Path) -> None:
        """Two findings with different pod hashes should share the same fingerprint."""
        store = PatternMemoryStore(base_dir=tmp_path)
        settings = _settings(tmp_path)
        MemorySubscriber(settings, store=store)

        finding_v1 = {**FINDING, "description": "Pod payment-svc-59967f9ccc-4zdx6 is crash-looping"}
        finding_v2 = {**FINDING, "description": "Pod payment-svc-aabbccdd11-xyz99 is crash-looping"}

        path1 = _write_report(tmp_path, "run1", [finding_v1])
        EventBus.get().emit(
            HealthReportCompleted(run_id="run1", report_path=str(path1))
        )
        path2 = _write_report(tmp_path, "run2", [finding_v2])
        EventBus.get().emit(
            HealthReportCompleted(run_id="run2", report_path=str(path2))
        )

        # Same fingerprint → merged into one entry
        entries = store.all_entries()
        assert len(entries) == 1
        assert entries[0].occurrences == 2

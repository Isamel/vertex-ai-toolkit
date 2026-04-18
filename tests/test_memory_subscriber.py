"""Tests for MemorySubscriber."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vaig.core.event_bus import EventBus
from vaig.core.events import HealthReportCompleted
from vaig.core.memory.pattern_store import PatternMemoryStore
from vaig.core.subscribers.memory_subscriber import MemorySubscriber


@pytest.fixture(autouse=True)
def reset_event_bus() -> None:
    """Ensure a fresh EventBus for each test."""
    EventBus._instance = None  # noqa: SLF001
    yield
    EventBus._instance = None  # noqa: SLF001


def _make_settings(enabled: bool = True) -> MagicMock:
    settings = MagicMock()
    settings.memory.enabled = enabled
    settings.memory.store_path = "/tmp/test-memory"
    return settings


class TestMemorySubscriberInit:
    def test_noop_when_disabled(self) -> None:
        settings = _make_settings(enabled=False)
        sub = MemorySubscriber(settings)
        # No subscription wired — bus has no handlers
        bus = EventBus.get()
        assert not bus._handlers  # noqa: SLF001

    def test_subscribes_when_enabled(self, tmp_path: Path) -> None:
        settings = _make_settings(enabled=True)
        store = PatternMemoryStore(base_dir=tmp_path)
        sub = MemorySubscriber(settings, store=store)
        bus = EventBus.get()
        assert HealthReportCompleted in bus._handlers  # noqa: SLF001

    def test_unsubscribe_all_clears_handlers(self, tmp_path: Path) -> None:
        settings = _make_settings(enabled=True)
        store = PatternMemoryStore(base_dir=tmp_path)
        sub = MemorySubscriber(settings, store=store)
        sub.unsubscribe_all()
        # After unsubscribing, handlers list may exist but be empty
        bus = EventBus.get()
        handlers = bus._handlers.get(HealthReportCompleted, [])  # noqa: SLF001
        assert handlers == []


class TestMemorySubscriberProcessing:
    def test_records_findings_from_report(self, tmp_path: Path) -> None:
        settings = _make_settings(enabled=True)
        store = PatternMemoryStore(base_dir=tmp_path)
        MemorySubscriber(settings, store=store)

        run_id = "20240101T120000Z"
        report_path = tmp_path / f"{run_id}.jsonl"
        report_path.write_text(
            json.dumps({
                "findings": [
                    {
                        "category": "pod-health",
                        "service": "payment-svc",
                        "title": "CrashLoopBackOff",
                        "description": "Pod is crash-looping",
                        "severity": "CRITICAL",
                    }
                ]
            })
            + "\n"
        )

        EventBus.get().emit(
            HealthReportCompleted(run_id=run_id, report_path=str(report_path))
        )

        entries = store.all_entries()
        assert len(entries) == 1
        assert entries[0].title == "CrashLoopBackOff"
        assert entries[0].occurrences == 1

    def test_missing_report_path_is_silent(self, tmp_path: Path) -> None:
        settings = _make_settings(enabled=True)
        store = PatternMemoryStore(base_dir=tmp_path)
        MemorySubscriber(settings, store=store)

        # Emit with a path that doesn't exist — should not raise
        EventBus.get().emit(
            HealthReportCompleted(
                run_id="missing-run",
                report_path=str(tmp_path / "nonexistent.jsonl"),
            )
        )
        assert store.all_entries() == []

    def test_malformed_json_line_is_skipped(self, tmp_path: Path) -> None:
        settings = _make_settings(enabled=True)
        store = PatternMemoryStore(base_dir=tmp_path)
        MemorySubscriber(settings, store=store)

        run_id = "bad-run"
        report_path = tmp_path / f"{run_id}.jsonl"
        report_path.write_text("not-valid-json\n")

        # Should not raise
        EventBus.get().emit(
            HealthReportCompleted(run_id=run_id, report_path=str(report_path))
        )
        assert store.all_entries() == []

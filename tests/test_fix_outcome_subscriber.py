"""Tests for FixOutcomeSubscriber."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vaig.core.event_bus import EventBus
from vaig.core.events import FixAppliedEvent, HealthReportCompleted
from vaig.core.memory.outcome_store import FixOutcomeStore
from vaig.core.subscribers.fix_outcome_subscriber import FixOutcomeSubscriber


@pytest.fixture(autouse=True)
def reset_event_bus() -> None:
    """Ensure a fresh EventBus for each test."""
    EventBus._instance = None  # noqa: SLF001
    yield
    EventBus._instance = None  # noqa: SLF001


def _make_settings(enabled: bool = True) -> MagicMock:
    settings = MagicMock()
    settings.memory.outcome_tracking_enabled = enabled
    settings.memory.outcome_store_path = "/tmp/test-outcomes"
    return settings


def _write_report(path: Path, fingerprints: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    findings = [{"fingerprint": fp, "category": "pod-health"} for fp in fingerprints]
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"findings": findings}) + "\n")


class TestFixOutcomeSubscriberInit:
    def test_noop_when_disabled(self) -> None:
        settings = _make_settings(enabled=False)
        sub = FixOutcomeSubscriber(settings)
        bus = EventBus.get()
        assert not bus._handlers  # noqa: SLF001

    def test_subscribes_when_enabled(self, tmp_path: Path) -> None:
        settings = _make_settings(enabled=True)
        store = FixOutcomeStore(base_dir=tmp_path / "outcomes")
        sub = FixOutcomeSubscriber(settings, store=store)
        bus = EventBus.get()
        assert bus._handlers  # noqa: SLF001


class TestFixOutcomeSubscriberOnFixApplied:
    def test_records_fix_on_event(self, tmp_path: Path) -> None:
        settings = _make_settings()
        store = FixOutcomeStore(base_dir=tmp_path / "outcomes")
        sub = FixOutcomeSubscriber(settings, store=store)

        EventBus.get().emit(
            FixAppliedEvent(
                run_id="run-1",
                fix_id="fix-abc",
                fingerprint="deadbeef01234567",
                strategy="restart-pod",
                cluster="my-cluster",
                namespace="default",
            )
        )

        entry = store.lookup("fix-abc")
        assert entry is not None
        assert entry.outcome == "unknown"
        assert entry.strategy == "restart-pod"

    def test_silently_handles_store_error(self, tmp_path: Path) -> None:
        settings = _make_settings()
        store = MagicMock()
        store.record_fix.side_effect = RuntimeError("store error")
        # Should not raise
        sub = FixOutcomeSubscriber(settings, store=store)
        EventBus.get().emit(
            FixAppliedEvent(run_id="r", fix_id="f", fingerprint="fp", strategy="s")
        )


class TestFixOutcomeSubscriberOnHealthReportCompleted:
    def test_correlates_resolved_when_fingerprint_absent(self, tmp_path: Path) -> None:
        settings = _make_settings()
        store = FixOutcomeStore(base_dir=tmp_path / "outcomes")
        store.record_fix(run_id="r1", fix_id="fix-1", fingerprint="aabbccdd11223344", strategy="s")
        sub = FixOutcomeSubscriber(settings, store=store)

        report = tmp_path / "report.jsonl"
        _write_report(report, fingerprints=[])  # fingerprint absent

        EventBus.get().emit(
            HealthReportCompleted(run_id="r2", report_path=str(report))
        )

        entry = store.lookup("fix-1")
        assert entry is not None
        assert entry.outcome == "resolved"

    def test_correlates_persisted_when_fingerprint_present(self, tmp_path: Path) -> None:
        settings = _make_settings()
        store = FixOutcomeStore(base_dir=tmp_path / "outcomes")
        fp = "aabbccdd11223344"
        store.record_fix(run_id="r1", fix_id="fix-2", fingerprint=fp, strategy="s")
        sub = FixOutcomeSubscriber(settings, store=store)

        report = tmp_path / "report2.jsonl"
        _write_report(report, fingerprints=[fp])  # fingerprint still present

        EventBus.get().emit(
            HealthReportCompleted(run_id="r2", report_path=str(report))
        )

        entry = store.lookup("fix-2")
        assert entry is not None
        assert entry.outcome == "persisted"

    def test_silently_handles_missing_report(self, tmp_path: Path) -> None:
        settings = _make_settings()
        store = FixOutcomeStore(base_dir=tmp_path / "outcomes")
        sub = FixOutcomeSubscriber(settings, store=store)

        EventBus.get().emit(
            HealthReportCompleted(run_id="r2", report_path="/nonexistent/report.jsonl")
        )
        # No exception raised — store remains unmodified

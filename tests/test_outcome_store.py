"""Tests for FixOutcomeStore."""
from __future__ import annotations

from pathlib import Path

from vaig.core.memory.outcome_store import FixOutcomeStore


def _make_store(tmp_path: Path) -> FixOutcomeStore:
    return FixOutcomeStore(base_dir=tmp_path / "outcomes")


class TestFixOutcomeStoreRecordFix:
    def test_record_fix_creates_entry(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        entry = store.record_fix(
            run_id="run-1",
            fix_id="fix-abc",
            fingerprint="deadbeef01234567",
            strategy="restart-pod",
        )
        assert entry.fix_id == "fix-abc"
        assert entry.outcome == "unknown"
        assert entry.fingerprint == "deadbeef01234567"
        assert entry.strategy == "restart-pod"

    def test_record_fix_persists_to_jsonl(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.record_fix(run_id="run-1", fix_id="fix-abc", fingerprint="fp", strategy="s")
        jsonl = tmp_path / "outcomes" / "run-1.jsonl"
        assert jsonl.exists()
        lines = [l for l in jsonl.read_text().splitlines() if l.strip()]
        assert len(lines) == 1

    def test_record_fix_indexable_by_fix_id(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.record_fix(run_id="run-1", fix_id="fix-xyz", fingerprint="fp", strategy="s")
        found = store.lookup("fix-xyz")
        assert found is not None
        assert found.fix_id == "fix-xyz"


class TestFixOutcomeStoreCorrelate:
    def test_correlate_resolved(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.record_fix(run_id="run-1", fix_id="fix-1", fingerprint="fp", strategy="s")
        updated = store.correlate("fix-1", "resolved", "run-2")
        assert updated is not None
        assert updated.outcome == "resolved"
        assert updated.correlated_run_id == "run-2"
        assert updated.correlated_at is not None

    def test_correlate_persisted(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.record_fix(run_id="run-1", fix_id="fix-2", fingerprint="fp", strategy="s")
        updated = store.correlate("fix-2", "persisted", "run-2")
        assert updated is not None
        assert updated.outcome == "persisted"

    def test_correlate_unknown_fix_id_returns_none(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        result = store.correlate("no-such-id", "resolved", "run-99")
        assert result is None


class TestFixOutcomeStorePending:
    def test_pending_returns_unknown_only(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.record_fix(run_id="r1", fix_id="f1", fingerprint="fp1", strategy="s")
        store.record_fix(run_id="r1", fix_id="f2", fingerprint="fp2", strategy="s")
        store.correlate("f1", "resolved", "r2")

        pending = store.pending()
        ids = {e.fix_id for e in pending}
        assert "f2" in ids
        assert "f1" not in ids


class TestFixOutcomeStorePersistence:
    def test_index_rebuilt_from_disk(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.record_fix(run_id="r1", fix_id="fix-disk", fingerprint="fp", strategy="s")

        # New store instance pointing to same dir.
        store2 = _make_store(tmp_path)
        found = store2.lookup("fix-disk")
        assert found is not None
        assert found.fix_id == "fix-disk"

"""Tests for PatternMemoryStore."""
from __future__ import annotations

from pathlib import Path

import pytest

from vaig.core.memory.pattern_store import PatternMemoryStore


@pytest.fixture()
def store(tmp_path: Path) -> PatternMemoryStore:
    return PatternMemoryStore(base_dir=tmp_path)


class TestPatternMemoryStoreRecord:
    def test_record_creates_entry(self, store: PatternMemoryStore) -> None:
        entry = store.record("run1", "abc123def456abcd", title="OOMKilled", severity="HIGH")
        assert entry.fingerprint == "abc123def456abcd"
        assert entry.occurrences == 1

    def test_record_twice_increments_occurrences(self, store: PatternMemoryStore) -> None:
        store.record("run1", "fp1", title="OOMKilled")
        entry = store.record("run2", "fp1", title="OOMKilled")
        assert entry.occurrences == 2

    def test_record_persists_to_file(self, store: PatternMemoryStore, tmp_path: Path) -> None:
        store.record("myrun", "fp1", title="CrashLoop")
        jsonl = tmp_path / "myrun.jsonl"
        assert jsonl.exists()
        assert jsonl.stat().st_size > 0

    def test_lookup_returns_none_for_unknown(self, store: PatternMemoryStore) -> None:
        assert store.lookup("doesnotexist") is None

    def test_lookup_returns_entry_after_record(self, store: PatternMemoryStore) -> None:
        store.record("run1", "fp1", title="Test")
        entry = store.lookup("fp1")
        assert entry is not None
        assert entry.title == "Test"


class TestPatternMemoryStorePersistence:
    def test_new_store_reads_existing_files(self, tmp_path: Path) -> None:
        """A freshly created store should pick up entries from a previous run."""
        store1 = PatternMemoryStore(base_dir=tmp_path)
        store1.record("run1", "fp1", title="OOMKilled", severity="HIGH")

        # New store instance — should reload from disk
        store2 = PatternMemoryStore(base_dir=tmp_path)
        entry = store2.lookup("fp1")
        assert entry is not None
        assert entry.occurrences == 1

    def test_missing_dir_does_not_raise(self, tmp_path: Path) -> None:
        store = PatternMemoryStore(base_dir=tmp_path / "nonexistent")
        # Should silently return empty
        assert store.all_entries() == []

    def test_all_entries_returns_one_per_fingerprint(self, store: PatternMemoryStore) -> None:
        store.record("run1", "fp1", title="A")
        store.record("run1", "fp2", title="B")
        store.record("run2", "fp1", title="A")  # duplicate fingerprint
        entries = store.all_entries()
        fps = {e.fingerprint for e in entries}
        assert fps == {"fp1", "fp2"}

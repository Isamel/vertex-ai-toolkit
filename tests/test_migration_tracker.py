"""Tests for _MigrationTracker persistence (save_state / load_state)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vaig.skills.code_migration.skill import MigrationPhase, _MigrationTracker


class TestMigrationTrackerPersistence:
    def _make_tracker_with_record(self) -> _MigrationTracker:
        t = _MigrationTracker()
        t.record("main.py", MigrationPhase.IMPLEMENT, "complete", "done")
        return t

    def test_save_creates_file(self, tmp_path: Path) -> None:
        tracker = self._make_tracker_with_record()
        state_file = tmp_path / "state.json"
        tracker.save_state(state_file)
        assert state_file.exists()

    def test_save_produces_valid_json(self, tmp_path: Path) -> None:
        tracker = self._make_tracker_with_record()
        state_file = tmp_path / "state.json"
        tracker.save_state(state_file)
        data = json.loads(state_file.read_text())
        assert "records" in data
        assert "main.py" in data["records"]

    def test_load_restores_records(self, tmp_path: Path) -> None:
        original = self._make_tracker_with_record()
        state_file = tmp_path / "state.json"
        original.save_state(state_file)

        restored = _MigrationTracker()
        restored.load_state(state_file)
        log = restored.get_all()
        assert len(log) == 1
        assert log[0]["filename"] == "main.py"

    def test_load_missing_file_is_noop(self, tmp_path: Path) -> None:
        tracker = _MigrationTracker()
        # Should not raise
        tracker.load_state(tmp_path / "nonexistent.json")
        assert tracker.get_all() == []

    def test_load_invalid_json_is_silent_noop(self, tmp_path: Path) -> None:
        """Loading a corrupt JSON file should not raise and should leave tracker empty."""
        state_file = tmp_path / "bad.json"
        state_file.write_text("not-json", encoding="utf-8")
        tracker = _MigrationTracker()
        # Should not raise
        tracker.load_state(state_file)
        assert tracker.get_all() == []

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        tracker = self._make_tracker_with_record()
        state_file = tmp_path / "nested" / "deep" / "state.json"
        tracker.save_state(state_file)
        assert state_file.exists()

    def test_roundtrip_multiple_records(self, tmp_path: Path) -> None:
        tracker = _MigrationTracker()
        tracker.record("a.py", MigrationPhase.INVENTORY, "complete")
        tracker.record("b.py", MigrationPhase.IMPLEMENT, "skipped", "no tests")
        state_file = tmp_path / "state.json"
        tracker.save_state(state_file)

        restored = _MigrationTracker()
        restored.load_state(state_file)
        log = {r["filename"]: r for r in restored.get_all()}
        assert set(log.keys()) == {"a.py", "b.py"}
        assert log["b.py"]["notes"] == "no tests"


class TestCodeMigrationSkillResume:
    def test_init_with_resume_loads_state(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)

        # Pre-populate a state file
        from vaig.core.project import ensure_project_dir

        state_path = ensure_project_dir() / "migration-state.json"
        state_path.write_text(
            json.dumps({"records": {"old.py": {"filename": "old.py", "phase": "implement", "status": "complete", "notes": ""}}}),
            encoding="utf-8",
        )

        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill(resume=True)
        log = skill.get_migration_log()
        assert len(log) == 1
        assert log[0]["filename"] == "old.py"

    def test_init_without_resume_starts_fresh(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)

        # Pre-populate a state file
        from vaig.core.project import ensure_project_dir

        state_path = ensure_project_dir() / "migration-state.json"
        state_path.write_text(
            json.dumps({"records": {"old.py": {"filename": "old.py", "phase": "implement", "status": "complete", "notes": ""}}}),
            encoding="utf-8",
        )

        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill(resume=False)
        assert skill.get_migration_log() == []

    def test_record_auto_saves(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)

        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill()
        skill.record_file_migration("new.py", MigrationPhase.SPEC, "complete")

        from vaig.core.project import ensure_project_dir

        state_path = ensure_project_dir() / "migration-state.json"
        assert state_path.exists()
        data = json.loads(state_path.read_text())
        assert "new.py" in data["records"]

"""Tests for ToolCallStore and ToolCallRecord.

Covers the JSONL storage backend, thread safety, path validation security,
and the ToolCallRecord dataclass serialization.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from vaig.core.tool_call_store import ToolCallStore
from vaig.tools.base import ToolCallRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(**overrides: object) -> ToolCallRecord:
    """Create a ToolCallRecord with sensible defaults, overridable."""
    defaults: dict[str, object] = {
        "tool_name": "kubectl_get_pods",
        "tool_args": {"namespace": "default"},
        "output": "pod-1 Running\npod-2 Running",
        "output_size_bytes": 30,
        "error": False,
        "error_type": "",
        "error_message": "",
        "duration_s": 0.42,
        "timestamp": "2026-03-16T00:00:00Z",
        "agent_name": "gatherer",
        "run_id": "abc123",
        "iteration": 1,
    }
    defaults.update(overrides)
    return ToolCallRecord(**defaults)  # type: ignore[arg-type]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write a list of dicts to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# ToolCallRecord tests
# ---------------------------------------------------------------------------

class TestToolCallRecord:
    """Tests for the ToolCallRecord dataclass."""

    def test_to_dict_contains_all_fields(self) -> None:
        record = _make_record()
        d = record.to_dict()
        assert d["tool_name"] == "kubectl_get_pods"
        assert d["tool_args"] == {"namespace": "default"}
        assert d["output"] == "pod-1 Running\npod-2 Running"
        assert d["output_size_bytes"] == 30
        assert d["error"] is False
        assert d["error_type"] == ""
        assert d["error_message"] == ""
        assert d["duration_s"] == 0.42
        assert d["timestamp"] == "2026-03-16T00:00:00Z"
        assert d["agent_name"] == "gatherer"
        assert d["run_id"] == "abc123"
        assert d["iteration"] == 1

    def test_to_dict_with_error(self) -> None:
        record = _make_record(
            error=True,
            error_type="TimeoutError",
            error_message="Command timed out after 30s",
            output="",
            output_size_bytes=0,
        )
        d = record.to_dict()
        assert d["error"] is True
        assert d["error_type"] == "TimeoutError"
        assert d["error_message"] == "Command timed out after 30s"

    def test_to_dict_is_json_serializable(self) -> None:
        record = _make_record()
        serialized = json.dumps(record.to_dict())
        deserialized = json.loads(serialized)
        assert deserialized["tool_name"] == "kubectl_get_pods"


# ---------------------------------------------------------------------------
# ToolCallStore tests
# ---------------------------------------------------------------------------

class TestToolCallStore:
    """Tests for the ToolCallStore JSONL backend."""

    def test_start_run_creates_directory(self, tmp_path: Path) -> None:
        store = ToolCallStore(base_dir=tmp_path)
        run_id = store.start_run()
        assert run_id  # Non-empty
        assert store.run_id == run_id
        run_file = store.get_run_file()
        assert run_file is not None
        assert run_file.parent.exists()
        assert "tool_results" in str(run_file)

    def test_start_run_with_custom_id(self, tmp_path: Path) -> None:
        store = ToolCallStore(base_dir=tmp_path)
        run_id = store.start_run(run_id="custom-run-42")
        assert run_id == "custom-run-42"
        assert store.run_id == "custom-run-42"
        run_file = store.get_run_file()
        assert run_file is not None
        assert run_file.name == "custom-run-42.jsonl"

    def test_record_writes_jsonl(self, tmp_path: Path) -> None:
        store = ToolCallStore(base_dir=tmp_path)
        store.start_run(run_id="test-run")
        record = _make_record(run_id="test-run")
        store.record(record)

        run_file = store.get_run_file()
        assert run_file is not None
        assert run_file.exists()

        lines = run_file.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["tool_name"] == "kubectl_get_pods"
        assert parsed["run_id"] == "test-run"

    def test_record_multiple_appends(self, tmp_path: Path) -> None:
        store = ToolCallStore(base_dir=tmp_path)
        store.start_run(run_id="multi")

        for i in range(5):
            record = _make_record(
                tool_name=f"tool_{i}",
                iteration=i,
                run_id="multi",
            )
            store.record(record)

        run_file = store.get_run_file()
        assert run_file is not None
        lines = run_file.read_text().strip().split("\n")
        assert len(lines) == 5

        for i, line in enumerate(lines):
            parsed = json.loads(line)
            assert parsed["tool_name"] == f"tool_{i}"
            assert parsed["iteration"] == i

    def test_record_auto_starts_run_if_needed(self, tmp_path: Path) -> None:
        store = ToolCallStore(base_dir=tmp_path)
        # Don't call start_run — record() should auto-start
        record = _make_record()
        store.record(record)

        assert store.run_id  # Auto-generated
        run_file = store.get_run_file()
        assert run_file is not None
        assert run_file.exists()

    def test_thread_safety(self, tmp_path: Path) -> None:
        """Multiple threads writing concurrently should not lose records."""
        store = ToolCallStore(base_dir=tmp_path)
        store.start_run(run_id="threaded")
        num_threads = 10
        records_per_thread = 20

        def _writer(thread_id: int) -> None:
            for i in range(records_per_thread):
                record = _make_record(
                    tool_name=f"thread_{thread_id}_tool_{i}",
                    iteration=i,
                    run_id="threaded",
                )
                store.record(record)

        threads = [
            threading.Thread(target=_writer, args=(t,))
            for t in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        run_file = store.get_run_file()
        assert run_file is not None
        lines = run_file.read_text().strip().split("\n")
        assert len(lines) == num_threads * records_per_thread

        # Every line should be valid JSON
        for line in lines:
            parsed = json.loads(line)
            assert "tool_name" in parsed

    def test_get_run_file_none_before_start(self) -> None:
        store = ToolCallStore()
        assert store.get_run_file() is None

    def test_run_file_path_format(self, tmp_path: Path) -> None:
        """Verify the {base_dir}/tool_results/{YYYY-MM-DD}/{run_id}.jsonl layout."""
        store = ToolCallStore(base_dir=tmp_path)
        store.start_run(run_id="layout-test")
        run_file = store.get_run_file()
        assert run_file is not None
        # Should be: tmp_path / tool_results / YYYY-MM-DD / layout-test.jsonl
        parts = run_file.relative_to(tmp_path).parts
        assert parts[0] == "tool_results"
        # parts[1] should be a date like "2026-03-16"
        assert len(parts[1]) == 10  # YYYY-MM-DD
        assert parts[1].count("-") == 2
        assert parts[2] == "layout-test.jsonl"


# ---------------------------------------------------------------------------
# Path validation security tests
# ---------------------------------------------------------------------------


class TestToolCallStorePathSecurity:
    """Tests for the path safety check in ToolCallStore.__init__."""

    def test_tmp_path_is_accepted(self, tmp_path: Path) -> None:
        """A pytest tmp_path (under /tmp) must be accepted."""
        # tmp_path is under tempfile.gettempdir() — passes the tmp safe check.
        store = ToolCallStore(base_dir=tmp_path)
        assert store is not None

    def test_home_relative_dir_is_accepted(self) -> None:
        """A directory under the user's home dir must be accepted."""
        home = Path.home()
        home_subdir = home / ".vaig" / "runs"
        # No mocking needed — home is always a valid allowed root.
        store = ToolCallStore(base_dir=home_subdir)
        assert store is not None

    def test_system_path_raises_value_error(self) -> None:
        """/etc/evil is not under home, cwd, or tmp — must raise ValueError."""
        fake_home = Path("/home/security-test-user-xyzzy")
        fake_cwd = Path("/workspaces/security-test-project-xyzzy")
        # Both fake_home and fake_cwd are paths that /etc/evil is NOT under.
        with (
            patch("pathlib.Path.home", return_value=fake_home),
            patch("pathlib.Path.cwd", return_value=fake_cwd),
        ):
            with pytest.raises(ValueError, match="tool_results_dir must be under"):
                ToolCallStore(base_dir="/etc/evil")

    def test_warning_logged_for_path_outside_vaig(self) -> None:
        """A path outside ~/.vaig/ and outside cwd should emit a warning."""
        home = Path.home()
        # Use a subdir of home that is NOT under ~/.vaig/
        outside_vaig = home / ".some_other_tool" / "runs"
        # Force cwd to something that doesn't cover this path
        unrelated_cwd = Path("/workspaces/unrelated-xyzzy")

        with (
            patch("pathlib.Path.cwd", return_value=unrelated_cwd),
            patch("vaig.core.tool_call_store.logger") as mock_logger,
        ):
            ToolCallStore(base_dir=outside_vaig)

        # The warning should have been issued for the out-of-~/.vaig/ path
        assert mock_logger.warning.called, "Expected logger.warning to be called"
        call_args = mock_logger.warning.call_args
        assert "outside ~/.vaig/" in call_args[0][0]

    def test_cwd_root_does_not_allow_arbitrary_path(self) -> None:
        """When cwd is /, the cwd branch must NOT allow arbitrary system paths."""
        fake_home = Path("/home/security-test-user-xyzzy")
        with (
            patch("pathlib.Path.home", return_value=fake_home),
            patch("pathlib.Path.cwd", return_value=Path("/")),
        ):
            with pytest.raises(ValueError, match="tool_results_dir must be under"):
                ToolCallStore(base_dir="/etc/passwd_dir")


# ---------------------------------------------------------------------------
# list_runs() tests
# ---------------------------------------------------------------------------


class TestListRuns:
    """Tests for ToolCallStore.list_runs()."""

    def test_list_runs_returns_correct_format(self, tmp_path: Path) -> None:
        """list_runs() returns (run_id, date) tuples with UTC-aware datetimes."""
        store = ToolCallStore(base_dir=tmp_path)
        date_dir = tmp_path / "tool_results" / "2026-03-16"
        _write_jsonl(date_dir / "run-abc.jsonl", [{"tool_name": "t1"}])

        runs = store.list_runs()

        assert len(runs) == 1
        run_id, date = runs[0]
        assert run_id == "run-abc"
        assert date == datetime(2026, 3, 16, tzinfo=UTC)
        assert date.tzinfo is not None  # must be timezone-aware

    def test_list_runs_returns_multiple_sorted_ascending(self, tmp_path: Path) -> None:
        """list_runs() returns all runs sorted by date ascending."""
        store = ToolCallStore(base_dir=tmp_path)
        for date_str, run_name in [("2026-03-14", "run-old"), ("2026-03-16", "run-new")]:
            _write_jsonl(
                tmp_path / "tool_results" / date_str / f"{run_name}.jsonl",
                [{"tool_name": "t"}],
            )

        runs = store.list_runs()

        assert len(runs) == 2
        assert runs[0][0] == "run-old"
        assert runs[1][0] == "run-new"

    def test_list_runs_empty_when_no_results_dir(self, tmp_path: Path) -> None:
        store = ToolCallStore(base_dir=tmp_path)
        assert store.list_runs() == []

    def test_list_runs_since_filters_old_dates(self, tmp_path: Path) -> None:
        """list_runs(since=...) excludes directories before the cutoff date."""
        store = ToolCallStore(base_dir=tmp_path)
        for date_str, run_name in [
            ("2026-03-14", "run-old"),
            ("2026-03-16", "run-new"),
        ]:
            _write_jsonl(
                tmp_path / "tool_results" / date_str / f"{run_name}.jsonl",
                [{"tool_name": "t"}],
            )

        since = datetime(2026, 3, 15, tzinfo=UTC)
        runs = store.list_runs(since=since)

        run_ids = [r[0] for r in runs]
        assert "run-old" not in run_ids
        assert "run-new" in run_ids

    def test_list_runs_since_with_naive_datetime(self, tmp_path: Path) -> None:
        """list_runs(since=...) handles naive datetimes by treating them as UTC."""
        store = ToolCallStore(base_dir=tmp_path)
        _write_jsonl(
            tmp_path / "tool_results" / "2026-03-16" / "run-x.jsonl",
            [{"tool_name": "t"}],
        )

        # Naive datetime — should be treated as UTC (no exception)
        since_naive = datetime(2026, 3, 15)  # no tzinfo
        runs = store.list_runs(since=since_naive)

        assert len(runs) == 1

    def test_list_runs_since_with_aware_datetime(self, tmp_path: Path) -> None:
        """list_runs(since=...) properly converts timezone-aware datetimes."""
        store = ToolCallStore(base_dir=tmp_path)
        _write_jsonl(
            tmp_path / "tool_results" / "2026-03-16" / "run-y.jsonl",
            [{"tool_name": "t"}],
        )

        # Timezone-aware datetime — should be converted, not overwritten
        since_aware = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)
        runs = store.list_runs(since=since_aware)

        assert len(runs) == 1


# ---------------------------------------------------------------------------
# read_records() tests
# ---------------------------------------------------------------------------


class TestReadRecords:
    """Tests for ToolCallStore.read_records()."""

    def test_read_records_returns_correct_records(self, tmp_path: Path) -> None:
        """read_records(run_id=...) returns the records from the matching file."""
        store = ToolCallStore(base_dir=tmp_path)
        records = [
            {"tool_name": "kubectl_get_pods", "run_id": "run-1"},
            {"tool_name": "kubectl_get_nodes", "run_id": "run-1"},
        ]
        _write_jsonl(tmp_path / "tool_results" / "2026-03-16" / "run-1.jsonl", records)

        result = store.read_records(run_id="run-1")

        assert len(result) == 2
        assert result[0]["tool_name"] == "kubectl_get_pods"
        assert result[1]["tool_name"] == "kubectl_get_nodes"

    def test_read_records_skips_malformed_json_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """read_records() skips malformed JSON lines and logs a warning."""
        store = ToolCallStore(base_dir=tmp_path)
        jsonl_path = tmp_path / "tool_results" / "2026-03-16" / "run-bad.jsonl"
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path.write_text(
            '{"tool_name": "good_tool"}\n'
            "NOT JSON AT ALL\n"
            '{"tool_name": "another_good_tool"}\n',
            encoding="utf-8",
        )

        # The "vaig" parent logger has propagate=False (set by setup_logging()),
        # so we must patch the parent's propagate to let records reach caplog.
        vaig_logger = logging.getLogger("vaig")
        original_propagate = vaig_logger.propagate
        vaig_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger="vaig.core.tool_call_store"):
                result = store.read_records(run_id="run-bad")
        finally:
            vaig_logger.propagate = original_propagate

        assert len(result) == 2
        assert result[0]["tool_name"] == "good_tool"
        assert result[1]["tool_name"] == "another_good_tool"
        assert any("malformed" in msg.lower() for msg in caplog.messages)

    def test_read_records_empty_when_run_id_not_found(self, tmp_path: Path) -> None:
        """read_records() returns [] and logs warning when run_id doesn't exist."""
        store = ToolCallStore(base_dir=tmp_path)
        (tmp_path / "tool_results" / "2026-03-16").mkdir(parents=True)

        result = store.read_records(run_id="nonexistent-run")

        assert result == []

    def test_read_records_selects_newest_on_duplicate_run_id(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """read_records() picks the newest date dir when run_id exists in multiple."""
        store = ToolCallStore(base_dir=tmp_path)
        run_id = "dup-run"

        # Older date: has record A
        _write_jsonl(
            tmp_path / "tool_results" / "2026-03-14" / f"{run_id}.jsonl",
            [{"tool_name": "old_tool", "date": "2026-03-14"}],
        )
        # Newer date: has record B
        _write_jsonl(
            tmp_path / "tool_results" / "2026-03-16" / f"{run_id}.jsonl",
            [{"tool_name": "new_tool", "date": "2026-03-16"}],
        )

        # The "vaig" parent logger has propagate=False (set by setup_logging()),
        # so we must patch the parent's propagate to let records reach caplog.
        vaig_logger = logging.getLogger("vaig")
        original_propagate = vaig_logger.propagate
        vaig_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger="vaig.core.tool_call_store"):
                result = store.read_records(run_id=run_id)
        finally:
            vaig_logger.propagate = original_propagate

        # Should use the newest (2026-03-16)
        assert len(result) == 1
        assert result[0]["tool_name"] == "new_tool"
        # Should warn about duplicates
        assert any("multiple" in msg.lower() for msg in caplog.messages)

    def test_read_records_all_without_run_id(self, tmp_path: Path) -> None:
        """read_records() without run_id returns all records across all dates."""
        store = ToolCallStore(base_dir=tmp_path)
        _write_jsonl(
            tmp_path / "tool_results" / "2026-03-14" / "run-a.jsonl",
            [{"tool_name": "tool_a"}],
        )
        _write_jsonl(
            tmp_path / "tool_results" / "2026-03-16" / "run-b.jsonl",
            [{"tool_name": "tool_b"}],
        )

        result = store.read_records()

        tool_names = [r["tool_name"] for r in result]
        assert "tool_a" in tool_names
        assert "tool_b" in tool_names

    def test_read_records_since_filters_old_dates(self, tmp_path: Path) -> None:
        """read_records(since=...) excludes records before the cutoff date."""
        store = ToolCallStore(base_dir=tmp_path)
        _write_jsonl(
            tmp_path / "tool_results" / "2026-03-14" / "run-old.jsonl",
            [{"tool_name": "old_tool"}],
        )
        _write_jsonl(
            tmp_path / "tool_results" / "2026-03-16" / "run-new.jsonl",
            [{"tool_name": "new_tool"}],
        )

        since = datetime(2026, 3, 15, tzinfo=UTC)
        result = store.read_records(since=since)

        tool_names = [r["tool_name"] for r in result]
        assert "old_tool" not in tool_names
        assert "new_tool" in tool_names

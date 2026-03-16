"""Tests for ToolCallStore and ToolCallRecord.

Covers the JSONL storage backend, thread safety, and the
ToolCallRecord dataclass serialization.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

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

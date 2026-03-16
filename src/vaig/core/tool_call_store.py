"""Tool call result storage for metrics and feedback.

Stores each tool call as a JSON Lines (.jsonl) record.
Current backend: local files in the project directory.
Future: GCS, BigQuery, or other cloud storage.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from vaig.tools.base import ToolCallRecord

logger = logging.getLogger(__name__)


class ToolCallStore:
    """Stores tool call results as JSONL files.

    Storage layout:
        {base_dir}/tool_results/{YYYY-MM-DD}/{run_id}.jsonl

    Each line is a JSON object representing one ToolCallRecord.
    Thread-safe via a lock on writes.
    """

    def __init__(self, base_dir: str | Path = ".") -> None:
        self._base_dir = Path(base_dir)
        self._lock = threading.Lock()
        self._run_id = ""
        self._current_file: Path | None = None

    def start_run(self, run_id: str = "") -> str:
        """Start a new execution run. Returns the run_id."""
        self._run_id = run_id or uuid4().hex[:12]
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        run_dir = self._base_dir / "tool_results" / today
        run_dir.mkdir(parents=True, exist_ok=True)
        self._current_file = run_dir / f"{self._run_id}.jsonl"
        return self._run_id

    @property
    def run_id(self) -> str:
        return self._run_id

    def record(self, tool_record: ToolCallRecord) -> None:
        """Append a tool call record to the current run file."""
        if self._current_file is None:
            self.start_run()

        line = json.dumps(tool_record.to_dict(), ensure_ascii=False, default=str)
        with self._lock:
            with open(self._current_file, "a", encoding="utf-8") as f:  # noqa: PTH123
                f.write(line + "\n")

    def get_run_file(self) -> Path | None:
        """Get the path to the current run's JSONL file."""
        return self._current_file

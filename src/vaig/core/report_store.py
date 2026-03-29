"""Local persistence for HealthReport JSON — one JSONL file per run_id.

Stores each serialized HealthReport as a JSON Lines (.jsonl) record
in ``~/.vaig/reports/``.  Mirror of the :class:`ToolCallStore` pattern
for tool call records.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path.home() / ".vaig" / "reports"

__all__ = ["ReportStore"]


class ReportStore:
    """Append-only JSONL store for serialized HealthReport objects.

    Storage layout:
        {base_dir}/{run_id}.jsonl

    Each line is a JSON object containing a timestamp, run_id, and the
    full report dict.  Thread-safe for single-line appends on POSIX.
    """

    def __init__(self, base_dir: Path = _DEFAULT_DIR) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, run_id: str, report_dict: dict[str, Any]) -> Path:
        """Persist a single report as a JSONL line.

        Args:
            run_id: Unique identifier for this pipeline run.
            report_dict: Serialized HealthReport (from ``report.to_dict()``).

        Returns:
            Path to the JSONL file that was written.
        """
        path = self._base_dir / f"{run_id}.jsonl"
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "report": report_dict,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return path

    def list_runs(self) -> list[str]:
        """Return run IDs sorted by modification time (newest first)."""
        files = sorted(
            self._base_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return [f.stem for f in files]

    def read_reports(self, *, last: int = 20) -> list[dict[str, Any]]:
        """Read the latest *N* reports across all runs.

        Args:
            last: Maximum number of report records to return.

        Returns:
            List of record dicts (each containing ``timestamp``, ``run_id``,
            and ``report``).  Malformed JSONL lines are skipped with a
            warning.
        """
        reports: list[dict[str, Any]] = []
        for run_id in self.list_runs():
            path = self._base_dir / f"{run_id}.jsonl"
            try:
                for lineno, line in enumerate(
                    path.read_text(encoding="utf-8").strip().splitlines(),
                    start=1,
                ):
                    try:
                        record = json.loads(line)
                        reports.append(record)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "Skipping malformed JSON line %d in %s: %s",
                            lineno,
                            path,
                            exc,
                        )
            except OSError as exc:
                logger.warning("Could not read report file %s: %s", path, exc)
            if len(reports) >= last:
                break
        return reports[:last]

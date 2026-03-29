"""Local persistence for HealthReport JSON — one JSONL file per run_id.

Stores each serialized HealthReport as a JSON Lines (.jsonl) record
in ``~/.vaig/reports/``.  Mirror of the :class:`ToolCallStore` pattern
for tool call records.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path.home() / ".vaig" / "reports"

# Only allow safe characters in run_id to prevent path traversal.
_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

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

        # SECURITY: Validate that the base directory is within a safe location
        # to prevent path traversal or accidental writes to system directories.
        resolved = self._base_dir.resolve()
        home = Path.home()
        cwd = Path.cwd()
        tmp = Path(tempfile.gettempdir()).resolve()

        if cwd == Path("/"):
            safe = (
                (resolved == home or home in resolved.parents)
                or (resolved == tmp or tmp in resolved.parents)
            )
        else:
            safe = (
                (resolved == home or home in resolved.parents)
                or (resolved == cwd or cwd in resolved.parents)
                or (resolved == tmp or tmp in resolved.parents)
            )
        if not safe:
            raise ValueError(
                f"base_dir must be under home ({home}), cwd ({cwd}), "
                f"or temp ({tmp}), got: {resolved}"
            )

        self._base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, run_id: str, report_dict: dict[str, Any]) -> Path:
        """Persist a single report as a JSONL line.

        Args:
            run_id: Unique identifier for this pipeline run.  Must match
                ``[A-Za-z0-9_-]+`` to prevent path traversal.
            report_dict: Serialized HealthReport (from ``report.to_dict()``).

        Returns:
            Path to the JSONL file that was written.

        Raises:
            ValueError: If *run_id* contains unsafe characters.
        """
        if not _SAFE_RUN_ID_RE.match(run_id):
            raise ValueError(
                f"run_id contains unsafe characters (must match [A-Za-z0-9_-]+): {run_id!r}"
            )
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
            and ``report``), with the **latest** records last.  Malformed
            JSONL lines are skipped with a warning.
        """
        reports: list[dict[str, Any]] = []
        for run_id in self.list_runs():
            path = self._base_dir / f"{run_id}.jsonl"
            try:
                with path.open(encoding="utf-8") as f:
                    for lineno, raw_line in enumerate(f, start=1):
                        line = raw_line.strip()
                        if not line:
                            continue
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
        # Files are sorted newest-first, but lines within each file are
        # appended chronologically (oldest first).  Slice the last N
        # entries to return the most recent records.
        return reports[-last:]

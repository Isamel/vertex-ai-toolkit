"""Tool call result storage for metrics and feedback.

Stores each tool call as a JSON Lines (.jsonl) record.
Current backend: local files in the project directory.
Future: GCS, BigQuery, or other cloud storage.
"""

from __future__ import annotations

import json
import logging
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
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
        self._base_dir = Path(base_dir).expanduser()
        self._lock = threading.Lock()
        self._run_id = ""
        self._current_file: Path | None = None

        # SECURITY: Validate that the base directory is within a safe location
        # to prevent path traversal or accidental writes to system directories.
        resolved = self._base_dir.resolve()
        home = Path.home()
        cwd = Path.cwd()
        tmp = Path(tempfile.gettempdir()).resolve()

        if cwd == Path("/"):
            # Don't use cwd=/ as an allowed root — it would permit everything
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
                f"tool_results_dir must be under home ({home}), cwd ({cwd}), "
                f"or temp ({tmp}), got: {resolved}"
            )
        # Warn if the path is outside the default ~/.vaig/ location
        default_vaig_dir = home / ".vaig"
        in_default_vaig = resolved == default_vaig_dir or default_vaig_dir in resolved.parents
        in_cwd = resolved == cwd or cwd in resolved.parents
        if not in_default_vaig and not in_cwd:
            logger.warning(
                "ToolCallStore base_dir is outside ~/.vaig/: %s — "
                "verify this is intentional.",
                resolved,
            )

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
            with open(self._current_file, "a", encoding="utf-8") as f:  # type: ignore[arg-type]  # noqa: PTH123
                f.write(line + "\n")

    def get_run_file(self) -> Path | None:
        """Get the path to the current run's JSONL file."""
        return self._current_file

    def list_runs(self, since: datetime | None = None) -> list[tuple[str, datetime]]:
        """List available run IDs with their dates.

        Args:
            since: If provided, only return runs from date directories on or after
                this datetime (compared at day granularity in UTC).

        Returns:
            List of ``(run_id, date)`` tuples sorted by date ascending.
            ``date`` is a timezone-aware UTC datetime at midnight for the
            directory's date.
        """
        results_dir = self._base_dir / "tool_results"
        if not results_dir.is_dir():
            return []

        runs: list[tuple[str, datetime]] = []
        for date_dir in sorted(results_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            try:
                dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d").replace(tzinfo=UTC)
            except ValueError:
                logger.debug("Skipping non-date directory in tool_results: %s", date_dir.name)
                continue

            if since is not None:
                since_utc = since.astimezone(UTC) if since.tzinfo else since.replace(tzinfo=UTC)
                since_start_of_day = since_utc.replace(hour=0, minute=0, second=0, microsecond=0)
                if dir_date < since_start_of_day:
                    continue

            for jsonl_file in sorted(date_dir.glob("*.jsonl")):
                run_id = jsonl_file.stem
                runs.append((run_id, dir_date))

        return runs

    def read_records(
        self,
        run_id: str | None = None,
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Read tool call records from JSONL files.

        Args:
            run_id: If provided, reads only the specific run file.  The date
                directory is located by scanning all date dirs for a matching
                ``{run_id}.jsonl`` file.
            since: If provided, only reads files from date directories on or
                after this datetime.  Ignored when ``run_id`` is given.

        Returns:
            List of record dicts (one per tool call).  Returns an empty list
            if no matching files exist.  Malformed JSON lines are skipped with
            a warning.
        """
        results_dir = self._base_dir / "tool_results"
        if not results_dir.is_dir():
            return []

        files_to_read: list[Path] = []

        if run_id is not None:
            # Collect ALL matching candidates with their dates, then pick newest.
            matching_candidates: list[tuple[datetime, Path]] = []
            for date_dir in results_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                try:
                    dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d").replace(tzinfo=UTC)
                except ValueError:
                    continue
                candidate = date_dir / f"{run_id}.jsonl"
                if candidate.is_file():
                    matching_candidates.append((dir_date, candidate))

            if not matching_candidates:
                logger.warning("No JSONL file found for run_id=%r in %s", run_id, results_dir)
                return []

            matching_candidates.sort(key=lambda item: item[0], reverse=True)
            chosen_date, chosen_path = matching_candidates[0]
            files_to_read.append(chosen_path)

            if len(matching_candidates) > 1:
                logger.warning(
                    "Multiple JSONL files for run_id=%r; using newest at %s",
                    run_id,
                    chosen_path,
                )
        else:
            # Collect all date directories, optionally filtered by `since`
            for date_dir in sorted(results_dir.iterdir()):
                if not date_dir.is_dir():
                    continue
                try:
                    dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d").replace(tzinfo=UTC)
                except ValueError:
                    logger.debug("Skipping non-date directory in tool_results: %s", date_dir.name)
                    continue

                if since is not None:
                    since_utc = since.astimezone(UTC) if since.tzinfo else since.replace(tzinfo=UTC)
                    since_start_of_day = since_utc.replace(hour=0, minute=0, second=0, microsecond=0)
                    if dir_date < since_start_of_day:
                        continue

                files_to_read.extend(sorted(date_dir.glob("*.jsonl")))

        records: list[dict[str, Any]] = []
        for path in files_to_read:
            try:
                with path.open("r", encoding="utf-8") as f:
                    for lineno, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError as exc:
                            logger.warning(
                                "Skipping malformed JSON line %d in %s: %s",
                                lineno,
                                path,
                                exc,
                            )
            except OSError as exc:
                logger.warning("Could not read tool call records from %s: %s", path, exc)

        return records

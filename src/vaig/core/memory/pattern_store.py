"""JSONL-based persistent store for pattern memory entries.

The store appends new entries to per-run JSONL files and builds an
in-memory index (fingerprint → PatternEntry) lazily on first access.
All I/O errors are swallowed silently so that a broken store never
interrupts the live pipeline.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from vaig.core.memory.models import PatternEntry

logger = logging.getLogger(__name__)

_SENTINEL_DATE = datetime(1970, 1, 1, tzinfo=UTC)


class PatternMemoryStore:
    """Append-only JSONL store for :class:`PatternEntry` objects.

    Each *run_id* gets its own ``{run_id}.jsonl`` file inside
    ``base_dir``.  On first use the store scans all existing files to
    build an in-memory index keyed by fingerprint.

    All public methods are error-silent — they catch and log exceptions
    so that a corrupt or missing store never disrupts the live pipeline.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._base_dir = Path(base_dir).expanduser()
        self._index: dict[str, PatternEntry] | None = None

    # ── Index ─────────────────────────────────────────────────

    def _ensure_index(self) -> dict[str, PatternEntry]:
        """Build the in-memory index from all JSONL files (lazy, once)."""
        if self._index is not None:
            return self._index

        self._index = {}
        try:
            if not self._base_dir.exists():
                return self._index

            for path in sorted(self._base_dir.glob("*.jsonl")):
                try:
                    for line in path.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            entry = PatternEntry.model_validate(data)
                            existing = self._index.get(entry.fingerprint)
                            if existing is None or entry.last_seen > existing.last_seen:
                                # Keep the entry with the most occurrences
                                if existing is None or entry.occurrences >= existing.occurrences:
                                    self._index[entry.fingerprint] = entry
                        except Exception:  # noqa: BLE001
                            logger.debug("Skipping malformed memory entry in %s", path)
                except Exception:  # noqa: BLE001
                    logger.debug("Could not read memory file %s", path)
        except Exception:  # noqa: BLE001
            logger.debug("Error scanning memory store at %s", self._base_dir)

        return self._index

    # ── Read ──────────────────────────────────────────────────

    def lookup(self, fingerprint: str) -> PatternEntry | None:
        """Return the stored entry for *fingerprint*, or ``None``."""
        return self._ensure_index().get(fingerprint)

    def all_entries(self) -> list[PatternEntry]:
        """Return all unique entries (one per fingerprint)."""
        return list(self._ensure_index().values())

    # ── Write ─────────────────────────────────────────────────

    def record(
        self,
        run_id: str,
        fingerprint: str,
        severity: str = "",
        title: str = "",
        service: str = "",
        category: str = "",
    ) -> PatternEntry:
        """Record a finding occurrence and persist it.

        Returns the (possibly updated) ``PatternEntry`` after merging
        the new observation.  Never raises.
        """
        index = self._ensure_index()
        now = datetime.now(UTC)

        existing = index.get(fingerprint)
        if existing is None:
            entry = PatternEntry(
                fingerprint=fingerprint,
                first_seen=now,
                last_seen=now,
                occurrences=1,
                severity=severity,
                title=title,
                service=service,
                category=category,
            )
        else:
            entry = existing.merge(seen_at=now, severity=severity, title=title)

        index[fingerprint] = entry
        self._append(run_id, entry)
        return entry

    def _append(self, run_id: str, entry: PatternEntry) -> None:
        """Append *entry* as a JSON line to ``{run_id}.jsonl``.

        Creates parent directories if needed.  Silently ignores I/O errors.
        """
        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            path = self._base_dir / f"{run_id}.jsonl"
            line = entry.model_dump_json() + "\n"
            path.open("a", encoding="utf-8").write(line)
        except Exception:  # noqa: BLE001
            logger.debug("Could not append to memory store for run %s", run_id)

"""JSONL-based persistent store for fix outcome records.

Mirrors :class:`~vaig.core.memory.pattern_store.PatternMemoryStore` — each
run_id gets its own ``{run_id}.jsonl`` file.  All public methods are
error-silent so that a broken store never disrupts the live pipeline.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from vaig.core.memory.models import FixOutcome

logger = logging.getLogger(__name__)


class FixOutcomeStore:
    """Append-only JSONL store for :class:`~vaig.core.memory.models.FixOutcome` objects.

    Each ``run_id`` gets its own ``{run_id}.jsonl`` file inside ``base_dir``.
    An in-memory index keyed by ``fix_id`` is built lazily on first access
    and updated on every write.

    All public methods are error-silent — they catch and log exceptions so
    that a corrupt or missing store never disrupts the live pipeline.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._base_dir = Path(base_dir).expanduser()
        self._index: dict[str, FixOutcome] | None = None

    # ── Index ─────────────────────────────────────────────────

    def _ensure_index(self) -> dict[str, FixOutcome]:
        """Build the in-memory index from all JSONL files (lazy, once)."""
        if self._index is not None:
            return self._index

        self._index = {}
        try:
            if not self._base_dir.exists():
                return self._index

            for path in sorted(self._base_dir.glob("*.jsonl")):
                try:
                    with path.open(encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                                entry = FixOutcome.model_validate(data)
                                # Latest write wins for the same fix_id.
                                self._index[entry.fix_id] = entry
                            except Exception:  # noqa: BLE001
                                logger.debug(
                                    "FixOutcomeStore: skipping malformed entry in %s", path
                                )
                except Exception:  # noqa: BLE001
                    logger.debug("FixOutcomeStore: could not read file %s", path)
        except Exception:  # noqa: BLE001
            logger.debug("FixOutcomeStore: error scanning store at %s", self._base_dir)

        return self._index

    # ── Write helpers ─────────────────────────────────────────

    def _append(self, run_id: str, entry: FixOutcome) -> None:
        """Append *entry* as a JSON line to ``{run_id}.jsonl``.

        Creates parent directories if needed.  Silently ignores I/O errors.
        """
        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            path = self._base_dir / f"{run_id}.jsonl"
            line = entry.model_dump_json() + "\n"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        except Exception:  # noqa: BLE001
            logger.debug(
                "FixOutcomeStore: could not append to store for run %s", run_id
            )

    # ── Public API ────────────────────────────────────────────

    def record_fix(
        self,
        run_id: str,
        fix_id: str,
        fingerprint: str,
        strategy: str,
    ) -> FixOutcome:
        """Append a new :class:`FixOutcome` with ``outcome="unknown"``.

        Returns the created entry.  Never raises.
        """
        index = self._ensure_index()
        entry = FixOutcome(
            fix_id=fix_id,
            fingerprint=fingerprint,
            strategy=strategy,
            applied_at=datetime.now(UTC),
        )
        index[fix_id] = entry
        self._append(run_id, entry)
        return entry

    def correlate(
        self,
        fix_id: str,
        outcome: Literal["resolved", "persisted", "worsened"],
        correlated_run_id: str,
    ) -> FixOutcome | None:
        """Update outcome for *fix_id* and append the updated entry.

        Returns the updated entry, or ``None`` if *fix_id* is not found.
        Never raises.
        """
        index = self._ensure_index()
        existing = index.get(fix_id)
        if existing is None:
            return None

        updated = existing.model_copy(
            update={
                "outcome": outcome,
                "correlated_run_id": correlated_run_id,
                "correlated_at": datetime.now(UTC),
            }
        )
        index[fix_id] = updated
        # Append to the run that correlated the outcome.
        self._append(correlated_run_id, updated)
        return updated

    def lookup(self, fix_id: str) -> FixOutcome | None:
        """Return the most-recent :class:`FixOutcome` for *fix_id*, or ``None``."""
        return self._ensure_index().get(fix_id)

    def pending(self) -> list[FixOutcome]:
        """Return all :class:`FixOutcome` entries with ``outcome == "unknown"``."""
        return [e for e in self._ensure_index().values() if e.outcome == "unknown"]

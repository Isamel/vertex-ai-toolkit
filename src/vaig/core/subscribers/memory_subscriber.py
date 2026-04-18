"""Memory subscriber — records finding fingerprints on HealthReportCompleted.

Subscribes to :class:`~vaig.core.events.HealthReportCompleted` on the
:class:`~vaig.core.event_bus.EventBus` and persists finding fingerprints
to the :class:`~vaig.core.memory.pattern_store.PatternMemoryStore`.

When ``settings.memory.memory_rag_enabled`` is also True, the subscriber
additionally ingests pattern narratives into the memory RAG corpus via
:class:`~vaig.core.memory.memory_rag.MemoryRAGIndex` after each report.

Mirrors the :class:`~vaig.core.subscribers.audit_subscriber.AuditSubscriber`
pattern: all handlers are wrapped in ``try/except`` and silently log failures
so that a broken store never disrupts the live pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from vaig.core.event_bus import EventBus
from vaig.core.events import HealthReportCompleted
from vaig.core.memory.fingerprint import ObservationFingerprint
from vaig.core.memory.pattern_store import PatternMemoryStore

if TYPE_CHECKING:
    from vaig.core.config import Settings
    from vaig.core.memory.memory_rag import MemoryRAGIndex

logger = logging.getLogger(__name__)

__all__ = ["MemorySubscriber"]


class MemorySubscriber:
    """Subscribes to health report completion events and records fingerprints.

    When ``settings.memory.enabled`` is False this class is a no-op:
    the constructor returns immediately without subscribing.

    When ``settings.memory.memory_rag_enabled`` is True, a
    :class:`~vaig.core.memory.memory_rag.MemoryRAGIndex` is constructed and
    pattern narratives are ingested into the memory RAG corpus after each
    health report is processed.

    Args:
        settings: Application settings (provides ``memory`` config).
        store: Optional pre-built store for testing.  If not provided a new
            :class:`~vaig.core.memory.pattern_store.PatternMemoryStore` is
            created using ``settings.memory.store_path``.
        rag_index: Optional pre-built :class:`~vaig.core.memory.memory_rag.MemoryRAGIndex`
            for testing.  Created automatically when ``memory_rag_enabled`` is
            True and a pre-built index is not provided.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        store: PatternMemoryStore | None = None,
        rag_index: MemoryRAGIndex | None = None,
    ) -> None:
        self._settings = settings
        self._unsubscribers: list[Callable[[], None]] = []
        self._rag_index: MemoryRAGIndex | None = None

        if not settings.memory.enabled:
            return

        self._store = store or PatternMemoryStore(base_dir=settings.memory.store_path)

        if settings.memory.memory_rag_enabled:
            if rag_index is not None:
                self._rag_index = rag_index
            else:
                try:
                    from vaig.core.memory.memory_rag import MemoryRAGIndex  # noqa: PLC0415
                    from vaig.core.rag import RAGKnowledgeBase  # noqa: PLC0415

                    rag_kb = RAGKnowledgeBase(config=settings.export)
                    self._rag_index = MemoryRAGIndex(rag_kb=rag_kb, config=settings.memory)
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "MemorySubscriber: failed to initialise MemoryRAGIndex — "
                        "RAG ingestion disabled",
                        exc_info=True,
                    )

        self._subscribe_all()

    # ── Subscription wiring ──────────────────────────────────

    def _subscribe_all(self) -> None:
        """Register handler for HealthReportCompleted."""
        bus = EventBus.get()
        self._unsubscribers = [
            bus.subscribe(HealthReportCompleted, self._on_health_report_completed),
        ]

    def unsubscribe_all(self) -> None:
        """Detach all handlers."""
        for unsub in self._unsubscribers:
            unsub()
        self._unsubscribers.clear()

    # ── Event handlers ───────────────────────────────────────

    def _on_health_report_completed(self, event: HealthReportCompleted) -> None:
        """Record fingerprints for all findings in the completed report."""
        try:
            self._process_report(event)
        except Exception:  # noqa: BLE001
            logger.debug(
                "MemorySubscriber: failed to process HealthReportCompleted for run %s",
                event.run_id,
                exc_info=True,
            )

        if self._rag_index is not None:
            try:
                entries = list(self._store.all_entries())
                count = self._rag_index.ingest(entries)
                if count:
                    logger.debug(
                        "MemorySubscriber: ingested %d narratives into memory RAG corpus",
                        count,
                    )
            except Exception:  # noqa: BLE001
                logger.debug(
                    "MemorySubscriber: RAG ingest failed for run %s",
                    event.run_id,
                    exc_info=True,
                )

    def _process_report(self, event: HealthReportCompleted) -> None:
        """Load the report JSONL and record one fingerprint per finding.

        NOTE: If :class:`~vaig.core.memory.recurrence.RecurrenceAnalyzer` is
        also called inline with the same ``run_id`` and the same
        :class:`~vaig.core.memory.pattern_store.PatternMemoryStore`, findings
        will be counted twice.  Callers that wire both the analyzer and this
        subscriber must use separate stores or deduplicate by run_id externally.
        """
        import json
        from pathlib import Path

        path = Path(event.report_path)
        if not path.exists():
            logger.debug("MemorySubscriber: report path does not exist: %s", path)
            return

        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                findings = record.get("findings", [])
                for finding in findings:
                    fp = ObservationFingerprint.from_finding(
                        category=finding.get("category", ""),
                        service=finding.get("service", ""),
                        title=finding.get("title", ""),
                        description=finding.get("description", ""),
                    )
                    self._store.record(
                        run_id=event.run_id,
                        fingerprint=fp,
                        severity=finding.get("severity", ""),
                        title=finding.get("title", ""),
                        service=finding.get("service", ""),
                        category=finding.get("category", ""),
                    )
            except Exception:  # noqa: BLE001
                logger.debug("MemorySubscriber: skipping malformed report line")

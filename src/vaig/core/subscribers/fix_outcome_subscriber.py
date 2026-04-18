"""Fix outcome subscriber — records fix outcomes from FixAppliedEvent and HealthReportCompleted.

Subscribes to :class:`~vaig.core.events.FixAppliedEvent` and
:class:`~vaig.core.events.HealthReportCompleted` on the
:class:`~vaig.core.event_bus.EventBus`.

On ``FixAppliedEvent``:
  Calls :meth:`~vaig.core.memory.outcome_store.FixOutcomeStore.record_fix`
  to persist a new :class:`~vaig.core.memory.models.FixOutcome` with
  ``outcome="unknown"``.

On ``HealthReportCompleted``:
  Calls :meth:`~vaig.core.memory.outcome_store.FixOutcomeStore.correlate`
  for all pending (``outcome="unknown"``) entries so they receive a
  ``"resolved"`` or ``"persisted"`` outcome based on whether the follow-up
  run is clean.

Mirrors the :class:`~vaig.core.subscribers.audit_subscriber.AuditSubscriber`
pattern: all handlers are wrapped in ``try/except`` and silently log failures
so that a broken store never disrupts the live pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from vaig.core.event_bus import EventBus
from vaig.core.events import FixAppliedEvent, HealthReportCompleted

if TYPE_CHECKING:
    from vaig.core.config import Settings
    from vaig.core.memory.outcome_store import FixOutcomeStore

logger = logging.getLogger(__name__)

__all__ = ["FixOutcomeSubscriber"]


class FixOutcomeSubscriber:
    """Subscribes to fix and health events to track fix outcomes.

    When ``settings.memory.outcome_tracking_enabled`` is False this class
    is a no-op: the constructor returns immediately without subscribing.

    Args:
        settings: Application settings (provides ``memory`` config).
        store: Optional pre-built store for testing.  If not provided a new
            :class:`~vaig.core.memory.outcome_store.FixOutcomeStore` is
            created using ``settings.memory.outcome_store_path``.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        store: FixOutcomeStore | None = None,
    ) -> None:
        self._settings = settings
        self._unsubscribers: list[Callable[[], None]] = []

        if not settings.memory.outcome_tracking_enabled:
            return

        if store is not None:
            self._store = store
        else:
            from vaig.core.memory.outcome_store import FixOutcomeStore  # noqa: PLC0415

            self._store = FixOutcomeStore(base_dir=settings.memory.outcome_store_path)

        self._subscribe_all()

    # ── Subscription wiring ──────────────────────────────────

    def _subscribe_all(self) -> None:
        """Register handlers for FixAppliedEvent and HealthReportCompleted."""
        bus = EventBus.get()
        self._unsubscribers = [
            bus.subscribe(FixAppliedEvent, self._on_fix_applied),
            bus.subscribe(HealthReportCompleted, self._on_health_report_completed),
        ]

    def unsubscribe_all(self) -> None:
        """Detach all handlers."""
        for unsub in self._unsubscribers:
            unsub()
        self._unsubscribers.clear()

    # ── Event handlers ───────────────────────────────────────

    def _on_fix_applied(self, event: FixAppliedEvent) -> None:
        """Record a new fix outcome entry with outcome='unknown'."""
        try:
            self._store.record_fix(
                run_id=event.run_id,
                fix_id=event.fix_id,
                fingerprint=event.fingerprint,
                strategy=event.strategy,
            )
        except Exception:  # noqa: BLE001
            logger.debug(
                "FixOutcomeSubscriber: failed to record FixAppliedEvent fix_id=%s run=%s",
                event.fix_id,
                event.run_id,
                exc_info=True,
            )

    def _on_health_report_completed(self, event: HealthReportCompleted) -> None:
        """Correlate pending fix outcomes against the new health report."""
        try:
            self._correlate_pending(event)
        except Exception:  # noqa: BLE001
            logger.debug(
                "FixOutcomeSubscriber: failed to correlate HealthReportCompleted for run %s",
                event.run_id,
                exc_info=True,
            )

    def _correlate_pending(self, event: HealthReportCompleted) -> None:
        """For each pending FixOutcome, correlate against the new run.

        Reads the new health report to determine which fingerprints still
        appear.  Fixes whose fingerprint is absent → ``"resolved"``.
        Fixes whose fingerprint still appears → ``"persisted"``.
        """
        import json
        from pathlib import Path

        path = Path(event.report_path)
        if not path.exists():
            logger.debug(
                "FixOutcomeSubscriber: report path does not exist: %s", path
            )
            return

        # Collect fingerprints present in this new report.
        active_fingerprints: set[str] = set()
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    for finding in record.get("findings", []):
                        fp = finding.get("fingerprint", "")
                        if fp:
                            active_fingerprints.add(fp)
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "FixOutcomeSubscriber: skipping malformed report line"
                    )

        for outcome_entry in self._store.pending():
            new_outcome = (
                "persisted"
                if outcome_entry.fingerprint in active_fingerprints
                else "resolved"
            )
            self._store.correlate(
                fix_id=outcome_entry.fix_id,
                outcome=new_outcome,  # type: ignore[arg-type]
                correlated_run_id=event.run_id,
            )

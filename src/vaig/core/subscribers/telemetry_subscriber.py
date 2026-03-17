"""Telemetry subscriber — adapts domain events to TelemetryCollector calls.

This is a pure adapter: it subscribes to all 10 event types on the
:class:`~vaig.core.event_bus.EventBus` and translates each into the
equivalent :class:`~vaig.core.telemetry.TelemetryCollector` method call.

The subscriber produces **identical** telemetry output to the direct
``emit_*`` calls scattered throughout the codebase.  Zero behavior change.

Each handler is wrapped in try/except so a telemetry failure never
propagates to the emitting code.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from vaig.core.event_bus import EventBus
from vaig.core.events import (
    ApiCalled,
    BudgetChecked,
    CliCommandTracked,
    ErrorOccurred,
    OrchestratorPhaseCompleted,
    OrchestratorToolsCompleted,
    SessionEnded,
    SessionStarted,
    SkillUsed,
    ToolExecuted,
)

if TYPE_CHECKING:
    from vaig.core.telemetry import TelemetryCollector

logger = logging.getLogger(__name__)

__all__ = [
    "TelemetrySubscriber",
]


class TelemetrySubscriber:
    """Subscribes to all domain events and forwards them to the TelemetryCollector.

    Each handler maps a frozen :class:`~vaig.core.events.Event` dataclass
    to the corresponding ``TelemetryCollector.emit_*`` or ``emit()`` call,
    preserving the exact ``event_type`` strings and field mappings that the
    collector expects.

    Usage::

        from vaig.core.subscribers import TelemetrySubscriber
        from vaig.core.telemetry import get_telemetry_collector

        subscriber = TelemetrySubscriber(get_telemetry_collector())
        # subscriber is now listening — call unsubscribe_all() to detach.
    """

    def __init__(self, collector: TelemetryCollector) -> None:
        self._collector = collector
        self._unsubscribers: list[Callable[[], None]] = []
        self._subscribe_all()

    # ── Subscription wiring ──────────────────────────────────

    def _subscribe_all(self) -> None:
        """Register handlers for all 10 event types on the singleton EventBus."""
        bus = EventBus.get()
        self._unsubscribers = [
            bus.subscribe(ToolExecuted, self._on_tool_executed),
            bus.subscribe(ApiCalled, self._on_api_called),
            bus.subscribe(ErrorOccurred, self._on_error_occurred),
            bus.subscribe(SessionStarted, self._on_session_started),
            bus.subscribe(SessionEnded, self._on_session_ended),
            bus.subscribe(SkillUsed, self._on_skill_used),
            bus.subscribe(CliCommandTracked, self._on_cli_command_tracked),
            bus.subscribe(BudgetChecked, self._on_budget_checked),
            bus.subscribe(OrchestratorPhaseCompleted, self._on_orchestrator_phase_completed),
            bus.subscribe(OrchestratorToolsCompleted, self._on_orchestrator_tools_completed),
        ]

    def unsubscribe_all(self) -> None:
        """Detach all handlers from the EventBus.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        for unsub in self._unsubscribers:
            unsub()
        self._unsubscribers.clear()

    # ── Event handlers ───────────────────────────────────────
    #
    # Each handler translates an Event dataclass into the matching
    # TelemetryCollector method call.  The event_type strings and
    # field mappings are kept identical to the existing direct calls.

    def _on_tool_executed(self, event: ToolExecuted) -> None:
        """ToolExecuted → emit_tool_call()."""
        try:
            metadata: dict[str, object] | None = None
            if event.args_keys or event.error:
                metadata = {}
                if event.args_keys:
                    metadata["args_keys"] = list(event.args_keys)
                if event.error:
                    metadata["error"] = event.error
            self._collector.emit_tool_call(
                event.tool_name,
                duration_ms=event.duration_ms,
                metadata=metadata,
                error_type=event.error_type,
                error_message=event.error_message,
            )
        except Exception:  # noqa: BLE001
            logger.debug("TelemetrySubscriber: failed to handle ToolExecuted", exc_info=True)

    def _on_api_called(self, event: ApiCalled) -> None:
        """ApiCalled → emit_api_call()."""
        try:
            metadata = dict(event.metadata) if event.metadata else None
            self._collector.emit_api_call(
                event.model,
                duration_ms=event.duration_ms,
                tokens_in=event.tokens_in,
                tokens_out=event.tokens_out,
                cost_usd=event.cost_usd,
                metadata=metadata,
            )
        except Exception:  # noqa: BLE001
            logger.debug("TelemetrySubscriber: failed to handle ApiCalled", exc_info=True)

    def _on_error_occurred(self, event: ErrorOccurred) -> None:
        """ErrorOccurred → emit_error()."""
        try:
            metadata = dict(event.metadata) if event.metadata else None
            if metadata is not None and event.source:
                metadata["source"] = event.source
            elif event.source:
                metadata = {"source": event.source}
            self._collector.emit_error(
                event.error_type,
                event.error_message,
                metadata=metadata,
            )
        except Exception:  # noqa: BLE001
            logger.debug("TelemetrySubscriber: failed to handle ErrorOccurred", exc_info=True)

    def _on_session_started(self, event: SessionStarted) -> None:
        """SessionStarted → set_session_id() + emit(session, session_start)."""
        try:
            self._collector.set_session_id(event.session_id)
            self._collector.emit(
                event_type="session",
                event_name="session_start",
                metadata={
                    "name": event.name,
                    "model": event.model,
                    "skill": event.skill,
                },
            )
        except Exception:  # noqa: BLE001
            logger.debug("TelemetrySubscriber: failed to handle SessionStarted", exc_info=True)

    def _on_session_ended(self, event: SessionEnded) -> None:
        """SessionEnded → emit(session, session_end)."""
        try:
            self._collector.emit(
                event_type="session",
                event_name="session_end",
                duration_ms=event.duration_ms,
            )
        except Exception:  # noqa: BLE001
            logger.debug("TelemetrySubscriber: failed to handle SessionEnded", exc_info=True)

    def _on_skill_used(self, event: SkillUsed) -> None:
        """SkillUsed → emit_skill_use()."""
        try:
            self._collector.emit_skill_use(
                event.skill_name,
                duration_ms=event.duration_ms,
            )
        except Exception:  # noqa: BLE001
            logger.debug("TelemetrySubscriber: failed to handle SkillUsed", exc_info=True)

    def _on_cli_command_tracked(self, event: CliCommandTracked) -> None:
        """CliCommandTracked → emit_cli_command()."""
        try:
            self._collector.emit_cli_command(
                event.command_name,
                duration_ms=event.duration_ms,
            )
        except Exception:  # noqa: BLE001
            logger.debug("TelemetrySubscriber: failed to handle CliCommandTracked", exc_info=True)

    def _on_budget_checked(self, event: BudgetChecked) -> None:
        """BudgetChecked → emit(budget, budget-checked)."""
        try:
            self._collector.emit(
                event_type="budget",
                event_name="budget-checked",
                metadata={
                    "status": event.status,
                    "cost_usd": event.cost_usd,
                    "limit_usd": event.limit_usd,
                    "message": event.message,
                },
            )
        except Exception:  # noqa: BLE001
            logger.debug("TelemetrySubscriber: failed to handle BudgetChecked", exc_info=True)

    def _on_orchestrator_phase_completed(self, event: OrchestratorPhaseCompleted) -> None:
        """OrchestratorPhaseCompleted → emit(orchestrator, execute_skill_phase / async_...)."""
        try:
            event_name = "async_execute_skill_phase" if event.is_async else "execute_skill_phase"
            self._collector.emit(
                event_type="orchestrator",
                event_name=event_name,
                duration_ms=event.duration_ms,
                metadata={
                    "skill": event.skill,
                    "phase": event.phase,
                    "strategy": event.strategy,
                },
            )
        except Exception:  # noqa: BLE001
            logger.debug("TelemetrySubscriber: failed to handle OrchestratorPhaseCompleted", exc_info=True)

    def _on_orchestrator_tools_completed(self, event: OrchestratorToolsCompleted) -> None:
        """OrchestratorToolsCompleted → emit(orchestrator, execute_with_tools / async_...)."""
        try:
            event_name = "async_execute_with_tools" if event.is_async else "execute_with_tools"
            self._collector.emit(
                event_type="orchestrator",
                event_name=event_name,
                duration_ms=event.duration_ms,
                metadata={
                    "skill": event.skill,
                    "strategy": event.strategy,
                    "agents_count": event.agents_count,
                    "success": event.success,
                },
            )
        except Exception:  # noqa: BLE001
            logger.debug("TelemetrySubscriber: failed to handle OrchestratorToolsCompleted", exc_info=True)

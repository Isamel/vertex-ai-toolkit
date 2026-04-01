"""EventBus → asyncio.Queue bridge for per-request event forwarding.

Provides :class:`EventQueueBridge` — an async context manager that
subscribes to the :class:`~vaig.core.event_bus.EventBus` singleton for
the duration of a single web request and pushes received events into a
per-request ``asyncio.Queue``.

Usage::

    async with EventQueueBridge(event_bus) as queue:
        # queue is an asyncio.Queue[Event | None]
        # Events emitted on the bus appear in `queue`
        ...
    # Automatically unsubscribed on exit

CRITICAL: EventBus dispatches handlers synchronously on the emitting
thread (which may be a worker thread via ``asyncio.to_thread()``).
Since ``asyncio.Queue`` is NOT thread-safe, we use
``loop.call_soon_threadsafe()`` to bridge the gap.
"""

from __future__ import annotations

import asyncio
import logging
from types import TracebackType
from typing import Any

from vaig.core.event_bus import EventBus
from vaig.core.events import (
    AgentProgressCompleted,
    AgentProgressStarted,
    ErrorOccurred,
    Event,
    OrchestratorPhaseCompleted,
    ToolExecuted,
)

__all__ = [
    "EventQueueBridge",
]

logger = logging.getLogger(__name__)

# Event types to forward to the SSE stream (same as ask route)
_SUBSCRIBED_EVENTS: tuple[type[Event], ...] = (
    ToolExecuted,
    OrchestratorPhaseCompleted,
    ErrorOccurred,
)

# Extended event set for live mode (includes agent progress events)
_LIVE_SUBSCRIBED_EVENTS: tuple[type[Event], ...] = (
    ToolExecuted,
    OrchestratorPhaseCompleted,
    ErrorOccurred,
    AgentProgressStarted,
    AgentProgressCompleted,
)


class EventQueueBridge:
    """Async context manager bridging EventBus → asyncio.Queue.

    Subscribes to configured event types on enter, unsubscribes on exit.
    Events are forwarded via ``loop.call_soon_threadsafe()`` to guarantee
    thread-safety when the EventBus dispatches on worker threads.

    Args:
        event_bus: The ``EventBus`` instance to subscribe to.
        event_types: Optional tuple of event types to subscribe to.
                     Defaults to ``(ToolExecuted, OrchestratorPhaseCompleted,
                     ErrorOccurred)``.

    Example::

        async with EventQueueBridge(container.event_bus) as queue:
            # Start orchestrator work...
            event = await queue.get()
    """

    def __init__(
        self,
        event_bus: EventBus,
        event_types: tuple[type[Event], ...] | None = None,
    ) -> None:
        self._bus = event_bus
        self._event_types = event_types or _SUBSCRIBED_EVENTS
        self._queue: asyncio.Queue[Event | None] = asyncio.Queue()
        self._unsub_fns: list[Any] = []

    @property
    def queue(self) -> asyncio.Queue[Event | None]:
        """The per-request event queue."""
        return self._queue

    async def __aenter__(self) -> asyncio.Queue[Event | None]:
        """Subscribe to EventBus and return the queue."""
        loop = asyncio.get_running_loop()

        def _threadsafe_handler(
            event: Event,
            *,
            _loop: asyncio.AbstractEventLoop = loop,
            _queue: asyncio.Queue[Event | None] = self._queue,
        ) -> None:
            _loop.call_soon_threadsafe(_queue.put_nowait, event)

        for event_type in self._event_types:
            unsub = self._bus.subscribe(event_type, _threadsafe_handler)
            self._unsub_fns.append(unsub)

        return self._queue

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Unsubscribe from all event types — guaranteed cleanup."""
        for unsub in self._unsub_fns:
            try:
                unsub()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to unsubscribe event handler", exc_info=True)
        self._unsub_fns.clear()

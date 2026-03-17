"""Process-wide, thread-safe, synchronous event bus.

Provides a singleton :class:`EventBus` that lets business modules emit
typed :class:`~vaig.core.events.Event` instances while subscribers react
without tight coupling.

Usage::

    from vaig.core.event_bus import EventBus
    from vaig.core.events import ToolExecuted

    bus = EventBus.get()
    unsub = bus.subscribe(ToolExecuted, lambda e: print(e.tool_name))
    bus.emit(ToolExecuted(tool_name="kubectl"))
    unsub()  # unsubscribe
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import TypeVar

from vaig.core.events import Event

__all__ = [
    "EventBus",
]

logger = logging.getLogger(__name__)

E = TypeVar("E", bound=Event)


class EventBus:
    """Thread-safe, singleton event bus for in-process event routing.

    Handlers are dispatched synchronously on the emitting thread.
    Each handler is wrapped in a try/except so that a failing subscriber
    never disrupts the emitter or other subscribers.
    """

    _instance: EventBus | None = None
    _class_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._handlers: dict[type[Event], list[Callable[[Event], None]]] = {}
        self._lock = threading.Lock()

    # ── Singleton access ─────────────────────────────────────

    @classmethod
    def get(cls) -> EventBus:
        """Return the process-wide singleton, creating it on first call.

        Thread-safe via double-checked locking.
        """
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Subscribe / Unsubscribe ──────────────────────────────

    def subscribe(
        self,
        event_type: type[E],
        handler: Callable[[E], None],
    ) -> Callable[[], None]:
        """Register *handler* to be called when events of *event_type* are emitted.

        Args:
            event_type: The concrete :class:`Event` subclass to listen for.
            handler: A callable that accepts a single event argument.

        Returns:
            A no-arg callable that, when invoked, removes this subscription.
        """
        with self._lock:
            self._handlers.setdefault(event_type, []).append(handler)  # type: ignore[arg-type]

        def _unsubscribe() -> None:
            with self._lock:
                handlers = self._handlers.get(event_type, [])
                try:
                    handlers.remove(handler)  # type: ignore[arg-type]
                except ValueError:
                    pass  # already removed

        return _unsubscribe

    # ── Emit ─────────────────────────────────────────────────

    def emit(self, event: Event) -> None:
        """Dispatch *event* to all handlers registered for its exact type.

        Each handler is called synchronously.  If a handler raises, the
        exception is logged as a warning and remaining handlers are still
        invoked.

        Args:
            event: The event instance to dispatch.
        """
        with self._lock:
            # Snapshot to release the lock quickly.
            handlers = list(self._handlers.get(type(event), []))

        for handler in handlers:
            try:
                handler(event)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Event handler %r failed for %s",
                    handler,
                    type(event).__name__,
                    exc_info=True,
                )

    # ── Lifecycle ────────────────────────────────────────────

    def reset(self) -> None:
        """Remove all subscriptions.  Intended for testing only."""
        with self._lock:
            self._handlers.clear()

    @classmethod
    def _reset_singleton(cls) -> None:
        """Destroy the singleton instance.  **Testing only.**"""
        with cls._class_lock:
            cls._instance = None

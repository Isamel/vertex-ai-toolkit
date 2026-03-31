"""Tests for EventQueueBridge — Task 3.3.

Covers:
- Events emitted on EventBus appear in the queue
- Per-request isolation (two concurrent bridges get independent queues)
- Cleanup on exit (unsubscribe)
- Thread-safety via loop.call_soon_threadsafe
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip(
    "fastapi",
    reason="FastAPI not available; install the 'web' extra to run web tests.",
)

from vaig.core.event_bus import EventBus
from vaig.core.events import ErrorOccurred, OrchestratorPhaseCompleted, ToolExecuted
from vaig.web.events import EventQueueBridge

# -- Fixtures ----------------------------------------------------------


@pytest.fixture()
def event_bus() -> EventBus:
    """Fresh EventBus instance for each test (avoid singleton bleed)."""
    bus = EventBus()
    return bus


# -- Basic functionality -----------------------------------------------


@pytest.mark.asyncio
async def test_bridge_receives_emitted_events(event_bus: EventBus) -> None:
    """Events emitted on the EventBus should appear in the bridge queue."""
    async with EventQueueBridge(event_bus) as queue:
        event_bus.emit(ToolExecuted(tool_name="kubectl", duration_ms=42.0))

        # Give call_soon_threadsafe a tick to schedule
        await asyncio.sleep(0.05)

        assert not queue.empty()
        event = queue.get_nowait()
        assert isinstance(event, ToolExecuted)
        assert event.tool_name == "kubectl"
        assert event.duration_ms == 42.0


@pytest.mark.asyncio
async def test_bridge_receives_multiple_event_types(event_bus: EventBus) -> None:
    """All subscribed event types should be forwarded."""
    async with EventQueueBridge(event_bus) as queue:
        event_bus.emit(ToolExecuted(tool_name="gcloud"))
        event_bus.emit(OrchestratorPhaseCompleted(skill="rca", phase="analyze"))
        event_bus.emit(ErrorOccurred(error_type="TestError", error_message="oops"))

        await asyncio.sleep(0.05)

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        assert len(events) == 3
        assert isinstance(events[0], ToolExecuted)
        assert isinstance(events[1], OrchestratorPhaseCompleted)
        assert isinstance(events[2], ErrorOccurred)


@pytest.mark.asyncio
async def test_bridge_unsubscribes_on_exit(event_bus: EventBus) -> None:
    """After exiting the context manager, no more events should be received."""
    bridge = EventQueueBridge(event_bus)
    async with bridge as queue:
        event_bus.emit(ToolExecuted(tool_name="before"))
        await asyncio.sleep(0.05)
        assert not queue.empty()

    # After __aexit__, emit another event
    event_bus.emit(ToolExecuted(tool_name="after"))
    await asyncio.sleep(0.05)

    # The queue should NOT have the second event (only the first from before)
    remaining = []
    while not bridge.queue.empty():
        remaining.append(bridge.queue.get_nowait())
    tool_names = [e.tool_name for e in remaining if isinstance(e, ToolExecuted)]
    assert "after" not in tool_names


@pytest.mark.asyncio
async def test_bridge_isolation_two_concurrent(event_bus: EventBus) -> None:
    """Two concurrent bridges should get independent queues."""
    async with EventQueueBridge(event_bus) as queue_a:
        async with EventQueueBridge(event_bus) as queue_b:
            event_bus.emit(ToolExecuted(tool_name="shared"))
            await asyncio.sleep(0.05)

            # Both queues should receive the same event independently
            assert not queue_a.empty()
            assert not queue_b.empty()

            event_a = queue_a.get_nowait()
            event_b = queue_b.get_nowait()
            assert isinstance(event_a, ToolExecuted)
            assert isinstance(event_b, ToolExecuted)
            assert event_a.tool_name == "shared"
            assert event_b.tool_name == "shared"


@pytest.mark.asyncio
async def test_bridge_custom_event_types(event_bus: EventBus) -> None:
    """Bridge with custom event types should only receive those types."""
    async with EventQueueBridge(
        event_bus, event_types=(ErrorOccurred,)
    ) as queue:
        # This should NOT be forwarded
        event_bus.emit(ToolExecuted(tool_name="ignored"))
        # This SHOULD be forwarded
        event_bus.emit(ErrorOccurred(error_type="TestError", error_message="test"))

        await asyncio.sleep(0.05)

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        assert len(events) == 1
        assert isinstance(events[0], ErrorOccurred)


@pytest.mark.asyncio
async def test_bridge_queue_property(event_bus: EventBus) -> None:
    """The queue property should return the same queue as __aenter__."""
    bridge = EventQueueBridge(event_bus)
    async with bridge as queue:
        assert bridge.queue is queue


@pytest.mark.asyncio
async def test_bridge_cleanup_on_exception(event_bus: EventBus) -> None:
    """Bridge should unsubscribe even if an exception occurs inside the block."""
    bridge = EventQueueBridge(event_bus)
    with pytest.raises(ValueError, match="test error"):
        async with bridge:
            raise ValueError("test error")

    # After exception, emit — should not reach the bridge queue
    event_bus.emit(ToolExecuted(tool_name="post-error"))
    await asyncio.sleep(0.05)
    assert bridge.queue.empty()


@pytest.mark.asyncio
async def test_bridge_uses_call_soon_threadsafe(event_bus: EventBus) -> None:
    """Bridge handler should use loop.call_soon_threadsafe for thread-safety."""
    with patch("asyncio.get_running_loop") as mock_get_loop:
        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop

        async with EventQueueBridge(event_bus):
            # The handler was registered — now emit an event to trigger it
            event_bus.emit(ToolExecuted(tool_name="test"))

            # call_soon_threadsafe should have been called
            assert mock_loop.call_soon_threadsafe.called

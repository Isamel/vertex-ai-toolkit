"""Tests for EventBus — subscribe, emit, unsubscribe, isolation, thread safety."""

from __future__ import annotations

import threading
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock

import pytest

from vaig.core.event_bus import EventBus
from vaig.core.events import ApiCalled, ErrorOccurred, ToolExecuted

# ── Fixture: fresh EventBus per test ────────────────────────


@pytest.fixture(autouse=True)
def _fresh_bus() -> Generator[None, None, None]:
    """Reset the EventBus singleton between tests to avoid cross-contamination."""
    EventBus._reset_singleton()
    yield
    EventBus._reset_singleton()


# ══════════════════════════════════════════════════════════════
# Subscribe & Emit
# ══════════════════════════════════════════════════════════════


class TestSubscribeAndEmit:
    """Tests for basic subscribe + emit flow."""

    def test_subscribe_and_receive_event(self) -> None:
        bus = EventBus.get()
        received: list[ToolExecuted] = []
        bus.subscribe(ToolExecuted, received.append)

        evt = ToolExecuted(tool_name="kubectl")
        bus.emit(evt)

        assert len(received) == 1
        assert received[0] is evt

    def test_multiple_subscribers_same_event(self) -> None:
        bus = EventBus.get()
        received_a: list[ToolExecuted] = []
        received_b: list[ToolExecuted] = []
        bus.subscribe(ToolExecuted, received_a.append)
        bus.subscribe(ToolExecuted, received_b.append)

        evt = ToolExecuted(tool_name="helm")
        bus.emit(evt)

        assert len(received_a) == 1
        assert len(received_b) == 1
        assert received_a[0] is evt
        assert received_b[0] is evt

    def test_emit_with_no_subscribers_no_error(self) -> None:
        bus = EventBus.get()
        # Should not raise.
        bus.emit(ToolExecuted(tool_name="kubectl"))

    def test_multiple_events_emitted(self) -> None:
        bus = EventBus.get()
        received: list[ToolExecuted] = []
        bus.subscribe(ToolExecuted, received.append)

        bus.emit(ToolExecuted(tool_name="kubectl"))
        bus.emit(ToolExecuted(tool_name="helm"))
        bus.emit(ToolExecuted(tool_name="gcloud"))

        assert len(received) == 3
        assert [e.tool_name for e in received] == ["kubectl", "helm", "gcloud"]


# ══════════════════════════════════════════════════════════════
# Unsubscribe
# ══════════════════════════════════════════════════════════════


class TestUnsubscribe:
    """Tests for the unsubscribe callable returned by subscribe."""

    def test_unsubscribe_stops_delivery(self) -> None:
        bus = EventBus.get()
        received: list[ToolExecuted] = []
        unsub = bus.subscribe(ToolExecuted, received.append)

        bus.emit(ToolExecuted(tool_name="kubectl"))
        assert len(received) == 1

        unsub()
        bus.emit(ToolExecuted(tool_name="helm"))
        assert len(received) == 1  # still 1 — second not delivered

    def test_double_unsubscribe_is_safe(self) -> None:
        bus = EventBus.get()
        unsub = bus.subscribe(ToolExecuted, lambda _: None)
        unsub()
        unsub()  # should not raise

    def test_unsubscribe_one_leaves_others(self) -> None:
        bus = EventBus.get()
        received_a: list[ToolExecuted] = []
        received_b: list[ToolExecuted] = []
        unsub_a = bus.subscribe(ToolExecuted, received_a.append)
        bus.subscribe(ToolExecuted, received_b.append)

        unsub_a()
        bus.emit(ToolExecuted(tool_name="kubectl"))

        assert len(received_a) == 0
        assert len(received_b) == 1


# ══════════════════════════════════════════════════════════════
# Exception Isolation
# ══════════════════════════════════════════════════════════════


class TestExceptionIsolation:
    """Verify that a failing subscriber doesn't break others."""

    def test_failing_handler_doesnt_block_others(self) -> None:
        bus = EventBus.get()
        received: list[ToolExecuted] = []

        def exploding_handler(_: ToolExecuted) -> None:
            msg = "boom"
            raise RuntimeError(msg)

        bus.subscribe(ToolExecuted, exploding_handler)
        bus.subscribe(ToolExecuted, received.append)

        # Should not raise, and second handler should still be called.
        bus.emit(ToolExecuted(tool_name="kubectl"))
        assert len(received) == 1

    def test_failing_handler_logged_as_warning(self) -> None:
        bus = EventBus.get()
        mock_logger = MagicMock()

        def exploding_handler(_: ToolExecuted) -> None:
            msg = "kaboom"
            raise ValueError(msg)

        bus.subscribe(ToolExecuted, exploding_handler)

        import vaig.core.event_bus as eb_module

        original_logger = eb_module.logger
        eb_module.logger = mock_logger
        try:
            bus.emit(ToolExecuted(tool_name="kubectl"))
        finally:
            eb_module.logger = original_logger

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert "failed" in call_args[0][0].lower()


# ══════════════════════════════════════════════════════════════
# Reset
# ══════════════════════════════════════════════════════════════


class TestReset:
    """Tests for the reset() method."""

    def test_reset_clears_all_subscriptions(self) -> None:
        bus = EventBus.get()
        received: list[ToolExecuted] = []
        bus.subscribe(ToolExecuted, received.append)
        bus.subscribe(ApiCalled, lambda _: None)

        bus.reset()
        bus.emit(ToolExecuted(tool_name="kubectl"))

        assert len(received) == 0

    def test_can_subscribe_after_reset(self) -> None:
        bus = EventBus.get()
        bus.subscribe(ToolExecuted, lambda _: None)
        bus.reset()

        received: list[ToolExecuted] = []
        bus.subscribe(ToolExecuted, received.append)
        bus.emit(ToolExecuted(tool_name="kubectl"))

        assert len(received) == 1


# ══════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════


class TestSingleton:
    """Tests for singleton behavior."""

    def test_get_returns_same_instance(self) -> None:
        bus_a = EventBus.get()
        bus_b = EventBus.get()
        assert bus_a is bus_b

    def test_reset_singleton_creates_new_instance(self) -> None:
        bus_a = EventBus.get()
        EventBus._reset_singleton()
        bus_b = EventBus.get()
        assert bus_a is not bus_b


# ══════════════════════════════════════════════════════════════
# Type Routing
# ══════════════════════════════════════════════════════════════


class TestTypeRouting:
    """Verify events are routed only to handlers of the matching type."""

    def test_subscribing_to_tool_executed_ignores_api_called(self) -> None:
        bus = EventBus.get()
        tool_received: list[ToolExecuted] = []
        bus.subscribe(ToolExecuted, tool_received.append)

        bus.emit(ApiCalled(model="gemini-2.5-pro", tokens_in=100))

        assert len(tool_received) == 0

    def test_each_type_gets_own_events(self) -> None:
        bus = EventBus.get()
        tool_events: list[ToolExecuted] = []
        api_events: list[ApiCalled] = []
        error_events: list[ErrorOccurred] = []

        bus.subscribe(ToolExecuted, tool_events.append)
        bus.subscribe(ApiCalled, api_events.append)
        bus.subscribe(ErrorOccurred, error_events.append)

        bus.emit(ToolExecuted(tool_name="kubectl"))
        bus.emit(ApiCalled(model="gemini-2.5-pro"))
        bus.emit(ErrorOccurred(error_type="ValueError", error_message="bad"))

        assert len(tool_events) == 1
        assert len(api_events) == 1
        assert len(error_events) == 1


# ══════════════════════════════════════════════════════════════
# Thread Safety
# ══════════════════════════════════════════════════════════════


class TestThreadSafety:
    """Verify concurrent subscribe/emit doesn't corrupt state."""

    def test_concurrent_subscribe_and_emit(self) -> None:
        bus = EventBus.get()
        total_events = 200
        results: list[ToolExecuted] = []
        lock = threading.Lock()

        def safe_append(evt: ToolExecuted) -> None:
            with lock:
                results.append(evt)

        bus.subscribe(ToolExecuted, safe_append)

        def emit_batch(start: int) -> int:
            """Emit events and return count emitted."""
            count = 0
            for i in range(start, start + 50):
                bus.emit(ToolExecuted(tool_name=f"tool-{i}"))
                count += 1
            return count

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(emit_batch, i * 50) for i in range(4)]
            emitted = sum(f.result() for f in as_completed(futures))

        assert emitted == total_events
        assert len(results) == total_events

    def test_concurrent_subscribe_doesnt_crash(self) -> None:
        bus = EventBus.get()

        def subscribe_many(start: int) -> int:
            count = 0
            for i in range(start, start + 50):
                bus.subscribe(ToolExecuted, lambda _e, idx=i: None)  # type: ignore[misc]
                count += 1
            return count

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(subscribe_many, i * 50) for i in range(4)]
            subscribed = sum(f.result() for f in as_completed(futures))

        assert subscribed == 200

    def test_singleton_thread_safety(self) -> None:
        """Multiple threads calling get() all receive the same instance."""
        EventBus._reset_singleton()
        instances: list[EventBus] = []
        lock = threading.Lock()

        def get_instance() -> None:
            inst = EventBus.get()
            with lock:
                instances.append(inst)

        threads = [threading.Thread(target=get_instance) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(inst is instances[0] for inst in instances)

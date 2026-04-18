"""Unit tests for LoopStepEvent dataclass (T-10)."""

from __future__ import annotations

import dataclasses

from vaig.core.event_bus import EventBus
from vaig.core.events import Event, LoopStepEvent

# ══════════════════════════════════════════════════════════════
# LoopStepEvent Structure Tests
# ══════════════════════════════════════════════════════════════


class TestLoopStepEventStructure:
    def test_is_frozen_dataclass(self) -> None:
        assert dataclasses.is_dataclass(LoopStepEvent)
        params = dataclasses.fields(LoopStepEvent)
        assert params is not None
        # Verify frozen by attempting attribute assignment
        event = LoopStepEvent()
        try:
            event.run_id = "changed"  # type: ignore[misc]
            raise AssertionError("Should have raised FrozenInstanceError")
        except (dataclasses.FrozenInstanceError, TypeError, AttributeError):
            pass

    def test_inherits_from_event(self) -> None:
        event = LoopStepEvent()
        assert isinstance(event, Event)

    def test_event_type_is_loop_step(self) -> None:
        event = LoopStepEvent()
        assert event.event_type == "loop.step"

    def test_event_type_not_in_init(self) -> None:
        # event_type is set via field(default=..., init=False)
        fields_by_name = {f.name: f for f in dataclasses.fields(LoopStepEvent)}
        assert fields_by_name["event_type"].init is False

    def test_timestamp_auto_populated(self) -> None:
        event = LoopStepEvent()
        assert event.timestamp != ""
        assert "T" in event.timestamp

    def test_default_field_values(self) -> None:
        event = LoopStepEvent()
        assert event.run_id == ""
        assert event.skill == ""
        assert event.loop_type == ""
        assert event.iteration == 0
        assert event.inputs_hash == ""
        assert event.outputs_hash == ""
        assert event.tokens_used == 0
        assert event.tool_calls_made == 0
        assert event.budget_remaining_usd == 0.0
        assert event.termination_reason == ""

    def test_all_fields_set(self) -> None:
        event = LoopStepEvent(
            run_id="run-abc123",
            skill="kubernetes",
            loop_type="hypothesis",
            iteration=3,
            inputs_hash="abcd12345678abcd",
            outputs_hash="efgh12345678efgh",
            tokens_used=1500,
            tool_calls_made=4,
            budget_remaining_usd=0.85,
            termination_reason="text_response",
        )
        assert event.run_id == "run-abc123"
        assert event.skill == "kubernetes"
        assert event.loop_type == "hypothesis"
        assert event.iteration == 3
        assert event.inputs_hash == "abcd12345678abcd"
        assert event.outputs_hash == "efgh12345678efgh"
        assert event.tokens_used == 1500
        assert event.tool_calls_made == 4
        assert event.budget_remaining_usd == 0.85
        assert event.termination_reason == "text_response"


class TestLoopStepEventHashFields:
    def test_inputs_hash_16_chars(self) -> None:
        event = LoopStepEvent(inputs_hash="a1b2c3d4e5f6a1b2")
        assert len(event.inputs_hash) == 16

    def test_outputs_hash_16_chars(self) -> None:
        event = LoopStepEvent(outputs_hash="1234567890abcdef")
        assert len(event.outputs_hash) == 16

    def test_empty_hash_allowed(self) -> None:
        event = LoopStepEvent(inputs_hash="", outputs_hash="")
        assert event.inputs_hash == ""
        assert event.outputs_hash == ""


class TestLoopStepEventFieldTypes:
    def test_run_id_is_str(self) -> None:
        event = LoopStepEvent(run_id="run-1")
        assert isinstance(event.run_id, str)

    def test_iteration_is_int(self) -> None:
        event = LoopStepEvent(iteration=5)
        assert isinstance(event.iteration, int)

    def test_tokens_used_is_int(self) -> None:
        event = LoopStepEvent(tokens_used=999)
        assert isinstance(event.tokens_used, int)

    def test_tool_calls_made_is_int(self) -> None:
        event = LoopStepEvent(tool_calls_made=2)
        assert isinstance(event.tool_calls_made, int)

    def test_budget_remaining_usd_is_float(self) -> None:
        event = LoopStepEvent(budget_remaining_usd=1.5)
        assert isinstance(event.budget_remaining_usd, float)

    def test_loop_type_is_str(self) -> None:
        event = LoopStepEvent(loop_type="self_correction")
        assert isinstance(event.loop_type, str)

    def test_termination_reason_is_str(self) -> None:
        event = LoopStepEvent(termination_reason="max_iterations")
        assert isinstance(event.termination_reason, str)


class TestLoopStepEventViaBus:
    def test_emit_and_receive_via_event_bus(self) -> None:
        EventBus._reset_singleton()
        bus = EventBus.get()
        received: list[LoopStepEvent] = []

        unsub = bus.subscribe(LoopStepEvent, received.append)
        try:
            event = LoopStepEvent(
                run_id="run-xyz",
                skill="k8s-skill",
                iteration=1,
                tokens_used=500,
                tool_calls_made=3,
                termination_reason="text_response",
            )
            bus.emit(event)
            assert len(received) == 1
            assert received[0] is event
            assert received[0].event_type == "loop.step"
            assert received[0].run_id == "run-xyz"
            assert received[0].iteration == 1
        finally:
            unsub()
            EventBus._reset_singleton()

    def test_only_loop_step_events_received(self) -> None:
        from vaig.core.events import ApiCalled

        EventBus._reset_singleton()
        bus = EventBus.get()
        loop_received: list[LoopStepEvent] = []

        unsub = bus.subscribe(LoopStepEvent, loop_received.append)
        try:
            bus.emit(ApiCalled(model="gemini", tokens_in=100, tokens_out=50))
            bus.emit(LoopStepEvent(iteration=1))
            assert len(loop_received) == 1
        finally:
            unsub()
            EventBus._reset_singleton()

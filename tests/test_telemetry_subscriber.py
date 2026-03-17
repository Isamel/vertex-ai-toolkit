"""Tests for TelemetrySubscriber — event-to-collector translation."""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock

import pytest

from vaig.core.event_bus import EventBus
from vaig.core.events import (
    ApiCalled,
    BudgetChecked,
    CliCommandTracked,
    ErrorOccurred,
    SessionEnded,
    SessionStarted,
    SkillUsed,
    ToolExecuted,
)
from vaig.core.subscribers.telemetry_subscriber import TelemetrySubscriber

# ── Fixture: fresh EventBus per test ────────────────────────


@pytest.fixture(autouse=True)
def _fresh_bus() -> Generator[None, None, None]:
    """Reset the EventBus singleton between tests."""
    EventBus._reset_singleton()
    yield
    EventBus._reset_singleton()


@pytest.fixture()
def mock_collector() -> MagicMock:
    """A mock TelemetryCollector with all emit methods."""
    collector = MagicMock()
    collector.emit_tool_call = MagicMock()
    collector.emit_api_call = MagicMock()
    collector.emit_error = MagicMock()
    collector.emit_skill_use = MagicMock()
    collector.emit_cli_command = MagicMock()
    collector.emit = MagicMock()
    collector.set_session_id = MagicMock()
    return collector


@pytest.fixture()
def subscriber(mock_collector: MagicMock) -> Generator[TelemetrySubscriber, None, None]:
    """Create a TelemetrySubscriber wired to a mock collector."""
    sub = TelemetrySubscriber(mock_collector)
    yield sub
    sub.unsubscribe_all()


# ══════════════════════════════════════════════════════════════
# Handler Translation Tests
# ══════════════════════════════════════════════════════════════


class TestToolExecutedHandler:
    """ToolExecuted → emit_tool_call()."""

    def test_basic_tool_executed(
        self, subscriber: TelemetrySubscriber, mock_collector: MagicMock
    ) -> None:
        bus = EventBus.get()
        bus.emit(ToolExecuted(tool_name="kubectl", duration_ms=42.5))

        mock_collector.emit_tool_call.assert_called_once_with(
            "kubectl",
            duration_ms=42.5,
            error_type="",
            error_message="",
        )

    def test_tool_executed_with_error(
        self, subscriber: TelemetrySubscriber, mock_collector: MagicMock
    ) -> None:
        bus = EventBus.get()
        bus.emit(
            ToolExecuted(
                tool_name="helm",
                duration_ms=100.0,
                error=True,
                error_type="TimeoutError",
                error_message="connection timed out",
            )
        )

        mock_collector.emit_tool_call.assert_called_once_with(
            "helm",
            duration_ms=100.0,
            error_type="TimeoutError",
            error_message="connection timed out",
        )


class TestApiCalledHandler:
    """ApiCalled → emit_api_call()."""

    def test_basic_api_called(
        self, subscriber: TelemetrySubscriber, mock_collector: MagicMock
    ) -> None:
        bus = EventBus.get()
        bus.emit(
            ApiCalled(
                model="gemini-2.5-pro",
                tokens_in=100,
                tokens_out=50,
                cost_usd=0.005,
                duration_ms=1200.0,
            )
        )

        mock_collector.emit_api_call.assert_called_once_with(
            "gemini-2.5-pro",
            duration_ms=1200.0,
            tokens_in=100,
            tokens_out=50,
            cost_usd=0.005,
            metadata=None,
        )

    def test_api_called_with_metadata(
        self, subscriber: TelemetrySubscriber, mock_collector: MagicMock
    ) -> None:
        bus = EventBus.get()
        bus.emit(
            ApiCalled(
                model="gemini-2.5-flash",
                tokens_in=50,
                tokens_out=25,
                metadata=(("thinking_tokens", 10), ("grounding", True)),
            )
        )

        mock_collector.emit_api_call.assert_called_once()
        call_kwargs = mock_collector.emit_api_call.call_args
        assert call_kwargs[1]["metadata"] == {"thinking_tokens": 10, "grounding": True}

    def test_api_called_empty_metadata_sends_none(
        self, subscriber: TelemetrySubscriber, mock_collector: MagicMock
    ) -> None:
        bus = EventBus.get()
        bus.emit(ApiCalled(model="gemini-2.5-pro"))

        call_kwargs = mock_collector.emit_api_call.call_args
        assert call_kwargs[1]["metadata"] is None


class TestErrorOccurredHandler:
    """ErrorOccurred → emit_error()."""

    def test_basic_error(
        self, subscriber: TelemetrySubscriber, mock_collector: MagicMock
    ) -> None:
        bus = EventBus.get()
        bus.emit(
            ErrorOccurred(
                error_type="ValueError",
                error_message="bad input",
            )
        )

        mock_collector.emit_error.assert_called_once_with(
            "ValueError",
            "bad input",
            metadata=None,
        )

    def test_error_with_source(
        self, subscriber: TelemetrySubscriber, mock_collector: MagicMock
    ) -> None:
        bus = EventBus.get()
        bus.emit(
            ErrorOccurred(
                error_type="TimeoutError",
                error_message="timed out",
                source="direct_chat",
            )
        )

        mock_collector.emit_error.assert_called_once_with(
            "TimeoutError",
            "timed out",
            metadata={"source": "direct_chat"},
        )

    def test_error_with_source_and_metadata(
        self, subscriber: TelemetrySubscriber, mock_collector: MagicMock
    ) -> None:
        bus = EventBus.get()
        bus.emit(
            ErrorOccurred(
                error_type="RuntimeError",
                error_message="crash",
                source="skill_chat",
                metadata=(("extra", "data"),),
            )
        )

        call_kwargs = mock_collector.emit_error.call_args
        expected_metadata = {"extra": "data", "source": "skill_chat"}
        assert call_kwargs[1]["metadata"] == expected_metadata


class TestSessionStartedHandler:
    """SessionStarted → set_session_id() + emit(session, session-started)."""

    def test_session_started(
        self, subscriber: TelemetrySubscriber, mock_collector: MagicMock
    ) -> None:
        bus = EventBus.get()
        bus.emit(
            SessionStarted(
                session_id="abc-123",
                name="test-session",
                model="gemini-2.5-pro",
                skill="gke_diagnostics",
            )
        )

        mock_collector.set_session_id.assert_called_once_with("abc-123")
        mock_collector.emit.assert_called_once_with(
            event_type="session",
            event_name="session-started",
            metadata={
                "name": "test-session",
                "model": "gemini-2.5-pro",
                "skill": "gke_diagnostics",
            },
        )

    def test_session_started_sets_id_before_emit(
        self, subscriber: TelemetrySubscriber, mock_collector: MagicMock
    ) -> None:
        """set_session_id must be called BEFORE emit so the event inherits it."""
        bus = EventBus.get()
        call_order: list[str] = []
        mock_collector.set_session_id.side_effect = lambda _: call_order.append("set_id")
        mock_collector.emit.side_effect = lambda **_: call_order.append("emit")

        bus.emit(SessionStarted(session_id="xyz-789"))

        assert call_order == ["set_id", "emit"]


class TestSessionEndedHandler:
    """SessionEnded → emit(session, session-ended)."""

    def test_session_ended(
        self, subscriber: TelemetrySubscriber, mock_collector: MagicMock
    ) -> None:
        bus = EventBus.get()
        bus.emit(SessionEnded(session_id="abc-123", duration_ms=60000.0))

        mock_collector.emit.assert_called_once_with(
            event_type="session",
            event_name="session-ended",
            duration_ms=60000.0,
        )


class TestSkillUsedHandler:
    """SkillUsed → emit_skill_use()."""

    def test_skill_used(
        self, subscriber: TelemetrySubscriber, mock_collector: MagicMock
    ) -> None:
        bus = EventBus.get()
        bus.emit(SkillUsed(skill_name="gke_diagnostics", duration_ms=5000.0))

        mock_collector.emit_skill_use.assert_called_once_with(
            "gke_diagnostics",
            duration_ms=5000.0,
        )


class TestCliCommandTrackedHandler:
    """CliCommandTracked → emit_cli_command()."""

    def test_cli_command_tracked(
        self, subscriber: TelemetrySubscriber, mock_collector: MagicMock
    ) -> None:
        bus = EventBus.get()
        bus.emit(CliCommandTracked(command_name="ask", duration_ms=300.0))

        mock_collector.emit_cli_command.assert_called_once_with(
            "ask",
            duration_ms=300.0,
        )


class TestBudgetCheckedHandler:
    """BudgetChecked → emit(budget, budget-checked)."""

    def test_budget_checked(
        self, subscriber: TelemetrySubscriber, mock_collector: MagicMock
    ) -> None:
        bus = EventBus.get()
        bus.emit(
            BudgetChecked(
                status="warning",
                cost_usd=0.45,
                limit_usd=0.50,
                message="Approaching budget limit",
            )
        )

        mock_collector.emit.assert_called_once_with(
            event_type="budget",
            event_name="budget-checked",
            metadata={
                "status": "warning",
                "cost_usd": 0.45,
                "limit_usd": 0.50,
                "message": "Approaching budget limit",
            },
        )


# ══════════════════════════════════════════════════════════════
# Error Isolation
# ══════════════════════════════════════════════════════════════


class TestErrorIsolation:
    """Handler failures must not propagate to the emitter."""

    def test_tool_handler_failure_does_not_propagate(
        self, mock_collector: MagicMock
    ) -> None:
        mock_collector.emit_tool_call.side_effect = RuntimeError("db crash")
        sub = TelemetrySubscriber(mock_collector)

        bus = EventBus.get()
        # Should NOT raise
        bus.emit(ToolExecuted(tool_name="kubectl"))

        mock_collector.emit_tool_call.assert_called_once()
        sub.unsubscribe_all()

    def test_api_handler_failure_does_not_propagate(
        self, mock_collector: MagicMock
    ) -> None:
        mock_collector.emit_api_call.side_effect = RuntimeError("db crash")
        sub = TelemetrySubscriber(mock_collector)

        bus = EventBus.get()
        bus.emit(ApiCalled(model="gemini-2.5-pro"))

        mock_collector.emit_api_call.assert_called_once()
        sub.unsubscribe_all()

    def test_session_started_handler_failure_does_not_propagate(
        self, mock_collector: MagicMock
    ) -> None:
        mock_collector.set_session_id.side_effect = RuntimeError("boom")
        sub = TelemetrySubscriber(mock_collector)

        bus = EventBus.get()
        bus.emit(SessionStarted(session_id="abc"))

        mock_collector.set_session_id.assert_called_once()
        sub.unsubscribe_all()

    def test_error_handler_failure_does_not_propagate(
        self, mock_collector: MagicMock
    ) -> None:
        mock_collector.emit_error.side_effect = RuntimeError("meta crash")
        sub = TelemetrySubscriber(mock_collector)

        bus = EventBus.get()
        bus.emit(ErrorOccurred(error_type="X", error_message="y"))

        mock_collector.emit_error.assert_called_once()
        sub.unsubscribe_all()

    def test_all_handlers_fail_without_crashing(
        self, mock_collector: MagicMock
    ) -> None:
        """Every single handler is broken but emit() never raises."""
        mock_collector.emit_tool_call.side_effect = RuntimeError("fail")
        mock_collector.emit_api_call.side_effect = RuntimeError("fail")
        mock_collector.emit_error.side_effect = RuntimeError("fail")
        mock_collector.set_session_id.side_effect = RuntimeError("fail")
        mock_collector.emit.side_effect = RuntimeError("fail")
        mock_collector.emit_skill_use.side_effect = RuntimeError("fail")
        mock_collector.emit_cli_command.side_effect = RuntimeError("fail")

        sub = TelemetrySubscriber(mock_collector)
        bus = EventBus.get()

        # All 8 events — none should raise
        bus.emit(ToolExecuted(tool_name="t"))
        bus.emit(ApiCalled(model="m"))
        bus.emit(ErrorOccurred(error_type="E", error_message="e"))
        bus.emit(SessionStarted(session_id="s"))
        bus.emit(SessionEnded(session_id="s"))
        bus.emit(SkillUsed(skill_name="sk"))
        bus.emit(CliCommandTracked(command_name="c"))
        bus.emit(BudgetChecked(status="ok", cost_usd=0.0, limit_usd=1.0))

        sub.unsubscribe_all()


# ══════════════════════════════════════════════════════════════
# Unsubscribe
# ══════════════════════════════════════════════════════════════


class TestUnsubscribe:
    """Verify unsubscribe_all() cleanly detaches from the event bus."""

    def test_unsubscribe_stops_all_handlers(
        self, subscriber: TelemetrySubscriber, mock_collector: MagicMock
    ) -> None:
        subscriber.unsubscribe_all()

        bus = EventBus.get()
        bus.emit(ToolExecuted(tool_name="kubectl"))
        bus.emit(ApiCalled(model="gemini-2.5-pro"))
        bus.emit(ErrorOccurred(error_type="E", error_message="e"))
        bus.emit(SessionStarted(session_id="s"))
        bus.emit(SessionEnded(session_id="s"))
        bus.emit(SkillUsed(skill_name="sk"))
        bus.emit(CliCommandTracked(command_name="c"))
        bus.emit(BudgetChecked(status="ok", cost_usd=0.0, limit_usd=1.0))

        mock_collector.emit_tool_call.assert_not_called()
        mock_collector.emit_api_call.assert_not_called()
        mock_collector.emit_error.assert_not_called()
        mock_collector.set_session_id.assert_not_called()
        mock_collector.emit.assert_not_called()
        mock_collector.emit_skill_use.assert_not_called()
        mock_collector.emit_cli_command.assert_not_called()

    def test_double_unsubscribe_is_safe(
        self, subscriber: TelemetrySubscriber
    ) -> None:
        subscriber.unsubscribe_all()
        subscriber.unsubscribe_all()  # Should not raise


# ══════════════════════════════════════════════════════════════
# Multiple Events Sequence
# ══════════════════════════════════════════════════════════════


class TestMultipleEvents:
    """Verify correct handling of multiple events in sequence."""

    def test_multiple_tool_events(
        self, subscriber: TelemetrySubscriber, mock_collector: MagicMock
    ) -> None:
        bus = EventBus.get()
        bus.emit(ToolExecuted(tool_name="kubectl"))
        bus.emit(ToolExecuted(tool_name="helm"))
        bus.emit(ToolExecuted(tool_name="gcloud"))

        assert mock_collector.emit_tool_call.call_count == 3
        names = [c[0][0] for c in mock_collector.emit_tool_call.call_args_list]
        assert names == ["kubectl", "helm", "gcloud"]

    def test_mixed_event_types(
        self, subscriber: TelemetrySubscriber, mock_collector: MagicMock
    ) -> None:
        bus = EventBus.get()
        bus.emit(ToolExecuted(tool_name="kubectl"))
        bus.emit(ApiCalled(model="gemini-2.5-pro"))
        bus.emit(ErrorOccurred(error_type="E", error_message="msg"))

        mock_collector.emit_tool_call.assert_called_once()
        mock_collector.emit_api_call.assert_called_once()
        mock_collector.emit_error.assert_called_once()

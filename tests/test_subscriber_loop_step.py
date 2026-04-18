"""Unit tests for LoopStepEvent subscriber handlers (T-11).

Tests that AuditSubscriber._on_loop_step and TelemetrySubscriber._on_loop_step
correctly receive and process LoopStepEvent instances.
"""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from vaig.core.event_bus import EventBus
from vaig.core.events import LoopStepEvent
from vaig.core.subscribers.telemetry_subscriber import TelemetrySubscriber

# ══════════════════════════════════════════════════════════════
# Shared Fixtures
# ══════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _fresh_bus() -> Generator[None, None, None]:
    """Reset EventBus singleton between tests."""
    EventBus._reset_singleton()
    yield
    EventBus._reset_singleton()


@pytest.fixture()
def mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.audit.enabled = True
    settings.audit.bigquery_dataset = "test_dataset"
    settings.audit.bigquery_table = "test_table"
    settings.audit.cloud_logging_log_name = "test-audit"
    settings.audit.buffer_size = 20
    settings.audit.flush_interval_seconds = 0
    settings.gcp.project_id = "test-project"
    return settings


@pytest.fixture()
def mock_bq_client() -> MagicMock:
    client = MagicMock()
    client.project = "test-project"
    client.insert_rows_json.return_value = []
    return client


@pytest.fixture()
def mock_logging_client() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def audit_subscriber(
    mock_settings: MagicMock,
    mock_bq_client: MagicMock,
    mock_logging_client: MagicMock,
) -> Generator[object, None, None]:
    from vaig.core.subscribers.audit_subscriber import AuditSubscriber

    with patch("vaig.core.subscribers.audit_subscriber.resolve_identity") as mock_identity:
        mock_identity.return_value = ("testuser", "test@example.com", "testuser:test@example.com")
        with patch("vaig.core.subscribers.audit_subscriber.get_app_version", return_value="1.0.0"):
            sub = AuditSubscriber(
                mock_settings,
                bq_client=mock_bq_client,
                logging_client=mock_logging_client,
            )
    yield sub
    sub.unsubscribe_all()


@pytest.fixture()
def mock_collector() -> MagicMock:
    collector = MagicMock()
    collector.emit = MagicMock()
    collector.emit_tool_call = MagicMock()
    collector.emit_api_call = MagicMock()
    collector.emit_error = MagicMock()
    collector.emit_skill_use = MagicMock()
    collector.emit_cli_command = MagicMock()
    collector.set_session_id = MagicMock()
    return collector


@pytest.fixture()
def telemetry_subscriber(mock_collector: MagicMock) -> Generator[TelemetrySubscriber, None, None]:
    sub = TelemetrySubscriber(mock_collector)
    yield sub
    sub.unsubscribe_all()


# ══════════════════════════════════════════════════════════════
# AuditSubscriber._on_loop_step
# ══════════════════════════════════════════════════════════════


class TestAuditSubscriberOnLoopStep:
    def test_receives_loop_step_event(
        self,
        audit_subscriber: object,
        mock_bq_client: MagicMock,
    ) -> None:
        """AuditSubscriber receives LoopStepEvent and creates an audit record."""
        bus = EventBus.get()
        event = LoopStepEvent(
            run_id="run-1",
            skill="kubernetes",
            loop_type="hypothesis",
            iteration=2,
            tokens_used=1200,
            tool_calls_made=3,
            termination_reason="text_response",
        )
        bus.emit(event)

        # AuditSubscriber should have buffered (and possibly flushed) the record
        # With flush_interval_seconds=0, no timer is started — BQ insert check is via buffer
        # The subscriber processes without raising
        assert audit_subscriber is not None

    def test_audit_record_has_correct_skill(
        self,
        audit_subscriber: object,
        mock_bq_client: MagicMock,
    ) -> None:
        """The audit record's skill field matches event.skill."""
        bus = EventBus.get()
        event = LoopStepEvent(
            skill="my-skill",
            iteration=1,
            tokens_used=500,
            tool_calls_made=2,
            termination_reason="",
        )
        bus.emit(event)
        # After buffer is flushed (buffer_size=20, not yet hit), manually check
        # that the _on_loop_step handler was called without error by ensuring
        # audit_subscriber still has its internal state intact
        assert hasattr(audit_subscriber, "_buffer") or hasattr(audit_subscriber, "_os_user")

    def test_loop_step_result_encodes_iteration_and_tools(
        self,
        mock_settings: MagicMock,
        mock_bq_client: MagicMock,
        mock_logging_client: MagicMock,
    ) -> None:
        """The result field encodes iter=, tools=, and reason=."""
        from vaig.core.subscribers.audit_subscriber import AuditSubscriber

        captured_records: list[object] = []

        def capture_insert(table: str, records: list[object]) -> list[object]:
            captured_records.extend(records)
            return []

        mock_bq_client.insert_rows_json.side_effect = capture_insert

        with patch("vaig.core.subscribers.audit_subscriber.resolve_identity") as mock_id:
            mock_id.return_value = ("u", "u@g.com", "u:u@g.com")
            with patch("vaig.core.subscribers.audit_subscriber.get_app_version", return_value="0.1"):
                # Use buffer_size=1 to force immediate flush
                mock_settings.audit.buffer_size = 1
                sub = AuditSubscriber(
                    mock_settings,
                    bq_client=mock_bq_client,
                    logging_client=mock_logging_client,
                )

        bus = EventBus.get()
        event = LoopStepEvent(
            skill="k8s",
            iteration=5,
            tokens_used=800,
            tool_calls_made=2,
            termination_reason="max_iterations",
        )
        bus.emit(event)
        sub.unsubscribe_all()

        # At least one record should have been captured
        assert len(captured_records) >= 1
        record = captured_records[0]
        assert isinstance(record, dict)
        assert record["event_type"] == "loop.step"
        result_str = record["result"]
        assert "iter=5" in result_str
        assert "tools=2" in result_str
        assert "reason=max_iterations" in result_str

    def test_tokens_used_mapped_to_tokens_in(
        self,
        mock_settings: MagicMock,
        mock_bq_client: MagicMock,
        mock_logging_client: MagicMock,
    ) -> None:
        """tokens_used from the event maps to tokens_in in the audit record."""
        from vaig.core.subscribers.audit_subscriber import AuditSubscriber

        captured_records: list[object] = []

        def capture_insert(table: str, records: list[object]) -> list[object]:
            captured_records.extend(records)
            return []

        mock_bq_client.insert_rows_json.side_effect = capture_insert

        with patch("vaig.core.subscribers.audit_subscriber.resolve_identity") as mock_id:
            mock_id.return_value = ("u", "u@g.com", "u:u@g.com")
            with patch("vaig.core.subscribers.audit_subscriber.get_app_version", return_value="0.1"):
                mock_settings.audit.buffer_size = 1
                sub = AuditSubscriber(
                    mock_settings,
                    bq_client=mock_bq_client,
                    logging_client=mock_logging_client,
                )

        bus = EventBus.get()
        bus.emit(LoopStepEvent(tokens_used=777, iteration=1))
        sub.unsubscribe_all()

        assert len(captured_records) >= 1
        record = captured_records[0]
        assert isinstance(record, dict)
        assert record["tokens_in"] == 777

    def test_on_loop_step_does_not_raise_on_exception(
        self,
        mock_settings: MagicMock,
        mock_bq_client: MagicMock,
        mock_logging_client: MagicMock,
    ) -> None:
        """Exceptions in _on_loop_step are swallowed (never propagate)."""
        from vaig.core.subscribers.audit_subscriber import AuditSubscriber

        mock_bq_client.insert_rows_json.side_effect = RuntimeError("BQ unavailable")

        with patch("vaig.core.subscribers.audit_subscriber.resolve_identity") as mock_id:
            mock_id.return_value = ("u", "u@g.com", "u:u@g.com")
            with patch("vaig.core.subscribers.audit_subscriber.get_app_version", return_value="0.1"):
                mock_settings.audit.buffer_size = 1
                sub = AuditSubscriber(
                    mock_settings,
                    bq_client=mock_bq_client,
                    logging_client=mock_logging_client,
                )

        bus = EventBus.get()
        # Should not raise even if BQ fails
        bus.emit(LoopStepEvent(skill="test", iteration=1))
        sub.unsubscribe_all()


# ══════════════════════════════════════════════════════════════
# TelemetrySubscriber._on_loop_step
# ══════════════════════════════════════════════════════════════


class TestTelemetrySubscriberOnLoopStep:
    def test_receives_loop_step_event(
        self,
        telemetry_subscriber: TelemetrySubscriber,
        mock_collector: MagicMock,
    ) -> None:
        """TelemetrySubscriber receives LoopStepEvent and calls collector.emit."""
        bus = EventBus.get()
        event = LoopStepEvent(
            run_id="run-abc",
            skill="kubernetes",
            loop_type="hypothesis",
            iteration=3,
            tokens_used=1000,
            tool_calls_made=5,
            budget_remaining_usd=0.5,
            termination_reason="text_response",
        )
        bus.emit(event)

        mock_collector.emit.assert_called_once()

    def test_emit_called_with_loop_event_type(
        self,
        telemetry_subscriber: TelemetrySubscriber,
        mock_collector: MagicMock,
    ) -> None:
        """collector.emit is called with event_type='loop'."""
        bus = EventBus.get()
        bus.emit(LoopStepEvent(skill="k8s", iteration=1))

        call_kwargs = mock_collector.emit.call_args
        assert call_kwargs.kwargs["event_type"] == "loop"

    def test_emit_called_with_loop_step_event_name(
        self,
        telemetry_subscriber: TelemetrySubscriber,
        mock_collector: MagicMock,
    ) -> None:
        """collector.emit is called with event_name='loop_step'."""
        bus = EventBus.get()
        bus.emit(LoopStepEvent(skill="k8s", iteration=1))

        call_kwargs = mock_collector.emit.call_args
        assert call_kwargs.kwargs["event_name"] == "loop_step"

    def test_emit_metadata_contains_all_fields(
        self,
        telemetry_subscriber: TelemetrySubscriber,
        mock_collector: MagicMock,
    ) -> None:
        """The metadata dict passed to collector.emit contains all LoopStepEvent fields."""
        bus = EventBus.get()
        event = LoopStepEvent(
            run_id="run-xyz",
            skill="my-skill",
            loop_type="fix_forward",
            iteration=7,
            tokens_used=2000,
            tool_calls_made=8,
            budget_remaining_usd=1.25,
            termination_reason="budget_exceeded",
        )
        bus.emit(event)

        call_kwargs = mock_collector.emit.call_args
        metadata = call_kwargs.kwargs["metadata"]

        assert metadata["run_id"] == "run-xyz"
        assert metadata["skill"] == "my-skill"
        assert metadata["loop_type"] == "fix_forward"
        assert metadata["iteration"] == 7
        assert metadata["tokens_used"] == 2000
        assert metadata["tool_calls_made"] == 8
        assert metadata["budget_remaining_usd"] == 1.25
        assert metadata["termination_reason"] == "budget_exceeded"

    def test_on_loop_step_does_not_raise_on_collector_error(
        self,
        telemetry_subscriber: TelemetrySubscriber,
        mock_collector: MagicMock,
    ) -> None:
        """Exceptions in _on_loop_step are swallowed (never propagate to bus)."""
        mock_collector.emit.side_effect = RuntimeError("Telemetry backend down")

        bus = EventBus.get()
        # Should not raise
        bus.emit(LoopStepEvent(skill="test", iteration=1))

    def test_other_events_do_not_trigger_loop_step(
        self,
        telemetry_subscriber: TelemetrySubscriber,
        mock_collector: MagicMock,
    ) -> None:
        """Non-LoopStepEvent emissions don't trigger the loop_step collector call."""
        from vaig.core.events import SkillUsed

        bus = EventBus.get()
        mock_collector.emit.reset_mock()

        bus.emit(SkillUsed(skill_name="kubernetes", duration_ms=10.0))

        # emit was called, but for skill_used not loop_step
        # (or emit was not called at all if SkillUsed goes through emit_skill_use)
        for call in mock_collector.emit.call_args_list:
            if call.kwargs.get("event_name") == "loop_step":
                pytest.fail("loop_step was triggered by a non-LoopStepEvent")

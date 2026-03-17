"""Tests for event dataclasses — construction, immutability, auto-timestamp."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from vaig.core.events import (
    ApiCalled,
    BudgetChecked,
    CliCommandTracked,
    ErrorOccurred,
    Event,
    OrchestratorPhaseCompleted,
    OrchestratorToolsCompleted,
    SessionEnded,
    SessionStarted,
    SkillUsed,
    ToolExecuted,
)

# ══════════════════════════════════════════════════════════════
# Base Event
# ══════════════════════════════════════════════════════════════


class TestBaseEvent:
    """Tests for the base Event dataclass."""

    def test_auto_timestamp_is_valid_iso8601_utc(self) -> None:
        evt = Event()
        ts = datetime.fromisoformat(evt.timestamp)
        assert ts.tzinfo is not None  # timezone-aware

    def test_auto_timestamp_is_recent(self) -> None:
        before = datetime.now(UTC)
        evt = Event()
        after = datetime.now(UTC)
        ts = datetime.fromisoformat(evt.timestamp)
        assert before <= ts <= after

    def test_frozen_prevents_mutation(self) -> None:
        evt = Event()
        with pytest.raises(dataclasses.FrozenInstanceError):
            evt.event_type = "hacked"  # type: ignore[misc]

    def test_default_event_type_is_empty(self) -> None:
        evt = Event()
        assert evt.event_type == ""


# ══════════════════════════════════════════════════════════════
# ToolExecuted
# ══════════════════════════════════════════════════════════════


class TestToolExecuted:
    """Tests for the ToolExecuted event."""

    def test_construction_with_required_fields(self) -> None:
        evt = ToolExecuted(tool_name="kubectl")
        assert evt.tool_name == "kubectl"
        assert evt.event_type == "tool.executed"

    def test_defaults(self) -> None:
        evt = ToolExecuted(tool_name="kubectl")
        assert evt.duration_ms == 0.0
        assert evt.args_keys == ()
        assert evt.error is False
        assert evt.error_type == ""
        assert evt.error_message == ""

    def test_auto_timestamp(self) -> None:
        before = datetime.now(UTC)
        evt = ToolExecuted(tool_name="kubectl")
        after = datetime.now(UTC)
        ts = datetime.fromisoformat(evt.timestamp)
        assert before <= ts <= after

    def test_frozen_prevents_mutation(self) -> None:
        evt = ToolExecuted(tool_name="kubectl")
        with pytest.raises(dataclasses.FrozenInstanceError):
            evt.tool_name = "hacked"  # type: ignore[misc]

    def test_error_fields(self) -> None:
        evt = ToolExecuted(
            tool_name="helm",
            error=True,
            error_type="RuntimeError",
            error_message="connection refused",
        )
        assert evt.error is True
        assert evt.error_type == "RuntimeError"
        assert evt.error_message == "connection refused"

    def test_args_keys_tuple(self) -> None:
        evt = ToolExecuted(tool_name="kubectl", args_keys=("namespace", "pod"))
        assert evt.args_keys == ("namespace", "pod")


# ══════════════════════════════════════════════════════════════
# ApiCalled
# ══════════════════════════════════════════════════════════════


class TestApiCalled:
    """Tests for the ApiCalled event."""

    def test_construction_with_required_fields(self) -> None:
        evt = ApiCalled(model="gemini-2.5-pro")
        assert evt.model == "gemini-2.5-pro"
        assert evt.event_type == "api.called"

    def test_defaults(self) -> None:
        evt = ApiCalled(model="gemini-2.5-pro")
        assert evt.tokens_in == 0
        assert evt.tokens_out == 0
        assert evt.cost_usd == 0.0
        assert evt.duration_ms == 0.0
        assert evt.metadata == ()

    def test_auto_timestamp(self) -> None:
        before = datetime.now(UTC)
        evt = ApiCalled(model="gemini-2.5-pro")
        after = datetime.now(UTC)
        ts = datetime.fromisoformat(evt.timestamp)
        assert before <= ts <= after

    def test_frozen_prevents_mutation(self) -> None:
        evt = ApiCalled(model="gemini-2.5-pro")
        with pytest.raises(dataclasses.FrozenInstanceError):
            evt.model = "hacked"  # type: ignore[misc]

    def test_full_construction(self) -> None:
        evt = ApiCalled(
            model="gemini-2.5-pro",
            tokens_in=1000,
            tokens_out=500,
            cost_usd=0.005,
            duration_ms=1200.0,
            metadata=(("thinking_tokens", 200),),
        )
        assert evt.tokens_in == 1000
        assert evt.tokens_out == 500
        assert evt.cost_usd == 0.005
        assert evt.metadata == (("thinking_tokens", 200),)


# ══════════════════════════════════════════════════════════════
# ErrorOccurred
# ══════════════════════════════════════════════════════════════


class TestErrorOccurred:
    """Tests for the ErrorOccurred event."""

    def test_construction_with_required_fields(self) -> None:
        evt = ErrorOccurred(error_type="ValueError", error_message="bad input")
        assert evt.error_type == "ValueError"
        assert evt.error_message == "bad input"
        assert evt.event_type == "error.occurred"

    def test_defaults(self) -> None:
        evt = ErrorOccurred(error_type="ValueError", error_message="bad input")
        assert evt.source == ""
        assert evt.metadata == ()

    def test_auto_timestamp(self) -> None:
        before = datetime.now(UTC)
        evt = ErrorOccurred(error_type="ValueError", error_message="bad input")
        after = datetime.now(UTC)
        ts = datetime.fromisoformat(evt.timestamp)
        assert before <= ts <= after

    def test_frozen_prevents_mutation(self) -> None:
        evt = ErrorOccurred(error_type="ValueError", error_message="bad input")
        with pytest.raises(dataclasses.FrozenInstanceError):
            evt.error_type = "hacked"  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════
# SessionStarted
# ══════════════════════════════════════════════════════════════


class TestSessionStarted:
    """Tests for the SessionStarted event."""

    def test_construction_with_required_fields(self) -> None:
        evt = SessionStarted(session_id="abc-123")
        assert evt.session_id == "abc-123"
        assert evt.event_type == "session.started"

    def test_defaults(self) -> None:
        evt = SessionStarted(session_id="abc-123")
        assert evt.name == ""
        assert evt.model == ""
        assert evt.skill == ""

    def test_auto_timestamp(self) -> None:
        before = datetime.now(UTC)
        evt = SessionStarted(session_id="abc-123")
        after = datetime.now(UTC)
        ts = datetime.fromisoformat(evt.timestamp)
        assert before <= ts <= after

    def test_frozen_prevents_mutation(self) -> None:
        evt = SessionStarted(session_id="abc-123")
        with pytest.raises(dataclasses.FrozenInstanceError):
            evt.session_id = "hacked"  # type: ignore[misc]

    def test_full_construction(self) -> None:
        evt = SessionStarted(
            session_id="abc-123",
            name="Debug session",
            model="gemini-2.5-pro",
            skill="infra",
        )
        assert evt.name == "Debug session"
        assert evt.model == "gemini-2.5-pro"
        assert evt.skill == "infra"


# ══════════════════════════════════════════════════════════════
# SessionEnded
# ══════════════════════════════════════════════════════════════


class TestSessionEnded:
    """Tests for the SessionEnded event."""

    def test_construction_with_required_fields(self) -> None:
        evt = SessionEnded(session_id="abc-123")
        assert evt.session_id == "abc-123"
        assert evt.event_type == "session.ended"

    def test_defaults(self) -> None:
        evt = SessionEnded(session_id="abc-123")
        assert evt.duration_ms == 0.0

    def test_auto_timestamp(self) -> None:
        before = datetime.now(UTC)
        evt = SessionEnded(session_id="abc-123")
        after = datetime.now(UTC)
        ts = datetime.fromisoformat(evt.timestamp)
        assert before <= ts <= after

    def test_frozen_prevents_mutation(self) -> None:
        evt = SessionEnded(session_id="abc-123")
        with pytest.raises(dataclasses.FrozenInstanceError):
            evt.session_id = "hacked"  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════
# SkillUsed
# ══════════════════════════════════════════════════════════════


class TestSkillUsed:
    """Tests for the SkillUsed event."""

    def test_construction_with_required_fields(self) -> None:
        evt = SkillUsed(skill_name="infra")
        assert evt.skill_name == "infra"
        assert evt.event_type == "skill.used"

    def test_defaults(self) -> None:
        evt = SkillUsed(skill_name="infra")
        assert evt.duration_ms == 0.0

    def test_auto_timestamp(self) -> None:
        before = datetime.now(UTC)
        evt = SkillUsed(skill_name="infra")
        after = datetime.now(UTC)
        ts = datetime.fromisoformat(evt.timestamp)
        assert before <= ts <= after

    def test_frozen_prevents_mutation(self) -> None:
        evt = SkillUsed(skill_name="infra")
        with pytest.raises(dataclasses.FrozenInstanceError):
            evt.skill_name = "hacked"  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════
# CliCommandTracked
# ══════════════════════════════════════════════════════════════


class TestCliCommandTracked:
    """Tests for the CliCommandTracked event."""

    def test_construction_with_required_fields(self) -> None:
        evt = CliCommandTracked(command_name="ask")
        assert evt.command_name == "ask"
        assert evt.event_type == "cli.command"

    def test_defaults(self) -> None:
        evt = CliCommandTracked(command_name="ask")
        assert evt.duration_ms == 0.0

    def test_auto_timestamp(self) -> None:
        before = datetime.now(UTC)
        evt = CliCommandTracked(command_name="ask")
        after = datetime.now(UTC)
        ts = datetime.fromisoformat(evt.timestamp)
        assert before <= ts <= after

    def test_frozen_prevents_mutation(self) -> None:
        evt = CliCommandTracked(command_name="ask")
        with pytest.raises(dataclasses.FrozenInstanceError):
            evt.command_name = "hacked"  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════
# BudgetChecked
# ══════════════════════════════════════════════════════════════


class TestBudgetChecked:
    """Tests for the BudgetChecked event."""

    def test_construction_with_required_fields(self) -> None:
        evt = BudgetChecked(status="ok", cost_usd=0.50, limit_usd=5.0)
        assert evt.status == "ok"
        assert evt.cost_usd == 0.50
        assert evt.limit_usd == 5.0
        assert evt.event_type == "budget.checked"

    def test_defaults(self) -> None:
        evt = BudgetChecked()
        assert evt.status == ""
        assert evt.cost_usd == 0.0
        assert evt.limit_usd == 0.0
        assert evt.message == ""

    def test_auto_timestamp(self) -> None:
        before = datetime.now(UTC)
        evt = BudgetChecked(status="warning", cost_usd=4.0, limit_usd=5.0)
        after = datetime.now(UTC)
        ts = datetime.fromisoformat(evt.timestamp)
        assert before <= ts <= after

    def test_frozen_prevents_mutation(self) -> None:
        evt = BudgetChecked(status="ok", cost_usd=0.50, limit_usd=5.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            evt.status = "hacked"  # type: ignore[misc]

    def test_exceeded_with_message(self) -> None:
        evt = BudgetChecked(
            status="exceeded",
            cost_usd=6.0,
            limit_usd=5.0,
            message="Budget exceeded: $6.00 >= $5.00",
        )
        assert evt.status == "exceeded"
        assert evt.message == "Budget exceeded: $6.00 >= $5.00"


# ══════════════════════════════════════════════════════════════
# OrchestratorPhaseCompleted
# ══════════════════════════════════════════════════════════════


class TestOrchestratorPhaseCompleted:
    """Tests for the OrchestratorPhaseCompleted event."""

    def test_construction_with_fields(self) -> None:
        evt = OrchestratorPhaseCompleted(
            skill="gke_diagnostics",
            phase="gather",
            strategy="sequential",
            duration_ms=1500.0,
        )
        assert evt.skill == "gke_diagnostics"
        assert evt.phase == "gather"
        assert evt.strategy == "sequential"
        assert evt.duration_ms == 1500.0
        assert evt.event_type == "orchestrator.phase_completed"

    def test_defaults(self) -> None:
        evt = OrchestratorPhaseCompleted()
        assert evt.skill == ""
        assert evt.phase == ""
        assert evt.strategy == ""
        assert evt.duration_ms == 0.0
        assert evt.is_async is False

    def test_async_flag(self) -> None:
        evt = OrchestratorPhaseCompleted(is_async=True)
        assert evt.is_async is True

    def test_auto_timestamp(self) -> None:
        before = datetime.now(UTC)
        evt = OrchestratorPhaseCompleted(skill="infra")
        after = datetime.now(UTC)
        ts = datetime.fromisoformat(evt.timestamp)
        assert before <= ts <= after

    def test_frozen_prevents_mutation(self) -> None:
        evt = OrchestratorPhaseCompleted(skill="infra")
        with pytest.raises(dataclasses.FrozenInstanceError):
            evt.skill = "hacked"  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════
# OrchestratorToolsCompleted
# ══════════════════════════════════════════════════════════════


class TestOrchestratorToolsCompleted:
    """Tests for the OrchestratorToolsCompleted event."""

    def test_construction_with_fields(self) -> None:
        evt = OrchestratorToolsCompleted(
            skill="gke_diagnostics",
            strategy="fanout",
            agents_count=3,
            success=True,
            duration_ms=5000.0,
        )
        assert evt.skill == "gke_diagnostics"
        assert evt.strategy == "fanout"
        assert evt.agents_count == 3
        assert evt.success is True
        assert evt.duration_ms == 5000.0
        assert evt.event_type == "orchestrator.tools_completed"

    def test_defaults(self) -> None:
        evt = OrchestratorToolsCompleted()
        assert evt.skill == ""
        assert evt.strategy == ""
        assert evt.agents_count == 0
        assert evt.success is True
        assert evt.duration_ms == 0.0
        assert evt.is_async is False

    def test_async_flag(self) -> None:
        evt = OrchestratorToolsCompleted(is_async=True)
        assert evt.is_async is True

    def test_auto_timestamp(self) -> None:
        before = datetime.now(UTC)
        evt = OrchestratorToolsCompleted(skill="infra")
        after = datetime.now(UTC)
        ts = datetime.fromisoformat(evt.timestamp)
        assert before <= ts <= after

    def test_frozen_prevents_mutation(self) -> None:
        evt = OrchestratorToolsCompleted(skill="infra")
        with pytest.raises(dataclasses.FrozenInstanceError):
            evt.skill = "hacked"  # type: ignore[misc]

    def test_failure_flag(self) -> None:
        evt = OrchestratorToolsCompleted(success=False)
        assert evt.success is False

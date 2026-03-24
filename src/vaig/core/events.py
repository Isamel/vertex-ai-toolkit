"""Typed event dataclasses for the in-process event bus.

All events are frozen (immutable) dataclasses inheriting from :class:`Event`.
Each concrete event carries a fixed ``event_type`` string identifier and an
auto-populated ISO 8601 UTC timestamp.

This module has **zero** internal imports — only stdlib is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "ApiCalled",
    "BudgetChecked",
    "CliCommandTracked",
    "ContextWindowChecked",
    "ErrorOccurred",
    "Event",
    "OrchestratorPhaseCompleted",
    "OrchestratorToolsCompleted",
    "SessionEnded",
    "SessionStarted",
    "SkillUsed",
    "ToolExecuted",
]


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


# ══════════════════════════════════════════════════════════════
# Base Event
# ══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Event:
    """Base event type — all domain events inherit from this.

    Attributes:
        event_type: Dot-notation identifier (e.g. ``"tool.executed"``).
        timestamp: ISO 8601 UTC string, auto-populated at construction.
    """

    event_type: str = ""
    timestamp: str = field(default_factory=_utc_now_iso)


# ══════════════════════════════════════════════════════════════
# Concrete Events
# ══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ToolExecuted(Event):
    """Emitted after a tool finishes execution (success or failure).

    Attributes:
        tool_name: Name of the tool that was executed.
        duration_ms: Execution wall-clock time in milliseconds.
        args_keys: Top-level argument keys (no values — privacy safe).
        error: Whether the tool execution resulted in an error.
        error_type: Exception class name when ``error`` is ``True``.
        error_message: Human-readable error description.
        cached: Whether the result was served from ``ToolResultCache``.
    """

    event_type: str = field(default="tool.executed", init=False)
    tool_name: str = ""
    duration_ms: float = 0.0
    args_keys: tuple[str, ...] = ()
    error: bool = False
    error_type: str = ""
    error_message: str = ""
    cached: bool = False


@dataclass(frozen=True)
class ApiCalled(Event):
    """Emitted after each Gemini API call completes.

    Attributes:
        model: Model identifier (e.g. ``"gemini-2.5-pro"``).
        tokens_in: Number of input/prompt tokens.
        tokens_out: Number of output/completion tokens.
        cost_usd: Estimated cost for this call in USD.
        duration_ms: Round-trip latency in milliseconds.
        metadata: Arbitrary key-value pairs (e.g. thinking tokens).
    """

    event_type: str = field(default="api.called", init=False)
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    metadata: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class ErrorOccurred(Event):
    """Emitted when a notable error is captured for telemetry.

    Attributes:
        error_type: Exception class name or error category.
        error_message: Human-readable error description.
        source: Module/component that raised the error.
        metadata: Extra context for debugging.
    """

    event_type: str = field(default="error.occurred", init=False)
    error_type: str = ""
    error_message: str = ""
    source: str = ""
    metadata: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class SessionStarted(Event):
    """Emitted when a new interactive session begins.

    Attributes:
        session_id: Unique identifier for the session.
        name: Optional human-readable session name.
        model: Model selected for the session.
        skill: Active skill/persona at session start.
    """

    event_type: str = field(default="session.started", init=False)
    session_id: str = ""
    name: str = ""
    model: str = ""
    skill: str = ""


@dataclass(frozen=True)
class SessionEnded(Event):
    """Emitted when a session terminates (normally or by error).

    Attributes:
        session_id: Unique identifier for the session.
        duration_ms: Total session wall-clock time in milliseconds.
    """

    event_type: str = field(default="session.ended", init=False)
    session_id: str = ""
    duration_ms: float = 0.0


@dataclass(frozen=True)
class SkillUsed(Event):
    """Emitted when a skill/persona is activated.

    Attributes:
        skill_name: Name of the skill that was used.
        duration_ms: Skill execution time in milliseconds.
    """

    event_type: str = field(default="skill.used", init=False)
    skill_name: str = ""
    duration_ms: float = 0.0


@dataclass(frozen=True)
class CliCommandTracked(Event):
    """Emitted when a CLI command finishes execution.

    Attributes:
        command_name: The CLI command that was run.
        duration_ms: Command execution time in milliseconds.
    """

    event_type: str = field(default="cli.command", init=False)
    command_name: str = ""
    duration_ms: float = 0.0


@dataclass(frozen=True)
class BudgetChecked(Event):
    """Emitted after a budget check is performed.

    Attributes:
        status: Budget check result (``"ok"``, ``"warning"``, ``"exceeded"``).
        cost_usd: Current accumulated cost.
        limit_usd: Configured budget limit.
        message: Human-readable budget status message.
    """

    event_type: str = field(default="budget.checked", init=False)
    status: str = ""
    cost_usd: float = 0.0
    limit_usd: float = 0.0
    message: str = ""


@dataclass(frozen=True)
class OrchestratorPhaseCompleted(Event):
    """Emitted when the orchestrator completes a skill phase execution.

    Covers both sync ``execute_skill_phase`` and async
    ``async_execute_skill_phase``.

    Attributes:
        skill: Name of the skill being executed.
        phase: The skill phase (e.g. ``"gather"``, ``"analyze"``).
        strategy: Execution strategy (``"sequential"`` or ``"fanout"``).
        duration_ms: Execution wall-clock time in milliseconds.
        is_async: Whether the async code path was used.
    """

    event_type: str = field(default="orchestrator.phase.completed", init=False)
    skill: str = ""
    phase: str = ""
    strategy: str = ""
    duration_ms: float = 0.0
    is_async: bool = False


@dataclass(frozen=True)
class OrchestratorToolsCompleted(Event):
    """Emitted when the orchestrator completes a tool-loop execution.

    Covers both sync ``execute_with_tools`` and async
    ``async_execute_with_tools``.

    Attributes:
        skill: Name of the skill being executed.
        strategy: Execution strategy.
        agents_count: Number of agents involved.
        success: Whether the orchestrator run succeeded.
        duration_ms: Execution wall-clock time in milliseconds.
        is_async: Whether the async code path was used.
    """

    event_type: str = field(default="orchestrator.tools.completed", init=False)
    skill: str = ""
    strategy: str = ""
    agents_count: int = 0
    success: bool = True
    duration_ms: float = 0.0
    is_async: bool = False


@dataclass(frozen=True)
class ContextWindowChecked(Event):
    """Emitted after each API call to report context window usage.

    Attributes:
        model: Model identifier (e.g. ``"gemini-2.5-pro"``).
        prompt_tokens: Number of prompt tokens used in this call.
        context_window: Total context window size in tokens.
        context_pct: Percentage of context window consumed (0–100).
        iteration: Current tool-loop iteration number (1-based).
        status: Usage severity — ``"ok"``, ``"warning"``, or ``"error"``.
    """

    event_type: str = field(default="context.checked", init=False)
    model: str = ""
    prompt_tokens: int = 0
    context_window: int = 0
    context_pct: float = 0.0
    iteration: int = 0
    status: str = ""

"""Agent-as-tool factory — wrap a BaseAgent into a ToolDef callable by the LLM.

This module provides :func:`agent_as_tool`, a factory that adapts any
:class:`~vaig.agents.base.BaseAgent` into a :class:`~vaig.tools.base.ToolDef`
so that orchestrator-driven agents can delegate sub-tasks to other agents
via the standard function-calling interface.

Usage::

    from vaig.tools.agent_tool import agent_as_tool

    # Factory pattern (recommended — thread-safe, fresh instance per call)
    mesh_tool = agent_as_tool(
        agent_factory=lambda: create_mesh_agent(),
        state_getter=lambda: current_state,
        current_depth=0,
    )
    registry.register(mesh_tool)

    # Legacy pattern (backward-compatible, shared instance)
    mesh_tool = agent_as_tool(mesh_specialist, state=state, current_depth=0)
    registry.register(mesh_tool)
"""

from __future__ import annotations

import logging
import re
import traceback
from typing import TYPE_CHECKING

from vaig.tools.base import ToolDef, ToolParam, ToolResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from vaig.agents.base import BaseAgent
    from vaig.core.models import PipelineState

logger = logging.getLogger(__name__)

_DEFAULT_DESCRIPTION = "Delegate a sub-task to a specialised agent."


def _sanitize_tool_name(name: str) -> str:
    """Convert an agent name into a valid function-calling tool name.

    Rules:
    - Prefix with ``ask_``
    - Replace any non-alphanumeric characters (hyphens, spaces, dots, etc.)
      with underscores
    - Collapse consecutive underscores
    - Strip leading/trailing underscores from the suffix part

    Examples::

        >>> _sanitize_tool_name("mesh-specialist")
        'ask_mesh_specialist'
        >>> _sanitize_tool_name("My Agent 2.0")
        'ask_My_Agent_2_0'
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return f"ask_{sanitized}"


def _extract_description(agent: BaseAgent) -> str:
    """Extract a short description from the agent's system instruction.

    Returns the first sentence (split on ``.``, ``!``, or ``?``) of the
    system instruction, falling back to :data:`_DEFAULT_DESCRIPTION` when
    the instruction is empty or whitespace-only.
    """
    instruction = (agent.config.system_instruction or "").strip()
    if not instruction:
        return _DEFAULT_DESCRIPTION

    # Take first sentence — split on sentence-ending punctuation
    match = re.split(r"[.!?]", instruction, maxsplit=1)
    first_sentence = match[0].strip()
    return first_sentence if first_sentence else _DEFAULT_DESCRIPTION


def agent_as_tool(
    agent: BaseAgent | None = None,
    *,
    agent_factory: Callable[[], BaseAgent] | None = None,
    state: PipelineState | None = None,
    state_getter: Callable[[], PipelineState | None] | None = None,
    current_depth: int = 0,
    max_depth: int = 2,
    caller_name: str = "",
    agent_name: str = "",
) -> ToolDef:
    """Wrap a :class:`~vaig.agents.base.BaseAgent` into a :class:`ToolDef`.

    Supports two usage patterns:

    **Factory pattern (recommended)**:
    Pass an ``agent_factory`` callable.  Each tool invocation creates a fresh
    agent instance, ensuring thread safety for parallel execution strategies.
    Pass ``state_getter`` to supply the current :class:`~vaig.core.models.PipelineState`
    dynamically at call time (lazy binding — solves state propagation issue).

    **Legacy pattern (backward-compatible)**:
    Pass an ``agent`` instance directly.  State is bound at creation time via
    the ``state`` parameter.  The shared instance is reused across calls (not
    thread-safe for parallel workloads).

    The resulting :class:`ToolDef` exposes a single ``query`` parameter.
    When the LLM invokes it, the closure calls
    ``agent.execute(query, state=current_state)`` and returns the textual content.

    Args:
        agent: The sub-agent to wrap (legacy pattern).  Required when
            *agent_factory* is not provided.
        agent_factory: A zero-argument callable that returns a fresh
            :class:`~vaig.agents.base.BaseAgent` instance each call
            (factory pattern).  Takes precedence over *agent* when both
            are provided.
        state: Optional :class:`~vaig.core.models.PipelineState` forwarded
            to the sub-agent on each call (legacy pattern — bound at creation
            time).  Ignored when *state_getter* is provided.
        state_getter: Optional zero-argument callable that returns the
            **current** :class:`~vaig.core.models.PipelineState` at invocation
            time.  Use this to achieve lazy/dynamic state binding so that
            sub-agents always receive up-to-date pipeline state.  Takes
            precedence over *state* when both are provided.
        current_depth: The current recursion depth in the agent call tree.
            Passed in by the orchestrator or by the calling agent's tool
            wrapper.  Each nested invocation increments this by 1.
        max_depth: Maximum allowed recursion depth (inclusive upper bound
            is ``max_depth - 1``).  When ``current_depth >= max_depth``
            the returned tool immediately returns a depth-exceeded message
            without executing the agent.
        caller_name: Name of the agent that will invoke this tool.  When it
            matches the target agent's name (self-injection), the tool
            returns an error string instead of executing.
        agent_name: Override for the target agent name (used when
            *agent_factory* is provided and the agent name is known
            in advance — e.g. from the config dict).  When empty and
            *agent* is provided, ``agent.name`` is used instead.

    Returns:
        A :class:`ToolDef` whose ``execute`` callable wraps the sub-agent.

    Raises:
        ValueError: When neither *agent* nor *agent_factory* is provided.

    Example::

        # Factory pattern
        tool = agent_as_tool(
            agent_factory=lambda: build_mesh_agent(),
            state_getter=lambda: pipeline.current_state,
            current_depth=1,
            caller_name="orchestrator",
        )
        registry.register(tool)

        # Legacy pattern (backward-compatible)
        tool = agent_as_tool(mesh_specialist, state=state, current_depth=1)
        registry.register(tool)
    """
    if agent is None and agent_factory is None:
        raise ValueError("Either 'agent' or 'agent_factory' must be provided.")

    # Resolve the target agent name for guards and ToolDef metadata.
    # When using the factory pattern, agent_name must be supplied explicitly
    # (the factory hasn't been called yet).  Fall back to agent.name for the
    # legacy pattern.
    _agent_name: str
    if agent_name:
        _agent_name = agent_name
    elif agent is not None:
        _agent_name = agent.name
    else:
        # Factory pattern with no explicit agent_name: create a temporary
        # instance solely to read the name, then discard it.  This is a
        # one-time cost at registration time only.
        _temp = agent_factory()  # type: ignore[misc]
        _agent_name = _temp.name

    tool_name = _sanitize_tool_name(_agent_name)

    # Resolve the tool description.
    # For the legacy pattern we can read from the existing agent instance.
    # For the factory pattern without a pre-existing agent, call the factory
    # once to read config (same temporary instance used above if needed).
    if agent is not None:
        description = _extract_description(agent)
    else:
        # We may have already created _temp above; create it if we haven't.
        # Either way, just call the factory once more — at registration time
        # the cost is acceptable.
        _temp_for_desc = agent_factory()  # type: ignore[misc]
        description = _extract_description(_temp_for_desc)

    # ── Self-injection guard ──────────────────────────────────
    if caller_name and caller_name == _agent_name:
        logger.warning(
            "agent_as_tool: self-injection detected — agent '%s' cannot invoke itself. "
            "Returning a no-op tool.",
            _agent_name,
        )

        def _self_inject_execute(**kwargs: object) -> ToolResult:
            return ToolResult(
                output=f"Sub-agent '{_agent_name}' cannot invoke itself (self-injection prevented).",
                error=True,
            )

        return ToolDef(
            name=tool_name,
            description=description,
            parameters=[
                ToolParam(
                    name="query",
                    type="string",
                    description="The question or sub-task to delegate.",
                    required=True,
                )
            ],
            execute=_self_inject_execute,
            cacheable=False,
        )

    # ── Recursion depth guard ─────────────────────────────────
    if current_depth >= max_depth:
        logger.warning(
            "agent_as_tool: max recursion depth %d reached for agent '%s'. "
            "Returning a depth-exceeded tool.",
            max_depth,
            _agent_name,
        )

        def _depth_exceeded_execute(**kwargs: object) -> ToolResult:
            return ToolResult(
                output=(
                    f"Max recursion depth ({max_depth}) reached. "
                    f"Sub-agent '{_agent_name}' was not invoked."
                ),
                error=True,
            )

        return ToolDef(
            name=tool_name,
            description=description,
            parameters=[
                ToolParam(
                    name="query",
                    type="string",
                    description="The question or sub-task to delegate.",
                    required=True,
                )
            ],
            execute=_depth_exceeded_execute,
            cacheable=False,
        )

    # ── Normal execution wrapper ──────────────────────────────
    # Capture variables needed by the closure.
    _current_depth = current_depth
    _max_depth = max_depth
    _caller_name = caller_name

    def _agent_execute(**kwargs: object) -> ToolResult:
        """Execute the sub-agent and return its textual result.

        Uses lazy binding for state (via state_getter) and creates a fresh
        agent instance per invocation when agent_factory is provided
        (thread-safe for parallel execution strategies).
        """
        query = str(kwargs.get("query", ""))

        # Resolve the agent instance:
        # - Factory pattern: fresh instance per call (thread-safe)
        # - Legacy pattern: shared instance (backward-compatible)
        if agent_factory is not None:
            _exec_agent = agent_factory()
        else:
            _exec_agent = agent  # type: ignore[assignment]

        # Resolve current state lazily:
        # - state_getter: dynamic / late binding (correct for sub-agents)
        # - state: static binding at creation time (legacy fallback)
        if state_getter is not None:
            current_state = state_getter()
        else:
            current_state = state

        # Log depth for traceability
        logger.debug(
            "agent_as_tool: invoking sub-agent '%s' (depth=%d/%d, caller='%s')",
            _agent_name,
            _current_depth + 1,
            _max_depth,
            _caller_name or "(unknown)",
        )

        try:
            result = _exec_agent.execute(query, state=current_state)

            # ── Issue 3: Map AgentResult.success → ToolResult.error ─────
            # When the sub-agent reports failure, surface it as a tool error
            # so the calling agent can react appropriately instead of silently
            # receiving partial/incorrect content.
            if not result.success:
                failure_msg = (
                    f"Sub-agent '{_agent_name}' reported failure: {result.content}"
                )
                logger.warning(
                    "agent_as_tool: sub-agent '%s' returned success=False — "
                    "mapping to ToolResult(error=True). Content: %s",
                    _agent_name,
                    result.content[:200],
                )
                return ToolResult(output=failure_msg, error=True)

            return ToolResult(output=result.content)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            logger.warning(
                "agent_as_tool: sub-agent '%s' raised an exception.\n%s",
                _agent_name,
                tb,
            )
            return ToolResult(
                output=f"Sub-agent '{_agent_name}' failed: {exc}",
                error=True,
            )

    return ToolDef(
        name=tool_name,
        description=description,
        parameters=[
            ToolParam(
                name="query",
                type="string",
                description="The question or sub-task to delegate.",
                required=True,
            )
        ],
        execute=_agent_execute,
        cacheable=False,
    )

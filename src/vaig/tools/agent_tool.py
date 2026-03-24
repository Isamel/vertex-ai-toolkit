"""Agent-as-tool factory — wrap a BaseAgent into a ToolDef callable by the LLM.

This module provides :func:`agent_as_tool`, a factory that adapts any
:class:`~vaig.agents.base.BaseAgent` into a :class:`~vaig.tools.base.ToolDef`
so that orchestrator-driven agents can delegate sub-tasks to other agents
via the standard function-calling interface.

Usage::

    from vaig.tools.agent_tool import agent_as_tool

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
    agent: BaseAgent,
    *,
    state: PipelineState | None = None,
    current_depth: int = 0,
    max_depth: int = 2,
    caller_name: str = "",
) -> ToolDef:
    """Wrap a :class:`~vaig.agents.base.BaseAgent` into a :class:`ToolDef`.

    The resulting :class:`ToolDef` exposes a single ``query`` parameter.
    When the LLM invokes it, the factory's closure calls
    ``agent.execute(query, state=state)`` and returns the textual content.

    Args:
        agent: The sub-agent to wrap.
        state: Optional :class:`~vaig.core.models.PipelineState` forwarded
            to the sub-agent on each call.  The sub-agent may read it but
            should not mutate it.
        current_depth: The current recursion depth in the agent call tree.
            Passed in by the orchestrator or by the calling agent's tool
            wrapper.
        max_depth: Maximum allowed recursion depth (inclusive upper bound
            is ``max_depth - 1``).  When ``current_depth >= max_depth``
            the returned tool immediately returns a depth-exceeded message
            without executing the agent.
        caller_name: Name of the agent that will invoke this tool.  When it
            matches ``agent.name`` (self-injection), the tool returns an
            error string instead of executing.

    Returns:
        A :class:`ToolDef` whose ``execute`` callable wraps the sub-agent.

    Example::

        tool = agent_as_tool(mesh_specialist, state=state, current_depth=1)
        registry.register(tool)
    """
    tool_name = _sanitize_tool_name(agent.name)
    description = _extract_description(agent)
    agent_name = agent.name

    # ── Self-injection guard ──────────────────────────────────
    if caller_name and caller_name == agent_name:
        logger.warning(
            "agent_as_tool: self-injection detected — agent '%s' cannot invoke itself. "
            "Returning a no-op tool.",
            agent_name,
        )

        def _self_inject_execute(**kwargs: object) -> ToolResult:
            return ToolResult(
                output=f"Sub-agent '{agent_name}' cannot invoke itself (self-injection prevented).",
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
            agent_name,
        )

        def _depth_exceeded_execute(**kwargs: object) -> ToolResult:
            return ToolResult(
                output=(
                    f"Max recursion depth ({max_depth}) reached. "
                    f"Sub-agent '{agent_name}' was not invoked."
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

    def _agent_execute(**kwargs: object) -> ToolResult:
        """Execute the sub-agent and return its textual result."""
        query = str(kwargs.get("query", ""))

        try:
            result = agent.execute(query, state=state)
            return ToolResult(output=result.content)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            logger.warning(
                "agent_as_tool: sub-agent '%s' raised an exception.\n%s",
                agent_name,
                tb,
            )
            return ToolResult(
                output=f"Sub-agent '{agent_name}' failed: {exc}",
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

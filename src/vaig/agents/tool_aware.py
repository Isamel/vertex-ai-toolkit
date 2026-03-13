"""Tool-aware agent -- generic, configurable agent with tool-use loop.

Unlike ``InfraAgent`` which hardcodes a system prompt and registers its
own tools, ``ToolAwareAgent`` receives *both* the system prompt and a
pre-configured ``ToolRegistry`` at construction time.  This makes it
the building block for orchestrator-driven multi-agent pipelines where
each agent has a different skill set.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from google.genai import types

from vaig.agents.base import AgentConfig, AgentResult, AgentRole, BaseAgent
from vaig.agents.mixins import ToolLoopMixin
from vaig.core.client import GeminiClient
from vaig.core.exceptions import MaxIterationsError
from vaig.tools.base import ToolRegistry

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


class ToolAwareAgent(BaseAgent, ToolLoopMixin):
    """A generic, reusable agent that wraps Gemini function-calling.

    This is the configurable counterpart to domain-specific agents like
    ``InfraAgent``.  The caller supplies the system prompt, tool registry,
    and model -- the agent handles the tool-use loop mechanics via
    :class:`ToolLoopMixin`.

    Typical usage from an orchestrator::

        registry = ToolRegistry()
        registry.register(some_tool)
        agent = ToolAwareAgent(
            system_prompt="You are a ...",
            tool_registry=registry,
            model="gemini-2.5-pro",
            name="my-agent",
        )
        result = agent.execute("Do something useful")
    """

    def __init__(
        self,
        *,
        system_prompt: str,
        tool_registry: ToolRegistry,
        model: str,
        name: str,
        client: GeminiClient,
        max_iterations: int = 15,
        temperature: float = 0.7,
        max_output_tokens: int = 65536,
    ) -> None:
        """Initialise the tool-aware agent.

        Args:
            system_prompt: System instruction that defines the agent's skill.
            tool_registry: Pre-configured registry of tools available to the agent.
            model: Gemini model ID.
            name: Unique agent name.
            client: The ``GeminiClient`` for API calls.
            max_iterations: Safety cap on tool-use loop iterations.
            temperature: Sampling temperature.
            max_output_tokens: Max output token count.
        """
        config = AgentConfig(
            name=name,
            role=AgentRole.SPECIALIST,
            system_instruction=system_prompt,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        super().__init__(config, client)

        self._tool_registry = tool_registry
        self._max_iterations = max_iterations

        logger.info(
            "ToolAwareAgent '%s' initialized — model=%s, max_iterations=%d, tools=%d",
            name,
            model,
            max_iterations,
            len(tool_registry.list_tools()),
        )

    # ── Factory ──────────────────────────────────────────────

    @classmethod
    def from_config_dict(
        cls,
        config: dict[str, Any],
        model: str,
        tool_registry: ToolRegistry,
        client: GeminiClient,
    ) -> ToolAwareAgent:
        """Create a ``ToolAwareAgent`` from a configuration dictionary.

        Expected keys in *config*:
        - ``name`` (str): Agent name.
        - ``role`` (str): Role description (informational, stored in config).
        - ``system_prompt`` or ``system_instruction`` (str): The system
          instruction for the agent.  ``system_prompt`` takes precedence
          when both are present (for backward compatibility).
        - ``max_iterations`` (int, optional): Override default max iterations.
        - ``temperature`` (float, optional): Override default temperature.
        - ``max_output_tokens`` (int, optional): Override default max tokens.

        Args:
            config: Agent configuration dictionary.
            model: Gemini model ID.
            tool_registry: Pre-configured tool registry.
            client: ``GeminiClient`` instance.

        Returns:
            A fully configured ``ToolAwareAgent``.

        Raises:
            KeyError: If required keys (``name``, ``system_prompt`` or
                ``system_instruction``) are missing.
        """
        return cls(
            system_prompt=config.get("system_prompt") or config["system_instruction"],
            tool_registry=tool_registry,
            model=model,
            name=config["name"],
            client=client,
            max_iterations=config.get("max_iterations", 15),
            temperature=config.get("temperature", 0.7),
            max_output_tokens=config.get("max_output_tokens", 65536),
        )

    # ── Properties ───────────────────────────────────────────

    @property
    def tool_registry(self) -> ToolRegistry:
        """The tool registry for this agent."""
        return self._tool_registry

    @property
    def max_iterations(self) -> int:
        """Maximum tool-use loop iterations."""
        return self._max_iterations

    # ── Execute ──────────────────────────────────────────────

    def execute(self, prompt: str, *, context: str = "") -> AgentResult:
        """Execute a task using the tool-use loop.

        If *context* is provided (e.g. output from an upstream agent in a
        pipeline), it is prepended to the prompt.

        Args:
            prompt: The user query or task.
            context: Optional upstream context.

        Returns:
            ``AgentResult`` with the final text response and metadata.
        """
        full_prompt = self._build_prompt(prompt, context)
        self._add_to_conversation("user", full_prompt)

        history = self._build_chat_history()

        logger.debug(
            "ToolAwareAgent '%s' execute — starting tool loop (max=%d)",
            self.name,
            self._max_iterations,
        )

        try:
            loop_result = self._run_tool_loop(
                client=self._client,
                prompt=full_prompt,
                tool_registry=self._tool_registry,
                system_prompt=self._config.system_instruction,
                history=history,
                max_iterations=self._max_iterations,
                model=self._config.model,
                temperature=self._config.temperature,
                max_output_tokens=self._config.max_output_tokens,
            )
        except MaxIterationsError:
            raise
        except Exception as exc:
            logger.exception("ToolAwareAgent '%s' API call failed", self.name)
            return AgentResult(
                agent_name=self.name,
                content=f"Error during API call: {exc}",
                success=False,
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                metadata={"error": str(exc)},
            )

        self._add_to_conversation("agent", loop_result.text)

        logger.info(
            "ToolAwareAgent '%s' completed — %d iterations, %d tool calls, %s tokens",
            self.name,
            loop_result.iterations,
            len(loop_result.tools_executed),
            loop_result.usage.get("total_tokens", "?"),
        )

        return AgentResult(
            agent_name=self.name,
            content=loop_result.text,
            success=True,
            usage=loop_result.usage,
            metadata={
                "model": loop_result.model,
                "finish_reason": loop_result.finish_reason,
                "iterations": loop_result.iterations,
                "tools_executed": loop_result.tools_executed,
            },
        )

    def execute_stream(self, prompt: str, *, context: str = "") -> Iterator[str]:
        """Execute with streaming -- falls back to non-streaming.

        Tool-use loops are inherently non-streamable because the model
        needs to receive function execution results between turns.
        Falls back to :meth:`execute` and yields the complete result.
        """
        logger.debug(
            "ToolAwareAgent '%s' execute_stream — falling back to non-streaming",
            self.name,
        )
        result = self.execute(prompt, context=context)
        yield result.content

    # ── Internal helpers ─────────────────────────────────────

    def _build_prompt(self, prompt: str, context: str) -> str:
        """Build the full prompt with optional upstream context."""
        if context:
            return f"## Context\n\n{context}\n\n## Task\n\n{prompt}"
        return prompt

    def _build_chat_history(self) -> list[Any]:
        """Convert conversation history to Gemini Content list.

        Works with ``types.Content`` objects directly because the history
        may contain function call / response Parts (not just text).
        """
        contents: list[types.Content] = []
        for msg in self._conversation:
            role = "user" if msg.role == "user" else "model"
            contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]),
            )
        return contents

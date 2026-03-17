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

from vaig.agents.base import AgentConfig, AgentResult, AgentRole, BaseAgent
from vaig.agents.mixins import OnToolCall, ToolLoopMixin
from vaig.core.config import DEFAULT_MAX_OUTPUT_TOKENS
from vaig.core.exceptions import MaxIterationsError
from vaig.tools.base import ToolRegistry

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from vaig.core.cache import ToolResultCache
    from vaig.core.protocols import GeminiClientProtocol
    from vaig.core.tool_call_store import ToolCallStore

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
            system_instruction="You are a ...",
            tool_registry=registry,
            model="gemini-2.5-pro",
            name="my-agent",
        )
        result = agent.execute("Do something useful")
    """

    def __init__(
        self,
        *,
        system_instruction: str,
        tool_registry: ToolRegistry,
        model: str,
        name: str,
        client: GeminiClientProtocol,
        max_iterations: int = 15,
        temperature: float = 0.7,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        frequency_penalty: float | None = None,
    ) -> None:
        """Initialise the tool-aware agent.

        Args:
            system_instruction: System instruction that defines the agent's skill.
            tool_registry: Pre-configured registry of tools available to the agent.
            model: Gemini model ID.
            name: Unique agent name.
            client: The ``GeminiClientProtocol`` for API calls.
            max_iterations: Safety cap on tool-use loop iterations.
            temperature: Sampling temperature.
            max_output_tokens: Max output token count.
            frequency_penalty: Frequency penalty for reducing repetitive output.
                ``None`` falls back to the ``ToolLoopMixin`` default (0.15).
        """
        config = AgentConfig(
            name=name,
            role=AgentRole.SPECIALIST,
            system_instruction=system_instruction,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            frequency_penalty=frequency_penalty,
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
        client: GeminiClientProtocol,
    ) -> ToolAwareAgent:
        """Create a ``ToolAwareAgent`` from a configuration dictionary.

        Expected keys in *config*:
        - ``name`` (str): Agent name.
        - ``role`` (str): Role description (informational, stored in config).
        - ``system_instruction`` (str): The system instruction for the
          agent.  ``system_prompt`` is accepted as a fallback alias for
          backward compatibility but ``system_instruction`` takes precedence.
        - ``max_iterations`` (int, optional): Override default max iterations.
        - ``temperature`` (float, optional): Override default temperature.
        - ``max_output_tokens`` (int, optional): Override default max tokens.
        - ``frequency_penalty`` (float, optional): Frequency penalty for the
          model.  ``None`` (default) falls back to ``ToolLoopMixin`` default.

        Args:
            config: Agent configuration dictionary.
            model: Gemini model ID.
            tool_registry: Pre-configured tool registry.
            client: ``GeminiClientProtocol`` instance.

        Returns:
            A fully configured ``ToolAwareAgent``.

        Raises:
            KeyError: If required keys (``name``, ``system_instruction``)
                are missing.
        """
        return cls(
            system_instruction=config.get("system_instruction") or config.get("system_prompt", ""),
            tool_registry=tool_registry,
            model=model,
            name=config["name"],
            client=client,
            max_iterations=config.get("max_iterations", 15),
            temperature=config.get("temperature", 0.7),
            max_output_tokens=config.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS),
            frequency_penalty=config.get("frequency_penalty"),
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

    def execute(
        self,
        prompt: str,
        *,
        context: str = "",
        on_tool_call: OnToolCall | None = None,
        tool_call_store: ToolCallStore | None = None,
        tool_result_cache: ToolResultCache | None = None,
    ) -> AgentResult:
        """Execute a task using the tool-use loop.

        If *context* is provided (e.g. output from an upstream agent in a
        pipeline), it is prepended to the prompt.

        Args:
            prompt: The user query or task.
            context: Optional upstream context.
            on_tool_call: Optional callback invoked after each tool
                execution with ``(tool_name, tool_args, duration_secs,
                success)``.
            tool_call_store: Optional store for recording full tool call
                results for metrics and feedback.
            tool_result_cache: Optional cache for deduplicating identical
                tool calls within and across orchestrator passes.

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
            # Build optional kwargs — only pass frequency_penalty when
            # explicitly configured to avoid overriding the mixin default.
            loop_kwargs: dict[str, Any] = {}
            if self._config.frequency_penalty is not None:
                loop_kwargs["frequency_penalty"] = self._config.frequency_penalty

            loop_result = self._run_tool_loop(
                client=self._client,
                prompt=full_prompt,
                tool_registry=self._tool_registry,
                system_instruction=self._config.system_instruction,
                history=history,
                max_iterations=self._max_iterations,
                model=self._config.model,
                temperature=self._config.temperature,
                max_output_tokens=self._config.max_output_tokens,
                on_tool_call=on_tool_call,
                agent_name=self.name,
                tool_call_store=tool_call_store,
                tool_result_cache=tool_result_cache,
                **loop_kwargs,
            )
        except MaxIterationsError:
            raise
        except Exception as exc:
            logger.exception("ToolAwareAgent '%s' API call failed", self.name)
            return AgentResult(
                agent_name=self.name,
                content=f"Error during API call: {self.sanitize_error_for_agent(exc)}",
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

    # ── Async methods ────────────────────────────────────────

    async def async_execute(
        self,
        prompt: str,
        *,
        context: str = "",
        on_tool_call: OnToolCall | None = None,
        tool_call_store: ToolCallStore | None = None,
        tool_result_cache: ToolResultCache | None = None,
    ) -> AgentResult:
        """Execute a task using the async tool-use loop.

        Async version of :meth:`execute`.  Delegates to
        :meth:`ToolLoopMixin._async_run_tool_loop` for non-blocking LLM
        calls and tool execution.

        Args:
            prompt: The user query or task.
            context: Optional upstream context.
            on_tool_call: Optional callback invoked after each tool
                execution with ``(tool_name, tool_args, duration_secs,
                success)``.
            tool_call_store: Optional store for recording full tool call
                results for metrics and feedback.
            tool_result_cache: Optional cache for deduplicating identical
                tool calls within and across orchestrator passes.

        Returns:
            ``AgentResult`` with the final text response and metadata.
        """
        full_prompt = self._build_prompt(prompt, context)
        self._add_to_conversation("user", full_prompt)

        history = self._build_chat_history()

        logger.debug(
            "ToolAwareAgent '%s' async_execute — starting async tool loop (max=%d)",
            self.name,
            self._max_iterations,
        )

        try:
            # Build optional kwargs — only pass frequency_penalty when
            # explicitly configured to avoid overriding the mixin default.
            loop_kwargs: dict[str, Any] = {}
            if self._config.frequency_penalty is not None:
                loop_kwargs["frequency_penalty"] = self._config.frequency_penalty

            loop_result = await self._async_run_tool_loop(
                client=self._client,
                prompt=full_prompt,
                tool_registry=self._tool_registry,
                system_instruction=self._config.system_instruction,
                history=history,
                max_iterations=self._max_iterations,
                model=self._config.model,
                temperature=self._config.temperature,
                max_output_tokens=self._config.max_output_tokens,
                on_tool_call=on_tool_call,
                agent_name=self.name,
                tool_call_store=tool_call_store,
                tool_result_cache=tool_result_cache,
                **loop_kwargs,
            )
        except MaxIterationsError:
            raise
        except Exception as exc:
            logger.exception("ToolAwareAgent '%s' async API call failed", self.name)
            return AgentResult(
                agent_name=self.name,
                content=f"Error during API call: {self.sanitize_error_for_agent(exc)}",
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
            "ToolAwareAgent '%s' async completed — %d iterations, %d tool calls, %s tokens",
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

    async def async_execute_stream(
        self, prompt: str, *, context: str = "",
    ) -> AsyncIterator[str]:
        """Async streaming -- falls back to async non-streaming.

        Tool-use loops are inherently non-streamable because the model
        needs to receive function execution results between turns.
        Falls back to :meth:`async_execute` and yields the complete result.
        """
        logger.debug(
            "ToolAwareAgent '%s' async_execute_stream — falling back to async non-streaming",
            self.name,
        )
        result = await self.async_execute(prompt, context=context)
        yield result.content

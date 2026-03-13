"""Tool-loop mixin -- reusable tool-use loop for agents that interact with Gemini function calling."""

from __future__ import annotations

import logging
from typing import Any

from google.genai import types

from vaig.core.client import GeminiClient, ToolCallResult
from vaig.core.config import DEFAULT_MAX_OUTPUT_TOKENS
from vaig.core.exceptions import MaxIterationsError
from vaig.tools.base import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


class ToolLoopMixin:
    """Mixin that provides a generic Gemini tool-use loop.

    Any agent that needs function-calling can inherit from this mixin
    to get a ready-made ``_run_tool_loop()`` without duplicating the
    loop mechanics.  The mixin is deliberately *not* coupled to
    ``BaseAgent`` -- it only requires a ``GeminiClient`` instance
    and a ``ToolRegistry``.

    Expected usage::

        class MyAgent(BaseAgent, ToolLoopMixin):
            def execute(self, prompt, *, context=""):
                ...
                return self._run_tool_loop(...)

    The host class can override ``_execute_single_tool`` if it needs
    custom pre/post-processing around tool execution (e.g. confirmation
    dialogs for destructive tools).
    """

    # ── Public loop entry-point ──────────────────────────────

    def _run_tool_loop(
        self,
        *,
        client: GeminiClient,
        prompt: str | list[types.Part],
        tool_registry: ToolRegistry,
        system_prompt: str,
        history: list[Any],
        max_iterations: int = 15,
        model: str | None = None,
        temperature: float = 0.7,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        frequency_penalty: float | None = 0.15,
    ) -> ToolLoopResult:
        """Drive a Gemini tool-use loop until text or max iterations.

        Sends *prompt* (first iteration) then empty prompts on subsequent
        iterations, relying on *history* to carry context.  Returns a
        :class:`ToolLoopResult` with the final text, token usage, and the
        list of tools that were executed.

        Args:
            client: The ``GeminiClient`` for API calls.
            prompt: The initial user prompt (text or Parts).
            tool_registry: Registry containing available tools.
            system_prompt: System instruction for the model.
            history: Mutable list of ``types.Content`` -- updated in-place
                     with function call/response entries as the loop runs.
            max_iterations: Safety cap on loop iterations.
            model: Model ID override (``None`` = client default).
            temperature: Sampling temperature.
            max_output_tokens: Max output token count.
            frequency_penalty: Frequency penalty (``None`` to omit).

        Returns:
            ``ToolLoopResult`` with final text, usage, tool metadata,
            iteration count, model name, and finish reason.

        Raises:
            MaxIterationsError: If the loop exceeds *max_iterations*.
        """
        declarations = tool_registry.to_function_declarations()

        total_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        tools_executed: list[dict[str, Any]] = []
        iteration = 0

        logger.debug(
            "ToolLoopMixin._run_tool_loop() -- starting (max=%d)",
            max_iterations,
        )

        gen_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if frequency_penalty is not None:
            gen_kwargs["frequency_penalty"] = frequency_penalty

        while iteration < max_iterations:
            iteration += 1
            logger.debug("Tool loop iteration %d/%d", iteration, max_iterations)

            # -- Call Gemini with tool declarations -----------------------
            try:
                result: ToolCallResult = client.generate_with_tools(
                    prompt if iteration == 1 else [],
                    tool_declarations=declarations,
                    system_instruction=system_prompt,
                    history=history,
                    model_id=model,
                    **gen_kwargs,
                )
            except Exception:
                logger.exception(
                    "ToolLoopMixin API call failed on iteration %d", iteration,
                )
                raise

            # -- Accumulate token usage -----------------------------------
            for key in total_usage:
                total_usage[key] += result.usage.get(key, 0)

            # -- Case 1: text response (no function calls) -- done --------
            if not result.function_calls:
                logger.debug(
                    "Tool loop completed -- %d iterations, %d tool calls",
                    iteration,
                    len(tools_executed),
                )
                return ToolLoopResult(
                    text=result.text,
                    usage=total_usage,
                    tools_executed=tools_executed,
                    iterations=iteration,
                    model=result.model,
                    finish_reason=result.finish_reason,
                )

            # -- Case 2: function calls -- execute and continue -----------
            history.append(
                self._build_function_call_content(result.function_calls),
            )

            function_responses: list[dict[str, Any]] = []

            for fc in result.function_calls:
                tool_name = fc["name"]
                tool_args = fc["args"]

                tool_result = self._execute_single_tool(
                    tool_registry, tool_name, tool_args,
                )

                tools_executed.append({
                    "name": tool_name,
                    "args": tool_args,
                    "output": tool_result.output[:200],  # Truncate for metadata
                    "error": tool_result.error,
                })

                function_responses.append({
                    "name": tool_name,
                    "response": {
                        "output": tool_result.output,
                        "error": tool_result.error,
                    },
                })

            # Add function responses to history for next turn
            response_parts = GeminiClient.build_function_response_parts(
                function_responses,
            )
            history.append(types.Content(role="user", parts=response_parts))

        # -- Max iterations exceeded --------------------------------------
        msg = (
            f"Tool-use loop exceeded maximum iterations ({max_iterations}). "
            f"Executed {len(tools_executed)} tool calls."
        )
        logger.warning(msg)
        raise MaxIterationsError(msg, iterations=max_iterations)

    # ── Tool execution (overridable) ─────────────────────────

    def _execute_single_tool(
        self,
        tool_registry: ToolRegistry,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> ToolResult:
        """Execute a single tool call with error handling.

        Subclasses can override this to add pre/post-processing such as
        confirmation dialogs, audit logging, etc.

        Never raises -- returns a ``ToolResult`` with ``error=True``
        on failure so the model can see errors and self-correct.
        """
        tool = tool_registry.get(tool_name)

        if tool is None:
            logger.warning("Unknown tool requested: %s", tool_name)
            return ToolResult(
                output=f"Unknown tool: {tool_name}. Available tools: "
                f"{', '.join(t.name for t in tool_registry.list_tools())}",
                error=True,
            )

        try:
            logger.debug("Executing tool: %s(%s)", tool_name, tool_args)
            result = tool.execute(**tool_args)
            logger.debug(
                "Tool %s result: error=%s, output_len=%d",
                tool_name,
                result.error,
                len(result.output),
            )
            return result
        except TypeError as exc:
            logger.warning("Tool %s type error: %s", tool_name, exc)
            return ToolResult(
                output=f"Invalid arguments for {tool_name}: {exc}. "
                f"Expected parameters: {', '.join(p.name for p in tool.parameters)}",
                error=True,
            )
        except Exception as exc:
            logger.warning("Tool %s unexpected error: %s", tool_name, exc)
            return ToolResult(
                output=f"Tool execution error ({tool_name}): {exc}",
                error=True,
            )

    # ── Message builders ─────────────────────────────────────

    @staticmethod
    def _build_function_call_content(
        function_calls: list[dict[str, Any]],
    ) -> types.Content:
        """Build a ``types.Content`` from a list of function call dicts.

        Each dict must have ``"name"`` (str) and ``"args"`` (dict).
        Returns a ``Content(role="model", parts=[...])`` suitable for
        appending to the conversation history.
        """
        fc_parts: list[types.Part] = []
        for fc in function_calls:
            fc_parts.append(
                types.Part.from_function_call(
                    name=fc["name"],
                    args=fc["args"],
                ),
            )
        return types.Content(role="model", parts=fc_parts)


class ToolLoopResult:
    """Result produced by ``ToolLoopMixin._run_tool_loop``.

    Carries the raw text, token usage, executed tool metadata,
    iteration count, model name, and finish reason.  The host agent
    is responsible for wrapping this into an ``AgentResult``.
    """

    __slots__ = (
        "text",
        "usage",
        "tools_executed",
        "iterations",
        "model",
        "finish_reason",
    )

    def __init__(
        self,
        *,
        text: str,
        usage: dict[str, int],
        tools_executed: list[dict[str, Any]],
        iterations: int,
        model: str,
        finish_reason: str,
    ) -> None:
        self.text = text
        self.usage = usage
        self.tools_executed = tools_executed
        self.iterations = iterations
        self.model = model
        self.finish_reason = finish_reason

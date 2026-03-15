"""Tool-loop mixin -- reusable tool-use loop for agents that interact with Gemini function calling."""

from __future__ import annotations

import inspect
import logging
import time
from typing import Any, Protocol

from google.genai import types

from vaig.core.async_utils import to_async
from vaig.core.client import GeminiClient, ToolCallResult
from vaig.core.config import DEFAULT_MAX_OUTPUT_TOKENS
from vaig.core.exceptions import MaxIterationsError
from vaig.tools.base import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


class OnToolCall(Protocol):
    """Callback protocol invoked after each tool execution.

    Args:
        tool_name: Name of the tool that was executed.
        tool_args: Arguments passed to the tool.
        duration: Wall-clock seconds the tool took.
        success: ``True`` when the tool returned without error.
    """

    def __call__(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        duration: float,
        success: bool,
    ) -> None: ...


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
        system_instruction: str,
        history: list[Any],
        max_iterations: int = 15,
        model: str | None = None,
        temperature: float = 0.7,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        frequency_penalty: float | None = 0.15,
        on_tool_call: OnToolCall | None = None,
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
            system_instruction: System instruction for the model.
            history: Mutable list of ``types.Content`` -- updated in-place
                     with function call/response entries as the loop runs.
            max_iterations: Safety cap on loop iterations.
            model: Model ID override (``None`` = client default).
            temperature: Sampling temperature.
            max_output_tokens: Max output token count.
            frequency_penalty: Frequency penalty (``None`` to omit).
            on_tool_call: Optional callback invoked after each tool
                execution with ``(tool_name, tool_args, duration_secs,
                success)``.  Useful for live progress logging.

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
                    system_instruction=system_instruction,
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

                t_tool = time.perf_counter()
                tool_result = self._execute_single_tool(
                    tool_registry, tool_name, tool_args,
                )
                tool_duration = time.perf_counter() - t_tool

                # Notify caller about this tool execution
                if on_tool_call is not None:
                    try:
                        on_tool_call(tool_name, tool_args, tool_duration, not tool_result.error)
                    except Exception:  # noqa: BLE001
                        logger.debug("on_tool_call callback raised; ignoring")

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
        t0 = time.perf_counter()
        tool = tool_registry.get(tool_name)

        if tool is None:
            logger.warning("Unknown tool requested: %s", tool_name)
            result = ToolResult(
                output=f"Unknown tool: {tool_name}. Available tools: "
                f"{', '.join(t.name for t in tool_registry.list_tools())}",
                error=True,
            )
            self._emit_tool_telemetry(tool_name, tool_args, result, t0, error_type="UnknownTool")
            return result

        try:
            logger.debug("Executing tool: %s(%s)", tool_name, tool_args)
            result = tool.execute(**tool_args)
            logger.debug(
                "Tool %s result: error=%s, output_len=%d",
                tool_name,
                result.error,
                len(result.output),
            )
            self._emit_tool_telemetry(tool_name, tool_args, result, t0)
            return result
        except TypeError as exc:
            logger.warning("Tool %s type error: %s", tool_name, exc)
            result = ToolResult(
                output=f"Invalid arguments for {tool_name}: {exc}. "
                f"Expected parameters: {', '.join(p.name for p in tool.parameters)}",
                error=True,
            )
            self._emit_tool_telemetry(tool_name, tool_args, result, t0, error_type="TypeError", error_message=str(exc))
            return result
        except Exception as exc:
            logger.warning("Tool %s unexpected error: %s", tool_name, exc)
            result = ToolResult(
                output=f"Tool execution error ({tool_name}): {exc}",
                error=True,
            )
            self._emit_tool_telemetry(tool_name, tool_args, result, t0, error_type=type(exc).__name__, error_message=str(exc))
            return result

    @staticmethod
    def _emit_tool_telemetry(
        tool_name: str,
        tool_args: dict[str, Any],
        result: ToolResult,
        t0: float,
        *,
        error_type: str = "",
        error_message: str = "",
    ) -> None:
        """Emit a tool_call telemetry event. Never raises."""
        try:
            from vaig.core.telemetry import get_telemetry_collector

            duration_ms = (time.perf_counter() - t0) * 1000
            collector = get_telemetry_collector()
            collector.emit_tool_call(
                tool_name,
                duration_ms=duration_ms,
                metadata={"args_keys": sorted(tool_args.keys()), "error": result.error},
                error_type=error_type,
                error_message=error_message,
            )
        except Exception:  # noqa: BLE001
            pass

    # ── Async public loop entry-point ────────────────────────

    async def _async_run_tool_loop(
        self,
        *,
        client: GeminiClient,
        prompt: str | list[types.Part],
        tool_registry: ToolRegistry,
        system_instruction: str,
        history: list[Any],
        max_iterations: int = 15,
        model: str | None = None,
        temperature: float = 0.7,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        frequency_penalty: float | None = 0.15,
        on_tool_call: OnToolCall | None = None,
    ) -> ToolLoopResult:
        """Async version of :meth:`_run_tool_loop`.

        Uses ``client.async_generate_with_tools()`` for non-blocking LLM calls
        and :meth:`_async_execute_single_tool` for tool execution (which
        automatically wraps sync tools via ``asyncio.to_thread``).

        Same signature and semantics as the sync counterpart.
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
            "ToolLoopMixin._async_run_tool_loop() -- starting (max=%d)",
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
            logger.debug("Async tool loop iteration %d/%d", iteration, max_iterations)

            # -- Call Gemini with tool declarations (async) ----------------
            try:
                result: ToolCallResult = await client.async_generate_with_tools(
                    prompt if iteration == 1 else [],
                    tool_declarations=declarations,
                    system_instruction=system_instruction,
                    history=history,
                    model_id=model,
                    **gen_kwargs,
                )
            except Exception:
                logger.exception(
                    "ToolLoopMixin async API call failed on iteration %d", iteration,
                )
                raise

            # -- Accumulate token usage -----------------------------------
            for key in total_usage:
                total_usage[key] += result.usage.get(key, 0)

            # -- Case 1: text response (no function calls) -- done --------
            if not result.function_calls:
                logger.debug(
                    "Async tool loop completed -- %d iterations, %d tool calls",
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

                t_tool = time.perf_counter()
                tool_result = await self._async_execute_single_tool(
                    tool_registry, tool_name, tool_args,
                )
                tool_duration = time.perf_counter() - t_tool

                # Notify caller about this tool execution
                if on_tool_call is not None:
                    try:
                        on_tool_call(tool_name, tool_args, tool_duration, not tool_result.error)
                    except Exception:  # noqa: BLE001
                        logger.debug("on_tool_call callback raised; ignoring")

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

    # ── Async tool execution (overridable) ────────────────────

    async def _async_execute_single_tool(
        self,
        tool_registry: ToolRegistry,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> ToolResult:
        """Async version of :meth:`_execute_single_tool`.

        If the tool's ``execute`` callable is a coroutine function, it is
        awaited directly.  Otherwise it is wrapped via ``to_async()``
        (runs in ``asyncio.to_thread``) to avoid blocking the event loop.

        Never raises -- returns a ``ToolResult`` with ``error=True``
        on failure so the model can see errors and self-correct.
        """
        t0 = time.perf_counter()
        tool = tool_registry.get(tool_name)

        if tool is None:
            logger.warning("Unknown tool requested: %s", tool_name)
            result = ToolResult(
                output=f"Unknown tool: {tool_name}. Available tools: "
                f"{', '.join(t.name for t in tool_registry.list_tools())}",
                error=True,
            )
            self._emit_tool_telemetry(tool_name, tool_args, result, t0, error_type="UnknownTool")
            return result

        try:
            logger.debug("Async executing tool: %s(%s)", tool_name, tool_args)

            if inspect.iscoroutinefunction(tool.execute):
                # Tool is natively async — await directly
                result = await tool.execute(**tool_args)
            else:
                # Tool is sync — wrap to avoid blocking the event loop
                async_execute = to_async(tool.execute)
                result = await async_execute(**tool_args)

            logger.debug(
                "Tool %s async result: error=%s, output_len=%d",
                tool_name,
                result.error,
                len(result.output),
            )
            self._emit_tool_telemetry(tool_name, tool_args, result, t0)
            return result
        except TypeError as exc:
            logger.warning("Tool %s type error: %s", tool_name, exc)
            result = ToolResult(
                output=f"Invalid arguments for {tool_name}: {exc}. "
                f"Expected parameters: {', '.join(p.name for p in tool.parameters)}",
                error=True,
            )
            self._emit_tool_telemetry(tool_name, tool_args, result, t0, error_type="TypeError", error_message=str(exc))
            return result
        except Exception as exc:
            logger.warning("Tool %s unexpected error: %s", tool_name, exc)
            result = ToolResult(
                output=f"Tool execution error ({tool_name}): {exc}",
                error=True,
            )
            self._emit_tool_telemetry(tool_name, tool_args, result, t0, error_type=type(exc).__name__, error_message=str(exc))
            return result

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

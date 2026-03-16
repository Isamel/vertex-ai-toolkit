"""Tool-loop mixin -- reusable tool-use loop for agents that interact with Gemini function calling."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from datetime import UTC
from typing import TYPE_CHECKING, Any, Protocol

from google.genai import types

from vaig.core.async_utils import to_async
from vaig.core.client import GeminiClient, ToolCallResult
from vaig.core.config import DEFAULT_MAX_OUTPUT_TOKENS
from vaig.core.exceptions import MaxIterationsError
from vaig.tools.base import ToolCallRecord, ToolRegistry, ToolResult

if TYPE_CHECKING:
    from vaig.core.tool_call_store import ToolCallStore

logger = logging.getLogger(__name__)


class OnToolCall(Protocol):
    """Callback protocol invoked after each tool execution.

    Args:
        tool_name: Name of the tool that was executed.
        tool_args: Arguments passed to the tool.
        duration: Wall-clock seconds the tool took.
        success: ``True`` when the tool returned without error.
        error_message: Short error description when ``success`` is ``False``.
            Empty string when the tool succeeded.  Optional for backward
            compatibility — existing callbacks that accept only 4 positional
            args will continue to work.
    """

    def __call__(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        duration: float,
        success: bool,
        error_message: str = "",
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
        agent_name: str = "",
        tool_call_store: ToolCallStore | None = None,
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
                    "ToolLoopMixin API call failed on iteration %d",
                    iteration,
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
            # Sync path: sequential execution. Use async path for parallel tool calls.
            history.append(
                self._build_function_call_content(result.function_calls),
            )

            function_responses: list[dict[str, Any]] = []

            for fc in result.function_calls:
                tool_name = fc["name"]
                tool_args = fc["args"]

                t_tool = time.perf_counter()
                tool_result = self._execute_single_tool(
                    tool_registry,
                    tool_name,
                    tool_args,
                )
                tool_duration = time.perf_counter() - t_tool

                # Notify caller about this tool execution
                self._notify_tool_call(
                    on_tool_call,
                    tool_name,
                    tool_args,
                    tool_duration,
                    tool_result,
                )

                # Record tool call for metrics/feedback storage
                self._record_tool_call(
                    tool_call_store,
                    tool_name,
                    tool_args,
                    tool_result,
                    tool_duration,
                    agent_name,
                    iteration,
                )

                tools_executed.append(
                    {
                        "name": tool_name,
                        "args": tool_args,
                        "output": (tool_result.output or "")[:200],
                        "error": tool_result.error,
                    }
                )

                function_responses.append(
                    {
                        "name": tool_name,
                        "response": {
                            "output": tool_result.output,
                            "error": tool_result.error,
                        },
                    }
                )

            # Add function responses to history for next turn
            response_parts = GeminiClient.build_function_response_parts(
                function_responses,
            )
            history.append(types.Content(role="user", parts=response_parts))

        # -- Max iterations exceeded --------------------------------------
        msg = (
            f"Tool-use loop exceeded maximum iterations ({max_iterations}). Executed {len(tools_executed)} tool calls."
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
            # Guard against tools returning None output
            if result.output is None:
                result = ToolResult(output="(no output)", error=result.error)
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
            self._emit_tool_telemetry(
                tool_name, tool_args, result, t0, error_type=type(exc).__name__, error_message=str(exc)
            )
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
        agent_name: str = "",
        tool_call_store: ToolCallStore | None = None,
        parallel_tool_calls: bool = True,
        max_concurrent_tool_calls: int = 5,
    ) -> ToolLoopResult:
        """Async version of :meth:`_run_tool_loop`.

        Uses ``client.async_generate_with_tools()`` for non-blocking LLM calls
        and :meth:`_async_execute_single_tool` for tool execution (which
        automatically wraps sync tools via ``asyncio.to_thread``).

        When *parallel_tool_calls* is ``True`` (default) and Gemini returns
        multiple function calls in a single response, they are executed
        concurrently via ``asyncio.gather`` with a semaphore limit of
        *max_concurrent_tool_calls*.  Single function calls always use the
        sequential path to avoid gather overhead.

        Same signature and semantics as the sync counterpart, with the
        addition of ``parallel_tool_calls`` and ``max_concurrent_tool_calls``.
        """
        declarations = tool_registry.to_function_declarations()
        semaphore = asyncio.Semaphore(max_concurrent_tool_calls)

        total_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        tools_executed: list[dict[str, Any]] = []
        iteration = 0

        logger.debug(
            "ToolLoopMixin._async_run_tool_loop() -- starting (max=%d, parallel=%s, max_concurrent=%d)",
            max_iterations,
            parallel_tool_calls,
            max_concurrent_tool_calls,
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
                    "ToolLoopMixin async API call failed on iteration %d",
                    iteration,
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
            num_calls = len(result.function_calls)
            use_parallel = parallel_tool_calls and num_calls > 1

            if use_parallel:
                # ── Parallel execution via asyncio.gather ────────────
                logger.info(
                    "Executing %d tool calls in parallel (max_concurrent=%d)",
                    num_calls,
                    max_concurrent_tool_calls,
                )
                t_parallel_start = time.perf_counter()

                async def _run_with_semaphore(
                    fc: dict[str, Any],
                ) -> tuple[str, dict[str, Any], float, ToolResult]:
                    """Execute a single tool call under the semaphore."""
                    async with semaphore:
                        t0 = time.perf_counter()
                        res = await self._async_execute_single_tool(
                            tool_registry,
                            fc["name"],
                            fc["args"],
                        )
                        dur = time.perf_counter() - t0
                        return fc["name"], fc["args"], dur, res

                tasks = [_run_with_semaphore(fc) for fc in result.function_calls]
                gather_results = await asyncio.gather(
                    *tasks,
                    return_exceptions=True,
                )

                t_parallel_total = time.perf_counter() - t_parallel_start
                sequential_estimate = 0.0

                for i, res in enumerate(gather_results):
                    fc = result.function_calls[i]
                    if isinstance(res, BaseException):
                        # Tool raised an unhandled exception — create error response
                        logger.warning(
                            "Parallel tool call %s failed: %s",
                            fc["name"],
                            res,
                        )
                        tool_name = fc["name"]
                        tool_args = fc["args"]
                        tool_result = ToolResult(
                            output=f"Tool execution error ({tool_name}): {res}",
                            error=True,
                        )
                        tool_duration = 0.0
                    else:
                        tool_name, tool_args, tool_duration, tool_result = res

                    sequential_estimate += tool_duration

                    # Notify caller about this tool execution
                    self._notify_tool_call(
                        on_tool_call,
                        tool_name,
                        tool_args,
                        tool_duration,
                        tool_result,
                    )

                    # Record tool call for metrics/feedback storage
                    self._record_tool_call(
                        tool_call_store,
                        tool_name,
                        tool_args,
                        tool_result,
                        tool_duration,
                        agent_name,
                        iteration,
                    )

                    tools_executed.append(
                        {
                            "name": tool_name,
                            "args": tool_args,
                            "output": (tool_result.output or "")[:200],
                            "error": tool_result.error,
                        }
                    )

                    function_responses.append(
                        {
                            "name": tool_name,
                            "response": {
                                "output": tool_result.output,
                                "error": tool_result.error,
                            },
                        }
                    )

                logger.info(
                    "Parallel execution completed in %.2fs (vs estimated %.2fs sequential)",
                    t_parallel_total,
                    sequential_estimate,
                )
            else:
                # ── Sequential execution (single call or parallel disabled) ──
                for fc in result.function_calls:
                    tool_name = fc["name"]
                    tool_args = fc["args"]

                    t_tool = time.perf_counter()
                    tool_result = await self._async_execute_single_tool(
                        tool_registry,
                        tool_name,
                        tool_args,
                    )
                    tool_duration = time.perf_counter() - t_tool

                    # Notify caller about this tool execution
                    self._notify_tool_call(
                        on_tool_call,
                        tool_name,
                        tool_args,
                        tool_duration,
                        tool_result,
                    )

                    # Record tool call for metrics/feedback storage
                    self._record_tool_call(
                        tool_call_store,
                        tool_name,
                        tool_args,
                        tool_result,
                        tool_duration,
                        agent_name,
                        iteration,
                    )

                    tools_executed.append(
                        {
                            "name": tool_name,
                            "args": tool_args,
                            "output": (tool_result.output or "")[:200],
                            "error": tool_result.error,
                        }
                    )

                    function_responses.append(
                        {
                            "name": tool_name,
                            "response": {
                                "output": tool_result.output,
                                "error": tool_result.error,
                            },
                        }
                    )

            # Add function responses to history for next turn
            response_parts = GeminiClient.build_function_response_parts(
                function_responses,
            )
            history.append(types.Content(role="user", parts=response_parts))

        # -- Max iterations exceeded --------------------------------------
        msg = (
            f"Tool-use loop exceeded maximum iterations ({max_iterations}). Executed {len(tools_executed)} tool calls."
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

            # Guard against tools returning None output
            if result.output is None:
                result = ToolResult(output="(no output)", error=result.error)

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
            self._emit_tool_telemetry(
                tool_name, tool_args, result, t0, error_type=type(exc).__name__, error_message=str(exc)
            )
            return result

    # ── Shared helpers for tool call callbacks/recording ────────

    @staticmethod
    def _notify_tool_call(
        on_tool_call: OnToolCall | None,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_duration: float,
        tool_result: ToolResult,
    ) -> None:
        """Invoke the on_tool_call callback with backward compatibility."""
        if on_tool_call is None:
            return
        try:
            err_msg = (tool_result.output or "")[:200] if tool_result.error else ""
            on_tool_call(tool_name, tool_args, tool_duration, not tool_result.error, err_msg)
        except TypeError:
            # Backward compat: caller may not accept error_message
            try:
                on_tool_call(tool_name, tool_args, tool_duration, not tool_result.error)
            except Exception:  # noqa: BLE001
                logger.debug("on_tool_call callback raised; ignoring")
        except Exception:  # noqa: BLE001
            logger.debug("on_tool_call callback raised; ignoring")

    @staticmethod
    def _record_tool_call(
        tool_call_store: ToolCallStore | None,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: ToolResult,
        tool_duration: float,
        agent_name: str,
        iteration: int,
    ) -> None:
        """Record a tool call to the ToolCallStore if available."""
        if tool_call_store is None:
            return
        try:
            from datetime import datetime

            err_msg_store = (tool_result.output or "")[:500] if tool_result.error else ""
            record = ToolCallRecord(
                tool_name=tool_name,
                tool_args=tool_args,
                output=tool_result.output or "",
                output_size_bytes=len((tool_result.output or "").encode("utf-8")),
                error=tool_result.error,
                error_type="",
                error_message=err_msg_store,
                duration_s=tool_duration,
                timestamp=datetime.now(UTC).isoformat(),
                agent_name=agent_name,
                run_id=tool_call_store.run_id,
                iteration=iteration,
            )
            tool_call_store.record(record)
        except Exception:  # noqa: BLE001
            logger.debug("tool_call_store.record() failed; ignoring")

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

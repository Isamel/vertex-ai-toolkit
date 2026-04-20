"""Tool-loop mixin -- reusable tool-use loop for agents that interact with Gemini function calling."""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import inspect
import logging
import math
import time
from datetime import UTC
from typing import TYPE_CHECKING, Any, Protocol

from google.api_core.exceptions import InvalidArgument
from google.genai import errors as genai_errors
from google.genai import types

from vaig.core.async_utils import to_async
from vaig.core.config import DEFAULT_CHARS_PER_TOKEN, DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_OUTPUT_TOKENS, get_settings
from vaig.core.context_budget import ContextBudgetManager
from vaig.core.event_bus import EventBus
from vaig.core.events import ContextWindowChecked, LoopStepEvent
from vaig.core.evidence_ledger import EvidenceEntry, EvidenceLedger, _hash_tool_args
from vaig.core.exceptions import (
    CONTEXT_WINDOW_ERROR_KEYWORDS,
    BudgetExhaustedError,
    CircuitBreakerOpenError,
    ContextWindowExceededError,
    MaxIterationsError,
)
from vaig.core.output_redactor import redact_sensitive_output
from vaig.core.prompt_defense import wrap_untrusted_content
from vaig.session.summarizer import SUMMARIZATION_PROMPT, estimate_history_tokens
from vaig.tools.base import ToolCallRecord, ToolDef, ToolRegistry, ToolResult

# Maximum allowed length for any single string argument passed to a tool.
# Defense-in-depth against LLM-generated payloads that could exhaust tool
# resources (e.g. disk, network, or API quotas).
MAX_TOOL_ARG_LENGTH: int = 50_000

if TYPE_CHECKING:
    from vaig.core.cache import ToolResultCache
    from vaig.core.circuit_breaker import CircuitBreaker
    from vaig.core.client import ToolCallResult
    from vaig.core.global_budget import GlobalBudgetManager
    from vaig.core.protocols import GeminiClientProtocol
    from vaig.core.tool_call_store import ToolCallStore

logger = logging.getLogger(__name__)

BUDGET_WARNING_THRESHOLD = 0.8

# Percentage of context window at which the circuit breaker fires.
# This is the LAST LINE OF DEFENSE — upstream truncation (merge/summary)
# should prevent hitting this.  Set slightly below 100% so we abort before
# the API rejects the request outright.
CONTEXT_CIRCUIT_BREAKER_PCT = 95.0

# Re-export under the old name for backward compatibility.
_CONTEXT_ERROR_KEYWORDS = CONTEXT_WINDOW_ERROR_KEYWORDS


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
        cached: ``True`` when the result came from the tool-result cache.
            Keyword-only for backward compatibility.
    """

    def __call__(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        duration: float,
        success: bool,
        error_message: str = "",
        *,
        cached: bool = False,
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

    @staticmethod
    def _synthesize_tool_summary(tools_executed: list[dict[str, Any]]) -> str:
        """Synthesize a human-readable summary of executed tools.

        Lists each tool name with its primary target argument (the first
        match among ``path``, ``file``, ``filepath``, ``filename``,
        ``command``, ``query``, ``url``), counts failures, and returns a
        single summary line.  Returns an empty string when *tools_executed*
        is empty so callers can use it as a falsy guard.

        Args:
            tools_executed: List of tool-execution dicts, each with keys
                ``name`` (str), ``args`` (dict), and ``error`` (bool).

        Returns:
            Summary string such as::

                "Completed 3 tool operation(s): edit_file(path='foo.py'), ..."

            or with failures::

                "Completed 3 tool operation(s) with 1 failure(s): edit_file(path='foo.py'), read_file(path='baz.py') [FAILED]"

            Returns ``""`` when *tools_executed* is empty.
        """
        if not tools_executed:
            return ""

        _TARGET_ARG_KEYS = ("path", "file", "filepath", "filename", "command", "query", "url")
        failure_count = 0
        parts: list[str] = []

        for t in tools_executed:
            name = t["name"]
            args: dict[str, Any] = t.get("args") or {}
            failed: bool = bool(t.get("error"))

            # Find first matching target arg
            target_val: str | None = None
            for key in _TARGET_ARG_KEYS:
                if key in args:
                    target_val = str(args[key])
                    label = f"{name}({key}={target_val!r})"
                    break
            else:
                label = name

            if failed:
                failure_count += 1
                label = f"{label} [FAILED]"

            parts.append(label)

        total = len(tools_executed)
        tool_list = ", ".join(parts)

        if failure_count:
            return f"Completed {total} tool operation(s) with {failure_count} failure(s): {tool_list}"
        return f"Completed {total} tool operation(s): {tool_list}"

    @staticmethod
    def _load_cw_thresholds() -> tuple[float, float]:
        """Load context window warn/error thresholds from settings.

        Returns a ``(warn_threshold, error_threshold)`` tuple.  Falls back
        to safe hardcoded defaults when settings are unavailable so the tool
        loop never fails due to a configuration error.

        Returns:
            Tuple of ``(warn_threshold_pct, error_threshold_pct)``.
        """
        try:
            _cw_cfg = get_settings().context_window
            return _cw_cfg.warn_threshold_pct, _cw_cfg.error_threshold_pct
        except Exception:  # noqa: BLE001
            return 80.0, 95.0

    def _emit_loop_step(
        self,
        *,
        run_id: str,
        skill: str,
        loop_type: str,
        iteration: int,
        prompt: Any,
        history: list[Any],
        result: Any,
        tool_calls_made: int,
        budget_manager: ContextBudgetManager | None,
        termination_reason: str,
    ) -> None:
        """Compute hashes and emit LoopStepEvent. Swallows all exceptions."""
        try:
            inputs_hash = hashlib.sha256(
                f"{prompt!r}:{len(history)}".encode()
            ).hexdigest()[:16]
            outputs_hash = hashlib.sha256(
                (result.text or "").encode()
            ).hexdigest()[:16]
            tokens_used = result.usage.get("total_tokens", 0) if result.usage else 0
            budget_remaining = (
                budget_manager.remaining_usd()
                if budget_manager is not None and hasattr(budget_manager, "remaining_usd")
                else 0.0
            )
            EventBus.get().emit(LoopStepEvent(
                run_id=run_id,
                skill=skill,
                loop_type=loop_type,
                iteration=iteration,
                inputs_hash=inputs_hash,
                outputs_hash=outputs_hash,
                tokens_used=tokens_used,
                tool_calls_made=tool_calls_made,
                budget_remaining_usd=budget_remaining,
                termination_reason=termination_reason,
            ))
        except Exception:  # noqa: BLE001
            logger.debug("_emit_loop_step: failed to emit LoopStepEvent", exc_info=True)

    def _run_tool_loop(
        self,
        *,
        client: GeminiClientProtocol,
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
        tool_result_cache: ToolResultCache | None = None,
        required_sections: list[str] | None = None,
        max_history_tokens: int = 28_000,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        budget_manager: ContextBudgetManager | None = None,
        global_budget_mgr: GlobalBudgetManager | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        run_id: str = "",
        skill: str = "",
        loop_type: str = "",
        ledger: EvidenceLedger | None = None,
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
            budget_manager: Optional :class:`~vaig.core.context_budget.ContextBudgetManager`
                for tracking token usage per phase.  When provided, records
                prompt tokens under the ``"tool_loop"`` phase after each API
                call and logs a WARNING if the budget is exceeded.
            global_budget_mgr: Optional :class:`~vaig.core.global_budget.GlobalBudgetManager`
                for enforcing cross-run resource limits (tokens, cost, tool calls,
                wall time).  When provided, ``check()`` is called after each API
                call and raises :class:`~vaig.core.exceptions.BudgetExhaustedError`
                if any limit is exceeded.
            circuit_breaker: Optional :class:`~vaig.core.circuit_breaker.CircuitBreaker`
                for transient failure protection.  When provided, ``allow_request()``
                is called before each API call and failures are recorded.

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
        result: ToolCallResult | None = None  # set on first LLM call; safe because loop gates access
        budget_warning_issued = False
        accumulated_llm_text = ""
        peak_context_pct: float = 0.0

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

        # -- Hoist settings and event infrastructure before the loop --------
        # Resolving settings and importing event types once avoids repeated
        # per-iteration overhead (C5).
        _warn_threshold, _error_threshold = self._load_cw_thresholds()

        try:
            _inter_call_delay = get_settings().agents.min_inter_call_delay
        except Exception:  # noqa: BLE001
            _inter_call_delay = 0.0

        while iteration < max_iterations:
            iteration += 1
            logger.debug("Tool loop iteration %d/%d", iteration, max_iterations)

            # -- RPM throttle: sleep between LLM calls (skip first) ------
            if iteration > 1 and _inter_call_delay > 0:
                time.sleep(_inter_call_delay)

            # -- Circuit breaker: check before API call -------------------
            if circuit_breaker is not None:
                try:
                    asyncio.get_event_loop().run_until_complete(circuit_breaker.allow_request())
                except CircuitBreakerOpenError:
                    raise

            # -- Call Gemini with tool declarations -----------------------
            try:
                result = client.generate_with_tools(
                    prompt if iteration == 1 else [],
                    tool_declarations=declarations,
                    system_instruction=system_instruction,
                    history=history,
                    model_id=model,
                    **gen_kwargs,
                )
            except Exception as _api_exc:
                # Record failure in circuit breaker
                if circuit_breaker is not None:
                    try:
                        asyncio.get_event_loop().run_until_complete(circuit_breaker.record_failure())
                    except Exception:  # noqa: BLE001
                        pass

                # If it's already a ContextWindowExceededError (e.g. from circuit breaker),
                # re-raise immediately — don't let the broad except swallow it.
                if isinstance(_api_exc, ContextWindowExceededError):
                    raise

                _msg_lower = str(_api_exc).lower()
                _is_ctx_error = any(kw in _msg_lower for kw in _CONTEXT_ERROR_KEYWORDS)

                if isinstance(_api_exc, InvalidArgument):
                    if _is_ctx_error:
                        raise ContextWindowExceededError(
                            f"Context window exceeded (HTTP 400 InvalidArgument): {_api_exc}",
                            context_pct=peak_context_pct,
                            usage=dict(total_usage),
                        ) from _api_exc
                    logger.warning(
                        "ToolLoopMixin received InvalidArgument without context-window keywords on iteration %d: %s",
                        iteration,
                        _api_exc,
                    )
                    raise

                # google-genai SDK ClientError(400/413) — same error, different type.
                if isinstance(_api_exc, genai_errors.ClientError) and _api_exc.code in (400, 413) and _is_ctx_error:
                    raise ContextWindowExceededError(
                        f"Context window exceeded (genai ClientError {_api_exc.code}): {_api_exc}",
                        context_pct=peak_context_pct,
                        usage=dict(total_usage),
                    ) from _api_exc

                # Vertex AI server-side cancellation (400 CANCELLED). Retryability
                # is decided in the client retry layer; at this point we only
                # avoid emitting a noisy stack trace after retries are exhausted.
                if (
                    isinstance(_api_exc, genai_errors.ClientError)
                    and _api_exc.code == 400
                    and "cancelled" in _msg_lower
                ):
                    logger.warning(
                        "ToolLoopMixin 400 CANCELLED on iteration %d — transient server cancellation",
                        iteration,
                    )
                    raise

                logger.exception(
                    "ToolLoopMixin API call failed on iteration %d",
                    iteration,
                )
                raise

            assert result is not None  # always set by generate_with_tools above

            # -- Record success in circuit breaker -----------------------
            if circuit_breaker is not None:
                try:
                    asyncio.get_event_loop().run_until_complete(circuit_breaker.record_success())
                except Exception:  # noqa: BLE001
                    pass

            # -- Accumulate token usage -----------------------------------
            for key in total_usage:
                total_usage[key] += result.usage.get(key, 0)

            # -- Global budget check (tokens + cost) ----------------------
            if global_budget_mgr is not None:
                try:
                    _prompt_tokens_gb = result.usage.get("prompt_tokens", 0)
                    _completion_tokens_gb = result.usage.get("completion_tokens", 0)
                    asyncio.get_event_loop().run_until_complete(
                        global_budget_mgr.record_tokens(_prompt_tokens_gb + _completion_tokens_gb)
                    )
                    asyncio.get_event_loop().run_until_complete(global_budget_mgr.check())
                except BudgetExhaustedError:
                    raise

            # -- Context window monitoring (G1) ---------------------------
            try:
                peak_context_pct = self._monitor_context_window(
                    result=result,
                    context_window=context_window,
                    peak_context_pct=peak_context_pct,
                    iteration=iteration,
                    model=model,
                    warn_threshold=_warn_threshold,
                    error_threshold=_error_threshold,
                )
            except ContextWindowExceededError:
                self._emit_loop_step(
                    run_id=run_id,
                    skill=skill,
                    loop_type=loop_type,
                    iteration=iteration,
                    prompt=prompt,
                    history=history,
                    result=result,
                    tool_calls_made=0,
                    budget_manager=budget_manager,
                    termination_reason="context_exceeded",
                )
                raise

            # -- Context budget tracking ----------------------------------
            if budget_manager is not None:
                _prompt_tokens = result.usage.get("prompt_tokens", 0)
                budget_manager.record_usage("tool_loop", _prompt_tokens)
                if budget_manager.is_over_budget("tool_loop") and not budget_warning_issued:
                    logger.warning(
                        "Context budget exceeded for phase 'tool_loop' on iteration %d "
                        "(remaining: %d tokens)",
                        iteration,
                        budget_manager.remaining("tool_loop"),
                    )

            # -- Case 1: text response (no function calls) -- done --------
            if not result.function_calls:
                logger.debug(
                    "Tool loop completed -- %d iterations, %d tool calls",
                    iteration,
                    len(tools_executed),
                )
                final_text = result.text
                # Synthesize a minimal summary when no text was produced but tools ran.
                # This ensures callers always have a non-empty content string to display.
                if not final_text and tools_executed:
                    final_text = self._synthesize_tool_summary(tools_executed)
                self._emit_loop_step(
                    run_id=run_id,
                    skill=skill,
                    loop_type=loop_type,
                    iteration=iteration,
                    prompt=prompt,
                    history=history,
                    result=result,
                    tool_calls_made=0,
                    budget_manager=budget_manager,
                    termination_reason="text_response",
                )
                return ToolLoopResult(
                    text=final_text,
                    usage=total_usage,
                    tools_executed=tools_executed,
                    iterations=iteration,
                    model=result.model,
                    finish_reason=result.finish_reason,
                    peak_context_pct=peak_context_pct,
                    ledger=ledger,
                )

            # -- Case 2: function calls -- execute and continue -----------
            # Accumulate LLM text for budget warning tracking
            accumulated_llm_text += result.text or ""

            # Sync path: sequential execution. Use async path for parallel tool calls.
            history.append(
                self._build_function_call_content(
                    result.function_calls,
                    raw_parts=result.raw_parts,
                ),
            )

            function_responses: list[dict[str, Any]] = []

            for fc in result.function_calls:
                tool_name = fc["name"]
                tool_args = fc["args"]

                # Look up tool definition once for cache checks
                tool_def = tool_registry.get(tool_name) if tool_result_cache is not None else None

                # ── Cache lookup ───────────────────────────────────
                cached_result: ToolResult | None = None
                if tool_def is not None and tool_def.cacheable:
                    cached_result = tool_result_cache.get_or_none(  # type: ignore[union-attr]
                        tool_name,
                        tool_args,
                        ttl_override=tool_def.cache_ttl_seconds,
                    )

                if cached_result is not None:
                    # Cache hit — skip execution
                    logger.debug("[CACHE HIT] %s", tool_name)
                    tool_result = cached_result
                    tool_duration = 0.0
                    is_cached = True
                    self._emit_tool_telemetry(
                        tool_name, tool_args, tool_result, time.perf_counter(), cached=True,
                    )
                else:
                    # Cache miss or not cacheable — execute
                    t_tool = time.perf_counter()
                    tool_result = self._execute_single_tool(
                        tool_registry,
                        tool_name,
                        tool_args,
                    )
                    tool_duration = time.perf_counter() - t_tool
                    is_cached = False

                    # Store in cache (reuse tool_def from lookup above)
                    if tool_result_cache is not None and not tool_result.error:
                        if tool_def is None:
                            tool_def = tool_registry.get(tool_name)
                        if tool_def is not None and tool_def.cacheable:
                            from vaig.core.cache import _make_tool_cache_key

                            cache_key = _make_tool_cache_key(tool_name, tool_args)
                            tool_result_cache.put(
                                cache_key,
                                tool_result,
                                ttl_seconds=tool_def.cache_ttl_seconds,
                            )

                # Notify caller about this tool execution
                self._notify_tool_call(
                    on_tool_call,
                    tool_name,
                    tool_args,
                    tool_duration,
                    tool_result,
                    cached=is_cached,
                )

                # Record tool call for metrics/feedback storage
                if not (tool_result.output or "").startswith("Tool skipped (effectiveness)"):
                    self._record_tool_call(
                        tool_call_store,
                        tool_name,
                        tool_args,
                        tool_result,
                        tool_duration,
                        agent_name,
                        iteration,
                        cached=is_cached,
                    )
                    # Append to evidence ledger (SH-05-R-14)
                    if ledger is not None:
                        entry = EvidenceEntry(
                            source_agent=agent_name,
                            tool_name=tool_name,
                            tool_args_hash=_hash_tool_args(tool_args),
                            question="",
                            answer_summary=(tool_result.output or "")[:500],
                            raw_output_ref=getattr(tool_call_store, "run_id", "") if tool_call_store else "",
                            supports=(),
                            contradicts=(),
                        )
                        ledger = ledger.append(entry)

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
            response_parts = client.build_function_response_parts(
                function_responses,
            )
            history.append(types.Content(role="user", parts=response_parts))

            # Emit LoopStepEvent for tool-calls iteration (X-01)
            iteration_tool_calls = len(result.function_calls)
            self._emit_loop_step(
                run_id=run_id,
                skill=skill,
                loop_type=loop_type,
                iteration=iteration,
                prompt=prompt,
                history=history,
                result=result,
                tool_calls_made=iteration_tool_calls,
                budget_manager=budget_manager,
                termination_reason="",
            )

            # -- Budget warning injection ---------------------------------
            budget_warning_issued = self._check_and_inject_budget_warning(
                history,
                budget_warning_issued,
                required_sections,
                max_iterations,
                iteration,
                accumulated_llm_text,
                agent_name,
            )
            self._check_and_summarize(history, client, max_history_tokens)

        # -- Max iterations exceeded --------------------------------------
        msg = (
            f"Tool-use loop exceeded maximum iterations ({max_iterations}). Executed {len(tools_executed)} tool calls."
        )
        logger.warning(msg)
        if iteration > 0 and result is not None:
            self._emit_loop_step(
                run_id=run_id,
                skill=skill,
                loop_type=loop_type,
                iteration=iteration,
                prompt=prompt,
                history=history,
                result=result,
                tool_calls_made=len(tools_executed),
                budget_manager=budget_manager,
                termination_reason="max_iterations",
            )
        raise MaxIterationsError(msg, iterations=max_iterations, partial_output=accumulated_llm_text)

    # ── Budget warning helper ────────────────────────────────

    def _monitor_context_window(
        self,
        *,
        result: Any,
        context_window: int,
        peak_context_pct: float,
        iteration: int,
        model: str | None,
        warn_threshold: float,
        error_threshold: float,
    ) -> float:
        """Check context window usage for one iteration and emit the event.

        Computes the current context percentage, updates *peak_context_pct*,
        logs at the appropriate level, and fires a :class:`ContextWindowChecked`
        event via :class:`EventBus`.  Bus failures are silently swallowed so
        they never interrupt the tool loop.

        When the estimated usage exceeds ``CONTEXT_CIRCUIT_BREAKER_PCT`` (or
        *error_threshold*, whichever is higher), a
        :class:`ContextWindowExceededError` is raised as a **last line of
        defense** — upstream truncation should prevent this from firing
        under normal conditions.

        Args:
            result: The ``ToolCallResult`` from this iteration.
            context_window: Total context window size in tokens.
            peak_context_pct: Running peak percentage from prior iterations.
            iteration: Current 1-based loop iteration index.
            model: Model ID string (used in the event payload).
            warn_threshold: Percentage threshold for WARNING log level.
            error_threshold: Percentage threshold for ERROR log level.

        Returns:
            Updated *peak_context_pct* (max of prior value and current).

        Raises:
            ContextWindowExceededError: When usage exceeds the circuit-breaker
                threshold.
        """
        _prompt_tokens = result.usage.get("prompt_tokens", 0)
        _context_pct = (_prompt_tokens / context_window * 100) if context_window > 0 else 0.0
        peak_context_pct = max(peak_context_pct, _context_pct)

        _ctx_status = (
            "error" if _context_pct >= error_threshold
            else "warning" if _context_pct >= warn_threshold
            else "ok"
        )
        if _ctx_status == "ok":
            logger.debug(
                "Context window: %.1f%% (%d/%d tokens) — ok",
                _context_pct, _prompt_tokens, context_window,
            )
        elif _ctx_status == "warning":
            logger.warning(
                "Context window: %.1f%% (%d/%d tokens) — approaching limit",
                _context_pct, _prompt_tokens, context_window,
            )
        else:
            logger.error(
                "Context window: %.1f%% (%d/%d tokens) — critical",
                _context_pct, _prompt_tokens, context_window,
            )
        try:
            EventBus.get().emit(
                ContextWindowChecked(
                    model=result.model or (model or ""),
                    prompt_tokens=_prompt_tokens,
                    context_window=context_window,
                    context_pct=_context_pct,
                    iteration=iteration,
                    status=_ctx_status,
                )
            )
        except Exception:  # noqa: BLE001
            pass

        # Circuit breaker — last line of defense against token explosion.
        _breaker_threshold = max(error_threshold, CONTEXT_CIRCUIT_BREAKER_PCT)
        if _context_pct >= _breaker_threshold:
            raise ContextWindowExceededError(
                f"Context circuit breaker tripped at {_context_pct:.1f}% "
                f"({_prompt_tokens}/{context_window} tokens) on iteration {iteration}",
                context_pct=_context_pct,
                usage={"prompt_tokens": _prompt_tokens, "completion_tokens": 0, "total_tokens": _prompt_tokens},
            )

        return peak_context_pct

    # ── Budget warning helper ────────────────────────────────

    def _check_and_inject_budget_warning(
        self,
        history: list[Any],
        budget_warning_issued: bool,
        required_sections: list[str] | None,
        max_iterations: int,
        iteration: int,
        accumulated_llm_text: str,
        agent_name: str,
    ) -> bool:
        """Check iteration budget and inject a warning message into history if needed.

        Injects a warning only once, only when *required_sections* is set,
        only when the iteration reaches or exceeds the 80% threshold, and only
        when at least one required section is missing from *accumulated_llm_text*.

        Args:
            history: Mutable conversation history list (appended in-place).
            budget_warning_issued: Whether a warning has already been injected.
            required_sections: Sections the model must produce; ``None`` disables
                the feature entirely.
            max_iterations: Safety cap used to compute the threshold.
            iteration: Current 1-based iteration index.
            accumulated_llm_text: All LLM text produced so far (for section matching).
            agent_name: Name used in the log warning.

        Returns:
            Updated *budget_warning_issued* flag (``True`` if a warning was
            injected this call or on a prior call; ``False`` otherwise).
        """
        if (
            budget_warning_issued
            or not required_sections
            or max_iterations <= 2
            or iteration < math.ceil(max_iterations * BUDGET_WARNING_THRESHOLD)
        ):
            return budget_warning_issued

        missing = [
            s for s in required_sections
            if s.lower() not in accumulated_llm_text.lower()
        ]
        if not missing:
            return budget_warning_issued

        remaining = max_iterations - iteration
        missing_list = ", ".join(missing)
        warning = (
            f"⚠️ BUDGET WARNING: You have used {iteration}/{max_iterations} iterations. "
            f"The following required sections are MISSING from your output: {missing_list}. "
            f"You have {remaining} iterations left. "
            f"Focus on producing these sections NOW."
        )
        history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=warning)])
        )
        logger.warning(
            "Budget warning injected for agent '%s': missing=%s, iter=%d/%d",
            agent_name, missing_list, iteration, max_iterations,
        )
        return True

    def _check_and_summarize(
        self,
        history: list[Any],
        client: GeminiClientProtocol,
        max_history_tokens: int,
    ) -> None:
        """Summarize history in-place if it exceeds the token budget.

        When the accumulated tool-loop history grows beyond *max_history_tokens*,
        the oldest portion is condensed into a single ``[CONVERSATION SUMMARY]``
        ``Content`` item and the most recent third of entries is kept verbatim.
        The mutation is performed in-place so the caller's ``history`` reference
        reflects the change immediately.

        Args:
            history: Mutable list of ``types.Content`` items (modified in-place).
            client: The ``GeminiClientProtocol`` used for the summarization call.
            max_history_tokens: Token ceiling before summarization is triggered.
        """
        if not history:
            return
        current_tokens = estimate_history_tokens(history)
        if current_tokens <= max_history_tokens:
            return

        # Keep the last 1/3 of messages to preserve recent context
        keep_count = max(1, len(history) // 3)
        to_summarize = history[:-keep_count]
        to_keep = history[-keep_count:]

        # Build a text representation of the items to summarize
        parts_text: list[str] = []
        for item in to_summarize:
            text = ""
            if hasattr(item, "parts") and item.parts:
                text = " ".join(p.text for p in item.parts if getattr(p, "text", None))
            if text:
                role = getattr(item, "role", "unknown").upper()
                parts_text.append(f"[{role}]: {text}")
        conversation_text = "\n\n".join(parts_text)

        target_tokens = max_history_tokens // 4
        target_chars = int(target_tokens * DEFAULT_CHARS_PER_TOKEN)
        prompt = SUMMARIZATION_PROMPT.format(
            target_tokens=target_tokens,
            target_chars=target_chars,
        )

        # History contains untrusted tool outputs (kubectl results, logs, etc.).
        # Wrap before sending to the LLM to prevent prompt injection attacks.
        safe_conversation_text = wrap_untrusted_content(conversation_text)

        try:
            result = client.generate(
                safe_conversation_text,
                system_instruction=prompt,
                temperature=0.3,
                max_output_tokens=target_tokens,
            )
            summary_text = result.text.strip()
            if not summary_text.startswith("[CONVERSATION SUMMARY]"):
                summary_text = f"[CONVERSATION SUMMARY]\n{summary_text}"
        except Exception:  # noqa: BLE001
            # Broad catch is intentional: summarization failure must NEVER crash
            # the tool loop. Any exception (network, quota, malformed response)
            # should fall back gracefully so the agent can continue operating.
            logger.warning("History summarization failed — keeping truncated history")
            summary_text = "[CONVERSATION SUMMARY]\nPrevious conversation could not be summarized."

        summary_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=summary_text)],
        )
        history.clear()
        history.append(summary_content)
        history.extend(to_keep)
        logger.info(
            "History summarized: %d tokens → keeping %d items",
            current_tokens,
            len(history),
        )

    async def _async_check_and_summarize(
        self,
        history: list[Any],
        client: GeminiClientProtocol,
        max_history_tokens: int,
    ) -> None:
        """Async variant of :meth:`_check_and_summarize` for use in async tool loops.

        Uses ``client.async_generate()`` to avoid blocking the event loop.
        See :meth:`_check_and_summarize` for full behaviour documentation.
        """
        if not history:
            return
        current_tokens = estimate_history_tokens(history)
        if current_tokens <= max_history_tokens:
            return

        # Keep the last 1/3 of messages to preserve recent context
        keep_count = max(1, len(history) // 3)
        to_summarize = history[:-keep_count]
        to_keep = history[-keep_count:]

        # Build a text representation of the items to summarize
        parts_text: list[str] = []
        for item in to_summarize:
            text = ""
            if hasattr(item, "parts") and item.parts:
                text = " ".join(p.text for p in item.parts if getattr(p, "text", None))
            if text:
                role = getattr(item, "role", "unknown").upper()
                parts_text.append(f"[{role}]: {text}")
        conversation_text = "\n\n".join(parts_text)

        target_tokens = max_history_tokens // 4
        target_chars = int(target_tokens * DEFAULT_CHARS_PER_TOKEN)
        prompt = SUMMARIZATION_PROMPT.format(
            target_tokens=target_tokens,
            target_chars=target_chars,
        )

        # History contains untrusted tool outputs (kubectl results, logs, etc.).
        # Wrap before sending to the LLM to prevent prompt injection attacks.
        safe_conversation_text = wrap_untrusted_content(conversation_text)

        try:
            result = await client.async_generate(
                safe_conversation_text,
                system_instruction=prompt,
                temperature=0.3,
                max_output_tokens=target_tokens,
            )
            summary_text = result.text.strip()
            if not summary_text.startswith("[CONVERSATION SUMMARY]"):
                summary_text = f"[CONVERSATION SUMMARY]\n{summary_text}"
        except Exception:  # noqa: BLE001
            # Broad catch is intentional: summarization failure must NEVER crash
            # the tool loop. Any exception (network, quota, malformed response)
            # should fall back gracefully so the agent can continue operating.
            logger.warning("History summarization failed — keeping truncated history")
            summary_text = "[CONVERSATION SUMMARY]\nPrevious conversation could not be summarized."

        summary_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=summary_text)],
        )
        history.clear()
        history.append(summary_content)
        history.extend(to_keep)
        logger.info(
            "History summarized: %d tokens → keeping %d items",
            current_tokens,
            len(history),
        )

    # ── Unknown tool error message builder ───────────────────

    def _build_unknown_tool_message(self, tool_name: str, tool_registry: ToolRegistry) -> str:
        """Build a helpful error message for an unknown tool request.

        Sorts available tool names once, caps the list at 10 entries to avoid
        bloating the model's context, and includes fuzzy-match suggestions when
        the requested name is close to an existing tool.

        Args:
            tool_name: The name of the tool that was not found.
            tool_registry: The registry to look up available tools from.

        Returns:
            A human-readable error string suitable for returning to the model.
        """
        available_names = sorted(t.name for t in tool_registry.list_tools())
        suggestions = difflib.get_close_matches(tool_name, available_names, n=3, cutoff=0.5)
        suggestion_hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""

        _max_display = 10
        if not available_names:
            available_hint = "(none registered)"
        elif len(available_names) <= _max_display:
            available_hint = ", ".join(available_names)
        else:
            remainder = len(available_names) - _max_display
            available_hint = f"{', '.join(available_names[:_max_display])}, ... and {remainder} more"

        return (
            f"Tool '{tool_name}' does not exist in the registry.{suggestion_hint}"
            f" Available tools: {available_hint}."
            " Please use one of the available tools listed above."
        )

    # ── Tool arg pre-validation ────────────────────────────────

    @staticmethod
    def _pre_validate_tool_args(
        tool: ToolDef,
        args: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """Validate tool arguments BEFORE execution (defense-in-depth).

        Checks performed:
        1. Reject unknown arg keys not declared in ``tool.parameters``.
        2. Check required params are present (params with ``required=True``).
        3. Enforce max string length per arg value (``MAX_TOOL_ARG_LENGTH``).

        Returns:
            ``(True, None)`` when valid, ``(False, error_message)`` otherwise.
        """
        # If tool has no parameter schema (None), skip schema checks but
        # still enforce string length. An explicit empty list means the tool
        # takes no arguments — reject any args in that case.
        param_schema = getattr(tool, "parameters", None)
        if param_schema is not None:
            if not param_schema:
                # Explicit "no-args" schema: reject any provided arguments.
                if args:
                    return (
                        False,
                        f"Tool '{tool.name}' does not take any arguments, "
                        f"but received: {sorted(args)}",
                    )
            else:
                declared_names = {p.name for p in param_schema}

                # 1. Reject unknown arg keys
                unknown = set(args) - declared_names
                if unknown:
                    return (
                        False,
                        f"Unknown argument(s) {sorted(unknown)} for tool '{tool.name}'. "
                        f"Allowed parameters: {sorted(declared_names)}",
                    )

                # 2. Check required params are present
                required_names = {p.name for p in param_schema if p.required}
                missing = required_names - set(args)
                if missing:
                    return (
                        False,
                        f"Missing required argument(s) {sorted(missing)} for tool '{tool.name}'.",
                    )

        # 3. Enforce max string length (always — even without schema)
        for key, value in args.items():
            if isinstance(value, str) and len(value) > MAX_TOOL_ARG_LENGTH:
                return (
                    False,
                    f"Argument '{key}' for tool '{tool.name}' exceeds maximum length "
                    f"({len(value):,} chars > {MAX_TOOL_ARG_LENGTH:,} limit).",
                )

        return (True, None)

    # ── Effectiveness pre-check ─────────────────────────────

    @staticmethod
    def _check_tool_effectiveness(tool_name: str, agent_name: str) -> ToolResult | None:
        """Return a synthetic :class:`ToolResult` if *tool_name* should be skipped.

        SKIP-tier tools get a synthetic result without execution.
        DEPRIORITIZE-tier tools log a warning but return ``None`` (proceed).
        ALLOW/BOOST and disabled service return ``None`` (proceed).
        """
        from vaig.core.effectiveness import EffectivenessTier, get_effectiveness_service

        svc = get_effectiveness_service()
        if svc is None:
            return None

        score = svc.get_tool_score(tool_name, agent_name)

        if score.tier == EffectivenessTier.SKIP:
            logger.info("Skipping tool %s: %s", tool_name, score.reason)
            return ToolResult(
                output=f"Tool skipped (effectiveness): {score.reason}",
                error=True,
            )

        if score.tier == EffectivenessTier.DEPRIORITIZE:
            logger.warning("Deprioritized tool %s: %s", tool_name, score.reason)

        return None

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
                output=self._build_unknown_tool_message(tool_name, tool_registry),
                error=True,
            )
            self._emit_tool_telemetry(tool_name, tool_args, result, t0, error_type="UnknownTool")
            return result

        # ── Effectiveness pre-check (R-EFF-06) ────────────────
        skip_result = self._check_tool_effectiveness(tool_name, getattr(self, "name", ""))
        if skip_result is not None:
            return skip_result

        # ── Pre-validate arguments (AI-SEC4) ──────────────────
        is_valid, validation_error = self._pre_validate_tool_args(tool, tool_args)
        if not is_valid:
            logger.warning("Tool %s arg validation failed: %s", tool_name, validation_error)
            result = ToolResult(output=validation_error or "Argument validation failed", error=True)
            self._emit_tool_telemetry(
                tool_name, tool_args, result, t0,
                error_type="ValidationError", error_message=validation_error or "",
            )
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
                f"Expected parameters: {', '.join(p.name for p in (tool.parameters or []))}",
                error=True,
            )
            self._emit_tool_telemetry(tool_name, tool_args, result, t0, error_type="TypeError", error_message=str(exc))
            return result
        except Exception as exc:  # noqa: BLE001
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
        cached: bool = False,
    ) -> None:
        """Emit a tool_call telemetry event via EventBus. Never raises."""
        try:
            from vaig.core.event_bus import EventBus
            from vaig.core.events import ToolExecuted

            duration_ms = (time.perf_counter() - t0) * 1000
            EventBus.get().emit(
                ToolExecuted(
                    tool_name=tool_name,
                    duration_ms=duration_ms,
                    args_keys=tuple(sorted(tool_args.keys())),
                    error=result.error,
                    error_type=error_type,
                    error_message=error_message,
                    cached=cached,
                )
            )
        except Exception:  # noqa: BLE001
            pass

    # ── Async public loop entry-point ────────────────────────

    async def _async_run_tool_loop(
        self,
        *,
        client: GeminiClientProtocol,
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
        tool_result_cache: ToolResultCache | None = None,
        required_sections: list[str] | None = None,
        max_history_tokens: int = 28_000,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        budget_manager: ContextBudgetManager | None = None,
        global_budget_mgr: GlobalBudgetManager | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        run_id: str = "",
        skill: str = "",
        loop_type: str = "",
        ledger: EvidenceLedger | None = None,
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
        result: ToolCallResult | None = None  # set on first LLM call; safe because loop gates access
        budget_warning_issued = False
        accumulated_llm_text = ""
        peak_context_pct: float = 0.0

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

        # -- Hoist settings and event infrastructure before the loop --------
        # Resolving settings and importing event types once avoids repeated
        # per-iteration overhead (C5).
        _warn_threshold, _error_threshold = self._load_cw_thresholds()

        try:
            _inter_call_delay = get_settings().agents.min_inter_call_delay
        except Exception:  # noqa: BLE001
            _inter_call_delay = 0.0

        while iteration < max_iterations:
            iteration += 1
            logger.debug("Async tool loop iteration %d/%d", iteration, max_iterations)

            # -- RPM throttle: sleep between LLM calls (skip first) ------
            if iteration > 1 and _inter_call_delay > 0:
                await asyncio.sleep(_inter_call_delay)

            # -- Circuit breaker: check before API call -------------------
            if circuit_breaker is not None:
                await circuit_breaker.allow_request()

            # -- Call Gemini with tool declarations (async) ----------------
            try:
                result = await client.async_generate_with_tools(
                    prompt if iteration == 1 else [],
                    tool_declarations=declarations,
                    system_instruction=system_instruction,
                    history=history,
                    model_id=model,
                    **gen_kwargs,
                )
            except Exception as _api_exc:
                # Record failure in circuit breaker
                if circuit_breaker is not None:
                    try:
                        await circuit_breaker.record_failure()
                    except Exception:  # noqa: BLE001
                        pass

                # If it's already a ContextWindowExceededError (e.g. from circuit breaker),
                # re-raise immediately — don't let the broad except swallow it.
                if isinstance(_api_exc, ContextWindowExceededError):
                    raise

                _msg_lower = str(_api_exc).lower()
                _is_ctx_error = any(kw in _msg_lower for kw in _CONTEXT_ERROR_KEYWORDS)

                if isinstance(_api_exc, InvalidArgument):
                    if _is_ctx_error:
                        raise ContextWindowExceededError(
                            f"Context window exceeded (HTTP 400 InvalidArgument): {_api_exc}",
                            context_pct=peak_context_pct,
                            usage=dict(total_usage),
                        ) from _api_exc
                    logger.warning(
                        "ToolLoopMixin received InvalidArgument without context-window keywords on iteration %d: %s",
                        iteration,
                        _api_exc,
                    )
                    raise

                # google-genai SDK ClientError(400/413) — same error, different type.
                if isinstance(_api_exc, genai_errors.ClientError) and _api_exc.code in (400, 413) and _is_ctx_error:
                    raise ContextWindowExceededError(
                        f"Context window exceeded (genai ClientError {_api_exc.code}): {_api_exc}",
                        context_pct=peak_context_pct,
                        usage=dict(total_usage),
                    ) from _api_exc

                # Vertex AI server-side cancellation (400 CANCELLED). Retryability
                # is decided in the client retry layer; at this point we only
                # avoid emitting a noisy stack trace after retries are exhausted.
                if (
                    isinstance(_api_exc, genai_errors.ClientError)
                    and _api_exc.code == 400
                    and "cancelled" in _msg_lower
                ):
                    logger.warning(
                        "ToolLoopMixin 400 CANCELLED on iteration %d — transient server cancellation",
                        iteration,
                    )
                    raise

                logger.exception(
                    "ToolLoopMixin async API call failed on iteration %d",
                    iteration,
                )
                raise

            assert result is not None  # always set by async_generate_with_tools above

            # -- Record success in circuit breaker -----------------------
            if circuit_breaker is not None:
                try:
                    await circuit_breaker.record_success()
                except Exception:  # noqa: BLE001
                    pass

            # -- Accumulate token usage -----------------------------------
            for key in total_usage:
                total_usage[key] += result.usage.get(key, 0)

            # -- Global budget check (tokens + cost) ----------------------
            if global_budget_mgr is not None:
                _prompt_tokens_gb = result.usage.get("prompt_tokens", 0)
                _completion_tokens_gb = result.usage.get("completion_tokens", 0)
                await global_budget_mgr.record_tokens(_prompt_tokens_gb + _completion_tokens_gb)
                await global_budget_mgr.check()

            # -- Context window monitoring (G1) ---------------------------
            try:
                peak_context_pct = self._monitor_context_window(
                    result=result,
                    context_window=context_window,
                    peak_context_pct=peak_context_pct,
                    iteration=iteration,
                    model=model,
                    warn_threshold=_warn_threshold,
                    error_threshold=_error_threshold,
                )
            except ContextWindowExceededError:
                self._emit_loop_step(
                    run_id=run_id,
                    skill=skill,
                    loop_type=loop_type,
                    iteration=iteration,
                    prompt=prompt,
                    history=history,
                    result=result,
                    tool_calls_made=0,
                    budget_manager=budget_manager,
                    termination_reason="context_exceeded",
                )
                raise

            # -- Context budget tracking ----------------------------------
            if budget_manager is not None:
                _prompt_tokens = result.usage.get("prompt_tokens", 0)
                budget_manager.record_usage("tool_loop", _prompt_tokens)
                if budget_manager.is_over_budget("tool_loop") and not budget_warning_issued:
                    logger.warning(
                        "Context budget exceeded for phase 'tool_loop' on iteration %d "
                        "(remaining: %d tokens)",
                        iteration,
                        budget_manager.remaining("tool_loop"),
                    )

            # -- Case 1: text response (no function calls) -- done --------
            if not result.function_calls:
                logger.debug(
                    "Async tool loop completed -- %d iterations, %d tool calls",
                    iteration,
                    len(tools_executed),
                )
                final_text = result.text
                # Synthesize a minimal summary when no text was produced but tools ran.
                # This ensures callers always have a non-empty content string to display.
                if not final_text and tools_executed:
                    final_text = self._synthesize_tool_summary(tools_executed)
                self._emit_loop_step(
                    run_id=run_id,
                    skill=skill,
                    loop_type=loop_type,
                    iteration=iteration,
                    prompt=prompt,
                    history=history,
                    result=result,
                    tool_calls_made=0,
                    budget_manager=budget_manager,
                    termination_reason="text_response",
                )
                return ToolLoopResult(
                    text=final_text,
                    usage=total_usage,
                    tools_executed=tools_executed,
                    iterations=iteration,
                    model=result.model,
                    finish_reason=result.finish_reason,
                    peak_context_pct=peak_context_pct,
                    ledger=ledger,
                )

            # -- Case 2: function calls -- execute and continue -----------
            # Accumulate LLM text for budget warning tracking
            accumulated_llm_text += result.text or ""

            history.append(
                self._build_function_call_content(
                    result.function_calls,
                    raw_parts=result.raw_parts,
                ),
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
                ) -> tuple[str, dict[str, Any], float, ToolResult, bool]:
                    """Execute a single tool call under the semaphore."""
                    async with semaphore:
                        _tool_name = fc["name"]
                        _tool_args = fc["args"]

                        # ── Cache lookup ─────────────────────────
                        _td = tool_registry.get(_tool_name) if tool_result_cache is not None else None
                        if _td is not None and _td.cacheable:
                            _cached = tool_result_cache.get_or_none(  # type: ignore[union-attr]
                                _tool_name,
                                _tool_args,
                                ttl_override=_td.cache_ttl_seconds,
                            )
                            if _cached is not None:
                                logger.debug("[CACHE HIT] %s", _tool_name)
                                self._emit_tool_telemetry(
                                    _tool_name, _tool_args, _cached, time.perf_counter(), cached=True,
                                )
                                return _tool_name, _tool_args, 0.0, _cached, True

                        t0 = time.perf_counter()
                        res = await self._async_execute_single_tool(
                            tool_registry,
                            _tool_name,
                            _tool_args,
                        )
                        dur = time.perf_counter() - t0

                        # Store in cache (reuse _td from lookup above)
                        if tool_result_cache is not None and not res.error:
                            if _td is None:
                                _td = tool_registry.get(_tool_name)
                            if _td is not None and _td.cacheable:
                                from vaig.core.cache import _make_tool_cache_key

                                _ck = _make_tool_cache_key(_tool_name, _tool_args)
                                tool_result_cache.put(
                                    _ck, res, ttl_seconds=_td.cache_ttl_seconds,
                                )

                        return _tool_name, _tool_args, dur, res, False

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
                        is_cached = False
                    else:
                        tool_name, tool_args, tool_duration, tool_result, is_cached = res

                    sequential_estimate += tool_duration

                    # Notify caller about this tool execution
                    self._notify_tool_call(
                        on_tool_call,
                        tool_name,
                        tool_args,
                        tool_duration,
                        tool_result,
                        cached=is_cached,
                    )

                    # Record tool call for metrics/feedback storage
                    if not (tool_result.output or "").startswith("Tool skipped (effectiveness)"):
                        self._record_tool_call(
                            tool_call_store,
                            tool_name,
                            tool_args,
                            tool_result,
                            tool_duration,
                            agent_name,
                            iteration,
                            cached=is_cached,
                        )
                        # Append to evidence ledger (SH-05-R-14)
                        if ledger is not None:
                            entry = EvidenceEntry(
                                source_agent=agent_name,
                                tool_name=tool_name,
                                tool_args_hash=_hash_tool_args(tool_args),
                                question="",
                                answer_summary=(tool_result.output or "")[:500],
                                raw_output_ref=getattr(tool_call_store, "run_id", "") if tool_call_store else "",
                                supports=(),
                                contradicts=(),
                            )
                            ledger = ledger.append(entry)

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

                    # Look up tool definition once for cache checks
                    tool_def = tool_registry.get(tool_name) if tool_result_cache is not None else None

                    # ── Cache lookup ───────────────────────────────
                    cached_result_async: ToolResult | None = None
                    if tool_def is not None and tool_def.cacheable:
                        cached_result_async = tool_result_cache.get_or_none(  # type: ignore[union-attr]
                            tool_name,
                            tool_args,
                            ttl_override=tool_def.cache_ttl_seconds,
                        )

                    if cached_result_async is not None:
                        logger.debug("[CACHE HIT] %s", tool_name)
                        tool_result = cached_result_async
                        tool_duration = 0.0
                        is_cached = True
                        self._emit_tool_telemetry(
                            tool_name, tool_args, tool_result, time.perf_counter(), cached=True,
                        )
                    else:
                        t_tool = time.perf_counter()
                        tool_result = await self._async_execute_single_tool(
                            tool_registry,
                            tool_name,
                            tool_args,
                        )
                        tool_duration = time.perf_counter() - t_tool
                        is_cached = False

                        # Store in cache (reuse tool_def from lookup above)
                        if tool_result_cache is not None and not tool_result.error:
                            if tool_def is None:
                                tool_def = tool_registry.get(tool_name)
                            if tool_def is not None and tool_def.cacheable:
                                from vaig.core.cache import _make_tool_cache_key

                                cache_key = _make_tool_cache_key(tool_name, tool_args)
                                tool_result_cache.put(
                                    cache_key,
                                    tool_result,
                                    ttl_seconds=tool_def.cache_ttl_seconds,
                                )

                    # Notify caller about this tool execution
                    self._notify_tool_call(
                        on_tool_call,
                        tool_name,
                        tool_args,
                        tool_duration,
                        tool_result,
                        cached=is_cached,
                    )

                    # Record tool call for metrics/feedback storage
                    if not (tool_result.output or "").startswith("Tool skipped (effectiveness)"):
                        self._record_tool_call(
                            tool_call_store,
                            tool_name,
                            tool_args,
                            tool_result,
                            tool_duration,
                            agent_name,
                            iteration,
                            cached=is_cached,
                        )
                        # Append to evidence ledger (SH-05-R-14)
                        if ledger is not None:
                            entry = EvidenceEntry(
                                source_agent=agent_name,
                                tool_name=tool_name,
                                tool_args_hash=_hash_tool_args(tool_args),
                                question="",
                                answer_summary=(tool_result.output or "")[:500],
                                raw_output_ref=getattr(tool_call_store, "run_id", "") if tool_call_store else "",
                                supports=(),
                                contradicts=(),
                            )
                            ledger = ledger.append(entry)

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
            response_parts = client.build_function_response_parts(
                function_responses,
            )
            history.append(types.Content(role="user", parts=response_parts))

            # Emit LoopStepEvent for tool-calls iteration (X-01)
            iteration_tool_calls = len(result.function_calls)
            self._emit_loop_step(
                run_id=run_id,
                skill=skill,
                loop_type=loop_type,
                iteration=iteration,
                prompt=prompt,
                history=history,
                result=result,
                tool_calls_made=iteration_tool_calls,
                budget_manager=budget_manager,
                termination_reason="",
            )

            # -- Budget warning injection ---------------------------------
            budget_warning_issued = self._check_and_inject_budget_warning(
                history,
                budget_warning_issued,
                required_sections,
                max_iterations,
                iteration,
                accumulated_llm_text,
                agent_name,
            )
            await self._async_check_and_summarize(history, client, max_history_tokens)

        # -- Max iterations exceeded --------------------------------------
        msg = (
            f"Tool-use loop exceeded maximum iterations ({max_iterations}). Executed {len(tools_executed)} tool calls."
        )
        logger.warning(msg)
        if iteration > 0 and result is not None:
            self._emit_loop_step(
                run_id=run_id,
                skill=skill,
                loop_type=loop_type,
                iteration=iteration,
                prompt=prompt,
                history=history,
                result=result,
                tool_calls_made=len(tools_executed),
                budget_manager=budget_manager,
                termination_reason="max_iterations",
            )
        raise MaxIterationsError(msg, iterations=max_iterations, partial_output=accumulated_llm_text)

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
                output=self._build_unknown_tool_message(tool_name, tool_registry),
                error=True,
            )
            self._emit_tool_telemetry(tool_name, tool_args, result, t0, error_type="UnknownTool")
            return result

        # ── Effectiveness pre-check (R-EFF-06) ────────────────
        skip_result = self._check_tool_effectiveness(tool_name, getattr(self, "name", ""))
        if skip_result is not None:
            return skip_result

        # ── Pre-validate arguments (AI-SEC4) ──────────────────
        is_valid, validation_error = self._pre_validate_tool_args(tool, tool_args)
        if not is_valid:
            logger.warning("Tool %s arg validation failed: %s", tool_name, validation_error)
            result = ToolResult(output=validation_error or "Argument validation failed", error=True)
            self._emit_tool_telemetry(
                tool_name, tool_args, result, t0,
                error_type="ValidationError", error_message=validation_error or "",
            )
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
                f"Expected parameters: {', '.join(p.name for p in (tool.parameters or []))}",
                error=True,
            )
            self._emit_tool_telemetry(tool_name, tool_args, result, t0, error_type="TypeError", error_message=str(exc))
            return result
        except Exception as exc:  # noqa: BLE001
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
        *,
        cached: bool = False,
    ) -> None:
        """Invoke the on_tool_call callback with backward compatibility.

        Uses ``inspect.signature`` to determine the number of positional
        parameters *before* invoking the callback.  This avoids the previous
        try/except-TypeError approach which could mask a real ``TypeError``
        raised *inside* the callback body.
        """
        if on_tool_call is None:
            return
        err_msg = (tool_result.output or "")[:200] if tool_result.error else ""

        # Determine the number of positional parameters the callback accepts
        # so we can choose the right call form without catching TypeError.
        try:
            sig = inspect.signature(on_tool_call)
            # Count positional-capable params (POSITIONAL_ONLY, POSITIONAL_OR_KEYWORD, VAR_POSITIONAL)
            _POSITIONAL_KINDS = {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
            positional_count = sum(
                1 for p in sig.parameters.values() if p.kind in _POSITIONAL_KINDS
            )
            has_var_positional = any(
                p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
            )
            has_cached_kw = "cached" in sig.parameters
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            accepts_cached = has_cached_kw or has_var_keyword
        except (ValueError, TypeError):
            # inspect.signature can fail for builtins / C extensions —
            # fall back to the 6-arg form and swallow any error.
            positional_count = 6
            has_var_positional = False
            accepts_cached = True

        try:
            if (positional_count >= 5 or has_var_positional) and accepts_cached:
                on_tool_call(tool_name, tool_args, tool_duration, not tool_result.error, err_msg, cached=cached)
            elif positional_count >= 5 or has_var_positional:
                on_tool_call(tool_name, tool_args, tool_duration, not tool_result.error, err_msg)
            elif accepts_cached:
                on_tool_call(tool_name, tool_args, tool_duration, not tool_result.error, cached=cached)
            else:
                on_tool_call(tool_name, tool_args, tool_duration, not tool_result.error)
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
        *,
        cached: bool = False,
    ) -> None:
        """Record a tool call to the ToolCallStore if available."""
        if tool_call_store is None:
            return
        try:
            from datetime import datetime

            err_msg_store = (tool_result.output or "")[:500] if tool_result.error else ""
            raw_output = tool_result.output or ""
            redacted_output, redaction_count = redact_sensitive_output(raw_output)
            # Also redact the error message snippet
            if err_msg_store:
                err_msg_store, _ = redact_sensitive_output(err_msg_store)
            record = ToolCallRecord(
                tool_name=tool_name,
                tool_args=tool_args,
                output=redacted_output,
                output_size_bytes=len(redacted_output.encode("utf-8")),
                error=tool_result.error,
                error_type="",
                error_message=err_msg_store,
                duration_s=tool_duration,
                timestamp=datetime.now(UTC).isoformat(),
                agent_name=agent_name,
                run_id=tool_call_store.run_id,
                iteration=iteration,
                cached=cached,
                redactions=redaction_count,
            )
            tool_call_store.record(record)
        except Exception:  # noqa: BLE001
            logger.debug("tool_call_store.record() failed; ignoring")

    # ── Message builders ─────────────────────────────────────

    @staticmethod
    def _build_function_call_content(
        function_calls: list[dict[str, Any]],
        *,
        raw_parts: list[types.Part] | None = None,
    ) -> types.Content:
        """Build a ``types.Content`` from a list of function call dicts.

        Each dict must have ``"name"`` (str) and ``"args"`` (dict).
        Returns a ``Content(role="model", parts=[...])`` suitable for
        appending to the conversation history.

        When *raw_parts* is provided (non-empty), the original ``types.Part``
        objects are replayed verbatim.  This preserves Gemini 2.5+
        ``thought_signature`` bytes that would otherwise be lost when
        reconstructing Parts from the extracted function_calls dicts.

        Args:
            function_calls: Extracted function call dicts (fallback path).
            raw_parts: Raw ``types.Part`` objects from the API response.
                When truthy, used directly instead of reconstructing from dicts.
        """
        if raw_parts:
            return types.Content(role="model", parts=raw_parts)
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
    iteration count, model name, finish reason, and peak context
    window usage percentage.  The host agent is responsible for
    wrapping this into an ``AgentResult``.
    """

    __slots__ = (
        "text",
        "usage",
        "tools_executed",
        "iterations",
        "model",
        "finish_reason",
        "peak_context_pct",
        "ledger",
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
        peak_context_pct: float = 0.0,
        ledger: EvidenceLedger | None = None,
    ) -> None:
        self.text = text
        self.usage = usage
        self.tools_executed = tools_executed
        self.iterations = iterations
        self.model = model
        self.finish_reason = finish_reason
        self.peak_context_pct = peak_context_pct
        self.ledger = ledger


# ── SPEC-V2-AUDIT-13: Memory Recall Mixin ─────────────────────────────────────

class MemoryRecallMixin:
    """Mixin that injects prior-run memory recalls into agent system prompts.

    When mixed into a :class:`~vaig.agents.tool_aware.ToolAwareAgent` the mixin
    reads the :class:`~vaig.core.memory.pattern_store.PatternMemoryStore` and
    formats up to ``top_k`` semantically-similar entries into a structured
    ``## Prior-run memory`` block that is appended to the base system instruction
    before the first LLM call.

    Opt-out per agent class (or globally via ``memory.recall_in_gatherers=false``)
    by overriding :meth:`_recall_enabled`.
    """

    def _recall_enabled(self) -> bool:  # noqa: D401
        """Return ``True`` when memory recall is active for this agent."""
        return True

    def _recall_patterns(self, query: str, top_k: int = 3) -> list[Any]:
        """Return up to *top_k* :class:`~vaig.core.memory.models.RecalledPattern` objects.

        Falls back to an empty list on any error so that a broken or empty
        memory store never disrupts the live pipeline.

        The current implementation uses a simple keyword-overlap heuristic
        rather than full semantic embeddings (fast, no extra dependencies).
        Future work: replace with a vector-based approach once MEM-04 is active.
        """
        from vaig.core.config import get_settings
        from vaig.core.memory.models import RecalledPattern
        from vaig.core.memory.pattern_store import PatternMemoryStore

        try:
            cfg = get_settings()
            if not cfg.memory.enabled or cfg.memory.recall_budget_tokens == 0:
                return []

            store = PatternMemoryStore(cfg.memory.store_path)
            entries = store.all_entries()
            if not entries:
                return []

            # Rank by keyword overlap with *query* (case-insensitive)
            # Filter out short stop-words (len <= 2) to reduce noise.
            query_lower = query.lower()
            query_words = {w for w in query_lower.split() if len(w) > 2}

            def _score(entry: Any) -> int:
                text = (entry.title + " " + entry.service + " " + entry.category).lower()
                return sum(1 for w in query_words if w in text)

            ranked = sorted(entries, key=_score, reverse=True)[:top_k]
            return [RecalledPattern.from_entry(e) for e in ranked]
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:  # noqa: BLE001
            return []

    @staticmethod
    def _format_recall_block(recalls: list[Any], budget_tokens: int = 800) -> str:
        """Format *recalls* into a markdown block, truncating to *budget_tokens*.

        Tokens are estimated at ``DEFAULT_CHARS_PER_TOKEN`` characters per token.
        Entries are dropped from the tail when the budget is exceeded.
        """
        from vaig.core.config import DEFAULT_CHARS_PER_TOKEN

        lines: list[str] = ["## Prior-run memory (top similar cases)", ""]
        chars_used = 0
        chars_budget = budget_tokens * DEFAULT_CHARS_PER_TOKEN

        for i, recall in enumerate(recalls, start=1):
            ts = recall.timestamp.strftime("%Y-%m-%d %H:%M UTC")
            cluster_part = f" (cluster={recall.cluster})" if recall.cluster else ""
            header = f"{i}. {ts} — \"{recall.title}\"{cluster_part}"
            resolution = f"   Resolution: {recall.resolution}" if recall.resolution else ""
            outcome = f"   Fix outcome: {recall.fix_outcome}" if recall.fix_outcome else ""

            entry_lines = [header]
            if resolution:
                entry_lines.append(resolution)
            if outcome:
                entry_lines.append(outcome)
            entry_lines.append("")

            entry_text = "\n".join(entry_lines)
            # Always include the first entry even if it exceeds the budget.
            if i > 1 and chars_used + len(entry_text) > chars_budget:
                break
            lines.extend(entry_lines)
            chars_used += len(entry_text)

        return "\n".join(lines).rstrip()

    def _augment_system_instruction(self, base: str, query: str) -> str:
        """Return *base* extended with a prior-run memory recall block.

        When the :class:`~vaig.core.memory.pattern_store.PatternMemoryStore`
        is empty or memory is disabled, *base* is returned unchanged —
        guaranteeing byte-for-byte prompt equality with the pre-memory state.
        """
        from vaig.core.config import get_settings

        if not self._recall_enabled():
            return base

        try:
            cfg = get_settings()
            budget = cfg.memory.recall_budget_tokens
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:  # noqa: BLE001
            budget = 800

        recalls = self._recall_patterns(query)
        if not recalls:
            return base

        recall_block = self._format_recall_block(recalls, budget_tokens=budget)
        return base + "\n\n" + recall_block

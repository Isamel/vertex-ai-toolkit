"""Orchestrator — coordinates multi-agent execution for skill-based tasks."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

from vaig.agents.base import AgentConfig, AgentResult, BaseAgent
from vaig.agents.mixins import OnToolCall
from vaig.agents.specialist import SpecialistAgent
from vaig.agents.tool_aware import ToolAwareAgent
from vaig.core.async_utils import gather_with_errors
from vaig.core.client import StreamResult
from vaig.core.event_bus import EventBus
from vaig.core.events import OrchestratorPhaseCompleted, OrchestratorToolsCompleted
from vaig.core.exceptions import MaxIterationsError, VaigAuthError, VAIGError
from vaig.core.language import (
    detect_language,
    inject_autopilot_into_config,
    inject_language_into_config,
)
from vaig.core.pricing import calculate_cost
from vaig.core.prompt_defense import ANTI_INJECTION_RULE, wrap_untrusted_content
from vaig.skills.base import BaseSkill, SkillPhase, SkillResult
from vaig.tools.base import ToolRegistry

if TYPE_CHECKING:
    from vaig.core.config import Settings
    from vaig.core.protocols import GeminiClientProtocol
    from vaig.core.tool_call_store import ToolCallStore

logger = logging.getLogger(__name__)


class OnAgentProgress(Protocol):
    """Callback protocol invoked when an agent phase starts or ends.

    Args:
        agent_name: Display name of the agent (e.g. ``"health_gatherer"``).
        agent_index: Zero-based position in the execution pipeline.
        total_agents: Total number of agents in the pipeline.
        event: ``"start"`` before execution, ``"end"`` after completion.
    """

    def __call__(
        self,
        agent_name: str,
        agent_index: int,
        total_agents: int,
        event: Literal["start", "end"],
    ) -> None: ...


def _fire_agent_progress(
    callback: OnAgentProgress | None,
    agent_name: str,
    agent_index: int,
    total_agents: int,
    event: Literal["start", "end"],
) -> None:
    """Safely invoke an *on_agent_progress* callback, swallowing errors."""
    if callback is None:
        return
    try:
        callback(agent_name, agent_index, total_agents, event)
    except Exception:
        logger.debug(
            "on_agent_progress callback error (swallowed): agent=%s event=%s",
            agent_name,
            event,
            exc_info=True,
        )


# ── Gatherer output validation constants ────────────────────────────────
DEFAULT_MIN_CONTENT_CHARS = 200

EMPTY_MARKERS: tuple[str, ...] = (
    "n/a",
    "data not available",
    "no data",
    "not available",
    "none found",
    "no results",
    "no data available",
    "no information available",
    "unavailable",
)


@dataclass
class OrchestratorResult:
    """Aggregated result from orchestrated multi-agent execution."""

    skill_name: str
    phase: SkillPhase
    agent_results: list[AgentResult] = field(default_factory=list)
    synthesized_output: str = ""
    success: bool = True
    total_usage: dict[str, int] = field(default_factory=dict)
    structured_report: Any = field(default=None)
    """Optional parsed HealthReport (or other structured model) retained
    from ``post_process_report``.  Set by the pipeline when the reporter
    agent produces valid JSON; consumed by the CLI for Rich rendering
    and ``--summary`` mode.  ``None`` when parsing failed or no reporter
    agent was involved."""
    run_cost_usd: float = 0.0
    """Accumulated USD cost for this pipeline run.

    Computed by summing :func:`~vaig.core.pricing.calculate_cost` for every
    agent step that completed before the pipeline returned.  Zero when pricing
    data is unavailable for the model or when no agents ran.
    """
    budget_exceeded: bool = False
    """``True`` when the pipeline was halted because ``run_cost_usd`` exceeded
    ``settings.budget.max_cost_per_run``.  Always ``False`` when the per-run
    budget is disabled (``max_cost_per_run == 0.0``).
    """

    def to_skill_result(self) -> SkillResult:
        """Convert to a SkillResult for the skill system."""
        return SkillResult(
            phase=self.phase,
            success=self.success,
            output=self.synthesized_output,
            metadata={
                "agents_used": [r.agent_name for r in self.agent_results],
                "total_usage": self.total_usage,
            },
        )


@dataclass
class GathererValidationResult:
    """Rich validation result from gatherer output checking.

    Separates two classes of issues:
    - **missing_sections**: section header not found at all.
    - **shallow_sections**: header found but content is empty, too short,
      or consists only of empty-marker phrases (e.g. "N/A").
    """

    missing_sections: list[str] = field(default_factory=list)
    shallow_sections: list[str] = field(default_factory=list)

    @property
    def needs_retry(self) -> bool:
        """Return ``True`` when at least one section is missing or shallow."""
        return bool(self.missing_sections or self.shallow_sections)


class Orchestrator:
    """Orchestrates multi-agent execution for skills.

    The orchestrator:
    1. Takes a skill and creates its specialized agents
    2. Routes tasks to the appropriate agent(s)
    3. Coordinates sequential or parallel execution
    4. Synthesizes results from multiple agents

    Execution strategies:
    - Sequential: Each agent builds on the previous agent's output
    - Fan-out: All agents work independently, results are merged
    - Lead-delegate: A lead agent delegates subtasks to specialists
    """

    def __init__(self, client: GeminiClientProtocol, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._agents: dict[str, BaseAgent] = {}

    def _build_previous_agent_summary(
        self,
        agent_role: str,
        prev_result: AgentResult,
    ) -> str:
        """Build a markdown summary of a previous agent's execution.

        Consolidates the repeated context-building pattern used by all
        sequential execution paths (sync/async, with/without tools).
        """
        tools_summary = _build_tools_summary(agent_role, prev_result.metadata)
        return (
            f"## Previous Analysis ({agent_role})\n\n"
            f"{wrap_untrusted_content(prev_result.content)}"
            f"{wrap_untrusted_content(tools_summary) if tools_summary else ''}"
        )

    def create_agents_for_skill(
        self,
        skill: BaseSkill,
        tool_registry: ToolRegistry | None = None,
        *,
        agent_configs: list[dict[str, Any]] | None = None,
    ) -> list[BaseAgent]:
        """Create agents based on a skill's configuration.

        Each skill defines its own agent configs (roles, models, instructions).
        When an agent config has ``requires_tools=True`` **and** a *tool_registry*
        is provided, a :class:`ToolAwareAgent` is created instead of a
        :class:`SpecialistAgent`.  This keeps the method fully backward-compatible
        — callers that omit *tool_registry* get the same behaviour as before.

        Args:
            skill: The skill whose agents to create.
            tool_registry: Optional tool registry for tool-aware agents.
            agent_configs: Optional pre-built agent config dicts.  When
                provided, these are used instead of calling
                ``skill.get_agents_config()``.  This allows callers
                (e.g. :meth:`execute_with_tools`) to inject runtime
                modifications like language instructions before agent
                creation.
        """
        self._agents.clear()
        agents: list[BaseAgent] = []

        configs = agent_configs if agent_configs is not None else skill.get_agents_config()

        for config_dict in configs:
            if config_dict.get("requires_tools") and tool_registry is not None:
                model = config_dict.get("model", "gemini-2.5-pro")
                agent: BaseAgent = ToolAwareAgent.from_config_dict(
                    config_dict, model, tool_registry, self._client,
                )
                logger.info(
                    "Created ToolAwareAgent: %s (role=%s, model=%s)",
                    agent.name, agent.role, agent.model,
                )
            else:
                agent = SpecialistAgent.from_config_dict(config_dict, self._client)
                logger.info(
                    "Created SpecialistAgent: %s (role=%s, model=%s)",
                    agent.name, agent.role, agent.model,
                )

            self._agents[agent.name] = agent
            agents.append(agent)

        return agents

    def execute_sequential(
        self,
        skill: BaseSkill,
        phase: SkillPhase,
        context: str,
        user_input: str,
    ) -> OrchestratorResult:
        """Execute agents sequentially — each builds on the previous agent's output.

        This is the most common pattern: agent A analyzes → agent B synthesizes → agent C reports.
        """
        agents = self.create_agents_for_skill(skill)
        result = OrchestratorResult(
            skill_name=skill.get_metadata().name,
            phase=phase,
        )

        # Build the initial phase prompt from the skill
        current_context = context
        prompt = skill.get_phase_prompt(phase, context, user_input)
        context_chain: list[str] = []

        run_cost_usd: float = 0.0
        max_cost_per_run: float = self._settings.budget.max_cost_per_run
        failure_counts: dict[str, int] = {}
        max_failures: int = self._settings.agents.max_failures_before_fallback

        for i, agent in enumerate(agents):
            logger.info("Sequential step %d/%d: agent=%s", i + 1, len(agents), agent.name)
            logger.debug(
                "Agent '%s' config: role=%s, model=%s",
                agent.name, agent.role, agent.model,
            )

            # First agent gets the original prompt; subsequent agents get the accumulated context
            if i == 0:
                _seq_ctx = current_context
                agent_result = agent.execute(prompt, context=_seq_ctx)
            else:
                # Feed ALL previous agents' outputs as accumulated context
                prev = result.agent_results[-1]
                context_chain.append(
                    self._build_previous_agent_summary(agents[i - 1].role, prev),
                )
                accumulated = (
                    f"{current_context}\n\n"
                    + "\n\n---\n\n".join(context_chain)
                )
                _seq_ctx = accumulated
                agent_result = agent.execute(prompt, context=_seq_ctx)

            result.agent_results.append(agent_result)
            _accumulate_usage(result, agent_result)

            # ── Cost circuit breaker ──────────────────────────────────────
            run_cost_usd += _compute_step_cost(agent_result, agent.model)
            result.run_cost_usd = run_cost_usd
            if max_cost_per_run > 0.0 and run_cost_usd > max_cost_per_run:
                logger.warning(
                    "Cost circuit breaker triggered after agent %s: "
                    "run_cost_usd=%.6f > max_cost_per_run=%.6f",
                    agent.name, run_cost_usd, max_cost_per_run,
                )
                result.success = False
                result.budget_exceeded = True
                result.synthesized_output = (
                    f"[WARNING] Pipeline halted: accumulated run cost "
                    f"${run_cost_usd:.4f} exceeded budget ${max_cost_per_run:.4f}.\n\n"
                    + (result.agent_results[-1].content if result.agent_results else "")
                )
                return result

            logger.debug(
                "Agent '%s' finished: success=%s, tokens=%s",
                agent.name,
                agent_result.success,
                agent_result.usage.get("total_tokens", "?"),
            )

            if not agent_result.success:
                # ── Model fallback on repeated failures ───────────────────
                error_type = agent_result.metadata.get("error_type", "")
                if max_failures > 0 and error_type in {"rate_limit", "connection"}:
                    failure_counts[agent.name] = failure_counts.get(agent.name, 0) + 1
                    if failure_counts[agent.name] >= max_failures:
                        fallback_model = self._settings.models.fallback
                        logger.warning(
                            "Agent %s failed %d time(s) with error_type=%s — "
                            "switching to fallback model %s",
                            agent.name, failure_counts[agent.name],
                            error_type, fallback_model,
                        )
                        agent._config.model = fallback_model
                        # ── Inline retry with fallback model ─────────────
                        logger.info(
                            "Retrying agent %s inline with fallback model %s",
                            agent.name, fallback_model,
                        )
                        retry_ar = agent.execute(prompt, context=_seq_ctx)
                        run_cost_usd += _compute_step_cost(retry_ar, agent.model)
                        result.run_cost_usd = run_cost_usd
                        _accumulate_usage(result, retry_ar)
                        result.agent_results[-1] = retry_ar
                        agent_result = retry_ar
                        if not retry_ar.success:
                            result.success = False
                            logger.warning(
                                "Agent %s fallback retry also failed: %s",
                                agent.name, retry_ar.content,
                            )
                            break
                        # Retry succeeded — reset failure counter
                        failure_counts.pop(agent.name, None)
                else:
                    result.success = False
                    logger.warning("Agent %s failed: %s", agent.name, agent_result.content)
                    break
            else:
                # Reset failure counter on success
                failure_counts.pop(agent.name, None)

        # The final agent's output is the synthesized result
        if result.agent_results:
            result.synthesized_output = result.agent_results[-1].content

        return result

    def execute_fanout(
        self,
        skill: BaseSkill,
        phase: SkillPhase,
        context: str,
        user_input: str,
    ) -> OrchestratorResult:
        """Execute all agents concurrently and merge their outputs.

        Each agent gets the same prompt and context, running in parallel
        via a :class:`~concurrent.futures.ThreadPoolExecutor`.  Results are
        collected in submission order for deterministic merging.  Usage
        is accumulated serially after all futures complete.
        """
        agents = self.create_agents_for_skill(skill)
        result = OrchestratorResult(
            skill_name=skill.get_metadata().name,
            phase=phase,
        )

        prompt = skill.get_phase_prompt(phase, context, user_input)

        # Submit all agents concurrently, collect in submission order
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(agents)) as executor:
            futures = []
            for agent in agents:
                logger.info("Fan-out: submitting agent=%s", agent.name)
                futures.append(executor.submit(agent.execute, prompt, context=context))

            for agent, future in zip(agents, futures, strict=True):
                try:
                    agent_result = future.result()
                except Exception:
                    logger.exception(
                        "Agent %s raised an exception during fan-out execution",
                        agent.name,
                    )
                    agent_result = AgentResult(
                        agent_name=agent.name,
                        content=f"Agent '{agent.name}' failed with an unexpected error.",
                        success=False,
                    )
                result.agent_results.append(agent_result)
                if not agent_result.success:
                    logger.warning("Agent %s failed (non-fatal in fan-out): %s", agent.name, agent_result.content)

        # Accumulate usage serially after all agents complete
        for agent_result in result.agent_results:
            _accumulate_usage(result, agent_result)

        # Merge all agent outputs
        result.success = any(r.success for r in result.agent_results)
        result.synthesized_output = self._merge_agent_outputs(result.agent_results)

        return result

    def execute_single(
        self,
        prompt: str,
        *,
        context: str = "",
        system_instruction: str = "",
        model_id: str | None = None,
        stream: bool = False,
    ) -> AgentResult | StreamResult:
        """Execute with a single ad-hoc agent (no skill, direct chat).

        Used for the general chat mode when no skill is active.

        When *stream* is ``True``, returns a :class:`StreamResult` that
        is iterable (yields ``str`` chunks) and exposes ``.usage`` after
        iteration completes.
        """
        config = AgentConfig(
            name="assistant",
            role="General Assistant",
            system_instruction=system_instruction or self.default_system_instruction(),
            model=model_id or self._settings.models.default,
        )

        agent = SpecialistAgent(config, self._client)

        if stream:
            # Return StreamResult directly so callers can access .usage
            # after iteration.  We build the prompt the same way the agent
            # would, but skip the agent's generator wrapper.
            full_prompt = agent._build_prompt(prompt, context)
            return self._client.generate_stream(
                full_prompt,
                system_instruction=config.system_instruction,
                model_id=config.model,
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
            )
        return agent.execute(prompt, context=context)

    def execute_skill_phase(
        self,
        skill: BaseSkill,
        phase: SkillPhase,
        context: str,
        user_input: str,
        *,
        strategy: str = "sequential",
    ) -> SkillResult:
        """High-level: execute a skill phase with the given strategy.

        This is the main entry point for skill-based execution.
        """
        skill_name = skill.get_metadata().name
        logger.info(
            "Executing skill=%s phase=%s strategy=%s",
            skill_name,
            phase,
            strategy,
        )

        try:
            t0 = time.perf_counter()
            if strategy == "fanout":
                orch_result = self.execute_fanout(skill, phase, context, user_input)
            else:
                orch_result = self.execute_sequential(skill, phase, context, user_input)

            # Telemetry: emit orchestrator event
            try:
                duration_ms = (time.perf_counter() - t0) * 1000
                EventBus.get().emit(
                    OrchestratorPhaseCompleted(
                        skill=skill_name,
                        phase=phase.value if hasattr(phase, "value") else str(phase),
                        strategy=strategy,
                        duration_ms=duration_ms,
                        is_async=False,
                    )
                )
            except Exception:  # noqa: BLE001
                pass

            return orch_result.to_skill_result()
        except (VaigAuthError, VAIGError):
            raise  # Let known errors propagate with their actionable messages
        except Exception as exc:
            logger.error("Orchestrator error in execute_skill_phase: %s", exc, exc_info=True)
            raise VAIGError(f"Pipeline execution failed: {exc}") from exc

    def execute_with_tools(
        self,
        query: str,
        skill: BaseSkill,
        tool_registry: ToolRegistry,
        *,
        strategy: str = "sequential",
        is_autopilot: bool | None = None,
        on_tool_call: OnToolCall | None = None,
        tool_call_store: ToolCallStore | None = None,
        on_agent_progress: OnAgentProgress | None = None,
    ) -> OrchestratorResult:
        """Execute a skill with tool-aware agents.

        This is the entry point for tool-backed execution.  It creates a
        *mixed* agent list (some :class:`ToolAwareAgent`, some
        :class:`SpecialistAgent`) based on each agent's ``requires_tools``
        flag, then routes execution through the requested strategy.

        Unlike :meth:`execute_skill_phase` this method does **not** use
        skill phases — it sends *query* directly to the agent pipeline.

        Args:
            query: The user query / task to execute.
            skill: The skill defining agent configs and system prompts.
            tool_registry: Pre-configured tool registry for tool-aware agents.
            strategy: ``"sequential"`` (default), ``"fanout"``, or ``"single"``.
            is_autopilot: Autopilot detection result — ``True``, ``False``,
                or ``None`` (unknown).  When ``True``, an Autopilot context
                instruction is injected into each agent's system prompt.
            on_tool_call: Optional callback invoked after each tool
                execution.  Threaded through to each agent's tool loop.
            tool_call_store: Optional store for recording full tool call
                results for metrics and feedback.  When provided, the
                store is passed to each ``ToolAwareAgent.execute()`` call.
            on_agent_progress: Optional callback invoked when each agent
                starts and finishes execution.  Used by the CLI to show
                live progress indicators.

        Returns:
            :class:`OrchestratorResult` with the aggregated outcome.
        """
        try:
            return self._execute_with_tools_impl(
                query, skill, tool_registry,
                strategy=strategy,
                is_autopilot=is_autopilot,
                on_tool_call=on_tool_call,
                tool_call_store=tool_call_store,
                on_agent_progress=on_agent_progress,
            )
        except (VaigAuthError, VAIGError):
            raise  # Let known errors propagate with their actionable messages
        except Exception as exc:
            logger.error("Orchestrator error in execute_with_tools: %s", exc, exc_info=True)
            raise VAIGError(f"Pipeline execution failed: {exc}") from exc

    def _execute_with_tools_impl(
        self,
        query: str,
        skill: BaseSkill,
        tool_registry: ToolRegistry,
        *,
        strategy: str = "sequential",
        is_autopilot: bool | None = None,
        on_tool_call: OnToolCall | None = None,
        tool_call_store: ToolCallStore | None = None,
        on_agent_progress: OnAgentProgress | None = None,
    ) -> OrchestratorResult:
        """Inner implementation of execute_with_tools (no error boundary)."""
        logger.info(
            "execute_with_tools: skill=%s strategy=%s",
            skill.get_metadata().name,
            strategy,
        )

        t0_ewt = time.perf_counter()

        # ── Create a shared tool result cache for dedup ──────
        from vaig.core.cache import ToolResultCache

        tool_result_cache = ToolResultCache()

        # ── Start a tool call store run if provided ──────────
        if tool_call_store is not None:
            tool_call_store.start_run()

        # ── Dynamic language detection & injection ───────────
        # Detect the user's language and inject a language instruction
        # into each agent's system prompt so the entire pipeline responds
        # in the same language as the query.
        lang = detect_language(query)
        agent_configs = skill.get_agents_config()
        if lang != "en":
            inject_language_into_config(agent_configs, lang)
            logger.info(
                "Language detected: %s — injected language instruction into %d agent(s)",
                lang,
                len(agent_configs),
            )

        # ── Autopilot context injection ──────────────────────
        # When the cluster is confirmed as GKE Autopilot, inject an
        # instruction so agents adapt their behaviour (skip node ops,
        # focus on workload health, etc.).
        if is_autopilot:
            inject_autopilot_into_config(agent_configs, is_autopilot)
            logger.info(
                "GKE Autopilot detected — injected Autopilot instruction into %d agent(s)",
                len(agent_configs),
            )

        # ── Auto-detect parallel_sequential from agent configs ───────────
        # When any agent config declares a ``parallel_group`` key and the
        # caller has not explicitly requested a non-sequential strategy,
        # automatically upgrade to ``parallel_sequential`` so that skills
        # returning ``get_parallel_agents_config()`` from ``get_agents_config()``
        # work without requiring every call-site to pass the strategy explicitly.
        if strategy == "sequential" and any(
            cfg.get("parallel_group") for cfg in agent_configs
        ):
            logger.warning(
                "Auto-detected parallel_group agents — upgrading strategy to parallel_sequential",
            )
            strategy = "parallel_sequential"

        agents = self.create_agents_for_skill(
            skill, tool_registry, agent_configs=agent_configs,
        )
        result = OrchestratorResult(
            skill_name=skill.get_metadata().name,
            phase=SkillPhase.EXECUTE,
        )

        known_strategies = {"sequential", "fanout", "single", "parallel_sequential"}
        if strategy not in known_strategies:
            logger.warning(
                "Unknown strategy '%s' — falling back to sequential. "
                "Valid strategies: %s",
                strategy,
                ", ".join(sorted(known_strategies)),
            )

        if strategy == "parallel_sequential":
            result = self._execute_parallel_then_sequential(
                agents=agents,
                skill=skill,
                query=query,
                result=result,
                on_agent_progress=on_agent_progress,
                on_tool_call=on_tool_call,
                tool_call_store=tool_call_store,
                tool_result_cache=tool_result_cache,
            )

        elif strategy == "fanout":
            # Submit all agents concurrently, collect in submission order
            total = len(agents)
            with concurrent.futures.ThreadPoolExecutor(max_workers=total) as executor:
                futures = []
                for idx, agent in enumerate(agents):
                    logger.info("Fan-out (tools): submitting agent=%s", agent.name)
                    _fire_agent_progress(on_agent_progress, agent.name, idx, total, "start")
                    kw: dict[str, Any] = {"context": query}
                    if isinstance(agent, ToolAwareAgent):
                        if on_tool_call is not None:
                            kw["on_tool_call"] = on_tool_call
                        if tool_call_store is not None:
                            kw["tool_call_store"] = tool_call_store
                        kw["tool_result_cache"] = tool_result_cache
                    futures.append(executor.submit(agent.execute, query, **kw))

                for idx, (agent, future) in enumerate(zip(agents, futures, strict=True)):
                    try:
                        agent_result = future.result()
                    except Exception:
                        logger.exception(
                            "Agent %s raised an exception during fan-out (tools) execution",
                            agent.name,
                        )
                        agent_result = AgentResult(
                            agent_name=agent.name,
                            content=f"Agent '{agent.name}' failed with an unexpected error.",
                            success=False,
                        )
                    _fire_agent_progress(on_agent_progress, agent.name, idx, total, "end")
                    result.agent_results.append(agent_result)
                    if not agent_result.success:
                        logger.warning(
                            "Agent %s failed (non-fatal in fan-out): %s",
                            agent.name, agent_result.content,
                        )

            # Accumulate usage serially after all agents complete
            for agent_result in result.agent_results:
                _accumulate_usage(result, agent_result)

            result.success = any(r.success for r in result.agent_results)
            result.synthesized_output = self._merge_agent_outputs(result.agent_results)

        elif strategy == "single":
            if agents:
                kw_single: dict[str, Any] = {}
                if isinstance(agents[0], ToolAwareAgent):
                    if on_tool_call is not None:
                        kw_single["on_tool_call"] = on_tool_call
                    if tool_call_store is not None:
                        kw_single["tool_call_store"] = tool_call_store
                    kw_single["tool_result_cache"] = tool_result_cache
                _fire_agent_progress(on_agent_progress, agents[0].name, 0, 1, "start")
                try:
                    agent_result = agents[0].execute(query, **kw_single)
                finally:
                    _fire_agent_progress(on_agent_progress, agents[0].name, 0, 1, "end")
                result.agent_results.append(agent_result)
                _accumulate_usage(result, agent_result)
                result.success = agent_result.success
                result.synthesized_output = agent_result.content
            else:
                result.success = False
                result.synthesized_output = "No agents created for skill."

        else:  # sequential (default)
            current_context = ""
            context_chain: list[str] = []
            required_sections = skill.get_required_output_sections()

            run_cost_usd: float = 0.0
            max_cost_per_run: float = self._settings.budget.max_cost_per_run
            failure_counts: dict[str, int] = {}
            max_failures: int = self._settings.agents.max_failures_before_fallback

            for i, agent in enumerate(agents):
                logger.info(
                    "Sequential (tools) step %d/%d: agent=%s",
                    i + 1, len(agents), agent.name,
                )
                logger.debug(
                    "Agent '%s' config: role=%s, model=%s",
                    agent.name, agent.role, agent.model,
                )
                if i > 0 and result.agent_results:
                    prev = result.agent_results[-1]
                    context_chain.append(
                        self._build_previous_agent_summary(agents[i - 1].role, prev),
                    )
                    current_context = "\n\n---\n\n".join(context_chain)

                kw_seq: dict[str, Any] = {"context": current_context}
                if isinstance(agent, ToolAwareAgent):
                    if on_tool_call is not None:
                        kw_seq["on_tool_call"] = on_tool_call
                    if tool_call_store is not None:
                        kw_seq["tool_call_store"] = tool_call_store
                    kw_seq["tool_result_cache"] = tool_result_cache
                    if i == 0 and required_sections:
                        kw_seq["required_sections"] = required_sections

                _fire_agent_progress(on_agent_progress, agent.name, i, len(agents), "start")
                try:
                    agent_result = agent.execute(query, **kw_seq)
                finally:
                    _fire_agent_progress(on_agent_progress, agent.name, i, len(agents), "end")
                result.agent_results.append(agent_result)
                _accumulate_usage(result, agent_result)

                # ── Cost circuit breaker ──────────────────────────────────
                run_cost_usd += _compute_step_cost(agent_result, agent.model)
                result.run_cost_usd = run_cost_usd
                if max_cost_per_run > 0.0 and run_cost_usd > max_cost_per_run:
                    logger.warning(
                        "Cost circuit breaker triggered after agent %s: "
                        "run_cost_usd=%.6f > max_cost_per_run=%.6f",
                        agent.name, run_cost_usd, max_cost_per_run,
                    )
                    result.success = False
                    result.budget_exceeded = True
                    result.synthesized_output = (
                        f"[WARNING] Pipeline halted: accumulated run cost "
                        f"${run_cost_usd:.4f} exceeded budget ${max_cost_per_run:.4f}.\n\n"
                        + (result.agent_results[-1].content if result.agent_results else "")
                    )
                    return result

                logger.debug(
                    "Agent '%s' finished: success=%s, tokens=%s",
                    agent.name,
                    agent_result.success,
                    agent_result.usage.get("total_tokens", "?"),
                )

                if not agent_result.success:
                    # ── Model fallback on repeated failures ───────────────
                    error_type = agent_result.metadata.get("error_type", "")
                    if max_failures > 0 and error_type in {"rate_limit", "connection"}:
                        failure_counts[agent.name] = failure_counts.get(agent.name, 0) + 1
                        if failure_counts[agent.name] >= max_failures:
                            fallback_model = self._settings.models.fallback
                            logger.warning(
                                "Agent %s failed %d time(s) with error_type=%s — "
                                "switching to fallback model %s",
                                agent.name, failure_counts[agent.name],
                                error_type, fallback_model,
                            )
                            agent._config.model = fallback_model
                            # ── Inline retry with fallback model ─────────
                            logger.info(
                                "Retrying agent %s inline with fallback model %s",
                                agent.name, fallback_model,
                            )
                            retry_ar = agent.execute(query, **kw_seq)
                            run_cost_usd += _compute_step_cost(retry_ar, agent.model)
                            result.run_cost_usd = run_cost_usd
                            _accumulate_usage(result, retry_ar)
                            result.agent_results[-1] = retry_ar
                            agent_result = retry_ar
                            if not retry_ar.success:
                                result.success = False
                                logger.warning(
                                    "Agent %s fallback retry also failed: %s",
                                    agent.name, retry_ar.content,
                                )
                                break
                            # Retry succeeded — reset failure counter
                            failure_counts.pop(agent.name, None)
                    else:
                        result.success = False
                        logger.warning("Agent %s failed: %s", agent.name, agent_result.content)
                        break
                else:
                    # Reset failure counter on success
                    failure_counts.pop(agent.name, None)

                # ── Gatherer output validation + mandatory retry ──────
                if i == 0 and required_sections and agent_result.success:
                    validation = self._validate_gatherer_output(
                        agent_result.content, required_sections,
                    )
                    if validation.needs_retry:
                        all_issues = validation.missing_sections + validation.shallow_sections
                        logger.warning(
                            "Gatherer output has %d issue(s): "
                            "missing=%s, shallow=%s — retrying once",
                            len(all_issues),
                            ", ".join(validation.missing_sections) or "(none)",
                            ", ".join(validation.shallow_sections) or "(none)",
                        )
                        logger.debug(
                            "Validation failed: required_sections=%s, "
                            "missing=%s, shallow=%s",
                            required_sections,
                            validation.missing_sections,
                            validation.shallow_sections,
                        )
                        retry_prompt = self._build_retry_prompt(
                            query,
                            validation.missing_sections,
                            shallow_sections=validation.shallow_sections,
                        )
                    else:
                        logger.info(
                            "Gatherer validation passed — running mandatory "
                            "deepening second pass for thoroughness",
                        )
                        retry_prompt = self._build_deepening_prompt(
                            query, agent_result.content,
                        )

                    logger.debug("Retry prompt: %s", retry_prompt[:200])

                    is_deepening = not validation.needs_retry
                    if not is_deepening:
                        # Validation retry: reset agent state so it starts fresh
                        agent.reset()

                    # Use configurable max_iterations for the second pass
                    kw_retry: dict[str, Any] = {"context": ""}
                    if isinstance(agent, ToolAwareAgent):
                        if on_tool_call is not None:
                            kw_retry["on_tool_call"] = on_tool_call
                        if tool_call_store is not None:
                            kw_retry["tool_call_store"] = tool_call_store
                        kw_retry["tool_result_cache"] = tool_result_cache
                        if required_sections:
                            kw_retry["required_sections"] = required_sections

                    original_max_iters: int | None = None
                    if (
                        isinstance(agent, ToolAwareAgent)
                        and hasattr(agent, "_max_iterations")
                    ):
                        original_max_iters = agent._max_iterations
                        agent._max_iterations = (
                            self._settings.agents.max_iterations_retry
                        )

                    try:
                        retry_result = agent.execute(retry_prompt, **kw_retry)
                    except MaxIterationsError:
                        if is_deepening:
                            # Deepening is optional — first pass was valid.
                            # Fall back to first-pass output instead of
                            # crashing the pipeline.
                            logger.warning(
                                "Deepening second pass for agent %s hit "
                                "max_iterations_retry limit; falling back "
                                "to first-pass output",
                                agent.name,
                            )
                            # Keep first-pass result — already in
                            # result.agent_results[-1]
                            logger.info(
                                "Agent %s continuing with first-pass "
                                "output — tokens=%s",
                                agent.name,
                                agent_result.usage.get(
                                    "total_tokens", "?",
                                ),
                            )
                        else:
                            # Genuine retry (validation failed) — propagate
                            raise
                    else:
                        # No exception — process retry result normally
                        _accumulate_usage(result, retry_result)
                        run_cost_usd += _compute_step_cost(retry_result, agent.model)
                        result.run_cost_usd = run_cost_usd
                        if max_cost_per_run > 0.0 and run_cost_usd > max_cost_per_run:
                            logger.warning(
                                "Cost circuit breaker triggered after gatherer retry %s: "
                                "run_cost_usd=%.6f > max_cost_per_run=%.6f",
                                agent.name, run_cost_usd, max_cost_per_run,
                            )
                            result.success = False
                            result.budget_exceeded = True
                            result.synthesized_output = (
                                f"[WARNING] Pipeline halted: accumulated run cost "
                                f"${run_cost_usd:.4f} exceeded budget ${max_cost_per_run:.4f}.\n\n"
                                + (result.agent_results[-1].content if result.agent_results else "")
                            )
                            return result

                        if is_deepening:
                            # Merge: concatenate first-pass + second-pass
                            # content
                            merged_content = (
                                agent_result.content
                                + "\n\n"
                                + retry_result.content
                            )
                            retry_result = AgentResult(
                                agent_name=retry_result.agent_name,
                                content=merged_content,
                                success=retry_result.success,
                                usage=retry_result.usage,
                                metadata=retry_result.metadata,
                            )

                        result.agent_results[-1] = retry_result

                        if not retry_result.success:
                            result.success = False
                            logger.warning(
                                "Agent %s retry also failed: %s",
                                agent.name, retry_result.content,
                            )
                            break
                        logger.info(
                            "Agent %s retry succeeded — tokens=%s",
                            agent.name,
                            retry_result.usage.get("total_tokens", "?"),
                        )
                    finally:
                        if (
                            isinstance(agent, ToolAwareAgent)
                            and original_max_iters is not None
                        ):
                            agent._max_iterations = original_max_iters

                # ── Reporter output validation + retry ──────
                is_reporter = "report" in getattr(agent, "role", "").lower()
                if i == len(agents) - 1 and agent_result.success and is_reporter:
                    # ── Structured output pre-validation ──────
                    # When the reporter uses response_schema, validate raw
                    # JSON BEFORE post-processing.  If invalid, retry with
                    # a JSON-focused prompt so post_process_report receives
                    # well-formed input.
                    schema_cls = getattr(
                        getattr(agent, "config", None),
                        "response_schema",
                        None,
                    )
                    if (
                        schema_cls is not None
                        and isinstance(schema_cls, type)
                        and hasattr(schema_cls, "model_validate_json")
                    ):
                        import json as _json

                        try:
                            _json.loads(agent_result.content)
                            schema_cls.model_validate_json(agent_result.content)
                        except Exception as schema_exc:
                            logger.warning(
                                "Reporter %s structured output failed "
                                "schema validation (%s) — retrying with "
                                "JSON-focused prompt",
                                agent.name,
                                schema_exc,
                            )
                            json_retry_prompt = (
                                "Your previous response was not valid JSON "
                                "conforming to the required schema. "
                                f"Error: {schema_exc}\n\n"
                                "Regenerate the report as valid JSON that "
                                "strictly conforms to the schema. "
                                "Do NOT include markdown fences or commentary "
                                "— output ONLY the raw JSON object.\n\n"
                                f"## Previous Analysis\n\n{current_context}"
                            )
                            agent.reset()
                            kw_js: dict[str, Any] = {"context": ""}
                            if isinstance(agent, ToolAwareAgent):
                                if on_tool_call is not None:
                                    kw_js["on_tool_call"] = on_tool_call
                                if tool_call_store is not None:
                                    kw_js["tool_call_store"] = tool_call_store
                                kw_js["tool_result_cache"] = tool_result_cache
                            json_retry_result = agent.execute(
                                json_retry_prompt, **kw_js,
                            )
                            _accumulate_usage(result, json_retry_result)
                            run_cost_usd += _compute_step_cost(json_retry_result, agent.model)
                            result.run_cost_usd = run_cost_usd
                            if max_cost_per_run > 0.0 and run_cost_usd > max_cost_per_run:
                                logger.warning(
                                    "Cost circuit breaker triggered after JSON retry %s: "
                                    "run_cost_usd=%.6f > max_cost_per_run=%.6f",
                                    agent.name, run_cost_usd, max_cost_per_run,
                                )
                                result.success = False
                                result.budget_exceeded = True
                                result.synthesized_output = (
                                    f"[WARNING] Pipeline halted: accumulated run cost "
                                    f"${run_cost_usd:.4f} exceeded budget ${max_cost_per_run:.4f}.\n\n"
                                    + (result.agent_results[-1].content if result.agent_results else "")
                                )
                                return result
                            if json_retry_result.success:
                                agent_result = json_retry_result
                                result.agent_results[-1] = agent_result
                                logger.info(
                                    "Reporter %s JSON retry succeeded",
                                    agent.name,
                                )
                            else:
                                logger.warning(
                                    "Reporter %s JSON retry also failed — "
                                    "proceeding with original output",
                                    agent.name,
                                )

                    # Retain the parsed structured report (if any)
                    # before post-processing converts it to Markdown.
                    # The CLI uses this for --summary and Rich rendering.
                    if (
                        schema_cls is not None
                        and hasattr(schema_cls, "model_validate_json")
                        and result.structured_report is None
                    ):
                        try:
                            result.structured_report = schema_cls.model_validate_json(
                                agent_result.content,
                            )
                        except Exception:
                            pass  # best-effort

                    # Post-process structured output (e.g. JSON → Markdown)
                    agent_result.content = skill.post_process_report(
                        agent_result.content,
                    )

                    finish_reason = agent_result.metadata.get("finish_reason", "")
                    md_issues = self._validate_reporter_output(agent_result.content)

                    if finish_reason == "MAX_TOKENS" or md_issues:
                        reasons: list[str] = []
                        if finish_reason == "MAX_TOKENS":
                            reasons.append("finish_reason=MAX_TOKENS")
                        if md_issues:
                            reasons.extend(md_issues)
                        logger.warning(
                            "Reporter output has %d issue(s): %s — retrying once",
                            len(reasons), "; ".join(reasons),
                        )

                        reporter_retry_prompt = (
                            "Your previous report had formatting issues "
                            f"({'; '.join(reasons)}). "
                            "Regenerate the report more concisely:\n"
                            "- Focus on CONFIRMED and HIGH severity findings only\n"
                            "- Close ALL table rows with a trailing |\n"
                            "- Close ALL code blocks with ```\n"
                            "- Keep total output under 12 000 tokens\n\n"
                            f"## Previous Analysis\n\n{current_context}"
                        )
                        agent.reset()
                        kw_rr: dict[str, Any] = {"context": ""}
                        if isinstance(agent, ToolAwareAgent):
                            if on_tool_call is not None:
                                kw_rr["on_tool_call"] = on_tool_call
                            if tool_call_store is not None:
                                kw_rr["tool_call_store"] = tool_call_store
                            kw_rr["tool_result_cache"] = tool_result_cache
                        reporter_retry = agent.execute(
                            reporter_retry_prompt, **kw_rr,
                        )
                        _accumulate_usage(result, reporter_retry)
                        run_cost_usd += _compute_step_cost(reporter_retry, agent.model)
                        result.run_cost_usd = run_cost_usd
                        if max_cost_per_run > 0.0 and run_cost_usd > max_cost_per_run:
                            result.success = False
                            result.budget_exceeded = True
                            result.synthesized_output = (
                                f"[WARNING] Cost budget ${max_cost_per_run:.4f} exceeded "
                                f"during reporter retry (actual: ${run_cost_usd:.4f})"
                            )
                            return result
                        result.agent_results[-1] = reporter_retry

                        if not reporter_retry.success:
                            result.success = False
                            logger.warning(
                                "Reporter %s retry also failed: %s",
                                agent.name, reporter_retry.content,
                            )
                            break
                        logger.info(
                            "Reporter %s retry succeeded — tokens=%s",
                            agent.name,
                            reporter_retry.usage.get("total_tokens", "?"),
                        )

            if result.agent_results:
                result.synthesized_output = result.agent_results[-1].content

        # Telemetry: emit orchestrator event for execute_with_tools
        try:
            duration_ms = (time.perf_counter() - t0_ewt) * 1000
            EventBus.get().emit(
                OrchestratorToolsCompleted(
                    skill=skill.get_metadata().name,
                    strategy=strategy,
                    agents_count=len(agents),
                    success=result.success,
                    duration_ms=duration_ms,
                    is_async=False,
                )
            )
        except Exception:  # noqa: BLE001
            pass

        return result

    def get_agent(self, name: str) -> BaseAgent | None:
        """Get a currently loaded agent by name."""
        return self._agents.get(name)

    def list_agents(self) -> list[str]:
        """List currently loaded agent names."""
        return list(self._agents.keys())

    def reset_agents(self) -> None:
        """Reset all agent conversation histories."""
        for agent in self._agents.values():
            agent.reset()

    # ── Async methods ──────────────────────────────────────────
    # These mirror the sync methods above but use ``asyncio.gather``
    # (via ``gather_with_errors``) instead of ``ThreadPoolExecutor``
    # for concurrent agent execution.  Agents that don't yet have
    # native async_execute methods are wrapped via
    # ``asyncio.to_thread()`` so the event loop stays unblocked.

    async def async_execute_single(
        self,
        prompt: str,
        *,
        context: str = "",
        system_instruction: str = "",
        model_id: str | None = None,
        stream: bool = False,
    ) -> AgentResult | StreamResult:
        """Async version of :meth:`execute_single`.

        Uses ``SpecialistAgent.async_execute()`` or
        ``GeminiClient.async_generate_stream()`` for non-blocking I/O.

        When *stream* is ``True``, returns a :class:`StreamResult` that
        supports ``async for`` iteration and exposes ``.usage`` after
        iteration completes.
        """

        config = AgentConfig(
            name="assistant",
            role="General Assistant",
            system_instruction=system_instruction or self.default_system_instruction(),
            model=model_id or self._settings.models.default,
        )

        agent = SpecialistAgent(config, self._client)

        if stream:
            full_prompt = agent._build_prompt(prompt, context)
            return await self._client.async_generate_stream(
                full_prompt,
                system_instruction=config.system_instruction,
                model_id=config.model,
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
            )
        return await agent.async_execute(prompt, context=context)

    async def async_execute_fanout(
        self,
        skill: BaseSkill,
        phase: SkillPhase,
        context: str,
        user_input: str,
    ) -> OrchestratorResult:
        """Async version of :meth:`execute_fanout`.

        Uses ``gather_with_errors()`` instead of ``ThreadPoolExecutor``
        for true async-native parallelism.  Each agent's ``execute``
        is wrapped in ``asyncio.to_thread()`` since agents don't yet
        have native ``async_execute`` methods.

        Args:
            skill: The skill defining agents.
            phase: Skill phase to execute.
            context: Context data for the agents.
            user_input: The user's original input.

        Returns:
            :class:`OrchestratorResult` with merged agent outputs.
        """
        agents = self.create_agents_for_skill(skill)
        result = OrchestratorResult(
            skill_name=skill.get_metadata().name,
            phase=phase,
        )

        prompt = skill.get_phase_prompt(phase, context, user_input)

        async def _run_agent(agent: BaseAgent) -> AgentResult:
            logger.info("Async fan-out: launching agent=%s", agent.name)
            try:
                return await asyncio.to_thread(agent.execute, prompt, context=context)
            except Exception:
                logger.exception(
                    "Agent %s raised an exception during async fan-out execution",
                    agent.name,
                )
                return AgentResult(
                    agent_name=agent.name,
                    content=f"Agent '{agent.name}' failed with an unexpected error.",
                    success=False,
                )

        coros = [_run_agent(agent) for agent in agents]
        agent_results = await gather_with_errors(*coros, return_exceptions=False)

        for agent_result in agent_results:
            result.agent_results.append(agent_result)
            _accumulate_usage(result, agent_result)
            if not agent_result.success:
                logger.warning(
                    "Agent %s failed (non-fatal in async fan-out): %s",
                    agent_result.agent_name,
                    agent_result.content,
                )

        result.success = any(r.success for r in result.agent_results)
        result.synthesized_output = self._merge_agent_outputs(result.agent_results)

        return result

    async def async_execute_sequential(
        self,
        skill: BaseSkill,
        phase: SkillPhase,
        context: str,
        user_input: str,
    ) -> OrchestratorResult:
        """Async version of :meth:`execute_sequential`.

        Runs agents sequentially (each builds on the previous output)
        but uses ``asyncio.to_thread()`` to avoid blocking the event
        loop during each agent's execution.
        """
        agents = self.create_agents_for_skill(skill)
        result = OrchestratorResult(
            skill_name=skill.get_metadata().name,
            phase=phase,
        )

        current_context = context
        prompt = skill.get_phase_prompt(phase, context, user_input)
        context_chain: list[str] = []

        run_cost_usd: float = 0.0
        max_cost_per_run: float = self._settings.budget.max_cost_per_run
        failure_counts: dict[str, int] = {}
        max_failures: int = self._settings.agents.max_failures_before_fallback

        for i, agent in enumerate(agents):
            logger.info(
                "Async sequential step %d/%d: agent=%s",
                i + 1, len(agents), agent.name,
            )

            if i == 0:
                agent_result = await asyncio.to_thread(
                    agent.execute, prompt, context=current_context,
                )
            else:
                prev = result.agent_results[-1]
                context_chain.append(
                    self._build_previous_agent_summary(agents[i - 1].role, prev),
                )
                accumulated = (
                    f"{current_context}\n\n"
                    + "\n\n---\n\n".join(context_chain)
                )
                agent_result = await asyncio.to_thread(
                    agent.execute, prompt, context=accumulated,
                )

            result.agent_results.append(agent_result)
            _accumulate_usage(result, agent_result)

            # ── Cost circuit breaker ──────────────────────────────────────
            run_cost_usd += _compute_step_cost(agent_result, agent.model)
            result.run_cost_usd = run_cost_usd
            if max_cost_per_run > 0.0 and run_cost_usd > max_cost_per_run:
                logger.warning(
                    "Cost circuit breaker triggered after agent %s: "
                    "run_cost_usd=%.6f > max_cost_per_run=%.6f",
                    agent.name, run_cost_usd, max_cost_per_run,
                )
                result.success = False
                result.budget_exceeded = True
                result.synthesized_output = (
                    f"[WARNING] Pipeline halted: accumulated run cost "
                    f"${run_cost_usd:.4f} exceeded budget ${max_cost_per_run:.4f}.\n\n"
                    + (result.agent_results[-1].content if result.agent_results else "")
                )
                return result

            if not agent_result.success:
                # ── Model fallback on repeated failures ───────────────────
                error_type = agent_result.metadata.get("error_type", "")
                if max_failures > 0 and error_type in {"rate_limit", "connection"}:
                    failure_counts[agent.name] = failure_counts.get(agent.name, 0) + 1
                    if failure_counts[agent.name] >= max_failures:
                        fallback_model = self._settings.models.fallback
                        logger.warning(
                            "Agent %s failed %d time(s) with error_type=%s — "
                            "switching to fallback model %s",
                            agent.name, failure_counts[agent.name],
                            error_type, fallback_model,
                        )
                        agent._config.model = fallback_model
                else:
                    result.success = False
                    logger.warning("Agent %s failed: %s", agent.name, agent_result.content)
                    break
            else:
                # Reset failure counter on success
                failure_counts.pop(agent.name, None)

        if result.agent_results:
            result.synthesized_output = result.agent_results[-1].content

        return result

    async def async_execute_skill_phase(
        self,
        skill: BaseSkill,
        phase: SkillPhase,
        context: str,
        user_input: str,
        *,
        strategy: str = "sequential",
    ) -> SkillResult:
        """Async version of :meth:`execute_skill_phase`.

        Delegates to :meth:`async_execute_fanout` or
        :meth:`async_execute_sequential` based on *strategy*.
        """
        skill_name = skill.get_metadata().name
        logger.info(
            "Async executing skill=%s phase=%s strategy=%s",
            skill_name, phase, strategy,
        )

        try:
            t0 = time.perf_counter()
            if strategy == "fanout":
                orch_result = await self.async_execute_fanout(skill, phase, context, user_input)
            else:
                orch_result = await self.async_execute_sequential(skill, phase, context, user_input)

            # Telemetry
            try:
                duration_ms = (time.perf_counter() - t0) * 1000
                EventBus.get().emit(
                    OrchestratorPhaseCompleted(
                        skill=skill_name,
                        phase=phase.value if hasattr(phase, "value") else str(phase),
                        strategy=strategy,
                        duration_ms=duration_ms,
                        is_async=True,
                    )
                )
            except Exception:  # noqa: BLE001
                pass

            return orch_result.to_skill_result()
        except (VaigAuthError, VAIGError):
            raise  # Let known errors propagate with their actionable messages
        except Exception as exc:
            logger.error("Orchestrator error in async_execute_skill_phase: %s", exc, exc_info=True)
            raise VAIGError(f"Pipeline execution failed: {exc}") from exc

    async def async_execute_with_tools(
        self,
        query: str,
        skill: BaseSkill,
        tool_registry: ToolRegistry,
        *,
        strategy: str = "sequential",
        is_autopilot: bool | None = None,
        on_tool_call: OnToolCall | None = None,
        tool_call_store: ToolCallStore | None = None,
        on_agent_progress: OnAgentProgress | None = None,
    ) -> OrchestratorResult:
        """Async version of :meth:`execute_with_tools`.

        Uses ``gather_with_errors()`` for fan-out parallelism and
        ``asyncio.to_thread()`` for sequential / single agent execution.
        All sync agent methods are offloaded to threads so the event
        loop stays responsive.

        Args:
            query: The user query / task to execute.
            skill: The skill defining agent configs and system prompts.
            tool_registry: Pre-configured tool registry for tool-aware agents.
            strategy: ``"sequential"`` (default), ``"fanout"``, or ``"single"``.
            is_autopilot: Autopilot detection result.
            on_tool_call: Optional callback invoked after each tool
                execution.  Threaded through to each agent's tool loop.
            tool_call_store: Optional store for recording full tool call
                results for metrics and feedback.  When provided, the
                store is passed to each ``ToolAwareAgent.execute()`` call.
            on_agent_progress: Optional callback invoked when each agent
                starts and finishes execution.  Used by the CLI to show
                live progress indicators.

        Returns:
            :class:`OrchestratorResult` with the aggregated outcome.
        """
        try:
            return await self._async_execute_with_tools_impl(
                query, skill, tool_registry,
                strategy=strategy,
                is_autopilot=is_autopilot,
                on_tool_call=on_tool_call,
                tool_call_store=tool_call_store,
                on_agent_progress=on_agent_progress,
            )
        except (VaigAuthError, VAIGError):
            raise  # Let known errors propagate with their actionable messages
        except Exception as exc:
            logger.error("Orchestrator error in async_execute_with_tools: %s", exc, exc_info=True)
            raise VAIGError(f"Pipeline execution failed: {exc}") from exc

    async def _async_execute_with_tools_impl(
        self,
        query: str,
        skill: BaseSkill,
        tool_registry: ToolRegistry,
        *,
        strategy: str = "sequential",
        is_autopilot: bool | None = None,
        on_tool_call: OnToolCall | None = None,
        tool_call_store: ToolCallStore | None = None,
        on_agent_progress: OnAgentProgress | None = None,
    ) -> OrchestratorResult:
        """Inner implementation of async_execute_with_tools (no error boundary)."""
        logger.info(
            "async_execute_with_tools: skill=%s strategy=%s",
            skill.get_metadata().name,
            strategy,
        )

        t0_ewt = time.perf_counter()

        # ── Create a shared tool result cache for dedup ──────
        from vaig.core.cache import ToolResultCache

        tool_result_cache = ToolResultCache()

        # ── Start a tool call store run if provided ──────────
        if tool_call_store is not None:
            tool_call_store.start_run()

        # ── Dynamic language detection & injection ───────────
        lang = detect_language(query)
        agent_configs = skill.get_agents_config()
        if lang != "en":
            inject_language_into_config(agent_configs, lang)
            logger.info(
                "Language detected: %s — injected language instruction into %d agent(s)",
                lang, len(agent_configs),
            )

        # ── Autopilot context injection ──────────────────────
        if is_autopilot:
            inject_autopilot_into_config(agent_configs, is_autopilot)
            logger.info(
                "GKE Autopilot detected — injected Autopilot instruction into %d agent(s)",
                len(agent_configs),
            )

        # ── Auto-detect parallel_sequential from agent configs ───────────
        # When any agent config declares a ``parallel_group`` key and the
        # caller has not explicitly requested a non-sequential strategy,
        # automatically upgrade to ``parallel_sequential`` so that skills
        # returning ``get_parallel_agents_config()`` from ``get_agents_config()``
        # work without requiring every call-site to pass the strategy explicitly.
        if strategy == "sequential" and any(
            cfg.get("parallel_group") for cfg in agent_configs
        ):
            logger.warning(
                "Auto-detected parallel_group agents — upgrading strategy to parallel_sequential",
            )
            strategy = "parallel_sequential"

        agents = self.create_agents_for_skill(
            skill, tool_registry, agent_configs=agent_configs,
        )
        result = OrchestratorResult(
            skill_name=skill.get_metadata().name,
            phase=SkillPhase.EXECUTE,
        )

        known_strategies = {"sequential", "fanout", "single", "parallel_sequential"}
        if strategy not in known_strategies:
            logger.warning(
                "Unknown strategy '%s' — falling back to sequential. "
                "Valid strategies: %s",
                strategy,
                ", ".join(sorted(known_strategies)),
            )

        if strategy == "parallel_sequential":
            result = await self._async_execute_parallel_then_sequential(
                agents=agents,
                skill=skill,
                query=query,
                result=result,
                on_agent_progress=on_agent_progress,
                on_tool_call=on_tool_call,
                tool_call_store=tool_call_store,
                tool_result_cache=tool_result_cache,
            )

        elif strategy == "fanout":
            # ── Async fan-out: all agents concurrently via gather ──
            total = len(agents)

            async def _run_agent(agent: BaseAgent, idx: int) -> AgentResult:
                logger.info("Async fan-out (tools): launching agent=%s", agent.name)
                _fire_agent_progress(on_agent_progress, agent.name, idx, total, "start")
                try:
                    kw: dict[str, Any] = {"context": query}
                    if isinstance(agent, ToolAwareAgent):
                        if on_tool_call is not None:
                            kw["on_tool_call"] = on_tool_call
                        if tool_call_store is not None:
                            kw["tool_call_store"] = tool_call_store
                        kw["tool_result_cache"] = tool_result_cache
                    agent_result = await asyncio.to_thread(agent.execute, query, **kw)
                except Exception:
                    logger.exception(
                        "Agent %s raised an exception during async fan-out (tools) execution",
                        agent.name,
                    )
                    agent_result = AgentResult(
                        agent_name=agent.name,
                        content=f"Agent '{agent.name}' failed with an unexpected error.",
                        success=False,
                    )
                _fire_agent_progress(on_agent_progress, agent.name, idx, total, "end")
                return agent_result

            coros = [_run_agent(agent, idx) for idx, agent in enumerate(agents)]
            agent_results = await gather_with_errors(*coros, return_exceptions=False)

            for agent_result in agent_results:
                result.agent_results.append(agent_result)
                if not agent_result.success:
                    logger.warning(
                        "Agent %s failed (non-fatal in async fan-out): %s",
                        agent_result.agent_name,
                        agent_result.content,
                    )

            # Accumulate usage after all agents complete
            for agent_result in result.agent_results:
                _accumulate_usage(result, agent_result)

            result.success = any(r.success for r in result.agent_results)
            result.synthesized_output = self._merge_agent_outputs(result.agent_results)

        elif strategy == "single":
            if agents:
                kw_single: dict[str, Any] = {}
                if isinstance(agents[0], ToolAwareAgent):
                    if on_tool_call is not None:
                        kw_single["on_tool_call"] = on_tool_call
                    if tool_call_store is not None:
                        kw_single["tool_call_store"] = tool_call_store
                    kw_single["tool_result_cache"] = tool_result_cache
                _fire_agent_progress(on_agent_progress, agents[0].name, 0, 1, "start")
                try:
                    agent_result = await asyncio.to_thread(agents[0].execute, query, **kw_single)
                finally:
                    _fire_agent_progress(on_agent_progress, agents[0].name, 0, 1, "end")
                result.agent_results.append(agent_result)
                _accumulate_usage(result, agent_result)
                result.success = agent_result.success
                result.synthesized_output = agent_result.content
            else:
                result.success = False
                result.synthesized_output = "No agents created for skill."

        else:  # sequential (default)
            current_context = ""
            context_chain: list[str] = []
            required_sections = skill.get_required_output_sections()

            run_cost_usd: float = 0.0
            max_cost_per_run: float = self._settings.budget.max_cost_per_run
            failure_counts: dict[str, int] = {}
            max_failures: int = self._settings.agents.max_failures_before_fallback

            for i, agent in enumerate(agents):
                logger.info(
                    "Async sequential (tools) step %d/%d: agent=%s",
                    i + 1, len(agents), agent.name,
                )
                if i > 0 and result.agent_results:
                    prev = result.agent_results[-1]
                    context_chain.append(
                        self._build_previous_agent_summary(agents[i - 1].role, prev),
                    )
                    current_context = "\n\n---\n\n".join(context_chain)

                kw_seq: dict[str, Any] = {"context": current_context}
                if isinstance(agent, ToolAwareAgent):
                    if on_tool_call is not None:
                        kw_seq["on_tool_call"] = on_tool_call
                    if tool_call_store is not None:
                        kw_seq["tool_call_store"] = tool_call_store
                    kw_seq["tool_result_cache"] = tool_result_cache
                    if i == 0 and required_sections:
                        kw_seq["required_sections"] = required_sections

                _fire_agent_progress(on_agent_progress, agent.name, i, len(agents), "start")
                try:
                    agent_result = await asyncio.to_thread(
                        agent.execute, query, **kw_seq,
                    )
                finally:
                    _fire_agent_progress(on_agent_progress, agent.name, i, len(agents), "end")
                result.agent_results.append(agent_result)
                _accumulate_usage(result, agent_result)

                # ── Cost circuit breaker ──────────────────────────────────
                run_cost_usd += _compute_step_cost(agent_result, agent.model)
                result.run_cost_usd = run_cost_usd
                if max_cost_per_run > 0.0 and run_cost_usd > max_cost_per_run:
                    logger.warning(
                        "Cost circuit breaker triggered after agent %s: "
                        "run_cost_usd=%.6f > max_cost_per_run=%.6f",
                        agent.name, run_cost_usd, max_cost_per_run,
                    )
                    result.success = False
                    result.budget_exceeded = True
                    result.synthesized_output = (
                        f"[WARNING] Pipeline halted: accumulated run cost "
                        f"${run_cost_usd:.4f} exceeded budget ${max_cost_per_run:.4f}.\n\n"
                        + (result.agent_results[-1].content if result.agent_results else "")
                    )
                    return result

                if not agent_result.success:
                    # ── Model fallback on repeated failures ───────────────
                    error_type = agent_result.metadata.get("error_type", "")
                    if max_failures > 0 and error_type in {"rate_limit", "connection"}:
                        failure_counts[agent.name] = failure_counts.get(agent.name, 0) + 1
                        if failure_counts[agent.name] >= max_failures:
                            fallback_model = self._settings.models.fallback
                            logger.warning(
                                "Agent %s failed %d time(s) with error_type=%s — "
                                "switching to fallback model %s",
                                agent.name, failure_counts[agent.name],
                                error_type, fallback_model,
                            )
                            agent._config.model = fallback_model
                            # ── Inline retry with fallback model ─────────
                            logger.info(
                                "Retrying agent %s inline with fallback model %s",
                                agent.name, fallback_model,
                            )
                            retry_ar = await asyncio.to_thread(
                                agent.execute, query, **kw_seq,
                            )
                            run_cost_usd += _compute_step_cost(retry_ar, agent.model)
                            result.run_cost_usd = run_cost_usd
                            _accumulate_usage(result, retry_ar)
                            result.agent_results[-1] = retry_ar
                            agent_result = retry_ar
                            if not retry_ar.success:
                                result.success = False
                                logger.warning(
                                    "Agent %s fallback retry also failed: %s",
                                    agent.name, retry_ar.content,
                                )
                                break
                            # Retry succeeded — reset failure counter
                            failure_counts.pop(agent.name, None)
                    else:
                        result.success = False
                        logger.warning("Agent %s failed: %s", agent.name, agent_result.content)
                        break
                else:
                    # Reset failure counter on success
                    failure_counts.pop(agent.name, None)

                # ── Gatherer output validation + mandatory retry ──────
                if i == 0 and required_sections and agent_result.success:
                    validation = self._validate_gatherer_output(
                        agent_result.content, required_sections,
                    )
                    if validation.needs_retry:
                        all_issues = validation.missing_sections + validation.shallow_sections
                        logger.warning(
                            "Gatherer output has %d issue(s): "
                            "missing=%s, shallow=%s — retrying once",
                            len(all_issues),
                            ", ".join(validation.missing_sections) or "(none)",
                            ", ".join(validation.shallow_sections) or "(none)",
                        )
                        retry_prompt = self._build_retry_prompt(
                            query,
                            validation.missing_sections,
                            shallow_sections=validation.shallow_sections,
                        )
                    else:
                        logger.info(
                            "Gatherer validation passed — running mandatory "
                            "deepening second pass for thoroughness",
                        )
                        retry_prompt = self._build_deepening_prompt(
                            query, agent_result.content,
                        )

                    is_deepening = not validation.needs_retry
                    if not is_deepening:
                        agent.reset()

                    kw_retry: dict[str, Any] = {"context": ""}
                    if isinstance(agent, ToolAwareAgent):
                        if on_tool_call is not None:
                            kw_retry["on_tool_call"] = on_tool_call
                        if tool_call_store is not None:
                            kw_retry["tool_call_store"] = tool_call_store
                        kw_retry["tool_result_cache"] = tool_result_cache
                        if required_sections:
                            kw_retry["required_sections"] = required_sections

                    original_max_iters: int | None = None
                    if (
                        isinstance(agent, ToolAwareAgent)
                        and hasattr(agent, "_max_iterations")
                    ):
                        original_max_iters = agent._max_iterations
                        agent._max_iterations = (
                            self._settings.agents.max_iterations_retry
                        )

                    try:
                        retry_result = await asyncio.to_thread(
                            agent.execute, retry_prompt, **kw_retry,
                        )
                    except MaxIterationsError:
                        if is_deepening:
                            logger.warning(
                                "Deepening second pass for agent %s hit "
                                "max_iterations_retry limit; falling back "
                                "to first-pass output",
                                agent.name,
                            )
                            logger.info(
                                "Agent %s continuing with first-pass "
                                "output — tokens=%s",
                                agent.name,
                                agent_result.usage.get(
                                    "total_tokens", "?",
                                ),
                            )
                        else:
                            raise
                    else:
                        _accumulate_usage(result, retry_result)
                        run_cost_usd += _compute_step_cost(retry_result, agent.model)
                        result.run_cost_usd = run_cost_usd
                        if max_cost_per_run > 0.0 and run_cost_usd > max_cost_per_run:
                            result.success = False
                            result.budget_exceeded = True
                            result.synthesized_output = (
                                f"[WARNING] Cost budget ${max_cost_per_run:.4f} exceeded "
                                f"during retry (actual: ${run_cost_usd:.4f})"
                            )
                            break

                        if is_deepening:
                            merged_content = (
                                agent_result.content
                                + "\n\n"
                                + retry_result.content
                            )
                            retry_result = AgentResult(
                                agent_name=retry_result.agent_name,
                                content=merged_content,
                                success=retry_result.success,
                                usage=retry_result.usage,
                                metadata=retry_result.metadata,
                            )

                        result.agent_results[-1] = retry_result

                        if not retry_result.success:
                            result.success = False
                            logger.warning(
                                "Agent %s retry also failed: %s",
                                agent.name, retry_result.content,
                            )
                            break
                        logger.info(
                            "Agent %s retry succeeded — tokens=%s",
                            agent.name,
                            retry_result.usage.get("total_tokens", "?"),
                        )
                    finally:
                        if (
                            isinstance(agent, ToolAwareAgent)
                            and original_max_iters is not None
                        ):
                            agent._max_iterations = original_max_iters

                # ── Reporter output validation + retry (async) ──────
                is_reporter = "report" in getattr(agent, "role", "").lower()
                if i == len(agents) - 1 and agent_result.success and is_reporter:
                    # ── Structured output pre-validation (async) ──────
                    schema_cls = getattr(
                        getattr(agent, "config", None),
                        "response_schema",
                        None,
                    )
                    if (
                        schema_cls is not None
                        and isinstance(schema_cls, type)
                        and hasattr(schema_cls, "model_validate_json")
                    ):
                        import json as _json

                        try:
                            _json.loads(agent_result.content)
                            schema_cls.model_validate_json(agent_result.content)
                        except Exception as schema_exc:
                            logger.warning(
                                "Reporter %s structured output failed "
                                "schema validation (%s) — retrying with "
                                "JSON-focused prompt",
                                agent.name,
                                schema_exc,
                            )
                            json_retry_prompt = (
                                "Your previous response was not valid JSON "
                                "conforming to the required schema. "
                                f"Error: {schema_exc}\n\n"
                                "Regenerate the report as valid JSON that "
                                "strictly conforms to the schema. "
                                "Do NOT include markdown fences or commentary "
                                "— output ONLY the raw JSON object.\n\n"
                                f"## Previous Analysis\n\n{current_context}"
                            )
                            agent.reset()
                            kw_js: dict[str, Any] = {"context": ""}
                            if isinstance(agent, ToolAwareAgent):
                                if on_tool_call is not None:
                                    kw_js["on_tool_call"] = on_tool_call
                                if tool_call_store is not None:
                                    kw_js["tool_call_store"] = tool_call_store
                                kw_js["tool_result_cache"] = tool_result_cache
                            json_retry_result = await asyncio.to_thread(
                                agent.execute, json_retry_prompt, **kw_js,
                            )
                            _accumulate_usage(result, json_retry_result)
                            run_cost_usd += _compute_step_cost(json_retry_result, agent.model)
                            result.run_cost_usd = run_cost_usd
                            if max_cost_per_run > 0.0 and run_cost_usd > max_cost_per_run:
                                result.success = False
                                result.budget_exceeded = True
                                result.synthesized_output = (
                                    f"[WARNING] Cost budget ${max_cost_per_run:.4f} exceeded "
                                    f"during JSON schema retry (actual: ${run_cost_usd:.4f})"
                                )
                                break
                            if json_retry_result.success:
                                agent_result = json_retry_result
                                result.agent_results[-1] = agent_result
                                logger.info(
                                    "Reporter %s JSON retry succeeded",
                                    agent.name,
                                )
                            else:
                                logger.warning(
                                    "Reporter %s JSON retry also failed — "
                                    "proceeding with original output",
                                    agent.name,
                                )

                    # Retain the parsed structured report (if any)
                    # before post-processing converts it to Markdown.
                    # The CLI uses this for --summary and Rich rendering.
                    if (
                        schema_cls is not None
                        and hasattr(schema_cls, "model_validate_json")
                        and result.structured_report is None
                    ):
                        try:
                            result.structured_report = schema_cls.model_validate_json(
                                agent_result.content,
                            )
                        except Exception:
                            pass  # best-effort

                    # Post-process structured output (e.g. JSON → Markdown)
                    agent_result.content = skill.post_process_report(
                        agent_result.content,
                    )

                    finish_reason = agent_result.metadata.get("finish_reason", "")
                    md_issues = self._validate_reporter_output(agent_result.content)

                    if finish_reason == "MAX_TOKENS" or md_issues:
                        reasons: list[str] = []
                        if finish_reason == "MAX_TOKENS":
                            reasons.append("finish_reason=MAX_TOKENS")
                        if md_issues:
                            reasons.extend(md_issues)
                        logger.warning(
                            "Reporter output has %d issue(s): %s — retrying once",
                            len(reasons), "; ".join(reasons),
                        )

                        reporter_retry_prompt = (
                            "Your previous report had formatting issues "
                            f"({'; '.join(reasons)}). "
                            "Regenerate the report more concisely:\n"
                            "- Focus on CONFIRMED and HIGH severity findings only\n"
                            "- Close ALL table rows with a trailing |\n"
                            "- Close ALL code blocks with ```\n"
                            "- Keep total output under 12 000 tokens\n\n"
                            f"## Previous Analysis\n\n{current_context}"
                        )
                        agent.reset()
                        kw_rr: dict[str, Any] = {"context": ""}
                        if isinstance(agent, ToolAwareAgent):
                            if on_tool_call is not None:
                                kw_rr["on_tool_call"] = on_tool_call
                            if tool_call_store is not None:
                                kw_rr["tool_call_store"] = tool_call_store
                            kw_rr["tool_result_cache"] = tool_result_cache
                        reporter_retry = await asyncio.to_thread(
                            agent.execute, reporter_retry_prompt, **kw_rr,
                        )
                        _accumulate_usage(result, reporter_retry)
                        run_cost_usd += _compute_step_cost(reporter_retry, agent.model)
                        result.run_cost_usd = run_cost_usd
                        if max_cost_per_run > 0.0 and run_cost_usd > max_cost_per_run:
                            result.success = False
                            result.budget_exceeded = True
                            result.synthesized_output = (
                                f"[WARNING] Cost budget ${max_cost_per_run:.4f} exceeded "
                                f"during reporter retry (actual: ${run_cost_usd:.4f})"
                            )
                            return result
                        result.agent_results[-1] = reporter_retry

                        if not reporter_retry.success:
                            result.success = False
                            logger.warning(
                                "Reporter %s retry also failed: %s",
                                agent.name, reporter_retry.content,
                            )
                            break
                        logger.info(
                            "Reporter %s retry succeeded — tokens=%s",
                            agent.name,
                            reporter_retry.usage.get("total_tokens", "?"),
                        )

            if result.agent_results:
                result.synthesized_output = result.agent_results[-1].content

        # Telemetry
        try:
            duration_ms = (time.perf_counter() - t0_ewt) * 1000
            EventBus.get().emit(
                OrchestratorToolsCompleted(
                    skill=skill.get_metadata().name,
                    strategy=strategy,
                    agents_count=len(agents),
                    success=result.success,
                    duration_ms=duration_ms,
                    is_async=True,
                )
            )
        except Exception:  # noqa: BLE001
            pass

        return result

    # ── parallel_sequential helpers ────────────────────────────────────────

    def _execute_parallel_then_sequential(
        self,
        *,
        agents: list[BaseAgent],
        skill: BaseSkill,
        query: str,
        result: OrchestratorResult,
        on_agent_progress: Any | None,
        on_tool_call: Any | None,
        tool_call_store: Any | None,
        tool_result_cache: Any,
    ) -> OrchestratorResult:
        """Run gatherer agents concurrently, then remaining agents sequentially.

        This is the synchronous implementation of the ``parallel_sequential``
        execution strategy.  Agents whose names end with ``_gatherer`` are
        treated as the *parallel group* and are submitted to a
        :class:`~concurrent.futures.ThreadPoolExecutor` simultaneously.  All
        other agents (analyzer, verifier, reporter, …) form the *sequential
        tail* and run one after another once the parallel phase is done.

        **Execution flow**:

        1. Call :meth:`~vaig.skills.base.BaseSkill.pre_execute_parallel` on the
           skill so that shared resources (e.g. K8s client) can be pre-warmed
           before threads start.
        2. Submit every ``_gatherer`` agent to the thread pool and collect
           :class:`~vaig.agents.base.AgentResult` objects.  Individual agent
           failures are caught and recorded as non-fatal error results so the
           rest of the pipeline can continue with partial data.
        3. Merge all gatherer outputs into a single context string via
           :meth:`_merge_parallel_outputs`.
        4. Optionally warn about missing required sections via
           :meth:`~vaig.skills.base.BaseSkill.get_required_output_sections`.
        5. Run each sequential agent in order, threading the accumulated context
           chain between them.
        6. Compute overall success: at least one gatherer must succeed AND the
           last sequential agent must succeed.

        Args:
            agents: Full list of agents created for the skill (gatherers +
                sequential tail).  Order within each group is preserved.
            skill: The :class:`~vaig.skills.base.BaseSkill` being executed.
                Used for the pre-warm hook and required-sections validation.
            query: The original user query passed to each agent's ``execute``
                call.
            result: Pre-initialised :class:`OrchestratorResult` to accumulate
                agent results and usage into.
            on_agent_progress: Optional progress callback fired at the start
                and end of each agent.
            on_tool_call: Optional callback forwarded to tool-aware agents.
            tool_call_store: Optional store forwarded to tool-aware agents.
            tool_result_cache: Shared :class:`~vaig.core.cache.ToolResultCache`
                instance for deduplicating tool calls across agents.

        Returns:
            The mutated *result* object populated with all agent results,
            aggregated token usage, ``success`` flag, and
            ``synthesized_output`` (the last agent's content).
        """
        parallel_agents = [a for a in agents if a.name.endswith("_gatherer")]
        sequential_agents = [a for a in agents if not a.name.endswith("_gatherer")]

        logger.info(
            "parallel_sequential: %d parallel gatherer(s), %d sequential agent(s)",
            len(parallel_agents),
            len(sequential_agents),
        )

        # ── Pre-execution hook (sequential, before threads start) ─────────
        skill.pre_execute_parallel(query)

        # ── Concurrent phase ─────────────────────────────────────────────
        total_parallel = len(parallel_agents)
        parallel_results: list[AgentResult] = []
        # Capture model IDs before the executor starts — needed for cost
        # accounting after futures complete (agent refs are still valid here).
        parallel_agent_models: dict[str, str] = {a.name: a.model for a in parallel_agents}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(total_parallel, 1)) as executor:
            futures_map: list[tuple[BaseAgent, concurrent.futures.Future[AgentResult]]] = []
            for idx, agent in enumerate(parallel_agents):
                logger.info("parallel_sequential: submitting gatherer agent=%s", agent.name)
                _fire_agent_progress(on_agent_progress, agent.name, idx, total_parallel, "start")
                kw: dict[str, Any] = {"context": query}
                if isinstance(agent, ToolAwareAgent):
                    if on_tool_call is not None:
                        kw["on_tool_call"] = on_tool_call
                    if tool_call_store is not None:
                        kw["tool_call_store"] = tool_call_store
                    kw["tool_result_cache"] = tool_result_cache
                futures_map.append((agent, executor.submit(agent.execute, query, **kw)))

            for idx, (agent, future) in enumerate(futures_map):
                try:
                    agent_result = future.result()
                except Exception:
                    logger.exception(
                        "Gatherer agent %s raised an exception during parallel execution",
                        agent.name,
                    )
                    agent_result = AgentResult(
                        agent_name=agent.name,
                        content=f"Agent '{agent.name}' failed with an unexpected error.",
                        success=False,
                    )
                _fire_agent_progress(on_agent_progress, agent.name, idx, total_parallel, "end")
                parallel_results.append(agent_result)
                result.agent_results.append(agent_result)
                _accumulate_usage(result, agent_result)
                if not agent_result.success:
                    logger.warning(
                        "Gatherer agent %s failed (non-fatal): %s",
                        agent.name, agent_result.content,
                    )

        # ── Merge gatherer outputs ────────────────────────────────────────
        merged_context = self._merge_parallel_outputs(parallel_results)

        # ── Validate gatherer outputs (task 1.4) ─────────────────────────
        required_sections = skill.get_required_output_sections()
        if required_sections:
            missing = [s for s in required_sections if s.lower() not in merged_context.lower()]
            if missing:
                logger.warning(
                    "Merged gatherer output is missing %d required section(s): %s",
                    len(missing),
                    ", ".join(missing),
                )

        # ── Sequential phase ──────────────────────────────────────────────
        total_agents = len(parallel_agents) + len(sequential_agents)
        current_context = merged_context
        context_chain: list[str] = [merged_context]

        run_cost_usd: float = sum(
            _compute_step_cost(r, parallel_agent_models.get(r.agent_name))
            for r in parallel_results
        )
        max_cost_per_run: float = self._settings.budget.max_cost_per_run
        result.run_cost_usd = run_cost_usd
        failure_counts: dict[str, int] = {}
        max_failures: int = self._settings.agents.max_failures_before_fallback

        # ── Post-parallel cost check ──────────────────────────────────────
        if max_cost_per_run > 0.0 and run_cost_usd > max_cost_per_run:
            logger.warning(
                "Cost circuit breaker triggered after parallel phase: "
                "run_cost_usd=%.6f > max_cost_per_run=%.6f",
                run_cost_usd, max_cost_per_run,
            )
            result.success = False
            result.budget_exceeded = True
            result.synthesized_output = (
                f"[WARNING] Pipeline halted: accumulated run cost "
                f"${run_cost_usd:.4f} exceeded budget ${max_cost_per_run:.4f}.\n\n"
                + (result.agent_results[-1].content if result.agent_results else "")
            )
            return result

        for i, agent in enumerate(sequential_agents):
            idx_global = total_parallel + i
            logger.info(
                "parallel_sequential: sequential step %d/%d: agent=%s",
                i + 1, len(sequential_agents), agent.name,
            )
            if i > 0 and result.agent_results:
                prev = result.agent_results[-1]
                context_chain.append(
                    self._build_previous_agent_summary(
                        sequential_agents[i - 1].role, prev,
                    ),
                )
                current_context = "\n\n---\n\n".join(context_chain)

            kw_seq: dict[str, Any] = {"context": current_context}
            if isinstance(agent, ToolAwareAgent):
                if on_tool_call is not None:
                    kw_seq["on_tool_call"] = on_tool_call
                if tool_call_store is not None:
                    kw_seq["tool_call_store"] = tool_call_store
                kw_seq["tool_result_cache"] = tool_result_cache

            _fire_agent_progress(on_agent_progress, agent.name, idx_global, total_agents, "start")
            try:
                agent_result = agent.execute(query, **kw_seq)
            finally:
                _fire_agent_progress(on_agent_progress, agent.name, idx_global, total_agents, "end")

            result.agent_results.append(agent_result)
            _accumulate_usage(result, agent_result)

            # ── Cost circuit breaker ──────────────────────────────────────
            run_cost_usd += _compute_step_cost(agent_result, agent.model)
            result.run_cost_usd = run_cost_usd
            if max_cost_per_run > 0.0 and run_cost_usd > max_cost_per_run:
                logger.warning(
                    "Cost circuit breaker triggered after agent %s: "
                    "run_cost_usd=%.6f > max_cost_per_run=%.6f",
                    agent.name, run_cost_usd, max_cost_per_run,
                )
                result.success = False
                result.budget_exceeded = True
                result.synthesized_output = (
                    f"[WARNING] Pipeline halted: accumulated run cost "
                    f"${run_cost_usd:.4f} exceeded budget ${max_cost_per_run:.4f}.\n\n"
                    + (result.agent_results[-1].content if result.agent_results else "")
                )
                return result

            if not agent_result.success:
                # ── Model fallback on repeated failures ───────────────────
                error_type = agent_result.metadata.get("error_type", "")
                if max_failures > 0 and error_type in {"rate_limit", "connection"}:
                    failure_counts[agent.name] = failure_counts.get(agent.name, 0) + 1
                    if failure_counts[agent.name] >= max_failures:
                        fallback_model = self._settings.models.fallback
                        logger.warning(
                            "Agent %s failed %d time(s) with error_type=%s — "
                            "switching to fallback model %s",
                            agent.name, failure_counts[agent.name],
                            error_type, fallback_model,
                        )
                        agent._config.model = fallback_model
                        # ── Inline retry with fallback model ─────────────
                        logger.info(
                            "Retrying agent %s inline with fallback model %s",
                            agent.name, fallback_model,
                        )
                        retry_ar = agent.execute(query, **kw_seq)
                        run_cost_usd += _compute_step_cost(retry_ar, agent.model)
                        result.run_cost_usd = run_cost_usd
                        _accumulate_usage(result, retry_ar)
                        result.agent_results[-1] = retry_ar
                        agent_result = retry_ar
                        if not retry_ar.success:
                            result.success = False
                            logger.warning(
                                "Agent %s fallback retry also failed: %s",
                                agent.name, retry_ar.content,
                            )
                            break
                        # Retry succeeded — reset failure counter
                        failure_counts.pop(agent.name, None)
                else:
                    result.success = False
                    logger.warning(
                        "Sequential agent %s failed: %s", agent.name, agent_result.content,
                    )
                    break
            else:
                # Reset failure counter on success
                failure_counts.pop(agent.name, None)

        if result.agent_results:
            # Success if any gatherer succeeded AND the last sequential agent succeeded
            any_gatherer_ok = any(r.success for r in parallel_results)
            last_seq_ok = result.agent_results[-1].success if sequential_agents else True
            result.success = any_gatherer_ok and last_seq_ok
            result.synthesized_output = result.agent_results[-1].content
        else:
            result.success = False
            result.synthesized_output = "No agents ran."

        return result

    async def _async_execute_parallel_then_sequential(
        self,
        *,
        agents: list[BaseAgent],
        skill: BaseSkill,
        query: str,
        result: OrchestratorResult,
        on_agent_progress: Any | None,
        on_tool_call: Any | None,
        tool_call_store: Any | None,
        tool_result_cache: Any,
    ) -> OrchestratorResult:
        """Async version of :meth:`_execute_parallel_then_sequential`.

        Parallel gatherers run concurrently via :func:`asyncio.gather` inside a
        :class:`~concurrent.futures.ThreadPoolExecutor` (agents are synchronous),
        while sequential agents run serially afterwards.  The execution flow and
        success semantics are identical to the synchronous counterpart — see its
        docstring for full details.

        Args:
            agents: Full list of agents created for the skill.
            skill: The :class:`~vaig.skills.base.BaseSkill` being executed.
            query: The original user query.
            result: Pre-initialised :class:`OrchestratorResult` to populate.
            on_agent_progress: Optional progress callback.
            on_tool_call: Optional tool-call callback.
            tool_call_store: Optional tool-call store.
            tool_result_cache: Shared tool result cache.

        Returns:
            The mutated *result* object.
        """
        parallel_agents = [a for a in agents if a.name.endswith("_gatherer")]
        sequential_agents = [a for a in agents if not a.name.endswith("_gatherer")]

        logger.info(
            "async parallel_sequential: %d parallel gatherer(s), %d sequential agent(s)",
            len(parallel_agents),
            len(sequential_agents),
        )

        # ── Pre-execution hook (sequential, before tasks launch) ──────────
        skill.pre_execute_parallel(query)

        # ── Concurrent phase ─────────────────────────────────────────────
        total_parallel = len(parallel_agents)
        # Capture model IDs before coroutines start — needed for cost
        # accounting after gather returns.
        parallel_agent_models: dict[str, str] = {a.name: a.model for a in parallel_agents}

        async def _run_gatherer(agent: BaseAgent, idx: int) -> AgentResult:
            logger.info("async parallel_sequential: launching gatherer=%s", agent.name)
            _fire_agent_progress(on_agent_progress, agent.name, idx, total_parallel, "start")
            try:
                kw: dict[str, Any] = {"context": query}
                if isinstance(agent, ToolAwareAgent):
                    if on_tool_call is not None:
                        kw["on_tool_call"] = on_tool_call
                    if tool_call_store is not None:
                        kw["tool_call_store"] = tool_call_store
                    kw["tool_result_cache"] = tool_result_cache
                agent_result = await asyncio.to_thread(agent.execute, query, **kw)
            except Exception:
                logger.exception(
                    "Gatherer agent %s raised an exception during async parallel execution",
                    agent.name,
                )
                agent_result = AgentResult(
                    agent_name=agent.name,
                    content=f"Agent '{agent.name}' failed with an unexpected error.",
                    success=False,
                )
            _fire_agent_progress(on_agent_progress, agent.name, idx, total_parallel, "end")
            return agent_result

        coros = [_run_gatherer(agent, idx) for idx, agent in enumerate(parallel_agents)]
        parallel_results: list[AgentResult] = await gather_with_errors(*coros, return_exceptions=False)

        for agent_result in parallel_results:
            result.agent_results.append(agent_result)
            _accumulate_usage(result, agent_result)
            if not agent_result.success:
                logger.warning(
                    "Async gatherer agent %s failed (non-fatal): %s",
                    agent_result.agent_name, agent_result.content,
                )

        # ── Merge gatherer outputs ────────────────────────────────────────
        merged_context = self._merge_parallel_outputs(parallel_results)

        # ── Validate gatherer outputs (task 1.4) ─────────────────────────
        required_sections = skill.get_required_output_sections()
        if required_sections:
            missing = [s for s in required_sections if s.lower() not in merged_context.lower()]
            if missing:
                logger.warning(
                    "Async merged gatherer output is missing %d required section(s): %s",
                    len(missing),
                    ", ".join(missing),
                )

        # ── Sequential phase ──────────────────────────────────────────────
        total_agents = len(parallel_agents) + len(sequential_agents)
        current_context = merged_context
        context_chain: list[str] = [merged_context]

        run_cost_usd: float = sum(
            _compute_step_cost(r, parallel_agent_models.get(r.agent_name))
            for r in parallel_results
        )
        max_cost_per_run: float = self._settings.budget.max_cost_per_run
        result.run_cost_usd = run_cost_usd
        failure_counts: dict[str, int] = {}
        max_failures: int = self._settings.agents.max_failures_before_fallback

        # ── Post-parallel cost check ──────────────────────────────────────
        if max_cost_per_run > 0.0 and run_cost_usd > max_cost_per_run:
            logger.warning(
                "Cost circuit breaker triggered after async parallel phase: "
                "run_cost_usd=%.6f > max_cost_per_run=%.6f",
                run_cost_usd, max_cost_per_run,
            )
            result.success = False
            result.budget_exceeded = True
            result.synthesized_output = (
                f"[WARNING] Pipeline halted: accumulated run cost "
                f"${run_cost_usd:.4f} exceeded budget ${max_cost_per_run:.4f}.\n\n"
                + (result.agent_results[-1].content if result.agent_results else "")
            )
            return result

        for i, agent in enumerate(sequential_agents):
            idx_global = total_parallel + i
            logger.info(
                "async parallel_sequential: sequential step %d/%d: agent=%s",
                i + 1, len(sequential_agents), agent.name,
            )
            if i > 0 and result.agent_results:
                prev = result.agent_results[-1]
                context_chain.append(
                    self._build_previous_agent_summary(
                        sequential_agents[i - 1].role, prev,
                    ),
                )
                current_context = "\n\n---\n\n".join(context_chain)

            kw_seq: dict[str, Any] = {"context": current_context}
            if isinstance(agent, ToolAwareAgent):
                if on_tool_call is not None:
                    kw_seq["on_tool_call"] = on_tool_call
                if tool_call_store is not None:
                    kw_seq["tool_call_store"] = tool_call_store
                kw_seq["tool_result_cache"] = tool_result_cache

            _fire_agent_progress(on_agent_progress, agent.name, idx_global, total_agents, "start")
            try:
                agent_result = await asyncio.to_thread(agent.execute, query, **kw_seq)
            finally:
                _fire_agent_progress(on_agent_progress, agent.name, idx_global, total_agents, "end")

            result.agent_results.append(agent_result)
            _accumulate_usage(result, agent_result)

            # ── Cost circuit breaker ──────────────────────────────────────
            run_cost_usd += _compute_step_cost(agent_result, agent.model)
            result.run_cost_usd = run_cost_usd
            if max_cost_per_run > 0.0 and run_cost_usd > max_cost_per_run:
                logger.warning(
                    "Cost circuit breaker triggered after async agent %s: "
                    "run_cost_usd=%.6f > max_cost_per_run=%.6f",
                    agent.name, run_cost_usd, max_cost_per_run,
                )
                result.success = False
                result.budget_exceeded = True
                result.synthesized_output = (
                    f"[WARNING] Pipeline halted: accumulated run cost "
                    f"${run_cost_usd:.4f} exceeded budget ${max_cost_per_run:.4f}.\n\n"
                    + (result.agent_results[-1].content if result.agent_results else "")
                )
                return result

            if not agent_result.success:
                # ── Model fallback on repeated failures ───────────────────
                error_type = agent_result.metadata.get("error_type", "")
                if max_failures > 0 and error_type in {"rate_limit", "connection"}:
                    failure_counts[agent.name] = failure_counts.get(agent.name, 0) + 1
                    if failure_counts[agent.name] >= max_failures:
                        fallback_model = self._settings.models.fallback
                        logger.warning(
                            "Agent %s failed %d time(s) with error_type=%s — "
                            "switching to fallback model %s",
                            agent.name, failure_counts[agent.name],
                            error_type, fallback_model,
                        )
                        agent._config.model = fallback_model
                        # ── Inline retry with fallback model ─────────────
                        logger.info(
                            "Retrying agent %s inline with fallback model %s",
                            agent.name, fallback_model,
                        )
                        retry_ar = await asyncio.to_thread(
                            agent.execute, query, **kw_seq,
                        )
                        run_cost_usd += _compute_step_cost(retry_ar, agent.model)
                        result.run_cost_usd = run_cost_usd
                        _accumulate_usage(result, retry_ar)
                        result.agent_results[-1] = retry_ar
                        agent_result = retry_ar
                        if not retry_ar.success:
                            result.success = False
                            logger.warning(
                                "Agent %s fallback retry also failed: %s",
                                agent.name, retry_ar.content,
                            )
                            break
                        # Retry succeeded — reset failure counter
                        failure_counts.pop(agent.name, None)
                else:
                    result.success = False
                    logger.warning(
                        "Async sequential agent %s failed: %s", agent.name, agent_result.content,
                    )
                    break
            else:
                # Reset failure counter on success
                failure_counts.pop(agent.name, None)

        if result.agent_results:
            any_gatherer_ok = any(r.success for r in parallel_results)
            last_seq_ok = result.agent_results[-1].success if sequential_agents else True
            result.success = any_gatherer_ok and last_seq_ok
            result.synthesized_output = result.agent_results[-1].content
        else:
            result.success = False
            result.synthesized_output = "No agents ran."

        return result

    def _merge_parallel_outputs(self, results: list[AgentResult]) -> str:
        """Merge outputs from parallel gatherer agents with clear section headers.

        Each agent's output is wrapped in a labelled section block so that
        downstream sequential agents (analyzer, verifier, reporter) can
        clearly attribute findings to their origin agent.

        **Format per agent**::

            --- node_gatherer ---

            <agent output text>

        Failed agents produce an error note in place of their content so the
        section is always present in the merged string — downstream agents can
        see which gatherers failed without the pipeline crashing.

        Args:
            results: List of :class:`~vaig.agents.base.AgentResult` objects
                produced by the parallel gatherer agents, in submission order.

        Returns:
            A single string with all agent sections concatenated, separated by
            blank lines (``\\n\\n``).  If *results* is empty the return value
            is an empty string.
        """
        sections: list[str] = []
        for r in results:
            if r.success:
                sections.append(f"--- {r.agent_name} ---\n\n{r.content}")
            else:
                sections.append(
                    f"--- {r.agent_name} ---\n\n"
                    f"[ERROR: Agent '{r.agent_name}' failed — {r.content}]",
                )
        return "\n\n".join(sections)

    def _merge_agent_outputs(self, results: list[AgentResult]) -> str:
        """Merge outputs from multiple agents into a coherent summary."""
        sections: list[str] = []
        for r in results:
            if r.success:
                sections.append(f"### {r.agent_name}\n\n{r.content}")
            else:
                sections.append(f"### {r.agent_name} (failed)\n\n{r.content}")
        return "\n\n---\n\n".join(sections)

    def _validate_gatherer_output(
        self,
        output: str,
        required_sections: list[str],
        *,
        min_content_chars: int = DEFAULT_MIN_CONTENT_CHARS,
    ) -> GathererValidationResult:
        """Check whether the gatherer output contains all required sections
        with sufficient content depth.

        The method performs two levels of validation:

        1. **Header presence** (case-insensitive) — same as before.
        2. **Content depth** — for each found header, extract the body text
           (everything between this header and the next markdown heading or
           end-of-text), strip whitespace, and verify:
           a) The body is not empty.
           b) The body does not consist solely of an empty-marker phrase
              (e.g. ``"N/A"``, ``"Data not available"``).
           c) The body has at least *min_content_chars* meaningful characters.

        Args:
            output: The text produced by the first agent (gatherer).
            required_sections: Section headings that MUST appear in the output.
            min_content_chars: Minimum character count for a section body to be
                considered "deep enough".  Defaults to
                :data:`DEFAULT_MIN_CONTENT_CHARS`.

        Returns:
            A :class:`GathererValidationResult` with separate lists for
            missing and shallow sections.
        """
        output_lower = output.lower()
        result = GathererValidationResult()

        for section in required_sections:
            section_lower = section.lower()

            if section_lower not in output_lower:
                result.missing_sections.append(section)
                continue

            # ── Extract section body ─────────────────────────────────
            body = self._extract_section_body(output, section)

            if self._is_section_shallow(body, min_content_chars):
                result.shallow_sections.append(section)

        # ── Investigation checklist validation ───────────────────
        checklist_warnings = self._validate_investigation_checklist(output, output)
        if checklist_warnings:
            # Treat invalid skips as shallow content — triggers retry
            for warning in checklist_warnings:
                result.shallow_sections.append(f"Investigation Checklist: {warning}")

        return result

    @staticmethod
    def _extract_section_body(output: str, section_name: str) -> str:
        """Extract text content after a section header up to the next heading.

        Uses a regex that matches the section name (case-insensitive) preceded
        by optional ``#`` markdown heading markers.  The body extends until the
        next markdown heading (``^#{1,6} ``) or the end of text.

        Args:
            output: Full gatherer output text.
            section_name: The section header to find.

        Returns:
            The extracted body text, stripped of leading/trailing whitespace.
            Returns ``""`` if the header is not found.
        """
        # Match optional markdown heading markers, then the section name
        pattern = re.compile(
            rf"^(?:#{1,6}\s+)?{re.escape(section_name)}\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        match = pattern.search(output)
        if not match:
            # Fallback: try finding the section name as a substring on any line
            idx = output.lower().find(section_name.lower())
            if idx == -1:
                return ""
            # Skip past the header line
            newline_after = output.find("\n", idx)
            if newline_after == -1:
                return ""
            body_start = newline_after + 1
        else:
            body_start = match.end()

        # Find the next markdown heading (or end of string)
        next_heading = re.search(
            r"^#{1,6}\s+",
            output[body_start:],
            re.MULTILINE,
        )
        if next_heading:
            body_end = body_start + next_heading.start()
        else:
            body_end = len(output)

        return output[body_start:body_end].strip()

    @staticmethod
    def _is_section_shallow(body: str, min_content_chars: int) -> bool:
        """Determine if a section body is too shallow to be useful.

        A section is considered shallow if:
        - The body is empty after stripping whitespace.
        - The body text (ignoring case) matches an empty-marker phrase.
        - The body has fewer than *min_content_chars* characters.

        Args:
            body: Extracted section body text (already stripped).
            min_content_chars: Minimum character count threshold.

        Returns:
            ``True`` if the section is shallow, ``False`` otherwise.
        """
        if not body:
            return True

        body_lower = body.lower().strip()

        # Check if body is just an empty marker
        if body_lower in EMPTY_MARKERS:
            return True

        # Check minimum character threshold
        return len(body) < min_content_chars

    # ── Investigation checklist validation ──────────────────────────────

    # Steps that are ALWAYS mandatory — can never be skipped.
    _MANDATORY_STEPS: frozenset[str] = frozenset({"1", "2", "3", "7a", "7b"})

    # Evidence patterns that invalidate a skip for conditional steps.
    _SKIP_CONTRADICTION_PATTERNS: dict[str, list[re.Pattern[str]]] = {
        "4": [
            re.compile(r"unavailable\s+replicas?", re.IGNORECASE),
            re.compile(r"\b0/\d+", re.IGNORECASE),  # "0/3" replica counts
            re.compile(r"FailedCreate", re.IGNORECASE),
        ],
        "5": [
            re.compile(r"CrashLoopBackOff", re.IGNORECASE),
            re.compile(r"\bError\b"),
            re.compile(r"\bPending\b"),
            re.compile(r"ImagePullBackOff", re.IGNORECASE),
        ],
        "6": [
            re.compile(r"\bHPA\b", re.IGNORECASE),
            re.compile(r"ScalingLimited", re.IGNORECASE),
            re.compile(r"FailedGetExternalMetric", re.IGNORECASE),
        ],
    }

    @staticmethod
    def _validate_investigation_checklist(
        output: str,
        gatherer_output: str,
    ) -> list[str]:
        """Validate the Investigation Checklist in gatherer output.

        Parses the checklist section, identifies SKIPPED steps, and
        cross-references them against the gatherer's earlier data to
        detect invalid skips (e.g., Step 4 skipped but output mentions
        "unavailable replicas").

        Args:
            output: The gatherer output containing the checklist.
            gatherer_output: The full gatherer output to scan for
                contradicting evidence. Usually the same as *output*,
                but separated for testability.

        Returns:
            A list of warning strings for each invalid skip found.
            Empty list means no problems.
        """
        warnings: list[str] = []

        # ── Locate the Investigation Checklist section ───────────
        checklist_match = re.search(
            r"###?\s*Investigation\s+Checklist",
            output,
            re.IGNORECASE,
        )
        if not checklist_match:
            warnings.append(
                "Investigation Checklist section is missing from the output."
            )
            return warnings

        # Extract everything after the checklist header
        checklist_text = output[checklist_match.end():]
        # Stop at the next markdown heading (if any)
        next_heading = re.search(r"^#{1,6}\s+", checklist_text, re.MULTILINE)
        if next_heading:
            checklist_text = checklist_text[:next_heading.start()]

        # ── Parse checklist items ────────────────────────────────
        # Match lines like:
        #   - [x] Step 1: ...
        #   - [ ] Step 4: ... (SKIPPED — reason: ...)
        item_pattern = re.compile(
            r"-\s*\[(?P<status>[x ])\]\s*Step\s+(?P<step>\w+):\s*(?P<desc>.*)",
            re.IGNORECASE,
        )

        parsed_steps: dict[str, tuple[str, str]] = {}  # step_id -> (status, description)
        for match in item_pattern.finditer(checklist_text):
            step_id = match.group("step").lower()
            status = match.group("status").strip().lower()
            desc = match.group("desc").strip()
            parsed_steps[step_id] = (status, desc)

        # ── Validate mandatory steps ─────────────────────────────
        for step_id in Orchestrator._MANDATORY_STEPS:
            if step_id not in parsed_steps:
                continue  # Step not listed — will be caught by section validation
            status, desc = parsed_steps[step_id]
            if status != "x":
                warnings.append(
                    f"Step {step_id} is MANDATORY and cannot be skipped. "
                    f"Reason given: {desc}"
                )

        # ── Validate conditional step skips against evidence ─────
        for step_id, patterns in Orchestrator._SKIP_CONTRADICTION_PATTERNS.items():
            if step_id not in parsed_steps:
                continue
            status, desc = parsed_steps[step_id]
            if status == "x":
                continue  # Step was completed — no issue

            # Step was skipped — check for contradicting evidence
            for pattern in patterns:
                if pattern.search(gatherer_output):
                    warnings.append(
                        f"Step {step_id} was SKIPPED but output contains "
                        f"evidence matching '{pattern.pattern}' — this step "
                        f"should NOT be skipped."
                    )
                    break  # One contradiction is enough

        return warnings

    def _build_retry_prompt(
        self,
        original_query: str,
        missing_sections: list[str],
        *,
        shallow_sections: list[str] | None = None,
    ) -> str:
        """Build a clean retry prompt for the gatherer agent.

        The prompt uses neutral instructional language so the LLM does not
        echo warning/error markers into its output.  All diagnostic text
        (e.g. ``[WARNING] ...``) stays in the Python logger — it must
        **never** appear inside a prompt sent to the model.

        Args:
            original_query: The original user query to re-execute.
            missing_sections: Section names absent from the first attempt.
            shallow_sections: Section names present but with insufficient
                content depth.  ``None`` or empty list means no shallow
                sections.  Items prefixed with ``"Investigation Checklist: "``
                are treated as checklist warnings and rendered separately.

        Returns:
            A prompt string free of bracket-prefixed markers.
        """
        parts: list[str] = [original_query, ""]

        if missing_sections:
            sections_list = ", ".join(missing_sections)
            parts.append(
                f"Your previous response was missing the following sections: "
                f"{sections_list}."
            )

        # Separate regular shallow sections from checklist warnings
        regular_shallow: list[str] = []
        checklist_warnings: list[str] = []
        for item in (shallow_sections or []):
            if item.startswith("Investigation Checklist: "):
                checklist_warnings.append(
                    item.removeprefix("Investigation Checklist: ")
                )
            else:
                regular_shallow.append(item)

        if regular_shallow:
            sections_list = ", ".join(regular_shallow)
            parts.append(
                f"The following sections had insufficient data and need more "
                f"detail: {sections_list}. Please collect more detailed data "
                f"for these sections."
            )

        if checklist_warnings:
            parts.append(
                "Your Investigation Checklist had the following issues:"
            )
            for warning in checklist_warnings:
                parts.append(f"  - {warning}")
            parts.append(
                "Please re-run the skipped steps that have contradicting "
                "evidence in your output. Do not skip a step when the data "
                "shows it is needed."
            )

        parts.append(
            "Please regenerate your response including ALL required sections "
            "with comprehensive, detailed data for each."
        )

        return "\n".join(parts)

    @staticmethod
    def _build_deepening_prompt(query: str, first_pass_output: str) -> str:
        """Build an incremental deepening prompt for a mandatory second pass.

        When the first pass already covers all required sections with
        sufficient depth, this prompt instructs the gatherer to fill gaps
        WITHOUT repeating tool calls already in the conversation history.

        The agent keeps its conversation history (no reset), so the prompt
        references "your conversation history above" rather than embedding
        the full first-pass output.

        Args:
            query: The original user query.
            first_pass_output: The full text produced by the first pass
                (kept for API compatibility but NOT embedded in the prompt
                since the conversation history already contains it).

        Returns:
            A prompt string for the second (incremental deepening) pass.
        """
        return (
            "You already completed a first diagnostic pass (see your conversation "
            "history above). A second pass is now running to fill any gaps.\n\n"
            "RULES FOR THIS SECOND PASS:\n"
            "1. Do NOT repeat tool calls you already made — the results are in your "
            "history. Only call tools you have NOT called yet.\n"
            "2. Review your Investigation Checklist from the first pass. For any "
            "step marked SKIPPED, evaluate if it should now be executed.\n"
            "3. Look for these specific gaps:\n"
            "   - Deployments with unavailable replicas that lack ReplicaSet describe\n"
            "   - Warning events that lack corresponding YAML inspection\n"
            "   - HPAs that lack describe output\n"
            "   - Missing Cloud Logging queries for namespaces with issues\n"
            "4. Output ONLY the NEW findings from this second pass. Start with:\n"
            "   ### Second Pass — Additional Findings\n"
            "   Then list only the new tool call results and updated checklist.\n\n"
            f"Original query: {query}"
        )

    def _validate_reporter_output(self, output: str) -> list[str]:
        """Check reporter output for broken Markdown.

        Detects:
        - Table rows that start with ``|`` but do not end with ``|``
        - Output that appears truncated (does not end with sentence-ending
          punctuation such as ``.``, ``!``, ``?``, triple-backtick, ``|``,
          or ``)``).  Only checked when the output is longer than 100
          characters to avoid false positives on short/mock outputs.
        - Unclosed code blocks (odd number of ````` ``` ````` markers)

        Returns:
            A list of human-readable issue descriptions (empty = clean).
        """
        issues: list[str] = []

        # ── Broken table rows ──
        lines = output.split("\n")
        in_table = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                in_table = True
            elif in_table and stripped and not stripped.startswith("|"):
                in_table = False
            elif in_table and stripped.startswith("|") and not stripped.endswith("|"):
                issues.append(f"Broken table row at line {i + 1}")

        # ── Truncated output (only for substantial outputs) ──
        stripped_output = output.rstrip()
        if len(stripped_output) > 100:
            last_char = stripped_output[-1] if stripped_output else ""
            if last_char and last_char not in ".!?\n`|)":
                issues.append(
                    "Output appears truncated "
                    "(does not end with sentence-ending punctuation)"
                )

        # ── Unclosed code blocks ──
        if output.count("```") % 2 != 0:
            issues.append("Unclosed code block (odd number of ``` markers)")

        return issues

    def default_system_instruction(self) -> str:
        """Default system instruction for general chat mode."""
        return (
            f"{ANTI_INJECTION_RULE}\n\n"
            "You are VAIG (Vertex AI Gemini Toolkit), a helpful AI assistant powered by "
            "Google's Gemini models through Vertex AI. You can analyze files, code, logs, "
            "metrics, and data.\n\n"
            "## Response Quality Rules\n"
            "1. **Be specific and technical** — Reference line numbers, file paths, data "
            "points, and concrete examples. Never give vague or generic advice.\n"
            "2. **Explain the WHY, not just the WHAT** — Don't just say what to do, explain "
            "the reasoning behind it. Help the user understand, not just execute.\n"
            "3. **Use proper formatting** — Use markdown headers, code blocks with language "
            "tags, bullet lists, and tables where appropriate. Structure long responses with "
            "clear sections.\n"
            "4. **Complete code examples** — When showing code, always provide complete, "
            "runnable examples with all imports and context. Never use placeholder comments "
            "like `# ... rest of code` or `pass`.\n"
            "5. **Actionable responses** — Every response should end with clear, concrete "
            "next steps the user can take.\n"
            "6. **Admit uncertainty** — If you're not sure about something, say so. Don't "
            "fabricate information."
        )


def _build_tools_summary(agent_role: str, metadata: dict[str, Any] | None) -> str:
    """Build a summary of tool calls for downstream agents.

    When an agent's metadata includes ``tools_executed``, this function
    produces a short Markdown section listing total calls, unique tools,
    and any failures — so the next agent in a sequential pipeline can
    reason about data gaps.

    Args:
        agent_role: The human-readable role of the agent whose tools
            are being summarised (e.g. ``"Gatherer"``).
        metadata: The ``AgentResult.metadata`` dict.

    Returns:
        A Markdown string ready to be appended to the downstream
        context.  Empty string when there is nothing to report.
    """
    if not metadata or not metadata.get("tools_executed"):
        return ""

    tools: list[dict[str, Any]] = metadata["tools_executed"]
    tool_names = [t["name"] for t in tools]
    failed_tools = [t for t in tools if t.get("error")]

    summary = (
        f"\n\n## Tools Executed by {agent_role}\n"
        f"- Total tool calls: {len(tools)}\n"
        f"- Unique tools: {', '.join(sorted(set(tool_names)))}\n"
    )
    if failed_tools:
        summary += (
            f"- Failed calls: {len(failed_tools)} — "
            + ", ".join(
                f"{t['name']}({t.get('args', {})}) → "
                f"{(t.get('output') or 'error')[:80]}"
                for t in failed_tools
            )
            + "\n"
        )
        summary += (
            "NOTE: Failed tool calls may indicate tool gaps "
            "(real K8s resources not yet supported) or agent errors. "
            "Consider these as potential data gaps in your analysis.\n"
        )
    return summary


def _accumulate_usage(result: OrchestratorResult, agent_result: AgentResult) -> None:
    """Accumulate token usage from agent results."""
    for key, value in agent_result.usage.items():
        result.total_usage[key] = result.total_usage.get(key, 0) + value


def _compute_step_cost(agent_result: AgentResult, model_id: str | None) -> float:
    """Return the USD cost for a single agent step, or 0.0 if unknown.

    Delegates to :func:`~vaig.core.pricing.calculate_cost` using the token
    counts reported in *agent_result.usage*.  Returns ``0.0`` whenever
    ``calculate_cost`` returns ``None`` (model not in pricing table), when
    token counts are absent, or when *model_id* is ``None``.
    """
    if model_id is None:
        return 0.0
    usage = agent_result.usage
    cost = calculate_cost(
        model_id,
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
        usage.get("thinking_tokens", 0),
    )
    return cost if cost is not None else 0.0

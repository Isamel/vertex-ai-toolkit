"""Orchestrator — coordinates multi-agent execution for skill-based tasks."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vaig.agents.base import AgentConfig, AgentResult, BaseAgent
from vaig.agents.specialist import SpecialistAgent
from vaig.agents.tool_aware import ToolAwareAgent
from vaig.core.client import GeminiClient
from vaig.core.language import (
    detect_language,
    inject_autopilot_into_config,
    inject_language_into_config,
)
from vaig.skills.base import BaseSkill, SkillPhase, SkillResult
from vaig.tools.base import ToolRegistry

if TYPE_CHECKING:
    from vaig.core.config import Settings

logger = logging.getLogger(__name__)

# ── Gatherer output validation constants ────────────────────────────────
DEFAULT_MIN_CONTENT_CHARS = 50

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

    def __init__(self, client: GeminiClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._agents: dict[str, BaseAgent] = {}

    def create_agents_for_skill(
        self,
        skill: BaseSkill,
        tool_registry: ToolRegistry | None = None,
        *,
        agent_configs: list[dict] | None = None,
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

        for i, agent in enumerate(agents):
            logger.info("Sequential step %d/%d: agent=%s", i + 1, len(agents), agent.name)
            logger.debug(
                "Agent '%s' config: role=%s, model=%s",
                agent.name, agent.role, agent.model,
            )

            # First agent gets the original prompt; subsequent agents get the accumulated context
            if i == 0:
                agent_result = agent.execute(prompt, context=current_context)
            else:
                # Feed previous agent's output as additional context
                accumulated = f"{current_context}\n\n## Previous Analysis ({agents[i - 1].role})\n\n{result.agent_results[-1].content}"
                agent_result = agent.execute(prompt, context=accumulated)

            result.agent_results.append(agent_result)
            _accumulate_usage(result, agent_result)

            logger.debug(
                "Agent '%s' finished: success=%s, tokens=%s",
                agent.name,
                agent_result.success,
                agent_result.usage.get("total_tokens", "?"),
            )

            if not agent_result.success:
                result.success = False
                logger.warning("Agent %s failed: %s", agent.name, agent_result.content)
                break

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
        """Execute all agents independently and merge their outputs.

        Good for parallel analysis: each agent looks at the same data
        from a different perspective.
        """
        agents = self.create_agents_for_skill(skill)
        result = OrchestratorResult(
            skill_name=skill.get_metadata().name,
            phase=phase,
        )

        prompt = skill.get_phase_prompt(phase, context, user_input)

        # Each agent gets the same prompt and context
        for agent in agents:
            logger.info("Fan-out: agent=%s", agent.name)
            agent_result = agent.execute(prompt, context=context)
            result.agent_results.append(agent_result)
            _accumulate_usage(result, agent_result)

            if not agent_result.success:
                logger.warning("Agent %s failed (non-fatal in fan-out): %s", agent.name, agent_result.content)

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
    ) -> AgentResult | Iterator[str]:
        """Execute with a single ad-hoc agent (no skill, direct chat).

        Used for the general chat mode when no skill is active.
        """
        config = AgentConfig(
            name="assistant",
            role="General Assistant",
            system_instruction=system_instruction or self.default_system_instruction(),
            model=model_id or self._settings.models.default,
        )

        agent = SpecialistAgent(config, self._client)

        if stream:
            return agent.execute_stream(prompt, context=context)
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
        logger.info(
            "Executing skill=%s phase=%s strategy=%s",
            skill.get_metadata().name,
            phase,
            strategy,
        )

        if strategy == "fanout":
            orch_result = self.execute_fanout(skill, phase, context, user_input)
        else:
            orch_result = self.execute_sequential(skill, phase, context, user_input)

        return orch_result.to_skill_result()

    def execute_with_tools(
        self,
        query: str,
        skill: BaseSkill,
        tool_registry: ToolRegistry,
        *,
        strategy: str = "sequential",
        is_autopilot: bool | None = None,
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

        Returns:
            :class:`OrchestratorResult` with the aggregated outcome.
        """
        logger.info(
            "execute_with_tools: skill=%s strategy=%s",
            skill.get_metadata().name,
            strategy,
        )

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

        agents = self.create_agents_for_skill(
            skill, tool_registry, agent_configs=agent_configs,
        )
        result = OrchestratorResult(
            skill_name=skill.get_metadata().name,
            phase=SkillPhase.EXECUTE,
        )

        known_strategies = {"sequential", "fanout", "single"}
        if strategy not in known_strategies:
            logger.warning(
                "Unknown strategy '%s' — falling back to sequential. "
                "Valid strategies: %s",
                strategy,
                ", ".join(sorted(known_strategies)),
            )

        if strategy == "fanout":
            for agent in agents:
                logger.info("Fan-out (tools): agent=%s", agent.name)
                agent_result = agent.execute(query, context=query)
                result.agent_results.append(agent_result)
                _accumulate_usage(result, agent_result)
                if not agent_result.success:
                    logger.warning(
                        "Agent %s failed (non-fatal in fan-out): %s",
                        agent.name, agent_result.content,
                    )
            result.success = any(r.success for r in result.agent_results)
            result.synthesized_output = self._merge_agent_outputs(result.agent_results)

        elif strategy == "single":
            if agents:
                agent_result = agents[0].execute(query)
                result.agent_results.append(agent_result)
                _accumulate_usage(result, agent_result)
                result.success = agent_result.success
                result.synthesized_output = agent_result.content
            else:
                result.success = False
                result.synthesized_output = "No agents created for skill."

        else:  # sequential (default)
            current_context = ""
            required_sections = skill.get_required_output_sections()

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
                    current_context = (
                        f"## Previous Analysis ({agents[i - 1].role})\n\n"
                        f"{prev.content}"
                    )

                agent_result = agent.execute(query, context=current_context)
                result.agent_results.append(agent_result)
                _accumulate_usage(result, agent_result)

                logger.debug(
                    "Agent '%s' finished: success=%s, tokens=%s",
                    agent.name,
                    agent_result.success,
                    agent_result.usage.get("total_tokens", "?"),
                )

                if not agent_result.success:
                    result.success = False
                    logger.warning("Agent %s failed: %s", agent.name, agent_result.content)
                    break

                # ── Gatherer output validation + retry ───────────
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
                        logger.debug("Retry prompt: %s", retry_prompt[:200])
                        agent.reset()
                        retry_result = agent.execute(retry_prompt, context="")
                        _accumulate_usage(result, retry_result)

                        # Replace the original result with the retry
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
                    else:
                        logger.debug(
                            "Gatherer validation passed: all %d required sections "
                            "present with sufficient depth",
                            len(required_sections),
                        )

            if result.agent_results:
                result.synthesized_output = result.agent_results[-1].content

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
                sections.

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

        if shallow_sections:
            sections_list = ", ".join(shallow_sections)
            parts.append(
                f"The following sections had insufficient data and need more "
                f"detail: {sections_list}. Please collect more detailed data "
                f"for these sections."
            )

        parts.append(
            "Please regenerate your response including ALL required sections "
            "with comprehensive, detailed data for each."
        )

        return "\n".join(parts)

    def default_system_instruction(self) -> str:
        """Default system instruction for general chat mode."""
        return (
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


def _accumulate_usage(result: OrchestratorResult, agent_result: AgentResult) -> None:
    """Accumulate token usage from agent results."""
    for key, value in agent_result.usage.items():
        result.total_usage[key] = result.total_usage.get(key, 0) + value

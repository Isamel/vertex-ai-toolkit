"""Orchestrator — coordinates multi-agent execution for skill-based tasks."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vaig.agents.base import AgentConfig, AgentResult, BaseAgent
from vaig.agents.specialist import SpecialistAgent
from vaig.core.client import GeminiClient
from vaig.skills.base import BaseSkill, SkillPhase, SkillResult

if TYPE_CHECKING:
    from vaig.core.config import Settings

logger = logging.getLogger(__name__)


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
        self._agents: dict[str, SpecialistAgent] = {}

    def create_agents_for_skill(self, skill: BaseSkill) -> list[SpecialistAgent]:
        """Create specialist agents based on a skill's configuration.

        Each skill defines its own agent configs (roles, models, instructions).
        """
        self._agents.clear()
        agents: list[SpecialistAgent] = []

        for config_dict in skill.get_agents_config():
            agent = SpecialistAgent.from_config_dict(config_dict, self._client)
            self._agents[agent.name] = agent
            agents.append(agent)
            logger.info("Created agent: %s (role=%s, model=%s)", agent.name, agent.role, agent.model)

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

            # First agent gets the original prompt; subsequent agents get the accumulated context
            if i == 0:
                agent_result = agent.execute(prompt, context=current_context)
            else:
                # Feed previous agent's output as additional context
                accumulated = f"{current_context}\n\n## Previous Analysis ({agents[i - 1].role})\n\n{result.agent_results[-1].content}"
                agent_result = agent.execute(prompt, context=accumulated)

            result.agent_results.append(agent_result)
            _accumulate_usage(result, agent_result)

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
            system_instruction=system_instruction or self._default_system_instruction(),
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

    def get_agent(self, name: str) -> SpecialistAgent | None:
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

    def _default_system_instruction(self) -> str:
        """Default system instruction for general chat mode."""
        return (
            "You are VAIG (Vertex AI Gemini Toolkit), a helpful AI assistant powered by "
            "Google's Gemini models through Vertex AI. You can analyze files, code, logs, "
            "metrics, and data. You provide clear, technical, and actionable responses. "
            "When analyzing code or data, be specific and reference line numbers or data points."
        )


def _accumulate_usage(result: OrchestratorResult, agent_result: AgentResult) -> None:
    """Accumulate token usage from agent results."""
    for key, value in agent_result.usage.items():
        result.total_usage[key] = result.total_usage.get(key, 0) + value

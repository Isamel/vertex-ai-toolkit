"""Agent registry — manages agent lifecycle and provides factory methods."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vaig.agents.base import AgentConfig
from vaig.agents.specialist import SpecialistAgent

if TYPE_CHECKING:
    from vaig.core.client import GeminiClient
    from vaig.core.config import Settings
    from vaig.skills.base import BaseSkill

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Registry for creating and managing agent instances.

    Provides factory methods for creating agents from various sources:
    - From skill configurations
    - From explicit AgentConfig objects
    - Ad-hoc agents for general chat
    """

    def __init__(self, client: GeminiClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._agents: dict[str, SpecialistAgent] = {}

    def create_from_skill(self, skill: BaseSkill) -> list[SpecialistAgent]:
        """Create agents defined by a skill's configuration.

        Returns the list of created agents.
        """
        agents: list[SpecialistAgent] = []

        for config_dict in skill.get_agents_config():
            agent = SpecialistAgent.from_config_dict(config_dict, self._client)
            self._agents[agent.name] = agent
            agents.append(agent)

        logger.info(
            "Created %d agents for skill '%s': %s",
            len(agents),
            skill.get_metadata().name,
            [a.name for a in agents],
        )
        return agents

    def create_from_config(self, config: AgentConfig) -> SpecialistAgent:
        """Create an agent from an explicit AgentConfig."""
        agent = SpecialistAgent(config, self._client)
        self._agents[agent.name] = agent
        logger.info("Created agent: %s", agent.name)
        return agent

    def create_chat_agent(
        self,
        *,
        model_id: str | None = None,
        system_instruction: str = "",
    ) -> SpecialistAgent:
        """Create a general-purpose chat agent."""
        config = AgentConfig(
            name="chat_assistant",
            role="General Assistant",
            system_instruction=system_instruction
            or (
                "You are VAIG (Vertex AI Gemini Toolkit), a helpful AI assistant powered by "
                "Google's Gemini models through Vertex AI. You can analyze files, code, logs, "
                "metrics, and data.\n\n"
                "## Response Quality Rules\n"
                "1. **Be specific and technical** — Reference line numbers, file paths, data "
                "points, and concrete examples. Never give vague or generic advice.\n"
                "2. **Explain the WHY, not just the WHAT** — Don't just say what to do, explain "
                "the reasoning behind it.\n"
                "3. **Use proper formatting** — Use markdown headers, code blocks with language "
                "tags, bullet lists, and tables where appropriate.\n"
                "4. **Complete code examples** — When showing code, always provide complete, "
                "runnable examples with all imports. Never use placeholder comments.\n"
                "5. **Actionable responses** — Every response should end with clear, concrete "
                "next steps.\n"
                "6. **Admit uncertainty** — If you're not sure, say so."
            ),
            model=model_id or self._settings.models.default,
        )
        return self.create_from_config(config)

    def get(self, name: str) -> SpecialistAgent | None:
        """Get a registered agent by name."""
        return self._agents.get(name)

    def list_agents(self) -> list[SpecialistAgent]:
        """List all registered agents."""
        return list(self._agents.values())

    def remove(self, name: str) -> bool:
        """Remove an agent from the registry."""
        if name in self._agents:
            del self._agents[name]
            return True
        return False

    def clear(self) -> None:
        """Remove all agents."""
        self._agents.clear()

    def reset_all(self) -> None:
        """Reset conversation history for all agents."""
        for agent in self._agents.values():
            agent.reset()

    @property
    def count(self) -> int:
        """Number of registered agents."""
        return len(self._agents)

"""Specialist agent — a concrete agent that wraps GeminiClient with a specific role."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from vaig.agents.base import AgentConfig, AgentResult, BaseAgent
from vaig.core.client import ChatMessage, GeminiClient

logger = logging.getLogger(__name__)


class SpecialistAgent(BaseAgent):
    """A specialist agent with a fixed role and system instruction.

    This is the workhorse of the multi-agent system. Each specialist has:
    - A dedicated system instruction defining its expertise
    - Its own model (can be different from other agents)
    - Conversation history for multi-turn within a task

    Used by skills to define specialized agents (e.g., log_analyzer, code_generator).
    """

    def __init__(self, config: AgentConfig, client: GeminiClient) -> None:
        super().__init__(config, client)

    def execute(self, prompt: str, *, context: str = "") -> AgentResult:
        """Execute a task using the configured model and system instruction.

        Builds a full prompt with optional context, sends to Gemini,
        and returns a structured result.
        """
        full_prompt = self._build_prompt(prompt, context)
        history = self._build_chat_history()

        self._add_to_conversation("user", full_prompt)

        try:
            result = self._client.generate(
                full_prompt,
                system_instruction=self._config.system_instruction,
                history=history,
                model_id=self._config.model,
                temperature=self._config.temperature,
                max_output_tokens=self._config.max_output_tokens,
            )

            self._add_to_conversation("agent", result.text)

            return AgentResult(
                agent_name=self.name,
                content=result.text,
                success=True,
                usage=result.usage,
                metadata={"model": result.model, "finish_reason": result.finish_reason},
            )

        except Exception as e:
            logger.exception("Agent %s failed", self.name)
            return AgentResult(
                agent_name=self.name,
                content=f"Error: {e}",
                success=False,
                metadata={"error": str(e)},
            )

    def execute_stream(self, prompt: str, *, context: str = "") -> Iterator[str]:
        """Execute a task with streaming output.

        Yields text chunks as they arrive from Gemini.
        Accumulates the full response for conversation history.
        """
        full_prompt = self._build_prompt(prompt, context)
        history = self._build_chat_history()

        self._add_to_conversation("user", full_prompt)

        accumulated: list[str] = []

        try:
            for chunk in self._client.generate_stream(
                full_prompt,
                system_instruction=self._config.system_instruction,
                history=history,
                model_id=self._config.model,
                temperature=self._config.temperature,
                max_output_tokens=self._config.max_output_tokens,
            ):
                accumulated.append(chunk)
                yield chunk

            full_response = "".join(accumulated)
            self._add_to_conversation("agent", full_response)

        except Exception as e:
            logger.exception("Agent %s streaming failed", self.name)
            yield f"\n[Error: {e}]"

    def _build_prompt(self, prompt: str, context: str) -> str:
        """Build the full prompt with optional context."""
        if context:
            return f"## Context\n\n{context}\n\n## Task\n\n{prompt}"
        return prompt

    def _build_chat_history(self) -> list[ChatMessage]:
        """Convert agent conversation history to ChatMessage format for Gemini."""
        messages: list[ChatMessage] = []

        for msg in self._conversation:
            # Map agent roles to Gemini roles
            role = "user" if msg.role == "user" else "model"
            messages.append(ChatMessage(role=role, content=msg.content))

        return messages

    @classmethod
    def from_config_dict(cls, config_dict: dict[str, Any], client: GeminiClient) -> SpecialistAgent:
        """Factory method to create a SpecialistAgent from a skill's agent config dict.

        This maps the dict format used by BaseSkill.get_agents_config() to AgentConfig.

        Raises:
            TypeError: If config_dict is not a dict (e.g. a list was passed).
            KeyError: If required keys (name, role, system_instruction) are missing.
        """
        if not isinstance(config_dict, dict):
            raise TypeError(
                f"Expected dict for agent config, got {type(config_dict).__name__}. "
                "Check that get_agents_config() returns a list of dicts."
            )
        agent_config = AgentConfig(
            name=config_dict["name"],
            role=config_dict["role"],
            system_instruction=config_dict["system_instruction"],
            model=config_dict.get("model", "gemini-2.5-pro"),
            temperature=config_dict.get("temperature", 0.7),
            max_output_tokens=config_dict.get("max_output_tokens", 8192),
        )
        return cls(agent_config, client)

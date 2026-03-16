"""Specialist agent — a concrete agent that wraps GeminiClient with a specific role."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

from vaig.agents.base import AgentConfig, AgentResult, BaseAgent
from vaig.core.client import ChatMessage, GeminiClient
from vaig.core.exceptions import GeminiConnectionError, GeminiRateLimitError

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
        logger.debug("Agent %s executing (model=%s, history=%d)", self.name, self._config.model, len(history))

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
            logger.info("Agent %s completed — %s tokens", self.name, result.usage.get("total_tokens", "?"))

            return AgentResult(
                agent_name=self.name,
                content=result.text,
                success=True,
                usage=result.usage,
                metadata={"model": result.model, "finish_reason": result.finish_reason},
            )

        except GeminiRateLimitError as e:
            logger.warning("Agent %s rate-limited after %d retries", self.name, e.retries_attempted)
            return AgentResult(
                agent_name=self.name,
                content=f"Rate limit exceeded (retried {e.retries_attempted}x): {e}",
                success=False,
                metadata={"error": str(e), "error_type": "rate_limit"},
            )

        except GeminiConnectionError as e:
            logger.warning("Agent %s connection error after %d retries", self.name, e.retries_attempted)
            return AgentResult(
                agent_name=self.name,
                content=f"Connection error (retried {e.retries_attempted}x): {e}",
                success=False,
                metadata={"error": str(e), "error_type": "connection"},
            )

        except Exception as e:
            logger.exception("Agent %s failed", self.name)
            return AgentResult(
                agent_name=self.name,
                content=self.sanitize_error_for_agent(e),
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

        except GeminiRateLimitError as e:
            logger.warning("Agent %s streaming rate-limited after %d retries", self.name, e.retries_attempted)
            yield f"\n[Rate limit exceeded (retried {e.retries_attempted}x): {e}]"

        except GeminiConnectionError as e:
            logger.warning("Agent %s streaming connection error after %d retries", self.name, e.retries_attempted)
            yield f"\n[Connection error (retried {e.retries_attempted}x): {e}]"

        except Exception as e:
            logger.exception("Agent %s streaming failed", self.name)
            yield f"\n[Error: {self.sanitize_error_for_agent(e)}]"

    # ── Async methods ────────────────────────────────────────

    async def async_execute(self, prompt: str, *, context: str = "") -> AgentResult:
        """Execute a task using the configured model (async).

        Async version of :meth:`execute`.  Uses ``client.async_generate()``
        for non-blocking LLM calls.  Same error handling and conversation
        tracking as the sync counterpart.
        """
        full_prompt = self._build_prompt(prompt, context)
        history = self._build_chat_history()

        self._add_to_conversation("user", full_prompt)
        logger.debug(
            "Agent %s async_execute (model=%s, history=%d)",
            self.name,
            self._config.model,
            len(history),
        )

        try:
            result = await self._client.async_generate(
                full_prompt,
                system_instruction=self._config.system_instruction,
                history=history,
                model_id=self._config.model,
                temperature=self._config.temperature,
                max_output_tokens=self._config.max_output_tokens,
            )

            self._add_to_conversation("agent", result.text)
            logger.info(
                "Agent %s async completed — %s tokens",
                self.name,
                result.usage.get("total_tokens", "?"),
            )

            return AgentResult(
                agent_name=self.name,
                content=result.text,
                success=True,
                usage=result.usage,
                metadata={"model": result.model, "finish_reason": result.finish_reason},
            )

        except GeminiRateLimitError as e:
            logger.warning(
                "Agent %s async rate-limited after %d retries",
                self.name,
                e.retries_attempted,
            )
            return AgentResult(
                agent_name=self.name,
                content=f"Rate limit exceeded (retried {e.retries_attempted}x): {e}",
                success=False,
                metadata={"error": str(e), "error_type": "rate_limit"},
            )

        except GeminiConnectionError as e:
            logger.warning(
                "Agent %s async connection error after %d retries",
                self.name,
                e.retries_attempted,
            )
            return AgentResult(
                agent_name=self.name,
                content=f"Connection error (retried {e.retries_attempted}x): {e}",
                success=False,
                metadata={"error": str(e), "error_type": "connection"},
            )

        except Exception as e:
            logger.exception("Agent %s async failed", self.name)
            return AgentResult(
                agent_name=self.name,
                content=self.sanitize_error_for_agent(e),
                success=False,
                metadata={"error": str(e)},
            )

    async def async_execute_stream(self, prompt: str, *, context: str = "") -> AsyncIterator[str]:
        """Execute a task with async streaming output.

        Async version of :meth:`execute_stream`.  Uses
        ``client.async_generate_stream()`` for non-blocking streaming.
        Yields text chunks as they arrive from Gemini.
        """
        full_prompt = self._build_prompt(prompt, context)
        history = self._build_chat_history()

        self._add_to_conversation("user", full_prompt)

        accumulated: list[str] = []

        try:
            stream_result = await self._client.async_generate_stream(
                full_prompt,
                system_instruction=self._config.system_instruction,
                history=history,
                model_id=self._config.model,
                temperature=self._config.temperature,
                max_output_tokens=self._config.max_output_tokens,
            )

            async for chunk in stream_result:
                accumulated.append(chunk)
                yield chunk

            full_response = "".join(accumulated)
            self._add_to_conversation("agent", full_response)

        except GeminiRateLimitError as e:
            logger.warning(
                "Agent %s async streaming rate-limited after %d retries",
                self.name,
                e.retries_attempted,
            )
            yield f"\n[Rate limit exceeded (retried {e.retries_attempted}x): {e}]"

        except GeminiConnectionError as e:
            logger.warning(
                "Agent %s async streaming connection error after %d retries",
                self.name,
                e.retries_attempted,
            )
            yield f"\n[Connection error (retried {e.retries_attempted}x): {e}]"

        except Exception as e:
            logger.exception("Agent %s async streaming failed", self.name)
            yield f"\n[Error: {self.sanitize_error_for_agent(e)}]"

    def _build_chat_history(self) -> list[ChatMessage]:
        """Convert agent conversation history to ChatMessage format for Gemini.

        Overrides ``BaseAgent._build_chat_history`` because SpecialistAgent
        uses the lightweight ``ChatMessage`` dataclass (text-only history)
        rather than ``types.Content`` objects.
        """
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
            max_output_tokens=config_dict.get("max_output_tokens", 16384),
        )
        return cls(agent_config, client)

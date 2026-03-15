"""Base agent — abstract contract for all agents in the system."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from google.genai import types

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from vaig.core.client import GeminiClient

logger = logging.getLogger(__name__)


class AgentRole(StrEnum):
    """Standard agent roles."""

    ORCHESTRATOR = "orchestrator"
    SPECIALIST = "specialist"
    ASSISTANT = "assistant"
    CODER = "coder"
    SRE = "sre"


@dataclass
class AgentConfig:
    """Configuration for creating an agent."""

    name: str
    role: str
    system_instruction: str
    model: str = "gemini-2.5-pro"
    temperature: float = 0.7
    max_output_tokens: int = 16384


@dataclass
class AgentMessage:
    """A message in an agent's conversation."""

    role: str  # "user" | "agent" | "system"
    content: str
    agent_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result from an agent's execution."""

    agent_name: str
    content: str
    success: bool = True
    usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for all agents.

    An agent wraps a GeminiClient with a specific role, system instruction,
    and optional tool integration. Agents are the execution units in the
    multi-agent system.
    """

    def __init__(self, config: AgentConfig, client: GeminiClient) -> None:
        self._config = config
        self._client = client
        self._conversation: list[AgentMessage] = []

    @property
    def name(self) -> str:
        """Agent's unique name."""
        return self._config.name

    @property
    def role(self) -> str:
        """Agent's role description."""
        return self._config.role

    @property
    def model(self) -> str:
        """Agent's model ID."""
        return self._config.model

    @property
    def config(self) -> AgentConfig:
        """Agent's configuration."""
        return self._config

    @property
    def conversation_history(self) -> list[AgentMessage]:
        """Agent's conversation history."""
        return self._conversation

    @abstractmethod
    def execute(self, prompt: str, *, context: str = "") -> AgentResult:
        """Execute a task and return a result.

        Args:
            prompt: The task or question for the agent.
            context: Optional context (files, data, previous results).

        Returns:
            AgentResult with the agent's response.
        """
        ...

    @abstractmethod
    def execute_stream(self, prompt: str, *, context: str = "") -> Iterator[str]:
        """Execute a task with streaming output.

        Args:
            prompt: The task or question for the agent.
            context: Optional context.

        Yields:
            Text chunks as they are generated.
        """
        ...

    def reset(self) -> None:
        """Reset the agent's conversation history."""
        self._conversation.clear()
        logger.debug("Agent %s conversation reset", self.name)

    def _add_to_conversation(self, role: str, content: str) -> None:
        """Track a message in the conversation."""
        self._conversation.append(
            AgentMessage(role=role, content=content, agent_name=self.name)
        )

    # ── Async abstract methods ─────────────────────────────────

    @abstractmethod
    async def async_execute(self, prompt: str, *, context: str = "") -> AgentResult:
        """Execute a task and return a result (async).

        Async version of :meth:`execute`.  Uses non-blocking I/O for
        LLM API calls and tool execution.

        Args:
            prompt: The task or question for the agent.
            context: Optional context (files, data, previous results).

        Returns:
            AgentResult with the agent's response.
        """
        ...

    @abstractmethod
    async def async_execute_stream(self, prompt: str, *, context: str = "") -> AsyncIterator[str]:
        """Execute a task with streaming output (async).

        Async version of :meth:`execute_stream`.  Uses non-blocking I/O
        for LLM API calls.

        Args:
            prompt: The task or question for the agent.
            context: Optional context.

        Yields:
            Text chunks as they are generated.
        """
        ...
        # Trick: yield inside an abstract async generator so Python treats
        # this as an async generator function rather than a plain coroutine.
        yield ""  # type: ignore[misc]  # pragma: no cover

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} role={self.role!r} model={self.model!r}>"

    # ── Shared helpers (subclasses may override) ──────────────

    def _build_prompt(self, prompt: str, context: str) -> str:
        """Build the full prompt with optional context.

        Subclasses inherit this as-is unless they need custom formatting.
        """
        if context:
            return f"## Context\n\n{context}\n\n## Task\n\n{prompt}"
        return prompt

    def _build_chat_history(self) -> list[Any]:
        """Convert conversation history to Gemini ``Content`` objects.

        Returns ``types.Content`` instances suitable for the chat API.
        Subclasses that need a different format (e.g. ``ChatMessage``)
        should override this method.
        """
        contents: list[types.Content] = []
        for msg in self._conversation:
            role = "user" if msg.role == "user" else "model"
            contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]),
            )
        return contents

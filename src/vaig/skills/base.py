"""Base skill protocol — defines the contract for all skills."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum


class SkillPhase(StrEnum):
    """Phases a skill can execute."""

    ANALYZE = "analyze"
    PLAN = "plan"
    EXECUTE = "execute"
    VALIDATE = "validate"
    REPORT = "report"


@dataclass
class SkillMetadata:
    """Metadata describing a skill."""

    name: str
    display_name: str
    description: str
    version: str = "1.0.0"
    author: str = "vaig"
    tags: list[str] = field(default_factory=list)
    supported_phases: list[SkillPhase] = field(
        default_factory=lambda: [SkillPhase.ANALYZE, SkillPhase.EXECUTE, SkillPhase.REPORT]
    )
    recommended_model: str = "gemini-2.5-pro"


@dataclass
class SkillResult:
    """Result from a skill execution phase."""

    phase: SkillPhase
    success: bool
    output: str
    artifacts: dict[str, str] = field(default_factory=dict)  # name → content
    metadata: dict = field(default_factory=dict)
    next_phase: SkillPhase | None = None


class BaseSkill(ABC):
    """Abstract base class for all skills.

    A Skill defines:
    - System instructions for the AI model
    - Phase-specific prompts
    - Output formatting
    - Agent configuration (which agents to spawn)

    Skills are stateless — state lives in the session.
    """

    @abstractmethod
    def get_metadata(self) -> SkillMetadata:
        """Return skill metadata."""
        ...

    @abstractmethod
    def get_system_instruction(self) -> str:
        """Return the system instruction for this skill.

        This is injected as the system prompt when the skill is active.
        """
        ...

    @abstractmethod
    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        """Build a prompt for a specific phase.

        Args:
            phase: The skill phase to execute.
            context: The loaded context (files, data, etc.).
            user_input: The user's specific request.
        """
        ...

    def get_agents_config(self) -> list[dict]:
        """Return agent configurations for multi-agent execution.

        Override this to define specialized agents for the skill.
        Default: single agent with the skill's system instruction.
        """
        meta = self.get_metadata()
        return [
            {
                "name": meta.name,
                "role": meta.display_name,
                "system_instruction": self.get_system_instruction(),
                "model": meta.recommended_model,
            }
        ]

    def format_output(self, result: SkillResult) -> str:
        """Format the skill result for display. Override for custom formatting."""
        return result.output

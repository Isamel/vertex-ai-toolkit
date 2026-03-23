"""Base skill protocol — defines the contract for all skills."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from vaig.core.models import PipelineState


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
    requires_live_tools: bool = False


@dataclass
class SkillResult:
    """Result from a skill execution phase."""

    phase: SkillPhase
    success: bool
    output: str
    artifacts: dict[str, str] = field(default_factory=dict)  # name → content
    metadata: dict[str, Any] = field(default_factory=dict)
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

    def get_initial_state(self) -> PipelineState | None:
        """Return the initial pipeline state for this skill.

        Called once by the orchestrator before any agent executes.
        Override in subclasses that need to seed the shared state with
        skill-specific data (e.g. namespace, cluster context, config flags).

        The default implementation returns ``None``, which disables
        state-threading for skills that have not opted in.

        Returns:
            A :class:`~vaig.core.models.PipelineState` instance to seed the
            pipeline, or ``None`` to leave state-threading disabled.
        """
        return None

    def get_agents_config(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Return agent configurations for multi-agent execution.

        Override this to define specialized agents for the skill.
        Default: single agent with the skill's system instruction.

        The ``**kwargs`` signature allows the orchestrator to always pass
        ``namespace=``, ``location=``, and ``cluster_name=`` without breaking
        subclasses that don't need those parameters.  Subclasses that do need
        them (e.g. :class:`~vaig.skills.service_health.skill.ServiceHealthSkill`)
        may declare explicit keyword arguments — mypy accepts a more specific
        signature when the base uses ``**kwargs``.

        Args:
            **kwargs: Caller-supplied keyword arguments.  The orchestrator
                passes ``namespace``, ``location``, and ``cluster_name``.
                Subclasses that don't use them simply ignore them via
                ``**kwargs``.
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

    def get_required_output_sections(self) -> list[str] | None:
        """Return mandatory sections expected in the first agent's output.

        When this returns a non-``None`` list, the orchestrator validates the
        first agent's output and retries once if any sections are missing.

        Override in subclasses whose first agent must produce structured output
        (e.g. the service-health gatherer).  Return ``None`` (the default) to
        skip validation entirely.
        """
        return None

    def post_process_report(self, content: str) -> str:
        """Post-process the reporter agent's raw output before validation.

        Called by the orchestrator on the last agent's output when the agent
        has role containing ``"report"``.  Override in subclasses that use
        Gemini's structured output mode (``response_schema``) to convert the
        raw JSON response into the desired display format (e.g. Markdown).

        The default implementation returns *content* unchanged, preserving
        backward compatibility for skills that do not use structured output.
        """
        return content

    def pre_execute_parallel(self, query: str) -> None:  # noqa: ARG002, B027
        """Hook called once before any parallel agents are launched.

        Override in subclasses that need to perform sequential initialization
        before threads start — e.g. pre-warming a client cache that is not
        thread-safe on first write.

        The default implementation is a no-op, preserving backward
        compatibility for skills that do not require pre-warming.

        Args:
            query: The user query string passed to the skill execution.
        """

    def format_output(self, result: SkillResult) -> str:
        """Format the skill result for display. Override for custom formatting."""
        return result.output


class CompositeSkill(BaseSkill):
    """Combine multiple skills into a single unified skill.

    Merges system instructions, agent configs, tags, and phases from all
    component skills.  Useful when an investigation requires expertise from
    several domains (e.g. RCA + log-analysis + cost-analysis).
    """

    def __init__(self, skills: list[BaseSkill], *, name: str | None = None) -> None:
        if len(skills) < 2:  # noqa: PLR2004
            msg = "CompositeSkill requires at least 2 component skills."
            raise ValueError(msg)

        self._skills = skills
        metas = [s.get_metadata() for s in skills]

        self._name = name or "+".join(m.name for m in metas)
        self._display_name = " + ".join(m.display_name for m in metas)
        self._description = "Composite: " + "; ".join(m.description for m in metas)

        # Merge tags (deduplicated, order preserved)
        seen: set[str] = set()
        merged_tags: list[str] = []
        for m in metas:
            for tag in m.tags:
                if tag not in seen:
                    seen.add(tag)
                    merged_tags.append(tag)
        self._tags = merged_tags

        # Union of supported phases (order preserved by StrEnum)
        all_phases = {p for m in metas for p in m.supported_phases}
        self._phases = sorted(all_phases, key=lambda p: list(SkillPhase).index(p))

        # Use the first skill's recommended model
        self._model = metas[0].recommended_model

        # Requires live tools if ANY component does
        self._requires_live_tools = any(m.requires_live_tools for m in metas)

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name=self._name,
            display_name=self._display_name,
            description=self._description,
            version="1.0.0",
            author="vaig",
            tags=self._tags,
            supported_phases=self._phases,
            recommended_model=self._model,
            requires_live_tools=self._requires_live_tools,
        )

    def get_system_instruction(self) -> str:
        sections = []
        for skill in self._skills:
            meta = skill.get_metadata()
            sections.append(
                f"## {meta.display_name}\n\n{skill.get_system_instruction()}"
            )
        return (
            "You are a composite specialist combining expertise from multiple domains.\n"
            "Apply ALL of the following expertise areas to your analysis:\n\n"
            + "\n\n---\n\n".join(sections)
        )

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        """Merge phase prompts from all component skills that support the phase."""
        prompts: list[str] = []
        for skill in self._skills:
            meta = skill.get_metadata()
            if phase in meta.supported_phases:
                prompts.append(
                    f"### {meta.display_name} perspective\n\n"
                    + skill.get_phase_prompt(phase, context, user_input)
                )

        if not prompts:
            return f"No component skills support phase '{phase.value}'."

        return (
            "Apply ALL of the following analysis perspectives to the input:\n\n"
            + "\n\n---\n\n".join(prompts)
        )

    def get_agents_config(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Merge agent configs from all component skills (deduplicated by name)."""
        agents: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for skill in self._skills:
            for agent in skill.get_agents_config(**kwargs):
                if agent["name"] not in seen_names:
                    seen_names.add(agent["name"])
                    agents.append(agent)
        return agents

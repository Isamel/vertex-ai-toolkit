"""Greenfield Skill — 6-stage structured project generation.

Stages (in order):
  1. REQUIREMENTS — Extract and structure requirements from user input
  2. ARCHITECTURE_DECISION — Produce ADRs for key technical choices
  3. PROJECT_SPEC — Translate ADRs into file-level implementation spec
  4. SCAFFOLD — Create project skeleton (config, CI, stubs)
  5. IMPLEMENT — Replace every stub with production-ready code
  6. VERIFY — Validate the complete project against requirements

Usage::

    skill = GreenfieldSkill()
    prompt = skill.get_phase_prompt(
        SkillPhase.ANALYZE, context="", user_input="Build a REST API in Python"
    )

The skill maps :class:`~vaig.skills.base.SkillPhase` enum values to internal
Greenfield stages as follows:

  SkillPhase.ANALYZE  → requirements
  SkillPhase.PLAN     → architecture_decision, project_spec
  SkillPhase.EXECUTE  → scaffold, implement
  SkillPhase.VALIDATE → verify
  SkillPhase.REPORT   → verify (summary report)

When a SkillPhase maps to multiple stages, :meth:`get_phase_prompt` returns
the prompt for the **first** stage in that phase (e.g. ``architecture_decision``
for ``PLAN``, ``scaffold`` for ``EXECUTE``).  Use :meth:`get_stage_prompt` to
access any individual stage by name.
"""

from __future__ import annotations

from typing import Any

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.greenfield.prompts import (
    STAGE_ORDER,
    STAGE_PROMPTS,
    SYSTEM_INSTRUCTION,
)


class GreenfieldSkill(BaseSkill):
    """Greenfield project generation skill with 6 sequential stages.

    Maps :class:`~vaig.skills.base.SkillPhase` values to Greenfield-specific
    stages.  Each stage builds on the output of the previous one via the
    ``context`` parameter of :meth:`get_phase_prompt`.

    Supported phase mapping:

    +-------------------+---------------------------------------------+
    | SkillPhase        | Greenfield Stage(s)                         |
    +===================+=============================================+
    | ANALYZE           | requirements                                |
    +-------------------+---------------------------------------------+
    | PLAN              | architecture_decision, project_spec         |
    +-------------------+---------------------------------------------+
    | EXECUTE           | scaffold, implement                         |
    +-------------------+---------------------------------------------+
    | VALIDATE          | verify                                      |
    +-------------------+---------------------------------------------+
    | REPORT            | verify (summary report)                     |
    +-------------------+---------------------------------------------+

    When a phase maps to multiple stages, :meth:`get_phase_prompt` returns
    the prompt for the **first** stage in that list.  Use
    :meth:`get_stage_prompt` to address individual stages by name.
    """

    # Maps SkillPhase → ordered list of Greenfield stage name(s).
    # When a phase covers multiple stages the first entry is returned by
    # get_phase_prompt(); callers that need sub-stage granularity should use
    # get_stage_prompt() directly.
    _PHASE_TO_STAGE: dict[SkillPhase, list[str]] = {
        SkillPhase.ANALYZE: ["requirements"],
        SkillPhase.PLAN: ["architecture_decision", "project_spec"],
        SkillPhase.EXECUTE: ["scaffold", "implement"],
        SkillPhase.VALIDATE: ["verify"],
        SkillPhase.REPORT: ["verify"],
    }

    def get_metadata(self) -> SkillMetadata:
        """Return skill metadata."""
        return SkillMetadata(
            name="greenfield",
            display_name="Greenfield Project Generator",
            description=(
                "6-stage structured project generation: Requirements → "
                "Architecture Decision → Project Spec → Scaffold → Implement → Verify"
            ),
            version="1.0.0",
            author="vaig",
            tags=[
                "greenfield",
                "project-generation",
                "scaffold",
                "architecture",
                "full-stack",
            ],
            supported_phases=[
                SkillPhase.ANALYZE,
                SkillPhase.PLAN,
                SkillPhase.EXECUTE,
                SkillPhase.VALIDATE,
                SkillPhase.REPORT,
            ],
            recommended_model="gemini-2.5-pro",
            requires_live_tools=True,
        )

    def get_system_instruction(self) -> str:
        """Return the system instruction for this skill."""
        return SYSTEM_INSTRUCTION

    def get_phase_prompt(
        self, phase: SkillPhase, context: str, user_input: str
    ) -> str:
        """Build a prompt for a specific Greenfield phase.

        When a phase maps to multiple stages (e.g. ``PLAN`` → ``architecture_decision``,
        ``project_spec``), the prompt for the **first** stage in that list is
        returned.  Use :meth:`get_stage_prompt` to address sub-stages individually.

        Args:
            phase: The :class:`~vaig.skills.base.SkillPhase` to execute.
                Maps to one or more Greenfield stages via :attr:`_PHASE_TO_STAGE`.
                The first stage in the list is used.
            context: Output from the previous stage (passed as ``{context}``
                in the prompt template).  Empty string for the first stage.
            user_input: The user's project description / request (passed as
                ``{user_input}`` in the prompt template).

        Returns:
            Formatted prompt string ready to send to the model.
        """
        stages = self._PHASE_TO_STAGE.get(phase, ["requirements"])
        stage_name = stages[0]
        template = STAGE_PROMPTS.get(stage_name, STAGE_PROMPTS["requirements"])
        return template.format(context=context, user_input=user_input)

    def get_stage_prompt(self, stage: str, context: str, user_input: str) -> str:
        """Build a prompt for a named Greenfield stage directly.

        This method allows callers to address Greenfield stages by name
        (``"requirements"``, ``"architecture_decision"``, ``"project_spec"``,
        ``"scaffold"``, ``"implement"``, ``"verify"``) rather than via the
        :class:`~vaig.skills.base.SkillPhase` enum.

        Args:
            stage: One of the stage names from :data:`~vaig.skills.greenfield.prompts.STAGE_ORDER`.
            context: Output from the previous stage.  Empty for ``requirements``.
            user_input: The user's project description or follow-up request.

        Returns:
            Formatted prompt string.

        Raises:
            ValueError: If ``stage`` is not a recognised Greenfield stage name.
        """
        if stage not in STAGE_PROMPTS:
            valid = ", ".join(STAGE_ORDER)
            msg = f"Unknown Greenfield stage {stage!r}. Valid stages: {valid}"
            raise ValueError(msg)
        template = STAGE_PROMPTS[stage]
        return template.format(context=context, user_input=user_input)

    @property
    def stage_order(self) -> list[str]:
        """Ordered list of Greenfield stage names."""
        return list(STAGE_ORDER)

    def get_agents_config(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Return single-agent config for Greenfield execution.

        Greenfield uses a single high-capability agent per stage.  The
        orchestrator is responsible for passing the previous stage's output
        as context when calling :meth:`get_phase_prompt`.
        """
        return [
            {
                "name": "greenfield-agent",
                "role": "Principal Software Architect",
                "system_instruction": self.get_system_instruction(),
                "model": self.get_metadata().recommended_model,
            }
        ]

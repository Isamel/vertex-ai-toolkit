"""ADR Generator Skill — architecture decision record generation."""

from __future__ import annotations

from typing import Any

from vaig.skills.adr_generator.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION
from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase


class AdrGeneratorSkill(BaseSkill):
    """ADR Generator skill for creating architecture decision records.

    Supports multi-agent execution with specialized agents:
    - Context Researcher: Extracts decision drivers, constraints, and stakeholder concerns
    - ADR Author: Generates publication-ready ADRs in MADR format
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="adr-generator",
            display_name="ADR Generator",
            description=(
                "Generate architecture decision records (ADRs) from context, "
                "conversations, and requirements using MADR format"
            ),
            version="1.0.0",
            tags=[
                "documentation",
                "architecture",
                "decision-record",
                "adr",
                "technical-writing",
            ],
            supported_phases=[
                SkillPhase.ANALYZE,
                SkillPhase.PLAN,
                SkillPhase.EXECUTE,
                SkillPhase.VALIDATE,
                SkillPhase.REPORT,
            ],
            recommended_model="gemini-2.5-flash",
        )

    def get_system_instruction(self) -> str:
        return SYSTEM_INSTRUCTION

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        template = PHASE_PROMPTS.get(phase.value, PHASE_PROMPTS["analyze"])
        return template.format(context=context, user_input=user_input)

    def get_agents_config(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "name": "context_researcher",
                "role": "Context Researcher",
                "system_instruction": (
                    "You are an architecture decision context research specialist. Your job is "
                    "to analyze provided inputs — source code, design documents, Slack/Teams "
                    "conversations, meeting notes, RFC threads, requirements documents, and "
                    "existing codebase patterns — to extract the decision context. Identify: "
                    "what decision needs to be made and why now, the quality attributes at "
                    "stake (performance, security, scalability, maintainability, operability, "
                    "cost), business constraints (timeline, budget, team skills, regulatory "
                    "requirements), technical constraints (existing infrastructure, integration "
                    "requirements, data formats), and stakeholder concerns (what different "
                    "roles care about — developers want simplicity, ops wants observability, "
                    "security wants compliance, business wants speed). Distinguish hard "
                    "constraints (non-negotiable) from soft constraints (preferences). Identify "
                    "any existing ADRs or documented decisions this new decision relates to."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "adr_author",
                "role": "ADR Author",
                "system_instruction": SYSTEM_INSTRUCTION,
                "model": "gemini-2.5-pro",
            },
        ]

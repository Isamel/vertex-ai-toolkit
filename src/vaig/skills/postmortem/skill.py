"""Postmortem Skill — Blameless incident postmortem generation following Google SRE practices."""

from __future__ import annotations

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.postmortem.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION


class PostmortemSkill(BaseSkill):
    """Postmortem skill for generating blameless incident postmortems.

    Supports multi-agent execution with specialized agents:
    - Timeline Builder: Reconstructs incident timeline from provided data
    - Impact Assessor: Quantifies user, SLO, financial, and reputation impact
    - Postmortem Author: Synthesizes into a complete blameless postmortem document
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="postmortem",
            display_name="Postmortem",
            description="Generate comprehensive, blameless incident postmortems following Google SRE best practices",
            version="1.0.0",
            tags=["postmortem", "sre", "incident", "blameless", "retrospective", "action-items"],
            supported_phases=[
                SkillPhase.ANALYZE,
                SkillPhase.PLAN,
                SkillPhase.REPORT,
            ],
            recommended_model="gemini-2.5-pro",
        )

    def get_system_instruction(self) -> str:
        return SYSTEM_INSTRUCTION

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        template = PHASE_PROMPTS.get(phase.value, PHASE_PROMPTS["analyze"])
        return template.format(context=context, user_input=user_input)

    def get_agents_config(self) -> list[dict]:
        return [
            {
                "name": "timeline_builder",
                "role": "Timeline Builder",
                "system_instruction": (
                    "You are an incident timeline reconstruction specialist. Your job is to "
                    "build precise chronological timelines from incident data — logs, alerts, "
                    "chat transcripts, and status page updates. Focus on: detection time, "
                    "response time, escalation points, mitigation actions, and resolution "
                    "confirmation. Calculate TTD, TTM, and TTR metrics."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "impact_assessor",
                "role": "Impact Assessor",
                "system_instruction": (
                    "You are an incident impact quantification specialist. Your job is to "
                    "assess the full blast radius of incidents: affected users (count and "
                    "percentage), duration of user-facing impact, SLO burn rate, financial "
                    "impact estimation, data integrity impact, and reputation impact. Use "
                    "available metrics to provide precise numbers, and clearly state when "
                    "estimates are used vs confirmed data."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "postmortem_author",
                "role": "Postmortem Author",
                "system_instruction": SYSTEM_INSTRUCTION,
                "model": "gemini-2.5-pro",
            },
        ]

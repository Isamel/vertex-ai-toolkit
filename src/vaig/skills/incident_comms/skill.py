"""Incident Communications Skill — status pages, stakeholder updates, and crisis comms."""

from __future__ import annotations

from typing import Any

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.incident_comms.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION


class IncidentCommsSkill(BaseSkill):
    """Incident Communications skill for generating coordinated incident updates.

    Supports multi-agent execution with specialized agents:
    - Status Writer: Generates status page updates calibrated to audience and lifecycle stage
    - Comms Coordinator: Produces full communication packages across all channels
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="incident-comms",
            display_name="Incident Communications",
            description=(
                "Generate coordinated incident communications — status page updates, "
                "executive briefs, customer emails, and regulatory notifications"
            ),
            version="1.0.0",
            tags=[
                "incident-response",
                "communication",
                "status-page",
                "stakeholder",
                "crisis",
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
                "name": "status_writer",
                "role": "Status Page Writer",
                "system_instruction": (
                    "You are a status page communications specialist. Your job is to generate "
                    "clear, audience-calibrated status page updates following the standard "
                    "incident lifecycle: Investigating, Identified, Monitoring, Resolved. "
                    "Tailor detail level and language to the target audience — technical teams "
                    "receive service names, error codes, and metrics; customers receive "
                    "plain-language impact descriptions and ETAs; executives receive business "
                    "impact summaries. Use blameless language — never reference individuals "
                    "or specific teams. Always include: timestamp (UTC), affected components, "
                    "current status, and next update time. Quantify impact when possible "
                    "(percentage of requests affected, number of users impacted, regions affected). "
                    "Never speculate about root cause until confirmed."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "comms_coordinator",
                "role": "Communications Coordinator",
                "system_instruction": SYSTEM_INSTRUCTION,
                "model": "gemini-2.5-pro",
            },
        ]

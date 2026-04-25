"""Error Triage Skill — Rapid error classification and prioritization."""

from __future__ import annotations

from typing import Any

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.error_triage.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION


class ErrorTriageSkill(BaseSkill):
    """Error Triage skill for rapid error classification and incident prioritization.

    Supports multi-agent execution with specialized agents:
    - Error Classifier: Classifies error type, determines blast radius, identifies affected services
    - Triage Coordinator: Produces priority assessment, response actions, escalation paths
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="error-triage",
            display_name="Error Triage",
            description="Rapid error classification and prioritization during incidents",
            version="1.0.0",
            tags=["errors", "sre", "triage", "incident", "priority", "classification"],
            supported_phases=[
                SkillPhase.ANALYZE,
                SkillPhase.REPORT,
            ],
            recommended_model="gemini-2.5-pro",
        )

    def get_system_instruction(self) -> str:
        return SYSTEM_INSTRUCTION

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        template = PHASE_PROMPTS.get(phase.value, PHASE_PROMPTS["analyze"])
        return template.format(context=context, user_input=user_input)

    def get_agents_config(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "name": "error_classifier",
                "role": "Error Classifier",
                "system_instruction": (
                    "You are an error classification specialist. Your job is to rapidly "
                    "categorize errors by type (infrastructure, application, data, network, "
                    "auth, config), determine blast radius across users and services, and "
                    "identify all affected components. Focus on: error signatures, stack trace "
                    "patterns, failure domain isolation, and impact scope estimation."
                ),
                "model": "",
            },
            {
                "name": "triage_coordinator",
                "role": "Triage Coordinator",
                "system_instruction": SYSTEM_INSTRUCTION,
                "model": "",
            },
        ]

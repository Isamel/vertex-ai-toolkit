"""Change Risk Skill — change risk assessment and deployment safety analysis."""

from __future__ import annotations

from typing import Any

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.change_risk.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION


class ChangeRiskSkill(BaseSkill):
    """Change Risk skill for deployment risk assessment and change management.

    Supports multi-agent execution with specialized agents:
    - Change Analyzer: Evaluates change scope, blast radius, and dependency impact
    - Risk Scorer: Assigns risk scores and generates deployment checklists
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="change-risk",
            display_name="Change Risk Assessment",
            description=(
                "Assess deployment risk by analyzing change scope, blast radius, "
                "reversibility, and generating pre-deployment checklists and CAB summaries"
            ),
            version="1.0.0",
            tags=[
                "change-management",
                "risk",
                "deployment",
                "rollback",
                "blast-radius",
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
                "name": "change_analyzer",
                "role": "Change Analyzer",
                "system_instruction": (
                    "You are a change impact analysis specialist. Your job is to evaluate "
                    "the scope of proposed changes — files modified, services affected, "
                    "dependency graph impact, database migrations, API surface changes, "
                    "configuration modifications, and infrastructure changes. Trace impact "
                    "through the service dependency graph to identify all directly and "
                    "transitively affected systems. Classify changes by type (feature, "
                    "bug fix, refactor, dependency update, config change, infrastructure, "
                    "data migration). Assess whether changes are behind feature flags and "
                    "whether kill switches are properly configured. Identify blast radius "
                    "as Isolated, Service-Local, Cross-Service, Platform-Wide, or External. "
                    "Evaluate reversibility — can the change be instantly rolled back via "
                    "feature flag toggle, quickly rolled back via redeployment, or does it "
                    "require complex rollback with data reconciliation?"
                ),
                "model": "",
            },
            {
                "name": "risk_scorer",
                "role": "Risk Scorer & Deployment Planner",
                "system_instruction": SYSTEM_INSTRUCTION,
                "model": "",
            },
        ]

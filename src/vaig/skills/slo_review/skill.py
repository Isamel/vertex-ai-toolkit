"""SLO Review Skill — SLO/SLI analysis and error budget review."""

from __future__ import annotations

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.slo_review.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION


class SloReviewSkill(BaseSkill):
    """SLO Review skill for analyzing SLO/SLI definitions and error budgets.

    Supports multi-agent execution with specialized agents:
    - SLI Analyzer: Evaluates SLI definitions for correctness, coverage, and methodology
    - Budget Strategist: Analyzes error budget consumption, burn rate, and recommends adjustments
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="slo-review",
            display_name="SLO Review",
            description="Analyze SLO/SLI definitions, error budget consumption, and reliability targets based on Google SRE principles",
            version="1.0.0",
            tags=["slo", "sli", "sre", "reliability", "error-budget", "observability"],
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

    def get_agents_config(self) -> list[dict]:
        return [
            {
                "name": "sli_analyzer",
                "role": "SLI Analyzer",
                "system_instruction": (
                    "You are an SLI specification and measurement specialist. Your job is to "
                    "evaluate SLI definitions for correctness, completeness, and alignment with "
                    "user-facing quality. Focus on: SLI specification vs implementation gaps, "
                    "measurement methodology (request-based vs window-based), coverage of "
                    "critical user journeys, and whether SLIs capture actual user experience "
                    "rather than proxy infrastructure metrics."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "budget_strategist",
                "role": "Budget Strategist",
                "system_instruction": (
                    "You are an error budget analysis and strategy specialist. Your job is to "
                    "analyze error budget consumption patterns, burn rate trends, and projected "
                    "budget exhaustion timelines. Focus on: multi-window burn rate alerting "
                    "correctness, error budget policy enforcement, SLO-based release gating, "
                    "and recommending specific SLO target adjustments backed by consumption data "
                    "and Google SRE best practices."
                ),
                "model": "gemini-2.5-flash",
            },
        ]

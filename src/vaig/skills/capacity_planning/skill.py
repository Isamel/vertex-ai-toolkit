"""Capacity Planning Skill — resource capacity forecasting and scaling."""

from __future__ import annotations

from typing import Any

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.capacity_planning.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION


class CapacityPlanningSkill(BaseSkill):
    """Capacity Planning skill for resource forecasting and scaling recommendations.

    Supports multi-agent execution with specialized agents:
    - Trend Analyzer: Analyzes historical usage data, growth trends, and traffic projections
    - Capacity Modeler: Builds capacity models, forecasts needs, and recommends scaling strategies
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="capacity-planning",
            display_name="Capacity Planning",
            description=(
                "Forecast resource capacity needs, identify scaling bottlenecks, "
                "and recommend infrastructure scaling strategies"
            ),
            version="1.0.0",
            tags=["capacity", "scaling", "forecasting", "infrastructure", "sre", "performance"],
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

    def get_agents_config(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "name": "trend_analyzer",
                "role": "Trend Analyzer",
                "system_instruction": (
                    "You are a resource utilization trend analysis specialist. Your job is to "
                    "analyze historical usage data, identify growth trends, seasonal patterns, "
                    "and traffic projections across compute, memory, storage, and network "
                    "resources. Focus on: time-series decomposition (trend, seasonality, "
                    "residual), burst vs sustained load differentiation, step-change detection "
                    "from deploys or migrations, correlation between business metrics and "
                    "infrastructure utilization, and identifying current utilization hotspots "
                    "that are approaching saturation thresholds."
                ),
                "model": "",
            },
            {
                "name": "capacity_modeler",
                "role": "Capacity Modeler",
                "system_instruction": SYSTEM_INSTRUCTION,
                "model": "",
            },
        ]

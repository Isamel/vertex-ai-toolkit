"""Cost Analysis Skill — cloud cost analysis and FinOps optimization."""

from __future__ import annotations

from typing import Any

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.cost_analysis.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION


class CostAnalysisSkill(BaseSkill):
    """Cost Analysis skill for cloud cost optimization and FinOps recommendations.

    Supports multi-agent execution with specialized agents:
    - Resource Scanner: Scans billing data, resource configs, and usage patterns
    - Cost Optimizer: Synthesizes findings into actionable optimization plan
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="cost-analysis",
            display_name="Cost Analysis",
            description="Analyze cloud infrastructure costs, identify optimization opportunities, and provide FinOps recommendations",
            version="1.0.0",
            tags=["cost", "finops", "cloud", "optimization", "billing", "savings"],
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

    def get_agents_config(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "resource_scanner",
                "role": "Resource Scanner",
                "system_instruction": (
                    "You are a cloud resource scanning specialist. Your job is to scan "
                    "cloud resource configurations, billing data, and usage patterns to "
                    "identify waste, idle resources, and right-sizing opportunities. Focus on: "
                    "underutilized compute instances, unattached storage volumes, orphaned "
                    "snapshots, idle load balancers, unused static IPs, over-provisioned "
                    "databases, and resources with zero or near-zero traffic. Quantify every "
                    "finding with estimated monthly cost and potential savings."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "cost_optimizer",
                "role": "Cost Optimizer",
                "system_instruction": SYSTEM_INSTRUCTION,
                "model": "gemini-2.5-pro",
            },
        ]

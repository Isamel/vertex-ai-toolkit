"""IaC Review Skill — Infrastructure-as-Code review."""

from __future__ import annotations

from typing import Any

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.iac_review.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION


class IacReviewSkill(BaseSkill):
    """IaC Review skill for Infrastructure-as-Code security and quality analysis.

    Supports multi-agent execution with specialized agents:
    - Plan Analyzer: Analyzes IaC plans for risky changes and destructive operations
    - Drift Detector: Identifies configuration drift, deprecated resources, compliance gaps
    - IaC Reviewer: Senior reviewer synthesizing findings into prioritized report
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="iac-review",
            display_name="IaC Review",
            description=(
                "Review Infrastructure-as-Code for security misconfigurations, "
                "cost optimization, reliability gaps, and IaC best practices"
            ),
            version="1.0.0",
            tags=["iac", "terraform", "infrastructure", "security", "cloud", "devops", "compliance"],
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
                "name": "plan_analyzer",
                "role": "Plan Analyzer",
                "system_instruction": (
                    "You are an IaC plan analysis specialist. Your job is to analyze "
                    "infrastructure plan outputs (terraform plan, pulumi preview, "
                    "CloudFormation changesets) for risky changes, destructive operations, "
                    "and configuration drift. Focus on: resources being destroyed or replaced, "
                    "security group modifications, IAM policy changes, encryption setting "
                    "alterations, state drift between declared and actual infrastructure, "
                    "and any change that could cause downtime or data loss."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "drift_detector",
                "role": "Drift Detector",
                "system_instruction": (
                    "You are an infrastructure configuration drift and compliance specialist. "
                    "Your job is to identify configuration drift from best practices, "
                    "deprecated resource types, missing mandatory tags, non-compliant "
                    "settings, and policy violations. Focus on: resources using deprecated "
                    "instance types or API versions, missing encryption configurations, "
                    "non-compliant network settings, absent monitoring and logging, "
                    "tagging policy violations, and deviations from CIS Benchmarks, "
                    "SOC 2, HIPAA, and PCI-DSS requirements."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "iac_reviewer",
                "role": "IaC Reviewer",
                "system_instruction": SYSTEM_INSTRUCTION,
                "model": "gemini-2.5-pro",
            },
        ]

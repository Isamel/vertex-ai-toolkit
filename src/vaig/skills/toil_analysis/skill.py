"""Toil Analysis Skill — operational toil detection and automation planning."""

from __future__ import annotations

from typing import Any

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.toil_analysis.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION


class ToilAnalysisSkill(BaseSkill):
    """Toil Analysis skill for detecting operational toil and planning automation.

    Supports multi-agent execution with specialized agents:
    - Toil Detector: Classifies work as toil vs engineering, quantifies time and frequency
    - Automation Planner: Prioritizes automation by ROI and proposes implementation approaches
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="toil-analysis",
            display_name="Toil Analysis",
            description=(
                "Analyze operational work to identify toil, quantify its cost, "
                "and prioritize automation opportunities by ROI"
            ),
            version="1.0.0",
            tags=[
                "sre",
                "toil",
                "automation",
                "operational-efficiency",
                "on-call",
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
                "name": "toil_detector",
                "role": "Toil Detector",
                "system_instruction": (
                    "You are an operational toil detection specialist. Your job is to analyze "
                    "on-call tickets, runbooks, alert histories, and operational procedures to "
                    "classify each task as toil or legitimate engineering work. Apply the Google "
                    "SRE toil criteria: manual, repetitive, automatable, tactical/reactive, no "
                    "enduring value, scales linearly with service growth. For each identified toil "
                    "task, quantify: time spent per occurrence, frequency (daily/weekly/monthly), "
                    "number of engineers performing it, total quarterly investment, and scaling "
                    "characteristics. Group toil by category — deployment operations, alert "
                    "remediation, access provisioning, data operations, scaling operations, "
                    "certificate management, and reporting. Calculate the toil budget percentage "
                    "per team and flag any exceeding the 50% threshold."
                ),
                "model": "",
            },
            {
                "name": "automation_planner",
                "role": "Automation Planner",
                "system_instruction": SYSTEM_INSTRUCTION,
                "model": "",
            },
        ]

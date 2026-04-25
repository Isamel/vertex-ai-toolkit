"""RCA Skill — Root Cause Analysis for incident investigation."""

from __future__ import annotations

from typing import Any

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.rca.prompts import PHASE_PROMPTS, RCA_GATHERER_PROMPT, SYSTEM_INSTRUCTION


class RCASkill(BaseSkill):
    """Root Cause Analysis skill for investigating production incidents.

    Supports multi-agent execution with specialized agents:
    - Log Analyzer: Parses and correlates log data
    - Metric Correlator: Identifies metric anomalies and correlations
    - Impact Assessor: Quantifies blast radius and business impact
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="rca",
            display_name="Root Cause Analysis",
            description="Investigate production incidents using 5 Whys + Fishbone methodology",
            version="1.0.0",
            tags=["incident", "sre", "debugging", "post-mortem", "logs", "metrics", "live"],
            supported_phases=[
                SkillPhase.ANALYZE,
                SkillPhase.PLAN,
                SkillPhase.EXECUTE,
                SkillPhase.REPORT,
            ],
            recommended_model="gemini-2.5-pro",
            requires_live_tools=True,
        )

    def get_system_instruction(self) -> str:
        return SYSTEM_INSTRUCTION

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        template = PHASE_PROMPTS.get(phase.value, PHASE_PROMPTS["analyze"])
        return template.format(context=context, user_input=user_input)

    def get_agents_config(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "name": "rca_gatherer",
                "role": "RCA Data Gatherer",
                "requires_tools": True,
                "system_instruction": RCA_GATHERER_PROMPT,
                "model": "",
                "temperature": 0.0,
                "max_iterations": 10,
            },
            {
                "name": "log_analyzer",
                "role": "Log Analyzer",
                "system_instruction": (
                    "You are a log analysis specialist. Your job is to parse log files, "
                    "identify error patterns, build timelines from timestamps, and correlate "
                    "log entries across multiple services. Focus on: error messages, stack traces, "
                    "request IDs, correlation IDs, and timing anomalies."
                ),
                "model": "",
            },
            {
                "name": "metric_correlator",
                "role": "Metric Correlator",
                "system_instruction": (
                    "You are a metrics and monitoring specialist. Your job is to analyze "
                    "metrics data (CPU, memory, latency, error rates, throughput) and identify "
                    "anomalies, correlations, and trends. Use statistical reasoning to determine "
                    "which metric changes are significant vs noise."
                ),
                "model": "",
            },
            {
                "name": "rca_lead",
                "role": "RCA Lead Investigator",
                "system_instruction": SYSTEM_INSTRUCTION,
                "model": "",
            },
        ]

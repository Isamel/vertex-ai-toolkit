"""Alert Tuning Skill — alert and monitoring review for noise reduction and coverage."""

from __future__ import annotations

from typing import Any

from vaig.skills.alert_tuning.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION
from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase


class AlertTuningSkill(BaseSkill):
    """Alert Tuning skill for alert quality review, noise reduction, and coverage analysis.

    Supports multi-agent execution with specialized agents:
    - Noise Analyzer: Identifies noisy, dead, stuck, and duplicate alerts
    - Coverage Assessor: Maps alerting against system components for gaps
    - Observability Lead: Produces prioritized tuning plan with USE/RED dashboards
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="alert-tuning",
            display_name="Alert & Monitoring Review",
            description=(
                "Review alerting rules for noise reduction, coverage gaps, "
                "and monitoring quality using USE/RED methodology"
            ),
            version="1.0.0",
            tags=[
                "observability",
                "alerting",
                "monitoring",
                "noise-reduction",
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
                "name": "noise_analyzer",
                "role": "Noise Analyzer",
                "system_instruction": (
                    "You are an alert noise analysis specialist. Your job is to identify "
                    "noisy alerts with high fire rates and low action rates, duplicate alerts "
                    "that always fire together, dead alerts that have not triggered in 90+ "
                    "days (monitoring decommissioned components), stuck alerts that have been "
                    "firing continuously (wrong threshold or ignored issue), and orphaned "
                    "alerts with no owner or runbook. Calculate alert fatigue metrics: "
                    "signal-to-noise ratio per alert, action rate (percentage of firings "
                    "resulting in human action), auto-resolve rate, correlation patterns "
                    "between alerts, and overall on-call burden (pages per shift, off-hours "
                    "interruptions). Classify each alert health as HEALTHY, NOISY, DEAD, "
                    "STUCK, ORPHANED, or DUPLICATE."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "coverage_assessor",
                "role": "Coverage Assessor",
                "system_instruction": (
                    "You are a monitoring coverage analysis specialist. Your job is to map "
                    "alerting rules against the system architecture to find coverage gaps. "
                    "Identify unmonitored services with no alerts at all, services missing "
                    "golden signal coverage (latency, traffic, errors, saturation), absent "
                    "dependency health checks for external services and databases, data "
                    "pipeline monitoring gaps (no freshness or completeness alerts), "
                    "infrastructure blind spots (missing disk, memory, CPU, connection pool "
                    "alerts), and services with SLOs defined but no error budget burn rate "
                    "alerts. Evaluate whether existing alerts cover both the USE method "
                    "(Utilization, Saturation, Errors for infrastructure) and RED method "
                    "(Request rate, Error rate, Duration for services). Identify services "
                    "lacking proper observability dashboards."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "observability_lead",
                "role": "Observability Lead",
                "system_instruction": SYSTEM_INSTRUCTION,
                "model": "gemini-2.5-pro",
            },
        ]

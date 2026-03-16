"""Anomaly Detection Skill implementation."""

from __future__ import annotations

from vaig.skills.anomaly.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION
from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase


class AnomalySkill(BaseSkill):
    """Anomaly Detection skill for identifying unusual patterns in data."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="anomaly",
            display_name="Anomaly Detection",
            description="Detect anomalies in logs, metrics, data, and system behavior",
            version="1.0.0",
            tags=["anomaly", "detection", "metrics", "logs", "monitoring", "observability"],
            supported_phases=[SkillPhase.ANALYZE, SkillPhase.EXECUTE, SkillPhase.REPORT],
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
                "name": "pattern_analyzer",
                "role": "Pattern Analyzer",
                "system_instruction": (
                    "You specialize in pattern recognition in time-series data and logs. "
                    "Identify recurring patterns, seasonality, trends, and deviations. "
                    "Use statistical reasoning to separate signal from noise. "
                    "NEVER invent or fabricate data points — only report patterns "
                    "that are directly observable in the provided data. "
                    "If insufficient data is provided, state that explicitly."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "anomaly_detector",
                "role": "Anomaly Detector",
                "system_instruction": SYSTEM_INSTRUCTION,
                "model": "gemini-2.5-pro",
            },
        ]

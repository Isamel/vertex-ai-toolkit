"""Log Analysis Skill — SRE log diagnostic analysis."""

from __future__ import annotations

from typing import Any

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.log_analysis.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION


class LogAnalysisSkill(BaseSkill):
    """Log Analysis skill for SRE diagnostic investigation of production logs.

    Supports multi-agent execution with specialized agents:
    - Pattern Detector: Identifies error patterns, frequency, timing correlations
    - Context Analyzer: Correlates log entries with system context
    - Diagnostic Lead: Synthesizes findings into actionable diagnostic report
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="log-analysis",
            display_name="Log Analysis",
            description="Analyze production logs to identify error patterns, timing anomalies, and root cause hypotheses",
            version="1.0.0",
            tags=["logs", "sre", "diagnostics", "patterns", "incident", "observability"],
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

    def get_agents_config(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "pattern_detector",
                "role": "Pattern Detector",
                "system_instruction": (
                    "You are a log pattern recognition specialist. Your job is to identify "
                    "recurring error patterns, frequency distributions, timing correlations, "
                    "and anomalous log sequences. Focus on: error rates, burst patterns, "
                    "periodic failures, new error signatures, and log volume anomalies."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "context_analyzer",
                "role": "Context Analyzer",
                "system_instruction": (
                    "You are a system context correlation specialist. Your job is to correlate "
                    "log entries with system context — recent deployments, configuration changes, "
                    "traffic patterns, dependency health, and infrastructure events. Identify "
                    "causal relationships between system changes and observed log anomalies."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "diagnostic_lead",
                "role": "Diagnostic Lead",
                "system_instruction": SYSTEM_INSTRUCTION,
                "model": "gemini-2.5-pro",
            },
        ]

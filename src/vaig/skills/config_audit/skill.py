"""Config Audit Skill — infrastructure and application configuration auditing."""

from __future__ import annotations

from typing import Any

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.config_audit.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION


class ConfigAuditSkill(BaseSkill):
    """Config Audit skill for auditing infrastructure and application configurations.

    Supports multi-agent execution with specialized agents:
    - Security Scanner: Identifies security misconfigurations, exposed secrets, permissive IAM
    - Reliability Auditor: Identifies reliability risks, missing limits, absent health checks
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="config-audit",
            display_name="Config Audit",
            description="Audit infrastructure and application configs for security issues, misconfigurations, and reliability risks",
            version="1.0.0",
            tags=["config", "sre", "security", "audit", "compliance", "infrastructure", "reliability"],
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
                "name": "security_scanner",
                "role": "Security Scanner",
                "system_instruction": (
                    "You are a configuration security specialist. Your job is to identify "
                    "security misconfigurations in infrastructure and application configs. "
                    "Focus on: exposed secrets and credentials, overly permissive IAM/RBAC "
                    "policies, missing encryption at rest and in transit, insecure defaults, "
                    "missing network policies, container privilege escalation vectors, and "
                    "non-compliant settings per CIS benchmarks and NIST 800-53 controls."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "reliability_auditor",
                "role": "Reliability Auditor",
                "system_instruction": (
                    "You are a reliability configuration specialist. Your job is to identify "
                    "reliability risks in infrastructure and application configs. Focus on: "
                    "missing resource limits (CPU, memory, storage), absent health checks and "
                    "readiness probes, single points of failure, missing retry and circuit-breaker "
                    "configurations, inadequate logging and monitoring setup, missing autoscaling "
                    "policies, and insufficient redundancy or fault tolerance settings."
                ),
                "model": "gemini-2.5-flash",
            },
        ]

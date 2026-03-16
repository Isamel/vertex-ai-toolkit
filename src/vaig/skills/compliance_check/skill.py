"""Compliance Check Skill — regulatory and policy compliance auditing."""

from __future__ import annotations

from typing import Any

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.compliance_check.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION


class ComplianceCheckSkill(BaseSkill):
    """Compliance Check skill for auditing systems against regulatory and policy requirements.

    Supports multi-agent execution with specialized agents:
    - Regulation Mapper: Maps system components to applicable compliance frameworks and controls
    - Gap Auditor: Assesses each control and identifies non-compliant or partial items
    - Compliance Lead: Synthesizes findings into a compliance report with remediation roadmap
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="compliance-check",
            display_name="Compliance Check",
            description="Audit systems for regulatory compliance (SOC 2, ISO 27001, HIPAA, PCI-DSS, GDPR) and generate remediation plans",
            version="1.0.0",
            tags=["compliance", "audit", "security", "regulatory", "soc2", "gdpr", "hipaa", "governance"],
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
                "name": "regulation_mapper",
                "role": "Regulation Mapper",
                "system_instruction": (
                    "You are a regulatory framework specialist. Your job is to identify "
                    "which compliance frameworks apply to the system under review, map "
                    "system components to specific controls, and assess the applicability "
                    "of each control. You are deeply familiar with SOC 2, ISO 27001, HIPAA, "
                    "PCI-DSS, GDPR, FedRAMP, NIST 800-53, and CIS Benchmarks."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "gap_auditor",
                "role": "Gap Auditor",
                "system_instruction": (
                    "You are a compliance gap analysis specialist. Your job is to assess "
                    "each mapped control against the system's current state and determine "
                    "compliance status: COMPLIANT, PARTIAL, NON-COMPLIANT, or NOT APPLICABLE. "
                    "You identify missing evidence, misconfigured controls, and security gaps. "
                    "You rate each finding by severity and provide specific remediation steps."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "compliance_lead",
                "role": "Compliance Lead",
                "system_instruction": SYSTEM_INSTRUCTION,
                "model": "gemini-2.5-pro",
            },
        ]

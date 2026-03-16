"""Threat Model Skill — STRIDE-based threat modeling and attack surface analysis."""

from __future__ import annotations

from typing import Any

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.threat_model.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION


class ThreatModelSkill(BaseSkill):
    """Threat Model skill for STRIDE-based threat modeling and attack surface analysis.

    Supports multi-agent execution with specialized agents:
    - Attack Surface Mapper: Identifies entry points, trust boundaries, data flows
    - Threat Enumerator: Applies STRIDE categories with likelihood/impact scoring
    - Threat Model Lead: Synthesizes ranked threat model with countermeasures
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="threat-model",
            display_name="Threat Modeling",
            description=(
                "Conduct STRIDE-based threat modeling to identify attack surfaces, "
                "enumerate threats, and recommend prioritized countermeasures"
            ),
            version="1.0.0",
            tags=[
                "security",
                "threat-modeling",
                "stride",
                "attack-surface",
                "risk",
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

    def get_agents_config(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "attack_surface_mapper",
                "role": "Attack Surface Mapper",
                "system_instruction": (
                    "You are an attack surface analysis specialist. Your job is to identify "
                    "every entry point into the system — HTTP endpoints, gRPC services, "
                    "WebSocket connections, message queue consumers, file upload handlers, "
                    "OAuth callbacks, webhook receivers, admin interfaces, debug endpoints, "
                    "and health check endpoints. Map all trust boundaries — network segments, "
                    "process boundaries, privilege levels, tenant isolation boundaries, and "
                    "third-party integration boundaries. Trace data flows for sensitive data "
                    "(PII, credentials, tokens, financial data) through the system, noting "
                    "encryption status at each stage. Document authentication and authorization "
                    "mechanisms per entry point. Identify external dependencies and their "
                    "trust level. Produce a comprehensive attack surface inventory."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "threat_enumerator",
                "role": "Threat Enumerator",
                "system_instruction": (
                    "You are a STRIDE threat analysis specialist. Your job is to apply the "
                    "STRIDE framework (Spoofing, Tampering, Repudiation, Information Disclosure, "
                    "Denial of Service, Elevation of Privilege) systematically to each component "
                    "and data flow identified in the attack surface analysis. For each threat, "
                    "produce a concrete attack scenario (not generic risks), assess likelihood "
                    "(1–5) based on attacker capability and complexity, assess impact (1–5) "
                    "based on confidentiality, integrity, availability, and business impact, "
                    "and compute a risk score. Document existing mitigations and residual risk. "
                    "Identify attack chains where multiple medium-risk threats combine into "
                    "high-impact sequences. Assign unique threat IDs (TM-XXXX) for tracking."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "threat_model_lead",
                "role": "Threat Model Lead",
                "system_instruction": SYSTEM_INSTRUCTION,
                "model": "gemini-2.5-pro",
            },
        ]

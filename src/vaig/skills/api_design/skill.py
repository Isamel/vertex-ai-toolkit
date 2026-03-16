"""API Design Skill — API design review and best practices analysis."""

from __future__ import annotations

from typing import Any

from vaig.skills.api_design.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION
from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase


class APIDesignSkill(BaseSkill):
    """API Design skill for reviewing and improving API contracts and implementation.

    Supports multi-agent execution with specialized agents:
    - Contract Analyzer: Reviews API definitions for correctness, consistency, and completeness
    - Security Reviewer: Audits authentication, authorization, input validation, and data exposure
    - API Design Lead: Synthesizes findings into a comprehensive design review report
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="api-design",
            display_name="API Design Review",
            description="Review API design for REST/GraphQL/gRPC best practices, consistency, security, and developer experience",
            version="1.0.0",
            tags=["api", "rest", "graphql", "grpc", "design", "openapi", "developer-experience"],
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
                "name": "contract_analyzer",
                "role": "Contract Analyzer",
                "system_instruction": (
                    "You are an API contract analysis specialist. Your job is to review "
                    "API definitions (OpenAPI, Protobuf, GraphQL schemas) for correctness, "
                    "consistency, and completeness. You evaluate resource modeling, naming "
                    "conventions, HTTP semantics, pagination patterns, error formats, and "
                    "versioning strategy. You are deeply familiar with Google API Design "
                    "Guide, Microsoft REST API Guidelines, and Zalando RESTful API Guidelines."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "security_reviewer",
                "role": "Security Reviewer",
                "system_instruction": (
                    "You are an API security specialist. Your job is to audit APIs for "
                    "authentication and authorization gaps, input validation weaknesses, "
                    "data exposure risks, rate limiting adequacy, and OWASP API Security "
                    "Top 10 vulnerabilities. You evaluate OAuth 2.0 flows, JWT handling, "
                    "scope definitions, CORS policies, and sensitive data in responses."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "api_design_lead",
                "role": "API Design Lead",
                "system_instruction": SYSTEM_INSTRUCTION,
                "model": "gemini-2.5-pro",
            },
        ]

"""Code Review Skill — automated code review with architecture/pattern awareness."""

from __future__ import annotations

from typing import Any

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.code_review.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION


class CodeReviewSkill(BaseSkill):
    """Code Review skill for automated architecture and quality analysis.

    Supports multi-agent execution with specialized agents:
    - Code Reviewer: Architecture & patterns specialist (SOLID, Clean Architecture, DDD)
    - Security Auditor: Security vulnerability scanner (OWASP Top 10)
    - Review Lead: Synthesizes findings into prioritized review report
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="code-review",
            display_name="Code Review",
            description=(
                "Automated code review analyzing architecture patterns, security "
                "vulnerabilities, performance issues, and best practices"
            ),
            version="1.0.0",
            tags=[
                "code-review",
                "quality",
                "security",
                "architecture",
                "patterns",
                "best-practices",
            ],
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
                "name": "code_reviewer",
                "role": "Architecture & Patterns Specialist",
                "system_instruction": (
                    "You are a code architecture specialist focused on structural quality. "
                    "Your job is to review code for SOLID principle violations, design pattern "
                    "misapplication, coupling/cohesion issues, layer boundary violations, and "
                    "complexity problems. Evaluate naming conventions, code organization, "
                    "duplication, error handling strategy, and documentation coverage. "
                    "Flag both over-engineering and under-abstraction with equal rigor."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "security_auditor",
                "role": "Security Vulnerability Scanner",
                "system_instruction": (
                    "You are a security-focused code auditor. Your job is to scan code for "
                    "injection vulnerabilities (SQL, XSS, command, template), authentication "
                    "and authorization flaws, hardcoded secrets or API keys, insecure "
                    "deserialization, path traversal, missing input validation, insecure "
                    "dependency usage, and OWASP Top 10 violations. Assess each finding "
                    "with an attack vector description and concrete remediation steps."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "review_lead",
                "role": "Senior Review Lead",
                "system_instruction": SYSTEM_INSTRUCTION,
                "model": "gemini-2.5-pro",
            },
        ]

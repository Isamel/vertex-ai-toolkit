"""Dependency Audit Skill — supply-chain security and dependency health analysis."""

from __future__ import annotations

from typing import Any

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.dependency_audit.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION


class DependencyAuditSkill(BaseSkill):
    """Dependency Audit skill for supply-chain security and dependency health analysis.

    Supports multi-agent execution with specialized agents:
    - Vulnerability Scanner: CVE identification, CVSS scoring, EOL detection
    - License Analyst: License compliance, typosquatting, supply-chain risk
    - Dependency Lead: Synthesizes into prioritized remediation plan
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="dependency-audit",
            display_name="Dependency Audit",
            description=(
                "Audit software dependencies for known vulnerabilities, license "
                "compliance issues, supply-chain risks, and dependency hygiene"
            ),
            version="1.0.0",
            tags=[
                "security",
                "supply-chain",
                "dependencies",
                "cve",
                "vulnerability",
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
                "name": "vulnerability_scanner",
                "role": "Vulnerability Scanner",
                "system_instruction": (
                    "You are a dependency vulnerability scanning specialist. Your job is to "
                    "parse dependency manifests (package.json, requirements.txt, go.mod, "
                    "pom.xml, Gemfile, Cargo.toml, pyproject.toml) and their lock files to "
                    "build a complete dependency inventory. For each dependency, identify known "
                    "CVEs from NVD, GitHub Advisory Database, and OSV. Assess CVSS severity "
                    "scores, determine affected version ranges, and identify fixed versions. "
                    "Flag end-of-life runtimes and frameworks. Evaluate whether vulnerable "
                    "dependencies are actually reachable in the application code path. "
                    "Distinguish between direct and transitive dependency vulnerabilities."
                ),
                "model": "",
            },
            {
                "name": "license_analyst",
                "role": "License & Supply-Chain Analyst",
                "system_instruction": (
                    "You are a software license compliance and supply-chain risk specialist. "
                    "Your job is to evaluate license compatibility across the dependency tree — "
                    "detect GPL contamination in proprietary projects, AGPL exposure in SaaS "
                    "products, and packages with no declared license or custom terms requiring "
                    "legal review. Identify typosquatting risks by comparing package names "
                    "against popular packages. Detect phantom dependencies (used in code but "
                    "not in manifests) and unused dependencies (in manifests but never imported). "
                    "Assess maintainer health: bus factor, last commit date, publish frequency, "
                    "open issue trends, and known security incidents. Flag packages with "
                    "install scripts that execute arbitrary code."
                ),
                "model": "",
            },
            {
                "name": "dependency_lead",
                "role": "Dependency Audit Lead",
                "system_instruction": SYSTEM_INSTRUCTION,
                "model": "",
            },
        ]

"""Pipeline Review Skill — CI/CD pipeline security, efficiency, and hygiene analysis."""

from __future__ import annotations

from typing import Any

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.pipeline_review.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION


class PipelineReviewSkill(BaseSkill):
    """Pipeline Review skill for CI/CD security, efficiency, and hygiene analysis.

    Supports multi-agent execution with specialized agents:
    - Security Auditor: Secrets, tokens, third-party actions, artifact integrity
    - Efficiency Analyzer: Build times, caching, parallelization, redundancy
    - Pipeline Lead: Synthesizes into deployment safety and hygiene report
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="pipeline-review",
            display_name="Pipeline Review",
            description=(
                "Review CI/CD pipeline configurations for security risks, build "
                "efficiency, deployment safety, and pipeline-as-code hygiene"
            ),
            version="1.0.0",
            tags=[
                "cicd",
                "pipeline",
                "github-actions",
                "gitlab-ci",
                "devops",
                "deployment",
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
                "name": "security_auditor",
                "role": "Pipeline Security Auditor",
                "system_instruction": (
                    "You are a CI/CD security specialist. Your job is to scan pipeline "
                    "configurations for secrets exposure risks — environment variables "
                    "containing credentials, secrets passed as command-line arguments visible "
                    "in process listings, unmasked secrets in logs, secrets accessible to "
                    "untrusted code in PR builds from forks. Evaluate token permissions for "
                    "least-privilege violations — overly permissive GITHUB_TOKEN scopes, "
                    "long-lived PATs instead of OIDC, tokens shared across environments. "
                    "Check for unsigned container images, missing SBOM generation, no SLSA "
                    "provenance, and artifacts stored in insecure locations. Vet third-party "
                    "actions for unpinned versions (@main instead of SHA), unverified "
                    "publishers, and excessive permissions."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "efficiency_analyzer",
                "role": "Build Efficiency Analyzer",
                "system_instruction": (
                    "You are a CI/CD build optimization specialist. Your job is to identify "
                    "slow builds caused by missing dependency caching (npm, pip, Go, Docker "
                    "layer cache), unnecessary sequential steps that could run in parallel, "
                    "redundant checkout and setup steps across jobs, and tests running "
                    "sequentially when they could be parallelized or sharded. Evaluate flaky "
                    "test handling — automatic retry policies, quarantine mechanisms, test "
                    "impact analysis. Detect redundant workflows triggered on events they "
                    "don't need, full test suites running on documentation-only changes, and "
                    "duplicate CI checks on the same code. Assess runner resource utilization — "
                    "oversized runners for simple tasks, undersized runners causing OOM, and "
                    "spot instance opportunities."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "pipeline_lead",
                "role": "Pipeline Review Lead",
                "system_instruction": SYSTEM_INSTRUCTION,
                "model": "gemini-2.5-pro",
            },
        ]

"""Discovery Skill — autonomous cluster scanning and health discovery."""

from __future__ import annotations

from typing import Any

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.discovery.prompts import (
    CLUSTER_REPORTER_PROMPT,
    DEEP_INVESTIGATOR_PROMPT,
    INVENTORY_SCANNER_PROMPT,
    PHASE_PROMPTS,
    SYSTEM_INSTRUCTION,
    TRIAGE_CLASSIFIER_PROMPT,
)


class DiscoverySkill(BaseSkill):
    """Autonomous cluster scanning skill for discovering workload health.

    Unlike ``vaig live`` which takes a specific question, this skill
    auto-generates its investigation query based on namespace/flags and
    scans the cluster to discover what's wrong.

    Pipeline (4 sequential agents):
    1. **Inventory Scanner** (tools): enumerates namespaces, deployments,
       statefulsets, daemonsets, services.
    2. **Triage Classifier** (no tools): classifies workloads as
       🟢 Healthy / 🟡 Degraded / 🔴 Failing.
    3. **Deep Investigator** (tools): checks pods, logs, events, metrics
       for degraded/failing workloads.
    4. **Cluster Reporter** (no tools): aggregates findings into a
       structured markdown report.
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="discovery",
            display_name="Cluster Discovery",
            description="Autonomous cluster scanning — discovers workload health without a specific question",
            version="1.0.0",
            tags=["discovery", "scanning", "health", "kubernetes", "sre", "live"],
            supported_phases=[
                SkillPhase.ANALYZE,
                SkillPhase.EXECUTE,
                SkillPhase.REPORT,
            ],
            recommended_model="gemini-2.5-flash",
            requires_live_tools=True,
        )

    def get_system_instruction(self) -> str:
        return SYSTEM_INSTRUCTION

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        template = PHASE_PROMPTS.get(phase.value, PHASE_PROMPTS["analyze"])
        return template.format(context=context, user_input=user_input)

    def get_agents_config(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "name": "inventory_scanner",
                "role": "Inventory Scanner",
                "requires_tools": True,
                "system_instruction": INVENTORY_SCANNER_PROMPT,
                "model": "gemini-2.5-flash",
                "temperature": 0.0,
                "max_iterations": 15,
            },
            {
                "name": "triage_classifier",
                "role": "Triage Classifier",
                "requires_tools": False,
                "system_instruction": TRIAGE_CLASSIFIER_PROMPT,
                "model": "gemini-2.5-flash",
                "temperature": 0.0,
            },
            {
                "name": "deep_investigator",
                "role": "Deep Investigator",
                "requires_tools": True,
                "system_instruction": DEEP_INVESTIGATOR_PROMPT,
                "model": "gemini-2.5-flash",
                "temperature": 0.0,
                "max_iterations": 20,
            },
            {
                "name": "cluster_reporter",
                "role": "Cluster Reporter",
                "requires_tools": False,
                "system_instruction": CLUSTER_REPORTER_PROMPT,
                "model": "gemini-2.5-flash",
                "temperature": 0.2,
            },
        ]

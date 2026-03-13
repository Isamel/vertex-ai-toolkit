"""Service Health Skill — live Kubernetes service health assessment.

A 3-agent sequential pipeline that demonstrates the ToolAwareAgent +
Orchestrator integration.  The first agent uses live tools to collect
cluster health data; the second analyzes patterns; the third produces
a structured markdown report.
"""

from __future__ import annotations

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.service_health.prompts import (
    HEALTH_ANALYZER_PROMPT,
    HEALTH_GATHERER_PROMPT,
    HEALTH_REPORTER_PROMPT,
    PHASE_PROMPTS,
    SYSTEM_INSTRUCTION,
)


class ServiceHealthSkill(BaseSkill):
    """Service health assessment skill using live Kubernetes tools.

    Implements a 3-agent sequential pipeline:

    1. **health_gatherer** (``requires_tools=True``):
       Uses live kubectl tools to collect pod status, resource usage,
       logs, events, and deployment state.

    2. **health_analyzer** (``requires_tools=False``):
       Text-only agent that receives gathered data and performs SRE-style
       pattern analysis — degraded services, resource pressure, error
       rate spikes, cross-service correlations.

    3. **health_reporter** (``requires_tools=False``):
       Text-only agent that synthesizes findings into a structured
       markdown report with severity classification, root-cause
       hypotheses, and actionable remediation commands.

    The pipeline strategy is **sequential**: each agent's output feeds
    as context into the next agent.
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="service-health",
            display_name="Service Health Assessment",
            description="Live Kubernetes service health check with tool-backed data collection",
            version="1.0.0",
            tags=["sre", "live", "health", "kubernetes"],
            supported_phases=[
                SkillPhase.ANALYZE,
                SkillPhase.EXECUTE,
                SkillPhase.REPORT,
            ],
            recommended_model="gemini-2.5-pro",
            requires_live_tools=True,
        )

    def get_system_instruction(self) -> str:
        return SYSTEM_INSTRUCTION

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        template = PHASE_PROMPTS.get(phase.value, PHASE_PROMPTS["analyze"])
        return template.format(context=context, user_input=user_input)

    def get_agents_config(self) -> list[dict]:
        """Return the 3-agent sequential pipeline configuration.

        Agent 1 (health_gatherer) has ``requires_tools=True`` which tells
        the :class:`~vaig.agents.orchestrator.Orchestrator` to instantiate
        a :class:`~vaig.agents.tool_aware.ToolAwareAgent`.

        Agents 2 and 3 have ``requires_tools=False`` (the default) and
        are instantiated as :class:`~vaig.agents.specialist.SpecialistAgent`.

        Note: ``ToolAwareAgent.from_config_dict`` reads the ``system_prompt``
        key, while ``SpecialistAgent.from_config_dict`` reads
        ``system_instruction``.  Both keys are provided on the gatherer
        config for maximum compatibility.
        """
        return [
            {
                "name": "health_gatherer",
                "role": "Health Data Gatherer",
                "requires_tools": True,
                # ToolAwareAgent.from_config_dict reads "system_prompt"
                "system_prompt": HEALTH_GATHERER_PROMPT,
                # SpecialistAgent.from_config_dict reads "system_instruction"
                # (included for defensive compatibility if routing changes)
                "system_instruction": HEALTH_GATHERER_PROMPT,
                "model": "gemini-2.5-pro",
            },
            {
                "name": "health_analyzer",
                "role": "Health Pattern Analyzer",
                "requires_tools": False,
                "system_instruction": HEALTH_ANALYZER_PROMPT,
                "model": "gemini-2.5-flash",
            },
            {
                "name": "health_reporter",
                "role": "Health Report Generator",
                "requires_tools": False,
                "system_instruction": HEALTH_REPORTER_PROMPT,
                "model": "gemini-2.5-flash",
            },
        ]

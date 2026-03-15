"""Service Health Skill — live Kubernetes service health assessment.

A 4-agent sequential pipeline with two-pass verification that demonstrates
the ToolAwareAgent + Orchestrator integration.  The first agent uses live
tools to collect cluster health data; the second analyzes patterns; the
third verifies findings with targeted tool calls; the fourth produces a
structured markdown report.
"""

from __future__ import annotations

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.service_health.prompts import (
    HEALTH_ANALYZER_PROMPT,
    HEALTH_GATHERER_PROMPT,
    HEALTH_REPORTER_PROMPT,
    HEALTH_VERIFIER_PROMPT,
    PHASE_PROMPTS,
    SYSTEM_INSTRUCTION,
)


class ServiceHealthSkill(BaseSkill):
    """Service health assessment skill using live Kubernetes tools.

    Implements a 4-agent sequential pipeline with two-pass verification:

    1. **health_gatherer** (``requires_tools=True``):
       Uses live kubectl tools to collect pod status, resource usage,
       logs, events, and deployment state.

    2. **health_analyzer** (``requires_tools=False``):
       Text-only agent that receives gathered data and performs SRE-style
       pattern analysis — degraded services, resource pressure, error
       rate spikes, cross-service correlations.

    3. **health_verifier** (``requires_tools=True``):
       Tool-aware agent that makes targeted verification calls specified
       in the analyzer's Verification Gap fields.  Confirms, upgrades,
       or downgrades finding confidence levels before reporting.

    4. **health_reporter** (``requires_tools=False``):
       Text-only agent that synthesizes verified findings into a
       structured markdown report with severity classification,
       root-cause hypotheses, and actionable remediation commands.

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

    def get_required_output_sections(self) -> list[str]:
        """Mandatory sections the gatherer (first agent) must produce.

        These correspond to the MANDATORY OUTPUT FORMAT defined in
        ``HEALTH_GATHERER_PROMPT``.  The orchestrator validates the gatherer's
        output against these sections and retries once if any are missing.
        """
        return [
            "Cluster Overview",
            "Service Status",
            "Events Timeline",
            "Raw Findings",
            "Cloud Logging Findings",
        ]

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        template = PHASE_PROMPTS.get(phase.value, PHASE_PROMPTS["analyze"])
        return template.format(context=context, user_input=user_input)

    def get_agents_config(self) -> list[dict]:
        """Return the 4-agent sequential pipeline configuration.

        Agent 1 (health_gatherer) has ``requires_tools=True`` which tells
        the :class:`~vaig.agents.orchestrator.Orchestrator` to instantiate
        a :class:`~vaig.agents.tool_aware.ToolAwareAgent`.

        Agent 2 (health_analyzer) has ``requires_tools=False`` (the default)
        and is instantiated as :class:`~vaig.agents.specialist.SpecialistAgent`.

        Agent 3 (health_verifier) has ``requires_tools=True`` — it makes
        targeted tool calls to verify findings from the analyzer.  Uses a
        fast model with limited iterations for efficiency.

        Agent 4 (health_reporter) has ``requires_tools=False`` and
        is instantiated as :class:`~vaig.agents.specialist.SpecialistAgent`.

        Note: Both ``ToolAwareAgent.from_config_dict`` and
        ``SpecialistAgent.from_config_dict`` read the ``system_instruction``
        key.  The ``system_prompt`` alias is accepted as a backward-compat
        fallback by ``ToolAwareAgent`` but is no longer needed here.
        """
        return [
            {
                "name": "health_gatherer",
                "role": "Health Data Gatherer",
                "requires_tools": True,
                "system_instruction": HEALTH_GATHERER_PROMPT,
                "model": "gemini-2.5-pro",
                "temperature": 0.2,  # Low temp for precise data collection
                # Mandatory Cloud Logging (Steps 7a-7d) requires ~20 iterations
                "max_iterations": 25,
            },
            {
                "name": "health_analyzer",
                "role": "Health Pattern Analyzer",
                "requires_tools": False,
                "system_instruction": HEALTH_ANALYZER_PROMPT,
                "model": "gemini-2.5-flash",
                "temperature": 0.2,  # Low temp for precise analysis
            },
            {
                "name": "health_verifier",
                "role": "Health Finding Verifier",
                "requires_tools": True,
                "system_instruction": HEALTH_VERIFIER_PROMPT,
                "model": "gemini-2.5-flash",
                "max_iterations": 10,
                "temperature": 0.2,  # Low temp for precise verification
            },
            {
                "name": "health_reporter",
                "role": "Health Report Generator",
                "requires_tools": False,
                "system_instruction": HEALTH_REPORTER_PROMPT,
                "model": "gemini-2.5-flash",
                "temperature": 0.3,  # Slightly higher for natural writing
            },
        ]

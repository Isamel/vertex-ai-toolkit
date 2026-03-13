"""Tests for ServiceHealthSkill — the first skill with live tool support.

Validates:
- Skill metadata (name, tags, requires_live_tools, supported_phases)
- System instruction is non-empty
- Phase prompts inject context and user_input correctly
- Agent pipeline configuration (3 agents, sequential, requires_tools flags)
- ToolAwareAgent compatibility (system_prompt key on gatherer)
- SpecialistAgent compatibility (system_instruction key on analyzer/reporter)
"""

from __future__ import annotations

from vaig.skills.base import SkillPhase


class TestServiceHealthSkillMetadata:
    """Metadata contract tests."""

    def test_name(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        meta = skill.get_metadata()
        assert meta.name == "service-health"

    def test_display_name(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        meta = skill.get_metadata()
        assert meta.display_name == "Service Health Assessment"

    def test_requires_live_tools(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        meta = skill.get_metadata()
        assert meta.requires_live_tools is True

    def test_supported_phases(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        meta = skill.get_metadata()
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.EXECUTE in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases

    def test_tags(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        meta = skill.get_metadata()
        assert "sre" in meta.tags
        assert "live" in meta.tags
        assert "health" in meta.tags
        assert "kubernetes" in meta.tags

    def test_recommended_model(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        meta = skill.get_metadata()
        assert meta.recommended_model == "gemini-2.5-pro"

    def test_version(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        meta = skill.get_metadata()
        assert meta.version == "1.0.0"


class TestServiceHealthSkillSystemInstruction:
    """System instruction tests."""

    def test_non_empty(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_contains_sre_context(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        instruction = skill.get_system_instruction()
        assert "Site Reliability Engineer" in instruction

    def test_contains_kubernetes_context(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        instruction = skill.get_system_instruction()
        assert "Kubernetes" in instruction


class TestServiceHealthSkillPhasePrompts:
    """Phase prompt template tests."""

    def test_analyze_prompt_injects_context(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        prompt = skill.get_phase_prompt(
            SkillPhase.ANALYZE,
            context="pod nginx-abc123 CrashLoopBackOff",
            user_input="Check cluster health",
        )
        assert "pod nginx-abc123 CrashLoopBackOff" in prompt
        assert "Check cluster health" in prompt

    def test_execute_prompt_injects_context(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        prompt = skill.get_phase_prompt(
            SkillPhase.EXECUTE,
            context="namespace: production",
            user_input="Collect health data",
        )
        assert "namespace: production" in prompt
        assert "Collect health data" in prompt

    def test_report_prompt_injects_context(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        prompt = skill.get_phase_prompt(
            SkillPhase.REPORT,
            context="3 critical findings, 2 warnings",
            user_input="Generate health report",
        )
        assert "3 critical findings, 2 warnings" in prompt
        assert "Generate health report" in prompt

    def test_unknown_phase_falls_back_to_analyze(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        prompt = skill.get_phase_prompt(
            SkillPhase.VALIDATE,
            context="some context",
            user_input="some input",
        )
        # Should fall back to analyze template
        assert "some context" in prompt
        assert "some input" in prompt


class TestServiceHealthSkillAgentsConfig:
    """Agent pipeline configuration tests — the critical integration point."""

    def test_has_three_agents(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 3

    def test_agent_names(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_agents_config()
        names = [a["name"] for a in agents]
        assert names == ["health_gatherer", "health_analyzer", "health_reporter"]

    def test_agent_roles(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_agents_config()
        for agent in agents:
            assert "role" in agent
            assert isinstance(agent["role"], str)
            assert len(agent["role"]) > 0

    def test_agent_models(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_agents_config()
        # Gatherer uses pro model for complex tool use
        assert agents[0]["model"] == "gemini-2.5-pro"
        # Analyzer and reporter use flash for speed
        assert agents[1]["model"] == "gemini-2.5-flash"
        assert agents[2]["model"] == "gemini-2.5-flash"

    def test_gatherer_requires_tools_true(self) -> None:
        """The health_gatherer MUST have requires_tools=True.

        This is THE critical flag that tells the Orchestrator to create
        a ToolAwareAgent instead of a SpecialistAgent.
        """
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_agents_config()
        gatherer = agents[0]
        assert gatherer["name"] == "health_gatherer"
        assert gatherer["requires_tools"] is True

    def test_analyzer_requires_tools_false(self) -> None:
        """The health_analyzer MUST NOT require tools."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_agents_config()
        analyzer = agents[1]
        assert analyzer["name"] == "health_analyzer"
        assert analyzer.get("requires_tools", False) is False

    def test_reporter_requires_tools_false(self) -> None:
        """The health_reporter MUST NOT require tools."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_agents_config()
        reporter = agents[2]
        assert reporter["name"] == "health_reporter"
        assert reporter.get("requires_tools", False) is False

    def test_gatherer_has_system_prompt_for_tool_aware_agent(self) -> None:
        """ToolAwareAgent.from_config_dict expects 'system_prompt' key."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_agents_config()
        gatherer = agents[0]
        assert "system_prompt" in gatherer
        assert isinstance(gatherer["system_prompt"], str)
        assert len(gatherer["system_prompt"]) > 0

    def test_analyzer_has_system_instruction_for_specialist_agent(self) -> None:
        """SpecialistAgent.from_config_dict expects 'system_instruction' key."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_agents_config()
        analyzer = agents[1]
        assert "system_instruction" in analyzer
        assert isinstance(analyzer["system_instruction"], str)
        assert len(analyzer["system_instruction"]) > 0

    def test_reporter_has_system_instruction_for_specialist_agent(self) -> None:
        """SpecialistAgent.from_config_dict expects 'system_instruction' key."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_agents_config()
        reporter = agents[2]
        assert "system_instruction" in reporter
        assert isinstance(reporter["system_instruction"], str)
        assert len(reporter["system_instruction"]) > 0

    def test_sequential_pipeline_order(self) -> None:
        """Agents must be in gathering → analysis → reporting order.

        The Orchestrator's execute_sequential/execute_with_tools methods
        process agents in list order, feeding each agent's output as
        context to the next.
        """
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_agents_config()
        # Only the first agent should require tools
        requires_tools_flags = [a.get("requires_tools", False) for a in agents]
        assert requires_tools_flags == [True, False, False]

    def test_only_gatherer_requires_tools(self) -> None:
        """Exactly one agent in the pipeline requires tools."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_agents_config()
        tool_agents = [a for a in agents if a.get("requires_tools")]
        assert len(tool_agents) == 1
        assert tool_agents[0]["name"] == "health_gatherer"


class TestServiceHealthSkillPromptContent:
    """Validate that agent prompts contain domain-appropriate content."""

    def test_gatherer_prompt_mentions_kubectl(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_agents_config()
        gatherer_prompt = agents[0]["system_prompt"]
        assert "kubectl" in gatherer_prompt.lower()

    def test_gatherer_prompt_mentions_pods(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_agents_config()
        gatherer_prompt = agents[0]["system_prompt"]
        assert "pod" in gatherer_prompt.lower()

    def test_analyzer_prompt_mentions_severity(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_agents_config()
        analyzer_prompt = agents[1]["system_instruction"]
        assert "CRITICAL" in analyzer_prompt
        assert "WARNING" in analyzer_prompt

    def test_analyzer_prompt_mentions_crashloopbackoff(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_agents_config()
        analyzer_prompt = agents[1]["system_instruction"]
        assert "CrashLoopBackOff" in analyzer_prompt

    def test_reporter_prompt_mentions_markdown_structure(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_agents_config()
        reporter_prompt = agents[2]["system_instruction"]
        assert "Executive Summary" in reporter_prompt
        assert "Recommended Actions" in reporter_prompt

    def test_reporter_prompt_mentions_remediation_commands(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_agents_config()
        reporter_prompt = agents[2]["system_instruction"]
        assert "kubectl" in reporter_prompt.lower()


class TestServiceHealthQualityConstraints:
    """Validate the 4 quality constraints are present in prompts.

    These tests ensure the prompt engineering fixes for hallucination,
    actionability, scope precision, and findings structure remain in place.
    Regression protection — if someone removes these constraints, tests fail.
    """

    # ── Problem 1: Anti-Hallucination ────────────────────────

    def test_system_instruction_has_anti_hallucination_rules(self) -> None:
        """SYSTEM_INSTRUCTION must contain anti-hallucination rules."""
        from vaig.skills.service_health.prompts import SYSTEM_INSTRUCTION

        assert "NEVER invent" in SYSTEM_INSTRUCTION
        assert "placeholder" in SYSTEM_INSTRUCTION.lower()
        assert "REDACTED" in SYSTEM_INSTRUCTION
        assert "Data not available" in SYSTEM_INSTRUCTION

    def test_gatherer_has_data_integrity_rules(self) -> None:
        """Gatherer must enforce data integrity — no fabrication."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "NEVER fabricate" in HEALTH_GATHERER_PROMPT
        assert "Tool returned no data" in HEALTH_GATHERER_PROMPT
        assert "exact tool output" in HEALTH_GATHERER_PROMPT.lower()

    def test_reporter_has_anti_hallucination_section(self) -> None:
        """Reporter must have explicit anti-hallucination rules."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Anti-Hallucination" in HEALTH_REPORTER_PROMPT
        assert "NEVER invent data" in HEALTH_REPORTER_PROMPT
        assert "placeholder" in HEALTH_REPORTER_PROMPT.lower()

    # ── Problem 2: Actionability ─────────────────────────────

    def test_reporter_has_actionability_section(self) -> None:
        """Reporter must enforce actionable recommendations with exact commands."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Actionability" in HEALTH_REPORTER_PROMPT
        assert "exact kubectl commands" in HEALTH_REPORTER_PROMPT.lower()
        assert "copy-paste" in HEALTH_REPORTER_PROMPT.lower()

    def test_reporter_has_three_time_horizons(self) -> None:
        """Reporter must structure actions into 3 time horizons."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Immediate (next 5 minutes)" in HEALTH_REPORTER_PROMPT
        assert "Short-term (next 1 hour)" in HEALTH_REPORTER_PROMPT
        assert "Long-term (next sprint)" in HEALTH_REPORTER_PROMPT

    def test_reporter_requires_commands_for_every_action(self) -> None:
        """Reporter must require a command for every action — no vague suggestions."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "EVERY action MUST include an exact command" in HEALTH_REPORTER_PROMPT

    # ── Problem 3: Scope Precision ───────────────────────────

    def test_system_instruction_has_scope_precision_rules(self) -> None:
        """SYSTEM_INSTRUCTION must define scope levels."""
        from vaig.skills.service_health.prompts import SYSTEM_INSTRUCTION

        assert "Cluster-level" in SYSTEM_INSTRUCTION
        assert "Namespace-level" in SYSTEM_INSTRUCTION
        assert "Resource-level" in SYSTEM_INSTRUCTION

    def test_system_instruction_prevents_scope_exaggeration(self) -> None:
        """SYSTEM_INSTRUCTION must explicitly prevent scope exaggeration."""
        from vaig.skills.service_health.prompts import SYSTEM_INSTRUCTION

        assert "NEVER say the cluster is" in SYSTEM_INSTRUCTION
        assert "RESOURCE-LEVEL issue" in SYSTEM_INSTRUCTION

    def test_reporter_has_scope_precision_section(self) -> None:
        """Reporter must have explicit scope precision rules."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Scope Precision" in HEALTH_REPORTER_PROMPT
        assert "NEVER exaggerate scope" in HEALTH_REPORTER_PROMPT

    def test_reporter_executive_summary_has_scope_field(self) -> None:
        """Reporter template must include Scope in Executive Summary."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "**Scope**" in HEALTH_REPORTER_PROMPT

    def test_analyzer_has_scope_in_severity_assessment(self) -> None:
        """Analyzer must include scope classification in severity assessment."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "**Scope**" in HEALTH_ANALYZER_PROMPT
        assert "Cluster-wide" in HEALTH_ANALYZER_PROMPT

    # ── Problem 4: Findings Structure ────────────────────────

    def test_analyzer_has_mandatory_findings_structure(self) -> None:
        """Analyzer must enforce structured findings with What/Evidence/Impact/Affected."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "MANDATORY Output Format" in HEALTH_ANALYZER_PROMPT
        assert "**What**" in HEALTH_ANALYZER_PROMPT
        assert "**Evidence**" in HEALTH_ANALYZER_PROMPT
        assert "**Impact**" in HEALTH_ANALYZER_PROMPT
        assert "**Affected Resources**" in HEALTH_ANALYZER_PROMPT

    def test_reporter_has_mandatory_findings_structure(self) -> None:
        """Reporter must enforce structured findings with What/Evidence/Impact/Affected."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "MANDATORY Report Structure" in HEALTH_REPORTER_PROMPT
        assert "**What**" in HEALTH_REPORTER_PROMPT
        assert "**Evidence**" in HEALTH_REPORTER_PROMPT
        assert "**Impact**" in HEALTH_REPORTER_PROMPT
        assert "**Affected Resources**" in HEALTH_REPORTER_PROMPT

    def test_reporter_requires_all_four_fields(self) -> None:
        """Reporter must explicitly require all 4 fields for every finding."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "ALL four fields" in HEALTH_REPORTER_PROMPT

    def test_analyzer_requires_all_four_fields(self) -> None:
        """Analyzer must explicitly require all 4 fields for every finding."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "all four fields" in HEALTH_ANALYZER_PROMPT

    def test_reporter_forbids_unstructured_paragraphs(self) -> None:
        """Reporter must forbid unstructured paragraphs in findings."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "No unstructured paragraphs" in HEALTH_REPORTER_PROMPT

    # ── Timeline Rules ───────────────────────────────────────

    def test_reporter_has_timeline_anti_hallucination(self) -> None:
        """Reporter must prevent fabricated timeline events."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "NEVER fabricate timestamps" in HEALTH_REPORTER_PROMPT


class TestServiceHealthSkillRegistration:
    """Verify the skill is properly registered in the builtin skills map."""

    def test_skill_in_builtin_registry(self) -> None:
        from vaig.skills.registry import _discover_builtin_skills

        discovered = _discover_builtin_skills()
        assert "service-health" in discovered
        assert discovered["service-health"] == "vaig.skills.service_health.skill"

    def test_skill_can_be_imported(self) -> None:
        """Verify the registry's module path actually resolves."""
        import importlib

        from vaig.skills.registry import _discover_builtin_skills

        module_path = _discover_builtin_skills()["service-health"]
        module = importlib.import_module(module_path)
        assert hasattr(module, "ServiceHealthSkill")

    def test_skill_instantiation_from_module(self) -> None:
        """Verify the skill can be instantiated from its module path."""
        import importlib
        import inspect

        from vaig.skills.base import BaseSkill
        from vaig.skills.registry import _discover_builtin_skills

        module_path = _discover_builtin_skills()["service-health"]
        module = importlib.import_module(module_path)

        # Find the BaseSkill subclass (same logic as _find_skill_class)
        skill_class = None
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseSkill) and obj is not BaseSkill:
                skill_class = obj
                break

        assert skill_class is not None
        skill = skill_class()
        assert skill.get_metadata().name == "service-health"


class TestServiceHealthToolAccuracyConstraints:
    """Validate that prompts reference the correct tools and avoid the
    known kubectl_get-for-events bug.

    These are regression tests for the v2 prompt rewrite.
    """

    # ── Gatherer: correct tool references ────────────────────

    def test_gatherer_references_get_events(self) -> None:
        """Gatherer MUST use get_events for event retrieval."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "get_events" in HEALTH_GATHERER_PROMPT

    def test_gatherer_references_get_rollout_status(self) -> None:
        """Gatherer MUST use get_rollout_status for deployment rollout checks."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "get_rollout_status" in HEALTH_GATHERER_PROMPT

    def test_gatherer_references_get_rollout_history(self) -> None:
        """Gatherer MUST use get_rollout_history for revision inspection."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "get_rollout_history" in HEALTH_GATHERER_PROMPT

    def test_gatherer_references_get_container_status(self) -> None:
        """Gatherer MUST use get_container_status for pod-level investigation."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "get_container_status" in HEALTH_GATHERER_PROMPT

    def test_gatherer_references_get_node_conditions(self) -> None:
        """Gatherer MUST use get_node_conditions for cluster baseline."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "get_node_conditions" in HEALTH_GATHERER_PROMPT

    def test_gatherer_references_gcloud_monitoring_query(self) -> None:
        """Gatherer MUST reference gcloud_monitoring_query for HPA metric verification."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "gcloud_monitoring_query" in HEALTH_GATHERER_PROMPT

    def test_gatherer_does_not_use_kubectl_get_for_events(self) -> None:
        """Gatherer MUST NOT instruct using kubectl_get to retrieve events — that was the bug."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "use kubectl_get to retrieve events" not in HEALTH_GATHERER_PROMPT.lower()
        assert "use `kubectl_get` to retrieve" not in HEALTH_GATHERER_PROMPT.lower() or \
            "events" not in HEALTH_GATHERER_PROMPT.lower().split("use `kubectl_get` to retrieve")[1].split("\n")[0]

    # ── Analyzer: confidence taxonomy and verification gaps ──

    def test_analyzer_has_confidence_taxonomy(self) -> None:
        """Analyzer MUST define CONFIRMED/HIGH/MEDIUM/LOW confidence levels."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "CONFIRMED" in HEALTH_ANALYZER_PROMPT
        assert "HIGH" in HEALTH_ANALYZER_PROMPT
        assert "MEDIUM" in HEALTH_ANALYZER_PROMPT
        assert "LOW" in HEALTH_ANALYZER_PROMPT

    def test_analyzer_mentions_verification_gap(self) -> None:
        """Analyzer MUST include verification gap analysis for non-confirmed findings."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "Verification" in HEALTH_ANALYZER_PROMPT
        # Must mention data gap concept
        assert "DATA GAP" in HEALTH_ANALYZER_PROMPT

    # ── Reporter: banned practices and evidence rules ────────

    def test_reporter_bans_kubectl_edit(self) -> None:
        """Reporter MUST ban kubectl edit as a first option."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "kubectl edit" in HEALTH_REPORTER_PROMPT
        # Must be in a BANNED context
        assert "BANNED" in HEALTH_REPORTER_PROMPT

    def test_reporter_mentions_corrected_yaml(self) -> None:
        """Reporter MUST instruct showing corrected YAML when fixes are proposed."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Corrected YAML" in HEALTH_REPORTER_PROMPT or \
            "CORRECTED YAML" in HEALTH_REPORTER_PROMPT or \
            "corrected YAML" in HEALTH_REPORTER_PROMPT

    def test_reporter_mentions_get_rollout_history_for_rollbacks(self) -> None:
        """Reporter MUST reference get_rollout_history before recommending rollback."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "get_rollout_history" in HEALTH_REPORTER_PROMPT
        assert "to-revision" in HEALTH_REPORTER_PROMPT.lower()

    def test_reporter_bans_no_direct_kubectl_command(self) -> None:
        """Reporter MUST ban the phrase 'No direct kubectl command'."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "No direct kubectl command" in HEALTH_REPORTER_PROMPT
        # Must be in a BANNED context
        assert "NEVER say" in HEALTH_REPORTER_PROMPT

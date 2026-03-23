"""Tests for ServiceHealthSkill — the first skill with live tool support.

Validates:
- Skill metadata (name, tags, requires_live_tools, supported_phases)
- System instruction is non-empty
- Phase prompts inject context and user_input correctly
- Agent pipeline configuration (4 agents, sequential, requires_tools flags)
- ToolAwareAgent compatibility (system_instruction key on gatherer/verifier)
- SpecialistAgent compatibility (system_instruction key on analyzer/reporter)
- Verifier agent prompt content and anti-hallucination rules
- Two-pass verification pipeline: analyzer → verifier → reporter
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

    def test_has_four_agents(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        assert len(agents) == 4

    def test_agent_names(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        names = [a["name"] for a in agents]
        assert names == ["health_gatherer", "health_analyzer", "health_verifier", "health_reporter"]

    def test_agent_roles(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        for agent in agents:
            assert "role" in agent
            assert isinstance(agent["role"], str)
            assert len(agent["role"]) > 0

    def test_agent_models(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        # Gatherer uses pro model for complex tool use
        assert agents[0]["model"] == "gemini-2.5-pro"
        # Analyzer, verifier, and reporter use flash for speed
        assert agents[1]["model"] == "gemini-2.5-flash"
        assert agents[2]["model"] == "gemini-2.5-flash"
        assert agents[3]["model"] == "gemini-2.5-flash"

    def test_gatherer_requires_tools_true(self) -> None:
        """The health_gatherer MUST have requires_tools=True.

        This is THE critical flag that tells the Orchestrator to create
        a ToolAwareAgent instead of a SpecialistAgent.
        """
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        gatherer = agents[0]
        assert gatherer["name"] == "health_gatherer"
        assert gatherer["requires_tools"] is True

    def test_analyzer_requires_tools_false(self) -> None:
        """The health_analyzer MUST NOT require tools."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        analyzer = agents[1]
        assert analyzer["name"] == "health_analyzer"
        assert analyzer.get("requires_tools", False) is False

    def test_reporter_requires_tools_false(self) -> None:
        """The health_reporter MUST NOT require tools."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        reporter = agents[3]
        assert reporter["name"] == "health_reporter"
        assert reporter.get("requires_tools", False) is False

    def test_gatherer_has_system_instruction_for_tool_aware_agent(self) -> None:
        """ToolAwareAgent.from_config_dict expects 'system_instruction' key."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        gatherer = agents[0]
        assert "system_instruction" in gatherer
        assert isinstance(gatherer["system_instruction"], str)
        assert len(gatherer["system_instruction"]) > 0

    def test_analyzer_has_system_instruction_for_specialist_agent(self) -> None:
        """SpecialistAgent.from_config_dict expects 'system_instruction' key."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        analyzer = agents[1]
        assert "system_instruction" in analyzer
        assert isinstance(analyzer["system_instruction"], str)
        assert len(analyzer["system_instruction"]) > 0

    def test_reporter_has_system_instruction_for_specialist_agent(self) -> None:
        """SpecialistAgent.from_config_dict expects 'system_instruction' key."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        reporter = agents[3]
        assert "system_instruction" in reporter
        assert isinstance(reporter["system_instruction"], str)
        assert len(reporter["system_instruction"]) > 0

    def test_sequential_pipeline_order(self) -> None:
        """Agents must be in gathering → analysis → verification → reporting order.

        The Orchestrator's execute_sequential/execute_with_tools methods
        process agents in list order, feeding each agent's output as
        context to the next.
        """
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        # Gatherer and verifier require tools; analyzer and reporter do not
        requires_tools_flags = [a.get("requires_tools", False) for a in agents]
        assert requires_tools_flags == [True, False, True, False]

    def test_gatherer_and_verifier_require_tools(self) -> None:
        """Exactly two agents in the pipeline require tools."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        tool_agents = [a for a in agents if a.get("requires_tools")]
        assert len(tool_agents) == 2
        assert tool_agents[0]["name"] == "health_gatherer"
        assert tool_agents[1]["name"] == "health_verifier"

    # ── Verifier agent configuration ─────────────────────────

    def test_verifier_agent_requires_tools(self) -> None:
        """The health_verifier MUST have requires_tools=True.

        The verifier makes targeted tool calls to confirm findings from the
        analyzer — it needs ToolAwareAgent, not SpecialistAgent.
        """
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        verifier = agents[2]
        assert verifier["name"] == "health_verifier"
        assert verifier["requires_tools"] is True

    def test_verifier_agent_system_instruction_for_tool_aware(self) -> None:
        """ToolAwareAgent.from_config_dict expects 'system_instruction' key."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        verifier = agents[2]
        assert "system_instruction" in verifier
        assert isinstance(verifier["system_instruction"], str)
        assert len(verifier["system_instruction"]) > 0

    def test_verifier_agent_system_instruction(self) -> None:
        """Verifier provides system_instruction."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        verifier = agents[2]
        assert "system_instruction" in verifier
        assert isinstance(verifier["system_instruction"], str)
        assert len(verifier["system_instruction"]) > 0

    def test_verifier_agent_max_iterations(self) -> None:
        """Verifier must have max_iterations=15 for efficient targeted calls."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        verifier = agents[2]
        assert verifier["max_iterations"] == 15

    def test_gatherer_max_iterations(self) -> None:
        """Gatherer must have max_iterations=25 — mandatory Cloud Logging (Steps 7a-7d)
        adds ~4 extra tool calls beyond the default 15-iteration limit."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        gatherer = agents[0]
        assert gatherer["name"] == "health_gatherer"
        assert gatherer["max_iterations"] == 25

    def test_verifier_agent_model(self) -> None:
        """Verifier uses flash model for speed — verification is targeted, not complex."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        verifier = agents[2]
        assert verifier["model"] == "gemini-2.5-flash"


class TestServiceHealthSkillPromptContent:
    """Validate that agent prompts contain domain-appropriate content."""

    def test_gatherer_prompt_mentions_kubectl(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        gatherer_prompt = agents[0]["system_instruction"]
        assert "kubectl" in gatherer_prompt.lower()

    def test_gatherer_prompt_mentions_pods(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        gatherer_prompt = agents[0]["system_instruction"]
        assert "pod" in gatherer_prompt.lower()

    def test_analyzer_prompt_mentions_severity(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        analyzer_prompt = agents[1]["system_instruction"]
        assert "CRITICAL" in analyzer_prompt
        assert "HIGH" in analyzer_prompt
        assert "MEDIUM" in analyzer_prompt
        assert "LOW" in analyzer_prompt

    def test_analyzer_prompt_mentions_crashloopbackoff(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        analyzer_prompt = agents[1]["system_instruction"]
        assert "CrashLoopBackOff" in analyzer_prompt

    def test_reporter_prompt_mentions_markdown_structure(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        reporter_prompt = agents[3]["system_instruction"]
        assert "Executive Summary" in reporter_prompt
        assert "Recommended Actions" in reporter_prompt

    def test_reporter_prompt_mentions_remediation_commands(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        reporter_prompt = agents[3]["system_instruction"]
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

        # JSON schema uses urgency enum for time horizons
        assert "IMMEDIATE" in HEALTH_REPORTER_PROMPT
        assert "SHORT_TERM" in HEALTH_REPORTER_PROMPT
        assert "LONG_TERM" in HEALTH_REPORTER_PROMPT

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
        """Reporter must enforce structured findings via JSON schema fields."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "MANDATORY Report Structure" in HEALTH_REPORTER_PROMPT
        # JSON schema uses field names, not Markdown bold headers
        assert "``findings``" in HEALTH_REPORTER_PROMPT
        assert "``evidence``" in HEALTH_REPORTER_PROMPT
        assert "``impact``" in HEALTH_REPORTER_PROMPT
        assert "``affected_resources``" in HEALTH_REPORTER_PROMPT

    def test_reporter_requires_all_four_fields(self) -> None:
        """Reporter must explicitly require all required fields for every finding."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "all required fields populated" in HEALTH_REPORTER_PROMPT

    def test_analyzer_requires_all_four_fields(self) -> None:
        """Analyzer must explicitly require all fields for every finding."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "all fields" in HEALTH_ANALYZER_PROMPT

    def test_reporter_forbids_unstructured_paragraphs(self) -> None:
        """Reporter must forbid unstructured blobs in field values."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "No unstructured blobs" in HEALTH_REPORTER_PROMPT

    # ── Timeline Rules ───────────────────────────────────────

    def test_reporter_has_timeline_anti_hallucination(self) -> None:
        """Reporter must prevent fabricated/default timeline events.

        The consistency fix replaced the old 'NEVER fabricate timestamps'
        with deterministic timeline construction rules that prevent defaulting
        to 'no events' when data exists.
        """
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        # Rule 5: NEVER leave it empty when data exists
        assert "NEVER leave it empty" in HEALTH_REPORTER_PROMPT
        # Timeline section is MANDATORY
        assert "Timeline (MANDATORY)" in HEALTH_REPORTER_PROMPT


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


class TestServiceHealthVerifierPrompt:
    """Validate that the verifier prompt contains the required verification
    workflow, confidence operations, and anti-hallucination rules.

    These are regression tests — if someone weakens the verifier prompt,
    these tests will catch it.
    """

    def test_verifier_prompt_exists(self) -> None:
        """HEALTH_VERIFIER_PROMPT must exist and be non-empty."""
        from vaig.skills.service_health.prompts import HEALTH_VERIFIER_PROMPT

        assert isinstance(HEALTH_VERIFIER_PROMPT, str)
        assert len(HEALTH_VERIFIER_PROMPT) > 0

    def test_verifier_prompt_mentions_verification_gap(self) -> None:
        """Verifier must consume the Verification Gap field from analyzer output."""
        from vaig.skills.service_health.prompts import HEALTH_VERIFIER_PROMPT

        assert "Verification Gap" in HEALTH_VERIFIER_PROMPT

    def test_verifier_prompt_confidence_operations(self) -> None:
        """Verifier must define all confidence level operations."""
        from vaig.skills.service_health.prompts import HEALTH_VERIFIER_PROMPT

        assert "CONFIRMED" in HEALTH_VERIFIER_PROMPT
        assert "HIGH" in HEALTH_VERIFIER_PROMPT
        assert "MEDIUM" in HEALTH_VERIFIER_PROMPT
        assert "LOW" in HEALTH_VERIFIER_PROMPT
        assert "UNVERIFIABLE" in HEALTH_VERIFIER_PROMPT

    def test_verifier_prompt_anti_hallucination(self) -> None:
        """Verifier must have explicit anti-hallucination rules."""
        from vaig.skills.service_health.prompts import HEALTH_VERIFIER_PROMPT

        assert "NEVER fabricate" in HEALTH_VERIFIER_PROMPT
        assert "NEVER perform broad" in HEALTH_VERIFIER_PROMPT

    def test_verifier_prompt_targeted_tool_calls(self) -> None:
        """Verifier must only make targeted tool calls — not broad collection."""
        from vaig.skills.service_health.prompts import HEALTH_VERIFIER_PROMPT

        assert "targeted" in HEALTH_VERIFIER_PROMPT.lower()
        # Must explicitly state it is NOT a gatherer
        assert "NOT a" in HEALTH_VERIFIER_PROMPT

    def test_verifier_prompt_verification_summary(self) -> None:
        """Verifier must produce a Verification Summary with counts."""
        from vaig.skills.service_health.prompts import HEALTH_VERIFIER_PROMPT

        assert "Verification Summary" in HEALTH_VERIFIER_PROMPT

    def test_verifier_prompt_pass_through(self) -> None:
        """Verifier must pass through already-CONFIRMED findings unchanged."""
        from vaig.skills.service_health.prompts import HEALTH_VERIFIER_PROMPT

        assert "Pass through" in HEALTH_VERIFIER_PROMPT or \
            "pass through" in HEALTH_VERIFIER_PROMPT


class TestServiceHealthTwoPassPromptModifications:
    """Validate analyzer and reporter prompt modifications for the
    two-pass verification pipeline.

    The analyzer must produce machine-parseable Verification Gap fields,
    and the reporter must handle verified/downgraded/unverifiable findings.
    """

    # ── Analyzer modifications for two-pass pipeline ─────────

    def test_analyzer_mandatory_verification_gap(self) -> None:
        """Analyzer must make Verification Gap MANDATORY on every finding."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "Verification Gap" in HEALTH_ANALYZER_PROMPT
        assert "MANDATORY" in HEALTH_ANALYZER_PROMPT

    def test_analyzer_machine_parseable_format(self) -> None:
        """Analyzer must specify the Tool: format for verification gaps."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        # The prompt defines the "Tool: <tool_name>(<args>)" format
        assert "Tool:" in HEALTH_ANALYZER_PROMPT

    def test_analyzer_verifier_reference(self) -> None:
        """Analyzer must reference the downstream verification agent."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "verification" in HEALTH_ANALYZER_PROMPT.lower()

    # ── Reporter modifications for two-pass pipeline ─────────

    def test_reporter_downgraded_section(self) -> None:
        """Reporter must include a Downgraded Findings section."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Downgraded" in HEALTH_REPORTER_PROMPT

    def test_reporter_verified_findings(self) -> None:
        """Reporter must reference verified findings from the verifier."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        # Reporter must acknowledge that findings are verified
        assert "verified" in HEALTH_REPORTER_PROMPT.lower() or \
            "VERIFIED" in HEALTH_REPORTER_PROMPT

    def test_reporter_no_silent_omission(self) -> None:
        """Reporter must never silently omit downgraded findings."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "NEVER silently omit" in HEALTH_REPORTER_PROMPT

    def test_reporter_unverifiable_handling(self) -> None:
        """Reporter must handle UNVERIFIABLE findings with investigation steps."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "UNVERIFIABLE" in HEALTH_REPORTER_PROMPT


# ── Cloud Logging, exec_command verification, distroless ───


class TestServiceHealthPromptEnhancements:
    """Validate prompt enhancements for Cloud Logging patterns,
    exec_command verification gaps, and distroless container handling."""

    # ── Gatherer: Cloud Logging patterns ─────────────────

    def test_gatherer_cloud_logging_patterns(self) -> None:
        """Gatherer prompt must contain Cloud Logging filter examples."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "OOMKilled" in HEALTH_GATHERER_PROMPT
        assert "CrashLoopBackOff" in HEALTH_GATHERER_PROMPT
        assert "resource.type" in HEALTH_GATHERER_PROMPT

    def test_gatherer_log_time_range_guidance(self) -> None:
        """Gatherer prompt must mention narrow time ranges for log queries."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        prompt_lower = HEALTH_GATHERER_PROMPT.lower()
        assert "narrow" in prompt_lower or "time range" in prompt_lower or \
            "time_range" in prompt_lower

    # ── Analyzer: exec_command verification gaps ─────────

    def test_analyzer_exec_command_verification_gaps(self) -> None:
        """Analyzer prompt must contain exec_command examples for verification gaps."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "exec_command" in HEALTH_ANALYZER_PROMPT
        assert "curl" in HEALTH_ANALYZER_PROMPT
        assert "nslookup" in HEALTH_ANALYZER_PROMPT

    def test_analyzer_exec_enabled_note(self) -> None:
        """Analyzer must mention exec_enabled requirement."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "exec_enabled" in HEALTH_ANALYZER_PROMPT

    # ── Verifier: exec_command and distroless handling ────

    def test_verifier_exec_command_validation(self) -> None:
        """Verifier prompt must mention exec_command capabilities."""
        from vaig.skills.service_health.prompts import HEALTH_VERIFIER_PROMPT

        assert "exec_command" in HEALTH_VERIFIER_PROMPT

    def test_verifier_distroless_handling(self) -> None:
        """Verifier must handle distroless containers (missing tools)."""
        from vaig.skills.service_health.prompts import HEALTH_VERIFIER_PROMPT

        assert "distroless" in HEALTH_VERIFIER_PROMPT.lower()

    def test_verifier_exec_disabled_handling(self) -> None:
        """Verifier must handle the case where exec is disabled."""
        from vaig.skills.service_health.prompts import HEALTH_VERIFIER_PROMPT

        prompt_lower = HEALTH_VERIFIER_PROMPT.lower()
        assert "exec is disabled" in prompt_lower or \
            "exec_enabled" in HEALTH_VERIFIER_PROMPT


# ── 8 Prompt Consistency Fixes — Regression Tests ────────────────


class TestPromptConsistencyFix1GathererStructuredOutput:
    """Fix 1 (CRITICAL): Gatherer MUST produce mandatory structured output.

    Sections: Cluster Overview, Service Status table, Events Timeline, Raw Findings.
    Without this structure, downstream agents cannot parse gatherer output reliably.
    """

    def test_gatherer_has_mandatory_output_format_header(self) -> None:
        """Gatherer must declare a MANDATORY OUTPUT FORMAT section."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "MANDATORY OUTPUT FORMAT" in HEALTH_GATHERER_PROMPT

    def test_gatherer_has_cluster_overview_section(self) -> None:
        """Gatherer output must include a Cluster Overview section."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "### Cluster Overview" in HEALTH_GATHERER_PROMPT

    def test_gatherer_has_service_status_section(self) -> None:
        """Gatherer output must include a Service Status table."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "### Service Status" in HEALTH_GATHERER_PROMPT
        # Must define table columns
        assert "Deployment" in HEALTH_GATHERER_PROMPT
        assert "Ready Replicas" in HEALTH_GATHERER_PROMPT

    def test_gatherer_has_events_timeline_section(self) -> None:
        """Gatherer output must include an Events Timeline section."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "### Events Timeline" in HEALTH_GATHERER_PROMPT
        assert "CHRONOLOGICAL" in HEALTH_GATHERER_PROMPT

    def test_gatherer_has_raw_findings_section(self) -> None:
        """Gatherer output must include Raw Findings section."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "### Raw Findings" in HEALTH_GATHERER_PROMPT

    def test_gatherer_sections_not_optional(self) -> None:
        """Gatherer must state sections are NOT optional."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "NOT optional" in HEALTH_GATHERER_PROMPT or \
            "CRITICAL" in HEALTH_GATHERER_PROMPT.split("MANDATORY OUTPUT FORMAT")[1]


class TestPromptConsistencyFix2ReporterDeterministicTimeline:
    """Fix 2 (CRITICAL): Reporter MUST construct timeline deterministically.

    6 rules, never default to 'no events', must extract ALL timestamped events.
    """

    def test_reporter_timeline_section_is_mandatory(self) -> None:
        """Timeline section must be explicitly marked as MANDATORY."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Timeline (MANDATORY)" in HEALTH_REPORTER_PROMPT

    def test_reporter_timeline_must_build(self) -> None:
        """Reporter must be told to POPULATE the timeline field."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Populate the ``timeline`` field" in HEALTH_REPORTER_PROMPT

    def test_reporter_timeline_extract_every_event(self) -> None:
        """Rule 1: Extract EVERY event with a timestamp."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Extract EVERY event" in HEALTH_REPORTER_PROMPT

    def test_reporter_timeline_sort_chronologically(self) -> None:
        """Rule 2: Sort events chronologically (oldest first)."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Sort events chronologically" in HEALTH_REPORTER_PROMPT

    def test_reporter_timeline_table_format(self) -> None:
        """Rule 3: Timeline entries must have time, event, severity fields."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        # JSON schema uses structured fields instead of table format
        timeline_section = HEALTH_REPORTER_PROMPT[
            HEALTH_REPORTER_PROMPT.find("Timeline (MANDATORY)"):
        ]
        assert "``time``" in timeline_section
        assert "``event``" in timeline_section
        assert "``severity``" in timeline_section

    def test_reporter_timeline_events_without_timestamps(self) -> None:
        """Rule 4: Events without extractable timestamps shown in order."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "WITHOUT timestamps" in HEALTH_REPORTER_PROMPT

    def test_reporter_timeline_never_default_to_no_events(self) -> None:
        """Rule 5: NEVER leave timeline empty when data exists."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "NEVER leave it empty" in HEALTH_REPORTER_PROMPT

    def test_reporter_timeline_must_appear_in_every_report(self) -> None:
        """Rule 6: Timeline MUST have at least 1-2 entries in every report."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "MUST have at least 1-2 entries" in HEALTH_REPORTER_PROMPT


class TestPromptConsistencyFix3VerifierDecisionTree:
    """Fix 3 (CRITICAL): Verifier MUST use IF/THEN decision tree for confidence.

    One outcome per scenario, step-down-one-level rule.
    """

    def test_verifier_has_confidence_decision_tree_header(self) -> None:
        """Verifier must have an explicit 'Confidence Decision Tree' section."""
        from vaig.skills.service_health.prompts import HEALTH_VERIFIER_PROMPT

        assert "Confidence Decision Tree" in HEALTH_VERIFIER_PROMPT

    def test_verifier_decision_tree_confirms(self) -> None:
        """IF tool confirms hypothesis → set to CONFIRMED."""
        from vaig.skills.service_health.prompts import HEALTH_VERIFIER_PROMPT

        assert "IF tool call SUCCEEDS and result CONFIRMS" in HEALTH_VERIFIER_PROMPT

    def test_verifier_decision_tree_contradicts(self) -> None:
        """IF tool contradicts hypothesis → DOWNGRADE one level."""
        from vaig.skills.service_health.prompts import HEALTH_VERIFIER_PROMPT

        assert "IF tool call SUCCEEDS but result CONTRADICTS" in HEALTH_VERIFIER_PROMPT

    def test_verifier_decision_tree_inconclusive(self) -> None:
        """IF tool result is inconclusive → KEEP original level."""
        from vaig.skills.service_health.prompts import HEALTH_VERIFIER_PROMPT

        assert "IF tool call SUCCEEDS but result is INCONCLUSIVE" in HEALTH_VERIFIER_PROMPT

    def test_verifier_decision_tree_fails(self) -> None:
        """IF tool call fails → set to UNVERIFIABLE."""
        from vaig.skills.service_health.prompts import HEALTH_VERIFIER_PROMPT

        assert "IF tool call FAILS" in HEALTH_VERIFIER_PROMPT

    def test_verifier_step_down_one_level_rule(self) -> None:
        """Verifier must step down one level at a time, never skip levels."""
        from vaig.skills.service_health.prompts import HEALTH_VERIFIER_PROMPT

        assert "step down one level" in HEALTH_VERIFIER_PROMPT

    def test_verifier_never_upgrade_without_evidence(self) -> None:
        """Verifier must NEVER upgrade confidence without tool evidence."""
        from vaig.skills.service_health.prompts import HEALTH_VERIFIER_PROMPT

        assert "NEVER upgrade" in HEALTH_VERIFIER_PROMPT
        assert "without tool evidence" in HEALTH_VERIFIER_PROMPT

    def test_verifier_never_keep_confirmed_if_failed(self) -> None:
        """Verifier NEVER keeps CONFIRMED if verification tool call failed."""
        from vaig.skills.service_health.prompts import HEALTH_VERIFIER_PROMPT

        assert "NEVER keep" in HEALTH_VERIFIER_PROMPT


class TestPromptConsistencyFix4AnalyzerStructuredSummary:
    """Fix 4 (HIGH): Analyzer MUST output Service Status Summary table and
    Findings Overview at the TOP of output.
    """

    def test_analyzer_has_structured_summary_header(self) -> None:
        """Analyzer must have a Structured Summary section marked MANDATORY."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "Structured Summary (MANDATORY" in HEALTH_ANALYZER_PROMPT

    def test_analyzer_summary_at_top(self) -> None:
        """Structured Summary must appear at the TOP of output."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "TOP of your output" in HEALTH_ANALYZER_PROMPT

    def test_analyzer_has_service_status_summary_table(self) -> None:
        """Analyzer must include Service Status Summary table."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "### Service Status Summary" in HEALTH_ANALYZER_PROMPT
        # Table must have columns
        assert "Service/Deployment" in HEALTH_ANALYZER_PROMPT
        assert "Primary Issue" in HEALTH_ANALYZER_PROMPT

    def test_analyzer_has_findings_overview(self) -> None:
        """Analyzer must include Findings Overview with confidence counts."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "### Findings Overview" in HEALTH_ANALYZER_PROMPT
        assert "Total findings" in HEALTH_ANALYZER_PROMPT

    def test_analyzer_summary_even_if_no_findings(self) -> None:
        """Summary must appear even with zero findings."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "zero findings" in HEALTH_ANALYZER_PROMPT or \
            "No issues detected" in HEALTH_ANALYZER_PROMPT


class TestPromptConsistencyFix5ReporterEvidencePresentation:
    """Fix 5 (HIGH): Reporter MUST present evidence verbatim in code blocks.

    No paraphrasing raw K8s events. Show ACTUAL error text.
    """

    def test_reporter_has_evidence_presentation_section(self) -> None:
        """Reporter must have an explicit Evidence Presentation section."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Evidence Presentation" in HEALTH_REPORTER_PROMPT

    def test_reporter_evidence_verbatim(self) -> None:
        """Reporter must include raw events verbatim."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "verbatim" in HEALTH_REPORTER_PROMPT

    def test_reporter_evidence_code_blocks(self) -> None:
        """Evidence must be included as plain strings in the evidence list."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "plain string" in HEALTH_REPORTER_PROMPT

    def test_reporter_evidence_no_paraphrasing(self) -> None:
        """Reporter must NOT paraphrase raw event messages."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "do not paraphrase" in HEALTH_REPORTER_PROMPT

    def test_reporter_evidence_never_say_errors_without_text(self) -> None:
        """Reporter must never say 'tools reported errors' without the actual text."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "ACTUAL error text" in HEALTH_REPORTER_PROMPT or \
            "without showing the ACTUAL" in HEALTH_REPORTER_PROMPT

    def test_reporter_evidence_preserve_upstream_output(self) -> None:
        """Reporter must preserve upstream kubectl/tool output."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "preserve" in HEALTH_REPORTER_PROMPT.lower()


class TestPromptConsistencyFix6GathererNodeConditionsFirst:
    """Fix 6 (MEDIUM): Gatherer MUST collect node conditions ALWAYS as step 1.

    get_node_conditions() must be called FIRST, before any specific investigation.
    """

    def test_gatherer_step_1_always_first(self) -> None:
        """Step 1 must be explicitly marked as ALWAYS and FIRST."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "Step 1 (ALWAYS" in HEALTH_GATHERER_PROMPT
        assert "do this FIRST" in HEALTH_GATHERER_PROMPT

    def test_gatherer_step_1_calls_get_node_conditions(self) -> None:
        """Step 1 must call get_node_conditions."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        # get_node_conditions must appear in step 1 section
        # Use "### Step" headers to find the actual section, not the overview line
        step1_start = HEALTH_GATHERER_PROMPT.find("### Step 1")
        step2_start = HEALTH_GATHERER_PROMPT.find("### Step 2")
        step1_section = HEALTH_GATHERER_PROMPT[step1_start:step2_start]
        assert "get_node_conditions" in step1_section

    def test_gatherer_step_1_is_mandatory(self) -> None:
        """Step 1 must be marked as MANDATORY."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        step1_start = HEALTH_GATHERER_PROMPT.find("### Step 1")
        step2_start = HEALTH_GATHERER_PROMPT.find("### Step 2")
        step1_section = HEALTH_GATHERER_PROMPT[step1_start:step2_start]
        assert "MANDATORY" in step1_section

    def test_gatherer_step_1_before_specific_investigations(self) -> None:
        """Node conditions must come BEFORE deployment-specific investigations."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        step1_pos = HEALTH_GATHERER_PROMPT.find("### Step 1")
        step4_pos = HEALTH_GATHERER_PROMPT.find("### Step 4")  # Deep-dive step
        assert step1_pos < step4_pos


class TestPromptConsistencyFix7ReporterClusterOverviewMandatory:
    """Fix 7 (MEDIUM): Reporter Cluster Overview section MUST be mandatory
    with fallback text when data is unavailable.
    """

    def test_reporter_cluster_overview_mandatory(self) -> None:
        """Reporter must have Cluster Overview marked as MANDATORY."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Cluster Overview (MANDATORY)" in HEALTH_REPORTER_PROMPT

    def test_reporter_cluster_overview_uses_upstream_data(self) -> None:
        """Reporter must extract metrics from upstream Cluster Overview data."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "extract metrics into key/value pairs" in HEALTH_REPORTER_PROMPT

    def test_reporter_cluster_overview_fallback_text(self) -> None:
        """Reporter must have fallback text when cluster overview data is missing."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Cluster overview data was not collected" in HEALTH_REPORTER_PROMPT

    def test_reporter_never_data_not_available_without_explanation(self) -> None:
        """Reporter NEVER uses empty values without explanation."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "NEVER use empty values without explanation" in HEALTH_REPORTER_PROMPT


class TestPromptConsistencyFix8VerifierNoAmbiguousPhrases:
    """Fix 8 (MEDIUM): Verifier MUST NOT contain ambiguous phrases.

    'You may' → 'You MUST'. All directives must be deterministic.
    """

    def test_verifier_no_you_may(self) -> None:
        """Verifier prompt must NOT contain 'You may' (ambiguous directive)."""
        from vaig.skills.service_health.prompts import HEALTH_VERIFIER_PROMPT

        assert "You may" not in HEALTH_VERIFIER_PROMPT

    def test_verifier_uses_must(self) -> None:
        """Verifier prompt must use 'You MUST' or 'MUST' for directives."""
        from vaig.skills.service_health.prompts import HEALTH_VERIFIER_PROMPT

        # At least one MUST directive
        assert "MUST" in HEALTH_VERIFIER_PROMPT

    def test_verifier_never_directives_are_absolute(self) -> None:
        """Verifier NEVER rules must be absolute (no 'try to' or 'should')."""
        from vaig.skills.service_health.prompts import HEALTH_VERIFIER_PROMPT

        # Check that key rules use NEVER, not 'try not to' or 'should not'
        assert "NEVER fabricate" in HEALTH_VERIFIER_PROMPT
        assert "NEVER perform broad" in HEALTH_VERIFIER_PROMPT
        assert "NEVER add new findings" in HEALTH_VERIFIER_PROMPT

    def test_verifier_never_downgrade_directly_to_low(self) -> None:
        """Verifier must have rule: NEVER downgrade directly to LOW."""
        from vaig.skills.service_health.prompts import HEALTH_VERIFIER_PROMPT

        assert "NEVER downgrade directly to LOW" in HEALTH_VERIFIER_PROMPT


# ── Temperature & Validation — Non-Determinism Mitigation ────────────────


class TestServiceHealthTemperatureConfig:
    """Verify that all 4 agents have low temperature for deterministic output."""

    def test_gatherer_temperature(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        assert agents[0]["name"] == "health_gatherer"
        assert agents[0]["temperature"] == 0.0

    def test_analyzer_temperature(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        assert agents[1]["name"] == "health_analyzer"
        assert agents[1]["temperature"] == 0.2

    def test_verifier_temperature(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        assert agents[2]["name"] == "health_verifier"
        assert agents[2]["temperature"] == 0.2

    def test_reporter_temperature(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config()
        assert agents[3]["name"] == "health_reporter"
        assert agents[3]["temperature"] == 0.3

    def test_all_temperatures_below_default(self) -> None:
        """All agents must have temperature below the 0.7 default."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        for agent in skill.get_sequential_agents_config():
            assert agent["temperature"] < 0.7, (
                f"{agent['name']} temperature {agent['temperature']} is not below 0.7"
            )


class TestServiceHealthRequiredOutputSections:
    """Verify get_required_output_sections() returns expected sections."""

    def test_returns_list_not_none(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        sections = skill.get_required_output_sections()
        assert sections is not None
        assert isinstance(sections, list)

    def test_contains_cluster_overview(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        sections = skill.get_required_output_sections()
        assert "Cluster Overview" in sections

    def test_contains_service_status(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        sections = skill.get_required_output_sections()
        assert "Service Status" in sections

    def test_contains_events_timeline(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        sections = skill.get_required_output_sections()
        assert "Events Timeline" in sections

    def test_base_skill_returns_none(self) -> None:
        """BaseSkill.get_required_output_sections() returns None by default."""
        from vaig.skills.base import BaseSkill, SkillMetadata

        class DummySkill(BaseSkill):
            def get_metadata(self) -> SkillMetadata:
                return SkillMetadata(name="dummy", display_name="D", description="d")

            def get_system_instruction(self) -> str:
                return ""

            def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
                return ""

        assert DummySkill().get_required_output_sections() is None


# ── Cloud Logging Mandatory Collection — Regression Tests ────────────────


class TestGathererCloudLoggingMandatory:
    """Validate that the gatherer prompt makes gcloud_logging_query a MANDATORY
    data collection step, not an optional cross-reference.

    The gatherer was previously not calling gcloud_logging_query despite having
    access to it. These tests ensure the prompt explicitly forces the agent to
    call gcloud_logging_query as part of its mandatory data collection procedure.
    """

    def test_step_7_is_mandatory(self) -> None:
        """Step 7 must be explicitly marked as MANDATORY."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "Step 7 (MANDATORY" in HEALTH_GATHERER_PROMPT

    def test_step_7_says_always(self) -> None:
        """Step 7 must say ALWAYS to match Step 1's pattern."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        # Extract Step 7 header line — use "### Step" to skip overview line
        step7_start = HEALTH_GATHERER_PROMPT.find("### Step 7")
        step7_header = HEALTH_GATHERER_PROMPT[step7_start:step7_start + 100]
        assert "ALWAYS" in step7_header

    def test_step_7_references_gcloud_logging_query(self) -> None:
        """Step 7 must explicitly reference gcloud_logging_query tool."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        step7_start = HEALTH_GATHERER_PROMPT.find("### Step 7")
        step8_start = HEALTH_GATHERER_PROMPT.find("### Step 8")
        step7_section = HEALTH_GATHERER_PROMPT[step7_start:step8_start]
        assert "gcloud_logging_query" in step7_section

    def test_step_7_must_call_at_least_once(self) -> None:
        """Step 7 must require at least one gcloud_logging_query call per namespace."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        step7_start = HEALTH_GATHERER_PROMPT.find("### Step 7")
        step8_start = HEALTH_GATHERER_PROMPT.find("### Step 8")
        step7_section = HEALTH_GATHERER_PROMPT[step7_start:step8_start]
        assert "MUST call" in step7_section or "You MUST" in step7_section

    def test_step_7a_error_level_query(self) -> None:
        """Step 7a must include the baseline error-level query with severity>=ERROR."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "7a." in HEALTH_GATHERER_PROMPT
        step7a_start = HEALTH_GATHERER_PROMPT.find("#### 7a.")
        step7b_start = HEALTH_GATHERER_PROMPT.find("#### 7b.")
        step7a_section = HEALTH_GATHERER_PROMPT[step7a_start:step7b_start]
        assert "severity>=ERROR" in step7a_section
        assert "k8s_container" in step7a_section

    def test_step_7b_warning_level_query(self) -> None:
        """Step 7b must include warning-level pod query."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "7b." in HEALTH_GATHERER_PROMPT
        step7b_start = HEALTH_GATHERER_PROMPT.find("#### 7b.")
        step7c_start = HEALTH_GATHERER_PROMPT.find("#### 7c.")
        step7b_section = HEALTH_GATHERER_PROMPT[step7b_start:step7c_start]
        assert "severity>=WARNING" in step7b_section
        assert "k8s_pod" in step7b_section

    def test_step_7c_service_specific_query(self) -> None:
        """Step 7c must instruct service-specific log queries for unhealthy services."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "7c." in HEALTH_GATHERER_PROMPT
        step7c_start = HEALTH_GATHERER_PROMPT.find("#### 7c.")
        step7d_start = HEALTH_GATHERER_PROMPT.find("#### 7d.")
        step7c_section = HEALTH_GATHERER_PROMPT[step7c_start:step7d_start]
        assert "container_name" in step7c_section
        assert "gcloud_logging_query" in step7c_section

    def test_step_7d_correlation_with_k8s_events(self) -> None:
        """Step 7d must instruct correlation of Cloud Logging with K8s events."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "7d." in HEALTH_GATHERER_PROMPT
        step7d_start = HEALTH_GATHERER_PROMPT.find("#### 7d.")
        step8_start = HEALTH_GATHERER_PROMPT.find("Cloud Logging Query Patterns")
        step7d_section = HEALTH_GATHERER_PROMPT[step7d_start:step8_start]
        assert "correlate" in step7d_section.lower()
        assert "Step 3" in step7d_section or "events" in step7d_section.lower()

    def test_cloud_logging_is_mandatory_data_source(self) -> None:
        """Prompt must explicitly state Cloud Logging is a MANDATORY data source."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        step7_start = HEALTH_GATHERER_PROMPT.find("### Step 7")
        step8_start = HEALTH_GATHERER_PROMPT.find("### Step 8")
        step7_section = HEALTH_GATHERER_PROMPT[step7_start:step8_start]
        assert "MANDATORY data source" in step7_section

    def test_cloud_logging_filter_severity_error(self) -> None:
        """Cloud Logging patterns must include severity>=ERROR filter."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert 'severity>=ERROR AND resource.type="k8s_container"' in HEALTH_GATHERER_PROMPT

    def test_cloud_logging_filter_severity_warning(self) -> None:
        """Cloud Logging patterns must include severity>=WARNING filter."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert 'severity>=WARNING AND resource.type="k8s_pod"' in HEALTH_GATHERER_PROMPT

    def test_cloud_logging_filter_namespace(self) -> None:
        """Cloud Logging patterns must include namespace_name filter."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "resource.labels.namespace_name" in HEALTH_GATHERER_PROMPT


class TestGathererCloudLoggingMandatoryOutput:
    """Validate that Cloud Logging Findings is a mandatory output section."""

    def test_cloud_logging_findings_in_mandatory_output(self) -> None:
        """Cloud Logging Findings must appear in the MANDATORY OUTPUT FORMAT."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        mandatory_section = HEALTH_GATHERER_PROMPT[
            HEALTH_GATHERER_PROMPT.find("MANDATORY OUTPUT FORMAT"):
        ]
        assert "### Cloud Logging Findings" in mandatory_section

    def test_cloud_logging_findings_listed_as_not_optional(self) -> None:
        """The CRITICAL statement must list Cloud Logging Findings as not optional."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "Cloud Logging Findings sections are NOT optional" in HEALTH_GATHERER_PROMPT

    def test_cloud_logging_findings_in_output_format_template(self) -> None:
        """Cloud Logging Findings must appear in the mandatory output format section."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        output_format_start = HEALTH_GATHERER_PROMPT.find("## MANDATORY OUTPUT FORMAT")
        assert output_format_start != -1, "MANDATORY OUTPUT FORMAT section must exist"
        output_section = HEALTH_GATHERER_PROMPT[output_format_start:]
        assert "Cloud Logging Findings" in output_section


# ── Severity Classification — Operational Impact Rules ───────────────


class TestReporterSeverityClassification:
    """Validate that the reporter prompt contains severity classification rules
    based on operational IMPACT instead of copying K8s event severity labels.

    The reporter was previously copying K8s event severity (Normal/Warning) verbatim,
    which misclassifies critical operational failures like FailedCreate as mere
    WARNING instead of CRITICAL.
    """

    def test_severity_classification_section_exists(self) -> None:
        """Reporter must have a SEVERITY CLASSIFICATION section."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "SEVERITY CLASSIFICATION" in HEALTH_REPORTER_PROMPT

    def test_severity_evaluates_operational_impact(self) -> None:
        """Reporter must evaluate operational IMPACT, not K8s labels."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "OPERATIONAL IMPACT" in HEALTH_REPORTER_PROMPT

    def test_anti_copy_rule_exists(self) -> None:
        """Reporter must explicitly forbid copying K8s event severity."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Do NOT copy Kubernetes event severity" in HEALTH_REPORTER_PROMPT

    def test_severity_scale_has_five_levels(self) -> None:
        """Reporter must define a 5-level severity scale: CRITICAL, HIGH, MEDIUM, LOW, INFO."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        severity_section_start = HEALTH_REPORTER_PROMPT.find("SEVERITY CLASSIFICATION")
        severity_section_end = HEALTH_REPORTER_PROMPT.find("MANDATORY Report Structure")
        severity_section = HEALTH_REPORTER_PROMPT[severity_section_start:severity_section_end]

        assert "CRITICAL" in severity_section
        assert "HIGH" in severity_section
        assert "MEDIUM" in severity_section
        assert "LOW" in severity_section
        assert "INFO" in severity_section

    def test_failedcreate_classified_as_critical(self) -> None:
        """FailedCreate must be classified as CRITICAL, not WARNING."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "FailedCreate is CRITICAL" in HEALTH_REPORTER_PROMPT

    def test_crashloopbackoff_classified_as_critical(self) -> None:
        """CrashLoopBackOff must be classified as CRITICAL."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "CrashLoopBackOff is CRITICAL" in HEALTH_REPORTER_PROMPT

    def test_imagepullbackoff_classified_as_critical(self) -> None:
        """ImagePullBackOff must be classified as CRITICAL."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "ImagePullBackOff is CRITICAL" in HEALTH_REPORTER_PROMPT

    def test_persistent_oomkilled_classified_as_high(self) -> None:
        """Persistent OOMKilled must be classified as HIGH."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "OOMKilled is HIGH when persistent" in HEALTH_REPORTER_PROMPT

    def test_hpa_at_max_classified_as_high(self) -> None:
        """HPA at maxReplicas must be classified as HIGH."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "HPA at max is HIGH" in HEALTH_REPORTER_PROMPT

    def test_elevated_restart_count_classified_as_medium(self) -> None:
        """Elevated restart count must be classified as MEDIUM."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Elevated restart count is MEDIUM" in HEALTH_REPORTER_PROMPT

    def test_normal_k8s_events_classified_as_info(self) -> None:
        """Normal K8s events must be classified as INFO."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Normal K8s events are INFO" in HEALTH_REPORTER_PROMPT

    def test_anti_copy_rule_with_example(self) -> None:
        """Anti-copy rule must include a concrete FailedCreate example."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Anti-Copy Rule" in HEALTH_REPORTER_PROMPT
        # Must mention that K8s says Warning but operational impact is CRITICAL
        anti_copy_start = HEALTH_REPORTER_PROMPT.find("Anti-Copy Rule")
        anti_copy_section = HEALTH_REPORTER_PROMPT[anti_copy_start:anti_copy_start + 500]
        assert "FailedCreate" in anti_copy_section
        assert "CRITICAL" in anti_copy_section

    def test_severity_scale_in_severity_table(self) -> None:
        """Severity scale must be presented as a structured table."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "### Severity Scale" in HEALTH_REPORTER_PROMPT
        assert "| Severity | Criteria | Examples |" in HEALTH_REPORTER_PROMPT

    def test_findings_template_has_five_severity_levels(self) -> None:
        """The Findings field mapping must reference severity levels."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        # JSON schema uses severity enum values in the field mapping
        findings_start = HEALTH_REPORTER_PROMPT.find("``findings``")
        conciseness_start = HEALTH_REPORTER_PROMPT.find("### Conciseness Rule")
        findings_section = HEALTH_REPORTER_PROMPT[findings_start:conciseness_start]

        assert "CRITICAL" in findings_section
        assert "HIGH" in findings_section
        assert "MEDIUM" in findings_section
        assert "LOW" in findings_section
        assert "INFO" in findings_section

    def test_timeline_uses_operational_severity(self) -> None:
        """Timeline field must reference operational severity, not K8s event type."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        timeline_start = HEALTH_REPORTER_PROMPT.find("### Timeline")
        timeline_section = HEALTH_REPORTER_PROMPT[timeline_start:timeline_start + 500]
        assert "operational severity" in timeline_section.lower()


class TestPromptSchemaParameterConsistency:
    """Verify that prompt examples use correct parameter names matching tool schemas.

    These tests prevent regressions where prompt instructions reference
    wrong parameter names (e.g. pod_name instead of pod for kubectl_logs),
    which cause TypeErrors at runtime when the LLM follows the prompts.
    """

    def test_gatherer_no_pod_name_for_kubectl_logs(self) -> None:
        """kubectl_logs uses 'pod' not 'pod_name' — gatherer must not use pod_name."""
        # Find all kubectl_logs references in gatherer
        import re

        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        calls = re.findall(r"kubectl_logs\([^)]+\)", HEALTH_GATHERER_PROMPT)
        for call in calls:
            assert "pod_name" not in call, (
                f"Gatherer prompt uses 'pod_name' in kubectl_logs call: {call}. "
                "Schema parameter is 'pod'."
            )

    def test_gatherer_no_pod_name_for_get_container_status(self) -> None:
        """get_container_status uses 'name' not 'pod_name'."""
        import re

        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        calls = re.findall(r"get_container_status\([^)]+\)", HEALTH_GATHERER_PROMPT)
        for call in calls:
            assert "pod_name" not in call, (
                f"Gatherer prompt uses 'pod_name' in get_container_status call: {call}. "
                "Schema parameter is 'name'."
            )

    def test_gatherer_no_previous_for_kubectl_logs(self) -> None:
        """kubectl_logs does NOT have a 'previous' parameter — it's handled automatically."""
        import re

        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        calls = re.findall(r"kubectl_logs\([^)]+\)", HEALTH_GATHERER_PROMPT)
        for call in calls:
            assert "previous" not in call, (
                f"Gatherer prompt uses 'previous' in kubectl_logs call: {call}. "
                "kubectl_logs has no 'previous' parameter — it auto-fetches previous logs."
            )

    def test_analyzer_no_pod_name_for_kubectl_logs(self) -> None:
        """kubectl_logs uses 'pod' not 'pod_name' — analyzer must not use pod_name."""
        import re

        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        calls = re.findall(r"kubectl_logs\([^)]+\)", HEALTH_ANALYZER_PROMPT)
        for call in calls:
            assert "pod_name" not in call, (
                f"Analyzer prompt uses 'pod_name' in kubectl_logs call: {call}. "
                "Schema parameter is 'pod'."
            )

    def test_tool_call_reference_section_exists(self) -> None:
        """Gatherer prompt must have a Tool Call Reference section with exact param names."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "## Tool Call Reference" in HEALTH_GATHERER_PROMPT

    def test_tool_call_reference_lists_all_read_tools(self) -> None:
        """Tool Call Reference must list all read-only diagnostic tools."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        ref_start = HEALTH_GATHERER_PROMPT.find("## Tool Call Reference")
        ref_end = HEALTH_GATHERER_PROMPT.find("## Data Collection Procedure")
        ref_section = HEALTH_GATHERER_PROMPT[ref_start:ref_end]

        expected_tools = [
            "kubectl_get",
            "kubectl_describe",
            "kubectl_logs",
            "kubectl_top",
            "get_events",
            "get_rollout_status",
            "get_node_conditions",
            "get_container_status",
            "get_rollout_history",
            "exec_command",
            "check_rbac",
            "gcloud_logging_query",
            "gcloud_monitoring_query",
        ]
        for tool in expected_tools:
            assert tool in ref_section, (
                f"Tool Call Reference section missing tool: {tool}"
            )

    def test_tool_call_reference_warns_about_common_mistakes(self) -> None:
        """Tool Call Reference must explicitly warn about common parameter name mistakes."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        ref_start = HEALTH_GATHERER_PROMPT.find("## Tool Call Reference")
        ref_end = HEALTH_GATHERER_PROMPT.find("## Data Collection Procedure")
        ref_section = HEALTH_GATHERER_PROMPT[ref_start:ref_end]

        # Must warn that kubectl_logs uses 'pod' not 'pod_name'
        assert "pod" in ref_section and "NOT" in ref_section
        # Must warn that get_container_status uses 'name' not 'pod_name'
        assert "get_container_status" in ref_section


class TestServiceHealthAutopilotAwareness:
    """GKE Autopilot awareness is handled by dynamic injection (language.py),
    NOT by inline sections in prompts.  These tests verify:
    1. Prompts do NOT contain inline Autopilot sections (single source of truth).
    2. Dynamic injection via build_autopilot_instruction covers all required rules.
    """

    def test_gatherer_no_inline_autopilot_section(self) -> None:
        """Gatherer prompt must NOT have an inline Autopilot section."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "GKE Autopilot Awareness" not in HEALTH_GATHERER_PROMPT

    def test_analyzer_no_inline_autopilot_section(self) -> None:
        """Analyzer prompt must NOT have an inline Autopilot section."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "GKE Autopilot Context" not in HEALTH_ANALYZER_PROMPT

    def test_verifier_no_inline_autopilot_section(self) -> None:
        """Verifier prompt must NOT have an inline Autopilot section."""
        from vaig.skills.service_health.prompts import HEALTH_VERIFIER_PROMPT

        assert "GKE Autopilot Awareness" not in HEALTH_VERIFIER_PROMPT

    def test_reporter_no_inline_autopilot_section(self) -> None:
        """Reporter prompt must NOT have an inline Autopilot section."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "GKE Autopilot Cluster Overview" not in HEALTH_REPORTER_PROMPT

    def test_dynamic_injection_covers_node_data_context_only(self) -> None:
        """Dynamic Autopilot instruction must state node data is CONTEXT ONLY."""
        from vaig.core.language import build_autopilot_instruction

        result = build_autopilot_instruction(True)
        assert "CONTEXT ONLY" in result

    def test_dynamic_injection_covers_node_top_not_available(self) -> None:
        """Dynamic Autopilot instruction must warn kubectl_top nodes is unavailable."""
        from vaig.core.language import build_autopilot_instruction

        result = build_autopilot_instruction(True)
        assert "kubectl_top" in result
        assert "NOT available" in result

    def test_dynamic_injection_covers_no_node_actions(self) -> None:
        """Dynamic Autopilot instruction must prohibit node-level actions."""
        from vaig.core.language import build_autopilot_instruction

        result = build_autopilot_instruction(True)
        assert "node-level actions" in result

    def test_dynamic_injection_covers_mandatory_requests(self) -> None:
        """Dynamic Autopilot instruction must note mandatory resource requests."""
        from vaig.core.language import build_autopilot_instruction

        result = build_autopilot_instruction(True)
        assert "mandatory" in result.lower()
        assert "resource requests" in result.lower()

    def test_dynamic_injection_focuses_workload_level(self) -> None:
        """Dynamic Autopilot instruction must focus analysis on workload-level health."""
        from vaig.core.language import build_autopilot_instruction

        result = build_autopilot_instruction(True)
        assert "WORKLOAD-LEVEL" in result
        # Must NOT contain the old scope-narrowing language from the previous version
        assert "Focus on pod-level and workload-level health" not in result


class TestServiceHealthAntiDataFabrication:
    """Validate strengthened anti-fabrication rules in service_health prompts.

    These tests target specific gaps where the model may still invent data:
    - Fabricating numbers for summary tables and statistics
    - Inventing metrics (CPU %, memory %) not present in upstream data
    - Manufacturing findings to fill severity categories
    - Filling phase prompts without evidence constraints
    """

    # ── Analyzer: no fabricated counts or statistics ──────────

    def test_analyzer_forbids_fabricated_counts(self) -> None:
        """Analyzer must not invent numbers for Findings Overview."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "NEVER estimate or invent numbers" in HEALTH_ANALYZER_PROMPT

    def test_analyzer_forbids_guessing_service_status(self) -> None:
        """Analyzer must use 'Unknown' when data was not collected."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "Unknown" in HEALTH_ANALYZER_PROMPT
        assert "data not collected" in HEALTH_ANALYZER_PROMPT.lower()

    def test_analyzer_forbids_manufacturing_findings(self) -> None:
        """Analyzer must not create findings just to fill a severity category."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "NEVER create a finding" in HEALTH_ANALYZER_PROMPT

    # ── Reporter: no fabricated table values ──────────────────

    def test_reporter_forbids_fabricated_percentages(self) -> None:
        """Reporter must use N/A for missing metrics, not invented numbers."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "N/A" in HEALTH_REPORTER_PROMPT
        assert "NEVER estimate" in HEALTH_REPORTER_PROMPT

    def test_reporter_forbids_invented_table_values(self) -> None:
        """Reporter must not fill fields with plausible-looking numbers."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "NEVER fill fields" in HEALTH_REPORTER_PROMPT

    def test_reporter_prefers_accuracy_over_completeness(self) -> None:
        """Reporter must prefer a shorter accurate report over fabricated details."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "shorter" in HEALTH_REPORTER_PROMPT.lower()
        assert "accurate" in HEALTH_REPORTER_PROMPT.lower()

    # ── Phase prompts: anti-hallucination in entry points ─────

    def test_analyze_phase_has_anti_fabrication(self) -> None:
        """Analyze phase prompt must have anti-fabrication rules."""
        from vaig.skills.service_health.prompts import PHASE_PROMPTS

        analyze = PHASE_PROMPTS["analyze"]
        assert "NEVER invent" in analyze
        assert "evidence" in analyze.lower()

    def test_execute_phase_has_anti_fabrication(self) -> None:
        """Execute phase prompt must have anti-fabrication rules."""
        from vaig.skills.service_health.prompts import PHASE_PROMPTS

        execute = PHASE_PROMPTS["execute"]
        assert "NEVER fabricate" in execute
        assert "tool output" in execute.lower()

    def test_report_phase_has_anti_fabrication(self) -> None:
        """Report phase prompt must have anti-fabrication rules."""
        from vaig.skills.service_health.prompts import PHASE_PROMPTS

        report = PHASE_PROMPTS["report"]
        assert "NEVER invent" in report
        assert "Data not" in report


# ── Causa 3: Remediation Reasoning Framework — Regression Tests ──────────


class TestReporterRemediationReasoningFramework:
    """Validate that the reporter prompt contains the Remediation Reasoning Framework
    instead of the old Safe Action Hierarchy.

    Causa 3: Remediation should reason about the PROCESS (GitOps, Helm, Operator, Manual)
    rather than hardcoding "edit the YAML and re-apply" as the fix for everything.
    """

    def test_reporter_contains_remediation_reasoning_framework(self) -> None:
        """Reporter must contain the new Remediation Reasoning Framework section."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Remediation Reasoning Framework" in HEALTH_REPORTER_PROMPT

    def test_reporter_does_not_contain_safe_action_hierarchy(self) -> None:
        """Reporter must NOT contain the old Safe Action Hierarchy (replaced)."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Safe Action Hierarchy" not in HEALTH_REPORTER_PROMPT

    def test_reporter_mentions_gitops_management(self) -> None:
        """Reporter must mention GitOps management (ArgoCD, Flux)."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "GitOps" in HEALTH_REPORTER_PROMPT
        assert "ArgoCD" in HEALTH_REPORTER_PROMPT or "argocd" in HEALTH_REPORTER_PROMPT
        assert "Flux" in HEALTH_REPORTER_PROMPT or "fluxcd" in HEALTH_REPORTER_PROMPT

    def test_reporter_mentions_helm_management(self) -> None:
        """Reporter must mention Helm management."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Helm" in HEALTH_REPORTER_PROMPT
        assert "helm get values" in HEALTH_REPORTER_PROMPT
        assert "helm upgrade" in HEALTH_REPORTER_PROMPT

    def test_reporter_mentions_operator_management(self) -> None:
        """Reporter must mention operator management."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "operator" in HEALTH_REPORTER_PROMPT.lower()
        assert "OwnerReferences" in HEALTH_REPORTER_PROMPT
        assert "CRD/CR" in HEALTH_REPORTER_PROMPT or "CR" in HEALTH_REPORTER_PROMPT

    def test_reporter_mentions_manual_management(self) -> None:
        """Reporter must mention manual (no management annotations) as a management method."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        framework_start = HEALTH_REPORTER_PROMPT.find("Remediation Reasoning Framework")
        framework_section = HEALTH_REPORTER_PROMPT[framework_start:]
        assert "manual" in framework_section.lower()
        assert "no management annotations" in framework_section.lower()

    def test_reporter_warns_against_kubectl_edit_in_production(self) -> None:
        """Reporter must warn against kubectl edit in production."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "NEVER recommend `kubectl edit` in production" in HEALTH_REPORTER_PROMPT

    def test_reporter_separates_immediate_mitigation_from_permanent_fix(self) -> None:
        """Reporter must separate immediate mitigation from permanent fix."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Immediate mitigation" in HEALTH_REPORTER_PROMPT or \
            "**Immediate mitigation**" in HEALTH_REPORTER_PROMPT
        assert "Permanent fix" in HEALTH_REPORTER_PROMPT or \
            "**Permanent fix**" in HEALTH_REPORTER_PROMPT

    def test_reporter_warns_against_gitops_drift(self) -> None:
        """Reporter must warn that kubectl apply creates GitOps drift."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "drift" in HEALTH_REPORTER_PROMPT.lower()

    def test_reporter_mentions_change_source_identification(self) -> None:
        """Reporter must have Step 1: Identify the Change Source."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Identify the Change Source" in HEALTH_REPORTER_PROMPT

    def test_reporter_mentions_root_process_reasoning(self) -> None:
        """Reporter must have Step 2: Reason About Root Process."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Reason About Root Process" in HEALTH_REPORTER_PROMPT

    def test_reporter_mentions_source_of_truth(self) -> None:
        """Reporter must emphasize source of truth (Git repo, Helm chart, operator CR)."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "source of truth" in HEALTH_REPORTER_PROMPT.lower()

    def test_reporter_provides_both_scenarios_when_unknown(self) -> None:
        """Reporter must provide actions for BOTH scenarios when management method is unknown."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "BOTH scenarios" in HEALTH_REPORTER_PROMPT

    def test_reporter_mentions_gitops_annotations(self) -> None:
        """Reporter must list specific GitOps annotation prefixes."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "argocd.argoproj.io/" in HEALTH_REPORTER_PROMPT
        assert "fluxcd.io/" in HEALTH_REPORTER_PROMPT
        assert "kustomize.toolkit.fluxcd.io/" in HEALTH_REPORTER_PROMPT

    def test_reporter_mentions_helm_labels(self) -> None:
        """Reporter must list Helm management labels."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "app.kubernetes.io/managed-by" in HEALTH_REPORTER_PROMPT
        assert "helm.sh/chart" in HEALTH_REPORTER_PROMPT


class TestGathererManagementAnnotationDetection:
    """Validate that the gatherer prompt instructs collecting management annotations
    when inspecting deployment YAML.

    Causa 3: The gatherer must report management indicators (ArgoCD, Flux, Helm,
    OwnerReferences) so the reporter can reason about the correct remediation path.
    """

    def test_gatherer_mentions_management_annotations(self) -> None:
        """Gatherer must instruct looking for management annotations."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "management annotations" in HEALTH_GATHERER_PROMPT.lower() or \
            "management indicators" in HEALTH_GATHERER_PROMPT.lower()

    def test_gatherer_mentions_argocd_annotations(self) -> None:
        """Gatherer must mention ArgoCD annotations."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "ArgoCD" in HEALTH_GATHERER_PROMPT or "argocd" in HEALTH_GATHERER_PROMPT

    def test_gatherer_mentions_flux_annotations(self) -> None:
        """Gatherer must mention Flux annotations."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "Flux" in HEALTH_GATHERER_PROMPT or "fluxcd" in HEALTH_GATHERER_PROMPT

    def test_gatherer_mentions_helm_annotations(self) -> None:
        """Gatherer must mention Helm management annotations/labels."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "Helm" in HEALTH_GATHERER_PROMPT
        assert "app.kubernetes.io/managed-by" in HEALTH_GATHERER_PROMPT

    def test_gatherer_mentions_owner_references(self) -> None:
        """Gatherer must mention OwnerReferences for operator detection."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "ownerReferences" in HEALTH_GATHERER_PROMPT

    def test_gatherer_mentions_webhook_injection_annotations(self) -> None:
        """Gatherer must mention webhook injection annotations in template metadata."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "webhook injection annotations" in HEALTH_GATHERER_PROMPT.lower() or \
            ".spec.template.metadata.annotations" in HEALTH_GATHERER_PROMPT

    def test_gatherer_reports_indicators_for_reporter(self) -> None:
        """Gatherer must explicitly state that management indicators are for the reporter."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "reporter" in HEALTH_GATHERER_PROMPT.lower()
        assert "remediation" in HEALTH_GATHERER_PROMPT.lower()


class TestAnalyzerManagementContextDetection:
    """Validate that the analyzer prompt contains the Management Context Detection
    section for classifying how each affected resource is managed.

    Causa 3: The analyzer must classify management method (GitOps, Helm, Operator, Manual)
    so findings carry this metadata for the reporter to use in remediation reasoning.
    """

    def test_analyzer_has_management_context_detection_section(self) -> None:
        """Analyzer must contain a Management Context Detection section."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "Management Context Detection" in HEALTH_ANALYZER_PROMPT

    def test_analyzer_classifies_gitops_managed(self) -> None:
        """Analyzer must classify GitOps-managed resources."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        mgmt_start = HEALTH_ANALYZER_PROMPT.find("Management Context Detection")
        mgmt_section = HEALTH_ANALYZER_PROMPT[mgmt_start:mgmt_start + 600]
        assert "GitOps-managed" in mgmt_section
        assert "ArgoCD" in mgmt_section or "Flux" in mgmt_section

    def test_analyzer_classifies_helm_managed(self) -> None:
        """Analyzer must classify Helm-managed resources."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        mgmt_start = HEALTH_ANALYZER_PROMPT.find("Management Context Detection")
        mgmt_section = HEALTH_ANALYZER_PROMPT[mgmt_start:mgmt_start + 600]
        assert "Helm-managed" in mgmt_section
        assert "helm upgrade" in mgmt_section

    def test_analyzer_classifies_operator_managed(self) -> None:
        """Analyzer must classify Operator-managed resources."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        mgmt_start = HEALTH_ANALYZER_PROMPT.find("Management Context Detection")
        mgmt_section = HEALTH_ANALYZER_PROMPT[mgmt_start:mgmt_start + 600]
        assert "Operator-managed" in mgmt_section
        assert "OwnerReferences" in mgmt_section

    def test_analyzer_classifies_manual(self) -> None:
        """Analyzer must classify manually-managed resources."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        mgmt_start = HEALTH_ANALYZER_PROMPT.find("Management Context Detection")
        mgmt_section = HEALTH_ANALYZER_PROMPT[mgmt_start:mgmt_start + 600]
        assert "Manual" in mgmt_section

    def test_analyzer_includes_managed_by_in_findings(self) -> None:
        """Analyzer must include 'Managed by' field in finding metadata."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "**Managed by**" in HEALTH_ANALYZER_PROMPT


# ── Phase 3: Parallel Sub-Gatherer Config ────────────────────────────────────


class TestParallelAgentsConfig:
    """Validate the 7-agent parallel_sequential pipeline (Phase 3).

    Tests ``get_parallel_agents_config()`` which replaces the monolithic
    ``health_gatherer`` with 4 focused sub-gatherers that run concurrently,
    followed by the unchanged sequential tail (analyzer → verifier → reporter).

    These tests are intentionally separate from ``TestServiceHealthSkillAgentsConfig``
    to avoid breaking the existing sequential-pipeline assertions.
    """

    def _get_agents(self) -> list:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        return ServiceHealthSkill().get_parallel_agents_config()

    def test_has_seven_agents(self) -> None:
        """Parallel config must have 7 agents total (4 gatherers + 3 sequential)."""
        agents = self._get_agents()
        assert len(agents) == 7

    def test_four_parallel_gatherers(self) -> None:
        """Exactly 4 agents must carry parallel_group='gather'."""
        agents = self._get_agents()
        parallel = [a for a in agents if a.get("parallel_group") == "gather"]
        assert len(parallel) == 4

    def test_three_sequential_agents(self) -> None:
        """Exactly 3 agents must have no parallel_group (sequential tail)."""
        agents = self._get_agents()
        sequential = [a for a in agents if "parallel_group" not in a]
        assert len(sequential) == 3

    def test_parallel_gatherer_names(self) -> None:
        """All 4 gatherers must have names ending with '_gatherer'."""
        agents = self._get_agents()
        parallel = [a for a in agents if a.get("parallel_group") == "gather"]
        for agent in parallel:
            assert agent["name"].endswith("_gatherer"), (
                f"Parallel agent '{agent['name']}' must end with '_gatherer'"
            )

    def test_exact_gatherer_names(self) -> None:
        """Gatherer names must be the 4 expected sub-gatherer names."""
        agents = self._get_agents()
        parallel_names = {a["name"] for a in agents if a.get("parallel_group") == "gather"}
        assert parallel_names == {
            "node_gatherer",
            "workload_gatherer",
            "event_gatherer",
            "logging_gatherer",
        }

    def test_sequential_tail_names(self) -> None:
        """Sequential tail must be analyzer → verifier → reporter in order."""
        agents = self._get_agents()
        sequential = [a for a in agents if "parallel_group" not in a]
        names = [a["name"] for a in sequential]
        assert names == ["health_analyzer", "health_verifier", "health_reporter"]

    def test_all_gatherers_use_pro_model(self) -> None:
        """All 4 parallel gatherers must use gemini-2.5-pro for quality."""
        agents = self._get_agents()
        parallel = [a for a in agents if a.get("parallel_group") == "gather"]
        for agent in parallel:
            assert agent["model"] == "gemini-2.5-pro", (
                f"Gatherer '{agent['name']}' uses '{agent['model']}', expected 'gemini-2.5-pro'"
            )

    def test_all_gatherers_require_tools(self) -> None:
        """All 4 parallel gatherers must have requires_tools=True."""
        agents = self._get_agents()
        parallel = [a for a in agents if a.get("parallel_group") == "gather"]
        for agent in parallel:
            assert agent.get("requires_tools") is True, (
                f"Gatherer '{agent['name']}' must have requires_tools=True"
            )

    def test_gatherer_max_iterations_within_range(self) -> None:
        """All gatherers must have max_iterations between 8 and 12 (inclusive)."""
        agents = self._get_agents()
        parallel = [a for a in agents if a.get("parallel_group") == "gather"]
        for agent in parallel:
            iters = agent.get("max_iterations", 0)
            assert 8 <= iters <= 12, (
                f"Gatherer '{agent['name']}' has max_iterations={iters}, "
                "expected 8–12"
            )

    def test_gatherer_temperature_is_zero(self) -> None:
        """All gatherers must use temperature=0.0 for deterministic tool use."""
        agents = self._get_agents()
        parallel = [a for a in agents if a.get("parallel_group") == "gather"]
        for agent in parallel:
            assert agent.get("temperature") == 0.0, (
                f"Gatherer '{agent['name']}' must have temperature=0.0"
            )

    def test_all_gatherers_have_system_instruction(self) -> None:
        """All 4 gatherers must have a non-empty system_instruction string."""
        agents = self._get_agents()
        parallel = [a for a in agents if a.get("parallel_group") == "gather"]
        for agent in parallel:
            assert "system_instruction" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 100, (
                f"Gatherer '{agent['name']}' system_instruction too short"
            )

    def test_all_gatherer_prompts_contain_anti_injection_rule(self) -> None:
        """Every gatherer prompt must embed ANTI_INJECTION_RULE for security."""
        from vaig.core.prompt_defense import ANTI_INJECTION_RULE

        agents = self._get_agents()
        parallel = [a for a in agents if a.get("parallel_group") == "gather"]
        # Check a distinctive substring of ANTI_INJECTION_RULE
        fragment = "EXTERNAL, UNTRUSTED sources"
        for agent in parallel:
            assert fragment in agent["system_instruction"], (
                f"Gatherer '{agent['name']}' prompt is missing ANTI_INJECTION_RULE. "
                f"Expected fragment: '{fragment}'"
            )
        assert fragment in ANTI_INJECTION_RULE  # Sanity-check the fragment itself

    def test_node_gatherer_covers_node_scope(self) -> None:
        """node_gatherer prompt must reference node-related tools."""
        agents = self._get_agents()
        node = next(a for a in agents if a["name"] == "node_gatherer")
        prompt = node["system_instruction"]
        assert "get_node_conditions" in prompt
        assert "kubectl_get" in prompt
        assert "Cluster Overview" in prompt

    def test_workload_gatherer_covers_pods_and_deployments(self) -> None:
        """workload_gatherer prompt must cover pods, deployments, services, HPA."""
        agents = self._get_agents()
        workload = next(a for a in agents if a["name"] == "workload_gatherer")
        prompt = workload["system_instruction"]
        assert "pod" in prompt.lower()
        assert "deployment" in prompt.lower()
        assert "hpa" in prompt.lower() or "HPA" in prompt
        assert "Service Status" in prompt

    def test_event_gatherer_covers_events_and_infrastructure(self) -> None:
        """event_gatherer prompt must cover K8s events, networking, and storage."""
        agents = self._get_agents()
        event = next(a for a in agents if a["name"] == "event_gatherer")
        prompt = event["system_instruction"]
        assert "get_events" in prompt
        assert "pvc" in prompt.lower() or "PVC" in prompt
        assert "Events Timeline" in prompt
        assert "Investigation Checklist" in prompt

    def test_logging_gatherer_covers_cloud_logging(self) -> None:
        """logging_gatherer prompt must cover Cloud Logging queries (7a, 7b)."""
        agents = self._get_agents()
        logging_agent = next(a for a in agents if a["name"] == "logging_gatherer")
        prompt = logging_agent["system_instruction"]
        assert "gcloud_logging_query" in prompt
        assert 'severity>=ERROR AND resource.type="k8s_container"' in prompt
        assert 'severity>=WARNING AND resource.type="k8s_pod"' in prompt
        assert "Cloud Logging Findings" in prompt
        assert "MANDATORY" in prompt

    def test_logging_gatherer_prompt_states_mandatory(self) -> None:
        """logging_gatherer prompt must explicitly call Cloud Logging MANDATORY."""
        agents = self._get_agents()
        logging_agent = next(a for a in agents if a["name"] == "logging_gatherer")
        prompt = logging_agent["system_instruction"]
        assert "Cloud Logging Findings sections are NOT optional" in prompt

    def test_sequential_tail_analyzer_unchanged(self) -> None:
        """health_analyzer config must match the sequential pipeline config."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        parallel_agents = skill.get_parallel_agents_config()
        sequential_agents = skill.get_sequential_agents_config()

        p_analyzer = next(a for a in parallel_agents if a["name"] == "health_analyzer")
        s_analyzer = next(a for a in sequential_agents if a["name"] == "health_analyzer")

        assert p_analyzer["model"] == s_analyzer["model"]
        assert p_analyzer["system_instruction"] == s_analyzer["system_instruction"]

    def test_sequential_tail_reporter_has_response_schema(self) -> None:
        """health_reporter in parallel config must have response_schema for JSON output."""
        agents = self._get_agents()
        reporter = next(a for a in agents if a["name"] == "health_reporter")
        assert "response_schema" in reporter
        assert reporter.get("response_mime_type") == "application/json"

    def test_verifier_in_sequential_tail_has_max_iterations(self) -> None:
        """health_verifier in parallel config must retain max_iterations=15."""
        agents = self._get_agents()
        verifier = next(a for a in agents if a["name"] == "health_verifier")
        assert verifier["max_iterations"] == 15
        assert verifier.get("requires_tools") is True


class TestParallelSubGathererPromptBuilders:
    """Unit tests for the 4 standalone prompt builder functions."""

    def test_build_node_gatherer_prompt_returns_string(self) -> None:
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        result = build_node_gatherer_prompt()
        assert isinstance(result, str)
        assert len(result) > 200

    def test_build_workload_gatherer_prompt_returns_string(self) -> None:
        from vaig.skills.service_health.prompts import build_workload_gatherer_prompt

        result = build_workload_gatherer_prompt()
        assert isinstance(result, str)
        assert len(result) > 200

    def test_build_event_gatherer_prompt_returns_string(self) -> None:
        from vaig.skills.service_health.prompts import build_event_gatherer_prompt

        result = build_event_gatherer_prompt()
        assert isinstance(result, str)
        assert len(result) > 200

    def test_build_logging_gatherer_prompt_returns_string(self) -> None:
        from vaig.skills.service_health.prompts import build_logging_gatherer_prompt

        result = build_logging_gatherer_prompt()
        assert isinstance(result, str)
        assert len(result) > 200

    def test_all_builders_include_anti_injection_rule(self) -> None:
        """All 4 builders must include the security anti-injection rule."""
        from vaig.skills.service_health.prompts import (
            build_event_gatherer_prompt,
            build_logging_gatherer_prompt,
            build_node_gatherer_prompt,
            build_workload_gatherer_prompt,
        )

        fragment = "EXTERNAL, UNTRUSTED sources"
        for fn in [
            build_node_gatherer_prompt,
            build_workload_gatherer_prompt,
            build_event_gatherer_prompt,
            build_logging_gatherer_prompt,
        ]:
            result = fn()
            assert fragment in result, (
                f"{fn.__name__}() is missing ANTI_INJECTION_RULE fragment '{fragment}'"
            )

    def test_node_gatherer_prompt_no_pod_name_parameter(self) -> None:
        """node_gatherer prompt must not use 'pod_name' — correct param is 'pod'."""
        import re

        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        calls = re.findall(r"kubectl_logs\([^)]+\)", build_node_gatherer_prompt())
        for call in calls:
            assert "pod_name" not in call

    def test_workload_gatherer_prompt_no_pod_name_parameter(self) -> None:
        """workload_gatherer prompt must use 'pod' not 'pod_name' for kubectl_logs."""
        import re

        from vaig.skills.service_health.prompts import build_workload_gatherer_prompt

        calls = re.findall(r"kubectl_logs\([^)]+\)", build_workload_gatherer_prompt())
        for call in calls:
            assert "pod_name" not in call, (
                f"workload_gatherer uses 'pod_name' in kubectl_logs: {call}"
            )


class TestBuildReporterPromptConditional:
    """P2 — Conditional Remediation Framework: management_context parameter."""

    def test_no_context_includes_all_sections(self) -> None:
        """Default (None) includes all remediation sections — backward compat."""
        from vaig.skills.service_health.prompts import (
            _REMEDIATION_CORE_SECTION,
            _REMEDIATION_GITOPS_SECTION,
            _REMEDIATION_HELM_SECTION,
            _REMEDIATION_MANUAL_SECTION,
            build_reporter_prompt,
        )

        prompt = build_reporter_prompt()
        assert _REMEDIATION_CORE_SECTION in prompt
        assert _REMEDIATION_HELM_SECTION in prompt
        assert _REMEDIATION_GITOPS_SECTION in prompt
        assert _REMEDIATION_MANUAL_SECTION in prompt

    def test_helm_context_includes_helm_excludes_gitops(self) -> None:
        """helm context → Helm section included, GitOps section excluded."""
        from vaig.skills.service_health.prompts import (
            _REMEDIATION_GITOPS_SECTION,
            _REMEDIATION_HELM_SECTION,
            build_reporter_prompt,
        )

        prompt = build_reporter_prompt(management_context="helm")
        assert _REMEDIATION_HELM_SECTION in prompt
        assert _REMEDIATION_GITOPS_SECTION not in prompt

    def test_argocd_context_includes_gitops_excludes_helm(self) -> None:
        """argocd context → GitOps section included, Helm section excluded."""
        from vaig.skills.service_health.prompts import (
            _REMEDIATION_GITOPS_SECTION,
            _REMEDIATION_HELM_SECTION,
            build_reporter_prompt,
        )

        prompt = build_reporter_prompt(management_context="argocd")
        assert _REMEDIATION_GITOPS_SECTION in prompt
        assert _REMEDIATION_HELM_SECTION not in prompt

    def test_gitops_context_includes_gitops_excludes_helm(self) -> None:
        """gitops context → GitOps section included, Helm section excluded."""
        from vaig.skills.service_health.prompts import (
            _REMEDIATION_GITOPS_SECTION,
            _REMEDIATION_HELM_SECTION,
            build_reporter_prompt,
        )

        prompt = build_reporter_prompt(management_context="gitops")
        assert _REMEDIATION_GITOPS_SECTION in prompt
        assert _REMEDIATION_HELM_SECTION not in prompt

    def test_manual_section_always_included(self) -> None:
        """_REMEDIATION_MANUAL_SECTION is always present regardless of context."""
        from vaig.skills.service_health.prompts import (
            _REMEDIATION_MANUAL_SECTION,
            build_reporter_prompt,
        )

        for ctx in [None, "helm", "argocd", "gitops", "manual"]:
            prompt = build_reporter_prompt(management_context=ctx)
            assert _REMEDIATION_MANUAL_SECTION in prompt, (
                f"_REMEDIATION_MANUAL_SECTION missing for management_context={ctx!r}"
            )

    def test_core_section_always_included(self) -> None:
        """_REMEDIATION_CORE_SECTION is always present regardless of context."""
        from vaig.skills.service_health.prompts import (
            _REMEDIATION_CORE_SECTION,
            build_reporter_prompt,
        )

        for ctx in [None, "helm", "argocd", "gitops"]:
            prompt = build_reporter_prompt(management_context=ctx)
            assert _REMEDIATION_CORE_SECTION in prompt, (
                f"_REMEDIATION_CORE_SECTION missing for management_context={ctx!r}"
            )

    def test_compound_context_helm_argocd_includes_both_sections(self) -> None:
        """helm+argocd compound context → both GitOps and Helm sections included."""
        from vaig.skills.service_health.prompts import (
            _REMEDIATION_GITOPS_SECTION,
            _REMEDIATION_HELM_SECTION,
            build_reporter_prompt,
        )

        prompt = build_reporter_prompt(management_context="helm+argocd")
        assert _REMEDIATION_GITOPS_SECTION in prompt
        assert _REMEDIATION_HELM_SECTION in prompt

    def test_health_reporter_prompt_constant_backward_compat(self) -> None:
        """HEALTH_REPORTER_PROMPT module constant must still equal no-args call."""
        from vaig.skills.service_health.prompts import (
            HEALTH_REPORTER_PROMPT,
            build_reporter_prompt,
        )

        assert build_reporter_prompt() == HEALTH_REPORTER_PROMPT


class TestSystemInstructionSplit:
    """P3 — SYSTEM_INSTRUCTION split into universal vs analysis-specific."""

    def test_system_instruction_gatherer_exported(self) -> None:
        """SYSTEM_INSTRUCTION_GATHERER must be a non-empty string export."""
        from vaig.skills.service_health.prompts import SYSTEM_INSTRUCTION_GATHERER

        assert isinstance(SYSTEM_INSTRUCTION_GATHERER, str)
        assert len(SYSTEM_INSTRUCTION_GATHERER) > 0

    def test_gatherer_contains_anti_hallucination(self) -> None:
        """SYSTEM_INSTRUCTION_GATHERER must contain Anti-Hallucination rules."""
        from vaig.skills.service_health.prompts import SYSTEM_INSTRUCTION_GATHERER

        assert "NEVER invent" in SYSTEM_INSTRUCTION_GATHERER

    def test_gatherer_does_not_contain_assessment_framework(self) -> None:
        """SYSTEM_INSTRUCTION_GATHERER must NOT contain Assessment Framework."""
        from vaig.skills.service_health.prompts import SYSTEM_INSTRUCTION_GATHERER

        assert "Assessment Framework" not in SYSTEM_INSTRUCTION_GATHERER

    def test_gatherer_does_not_contain_causal_reasoning(self) -> None:
        """SYSTEM_INSTRUCTION_GATHERER must NOT contain Causal Reasoning Principle."""
        from vaig.skills.service_health.prompts import SYSTEM_INSTRUCTION_GATHERER

        assert "Causal Reasoning" not in SYSTEM_INSTRUCTION_GATHERER

    def test_full_system_instruction_contains_all_original_content(self) -> None:
        """SYSTEM_INSTRUCTION must still contain all original content after split."""
        from vaig.skills.service_health.prompts import SYSTEM_INSTRUCTION

        assert "NEVER invent" in SYSTEM_INSTRUCTION
        assert "Assessment Framework" in SYSTEM_INSTRUCTION
        assert "Causal Reasoning" in SYSTEM_INSTRUCTION

    def test_system_instruction_equals_universal_plus_analysis(self) -> None:
        """SYSTEM_INSTRUCTION must equal universal + analysis concatenation."""
        from vaig.skills.service_health.prompts import (
            _SYSTEM_INSTRUCTION_ANALYSIS,
            _SYSTEM_INSTRUCTION_UNIVERSAL,
            SYSTEM_INSTRUCTION,
        )

        assert SYSTEM_INSTRUCTION == _SYSTEM_INSTRUCTION_UNIVERSAL + _SYSTEM_INSTRUCTION_ANALYSIS


# ═══════════════════════════════════════════════════════════════
# Service Status table format — language enforcement
# ═══════════════════════════════════════════════════════════════


class TestServiceStatusTableFormat:
    """Validate the Service Status table format in gatherer prompts.

    Prevents regression of the 'ninguno' bug where Spanish-locale models
    translated [yes/no] to non-English values because the placeholder was
    ambiguous lowercase.
    """

    def test_sequential_gatherer_uses_yes_no_na_placeholder(self) -> None:
        """Sequential gatherer Service Status table must use [Yes/No/N/A], not [yes/no]."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "[Yes/No/N/A]" in HEALTH_GATHERER_PROMPT
        assert "[yes/no]" not in HEALTH_GATHERER_PROMPT

    def test_sequential_gatherer_enforces_english_values(self) -> None:
        """Sequential gatherer must explicitly require English in the Service Status table."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        assert "MUST be in English" in HEALTH_GATHERER_PROMPT

    def test_sequential_gatherer_lists_example_translations_to_avoid(self) -> None:
        """Prompt must name the specific translated values to reject."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        # Ensure common Spanish translations are explicitly blocked
        assert "sí" in HEALTH_GATHERER_PROMPT or "ninguno" in HEALTH_GATHERER_PROMPT

    def test_build_gatherer_prompt_uses_yes_no_na_placeholder(self) -> None:
        """build_gatherer_prompt() output must also contain [Yes/No/N/A]."""
        from vaig.skills.service_health.prompts import build_gatherer_prompt

        prompt = build_gatherer_prompt(helm_enabled=True, argocd_enabled=True)
        assert "[Yes/No/N/A]" in prompt
        assert "[yes/no]" not in prompt

    def test_parallel_workload_gatherer_has_no_ambiguous_placeholder(self) -> None:
        """Parallel workload gatherer must not contain ambiguous [yes/no] placeholder."""
        from vaig.skills.service_health.prompts import build_workload_gatherer_prompt

        prompt = build_workload_gatherer_prompt()
        assert "[yes/no]" not in prompt


# ═══════════════════════════════════════════════════════════════
# Service Status column preservation — analyzer and reporter
# ═══════════════════════════════════════════════════════════════


class TestServiceStatusColumnPreservation:
    """Validate that the analyzer prompt mandates full 8-column Service Status
    Summary and that the reporter prompt directs it to the workload_gatherer data.

    Prevents regression where the analyzer strips the rich workload_gatherer
    columns down to a 3-column summary, causing downstream data loss.
    """

    def test_analyzer_service_status_summary_has_eight_columns(self) -> None:
        """Analyzer prompt must mandate the full 8-column Service Status Summary table."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        # All eight required column headers must be present
        required_columns = [
            "Service/Deployment",
            "Namespace",
            "Status",
            "Ready",
            "Restarts",
            "CPU Usage",
            "Memory Usage",
            "Primary Issue",
        ]
        for col in required_columns:
            assert col in HEALTH_ANALYZER_PROMPT, (
                f"Analyzer Service Status Summary is missing column: '{col}'"
            )

    def test_analyzer_service_status_preserves_workload_gatherer_columns(self) -> None:
        """Analyzer prompt must instruct the model to PRESERVE gatherer columns, not reduce them."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "PRESERVE all columns from the workload_gatherer" in HEALTH_ANALYZER_PROMPT
        assert "Do NOT reduce or summarize the table" in HEALTH_ANALYZER_PROMPT

    def test_reporter_directs_to_workload_gatherer_for_service_statuses(self) -> None:
        """Reporter prompt must explicitly tell the model to use workload_gatherer data for service_statuses."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        # The reporter must reference the workload_gatherer section as the primary data source
        assert "workload_gatherer" in HEALTH_REPORTER_PROMPT
        assert "--- workload_gatherer ---" in HEALTH_REPORTER_PROMPT

    def test_reporter_warns_analyzer_summary_is_insufficient(self) -> None:
        """Reporter prompt must explicitly state the analyzer 3-col summary is NOT sufficient."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "NOT sufficient to populate all ServiceStatus fields" in HEALTH_REPORTER_PROMPT


class TestDatadogAPMTagResolution:
    """Validate that the Datadog APM step uses proper tag resolution and prohibits
    calling get_datadog_apm_services() without a service_name.

    Prevents regression where _DATADOG_API_STEP directed the LLM to call
    get_datadog_apm_services() without arguments when no service was found,
    causing noisy/misleading APM lookups.
    """

    def test_apm_step_references_unified_service_tagging_label(self) -> None:
        """_DATADOG_API_STEP must instruct the LLM to look for the
        tags.datadoghq.com/service label as the primary service identity source."""
        from vaig.skills.service_health.prompts import _DATADOG_API_STEP

        assert "tags.datadoghq.com/service" in _DATADOG_API_STEP, (
            "_DATADOG_API_STEP must reference 'tags.datadoghq.com/service' "
            "as Tier 1 (Unified Service Tagging) label for APM lookup."
        )

    def test_apm_step_references_env_unified_service_tagging_label(self) -> None:
        """_DATADOG_API_STEP must instruct the LLM to look for
        tags.datadoghq.com/env as the primary environment source."""
        from vaig.skills.service_health.prompts import _DATADOG_API_STEP

        assert "tags.datadoghq.com/env" in _DATADOG_API_STEP, (
            "_DATADOG_API_STEP must reference 'tags.datadoghq.com/env' "
            "as Tier 1 label for APM env parameter."
        )

    def test_apm_step_instructs_always_attempt_calls(self) -> None:
        """_DATADOG_API_STEP must instruct the LLM to ALWAYS attempt
        get_datadog_apm_services() calls — even when service_name cannot be
        resolved — because the tool handles empty service_name gracefully."""
        from vaig.skills.service_health.prompts import _DATADOG_API_STEP

        assert "ALWAYS call" in _DATADOG_API_STEP, (
            "_DATADOG_API_STEP must contain 'ALWAYS call' instruction so the LLM "
            "always attempts get_datadog_apm_services() instead of skipping it."
        )
        assert "handles empty service_name gracefully" in _DATADOG_API_STEP, (
            "_DATADOG_API_STEP must state that the tool handles empty service_name "
            "gracefully, which is why the call should always be attempted."
        )

    def test_apm_step_does_not_contain_no_arg_example(self) -> None:
        """_DATADOG_API_STEP must NOT contain the bad example
        get_datadog_apm_services() without any arguments."""
        from vaig.skills.service_health.prompts import _DATADOG_API_STEP

        # The bad example was: "get_datadog_apm_services()" — a bare no-arg call
        # It must not exist; the LLM should always pass service_name or skip entirely.
        assert "get_datadog_apm_services()" not in _DATADOG_API_STEP, (
            "_DATADOG_API_STEP must NOT contain 'get_datadog_apm_services()' "
            "(no-argument call) — it sends the LLM on unscoped APM queries."
        )

    def test_workload_gatherer_prompt_apm_step_uses_unified_tagging(self) -> None:
        """build_workload_gatherer_prompt with datadog enabled must include the
        unified service tagging instructions (shared _DATADOG_API_STEP constant)."""
        from vaig.skills.service_health.prompts import build_workload_gatherer_prompt

        prompt = build_workload_gatherer_prompt(namespace="default", datadog_api_enabled=True)

        assert "tags.datadoghq.com/service" in prompt, (
            "workload_gatherer prompt with datadog enabled must reference "
            "tags.datadoghq.com/service for APM tag resolution."
        )
        assert "get_datadog_apm_services()" not in prompt, (
            "workload_gatherer prompt must NOT contain a no-arg "
            "get_datadog_apm_services() example."
        )

    def test_gatherer_prompt_apm_step_uses_unified_tagging(self) -> None:
        """build_gatherer_prompt with datadog enabled must include the
        unified service tagging instructions (shared _DATADOG_API_STEP constant)."""
        from vaig.skills.service_health.prompts import build_gatherer_prompt

        prompt = build_gatherer_prompt(datadog_api_enabled=True)

        assert "tags.datadoghq.com/service" in prompt, (
            "Sequential gatherer prompt with datadog enabled must reference "
            "tags.datadoghq.com/service for APM tag resolution."
        )
        assert "get_datadog_apm_services()" not in prompt, (
            "Sequential gatherer prompt must NOT contain a no-arg "
            "get_datadog_apm_services() example."
        )

    def test_workload_gatherer_prompt_has_k8s_priority_hierarchy(self) -> None:
        """build_workload_gatherer_prompt must include the K8s-first priority hierarchy.

        Kubernetes data is the absolute source of truth — Datadog results must NEVER
        override K8s deployment status. The priority hierarchy must be explicit.
        """
        from vaig.skills.service_health.prompts import build_workload_gatherer_prompt

        prompt = build_workload_gatherer_prompt(namespace="default", datadog_api_enabled=True)

        assert "PRIORITY HIERARCHY" in prompt, (
            "workload_gatherer prompt must contain PRIORITY HIERARCHY section "
            "to ensure the LLM never overrides K8s truth with Datadog results."
        )
        assert "ABSOLUTE source of truth" in prompt, (
            "Priority hierarchy must state that Kubernetes data is the ABSOLUTE "
            "source of truth for deployment status."
        )
        assert "monitoring not configured" in prompt or "not monitored" in prompt, (
            "Prompt must clarify that empty Datadog results mean 'monitoring not "
            "configured', NOT 'service not deployed'."
        )

    def test_gatherer_prompt_has_k8s_priority_hierarchy(self) -> None:
        """build_gatherer_prompt (sequential) must also include the K8s-first priority hierarchy."""
        from vaig.skills.service_health.prompts import build_gatherer_prompt

        prompt = build_gatherer_prompt(datadog_api_enabled=True)

        assert "PRIORITY HIERARCHY" in prompt, (
            "Sequential gatherer prompt must contain PRIORITY HIERARCHY section."
        )
        assert "ABSOLUTE source of truth" in prompt, (
            "Priority hierarchy must state that Kubernetes data is the ABSOLUTE "
            "source of truth for deployment status."
        )

    def test_workload_gatherer_datadog_step_before_output_format(self) -> None:
        """In build_workload_gatherer_prompt, the Datadog step must appear BEFORE
        the MANDATORY OUTPUT FORMAT section.

        Root cause of the regression: the Datadog step was placed inside the output
        format block, causing the LLM to treat it as a template rather than an action.
        """
        from vaig.skills.service_health.prompts import build_workload_gatherer_prompt

        prompt = build_workload_gatherer_prompt(namespace="default", datadog_api_enabled=True)

        datadog_pos = prompt.find("Step 12 — Datadog API Correlation")
        output_format_pos = prompt.find("MANDATORY OUTPUT FORMAT")

        assert datadog_pos != -1, (
            "Datadog step must be present in workload_gatherer_prompt when "
            "datadog_api_enabled=True."
        )
        assert output_format_pos != -1, (
            "MANDATORY OUTPUT FORMAT section must exist in workload_gatherer_prompt."
        )
        assert datadog_pos < output_format_pos, (
            "Datadog step (Step 12) must appear BEFORE the MANDATORY OUTPUT FORMAT "
            "section. If it's inside the output format block, the LLM treats it as "
            "a template rather than an action to execute — that's the root cause of "
            "Datadog tools never being called."
        )


class TestNamespacePropagation:
    """Regression tests for Bug 1 & 3 — namespace/location/cluster_name propagation.

    Verifies that:
    - get_agents_config() and get_parallel_agents_config() accept and forward
      namespace, location, and cluster_name kwargs.
    - The effective namespace appears in namespace-scoped gatherer prompts (workload,
      event) but NOT necessarily in cluster-scoped gatherers (node_gatherer).
    - get_sequential_agents_config() also respects the namespace override.
    """

    def test_parallel_gatherer_prompts_use_overridden_namespace(self) -> None:
        """When namespace='odin-dev' is passed, namespace-scoped gatherer prompts must
        reference 'odin-dev' and NOT use 'default' as the target namespace.

        node_gatherer is cluster-scoped (nodes, kube-system) so it is excluded.
        """
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_parallel_agents_config(namespace="odin-dev")

        # Only check namespace-scoped gatherers — node_gatherer is cluster-scoped
        namespace_scoped = [
            a for a in agents
            if a.get("parallel_group") == "gather"
            and a.get("name") in ("workload_gatherer", "event_gatherer", "logging_gatherer")
        ]
        assert namespace_scoped, "Expected at least one namespace-scoped gatherer agent"

        for agent in namespace_scoped:
            prompt = agent.get("system_instruction", "")
            assert "odin-dev" in prompt, (
                f"Agent '{agent.get('name')}' prompt does not contain 'odin-dev'. "
                "Namespace override is not being propagated to the prompt builder."
            )

    def test_get_agents_config_accepts_namespace_kwarg(self) -> None:
        """get_agents_config() must accept a namespace kwarg without raising."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        # Should not raise
        agents = skill.get_agents_config(namespace="odin-dev")
        assert isinstance(agents, list)

    def test_get_agents_config_accepts_location_and_cluster_name(self) -> None:
        """get_agents_config() must accept location and cluster_name kwargs."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_agents_config(
            namespace="staging",
            location="us-central1",
            cluster_name="my-cluster",
        )
        assert isinstance(agents, list)

    def test_sequential_agents_config_accepts_namespace_kwarg(self) -> None:
        """get_sequential_agents_config() must accept namespace without raising."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_sequential_agents_config(namespace="production")
        assert isinstance(agents, list)


class TestTargetPlaceholderReplacement:
    """Regression tests for Bug 2 — literal '<target>' placeholder in prompts.

    After the fix, no gatherer prompt should contain the literal string '<target>'
    when a namespace is provided.  The placeholder must be replaced with the
    sanitized namespace value before the prompt is returned.
    """

    def test_workload_gatherer_no_target_placeholder_with_namespace(self) -> None:
        """build_workload_gatherer_prompt with a namespace must not contain '<target>'."""
        from vaig.skills.service_health.prompts import build_workload_gatherer_prompt

        prompt = build_workload_gatherer_prompt(namespace="odin-dev")
        assert "<target>" not in prompt, (
            "build_workload_gatherer_prompt still contains literal '<target>' "
            "after fix — namespace substitution is broken."
        )

    def test_event_gatherer_no_target_placeholder_with_namespace(self) -> None:
        """build_event_gatherer_prompt with a namespace must not contain '<target>'."""
        from vaig.skills.service_health.prompts import build_event_gatherer_prompt

        prompt = build_event_gatherer_prompt(namespace="odin-dev")
        assert "<target>" not in prompt, (
            "build_event_gatherer_prompt still contains literal '<target>' "
            "after fix — namespace substitution is broken."
        )

    def test_workload_gatherer_injects_namespace_value(self) -> None:
        """The actual namespace value must appear in the workload gatherer prompt."""
        from vaig.skills.service_health.prompts import build_workload_gatherer_prompt

        prompt = build_workload_gatherer_prompt(namespace="odin-dev")
        assert "odin-dev" in prompt, (
            "build_workload_gatherer_prompt does not inject the namespace value "
            "into tool call examples."
        )

    def test_event_gatherer_injects_namespace_value(self) -> None:
        """The actual namespace value must appear in the event gatherer prompt."""
        from vaig.skills.service_health.prompts import build_event_gatherer_prompt

        prompt = build_event_gatherer_prompt(namespace="odin-dev")
        assert "odin-dev" in prompt, (
            "build_event_gatherer_prompt does not inject the namespace value "
            "into tool call examples."
        )

    def test_workload_gatherer_default_namespace_no_target_placeholder(self) -> None:
        """Even with no namespace arg, '<target>' must not remain in the prompt."""
        from vaig.skills.service_health.prompts import build_workload_gatherer_prompt

        prompt = build_workload_gatherer_prompt()
        assert "<target>" not in prompt, (
            "build_workload_gatherer_prompt() with no namespace still contains "
            "literal '<target>' — the default substitution is broken."
        )

    def test_event_gatherer_default_namespace_no_target_placeholder(self) -> None:
        """Even with no namespace arg, '<target>' must not remain in the prompt."""
        from vaig.skills.service_health.prompts import build_event_gatherer_prompt

        prompt = build_event_gatherer_prompt()
        assert "<target>" not in prompt, (
            "build_event_gatherer_prompt() with no namespace still contains "
            "literal '<target>' — the default substitution is broken."
        )

    def test_logging_gatherer_default_namespace_no_target_placeholder(self) -> None:
        """Logging gatherer should not contain <target> when using default namespace."""
        from vaig.skills.service_health.prompts import build_logging_gatherer_prompt

        prompt = build_logging_gatherer_prompt(namespace="default")
        assert "<target>" not in prompt
        assert "<target-namespace>" not in prompt
        assert "<target_namespace>" not in prompt

    def test_logging_gatherer_injects_namespace_value(self) -> None:
        """Logging gatherer should contain the actual namespace value."""
        from vaig.skills.service_health.prompts import build_logging_gatherer_prompt

        prompt = build_logging_gatherer_prompt(namespace="prod-ns")
        assert "prod-ns" in prompt
        assert "<target>" not in prompt
        assert "<target-namespace>" not in prompt
        assert "<target_namespace>" not in prompt


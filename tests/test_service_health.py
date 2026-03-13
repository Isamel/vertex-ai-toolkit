"""Tests for ServiceHealthSkill — the first skill with live tool support.

Validates:
- Skill metadata (name, tags, requires_live_tools, supported_phases)
- System instruction is non-empty
- Phase prompts inject context and user_input correctly
- Agent pipeline configuration (4 agents, sequential, requires_tools flags)
- ToolAwareAgent compatibility (system_prompt key on gatherer/verifier)
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
        agents = skill.get_agents_config()
        assert len(agents) == 4

    def test_agent_names(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_agents_config()
        names = [a["name"] for a in agents]
        assert names == ["health_gatherer", "health_analyzer", "health_verifier", "health_reporter"]

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
        reporter = agents[3]
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
        agents = skill.get_agents_config()
        # Gatherer and verifier require tools; analyzer and reporter do not
        requires_tools_flags = [a.get("requires_tools", False) for a in agents]
        assert requires_tools_flags == [True, False, True, False]

    def test_gatherer_and_verifier_require_tools(self) -> None:
        """Exactly two agents in the pipeline require tools."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_agents_config()
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
        agents = skill.get_agents_config()
        verifier = agents[2]
        assert verifier["name"] == "health_verifier"
        assert verifier["requires_tools"] is True

    def test_verifier_agent_system_prompt(self) -> None:
        """ToolAwareAgent.from_config_dict expects 'system_prompt' key."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_agents_config()
        verifier = agents[2]
        assert "system_prompt" in verifier
        assert isinstance(verifier["system_prompt"], str)
        assert len(verifier["system_prompt"]) > 0

    def test_verifier_agent_system_instruction(self) -> None:
        """Verifier provides system_instruction for defensive compatibility."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_agents_config()
        verifier = agents[2]
        assert "system_instruction" in verifier
        assert isinstance(verifier["system_instruction"], str)
        assert len(verifier["system_instruction"]) > 0

    def test_verifier_agent_max_iterations(self) -> None:
        """Verifier must have max_iterations=10 for efficient targeted calls."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_agents_config()
        verifier = agents[2]
        assert verifier["max_iterations"] == 10

    def test_verifier_agent_model(self) -> None:
        """Verifier uses flash model for speed — verification is targeted, not complex."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_agents_config()
        verifier = agents[2]
        assert verifier["model"] == "gemini-2.5-flash"


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
        reporter_prompt = agents[3]["system_instruction"]
        assert "Executive Summary" in reporter_prompt
        assert "Recommended Actions" in reporter_prompt

    def test_reporter_prompt_mentions_remediation_commands(self) -> None:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        agents = skill.get_agents_config()
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
        """Analyzer must explicitly require all fields for every finding."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "all fields" in HEALTH_ANALYZER_PROMPT

    def test_reporter_forbids_unstructured_paragraphs(self) -> None:
        """Reporter must forbid unstructured paragraphs in findings."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "No unstructured paragraphs" in HEALTH_REPORTER_PROMPT

    # ── Timeline Rules ───────────────────────────────────────

    def test_reporter_has_timeline_anti_hallucination(self) -> None:
        """Reporter must prevent fabricated/default timeline events.

        The consistency fix replaced the old 'NEVER fabricate timestamps'
        with deterministic timeline construction rules that prevent defaulting
        to 'no events' when data exists.
        """
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        # Rule 5: NEVER use "no events" as a default
        assert "NEVER use this as a default" in HEALTH_REPORTER_PROMPT
        # Timeline section is MANDATORY
        assert "Timeline Section (MANDATORY)" in HEALTH_REPORTER_PROMPT


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

        assert "Timeline Section (MANDATORY)" in HEALTH_REPORTER_PROMPT

    def test_reporter_timeline_must_build(self) -> None:
        """Reporter must be told to BUILD the timeline, not just include it."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "You MUST build a chronological timeline" in HEALTH_REPORTER_PROMPT

    def test_reporter_timeline_extract_every_event(self) -> None:
        """Rule 1: Extract EVERY event with a timestamp."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Extract EVERY event" in HEALTH_REPORTER_PROMPT

    def test_reporter_timeline_sort_chronologically(self) -> None:
        """Rule 2: Sort events chronologically (oldest first)."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Sort events chronologically" in HEALTH_REPORTER_PROMPT

    def test_reporter_timeline_table_format(self) -> None:
        """Rule 3: Timeline must use table format with Time/Type/Resource/Event."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        # Table header columns
        timeline_section = HEALTH_REPORTER_PROMPT[
            HEALTH_REPORTER_PROMPT.find("Timeline Section"):
        ]
        assert "| Time |" in timeline_section
        assert "| Type |" in timeline_section or "Type" in timeline_section

    def test_reporter_timeline_events_without_timestamps(self) -> None:
        """Rule 4: Events without extractable timestamps shown in order."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "WITHOUT timestamps" in HEALTH_REPORTER_PROMPT

    def test_reporter_timeline_never_default_to_no_events(self) -> None:
        """Rule 5: NEVER use 'no events' as a default — only when upstream says so."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "NEVER use this as a default" in HEALTH_REPORTER_PROMPT

    def test_reporter_timeline_must_appear_in_every_report(self) -> None:
        """Rule 6: Timeline MUST appear in every report."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "MUST appear in every report" in HEALTH_REPORTER_PROMPT


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
        """Evidence must be formatted as code blocks."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "code blocks" in HEALTH_REPORTER_PROMPT

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
        step1_start = HEALTH_GATHERER_PROMPT.find("Step 1")
        step2_start = HEALTH_GATHERER_PROMPT.find("Step 2")
        step1_section = HEALTH_GATHERER_PROMPT[step1_start:step2_start]
        assert "get_node_conditions" in step1_section

    def test_gatherer_step_1_is_mandatory(self) -> None:
        """Step 1 must be marked as MANDATORY."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        step1_start = HEALTH_GATHERER_PROMPT.find("Step 1")
        step2_start = HEALTH_GATHERER_PROMPT.find("Step 2")
        step1_section = HEALTH_GATHERER_PROMPT[step1_start:step2_start]
        assert "MANDATORY" in step1_section

    def test_gatherer_step_1_before_specific_investigations(self) -> None:
        """Node conditions must come BEFORE deployment-specific investigations."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT

        step1_pos = HEALTH_GATHERER_PROMPT.find("Step 1")
        step4_pos = HEALTH_GATHERER_PROMPT.find("Step 4")  # Deep-dive step
        assert step1_pos < step4_pos


class TestPromptConsistencyFix7ReporterClusterOverviewMandatory:
    """Fix 7 (MEDIUM): Reporter Cluster Overview section MUST be mandatory
    with fallback text when data is unavailable.
    """

    def test_reporter_cluster_overview_mandatory(self) -> None:
        """Reporter must have Cluster Overview Section marked as MANDATORY."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Cluster Overview Section (MANDATORY)" in HEALTH_REPORTER_PROMPT

    def test_reporter_cluster_overview_uses_upstream_data(self) -> None:
        """Reporter must use upstream Cluster Overview data when available."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "use it directly" in HEALTH_REPORTER_PROMPT

    def test_reporter_cluster_overview_fallback_text(self) -> None:
        """Reporter must have fallback text when cluster overview data is missing."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert "Cluster overview data was not collected" in HEALTH_REPORTER_PROMPT

    def test_reporter_never_data_not_available_without_explanation(self) -> None:
        """Reporter NEVER writes 'Data not available' without explanation."""
        from vaig.skills.service_health.prompts import HEALTH_REPORTER_PROMPT

        assert 'NEVER write "Data not available" without explanation' in HEALTH_REPORTER_PROMPT


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

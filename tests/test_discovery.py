"""Tests for DiscoverySkill — autonomous cluster scanning and health discovery.

Validates:
- Skill metadata (name, tags, requires_live_tools, supported_phases)
- System instruction is non-empty and contains key context
- Phase prompts inject context and user_input correctly
- Agent pipeline configuration (4 agents, sequential, requires_tools flags)
- Auto-generated discover query building
- Prompt defense constants are present in agent prompts
"""

from __future__ import annotations

from vaig.skills.base import SkillPhase

# ── Metadata tests ────────────────────────────────────────────


class TestDiscoverySkillMetadata:
    """Metadata contract tests."""

    def test_name(self) -> None:
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        meta = skill.get_metadata()
        assert meta.name == "discovery"

    def test_display_name(self) -> None:
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        meta = skill.get_metadata()
        assert meta.display_name == "Cluster Discovery"

    def test_requires_live_tools(self) -> None:
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        meta = skill.get_metadata()
        assert meta.requires_live_tools is True

    def test_supported_phases(self) -> None:
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        meta = skill.get_metadata()
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.EXECUTE in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases

    def test_tags(self) -> None:
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        meta = skill.get_metadata()
        assert "discovery" in meta.tags
        assert "sre" in meta.tags
        assert "live" in meta.tags
        assert "kubernetes" in meta.tags

    def test_recommended_model(self) -> None:
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        meta = skill.get_metadata()
        assert meta.recommended_model == "gemini-2.5-flash"

    def test_version(self) -> None:
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        meta = skill.get_metadata()
        assert meta.version == "1.0.0"


# ── System instruction tests ─────────────────────────────────


class TestDiscoverySkillSystemInstruction:
    """System instruction tests."""

    def test_non_empty(self) -> None:
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        instruction = skill.get_system_instruction()
        assert len(instruction) > 0
        assert isinstance(instruction, str)

    def test_contains_kubernetes_context(self) -> None:
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        instruction = skill.get_system_instruction()
        assert "Kubernetes" in instruction

    def test_contains_anti_injection_rule(self) -> None:
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        instruction = skill.get_system_instruction()
        assert "SECURITY RULE" in instruction

    def test_contains_analysis_approach(self) -> None:
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        instruction = skill.get_system_instruction()
        assert "Inventory" in instruction
        assert "Triage" in instruction
        assert "Investigate" in instruction
        assert "Report" in instruction


# ── Phase prompt tests ────────────────────────────────────────


class TestDiscoverySkillPhasePrompts:
    """Phase prompt template tests."""

    def test_analyze_prompt_injects_context(self) -> None:
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        prompt = skill.get_phase_prompt(
            SkillPhase.ANALYZE,
            context="namespace: production, 5 deployments",
            user_input="Scan production namespace",
        )
        assert "namespace: production, 5 deployments" in prompt
        assert "Scan production namespace" in prompt

    def test_execute_prompt_injects_context(self) -> None:
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        prompt = skill.get_phase_prompt(
            SkillPhase.EXECUTE,
            context="2 failing, 1 degraded",
            user_input="Investigate failures",
        )
        assert "2 failing, 1 degraded" in prompt
        assert "Investigate failures" in prompt

    def test_report_prompt_injects_context(self) -> None:
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        prompt = skill.get_phase_prompt(
            SkillPhase.REPORT,
            context="3 critical, 1 warning",
            user_input="Generate discovery report",
        )
        assert "3 critical, 1 warning" in prompt
        assert "Generate discovery report" in prompt

    def test_unknown_phase_falls_back_to_analyze(self) -> None:
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        prompt = skill.get_phase_prompt(
            SkillPhase.VALIDATE,
            context="some context",
            user_input="some input",
        )
        assert "some context" in prompt
        assert "some input" in prompt


# ── Agent configuration tests ─────────────────────────────────


class TestDiscoverySkillAgentsConfig:
    """Agent pipeline configuration tests."""

    def test_has_four_agents(self) -> None:
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        agents = skill.get_agents_config()
        assert len(agents) == 4

    def test_agent_names(self) -> None:
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        agents = skill.get_agents_config()
        names = [a["name"] for a in agents]
        assert names == [
            "inventory_scanner",
            "triage_classifier",
            "deep_investigator",
            "cluster_reporter",
        ]

    def test_agent_roles(self) -> None:
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        agents = skill.get_agents_config()
        for agent in agents:
            assert "role" in agent
            assert isinstance(agent["role"], str)
            assert len(agent["role"]) > 0

    def test_agent_models(self) -> None:
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        agents = skill.get_agents_config()
        for agent in agents:
            assert agent["model"] == "gemini-2.5-flash"

    def test_inventory_scanner_requires_tools(self) -> None:
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        agents = skill.get_agents_config()
        scanner = agents[0]
        assert scanner["name"] == "inventory_scanner"
        assert scanner["requires_tools"] is True

    def test_triage_classifier_no_tools(self) -> None:
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        agents = skill.get_agents_config()
        triage = agents[1]
        assert triage["name"] == "triage_classifier"
        assert triage["requires_tools"] is False

    def test_deep_investigator_requires_tools(self) -> None:
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        agents = skill.get_agents_config()
        investigator = agents[2]
        assert investigator["name"] == "deep_investigator"
        assert investigator["requires_tools"] is True

    def test_cluster_reporter_no_tools(self) -> None:
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        agents = skill.get_agents_config()
        reporter = agents[3]
        assert reporter["name"] == "cluster_reporter"
        assert reporter["requires_tools"] is False

    def test_all_agents_have_system_instruction(self) -> None:
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        agents = skill.get_agents_config()
        for agent in agents:
            assert "system_instruction" in agent
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 100

    def test_tool_agents_have_max_iterations(self) -> None:
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        agents = skill.get_agents_config()
        # Only agents with requires_tools=True should have max_iterations
        for agent in agents:
            if agent.get("requires_tools"):
                assert "max_iterations" in agent
                assert agent["max_iterations"] > 0

    def test_kwargs_accepted(self) -> None:
        """get_agents_config must accept arbitrary kwargs from orchestrator."""
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        agents = skill.get_agents_config(
            namespace="production",
            location="us-central1",
            cluster_name="my-cluster",
        )
        assert len(agents) == 4


# ── Prompt defense tests ──────────────────────────────────────


class TestDiscoveryPromptDefense:
    """Verify prompt defense constants are present in agent prompts."""

    def test_inventory_scanner_has_anti_injection(self) -> None:
        from vaig.skills.discovery.prompts import INVENTORY_SCANNER_PROMPT

        assert "SECURITY RULE" in INVENTORY_SCANNER_PROMPT

    def test_triage_classifier_has_anti_injection(self) -> None:
        from vaig.skills.discovery.prompts import TRIAGE_CLASSIFIER_PROMPT

        assert "SECURITY RULE" in TRIAGE_CLASSIFIER_PROMPT

    def test_deep_investigator_has_anti_injection(self) -> None:
        from vaig.skills.discovery.prompts import DEEP_INVESTIGATOR_PROMPT

        assert "SECURITY RULE" in DEEP_INVESTIGATOR_PROMPT

    def test_cluster_reporter_has_anti_injection(self) -> None:
        from vaig.skills.discovery.prompts import CLUSTER_REPORTER_PROMPT

        assert "SECURITY RULE" in CLUSTER_REPORTER_PROMPT

    def test_inventory_scanner_has_anti_hallucination(self) -> None:
        from vaig.skills.discovery.prompts import INVENTORY_SCANNER_PROMPT

        assert "ANTI-HALLUCINATION" in INVENTORY_SCANNER_PROMPT

    def test_triage_has_delimiters(self) -> None:
        from vaig.skills.discovery.prompts import TRIAGE_CLASSIFIER_PROMPT

        assert "RAW FINDINGS" in TRIAGE_CLASSIFIER_PROMPT
        assert "END RAW FINDINGS" in TRIAGE_CLASSIFIER_PROMPT

    def test_system_namespaces_in_inventory_prompt(self) -> None:
        from vaig.skills.discovery.prompts import INVENTORY_SCANNER_PROMPT

        assert "kube-system" in INVENTORY_SCANNER_PROMPT
        assert "gke-mcs" in INVENTORY_SCANNER_PROMPT


# ── Auto-generated query tests ────────────────────────────────


class TestDiscoverQueryBuilder:
    """Test the auto-generated query builder."""

    def test_single_namespace_query(self) -> None:
        from vaig.cli.commands.discover import _build_discover_query

        query = _build_discover_query(namespace="production")
        assert "production" in query
        assert "Scan namespace" in query

    def test_all_namespaces_query(self) -> None:
        from vaig.cli.commands.discover import _build_discover_query

        query = _build_discover_query(all_namespaces=True)
        assert "ALL non-system namespaces" in query
        assert "kube-system" in query

    def test_skip_healthy_appended(self) -> None:
        from vaig.cli.commands.discover import _build_discover_query

        query = _build_discover_query(namespace="staging", skip_healthy=True)
        assert "Degraded" in query
        assert "Failing" in query
        assert "count" in query.lower()

    def test_default_namespace_when_none(self) -> None:
        from vaig.cli.commands.discover import _build_discover_query

        query = _build_discover_query(namespace=None)
        assert "default" in query

    def test_all_namespaces_ignores_specific_namespace(self) -> None:
        from vaig.cli.commands.discover import _build_discover_query

        query = _build_discover_query(namespace="production", all_namespaces=True)
        # all_namespaces takes precedence — should NOT mention specific namespace
        assert "ALL non-system namespaces" in query

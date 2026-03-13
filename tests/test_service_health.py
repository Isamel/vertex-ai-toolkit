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

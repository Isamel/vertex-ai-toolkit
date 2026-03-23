"""Integration tests for the RCA skill gatherer agent.

Validates:
- Gatherer agent is at position 0 in the pipeline
- Gatherer has requires_tools=True
- Gatherer system_instruction starts with ANTI_INJECTION_RULE
- Skill metadata has requires_live_tools=True
"""

from __future__ import annotations


class TestRCAGathererAgent:
    """Gatherer agent wiring tests for RCASkill."""

    def test_gatherer_is_first_agent(self) -> None:
        from vaig.skills.rca.skill import RCASkill

        skill = RCASkill()
        agents = skill.get_agents_config()
        assert agents[0]["name"] == "rca_gatherer"

    def test_gatherer_has_requires_tools(self) -> None:
        from vaig.skills.rca.skill import RCASkill

        skill = RCASkill()
        agents = skill.get_agents_config()
        gatherer = agents[0]
        assert gatherer.get("requires_tools") is True

    def test_gatherer_prompt_starts_with_anti_injection(self) -> None:
        from vaig.skills.rca.prompts import ANTI_INJECTION_RULE, RCA_GATHERER_PROMPT

        assert RCA_GATHERER_PROMPT.startswith(ANTI_INJECTION_RULE)

    def test_gatherer_system_instruction_is_non_empty(self) -> None:
        from vaig.skills.rca.skill import RCASkill

        skill = RCASkill()
        agents = skill.get_agents_config()
        gatherer = agents[0]
        assert isinstance(gatherer.get("system_instruction"), str)
        assert len(gatherer["system_instruction"]) > 0

    def test_gatherer_model_is_flash(self) -> None:
        from vaig.skills.rca.skill import RCASkill

        skill = RCASkill()
        agents = skill.get_agents_config()
        gatherer = agents[0]
        assert gatherer.get("model") == "gemini-2.5-flash"

    def test_gatherer_temperature_is_zero(self) -> None:
        from vaig.skills.rca.skill import RCASkill

        skill = RCASkill()
        agents = skill.get_agents_config()
        gatherer = agents[0]
        assert gatherer.get("temperature") == 0.0

    def test_pipeline_length(self) -> None:
        from vaig.skills.rca.skill import RCASkill

        skill = RCASkill()
        agents = skill.get_agents_config()
        assert len(agents) == 4

    def test_downstream_agents_not_requires_tools(self) -> None:
        from vaig.skills.rca.skill import RCASkill

        skill = RCASkill()
        agents = skill.get_agents_config()
        for agent in agents[1:]:
            assert not agent.get("requires_tools", False), (
                f"Agent '{agent['name']}' should not have requires_tools=True"
            )


class TestRCASkillLiveToolsMetadata:
    """Metadata contract for requires_live_tools flag."""

    def test_requires_live_tools_is_true(self) -> None:
        from vaig.skills.rca.skill import RCASkill

        skill = RCASkill()
        meta = skill.get_metadata()
        assert meta.requires_live_tools is True

    def test_live_tag_present(self) -> None:
        from vaig.skills.rca.skill import RCASkill

        skill = RCASkill()
        meta = skill.get_metadata()
        assert "live" in meta.tags

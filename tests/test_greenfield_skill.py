"""Tests for GreenfieldSkill — 6-stage structured project generation."""

from __future__ import annotations

import pytest

from vaig.skills.base import SkillMetadata, SkillPhase
from vaig.skills.greenfield.prompts import (
    STAGE_ORDER,
    STAGE_PROMPTS,
    SYSTEM_INSTRUCTION,
)
from vaig.skills.greenfield.skill import GreenfieldSkill

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def skill() -> GreenfieldSkill:
    """Return a fresh GreenfieldSkill instance."""
    return GreenfieldSkill()


# ===========================================================================
# TestGreenfieldSkillMetadata
# ===========================================================================


class TestGreenfieldSkillMetadata:
    """Tests for GreenfieldSkill.get_metadata()."""

    def test_returns_skill_metadata(self, skill: GreenfieldSkill) -> None:
        meta = skill.get_metadata()
        assert isinstance(meta, SkillMetadata)

    def test_name(self, skill: GreenfieldSkill) -> None:
        assert skill.get_metadata().name == "greenfield"

    def test_display_name(self, skill: GreenfieldSkill) -> None:
        assert "Greenfield" in skill.get_metadata().display_name

    def test_description_mentions_stages(self, skill: GreenfieldSkill) -> None:
        desc = skill.get_metadata().description
        assert "Requirements" in desc or "stage" in desc.lower()

    def test_requires_live_tools(self, skill: GreenfieldSkill) -> None:
        """Greenfield needs live tools for file creation."""
        assert skill.get_metadata().requires_live_tools is True

    def test_recommended_model_is_pro(self, skill: GreenfieldSkill) -> None:
        assert "pro" in skill.get_metadata().recommended_model

    def test_supported_phases_cover_all(self, skill: GreenfieldSkill) -> None:
        phases = skill.get_metadata().supported_phases
        assert SkillPhase.ANALYZE in phases
        assert SkillPhase.PLAN in phases
        assert SkillPhase.EXECUTE in phases
        assert SkillPhase.VALIDATE in phases

    def test_tags_include_greenfield(self, skill: GreenfieldSkill) -> None:
        assert "greenfield" in skill.get_metadata().tags


# ===========================================================================
# TestGreenfieldSkillSystemInstruction
# ===========================================================================


class TestGreenfieldSkillSystemInstruction:
    """Tests for GreenfieldSkill.get_system_instruction()."""

    def test_returns_string(self, skill: GreenfieldSkill) -> None:
        assert isinstance(skill.get_system_instruction(), str)

    def test_not_empty(self, skill: GreenfieldSkill) -> None:
        assert len(skill.get_system_instruction()) > 100

    def test_matches_module_constant(self, skill: GreenfieldSkill) -> None:
        assert skill.get_system_instruction() == SYSTEM_INSTRUCTION

    def test_contains_anti_injection(self, skill: GreenfieldSkill) -> None:
        instruction = skill.get_system_instruction()
        # The anti-injection rule is injected via ANTI_INJECTION_RULE constant
        assert "instruction" in instruction.lower() or "inject" in instruction.lower()


# ===========================================================================
# TestGreenfieldSkillGetPhasePrompt
# ===========================================================================


class TestGreenfieldSkillGetPhasePrompt:
    """Tests for GreenfieldSkill.get_phase_prompt()."""

    def test_analyze_maps_to_requirements(self, skill: GreenfieldSkill) -> None:
        prompt = skill.get_phase_prompt(SkillPhase.ANALYZE, context="", user_input="Build a CLI tool")
        # Requirements stage has "Requirements Analysis" header
        assert "Requirements" in prompt

    def test_plan_maps_to_architecture_decision(self, skill: GreenfieldSkill) -> None:
        prompt = skill.get_phase_prompt(SkillPhase.PLAN, context="requirements doc", user_input="")
        assert "Architecture Decision" in prompt or "ADR" in prompt

    def test_execute_maps_to_scaffold(self, skill: GreenfieldSkill) -> None:
        """EXECUTE phase maps to scaffold (first stage); implement is the second sub-stage."""
        prompt = skill.get_phase_prompt(SkillPhase.EXECUTE, context="spec doc", user_input="")
        assert "Scaffold" in prompt or "scaffold" in prompt.lower()

    def test_validate_maps_to_verify(self, skill: GreenfieldSkill) -> None:
        prompt = skill.get_phase_prompt(SkillPhase.VALIDATE, context="spec doc", user_input="")
        assert "Verify" in prompt or "Verification" in prompt

    def test_report_maps_to_verify(self, skill: GreenfieldSkill) -> None:
        prompt = skill.get_phase_prompt(SkillPhase.REPORT, context="spec doc", user_input="")
        assert "Verify" in prompt or "Verification" in prompt

    def test_user_input_is_injected(self, skill: GreenfieldSkill) -> None:
        prompt = skill.get_phase_prompt(
            SkillPhase.ANALYZE,
            context="",
            user_input="Build a REST API in Python",
        )
        assert "Build a REST API in Python" in prompt

    def test_context_is_injected(self, skill: GreenfieldSkill) -> None:
        prompt = skill.get_phase_prompt(
            SkillPhase.PLAN,
            context="Requirements document content here",
            user_input="",
        )
        assert "Requirements document content here" in prompt


# ===========================================================================
# TestGreenfieldSkillGetStagePrompt
# ===========================================================================


class TestGreenfieldSkillGetStagePrompt:
    """Tests for GreenfieldSkill.get_stage_prompt()."""

    def test_all_valid_stages_work(self, skill: GreenfieldSkill) -> None:
        for stage in STAGE_ORDER:
            prompt = skill.get_stage_prompt(stage, context="ctx", user_input="input")
            assert isinstance(prompt, str)
            assert len(prompt) > 50

    def test_unknown_stage_raises_value_error(self, skill: GreenfieldSkill) -> None:
        with pytest.raises(ValueError, match="Unknown Greenfield stage"):
            skill.get_stage_prompt("nonexistent_stage", context="", user_input="")

    def test_requirements_stage_user_input(self, skill: GreenfieldSkill) -> None:
        prompt = skill.get_stage_prompt("requirements", context="", user_input="My project idea")
        assert "My project idea" in prompt

    def test_scaffold_stage(self, skill: GreenfieldSkill) -> None:
        prompt = skill.get_stage_prompt("scaffold", context="spec doc", user_input="")
        assert "Scaffold" in prompt or "scaffold" in prompt.lower()

    def test_project_spec_stage(self, skill: GreenfieldSkill) -> None:
        prompt = skill.get_stage_prompt("project_spec", context="adrs", user_input="")
        assert "Spec" in prompt or "spec" in prompt.lower()


# ===========================================================================
# TestGreenfieldSkillStageOrder
# ===========================================================================


class TestGreenfieldSkillStageOrder:
    """Tests for GreenfieldSkill.stage_order property."""

    def test_returns_six_stages(self, skill: GreenfieldSkill) -> None:
        assert len(skill.stage_order) == 6

    def test_requirements_is_first(self, skill: GreenfieldSkill) -> None:
        assert skill.stage_order[0] == "requirements"

    def test_verify_is_last(self, skill: GreenfieldSkill) -> None:
        assert skill.stage_order[-1] == "verify"

    def test_matches_module_constant(self, skill: GreenfieldSkill) -> None:
        assert skill.stage_order == STAGE_ORDER

    def test_is_a_copy(self, skill: GreenfieldSkill) -> None:
        """Mutating the returned list does not affect the skill."""
        order = skill.stage_order
        order.append("extra")
        assert len(skill.stage_order) == 6


# ===========================================================================
# TestGreenfieldSkillAgentsConfig
# ===========================================================================


class TestGreenfieldSkillAgentsConfig:
    """Tests for GreenfieldSkill.get_agents_config()."""

    def test_returns_single_agent(self, skill: GreenfieldSkill) -> None:
        configs = skill.get_agents_config()
        assert len(configs) == 1

    def test_agent_has_required_fields(self, skill: GreenfieldSkill) -> None:
        agent = skill.get_agents_config()[0]
        assert "name" in agent
        assert "role" in agent
        assert "system_instruction" in agent
        assert "model" in agent

    def test_agent_name(self, skill: GreenfieldSkill) -> None:
        agent = skill.get_agents_config()[0]
        assert agent["name"] == "greenfield-agent"

    def test_agent_model_is_pro(self, skill: GreenfieldSkill) -> None:
        agent = skill.get_agents_config()[0]
        assert "pro" in agent["model"]


# ===========================================================================
# TestStagePrompts (module-level tests)
# ===========================================================================


class TestStagePrompts:
    """Tests for the STAGE_PROMPTS constant in prompts.py."""

    def test_all_six_stages_present(self) -> None:
        for stage in STAGE_ORDER:
            assert stage in STAGE_PROMPTS, f"Stage {stage!r} missing from STAGE_PROMPTS"

    def test_each_prompt_has_context_placeholder(self) -> None:
        # requirements stage only has user_input (context not used in template)
        for stage, template in STAGE_PROMPTS.items():
            if stage != "requirements":
                assert "{context}" in template, f"Stage {stage!r} missing {{context}} placeholder"

    def test_each_prompt_has_user_input_placeholder(self) -> None:
        for stage, template in STAGE_PROMPTS.items():
            assert "{user_input}" in template, f"Stage {stage!r} missing {{user_input}} placeholder"

    def test_prompts_are_non_trivial(self) -> None:
        for stage, template in STAGE_PROMPTS.items():
            assert len(template) > 200, f"Stage {stage!r} prompt seems too short"

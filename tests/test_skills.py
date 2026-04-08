"""Tests for skills base classes and registry."""

from __future__ import annotations

import logging

import pytest

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase, SkillResult

# ── Fixtures ─────────────────────────────────────────────────


class DummySkill(BaseSkill):
    """Minimal concrete skill for testing."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="dummy",
            display_name="Dummy Skill",
            description="A test skill",
            version="0.1.0",
            tags=["test"],
            supported_phases=[SkillPhase.ANALYZE, SkillPhase.REPORT],
            recommended_model="gemini-2.5-flash",
        )

    def get_system_instruction(self) -> str:
        return "You are a dummy assistant for testing."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return f"[{phase.value}] Context: {context}\nInput: {user_input}"


@pytest.fixture
def skill() -> DummySkill:
    return DummySkill()


# ── SkillPhase ───────────────────────────────────────────────


class TestSkillPhase:
    def test_all_phases(self) -> None:
        phases = list(SkillPhase)
        assert len(phases) == 5
        assert SkillPhase.ANALYZE in phases
        assert SkillPhase.PLAN in phases
        assert SkillPhase.EXECUTE in phases
        assert SkillPhase.VALIDATE in phases
        assert SkillPhase.REPORT in phases

    def test_str_enum_values(self) -> None:
        assert SkillPhase.ANALYZE == "analyze"
        assert SkillPhase.EXECUTE == "execute"


# ── SkillMetadata ────────────────────────────────────────────


class TestSkillMetadata:
    def test_defaults(self) -> None:
        meta = SkillMetadata(
            name="test",
            display_name="Test",
            description="A test",
        )
        assert meta.version == "1.0.0"
        assert meta.author == "vaig"
        assert meta.tags == []
        assert len(meta.supported_phases) == 3  # default: analyze, execute, report
        assert meta.recommended_model == "gemini-2.5-pro"

    def test_custom_values(self) -> None:
        meta = SkillMetadata(
            name="rca",
            display_name="RCA",
            description="Root cause",
            version="2.0.0",
            author="team",
            tags=["incident"],
            supported_phases=[SkillPhase.ANALYZE],
            recommended_model="gemini-2.5-flash",
        )
        assert meta.name == "rca"
        assert meta.version == "2.0.0"
        assert len(meta.supported_phases) == 1


# ── SkillResult ──────────────────────────────────────────────


class TestSkillResult:
    def test_construction(self) -> None:
        result = SkillResult(
            phase=SkillPhase.ANALYZE,
            success=True,
            output="Analysis complete",
        )
        assert result.phase == SkillPhase.ANALYZE
        assert result.success is True
        assert result.output == "Analysis complete"
        assert result.artifacts == {}
        assert result.metadata == {}
        assert result.next_phase is None

    def test_with_artifacts(self) -> None:
        result = SkillResult(
            phase=SkillPhase.EXECUTE,
            success=True,
            output="Done",
            artifacts={"report": "# Report\nSome content"},
            next_phase=SkillPhase.VALIDATE,
        )
        assert "report" in result.artifacts
        assert result.next_phase == SkillPhase.VALIDATE


# ── BaseSkill ────────────────────────────────────────────────


class TestBaseSkill:
    def test_get_metadata(self, skill: DummySkill) -> None:
        meta = skill.get_metadata()
        assert meta.name == "dummy"
        assert meta.display_name == "Dummy Skill"

    def test_get_system_instruction(self, skill: DummySkill) -> None:
        instruction = skill.get_system_instruction()
        assert "dummy assistant" in instruction

    def test_get_phase_prompt(self, skill: DummySkill) -> None:
        prompt = skill.get_phase_prompt(
            SkillPhase.ANALYZE,
            context="some logs",
            user_input="what happened?",
        )
        assert "[analyze]" in prompt
        assert "some logs" in prompt
        assert "what happened?" in prompt

    def test_default_agents_config(self, skill: DummySkill) -> None:
        agents = skill.get_agents_config()
        assert len(agents) == 1
        assert agents[0]["name"] == "dummy"
        assert agents[0]["role"] == "Dummy Skill"
        assert agents[0]["model"] == "gemini-2.5-flash"
        assert "system_instruction" in agents[0]

    def test_format_output(self, skill: DummySkill) -> None:
        result = SkillResult(
            phase=SkillPhase.ANALYZE,
            success=True,
            output="This is the output",
        )
        formatted = skill.format_output(result)
        assert formatted == "This is the output"


# ── Collision / Override ─────────────────────────────────────


class OverrideDummySkill(BaseSkill):
    """Skill with same name as DummySkill but different version — simulates external override."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="dummy",
            display_name="Override Dummy Skill",
            description="An override skill",
            version="2.0.0",
            tags=["test"],
            supported_phases=[SkillPhase.ANALYZE, SkillPhase.REPORT],
            recommended_model="gemini-2.5-flash",
        )

    def get_system_instruction(self) -> str:
        return "You are an override dummy."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return f"[{phase.value}] {context} | {user_input}"


class TestSkillCollision:
    """Tests for skill name collision handling in _register()."""

    def test_external_overrides_builtin_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Registering a skill with a name that already exists should override
        and log a warning."""
        from unittest.mock import MagicMock

        from vaig.skills.registry import SkillRegistry

        settings = MagicMock()
        registry = SkillRegistry(settings)

        builtin = DummySkill()
        external = OverrideDummySkill()

        # Register the builtin first (no collision yet)
        registry._register(builtin)
        assert registry._skills["dummy"] is builtin

        # The "vaig" parent logger has propagate=False (set by setup_logging()),
        # so we must patch the parent's propagate to let records reach caplog.
        vaig_logger = logging.getLogger("vaig")
        orig_propagate = vaig_logger.propagate
        vaig_logger.propagate = True

        # Now register the external with the same name
        try:
            with caplog.at_level(logging.WARNING):
                registry._register(external)
        finally:
            vaig_logger.propagate = orig_propagate

        # External should win
        assert registry._skills["dummy"] is external
        assert registry._metadata_cache["dummy"].version == "2.0.0"

        # Warning should mention the override
        assert "overridden" in caplog.text.lower() or "override" in caplog.text.lower()

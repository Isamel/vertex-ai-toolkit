"""Tests for Phase 4E — Skill enhancements: scaffolding and composition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vaig.skills.base import BaseSkill, CompositeSkill, SkillMetadata, SkillPhase

# ── Helpers ──────────────────────────────────────────────────


class _StubSkillA(BaseSkill):
    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="alpha",
            display_name="Alpha Skill",
            description="Skill A for testing",
            tags=["sre", "logs"],
            supported_phases=[SkillPhase.ANALYZE, SkillPhase.EXECUTE, SkillPhase.REPORT],
        )

    def get_system_instruction(self) -> str:
        return "You are Alpha."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return f"Alpha {phase.value}: {user_input}"


class _StubSkillB(BaseSkill):
    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="beta",
            display_name="Beta Skill",
            description="Skill B for testing",
            tags=["metrics", "alerts"],
            supported_phases=[SkillPhase.ANALYZE, SkillPhase.PLAN, SkillPhase.REPORT],
            requires_live_tools=True,
        )

    def get_system_instruction(self) -> str:
        return "You are Beta."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return f"Beta {phase.value}: {user_input}"

    def get_agents_config(self, **kwargs: Any) -> list[dict]:
        return [
            {"name": "beta_primary", "role": "Beta Primary", "system_instruction": "Primary", "model": "gemini-2.5-flash"},
            {"name": "beta_secondary", "role": "Beta Secondary", "system_instruction": "Secondary", "model": "gemini-2.5-flash"},
        ]


class _StubSkillC(BaseSkill):
    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="gamma",
            display_name="Gamma Skill",
            description="Skill C for testing",
            tags=["sre", "cost"],  # 'sre' overlaps with Alpha
            supported_phases=[SkillPhase.EXECUTE, SkillPhase.VALIDATE],
        )

    def get_system_instruction(self) -> str:
        return "You are Gamma."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return f"Gamma {phase.value}: {user_input}"


# ── CompositeSkill Tests ─────────────────────────────────────


class TestCompositeSkillCreation:
    def test_requires_at_least_two_skills(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            CompositeSkill([_StubSkillA()])

    def test_metadata_name_combined(self) -> None:
        comp = CompositeSkill([_StubSkillA(), _StubSkillB()])
        meta = comp.get_metadata()
        assert meta.name == "alpha+beta"

    def test_metadata_custom_name(self) -> None:
        comp = CompositeSkill([_StubSkillA(), _StubSkillB()], name="custom-combo")
        assert comp.get_metadata().name == "custom-combo"

    def test_metadata_display_name(self) -> None:
        comp = CompositeSkill([_StubSkillA(), _StubSkillB()])
        assert comp.get_metadata().display_name == "Alpha Skill + Beta Skill"

    def test_metadata_description(self) -> None:
        comp = CompositeSkill([_StubSkillA(), _StubSkillB()])
        desc = comp.get_metadata().description
        assert desc.startswith("Composite:")
        assert "Skill A for testing" in desc
        assert "Skill B for testing" in desc

    def test_tags_merged_and_deduplicated(self) -> None:
        comp = CompositeSkill([_StubSkillA(), _StubSkillC()])
        tags = comp.get_metadata().tags
        assert tags == ["sre", "logs", "cost"]  # 'sre' appears only once

    def test_phases_union(self) -> None:
        comp = CompositeSkill([_StubSkillA(), _StubSkillB(), _StubSkillC()])
        phases = comp.get_metadata().supported_phases
        # Should be the union, ordered by SkillPhase enum order
        assert SkillPhase.ANALYZE in phases
        assert SkillPhase.PLAN in phases
        assert SkillPhase.EXECUTE in phases
        assert SkillPhase.VALIDATE in phases
        assert SkillPhase.REPORT in phases

    def test_requires_live_tools_any(self) -> None:
        # Beta requires live tools
        comp = CompositeSkill([_StubSkillA(), _StubSkillB()])
        assert comp.get_metadata().requires_live_tools is True

    def test_requires_live_tools_none(self) -> None:
        comp = CompositeSkill([_StubSkillA(), _StubSkillC()])
        assert comp.get_metadata().requires_live_tools is False

    def test_model_from_first_skill(self) -> None:
        comp = CompositeSkill([_StubSkillA(), _StubSkillB()])
        assert comp.get_metadata().recommended_model == "gemini-2.5-pro"


class TestCompositeSkillSystemInstruction:
    def test_contains_all_instructions(self) -> None:
        comp = CompositeSkill([_StubSkillA(), _StubSkillB()])
        instruction = comp.get_system_instruction()
        assert "You are Alpha." in instruction
        assert "You are Beta." in instruction

    def test_includes_section_headers(self) -> None:
        comp = CompositeSkill([_StubSkillA(), _StubSkillB()])
        instruction = comp.get_system_instruction()
        assert "## Alpha Skill" in instruction
        assert "## Beta Skill" in instruction

    def test_includes_composite_intro(self) -> None:
        comp = CompositeSkill([_StubSkillA(), _StubSkillB()])
        instruction = comp.get_system_instruction()
        assert "composite specialist" in instruction.lower()


class TestCompositeSkillPhasePrompts:
    def test_merges_prompts_for_shared_phase(self) -> None:
        comp = CompositeSkill([_StubSkillA(), _StubSkillB()])
        prompt = comp.get_phase_prompt(SkillPhase.ANALYZE, "ctx", "question")
        # Both A and B support ANALYZE
        assert "Alpha Skill perspective" in prompt
        assert "Beta Skill perspective" in prompt
        assert "Alpha analyze: question" in prompt
        assert "Beta analyze: question" in prompt

    def test_only_includes_supporting_skills(self) -> None:
        comp = CompositeSkill([_StubSkillA(), _StubSkillB()])
        prompt = comp.get_phase_prompt(SkillPhase.PLAN, "ctx", "question")
        # Only Beta supports PLAN
        assert "Beta Skill perspective" in prompt
        assert "Alpha" not in prompt

    def test_unsupported_phase_message(self) -> None:
        comp = CompositeSkill([_StubSkillA(), _StubSkillB()])
        prompt = comp.get_phase_prompt(SkillPhase.VALIDATE, "ctx", "question")
        assert "No component skills support phase" in prompt


class TestCompositeSkillAgents:
    def test_merges_agents_from_all_skills(self) -> None:
        comp = CompositeSkill([_StubSkillA(), _StubSkillB()])
        agents = comp.get_agents_config()
        names = [a["name"] for a in agents]
        # A uses default (name=alpha), B has beta_primary + beta_secondary
        assert "alpha" in names
        assert "beta_primary" in names
        assert "beta_secondary" in names
        assert len(agents) == 3

    def test_deduplicates_agents_by_name(self) -> None:
        # Two copies of the same skill should not duplicate agents
        comp = CompositeSkill([_StubSkillA(), _StubSkillA()])
        agents = comp.get_agents_config()
        names = [a["name"] for a in agents]
        assert names.count("alpha") == 1

    def test_three_skills_all_agents_present(self) -> None:
        comp = CompositeSkill([_StubSkillA(), _StubSkillB(), _StubSkillC()])
        agents = comp.get_agents_config()
        names = {a["name"] for a in agents}
        assert names == {"alpha", "beta_primary", "beta_secondary", "gamma"}


# ── Scaffold Tests ───────────────────────────────────────────


class TestScaffoldHelpers:
    def test_to_snake_case(self) -> None:
        from vaig.skills.scaffold import _to_snake_case

        assert _to_snake_case("my-analyzer") == "my_analyzer"
        assert _to_snake_case("cost_analysis") == "cost_analysis"
        assert _to_snake_case("CostAnalysis") == "costanalysis"
        assert _to_snake_case("--trimmed--") == "trimmed"

    def test_to_class_name(self) -> None:
        from vaig.skills.scaffold import _to_class_name

        assert _to_class_name("my-analyzer") == "MyAnalyzerSkill"
        assert _to_class_name("cost_analysis") == "CostAnalysisSkill"
        assert _to_class_name("rca") == "RcaSkill"

    def test_to_kebab_case(self) -> None:
        from vaig.skills.scaffold import _to_kebab_case

        assert _to_kebab_case("my_analyzer") == "my-analyzer"
        assert _to_kebab_case("CostAnalysis") == "costanalysis"
        assert _to_kebab_case("--trimmed--") == "trimmed"


class TestScaffoldSkill:
    def test_creates_three_files(self, tmp_path: Path) -> None:
        from vaig.skills.scaffold import scaffold_skill

        skill_dir = scaffold_skill("my-analyzer", tmp_path, description="Test analyzer")
        assert (skill_dir / "__init__.py").exists()
        assert (skill_dir / "skill.py").exists()
        assert (skill_dir / "prompts.py").exists()

    def test_directory_name_is_snake_case(self, tmp_path: Path) -> None:
        from vaig.skills.scaffold import scaffold_skill

        skill_dir = scaffold_skill("my-analyzer", tmp_path)
        assert skill_dir.name == "my_analyzer"

    def test_init_has_docstring(self, tmp_path: Path) -> None:
        from vaig.skills.scaffold import scaffold_skill

        skill_dir = scaffold_skill("my-tool", tmp_path, description="A tool")
        content = (skill_dir / "__init__.py").read_text()
        assert "My Tool Skill" in content

    def test_skill_py_has_class(self, tmp_path: Path) -> None:
        from vaig.skills.scaffold import scaffold_skill

        skill_dir = scaffold_skill("cost-checker", tmp_path, description="Check costs")
        content = (skill_dir / "skill.py").read_text()
        assert "class CostCheckerSkill(BaseSkill):" in content
        assert 'name="cost-checker"' in content
        assert 'description="Check costs"' in content

    def test_prompts_py_has_templates(self, tmp_path: Path) -> None:
        from vaig.skills.scaffold import scaffold_skill

        skill_dir = scaffold_skill("perf-audit", tmp_path)
        content = (skill_dir / "prompts.py").read_text()
        assert "SYSTEM_INSTRUCTION" in content
        assert "PHASE_PROMPTS" in content
        assert '"analyze"' in content

    def test_custom_tags(self, tmp_path: Path) -> None:
        from vaig.skills.scaffold import scaffold_skill

        skill_dir = scaffold_skill(
            "net-scan", tmp_path, tags=["network", "security"],
        )
        content = (skill_dir / "skill.py").read_text()
        assert "'network'" in content
        assert "'security'" in content

    def test_default_tag_is_skill_name(self, tmp_path: Path) -> None:
        from vaig.skills.scaffold import scaffold_skill

        skill_dir = scaffold_skill("db-check", tmp_path)
        content = (skill_dir / "skill.py").read_text()
        assert "'db-check'" in content

    def test_raises_if_directory_exists(self, tmp_path: Path) -> None:
        from vaig.skills.scaffold import scaffold_skill

        (tmp_path / "my_skill").mkdir()
        with pytest.raises(FileExistsError, match="already exists"):
            scaffold_skill("my-skill", tmp_path)

    def test_skill_py_is_importable(self, tmp_path: Path) -> None:
        """Scaffolded skill.py should be syntactically valid Python."""
        from vaig.skills.scaffold import scaffold_skill

        skill_dir = scaffold_skill("importable-test", tmp_path)
        content = (skill_dir / "skill.py").read_text()
        compile(content, str(skill_dir / "skill.py"), "exec")

    def test_prompts_py_is_importable(self, tmp_path: Path) -> None:
        """Scaffolded prompts.py should be syntactically valid Python."""
        from vaig.skills.scaffold import scaffold_skill

        skill_dir = scaffold_skill("prompts-test", tmp_path)
        content = (skill_dir / "prompts.py").read_text()
        compile(content, str(skill_dir / "prompts.py"), "exec")


class TestSkillsCreateCLI:
    """Test the CLI create command."""

    def test_creates_skill_directory(self, tmp_path: Path) -> None:
        """Basic integration test — call the command function directly."""
        from typer.testing import CliRunner

        from vaig.cli.app import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "skills", "create", "test-skill",
            "-d", "A test skill",
            "-o", str(tmp_path),
        ])
        assert result.exit_code == 0
        assert "Skill scaffolded at:" in result.output
        assert (tmp_path / "test_skill" / "skill.py").exists()

    def test_fails_on_existing_directory(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from vaig.cli.app import app

        (tmp_path / "dup_skill").mkdir()
        runner = CliRunner()
        result = runner.invoke(app, [
            "skills", "create", "dup-skill",
            "-o", str(tmp_path),
        ])
        assert result.exit_code == 1

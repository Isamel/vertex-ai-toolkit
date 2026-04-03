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


class TestSkillsCreateCLIPreset:
    """Tests for CLI --preset and --interactive flags."""

    def test_preset_coding_generates_skill(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from vaig.cli.app import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "skills", "create", "cli-coder",
            "--preset", "coding",
            "-o", str(tmp_path),
        ])
        assert result.exit_code == 0
        skill_dir = tmp_path / "cli_coder"
        content = (skill_dir / "skill.py").read_text()
        assert "SkillPhase.PLAN" in content
        assert (skill_dir / "schema.py").exists()
        assert (skill_dir / "README.md").exists()

    def test_preset_analysis_generates_skill(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from vaig.cli.app import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "skills", "create", "cli-analyzer",
            "--preset", "analysis",
            "-o", str(tmp_path),
        ])
        assert result.exit_code == 0
        skill_dir = tmp_path / "cli_analyzer"
        assert (skill_dir / "skill.py").exists()
        assert not (skill_dir / "schema.py").exists()

    def test_preset_and_interactive_mutual_exclusion(self, tmp_path: Path) -> None:
        """SC-008: --preset + --interactive exits with error."""
        from typer.testing import CliRunner

        from vaig.cli.app import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "skills", "create", "conflict-test",
            "--preset", "coding",
            "--interactive",
            "-o", str(tmp_path),
        ])
        assert result.exit_code != 0

    def test_interactive_with_piped_input(self, tmp_path: Path) -> None:
        """SC-007: interactive mode with piped input generates matching output."""
        from typer.testing import CliRunner

        from vaig.cli.app import app

        runner = CliRunner()
        # Pipe: phases=1,3,5 (analyze,execute,report), agents=1, schema=n, live-tools=n
        result = runner.invoke(
            app,
            [
                "skills", "create", "interactive-test",
                "--interactive",
                "-o", str(tmp_path),
            ],
            input="1,3,5\n1\nn\nn\n",
        )
        assert result.exit_code == 0
        skill_dir = tmp_path / "interactive_test"
        assert (skill_dir / "skill.py").exists()
        assert not (skill_dir / "schema.py").exists()

    def test_invalid_preset_rejected(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from vaig.cli.app import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "skills", "create", "bad-preset",
            "--preset", "nonexistent",
            "-o", str(tmp_path),
        ])
        assert result.exit_code != 0

    def test_custom_preset_requires_interactive(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from vaig.cli.app import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "skills", "create", "custom-test",
            "--preset", "custom",
            "-o", str(tmp_path),
        ])
        assert result.exit_code != 0

    def test_output_lists_all_files(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from vaig.cli.app import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "skills", "create", "file-list-test",
            "--preset", "live-tools",
            "-o", str(tmp_path),
        ])
        assert result.exit_code == 0
        skill_dir = tmp_path / "file_list_test"
        # Verify the files are actually generated (output listing is cosmetic)
        assert (skill_dir / "schema.py").exists()
        assert (skill_dir / "README.md").exists()
        assert (skill_dir / "test_file_list_test.py").exists()
        assert "Created files:" in result.output


# ── Preset Scaffolding Tests ────────────────────────────────


class TestPresetScaffolding:
    """Tests for SkillPreset and get_preset (SC-001, SC-002)."""

    def test_get_known_preset(self) -> None:
        """SC-001: Retrieve known preset returns correct SkillPreset."""
        from vaig.skills._presets import get_preset

        preset = get_preset("live-tools")
        assert preset.name == "live-tools"
        assert len(preset.phases) == 4
        assert preset.agent_count == 3
        assert preset.requires_live_tools is True

    def test_get_analysis_preset(self) -> None:
        from vaig.skills._presets import get_preset

        preset = get_preset("analysis")
        assert preset.name == "analysis"
        assert len(preset.phases) == 3
        assert preset.agent_count == 1
        assert preset.requires_live_tools is False
        assert preset.generate_schema is False

    def test_get_coding_preset(self) -> None:
        from vaig.skills._presets import get_preset

        preset = get_preset("coding")
        assert preset.name == "coding"
        assert len(preset.phases) == 5
        assert preset.agent_count == 2
        assert preset.generate_schema is True
        assert preset.requires_live_tools is False

    def test_unknown_preset_raises_value_error(self) -> None:
        """SC-002: Unknown preset raises ValueError listing valid options."""
        from vaig.skills._presets import get_preset

        with pytest.raises(ValueError, match="Unknown preset 'nonexistent'"):
            get_preset("nonexistent")

    def test_unknown_preset_lists_valid_options(self) -> None:
        from vaig.skills._presets import get_preset

        with pytest.raises(ValueError, match="analysis") as exc_info:
            get_preset("bad")
        msg = str(exc_info.value)
        assert "live-tools" in msg
        assert "coding" in msg

    def test_preset_is_frozen(self) -> None:
        from vaig.skills._presets import get_preset

        preset = get_preset("analysis")
        with pytest.raises(AttributeError):
            preset.name = "mutated"  # type: ignore[misc]


class TestTemplateRendering:
    """Verify templates render without KeyError when called with expected vars."""

    def test_multi_agent_skill_template_renders(self) -> None:
        from vaig.skills.scaffold import _MULTI_AGENT_SKILL_TEMPLATE

        result = _MULTI_AGENT_SKILL_TEMPLATE.format(
            display_name="Test Skill",
            description="A test",
            class_name="TestSkillSkill",
            skill_name="test-skill",
            tags="['test']",
            supported_phases="SkillPhase.ANALYZE,",
            requires_live_tools="True",
            prompts_import=".prompts",
            agent_configs='{"name": "test", "role": "Tester", "system_instruction": "Test", "model": "gemini-2.5-pro"},',
        )
        assert "class TestSkillSkill(BaseSkill):" in result

    def test_coding_skill_template_renders(self) -> None:
        from vaig.skills.scaffold import _CODING_SKILL_TEMPLATE

        result = _CODING_SKILL_TEMPLATE.format(
            display_name="Code Skill",
            description="A coder",
            class_name="CodeSkillSkill",
            skill_name="code-skill",
            tags="['code']",
            prompts_import=".prompts",
        )
        assert "class CodeSkillSkill(BaseSkill):" in result
        assert "planner" in result.lower()

    def test_test_template_renders(self) -> None:
        from vaig.skills.scaffold import _TEST_TEMPLATE

        result = _TEST_TEMPLATE.format(
            display_name="My Tool",
            class_name="MyToolSkill",
            skill_name="my-tool",
            skill_import="vaig.skills.my_tool.skill",
        )
        assert "class TestMyToolSkill:" in result
        assert 'meta.name == "my-tool"' in result

    def test_readme_template_renders(self) -> None:
        from vaig.skills.scaffold import _README_TEMPLATE

        result = _README_TEMPLATE.format(
            display_name="My Tool",
            description="A tool",
            skill_name="my-tool",
            phases_list="- analyze\n- execute",
        )
        assert "# My Tool" in result
        assert "A tool" in result

    def test_schema_template_renders(self) -> None:
        from vaig.skills.scaffold import _SCHEMA_TEMPLATE

        result = _SCHEMA_TEMPLATE.format(
            display_name="My Tool",
            class_name_no_suffix="MyTool",
        )
        assert "class MyToolInput(BaseModel):" in result
        assert "class MyToolOutput(BaseModel):" in result

    def test_multi_agent_prompts_template_renders(self) -> None:
        from vaig.skills.scaffold import _MULTI_AGENT_PROMPTS_TEMPLATE

        result = _MULTI_AGENT_PROMPTS_TEMPLATE.format(display_name="My Agent")
        assert "SYSTEM_INSTRUCTION" in result
        assert '"validate"' in result

    def test_coding_prompts_template_renders(self) -> None:
        from vaig.skills.scaffold import _CODING_PROMPTS_TEMPLATE

        result = _CODING_PROMPTS_TEMPLATE.format(display_name="Code Buddy")
        assert '"plan"' in result
        assert '"validate"' in result


class TestPresetScaffoldIntegration:
    """Integration tests for scaffold_skill() with presets (SC-003 through SC-010)."""

    def test_no_preset_backward_compat_three_files(self, tmp_path: Path) -> None:
        """SC-003 / SC-010: No preset produces same 3 files as before."""
        from vaig.skills.scaffold import scaffold_skill

        skill_dir = scaffold_skill("compat-test", tmp_path)
        assert (skill_dir / "__init__.py").exists()
        assert (skill_dir / "skill.py").exists()
        assert (skill_dir / "prompts.py").exists()
        # Should NOT have the extra files
        assert not (skill_dir / "README.md").exists()
        assert not (skill_dir / "schema.py").exists()
        assert not list(skill_dir.glob("test_*.py"))

    def test_no_preset_with_description_and_tags(self, tmp_path: Path) -> None:
        """SC-010: Existing callers unaffected."""
        from vaig.skills.scaffold import scaffold_skill

        skill_dir = scaffold_skill("x", tmp_path, description="d", tags=["t"])
        content = (skill_dir / "skill.py").read_text()
        assert 'description="d"' in content
        assert "'t'" in content

    def test_analysis_preset_no_schema(self, tmp_path: Path) -> None:
        """SC-006: analysis preset does NOT produce schema.py."""
        from vaig.skills._presets import get_preset
        from vaig.skills.scaffold import scaffold_skill

        preset = get_preset("analysis")
        skill_dir = scaffold_skill("analyzer", tmp_path, preset=preset)
        assert (skill_dir / "skill.py").exists()
        assert (skill_dir / "README.md").exists()
        assert (skill_dir / "test_analyzer.py").exists()
        assert not (skill_dir / "schema.py").exists()

    def test_live_tools_preset_multi_agent_and_schema(self, tmp_path: Path) -> None:
        """SC-004: live-tools generates multi-agent + schema."""
        from vaig.skills._presets import get_preset
        from vaig.skills.scaffold import scaffold_skill

        preset = get_preset("live-tools")
        skill_dir = scaffold_skill("my-tool", tmp_path, preset=preset)
        skill_content = (skill_dir / "skill.py").read_text()
        assert "get_agents_config" in skill_content
        assert "requires_live_tools=True" in skill_content
        assert (skill_dir / "schema.py").exists()
        assert (skill_dir / "README.md").exists()
        assert (skill_dir / "test_my_tool.py").exists()

    def test_coding_preset_five_phases(self, tmp_path: Path) -> None:
        """SC-009: coding preset generates 5-phase skill with schema."""
        from vaig.skills._presets import get_preset
        from vaig.skills.scaffold import scaffold_skill

        preset = get_preset("coding")
        skill_dir = scaffold_skill("code-gen", tmp_path, preset=preset)
        skill_content = (skill_dir / "skill.py").read_text()
        assert "SkillPhase.PLAN" in skill_content
        assert "SkillPhase.VALIDATE" in skill_content
        assert "planner" in skill_content.lower()
        assert "executor" in skill_content.lower()
        assert (skill_dir / "schema.py").exists()
        assert (skill_dir / "README.md").exists()

    def test_test_file_generated_and_valid(self, tmp_path: Path) -> None:
        """SC-005: test file exists, imports skill class, has test_ function."""
        from vaig.skills._presets import get_preset
        from vaig.skills.scaffold import scaffold_skill

        preset = get_preset("analysis")
        skill_dir = scaffold_skill("test-gen", tmp_path, preset=preset)
        test_file = skill_dir / "test_test_gen.py"
        assert test_file.exists()
        content = test_file.read_text()
        assert "TestGenSkill" in content
        assert "def test_metadata" in content

    def test_generated_py_files_compile_analysis(self, tmp_path: Path) -> None:
        """NFR-003: All .py files from analysis preset are syntactically valid."""
        from vaig.skills._presets import get_preset
        from vaig.skills.scaffold import scaffold_skill

        preset = get_preset("analysis")
        skill_dir = scaffold_skill("compile-a", tmp_path, preset=preset)
        for py_file in skill_dir.glob("*.py"):
            content = py_file.read_text()
            compile(content, str(py_file), "exec")

    def test_generated_py_files_compile_live_tools(self, tmp_path: Path) -> None:
        """NFR-003: All .py files from live-tools preset are syntactically valid."""
        from vaig.skills._presets import get_preset
        from vaig.skills.scaffold import scaffold_skill

        preset = get_preset("live-tools")
        skill_dir = scaffold_skill("compile-lt", tmp_path, preset=preset)
        for py_file in skill_dir.glob("*.py"):
            content = py_file.read_text()
            compile(content, str(py_file), "exec")

    def test_generated_py_files_compile_coding(self, tmp_path: Path) -> None:
        """NFR-003: All .py files from coding preset are syntactically valid."""
        from vaig.skills._presets import get_preset
        from vaig.skills.scaffold import scaffold_skill

        preset = get_preset("coding")
        skill_dir = scaffold_skill("compile-c", tmp_path, preset=preset)
        for py_file in skill_dir.glob("*.py"):
            content = py_file.read_text()
            compile(content, str(py_file), "exec")

    def test_generated_get_agents_config_runtime(self, tmp_path: Path) -> None:
        """C5: exec generated multi-agent skill.py and verify get_agents_config() works at runtime."""
        import importlib
        import sys

        from vaig.skills._presets import get_preset
        from vaig.skills.scaffold import scaffold_skill

        preset = get_preset("live-tools")
        skill_dir = scaffold_skill("rt-check", tmp_path, preset=preset)

        # Make the generated package importable
        sys.path.insert(0, str(tmp_path))
        try:
            # Import the generated prompts module so the skill module can find it
            prompts_mod = importlib.import_module("rt_check.prompts")
            sys.modules[".prompts"] = prompts_mod
            sys.modules["rt_check.prompts"] = prompts_mod

            # exec the generated skill.py in a namespace that has proper imports
            skill_code = (skill_dir / "skill.py").read_text()
            # Replace the relative import with the absolute one we can resolve
            skill_code = skill_code.replace("from .prompts import", "from rt_check.prompts import")
            ns: dict[str, Any] = {}
            exec(compile(skill_code, str(skill_dir / "skill.py"), "exec"), ns)  # noqa: S102

            # Find the concrete skill class (ends with 'Skill', not BaseSkill)
            from vaig.skills.base import BaseSkill

            skill_classes = [
                v
                for k, v in ns.items()
                if k.endswith("Skill") and isinstance(v, type) and issubclass(v, BaseSkill) and v is not BaseSkill
            ]
            assert skill_classes, "No concrete Skill class found in generated skill.py"

            skill_instance = skill_classes[0]()
            agents = skill_instance.get_agents_config()
            assert isinstance(agents, list)
            assert len(agents) == preset.agent_count
            for agent in agents:
                assert "name" in agent
                assert "role" in agent
                assert "model" in agent
        finally:
            sys.path.pop(0)
            sys.modules.pop(".prompts", None)
            sys.modules.pop("rt_check.prompts", None)
            sys.modules.pop("rt_check", None)

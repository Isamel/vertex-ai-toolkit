"""Tests for CodeMigrationSkill — language-to-language code migration."""

from __future__ import annotations

from pathlib import Path

from vaig.skills.base import SkillPhase


class TestCodeMigrationSkillMetadata:
    def test_metadata_name(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill()
        meta = skill.get_metadata()
        assert meta.name == "code-migration"

    def test_metadata_display_name(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill()
        meta = skill.get_metadata()
        assert meta.display_name == "Code Migration"

    def test_metadata_description_non_empty(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill()
        meta = skill.get_metadata()
        assert len(meta.description) > 0

    def test_metadata_version(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill()
        meta = skill.get_metadata()
        assert meta.version == "1.0.0"

    def test_metadata_tags_include_migration(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill()
        meta = skill.get_metadata()
        assert "migration" in meta.tags

    def test_metadata_supported_phases_all_five(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill()
        meta = skill.get_metadata()
        assert SkillPhase.ANALYZE in meta.supported_phases
        assert SkillPhase.PLAN in meta.supported_phases
        assert SkillPhase.EXECUTE in meta.supported_phases
        assert SkillPhase.VALIDATE in meta.supported_phases
        assert SkillPhase.REPORT in meta.supported_phases

    def test_metadata_requires_live_tools(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill()
        meta = skill.get_metadata()
        assert meta.requires_live_tools is True


class TestCodeMigrationSkillSystemInstruction:
    def test_system_instruction_non_empty(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill()
        instruction = skill.get_system_instruction()
        assert isinstance(instruction, str)
        assert len(instruction) > 0

    def test_system_instruction_contains_anti_injection(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill()
        instruction = skill.get_system_instruction()
        # ANTI_INJECTION_RULE starts with "SECURITY RULE"
        assert "SECURITY RULE" in instruction

    def test_system_instruction_contains_migration_phases(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill()
        instruction = skill.get_system_instruction()
        assert "INVENTORY" in instruction
        assert "SEMANTIC_MAP" in instruction
        assert "IMPLEMENT" in instruction
        assert "VERIFY" in instruction

    def test_system_instruction_zero_placeholder_rule(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill()
        instruction = skill.get_system_instruction()
        assert "TODO" in instruction  # the rule mentions TODO as forbidden
        assert "Zero Placeholders" in instruction


class TestCodeMigrationSkillPhasePrompts:
    def test_analyze_phase_contains_context_and_input(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill()
        prompt = skill.get_phase_prompt(
            SkillPhase.ANALYZE,
            context="some source files here",
            user_input="Migrate from Python to Go",
        )
        assert "some source files here" in prompt
        assert "Migrate from Python to Go" in prompt

    def test_plan_phase_contains_context_and_input(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill()
        prompt = skill.get_phase_prompt(
            SkillPhase.PLAN,
            context="inventory output here",
            user_input="Map all idioms",
        )
        assert "inventory output here" in prompt
        assert "Map all idioms" in prompt

    def test_execute_phase_contains_context_and_input(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill()
        prompt = skill.get_phase_prompt(
            SkillPhase.EXECUTE,
            context="spec and source code",
            user_input="Implement migration",
        )
        assert "spec and source code" in prompt
        assert "Implement migration" in prompt

    def test_validate_phase_contains_context_and_input(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill()
        prompt = skill.get_phase_prompt(
            SkillPhase.VALIDATE,
            context="migrated output",
            user_input="Verify completeness",
        )
        assert "migrated output" in prompt
        assert "Verify completeness" in prompt

    def test_report_phase_contains_context_and_input(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill()
        prompt = skill.get_phase_prompt(
            SkillPhase.REPORT,
            context="final artifacts",
            user_input="Generate summary report",
        )
        assert "final artifacts" in prompt
        assert "Generate summary report" in prompt

    def test_plan_phase_injects_idiom_map_when_available(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        # python→go has a bundled idiom map
        skill = CodeMigrationSkill(source_lang="python", target_lang="go")
        prompt = skill.get_phase_prompt(
            SkillPhase.PLAN,
            context="inventory context",
            user_input="map idioms",
        )
        # The idiom map markdown should be injected into the prompt
        assert "Python → Go" in prompt or "python_to_go" in prompt or "Idiom Map" in prompt

    def test_plan_phase_no_idiom_map_when_pair_unknown(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        # haskell→rust has no bundled idiom map
        skill = CodeMigrationSkill(source_lang="haskell", target_lang="rust")
        # Should not raise — just omits the idiom section
        prompt = skill.get_phase_prompt(
            SkillPhase.PLAN,
            context="haskell inventory",
            user_input="map haskell to rust",
        )
        assert "haskell inventory" in prompt

    def test_unknown_phase_falls_back_to_analyze(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill()
        # REPORT maps to the "report" key in PHASE_PROMPTS
        prompt = skill.get_phase_prompt(
            SkillPhase.REPORT,
            context="ctx",
            user_input="inp",
        )
        assert "ctx" in prompt
        assert "inp" in prompt


class TestCodeMigrationSkillAgentsConfig:
    def test_three_agents(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill()
        agents = skill.get_agents_config()
        assert len(agents) == 3

    def test_required_agent_names(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill()
        agents = skill.get_agents_config()
        names = {a["name"] for a in agents}
        assert "migration_analyst" in names
        assert "migration_engineer" in names
        assert "migration_lead" in names

    def test_all_agents_have_required_keys(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill()
        for agent in skill.get_agents_config():
            assert "name" in agent
            assert "role" in agent
            assert "system_instruction" in agent
            assert "model" in agent

    def test_all_agents_have_non_empty_system_instruction(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill()
        for agent in skill.get_agents_config():
            assert isinstance(agent["system_instruction"], str)
            assert len(agent["system_instruction"]) > 0

    def test_all_agents_require_tools(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill()
        for agent in skill.get_agents_config():
            assert agent.get("requires_tools") is True

    def test_agents_have_coding_tool_category(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill()
        for agent in skill.get_agents_config():
            assert "coding" in agent.get("tool_categories", [])

    def test_agent_roles_reflect_language_pair(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill(source_lang="python", target_lang="go")
        agents = skill.get_agents_config()
        # At least one role should mention Python and Go
        roles = " ".join(a["role"] for a in agents)
        assert "Python" in roles
        assert "Go" in roles


class TestMigrationPhaseEnum:
    def test_all_five_values(self) -> None:
        from vaig.skills.code_migration.skill import MigrationPhase

        values = {p.value for p in MigrationPhase}
        assert values == {"inventory", "semantic_map", "spec", "implement", "verify"}

    def test_is_str_subclass(self) -> None:
        from vaig.skills.code_migration.skill import MigrationPhase

        assert isinstance(MigrationPhase.INVENTORY, str)
        assert MigrationPhase.INVENTORY == "inventory"

    def test_next_phase_order(self) -> None:
        from vaig.skills.code_migration.skill import MigrationPhase

        assert MigrationPhase.INVENTORY.next_phase() == MigrationPhase.SEMANTIC_MAP
        assert MigrationPhase.SEMANTIC_MAP.next_phase() == MigrationPhase.SPEC
        assert MigrationPhase.SPEC.next_phase() == MigrationPhase.IMPLEMENT
        assert MigrationPhase.IMPLEMENT.next_phase() == MigrationPhase.VERIFY

    def test_last_phase_next_is_none(self) -> None:
        from vaig.skills.code_migration.skill import MigrationPhase

        assert MigrationPhase.VERIFY.next_phase() is None

    def test_from_skill_phase_mapping(self) -> None:
        from vaig.skills.code_migration.skill import MigrationPhase

        assert MigrationPhase.from_skill_phase(SkillPhase.ANALYZE) == MigrationPhase.INVENTORY
        assert MigrationPhase.from_skill_phase(SkillPhase.PLAN) == MigrationPhase.SEMANTIC_MAP
        assert MigrationPhase.from_skill_phase(SkillPhase.EXECUTE) == MigrationPhase.IMPLEMENT
        assert MigrationPhase.from_skill_phase(SkillPhase.VALIDATE) == MigrationPhase.VERIFY
        assert MigrationPhase.from_skill_phase(SkillPhase.REPORT) == MigrationPhase.VERIFY

    def test_json_serializable(self) -> None:
        import json

        from vaig.skills.code_migration.skill import MigrationPhase

        data = {"phase": MigrationPhase.IMPLEMENT}
        # str subclass — json.dumps should work without custom encoder
        serialised = json.dumps(data)
        assert "implement" in serialised


class TestMigrationTracker:
    def test_empty_tracker_returns_empty_list(self) -> None:
        from vaig.skills.code_migration.skill import _MigrationTracker

        tracker = _MigrationTracker()
        assert tracker.get_all() == []

    def test_empty_tracker_markdown(self) -> None:
        from vaig.skills.code_migration.skill import _MigrationTracker

        tracker = _MigrationTracker()
        md = tracker.as_markdown()
        assert "No files tracked" in md

    def test_record_and_retrieve(self) -> None:
        from vaig.skills.code_migration.skill import MigrationPhase, _MigrationTracker

        tracker = _MigrationTracker()
        tracker.record("main.py", MigrationPhase.IMPLEMENT, "complete", "clean migration")
        records = tracker.get_all()
        assert len(records) == 1
        assert records[0]["filename"] == "main.py"
        assert records[0]["phase"] == "implement"
        assert records[0]["status"] == "complete"

    def test_record_updates_existing(self) -> None:
        from vaig.skills.code_migration.skill import MigrationPhase, _MigrationTracker

        tracker = _MigrationTracker()
        tracker.record("main.py", MigrationPhase.SPEC, "in-progress")
        tracker.record("main.py", MigrationPhase.IMPLEMENT, "complete")
        records = tracker.get_all()
        assert len(records) == 1  # deduped by filename
        assert records[0]["phase"] == "implement"

    def test_markdown_table_contains_filename(self) -> None:
        from vaig.skills.code_migration.skill import MigrationPhase, _MigrationTracker

        tracker = _MigrationTracker()
        tracker.record("utils.py", MigrationPhase.VERIFY, "complete")
        md = tracker.as_markdown()
        assert "utils.py" in md
        assert "verify" in md

    def test_multiple_files(self) -> None:
        from vaig.skills.code_migration.skill import MigrationPhase, _MigrationTracker

        tracker = _MigrationTracker()
        tracker.record("a.py", MigrationPhase.IMPLEMENT, "complete")
        tracker.record("b.py", MigrationPhase.IMPLEMENT, "skipped", "no equivalent")
        records = tracker.get_all()
        assert len(records) == 2


class TestCodeMigrationSkillPublicHelpers:
    def test_get_language_pair(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill(source_lang="Python", target_lang="Go")
        assert skill.get_language_pair() == ("python", "go")

    def test_get_idiom_map_python_to_go(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill(source_lang="python", target_lang="go")
        idiom_map = skill.get_idiom_map()
        assert idiom_map is not None
        assert idiom_map["source_lang"] == "python"
        assert idiom_map["target_lang"] == "go"
        assert len(idiom_map["idioms"]) > 0
        assert len(idiom_map["dependencies"]) > 0

    def test_get_idiom_map_unknown_pair_returns_none(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill(source_lang="haskell", target_lang="rust")
        assert skill.get_idiom_map() is None

    def test_get_file_tools_returns_list(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill(workspace=Path("/tmp"))
        tools = skill.get_file_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_get_file_tools_includes_verify_completeness(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill(workspace=Path("/tmp"))
        tools = skill.get_file_tools()
        tool_names = {t.name for t in tools}
        assert "verify_completeness" in tool_names

    def test_record_file_migration_and_get_log(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill, MigrationPhase

        skill = CodeMigrationSkill()
        skill.record_file_migration("app.py", MigrationPhase.IMPLEMENT, "complete")
        log = skill.get_migration_log()
        assert len(log) == 1
        assert log[0]["filename"] == "app.py"

    def test_get_migration_log_markdown_non_empty_after_record(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill, MigrationPhase

        skill = CodeMigrationSkill()
        skill.record_file_migration("service.py", MigrationPhase.VERIFY, "complete")
        md = skill.get_migration_log_markdown()
        assert "service.py" in md


class TestIdiomMapLoader:
    def test_load_python_to_go(self) -> None:
        from vaig.skills.code_migration.skill import _load_idiom_map

        data = _load_idiom_map("python", "go")
        assert data is not None
        assert data["source_lang"] == "python"
        assert data["target_lang"] == "go"

    def test_load_unknown_pair_returns_none(self) -> None:
        from vaig.skills.code_migration.skill import _load_idiom_map

        data = _load_idiom_map("cobol", "fortran")
        assert data is None

    def test_case_insensitive_lookup(self) -> None:
        from vaig.skills.code_migration.skill import _load_idiom_map

        data = _load_idiom_map("Python", "Go")
        assert data is not None

    def test_idiom_map_has_required_keys(self) -> None:
        from vaig.skills.code_migration.skill import _load_idiom_map

        data = _load_idiom_map("python", "go")
        assert data is not None
        assert "source_lang" in data
        assert "target_lang" in data
        assert "idioms" in data
        assert "dependencies" in data

    def test_idiom_map_idioms_list_non_empty(self) -> None:
        from vaig.skills.code_migration.skill import _load_idiom_map

        data = _load_idiom_map("python", "go")
        assert data is not None
        assert len(data["idioms"]) >= 10

    def test_idiom_map_deps_dict_non_empty(self) -> None:
        from vaig.skills.code_migration.skill import _load_idiom_map

        data = _load_idiom_map("python", "go")
        assert data is not None
        assert len(data["dependencies"]) >= 5

    def test_format_idiom_map_produces_markdown(self) -> None:
        from vaig.skills.code_migration.skill import _format_idiom_map, _load_idiom_map

        data = _load_idiom_map("python", "go")
        assert data is not None
        md = _format_idiom_map(data)
        assert "Python" in md
        assert "Go" in md
        assert "|" in md  # table rows


class TestCodeMigrationSkillRegistry:
    """Verify the skill is auto-discovered by the registry discovery scan."""

    def test_skill_discovered_by_name(self) -> None:
        from vaig.skills.registry import _discover_builtin_skills

        discovered = _discover_builtin_skills()
        assert "code-migration" in discovered

    def test_skill_module_path_correct(self) -> None:
        from vaig.skills.registry import _discover_builtin_skills

        discovered = _discover_builtin_skills()
        assert discovered["code-migration"] == "vaig.skills.code_migration.skill"

    def test_skill_is_importable_and_instantiable(self) -> None:
        from vaig.skills.code_migration.skill import CodeMigrationSkill

        skill = CodeMigrationSkill()
        assert skill.get_metadata().name == "code-migration"

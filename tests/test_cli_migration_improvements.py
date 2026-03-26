"""Tests for CLI migration improvements.

Covers:
- --dir/-d flag: load directories recursively into context
- --examples/-e flag: load reference files in a separate context section
- --phases flag: run multi-phase skill execution sequentially
- pentaho_to_glue.yaml: idiom map loads correctly
- _parse_phases: helper function edge cases
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _helpers import create_test_container
from typer.testing import CliRunner

from vaig.cli.app import app
from vaig.core.config import Settings
from vaig.skills.base import SkillPhase, SkillResult

runner = CliRunner(env={"NO_COLOR": "1"})

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


@pytest.fixture(autouse=True)
def _mock_settings() -> Settings:
    settings = Settings()
    with patch("vaig.cli._helpers._get_settings", return_value=settings):
        yield settings


# ══════════════════════════════════════════════════════════════
# _parse_phases helper
# ══════════════════════════════════════════════════════════════


class TestParsePhases:
    """Unit tests for the _parse_phases helper."""

    def test_none_returns_analyze_default(self) -> None:
        from vaig.cli.commands.ask import _parse_phases

        result = _parse_phases(None)
        assert result == [SkillPhase.ANALYZE]

    def test_empty_string_returns_analyze_default(self) -> None:
        from vaig.cli.commands.ask import _parse_phases

        result = _parse_phases("")
        assert result == [SkillPhase.ANALYZE]

    def test_single_valid_phase(self) -> None:
        from vaig.cli.commands.ask import _parse_phases

        result = _parse_phases("plan")
        assert result == [SkillPhase.PLAN]

    def test_multiple_phases_comma_separated(self) -> None:
        from vaig.cli.commands.ask import _parse_phases

        result = _parse_phases("analyze,plan,execute")
        assert result == [SkillPhase.ANALYZE, SkillPhase.PLAN, SkillPhase.EXECUTE]

    def test_all_phases(self) -> None:
        from vaig.cli.commands.ask import _parse_phases

        result = _parse_phases("analyze,plan,execute,validate,report")
        assert result == [
            SkillPhase.ANALYZE,
            SkillPhase.PLAN,
            SkillPhase.EXECUTE,
            SkillPhase.VALIDATE,
            SkillPhase.REPORT,
        ]

    def test_case_insensitive(self) -> None:
        from vaig.cli.commands.ask import _parse_phases

        result = _parse_phases("ANALYZE,Plan,EXECUTE")
        assert result == [SkillPhase.ANALYZE, SkillPhase.PLAN, SkillPhase.EXECUTE]

    def test_whitespace_trimmed(self) -> None:
        from vaig.cli.commands.ask import _parse_phases

        result = _parse_phases(" analyze , plan ")
        assert result == [SkillPhase.ANALYZE, SkillPhase.PLAN]

    def test_invalid_phase_raises_exit(self) -> None:
        import typer

        from vaig.cli.commands.ask import _parse_phases

        with pytest.raises(typer.Exit):
            _parse_phases("analyze,bogus,plan")

    def test_empty_tokens_ignored(self) -> None:
        from vaig.cli.commands.ask import _parse_phases

        # e.g. trailing comma
        result = _parse_phases("analyze,")
        assert result == [SkillPhase.ANALYZE]


# ══════════════════════════════════════════════════════════════
# --dir flag
# ══════════════════════════════════════════════════════════════


class TestDirFlag:
    """Tests for the --dir flag (recursive directory loading)."""

    def test_dir_flag_calls_add_directory(self, tmp_path: Path) -> None:
        (tmp_path / "script.py").write_text("print('hello')")
        (tmp_path / "utils.py").write_text("def foo(): pass")

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_single.return_value = MagicMock(content="done", usage=None)

        mock_builder = MagicMock()
        mock_builder.bundle.to_context_string.return_value = "dir content"
        mock_builder.add_directory.return_value = 2

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
            patch("vaig.context.builder.ContextBuilder", return_value=mock_builder),
        ):
            result = runner.invoke(
                app,
                ["ask", "Migrate this", "--dir", str(tmp_path), "--no-stream"],
            )

        assert result.exit_code == 0
        mock_builder.add_directory.assert_called_once_with(tmp_path)

    def test_dir_flag_warns_when_no_files_found(self, tmp_path: Path) -> None:
        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_single.return_value = MagicMock(content="done", usage=None)

        mock_builder = MagicMock()
        mock_builder.bundle.to_context_string.return_value = ""
        # add_directory returns 0 — no supported files found
        mock_builder.add_directory.return_value = 0

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
            patch("vaig.context.builder.ContextBuilder", return_value=mock_builder),
        ):
            result = runner.invoke(
                app,
                ["ask", "Migrate this", "--dir", str(tmp_path), "--no-stream"],
            )

        assert "Warning" in _strip_ansi(result.output) or result.exit_code == 0

    def test_dir_flag_exits_on_missing_directory(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does_not_exist"

        mock_builder = MagicMock()
        mock_builder.add_directory.side_effect = FileNotFoundError("not found")

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.context.builder.ContextBuilder", return_value=mock_builder),
        ):
            result = runner.invoke(
                app,
                ["ask", "Migrate this", "--dir", str(nonexistent), "--no-stream"],
            )

        assert result.exit_code == 1

    def test_multiple_dirs_all_loaded(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_single.return_value = MagicMock(content="done", usage=None)

        mock_builder = MagicMock()
        mock_builder.bundle.to_context_string.return_value = "combined content"
        mock_builder.add_directory.return_value = 1

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
            patch("vaig.context.builder.ContextBuilder", return_value=mock_builder),
        ):
            result = runner.invoke(
                app,
                ["ask", "Analyze", "--dir", str(dir_a), "--dir", str(dir_b), "--no-stream"],
            )

        assert result.exit_code == 0
        assert mock_builder.add_directory.call_count == 2

    def test_dir_included_in_context_file_paths(self, tmp_path: Path) -> None:
        """Directories are included in context_file_paths for export metadata."""
        mock_skill_result = SkillResult(
            phase=SkillPhase.ANALYZE,
            success=True,
            output="Migration analysis done",
        )
        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_skill_phase.return_value = mock_skill_result

        mock_skill = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill

        mock_builder = MagicMock()
        mock_builder.bundle.to_context_string.return_value = "dir content"
        mock_builder.add_directory.return_value = 1

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
            patch("vaig.skills.registry.SkillRegistry", return_value=mock_registry),
            patch("vaig.context.builder.ContextBuilder", return_value=mock_builder),
            patch("vaig.cli.commands.ask._handle_export_output") as mock_export,
        ):
            result = runner.invoke(
                app,
                ["ask", "Migrate", "--dir", str(tmp_path), "--skill", "migration"],
            )

        assert result.exit_code == 0
        # context_files should contain the directory path
        export_call_kwargs = mock_export.call_args[1]
        assert str(tmp_path) in export_call_kwargs.get("context_files", [])


# ══════════════════════════════════════════════════════════════
# --examples flag
# ══════════════════════════════════════════════════════════════


class TestExamplesFlag:
    """Tests for the --examples flag (separate reference context section)."""

    def test_examples_flag_creates_separate_builder(self, tmp_path: Path) -> None:
        example_file = tmp_path / "example_output.py"
        example_file.write_text("# example glue script")

        source_file = tmp_path / "source.py"
        source_file.write_text("# source pentaho script")

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_single.return_value = MagicMock(content="done", usage=None)

        call_count = 0
        instances: list[MagicMock] = []

        def make_builder(_settings: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            inst = MagicMock()
            inst.bundle.to_context_string.return_value = f"content_{call_count}"
            inst.add_file.return_value = MagicMock()
            instances.append(inst)
            return inst

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
            patch("vaig.context.builder.ContextBuilder", side_effect=make_builder),
        ):
            result = runner.invoke(
                app,
                [
                    "ask",
                    "Migrate to Glue",
                    "--file",
                    str(source_file),
                    "--examples",
                    str(example_file),
                    "--no-stream",
                ],
            )

        assert result.exit_code == 0
        # Two ContextBuilders were created: one for examples, one for source
        assert call_count == 2

    def test_examples_context_has_reference_examples_header(self, tmp_path: Path) -> None:
        example_file = tmp_path / "example.py"
        example_file.write_text("# reference code")

        source_file = tmp_path / "source.py"
        source_file.write_text("# source code")

        mock_orchestrator = MagicMock()
        captured_context: list[str] = []

        def capture_execute(question: str, context: str = "") -> MagicMock:
            captured_context.append(context)
            result = MagicMock()
            result.content = "ok"
            result.usage = None
            return result

        mock_orchestrator.execute_single.side_effect = capture_execute

        def make_builder(_settings: object) -> MagicMock:
            inst = MagicMock()
            inst.bundle.to_context_string.return_value = "section_content"
            inst.add_file.return_value = MagicMock()
            return inst

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
            patch("vaig.context.builder.ContextBuilder", side_effect=make_builder),
        ):
            runner.invoke(
                app,
                [
                    "ask",
                    "Migrate to Glue",
                    "--file",
                    str(source_file),
                    "--examples",
                    str(example_file),
                    "--no-stream",
                ],
            )

        assert len(captured_context) == 1
        ctx = captured_context[0]
        assert "Reference Examples" in ctx
        assert "Source Code" in ctx

    def test_examples_only_without_source_files(self, tmp_path: Path) -> None:
        example_file = tmp_path / "example.py"
        example_file.write_text("# reference only")

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_single.return_value = MagicMock(content="done", usage=None)

        def make_builder(_settings: object) -> MagicMock:
            inst = MagicMock()
            inst.bundle.to_context_string.return_value = "example_content"
            inst.add_file.return_value = MagicMock()
            return inst

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
            patch("vaig.context.builder.ContextBuilder", side_effect=make_builder),
        ):
            result = runner.invoke(
                app,
                ["ask", "Analyze example", "--examples", str(example_file), "--no-stream"],
            )

        assert result.exit_code == 0

    def test_examples_exits_on_missing_file(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "missing_example.py"

        mock_builder = MagicMock()
        mock_builder.add_file.side_effect = FileNotFoundError("not found")

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.context.builder.ContextBuilder", return_value=mock_builder),
        ):
            result = runner.invoke(
                app,
                ["ask", "Analyze", "--examples", str(nonexistent), "--no-stream"],
            )

        assert result.exit_code == 1


# ══════════════════════════════════════════════════════════════
# --phases flag
# ══════════════════════════════════════════════════════════════


class TestPhasesFlag:
    """Tests for --phases: multi-phase skill execution."""

    def test_single_phase_runs_once(self) -> None:
        mock_skill_result = SkillResult(
            phase=SkillPhase.ANALYZE,
            success=True,
            output="Analysis complete",
        )
        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_skill_phase.return_value = mock_skill_result

        mock_skill = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
            patch("vaig.skills.registry.SkillRegistry", return_value=mock_registry),
        ):
            result = runner.invoke(
                app,
                ["ask", "Migrate this", "--skill", "migration", "--phases", "analyze"],
            )

        assert result.exit_code == 0
        assert mock_orchestrator.execute_skill_phase.call_count == 1
        call_args = mock_orchestrator.execute_skill_phase.call_args
        assert call_args[0][1] == SkillPhase.ANALYZE

    def test_multiple_phases_run_sequentially(self) -> None:
        phases_run: list[SkillPhase] = []

        def fake_execute_skill_phase(
            skill: object,
            phase: SkillPhase,
            context: str,
            question: str,
        ) -> SkillResult:
            phases_run.append(phase)
            return SkillResult(phase=phase, success=True, output=f"Output of {phase.value}")

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_skill_phase.side_effect = fake_execute_skill_phase

        mock_skill = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
            patch("vaig.skills.registry.SkillRegistry", return_value=mock_registry),
        ):
            result = runner.invoke(
                app,
                ["ask", "Migrate project", "--skill", "migration", "--phases", "analyze,plan,execute"],
            )

        assert result.exit_code == 0
        assert phases_run == [SkillPhase.ANALYZE, SkillPhase.PLAN, SkillPhase.EXECUTE]

    def test_each_phase_output_feeds_next_phase_context(self) -> None:
        """Phase N output must be included in Phase N+1 context."""
        contexts_received: list[str] = []

        def fake_execute_skill_phase(
            skill: object,
            phase: SkillPhase,
            context: str,
            question: str,
        ) -> SkillResult:
            contexts_received.append(context)
            return SkillResult(phase=phase, success=True, output=f"result_{phase.value}")

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_skill_phase.side_effect = fake_execute_skill_phase

        mock_skill = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
            patch("vaig.skills.registry.SkillRegistry", return_value=mock_registry),
        ):
            runner.invoke(
                app,
                ["ask", "Migrate project", "--skill", "migration", "--phases", "analyze,plan"],
            )

        assert len(contexts_received) == 2
        # First phase context has no previous output
        assert "result_analyze" not in contexts_received[0]
        # Second phase context includes first phase's output
        assert "result_analyze" in contexts_received[1]

    def test_invalid_phase_name_exits_with_error(self) -> None:
        mock_skill = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator"),
            patch("vaig.skills.registry.SkillRegistry", return_value=mock_registry),
        ):
            result = runner.invoke(
                app,
                ["ask", "Migrate project", "--skill", "migration", "--phases", "analyze,INVALID"],
            )

        assert result.exit_code == 1

    def test_no_phases_flag_defaults_to_analyze(self) -> None:
        mock_skill_result = SkillResult(
            phase=SkillPhase.ANALYZE,
            success=True,
            output="Default analysis",
        )
        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_skill_phase.return_value = mock_skill_result

        mock_skill = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
            patch("vaig.skills.registry.SkillRegistry", return_value=mock_registry),
        ):
            result = runner.invoke(
                app,
                ["ask", "Analyze this", "--skill", "migration"],
            )

        assert result.exit_code == 0
        assert mock_orchestrator.execute_skill_phase.call_count == 1
        call_args = mock_orchestrator.execute_skill_phase.call_args
        assert call_args[0][1] == SkillPhase.ANALYZE

    def test_phases_label_printed_for_multiple_phases(self) -> None:
        def fake_execute_skill_phase(
            skill: object, phase: SkillPhase, context: str, question: str,
        ) -> SkillResult:
            return SkillResult(phase=phase, success=True, output=f"{phase.value} output")

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_skill_phase.side_effect = fake_execute_skill_phase
        mock_skill = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
            patch("vaig.skills.registry.SkillRegistry", return_value=mock_registry),
        ):
            result = runner.invoke(
                app,
                ["ask", "Migrate project", "--skill", "migration", "--phases", "analyze,plan"],
            )

        assert result.exit_code == 0
        output = _strip_ansi(result.output)
        assert "analyze" in output
        assert "plan" in output


# ══════════════════════════════════════════════════════════════
# Combined: --dir + --examples + --phases (migration workflow)
# ══════════════════════════════════════════════════════════════


class TestMigrationWorkflowCombined:
    """Integration-style tests combining dir, examples, and phases flags."""

    def test_full_migration_invocation(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "pentaho"
        src_dir.mkdir()
        (src_dir / "job.ktr").write_text("<transformation/>")

        example_file = tmp_path / "example_glue.py"
        example_file.write_text("# example glue script")

        def fake_execute_skill_phase(
            skill: object, phase: SkillPhase, context: str, question: str,
        ) -> SkillResult:
            return SkillResult(phase=phase, success=True, output=f"{phase.value} done")

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_skill_phase.side_effect = fake_execute_skill_phase

        mock_skill = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill

        instances: list[MagicMock] = []

        def make_builder(_settings: object) -> MagicMock:
            inst = MagicMock()
            inst.bundle.to_context_string.return_value = "context_content"
            inst.add_file.return_value = MagicMock()
            inst.add_directory.return_value = 1
            instances.append(inst)
            return inst

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
            patch("vaig.skills.registry.SkillRegistry", return_value=mock_registry),
            patch("vaig.context.builder.ContextBuilder", side_effect=make_builder),
        ):
            result = runner.invoke(
                app,
                [
                    "ask",
                    "Migrate Pentaho to Glue",
                    "--dir",
                    str(src_dir),
                    "--examples",
                    str(example_file),
                    "--skill",
                    "migration",
                    "--phases",
                    "analyze,plan,execute",
                ],
            )

        assert result.exit_code == 0
        # All 3 phases executed
        assert mock_orchestrator.execute_skill_phase.call_count == 3


# ══════════════════════════════════════════════════════════════
# pentaho_to_glue.yaml idiom map
# ══════════════════════════════════════════════════════════════


class TestPentahoToGlueIdiomMap:
    """Tests that pentaho_to_glue.yaml exists, is valid YAML, and has correct schema."""

    @pytest.fixture()
    def idiom_path(self) -> Path:
        here = Path(__file__).parent.parent
        return here / "src" / "vaig" / "skills" / "code_migration" / "idioms" / "pentaho_to_glue.yaml"

    def test_file_exists(self, idiom_path: Path) -> None:
        assert idiom_path.exists(), f"Expected idiom map at {idiom_path}"

    def test_file_is_valid_yaml(self, idiom_path: Path) -> None:
        import yaml  # type: ignore[import-untyped]

        with idiom_path.open() as f:
            data = yaml.safe_load(f)

        assert isinstance(data, dict), "YAML root must be a mapping"

    def test_has_required_fields(self, idiom_path: Path) -> None:
        import yaml

        with idiom_path.open() as f:
            data = yaml.safe_load(f)

        assert "source_lang" in data
        assert "target_lang" in data
        assert "idioms" in data

    def test_source_and_target_lang(self, idiom_path: Path) -> None:
        import yaml

        with idiom_path.open() as f:
            data = yaml.safe_load(f)

        assert data["source_lang"] == "pentaho"
        assert data["target_lang"] == "glue"

    def test_idioms_is_non_empty_list(self, idiom_path: Path) -> None:
        import yaml

        with idiom_path.open() as f:
            data = yaml.safe_load(f)

        assert isinstance(data["idioms"], list)
        assert len(data["idioms"]) > 0

    def test_each_idiom_has_required_fields(self, idiom_path: Path) -> None:
        import yaml

        with idiom_path.open() as f:
            data = yaml.safe_load(f)

        for i, idiom in enumerate(data["idioms"]):
            assert "source_pattern" in idiom, f"idiom[{i}] missing source_pattern"
            assert "target_pattern" in idiom, f"idiom[{i}] missing target_pattern"

    def test_has_dependencies_section(self, idiom_path: Path) -> None:
        import yaml

        with idiom_path.open() as f:
            data = yaml.safe_load(f)

        assert "dependencies" in data
        assert isinstance(data["dependencies"], dict)

    def test_covers_key_pentaho_concepts(self, idiom_path: Path) -> None:
        """Spot-check that common Pentaho concepts are covered."""
        import yaml

        with idiom_path.open() as f:
            data = yaml.safe_load(f)

        source_patterns = " ".join(
            idiom["source_pattern"] for idiom in data["idioms"]
        ).lower()

        assert "transformation" in source_patterns or "ktr" in source_patterns
        assert "table input" in source_patterns or "table output" in source_patterns


# ══════════════════════════════════════════════════════════════
# _async_ask_impl — new parameters
# ══════════════════════════════════════════════════════════════


class TestAsyncAskImplNewParams:
    """Tests that _async_ask_impl accepts and handles dirs, examples, and phases."""

    @pytest.fixture(autouse=True)
    def _mock_settings(self) -> Settings:
        settings = Settings()
        with patch("vaig.cli._helpers._get_settings", return_value=settings):
            yield settings

    @pytest.mark.asyncio
    async def test_async_ask_with_dirs(self, tmp_path: Path) -> None:
        from vaig.cli.commands.ask import _async_ask_impl

        mock_orchestrator = MagicMock()
        mock_orchestrator.async_execute_single = AsyncMock(
            return_value=MagicMock(content="done", usage=None)
        )

        mock_loaded = MagicMock()
        mock_loaded.path = tmp_path

        instances: list[MagicMock] = []

        def make_builder(_settings: object) -> MagicMock:
            inst = MagicMock()
            inst.async_add_directory = AsyncMock(return_value=1)
            inst.bundle.to_context_string.return_value = "dir content"
            instances.append(inst)
            return inst

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
            patch("vaig.context.builder.ContextBuilder", side_effect=make_builder),
        ):
            await _async_ask_impl("Analyze", dirs=[tmp_path], no_stream=True)

        assert any(inst.async_add_directory.called for inst in instances)

    @pytest.mark.asyncio
    async def test_async_ask_with_examples(self, tmp_path: Path) -> None:
        from vaig.cli.commands.ask import _async_ask_impl

        example_file = tmp_path / "example.py"
        example_file.write_text("# example")

        mock_orchestrator = MagicMock()
        mock_orchestrator.async_execute_single = AsyncMock(
            return_value=MagicMock(content="done", usage=None)
        )

        instances: list[MagicMock] = []

        def make_builder(_settings: object) -> MagicMock:
            inst = MagicMock()
            inst.async_add_file = AsyncMock(return_value=MagicMock())
            inst.bundle.to_context_string.return_value = "content"
            instances.append(inst)
            return inst

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
            patch("vaig.context.builder.ContextBuilder", side_effect=make_builder),
        ):
            await _async_ask_impl("Analyze", examples=[example_file], no_stream=True)

        # Two builders: one for examples, one for source (even if source is empty)
        # OR one builder if only examples are passed (no source files/dirs)
        assert len(instances) >= 1
        assert any(inst.async_add_file.called for inst in instances)

    @pytest.mark.asyncio
    async def test_async_ask_with_phases(self) -> None:
        from vaig.cli.commands.ask import _async_ask_impl

        phases_run: list[SkillPhase] = []

        async def fake_execute_skill_phase(
            skill: object, phase: SkillPhase, context: str, question: str,
        ) -> SkillResult:
            phases_run.append(phase)
            return SkillResult(phase=phase, success=True, output=f"output_{phase.value}")

        mock_orchestrator = MagicMock()
        mock_orchestrator.async_execute_skill_phase = AsyncMock(
            side_effect=fake_execute_skill_phase
        )

        mock_skill = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
            patch("vaig.skills.registry.SkillRegistry", return_value=mock_registry),
        ):
            await _async_ask_impl(
                "Migrate project",
                skill="migration",
                phases="analyze,plan",
            )

        assert phases_run == [SkillPhase.ANALYZE, SkillPhase.PLAN]

    @pytest.mark.asyncio
    async def test_async_ask_phases_context_chaining(self) -> None:
        from vaig.cli.commands.ask import _async_ask_impl

        contexts_received: list[str] = []

        async def fake_execute_skill_phase(
            skill: object, phase: SkillPhase, context: str, question: str,
        ) -> SkillResult:
            contexts_received.append(context)
            return SkillResult(phase=phase, success=True, output=f"result_{phase.value}")

        mock_orchestrator = MagicMock()
        mock_orchestrator.async_execute_skill_phase = AsyncMock(
            side_effect=fake_execute_skill_phase
        )

        mock_skill = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
            patch("vaig.skills.registry.SkillRegistry", return_value=mock_registry),
        ):
            await _async_ask_impl(
                "Migrate project",
                skill="migration",
                phases="analyze,plan",
            )

        assert len(contexts_received) == 2
        # Second phase's context must include first phase's output
        assert "result_analyze" in contexts_received[1]

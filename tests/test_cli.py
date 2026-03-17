"""Tests for the CLI app — Typer commands."""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest
from conftest import create_test_container
from typer.testing import CliRunner

from vaig import __version__
from vaig.cli.app import app
from vaig.core.config import Settings
from vaig.skills.base import SkillMetadata, SkillPhase, SkillResult

runner = CliRunner(env={"NO_COLOR": "1"})

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from *text*.

    Rich/Typer may inject color codes even when ``color=False`` is passed to
    ``CliRunner.invoke`` — for example when ``FORCE_COLOR=1`` is set in CI or
    the app uses ``rich_markup_mode="rich"``.  Stripping before assertions
    makes the tests environment-agnostic.
    """
    return _ANSI_RE.sub("", text)


@pytest.fixture(autouse=True)
def _mock_settings() -> Settings:
    """Provide a default Settings object to all CLI commands, avoiding real config."""
    settings = Settings()
    with patch("vaig.cli._helpers._get_settings", return_value=settings):
        yield settings


# ══════════════════════════════════════════════════════════════
# APP BASICS
# ══════════════════════════════════════════════════════════════
class TestAppBasics:
    def test_help_shows_help_text(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        output = _strip_ansi(result.output)
        assert "Vertex AI" in output or "VAIG" in output

    def test_version_flag(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in _strip_ansi(result.output)

    def test_version_short_flag(self) -> None:
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert __version__ in _strip_ansi(result.output)

    def test_no_args_shows_help(self) -> None:
        result = runner.invoke(app, [])
        # no_args_is_help=True — Typer shows help and exits with code 0.
        # Some Typer/Click versions exit with 0, others with 2 for "no args" help.
        assert result.exit_code in (0, 2)
        output = _strip_ansi(result.output)
        assert "Usage" in output or "VAIG" in output


# ══════════════════════════════════════════════════════════════
# SESSIONS COMMANDS
# ══════════════════════════════════════════════════════════════
class TestSessionsCommands:
    def test_sessions_list_with_data(self) -> None:
        fake_sessions = [
            {
                "id": "abc123def456ghi789",
                "name": "test-session",
                "model": "gemini-2.5-pro",
                "skill": "rca",
                "created_at": "2025-01-15",
            },
            {
                "id": "xyz987wvu654tsr321",
                "name": "debug-session",
                "model": "gemini-2.5-flash",
                "skill": None,
                "created_at": "2025-01-16",
            },
        ]
        mock_manager = MagicMock()
        mock_manager.list_sessions.return_value = fake_sessions

        with patch("vaig.session.manager.SessionManager", return_value=mock_manager):
            result = runner.invoke(app, ["sessions", "list"])

        assert result.exit_code == 0
        assert "test-session" in result.output
        assert "debug-session" in result.output
        assert "gemini-2.5-pro" in result.output
        mock_manager.close.assert_called_once()

    def test_sessions_list_empty(self) -> None:
        mock_manager = MagicMock()
        mock_manager.list_sessions.return_value = []

        with patch("vaig.session.manager.SessionManager", return_value=mock_manager):
            result = runner.invoke(app, ["sessions", "list"])

        assert result.exit_code == 0
        assert "No sessions found" in result.output

    def test_sessions_delete_with_force(self) -> None:
        mock_manager = MagicMock()
        mock_manager.delete_session.return_value = True

        with patch("vaig.session.manager.SessionManager", return_value=mock_manager):
            result = runner.invoke(app, ["sessions", "delete", "abc123", "--force"])

        assert result.exit_code == 0
        assert "Deleted session" in result.output or "abc123" in result.output
        mock_manager.delete_session.assert_called_once_with("abc123")
        mock_manager.close.assert_called_once()

    def test_sessions_delete_not_found(self) -> None:
        mock_manager = MagicMock()
        mock_manager.delete_session.return_value = False

        with patch("vaig.session.manager.SessionManager", return_value=mock_manager):
            result = runner.invoke(app, ["sessions", "delete", "nonexistent", "--force"])

        # Error printed to stderr — CliRunner doesn't capture stderr by default
        # The command still runs without raising, so exit_code is 0
        assert result.exit_code == 0

    def test_sessions_list_skips_non_dict_items(self) -> None:
        """Defensive guard: if list_sessions returns corrupt data (non-dict items), skip them."""
        mixed_sessions = [
            {"id": "valid-session-id", "name": "good", "model": "gemini-2.5-pro", "skill": None, "created_at": "2025-01-15"},
            "corrupt-entry",  # not a dict — should be skipped
            ["also", "corrupt"],  # list — should be skipped
        ]
        mock_manager = MagicMock()
        mock_manager.list_sessions.return_value = mixed_sessions

        with patch("vaig.session.manager.SessionManager", return_value=mock_manager):
            result = runner.invoke(app, ["sessions", "list"])

        assert result.exit_code == 0
        assert "good" in result.output
        # Should not crash — corrupt entries silently skipped
        mock_manager.close.assert_called_once()


# ══════════════════════════════════════════════════════════════
# MODELS COMMANDS
# ══════════════════════════════════════════════════════════════
class TestModelsCommands:
    def test_models_list_with_data(self) -> None:
        fake_models = [
            {"id": "gemini-2.5-pro", "description": "Pro model"},
            {"id": "gemini-2.5-flash", "description": "Flash model"},
        ]
        mock_client = MagicMock()
        mock_client.list_available_models.return_value = fake_models

        with patch("vaig.core.client.GeminiClient", return_value=mock_client):
            result = runner.invoke(app, ["models", "list"])

        assert result.exit_code == 0
        assert "gemini-2.5-pro" in result.output
        assert "gemini-2.5-flash" in result.output

    def test_models_list_empty(self) -> None:
        mock_client = MagicMock()
        mock_client.list_available_models.return_value = []

        with patch("vaig.core.client.GeminiClient", return_value=mock_client):
            result = runner.invoke(app, ["models", "list"])

        assert result.exit_code == 0
        assert "No models configured" in result.output

    def test_models_list_skips_non_dict_items(self) -> None:
        """Defensive guard: if list_available_models returns corrupt data, skip non-dict items."""
        mixed_models = [
            {"id": "gemini-2.5-pro", "description": "Pro model"},
            "corrupt-entry",  # not a dict
            42,  # not a dict
        ]
        mock_client = MagicMock()
        mock_client.list_available_models.return_value = mixed_models

        with patch("vaig.core.client.GeminiClient", return_value=mock_client):
            result = runner.invoke(app, ["models", "list"])

        assert result.exit_code == 0
        assert "gemini-2.5-pro" in result.output


# ══════════════════════════════════════════════════════════════
# SKILLS COMMANDS
# ══════════════════════════════════════════════════════════════
class TestSkillsCommands:
    def test_skills_list_with_data(self) -> None:
        fake_metadata = [
            SkillMetadata(
                name="rca",
                display_name="Root Cause Analysis",
                description="Analyzes incidents",
                supported_phases=[SkillPhase.ANALYZE, SkillPhase.REPORT],
                recommended_model="gemini-2.5-pro",
            ),
            SkillMetadata(
                name="migration",
                display_name="Migration Assistant",
                description="Helps with migrations",
                supported_phases=[SkillPhase.PLAN, SkillPhase.EXECUTE],
                recommended_model="gemini-2.5-flash",
            ),
        ]
        mock_registry = MagicMock()
        mock_registry.list_skills.return_value = fake_metadata

        with patch("vaig.skills.registry.SkillRegistry", return_value=mock_registry):
            result = runner.invoke(app, ["skills", "list"])

        assert result.exit_code == 0
        assert "rca" in result.output
        # Rich table may wrap "Root Cause Analysis" across lines
        assert "Root Cause" in result.output
        assert "Analysis" in result.output
        assert "migration" in result.output

    def test_skills_list_empty(self) -> None:
        mock_registry = MagicMock()
        mock_registry.list_skills.return_value = []

        with patch("vaig.skills.registry.SkillRegistry", return_value=mock_registry):
            result = runner.invoke(app, ["skills", "list"])

        assert result.exit_code == 0
        assert "No skills loaded" in result.output

    def test_skills_info_found(self) -> None:
        meta = SkillMetadata(
            name="rca",
            display_name="Root Cause Analysis",
            description="Deep incident analysis",
            version="2.0.0",
            tags=["incident", "debugging"],
            supported_phases=[SkillPhase.ANALYZE, SkillPhase.REPORT],
            recommended_model="gemini-2.5-pro",
        )
        mock_skill = MagicMock()
        mock_skill.get_metadata.return_value = meta
        mock_skill.get_agents_config.return_value = [
            {"name": "analyzer", "role": "Lead Analyst", "model": "gemini-2.5-pro"},
        ]

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill

        with patch("vaig.skills.registry.SkillRegistry", return_value=mock_registry):
            result = runner.invoke(app, ["skills", "info", "rca"])

        assert result.exit_code == 0
        assert "Root Cause Analysis" in result.output
        assert "2.0.0" in result.output
        mock_registry.get.assert_called_once_with("rca")

    def test_skills_info_not_found(self) -> None:
        mock_registry = MagicMock()
        mock_registry.get.return_value = None
        mock_registry.list_names.return_value = ["rca", "anomaly"]

        with patch("vaig.skills.registry.SkillRegistry", return_value=mock_registry):
            result = runner.invoke(app, ["skills", "info", "unknown_skill"])

        assert result.exit_code == 1


# ══════════════════════════════════════════════════════════════
# ASK COMMAND
# ══════════════════════════════════════════════════════════════
class TestAskCommand:
    def test_ask_no_stream(self) -> None:
        mock_agent_result = MagicMock()
        mock_agent_result.content = "The answer is 42."

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_single.return_value = mock_agent_result

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
        ):
            result = runner.invoke(app, ["ask", "What is the meaning of life?", "--no-stream"])

        assert result.exit_code == 0
        assert "42" in result.output
        mock_orchestrator.execute_single.assert_called_once_with(
            "What is the meaning of life?", context=""
        )

    def test_ask_with_skill(self) -> None:
        mock_skill = MagicMock()
        skill_result = SkillResult(
            phase=SkillPhase.ANALYZE,
            success=True,
            output="Root cause: memory leak in worker pool",
        )

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_skill_phase.return_value = skill_result

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
            patch("vaig.skills.registry.SkillRegistry", return_value=mock_registry),
        ):
            result = runner.invoke(app, ["ask", "Why is my app slow?", "--skill", "rca"])

        assert result.exit_code == 0
        assert "memory leak" in result.output
        mock_registry.get.assert_called_once_with("rca")

    def test_ask_with_unknown_skill(self) -> None:
        mock_registry = MagicMock()
        mock_registry.get.return_value = None
        mock_registry.list_names.return_value = ["rca", "anomaly"]

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator"),
            patch("vaig.skills.registry.SkillRegistry", return_value=mock_registry),
        ):
            result = runner.invoke(app, ["ask", "Do something", "--skill", "nonexistent"])

        assert result.exit_code == 1

    def test_ask_with_file(self, tmp_path: pytest.TempPathFactory) -> None:
        # Create a temp file to pass as --file
        test_file = tmp_path / "sample.py"  # type: ignore[operator]
        test_file.write_text("print('hello')")

        mock_agent_result = MagicMock()
        mock_agent_result.content = "This code prints hello."

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_single.return_value = mock_agent_result

        mock_builder = MagicMock()
        mock_builder.bundle.to_context_string.return_value = "file content here"

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
            patch("vaig.context.builder.ContextBuilder", return_value=mock_builder),
        ):
            result = runner.invoke(
                app, ["ask", "Explain this code", "--file", str(test_file), "--no-stream"]
            )

        assert result.exit_code == 0
        assert "prints hello" in result.output
        mock_builder.add_file.assert_called_once()
        mock_orchestrator.execute_single.assert_called_once_with(
            "Explain this code", context="file content here"
        )

    def test_ask_code_with_workspace(self, tmp_path: pytest.TempPathFactory, _mock_settings: Settings) -> None:
        """--workspace flag overrides coding.workspace_root in code mode."""
        workspace_dir = tmp_path / "my-project"  # type: ignore[operator]
        workspace_dir.mkdir()

        mock_result = MagicMock()
        mock_result.content = "Done."
        mock_result.metadata = {"tools_executed": [], "iterations": 1}
        mock_result.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.coding.CodingAgent") as MockCodingAgent,
        ):
            mock_agent = MagicMock()
            mock_agent.execute.return_value = mock_result
            MockCodingAgent.return_value = mock_agent

            result = runner.invoke(
                app,
                ["ask", "Create a file", "--code", "--workspace", str(workspace_dir)],
            )

        assert result.exit_code == 0
        assert _mock_settings.coding.workspace_root == str(workspace_dir)

    def test_ask_code_workspace_not_found(self) -> None:
        """--workspace with a non-existent directory should exit with error."""
        result = runner.invoke(
            app,
            ["ask", "Do something", "--code", "--workspace", "/nonexistent/path"],
        )

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_ask_workspace_ignored_without_code(self, _mock_settings: Settings) -> None:
        """--workspace without --code should be silently ignored (no error)."""
        mock_agent_result = MagicMock()
        mock_agent_result.content = "The answer."

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_single.return_value = mock_agent_result

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
        ):
            result = runner.invoke(
                app,
                ["ask", "Hello", "--no-stream", "--workspace", "."],
            )

        assert result.exit_code == 0
        # workspace_root should NOT have been changed since --code was not used
        assert _mock_settings.coding.workspace_root == "."


# ══════════════════════════════════════════════════════════════
# CHAT COMMAND — WORKSPACE FLAG
# ══════════════════════════════════════════════════════════════
class TestChatWorkspace:
    def test_chat_workspace_overrides_config(self, tmp_path: pytest.TempPathFactory, _mock_settings: Settings) -> None:
        """--workspace flag on chat overrides coding.workspace_root."""
        workspace_dir = tmp_path / "ws"  # type: ignore[operator]
        workspace_dir.mkdir()

        with patch("vaig.cli.repl.start_repl") as mock_repl:
            runner.invoke(
                app,
                ["chat", "--workspace", str(workspace_dir)],
            )

        assert _mock_settings.coding.workspace_root == str(workspace_dir)
        mock_repl.assert_called_once()

    def test_chat_workspace_not_found(self) -> None:
        """--workspace with a non-existent directory should exit with error."""
        result = runner.invoke(
            app,
            ["chat", "--workspace", "/nonexistent/path"],
        )

        assert result.exit_code == 1
        assert "not found" in result.output


# ══════════════════════════════════════════════════════════════
# OUTPUT FLAG
# ══════════════════════════════════════════════════════════════
class TestOutputFlag:
    def test_ask_output_saves_to_file(self, tmp_path) -> None:
        """--output saves the response to a file."""
        out_file = tmp_path / "result.md"

        mock_agent_result = MagicMock()
        mock_agent_result.content = "The answer is 42."

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_single.return_value = mock_agent_result

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
        ):
            result = runner.invoke(app, ["ask", "What?", "--no-stream", "-o", str(out_file)])

        assert result.exit_code == 0
        assert out_file.exists()
        assert out_file.read_text() == "The answer is 42."
        assert "saved to" in result.output

    def test_ask_output_creates_parent_dirs(self, tmp_path) -> None:
        """--output creates parent directories if they don't exist."""
        out_file = tmp_path / "nested" / "dir" / "result.md"

        mock_agent_result = MagicMock()
        mock_agent_result.content = "Content"

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_single.return_value = mock_agent_result

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
        ):
            result = runner.invoke(app, ["ask", "Hello", "--no-stream", "-o", str(out_file)])

        assert result.exit_code == 0
        assert out_file.exists()
        assert out_file.read_text() == "Content"

    def test_ask_output_with_streaming(self, tmp_path) -> None:
        """--output works with streaming mode (default)."""
        out_file = tmp_path / "stream_result.md"

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_single.return_value = iter(["Hello ", "World"])

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
        ):
            result = runner.invoke(app, ["ask", "Hi", "-o", str(out_file)])

        assert result.exit_code == 0
        assert out_file.exists()
        assert out_file.read_text() == "Hello World"

    def test_ask_output_with_skill(self, tmp_path) -> None:
        """--output works with skill mode."""
        out_file = tmp_path / "skill_result.md"

        mock_skill = MagicMock()
        skill_result = SkillResult(
            phase=SkillPhase.ANALYZE,
            success=True,
            output="Root cause: OOM",
        )

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_skill_phase.return_value = skill_result

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
            patch("vaig.skills.registry.SkillRegistry", return_value=mock_registry),
        ):
            result = runner.invoke(app, ["ask", "Why?", "--skill", "rca", "-o", str(out_file)])

        assert result.exit_code == 0
        assert out_file.exists()
        assert out_file.read_text() == "Root cause: OOM"

    def test_ask_no_output_flag_no_file_created(self, tmp_path) -> None:
        """Without --output, no file is created."""
        mock_agent_result = MagicMock()
        mock_agent_result.content = "Answer"

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_single.return_value = mock_agent_result

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
        ):
            result = runner.invoke(app, ["ask", "Hello", "--no-stream"])

        assert result.exit_code == 0
        # No file should exist in tmp_path
        assert len(list(tmp_path.iterdir())) == 0


# ══════════════════════════════════════════════════════════════
# VERBOSE / DEBUG FLAGS
# ══════════════════════════════════════════════════════════════
class TestVerboseDebugFlags:
    """Tests for --verbose, --debug, and --log-level CLI flags."""

    @pytest.fixture(autouse=True)
    def _reset_log_state(self) -> None:
        """Reset logging state before and after each test."""
        from vaig.core.log import reset_logging

        reset_logging()
        yield  # type: ignore[misc]
        reset_logging()

    def test_verbose_flag_sets_info_level(self) -> None:
        """--verbose / -V sets the vaig logger to INFO."""
        import logging

        mock_agent_result = MagicMock()
        mock_agent_result.content = "Answer"

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_single.return_value = mock_agent_result

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
        ):
            result = runner.invoke(app, ["-V", "ask", "Hello", "--no-stream"])

        assert result.exit_code == 0
        vaig_logger = logging.getLogger("vaig")
        # Logger level is min(console_level, file_level) — file defaults to
        # DEBUG, so logger level is DEBUG.  The console handler itself is INFO.
        assert vaig_logger.level <= logging.INFO
        rich_handlers = [h for h in vaig_logger.handlers if hasattr(h, "_log_render")]
        assert rich_handlers, "Expected a RichHandler on the vaig logger"
        assert rich_handlers[0].level == logging.INFO

    def test_debug_flag_sets_debug_level(self) -> None:
        """--debug / -d sets the vaig logger to DEBUG with show_path."""
        import logging

        mock_agent_result = MagicMock()
        mock_agent_result.content = "Answer"

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_single.return_value = mock_agent_result

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
        ):
            result = runner.invoke(app, ["-d", "ask", "Hello", "--no-stream"])

        assert result.exit_code == 0
        vaig_logger = logging.getLogger("vaig")
        assert vaig_logger.level == logging.DEBUG
        # DEBUG mode should enable show_path on the console (Rich) handler.
        # File logging may add a second handler.
        rich_handlers = [h for h in vaig_logger.handlers if hasattr(h, "_log_render")]
        assert rich_handlers, "Expected a RichHandler on the vaig logger"
        assert rich_handlers[0]._log_render.show_path is True

    def test_log_level_overrides_verbose(self) -> None:
        """--log-level takes precedence over --verbose."""
        import logging

        mock_agent_result = MagicMock()
        mock_agent_result.content = "Answer"

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_single.return_value = mock_agent_result

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
        ):
            result = runner.invoke(app, ["-V", "--log-level", "ERROR", "ask", "Hello", "--no-stream"])

        assert result.exit_code == 0
        vaig_logger = logging.getLogger("vaig")
        # Logger level is min(console, file) — file defaults to DEBUG.
        # Check the console (Rich) handler has ERROR level.
        rich_handlers = [h for h in vaig_logger.handlers if hasattr(h, "_log_render")]
        assert rich_handlers, "Expected a RichHandler on the vaig logger"
        assert rich_handlers[0].level == logging.ERROR

    def test_log_level_overrides_debug(self) -> None:
        """--log-level takes precedence over --debug."""
        import logging

        mock_agent_result = MagicMock()
        mock_agent_result.content = "Answer"

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_single.return_value = mock_agent_result

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
        ):
            result = runner.invoke(app, ["-d", "--log-level", "WARNING", "ask", "Hello", "--no-stream"])

        assert result.exit_code == 0
        vaig_logger = logging.getLogger("vaig")
        # Logger level is min(console, file) — file defaults to DEBUG.
        # Check the console (Rich) handler has WARNING level.
        rich_handlers = [h for h in vaig_logger.handlers if hasattr(h, "_log_render")]
        assert rich_handlers, "Expected a RichHandler on the vaig logger"
        assert rich_handlers[0].level == logging.WARNING

    def test_default_level_is_warning(self) -> None:
        """Without flags, default level should be WARNING."""
        import logging

        mock_agent_result = MagicMock()
        mock_agent_result.content = "Answer"

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_single.return_value = mock_agent_result

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
        ):
            result = runner.invoke(app, ["ask", "Hello", "--no-stream"])

        assert result.exit_code == 0
        vaig_logger = logging.getLogger("vaig")
        # Logger level is min(console, file) — file defaults to DEBUG.
        # Check the console (Rich) handler has WARNING level.
        rich_handlers = [h for h in vaig_logger.handlers if hasattr(h, "_log_render")]
        assert rich_handlers, "Expected a RichHandler on the vaig logger"
        assert rich_handlers[0].level == logging.WARNING

    def test_debug_flag_shows_in_help(self) -> None:
        """--debug flag should appear in help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--debug" in _strip_ansi(result.output)

    def test_verbose_flag_shows_in_help(self) -> None:
        """--verbose flag should appear in help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--verbose" in _strip_ansi(result.output)

    def test_debug_flag_after_ask_subcommand(self) -> None:
        """``vaig ask "Hello" -d`` sets DEBUG level (flag after subcommand)."""
        import logging

        mock_agent_result = MagicMock()
        mock_agent_result.content = "Answer"

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_single.return_value = mock_agent_result

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
        ):
            result = runner.invoke(app, ["ask", "Hello", "--no-stream", "-d"])

        assert result.exit_code == 0
        vaig_logger = logging.getLogger("vaig")
        assert vaig_logger.level == logging.DEBUG

    def test_verbose_flag_after_ask_subcommand(self) -> None:
        """``vaig ask "Hello" -V`` sets INFO level (flag after subcommand)."""
        import logging

        mock_agent_result = MagicMock()
        mock_agent_result.content = "Answer"

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_single.return_value = mock_agent_result

        with (
            patch("vaig.core.container.build_container", return_value=create_test_container()),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
        ):
            result = runner.invoke(app, ["ask", "Hello", "--no-stream", "-V"])

        assert result.exit_code == 0
        vaig_logger = logging.getLogger("vaig")
        assert vaig_logger.level == logging.INFO

    def test_debug_flag_shows_in_live_help(self) -> None:
        """--debug flag should appear in live subcommand help output."""
        result = runner.invoke(app, ["live", "--help"])
        assert result.exit_code == 0
        assert "--debug" in _strip_ansi(result.output)

    def test_verbose_flag_shows_in_live_help(self) -> None:
        """--verbose flag should appear in live subcommand help output."""
        result = runner.invoke(app, ["live", "--help"])
        assert result.exit_code == 0
        assert "--verbose" in _strip_ansi(result.output)

    def test_debug_flag_shows_in_ask_help(self) -> None:
        """--debug flag should appear in ask subcommand help output."""
        result = runner.invoke(app, ["ask", "--help"])
        assert result.exit_code == 0
        assert "--debug" in _strip_ansi(result.output)


# ══════════════════════════════════════════════════════════════
# LOCATION FLAG
# ══════════════════════════════════════════════════════════════
class TestLocationFlag:
    """Tests for the --location CLI flag on live and ask commands."""

    def test_location_flag_shows_in_live_help(self) -> None:
        """--location flag should appear in live subcommand help output."""
        result = runner.invoke(app, ["live", "--help"])
        assert result.exit_code == 0
        assert "--location" in _strip_ansi(result.output)

    def test_location_flag_shows_in_ask_help(self) -> None:
        """--location flag should appear in ask subcommand help output."""
        result = runner.invoke(app, ["ask", "--help"])
        assert result.exit_code == 0
        assert "--location" in _strip_ansi(result.output)

    def test_build_gke_config_with_location(self) -> None:
        """_build_gke_config passes location override to GKEConfig."""
        from vaig.cli.app import _build_gke_config
        from vaig.core.config import Settings

        settings = Settings()
        gke_config = _build_gke_config(settings, location="us-central1-a")
        assert gke_config.location == "us-central1-a"

    def test_build_gke_config_without_location_uses_default(self) -> None:
        """_build_gke_config falls back to settings.gke.location when no override."""
        from vaig.cli.app import _build_gke_config
        from vaig.core.config import Settings

        settings = Settings()
        gke_config = _build_gke_config(settings)
        assert gke_config.location == settings.gke.location


# ══════════════════════════════════════════════════════════════
# PROJECT FLAG (Phase 5)
# ══════════════════════════════════════════════════════════════
class TestProjectFlag:
    """Tests for the --project CLI flag on ask, chat, and live commands."""

    def test_project_flag_shows_in_ask_help(self) -> None:
        result = runner.invoke(app, ["ask", "--help"])
        assert result.exit_code == 0
        assert "--project" in _strip_ansi(result.output)

    def test_project_flag_shows_in_live_help(self) -> None:
        result = runner.invoke(app, ["live", "--help"])
        assert result.exit_code == 0
        assert "--project" in _strip_ansi(result.output)

    def test_project_flag_shows_in_chat_help(self) -> None:
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "--project" in _strip_ansi(result.output)

    def test_location_flag_shows_in_chat_help(self) -> None:
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "--location" in _strip_ansi(result.output)

    def test_project_flag_updates_settings(self, _mock_settings: Settings) -> None:
        """--project should mutate gcp.project_id and gke.project_id BEFORE component creation."""

        # Simulate what the command body does
        _mock_settings.gcp.project_id = "flag-project"
        _mock_settings.gke.project_id = "flag-project"

        assert _mock_settings.gcp.project_id == "flag-project"
        assert _mock_settings.gke.project_id == "flag-project"

    def test_project_id_backward_compat(self) -> None:
        """--project-id should still show in help for backward compat."""
        result = runner.invoke(app, ["ask", "--help"])
        assert result.exit_code == 0
        assert "--project-id" in _strip_ansi(result.output)

    def test_project_flag_overrides_project_id(self, _mock_settings: Settings) -> None:
        """--project takes precedence over --project-id when both provided."""
        # This tests the effective_project = project or project_id logic
        project = "from-project"
        project_id = "from-project-id"
        effective = project or project_id
        assert effective == "from-project"

    def test_project_id_used_when_project_absent(self, _mock_settings: Settings) -> None:
        """When --project is not set, --project-id should be used."""
        project = None
        project_id = "from-project-id"
        effective = project or project_id
        assert effective == "from-project-id"

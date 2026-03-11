"""Tests for the CLI app — Typer commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from vaig import __version__
from vaig.cli.app import app
from vaig.core.config import Settings
from vaig.skills.base import SkillMetadata, SkillPhase, SkillResult


runner = CliRunner()


@pytest.fixture(autouse=True)
def _mock_settings() -> Settings:
    """Provide a default Settings object to all CLI commands, avoiding real config."""
    settings = Settings()
    with patch("vaig.cli.app._get_settings", return_value=settings):
        yield settings


# ══════════════════════════════════════════════════════════════
# APP BASICS
# ══════════════════════════════════════════════════════════════
class TestAppBasics:
    def test_help_shows_help_text(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Vertex AI" in result.output or "VAIG" in result.output

    def test_version_flag(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_version_short_flag(self) -> None:
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_no_args_shows_help(self) -> None:
        result = runner.invoke(app, [])
        # no_args_is_help=True — Typer shows help and exits with code 0.
        # Some Typer/Click versions exit with 0, others with 2 for "no args" help.
        assert result.exit_code in (0, 2)
        assert "Usage" in result.output or "VAIG" in result.output


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
            patch("vaig.core.client.GeminiClient"),
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
            patch("vaig.core.client.GeminiClient"),
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
            patch("vaig.core.client.GeminiClient"),
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
            patch("vaig.core.client.GeminiClient"),
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

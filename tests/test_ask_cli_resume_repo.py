"""Tests for new CLI flags in vaig ask: --resume, --repo, --repo-ref."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from vaig.cli.app import app

runner = CliRunner()


class TestResumeFlagParsing:
    """Verify --resume flag is accepted by the CLI without error."""

    def test_resume_flag_is_accepted_by_cli(self) -> None:
        """--resume should be a known flag (no 'No such option' error)."""
        result = runner.invoke(app, ["ask", "--help"])
        assert "--resume" in result.output

    def test_resume_flag_triggers_skill_reinstantiation(self, tmp_path: pytest.MonkeyPatch) -> None:
        """When --resume and --skill code-migration are passed, CodeMigrationSkill(resume=True)."""
        with (
            patch("vaig.cli.commands.ask._helpers._get_settings") as mock_settings,
            patch("vaig.cli.commands.ask._helpers._init_telemetry"),
            patch("vaig.cli.commands.ask._helpers._init_audit"),
            patch("vaig.cli.commands.ask._helpers._check_platform_auth"),
            patch("vaig.skills.code_migration.CodeMigrationSkill") as mock_cm,
            patch("vaig.skills.registry.SkillRegistry") as mock_registry_cls,
            patch("vaig.agents.orchestrator.Orchestrator"),
            patch("vaig.core.container.build_container"),
        ):
            settings = MagicMock()
            settings.models.default = "gemini-2.5-flash"
            settings.skills.auto_routing = False
            mock_settings.return_value = settings

            mock_registry = MagicMock()
            mock_skill = MagicMock()
            mock_skill.get_metadata.return_value = MagicMock(name="code-migration")
            mock_registry.get.return_value = mock_skill
            mock_registry_cls.return_value = mock_registry

            fresh_skill = MagicMock()
            mock_cm.return_value = fresh_skill

            runner.invoke(
                app,
                ["ask", "migrate this", "--skill", "code-migration", "--resume"],
            )

            mock_cm.assert_called_once_with(resume=True)


class TestRepoFlagValidation:
    """--repo without a configured GitHub integration should exit with an error."""

    def test_repo_flag_without_github_enabled_exits(self) -> None:
        with (
            patch("vaig.cli.commands.ask._helpers._get_settings") as mock_settings,
            patch("vaig.cli.commands.ask._helpers._init_telemetry"),
            patch("vaig.cli.commands.ask._helpers._init_audit"),
            patch("vaig.cli.commands.ask._helpers._check_platform_auth"),
            patch("vaig.skills.registry.SkillRegistry") as mock_registry_cls,
            patch("vaig.agents.orchestrator.Orchestrator"),
            patch("vaig.core.container.build_container"),
        ):
            settings = MagicMock()
            settings.models.default = "gemini-2.5-flash"
            settings.skills.auto_routing = False
            settings.github.enabled = False
            mock_settings.return_value = settings

            mock_registry = MagicMock()
            mock_registry.get.return_value = MagicMock()
            mock_registry_cls.return_value = mock_registry

            result = runner.invoke(
                app,
                ["ask", "migrate this", "--skill", "rca", "--repo", "acme/myrepo"],
            )
            assert result.exit_code != 0

    def test_repo_flag_invalid_format_exits(self) -> None:
        with (
            patch("vaig.cli.commands.ask._helpers._get_settings") as mock_settings,
            patch("vaig.cli.commands.ask._helpers._init_telemetry"),
            patch("vaig.cli.commands.ask._helpers._init_audit"),
            patch("vaig.cli.commands.ask._helpers._check_platform_auth"),
            patch("vaig.skills.registry.SkillRegistry") as mock_registry_cls,
            patch("vaig.agents.orchestrator.Orchestrator"),
            patch("vaig.core.container.build_container"),
        ):
            settings = MagicMock()
            settings.models.default = "gemini-2.5-flash"
            settings.skills.auto_routing = False
            settings.github.enabled = True
            mock_settings.return_value = settings

            mock_registry = MagicMock()
            mock_registry.get.return_value = MagicMock()
            mock_registry_cls.return_value = mock_registry

            result = runner.invoke(
                app,
                ["ask", "migrate this", "--skill", "rca", "--repo", "no-slash-here"],
            )
            assert result.exit_code != 0

"""Tests for the --dry-run flag on the ``vaig live`` command.

Covers:
- Dry-run displays execution plan without running agents
- Dry-run shows GKE configuration, agents, tools, and cost estimate
- No API calls are made during dry-run
- Both orchestrated skill path and InfraAgent path are tested
- Dry-run with --skill flag shows agent pipeline
- Dry-run without skill shows InfraAgent
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from vaig.cli.app import app
from vaig.core.config import Settings
from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.tools.base import ToolDef, ToolRegistry

runner = CliRunner()


@pytest.fixture(autouse=True)
def _mock_settings() -> Settings:
    """Provide a default Settings object to all CLI commands, avoiding real config."""
    settings = Settings()
    # Disable auto-routing so no-skill tests don't accidentally route
    settings.skills.auto_routing = False
    with patch("vaig.cli._helpers._get_settings", return_value=settings):
        yield settings


def _make_tool_registry(*names: str) -> ToolRegistry:
    """Create a ToolRegistry with stub tools for the given names."""
    reg = ToolRegistry()
    for name in names:
        reg.register(ToolDef(name=name, description=f"Mock {name} tool"))
    return reg


class FakeOrchSkill(BaseSkill):
    """Fake skill with requires_live_tools=True for orchestrated path tests."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="fake-orch",
            display_name="Fake Orchestrated Skill",
            description="A test skill for dry-run testing",
            requires_live_tools=True,
        )

    def get_system_instruction(self) -> str:
        return "You are a test agent."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return f"phase={phase}, context={context}, input={user_input}"

    def get_agents_config(self) -> list[dict]:
        return [
            {
                "name": "gatherer",
                "role": "Data Gatherer",
                "requires_tools": True,
                "system_instruction": "Gather data",
                "model": "gemini-2.5-pro",
                "temperature": 0.2,
            },
            {
                "name": "analyzer",
                "role": "Pattern Analyzer",
                "requires_tools": False,
                "system_instruction": "Analyze patterns",
                "model": "gemini-2.5-flash",
                "temperature": 0.3,
            },
        ]


class FakeContextSkill(BaseSkill):
    """Fake skill with requires_live_tools=False (context-prepend path)."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="fake-ctx",
            display_name="Fake Context Skill",
            description="A context-only skill",
            requires_live_tools=False,
        )

    def get_system_instruction(self) -> str:
        return "Context skill"

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return ""

    def get_agents_config(self) -> list[dict]:
        return []


# ══════════════════════════════════════════════════════════════
# DRY-RUN — InfraAgent path (no skill)
# ══════════════════════════════════════════════════════════════
class TestDryRunNoSkill:
    """Dry-run without any skill — should show InfraAgent path."""

    @patch(
        "vaig.cli.commands.live._register_live_tools",
        return_value=_make_tool_registry("kubectl_get", "kubectl_describe", "kubectl_logs"),
    )
    @patch("vaig.core.client.GeminiClient")
    def test_dry_run_shows_plan(self, mock_client_cls: MagicMock, mock_tools: MagicMock) -> None:
        """--dry-run should show execution plan and exit with code 0."""
        result = runner.invoke(app, ["live", "Check pod health", "--dry-run"])
        assert result.exit_code == 0
        assert "Dry Run" in result.output
        assert "InfraAgent" in result.output

    @patch(
        "vaig.cli.commands.live._register_live_tools",
        return_value=_make_tool_registry("kubectl_get"),
    )
    @patch("vaig.core.client.GeminiClient")
    def test_dry_run_shows_configuration(self, mock_client_cls: MagicMock, mock_tools: MagicMock) -> None:
        """Dry-run should display GKE configuration."""
        result = runner.invoke(app, ["live", "Check pods", "--dry-run", "--namespace", "production"])
        assert result.exit_code == 0
        assert "Configuration" in result.output
        assert "production" in result.output

    @patch(
        "vaig.cli.commands.live._register_live_tools",
        return_value=_make_tool_registry("kubectl_get", "kubectl_describe"),
    )
    @patch("vaig.core.client.GeminiClient")
    def test_dry_run_shows_tools(self, mock_client_cls: MagicMock, mock_tools: MagicMock) -> None:
        """Dry-run should list available tools."""
        result = runner.invoke(app, ["live", "Check pods", "--dry-run"])
        assert result.exit_code == 0
        assert "kubectl_get" in result.output
        assert "kubectl_describe" in result.output

    @patch(
        "vaig.cli.commands.live._register_live_tools",
        return_value=_make_tool_registry("kubectl_get"),
    )
    @patch("vaig.core.client.GeminiClient")
    def test_dry_run_shows_cost_estimate(self, mock_client_cls: MagicMock, mock_tools: MagicMock) -> None:
        """Dry-run should show estimated cost note."""
        result = runner.invoke(app, ["live", "Check pods", "--dry-run"])
        assert result.exit_code == 0
        assert "Estimated cost" in result.output
        assert "without --dry-run" in result.output

    @patch(
        "vaig.cli.commands.live._register_live_tools",
        return_value=_make_tool_registry("kubectl_get"),
    )
    @patch("vaig.core.client.GeminiClient")
    def test_dry_run_no_api_calls(self, mock_client_cls: MagicMock, mock_tools: MagicMock) -> None:
        """Dry-run must NOT create any agents or make API calls."""
        result = runner.invoke(app, ["live", "Check pods", "--dry-run"])
        assert result.exit_code == 0
        # GeminiClient is instantiated but no methods should be called on it
        client_instance = mock_client_cls.return_value
        client_instance.generate.assert_not_called()
        client_instance.generate_with_tools.assert_not_called()

    @patch(
        "vaig.cli.commands.live._register_live_tools",
        return_value=_make_tool_registry(),
    )
    @patch("vaig.core.client.GeminiClient")
    def test_dry_run_no_tools_available(self, mock_client_cls: MagicMock, mock_tools: MagicMock) -> None:
        """Dry-run should warn when no tools are available."""
        result = runner.invoke(app, ["live", "Check pods", "--dry-run"])
        assert result.exit_code == 0
        assert "No infrastructure tools" in result.output


# ══════════════════════════════════════════════════════════════
# DRY-RUN — Orchestrated skill path
# ══════════════════════════════════════════════════════════════
class TestDryRunOrchSkill:
    """Dry-run with an orchestrated skill (requires_live_tools=True)."""

    @patch(
        "vaig.cli.commands.live._register_live_tools",
        return_value=_make_tool_registry("kubectl_get", "kubectl_describe", "gcloud_logging_query"),
    )
    @patch("vaig.core.client.GeminiClient")
    def test_dry_run_shows_agent_pipeline(self, mock_client_cls: MagicMock, mock_tools: MagicMock) -> None:
        """Dry-run with orchestrated skill should show the agent pipeline table."""
        fake_skill = FakeOrchSkill()
        fake_registry = MagicMock()
        fake_registry.get.return_value = fake_skill

        with patch("vaig.skills.registry.SkillRegistry", return_value=fake_registry):
            result = runner.invoke(app, ["live", "Check service health", "--dry-run", "--skill", "fake-orch"])

        assert result.exit_code == 0
        assert "Agents that would be created" in result.output
        assert "gatherer" in result.output
        assert "analyzer" in result.output
        assert "Data Gatherer" in result.output

    @patch(
        "vaig.cli.commands.live._register_live_tools",
        return_value=_make_tool_registry("kubectl_get", "kubectl_describe"),
    )
    @patch("vaig.core.client.GeminiClient")
    def test_dry_run_skill_shows_tool_info(self, mock_client_cls: MagicMock, mock_tools: MagicMock) -> None:
        """Dry-run with skill should still show tool count."""
        fake_skill = FakeOrchSkill()
        fake_registry = MagicMock()
        fake_registry.get.return_value = fake_skill

        with patch("vaig.skills.registry.SkillRegistry", return_value=fake_registry):
            result = runner.invoke(app, ["live", "Check health", "--dry-run", "--skill", "fake-orch"])

        assert result.exit_code == 0
        assert "Available tools (2)" in result.output
        assert "kubectl_get" in result.output

    @patch(
        "vaig.cli.commands.live._register_live_tools",
        return_value=_make_tool_registry("kubectl_get"),
    )
    @patch("vaig.core.client.GeminiClient")
    def test_dry_run_skill_no_execution(self, mock_client_cls: MagicMock, mock_tools: MagicMock) -> None:
        """Dry-run with skill must NOT call execute_with_tools."""
        fake_skill = FakeOrchSkill()
        fake_registry = MagicMock()
        fake_registry.get.return_value = fake_skill

        with (
            patch("vaig.skills.registry.SkillRegistry", return_value=fake_registry),
            patch("vaig.cli.commands.live._execute_orchestrated_skill") as mock_exec,
        ):
            result = runner.invoke(app, ["live", "Check health", "--dry-run", "--skill", "fake-orch"])

        assert result.exit_code == 0
        mock_exec.assert_not_called()


# ══════════════════════════════════════════════════════════════
# DRY-RUN — Context-prepend skill path (requires_live_tools=False)
# ══════════════════════════════════════════════════════════════
class TestDryRunContextSkill:
    """Dry-run with a context-prepend skill should show InfraAgent path."""

    @patch(
        "vaig.cli.commands.live._register_live_tools",
        return_value=_make_tool_registry("kubectl_get"),
    )
    @patch("vaig.core.client.GeminiClient")
    def test_dry_run_context_skill_shows_infra_agent(
        self, mock_client_cls: MagicMock, mock_tools: MagicMock
    ) -> None:
        """Context-prepend skill should show InfraAgent (not orchestrated pipeline)."""
        fake_skill = FakeContextSkill()
        fake_registry = MagicMock()
        fake_registry.get.return_value = fake_skill

        with patch("vaig.skills.registry.SkillRegistry", return_value=fake_registry):
            result = runner.invoke(app, ["live", "Do analysis", "--dry-run", "--skill", "fake-ctx"])

        assert result.exit_code == 0
        assert "InfraAgent" in result.output
        assert "Agents that would be created" not in result.output


# ══════════════════════════════════════════════════════════════
# DRY-RUN — --dry alias
# ══════════════════════════════════════════════════════════════
class TestDryRunAlias:
    """The --dry alias should behave identically to --dry-run."""

    @patch(
        "vaig.cli.commands.live._register_live_tools",
        return_value=_make_tool_registry("kubectl_get"),
    )
    @patch("vaig.core.client.GeminiClient")
    def test_dry_alias(self, mock_client_cls: MagicMock, mock_tools: MagicMock) -> None:
        """--dry should work the same as --dry-run."""
        result = runner.invoke(app, ["live", "Check pods", "--dry"])
        assert result.exit_code == 0
        assert "Dry Run" in result.output


# ══════════════════════════════════════════════════════════════
# DRY-RUN — Many tools truncation
# ══════════════════════════════════════════════════════════════
class TestDryRunManyTools:
    """When more than 10 tools are available, display should truncate."""

    @patch("vaig.core.client.GeminiClient")
    def test_many_tools_truncated(self, mock_client_cls: MagicMock) -> None:
        """With >10 tools, dry-run should show first 10 + 'and N more'."""
        tool_names = [f"tool_{i}" for i in range(15)]
        registry = _make_tool_registry(*tool_names)

        with patch("vaig.cli.commands.live._register_live_tools", return_value=registry):
            result = runner.invoke(app, ["live", "Check pods", "--dry-run"])

        assert result.exit_code == 0
        assert "Available tools (15)" in result.output
        assert "and 5 more" in result.output

"""Tests for CLI → Orchestrator tool-aware skill routing.

Covers the routing logic in the ``live`` command that dispatches skills with
``requires_live_tools=True`` through ``Orchestrator.execute_with_tools()``
instead of the legacy context-prepend approach via ``_execute_live_mode()``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from vaig.cli.app import (
    _register_live_tools,
    app,
)
from vaig.core.config import Settings
from vaig.skills.base import SkillMetadata, SkillPhase

runner = CliRunner()


@pytest.fixture(autouse=True)
def _mock_settings() -> Settings:
    """Provide a default Settings object, avoiding real config."""
    settings = Settings()
    with patch("vaig.cli._helpers._get_settings", return_value=settings):
        yield settings


def _make_skill_mock(*, requires_live_tools: bool = False) -> MagicMock:
    """Create a mock skill with configurable requires_live_tools."""
    skill = MagicMock()
    skill.get_metadata.return_value = SkillMetadata(
        name="test-skill",
        display_name="Test Skill",
        description="A test skill for CLI routing",
        requires_live_tools=requires_live_tools,
    )
    return skill


# ══════════════════════════════════════════════════════════════
# _register_live_tools — unit tests
# ══════════════════════════════════════════════════════════════
class TestRegisterLiveTools:
    """Tests for the _register_live_tools helper."""

    def test_returns_empty_registry_when_no_deps(self) -> None:
        """When both gke_tools and gcloud_tools fail to import, registry is empty."""
        gke_config = MagicMock()

        with (
            patch("vaig.cli.commands.live.create_gke_tools", side_effect=ImportError, create=True),
            patch("vaig.cli.commands.live.create_gcloud_tools", side_effect=ImportError, create=True),
            patch.dict("sys.modules", {"vaig.tools.gke_tools": None, "vaig.tools.gcloud_tools": None}),
        ):
            # Force ImportError by patching the import target
            with patch("builtins.__import__", side_effect=_import_blocker(["vaig.tools.gke_tools", "vaig.tools.gcloud_tools"])):
                registry = _register_live_tools(gke_config)

        assert len(registry.list_tools()) == 0

    def test_registers_gke_tools_when_available(self) -> None:
        """GKE tools are registered when the kubernetes package is installed."""
        gke_config = MagicMock()

        mock_tool = MagicMock()
        mock_tool.name = "kubectl_get"
        mock_create_gke = MagicMock(return_value=[mock_tool])

        mock_gke_module = MagicMock()
        mock_gke_module.create_gke_tools = mock_create_gke

        with patch.dict("sys.modules", {"vaig.tools.gke_tools": mock_gke_module}):
            # Block gcloud imports to isolate GKE
            with patch("builtins.__import__", side_effect=_import_blocker(["vaig.tools.gcloud_tools"])):
                registry = _register_live_tools(gke_config)

        assert len(registry.list_tools()) == 1
        assert registry.get("kubectl_get") is not None
        mock_create_gke.assert_called_once_with(gke_config)

    def test_registers_gcloud_tools_when_available(self) -> None:
        """GCloud tools are registered when google-cloud packages are installed."""
        gke_config = MagicMock()
        gke_config.project_id = "test-project"
        gke_config.log_limit = 100
        gke_config.metrics_interval_minutes = 30

        mock_tool = MagicMock()
        mock_tool.name = "query_cloud_logging"
        mock_create_gcloud = MagicMock(return_value=[mock_tool])

        mock_gcloud_module = MagicMock()
        mock_gcloud_module.create_gcloud_tools = mock_create_gcloud

        with (
            patch.dict("sys.modules", {"vaig.tools.gcloud_tools": mock_gcloud_module}),
            patch("builtins.__import__", side_effect=_import_blocker(["vaig.tools.gke_tools"])),
        ):
            registry = _register_live_tools(gke_config)

        assert len(registry.list_tools()) == 1
        assert registry.get("query_cloud_logging") is not None
        mock_create_gcloud.assert_called_once_with(
            project="test-project",
            log_limit=100,
            metrics_interval_minutes=30,
            credentials=None,
        )


def _import_blocker(blocked: list[str]):
    """Return a __import__ replacement that blocks specific modules."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _blocker(name, *args, **kwargs):
        if name in blocked:
            raise ImportError(f"Blocked for test: {name}")
        return real_import(name, *args, **kwargs)

    return _blocker


# ══════════════════════════════════════════════════════════════
# live --skill routing — integration tests
# ══════════════════════════════════════════════════════════════
class TestLiveSkillRouting:
    """Tests for the routing logic in the ``live`` command."""

    def test_requires_live_tools_true_routes_to_orchestrator(self) -> None:
        """Skills with requires_live_tools=True go through Orchestrator.execute_with_tools."""
        skill = _make_skill_mock(requires_live_tools=True)

        mock_registry = MagicMock()
        mock_registry.get.return_value = skill

        mock_orch_result = MagicMock()
        mock_orch_result.synthesized_output = "Service health looks good"
        mock_orch_result.success = True
        mock_orch_result.agent_results = []

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_with_tools.return_value = mock_orch_result

        # Mock _register_live_tools to return a non-empty registry
        mock_tool_registry = MagicMock()
        mock_tool_registry.list_tools.return_value = [MagicMock()] * 4

        with (
            patch("vaig.core.client.GeminiClient"),
            patch("vaig.skills.registry.SkillRegistry", return_value=mock_registry),
            patch("vaig.cli.commands.live._register_live_tools", return_value=mock_tool_registry),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
        ):
            result = runner.invoke(app, ["live", "Check service health", "--skill", "test-skill"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "Service health looks good" in result.output
        mock_orchestrator.execute_with_tools.assert_called_once()
        call_kwargs = mock_orchestrator.execute_with_tools.call_args
        assert call_kwargs.kwargs.get("strategy") == "sequential" or call_kwargs[1].get("strategy") == "sequential"

    def test_requires_live_tools_false_routes_to_infra_agent(self) -> None:
        """Skills with requires_live_tools=False use the legacy context-prepend path."""
        skill = _make_skill_mock(requires_live_tools=False)

        mock_registry = MagicMock()
        mock_registry.get.return_value = skill

        mock_agent_result = MagicMock()
        mock_agent_result.content = "Investigation complete"
        mock_agent_result.success = True
        mock_agent_result.metadata = {}
        mock_agent_result.usage = {}

        mock_infra_agent = MagicMock()
        mock_infra_agent.execute.return_value = mock_agent_result
        mock_infra_agent.registry.list_tools.return_value = [MagicMock()] * 3

        with (
            patch("vaig.core.client.GeminiClient"),
            patch("vaig.skills.registry.SkillRegistry", return_value=mock_registry),
            patch("vaig.agents.infra_agent.InfraAgent", return_value=mock_infra_agent),
        ):
            result = runner.invoke(app, ["live", "Check pods", "--skill", "test-skill"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        # Legacy path: InfraAgent.execute() is called, NOT orchestrator
        mock_infra_agent.execute.assert_called_once()

    def test_live_without_skill_uses_infra_agent(self) -> None:
        """``vaig live`` without --skill uses InfraAgent directly."""
        mock_agent_result = MagicMock()
        mock_agent_result.content = "Pods are healthy"
        mock_agent_result.success = True
        mock_agent_result.metadata = {}
        mock_agent_result.usage = {}

        mock_infra_agent = MagicMock()
        mock_infra_agent.execute.return_value = mock_agent_result
        mock_infra_agent.registry.list_tools.return_value = [MagicMock()] * 3

        with (
            patch("vaig.core.client.GeminiClient"),
            patch("vaig.agents.infra_agent.InfraAgent", return_value=mock_infra_agent),
        ):
            result = runner.invoke(app, ["live", "What pods are running?"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        mock_infra_agent.execute.assert_called_once()

    def test_unknown_skill_shows_error(self) -> None:
        """An unknown skill name shows error with available skill names."""
        mock_registry = MagicMock()
        mock_registry.get.return_value = None
        mock_registry.list_names.return_value = ["rca", "service-health"]

        with (
            patch("vaig.core.client.GeminiClient"),
            patch("vaig.skills.registry.SkillRegistry", return_value=mock_registry),
        ):
            result = runner.invoke(app, ["live", "Check something", "--skill", "nonexistent"])

        assert result.exit_code == 1
        assert "Skill not found" in result.output

    def test_no_tools_available_shows_error(self) -> None:
        """When requires_live_tools=True but no tools can be loaded, exit with error."""
        skill = _make_skill_mock(requires_live_tools=True)

        mock_registry = MagicMock()
        mock_registry.get.return_value = skill

        # Empty tool registry — no tools available
        mock_tool_registry = MagicMock()
        mock_tool_registry.list_tools.return_value = []

        with (
            patch("vaig.core.client.GeminiClient"),
            patch("vaig.skills.registry.SkillRegistry", return_value=mock_registry),
            patch("vaig.cli.commands.live._register_live_tools", return_value=mock_tool_registry),
        ):
            result = runner.invoke(app, ["live", "Check health", "--skill", "test-skill"])

        assert result.exit_code == 1
        assert "No infrastructure tools available" in result.output

    def test_output_saved_to_file(self, tmp_path) -> None:
        """Orchestrated skill output is saved to file when --output is given."""
        skill = _make_skill_mock(requires_live_tools=True)

        mock_registry = MagicMock()
        mock_registry.get.return_value = skill

        mock_orch_result = MagicMock()
        mock_orch_result.synthesized_output = "Report: all services healthy"
        mock_orch_result.success = True
        mock_orch_result.agent_results = []

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_with_tools.return_value = mock_orch_result

        mock_tool_registry = MagicMock()
        mock_tool_registry.list_tools.return_value = [MagicMock()] * 4

        out_file = tmp_path / "report.md"

        with (
            patch("vaig.core.client.GeminiClient"),
            patch("vaig.skills.registry.SkillRegistry", return_value=mock_registry),
            patch("vaig.cli.commands.live._register_live_tools", return_value=mock_tool_registry),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
        ):
            result = runner.invoke(
                app,
                ["live", "Check health", "--skill", "test-skill", "--output", str(out_file)],
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert out_file.exists()
        assert "Report: all services healthy" in out_file.read_text()

    def test_pipeline_summary_shows_agent_results(self) -> None:
        """The pipeline summary table is displayed after orchestrated execution."""
        skill = _make_skill_mock(requires_live_tools=True)

        mock_registry = MagicMock()
        mock_registry.get.return_value = skill

        agent_result_1 = MagicMock()
        agent_result_1.agent_name = "data-collector"
        agent_result_1.success = True
        agent_result_1.metadata = {"role": "collector"}

        agent_result_2 = MagicMock()
        agent_result_2.agent_name = "analyzer"
        agent_result_2.success = True
        agent_result_2.metadata = {"role": "analyst"}

        mock_orch_result = MagicMock()
        mock_orch_result.synthesized_output = "All good"
        mock_orch_result.success = True
        mock_orch_result.agent_results = [agent_result_1, agent_result_2]

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_with_tools.return_value = mock_orch_result

        mock_tool_registry = MagicMock()
        mock_tool_registry.list_tools.return_value = [MagicMock()] * 4

        with (
            patch("vaig.core.client.GeminiClient"),
            patch("vaig.skills.registry.SkillRegistry", return_value=mock_registry),
            patch("vaig.cli.commands.live._register_live_tools", return_value=mock_tool_registry),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
        ):
            result = runner.invoke(app, ["live", "Check health", "--skill", "test-skill"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "Pipeline Summary" in result.output
        assert "data-collector" in result.output
        assert "analyzer" in result.output

    def test_failed_pipeline_shows_warning(self) -> None:
        """When the orchestrated pipeline fails, a warning is displayed."""
        skill = _make_skill_mock(requires_live_tools=True)

        mock_registry = MagicMock()
        mock_registry.get.return_value = skill

        failed_result = MagicMock()
        failed_result.agent_name = "collector"
        failed_result.success = False
        failed_result.metadata = {}

        mock_orch_result = MagicMock()
        mock_orch_result.synthesized_output = "Partial results"
        mock_orch_result.success = False
        mock_orch_result.agent_results = [failed_result]

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_with_tools.return_value = mock_orch_result

        mock_tool_registry = MagicMock()
        mock_tool_registry.list_tools.return_value = [MagicMock()] * 4

        with (
            patch("vaig.core.client.GeminiClient"),
            patch("vaig.skills.registry.SkillRegistry", return_value=mock_registry),
            patch("vaig.cli.commands.live._register_live_tools", return_value=mock_tool_registry),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
        ):
            result = runner.invoke(app, ["live", "Check health", "--skill", "test-skill"])

        assert result.exit_code == 0  # CLI doesn't exit non-zero for pipeline failures
        assert "Pipeline completed with errors" in result.output


    def test_max_iterations_error_handled_gracefully(self) -> None:
        """MaxIterationsError from orchestrated pipeline is caught and exits cleanly."""
        from vaig.core.exceptions import MaxIterationsError

        skill = _make_skill_mock(requires_live_tools=True)

        mock_registry = MagicMock()
        mock_registry.get.return_value = skill

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_with_tools.side_effect = MaxIterationsError(
            "Max iterations exceeded", iterations=15,
        )

        mock_tool_registry = MagicMock()
        mock_tool_registry.list_tools.return_value = [MagicMock()] * 4

        with (
            patch("vaig.core.client.GeminiClient"),
            patch("vaig.skills.registry.SkillRegistry", return_value=mock_registry),
            patch("vaig.cli.commands.live._register_live_tools", return_value=mock_tool_registry),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
        ):
            result = runner.invoke(app, ["live", "Check health", "--skill", "test-skill"])

        assert result.exit_code == 1
        assert "Max iterations reached" in result.output


class TestDualToolRegistration:
    """Tests for simultaneous GKE + GCloud tool registration."""

    def test_both_gke_and_gcloud_tools_registered(self) -> None:
        """When both GKE and GCloud packages are available, all tools appear in registry."""
        gke_config = MagicMock()
        gke_config.project_id = "test-project"
        gke_config.log_limit = 100
        gke_config.metrics_interval_minutes = 30

        gke_tool = MagicMock()
        gke_tool.name = "kubectl_get"
        mock_create_gke = MagicMock(return_value=[gke_tool])

        gcloud_tool = MagicMock()
        gcloud_tool.name = "query_cloud_logging"
        mock_create_gcloud = MagicMock(return_value=[gcloud_tool])

        mock_gke_module = MagicMock()
        mock_gke_module.create_gke_tools = mock_create_gke

        mock_gcloud_module = MagicMock()
        mock_gcloud_module.create_gcloud_tools = mock_create_gcloud

        with patch.dict("sys.modules", {
            "vaig.tools.gke_tools": mock_gke_module,
            "vaig.tools.gcloud_tools": mock_gcloud_module,
        }):
            registry = _register_live_tools(gke_config)

        assert len(registry.list_tools()) == 2
        assert registry.get("kubectl_get") is not None
        assert registry.get("query_cloud_logging") is not None
        mock_create_gke.assert_called_once_with(gke_config)
        mock_create_gcloud.assert_called_once_with(
            project="test-project",
            log_limit=100,
            metrics_interval_minutes=30,
            credentials=None,
        )

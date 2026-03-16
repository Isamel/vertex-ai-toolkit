"""Tests for REPL config switching commands — /project, /location, /cluster, /config.

Tests Phases 3+4+6 of the runtime-config-switching change:
- Phase 3: REPL slash commands for config switching
- Phase 4: Tool/agent recreation (warning displayed)
- Phase 6: Prompt prefix shows active project
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from vaig.core.config import (
    GCPConfig,
    GKEConfig,
    ModelsConfig,
    ProjectEntry,
    Settings,
)

# ── Fixtures ─────────────────────────────────────────────────
# _reset_settings is provided by conftest.py (autouse)
# mock_client is provided by conftest.py


@pytest.fixture()
def settings() -> Settings:
    """Minimal settings for config command tests."""
    return Settings(
        gcp=GCPConfig(
            project_id="test-project",
            location="us-central1",
            available_projects=[
                ProjectEntry(project_id="test-project", description="Test"),
                ProjectEntry(project_id="other-project", description="Other"),
            ],
        ),
        gke=GKEConfig(
            cluster_name="test-cluster",
            project_id="test-project",
            location="us-central1",
            context="test-context",
        ),
        models=ModelsConfig(default="gemini-2.5-pro"),
    )


@pytest.fixture()
def repl_state(settings: Settings, mock_client: MagicMock) -> MagicMock:
    """Create a mock REPLState-like object for command handlers."""
    state = MagicMock()
    state.settings = settings
    state.client = mock_client
    return state


# ══════════════════════════════════════════════════════════════
# Phase 3.1: /project command
# ══════════════════════════════════════════════════════════════


class TestCmdProject:
    def test_project_no_args_shows_current(self, repl_state: MagicMock) -> None:
        """Task 3.1: /project with no args shows current project."""
        from vaig.cli.repl import _cmd_project

        # Capture console output
        buf = StringIO()
        test_console = Console(file=buf, no_color=True, width=120)
        with patch("vaig.cli.repl.console", test_console):
            _cmd_project(repl_state, "")

        output = buf.getvalue()
        assert "test-project" in output
        assert "Usage" in output or "project" in output.lower()

    def test_project_switch_success(self, repl_state: MagicMock) -> None:
        """Task 3.1: /project <id> switches project and shows result."""
        from vaig.cli.repl import _cmd_project

        buf = StringIO()
        test_console = Console(file=buf, no_color=True, width=120)
        with patch("vaig.cli.repl.console", test_console):
            _cmd_project(repl_state, "new-project")

        output = buf.getvalue()
        assert "new-project" in output
        assert repl_state.settings.gcp.project_id == "new-project"
        assert repl_state.settings.gke.project_id == "new-project"
        repl_state.client.reinitialize.assert_called_once()

    def test_project_switch_empty_fails(self, repl_state: MagicMock) -> None:
        """Task 3.1: /project with empty arg shows usage."""
        from vaig.cli.repl import _cmd_project

        buf = StringIO()
        test_console = Console(file=buf, no_color=True, width=120)
        with patch("vaig.cli.repl.console", test_console):
            _cmd_project(repl_state, "")

        assert repl_state.settings.gcp.project_id == "test-project"

    def test_project_switch_shows_stale_tools_warning(
        self, repl_state: MagicMock,
    ) -> None:
        """Task 3.1 + 4.1: After switch, warns about stale tools."""
        from vaig.cli.repl import _cmd_project

        buf = StringIO()
        test_console = Console(file=buf, no_color=True, width=120)
        with patch("vaig.cli.repl.console", test_console):
            _cmd_project(repl_state, "other-project")

        output = buf.getvalue()
        assert "tools" in output.lower() or "agents" in output.lower()

    def test_project_switch_client_failure(self, repl_state: MagicMock) -> None:
        """Task 3.1: Client reinit failure shows error and rolls back."""
        from vaig.cli.repl import _cmd_project

        repl_state.client.reinitialize.side_effect = Exception("connection failed")
        buf = StringIO()
        test_console = Console(file=buf, no_color=True, width=120)
        with patch("vaig.cli.repl.console", test_console):
            _cmd_project(repl_state, "bad-project")

        output = buf.getvalue()
        assert "reinitialize" in output.lower() or "failed" in output.lower()
        # Rolled back
        assert repl_state.settings.gcp.project_id == "test-project"

    def test_project_shows_available_projects(self, repl_state: MagicMock) -> None:
        """Task 3.1: /project with no args lists available projects."""
        from vaig.cli.repl import _cmd_project

        buf = StringIO()
        test_console = Console(file=buf, no_color=True, width=120)
        with patch("vaig.cli.repl.console", test_console):
            _cmd_project(repl_state, "")

        output = buf.getvalue()
        assert "test-project" in output
        assert "other-project" in output


# ══════════════════════════════════════════════════════════════
# Phase 3.2: /location command
# ══════════════════════════════════════════════════════════════


class TestCmdLocation:
    def test_location_no_args_shows_current(self, repl_state: MagicMock) -> None:
        from vaig.cli.repl import _cmd_location

        buf = StringIO()
        test_console = Console(file=buf, no_color=True, width=120)
        with patch("vaig.cli.repl.console", test_console):
            _cmd_location(repl_state, "")

        output = buf.getvalue()
        assert "us-central1" in output

    def test_location_switch_success(self, repl_state: MagicMock) -> None:
        from vaig.cli.repl import _cmd_location

        buf = StringIO()
        test_console = Console(file=buf, no_color=True, width=120)
        with patch("vaig.cli.repl.console", test_console):
            _cmd_location(repl_state, "europe-west1")

        output = buf.getvalue()
        assert "europe-west1" in output
        assert repl_state.settings.gcp.location == "europe-west1"
        repl_state.client.reinitialize.assert_called_once()

    def test_location_switch_same_noop(self, repl_state: MagicMock) -> None:
        from vaig.cli.repl import _cmd_location

        buf = StringIO()
        test_console = Console(file=buf, no_color=True, width=120)
        with patch("vaig.cli.repl.console", test_console):
            _cmd_location(repl_state, "us-central1")

        output = buf.getvalue()
        assert "already" in output.lower()

    def test_location_client_failure_rolls_back(self, repl_state: MagicMock) -> None:
        from vaig.cli.repl import _cmd_location

        repl_state.client.reinitialize.side_effect = Exception("fail")
        buf = StringIO()
        test_console = Console(file=buf, no_color=True, width=120)
        with patch("vaig.cli.repl.console", test_console):
            _cmd_location(repl_state, "asia-east1")

        assert repl_state.settings.gcp.location == "us-central1"


# ══════════════════════════════════════════════════════════════
# Phase 3.3: /cluster command
# ══════════════════════════════════════════════════════════════


class TestCmdCluster:
    def test_cluster_no_args_shows_current(self, repl_state: MagicMock) -> None:
        from vaig.cli.repl import _cmd_cluster

        buf = StringIO()
        test_console = Console(file=buf, no_color=True, width=120)
        with patch("vaig.cli.repl.console", test_console):
            _cmd_cluster(repl_state, "")

        output = buf.getvalue()
        assert "test-cluster" in output

    def test_cluster_switch_success(self, repl_state: MagicMock) -> None:
        from vaig.cli.repl import _cmd_cluster

        buf = StringIO()
        test_console = Console(file=buf, no_color=True, width=120)
        with patch("vaig.cli.repl.console", test_console):
            _cmd_cluster(repl_state, "new-cluster")

        output = buf.getvalue()
        assert "new-cluster" in output
        assert repl_state.settings.gke.cluster_name == "new-cluster"

    def test_cluster_switch_with_context(self, repl_state: MagicMock) -> None:
        from vaig.cli.repl import _cmd_cluster

        buf = StringIO()
        test_console = Console(file=buf, no_color=True, width=120)
        with patch("vaig.cli.repl.console", test_console):
            _cmd_cluster(repl_state, "new-cluster new-context")

        assert repl_state.settings.gke.cluster_name == "new-cluster"
        assert repl_state.settings.gke.context == "new-context"

    def test_cluster_shows_stale_tools_warning(self, repl_state: MagicMock) -> None:
        from vaig.cli.repl import _cmd_cluster

        buf = StringIO()
        test_console = Console(file=buf, no_color=True, width=120)
        with patch("vaig.cli.repl.console", test_console):
            _cmd_cluster(repl_state, "new-cluster")

        output = buf.getvalue()
        assert "tools" in output.lower()


# ══════════════════════════════════════════════════════════════
# Phase 3.4: /config command
# ══════════════════════════════════════════════════════════════


class TestCmdConfig:
    def test_config_shows_all_settings(self, repl_state: MagicMock) -> None:
        from vaig.cli.repl import _cmd_config

        buf = StringIO()
        test_console = Console(file=buf, no_color=True, width=120)
        with patch("vaig.cli.repl.console", test_console):
            _cmd_config(repl_state)

        output = buf.getvalue()
        assert "test-project" in output
        assert "us-central1" in output
        assert "gemini-2.5-pro" in output
        assert "test-cluster" in output

    def test_config_reflects_mutations(self, repl_state: MagicMock) -> None:
        """Config snapshot should reflect runtime changes."""
        from vaig.cli.repl import _cmd_config
        from vaig.core.config_switcher import switch_project

        switch_project(repl_state.settings, "switched-project")

        buf = StringIO()
        test_console = Console(file=buf, no_color=True, width=120)
        with patch("vaig.cli.repl.console", test_console):
            _cmd_config(repl_state)

        output = buf.getvalue()
        assert "switched-project" in output


# ══════════════════════════════════════════════════════════════
# Phase 3.5: Help text includes new commands
# ══════════════════════════════════════════════════════════════


class TestHelpTextUpdate:
    def test_help_includes_project_command(self) -> None:
        from vaig.cli.repl import _cmd_help

        buf = StringIO()
        test_console = Console(file=buf, no_color=True, width=120)
        with patch("vaig.cli.repl.console", test_console):
            _cmd_help()

        output = buf.getvalue()
        assert "/project" in output

    def test_help_includes_location_command(self) -> None:
        from vaig.cli.repl import _cmd_help

        buf = StringIO()
        test_console = Console(file=buf, no_color=True, width=120)
        with patch("vaig.cli.repl.console", test_console):
            _cmd_help()

        output = buf.getvalue()
        assert "/location" in output

    def test_help_includes_cluster_command(self) -> None:
        from vaig.cli.repl import _cmd_help

        buf = StringIO()
        test_console = Console(file=buf, no_color=True, width=120)
        with patch("vaig.cli.repl.console", test_console):
            _cmd_help()

        output = buf.getvalue()
        assert "/cluster" in output

    def test_help_includes_config_command(self) -> None:
        from vaig.cli.repl import _cmd_help

        buf = StringIO()
        test_console = Console(file=buf, no_color=True, width=120)
        with patch("vaig.cli.repl.console", test_console):
            _cmd_help()

        output = buf.getvalue()
        assert "/config" in output

    def test_help_has_config_commands_section(self) -> None:
        from vaig.cli.repl import _cmd_help

        buf = StringIO()
        test_console = Console(file=buf, no_color=True, width=120)
        with patch("vaig.cli.repl.console", test_console):
            _cmd_help()

        output = buf.getvalue()
        assert "Config Commands" in output


# ══════════════════════════════════════════════════════════════
# Phase 3.6: Command dispatch registration
# ══════════════════════════════════════════════════════════════


class TestCommandDispatch:
    def test_new_commands_in_slash_commands_list(self) -> None:
        from vaig.cli.repl import SLASH_COMMANDS

        assert "/project" in SLASH_COMMANDS
        assert "/location" in SLASH_COMMANDS
        assert "/cluster" in SLASH_COMMANDS
        assert "/config" in SLASH_COMMANDS

    def test_handle_command_routes_project(self, repl_state: MagicMock) -> None:
        """Verify /project is recognized by the dispatcher."""
        from vaig.cli.repl import _handle_command

        buf = StringIO()
        test_console = Console(file=buf, no_color=True, width=120)
        with patch("vaig.cli.repl.console", test_console):
            result = _handle_command(repl_state, "/project")

        assert result is False  # should not exit
        output = buf.getvalue()
        # Should show current project (no-args behavior)
        assert "test-project" in output

    def test_handle_command_routes_config(self, repl_state: MagicMock) -> None:
        from vaig.cli.repl import _handle_command

        buf = StringIO()
        test_console = Console(file=buf, no_color=True, width=120)
        with patch("vaig.cli.repl.console", test_console):
            result = _handle_command(repl_state, "/config")

        assert result is False
        output = buf.getvalue()
        assert "test-project" in output

    def test_handle_command_routes_location(self, repl_state: MagicMock) -> None:
        from vaig.cli.repl import _handle_command

        buf = StringIO()
        test_console = Console(file=buf, no_color=True, width=120)
        with patch("vaig.cli.repl.console", test_console):
            result = _handle_command(repl_state, "/location")

        assert result is False
        output = buf.getvalue()
        assert "us-central1" in output

    def test_handle_command_routes_cluster(self, repl_state: MagicMock) -> None:
        from vaig.cli.repl import _handle_command

        buf = StringIO()
        test_console = Console(file=buf, no_color=True, width=120)
        with patch("vaig.cli.repl.console", test_console):
            result = _handle_command(repl_state, "/cluster")

        assert result is False
        output = buf.getvalue()
        assert "test-cluster" in output


# ══════════════════════════════════════════════════════════════
# Phase 6.1: Prompt prefix shows active project
# ══════════════════════════════════════════════════════════════


class TestPromptPrefix:
    def test_prompt_shows_project(self, settings: Settings, mock_client: MagicMock) -> None:
        """Task 6.1: Prompt prefix should include active project."""
        from vaig.cli.repl import REPLState

        mock_orchestrator = MagicMock()
        mock_session_manager = MagicMock()
        mock_context_builder = MagicMock()
        mock_context_builder.bundle.file_count = 0
        mock_skill_registry = MagicMock()

        state = REPLState(
            settings=settings,
            client=mock_client,
            orchestrator=mock_orchestrator,
            session_manager=mock_session_manager,
            context_builder=mock_context_builder,
            skill_registry=mock_skill_registry,
        )

        prefix = state.prompt_prefix
        assert "[test-project]" in prefix
        assert "[gemini-2.5-pro]" in prefix

    def test_prompt_updates_after_project_switch(
        self, settings: Settings, mock_client: MagicMock,
    ) -> None:
        """Prompt should reflect new project after switch."""
        from vaig.cli.repl import REPLState
        from vaig.core.config_switcher import switch_project

        mock_orchestrator = MagicMock()
        mock_session_manager = MagicMock()
        mock_context_builder = MagicMock()
        mock_context_builder.bundle.file_count = 0
        mock_skill_registry = MagicMock()

        state = REPLState(
            settings=settings,
            client=mock_client,
            orchestrator=mock_orchestrator,
            session_manager=mock_session_manager,
            context_builder=mock_context_builder,
            skill_registry=mock_skill_registry,
        )

        switch_project(settings, "new-project")
        prefix = state.prompt_prefix
        assert "[new-project]" in prefix
        assert "[test-project]" not in prefix

    def test_prompt_no_project_when_empty(self, mock_client: MagicMock) -> None:
        """When project_id is empty, no project bracket should appear."""
        from vaig.cli.repl import REPLState

        empty_settings = Settings(
            gcp=GCPConfig(project_id=""),
            models=ModelsConfig(default="gemini-2.5-pro"),
        )

        mock_orchestrator = MagicMock()
        mock_session_manager = MagicMock()
        mock_context_builder = MagicMock()
        mock_context_builder.bundle.file_count = 0
        mock_skill_registry = MagicMock()

        state = REPLState(
            settings=empty_settings,
            client=mock_client,
            orchestrator=mock_orchestrator,
            session_manager=mock_session_manager,
            context_builder=mock_context_builder,
            skill_registry=mock_skill_registry,
        )

        prefix = state.prompt_prefix
        # Should not have empty brackets
        assert "[]" not in prefix
        # Should still have the model
        assert "[gemini-2.5-pro]" in prefix

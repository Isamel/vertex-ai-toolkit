"""Tests for REPL /live command — toggle, chat routing, and handler functions.

Verifies that:
- /live toggles live mode on/off
- /live and /code are mutually exclusive
- /live appears in /help output
- Live chat routing dispatches to the correct handler (InfraAgent or orchestrated)
- Async live handlers exist and are coroutine functions
- Graceful handling when kubernetes deps are missing
- /clear resets live_mode and related state
"""

from __future__ import annotations

import inspect
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from vaig.core.config import GCPConfig, GKEConfig, ModelsConfig, Settings

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def settings() -> Settings:
    """Minimal settings for live mode tests."""
    return Settings(
        gcp=GCPConfig(
            project_id="test-project",
            location="us-central1",
        ),
        gke=GKEConfig(
            cluster_name="test-cluster",
            project_id="test-project",
            default_namespace="default",
            location="us-central1",
        ),
        models=ModelsConfig(default="gemini-2.5-pro"),
    )


@pytest.fixture()
def repl_state(settings: Settings, mock_client: MagicMock) -> MagicMock:
    """Create a mock REPLState-like object for command handlers."""
    state = MagicMock()
    state.settings = settings
    state.client = mock_client
    state.model = "gemini-2.5-pro"
    state.code_mode = False
    state.live_mode = False
    state.gke_config = None
    state.tool_registry = None
    state.tool_result_cache = None
    state.active_skill = None
    state.debug = False
    state.cost_tracker.request_count = 0
    state.settings.budget.enabled = False
    state.context_builder.bundle.file_count = 0
    state.session_manager.add_message = MagicMock()
    state.session_manager.async_add_message = AsyncMock()
    state.orchestrator = MagicMock()
    return state


# ══════════════════════════════════════════════════════════════
# Signature / Inspection Tests
# ══════════════════════════════════════════════════════════════


class TestLiveModeFunctions:
    """Verify live mode functions exist and have correct types."""

    def test_cmd_live_exists(self) -> None:
        """_cmd_live should be importable."""
        from vaig.cli.repl import _cmd_live

        assert callable(_cmd_live)

    def test_handle_live_chat_exists(self) -> None:
        """_handle_live_chat should be importable."""
        from vaig.cli.repl import _handle_live_chat

        assert callable(_handle_live_chat)

    def test_handle_live_skill_chat_exists(self) -> None:
        """_handle_live_skill_chat should be importable."""
        from vaig.cli.repl import _handle_live_skill_chat

        assert callable(_handle_live_skill_chat)

    def test_async_handle_live_chat_exists(self) -> None:
        """_async_handle_live_chat should be an async function."""
        from vaig.cli.repl import _async_handle_live_chat

        assert inspect.iscoroutinefunction(_async_handle_live_chat)

    def test_async_handle_live_skill_chat_exists(self) -> None:
        """_async_handle_live_skill_chat should be an async function."""
        from vaig.cli.repl import _async_handle_live_skill_chat

        assert inspect.iscoroutinefunction(_async_handle_live_skill_chat)


# ══════════════════════════════════════════════════════════════
# /live Toggle Tests
# ══════════════════════════════════════════════════════════════


class TestCmdLiveToggle:
    """Test /live command toggles live mode on and off."""

    def test_live_toggle_on(self, repl_state: MagicMock) -> None:
        """Enabling /live should set live_mode=True and populate gke state."""
        from vaig.cli.repl import _cmd_live

        mock_gke_config = MagicMock()
        mock_gke_config.cluster_name = "test-cluster"
        mock_gke_config.default_namespace = "default"
        mock_gke_config.project_id = "test-project"
        mock_registry = MagicMock()
        mock_registry.list_tools.return_value = ["tool1", "tool2", "tool3"]
        mock_cache = MagicMock()

        with (
            patch("vaig.cli.repl.console"),
            patch(
                "vaig.cli.commands.live._build_gke_config",
                return_value=mock_gke_config,
            ) as mock_build,
            patch(
                "vaig.cli.commands.live._register_live_tools",
                return_value=mock_registry,
            ) as mock_register,
            patch(
                "vaig.core.cache.ToolResultCache",
                return_value=mock_cache,
            ),
        ):
            _cmd_live(repl_state)

        assert repl_state.live_mode is True
        assert repl_state.gke_config is mock_gke_config
        assert repl_state.tool_registry is mock_registry
        assert repl_state.tool_result_cache is not None
        mock_build.assert_called_once_with(repl_state.settings)
        mock_register.assert_called_once_with(
            mock_gke_config, settings=repl_state.settings,
        )

    def test_live_toggle_off(self, repl_state: MagicMock) -> None:
        """Disabling /live should clear live_mode and related state."""
        from vaig.cli.repl import _cmd_live

        repl_state.live_mode = True
        repl_state.gke_config = MagicMock()
        repl_state.tool_registry = MagicMock()
        repl_state.tool_result_cache = MagicMock()

        with patch("vaig.cli.repl.console"):
            _cmd_live(repl_state)

        assert repl_state.live_mode is False
        assert repl_state.gke_config is None
        assert repl_state.tool_registry is None
        assert repl_state.tool_result_cache is None


# ══════════════════════════════════════════════════════════════
# Mutual Exclusivity Tests
# ══════════════════════════════════════════════════════════════


class TestLiveCodeMutualExclusivity:
    """Test /live and /code are mutually exclusive."""

    def test_live_disables_code_mode(self, repl_state: MagicMock) -> None:
        """Enabling /live when /code is active should disable code mode."""
        from vaig.cli.repl import _cmd_live

        repl_state.code_mode = True

        mock_registry = MagicMock()
        mock_registry.list_tools.return_value = ["tool1"]

        with (
            patch("vaig.cli.repl.console"),
            patch(
                "vaig.cli.commands.live._build_gke_config",
                return_value=MagicMock(
                    cluster_name="c", default_namespace="ns", project_id="p",
                ),
            ),
            patch(
                "vaig.cli.commands.live._register_live_tools",
                return_value=mock_registry,
            ),
            patch("vaig.core.cache.ToolResultCache"),
        ):
            _cmd_live(repl_state)

        assert repl_state.live_mode is True
        assert repl_state.code_mode is False

    def test_code_is_independent_of_live(self, repl_state: MagicMock) -> None:
        """/code does not check live mode (but /live checks /code)."""
        from vaig.cli.repl import _cmd_code

        repl_state.live_mode = True

        with patch("vaig.cli.repl.console"):
            _cmd_code(repl_state)

        # code toggles independently — live mode is NOT auto-disabled
        assert repl_state.code_mode is True


# ══════════════════════════════════════════════════════════════
# /help and /live in SLASH_COMMANDS
# ══════════════════════════════════════════════════════════════


class TestLiveInHelp:
    """Verify /live appears in help output and SLASH_COMMANDS."""

    def test_live_in_slash_commands(self) -> None:
        """/live should be in the SLASH_COMMANDS list."""
        from vaig.cli.repl import SLASH_COMMANDS

        assert "/live" in SLASH_COMMANDS

    def test_help_mentions_live(self) -> None:
        """/help output should mention /live."""
        from vaig.cli.repl import _cmd_help

        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True)

        with patch("vaig.cli.repl.console", test_console):
            _cmd_help()

        output = buf.getvalue()
        assert "/live" in output
        assert "live" in output.lower()


# ══════════════════════════════════════════════════════════════
# Command Handler Routing
# ══════════════════════════════════════════════════════════════


class TestLiveInHandleCommand:
    """Verify /live is routed to _cmd_live in _handle_command."""

    def test_live_command_dispatches(self, repl_state: MagicMock) -> None:
        """/live should be dispatched via _handle_command."""
        from vaig.cli.repl import _handle_command

        mock_registry = MagicMock()
        mock_registry.list_tools.return_value = ["tool1"]

        with (
            patch("vaig.cli.repl.console"),
            patch(
                "vaig.cli.commands.live._build_gke_config",
                return_value=MagicMock(
                    cluster_name="c", default_namespace="ns", project_id="p",
                ),
            ),
            patch(
                "vaig.cli.commands.live._register_live_tools",
                return_value=mock_registry,
            ),
            patch("vaig.core.cache.ToolResultCache"),
        ):
            result = _handle_command(repl_state, "/live")

        assert result is False  # Should not exit the REPL
        assert repl_state.live_mode is True


# ══════════════════════════════════════════════════════════════
# Chat Routing Tests
# ══════════════════════════════════════════════════════════════


class TestChatRouting:
    """Test _handle_chat routes to live handlers when live_mode is active."""

    def test_live_mode_routes_to_infra_agent(self, repl_state: MagicMock) -> None:
        """When live_mode=True and no skill, should route to _handle_live_chat."""
        from vaig.cli.repl import _handle_chat

        repl_state.live_mode = True
        repl_state.active_skill = None

        with (
            patch("vaig.cli.repl.console"),
            patch("vaig.cli.repl._handle_live_chat") as mock_handler,
        ):
            _handle_chat(repl_state, "show me pods")

        mock_handler.assert_called_once()

    def test_live_mode_with_live_skill_routes_to_orchestrated(
        self, repl_state: MagicMock,
    ) -> None:
        """When live_mode=True and skill requires_live_tools, route to orchestrated."""
        from vaig.cli.repl import _handle_chat

        repl_state.live_mode = True
        mock_skill = MagicMock()
        mock_skill.get_metadata.return_value.requires_live_tools = True
        repl_state.active_skill = mock_skill

        with (
            patch("vaig.cli.repl.console"),
            patch("vaig.cli.repl._handle_live_skill_chat") as mock_handler,
        ):
            _handle_chat(repl_state, "run health check")

        mock_handler.assert_called_once()

    def test_live_mode_with_non_live_skill_routes_to_infra(
        self, repl_state: MagicMock,
    ) -> None:
        """When live_mode=True but skill doesn't require_live_tools, use InfraAgent."""
        from vaig.cli.repl import _handle_chat

        repl_state.live_mode = True
        mock_skill = MagicMock()
        mock_skill.get_metadata.return_value.requires_live_tools = False
        repl_state.active_skill = mock_skill

        with (
            patch("vaig.cli.repl.console"),
            patch("vaig.cli.repl._handle_live_chat") as mock_handler,
        ):
            _handle_chat(repl_state, "check pods")

        mock_handler.assert_called_once()

    def test_code_mode_takes_priority_over_live(self, repl_state: MagicMock) -> None:
        """code_mode should take priority over live_mode in routing."""
        from vaig.cli.repl import _handle_chat

        repl_state.code_mode = True
        repl_state.live_mode = True

        with (
            patch("vaig.cli.repl.console"),
            patch("vaig.cli.repl._handle_code_chat") as mock_code,
            patch("vaig.cli.repl._handle_live_chat") as mock_live,
        ):
            _handle_chat(repl_state, "do something")

        mock_code.assert_called_once()
        mock_live.assert_not_called()


# ══════════════════════════════════════════════════════════════
# Async Chat Routing Tests
# ══════════════════════════════════════════════════════════════


class TestAsyncChatRouting:
    """Test _async_handle_chat routes to async live handlers."""

    @pytest.mark.asyncio
    async def test_async_live_mode_routes_to_infra(self, repl_state: MagicMock) -> None:
        """Async: live_mode=True, no skill → _async_handle_live_chat."""
        from vaig.cli.repl import _async_handle_chat

        repl_state.live_mode = True
        repl_state.active_skill = None

        with (
            patch("vaig.cli.repl.console"),
            patch("vaig.cli.repl._async_handle_live_chat", new_callable=AsyncMock) as mock_handler,
        ):
            await _async_handle_chat(repl_state, "show pods")

        mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_live_mode_with_skill_routes_to_orchestrated(
        self, repl_state: MagicMock,
    ) -> None:
        """Async: live_mode=True + requires_live_tools → _async_handle_live_skill_chat."""
        from vaig.cli.repl import _async_handle_chat

        repl_state.live_mode = True
        mock_skill = MagicMock()
        mock_skill.get_metadata.return_value.requires_live_tools = True
        repl_state.active_skill = mock_skill

        with (
            patch("vaig.cli.repl.console"),
            patch(
                "vaig.cli.repl._async_handle_live_skill_chat",
                new_callable=AsyncMock,
            ) as mock_handler,
        ):
            await _async_handle_chat(repl_state, "health check")

        mock_handler.assert_called_once()


# ══════════════════════════════════════════════════════════════
# Missing Dependencies (ImportError)
# ══════════════════════════════════════════════════════════════


class TestLiveMissingDeps:
    """Test graceful handling when kubernetes deps are missing."""

    def test_import_error_disables_live(self, repl_state: MagicMock) -> None:
        """If imports fail, live_mode should stay False."""
        from vaig.cli.repl import _cmd_live

        with (
            patch("vaig.cli.repl.console"),
            patch.dict(
                "sys.modules",
                {"vaig.cli.commands.live": None},
            ),
        ):
            # Force ImportError by making the import fail
            with patch(
                "builtins.__import__",
                side_effect=_selective_import_error,
            ):
                _cmd_live(repl_state)

        assert repl_state.live_mode is False


def _selective_import_error(
    name: str, *args: object, **kwargs: object,
) -> object:
    """Raise ImportError only for live-mode-related imports."""
    if "vaig.cli.commands.live" in name:
        raise ImportError("No module named 'kubernetes'")
    return __builtins__.__import__(name, *args, **kwargs)  # type: ignore[union-attr]


# ══════════════════════════════════════════════════════════════
# /clear Resets Live Mode
# ══════════════════════════════════════════════════════════════


class TestClearResetsLive:
    """/clear should reset live_mode and related state."""

    def test_clear_resets_live_state(self, repl_state: MagicMock) -> None:
        """After /clear, live_mode and related fields should be False/None."""
        from vaig.cli.repl import _cmd_clear

        repl_state.live_mode = True
        repl_state.gke_config = MagicMock()
        repl_state.tool_registry = MagicMock()
        repl_state.tool_result_cache = MagicMock()

        with patch("vaig.cli.repl.console"):
            _cmd_clear(repl_state)

        assert repl_state.live_mode is False
        assert repl_state.gke_config is None
        assert repl_state.tool_registry is None
        assert repl_state.tool_result_cache is None
        assert repl_state.code_mode is False


# ══════════════════════════════════════════════════════════════
# Prompt Prefix Tests
# ══════════════════════════════════════════════════════════════


class TestPromptPrefixLive:
    """Test prompt_prefix shows live mode indicator."""

    def test_prompt_prefix_shows_live(self) -> None:
        """prompt_prefix should include (🔍live) when live_mode is True."""
        from vaig.cli.repl import REPLState

        mock_client = MagicMock()
        mock_client.current_model = "gemini-2.5-pro"

        state = REPLState(
            settings=Settings(
                gcp=GCPConfig(project_id="my-project"),
                models=ModelsConfig(default="gemini-2.5-pro"),
            ),
            client=mock_client,
            orchestrator=MagicMock(),
            session_manager=MagicMock(),
            context_builder=MagicMock(),
            skill_registry=MagicMock(),
        )
        state.context_builder.bundle.file_count = 0
        state.live_mode = True

        assert "🔍live" in state.prompt_prefix

    def test_prompt_prefix_no_live(self) -> None:
        """prompt_prefix should NOT include live when live_mode is False."""
        from vaig.cli.repl import REPLState

        mock_client = MagicMock()
        mock_client.current_model = "gemini-2.5-pro"

        state = REPLState(
            settings=Settings(
                gcp=GCPConfig(project_id="my-project"),
                models=ModelsConfig(default="gemini-2.5-pro"),
            ),
            client=mock_client,
            orchestrator=MagicMock(),
            session_manager=MagicMock(),
            context_builder=MagicMock(),
            skill_registry=MagicMock(),
        )
        state.context_builder.bundle.file_count = 0
        state.live_mode = False

        assert "live" not in state.prompt_prefix


# ══════════════════════════════════════════════════════════════
# REPLState Fields
# ══════════════════════════════════════════════════════════════


class TestREPLStateFields:
    """Verify REPLState has the new live mode fields."""

    def test_replstate_has_live_fields(self) -> None:
        """REPLState should have live_mode, gke_config, tool_registry, tool_result_cache."""
        from vaig.cli.repl import REPLState

        mock_client = MagicMock()
        mock_client.current_model = "gemini-2.5-pro"

        state = REPLState(
            settings=Settings(
                gcp=GCPConfig(project_id="p"),
                models=ModelsConfig(default="gemini-2.5-pro"),
            ),
            client=mock_client,
            orchestrator=MagicMock(),
            session_manager=MagicMock(),
            context_builder=MagicMock(),
            skill_registry=MagicMock(),
        )

        assert state.live_mode is False
        assert state.gke_config is None
        assert state.tool_registry is None
        assert state.tool_result_cache is None


# ══════════════════════════════════════════════════════════════
# Live Chat Handler Tests (sync)
# ══════════════════════════════════════════════════════════════


class TestHandleLiveChat:
    """Test _handle_live_chat dispatches to InfraAgent."""

    def test_calls_infra_agent(self, repl_state: MagicMock) -> None:
        """_handle_live_chat should create and execute an InfraAgent."""
        from vaig.cli.repl import _handle_live_chat

        repl_state.gke_config = MagicMock()
        repl_state.tool_result_cache = MagicMock()

        mock_result = MagicMock()
        mock_result.content = "Found 3 pods running"
        mock_result.usage = {"prompt_tokens": 10, "candidates_tokens": 20}

        mock_agent_cls = MagicMock()
        mock_agent_cls.return_value.execute.return_value = mock_result

        with (
            patch("vaig.cli.repl.console"),
            patch("vaig.agents.infra_agent.InfraAgent", mock_agent_cls),
        ):
            _handle_live_chat(repl_state, "show pods", "")

        mock_agent_cls.return_value.execute.assert_called_once()
        repl_state.session_manager.add_message.assert_called_once_with(
            "model", "Found 3 pods running", model="gemini-2.5-pro",
        )

    def test_handles_max_iterations_error(self, repl_state: MagicMock) -> None:
        """_handle_live_chat should catch MaxIterationsError gracefully."""
        from vaig.cli.repl import _handle_live_chat
        from vaig.core.exceptions import MaxIterationsError

        repl_state.gke_config = MagicMock()
        repl_state.tool_result_cache = MagicMock()

        mock_agent_cls = MagicMock()
        mock_agent_cls.return_value.execute.side_effect = MaxIterationsError(
            "Exceeded 25 iterations", iterations=25
        )

        with (
            patch("vaig.cli.repl.console") as mock_console,
            patch("vaig.agents.infra_agent.InfraAgent", mock_agent_cls),
        ):
            _handle_live_chat(repl_state, "complex query", "")

        # Should print an error message, not raise
        assert mock_console.print.called


# ══════════════════════════════════════════════════════════════
# Live Skill Chat Handler Tests (sync)
# ══════════════════════════════════════════════════════════════


class TestHandleLiveSkillChat:
    """Test _handle_live_skill_chat dispatches to orchestrator."""

    def test_calls_execute_with_tools(self, repl_state: MagicMock) -> None:
        """_handle_live_skill_chat should call orchestrator.execute_with_tools."""
        from vaig.cli.repl import _handle_live_skill_chat

        mock_skill = MagicMock()
        mock_skill.get_metadata.return_value.display_name = "Health Check"
        mock_skill.get_metadata.return_value.requires_live_tools = True
        repl_state.active_skill = mock_skill
        repl_state.tool_registry = MagicMock()
        repl_state.gke_config = MagicMock()
        repl_state.gke_config.default_namespace = "default"
        repl_state.gke_config.location = "us-central1"
        repl_state.gke_config.cluster_name = "test-cluster"

        mock_orch_result = MagicMock()
        mock_orch_result.synthesized_output = "All services healthy"
        mock_orch_result.total_usage = {"prompt_tokens": 100, "candidates_tokens": 200}
        repl_state.orchestrator.execute_with_tools.return_value = mock_orch_result

        with patch("vaig.cli.repl.console"):
            _handle_live_skill_chat(repl_state, "run check", "")

        repl_state.orchestrator.execute_with_tools.assert_called_once()
        call_kwargs = repl_state.orchestrator.execute_with_tools.call_args
        assert call_kwargs.kwargs["query"] == "run check"
        assert call_kwargs.kwargs["skill"] is mock_skill
        assert call_kwargs.kwargs["tool_registry"] is repl_state.tool_registry

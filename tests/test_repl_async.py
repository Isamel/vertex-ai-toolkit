"""Tests for async REPL — async_start_repl, _async_repl_loop, and async chat handlers.

Verifies that:
- async_start_repl() is an async function with the same signature as start_repl()
- _async_repl_loop() uses prompt_async() instead of prompt()
- _async_handle_direct_chat() calls async_execute_single (not sync execute_single)
- _async_handle_skill_chat() calls async_execute_skill_phase
- _async_handle_code_chat() calls CodingAgent.async_execute
- _async_save_cost_data() uses async_save_cost_data
- KeyboardInterrupt / EOFError are handled gracefully
"""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vaig.core.config import Settings

# ══════════════════════════════════════════════════════════════
# Signature / Inspection Tests
# ══════════════════════════════════════════════════════════════


class TestAsyncREPLSignatures:
    """Verify that async REPL functions exist and have correct types."""

    def test_async_start_repl_exists(self) -> None:
        """async_start_repl should be importable."""
        from vaig.cli.repl import async_start_repl

        assert callable(async_start_repl)

    def test_async_start_repl_is_coroutine_function(self) -> None:
        """async_start_repl should be an async function."""
        from vaig.cli.repl import async_start_repl

        assert inspect.iscoroutinefunction(async_start_repl)

    def test_async_start_repl_has_same_parameters_as_sync(self) -> None:
        """async_start_repl should accept the same parameters as start_repl."""
        from vaig.cli.repl import async_start_repl, start_repl

        sync_params = set(inspect.signature(start_repl).parameters.keys())
        async_params = set(inspect.signature(async_start_repl).parameters.keys())
        assert sync_params == async_params

    def test_async_repl_loop_exists(self) -> None:
        """_async_repl_loop should be importable."""
        from vaig.cli.repl import _async_repl_loop

        assert callable(_async_repl_loop)
        assert inspect.iscoroutinefunction(_async_repl_loop)

    def test_async_handle_chat_exists(self) -> None:
        """_async_handle_chat should be an async function."""
        from vaig.cli.repl import _async_handle_chat

        assert inspect.iscoroutinefunction(_async_handle_chat)

    def test_async_handle_direct_chat_exists(self) -> None:
        """_async_handle_direct_chat should be an async function."""
        from vaig.cli.repl import _async_handle_direct_chat

        assert inspect.iscoroutinefunction(_async_handle_direct_chat)

    def test_async_handle_skill_chat_exists(self) -> None:
        """_async_handle_skill_chat should be an async function."""
        from vaig.cli.repl import _async_handle_skill_chat

        assert inspect.iscoroutinefunction(_async_handle_skill_chat)

    def test_async_handle_code_chat_exists(self) -> None:
        """_async_handle_code_chat should be an async function."""
        from vaig.cli.repl import _async_handle_code_chat

        assert inspect.iscoroutinefunction(_async_handle_code_chat)

    def test_async_save_cost_data_exists(self) -> None:
        """_async_save_cost_data should be an async function."""
        from vaig.cli.repl import _async_save_cost_data

        assert inspect.iscoroutinefunction(_async_save_cost_data)


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════


def _make_state() -> MagicMock:
    """Create a mock REPLState with all required attributes."""
    state = MagicMock()
    state.model = "gemini-2.5-pro"
    state.prompt_prefix = "[test] > "
    state.stream_enabled = True
    state.code_mode = False
    state.active_skill = None
    state.current_phase = MagicMock()
    state.current_phase.value = "analyze"
    state.context_builder.bundle.file_count = 0
    state.cost_tracker.request_count = 0
    state.settings.budget.enabled = False

    # Async methods
    state.session_manager.async_add_message = AsyncMock()
    state.session_manager.async_save_cost_data = AsyncMock(return_value=True)
    state.session_manager.async_close = AsyncMock()

    return state


# ══════════════════════════════════════════════════════════════
# _async_repl_loop Tests
# ══════════════════════════════════════════════════════════════


class TestAsyncREPLLoop:
    """Test _async_repl_loop uses prompt_async and handles input."""

    @pytest.mark.asyncio
    async def test_eof_exits_loop(self) -> None:
        """EOFError from prompt_async should exit the loop cleanly."""
        from vaig.cli.repl import _async_repl_loop

        mock_session = MagicMock()
        mock_session.prompt_async = AsyncMock(side_effect=EOFError)

        state = _make_state()

        with patch("vaig.cli.repl.console"):
            await _async_repl_loop(mock_session, state)

        mock_session.prompt_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_keyboard_interrupt_continues(self) -> None:
        """KeyboardInterrupt should print ^C and continue, not exit."""
        from vaig.cli.repl import _async_repl_loop

        mock_session = MagicMock()
        # First call raises KeyboardInterrupt, second raises EOFError to exit
        mock_session.prompt_async = AsyncMock(
            side_effect=[KeyboardInterrupt, EOFError]
        )

        state = _make_state()

        with patch("vaig.cli.repl.console"):
            await _async_repl_loop(mock_session, state)

        assert mock_session.prompt_async.call_count == 2

    @pytest.mark.asyncio
    async def test_empty_input_skipped(self) -> None:
        """Empty input should be skipped, not sent to chat handler."""
        from vaig.cli.repl import _async_repl_loop

        mock_session = MagicMock()
        # Empty string, whitespace, then exit
        mock_session.prompt_async = AsyncMock(
            side_effect=["", "   ", EOFError]
        )

        state = _make_state()

        with (
            patch("vaig.cli.repl.console"),
            patch("vaig.cli.repl._async_handle_chat") as mock_chat,
        ):
            await _async_repl_loop(mock_session, state)

        mock_chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_slash_command_delegates_to_handler(self) -> None:
        """Slash commands should go to _handle_command (sync), not chat."""
        from vaig.cli.repl import _async_repl_loop

        mock_session = MagicMock()
        mock_session.prompt_async = AsyncMock(
            side_effect=["/help", EOFError]
        )

        state = _make_state()

        with (
            patch("vaig.cli.repl.console"),
            patch("vaig.cli.repl._handle_command", return_value=False) as mock_cmd,
            patch("vaig.cli.repl._async_handle_chat") as mock_chat,
        ):
            await _async_repl_loop(mock_session, state)

        mock_cmd.assert_called_once_with(state, "/help")
        mock_chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_quit_command_exits(self) -> None:
        """A /quit command should exit the loop."""
        from vaig.cli.repl import _async_repl_loop

        mock_session = MagicMock()
        mock_session.prompt_async = AsyncMock(
            side_effect=["/quit", "should not reach"]
        )

        state = _make_state()

        with (
            patch("vaig.cli.repl.console"),
            patch("vaig.cli.repl._handle_command", return_value=True) as mock_cmd,
        ):
            await _async_repl_loop(mock_session, state)

        mock_cmd.assert_called_once_with(state, "/quit")

    @pytest.mark.asyncio
    async def test_regular_input_goes_to_async_chat(self) -> None:
        """Regular (non-slash) input should go to _async_handle_chat."""
        from vaig.cli.repl import _async_repl_loop

        mock_session = MagicMock()
        mock_session.prompt_async = AsyncMock(
            side_effect=["hello world", EOFError]
        )

        state = _make_state()

        with (
            patch("vaig.cli.repl.console"),
            patch("vaig.cli.repl._async_handle_chat", new_callable=AsyncMock) as mock_chat,
        ):
            await _async_repl_loop(mock_session, state)

        mock_chat.assert_called_once_with(state, "hello world")

    @pytest.mark.asyncio
    async def test_uses_prompt_async_not_prompt(self) -> None:
        """_async_repl_loop must call prompt_async, never sync prompt."""
        from vaig.cli.repl import _async_repl_loop

        mock_session = MagicMock()
        mock_session.prompt_async = AsyncMock(side_effect=EOFError)
        mock_session.prompt = MagicMock(side_effect=AssertionError("sync prompt called!"))

        state = _make_state()

        with patch("vaig.cli.repl.console"):
            await _async_repl_loop(mock_session, state)

        mock_session.prompt_async.assert_called_once()


# ══════════════════════════════════════════════════════════════
# _async_handle_direct_chat Tests
# ══════════════════════════════════════════════════════════════


class TestAsyncHandleDirectChat:
    """Test that direct chat uses async orchestrator methods."""

    @pytest.mark.asyncio
    async def test_streaming_calls_async_execute_single(self) -> None:
        """Streaming mode should call async_execute_single with stream=True."""
        from vaig.cli.repl import _async_handle_direct_chat

        state = _make_state()

        # Mock async stream result
        mock_stream = MagicMock()
        mock_stream.model = "gemini-2.5-pro"
        mock_stream.usage = {"prompt_tokens": 10, "completion_tokens": 20}

        async def _fake_aiter(self):
            yield "Hello "
            yield "world"

        mock_stream.__aiter__ = _fake_aiter
        state.orchestrator.async_execute_single = AsyncMock(return_value=mock_stream)

        with (
            patch("vaig.cli.repl.console") as mock_console,
            patch("vaig.cli.repl._check_budget", return_value=True),
            patch("vaig.cli.repl._record_cost"),
            patch("vaig.cli.repl.isinstance", return_value=True),
        ):
            mock_console.status.return_value = MagicMock()
            await _async_handle_direct_chat(state, "hello", "")

        state.orchestrator.async_execute_single.assert_called_once_with(
            "hello", context="", stream=True,
        )

    @pytest.mark.asyncio
    async def test_non_streaming_calls_async_execute_single(self) -> None:
        """Non-streaming mode should call async_execute_single without stream."""
        from vaig.cli.repl import _async_handle_direct_chat

        state = _make_state()
        state.stream_enabled = False

        mock_result = MagicMock()
        mock_result.content = "Response text"
        mock_result.usage = {"prompt_tokens": 10, "completion_tokens": 20}
        state.orchestrator.async_execute_single = AsyncMock(return_value=mock_result)

        with (
            patch("vaig.cli.repl.console") as mock_console,
            patch("vaig.cli.repl._check_budget", return_value=True),
            patch("vaig.cli.repl._record_cost"),
            patch("vaig.cli.repl.Markdown"),
        ):
            mock_console.status.return_value.__enter__ = MagicMock()
            mock_console.status.return_value.__exit__ = MagicMock()
            await _async_handle_direct_chat(state, "hello", "")

        state.orchestrator.async_execute_single.assert_called_once_with(
            "hello", context="",
        )

    @pytest.mark.asyncio
    async def test_budget_exceeded_blocks_chat(self) -> None:
        """When budget is exceeded, no async call should be made."""
        from vaig.cli.repl import _async_handle_direct_chat

        state = _make_state()
        state.orchestrator.async_execute_single = AsyncMock()

        with patch("vaig.cli.repl._check_budget", return_value=False):
            await _async_handle_direct_chat(state, "hello", "")

        state.orchestrator.async_execute_single.assert_not_called()

    @pytest.mark.asyncio
    async def test_records_message_async(self) -> None:
        """Model response should be saved via async_add_message."""
        from vaig.cli.repl import _async_handle_direct_chat

        state = _make_state()
        state.stream_enabled = False

        mock_result = MagicMock()
        mock_result.content = "async response"
        mock_result.usage = {}
        state.orchestrator.async_execute_single = AsyncMock(return_value=mock_result)

        with (
            patch("vaig.cli.repl.console") as mock_console,
            patch("vaig.cli.repl._check_budget", return_value=True),
            patch("vaig.cli.repl._record_cost"),
            patch("vaig.cli.repl.Markdown"),
        ):
            mock_console.status.return_value.__enter__ = MagicMock()
            mock_console.status.return_value.__exit__ = MagicMock()
            await _async_handle_direct_chat(state, "hello", "")

        state.session_manager.async_add_message.assert_called_once_with(
            "model", "async response", model="gemini-2.5-pro",
        )


# ══════════════════════════════════════════════════════════════
# _async_handle_skill_chat Tests
# ══════════════════════════════════════════════════════════════


class TestAsyncHandleSkillChat:
    """Test that skill chat uses async orchestrator methods."""

    @pytest.mark.asyncio
    async def test_calls_async_execute_skill_phase(self) -> None:
        """Should call orchestrator.async_execute_skill_phase."""
        from vaig.cli.repl import _async_handle_skill_chat

        state = _make_state()
        mock_skill = MagicMock()
        mock_skill.get_metadata.return_value.display_name = "Test Skill"
        state.active_skill = mock_skill

        mock_result = MagicMock()
        mock_result.output = "Skill output"
        mock_result.metadata = {}
        mock_result.next_phase = None
        state.orchestrator.async_execute_skill_phase = AsyncMock(return_value=mock_result)

        with (
            patch("vaig.cli.repl.console") as mock_console,
            patch("vaig.cli.repl._check_budget", return_value=True),
            patch("vaig.cli.repl.Markdown"),
        ):
            mock_console.status.return_value.__enter__ = MagicMock()
            mock_console.status.return_value.__exit__ = MagicMock()
            await _async_handle_skill_chat(state, "analyze this", "context data")

        state.orchestrator.async_execute_skill_phase.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_skill_returns_early(self) -> None:
        """No active skill should return without calling orchestrator."""
        from vaig.cli.repl import _async_handle_skill_chat

        state = _make_state()
        state.active_skill = None
        state.orchestrator.async_execute_skill_phase = AsyncMock()

        with patch("vaig.cli.repl._check_budget", return_value=True):
            await _async_handle_skill_chat(state, "test", "")

        state.orchestrator.async_execute_skill_phase.assert_not_called()


# ══════════════════════════════════════════════════════════════
# _async_handle_code_chat Tests
# ══════════════════════════════════════════════════════════════


class TestAsyncHandleCodeChat:
    """Test that code chat uses CodingAgent.async_execute."""

    @pytest.mark.asyncio
    async def test_calls_agent_async_execute(self) -> None:
        """Should call CodingAgent.async_execute."""
        from vaig.cli.repl import _async_handle_code_chat

        state = _make_state()
        state.settings.coding = MagicMock()

        mock_result = MagicMock()
        mock_result.content = "Code result"
        mock_result.usage = {}

        mock_agent = MagicMock()
        mock_agent.async_execute = AsyncMock(return_value=mock_result)

        with (
            patch("vaig.cli.repl.console"),
            patch("vaig.cli.repl._check_budget", return_value=True),
            patch("vaig.cli.repl._record_cost"),
            patch("vaig.cli.repl._show_repl_coding_summary"),
            patch("vaig.cli.repl.Markdown"),
            patch("vaig.agents.coding.CodingAgent", return_value=mock_agent),
        ):
            await _async_handle_code_chat(state, "write code", "")

        mock_agent.async_execute.assert_called_once_with("write code", context="")

    @pytest.mark.asyncio
    async def test_max_iterations_handled(self) -> None:
        """MaxIterationsError should be caught and reported gracefully."""
        from vaig.cli.repl import _async_handle_code_chat
        from vaig.core.exceptions import MaxIterationsError

        state = _make_state()
        state.settings.coding = MagicMock()

        mock_agent = MagicMock()
        mock_agent.async_execute = AsyncMock(
            side_effect=MaxIterationsError("Max iterations reached", iterations=5),
        )

        with (
            patch("vaig.cli.repl.console") as mock_console,
            patch("vaig.cli.repl._check_budget", return_value=True),
            patch("vaig.agents.coding.CodingAgent", return_value=mock_agent),
        ):
            # Should NOT raise
            await _async_handle_code_chat(state, "complex task", "")

        # Verify error was displayed
        mock_console.print.assert_called()


# ══════════════════════════════════════════════════════════════
# _async_save_cost_data Tests
# ══════════════════════════════════════════════════════════════


class TestAsyncSaveCostData:
    """Test that cost data is saved via async methods."""

    @pytest.mark.asyncio
    async def test_saves_when_requests_exist(self) -> None:
        """Should call async_save_cost_data when there are recorded requests."""
        from vaig.cli.repl import _async_save_cost_data

        state = _make_state()
        state.cost_tracker.request_count = 3
        state.cost_tracker.to_dict.return_value = {"requests": 3}

        await _async_save_cost_data(state)

        state.session_manager.async_save_cost_data.assert_called_once_with({"requests": 3})

    @pytest.mark.asyncio
    async def test_skips_when_no_requests(self) -> None:
        """Should skip saving when request_count is 0."""
        from vaig.cli.repl import _async_save_cost_data

        state = _make_state()
        state.cost_tracker.request_count = 0

        await _async_save_cost_data(state)

        state.session_manager.async_save_cost_data.assert_not_called()


# ══════════════════════════════════════════════════════════════
# async_start_repl Tests
# ══════════════════════════════════════════════════════════════


class TestAsyncStartREPL:
    """Test async_start_repl initialization and lifecycle."""

    @pytest.mark.asyncio
    async def test_creates_new_session_async(self, tmp_path: Path) -> None:
        """async_start_repl should call async_new_session, not sync new_session."""
        from vaig.core.config import SessionConfig

        history_file = tmp_path / "repl_history"
        settings = Settings(
            session=SessionConfig(repl_history_path=str(history_file)),
        )

        mock_client = MagicMock()
        mock_client.current_model = "gemini-2.5-pro"
        mock_container = MagicMock()
        mock_container.gemini_client = mock_client

        with (
            patch("vaig.cli.repl.build_container", return_value=mock_container),
            patch("vaig.cli.repl.Orchestrator"),
            patch("vaig.cli.repl.SessionManager") as mock_sm_cls,
            patch("vaig.cli.repl.ContextBuilder"),
            patch("vaig.cli.repl.SkillRegistry"),
            patch("vaig.cli.repl._async_repl_loop", new_callable=AsyncMock),
            patch("vaig.cli.repl._show_session_cost_summary"),
            patch("vaig.cli.repl._async_save_cost_data", new_callable=AsyncMock),
            patch("vaig.cli.repl.console"),
        ):
            mock_sm = mock_sm_cls.return_value
            mock_sm.async_new_session = AsyncMock()
            mock_sm.async_close = AsyncMock()

            from vaig.cli.repl import async_start_repl

            await async_start_repl(settings)

        mock_sm.async_new_session.assert_called_once()
        mock_sm.async_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_loads_existing_session_async(self, tmp_path: Path) -> None:
        """async_start_repl with session_id should call async_load_session."""
        from vaig.core.config import SessionConfig

        history_file = tmp_path / "repl_history"
        settings = Settings(
            session=SessionConfig(repl_history_path=str(history_file)),
        )

        mock_client = MagicMock()
        mock_client.current_model = "gemini-2.5-pro"
        mock_container = MagicMock()
        mock_container.gemini_client = mock_client

        with (
            patch("vaig.cli.repl.build_container", return_value=mock_container),
            patch("vaig.cli.repl.Orchestrator"),
            patch("vaig.cli.repl.SessionManager") as mock_sm_cls,
            patch("vaig.cli.repl.ContextBuilder"),
            patch("vaig.cli.repl.SkillRegistry"),
            patch("vaig.cli.repl._async_repl_loop", new_callable=AsyncMock),
            patch("vaig.cli.repl._show_session_cost_summary"),
            patch("vaig.cli.repl._async_save_cost_data", new_callable=AsyncMock),
            patch("vaig.cli.repl.console"),
        ):
            mock_sm = mock_sm_cls.return_value

            loaded_session = MagicMock()
            loaded_session.name = "test-session"
            loaded_session.history = []
            mock_sm.async_load_session = AsyncMock(return_value=loaded_session)
            mock_sm.async_load_cost_data = AsyncMock(return_value=None)
            mock_sm.async_close = AsyncMock()

            from vaig.cli.repl import async_start_repl

            await async_start_repl(settings, session_id="abc123")

        mock_sm.async_load_session.assert_called_once_with("abc123")


# ══════════════════════════════════════════════════════════════
# Orchestrator async_execute_single Tests
# ══════════════════════════════════════════════════════════════


class TestOrchestratorAsyncExecuteSingle:
    """Test the async_execute_single method added to Orchestrator."""

    def test_async_execute_single_exists(self) -> None:
        """Orchestrator should have async_execute_single method."""
        from vaig.agents.orchestrator import Orchestrator

        assert hasattr(Orchestrator, "async_execute_single")
        assert inspect.iscoroutinefunction(Orchestrator.async_execute_single)

    def test_async_execute_single_has_same_params_as_sync(self) -> None:
        """async_execute_single should accept same params as execute_single."""
        from vaig.agents.orchestrator import Orchestrator

        sync_params = set(inspect.signature(Orchestrator.execute_single).parameters.keys())
        async_params = set(inspect.signature(Orchestrator.async_execute_single).parameters.keys())
        assert sync_params == async_params

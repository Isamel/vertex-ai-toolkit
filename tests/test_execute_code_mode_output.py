"""Tests for _execute_code_mode and _async_execute_code_mode output behavior.

Covers the bug fix for `ask --code --file` producing no output when the LLM
uses tool calls (edit_file/write_file) as its entire response — leaving
``result.content = ""`` which silently suppressed any confirmation to the user.

Fix: when ``result.content`` is empty, display either a success or warning
message depending on ``result.success``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vaig.core.config import Settings

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_agent_result(content: str = "", success: bool = True) -> MagicMock:
    """Build a minimal AgentResult-like mock."""
    result = MagicMock()
    result.content = content
    result.success = success
    result.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    result.metadata = {
        "tools_executed": [],
        "iterations": 1,
        "model": "gemini-2.5-pro",
        "finish_reason": "STOP",
    }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# _execute_code_mode (sync path)
# ══════════════════════════════════════════════════════════════════════════════


class TestExecuteCodeModeOutput:
    """Sync _execute_code_mode output guard tests."""

    def test_content_printed_when_present(self) -> None:
        """When result.content has text, it should be printed as Markdown."""
        from rich.markdown import Markdown

        from vaig.cli.commands._code import _execute_code_mode

        content = "# Done\nFile updated."
        mock_result = _make_agent_result(content=content)
        mock_agent = MagicMock()
        mock_agent.execute = MagicMock(return_value=mock_result)

        settings = Settings()
        mock_client = MagicMock()

        with (
            patch("vaig.agents.coding.CodingAgent", return_value=mock_agent),
            patch("vaig.cli.commands._code.console") as mock_console,
        ):
            _execute_code_mode(mock_client, settings, "Fix the bug", "")

        # Verify console.print was called with a Markdown object containing the content
        call_args = [call.args for call in mock_console.print.call_args_list]
        assert any(
            isinstance(arg, Markdown) and content in arg.markup
            for args in call_args
            for arg in args
        ), f"Expected Markdown({content!r}) to be printed; calls: {call_args}"

    def test_success_message_when_content_empty_and_success_true(self) -> None:
        """When result.content is empty and result.success is True,
        a green success message must be printed instead of silence."""
        from vaig.cli.commands._code import _execute_code_mode

        mock_result = _make_agent_result(content="", success=True)
        mock_agent = MagicMock()
        mock_agent.execute = MagicMock(return_value=mock_result)

        settings = Settings()
        mock_client = MagicMock()

        printed: list[str] = []

        def _capture_print(*args: object, **_kwargs: object) -> None:
            if args:
                printed.append(str(args[0]))

        with (
            patch("vaig.agents.coding.CodingAgent", return_value=mock_agent),
            patch("vaig.cli.commands._code.console") as mock_console,
        ):
            mock_console.print.side_effect = _capture_print
            _execute_code_mode(mock_client, settings, "Fix the bug", "")

        assert any(
            "completed successfully" in s or "files modified" in s
            for s in printed
        ), f"Expected success message in prints; got: {printed}"

    def test_warning_message_when_content_empty_and_success_false(self) -> None:
        """When result.content is empty and result.success is False,
        a yellow warning message must be printed."""
        from vaig.cli.commands._code import _execute_code_mode

        mock_result = _make_agent_result(content="", success=False)
        mock_agent = MagicMock()
        mock_agent.execute = MagicMock(return_value=mock_result)

        settings = Settings()
        mock_client = MagicMock()

        printed: list[str] = []

        def _capture_print(*args: object, **_kwargs: object) -> None:
            if args:
                printed.append(str(args[0]))

        with (
            patch("vaig.agents.coding.CodingAgent", return_value=mock_agent),
            patch("vaig.cli.commands._code.console") as mock_console,
        ):
            mock_console.print.side_effect = _capture_print
            _execute_code_mode(mock_client, settings, "Fix the bug", "")

        assert any(
            "no output" in s.lower() or "finished" in s.lower()
            for s in printed
        ), f"Expected warning message in prints; got: {printed}"


# ══════════════════════════════════════════════════════════════════════════════
# _async_execute_code_mode (async path)
# ══════════════════════════════════════════════════════════════════════════════


class TestAsyncExecuteCodeModeOutput:
    """Async _async_execute_code_mode output guard tests."""

    @pytest.mark.asyncio
    async def test_content_printed_when_present_async(self) -> None:
        """When result.content has text, it should be printed as Markdown."""
        from rich.markdown import Markdown

        from vaig.cli.commands._code import _async_execute_code_mode

        content = "# Done\nFile updated."
        mock_result = _make_agent_result(content=content)
        mock_agent = MagicMock()
        mock_agent.async_execute = AsyncMock(return_value=mock_result)

        settings = Settings()
        mock_client = MagicMock()

        with (
            patch("vaig.agents.coding.CodingAgent", return_value=mock_agent),
            patch("vaig.cli.commands._code.console") as mock_console,
        ):
            await _async_execute_code_mode(mock_client, settings, "Fix the bug", "")

        call_args = [call.args for call in mock_console.print.call_args_list]
        assert any(
            isinstance(arg, Markdown) and content in arg.markup
            for args in call_args
            for arg in args
        ), f"Expected Markdown({content!r}) to be printed; calls: {call_args}"

    @pytest.mark.asyncio
    async def test_success_message_when_content_empty_and_success_true_async(self) -> None:
        """Async path: empty content + success=True → green success message."""
        from vaig.cli.commands._code import _async_execute_code_mode

        mock_result = _make_agent_result(content="", success=True)
        mock_agent = MagicMock()
        mock_agent.async_execute = AsyncMock(return_value=mock_result)

        settings = Settings()
        mock_client = MagicMock()

        printed: list[str] = []

        def _capture_print(*args: object, **_kwargs: object) -> None:
            if args:
                printed.append(str(args[0]))

        with (
            patch("vaig.agents.coding.CodingAgent", return_value=mock_agent),
            patch("vaig.cli.commands._code.console") as mock_console,
        ):
            mock_console.print.side_effect = _capture_print
            await _async_execute_code_mode(mock_client, settings, "Fix the bug", "")

        assert any(
            "completed successfully" in s or "files modified" in s
            for s in printed
        ), f"Expected success message in prints; got: {printed}"

    @pytest.mark.asyncio
    async def test_warning_message_when_content_empty_and_success_false_async(self) -> None:
        """Async path: empty content + success=False → yellow warning message."""
        from vaig.cli.commands._code import _async_execute_code_mode

        mock_result = _make_agent_result(content="", success=False)
        mock_agent = MagicMock()
        mock_agent.async_execute = AsyncMock(return_value=mock_result)

        settings = Settings()
        mock_client = MagicMock()

        printed: list[str] = []

        def _capture_print(*args: object, **_kwargs: object) -> None:
            if args:
                printed.append(str(args[0]))

        with (
            patch("vaig.agents.coding.CodingAgent", return_value=mock_agent),
            patch("vaig.cli.commands._code.console") as mock_console,
        ):
            mock_console.print.side_effect = _capture_print
            await _async_execute_code_mode(mock_client, settings, "Fix the bug", "")

        assert any(
            "no output" in s.lower() or "finished" in s.lower()
            for s in printed
        ), f"Expected warning message in prints; got: {printed}"

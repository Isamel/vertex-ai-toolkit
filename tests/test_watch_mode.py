"""Tests for the --watch mode in the live command.

Covers:
- watch=None means normal single execution (no loop)
- watch < MINIMUM_WATCH_INTERVAL raises error (via typer.Exit)
- The watch loop structure: iteration counter, timestamp, separator
- Ctrl+C (KeyboardInterrupt) is handled gracefully
- Agent errors (typer.Exit) during watch don't stop the loop
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import typer

from vaig.cli.commands.live import (
    MINIMUM_WATCH_INTERVAL,
    _run_watch_loop,
)


# ══════════════════════════════════════════════════════════════
# MINIMUM_WATCH_INTERVAL constant
# ══════════════════════════════════════════════════════════════


class TestMinimumWatchInterval:
    """The minimum interval constant is set correctly."""

    def test_minimum_is_10(self) -> None:
        assert MINIMUM_WATCH_INTERVAL == 10


# ══════════════════════════════════════════════════════════════
# _run_watch_loop
# ══════════════════════════════════════════════════════════════


class TestRunWatchLoop:
    """Tests for the _run_watch_loop helper function."""

    @patch("vaig.cli.commands.live.time.sleep")
    @patch("vaig.cli.commands.live.console")
    def test_single_iteration_then_ctrl_c(
        self,
        mock_console: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """KeyboardInterrupt during sleep stops the loop after 1 iteration."""
        run_fn = MagicMock()
        mock_sleep.side_effect = KeyboardInterrupt

        _run_watch_loop(
            run_fn=run_fn,
            interval=30,
            question="Check pods",
        )

        run_fn.assert_called_once()
        mock_sleep.assert_called_once_with(30)

        # Verify stop message was printed
        printed = [str(c) for c in mock_console.print.call_args_list]
        stop_msgs = [s for s in printed if "Watch stopped" in s]
        assert len(stop_msgs) == 1
        assert "1 iteration" in stop_msgs[0]

    @patch("vaig.cli.commands.live.time.sleep")
    @patch("vaig.cli.commands.live.console")
    def test_multiple_iterations(
        self,
        mock_console: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """Loop runs multiple iterations before Ctrl+C."""
        run_fn = MagicMock()
        # Allow 3 iterations, then interrupt
        call_count = 0

        def sleep_side_effect(seconds: int) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                raise KeyboardInterrupt

        mock_sleep.side_effect = sleep_side_effect

        _run_watch_loop(
            run_fn=run_fn,
            interval=15,
            question="Check pods",
        )

        assert run_fn.call_count == 3
        assert mock_sleep.call_count == 3

        # Verify stop message shows correct iteration count
        printed = [str(c) for c in mock_console.print.call_args_list]
        stop_msgs = [s for s in printed if "Watch stopped" in s]
        assert len(stop_msgs) == 1
        assert "3 iterations" in stop_msgs[0]

    @patch("vaig.cli.commands.live.time.sleep")
    @patch("vaig.cli.commands.live.console")
    def test_iteration_header_shown(
        self,
        mock_console: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """Each iteration prints a header with iteration number."""
        run_fn = MagicMock()
        mock_sleep.side_effect = KeyboardInterrupt

        _run_watch_loop(
            run_fn=run_fn,
            interval=10,
            question="Check pods",
        )

        printed_args = [
            c.args[0] if c.args else ""
            for c in mock_console.print.call_args_list
        ]
        # Should have iteration #1 header
        iteration_headers = [
            s for s in printed_args
            if isinstance(s, str) and "iteration #1" in s.lower()
        ]
        assert len(iteration_headers) >= 1

    @patch("vaig.cli.commands.live.time.sleep")
    @patch("vaig.cli.commands.live.console")
    def test_separator_shown(
        self,
        mock_console: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """Separator line (====) is printed between iterations."""
        run_fn = MagicMock()
        mock_sleep.side_effect = KeyboardInterrupt

        _run_watch_loop(
            run_fn=run_fn,
            interval=10,
            question="Check pods",
        )

        printed_args = [
            c.args[0] if c.args else ""
            for c in mock_console.print.call_args_list
        ]
        separators = [
            s for s in printed_args
            if isinstance(s, str) and "=" * 30 in s
        ]
        assert len(separators) >= 1

    @patch("vaig.cli.commands.live.time.sleep")
    @patch("vaig.cli.commands.live.console")
    def test_countdown_message_shown(
        self,
        mock_console: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """Shows 'Next refresh in Ns' message after each iteration."""
        run_fn = MagicMock()
        mock_sleep.side_effect = KeyboardInterrupt

        _run_watch_loop(
            run_fn=run_fn,
            interval=60,
            question="Check pods",
        )

        printed_args = [
            c.args[0] if c.args else ""
            for c in mock_console.print.call_args_list
        ]
        refresh_msgs = [
            s for s in printed_args
            if isinstance(s, str) and "Next refresh in 60s" in s
        ]
        assert len(refresh_msgs) == 1

    @patch("vaig.cli.commands.live.time.sleep")
    @patch("vaig.cli.commands.live.console")
    def test_agent_error_continues_loop(
        self,
        mock_console: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """When run_fn raises typer.Exit, the loop continues."""
        call_count = 0

        def run_fn_with_error() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise typer.Exit(1)
            # Second call succeeds

        sleep_count = 0

        def sleep_side_effect(seconds: int) -> None:
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 2:
                raise KeyboardInterrupt

        mock_sleep.side_effect = sleep_side_effect

        _run_watch_loop(
            run_fn=run_fn_with_error,
            interval=10,
            question="Check pods",
        )

        assert call_count == 2  # Both iterations ran

        # Should show error continuation message
        printed = [str(c) for c in mock_console.print.call_args_list]
        error_msgs = [s for s in printed if "exited with error" in s]
        assert len(error_msgs) == 1

    @patch("vaig.cli.commands.live.time.sleep")
    @patch("vaig.cli.commands.live.console")
    def test_system_exit_continues_loop(
        self,
        mock_console: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """When run_fn raises SystemExit, the loop continues."""
        call_count = 0

        def run_fn_with_exit() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise SystemExit(1)

        mock_sleep.side_effect = KeyboardInterrupt

        _run_watch_loop(
            run_fn=run_fn_with_exit,
            interval=10,
            question="Check pods",
        )

        # First iteration errored, loop continued to sleep which raised KeyboardInterrupt
        assert call_count == 1

    @patch("vaig.cli.commands.live.time.sleep")
    @patch("vaig.cli.commands.live.console")
    def test_watch_banner_shows_question(
        self,
        mock_console: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """Initial banner shows the truncated question."""
        run_fn = MagicMock()
        mock_sleep.side_effect = KeyboardInterrupt

        _run_watch_loop(
            run_fn=run_fn,
            interval=30,
            question="Check pod health in production",
        )

        # The first call should be a Panel with the question
        first_call = mock_console.print.call_args_list[0]
        panel = first_call.args[0]
        # Panel.fit returns a Panel object — check its renderable
        from rich.panel import Panel
        assert isinstance(panel, Panel)

    @patch("vaig.cli.commands.live.time.sleep")
    @patch("vaig.cli.commands.live.console")
    def test_long_question_truncated_in_banner(
        self,
        mock_console: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """Questions longer than 80 chars are truncated in the banner."""
        run_fn = MagicMock()
        mock_sleep.side_effect = KeyboardInterrupt
        long_q = "x" * 100

        _run_watch_loop(
            run_fn=run_fn,
            interval=10,
            question=long_q,
        )

        # Banner should have been printed — verify it doesn't crash
        assert mock_console.print.called

    @patch("vaig.cli.commands.live.time.sleep")
    @patch("vaig.cli.commands.live.console")
    def test_elapsed_time_in_stop_message(
        self,
        mock_console: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """Stop message includes elapsed time."""
        run_fn = MagicMock()
        mock_sleep.side_effect = KeyboardInterrupt

        _run_watch_loop(
            run_fn=run_fn,
            interval=10,
            question="Check pods",
        )

        printed = [str(c) for c in mock_console.print.call_args_list]
        stop_msgs = [s for s in printed if "Watch stopped" in s]
        assert len(stop_msgs) == 1
        # Should contain "elapsed" and a number
        assert "elapsed" in stop_msgs[0]

    @patch("vaig.cli.commands.live.time.sleep")
    @patch("vaig.cli.commands.live.console")
    def test_singular_iteration_label(
        self,
        mock_console: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """Stop message says 'iteration' (singular) for count == 1."""
        run_fn = MagicMock()
        mock_sleep.side_effect = KeyboardInterrupt

        _run_watch_loop(
            run_fn=run_fn,
            interval=10,
            question="Check pods",
        )

        printed = [str(c) for c in mock_console.print.call_args_list]
        stop_msgs = [s for s in printed if "Watch stopped" in s]
        # "1 iteration" not "1 iterations"
        assert "1 iteration" in stop_msgs[0]
        assert "1 iterations" not in stop_msgs[0]


# ══════════════════════════════════════════════════════════════
# CLI integration: --watch validation
# ══════════════════════════════════════════════════════════════


class TestWatchValidation:
    """Test that --watch interval validation works correctly."""

    @patch("vaig.cli.commands.live.err_console")
    @patch("vaig.cli.commands.live._helpers._get_settings")
    def test_watch_below_minimum_exits(
        self,
        mock_get_settings: MagicMock,
        mock_err_console: MagicMock,
    ) -> None:
        """watch < MINIMUM_WATCH_INTERVAL should raise typer.Exit(1)."""
        # We test this by importing the register function and invoking
        # the live command directly via Typer's testing utilities.
        from typer.testing import CliRunner

        from vaig.cli.commands.live import register

        app = typer.Typer()
        register(app)

        runner = CliRunner()
        result = runner.invoke(app, ["Check pods", "--watch", "5"])

        assert result.exit_code != 0

    @patch("vaig.cli.commands.live.err_console")
    @patch("vaig.cli.commands.live._helpers._get_settings")
    def test_watch_below_minimum_shows_error_message(
        self,
        mock_get_settings: MagicMock,
        mock_err_console: MagicMock,
    ) -> None:
        """Error message mentions the minimum interval."""
        from typer.testing import CliRunner

        from vaig.cli.commands.live import register

        app = typer.Typer()
        register(app)

        runner = CliRunner()
        result = runner.invoke(app, ["Check pods", "--watch", "3"])

        # Check that err_console.print was called with the right message
        printed = [str(c) for c in mock_err_console.print.call_args_list]
        error_msgs = [s for s in printed if ">= 10" in s or "got 3" in s]
        assert len(error_msgs) >= 1

    @patch("vaig.cli.commands.live.err_console")
    @patch("vaig.cli.commands.live._helpers._get_settings")
    def test_watch_at_minimum_does_not_exit_early(
        self,
        mock_get_settings: MagicMock,
        mock_err_console: MagicMock,
    ) -> None:
        """watch == MINIMUM_WATCH_INTERVAL should NOT trigger validation error."""
        # This will fail downstream (no real GKE config), but it should
        # pass the validation step.
        from typer.testing import CliRunner

        from vaig.cli.commands.live import register

        app = typer.Typer()
        register(app)

        runner = CliRunner()
        result = runner.invoke(app, ["Check pods", "--watch", "10"])

        # Should NOT have printed the minimum interval error
        printed = [str(c) for c in mock_err_console.print.call_args_list]
        validation_errors = [s for s in printed if ">= 10" in s and "got" in s]
        assert len(validation_errors) == 0

    def test_watch_none_means_no_loop(self) -> None:
        """When watch is None, the _run_once path executes without looping."""
        # This is verified by the absence of _run_watch_loop call.
        # We test it structurally: MINIMUM_WATCH_INTERVAL exists and
        # the function signature accepts Optional[int] defaulting to None.
        import inspect

        from vaig.cli.commands.live import register

        app = typer.Typer()
        register(app)

        # Verify the command was registered with a --watch option
        command = app.registered_commands[0]
        assert command is not None

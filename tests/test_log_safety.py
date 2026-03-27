"""Tests for _SafeRichHandler and _make_console — Windows non-ANSI terminal safety."""

from __future__ import annotations

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest
from rich.logging import RichHandler

from vaig.core.log import _make_console, _SafeRichHandler, _stderr_console


class TestSafeRichHandler:
    """Verify that _SafeRichHandler falls back to plain text on OSError (WinError 1)."""

    def _make_handler(self) -> _SafeRichHandler:
        return _SafeRichHandler(
            console=_stderr_console,
            show_time=False,
            show_level=False,
            show_path=False,
            markup=False,
            rich_tracebacks=False,
        )

    def _make_record(self, msg: str = "test message") -> logging.LogRecord:
        return logging.LogRecord(
            name="vaig.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg=msg,
            args=(),
            exc_info=None,
        )

    def test_emit_succeeds_normally(self) -> None:
        """When no error occurs, emit works as expected."""
        handler = self._make_handler()
        record = self._make_record("hello from test")
        # Should not raise
        with patch.object(RichHandler, "emit", autospec=True, return_value=None) as mock_emit:
            handler.emit(record)
            mock_emit.assert_called_once_with(handler, record)

    def test_emit_falls_back_on_oserror(self) -> None:
        """OSError raised by RichHandler.emit must NOT propagate — fallback to StreamHandler."""
        handler = self._make_handler()
        record = self._make_record("windows thread message")

        with patch("rich.logging.RichHandler.emit", side_effect=OSError("[WinError 1] thread stderr")):
            with patch.object(logging.StreamHandler, "emit") as mock_fallback:
                # Must NOT raise — falls back to StreamHandler.emit
                handler.emit(record)
                # StreamHandler.emit is called as an unbound method: (self, record)
                mock_fallback.assert_called_once_with(handler, record)

    def test_emit_fallback_preserves_record(self) -> None:
        """The log record must reach StreamHandler.emit, not be silently lost."""
        handler = self._make_handler()
        record = self._make_record("important message that must not be lost")

        captured: list[logging.LogRecord] = []

        def _capture_emit(self_arg: object, r: logging.LogRecord) -> None:
            captured.append(r)

        with patch("rich.logging.RichHandler.emit", side_effect=OSError("[WinError 1]")):
            with patch.object(logging.StreamHandler, "emit", side_effect=_capture_emit):
                handler.emit(record)

        assert len(captured) == 1
        assert captured[0].getMessage() == "important message that must not be lost"

    def test_emit_does_not_swallow_other_exceptions(self) -> None:
        """Non-OSError exceptions must propagate normally so bugs aren't hidden."""
        handler = self._make_handler()
        record = self._make_record("boom")

        with patch("rich.logging.RichHandler.emit", side_effect=ValueError("unexpected")):
            with pytest.raises(ValueError, match="unexpected"):
                handler.emit(record)

    def test_handler_is_subclass_of_rich_handler(self) -> None:
        """_SafeRichHandler must be a RichHandler subclass."""
        assert issubclass(_SafeRichHandler, RichHandler)

    def test_multiple_oserrors_trigger_fallback_each_time(self) -> None:
        """Multiple OSErrors must each trigger fallback — none silently lost."""
        handler = self._make_handler()
        records = [self._make_record(f"msg {i}") for i in range(5)]
        fallback_calls: list[logging.LogRecord] = []

        def _capture(self_arg: object, r: logging.LogRecord) -> None:
            fallback_calls.append(r)

        with patch("rich.logging.RichHandler.emit", side_effect=OSError("thread error")):
            with patch.object(logging.StreamHandler, "emit", side_effect=_capture):
                for record in records:
                    handler.emit(record)

        # All 5 records must have reached the fallback
        assert len(fallback_calls) == 5


class TestMakeConsole:
    """Tests for _make_console() factory — Windows non-ANSI resilience."""

    def test_tty_stream_returns_regular_console(self) -> None:
        """When stderr is a TTY, return a normal Console (ANSI enabled)."""
        mock_stream = MagicMock()
        mock_stream.isatty.return_value = True

        with patch.object(sys, "stderr", mock_stream):
            console = _make_console(stderr=True)

        # force_terminal should be None (Rich default) meaning auto-detect
        assert console._force_terminal is None or console._force_terminal is True

    def test_non_tty_stream_returns_no_color_console(self) -> None:
        """When stderr is NOT a TTY, return Console with no_color=True."""
        mock_stream = MagicMock()
        mock_stream.isatty.return_value = False

        with patch.object(sys, "stderr", mock_stream):
            console = _make_console(stderr=True)

        assert console.no_color is True

    def test_non_tty_stream_returns_force_terminal_false(self) -> None:
        """When stderr is NOT a TTY, force_terminal must be False to prevent WinError 1."""
        mock_stream = MagicMock()
        mock_stream.isatty.return_value = False

        with patch.object(sys, "stderr", mock_stream):
            console = _make_console(stderr=True)

        assert console._force_terminal is False

    def test_stdout_non_tty_disables_ansi(self) -> None:
        """Works for stdout as well — non-TTY disables ANSI formatting."""
        mock_stream = MagicMock()
        mock_stream.isatty.return_value = False

        with patch.object(sys, "stdout", mock_stream):
            console = _make_console(stderr=False)

        assert console._force_terminal is False
        assert console.no_color is True

    def test_stream_without_isatty_treated_as_non_tty(self) -> None:
        """Streams without isatty() (AttributeError) are treated as non-TTY."""
        mock_stream = MagicMock(spec=[])  # No isatty attribute

        with patch.object(sys, "stderr", mock_stream):
            console = _make_console(stderr=True)

        # Should have picked the safe non-TTY path
        assert console._force_terminal is False
        assert console.no_color is True

    def test_stderr_console_uses_stderr_stream(self) -> None:
        """Console created with stderr=True should target stderr."""
        mock_stream = MagicMock()
        mock_stream.isatty.return_value = False

        with patch.object(sys, "stderr", mock_stream):
            console = _make_console(stderr=True)

        assert console.stderr is True

    def test_module_level_stderr_console_is_console_instance(self) -> None:
        """_stderr_console must be a rich.console.Console instance."""
        from rich.console import Console

        assert isinstance(_stderr_console, Console)

    def test_module_level_stderr_console_targets_stderr(self) -> None:
        """The module-level _stderr_console must write to stderr, not stdout."""
        assert _stderr_console.stderr is True

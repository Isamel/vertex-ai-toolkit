"""Tests for _SafeRichHandler — Windows thread OSError safety (Bug A)."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
from rich.logging import RichHandler

from vaig.core.log import _SafeRichHandler, _stderr_console


class TestSafeRichHandler:
    """Verify that _SafeRichHandler silently absorbs OSError raised by RichHandler.emit."""

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

    def test_emit_swallows_oserror(self) -> None:
        """OSError raised by the underlying RichHandler.emit must be silently dropped."""
        handler = self._make_handler()
        record = self._make_record("windows thread message")

        with patch("rich.logging.RichHandler.emit", side_effect=OSError("[WinError 1] thread stderr")):
            # Must NOT raise — this is the fix for Bug A
            handler.emit(record)

    def test_emit_does_not_swallow_other_exceptions(self) -> None:
        """Non-OSError exceptions must propagate normally so bugs aren't hidden."""
        handler = self._make_handler()
        record = self._make_record("boom")

        with patch("rich.logging.RichHandler.emit", side_effect=ValueError("unexpected")):
            with pytest.raises(ValueError, match="unexpected"):
                handler.emit(record)

    def test_handler_is_subclass_of_rich_handler(self) -> None:
        """_SafeRichHandler must be a RichHandler subclass."""
        from rich.logging import RichHandler

        assert issubclass(_SafeRichHandler, RichHandler)

    def test_multiple_oserrors_do_not_accumulate(self) -> None:
        """Multiple OSErrors in rapid succession must all be silently dropped."""
        handler = self._make_handler()
        records = [self._make_record(f"msg {i}") for i in range(5)]

        with patch("rich.logging.RichHandler.emit", side_effect=OSError("thread error")):
            for record in records:
                # None should raise
                handler.emit(record)

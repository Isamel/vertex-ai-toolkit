"""Tests for centralized logging configuration (core/log.py)."""

from __future__ import annotations

import logging

import pytest
from rich.logging import RichHandler

from vaig.core.log import reset_logging, setup_logging


@pytest.fixture(autouse=True)
def _clean_logging() -> None:
    """Reset logging state before and after each test."""
    reset_logging()
    yield  # type: ignore[misc]
    reset_logging()


class TestSetupLogging:
    """Tests for setup_logging() function."""

    def test_attaches_handler_to_vaig_logger(self) -> None:
        setup_logging("INFO")
        vaig_logger = logging.getLogger("vaig")
        assert len(vaig_logger.handlers) == 1
        assert isinstance(vaig_logger.handlers[0], RichHandler)

    def test_sets_correct_level(self) -> None:
        setup_logging("DEBUG")
        vaig_logger = logging.getLogger("vaig")
        assert vaig_logger.level == logging.DEBUG

    def test_level_is_case_insensitive(self) -> None:
        setup_logging("info")
        vaig_logger = logging.getLogger("vaig")
        assert vaig_logger.level == logging.INFO

    def test_default_level_is_warning(self) -> None:
        setup_logging()
        vaig_logger = logging.getLogger("vaig")
        assert vaig_logger.level == logging.WARNING

    def test_propagate_is_false(self) -> None:
        """Ensure vaig logger doesn't leak to root logger."""
        setup_logging("INFO")
        vaig_logger = logging.getLogger("vaig")
        assert vaig_logger.propagate is False

    def test_child_loggers_inherit_handler(self) -> None:
        """Child loggers (e.g. vaig.core.client) should inherit the handler."""
        setup_logging("DEBUG")
        child = logging.getLogger("vaig.core.client")
        # Child should have no OWN handlers but effective handler from parent
        assert len(child.handlers) == 0
        # But the parent has the handler
        vaig_logger = logging.getLogger("vaig")
        assert len(vaig_logger.handlers) == 1

    def test_handler_level_matches_logger_level(self) -> None:
        setup_logging("ERROR")
        vaig_logger = logging.getLogger("vaig")
        handler = vaig_logger.handlers[0]
        assert handler.level == logging.ERROR

    def test_invalid_level_falls_back_to_warning(self) -> None:
        """Invalid level string should fall back to WARNING via getattr default."""
        setup_logging("INVALID_LEVEL")
        vaig_logger = logging.getLogger("vaig")
        assert vaig_logger.level == logging.WARNING

    def test_all_standard_levels(self) -> None:
        """Verify all standard log levels work."""
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        for name, expected in levels.items():
            reset_logging()
            setup_logging(name)
            vaig_logger = logging.getLogger("vaig")
            assert vaig_logger.level == expected, f"Failed for level {name}"


class TestIdempotency:
    """Tests for setup_logging() idempotency."""

    def test_multiple_calls_add_only_one_handler(self) -> None:
        """Calling setup_logging() multiple times should not add duplicate handlers."""
        setup_logging("INFO")
        setup_logging("DEBUG")  # Should be a no-op
        setup_logging("ERROR")  # Should be a no-op
        vaig_logger = logging.getLogger("vaig")
        assert len(vaig_logger.handlers) == 1

    def test_second_call_preserves_first_level(self) -> None:
        """Once configured, subsequent calls should not change the level."""
        setup_logging("DEBUG")
        setup_logging("ERROR")  # No-op
        vaig_logger = logging.getLogger("vaig")
        assert vaig_logger.level == logging.DEBUG


class TestResetLogging:
    """Tests for reset_logging() function."""

    def test_removes_all_handlers(self) -> None:
        setup_logging("INFO")
        vaig_logger = logging.getLogger("vaig")
        assert len(vaig_logger.handlers) == 1
        reset_logging()
        assert len(vaig_logger.handlers) == 0

    def test_resets_level_to_warning(self) -> None:
        setup_logging("DEBUG")
        reset_logging()
        vaig_logger = logging.getLogger("vaig")
        assert vaig_logger.level == logging.WARNING

    def test_allows_reconfiguration(self) -> None:
        """After reset, setup_logging() should work again."""
        setup_logging("DEBUG")
        reset_logging()
        setup_logging("ERROR")
        vaig_logger = logging.getLogger("vaig")
        assert vaig_logger.level == logging.ERROR
        assert len(vaig_logger.handlers) == 1

    def test_reset_is_idempotent(self) -> None:
        """Calling reset multiple times without setup should be safe."""
        reset_logging()
        reset_logging()
        reset_logging()
        vaig_logger = logging.getLogger("vaig")
        assert len(vaig_logger.handlers) == 0


class TestShowPath:
    """Tests for the show_path parameter."""

    def test_show_path_false_by_default(self) -> None:
        setup_logging("INFO")
        vaig_logger = logging.getLogger("vaig")
        handler = vaig_logger.handlers[0]
        assert isinstance(handler, RichHandler)
        # RichHandler stores show_path inside its _log_render object, not as a direct attr
        assert handler._log_render.show_path is False

    def test_show_path_enabled(self) -> None:
        setup_logging("DEBUG", show_path=True)
        vaig_logger = logging.getLogger("vaig")
        handler = vaig_logger.handlers[0]
        assert isinstance(handler, RichHandler)
        # RichHandler stores show_path inside its _log_render object, not as a direct attr
        assert handler._log_render.show_path is True


class TestStderrOutput:
    """Tests verifying logs go to stderr, not stdout."""

    def test_handler_uses_stderr_console(self) -> None:
        setup_logging("INFO")
        vaig_logger = logging.getLogger("vaig")
        handler = vaig_logger.handlers[0]
        assert isinstance(handler, RichHandler)
        # RichHandler stores the console as `console` attribute
        assert handler.console.stderr is True


class TestLogCapture:
    """Tests verifying actual log messages are captured."""

    def test_info_message_captured_at_info_level(self, capfd: pytest.CaptureFixture[str]) -> None:
        setup_logging("INFO")
        logger = logging.getLogger("vaig.test.capture")
        logger.info("test info message")
        _, stderr = capfd.readouterr()
        assert "test info message" in stderr

    def test_debug_not_captured_at_info_level(self, capfd: pytest.CaptureFixture[str]) -> None:
        setup_logging("INFO")
        logger = logging.getLogger("vaig.test.capture2")
        logger.debug("this should not appear")
        _, stderr = capfd.readouterr()
        assert "this should not appear" not in stderr

    def test_nothing_on_stdout(self, capfd: pytest.CaptureFixture[str]) -> None:
        """Logs should NEVER appear on stdout (would break piped output)."""
        setup_logging("DEBUG")
        logger = logging.getLogger("vaig.test.stdout_check")
        logger.info("stderr only")
        logger.warning("stderr only warning")
        stdout, _ = capfd.readouterr()
        assert stdout == ""

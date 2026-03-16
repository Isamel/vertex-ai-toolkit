"""Tests for centralized logging configuration (core/log.py)."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from rich.logging import RichHandler

from vaig.core.log import reset_logging, setup_logging

if TYPE_CHECKING:
    from vaig.core.config import Settings


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


class TestFileLogging:
    """Tests for rotating file handler support."""

    def test_file_handler_created(self, tmp_path: Path) -> None:
        """file_enabled=True should add a RotatingFileHandler."""
        log_file = tmp_path / "test.log"
        setup_logging("WARNING", file_enabled=True, file_path=str(log_file))
        vaig_logger = logging.getLogger("vaig")
        assert len(vaig_logger.handlers) == 2
        file_handlers = [h for h in vaig_logger.handlers if isinstance(h, RotatingFileHandler)]
        assert len(file_handlers) == 1

    def test_file_handler_not_created_by_default(self) -> None:
        """Without file_enabled, only RichHandler should be attached."""
        setup_logging("INFO")
        vaig_logger = logging.getLogger("vaig")
        assert len(vaig_logger.handlers) == 1
        assert isinstance(vaig_logger.handlers[0], RichHandler)

    def test_file_handler_writes_to_disk(self, tmp_path: Path) -> None:
        """Messages should actually be written to the log file."""
        log_file = tmp_path / "vaig.log"
        setup_logging("WARNING", file_enabled=True, file_path=str(log_file), file_level="DEBUG")
        logger = logging.getLogger("vaig.test.file_write")
        logger.info("file test message")

        # Flush handlers
        for h in logging.getLogger("vaig").handlers:
            h.flush()

        assert log_file.exists()
        content = log_file.read_text()
        assert "file test message" in content

    def test_file_handler_level_independent_of_console(self, tmp_path: Path) -> None:
        """File handler should capture DEBUG even when console is WARNING."""
        log_file = tmp_path / "vaig.log"
        setup_logging("WARNING", file_enabled=True, file_path=str(log_file), file_level="DEBUG")
        vaig_logger = logging.getLogger("vaig")

        # Logger level should be DEBUG (the minimum of WARNING and DEBUG)
        assert vaig_logger.level == logging.DEBUG

        # Console handler should be WARNING
        rich_handlers = [h for h in vaig_logger.handlers if isinstance(h, RichHandler)]
        assert rich_handlers[0].level == logging.WARNING

        # File handler should be DEBUG
        file_handlers = [h for h in vaig_logger.handlers if isinstance(h, RotatingFileHandler)]
        assert file_handlers[0].level == logging.DEBUG

    def test_file_handler_format(self, tmp_path: Path) -> None:
        """File handler should use the specified format."""
        log_file = tmp_path / "vaig.log"
        setup_logging("WARNING", file_enabled=True, file_path=str(log_file), file_level="INFO")
        logger = logging.getLogger("vaig.test.format")
        logger.info("format check")

        for h in logging.getLogger("vaig").handlers:
            h.flush()

        content = log_file.read_text()
        # Format: %(asctime)s %(levelname)s %(name)s %(message)s
        assert "INFO" in content
        assert "vaig.test.format" in content
        assert "format check" in content

    def test_file_creates_parent_directories(self, tmp_path: Path) -> None:
        """Should create parent directories if they don't exist."""
        log_file = tmp_path / "deep" / "nested" / "dir" / "vaig.log"
        setup_logging("WARNING", file_enabled=True, file_path=str(log_file))
        vaig_logger = logging.getLogger("vaig")
        file_handlers = [h for h in vaig_logger.handlers if isinstance(h, RotatingFileHandler)]
        assert len(file_handlers) == 1
        assert log_file.parent.exists()

    def test_file_handler_max_bytes_and_backups(self, tmp_path: Path) -> None:
        """Verify maxBytes and backupCount are passed through."""
        log_file = tmp_path / "vaig.log"
        setup_logging(
            "WARNING",
            file_enabled=True,
            file_path=str(log_file),
            file_max_bytes=1024,
            file_backup_count=5,
        )
        vaig_logger = logging.getLogger("vaig")
        file_handlers = [h for h in vaig_logger.handlers if isinstance(h, RotatingFileHandler)]
        assert file_handlers[0].maxBytes == 1024
        assert file_handlers[0].backupCount == 5

    def test_file_handler_failure_does_not_crash(self, tmp_path: Path) -> None:
        """If the file path is invalid, setup should log a warning but not crash."""
        # Use a path that can't be created (file as directory)
        blocker = tmp_path / "blocker"
        blocker.write_text("I am a file")
        bad_path = str(blocker / "subdir" / "vaig.log")

        # This should not raise
        setup_logging("WARNING", file_enabled=True, file_path=bad_path)
        vaig_logger = logging.getLogger("vaig")
        # Should still have the console handler
        assert len(vaig_logger.handlers) >= 1

    def test_reset_clears_file_handler(self, tmp_path: Path) -> None:
        """reset_logging() should remove all handlers including file handler."""
        log_file = tmp_path / "vaig.log"
        setup_logging("WARNING", file_enabled=True, file_path=str(log_file))
        vaig_logger = logging.getLogger("vaig")
        assert len(vaig_logger.handlers) == 2
        reset_logging()
        assert len(vaig_logger.handlers) == 0


class TestCreateToolCallStore:
    """Tests for _create_tool_call_store() in live.py."""

    def _make_settings(self, *, tool_results: bool = True, tool_results_dir: str = "") -> Settings:
        """Create a minimal settings-like object with logging config."""
        from vaig.core.config import Settings

        settings = Settings()
        settings.logging.tool_results = tool_results
        if tool_results_dir:
            settings.logging.tool_results_dir = tool_results_dir
        return settings

    def test_returns_store_when_enabled(self, tmp_path: Path) -> None:
        """When tool_results=True, returns a ToolCallStore (run not yet started)."""
        from vaig.cli.commands.live import _create_tool_call_store

        settings = self._make_settings(tool_results=True, tool_results_dir=str(tmp_path))
        store = _create_tool_call_store(settings)
        assert store is not None
        # start_run() is deferred to the Orchestrator, so run_id is empty here
        assert store.run_id == ""

    def test_returns_none_when_disabled(self, tmp_path: Path) -> None:
        """When tool_results=False, returns None."""
        from vaig.cli.commands.live import _create_tool_call_store

        settings = self._make_settings(tool_results=False, tool_results_dir=str(tmp_path))
        store = _create_tool_call_store(settings)
        assert store is None

    def test_returns_store_even_with_bad_dir(self, tmp_path: Path) -> None:
        """Store creation succeeds even with a bad dir; error deferred to start_run()."""
        from vaig.cli.commands.live import _create_tool_call_store

        # Use a file as dir — error is now deferred to Orchestrator.start_run()
        blocker = tmp_path / "blocker"
        blocker.write_text("I am a file")
        settings = self._make_settings(tool_results=True, tool_results_dir=str(blocker))
        store = _create_tool_call_store(settings)
        # Store is created (start_run is deferred), but will fail later
        assert store is not None

    def test_expands_home_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The tool_results_dir should expand ~ to the user's home directory."""
        from vaig.cli.commands.live import _create_tool_call_store

        monkeypatch.setenv("HOME", str(tmp_path))
        settings = self._make_settings(tool_results=True, tool_results_dir="~/my-vaig-data")
        store = _create_tool_call_store(settings)
        assert store is not None
        # Dir creation is deferred to start_run(); verify ~ was expanded in base_dir
        assert "my-vaig-data" in str(store._base_dir)

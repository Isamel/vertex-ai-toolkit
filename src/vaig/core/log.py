"""Logging configuration — centralized setup with RichHandler on stderr.

This module has ZERO imports from vaig to prevent circular dependencies.
It only depends on stdlib ``logging`` and ``rich``.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

# Stderr console for diagnostics — never interferes with stdout output.
_stderr_console = Console(stderr=True)


class _SafeRichHandler(RichHandler):
    """RichHandler that silently drops OSError on Windows thread contexts."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
        except OSError:
            # Windows: stderr not writable from background threads using
            # concurrent.futures — the handle is invalid in thread context.
            # Silently drop rather than crash the pipeline.
            pass

# Sentinel to make setup_logging() idempotent.
_configured = False


def setup_logging(
    level: str = "WARNING",
    *,
    show_path: bool = False,
    file_enabled: bool = False,
    file_path: str = "~/.vaig/logs/vaig.log",
    file_level: str = "DEBUG",
    file_max_bytes: int = 5_242_880,
    file_backup_count: int = 3,
) -> None:
    """Configure the ``vaig`` logger hierarchy with a RichHandler on stderr.

    This attaches a single handler to the ``vaig`` root logger.  All child
    loggers (``vaig.core.client``, ``vaig.agents.orchestrator``, etc.) inherit
    the handler automatically — no per-module configuration needed.

    When ``file_enabled`` is True, a ``RotatingFileHandler`` is also attached
    that writes DEBUG-level logs to disk, independent of the console level.

    Calling this function multiple times is safe (idempotent).

    Args:
        level: Python log level name for **console** output
            (DEBUG, INFO, WARNING, ERROR, CRITICAL).  Case-insensitive.
        show_path: If True, include the module path in console log output.
        file_enabled: If True, also write logs to a rotating file.
        file_path: Path to the log file (supports ``~`` expansion).
        file_level: Log level for the file handler (always DEBUG by default).
        file_max_bytes: Maximum size per log file before rotation (default 5 MB).
        file_backup_count: Number of backup files to keep (default 3).
    """
    global _configured  # noqa: PLW0603

    if _configured:
        return

    numeric_level = getattr(logging, level.upper(), logging.WARNING)

    vaig_logger = logging.getLogger("vaig")
    # Set logger to the most permissive level among handlers so
    # file handler can capture DEBUG even when console is WARNING.
    file_numeric = getattr(logging, file_level.upper(), logging.DEBUG) if file_enabled else numeric_level
    vaig_logger.setLevel(min(numeric_level, file_numeric))
    vaig_logger.propagate = False  # Don't leak to root logger

    handler = _SafeRichHandler(
        console=_stderr_console,
        show_time=True,
        show_level=True,
        show_path=show_path,
        markup=False,
        rich_tracebacks=True,
        tracebacks_show_locals=numeric_level <= logging.DEBUG,
    )
    handler.setLevel(numeric_level)

    vaig_logger.addHandler(handler)

    # ── Optional file handler ────────────────────────────────
    if file_enabled:
        try:
            log_path = Path(file_path).expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                filename=str(log_path),
                maxBytes=file_max_bytes,
                backupCount=file_backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(file_numeric)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"),
            )
            vaig_logger.addHandler(file_handler)
        except Exception:  # noqa: BLE001
            # Never crash the app because of logging setup
            vaig_logger.warning(
                "Failed to set up file logging at %s", file_path, exc_info=True,
            )

    _configured = True


def reset_logging() -> None:
    """Remove all handlers and reset state (for testing)."""
    global _configured  # noqa: PLW0603

    vaig_logger = logging.getLogger("vaig")
    vaig_logger.handlers.clear()
    vaig_logger.setLevel(logging.WARNING)
    _configured = False

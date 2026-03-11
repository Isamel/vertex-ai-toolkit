"""Logging configuration — centralized setup with RichHandler on stderr.

This module has ZERO imports from vaig to prevent circular dependencies.
It only depends on stdlib ``logging`` and ``rich``.
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

# Stderr console for diagnostics — never interferes with stdout output.
_stderr_console = Console(stderr=True)

# Sentinel to make setup_logging() idempotent.
_configured = False


def setup_logging(level: str = "WARNING", *, show_path: bool = False) -> None:
    """Configure the ``vaig`` logger hierarchy with a RichHandler on stderr.

    This attaches a single handler to the ``vaig`` root logger.  All child
    loggers (``vaig.core.client``, ``vaig.agents.orchestrator``, etc.) inherit
    the handler automatically — no per-module configuration needed.

    Calling this function multiple times is safe (idempotent).

    Args:
        level: Python log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Case-insensitive.
        show_path: If True, include the module path in log output.
    """
    global _configured  # noqa: PLW0603

    if _configured:
        return

    numeric_level = getattr(logging, level.upper(), logging.WARNING)

    vaig_logger = logging.getLogger("vaig")
    vaig_logger.setLevel(numeric_level)
    vaig_logger.propagate = False  # Don't leak to root logger

    handler = RichHandler(
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
    _configured = True


def reset_logging() -> None:
    """Remove all handlers and reset state (for testing)."""
    global _configured  # noqa: PLW0603

    vaig_logger = logging.getLogger("vaig")
    vaig_logger.handlers.clear()
    vaig_logger.setLevel(logging.WARNING)
    _configured = False

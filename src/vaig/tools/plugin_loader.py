"""Plugin loader — discover and load custom tools from MCP servers and Python modules.

This module provides the unified entry point ``load_all_plugin_tools()``
that aggregates ``ToolDef`` objects from two plugin sources:

1. **MCP servers** — auto-registered via ``create_mcp_tools()`` when
   ``settings.mcp.auto_register`` is enabled.
2. **Python modules** — loaded from configured directories, each module
   must expose a ``register_tools() -> list[ToolDef]`` function.

Errors in individual plugins are logged and skipped — one broken plugin
never blocks agent startup.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from vaig.tools.base import ToolDef

if TYPE_CHECKING:
    from vaig.core.config import PluginConfig, Settings

logger = logging.getLogger(__name__)


def _load_python_plugins(plugin_config: PluginConfig) -> list[ToolDef]:
    """Discover and load ToolDefs from Python modules in configured directories.

    Each directory in ``plugin_config.directories`` is scanned for ``.py``
    files (non-recursive).  Each file is loaded as a module and checked for
    a ``register_tools`` callable that must return ``list[ToolDef]``.

    Args:
        plugin_config: The ``PluginConfig`` from application settings.

    Returns:
        A list of ``ToolDef`` objects from all valid plugin modules.
    """
    if not plugin_config.enabled:
        return []

    if not plugin_config.directories:
        logger.debug("Plugins enabled but no directories configured.")
        return []

    all_tools: list[ToolDef] = []

    for dir_path_str in plugin_config.directories:
        dir_path = Path(dir_path_str).resolve()

        if not dir_path.is_dir():
            logger.warning(
                "Plugin directory does not exist or is not a directory: %s",
                dir_path,
            )
            continue

        py_files = sorted(dir_path.glob("*.py"))
        if not py_files:
            logger.debug("No .py files found in plugin directory: %s", dir_path)
            continue

        for py_file in py_files:
            module_name = f"vaig_plugin_{py_file.stem}"

            try:
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec is None or spec.loader is None:
                    logger.warning(
                        "Could not create module spec for plugin: %s",
                        py_file,
                    )
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception:
                logger.warning(
                    "Failed to load plugin module '%s'. Skipping.",
                    py_file,
                    exc_info=True,
                )
                continue

            register_fn = getattr(module, "register_tools", None)
            if not callable(register_fn):
                logger.debug(
                    "Plugin '%s' has no register_tools() function. Skipping.",
                    py_file.name,
                )
                continue

            try:
                tools = register_fn()
            except Exception:
                logger.warning(
                    "register_tools() raised an error in plugin '%s'. Skipping.",
                    py_file.name,
                    exc_info=True,
                )
                continue

            if not isinstance(tools, list):
                logger.warning(
                    "register_tools() in '%s' did not return a list. Skipping.",
                    py_file.name,
                )
                continue

            all_tools.extend(tools)
            logger.info(
                "Loaded %d tool(s) from Python plugin '%s'.",
                len(tools),
                py_file.name,
            )

    return all_tools


def load_all_plugin_tools(settings: Settings) -> list[ToolDef]:
    """Discover and load tools from all configured plugin sources.

    This is the single entry point for plugin tool registration.  It
    aggregates tools from MCP servers and Python module plugins, logging
    a summary of what was loaded.

    Each source is wrapped in try/except — errors are logged and never
    crash agent startup.

    Args:
        settings: The full application ``Settings``.

    Returns:
        A combined list of ``ToolDef`` objects from all plugin sources.
    """
    mcp_tools: list[ToolDef] = []
    python_tools: list[ToolDef] = []

    # MCP auto-registration
    if settings.mcp.auto_register:
        try:
            from vaig.tools.mcp_bridge import create_mcp_tools

            mcp_tools = create_mcp_tools(settings.mcp)
        except Exception:
            logger.warning(
                "Failed to load MCP plugin tools. Skipping.",
                exc_info=True,
            )

    # Python module plugins
    if settings.plugins.enabled:
        try:
            python_tools = _load_python_plugins(settings.plugins)
        except Exception:
            logger.warning(
                "Failed to load Python plugin tools. Skipping.",
                exc_info=True,
            )

    all_tools = mcp_tools + python_tools
    total = len(all_tools)

    if total > 0:
        logger.info(
            "Loaded %d plugin tool(s) (%d from MCP, %d from Python modules).",
            total,
            len(mcp_tools),
            len(python_tools),
        )

    return all_tools

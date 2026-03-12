"""Tools package — tool definitions and registry for the coding agent."""

from vaig.tools.base import ToolDef, ToolParam, ToolRegistry, ToolResult
from vaig.tools.file_tools import (
    create_file_tools,
    edit_file,
    list_files,
    read_file,
    search_files,
    write_file,
)
from vaig.tools.shell_tools import create_shell_tools, run_command

__all__ = [
    "ToolDef",
    "ToolParam",
    "ToolRegistry",
    "ToolResult",
    "create_file_tools",
    "create_shell_tools",
    "edit_file",
    "list_files",
    "read_file",
    "run_command",
    "search_files",
    "write_file",
]

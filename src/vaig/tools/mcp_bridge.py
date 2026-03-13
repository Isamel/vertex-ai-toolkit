"""MCP bridge — connect to external MCP servers and expose their tools as ToolDefs.

This module provides a bridge between the Model Context Protocol (MCP) and
vaig's internal tool system.  It allows vaig to consume tools from any MCP
server (stdio or HTTP transport) and register them in the ``ToolRegistry``.

Usage::

    bridge = MCPBridge()
    await bridge.connect("my-server", command="npx", args=["-y", "@my/mcp-server"])
    tools = bridge.get_tool_defs("my-server")
    for t in tools:
        registry.register(t)

The bridge is intentionally kept simple: it handles tool listing and
invocation.  Resource and prompt support can be added later.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from vaig.tools.base import ToolDef, ToolParam, ToolResult

logger = logging.getLogger(__name__)

# Optional import — MCP SDK may not be installed.
try:
    from mcp import types as mcp_types
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    _MCP_AVAILABLE = True
    _MCP_IMPORT_ERROR: str | None = None
except ImportError as _exc:  # pragma: no cover
    _MCP_AVAILABLE = False
    _MCP_IMPORT_ERROR = str(_exc)

    # Provide stubs so the module can be imported without MCP.
    class StdioServerParameters:  # type: ignore[no-redef]
        """Stub."""

    class ClientSession:  # type: ignore[no-redef]
        """Stub."""

    mcp_types = None  # type: ignore[assignment]


def is_mcp_available() -> bool:
    """Return True if the MCP SDK is installed and importable."""
    return _MCP_AVAILABLE


@dataclass
class MCPServerInfo:
    """Tracked state for a connected MCP server."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    tools: list[mcp_types.Tool] = field(default_factory=list)  # type: ignore[name-defined]


def _mcp_type_to_tool_param_type(json_type: str) -> str:
    """Map JSON Schema types to ToolParam type strings."""
    mapping = {
        "string": "string",
        "integer": "integer",
        "number": "number",
        "boolean": "boolean",
        "array": "array",
        "object": "object",
    }
    return mapping.get(json_type, "string")


def _mcp_tool_to_tool_def(
    mcp_tool: mcp_types.Tool,  # type: ignore[name-defined]
    call_fn: Any,
    server_name: str,
) -> ToolDef:
    """Convert an MCP Tool to a vaig ToolDef."""
    params: list[ToolParam] = []
    schema = mcp_tool.inputSchema or {}
    properties = schema.get("properties", {})
    required_set = set(schema.get("required", []))

    for param_name, param_schema in properties.items():
        param_type = _mcp_type_to_tool_param_type(param_schema.get("type", "string"))
        param_desc = param_schema.get("description", "")
        params.append(
            ToolParam(
                name=param_name,
                type=param_type,
                description=param_desc,
                required=param_name in required_set,
            )
        )

    # Prefix with server name to avoid collisions
    tool_name = f"mcp_{server_name}_{mcp_tool.name}"

    return ToolDef(
        name=tool_name,
        description=f"[MCP:{server_name}] {mcp_tool.description or mcp_tool.name}",
        parameters=params,
        execute=call_fn,
    )


async def discover_tools(
    command: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> list[mcp_types.Tool]:  # type: ignore[name-defined]
    """Connect to an MCP server, list its tools, and disconnect.

    This is a one-shot discovery — it opens a connection, lists tools,
    and closes.  Use ``MCPBridge`` for persistent connections.

    Raises:
        RuntimeError: If the MCP SDK is not installed.
    """
    if not _MCP_AVAILABLE:
        msg = f"MCP SDK not installed: {_MCP_IMPORT_ERROR}"
        raise RuntimeError(msg)

    server_params = StdioServerParameters(
        command=command,
        args=args or [],
        env=env,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return list(result.tools)


async def call_mcp_tool(
    command: str,
    tool_name: str,
    arguments: dict[str, Any],
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> ToolResult:
    """Connect to an MCP server, call a tool, and return a ToolResult.

    This is a one-shot call — it opens a connection, calls the tool,
    and closes.  Suitable for stateless tools.

    Raises:
        RuntimeError: If the MCP SDK is not installed.
    """
    if not _MCP_AVAILABLE:
        return ToolResult(
            output=f"MCP SDK not installed: {_MCP_IMPORT_ERROR}",
            error=True,
        )

    server_params = StdioServerParameters(
        command=command,
        args=args or [],
        env=env,
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)

                # Extract text content from MCP result
                texts: list[str] = []
                is_error = result.isError or False
                for content_block in result.content:
                    if hasattr(content_block, "text"):
                        texts.append(content_block.text)
                    else:
                        texts.append(str(content_block))

                return ToolResult(
                    output="\n".join(texts) if texts else "(no output)",
                    error=is_error,
                )

    except Exception as exc:
        logger.exception("MCP tool call failed: %s/%s", command, tool_name)
        return ToolResult(output=f"MCP tool call failed: {exc}", error=True)


def create_mcp_tool_defs(
    server_name: str,
    command: str,
    tools: list[Any],
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> list[ToolDef]:
    """Create vaig ToolDef objects from a list of MCP tools.

    Each tool gets a closure that calls ``call_mcp_tool`` with the
    appropriate parameters bound via defaults.
    """
    defs: list[ToolDef] = []

    for mcp_tool in tools:
        def _make_executor(
            _cmd: str = command,
            _tool_name: str = mcp_tool.name,
            _args: list[str] | None = args,
            _env: dict[str, str] | None = env,
        ):
            def executor(**kwargs: Any) -> ToolResult:
                return asyncio.run(
                    call_mcp_tool(_cmd, _tool_name, kwargs, args=_args, env=_env)
                )
            return executor

        tool_def = _mcp_tool_to_tool_def(
            mcp_tool,
            _make_executor(),
            server_name,
        )
        defs.append(tool_def)

    return defs

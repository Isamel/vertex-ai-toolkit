"""Tests for MCP bridge, config, and CLI commands."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from vaig.core.config import MCPConfig, MCPServerConfig, Settings
from vaig.tools.base import ToolDef, ToolParam, ToolResult
from vaig.tools.mcp_bridge import (
    _mcp_tool_to_tool_def,
    _mcp_type_to_tool_param_type,
    create_mcp_tool_defs,
    create_mcp_tools,
    is_mcp_available,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures & Stubs
# ═══════════════════════════════════════════════════════════════

@dataclass
class FakeMCPTool:
    """Mimics mcp.types.Tool for testing without the SDK."""

    name: str
    description: str | None = None
    inputSchema: dict[str, Any] | None = None


@dataclass
class FakeContentBlock:
    """Mimics MCP result content blocks."""

    text: str


@dataclass
class FakeCallToolResult:
    """Mimics MCP call_tool return value."""

    content: list[FakeContentBlock] = field(default_factory=list)
    isError: bool = False


@dataclass
class FakeListToolsResult:
    """Mimics MCP list_tools return value."""

    tools: list[FakeMCPTool] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# MCPConfig tests
# ═══════════════════════════════════════════════════════════════

class TestMCPConfig:
    """Tests for MCPConfig and MCPServerConfig."""

    def test_default_config(self) -> None:
        cfg = MCPConfig()
        assert cfg.enabled is False
        assert cfg.servers == []

    def test_enabled_config(self) -> None:
        cfg = MCPConfig(enabled=True)
        assert cfg.enabled is True

    def test_server_config_minimal(self) -> None:
        srv = MCPServerConfig(name="test", command="echo")
        assert srv.name == "test"
        assert srv.command == "echo"
        assert srv.args == []
        assert srv.env == {}
        assert srv.description == ""

    def test_server_config_full(self) -> None:
        srv = MCPServerConfig(
            name="github",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_TOKEN": "ghp_test"},
            description="GitHub API",
        )
        assert srv.name == "github"
        assert srv.command == "npx"
        assert srv.args == ["-y", "@modelcontextprotocol/server-github"]
        assert srv.env == {"GITHUB_TOKEN": "ghp_test"}
        assert srv.description == "GitHub API"

    def test_config_with_servers(self) -> None:
        cfg = MCPConfig(
            enabled=True,
            servers=[
                MCPServerConfig(name="s1", command="cmd1"),
                MCPServerConfig(name="s2", command="cmd2", args=["--flag"]),
            ],
        )
        assert cfg.enabled is True
        assert len(cfg.servers) == 2
        assert cfg.servers[0].name == "s1"
        assert cfg.servers[1].name == "s2"
        assert cfg.servers[1].args == ["--flag"]


class TestMCPInSettings:
    """Tests for MCP config integration into Settings."""

    def test_settings_has_mcp_field(self) -> None:
        settings = Settings()
        assert hasattr(settings, "mcp")
        assert isinstance(settings.mcp, MCPConfig)

    def test_settings_default_mcp(self) -> None:
        settings = Settings()
        assert settings.mcp.enabled is False
        assert settings.mcp.servers == []

    def test_settings_with_mcp_data(self) -> None:
        settings = Settings(
            mcp={  # type: ignore[arg-type]
                "enabled": True,
                "servers": [
                    {"name": "fs", "command": "npx", "args": ["-y", "fs-server"]},
                ],
            }
        )
        assert settings.mcp.enabled is True
        assert len(settings.mcp.servers) == 1
        assert settings.mcp.servers[0].name == "fs"


# ═══════════════════════════════════════════════════════════════
# Type mapping tests
# ═══════════════════════════════════════════════════════════════

class TestMCPTypeMapping:
    """Tests for _mcp_type_to_tool_param_type."""

    @pytest.mark.parametrize(
        ("json_type", "expected"),
        [
            ("string", "string"),
            ("integer", "integer"),
            ("number", "number"),
            ("boolean", "boolean"),
            ("array", "array"),
            ("object", "object"),
        ],
    )
    def test_known_types(self, json_type: str, expected: str) -> None:
        assert _mcp_type_to_tool_param_type(json_type) == expected

    def test_unknown_type_defaults_to_string(self) -> None:
        assert _mcp_type_to_tool_param_type("unknown") == "string"
        assert _mcp_type_to_tool_param_type("") == "string"

    def test_null_type_defaults_to_string(self) -> None:
        assert _mcp_type_to_tool_param_type("null") == "string"


# ═══════════════════════════════════════════════════════════════
# _mcp_tool_to_tool_def tests
# ═══════════════════════════════════════════════════════════════

class TestMCPToolToToolDef:
    """Tests for converting MCP tools to vaig ToolDefs."""

    def test_basic_conversion(self) -> None:
        mcp_tool = FakeMCPTool(
            name="read_file",
            description="Read a file",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
        )
        call_fn = MagicMock()
        tool_def = _mcp_tool_to_tool_def(mcp_tool, call_fn, "fs")

        assert tool_def.name == "mcp_fs_read_file"
        assert tool_def.description == "[MCP:fs] Read a file"
        assert len(tool_def.parameters) == 1
        assert tool_def.parameters[0].name == "path"
        assert tool_def.parameters[0].type == "string"
        assert tool_def.parameters[0].required is True
        assert tool_def.execute is call_fn

    def test_multiple_params_with_optional(self) -> None:
        mcp_tool = FakeMCPTool(
            name="search",
            description="Search files",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results"},
                    "recursive": {"type": "boolean", "description": "Recurse"},
                },
                "required": ["query"],
            },
        )
        call_fn = MagicMock()
        tool_def = _mcp_tool_to_tool_def(mcp_tool, call_fn, "search-srv")

        assert tool_def.name == "mcp_search-srv_search"
        assert len(tool_def.parameters) == 3

        by_name = {p.name: p for p in tool_def.parameters}
        assert by_name["query"].required is True
        assert by_name["limit"].required is False
        assert by_name["limit"].type == "integer"
        assert by_name["recursive"].required is False
        assert by_name["recursive"].type == "boolean"

    def test_no_input_schema(self) -> None:
        mcp_tool = FakeMCPTool(name="ping", description="Ping", inputSchema=None)
        call_fn = MagicMock()
        tool_def = _mcp_tool_to_tool_def(mcp_tool, call_fn, "test")

        assert tool_def.name == "mcp_test_ping"
        assert tool_def.parameters == []

    def test_empty_input_schema(self) -> None:
        mcp_tool = FakeMCPTool(name="noop", description=None, inputSchema={})
        call_fn = MagicMock()
        tool_def = _mcp_tool_to_tool_def(mcp_tool, call_fn, "test")

        assert tool_def.name == "mcp_test_noop"
        assert tool_def.description == "[MCP:test] noop"
        assert tool_def.parameters == []

    def test_no_description_uses_name(self) -> None:
        mcp_tool = FakeMCPTool(name="my_tool", description=None)
        call_fn = MagicMock()
        tool_def = _mcp_tool_to_tool_def(mcp_tool, call_fn, "srv")

        assert tool_def.description == "[MCP:srv] my_tool"


# ═══════════════════════════════════════════════════════════════
# create_mcp_tool_defs tests
# ═══════════════════════════════════════════════════════════════

class TestCreateMCPToolDefs:
    """Tests for creating vaig ToolDefs from MCP tool lists."""

    def test_empty_list(self) -> None:
        defs = create_mcp_tool_defs("srv", "cmd", [])
        assert defs == []

    def test_single_tool(self) -> None:
        mcp_tool = FakeMCPTool(
            name="greet",
            description="Say hello",
            inputSchema={
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Name"}},
                "required": ["name"],
            },
        )
        defs = create_mcp_tool_defs("my-server", "npx", [mcp_tool])
        assert len(defs) == 1
        assert defs[0].name == "mcp_my-server_greet"
        assert callable(defs[0].execute)

    def test_multiple_tools(self) -> None:
        tools = [
            FakeMCPTool(name="tool_a", description="A"),
            FakeMCPTool(name="tool_b", description="B"),
            FakeMCPTool(name="tool_c", description="C"),
        ]
        defs = create_mcp_tool_defs("multi", "cmd", tools)
        assert len(defs) == 3
        names = [d.name for d in defs]
        assert names == ["mcp_multi_tool_a", "mcp_multi_tool_b", "mcp_multi_tool_c"]

    def test_closure_binds_correct_tool_name(self) -> None:
        """Verify each closure binds its own tool name (not last loop value)."""
        tools = [
            FakeMCPTool(name="first", description="First"),
            FakeMCPTool(name="second", description="Second"),
        ]
        defs = create_mcp_tool_defs("srv", "cmd", tools, args=["--arg"])

        # The closures should be different objects with distinct bound names
        assert defs[0].execute is not defs[1].execute
        assert defs[0].name == "mcp_srv_first"
        assert defs[1].name == "mcp_srv_second"

    def test_tool_defs_with_args_and_env(self) -> None:
        tool = FakeMCPTool(name="t", description="Test")
        defs = create_mcp_tool_defs(
            "srv", "node", [tool],
            args=["server.js"],
            env={"API_KEY": "test"},
        )
        assert len(defs) == 1
        assert defs[0].name == "mcp_srv_t"


# ═══════════════════════════════════════════════════════════════
# is_mcp_available tests
# ═══════════════════════════════════════════════════════════════

class TestIsMCPAvailable:
    """Tests for is_mcp_available."""

    def test_returns_bool(self) -> None:
        result = is_mcp_available()
        assert isinstance(result, bool)

    def test_reflects_mcp_available_flag(self) -> None:
        """Test that is_mcp_available reflects the _MCP_AVAILABLE flag."""
        with patch("vaig.tools.mcp_bridge._MCP_AVAILABLE", True):
            assert is_mcp_available() is True
        with patch("vaig.tools.mcp_bridge._MCP_AVAILABLE", False):
            assert is_mcp_available() is False


# ═══════════════════════════════════════════════════════════════
# discover_tools async tests
# ═══════════════════════════════════════════════════════════════

class TestDiscoverTools:
    """Tests for the discover_tools async function."""

    def test_raises_when_mcp_unavailable(self) -> None:
        with patch("vaig.tools.mcp_bridge._MCP_AVAILABLE", False):
            from vaig.tools.mcp_bridge import discover_tools

            with pytest.raises(RuntimeError, match="MCP SDK not installed"):
                asyncio.run(discover_tools("echo"))

    def test_discover_calls_stdio_client(self) -> None:
        """Test that discover_tools properly chains MCP calls."""
        import vaig.tools.mcp_bridge as bridge_mod

        fake_tool = FakeMCPTool(name="test_tool", description="A test")
        fake_result = FakeListToolsResult(tools=[fake_tool])

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=fake_result)

        # Create a proper async context manager chain
        session_cm = AsyncMock()
        session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        session_cm.__aexit__ = AsyncMock(return_value=False)

        stdio_cm = AsyncMock()
        stdio_cm.__aenter__ = AsyncMock(return_value=("read", "write"))
        stdio_cm.__aexit__ = AsyncMock(return_value=False)

        mock_stdio_client = MagicMock(return_value=stdio_cm)
        mock_client_session = MagicMock(return_value=session_cm)

        # Patch at module attribute level (works even if original imports failed)
        original_available = bridge_mod._MCP_AVAILABLE
        original_stdio = getattr(bridge_mod, "stdio_client", None)
        original_session = getattr(bridge_mod, "ClientSession", None)
        original_params = getattr(bridge_mod, "StdioServerParameters", None)

        try:
            bridge_mod._MCP_AVAILABLE = True
            bridge_mod.stdio_client = mock_stdio_client  # type: ignore[attr-defined]
            bridge_mod.ClientSession = mock_client_session  # type: ignore[attr-defined]
            bridge_mod.StdioServerParameters = lambda **kwargs: MagicMock(**kwargs)  # type: ignore[attr-defined,assignment]

            tools = asyncio.run(bridge_mod.discover_tools("test-cmd", args=["--flag"]))
        finally:
            bridge_mod._MCP_AVAILABLE = original_available
            if original_stdio is not None:
                bridge_mod.stdio_client = original_stdio  # type: ignore[attr-defined]
            if original_session is not None:
                bridge_mod.ClientSession = original_session  # type: ignore[attr-defined]
            if original_params is not None:
                bridge_mod.StdioServerParameters = original_params  # type: ignore[attr-defined]

        assert len(tools) == 1
        assert tools[0].name == "test_tool"


# ═══════════════════════════════════════════════════════════════
# call_mcp_tool async tests
# ═══════════════════════════════════════════════════════════════

def _run_call_mcp_tool_with_mocks(
    fake_result: FakeCallToolResult | None = None,
    *,
    raise_error: Exception | None = None,
) -> ToolResult:
    """Helper to run call_mcp_tool with mocked MCP internals."""
    import vaig.tools.mcp_bridge as bridge_mod

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    if fake_result is not None:
        mock_session.call_tool = AsyncMock(return_value=fake_result)

    session_cm = AsyncMock()
    session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    session_cm.__aexit__ = AsyncMock(return_value=False)

    stdio_cm = AsyncMock()
    if raise_error:
        stdio_cm.__aenter__ = AsyncMock(side_effect=raise_error)
    else:
        stdio_cm.__aenter__ = AsyncMock(return_value=("read", "write"))
    stdio_cm.__aexit__ = AsyncMock(return_value=False)

    mock_stdio_client = MagicMock(return_value=stdio_cm)
    mock_client_session = MagicMock(return_value=session_cm)

    original_available = bridge_mod._MCP_AVAILABLE
    original_stdio = getattr(bridge_mod, "stdio_client", None)
    original_session = getattr(bridge_mod, "ClientSession", None)
    original_params = getattr(bridge_mod, "StdioServerParameters", None)

    try:
        bridge_mod._MCP_AVAILABLE = True
        bridge_mod.stdio_client = mock_stdio_client  # type: ignore[attr-defined]
        bridge_mod.ClientSession = mock_client_session  # type: ignore[attr-defined]
        # StdioServerParameters needs to be callable
        bridge_mod.StdioServerParameters = lambda **kwargs: MagicMock(**kwargs)  # type: ignore[attr-defined,assignment]

        return asyncio.run(bridge_mod.call_mcp_tool("cmd", "tool", {"key": "val"}))
    finally:
        bridge_mod._MCP_AVAILABLE = original_available
        if original_stdio is not None:
            bridge_mod.stdio_client = original_stdio  # type: ignore[attr-defined]
        if original_session is not None:
            bridge_mod.ClientSession = original_session  # type: ignore[attr-defined]
        if original_params is not None:
            bridge_mod.StdioServerParameters = original_params  # type: ignore[attr-defined]


class TestCallMCPTool:
    """Tests for the call_mcp_tool async function."""

    def test_returns_error_when_mcp_unavailable(self) -> None:
        with patch("vaig.tools.mcp_bridge._MCP_AVAILABLE", False):
            from vaig.tools.mcp_bridge import call_mcp_tool

            result = asyncio.run(call_mcp_tool("echo", "test", {}))
            assert result.error is True
            assert "MCP SDK not installed" in result.output

    def test_successful_call(self) -> None:
        result = _run_call_mcp_tool_with_mocks(
            FakeCallToolResult(
                content=[FakeContentBlock(text="hello world")],
                isError=False,
            )
        )
        assert result.error is False
        assert result.output == "hello world"

    def test_error_result(self) -> None:
        result = _run_call_mcp_tool_with_mocks(
            FakeCallToolResult(
                content=[FakeContentBlock(text="permission denied")],
                isError=True,
            )
        )
        assert result.error is True
        assert "permission denied" in result.output

    def test_multiple_content_blocks(self) -> None:
        result = _run_call_mcp_tool_with_mocks(
            FakeCallToolResult(
                content=[
                    FakeContentBlock(text="line 1"),
                    FakeContentBlock(text="line 2"),
                ],
                isError=False,
            )
        )
        assert result.output == "line 1\nline 2"

    def test_empty_content(self) -> None:
        result = _run_call_mcp_tool_with_mocks(
            FakeCallToolResult(content=[], isError=False)
        )
        assert result.output == "(no output)"
        assert result.error is False

    def test_exception_returns_error_result(self) -> None:
        result = _run_call_mcp_tool_with_mocks(
            raise_error=ConnectionError("refused"),
        )
        assert result.error is True
        assert "MCP tool call failed" in result.output


# ═══════════════════════════════════════════════════════════════
# CLI tests
# ═══════════════════════════════════════════════════════════════

runner = CliRunner()


def _make_settings_with_mcp(
    enabled: bool = True,
    servers: list[dict[str, Any]] | None = None,
) -> Settings:
    """Build a Settings with MCP config for testing."""
    mcp_data: dict[str, Any] = {"enabled": enabled}
    if servers is not None:
        mcp_data["servers"] = servers
    return Settings(mcp=mcp_data)  # type: ignore[arg-type]


def _patch_mcp_available(available: bool = True):
    """Patch is_mcp_available for CLI tests.

    The CLI imports is_mcp_available from mcp_bridge, so we patch the flag.
    """
    return patch("vaig.tools.mcp_bridge._MCP_AVAILABLE", available)


class TestMCPListServers:
    """Tests for 'vaig mcp list-servers'."""

    def test_mcp_disabled(self) -> None:
        from vaig.cli.app import app

        settings = _make_settings_with_mcp(enabled=False)
        with (
            _patch_mcp_available(True),
            patch("vaig.cli.app._get_settings", return_value=settings),
        ):
            result = runner.invoke(app, ["mcp", "list-servers"])

        assert result.exit_code == 0
        assert "disabled" in result.output.lower()

    def test_no_servers_configured(self) -> None:
        from vaig.cli.app import app

        settings = _make_settings_with_mcp(enabled=True, servers=[])
        with (
            _patch_mcp_available(True),
            patch("vaig.cli.app._get_settings", return_value=settings),
        ):
            result = runner.invoke(app, ["mcp", "list-servers"])

        assert result.exit_code == 0
        assert "no mcp servers" in result.output.lower()

    def test_lists_configured_servers(self) -> None:
        from vaig.cli.app import app

        settings = _make_settings_with_mcp(
            enabled=True,
            servers=[
                {"name": "fs", "command": "npx", "args": ["-y", "fs-server"], "description": "Filesystem"},
                {"name": "github", "command": "node", "args": ["gh.js"]},
            ],
        )
        with (
            _patch_mcp_available(True),
            patch("vaig.cli.app._get_settings", return_value=settings),
        ):
            result = runner.invoke(app, ["mcp", "list-servers"])

        assert result.exit_code == 0
        assert "fs" in result.output
        assert "github" in result.output
        assert "Filesystem" in result.output

    def test_sdk_not_installed(self) -> None:
        from vaig.cli.app import app

        with (
            _patch_mcp_available(False),
            patch("vaig.cli.app._get_settings", return_value=Settings()),
        ):
            result = runner.invoke(app, ["mcp", "list-servers"])

        assert result.exit_code == 1
        assert "not installed" in result.output.lower()


class TestMCPDiscover:
    """Tests for 'vaig mcp discover'."""

    def test_server_not_found(self) -> None:
        from vaig.cli.app import app

        settings = _make_settings_with_mcp(enabled=True, servers=[])
        with (
            _patch_mcp_available(True),
            patch("vaig.cli.app._get_settings", return_value=settings),
        ):
            result = runner.invoke(app, ["mcp", "discover", "nonexistent"])

        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_mcp_disabled(self) -> None:
        from vaig.cli.app import app

        settings = _make_settings_with_mcp(enabled=False)
        with (
            _patch_mcp_available(True),
            patch("vaig.cli.app._get_settings", return_value=settings),
        ):
            result = runner.invoke(app, ["mcp", "discover", "any"])

        assert result.exit_code == 1
        assert "disabled" in result.output.lower()

    def test_successful_discovery(self) -> None:
        from vaig.cli.app import app

        settings = _make_settings_with_mcp(
            enabled=True,
            servers=[{"name": "test", "command": "echo"}],
        )

        fake_tools = [
            FakeMCPTool(
                name="read_file",
                description="Read a file",
                inputSchema={
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "File path"}},
                    "required": ["path"],
                },
            ),
        ]

        with (
            _patch_mcp_available(True),
            patch("vaig.cli.app._get_settings", return_value=settings),
            patch("asyncio.run", return_value=fake_tools),
        ):
            result = runner.invoke(app, ["mcp", "discover", "test"])

        assert result.exit_code == 0
        assert "read_file" in result.output
        assert "1 tools discovered" in result.output

    def test_no_tools_found(self) -> None:
        from vaig.cli.app import app

        settings = _make_settings_with_mcp(
            enabled=True,
            servers=[{"name": "empty", "command": "echo"}],
        )

        with (
            _patch_mcp_available(True),
            patch("vaig.cli.app._get_settings", return_value=settings),
            patch("asyncio.run", return_value=[]),
        ):
            result = runner.invoke(app, ["mcp", "discover", "empty"])

        assert result.exit_code == 0
        assert "no tools" in result.output.lower()

    def test_connection_failure(self) -> None:
        from vaig.cli.app import app

        settings = _make_settings_with_mcp(
            enabled=True,
            servers=[{"name": "broken", "command": "nonexistent"}],
        )

        with (
            _patch_mcp_available(True),
            patch("vaig.cli.app._get_settings", return_value=settings),
            patch("asyncio.run", side_effect=ConnectionError("refused")),
        ):
            result = runner.invoke(app, ["mcp", "discover", "broken"])

        assert result.exit_code == 1
        assert "failed to connect" in result.output.lower()


class TestMCPCall:
    """Tests for 'vaig mcp call'."""

    def test_server_not_found(self) -> None:
        from vaig.cli.app import app

        settings = _make_settings_with_mcp(enabled=True, servers=[])
        with (
            _patch_mcp_available(True),
            patch("vaig.cli.app._get_settings", return_value=settings),
        ):
            result = runner.invoke(app, ["mcp", "call", "missing", "tool"])

        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_mcp_disabled(self) -> None:
        from vaig.cli.app import app

        settings = _make_settings_with_mcp(enabled=False)
        with (
            _patch_mcp_available(True),
            patch("vaig.cli.app._get_settings", return_value=settings),
        ):
            result = runner.invoke(app, ["mcp", "call", "srv", "tool"])

        assert result.exit_code == 1
        assert "disabled" in result.output.lower()

    def test_invalid_json_args(self) -> None:
        from vaig.cli.app import app

        settings = _make_settings_with_mcp(
            enabled=True,
            servers=[{"name": "srv", "command": "echo"}],
        )
        with (
            _patch_mcp_available(True),
            patch("vaig.cli.app._get_settings", return_value=settings),
        ):
            result = runner.invoke(app, ["mcp", "call", "srv", "tool", "{bad json"])

        assert result.exit_code == 1
        assert "invalid json" in result.output.lower()

    def test_successful_call(self) -> None:
        from vaig.cli.app import app

        settings = _make_settings_with_mcp(
            enabled=True,
            servers=[{"name": "srv", "command": "echo"}],
        )

        with (
            _patch_mcp_available(True),
            patch("vaig.cli.app._get_settings", return_value=settings),
            patch("asyncio.run", return_value=ToolResult(output="result text")),
        ):
            result = runner.invoke(app, ["mcp", "call", "srv", "my-tool", '{"key": "val"}'])

        assert result.exit_code == 0
        assert "result text" in result.output

    def test_call_without_args(self) -> None:
        from vaig.cli.app import app

        settings = _make_settings_with_mcp(
            enabled=True,
            servers=[{"name": "srv", "command": "echo"}],
        )

        with (
            _patch_mcp_available(True),
            patch("vaig.cli.app._get_settings", return_value=settings),
            patch("asyncio.run", return_value=ToolResult(output="pong")),
        ):
            result = runner.invoke(app, ["mcp", "call", "srv", "ping"])

        assert result.exit_code == 0
        assert "pong" in result.output

    def test_tool_error(self) -> None:
        from vaig.cli.app import app

        settings = _make_settings_with_mcp(
            enabled=True,
            servers=[{"name": "srv", "command": "echo"}],
        )

        with (
            _patch_mcp_available(True),
            patch("vaig.cli.app._get_settings", return_value=settings),
            patch("asyncio.run", return_value=ToolResult(output="access denied", error=True)),
        ):
            result = runner.invoke(app, ["mcp", "call", "srv", "restricted"])

        assert result.exit_code == 1
        assert "access denied" in result.output

    def test_sdk_not_installed(self) -> None:
        from vaig.cli.app import app

        with (
            _patch_mcp_available(False),
            patch("vaig.cli.app._get_settings", return_value=Settings()),
        ):
            result = runner.invoke(app, ["mcp", "call", "srv", "tool"])

        assert result.exit_code == 1
        assert "not installed" in result.output.lower()


# ═══════════════════════════════════════════════════════════════
# MCPServerInfo tests
# ═══════════════════════════════════════════════════════════════

class TestMCPServerInfo:
    """Tests for the MCPServerInfo dataclass."""

    def test_basic_creation(self) -> None:
        from vaig.tools.mcp_bridge import MCPServerInfo

        info = MCPServerInfo(name="test", command="echo")
        assert info.name == "test"
        assert info.command == "echo"
        assert info.args == []
        assert info.env is None
        assert info.tools == []

    def test_with_args_and_env(self) -> None:
        from vaig.tools.mcp_bridge import MCPServerInfo

        info = MCPServerInfo(
            name="full",
            command="node",
            args=["server.js"],
            env={"KEY": "val"},
        )
        assert info.args == ["server.js"]
        assert info.env == {"KEY": "val"}


# ═══════════════════════════════════════════════════════════════
# YAML config loading tests
# ═══════════════════════════════════════════════════════════════

class TestMCPYAMLConfig:
    """Tests that MCP config loads correctly from YAML data."""

    def test_load_from_dict(self) -> None:
        """Simulate what Settings.load() does with YAML data."""
        yaml_data = {
            "mcp": {
                "enabled": True,
                "servers": [
                    {
                        "name": "filesystem",
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                        "description": "FS access",
                    },
                    {
                        "name": "github",
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-github"],
                        "env": {"GITHUB_TOKEN": "ghp_test123"},
                        "description": "GitHub API",
                    },
                ],
            },
        }
        settings = Settings(**yaml_data)

        assert settings.mcp.enabled is True
        assert len(settings.mcp.servers) == 2
        assert settings.mcp.servers[0].name == "filesystem"
        assert settings.mcp.servers[0].command == "npx"
        assert settings.mcp.servers[1].env == {"GITHUB_TOKEN": "ghp_test123"}

    def test_missing_mcp_section_uses_defaults(self) -> None:
        """YAML without mcp section should use defaults."""
        settings = Settings()
        assert settings.mcp.enabled is False
        assert settings.mcp.servers == []


# ═══════════════════════════════════════════════════════════════
# create_mcp_tools factory tests
# ═══════════════════════════════════════════════════════════════

class TestCreateMCPTools:
    """Tests for the create_mcp_tools factory function."""

    def test_returns_empty_when_disabled(self) -> None:
        """auto_register=False should return no tools."""
        cfg = MCPConfig(enabled=True, auto_register=False)
        result = create_mcp_tools(cfg)
        assert result == []

    def test_returns_empty_when_mcp_not_enabled(self) -> None:
        """enabled=False should return no tools even with auto_register=True."""
        cfg = MCPConfig(enabled=False, auto_register=True)
        result = create_mcp_tools(cfg)
        assert result == []

    def test_returns_empty_when_both_disabled(self) -> None:
        cfg = MCPConfig(enabled=False, auto_register=False)
        result = create_mcp_tools(cfg)
        assert result == []

    def test_returns_empty_when_mcp_sdk_unavailable(self) -> None:
        """When MCP SDK is not installed, should log warning and return []."""
        cfg = MCPConfig(
            enabled=True,
            auto_register=True,
            servers=[MCPServerConfig(name="test", command="echo")],
        )
        with patch("vaig.tools.mcp_bridge._MCP_AVAILABLE", False):
            result = create_mcp_tools(cfg)
        assert result == []

    def test_returns_empty_when_no_servers(self) -> None:
        """auto_register=True but no servers configured."""
        cfg = MCPConfig(enabled=True, auto_register=True, servers=[])
        # Need MCP available for this path
        with patch("vaig.tools.mcp_bridge._MCP_AVAILABLE", True):
            result = create_mcp_tools(cfg)
        assert result == []

    def test_discovers_and_creates_tools_from_single_server(self) -> None:
        """Happy path: one server with two tools."""
        cfg = MCPConfig(
            enabled=True,
            auto_register=True,
            servers=[
                MCPServerConfig(name="fs", command="npx", args=["-y", "fs-server"]),
            ],
        )
        fake_tools = [
            FakeMCPTool(
                name="read_file",
                description="Read a file",
                inputSchema={
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "File path"}},
                    "required": ["path"],
                },
            ),
            FakeMCPTool(name="list_dir", description="List directory"),
        ]

        with patch("vaig.tools.mcp_bridge._MCP_AVAILABLE", True), \
             patch("vaig.tools.mcp_bridge.asyncio") as mock_asyncio:
            mock_asyncio.run.return_value = fake_tools
            result = create_mcp_tools(cfg)

        assert len(result) == 2
        assert result[0].name == "mcp_fs_read_file"
        assert result[1].name == "mcp_fs_list_dir"
        assert result[0].description == "[MCP:fs] Read a file"

    def test_discovers_tools_from_multiple_servers(self) -> None:
        """Multiple servers each return tools."""
        cfg = MCPConfig(
            enabled=True,
            auto_register=True,
            servers=[
                MCPServerConfig(name="fs", command="npx"),
                MCPServerConfig(name="github", command="node"),
            ],
        )
        fs_tools = [FakeMCPTool(name="read", description="Read")]
        gh_tools = [FakeMCPTool(name="list_repos", description="List repos")]

        with patch("vaig.tools.mcp_bridge._MCP_AVAILABLE", True), \
             patch("vaig.tools.mcp_bridge.asyncio") as mock_asyncio:
            mock_asyncio.run.side_effect = [fs_tools, gh_tools]
            result = create_mcp_tools(cfg)

        assert len(result) == 2
        names = [t.name for t in result]
        assert "mcp_fs_read" in names
        assert "mcp_github_list_repos" in names

    def test_skips_unreachable_server(self) -> None:
        """One server fails, the other succeeds — only working server's tools returned."""
        cfg = MCPConfig(
            enabled=True,
            auto_register=True,
            servers=[
                MCPServerConfig(name="broken", command="nonexistent"),
                MCPServerConfig(name="working", command="echo"),
            ],
        )
        good_tools = [FakeMCPTool(name="ping", description="Ping")]

        with patch("vaig.tools.mcp_bridge._MCP_AVAILABLE", True), \
             patch("vaig.tools.mcp_bridge.asyncio") as mock_asyncio:
            mock_asyncio.run.side_effect = [ConnectionError("refused"), good_tools]
            result = create_mcp_tools(cfg)

        assert len(result) == 1
        assert result[0].name == "mcp_working_ping"

    def test_skips_server_returning_no_tools(self) -> None:
        """Server returns empty tool list — no tools added for that server."""
        cfg = MCPConfig(
            enabled=True,
            auto_register=True,
            servers=[
                MCPServerConfig(name="empty", command="echo"),
            ],
        )

        with patch("vaig.tools.mcp_bridge._MCP_AVAILABLE", True), \
             patch("vaig.tools.mcp_bridge.asyncio") as mock_asyncio:
            mock_asyncio.run.return_value = []
            result = create_mcp_tools(cfg)

        assert result == []

    def test_tool_execute_closure_calls_mcp_tool(self) -> None:
        """Each ToolDef's execute should be a callable."""
        cfg = MCPConfig(
            enabled=True,
            auto_register=True,
            servers=[
                MCPServerConfig(name="srv", command="cmd", args=["--flag"]),
            ],
        )
        fake_tools = [
            FakeMCPTool(
                name="greet",
                description="Greet",
                inputSchema={
                    "type": "object",
                    "properties": {"name": {"type": "string", "description": "Name"}},
                    "required": ["name"],
                },
            ),
        ]

        with patch("vaig.tools.mcp_bridge._MCP_AVAILABLE", True), \
             patch("vaig.tools.mcp_bridge.asyncio") as mock_asyncio:
            mock_asyncio.run.return_value = fake_tools
            result = create_mcp_tools(cfg)

        assert len(result) == 1
        assert callable(result[0].execute)
        assert result[0].parameters[0].name == "name"
        assert result[0].parameters[0].required is True

    def test_passes_server_env_to_discover(self) -> None:
        """Server env dict should be passed through to discover_tools."""
        cfg = MCPConfig(
            enabled=True,
            auto_register=True,
            servers=[
                MCPServerConfig(
                    name="auth-srv",
                    command="node",
                    args=["server.js"],
                    env={"API_KEY": "secret123"},
                ),
            ],
        )

        with patch("vaig.tools.mcp_bridge._MCP_AVAILABLE", True), \
             patch("vaig.tools.mcp_bridge.asyncio") as mock_asyncio:
            mock_asyncio.run.return_value = [FakeMCPTool(name="test", description="Test")]
            create_mcp_tools(cfg)

            # Verify asyncio.run was called with the discover_tools coroutine
            mock_asyncio.run.assert_called_once()

"""Tests for plugin_loader — Python module plugin discovery and unified loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vaig.core.config import MCPConfig, PluginConfig, Settings
from vaig.tools.base import ToolDef, ToolParam, ToolResult
from vaig.tools.plugin_loader import _load_python_plugins, load_all_plugin_tools


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════


def _make_tool(name: str, description: str = "test tool") -> ToolDef:
    """Create a simple ToolDef for testing."""
    return ToolDef(
        name=name,
        description=description,
        parameters=[ToolParam(name="arg", type="string", description="an arg")],
        execute=lambda **_kw: ToolResult(output="ok"),
    )


# ═══════════════════════════════════════════════════════════════
# _load_python_plugins
# ═══════════════════════════════════════════════════════════════


class TestLoadPythonPlugins:
    """Tests for _load_python_plugins()."""

    def test_returns_empty_when_disabled(self) -> None:
        """Should return [] immediately when plugins are disabled."""
        config = PluginConfig(enabled=False, directories=["./plugins"])
        assert _load_python_plugins(config) == []

    def test_returns_empty_when_no_directories(self) -> None:
        """Should return [] when no directories configured."""
        config = PluginConfig(enabled=True, directories=[])
        assert _load_python_plugins(config) == []

    def test_handles_nonexistent_directory(self) -> None:
        """Should log warning and return [] for a non-existent directory."""
        config = PluginConfig(enabled=True, directories=["/nonexistent/path/xyz"])
        result = _load_python_plugins(config)
        assert result == []

    def test_loads_valid_plugin(self, tmp_path: Path) -> None:
        """Should load a plugin module with register_tools() and return its ToolDefs."""
        plugin_file = tmp_path / "my_plugin.py"
        plugin_file.write_text(
            "from vaig.tools.base import ToolDef, ToolParam, ToolResult\n"
            "\n"
            "def register_tools():\n"
            "    return [\n"
            "        ToolDef(\n"
            "            name='my_custom_tool',\n"
            "            description='A custom tool',\n"
            "            parameters=[ToolParam(name='x', type='string', description='input')],\n"
            "            execute=lambda **kw: ToolResult(output='hello'),\n"
            "        )\n"
            "    ]\n"
        )

        config = PluginConfig(enabled=True, directories=[str(tmp_path)])
        result = _load_python_plugins(config)

        assert len(result) == 1
        assert result[0].name == "my_custom_tool"
        assert result[0].description == "A custom tool"

    def test_skips_module_without_register_tools(self, tmp_path: Path) -> None:
        """Should skip modules that don't have a register_tools function."""
        plugin_file = tmp_path / "no_register.py"
        plugin_file.write_text("x = 42\n")

        config = PluginConfig(enabled=True, directories=[str(tmp_path)])
        result = _load_python_plugins(config)
        assert result == []

    def test_skips_module_that_raises_on_load(self, tmp_path: Path) -> None:
        """Should skip modules that raise errors during import."""
        plugin_file = tmp_path / "broken_import.py"
        plugin_file.write_text("raise RuntimeError('boom')\n")

        config = PluginConfig(enabled=True, directories=[str(tmp_path)])
        result = _load_python_plugins(config)
        assert result == []

    def test_skips_module_where_register_tools_raises(self, tmp_path: Path) -> None:
        """Should skip modules where register_tools() raises an error."""
        plugin_file = tmp_path / "broken_register.py"
        plugin_file.write_text(
            "def register_tools():\n"
            "    raise ValueError('something went wrong')\n"
        )

        config = PluginConfig(enabled=True, directories=[str(tmp_path)])
        result = _load_python_plugins(config)
        assert result == []

    def test_skips_module_where_register_tools_returns_non_list(self, tmp_path: Path) -> None:
        """Should skip modules where register_tools() returns a non-list."""
        plugin_file = tmp_path / "bad_return.py"
        plugin_file.write_text(
            "def register_tools():\n"
            "    return 'not a list'\n"
        )

        config = PluginConfig(enabled=True, directories=[str(tmp_path)])
        result = _load_python_plugins(config)
        assert result == []

    def test_loads_multiple_plugins_from_directory(self, tmp_path: Path) -> None:
        """Should load tools from multiple plugin files in one directory."""
        for i in range(3):
            plugin_file = tmp_path / f"plugin_{i}.py"
            plugin_file.write_text(
                "from vaig.tools.base import ToolDef, ToolResult\n"
                "\n"
                f"def register_tools():\n"
                f"    return [ToolDef(name='tool_{i}', description='tool {i}')]\n"
            )

        config = PluginConfig(enabled=True, directories=[str(tmp_path)])
        result = _load_python_plugins(config)

        assert len(result) == 3
        names = {t.name for t in result}
        assert names == {"tool_0", "tool_1", "tool_2"}

    def test_loads_from_multiple_directories(self, tmp_path: Path) -> None:
        """Should aggregate tools from multiple directories."""
        dir_a = tmp_path / "plugins_a"
        dir_b = tmp_path / "plugins_b"
        dir_a.mkdir()
        dir_b.mkdir()

        (dir_a / "a_plugin.py").write_text(
            "from vaig.tools.base import ToolDef\n"
            "def register_tools():\n"
            "    return [ToolDef(name='from_a', description='tool a')]\n"
        )
        (dir_b / "b_plugin.py").write_text(
            "from vaig.tools.base import ToolDef\n"
            "def register_tools():\n"
            "    return [ToolDef(name='from_b', description='tool b')]\n"
        )

        config = PluginConfig(enabled=True, directories=[str(dir_a), str(dir_b)])
        result = _load_python_plugins(config)

        assert len(result) == 2
        names = {t.name for t in result}
        assert names == {"from_a", "from_b"}

    def test_empty_directory_returns_empty(self, tmp_path: Path) -> None:
        """Should return [] when directory exists but has no .py files."""
        config = PluginConfig(enabled=True, directories=[str(tmp_path)])
        result = _load_python_plugins(config)
        assert result == []


# ═══════════════════════════════════════════════════════════════
# load_all_plugin_tools
# ═══════════════════════════════════════════════════════════════


class TestLoadAllPluginTools:
    """Tests for load_all_plugin_tools()."""

    def test_returns_empty_when_both_disabled(self) -> None:
        """Should return [] when both MCP and plugins are disabled."""
        settings = Settings()
        assert settings.mcp.auto_register is False
        assert settings.plugins.enabled is False
        result = load_all_plugin_tools(settings)
        assert result == []

    @patch("vaig.tools.mcp_bridge.create_mcp_tools")
    def test_calls_create_mcp_tools_when_auto_register(self, mock_mcp: MagicMock) -> None:
        """Should call create_mcp_tools when mcp.auto_register is True."""
        fake_tools = [_make_tool("mcp_tool_1"), _make_tool("mcp_tool_2")]
        mock_mcp.return_value = fake_tools

        settings = Settings()
        settings.mcp.auto_register = True
        settings.mcp.enabled = True

        result = load_all_plugin_tools(settings)

        mock_mcp.assert_called_once_with(settings.mcp)
        assert len(result) == 2
        assert result[0].name == "mcp_tool_1"

    @patch("vaig.tools.plugin_loader._load_python_plugins")
    def test_calls_load_python_plugins_when_enabled(self, mock_py: MagicMock) -> None:
        """Should call _load_python_plugins when plugins.enabled is True."""
        fake_tools = [_make_tool("py_tool_1")]
        mock_py.return_value = fake_tools

        settings = Settings()
        settings.plugins.enabled = True

        result = load_all_plugin_tools(settings)

        mock_py.assert_called_once_with(settings.plugins)
        assert len(result) == 1
        assert result[0].name == "py_tool_1"

    @patch("vaig.tools.plugin_loader._load_python_plugins")
    @patch("vaig.tools.mcp_bridge.create_mcp_tools")
    def test_aggregates_both_sources(self, mock_mcp: MagicMock, mock_py: MagicMock) -> None:
        """Should combine tools from both MCP and Python plugins."""
        mock_mcp.return_value = [_make_tool("mcp_t")]
        mock_py.return_value = [_make_tool("py_t")]

        settings = Settings()
        settings.mcp.auto_register = True
        settings.mcp.enabled = True
        settings.plugins.enabled = True

        result = load_all_plugin_tools(settings)

        assert len(result) == 2
        names = {t.name for t in result}
        assert names == {"mcp_t", "py_t"}

    @patch("vaig.tools.mcp_bridge.create_mcp_tools", side_effect=RuntimeError("mcp broke"))
    def test_handles_mcp_error_gracefully(self, mock_mcp: MagicMock) -> None:
        """Should catch and log MCP errors, returning what Python plugins provide."""
        settings = Settings()
        settings.mcp.auto_register = True
        settings.mcp.enabled = True

        # Should not raise
        result = load_all_plugin_tools(settings)
        assert result == []

    @patch("vaig.tools.plugin_loader._load_python_plugins", side_effect=RuntimeError("py broke"))
    def test_handles_python_plugin_error_gracefully(self, mock_py: MagicMock) -> None:
        """Should catch and log Python plugin errors."""
        settings = Settings()
        settings.plugins.enabled = True

        result = load_all_plugin_tools(settings)
        assert result == []

    def test_does_not_call_mcp_when_auto_register_false(self) -> None:
        """Should not attempt MCP loading when auto_register is False."""
        settings = Settings()
        settings.mcp.auto_register = False

        with patch("vaig.tools.mcp_bridge.create_mcp_tools") as mock_mcp:
            load_all_plugin_tools(settings)
            mock_mcp.assert_not_called()

    def test_does_not_call_python_plugins_when_disabled(self) -> None:
        """Should not attempt Python plugin loading when plugins.enabled is False."""
        settings = Settings()
        settings.plugins.enabled = False

        with patch("vaig.tools.plugin_loader._load_python_plugins") as mock_py:
            load_all_plugin_tools(settings)
            mock_py.assert_not_called()

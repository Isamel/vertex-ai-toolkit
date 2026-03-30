"""Tests for CodingAgent — autonomous file-editing agent using function calling."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vaig.agents.base import AgentResult, AgentRole
from vaig.agents.coding import (
    _DESTRUCTIVE_TOOLS,
    CODING_SYSTEM_PROMPT,
    CodingAgent,
    _default_confirm,
)
from vaig.core.cache import ToolResultCache
from vaig.core.client import ToolCallResult
from vaig.core.config import CodingConfig
from vaig.core.exceptions import MaxIterationsError
from vaig.tools import ToolDef, ToolParam, ToolRegistry, ToolResult

# ── Fixtures ─────────────────────────────────────────────────
# _reset_settings is provided by conftest.py (autouse)


def _make_mock_client(current_model: str = "gemini-2.5-pro") -> MagicMock:
    """Create a MagicMock that behaves like GeminiClient."""
    client = MagicMock()
    client.current_model = current_model
    return client


def _make_coding_config(
    workspace_root: str = "/tmp/test-workspace",
    max_tool_iterations: int = 10,
    confirm_actions: bool = True,
    allowed_commands: list[str] | None = None,
) -> CodingConfig:
    return CodingConfig(
        workspace_root=workspace_root,
        max_tool_iterations=max_tool_iterations,
        confirm_actions=confirm_actions,
        allowed_commands=allowed_commands or [],
    )


def _make_text_result(
    text: str = "Done",
    model: str = "gemini-2.5-pro",
    usage: dict[str, int] | None = None,
    finish_reason: str = "STOP",
) -> ToolCallResult:
    """Create a ToolCallResult with text only (no function calls)."""
    return ToolCallResult(
        text=text,
        model=model,
        function_calls=[],
        usage=usage if usage is not None else {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        finish_reason=finish_reason,
    )


def _make_fc_result(
    function_calls: list[dict[str, Any]],
    model: str = "gemini-2.5-pro",
    usage: dict[str, int] | None = None,
) -> ToolCallResult:
    """Create a ToolCallResult with function calls (no text)."""
    return ToolCallResult(
        text="",
        model=model,
        function_calls=function_calls,
        usage=usage or {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
        finish_reason="STOP",
    )


# ===========================================================================
# TestDefaultConfirm
# ===========================================================================


class TestDefaultConfirm:
    """Tests for _default_confirm()."""

    def test_always_returns_true(self) -> None:
        assert _default_confirm("write_file", {"path": "test.py"}) is True
        assert _default_confirm("edit_file", {}) is True
        assert _default_confirm("run_command", {"cmd": "rm -rf /"}) is True


# ===========================================================================
# TestDestructiveTools
# ===========================================================================


class TestDestructiveTools:
    """Tests for _DESTRUCTIVE_TOOLS constant."""

    def test_contains_expected_tools(self) -> None:
        assert "write_file" in _DESTRUCTIVE_TOOLS
        assert "edit_file" in _DESTRUCTIVE_TOOLS
        assert "run_command" in _DESTRUCTIVE_TOOLS

    def test_does_not_contain_read_tools(self) -> None:
        assert "read_file" not in _DESTRUCTIVE_TOOLS
        assert "list_files" not in _DESTRUCTIVE_TOOLS
        assert "search_files" not in _DESTRUCTIVE_TOOLS

    def test_is_frozenset(self) -> None:
        assert isinstance(_DESTRUCTIVE_TOOLS, frozenset)


# ===========================================================================
# TestCodingAgentInit
# ===========================================================================


class TestCodingAgentInit:
    """Tests for CodingAgent.__init__()."""

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_basic_init(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        client = _make_mock_client()
        config = _make_coding_config()

        agent = CodingAgent(client, config)

        assert agent.name == "coding-agent"
        assert agent.role == AgentRole.CODER
        assert agent.model == "gemini-2.5-pro"
        assert agent.workspace == Path("/tmp/test-workspace").resolve()

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_temperature_is_low(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """CodingAgent uses low temperature (0.2) for precise code generation."""
        client = _make_mock_client()
        config = _make_coding_config()

        agent = CodingAgent(client, config)

        assert agent.config.temperature == 0.2

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_max_output_tokens(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        client = _make_mock_client()
        config = _make_coding_config()

        agent = CodingAgent(client, config)

        assert agent.config.max_output_tokens == 65536

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_system_instruction_is_coding_prompt(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        client = _make_mock_client()
        config = _make_coding_config()

        agent = CodingAgent(client, config)

        assert agent.config.system_instruction == CODING_SYSTEM_PROMPT

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_model_override(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        client = _make_mock_client()
        config = _make_coding_config()

        agent = CodingAgent(client, config, model_id="gemini-2.5-flash")

        assert agent.model == "gemini-2.5-flash"

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_confirm_fn_used_when_confirm_enabled(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """When confirm_actions=True and confirm_fn provided, use it."""
        client = _make_mock_client()
        config = _make_coding_config(confirm_actions=True)
        custom_confirm = MagicMock(return_value=True)

        agent = CodingAgent(client, config, confirm_fn=custom_confirm)

        assert agent._confirm_fn is custom_confirm

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_confirm_disabled_uses_default(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """When confirm_actions=False, default confirm (always True) is used."""
        client = _make_mock_client()
        config = _make_coding_config(confirm_actions=False)
        custom_confirm = MagicMock(return_value=False)

        agent = CodingAgent(client, config, confirm_fn=custom_confirm)

        assert agent._confirm_fn is _default_confirm

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_no_confirm_fn_uses_default(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """When no confirm_fn provided, default is used."""
        client = _make_mock_client()
        config = _make_coding_config(confirm_actions=True)

        agent = CodingAgent(client, config)

        assert agent._confirm_fn is _default_confirm

    @patch("vaig.agents.coding.create_shell_tools")
    @patch("vaig.agents.coding.create_file_tools")
    def test_tools_registered(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """File and shell tools are registered in the registry."""
        file_tool = ToolDef(name="read_file", description="Read a file")
        shell_tool = ToolDef(name="run_command", description="Run a command")
        mock_file_tools.return_value = [file_tool]
        mock_shell_tools.return_value = [shell_tool]

        client = _make_mock_client()
        config = _make_coding_config()

        agent = CodingAgent(client, config)

        assert agent.registry.get("read_file") is file_tool
        assert agent.registry.get("run_command") is shell_tool
        assert len(agent.registry.list_tools()) == 2

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools")
    def test_file_tools_created_with_workspace(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """create_file_tools is called with the resolved workspace path."""
        mock_file_tools.return_value = []
        client = _make_mock_client()
        config = _make_coding_config(workspace_root="/tmp/ws")

        CodingAgent(client, config)

        mock_file_tools.assert_called_once_with(Path("/tmp/ws").resolve())

    @patch("vaig.agents.coding.create_shell_tools")
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_shell_tools_created_with_allowed_commands(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """create_shell_tools is called with workspace and allowed_commands."""
        mock_shell_tools.return_value = []
        client = _make_mock_client()
        config = _make_coding_config(allowed_commands=["pytest", "ruff"])

        CodingAgent(client, config)

        mock_shell_tools.assert_called_once()
        call_args = mock_shell_tools.call_args
        assert call_args[0][0] == Path("/tmp/test-workspace").resolve()
        assert call_args[1]["allowed_commands"] == ["pytest", "ruff"]
        # denied_commands should be the non-empty defaults from CodingConfig
        assert call_args[1]["denied_commands"] is not None
        assert len(call_args[1]["denied_commands"]) > 0

    @patch("vaig.agents.coding.create_shell_tools")
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_shell_tools_none_when_empty_allowed_commands(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """Empty allowed_commands list passes None to create_shell_tools."""
        mock_shell_tools.return_value = []
        client = _make_mock_client()
        config = _make_coding_config(allowed_commands=[])

        CodingAgent(client, config)

        mock_shell_tools.assert_called_once()
        call_args = mock_shell_tools.call_args
        assert call_args[0][0] == Path("/tmp/test-workspace").resolve()
        assert call_args[1]["allowed_commands"] is None
        # denied_commands should still be the defaults
        assert call_args[1]["denied_commands"] is not None


# ===========================================================================
# TestCodingAgentExecute
# ===========================================================================


class TestCodingAgentExecute:
    """Tests for CodingAgent.execute() — the tool-use loop."""

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_text_response_returns_immediately(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """When model returns text (no FCs), return result immediately."""
        client = _make_mock_client()
        client.generate_with_tools.return_value = _make_text_result(text="All done!")

        agent = CodingAgent(client, _make_coding_config())
        result = agent.execute("Fix the bug")

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert result.content == "All done!"
        assert result.agent_name == "coding-agent"
        assert result.metadata["iterations"] == 1
        assert result.metadata["tools_executed"] == []

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_usage_accumulation_single_iteration(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """Usage tokens are accumulated from each iteration."""
        client = _make_mock_client()
        client.generate_with_tools.return_value = _make_text_result(
            usage={"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
        )

        agent = CodingAgent(client, _make_coding_config())
        result = agent.execute("Task")

        assert result.usage["prompt_tokens"] == 100
        assert result.usage["completion_tokens"] == 200
        assert result.usage["total_tokens"] == 300

    @patch("vaig.core.client.types.Part")
    @patch("vaig.agents.mixins.types.Content")
    @patch("vaig.agents.mixins.types.Part")
    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_function_call_then_text_response(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
        mock_part: MagicMock,
        mock_content: MagicMock,
        mock_client_part: MagicMock,
    ) -> None:
        """Model returns FC first, then text — two iterations."""
        client = _make_mock_client()

        # Build a mock tool to register
        read_tool = ToolDef(
            name="read_file",
            description="Read file",
            parameters=[ToolParam(name="path", type="string", description="Path")],
            execute=lambda path="": ToolResult(output="file contents here"),
        )

        # Iteration 1: function call
        fc_result = _make_fc_result(
            function_calls=[{"name": "read_file", "args": {"path": "main.py"}}],
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )
        # Iteration 2: text response
        text_result = _make_text_result(
            text="I read the file, here's my analysis.",
            usage={"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
        )
        client.generate_with_tools.side_effect = [fc_result, text_result]

        agent = CodingAgent(client, _make_coding_config())
        agent._registry.register(read_tool)

        result = agent.execute("Read and analyze main.py")

        assert result.success is True
        assert result.content == "I read the file, here's my analysis."
        assert result.metadata["iterations"] == 2
        assert len(result.metadata["tools_executed"]) == 1
        assert result.metadata["tools_executed"][0]["name"] == "read_file"
        # Usage accumulated across both iterations
        assert result.usage["prompt_tokens"] == 25
        assert result.usage["total_tokens"] == 70

    @patch("vaig.core.client.types.Part")
    @patch("vaig.agents.mixins.types.Content")
    @patch("vaig.agents.mixins.types.Part")
    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_multiple_function_calls_per_iteration(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
        mock_part: MagicMock,
        mock_content: MagicMock,
        mock_client_part: MagicMock,
    ) -> None:
        """Model returns multiple FCs in one response."""
        client = _make_mock_client()

        read_tool = ToolDef(
            name="read_file",
            description="Read",
            execute=lambda **kw: ToolResult(output="content"),
        )
        list_tool = ToolDef(
            name="list_files",
            description="List",
            execute=lambda **kw: ToolResult(output="a.py\nb.py"),
        )

        fc_result = _make_fc_result(
            function_calls=[
                {"name": "read_file", "args": {"path": "a.py"}},
                {"name": "list_files", "args": {"dir": "."}},
            ],
        )
        text_result = _make_text_result(text="Both done")
        client.generate_with_tools.side_effect = [fc_result, text_result]

        agent = CodingAgent(client, _make_coding_config())
        agent._registry.register(read_tool)
        agent._registry.register(list_tool)

        result = agent.execute("Read and list")

        assert result.success is True
        assert len(result.metadata["tools_executed"]) == 2

    @patch("vaig.core.client.types.Part")
    @patch("vaig.agents.mixins.types.Content")
    @patch("vaig.agents.mixins.types.Part")
    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_max_iterations_raises_error(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
        mock_part: MagicMock,
        mock_content: MagicMock,
        mock_client_part: MagicMock,
    ) -> None:
        """Exceeding max iterations raises MaxIterationsError."""
        client = _make_mock_client()

        read_tool = ToolDef(
            name="read_file",
            description="Read",
            execute=lambda **kw: ToolResult(output="data"),
        )

        # Always return function calls — never settles
        fc_result = _make_fc_result(
            function_calls=[{"name": "read_file", "args": {"path": "x.py"}}],
        )
        client.generate_with_tools.return_value = fc_result

        agent = CodingAgent(client, _make_coding_config(max_tool_iterations=3))
        agent._registry.register(read_tool)

        with pytest.raises(MaxIterationsError) as exc_info:
            agent.execute("Infinite loop")

        assert exc_info.value.iterations == 3

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_api_error_returns_failure_result(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """API call exception returns AgentResult with success=False."""
        client = _make_mock_client()
        client.generate_with_tools.side_effect = RuntimeError("API quota exceeded")

        agent = CodingAgent(client, _make_coding_config())
        result = agent.execute("Do something")

        assert result.success is False
        assert "API quota exceeded" in result.content
        assert result.metadata["error"] == "API quota exceeded"

    @patch("vaig.core.client.types.Part")
    @patch("vaig.agents.mixins.types.Content")
    @patch("vaig.agents.mixins.types.Part")
    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_first_iteration_sends_prompt_subsequent_empty(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
        mock_part: MagicMock,
        mock_content: MagicMock,
        mock_client_part: MagicMock,
    ) -> None:
        """First iteration sends full prompt, subsequent iterations send empty list."""
        client = _make_mock_client()

        read_tool = ToolDef(
            name="read_file",
            description="Read",
            execute=lambda **kw: ToolResult(output="ok"),
        )

        fc_result = _make_fc_result(
            function_calls=[{"name": "read_file", "args": {"path": "a.py"}}],
        )
        text_result = _make_text_result(text="Done")
        client.generate_with_tools.side_effect = [fc_result, text_result]

        agent = CodingAgent(client, _make_coding_config())
        agent._registry.register(read_tool)
        agent.execute("Analyze code")

        calls = client.generate_with_tools.call_args_list

        # First call: prompt is a non-empty string
        first_prompt = calls[0][0][0]
        assert isinstance(first_prompt, str)
        assert "Analyze code" in first_prompt

        # Second call: prompt is empty list
        second_prompt = calls[1][0][0]
        assert second_prompt == []

    @patch("vaig.core.client.types.Part")
    @patch("vaig.agents.mixins.types.Content")
    @patch("vaig.agents.mixins.types.Part")
    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_unknown_tool_returns_error_result_to_model(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
        mock_part: MagicMock,
        mock_content: MagicMock,
        mock_client_part: MagicMock,
    ) -> None:
        """Unknown tool name returns an error ToolResult, model continues."""
        client = _make_mock_client()

        fc_result = _make_fc_result(
            function_calls=[{"name": "nonexistent_tool", "args": {}}],
        )
        text_result = _make_text_result(text="I see, let me try differently.")
        client.generate_with_tools.side_effect = [fc_result, text_result]

        agent = CodingAgent(client, _make_coding_config())
        result = agent.execute("Use a tool")

        assert result.success is True
        assert result.metadata["tools_executed"][0]["error"] is True
        assert "does not exist in the registry" in result.metadata["tools_executed"][0]["output"]

    @patch("vaig.core.client.types.Part")
    @patch("vaig.agents.mixins.types.Content")
    @patch("vaig.agents.mixins.types.Part")
    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_tool_execution_error_handled_gracefully(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
        mock_part: MagicMock,
        mock_content: MagicMock,
        mock_client_part: MagicMock,
    ) -> None:
        """Tool execution error is caught and returned as error ToolResult."""
        client = _make_mock_client()

        def _failing_tool(**kw: Any) -> ToolResult:
            raise RuntimeError("Disk full")

        bad_tool = ToolDef(
            name="write_file",
            description="Write",
            execute=_failing_tool,
        )

        fc_result = _make_fc_result(
            function_calls=[{"name": "write_file", "args": {"path": "x.py", "content": "hi"}}],
        )
        text_result = _make_text_result(text="Write failed, trying plan B.")
        client.generate_with_tools.side_effect = [fc_result, text_result]

        agent = CodingAgent(client, _make_coding_config(confirm_actions=False))
        agent._registry.register(bad_tool)
        result = agent.execute("Write to file")

        assert result.success is True
        assert result.metadata["tools_executed"][0]["error"] is True
        assert "Disk full" in result.metadata["tools_executed"][0]["output"]

    @patch("vaig.core.client.types.Part")
    @patch("vaig.agents.mixins.types.Content")
    @patch("vaig.agents.mixins.types.Part")
    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_tool_type_error_handled(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
        mock_part: MagicMock,
        mock_content: MagicMock,
        mock_client_part: MagicMock,
    ) -> None:
        """TypeError from tool (wrong args) is caught and returned to model."""
        client = _make_mock_client()

        def _strict_tool(*, path: str) -> ToolResult:
            return ToolResult(output="ok")

        strict = ToolDef(
            name="read_file",
            description="Read",
            parameters=[ToolParam(name="path", type="string", description="Path")],
            execute=_strict_tool,
        )

        # Model passes wrong arg names
        fc_result = _make_fc_result(
            function_calls=[{"name": "read_file", "args": {"filename": "x.py"}}],
        )
        text_result = _make_text_result(text="Corrected")
        client.generate_with_tools.side_effect = [fc_result, text_result]

        agent = CodingAgent(client, _make_coding_config())
        agent._registry.register(strict)
        result = agent.execute("Read file")

        assert result.success is True
        assert result.metadata["tools_executed"][0]["error"] is True
        tool_output = result.metadata["tools_executed"][0]["output"]
        assert "Unknown argument" in tool_output or "Invalid arguments" in tool_output

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_conversation_history_tracked(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """User prompt and agent response are added to conversation history."""
        client = _make_mock_client()
        client.generate_with_tools.return_value = _make_text_result(text="Result")

        agent = CodingAgent(client, _make_coding_config())
        agent.execute("My task")

        assert len(agent.conversation_history) == 2
        assert agent.conversation_history[0].role == "user"
        assert "My task" in agent.conversation_history[0].content
        assert agent.conversation_history[1].role == "agent"
        assert agent.conversation_history[1].content == "Result"

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_execute_with_context(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """Context is included in the prompt using XML structure."""
        client = _make_mock_client()
        client.generate_with_tools.return_value = _make_text_result(text="Got it")

        agent = CodingAgent(client, _make_coding_config())
        agent.execute("Fix the bug", context="Error traceback here")

        first_call = client.generate_with_tools.call_args_list[0]
        prompt = first_call[0][0]
        assert "<external_code>" in prompt
        assert "Error traceback here" in prompt
        assert "<task>" in prompt
        assert "Fix the bug" in prompt

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_execute_without_context(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """Without context, prompt is just the task."""
        client = _make_mock_client()
        client.generate_with_tools.return_value = _make_text_result(text="ok")

        agent = CodingAgent(client, _make_coding_config())
        agent.execute("Simple task")

        first_call = client.generate_with_tools.call_args_list[0]
        prompt = first_call[0][0]
        assert prompt == "Simple task"
        assert "<external_code>" not in prompt


# ===========================================================================
# TestCodingAgentConfirmation
# ===========================================================================


class TestCodingAgentConfirmation:
    """Tests for confirmation callback integration."""

    @patch("vaig.core.client.types.Part")
    @patch("vaig.agents.mixins.types.Content")
    @patch("vaig.agents.mixins.types.Part")
    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_destructive_tool_calls_confirm_fn(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
        mock_part: MagicMock,
        mock_content: MagicMock,
        mock_client_part: MagicMock,
    ) -> None:
        """Destructive tools trigger the confirmation callback."""
        client = _make_mock_client()
        confirm_fn = MagicMock(return_value=True)

        write_tool = ToolDef(
            name="write_file",
            description="Write",
            execute=lambda **kw: ToolResult(output="written"),
        )

        fc_result = _make_fc_result(
            function_calls=[{"name": "write_file", "args": {"path": "x.py", "content": "hello"}}],
        )
        text_result = _make_text_result(text="Written")
        client.generate_with_tools.side_effect = [fc_result, text_result]

        agent = CodingAgent(client, _make_coding_config(confirm_actions=True), confirm_fn=confirm_fn)
        agent._registry.register(write_tool)
        result = agent.execute("Write file")

        confirm_fn.assert_called_once_with("write_file", {"path": "x.py", "content": "hello"})
        assert result.metadata["tools_executed"][0]["error"] is False

    @patch("vaig.core.client.types.Part")
    @patch("vaig.agents.mixins.types.Content")
    @patch("vaig.agents.mixins.types.Part")
    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_declined_destructive_returns_error(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
        mock_part: MagicMock,
        mock_content: MagicMock,
        mock_client_part: MagicMock,
    ) -> None:
        """Declining a destructive tool returns an error result to model."""
        client = _make_mock_client()
        confirm_fn = MagicMock(return_value=False)  # user declines

        write_tool = ToolDef(
            name="write_file",
            description="Write",
            execute=lambda **kw: ToolResult(output="written"),
        )

        fc_result = _make_fc_result(
            function_calls=[{"name": "write_file", "args": {"path": "x.py"}}],
        )
        text_result = _make_text_result(text="User declined, alternative approach")
        client.generate_with_tools.side_effect = [fc_result, text_result]

        agent = CodingAgent(client, _make_coding_config(confirm_actions=True), confirm_fn=confirm_fn)
        agent._registry.register(write_tool)
        result = agent.execute("Write file")

        assert result.metadata["tools_executed"][0]["error"] is True
        assert "declined" in result.metadata["tools_executed"][0]["output"]

    @patch("vaig.core.client.types.Part")
    @patch("vaig.agents.mixins.types.Content")
    @patch("vaig.agents.mixins.types.Part")
    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_non_destructive_tool_skips_confirmation(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
        mock_part: MagicMock,
        mock_content: MagicMock,
        mock_client_part: MagicMock,
    ) -> None:
        """Non-destructive tools (read_file, list_files) skip confirmation."""
        client = _make_mock_client()
        confirm_fn = MagicMock(return_value=True)

        read_tool = ToolDef(
            name="read_file",
            description="Read",
            execute=lambda **kw: ToolResult(output="content"),
        )

        fc_result = _make_fc_result(
            function_calls=[{"name": "read_file", "args": {"path": "x.py"}}],
        )
        text_result = _make_text_result(text="Read done")
        client.generate_with_tools.side_effect = [fc_result, text_result]

        agent = CodingAgent(client, _make_coding_config(confirm_actions=True), confirm_fn=confirm_fn)
        agent._registry.register(read_tool)
        result = agent.execute("Read a file")

        confirm_fn.assert_not_called()
        assert result.success is True


# ===========================================================================
# TestCodingAgentStream
# ===========================================================================


class TestCodingAgentStream:
    """Tests for CodingAgent.execute_stream() fallback."""

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_stream_falls_back_to_execute(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """execute_stream yields the result of execute() since streaming not supported."""
        client = _make_mock_client()
        client.generate_with_tools.return_value = _make_text_result(text="Streamed result")

        agent = CodingAgent(client, _make_coding_config())
        chunks = list(agent.execute_stream("Hello"))

        assert chunks == ["Streamed result"]


# ===========================================================================
# TestCodingAgentBuildPrompt
# ===========================================================================


class TestCodingAgentBuildPrompt:
    """Tests for _build_prompt() internal method."""

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_build_prompt_with_context(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        client = _make_mock_client()
        agent = CodingAgent(client, _make_coding_config())

        result = agent._build_prompt("Do stuff", "Some context")

        assert "<external_code>" in result
        assert "Some context" in result
        assert "<task>" in result
        assert "Do stuff" in result

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_build_prompt_without_context(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        client = _make_mock_client()
        agent = CodingAgent(client, _make_coding_config())

        result = agent._build_prompt("Do stuff", "")

        assert result == "Do stuff"


# ===========================================================================
# TestCodingSystemPrompt
# ===========================================================================


class TestCodingSystemPrompt:
    """Tests for CODING_SYSTEM_PROMPT content."""

    def test_contains_tool_usage_guidance(self) -> None:
        assert "read_file" in CODING_SYSTEM_PROMPT
        assert "write_file" in CODING_SYSTEM_PROMPT
        assert "edit_file" in CODING_SYSTEM_PROMPT
        assert "list_files" in CODING_SYSTEM_PROMPT
        assert "search_files" in CODING_SYSTEM_PROMPT
        assert "run_command" in CODING_SYSTEM_PROMPT

    def test_contains_rules(self) -> None:
        assert "COMPLETE" in CODING_SYSTEM_PROMPT
        assert "production-ready" in CODING_SYSTEM_PROMPT

    def test_contains_error_handling_guidance(self) -> None:
        assert "Error Handling" in CODING_SYSTEM_PROMPT
        assert "Do NOT repeat the same failing call" in CODING_SYSTEM_PROMPT


# ===========================================================================
# TestCodingAgentUsageAccumulation
# ===========================================================================


class TestCodingAgentUsageAccumulation:
    """Tests for token usage accumulation across iterations."""

    @patch("vaig.core.client.types.Part")
    @patch("vaig.agents.mixins.types.Content")
    @patch("vaig.agents.mixins.types.Part")
    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_usage_accumulates_across_three_iterations(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
        mock_part: MagicMock,
        mock_content: MagicMock,
        mock_client_part: MagicMock,
    ) -> None:
        """Three iterations of tool calls accumulate usage correctly."""
        client = _make_mock_client()

        tool = ToolDef(
            name="read_file",
            description="Read",
            execute=lambda **kw: ToolResult(output="ok"),
        )

        # 3 iterations: FC, FC, text
        fc1 = _make_fc_result(
            function_calls=[{"name": "read_file", "args": {"path": "a.py"}}],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        fc2 = _make_fc_result(
            function_calls=[{"name": "read_file", "args": {"path": "b.py"}}],
            usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        )
        text = _make_text_result(
            text="Final",
            usage={"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45},
        )
        client.generate_with_tools.side_effect = [fc1, fc2, text]

        agent = CodingAgent(client, _make_coding_config())
        agent._registry.register(tool)

        result = agent.execute("Analyze")

        assert result.usage["prompt_tokens"] == 60
        assert result.usage["completion_tokens"] == 30
        assert result.usage["total_tokens"] == 90
        assert result.metadata["iterations"] == 3

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_usage_with_missing_keys_defaults_to_zero(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """Missing usage keys default to 0 in accumulation."""
        client = _make_mock_client()
        client.generate_with_tools.return_value = _make_text_result(
            text="ok",
            usage={},  # No usage data
        )

        agent = CodingAgent(client, _make_coding_config())
        result = agent.execute("Task")

        assert result.usage["prompt_tokens"] == 0
        assert result.usage["completion_tokens"] == 0
        assert result.usage["total_tokens"] == 0


# ===========================================================================
# TestCodingAgentProperties
# ===========================================================================


class TestCodingAgentProperties:
    """Tests for CodingAgent properties."""

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_workspace_property(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        client = _make_mock_client()
        agent = CodingAgent(client, _make_coding_config(workspace_root="/tmp/ws"))

        assert agent.workspace == Path("/tmp/ws").resolve()

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_registry_property(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        client = _make_mock_client()
        agent = CodingAgent(client, _make_coding_config())

        assert isinstance(agent.registry, ToolRegistry)

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_repr(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        client = _make_mock_client()
        agent = CodingAgent(client, _make_coding_config())

        r = repr(agent)
        assert "CodingAgent" in r
        assert "coding-agent" in r
        assert "coder" in r


# ===========================================================================
# TestDeduplicateResponse
# ===========================================================================


class TestDeduplicateResponse:
    """Tests for CodingAgent._deduplicate_response() static method."""

    def test_empty_string(self) -> None:
        """Empty input returns empty output."""
        assert CodingAgent._deduplicate_response("") == ""

    def test_none_returns_none(self) -> None:
        """None input returns None (falsy passthrough)."""
        assert CodingAgent._deduplicate_response(None) is None  # type: ignore[arg-type]

    def test_no_repetition(self) -> None:
        """Normal text without repetition passes through unchanged."""
        text = "Line one here.\nAnother different line.\nThird line is unique."
        assert CodingAgent._deduplicate_response(text) == text

    def test_few_repeats_below_threshold(self) -> None:
        """Two consecutive repeats (below default threshold=3) are kept."""
        text = "This line repeats.\nThis line repeats.\nDifferent ending."
        assert CodingAgent._deduplicate_response(text) == text

    def test_exactly_at_threshold_keeps_all(self) -> None:
        """Exactly 3 consecutive repeats (default threshold) — all 3 are kept."""
        line = "I have created the Priority enum class."
        text = f"{line}\n{line}\n{line}\nDone."
        # threshold=3 means we keep first 3 occurrences (repeat_count reaches 3
        # only on the 3rd occurrence, and >= triggers truncation on the NEXT one)
        result = CodingAgent._deduplicate_response(text)
        assert result.count(line) == 3
        assert "[truncated" not in result

    def test_above_threshold_truncates(self) -> None:
        """More than 3 consecutive repeats get truncated to 3."""
        line = "I have created the Priority enum class."
        repeated = "\n".join([line] * 10)
        text = f"Start\n{repeated}\nEnd"

        result = CodingAgent._deduplicate_response(text)

        assert result.count(line) == 3
        assert "[truncated — repeated text removed]" in result

    def test_massive_repetition(self) -> None:
        """200+ repetitions (the real Gemini bug) are truncated efficiently."""
        line = "I have created the Priority enum successfully and it is ready to use."
        repeated = "\n".join([line] * 200)
        text = f"Here is the file:\n{repeated}\n\nLet me know if you need changes."

        result = CodingAgent._deduplicate_response(text)

        assert result.count(line) == 3
        assert "[truncated — repeated text removed]" in result
        # Result should be MUCH shorter than input
        assert len(result) < len(text) // 10

    def test_short_lines_not_tracked(self) -> None:
        """Short lines (≤10 chars) are never counted as repeats."""
        text = "```\n```\n```\n```\n```\n```\n```"
        result = CodingAgent._deduplicate_response(text)
        assert result == text
        assert "[truncated" not in result

    def test_blank_lines_not_tracked(self) -> None:
        """Blank lines are ignored (≤10 chars)."""
        text = "Content here.\n\n\n\n\n\n\nMore content here."
        result = CodingAgent._deduplicate_response(text)
        assert result == text

    def test_non_consecutive_repeats_not_truncated(self) -> None:
        """Same line repeated non-consecutively is NOT truncated."""
        line = "This appears multiple times."
        text = f"{line}\nSomething else.\n{line}\nAnother thing.\n{line}\nMore.\n{line}"
        result = CodingAgent._deduplicate_response(text)
        assert result == text  # All kept — non-consecutive

    def test_custom_threshold(self) -> None:
        """Custom threshold parameter controls truncation point."""
        line = "Repeated line with enough characters."
        text = "\n".join([line] * 5)

        # threshold=1 — keep only 1 occurrence
        result = CodingAgent._deduplicate_response(text, threshold=1)
        assert result.count(line) == 1
        assert "[truncated" in result

        # threshold=5 — keep all 5
        result = CodingAgent._deduplicate_response(text, threshold=5)
        assert result.count(line) == 5
        assert "[truncated" not in result

    def test_multiple_groups_of_repeats(self) -> None:
        """Multiple different repeated groups each get truncated independently."""
        line_a = "First repeated sentence with enough length."
        line_b = "Second repeated sentence with enough length."
        text = "\n".join([line_a] * 6 + ["break"] + [line_b] * 6)

        result = CodingAgent._deduplicate_response(text)

        assert result.count(line_a) == 3
        assert result.count(line_b) == 3
        assert "[truncated" in result

    def test_preserves_indentation(self) -> None:
        """Original indentation is preserved in kept lines."""
        line = "    return self.process(data)"
        text = "\n".join([line] * 5)

        result = CodingAgent._deduplicate_response(text)

        # The kept lines should still be indented
        for kept_line in result.split("\n"):
            if "return self.process" in kept_line:
                assert kept_line.startswith("    ")

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_execute_applies_deduplication(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """Integration: execute() applies deduplication to text response."""
        client = _make_mock_client()
        line = "I have created the Priority enum class successfully."
        repeated_text = "\n".join([line] * 50)
        client.generate_with_tools.return_value = _make_text_result(text=repeated_text)

        agent = CodingAgent(client, _make_coding_config())
        result = agent.execute("Create priority enum")

        assert result.success is True
        assert result.content.count(line) == 3
        assert "[truncated — repeated text removed]" in result.content

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_execute_passes_frequency_penalty(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """Integration: execute() passes frequency_penalty=0.15 to generate_with_tools."""
        client = _make_mock_client()
        client.generate_with_tools.return_value = _make_text_result(text="Done")

        agent = CodingAgent(client, _make_coding_config())
        agent.execute("Do something")

        call_kwargs = client.generate_with_tools.call_args[1]
        assert call_kwargs.get("frequency_penalty") == 0.15


# ── tool_result_cache passthrough ────────────────────────────


class TestCodingAgentCachePassthrough:
    """Tests that tool_result_cache is forwarded to _run_tool_loop / _async_run_tool_loop."""

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    @patch.object(CodingAgent, "_run_tool_loop")
    def test_execute_forwards_cache(
        self,
        mock_loop: MagicMock,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        client = _make_mock_client()
        agent = CodingAgent(client, _make_coding_config())

        mock_loop.return_value = MagicMock(
            text="result",
            model="gemini-2.5-pro",
            finish_reason="STOP",
            iterations=1,
            tools_executed=[],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        cache = ToolResultCache()
        agent.execute("Write code", tool_result_cache=cache)

        _, kwargs = mock_loop.call_args
        assert kwargs["tool_result_cache"] is cache

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    @patch.object(CodingAgent, "_run_tool_loop")
    def test_execute_cache_defaults_to_none(
        self,
        mock_loop: MagicMock,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        client = _make_mock_client()
        agent = CodingAgent(client, _make_coding_config())

        mock_loop.return_value = MagicMock(
            text="result",
            model="gemini-2.5-pro",
            finish_reason="STOP",
            iterations=1,
            tools_executed=[],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        agent.execute("Write code")

        _, kwargs = mock_loop.call_args
        assert kwargs["tool_result_cache"] is None

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    @patch.object(CodingAgent, "_run_tool_loop")
    def test_execute_forwards_on_tool_call_and_store(
        self,
        mock_loop: MagicMock,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        client = _make_mock_client()
        agent = CodingAgent(client, _make_coding_config())

        mock_loop.return_value = MagicMock(
            text="result",
            model="gemini-2.5-pro",
            finish_reason="STOP",
            iterations=1,
            tools_executed=[],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        on_tool_call = MagicMock()
        tool_call_store = MagicMock()
        agent.execute(
            "Write code",
            on_tool_call=on_tool_call,
            tool_call_store=tool_call_store,
        )

        _, kwargs = mock_loop.call_args
        assert kwargs["on_tool_call"] is on_tool_call
        assert kwargs["tool_call_store"] is tool_call_store

    @pytest.mark.asyncio
    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    @patch.object(CodingAgent, "_async_run_tool_loop")
    async def test_async_execute_forwards_cache(
        self,
        mock_loop: AsyncMock,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        client = _make_mock_client()
        agent = CodingAgent(client, _make_coding_config())

        mock_loop.return_value = MagicMock(
            text="result",
            model="gemini-2.5-pro",
            finish_reason="STOP",
            iterations=1,
            tools_executed=[],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        cache = ToolResultCache()
        await agent.async_execute("Write code", tool_result_cache=cache)

        _, kwargs = mock_loop.call_args
        assert kwargs["tool_result_cache"] is cache

    @pytest.mark.asyncio
    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    @patch.object(CodingAgent, "_async_run_tool_loop")
    async def test_async_execute_cache_defaults_to_none(
        self,
        mock_loop: AsyncMock,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        client = _make_mock_client()
        agent = CodingAgent(client, _make_coding_config())

        mock_loop.return_value = MagicMock(
            text="result",
            model="gemini-2.5-pro",
            finish_reason="STOP",
            iterations=1,
            tools_executed=[],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        await agent.async_execute("Write code")

        _, kwargs = mock_loop.call_args
        assert kwargs["tool_result_cache"] is None


# ===========================================================================
# TestCodingSystemPromptPhase1Improvements
# ===========================================================================


class TestCodingSystemPromptPhase1Improvements:
    """Tests for the Phase 1 improvements to CODING_SYSTEM_PROMPT."""

    def test_contains_cot_instruction(self) -> None:
        """CODING_SYSTEM_PROMPT includes CoT enforcement content."""
        from vaig.core.prompt_defense import COT_INSTRUCTION

        assert COT_INSTRUCTION in CODING_SYSTEM_PROMPT

    def test_contains_anti_hallucination_rules(self) -> None:
        """CODING_SYSTEM_PROMPT includes anti-hallucination rules."""
        from vaig.core.prompt_defense import ANTI_HALLUCINATION_RULES

        assert ANTI_HALLUCINATION_RULES in CODING_SYSTEM_PROMPT

    def test_contains_spec_phase(self) -> None:
        """CODING_SYSTEM_PROMPT includes Phase 0 spec requirement."""
        assert "Phase 0" in CODING_SYSTEM_PROMPT
        assert "SPEC.md" in CODING_SYSTEM_PROMPT
        assert "specification" in CODING_SYSTEM_PROMPT.lower()

    def test_phases_renumbered(self) -> None:
        """Existing phases are renumbered: Phase 1 and Phase 2 still present."""
        assert "Phase 1" in CODING_SYSTEM_PROMPT
        assert "Phase 2" in CODING_SYSTEM_PROMPT
        assert "Phase 3" in CODING_SYSTEM_PROMPT

    def test_contains_verify_completeness_reference(self) -> None:
        """CODING_SYSTEM_PROMPT references the verify_completeness tool."""
        assert "verify_completeness" in CODING_SYSTEM_PROMPT

    def test_contains_chain_of_thought_section(self) -> None:
        """CODING_SYSTEM_PROMPT has a Chain-of-Thought section."""
        assert "Chain-of-Thought" in CODING_SYSTEM_PROMPT


# ===========================================================================
# TestCodingAgentBuildPromptXML
# ===========================================================================


class TestCodingAgentBuildPromptXML:
    """Tests for the XML context boundary implementation in _build_prompt."""

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_context_wrapped_in_xml_tags(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """External context is wrapped in <external_code> XML tags."""
        client = _make_mock_client()
        agent = CodingAgent(client, _make_coding_config())

        result = agent._build_prompt("Fix the bug", "some external code")

        assert "<external_code>" in result
        assert "</external_code>" in result
        assert "some external code" in result

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_task_wrapped_in_xml_tags(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """Task is wrapped in <task> XML tags."""
        client = _make_mock_client()
        agent = CodingAgent(client, _make_coding_config())

        result = agent._build_prompt("Fix the bug", "some context")

        assert "<task>" in result
        assert "</task>" in result
        assert "Fix the bug" in result

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_system_rules_tag_present_with_context(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """<system_rules> tag is present when context is provided."""
        client = _make_mock_client()
        agent = CodingAgent(client, _make_coding_config())

        result = agent._build_prompt("Do something", "context data")

        assert "<system_rules>" in result
        assert "</system_rules>" in result

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_context_uses_untrusted_delimiters(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """Context is wrapped with prompt_defense delimiter markers."""
        from vaig.core.prompt_defense import DELIMITER_DATA_END, DELIMITER_DATA_START

        client = _make_mock_client()
        agent = CodingAgent(client, _make_coding_config())

        result = agent._build_prompt("task", "untrusted data")

        assert DELIMITER_DATA_START in result
        assert DELIMITER_DATA_END in result
        assert "untrusted data" in result

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    def test_empty_context_returns_plain_prompt(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """Empty context returns the raw prompt (no XML wrapping)."""
        client = _make_mock_client()
        agent = CodingAgent(client, _make_coding_config())

        result = agent._build_prompt("My task", "")

        assert result == "My task"
        assert "<" not in result

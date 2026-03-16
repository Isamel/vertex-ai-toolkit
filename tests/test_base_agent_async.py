"""Tests for async methods on BaseAgent, SpecialistAgent, CodingAgent, and ToolAwareAgent.

Phase 3, Task 3.3 — verify that async_execute() and async_execute_stream()
work correctly for all agent types that extend BaseAgent.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vaig.agents.base import AgentConfig, AgentResult, BaseAgent
from vaig.agents.specialist import SpecialistAgent
from vaig.core.client import GenerationResult, ToolCallResult
from vaig.core.config import CodingConfig
from vaig.core.exceptions import (
    GeminiConnectionError,
    GeminiRateLimitError,
    MaxIterationsError,
)
from vaig.tools import ToolDef, ToolParam, ToolRegistry, ToolResult

# ── Helpers ──────────────────────────────────────────────────


def _make_mock_client(current_model: str = "gemini-2.5-pro") -> MagicMock:
    """Create a MagicMock that behaves like GeminiClient."""
    client = MagicMock()
    client.current_model = current_model
    return client


def _make_generation_result(
    text: str = "Done",
    model: str = "gemini-2.5-pro",
    usage: dict[str, int] | None = None,
    finish_reason: str = "STOP",
) -> GenerationResult:
    """Create a GenerationResult for SpecialistAgent tests."""
    return GenerationResult(
        text=text,
        model=model,
        usage=usage or {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        finish_reason=finish_reason,
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
        usage=usage or {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
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


def _make_coding_config(
    workspace_root: str = "/tmp/test-workspace",
    max_tool_iterations: int = 10,
    confirm_actions: bool = False,
) -> CodingConfig:
    return CodingConfig(
        workspace_root=workspace_root,
        max_tool_iterations=max_tool_iterations,
        confirm_actions=confirm_actions,
        allowed_commands=[],
    )


# ===========================================================================
# TestBaseAgentAbstractAsync
# ===========================================================================


class TestBaseAgentAbstractAsync:
    """Verify that BaseAgent declares abstract async methods."""

    def test_base_agent_has_async_execute(self) -> None:
        """BaseAgent must declare async_execute as an abstract method."""
        assert hasattr(BaseAgent, "async_execute")
        assert getattr(BaseAgent.async_execute, "__isabstractmethod__", False)

    def test_base_agent_has_async_execute_stream(self) -> None:
        """BaseAgent must declare async_execute_stream as an abstract method."""
        assert hasattr(BaseAgent, "async_execute_stream")
        assert getattr(BaseAgent.async_execute_stream, "__isabstractmethod__", False)

    def test_cannot_instantiate_without_async_methods(self) -> None:
        """A subclass that only implements sync methods cannot be instantiated."""

        class IncompleteAgent(BaseAgent):
            def execute(self, prompt: str, *, context: str = "") -> AgentResult:
                return AgentResult(agent_name="test", content="ok")

            def execute_stream(self, prompt, *, context=""):
                yield "ok"

        with pytest.raises(TypeError, match="abstract"):
            IncompleteAgent(  # type: ignore[abstract]
                AgentConfig(name="t", role="t", system_instruction="t"),
                MagicMock(),
            )


# ===========================================================================
# TestSpecialistAgentAsyncExecute
# ===========================================================================


class TestSpecialistAgentAsyncExecute:
    """Tests for SpecialistAgent.async_execute()."""

    async def test_returns_successful_result(self) -> None:
        """async_execute returns AgentResult with success=True on normal response."""
        client = _make_mock_client()
        client.async_generate = AsyncMock(
            return_value=_make_generation_result(text="Analysis complete"),
        )

        config = AgentConfig(
            name="analyzer", role="analyst", system_instruction="Analyze.",
        )
        agent = SpecialistAgent(config, client)
        result = await agent.async_execute("Analyze this data")

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert result.content == "Analysis complete"
        assert result.agent_name == "analyzer"
        client.async_generate.assert_awaited_once()

    async def test_includes_usage_and_metadata(self) -> None:
        """async_execute populates usage and metadata from the generation result."""
        client = _make_mock_client()
        client.async_generate = AsyncMock(
            return_value=_make_generation_result(
                text="ok",
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                finish_reason="STOP",
            ),
        )

        config = AgentConfig(
            name="agent", role="helper", system_instruction="Help.",
        )
        agent = SpecialistAgent(config, client)
        result = await agent.async_execute("Test")

        assert result.usage["total_tokens"] == 150
        assert result.metadata["model"] == "gemini-2.5-pro"
        assert result.metadata["finish_reason"] == "STOP"

    async def test_with_context(self) -> None:
        """Context is included in the prompt sent to async_generate."""
        client = _make_mock_client()
        client.async_generate = AsyncMock(
            return_value=_make_generation_result(text="Got it"),
        )

        config = AgentConfig(
            name="agent", role="helper", system_instruction="Help.",
        )
        agent = SpecialistAgent(config, client)
        await agent.async_execute("Do this", context="Important context")

        call_args = client.async_generate.call_args
        prompt = call_args[0][0]
        assert "## Context" in prompt
        assert "Important context" in prompt
        assert "## Task" in prompt
        assert "Do this" in prompt

    async def test_tracks_conversation_history(self) -> None:
        """async_execute adds user prompt and agent response to conversation."""
        client = _make_mock_client()
        client.async_generate = AsyncMock(
            return_value=_make_generation_result(text="Response"),
        )

        config = AgentConfig(
            name="agent", role="helper", system_instruction="Help.",
        )
        agent = SpecialistAgent(config, client)
        await agent.async_execute("My question")

        assert len(agent.conversation_history) == 2
        assert agent.conversation_history[0].role == "user"
        assert "My question" in agent.conversation_history[0].content
        assert agent.conversation_history[1].role == "agent"
        assert agent.conversation_history[1].content == "Response"

    async def test_rate_limit_error_returns_failure(self) -> None:
        """GeminiRateLimitError returns AgentResult with success=False."""
        client = _make_mock_client()
        client.async_generate = AsyncMock(
            side_effect=GeminiRateLimitError("rate limited", retries_attempted=3),
        )

        config = AgentConfig(
            name="agent", role="helper", system_instruction="Help.",
        )
        agent = SpecialistAgent(config, client)
        result = await agent.async_execute("Task")

        assert result.success is False
        assert "Rate limit exceeded" in result.content
        assert result.metadata["error_type"] == "rate_limit"

    async def test_connection_error_returns_failure(self) -> None:
        """GeminiConnectionError returns AgentResult with success=False."""
        client = _make_mock_client()
        client.async_generate = AsyncMock(
            side_effect=GeminiConnectionError("connection lost", retries_attempted=2),
        )

        config = AgentConfig(
            name="agent", role="helper", system_instruction="Help.",
        )
        agent = SpecialistAgent(config, client)
        result = await agent.async_execute("Task")

        assert result.success is False
        assert "Connection error" in result.content
        assert result.metadata["error_type"] == "connection"

    async def test_generic_error_returns_failure(self) -> None:
        """Generic exception returns AgentResult with success=False."""
        client = _make_mock_client()
        client.async_generate = AsyncMock(
            side_effect=RuntimeError("unexpected failure"),
        )

        config = AgentConfig(
            name="agent", role="helper", system_instruction="Help.",
        )
        agent = SpecialistAgent(config, client)
        result = await agent.async_execute("Task")

        assert result.success is False
        assert "unexpected failure" in result.content

    async def test_passes_model_and_temperature(self) -> None:
        """async_execute passes model ID and temperature from config."""
        client = _make_mock_client()
        client.async_generate = AsyncMock(
            return_value=_make_generation_result(text="ok"),
        )

        config = AgentConfig(
            name="agent",
            role="helper",
            system_instruction="Help.",
            model="gemini-2.5-flash",
            temperature=0.3,
        )
        agent = SpecialistAgent(config, client)
        await agent.async_execute("Task")

        call_kwargs = client.async_generate.call_args[1]
        assert call_kwargs["model_id"] == "gemini-2.5-flash"
        assert call_kwargs["temperature"] == 0.3


# ===========================================================================
# TestSpecialistAgentAsyncExecuteStream
# ===========================================================================


class TestSpecialistAgentAsyncExecuteStream:
    """Tests for SpecialistAgent.async_execute_stream()."""

    async def test_yields_chunks(self) -> None:
        """async_execute_stream yields text chunks from the async stream."""
        client = _make_mock_client()

        # Create a mock StreamResult that supports async iteration
        mock_stream = AsyncMock()
        mock_stream.__aiter__ = MagicMock(return_value=iter(["Hello ", "world"]))

        # Use an async generator mock
        async def _async_iter():
            yield "Hello "
            yield "world"

        mock_stream_result = MagicMock()
        mock_stream_result.__aiter__ = lambda self: _async_iter()

        client.async_generate_stream = AsyncMock(return_value=mock_stream_result)

        config = AgentConfig(
            name="agent", role="helper", system_instruction="Help.",
        )
        agent = SpecialistAgent(config, client)

        chunks = []
        async for chunk in agent.async_execute_stream("Hello"):
            chunks.append(chunk)

        assert chunks == ["Hello ", "world"]
        client.async_generate_stream.assert_awaited_once()

    async def test_stream_error_yields_error_message(self) -> None:
        """Generic error during streaming yields error message."""
        client = _make_mock_client()
        client.async_generate_stream = AsyncMock(
            side_effect=RuntimeError("stream failed"),
        )

        config = AgentConfig(
            name="agent", role="helper", system_instruction="Help.",
        )
        agent = SpecialistAgent(config, client)

        chunks = []
        async for chunk in agent.async_execute_stream("Hello"):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert "stream failed" in chunks[0]

    async def test_stream_rate_limit_yields_error(self) -> None:
        """Rate limit error during streaming yields error message."""
        client = _make_mock_client()
        client.async_generate_stream = AsyncMock(
            side_effect=GeminiRateLimitError("rate limited", retries_attempted=3),
        )

        config = AgentConfig(
            name="agent", role="helper", system_instruction="Help.",
        )
        agent = SpecialistAgent(config, client)

        chunks = []
        async for chunk in agent.async_execute_stream("Hello"):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert "Rate limit exceeded" in chunks[0]


# ===========================================================================
# TestCodingAgentAsyncExecute
# ===========================================================================


class TestCodingAgentAsyncExecute:
    """Tests for CodingAgent.async_execute()."""

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    async def test_text_response_returns_immediately(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """When model returns text (no FCs), return result immediately."""
        from vaig.agents.coding import CodingAgent

        client = _make_mock_client()
        client.async_generate_with_tools = AsyncMock(
            return_value=_make_text_result(text="All done!"),
        )

        agent = CodingAgent(client, _make_coding_config())
        result = await agent.async_execute("Fix the bug")

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert result.content == "All done!"
        assert result.agent_name == "coding-agent"
        assert result.metadata["iterations"] == 1
        assert result.metadata["tools_executed"] == []

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    async def test_usage_accumulation(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """Usage tokens are accumulated correctly."""
        from vaig.agents.coding import CodingAgent

        client = _make_mock_client()
        client.async_generate_with_tools = AsyncMock(
            return_value=_make_text_result(
                usage={"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
            ),
        )

        agent = CodingAgent(client, _make_coding_config())
        result = await agent.async_execute("Task")

        assert result.usage["total_tokens"] == 300

    @patch("vaig.core.client.types.Part")
    @patch("vaig.agents.mixins.types.Content")
    @patch("vaig.agents.mixins.types.Part")
    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    async def test_function_call_then_text(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
        mock_part: MagicMock,
        mock_content: MagicMock,
        mock_client_part: MagicMock,
    ) -> None:
        """Model returns FC first, then text — two async iterations."""
        from vaig.agents.coding import CodingAgent

        client = _make_mock_client()

        read_tool = ToolDef(
            name="read_file",
            description="Read file",
            parameters=[ToolParam(name="path", type="string", description="Path")],
            execute=lambda path="": ToolResult(output="file contents"),
        )

        fc_result = _make_fc_result(
            function_calls=[{"name": "read_file", "args": {"path": "main.py"}}],
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )
        text_result = _make_text_result(
            text="Analysis done.",
            usage={"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
        )
        client.async_generate_with_tools = AsyncMock(
            side_effect=[fc_result, text_result],
        )

        agent = CodingAgent(client, _make_coding_config())
        agent._registry.register(read_tool)

        result = await agent.async_execute("Read and analyze")

        assert result.success is True
        assert result.content == "Analysis done."
        assert result.metadata["iterations"] == 2
        assert len(result.metadata["tools_executed"]) == 1
        assert result.metadata["tools_executed"][0]["name"] == "read_file"
        assert result.usage["total_tokens"] == 70

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    async def test_api_error_returns_failure(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """API exception returns AgentResult with success=False."""
        from vaig.agents.coding import CodingAgent

        client = _make_mock_client()
        client.async_generate_with_tools = AsyncMock(
            side_effect=RuntimeError("API quota exceeded"),
        )

        agent = CodingAgent(client, _make_coding_config())
        result = await agent.async_execute("Do something")

        assert result.success is False
        assert "API quota exceeded" in result.content
        assert result.metadata["error"] == "API quota exceeded"

    @patch("vaig.core.client.types.Part")
    @patch("vaig.agents.mixins.types.Content")
    @patch("vaig.agents.mixins.types.Part")
    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    async def test_max_iterations_raises(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
        mock_part: MagicMock,
        mock_content: MagicMock,
        mock_client_part: MagicMock,
    ) -> None:
        """Exceeding max iterations raises MaxIterationsError."""
        from vaig.agents.coding import CodingAgent

        client = _make_mock_client()

        read_tool = ToolDef(
            name="read_file",
            description="Read",
            execute=lambda **kw: ToolResult(output="data"),
        )

        fc_result = _make_fc_result(
            function_calls=[{"name": "read_file", "args": {"path": "x.py"}}],
        )
        client.async_generate_with_tools = AsyncMock(return_value=fc_result)

        agent = CodingAgent(client, _make_coding_config(max_tool_iterations=3))
        agent._registry.register(read_tool)

        with pytest.raises(MaxIterationsError) as exc_info:
            await agent.async_execute("Infinite loop")

        assert exc_info.value.iterations == 3

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    async def test_deduplication_applied(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """async_execute applies response deduplication."""
        from vaig.agents.coding import CodingAgent

        client = _make_mock_client()
        line = "I have created the Priority enum class successfully."
        repeated_text = "\n".join([line] * 50)
        client.async_generate_with_tools = AsyncMock(
            return_value=_make_text_result(text=repeated_text),
        )

        agent = CodingAgent(client, _make_coding_config())
        result = await agent.async_execute("Create priority enum")

        assert result.success is True
        assert result.content.count(line) == 3
        assert "[truncated — repeated text removed]" in result.content

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    async def test_conversation_history_tracked(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """User prompt and agent response are added to conversation history."""
        from vaig.agents.coding import CodingAgent

        client = _make_mock_client()
        client.async_generate_with_tools = AsyncMock(
            return_value=_make_text_result(text="Result"),
        )

        agent = CodingAgent(client, _make_coding_config())
        await agent.async_execute("My task")

        assert len(agent.conversation_history) == 2
        assert agent.conversation_history[0].role == "user"
        assert "My task" in agent.conversation_history[0].content
        assert agent.conversation_history[1].role == "agent"
        assert agent.conversation_history[1].content == "Result"


# ===========================================================================
# TestCodingAgentAsyncConfirmation
# ===========================================================================


class TestCodingAgentAsyncConfirmation:
    """Tests for CodingAgent async confirmation callback."""

    @patch("vaig.core.client.types.Part")
    @patch("vaig.agents.mixins.types.Content")
    @patch("vaig.agents.mixins.types.Part")
    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    async def test_destructive_tool_calls_confirm_fn(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
        mock_part: MagicMock,
        mock_content: MagicMock,
        mock_client_part: MagicMock,
    ) -> None:
        """Destructive tools trigger the confirmation callback in async mode."""
        from vaig.agents.coding import CodingAgent

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
        client.async_generate_with_tools = AsyncMock(
            side_effect=[fc_result, text_result],
        )

        config = _make_coding_config()
        config = CodingConfig(
            workspace_root="/tmp/test-workspace",
            max_tool_iterations=10,
            confirm_actions=True,
            allowed_commands=[],
        )
        agent = CodingAgent(client, config, confirm_fn=confirm_fn)
        agent._registry.register(write_tool)
        result = await agent.async_execute("Write file")

        confirm_fn.assert_called_once_with("write_file", {"path": "x.py", "content": "hello"})
        assert result.metadata["tools_executed"][0]["error"] is False

    @patch("vaig.core.client.types.Part")
    @patch("vaig.agents.mixins.types.Content")
    @patch("vaig.agents.mixins.types.Part")
    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    async def test_declined_destructive_returns_error(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
        mock_part: MagicMock,
        mock_content: MagicMock,
        mock_client_part: MagicMock,
    ) -> None:
        """Declining a destructive tool returns error result to model in async mode."""
        from vaig.agents.coding import CodingAgent

        client = _make_mock_client()
        confirm_fn = MagicMock(return_value=False)

        write_tool = ToolDef(
            name="write_file",
            description="Write",
            execute=lambda **kw: ToolResult(output="written"),
        )

        fc_result = _make_fc_result(
            function_calls=[{"name": "write_file", "args": {"path": "x.py"}}],
        )
        text_result = _make_text_result(text="User declined, alternative approach")
        client.async_generate_with_tools = AsyncMock(
            side_effect=[fc_result, text_result],
        )

        config = CodingConfig(
            workspace_root="/tmp/test-workspace",
            max_tool_iterations=10,
            confirm_actions=True,
            allowed_commands=[],
        )
        agent = CodingAgent(client, config, confirm_fn=confirm_fn)
        agent._registry.register(write_tool)
        result = await agent.async_execute("Write file")

        assert result.metadata["tools_executed"][0]["error"] is True
        assert "declined" in result.metadata["tools_executed"][0]["output"]


# ===========================================================================
# TestCodingAgentAsyncStream
# ===========================================================================


class TestCodingAgentAsyncStream:
    """Tests for CodingAgent.async_execute_stream() fallback."""

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    async def test_stream_falls_back_to_async_execute(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """async_execute_stream yields the result of async_execute()."""
        from vaig.agents.coding import CodingAgent

        client = _make_mock_client()
        client.async_generate_with_tools = AsyncMock(
            return_value=_make_text_result(text="Streamed result"),
        )

        agent = CodingAgent(client, _make_coding_config())
        chunks = []
        async for chunk in agent.async_execute_stream("Hello"):
            chunks.append(chunk)

        assert chunks == ["Streamed result"]


# ===========================================================================
# TestToolAwareAgentAsync
# ===========================================================================


class TestToolAwareAgentAsync:
    """Tests for ToolAwareAgent.async_execute() and async_execute_stream()."""

    async def test_text_response_returns_result(self) -> None:
        """async_execute returns AgentResult on text response."""
        from vaig.agents.tool_aware import ToolAwareAgent

        client = _make_mock_client()
        client.async_generate_with_tools = AsyncMock(
            return_value=_make_text_result(text="Task complete"),
        )

        registry = ToolRegistry()
        agent = ToolAwareAgent(
            system_instruction="You are a helper.",
            tool_registry=registry,
            model="gemini-2.5-pro",
            name="test-agent",
            client=client,
        )
        result = await agent.async_execute("Do something")

        assert result.success is True
        assert result.content == "Task complete"
        assert result.agent_name == "test-agent"

    async def test_api_error_returns_failure(self) -> None:
        """API exception returns AgentResult with success=False."""
        from vaig.agents.tool_aware import ToolAwareAgent

        client = _make_mock_client()
        client.async_generate_with_tools = AsyncMock(
            side_effect=RuntimeError("API error"),
        )

        registry = ToolRegistry()
        agent = ToolAwareAgent(
            system_instruction="Help.",
            tool_registry=registry,
            model="gemini-2.5-pro",
            name="test-agent",
            client=client,
        )
        result = await agent.async_execute("Do something")

        assert result.success is False
        assert "API error" in result.content

    @patch("vaig.core.client.types.Part")
    @patch("vaig.agents.mixins.types.Content")
    @patch("vaig.agents.mixins.types.Part")
    async def test_function_call_then_text(
        self,
        mock_part: MagicMock,
        mock_content: MagicMock,
        mock_client_part: MagicMock,
    ) -> None:
        """Model returns FC first, then text — two async iterations."""
        from vaig.agents.tool_aware import ToolAwareAgent

        client = _make_mock_client()

        search_tool = ToolDef(
            name="search",
            description="Search",
            execute=lambda **kw: ToolResult(output="found it"),
        )

        fc_result = _make_fc_result(
            function_calls=[{"name": "search", "args": {"query": "test"}}],
        )
        text_result = _make_text_result(text="Search complete.")
        client.async_generate_with_tools = AsyncMock(
            side_effect=[fc_result, text_result],
        )

        registry = ToolRegistry()
        registry.register(search_tool)
        agent = ToolAwareAgent(
            system_instruction="Search things.",
            tool_registry=registry,
            model="gemini-2.5-pro",
            name="searcher",
            client=client,
        )
        result = await agent.async_execute("Find something")

        assert result.success is True
        assert result.metadata["iterations"] == 2
        assert len(result.metadata["tools_executed"]) == 1

    async def test_conversation_history_tracked(self) -> None:
        """async_execute tracks conversation history."""
        from vaig.agents.tool_aware import ToolAwareAgent

        client = _make_mock_client()
        client.async_generate_with_tools = AsyncMock(
            return_value=_make_text_result(text="Done"),
        )

        registry = ToolRegistry()
        agent = ToolAwareAgent(
            system_instruction="Help.",
            tool_registry=registry,
            model="gemini-2.5-pro",
            name="agent",
            client=client,
        )
        await agent.async_execute("Task")

        assert len(agent.conversation_history) == 2
        assert agent.conversation_history[0].role == "user"
        assert agent.conversation_history[1].role == "agent"

    async def test_async_execute_stream_fallback(self) -> None:
        """async_execute_stream yields result of async_execute."""
        from vaig.agents.tool_aware import ToolAwareAgent

        client = _make_mock_client()
        client.async_generate_with_tools = AsyncMock(
            return_value=_make_text_result(text="Result"),
        )

        registry = ToolRegistry()
        agent = ToolAwareAgent(
            system_instruction="Help.",
            tool_registry=registry,
            model="gemini-2.5-pro",
            name="agent",
            client=client,
        )
        chunks = []
        async for chunk in agent.async_execute_stream("Task"):
            chunks.append(chunk)

        assert chunks == ["Result"]

    @patch("vaig.core.client.types.Part")
    @patch("vaig.agents.mixins.types.Content")
    @patch("vaig.agents.mixins.types.Part")
    async def test_max_iterations_raises(
        self,
        mock_part: MagicMock,
        mock_content: MagicMock,
        mock_client_part: MagicMock,
    ) -> None:
        """Exceeding max iterations raises MaxIterationsError."""
        from vaig.agents.tool_aware import ToolAwareAgent

        client = _make_mock_client()

        tool = ToolDef(
            name="action",
            description="Do",
            execute=lambda **kw: ToolResult(output="ok"),
        )

        fc_result = _make_fc_result(
            function_calls=[{"name": "action", "args": {}}],
        )
        client.async_generate_with_tools = AsyncMock(return_value=fc_result)

        registry = ToolRegistry()
        registry.register(tool)
        agent = ToolAwareAgent(
            system_instruction="Help.",
            tool_registry=registry,
            model="gemini-2.5-pro",
            name="agent",
            client=client,
            max_iterations=2,
        )

        with pytest.raises(MaxIterationsError):
            await agent.async_execute("Loop forever")

    async def test_with_context(self) -> None:
        """Context is included in the prompt."""
        from vaig.agents.tool_aware import ToolAwareAgent

        client = _make_mock_client()
        client.async_generate_with_tools = AsyncMock(
            return_value=_make_text_result(text="ok"),
        )

        registry = ToolRegistry()
        agent = ToolAwareAgent(
            system_instruction="Help.",
            tool_registry=registry,
            model="gemini-2.5-pro",
            name="agent",
            client=client,
        )
        await agent.async_execute("Do this", context="Prior results")

        call_args = client.async_generate_with_tools.call_args
        prompt = call_args[0][0]
        assert "## Context" in prompt
        assert "Prior results" in prompt


# ===========================================================================
# TestSyncMethodsUnchanged
# ===========================================================================


class TestSyncMethodsUnchanged:
    """Verify that adding async methods did NOT alter sync behavior."""

    async def test_specialist_sync_still_works(self) -> None:
        """SpecialistAgent.execute() still works after async additions."""
        client = _make_mock_client()
        client.generate = MagicMock(
            return_value=_make_generation_result(text="Sync result"),
        )

        config = AgentConfig(
            name="agent", role="helper", system_instruction="Help.",
        )
        agent = SpecialistAgent(config, client)
        result = agent.execute("Task")

        assert result.success is True
        assert result.content == "Sync result"
        client.generate.assert_called_once()

    @patch("vaig.agents.coding.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding.create_file_tools", return_value=[])
    async def test_coding_sync_still_works(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """CodingAgent.execute() still works after async additions."""
        from vaig.agents.coding import CodingAgent

        client = _make_mock_client()
        client.generate_with_tools = MagicMock(
            return_value=_make_text_result(text="Sync done"),
        )

        agent = CodingAgent(client, _make_coding_config())
        result = agent.execute("Fix bug")

        assert result.success is True
        assert result.content == "Sync done"
        client.generate_with_tools.assert_called_once()

    async def test_tool_aware_sync_still_works(self) -> None:
        """ToolAwareAgent.execute() still works after async additions."""
        from vaig.agents.tool_aware import ToolAwareAgent

        client = _make_mock_client()
        client.generate_with_tools = MagicMock(
            return_value=_make_text_result(text="Sync result"),
        )

        registry = ToolRegistry()
        agent = ToolAwareAgent(
            system_instruction="Help.",
            tool_registry=registry,
            model="gemini-2.5-pro",
            name="agent",
            client=client,
        )
        result = agent.execute("Task")

        assert result.success is True
        assert result.content == "Sync result"

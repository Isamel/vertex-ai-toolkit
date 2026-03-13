"""Tests for ToolAwareAgent — init, from_config_dict, execute, streaming, error handling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vaig.agents.base import AgentConfig, AgentResult, AgentRole
from vaig.agents.tool_aware import ToolAwareAgent
from vaig.core.exceptions import MaxIterationsError
from vaig.tools.base import ToolDef, ToolParam, ToolRegistry, ToolResult


# ── Helpers ──────────────────────────────────────────────────


def _make_mock_client() -> MagicMock:
    """Create a mock GeminiClient with sensible defaults."""
    client = MagicMock()
    client.current_model = "gemini-2.5-pro"
    return client


def _make_tool_registry(*tools: ToolDef) -> ToolRegistry:
    """Create a ToolRegistry with optional pre-registered tools."""
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _make_agent(
    *,
    name: str = "test-agent",
    system_instruction: str = "You are a test agent.",
    max_iterations: int = 15,
    extra_tools: list[ToolDef] | None = None,
) -> tuple[ToolAwareAgent, MagicMock]:
    """Create a ToolAwareAgent with a mock client and empty/custom registry."""
    client = _make_mock_client()
    registry = _make_tool_registry(*(extra_tools or []))
    agent = ToolAwareAgent(
        system_instruction=system_instruction,
        tool_registry=registry,
        model="gemini-2.5-pro",
        name=name,
        client=client,
        max_iterations=max_iterations,
    )
    return agent, client


def _mock_text_response(
    text: str = "All good.",
    model: str = "gemini-2.5-pro",
    usage: dict | None = None,
) -> MagicMock:
    """Create a mock ToolCallResult with a text response (no function calls)."""
    result = MagicMock()
    result.function_calls = []
    result.text = text
    result.model = model
    result.finish_reason = "STOP"
    result.usage = usage or {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }
    return result


def _mock_function_call_response(
    calls: list[dict],
    model: str = "gemini-2.5-pro",
) -> MagicMock:
    """Create a mock ToolCallResult with function calls (no text)."""
    result = MagicMock()
    result.function_calls = calls
    result.text = ""
    result.model = model
    result.finish_reason = "STOP"
    result.usage = {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75}
    return result


# ── TestInit ─────────────────────────────────────────────────


class TestInit:
    """Tests for ToolAwareAgent construction."""

    def test_basic_initialization(self) -> None:
        agent, _ = _make_agent(name="my-agent", system_instruction="Be helpful.")

        assert agent.name == "my-agent"
        assert agent._config.system_instruction == "Be helpful."
        assert agent._config.model == "gemini-2.5-pro"
        assert agent._max_iterations == 15

    def test_custom_max_iterations(self) -> None:
        agent, _ = _make_agent(max_iterations=5)

        assert agent.max_iterations == 5

    def test_role_is_specialist(self) -> None:
        agent, _ = _make_agent()

        assert agent._config.role == AgentRole.SPECIALIST

    def test_tool_registry_is_stored(self) -> None:
        agent, _ = _make_agent()

        assert isinstance(agent.tool_registry, ToolRegistry)

    def test_tool_registry_contains_provided_tools(self) -> None:
        tool = ToolDef(
            name="ping",
            description="Ping a host",
            execute=lambda: ToolResult(output="pong"),
        )
        agent, _ = _make_agent(extra_tools=[tool])

        assert agent.tool_registry.get("ping") is not None

    def test_does_not_import_infra_agent(self) -> None:
        """ToolAwareAgent must NOT depend on InfraAgent."""
        import ast
        import inspect
        import vaig.agents.tool_aware as module

        source = inspect.getsource(module)
        tree = ast.parse(source)

        # Check that no import statement references infra_agent or InfraAgent
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "infra_agent" not in alias.name, f"Found import of {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                assert "infra_agent" not in module_name, f"Found import from {module_name}"
                for alias in node.names:
                    assert alias.name != "InfraAgent", f"Found import of InfraAgent"


# ── TestFromConfigDict ───────────────────────────────────────


class TestFromConfigDict:
    """Tests for the from_config_dict class method."""

    def test_creates_agent_from_dict(self) -> None:
        client = _make_mock_client()
        registry = _make_tool_registry()
        config = {
            "name": "analyzer",
            "role": "analyst",
            "system_instruction": "You analyze data.",
        }

        agent = ToolAwareAgent.from_config_dict(config, "gemini-2.5-flash", registry, client)

        assert agent.name == "analyzer"
        assert agent._config.system_instruction == "You analyze data."
        assert agent._config.model == "gemini-2.5-flash"
        assert agent.max_iterations == 15  # default

    def test_respects_optional_max_iterations(self) -> None:
        client = _make_mock_client()
        registry = _make_tool_registry()
        config = {
            "name": "quick",
            "role": "worker",
            "system_instruction": "Work fast.",
            "max_iterations": 3,
        }

        agent = ToolAwareAgent.from_config_dict(config, "gemini-2.5-pro", registry, client)

        assert agent.max_iterations == 3

    def test_respects_optional_temperature(self) -> None:
        client = _make_mock_client()
        registry = _make_tool_registry()
        config = {
            "name": "precise",
            "role": "analyzer",
            "system_instruction": "Be precise.",
            "temperature": 0.1,
        }

        agent = ToolAwareAgent.from_config_dict(config, "gemini-2.5-pro", registry, client)

        assert agent._config.temperature == 0.1

    def test_missing_name_raises_key_error(self) -> None:
        client = _make_mock_client()
        registry = _make_tool_registry()

        with pytest.raises(KeyError, match="name"):
            ToolAwareAgent.from_config_dict(
                {"system_instruction": "hello"},
                "gemini-2.5-pro",
                registry,
                client,
            )

    def test_missing_system_prompt_and_instruction_uses_empty_string(self) -> None:
        """When neither system_prompt nor system_instruction is present, defaults to empty string."""
        client = _make_mock_client()
        registry = _make_tool_registry()

        agent = ToolAwareAgent.from_config_dict(
            {"name": "no-prompt", "role": "Worker"},
            "gemini-2.5-pro",
            registry,
            client,
        )
        assert agent._config.system_instruction == ""


# ── TestExecuteSimpleQuery ───────────────────────────────────


class TestExecuteSimpleQuery:
    """Tests for execute with a simple text response (no tool calls)."""

    def test_returns_agent_result(self) -> None:
        agent, client = _make_agent()
        client.generate_with_tools.return_value = _mock_text_response("Everything looks fine.")

        result = agent.execute("Status check")

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert "Everything looks fine" in result.content
        assert result.agent_name == "test-agent"

    def test_metadata_includes_model_info(self) -> None:
        agent, client = _make_agent()
        client.generate_with_tools.return_value = _mock_text_response()

        result = agent.execute("Check something")

        assert result.metadata["model"] == "gemini-2.5-pro"
        assert result.metadata["finish_reason"] == "STOP"
        assert result.metadata["iterations"] == 1

    def test_usage_tokens_populated(self) -> None:
        agent, client = _make_agent()
        client.generate_with_tools.return_value = _mock_text_response(
            usage={"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280},
        )

        result = agent.execute("Query")

        assert result.usage["total_tokens"] == 280


# ── TestExecuteWithToolCalls ─────────────────────────────────


class TestExecuteWithToolCalls:
    """Tests for execute with function calls and tool execution."""

    def test_executes_tool_and_returns_final_text(self) -> None:
        tool = ToolDef(
            name="get_status",
            description="Get system status",
            execute=lambda: ToolResult(output="status: healthy"),
        )
        agent, client = _make_agent(extra_tools=[tool])

        # First call: model requests a function call
        fc_response = _mock_function_call_response(
            [{"name": "get_status", "args": {}}],
        )
        # Second call: model returns text after seeing tool result
        text_response = _mock_text_response("System is healthy based on status check.")

        client.generate_with_tools.side_effect = [fc_response, text_response]

        result = agent.execute("How is the system?")

        assert result.success is True
        assert "healthy" in result.content
        assert result.metadata["iterations"] == 2
        assert len(result.metadata["tools_executed"]) == 1
        assert result.metadata["tools_executed"][0]["name"] == "get_status"

    def test_multiple_tool_calls_in_sequence(self) -> None:
        tool_a = ToolDef(
            name="tool_a",
            description="Tool A",
            execute=lambda: ToolResult(output="result_a"),
        )
        tool_b = ToolDef(
            name="tool_b",
            description="Tool B",
            execute=lambda: ToolResult(output="result_b"),
        )
        agent, client = _make_agent(extra_tools=[tool_a, tool_b])

        # Iteration 1: call tool_a
        fc1 = _mock_function_call_response([{"name": "tool_a", "args": {}}])
        # Iteration 2: call tool_b
        fc2 = _mock_function_call_response([{"name": "tool_b", "args": {}}])
        # Iteration 3: final text
        text = _mock_text_response("Done with both tools.")

        client.generate_with_tools.side_effect = [fc1, fc2, text]

        result = agent.execute("Run both")

        assert result.success is True
        assert result.metadata["iterations"] == 3
        assert len(result.metadata["tools_executed"]) == 2


# ── TestExecuteWithUpstreamContext ───────────────────────────


class TestExecuteWithUpstreamContext:
    """Tests that upstream context is correctly prepended to the prompt."""

    def test_context_prepended_to_prompt(self) -> None:
        agent, client = _make_agent()
        client.generate_with_tools.return_value = _mock_text_response("Analyzed.")

        result = agent.execute("Analyze this", context="Previous agent found 3 errors.")

        assert result.success is True
        # Verify the prompt sent to generate_with_tools includes context
        call_args = client.generate_with_tools.call_args
        prompt_arg = call_args[0][0] if call_args[0] else call_args[1].get("prompt", "")
        if isinstance(prompt_arg, str):
            assert "Previous agent found 3 errors" in prompt_arg
            assert "Analyze this" in prompt_arg

    def test_no_context_passes_raw_prompt(self) -> None:
        agent, client = _make_agent()
        client.generate_with_tools.return_value = _mock_text_response()

        agent.execute("Just a question")

        call_args = client.generate_with_tools.call_args
        prompt_arg = call_args[0][0] if call_args[0] else call_args[1].get("prompt", "")
        if isinstance(prompt_arg, str):
            assert "## Context" not in prompt_arg


# ── TestExecuteMaxIterations ─────────────────────────────────


class TestExecuteMaxIterations:
    """Tests that the agent stops after max_iterations."""

    def test_raises_max_iterations_error(self) -> None:
        agent, client = _make_agent(max_iterations=2)

        # Always return function calls -- never stops voluntarily
        fc = _mock_function_call_response([{"name": "unknown_tool", "args": {}}])
        client.generate_with_tools.return_value = fc

        with pytest.raises(MaxIterationsError):
            agent.execute("Loop forever")

    def test_max_iterations_enforced_at_configured_limit(self) -> None:
        agent, client = _make_agent(max_iterations=3)

        fc = _mock_function_call_response([{"name": "noop", "args": {}}])
        client.generate_with_tools.return_value = fc

        with pytest.raises(MaxIterationsError) as exc_info:
            agent.execute("Loop")

        assert exc_info.value.iterations == 3


# ── TestExecuteErrorHandling ────────────────────────────────


class TestExecuteErrorHandling:
    """Tests for error handling in execute."""

    def test_api_error_returns_failed_result(self) -> None:
        agent, client = _make_agent()
        client.generate_with_tools.side_effect = RuntimeError("API connection refused")

        result = agent.execute("Try something")

        assert result.success is False
        assert "Error during API call" in result.content
        assert "API connection refused" in result.content
        assert result.agent_name == "test-agent"

    def test_api_error_includes_error_metadata(self) -> None:
        agent, client = _make_agent()
        client.generate_with_tools.side_effect = ValueError("Invalid request")

        result = agent.execute("Bad query")

        assert result.metadata.get("error") == "Invalid request"

    def test_api_error_has_zero_usage(self) -> None:
        agent, client = _make_agent()
        client.generate_with_tools.side_effect = Exception("Boom")

        result = agent.execute("Explode")

        assert result.usage["total_tokens"] == 0


# ── TestExecuteStream ────────────────────────────────────────


class TestExecuteStream:
    """Tests for execute_stream (fallback to non-streaming)."""

    def test_yields_complete_result(self) -> None:
        agent, client = _make_agent()
        client.generate_with_tools.return_value = _mock_text_response("Streamed output.")

        chunks = list(agent.execute_stream("Stream me"))

        assert len(chunks) == 1
        assert "Streamed output" in chunks[0]

    def test_stream_with_context(self) -> None:
        agent, client = _make_agent()
        client.generate_with_tools.return_value = _mock_text_response("Contextual result.")

        chunks = list(agent.execute_stream("Analyze", context="Prior data here."))

        assert len(chunks) == 1
        assert "Contextual result" in chunks[0]

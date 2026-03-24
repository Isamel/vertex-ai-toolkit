"""Tests for agent_as_tool() factory — Tasks 1.1 through 1.5.

Covers:
- Factory creates a valid ToolDef with correct name (Task 1.1, 1.2)
- ToolDef invocation calls agent.execute() with correct args (Task 1.2)
- Recursion depth guard at max_depth (Task 1.3)
- Error handling returns error string (Task 1.4)
- Self-injection prevention (Task 1.3)
- Tool works with state=None and state=PipelineState(...) (Task 1.2)
- Tool description extraction from system instruction (Task 1.1)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vaig.agents.base import AgentConfig, AgentResult
from vaig.core.models import PipelineState
from vaig.tools.agent_tool import (
    _DEFAULT_DESCRIPTION,
    _extract_description,
    _sanitize_tool_name,
    agent_as_tool,
)
from vaig.tools.base import ToolDef, ToolResult

# ── Helpers ──────────────────────────────────────────────────


def _make_mock_agent(
    name: str = "mesh-specialist",
    system_instruction: str = "You are a mesh specialist. Analyse mesh topology.",
    execute_result: AgentResult | None = None,
) -> MagicMock:
    """Build a mock BaseAgent with controllable execute() return value."""
    agent = MagicMock()
    agent.name = name
    config = MagicMock(spec=AgentConfig)
    config.system_instruction = system_instruction
    config.name = name
    agent.config = config
    # Default result: success with some content
    default_result = AgentResult(
        agent_name=name,
        content="Mesh topology looks healthy.",
        success=True,
        usage={"total_tokens": 100},
    )
    agent.execute.return_value = execute_result or default_result
    return agent


def _call_tool(tool: ToolDef, query: str = "What is the status?") -> ToolResult:
    """Invoke the tool's execute callable with a query kwarg."""
    return tool.execute(query=query)


# ── TestSanitizeToolName ──────────────────────────────────────


class TestSanitizeToolName:
    """Tests for the _sanitize_tool_name helper."""

    def test_hyphen_becomes_underscore(self) -> None:
        assert _sanitize_tool_name("mesh-specialist") == "ask_mesh_specialist"

    def test_space_becomes_underscore(self) -> None:
        assert _sanitize_tool_name("my agent") == "ask_my_agent"

    def test_dots_become_underscore(self) -> None:
        assert _sanitize_tool_name("agent.v2.0") == "ask_agent_v2_0"

    def test_consecutive_separators_collapsed(self) -> None:
        assert _sanitize_tool_name("a--b") == "ask_a_b"

    def test_plain_name_unchanged(self) -> None:
        assert _sanitize_tool_name("analyzer") == "ask_analyzer"

    def test_mixed_characters(self) -> None:
        result = _sanitize_tool_name("My Agent 2.0")
        assert result.startswith("ask_")
        assert " " not in result
        assert "." not in result


# ── TestExtractDescription ────────────────────────────────────


class TestExtractDescription:
    """Tests for the _extract_description helper."""

    def test_first_sentence_extracted(self) -> None:
        agent = _make_mock_agent(
            system_instruction="You are a mesh specialist. Analyse mesh topology."
        )
        desc = _extract_description(agent)
        assert desc == "You are a mesh specialist"

    def test_exclamation_marks_split(self) -> None:
        agent = _make_mock_agent(system_instruction="Be precise! Always verify data.")
        desc = _extract_description(agent)
        assert desc == "Be precise"

    def test_question_mark_split(self) -> None:
        agent = _make_mock_agent(system_instruction="Is this working? Check it.")
        desc = _extract_description(agent)
        assert desc == "Is this working"

    def test_empty_instruction_returns_default(self) -> None:
        agent = _make_mock_agent(system_instruction="")
        desc = _extract_description(agent)
        assert desc == _DEFAULT_DESCRIPTION

    def test_whitespace_only_returns_default(self) -> None:
        agent = _make_mock_agent(system_instruction="   ")
        desc = _extract_description(agent)
        assert desc == _DEFAULT_DESCRIPTION

    def test_no_punctuation_returns_full_instruction(self) -> None:
        agent = _make_mock_agent(system_instruction="Short instruction")
        desc = _extract_description(agent)
        assert desc == "Short instruction"


# ── TestFactoryCreatesToolDef ─────────────────────────────────


class TestFactoryCreatesToolDef:
    """Task 1.1 & 1.2 — factory creates valid ToolDef with correct structure."""

    def test_returns_tool_def_instance(self) -> None:
        agent = _make_mock_agent()
        tool = agent_as_tool(agent)
        assert isinstance(tool, ToolDef)

    def test_tool_name_is_ask_prefixed(self) -> None:
        agent = _make_mock_agent(name="mesh-specialist")
        tool = agent_as_tool(agent)
        assert tool.name == "ask_mesh_specialist"

    def test_tool_name_sanitizes_special_chars(self) -> None:
        agent = _make_mock_agent(name="my.agent-v2")
        tool = agent_as_tool(agent)
        assert tool.name == "ask_my_agent_v2"

    def test_tool_description_from_system_instruction(self) -> None:
        agent = _make_mock_agent(
            system_instruction="You are a mesh specialist. More detail here."
        )
        tool = agent_as_tool(agent)
        assert tool.description == "You are a mesh specialist"

    def test_tool_description_default_when_no_instruction(self) -> None:
        agent = _make_mock_agent(system_instruction="")
        tool = agent_as_tool(agent)
        assert tool.description == _DEFAULT_DESCRIPTION

    def test_tool_has_single_query_parameter(self) -> None:
        agent = _make_mock_agent()
        tool = agent_as_tool(agent)
        assert len(tool.parameters) == 1
        param = tool.parameters[0]
        assert param.name == "query"
        assert param.type == "string"
        assert param.required is True

    def test_tool_not_cacheable(self) -> None:
        """Sub-agent tools must not be cached — each call may have different context."""
        agent = _make_mock_agent()
        tool = agent_as_tool(agent)
        assert tool.cacheable is False

    def test_execute_callable_is_set(self) -> None:
        agent = _make_mock_agent()
        tool = agent_as_tool(agent)
        assert callable(tool.execute)


# ── TestToolDefInvocation ─────────────────────────────────────


class TestToolDefInvocation:
    """Task 1.2 — ToolDef invocation calls agent.execute() with correct args."""

    def test_query_forwarded_to_agent_execute(self) -> None:
        agent = _make_mock_agent()
        tool = agent_as_tool(agent)

        _call_tool(tool, query="How many replicas are running?")

        agent.execute.assert_called_once()
        call_kwargs = agent.execute.call_args
        # execute(query, state=state)
        pos_arg = call_kwargs[0][0] if call_kwargs[0] else call_kwargs[1].get("prompt", "")
        assert pos_arg == "How many replicas are running?"

    def test_returns_tool_result_with_content(self) -> None:
        agent = _make_mock_agent()
        tool = agent_as_tool(agent)

        result = _call_tool(tool, query="Check mesh status")

        assert isinstance(result, ToolResult)
        assert result.output == "Mesh topology looks healthy."
        assert result.error is False

    def test_state_none_forwarded_to_execute(self) -> None:
        agent = _make_mock_agent()
        tool = agent_as_tool(agent, state=None)

        _call_tool(tool)

        call_kwargs = agent.execute.call_args
        assert call_kwargs[1].get("state") is None

    def test_state_pipeline_state_forwarded_to_execute(self) -> None:
        agent = _make_mock_agent()
        state = PipelineState(metrics={"region": "us-central1"})
        tool = agent_as_tool(agent, state=state)

        _call_tool(tool)

        call_kwargs = agent.execute.call_args
        assert call_kwargs[1].get("state") is state

    def test_agent_result_content_extracted(self) -> None:
        """AgentResult.content is used as ToolResult.output."""
        custom_result = AgentResult(
            agent_name="analyzer",
            content="Detailed analysis: no issues found.",
            success=True,
        )
        agent = _make_mock_agent(execute_result=custom_result)
        tool = agent_as_tool(agent)

        result = _call_tool(tool)

        assert result.output == "Detailed analysis: no issues found."


# ── TestRecursionDepthGuard ───────────────────────────────────


class TestRecursionDepthGuard:
    """Task 1.3 — recursion depth guard prevents infinite recursion."""

    def test_at_max_depth_returns_depth_exceeded_message(self) -> None:
        agent = _make_mock_agent()
        tool = agent_as_tool(agent, current_depth=2, max_depth=2)

        result = _call_tool(tool)

        assert result.error is True
        assert "Max recursion depth" in result.output
        assert "2" in result.output

    def test_at_max_depth_agent_execute_not_called(self) -> None:
        agent = _make_mock_agent()
        tool = agent_as_tool(agent, current_depth=2, max_depth=2)

        _call_tool(tool)

        agent.execute.assert_not_called()

    def test_below_max_depth_agent_execute_called(self) -> None:
        agent = _make_mock_agent()
        tool = agent_as_tool(agent, current_depth=1, max_depth=2)

        _call_tool(tool)

        agent.execute.assert_called_once()

    def test_depth_zero_allowed(self) -> None:
        agent = _make_mock_agent()
        tool = agent_as_tool(agent, current_depth=0, max_depth=2)

        result = _call_tool(tool)

        assert result.error is False

    def test_depth_exceeded_includes_agent_name_in_message(self) -> None:
        agent = _make_mock_agent(name="mesh-specialist")
        tool = agent_as_tool(agent, current_depth=5, max_depth=2)

        result = _call_tool(tool)

        assert "mesh_specialist" in result.output or "mesh-specialist" in result.output

    def test_default_max_depth_is_2(self) -> None:
        """Default max_depth=2 means depth=2 triggers the guard."""
        agent = _make_mock_agent()
        tool = agent_as_tool(agent, current_depth=2)

        result = _call_tool(tool)

        assert result.error is True
        agent.execute.assert_not_called()


# ── TestSelfInjectionPrevention ───────────────────────────────


class TestSelfInjectionPrevention:
    """Task 1.3 — self-injection prevention."""

    def test_self_injection_returns_error_tool_result(self) -> None:
        agent = _make_mock_agent(name="analyzer")
        tool = agent_as_tool(agent, caller_name="analyzer")

        result = _call_tool(tool)

        assert result.error is True
        assert "self-injection" in result.output.lower() or "cannot invoke itself" in result.output

    def test_self_injection_agent_execute_not_called(self) -> None:
        agent = _make_mock_agent(name="analyzer")
        tool = agent_as_tool(agent, caller_name="analyzer")

        _call_tool(tool)

        agent.execute.assert_not_called()

    def test_different_caller_name_allows_execution(self) -> None:
        agent = _make_mock_agent(name="mesh-specialist")
        tool = agent_as_tool(agent, caller_name="workload-gatherer")

        _call_tool(tool)

        agent.execute.assert_called_once()

    def test_empty_caller_name_allows_execution(self) -> None:
        """Empty caller_name disables self-injection check."""
        agent = _make_mock_agent(name="analyzer")
        tool = agent_as_tool(agent, caller_name="")

        _call_tool(tool)

        agent.execute.assert_called_once()

    def test_self_injection_tool_still_has_correct_name(self) -> None:
        agent = _make_mock_agent(name="my-agent")
        tool = agent_as_tool(agent, caller_name="my-agent")
        assert tool.name == "ask_my_agent"


# ── TestErrorHandling ─────────────────────────────────────────


class TestErrorHandling:
    """Task 1.4 — exceptions from agent.execute() are caught and returned as ToolResult."""

    def test_exception_returns_tool_result_with_error_true(self) -> None:
        agent = _make_mock_agent()
        agent.execute.side_effect = RuntimeError("Mesh API unreachable")
        tool = agent_as_tool(agent)

        result = _call_tool(tool)

        assert result.error is True

    def test_exception_message_included_in_output(self) -> None:
        agent = _make_mock_agent()
        agent.execute.side_effect = RuntimeError("Mesh API unreachable")
        tool = agent_as_tool(agent)

        result = _call_tool(tool)

        assert "Mesh API unreachable" in result.output

    def test_output_contains_agent_name(self) -> None:
        agent = _make_mock_agent(name="mesh-specialist")
        agent.execute.side_effect = ValueError("Bad input")
        tool = agent_as_tool(agent)

        result = _call_tool(tool)

        assert "mesh-specialist" in result.output

    def test_exception_format_is_user_friendly(self) -> None:
        agent = _make_mock_agent(name="analyzer")
        agent.execute.side_effect = ConnectionError("Connection refused")
        tool = agent_as_tool(agent)

        result = _call_tool(tool)

        # Should follow "Sub-agent '{name}' failed: {error}" pattern
        assert "Sub-agent" in result.output
        assert "failed" in result.output.lower()

    def test_exception_logged_as_warning(self) -> None:
        agent = _make_mock_agent()
        agent.execute.side_effect = RuntimeError("Boom")
        tool = agent_as_tool(agent)

        with patch("vaig.tools.agent_tool.logger") as mock_logger:
            _call_tool(tool)
            mock_logger.warning.assert_called_once()

    def test_arbitrary_exception_types_caught(self) -> None:
        """All Exception subclasses must be caught."""
        agent = _make_mock_agent()
        agent.execute.side_effect = KeyError("missing_key")
        tool = agent_as_tool(agent)

        # Must NOT raise
        result = _call_tool(tool)
        assert result.error is True

    def test_does_not_raise(self) -> None:
        """The ToolDef wrapper must NEVER propagate exceptions."""
        agent = _make_mock_agent()
        agent.execute.side_effect = Exception("Something unexpected")
        tool = agent_as_tool(agent)

        # Should not raise
        try:
            _call_tool(tool)
        except Exception as exc:
            pytest.fail(f"agent_as_tool raised unexpectedly: {exc}")


# ── TestPublicExport ──────────────────────────────────────────


class TestPublicExport:
    """Ensure agent_as_tool is exported from the tools package."""

    def test_importable_from_vaig_tools(self) -> None:
        from vaig.tools import agent_as_tool as _fn  # noqa: PLC0415

        assert callable(_fn)

    def test_agent_as_tool_in_all(self) -> None:
        import vaig.tools as pkg  # noqa: PLC0415

        assert "agent_as_tool" in pkg.__all__


# ── TestAgentResultFailureMappedToToolError ───────────────────


class TestAgentResultFailureMappedToToolError:
    """Issue 3 — AgentResult.success=False must map to ToolResult.error=True."""

    def test_success_false_returns_error_true(self) -> None:
        failed_result = AgentResult(
            agent_name="analyzer",
            content="Something went wrong inside the agent.",
            success=False,
        )
        agent = _make_mock_agent(execute_result=failed_result)
        tool = agent_as_tool(agent)

        result = _call_tool(tool)

        assert result.error is True

    def test_success_false_error_output_contains_failure_content(self) -> None:
        failed_result = AgentResult(
            agent_name="analyzer",
            content="Mesh API timed out.",
            success=False,
        )
        agent = _make_mock_agent(execute_result=failed_result)
        tool = agent_as_tool(agent)

        result = _call_tool(tool)

        assert "Mesh API timed out." in result.output

    def test_success_false_output_contains_agent_name(self) -> None:
        failed_result = AgentResult(
            agent_name="mesh-specialist",
            content="Unavailable.",
            success=False,
        )
        agent = _make_mock_agent(name="mesh-specialist", execute_result=failed_result)
        tool = agent_as_tool(agent)

        result = _call_tool(tool)

        assert "mesh-specialist" in result.output or "mesh_specialist" in result.output

    def test_success_true_returns_error_false(self) -> None:
        """Baseline: success=True must NOT produce an error ToolResult."""
        ok_result = AgentResult(
            agent_name="analyzer",
            content="All good.",
            success=True,
        )
        agent = _make_mock_agent(execute_result=ok_result)
        tool = agent_as_tool(agent)

        result = _call_tool(tool)

        assert result.error is False
        assert result.output == "All good."


# ── TestStateGetterLazyBinding ────────────────────────────────


class TestStateGetterLazyBinding:
    """Issue 1 — state_getter lazy binding: sub-agent receives up-to-date state."""

    def test_state_getter_called_at_invocation_time(self) -> None:
        """state_getter must be evaluated each time the tool is invoked."""
        state_holder: dict[str, PipelineState | None] = {"current": None}
        agent = _make_mock_agent()
        tool = agent_as_tool(agent, state_getter=lambda: state_holder["current"])

        # First call — state is None
        _call_tool(tool)
        first_call_state = agent.execute.call_args_list[0][1].get("state")
        assert first_call_state is None

        # Update state and call again
        new_state = PipelineState(metrics={"region": "eu-west-1"})
        state_holder["current"] = new_state
        _call_tool(tool)
        second_call_state = agent.execute.call_args_list[1][1].get("state")
        assert second_call_state is new_state

    def test_state_getter_takes_precedence_over_static_state(self) -> None:
        """When both state and state_getter are provided, state_getter wins."""
        static_state = PipelineState(metrics={"source": "static"})
        dynamic_state = PipelineState(metrics={"source": "dynamic"})
        agent = _make_mock_agent()
        tool = agent_as_tool(
            agent,
            state=static_state,
            state_getter=lambda: dynamic_state,
        )

        _call_tool(tool)

        call_state = agent.execute.call_args[1].get("state")
        assert call_state is dynamic_state
        assert call_state is not static_state

    def test_static_state_used_when_no_getter(self) -> None:
        """With no state_getter, the static state is forwarded unchanged."""
        static_state = PipelineState(metrics={"region": "us-east-1"})
        agent = _make_mock_agent()
        tool = agent_as_tool(agent, state=static_state)

        _call_tool(tool)

        call_state = agent.execute.call_args[1].get("state")
        assert call_state is static_state


# ── TestFactoryPattern ────────────────────────────────────────


class TestFactoryPattern:
    """Issue 4 — factory pattern creates a fresh agent instance per invocation."""

    def test_factory_called_per_invocation(self) -> None:
        """agent_factory must be called each time the tool is invoked."""
        call_count = {"n": 0}

        def _factory() -> MagicMock:
            call_count["n"] += 1
            return _make_mock_agent()

        tool = agent_as_tool(agent_factory=_factory, agent_name="mesh-specialist")

        _call_tool(tool)
        _call_tool(tool)
        _call_tool(tool)

        # Factory is called once for description at creation time, then once per invocation
        # (creation cost + 2 invocations = 3+)
        assert call_count["n"] >= 2  # at minimum 1 per real invocation

    def test_factory_instances_are_independent(self) -> None:
        """Each call gets a separate agent instance, not a shared one."""
        instances: list[MagicMock] = []

        def _factory() -> MagicMock:
            inst = _make_mock_agent()
            instances.append(inst)
            return inst

        tool = agent_as_tool(agent_factory=_factory, agent_name="mesh-specialist")

        # Trigger two actual execute calls
        _call_tool(tool)
        _call_tool(tool)

        # At least 2 execute calls must have been made, across separate instances
        total_execute_calls = sum(inst.execute.call_count for inst in instances)
        assert total_execute_calls >= 2

    def test_factory_with_state_getter_passes_state(self) -> None:
        """Factory + state_getter: the agent receives the getter's returned state."""
        dynamic_state = PipelineState(metrics={"env": "prod"})
        captured_agent: list[MagicMock] = []

        def _factory() -> MagicMock:
            inst = _make_mock_agent()
            captured_agent.append(inst)
            return inst

        tool = agent_as_tool(
            agent_factory=_factory,
            state_getter=lambda: dynamic_state,
            agent_name="mesh-specialist",
        )

        _call_tool(tool)

        # The last captured agent (the one used for execution) should have been called
        exec_agent = captured_agent[-1]
        call_state = exec_agent.execute.call_args[1].get("state")
        assert call_state is dynamic_state

    def test_requires_agent_or_factory(self) -> None:
        """ValueError raised when neither agent nor agent_factory is provided."""
        import pytest  # noqa: PLC0415

        with pytest.raises(ValueError, match="Either 'agent' or 'agent_factory'"):
            agent_as_tool()

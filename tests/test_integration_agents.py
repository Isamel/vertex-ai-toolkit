"""Integration tests for SpecialistAgent and Orchestrator with mocked GeminiClient."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, PropertyMock

import pytest

from vaig.agents.base import AgentConfig, AgentResult
from vaig.agents.specialist import SpecialistAgent
from vaig.agents.orchestrator import Orchestrator, OrchestratorResult
from vaig.core.client import ChatMessage, GenerationResult
from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase, SkillResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_generation_result(
    text: str = "Agent response",
    model: str = "gemini-2.5-pro",
    usage: dict[str, int] | None = None,
    finish_reason: str = "STOP",
) -> GenerationResult:
    """Helper to build a GenerationResult for mocked client returns."""
    return GenerationResult(
        text=text,
        model=model,
        usage=usage or {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        finish_reason=finish_reason,
    )


def _make_mock_client(
    generate_return: GenerationResult | None = None,
    stream_chunks: list[str] | None = None,
) -> MagicMock:
    """Create a MagicMock that behaves like GeminiClient."""
    client = MagicMock()
    client.generate.return_value = generate_return or _make_generation_result()
    client.generate_stream.return_value = iter(stream_chunks or ["chunk1", "chunk2", "chunk3"])
    return client


def _make_agent_config(
    name: str = "test-agent",
    role: str = "analyzer",
    system_instruction: str = "You analyze things.",
    model: str = "gemini-2.5-pro",
    temperature: float = 0.7,
    max_output_tokens: int = 8192,
) -> AgentConfig:
    return AgentConfig(
        name=name,
        role=role,
        system_instruction=system_instruction,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


class StubSkill(BaseSkill):
    """Concrete skill implementation for testing."""

    def __init__(
        self,
        name: str = "test_skill",
        agents: list[dict[str, Any]] | None = None,
    ) -> None:
        self._name = name
        self._agents = agents or [
            {
                "name": "analyzer",
                "role": "Log Analyzer",
                "system_instruction": "Analyze logs.",
                "model": "gemini-2.5-pro",
            },
            {
                "name": "reporter",
                "role": "Report Writer",
                "system_instruction": "Write reports.",
                "model": "gemini-2.5-flash",
            },
        ]

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name=self._name,
            display_name="Test Skill",
            description="A skill for testing.",
        )

    def get_system_instruction(self) -> str:
        return "You are a test skill."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return f"[{phase.value}] {user_input}"

    def get_agents_config(self) -> list[dict]:
        return self._agents


def _make_mock_settings() -> MagicMock:
    """Create a minimal mock Settings object."""
    settings = MagicMock()
    settings.models.default = "gemini-2.5-pro"
    return settings


# ===========================================================================
# SpecialistAgent tests
# ===========================================================================


class TestSpecialistAgentExecute:
    """Tests for SpecialistAgent.execute() and related methods."""

    def test_execute_calls_generate_with_correct_args(self) -> None:
        client = _make_mock_client()
        config = _make_agent_config()
        agent = SpecialistAgent(config, client)

        result = agent.execute("Analyze this log")

        client.generate.assert_called_once_with(
            "Analyze this log",
            system_instruction="You analyze things.",
            history=[],
            model_id="gemini-2.5-pro",
            temperature=0.7,
            max_output_tokens=8192,
        )
        assert result.success is True
        assert result.agent_name == "test-agent"
        assert result.content == "Agent response"
        assert result.usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        assert result.metadata["model"] == "gemini-2.5-pro"
        assert result.metadata["finish_reason"] == "STOP"

    def test_execute_with_context_builds_prompt_correctly(self) -> None:
        client = _make_mock_client()
        agent = SpecialistAgent(_make_agent_config(), client)

        agent.execute("Find errors", context="Log data here")

        call_args = client.generate.call_args
        prompt_sent = call_args[0][0]
        assert "## Context" in prompt_sent
        assert "Log data here" in prompt_sent
        assert "## Task" in prompt_sent
        assert "Find errors" in prompt_sent

    def test_execute_failure_returns_error_result(self) -> None:
        client = _make_mock_client()
        client.generate.side_effect = RuntimeError("API quota exceeded")
        agent = SpecialistAgent(_make_agent_config(), client)

        result = agent.execute("Do something")

        assert result.success is False
        assert "API quota exceeded" in result.content
        assert result.metadata["error"] == "API quota exceeded"

    def test_execute_tracks_conversation_history(self) -> None:
        client = _make_mock_client()
        agent = SpecialistAgent(_make_agent_config(), client)

        assert len(agent.conversation_history) == 0

        agent.execute("First prompt")

        assert len(agent.conversation_history) == 2
        assert agent.conversation_history[0].role == "user"
        assert agent.conversation_history[0].content == "First prompt"
        assert agent.conversation_history[1].role == "agent"
        assert agent.conversation_history[1].content == "Agent response"

    def test_execute_multi_turn_sends_history(self) -> None:
        """Second call includes prior conversation as ChatMessage history."""
        client = _make_mock_client()
        agent = SpecialistAgent(_make_agent_config(), client)

        agent.execute("First prompt")

        client.generate.return_value = _make_generation_result(text="Second response")
        agent.execute("Second prompt")

        second_call = client.generate.call_args_list[1]
        history = second_call.kwargs["history"]
        assert len(history) == 2
        assert isinstance(history[0], ChatMessage)
        assert history[0].role == "user"
        assert history[0].content == "First prompt"
        assert history[1].role == "model"
        assert history[1].content == "Agent response"

    def test_execute_stream_yields_chunks_and_records_history(self) -> None:
        client = _make_mock_client(stream_chunks=["Hello", " ", "World"])
        agent = SpecialistAgent(_make_agent_config(), client)

        chunks = list(agent.execute_stream("Stream this"))

        assert chunks == ["Hello", " ", "World"]
        client.generate_stream.assert_called_once()

        # Full accumulated response in conversation
        assert len(agent.conversation_history) == 2
        assert agent.conversation_history[1].role == "agent"
        assert agent.conversation_history[1].content == "Hello World"

    def test_execute_stream_failure_yields_error(self) -> None:
        client = _make_mock_client()
        client.generate_stream.side_effect = RuntimeError("Stream boom")
        agent = SpecialistAgent(_make_agent_config(), client)

        chunks = list(agent.execute_stream("Will fail"))

        assert len(chunks) == 1
        assert "[Error: Stream boom]" in chunks[0]

    def test_from_config_dict_factory(self) -> None:
        client = _make_mock_client()
        config_dict = {
            "name": "my_agent",
            "role": "Summarizer",
            "system_instruction": "Summarize everything.",
            "model": "gemini-2.5-flash",
            "temperature": 0.3,
            "max_output_tokens": 4096,
        }

        agent = SpecialistAgent.from_config_dict(config_dict, client)

        assert agent.name == "my_agent"
        assert agent.role == "Summarizer"
        assert agent.model == "gemini-2.5-flash"
        assert agent.config.temperature == 0.3
        assert agent.config.max_output_tokens == 4096
        assert agent.config.system_instruction == "Summarize everything."

    def test_from_config_dict_uses_defaults(self) -> None:
        client = _make_mock_client()
        config_dict = {
            "name": "minimal",
            "role": "worker",
            "system_instruction": "Work.",
        }

        agent = SpecialistAgent.from_config_dict(config_dict, client)

        assert agent.model == "gemini-2.5-pro"
        assert agent.config.temperature == 0.7
        assert agent.config.max_output_tokens == 8192

    def test_build_chat_history_converts_correctly(self) -> None:
        client = _make_mock_client()
        agent = SpecialistAgent(_make_agent_config(), client)

        # Manually add conversation to test _build_chat_history
        agent._add_to_conversation("user", "Hello")
        agent._add_to_conversation("agent", "Hi there")
        agent._add_to_conversation("user", "Thanks")

        history = agent._build_chat_history()

        assert len(history) == 3
        assert history[0].role == "user"
        assert history[0].content == "Hello"
        assert history[1].role == "model"  # agent → model
        assert history[1].content == "Hi there"
        assert history[2].role == "user"
        assert history[2].content == "Thanks"

    def test_reset_clears_history(self) -> None:
        client = _make_mock_client()
        agent = SpecialistAgent(_make_agent_config(), client)
        agent.execute("Something")
        assert len(agent.conversation_history) == 2

        agent.reset()

        assert len(agent.conversation_history) == 0


# ===========================================================================
# Orchestrator tests
# ===========================================================================


class TestOrchestrator:
    """Tests for Orchestrator execution strategies."""

    def test_create_agents_for_skill(self) -> None:
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = StubSkill()

        agents = orchestrator.create_agents_for_skill(skill)

        assert len(agents) == 2
        assert agents[0].name == "analyzer"
        assert agents[0].role == "Log Analyzer"
        assert agents[1].name == "reporter"
        assert agents[1].role == "Report Writer"
        assert orchestrator.list_agents() == ["analyzer", "reporter"]

    def test_execute_sequential_runs_agents_in_order(self) -> None:
        client = _make_mock_client()
        # Each call returns a distinct response
        client.generate.side_effect = [
            _make_generation_result(text="Analysis result"),
            _make_generation_result(text="Final report"),
        ]
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = StubSkill()

        result = orchestrator.execute_sequential(
            skill, SkillPhase.ANALYZE, "log data", "find errors"
        )

        assert result.success is True
        assert result.skill_name == "test_skill"
        assert result.phase == SkillPhase.ANALYZE
        assert len(result.agent_results) == 2
        assert result.agent_results[0].content == "Analysis result"
        assert result.agent_results[1].content == "Final report"
        # Synthesized output is last agent's output
        assert result.synthesized_output == "Final report"

    def test_execute_sequential_feeds_context_between_agents(self) -> None:
        client = _make_mock_client()
        client.generate.side_effect = [
            _make_generation_result(text="Step 1 output"),
            _make_generation_result(text="Step 2 output"),
        ]
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = StubSkill()

        orchestrator.execute_sequential(
            skill, SkillPhase.ANALYZE, "initial context", "task"
        )

        # SpecialistAgent._build_prompt bakes context into the prompt string
        # sent to client.generate as the first positional arg.
        # Second agent's prompt should contain previous analysis.
        second_call_prompt = client.generate.call_args_list[1][0][0]
        assert "Previous Analysis" in second_call_prompt
        assert "Step 1 output" in second_call_prompt

    def test_execute_sequential_stops_on_failure(self) -> None:
        client = _make_mock_client()
        client.generate.side_effect = [
            RuntimeError("Agent 1 failed"),
        ]
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = StubSkill()

        result = orchestrator.execute_sequential(
            skill, SkillPhase.ANALYZE, "", "task"
        )

        assert result.success is False
        # Only 1 agent ran (failed), second was never called
        assert len(result.agent_results) == 1
        assert result.agent_results[0].success is False
        assert client.generate.call_count == 1

    def test_execute_fanout_runs_all_agents_independently(self) -> None:
        client = _make_mock_client()
        client.generate.side_effect = [
            _make_generation_result(text="Agent A perspective"),
            _make_generation_result(text="Agent B perspective"),
        ]
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = StubSkill()

        result = orchestrator.execute_fanout(
            skill, SkillPhase.ANALYZE, "shared context", "analyze"
        )

        assert result.success is True
        assert len(result.agent_results) == 2
        # Both agents get the same context baked into their prompt
        for call in client.generate.call_args_list:
            prompt_sent = call[0][0]
            assert "shared context" in prompt_sent
        # Merged output contains both agent names
        assert "analyzer" in result.synthesized_output
        assert "reporter" in result.synthesized_output

    def test_execute_fanout_partial_failure(self) -> None:
        client = _make_mock_client()
        client.generate.side_effect = [
            RuntimeError("First agent boom"),
            _make_generation_result(text="Second agent OK"),
        ]
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = StubSkill()

        result = orchestrator.execute_fanout(
            skill, SkillPhase.ANALYZE, "", "go"
        )

        # success = any succeeded
        assert result.success is True
        assert len(result.agent_results) == 2
        assert result.agent_results[0].success is False
        assert result.agent_results[1].success is True

    def test_execute_fanout_all_fail(self) -> None:
        client = _make_mock_client()
        client.generate.side_effect = [
            RuntimeError("Fail A"),
            RuntimeError("Fail B"),
        ]
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = StubSkill()

        result = orchestrator.execute_fanout(
            skill, SkillPhase.ANALYZE, "", "go"
        )

        assert result.success is False

    def test_execute_single_returns_agent_result(self) -> None:
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        result = orchestrator.execute_single("Hello there")

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert result.content == "Agent response"
        assert result.agent_name == "assistant"

    def test_execute_single_stream_returns_iterator(self) -> None:
        client = _make_mock_client(stream_chunks=["Hi", " ", "there"])
        orchestrator = Orchestrator(client, _make_mock_settings())

        stream = orchestrator.execute_single("Hello", stream=True)

        chunks = list(stream)
        assert chunks == ["Hi", " ", "there"]

    def test_execute_single_with_custom_params(self) -> None:
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        orchestrator.execute_single(
            "Test prompt",
            context="some context",
            system_instruction="Custom instruction",
            model_id="gemini-2.5-flash",
        )

        call_args = client.generate.call_args
        assert call_args.kwargs["system_instruction"] == "Custom instruction"
        assert call_args.kwargs["model_id"] == "gemini-2.5-flash"

    def test_execute_skill_phase_sequential_strategy(self) -> None:
        client = _make_mock_client()
        # Skill has 2 agents, so 2 generate calls
        client.generate.side_effect = [
            _make_generation_result(text="Step 1"),
            _make_generation_result(text="Step 2"),
        ]
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = StubSkill()

        result = orchestrator.execute_skill_phase(
            skill, SkillPhase.ANALYZE, "ctx", "input", strategy="sequential"
        )

        assert isinstance(result, SkillResult)
        assert result.success is True
        assert result.phase == SkillPhase.ANALYZE
        assert result.output == "Step 2"

    def test_execute_skill_phase_fanout_strategy(self) -> None:
        client = _make_mock_client()
        client.generate.side_effect = [
            _make_generation_result(text="A"),
            _make_generation_result(text="B"),
        ]
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = StubSkill()

        result = orchestrator.execute_skill_phase(
            skill, SkillPhase.ANALYZE, "ctx", "input", strategy="fanout"
        )

        assert isinstance(result, SkillResult)
        assert result.success is True
        # Merged output from fanout
        assert "analyzer" in result.output
        assert "reporter" in result.output

    def test_reset_agents_clears_all_histories(self) -> None:
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = StubSkill()

        # Create agents and give them some history
        client.generate.side_effect = [
            _make_generation_result(text="resp1"),
            _make_generation_result(text="resp2"),
        ]
        orchestrator.execute_sequential(
            skill, SkillPhase.ANALYZE, "", "task"
        )

        for agent in orchestrator._agents.values():
            assert len(agent.conversation_history) > 0

        orchestrator.reset_agents()

        for agent in orchestrator._agents.values():
            assert len(agent.conversation_history) == 0

    def test_merge_agent_outputs_format(self) -> None:
        orchestrator = Orchestrator(_make_mock_client(), _make_mock_settings())

        results = [
            AgentResult(agent_name="agent_a", content="Output A", success=True),
            AgentResult(agent_name="agent_b", content="Output B", success=True),
            AgentResult(agent_name="agent_c", content="Error X", success=False),
        ]

        merged = orchestrator._merge_agent_outputs(results)

        assert "### agent_a" in merged
        assert "Output A" in merged
        assert "### agent_b" in merged
        assert "Output B" in merged
        assert "### agent_c (failed)" in merged
        assert "Error X" in merged
        assert "---" in merged

    def test_usage_accumulation(self) -> None:
        client = _make_mock_client()
        client.generate.side_effect = [
            _make_generation_result(
                text="r1",
                usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            ),
            _make_generation_result(
                text="r2",
                usage={"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
            ),
        ]
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = StubSkill()

        result = orchestrator.execute_sequential(
            skill, SkillPhase.ANALYZE, "", "task"
        )

        assert result.total_usage["prompt_tokens"] == 25
        assert result.total_usage["completion_tokens"] == 45
        assert result.total_usage["total_tokens"] == 70


# ===========================================================================
# OrchestratorResult tests
# ===========================================================================


class TestOrchestratorResult:
    def test_to_skill_result_conversion(self) -> None:
        orch_result = OrchestratorResult(
            skill_name="test_skill",
            phase=SkillPhase.ANALYZE,
            agent_results=[
                AgentResult(agent_name="a1", content="out1"),
                AgentResult(agent_name="a2", content="out2"),
            ],
            synthesized_output="Final output",
            success=True,
            total_usage={"prompt_tokens": 50, "total_tokens": 100},
        )

        skill_result = orch_result.to_skill_result()

        assert isinstance(skill_result, SkillResult)
        assert skill_result.phase == SkillPhase.ANALYZE
        assert skill_result.success is True
        assert skill_result.output == "Final output"
        assert skill_result.metadata["agents_used"] == ["a1", "a2"]
        assert skill_result.metadata["total_usage"]["total_tokens"] == 100

    def test_to_skill_result_failure(self) -> None:
        orch_result = OrchestratorResult(
            skill_name="broken",
            phase=SkillPhase.EXECUTE,
            success=False,
            synthesized_output="Error occurred",
        )

        skill_result = orch_result.to_skill_result()

        assert skill_result.success is False
        assert skill_result.phase == SkillPhase.EXECUTE
        assert skill_result.output == "Error occurred"

    def test_to_skill_result_empty_agents(self) -> None:
        orch_result = OrchestratorResult(
            skill_name="empty",
            phase=SkillPhase.REPORT,
        )

        skill_result = orch_result.to_skill_result()

        assert skill_result.metadata["agents_used"] == []
        assert skill_result.output == ""

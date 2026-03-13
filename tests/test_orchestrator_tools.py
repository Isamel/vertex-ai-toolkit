"""Tests for Orchestrator tool-aware agent integration (Phase 3).

Verifies that the orchestrator correctly creates mixed pipelines
(SpecialistAgent + ToolAwareAgent) and routes execution through them.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vaig.agents.base import AgentResult, BaseAgent
from vaig.agents.orchestrator import Orchestrator, OrchestratorResult
from vaig.agents.specialist import SpecialistAgent
from vaig.agents.tool_aware import ToolAwareAgent
from vaig.core.client import GenerationResult
from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.tools.base import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_generation_result(
    text: str = "Agent response",
    model: str = "gemini-2.5-pro",
    usage: dict[str, int] | None = None,
    finish_reason: str = "STOP",
) -> GenerationResult:
    return GenerationResult(
        text=text,
        model=model,
        usage=usage or {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        finish_reason=finish_reason,
    )


def _make_mock_client() -> MagicMock:
    client = MagicMock()
    client.generate.return_value = _make_generation_result()
    return client


def _make_mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.models.default = "gemini-2.5-pro"
    return settings


def _make_mock_registry() -> ToolRegistry:
    """Create a real (empty) ToolRegistry."""
    return ToolRegistry()


class StubToolSkill(BaseSkill):
    """Skill with configurable agent configs for testing mixed pipelines."""

    def __init__(self, agents: list[dict[str, Any]] | None = None) -> None:
        self._agents = agents or [
            {
                "name": "tool-agent",
                "role": "Infrastructure Worker",
                "system_instruction": "You manage infra.",
                "system_prompt": "You manage infra.",
                "model": "gemini-2.5-pro",
                "requires_tools": True,
            },
            {
                "name": "report-agent",
                "role": "Report Writer",
                "system_instruction": "Write reports.",
                "model": "gemini-2.5-flash",
                "requires_tools": False,
            },
        ]

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="tool_skill",
            display_name="Tool Skill",
            description="A skill for testing tool-aware agents.",
            requires_live_tools=True,
        )

    def get_system_instruction(self) -> str:
        return "You are a tool skill."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return f"[{phase.value}] {user_input}"

    def get_agents_config(self) -> list[dict]:
        return self._agents


# ===========================================================================
# Task 3.1 — create_agents_for_skill with tool_registry
# ===========================================================================


class TestCreateAgentsMixedPipeline:
    """Test that create_agents_for_skill creates ToolAwareAgent when appropriate."""

    def test_create_agents_mixed_pipeline(self) -> None:
        """Skill with 2 agents: one requires_tools=True, one False.

        When tool_registry is provided, the first should be a ToolAwareAgent
        and the second a SpecialistAgent.
        """
        client = _make_mock_client()
        registry = _make_mock_registry()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = StubToolSkill()

        agents = orchestrator.create_agents_for_skill(skill, registry)

        assert len(agents) == 2
        assert isinstance(agents[0], ToolAwareAgent)
        assert agents[0].name == "tool-agent"
        assert isinstance(agents[1], SpecialistAgent)
        assert agents[1].name == "report-agent"

    def test_create_agents_no_registry_falls_back(self) -> None:
        """Skill with requires_tools=True but no registry passed.

        Should fall back to SpecialistAgent for ALL agents (backward compat).
        """
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = StubToolSkill()

        agents = orchestrator.create_agents_for_skill(skill)

        assert len(agents) == 2
        assert all(isinstance(a, SpecialistAgent) for a in agents)

    def test_create_agents_backward_compat(self) -> None:
        """Skill without requires_tools field — all agents are SpecialistAgent."""
        client = _make_mock_client()
        registry = _make_mock_registry()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = StubToolSkill(
            agents=[
                {
                    "name": "plain-agent",
                    "role": "Worker",
                    "system_instruction": "Work.",
                    "model": "gemini-2.5-pro",
                    # No requires_tools key at all
                },
            ]
        )

        agents = orchestrator.create_agents_for_skill(skill, registry)

        assert len(agents) == 1
        assert isinstance(agents[0], SpecialistAgent)

    def test_create_agents_requires_tools_false(self) -> None:
        """Agent config with requires_tools=False — creates SpecialistAgent even with registry."""
        client = _make_mock_client()
        registry = _make_mock_registry()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = StubToolSkill(
            agents=[
                {
                    "name": "no-tools",
                    "role": "Worker",
                    "system_instruction": "Work.",
                    "system_prompt": "Work.",
                    "model": "gemini-2.5-pro",
                    "requires_tools": False,
                },
            ]
        )

        agents = orchestrator.create_agents_for_skill(skill, registry)

        assert len(agents) == 1
        assert isinstance(agents[0], SpecialistAgent)

    def test_create_agents_all_tool_aware(self) -> None:
        """All agents with requires_tools=True and registry provided."""
        client = _make_mock_client()
        registry = _make_mock_registry()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = StubToolSkill(
            agents=[
                {
                    "name": "tools-1",
                    "role": "Worker A",
                    "system_instruction": "A.",
                    "system_prompt": "A.",
                    "model": "gemini-2.5-pro",
                    "requires_tools": True,
                },
                {
                    "name": "tools-2",
                    "role": "Worker B",
                    "system_instruction": "B.",
                    "system_prompt": "B.",
                    "model": "gemini-2.5-pro",
                    "requires_tools": True,
                },
            ]
        )

        agents = orchestrator.create_agents_for_skill(skill, registry)

        assert len(agents) == 2
        assert all(isinstance(a, ToolAwareAgent) for a in agents)

    def test_create_agents_passes_registry_to_tool_aware(self) -> None:
        """Verify the ToolAwareAgent actually receives the registry."""
        client = _make_mock_client()
        registry = _make_mock_registry()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = StubToolSkill(
            agents=[
                {
                    "name": "tools-agent",
                    "role": "Worker",
                    "system_instruction": "Do work.",
                    "system_prompt": "Do work.",
                    "model": "gemini-2.5-pro",
                    "requires_tools": True,
                },
            ]
        )

        agents = orchestrator.create_agents_for_skill(skill, registry)

        assert isinstance(agents[0], ToolAwareAgent)
        assert agents[0].tool_registry is registry


# ===========================================================================
# Task 3.2/3.3 — execute_with_tools (sequential, fan-out, single)
# ===========================================================================


class TestExecuteWithToolsSequential:
    """Test execute_with_tools() with sequential strategy."""

    def test_sequential_executes_agents_in_order(self) -> None:
        """Mock 2 agents, verify sequential execution with context passing."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        # Create mock agents directly
        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "agent-1"
        agent1.role = "First"
        agent1.execute.return_value = AgentResult(
            agent_name="agent-1", content="First output", success=True,
            usage={"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        )

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "agent-2"
        agent2.role = "Second"
        agent2.execute.return_value = AgentResult(
            agent_name="agent-2", content="Final output", success=True,
            usage={"prompt_tokens": 8, "completion_tokens": 12, "total_tokens": 20},
        )

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = orchestrator.execute_with_tools(
                "do something", StubToolSkill(), registry, strategy="sequential",
            )

        assert result.success is True
        assert len(result.agent_results) == 2
        assert result.agent_results[0].content == "First output"
        assert result.agent_results[1].content == "Final output"
        assert result.synthesized_output == "Final output"

        # First agent gets empty context
        agent1.execute.assert_called_once_with("do something", context="")
        # Second agent gets context with previous output
        second_call_context = agent2.execute.call_args.kwargs["context"]
        assert "Previous Analysis" in second_call_context
        assert "First output" in second_call_context

    def test_sequential_stops_on_failure(self) -> None:
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "agent-1"
        agent1.role = "First"
        agent1.execute.return_value = AgentResult(
            agent_name="agent-1", content="Error", success=False, usage={},
        )

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "agent-2"
        agent2.role = "Second"

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = orchestrator.execute_with_tools(
                "do something", StubToolSkill(), registry,
            )

        assert result.success is False
        assert len(result.agent_results) == 1
        agent2.execute.assert_not_called()


class TestExecuteWithToolsFanout:
    """Test execute_with_tools() with fanout strategy."""

    def test_fanout_executes_all_agents_independently(self) -> None:
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "agent-1"
        agent1.execute.return_value = AgentResult(
            agent_name="agent-1", content="Perspective A", success=True,
            usage={"total_tokens": 10},
        )

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "agent-2"
        agent2.execute.return_value = AgentResult(
            agent_name="agent-2", content="Perspective B", success=True,
            usage={"total_tokens": 15},
        )

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = orchestrator.execute_with_tools(
                "analyze this", StubToolSkill(), registry, strategy="fanout",
            )

        assert result.success is True
        assert len(result.agent_results) == 2
        # Both agents get the same query with context threaded through
        agent1.execute.assert_called_once_with("analyze this", context="analyze this")
        agent2.execute.assert_called_once_with("analyze this", context="analyze this")
        # Merged output
        assert "agent-1" in result.synthesized_output
        assert "agent-2" in result.synthesized_output

    def test_fanout_partial_failure_still_succeeds(self) -> None:
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "agent-1"
        agent1.execute.return_value = AgentResult(
            agent_name="agent-1", content="Error", success=False, usage={},
        )

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "agent-2"
        agent2.execute.return_value = AgentResult(
            agent_name="agent-2", content="OK", success=True, usage={},
        )

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = orchestrator.execute_with_tools(
                "go", StubToolSkill(), registry, strategy="fanout",
            )

        assert result.success is True  # at least one succeeded


class TestExecuteWithToolsSingle:
    """Test execute_with_tools() with single strategy."""

    def test_single_uses_first_agent(self) -> None:
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "agent-1"
        agent1.execute.return_value = AgentResult(
            agent_name="agent-1", content="Solo output", success=True,
            usage={"total_tokens": 10},
        )

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "agent-2"

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = orchestrator.execute_with_tools(
                "query", StubToolSkill(), registry, strategy="single",
            )

        assert result.success is True
        assert result.synthesized_output == "Solo output"
        assert len(result.agent_results) == 1
        agent2.execute.assert_not_called()

    def test_single_no_agents_returns_failure(self) -> None:
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[]):
            result = orchestrator.execute_with_tools(
                "query", StubToolSkill(), registry, strategy="single",
            )

        assert result.success is False
        assert "No agents" in result.synthesized_output


# ===========================================================================
# Task 3.3 — backward compatibility of existing methods
# ===========================================================================


class TestBackwardCompatibility:
    """Ensure existing execute_sequential/execute_fanout still work unchanged."""

    def test_execute_sequential_unchanged(self) -> None:
        """Original execute_sequential (no tool_registry) creates only SpecialistAgents."""
        client = _make_mock_client()
        client.generate.side_effect = [
            _make_generation_result(text="Step 1"),
            _make_generation_result(text="Step 2"),
        ]
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = StubToolSkill()

        result = orchestrator.execute_sequential(
            skill, SkillPhase.ANALYZE, "ctx", "task",
        )

        assert result.success is True
        assert len(result.agent_results) == 2
        # All agents created by execute_sequential are SpecialistAgent
        # (it calls create_agents_for_skill without tool_registry)
        for agent in orchestrator._agents.values():
            assert isinstance(agent, SpecialistAgent)

    def test_execute_skill_phase_unchanged(self) -> None:
        """execute_skill_phase still works without changes."""
        client = _make_mock_client()
        client.generate.side_effect = [
            _make_generation_result(text="A"),
            _make_generation_result(text="B"),
        ]
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = StubToolSkill()

        result = orchestrator.execute_skill_phase(
            skill, SkillPhase.ANALYZE, "ctx", "input",
        )

        assert result.success is True
        assert result.output == "B"


# ===========================================================================
# Task 3.2 extra — execute_with_tools usage accumulation
# ===========================================================================


class TestExecuteWithToolsUsage:
    """Verify token usage accumulation in execute_with_tools."""

    def test_usage_accumulated_sequentially(self) -> None:
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "a1"
        agent1.role = "R1"
        agent1.execute.return_value = AgentResult(
            agent_name="a1", content="x", success=True,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "a2"
        agent2.role = "R2"
        agent2.execute.return_value = AgentResult(
            agent_name="a2", content="y", success=True,
            usage={"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
        )

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = orchestrator.execute_with_tools(
                "q", StubToolSkill(), registry,
            )

        assert result.total_usage["prompt_tokens"] == 25
        assert result.total_usage["completion_tokens"] == 45
        assert result.total_usage["total_tokens"] == 70

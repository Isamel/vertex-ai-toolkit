"""Tests for Orchestrator tool-aware agent integration (Phase 3).

Verifies that the orchestrator correctly creates mixed pipelines
(SpecialistAgent + ToolAwareAgent) and routes execution through them.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vaig.agents.base import AgentResult, BaseAgent
from vaig.agents.orchestrator import (
    DEFAULT_MIN_CONTENT_CHARS,
    EMPTY_MARKERS,
    GathererValidationResult,
    Orchestrator,
    OrchestratorResult,
)
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


# ===========================================================================
# Gatherer output validation + retry logic
# ===========================================================================


class TestValidateGathererOutput:
    """Test Orchestrator._validate_gatherer_output()."""

    def _make_orchestrator(self) -> Orchestrator:
        return Orchestrator(_make_mock_client(), _make_mock_settings())

    def test_complete_output_returns_empty_lists(self) -> None:
        """All required sections present with sufficient content → no issues."""
        orch = self._make_orchestrator()
        output = (
            "## Cluster Overview\n"
            "Nodes: 3, all healthy. CPU usage is at 45% across the cluster. Memory is stable.\n\n"
            "## Service Status\n"
            "| Deployment | Ready | Restarts |\n"
            "| app-frontend | 3/3 | 0 |\n"
            "| app-backend | 2/2 | 1 |\n\n"
            "## Events Timeline\n"
            "10:00 Normal pod started successfully on node-1\n"
            "10:05 Normal deployment scaled to 3 replicas\n"
        )
        required = ["Cluster Overview", "Service Status", "Events Timeline"]
        result = orch._validate_gatherer_output(output, required)
        assert result.missing_sections == []
        assert result.shallow_sections == []
        assert not result.needs_retry

    def test_missing_sections_detected(self) -> None:
        """Missing sections are returned in missing_sections list."""
        orch = self._make_orchestrator()
        output = (
            "## Cluster Overview\n"
            "Nodes: 3, all healthy. CPU usage is at 45% across the cluster. Memory is stable.\n"
        )
        required = ["Cluster Overview", "Service Status", "Events Timeline"]
        result = orch._validate_gatherer_output(output, required)
        assert "Service Status" in result.missing_sections
        assert "Events Timeline" in result.missing_sections
        assert "Cluster Overview" not in result.missing_sections
        assert result.needs_retry

    def test_case_insensitive_matching(self) -> None:
        """Section matching is case-insensitive."""
        orch = self._make_orchestrator()
        output = (
            "## CLUSTER OVERVIEW\n"
            "Nodes: 3, all healthy. CPU usage is at 45% across the cluster. Memory is stable.\n"
            "## service status\n"
            "All services running smoothly, 100% uptime across all deployments in the namespace.\n"
            "## events timeline\n"
            "10:00 Normal pod started, 10:05 scaling complete, 10:10 health check passed.\n"
        )
        required = ["Cluster Overview", "Service Status", "Events Timeline"]
        result = orch._validate_gatherer_output(output, required)
        assert result.missing_sections == []
        assert result.shallow_sections == []

    def test_all_sections_missing(self) -> None:
        """Output with none of the required sections returns all as missing."""
        orch = self._make_orchestrator()
        output = "Some random text with no structure."
        required = ["Cluster Overview", "Service Status", "Events Timeline"]
        result = orch._validate_gatherer_output(output, required)
        assert len(result.missing_sections) == 3
        assert result.needs_retry

    def test_empty_required_list(self) -> None:
        """Empty required list → nothing is missing or shallow."""
        orch = self._make_orchestrator()
        result = orch._validate_gatherer_output("anything", [])
        assert result.missing_sections == []
        assert result.shallow_sections == []
        assert not result.needs_retry


class TestGathererRetryLogic:
    """Test the retry-on-incomplete-gatherer-output logic in execute_with_tools."""

    def test_retry_triggered_on_incomplete_output(self) -> None:
        """When gatherer output is incomplete and skill defines required sections,
        the orchestrator retries the gatherer once."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        # First call returns incomplete output (missing 2 sections), retry returns complete
        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1.execute.side_effect = [
            AgentResult(
                agent_name="gatherer",
                content=(
                    "## Cluster Overview\n"
                    "Nodes: 3, all healthy. CPU at 45%. Memory stable across the cluster.\n"
                ),  # Missing Service Status, Events Timeline
                success=True,
                usage={"total_tokens": 100},
            ),
            AgentResult(
                agent_name="gatherer",
                content=(
                    "## Cluster Overview\n"
                    "Nodes: 3, all healthy. CPU at 45%. Memory stable across the cluster.\n"
                    "## Service Status\n"
                    "All services running. Frontend 3/3, Backend 2/2, DB 1/1. No crashloops.\n"
                    "## Events Timeline\n"
                    "10:00 Normal pod started. 10:05 Scaling complete. 10:10 Health checks pass.\n"
                ),
                success=True,
                usage={"total_tokens": 150},
            ),
        ]

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "reporter"
        agent2.role = "Reporter"
        agent2.execute.return_value = AgentResult(
            agent_name="reporter", content="Final report", success=True,
            usage={"total_tokens": 50},
        )

        # Skill with required sections
        skill = StubToolSkill()
        skill.get_required_output_sections = MagicMock(
            return_value=["Cluster Overview", "Service Status", "Events Timeline"],
        )

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = orchestrator.execute_with_tools("check health", skill, registry)

        assert result.success is True
        # Agent1 should have been called twice (original + retry)
        assert agent1.execute.call_count == 2
        # Agent1 should have been reset before retry
        agent1.reset.assert_called_once()
        # Retry prompt should mention missing sections with clean language
        retry_call = agent1.execute.call_args_list[1]
        retry_prompt = retry_call.args[0]
        assert "missing the following sections" in retry_prompt
        assert "Service Status" in retry_prompt
        assert "Events Timeline" in retry_prompt

    def test_no_retry_when_output_is_complete(self) -> None:
        """When gatherer output is complete with sufficient depth, no retry happens."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1.execute.return_value = AgentResult(
            agent_name="gatherer",
            content=(
                "## Cluster Overview\n"
                "Nodes: 3, all healthy. CPU at 45%. Memory stable across the cluster.\n"
                "## Service Status\n"
                "All services running. Frontend 3/3, Backend 2/2. No crashloops detected.\n"
                "## Events Timeline\n"
                "10:00 Normal pod started. 10:05 Scaling complete. Health checks passing.\n"
            ),
            success=True,
            usage={"total_tokens": 100},
        )

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "reporter"
        agent2.role = "Reporter"
        agent2.execute.return_value = AgentResult(
            agent_name="reporter", content="Report", success=True,
            usage={"total_tokens": 50},
        )

        skill = StubToolSkill()
        skill.get_required_output_sections = MagicMock(
            return_value=["Cluster Overview", "Service Status", "Events Timeline"],
        )

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = orchestrator.execute_with_tools("check health", skill, registry)

        assert result.success is True
        assert agent1.execute.call_count == 1
        agent1.reset.assert_not_called()

    def test_no_retry_when_skill_has_no_required_sections(self) -> None:
        """Skills without required sections skip validation entirely."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1.execute.return_value = AgentResult(
            agent_name="gatherer",
            content="Random incomplete output",
            success=True,
            usage={"total_tokens": 100},
        )

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "reporter"
        agent2.role = "Reporter"
        agent2.execute.return_value = AgentResult(
            agent_name="reporter", content="Report", success=True,
            usage={"total_tokens": 50},
        )

        # Default StubToolSkill doesn't override get_required_output_sections
        skill = StubToolSkill()

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = orchestrator.execute_with_tools("query", skill, registry)

        assert result.success is True
        assert agent1.execute.call_count == 1
        agent1.reset.assert_not_called()

    def test_retry_only_happens_once(self) -> None:
        """Even if retry output is still incomplete, no second retry."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        # Both calls return incomplete output
        agent1.execute.side_effect = [
            AgentResult(
                agent_name="gatherer", content="nothing useful",
                success=True, usage={"total_tokens": 50},
            ),
            AgentResult(
                agent_name="gatherer", content="still incomplete",
                success=True, usage={"total_tokens": 60},
            ),
        ]

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "reporter"
        agent2.role = "Reporter"
        agent2.execute.return_value = AgentResult(
            agent_name="reporter", content="Report", success=True,
            usage={"total_tokens": 50},
        )

        skill = StubToolSkill()
        skill.get_required_output_sections = MagicMock(
            return_value=["Cluster Overview", "Service Status"],
        )

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = orchestrator.execute_with_tools("check", skill, registry)

        # Exactly 2 calls: original + 1 retry (not more)
        assert agent1.execute.call_count == 2
        # Pipeline continues with retry result even though incomplete
        assert result.success is True
        assert agent2.execute.call_count == 1

    def test_retry_failure_stops_pipeline(self) -> None:
        """If the retry itself fails (success=False), the pipeline stops."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1.execute.side_effect = [
            AgentResult(
                agent_name="gatherer", content="no sections",
                success=True, usage={"total_tokens": 50},
            ),
            AgentResult(
                agent_name="gatherer", content="API error",
                success=False, usage={"total_tokens": 10},
            ),
        ]

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "reporter"
        agent2.role = "Reporter"

        skill = StubToolSkill()
        skill.get_required_output_sections = MagicMock(
            return_value=["Cluster Overview"],
        )

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = orchestrator.execute_with_tools("check", skill, registry)

        assert result.success is False
        agent2.execute.assert_not_called()


# ===========================================================================
# Retry prompt cleanliness — no warning markers leak into LLM prompts
# ===========================================================================


class TestBuildRetryPrompt:
    """Test Orchestrator._build_retry_prompt() produces clean prompts."""

    def _make_orchestrator(self) -> Orchestrator:
        return Orchestrator(_make_mock_client(), _make_mock_settings())

    def test_retry_prompt_contains_missing_sections(self) -> None:
        """The retry prompt lists the missing section names."""
        orch = self._make_orchestrator()
        prompt = orch._build_retry_prompt(
            "check health", ["Service Status", "Events Timeline"],
        )
        assert "Service Status" in prompt
        assert "Events Timeline" in prompt

    def test_retry_prompt_includes_original_query(self) -> None:
        """The retry prompt starts with the original user query."""
        orch = self._make_orchestrator()
        prompt = orch._build_retry_prompt("check health", ["Section A"])
        assert prompt.startswith("check health")

    def test_retry_prompt_no_bracket_markers(self) -> None:
        """No [WARNING], [ERROR], [IMPORTANT] or similar bracket-prefixed
        markers should appear in the retry prompt sent to the LLM."""
        orch = self._make_orchestrator()
        prompt = orch._build_retry_prompt(
            "analyze cluster",
            ["Cluster Overview", "Service Status", "Events Timeline"],
        )
        # These patterns should NEVER appear in LLM prompts
        assert "[WARNING]" not in prompt
        assert "[ERROR]" not in prompt
        assert "[IMPORTANT]" not in prompt
        assert "[CRITICAL]" not in prompt
        assert "[ALERT]" not in prompt

    def test_retry_prompt_no_imperative_warning_language(self) -> None:
        """The prompt should use neutral instructional language, not
        all-caps imperative words that LLMs tend to echo verbatim."""
        orch = self._make_orchestrator()
        prompt = orch._build_retry_prompt(
            "check pods",
            ["Cluster Overview", "Events Timeline"],
        )
        # These imperative markers from the old prompt should be gone
        assert "IMPORTANT:" not in prompt
        assert "MANDATORY" not in prompt
        assert "You MUST" not in prompt

    def test_retry_prompt_uses_neutral_instruction(self) -> None:
        """The prompt should contain neutral re-generation instructions."""
        orch = self._make_orchestrator()
        prompt = orch._build_retry_prompt("query", ["Section X"])
        assert "missing the following sections" in prompt
        assert "Please regenerate" in prompt

    def test_retry_prompt_single_missing_section(self) -> None:
        """Works correctly with a single missing section."""
        orch = self._make_orchestrator()
        prompt = orch._build_retry_prompt("query", ["Events Timeline"])
        assert "Events Timeline" in prompt
        assert "missing the following sections" in prompt


class TestRetryPromptCleanInPipeline:
    """Integration test: verify the retry prompt sent during execute_with_tools
    is free of warning markers that the LLM would echo."""

    def test_retry_prompt_clean_during_execution(self) -> None:
        """When a retry is triggered, the actual prompt sent to the agent
        must not contain any bracket-prefixed warning markers."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1.execute.side_effect = [
            AgentResult(
                agent_name="gatherer",
                content=(
                    "## Cluster Overview\n"
                    "Nodes: 3, all healthy. CPU at 45%. Memory stable across the cluster.\n"
                ),
                success=True,
                usage={"total_tokens": 100},
            ),
            AgentResult(
                agent_name="gatherer",
                content=(
                    "## Cluster Overview\n"
                    "Nodes: 3, all healthy. CPU at 45%. Memory stable across the cluster.\n"
                    "## Service Status\n"
                    "All services running. Frontend 3/3, Backend 2/2. No crashloops detected.\n"
                    "## Events Timeline\n"
                    "10:00 Normal pod started. 10:05 Scaling complete. Health checks passing.\n"
                ),
                success=True,
                usage={"total_tokens": 150},
            ),
        ]

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "reporter"
        agent2.role = "Reporter"
        agent2.execute.return_value = AgentResult(
            agent_name="reporter", content="Final report", success=True,
            usage={"total_tokens": 50},
        )

        skill = StubToolSkill()
        skill.get_required_output_sections = MagicMock(
            return_value=["Cluster Overview", "Service Status", "Events Timeline"],
        )

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            orchestrator.execute_with_tools("check health", skill, registry)

        # Verify the retry prompt was clean
        retry_call = agent1.execute.call_args_list[1]
        retry_prompt = retry_call.args[0]

        # No bracket-prefixed markers
        for marker in ("[WARNING]", "[ERROR]", "[IMPORTANT]", "[CRITICAL]", "[ALERT]"):
            assert marker not in retry_prompt, (
                f"LLM prompt contains '{marker}' which will be echoed into the report"
            )

        # No all-caps imperative language from the old prompt
        assert "IMPORTANT:" not in retry_prompt
        assert "MANDATORY" not in retry_prompt
        assert "You MUST" not in retry_prompt

        # The prompt DOES contain the expected clean language
        assert "missing the following sections" in retry_prompt
        assert "Service Status" in retry_prompt
        assert "Events Timeline" in retry_prompt


# ===========================================================================
# Content depth validation — shallow section detection
# ===========================================================================


class TestShallowSectionDetection:
    """Test that _validate_gatherer_output detects shallow sections."""

    def _make_orchestrator(self) -> Orchestrator:
        return Orchestrator(_make_mock_client(), _make_mock_settings())

    def test_empty_section_body_is_shallow(self) -> None:
        """Section header exists but body is empty → shallow."""
        orch = self._make_orchestrator()
        output = (
            "## Cluster Overview\n"
            "\n"
            "## Service Status\n"
            "All services running. Frontend 3/3, Backend 2/2. No crashloops detected.\n"
            "## Events Timeline\n"
            "10:00 Normal pod started. 10:05 Scaling complete. Health checks passing.\n"
        )
        required = ["Cluster Overview", "Service Status", "Events Timeline"]
        result = orch._validate_gatherer_output(output, required)
        assert "Cluster Overview" in result.shallow_sections
        assert "Service Status" not in result.shallow_sections
        assert "Events Timeline" not in result.shallow_sections
        assert result.needs_retry

    def test_na_body_is_shallow(self) -> None:
        """Section body is just 'N/A' → shallow."""
        orch = self._make_orchestrator()
        output = (
            "## Cluster Overview\n"
            "N/A\n"
            "## Service Status\n"
            "All services running. Frontend 3/3, Backend 2/2. No crashloops detected.\n"
            "## Events Timeline\n"
            "10:00 Normal pod started. 10:05 Scaling complete. Health checks passing.\n"
        )
        required = ["Cluster Overview", "Service Status", "Events Timeline"]
        result = orch._validate_gatherer_output(output, required)
        assert "Cluster Overview" in result.shallow_sections
        assert result.needs_retry

    def test_data_not_available_is_shallow(self) -> None:
        """Section body is 'Data not available' → shallow."""
        orch = self._make_orchestrator()
        output = (
            "## Cluster Overview\n"
            "Nodes: 3, all healthy. CPU at 45%. Memory stable across the cluster.\n"
            "## Service Status\n"
            "Data not available\n"
            "## Events Timeline\n"
            "10:00 Normal pod started. 10:05 Scaling complete. Health checks passing.\n"
        )
        required = ["Cluster Overview", "Service Status", "Events Timeline"]
        result = orch._validate_gatherer_output(output, required)
        assert "Service Status" in result.shallow_sections
        assert "Cluster Overview" not in result.shallow_sections

    def test_all_empty_markers_detected(self) -> None:
        """Every item in EMPTY_MARKERS is detected as shallow."""
        orch = self._make_orchestrator()
        for marker in EMPTY_MARKERS:
            output = f"## Test Section\n{marker}\n"
            result = orch._validate_gatherer_output(output, ["Test Section"])
            assert "Test Section" in result.shallow_sections, (
                f"Empty marker '{marker}' was not detected as shallow"
            )

    def test_short_content_is_shallow(self) -> None:
        """Section with fewer than min_content_chars is shallow."""
        orch = self._make_orchestrator()
        # "OK" is only 2 chars, well below the 50-char default
        output = (
            "## Cluster Overview\n"
            "OK\n"
            "## Service Status\n"
            "All services running. Frontend 3/3, Backend 2/2. No crashloops detected.\n"
            "## Events Timeline\n"
            "10:00 Normal pod started. 10:05 Scaling complete. Health checks passing.\n"
        )
        required = ["Cluster Overview", "Service Status", "Events Timeline"]
        result = orch._validate_gatherer_output(output, required)
        assert "Cluster Overview" in result.shallow_sections

    def test_custom_min_content_chars(self) -> None:
        """Custom min_content_chars threshold is respected."""
        orch = self._make_orchestrator()
        # 30 chars of content: "This is some moderate content."
        body_text = "This is some moderate content."
        output = f"## Cluster Overview\n{body_text}\n"
        required = ["Cluster Overview"]

        # With default (50), it should be shallow
        result_default = orch._validate_gatherer_output(output, required)
        assert "Cluster Overview" in result_default.shallow_sections

        # With threshold of 20, it should pass
        result_low = orch._validate_gatherer_output(
            output, required, min_content_chars=20,
        )
        assert "Cluster Overview" not in result_low.shallow_sections
        assert not result_low.needs_retry

    def test_sufficient_content_passes(self) -> None:
        """Section with adequate content (>= min chars) passes depth check."""
        orch = self._make_orchestrator()
        output = (
            "## Cluster Overview\n"
            "The cluster has 3 nodes running Kubernetes v1.28. CPU usage averages 45% across all nodes. "
            "Memory utilization is stable at 60%. No node pressure conditions detected.\n"
        )
        required = ["Cluster Overview"]
        result = orch._validate_gatherer_output(output, required)
        assert result.missing_sections == []
        assert result.shallow_sections == []
        assert not result.needs_retry

    def test_mixed_missing_and_shallow(self) -> None:
        """Output with both missing and shallow sections reports both."""
        orch = self._make_orchestrator()
        output = (
            "## Cluster Overview\n"
            "N/A\n"
            "## Service Status\n"
            "All services running. Frontend 3/3, Backend 2/2. No crashloops detected.\n"
            # Events Timeline is completely missing
        )
        required = ["Cluster Overview", "Service Status", "Events Timeline"]
        result = orch._validate_gatherer_output(output, required)
        assert "Cluster Overview" in result.shallow_sections
        assert "Events Timeline" in result.missing_sections
        assert "Service Status" not in result.missing_sections
        assert "Service Status" not in result.shallow_sections
        assert result.needs_retry

    def test_empty_marker_case_insensitive(self) -> None:
        """Empty marker matching is case-insensitive."""
        orch = self._make_orchestrator()
        output = "## Test Section\nNO DATA AVAILABLE\n"
        result = orch._validate_gatherer_output(output, ["Test Section"])
        assert "Test Section" in result.shallow_sections

    def test_section_with_only_whitespace_is_shallow(self) -> None:
        """Section body with only whitespace is shallow."""
        orch = self._make_orchestrator()
        output = (
            "## Cluster Overview\n"
            "   \n"
            "  \t  \n"
            "## Service Status\n"
            "All services running. Frontend 3/3, Backend 2/2. No crashloops detected.\n"
        )
        required = ["Cluster Overview", "Service Status"]
        result = orch._validate_gatherer_output(output, required)
        assert "Cluster Overview" in result.shallow_sections


class TestExtractSectionBody:
    """Test Orchestrator._extract_section_body()."""

    def test_markdown_heading_extraction(self) -> None:
        """Extracts body between markdown heading and next heading."""
        output = (
            "## Section A\n"
            "Content of section A with details.\n\n"
            "## Section B\n"
            "Content of section B.\n"
        )
        body = Orchestrator._extract_section_body(output, "Section A")
        assert "Content of section A" in body
        assert "Section B" not in body

    def test_last_section_extracts_to_end(self) -> None:
        """Last section extracts until end of text."""
        output = (
            "## Section A\n"
            "First.\n\n"
            "## Section B\n"
            "Content of the last section with lots of detail.\n"
        )
        body = Orchestrator._extract_section_body(output, "Section B")
        assert "Content of the last section" in body

    def test_section_not_found_returns_empty(self) -> None:
        """Non-existent section returns empty string."""
        output = "## Other Section\nSome data.\n"
        body = Orchestrator._extract_section_body(output, "Missing Section")
        assert body == ""

    def test_case_insensitive_heading(self) -> None:
        """Heading matching is case-insensitive."""
        output = "## CLUSTER OVERVIEW\nData here with enough detail to matter.\n"
        body = Orchestrator._extract_section_body(output, "Cluster Overview")
        assert "Data here" in body

    def test_different_heading_levels(self) -> None:
        """Works with different markdown heading levels (# to ######)."""
        for level in range(1, 7):
            prefix = "#" * level
            output = f"{prefix} My Section\nBody content for this particular section.\n"
            body = Orchestrator._extract_section_body(output, "My Section")
            assert "Body content" in body, f"Failed for heading level {level}"


class TestIsSectionShallow:
    """Test Orchestrator._is_section_shallow()."""

    def test_empty_body_is_shallow(self) -> None:
        assert Orchestrator._is_section_shallow("", 50) is True

    def test_empty_marker_is_shallow(self) -> None:
        assert Orchestrator._is_section_shallow("N/A", 50) is True
        assert Orchestrator._is_section_shallow("No data", 50) is True

    def test_short_body_is_shallow(self) -> None:
        assert Orchestrator._is_section_shallow("OK", 50) is True

    def test_sufficient_body_not_shallow(self) -> None:
        body = "This section contains detailed information about the cluster health status and metrics."
        assert Orchestrator._is_section_shallow(body, 50) is False

    def test_exact_threshold_not_shallow(self) -> None:
        body = "x" * 50
        assert Orchestrator._is_section_shallow(body, 50) is False

    def test_one_below_threshold_is_shallow(self) -> None:
        body = "x" * 49
        assert Orchestrator._is_section_shallow(body, 50) is True


class TestGathererValidationResult:
    """Test the GathererValidationResult dataclass."""

    def test_empty_result_no_retry(self) -> None:
        result = GathererValidationResult()
        assert not result.needs_retry

    def test_missing_triggers_retry(self) -> None:
        result = GathererValidationResult(missing_sections=["X"])
        assert result.needs_retry

    def test_shallow_triggers_retry(self) -> None:
        result = GathererValidationResult(shallow_sections=["Y"])
        assert result.needs_retry

    def test_both_triggers_retry(self) -> None:
        result = GathererValidationResult(
            missing_sections=["X"],
            shallow_sections=["Y"],
        )
        assert result.needs_retry


# ===========================================================================
# Enhanced retry prompt — missing vs shallow distinction
# ===========================================================================


class TestBuildRetryPromptEnhanced:
    """Test enhanced _build_retry_prompt with shallow_sections parameter."""

    def _make_orchestrator(self) -> Orchestrator:
        return Orchestrator(_make_mock_client(), _make_mock_settings())

    def test_only_missing_sections_prompt(self) -> None:
        """When only missing sections, prompt mentions missing but not shallow."""
        orch = self._make_orchestrator()
        prompt = orch._build_retry_prompt(
            "check health",
            ["Service Status"],
            shallow_sections=[],
        )
        assert "missing the following sections" in prompt
        assert "Service Status" in prompt
        assert "insufficient data" not in prompt

    def test_only_shallow_sections_prompt(self) -> None:
        """When only shallow sections, prompt mentions insufficient data."""
        orch = self._make_orchestrator()
        prompt = orch._build_retry_prompt(
            "check health",
            [],
            shallow_sections=["Cluster Overview", "Events Timeline"],
        )
        assert "insufficient data" in prompt
        assert "Cluster Overview" in prompt
        assert "Events Timeline" in prompt
        assert "missing the following sections" not in prompt

    def test_both_missing_and_shallow_prompt(self) -> None:
        """When both missing and shallow, prompt mentions both."""
        orch = self._make_orchestrator()
        prompt = orch._build_retry_prompt(
            "check health",
            ["Events Timeline"],
            shallow_sections=["Cluster Overview"],
        )
        assert "missing the following sections" in prompt
        assert "Events Timeline" in prompt
        assert "insufficient data" in prompt
        assert "Cluster Overview" in prompt

    def test_enhanced_prompt_no_bracket_markers(self) -> None:
        """Enhanced prompt still has no bracket markers."""
        orch = self._make_orchestrator()
        prompt = orch._build_retry_prompt(
            "check health",
            ["A"],
            shallow_sections=["B"],
        )
        for marker in ("[WARNING]", "[ERROR]", "[IMPORTANT]", "[CRITICAL]"):
            assert marker not in prompt

    def test_enhanced_prompt_starts_with_query(self) -> None:
        """Enhanced prompt still starts with the original query."""
        orch = self._make_orchestrator()
        prompt = orch._build_retry_prompt(
            "check health",
            [],
            shallow_sections=["Cluster Overview"],
        )
        assert prompt.startswith("check health")

    def test_backward_compat_no_shallow_kwarg(self) -> None:
        """Calling without shallow_sections kwarg still works (backward compat)."""
        orch = self._make_orchestrator()
        prompt = orch._build_retry_prompt("query", ["Missing Section"])
        assert "missing the following sections" in prompt
        assert "Missing Section" in prompt
        assert "Please regenerate" in prompt


# ===========================================================================
# Integration: shallow section triggers retry in pipeline
# ===========================================================================


class TestShallowRetryIntegration:
    """Integration tests for shallow-section retry in execute_with_tools."""

    def test_shallow_section_triggers_retry(self) -> None:
        """When a section header exists but content is shallow,
        the orchestrator retries with a prompt mentioning insufficient data."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1.execute.side_effect = [
            AgentResult(
                agent_name="gatherer",
                content=(
                    "## Cluster Overview\n"
                    "N/A\n"  # Shallow — empty marker
                    "## Service Status\n"
                    "All services running. Frontend 3/3, Backend 2/2. No crashloops detected.\n"
                    "## Events Timeline\n"
                    "10:00 Normal pod started. 10:05 Scaling complete. Health checks passing.\n"
                ),
                success=True,
                usage={"total_tokens": 100},
            ),
            AgentResult(
                agent_name="gatherer",
                content=(
                    "## Cluster Overview\n"
                    "Nodes: 3 running k8s v1.28. CPU 45%. Memory 60%. No node pressure detected.\n"
                    "## Service Status\n"
                    "All services running. Frontend 3/3, Backend 2/2. No crashloops detected.\n"
                    "## Events Timeline\n"
                    "10:00 Normal pod started. 10:05 Scaling complete. Health checks passing.\n"
                ),
                success=True,
                usage={"total_tokens": 200},
            ),
        ]

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "reporter"
        agent2.role = "Reporter"
        agent2.execute.return_value = AgentResult(
            agent_name="reporter", content="Final report", success=True,
            usage={"total_tokens": 50},
        )

        skill = StubToolSkill()
        skill.get_required_output_sections = MagicMock(
            return_value=["Cluster Overview", "Service Status", "Events Timeline"],
        )

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = orchestrator.execute_with_tools("check health", skill, registry)

        assert result.success is True
        assert agent1.execute.call_count == 2
        agent1.reset.assert_called_once()

        # Retry prompt should mention "insufficient data" for shallow section
        retry_call = agent1.execute.call_args_list[1]
        retry_prompt = retry_call.args[0]
        assert "insufficient data" in retry_prompt
        assert "Cluster Overview" in retry_prompt
        # Should NOT say "missing" for Cluster Overview (it was found, just shallow)
        assert "missing the following sections" not in retry_prompt

    def test_mixed_missing_and_shallow_retry(self) -> None:
        """Both missing and shallow sections trigger retry with differentiated prompt."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1.execute.side_effect = [
            AgentResult(
                agent_name="gatherer",
                content=(
                    "## Cluster Overview\n"
                    "No data available\n"  # Shallow — empty marker
                    "## Service Status\n"
                    "All services running. Frontend 3/3, Backend 2/2. No crashloops detected.\n"
                    # Events Timeline is completely missing
                ),
                success=True,
                usage={"total_tokens": 100},
            ),
            AgentResult(
                agent_name="gatherer",
                content=(
                    "## Cluster Overview\n"
                    "Nodes: 3 running k8s v1.28. CPU 45%. Memory 60%. No node pressure.\n"
                    "## Service Status\n"
                    "All services running. Frontend 3/3, Backend 2/2. No crashloops detected.\n"
                    "## Events Timeline\n"
                    "10:00 Normal pod started. 10:05 Scaling complete. Health checks passing.\n"
                ),
                success=True,
                usage={"total_tokens": 200},
            ),
        ]

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "reporter"
        agent2.role = "Reporter"
        agent2.execute.return_value = AgentResult(
            agent_name="reporter", content="Report", success=True,
            usage={"total_tokens": 50},
        )

        skill = StubToolSkill()
        skill.get_required_output_sections = MagicMock(
            return_value=["Cluster Overview", "Service Status", "Events Timeline"],
        )

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = orchestrator.execute_with_tools("check health", skill, registry)

        assert result.success is True
        assert agent1.execute.call_count == 2

        retry_call = agent1.execute.call_args_list[1]
        retry_prompt = retry_call.args[0]
        # Should mention BOTH missing and shallow
        assert "missing the following sections" in retry_prompt
        assert "Events Timeline" in retry_prompt
        assert "insufficient data" in retry_prompt
        assert "Cluster Overview" in retry_prompt

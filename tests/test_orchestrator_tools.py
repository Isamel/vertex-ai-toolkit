"""Tests for Orchestrator tool-aware agent integration (Phase 3).

Verifies that the orchestrator correctly creates mixed pipelines
(SpecialistAgent + ToolAwareAgent) and routes execution through them.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vaig.agents.base import AgentResult
from vaig.agents.orchestrator import (
    EMPTY_MARKERS,
    GathererValidationResult,
    Orchestrator,
    _build_tools_summary,
)
from vaig.agents.specialist import SpecialistAgent
from vaig.agents.tool_aware import ToolAwareAgent
from vaig.core.client import GenerationResult
from vaig.core.exceptions import VAIGError
from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.tools.base import ToolRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Section body with >200 chars for tests that need validation to pass depth check
_CLUSTER_BODY = (
    "Nodes: 3, all healthy. CPU usage is at 45% across the cluster. "
    "Memory utilization is stable at 60%. No node pressure conditions "
    "detected. Disk pressure: none. PID pressure: none. All kubelet "
    "versions match v1.28.4. Network policies are active."
)
_SERVICE_BODY = (
    "All services running smoothly. app-frontend: 3/3 replicas ready, "
    "0 restarts in last 24h. app-backend: 2/2 replicas ready, 1 restart "
    "(OOMKilled 6h ago, recovered). redis-cache: 1/1 ready, 0 restarts. "
    "All health checks passing. No pending rollouts detected."
)
_EVENTS_BODY = (
    "10:00 Normal SuccessfulCreate: Created pod app-frontend-abc123 on "
    "node gke-pool-1. 10:05 Normal ScalingReplicaSet: Scaled up deployment "
    "app-frontend to 3 replicas. 10:10 Normal Pulling: Pulling image "
    "gcr.io/project/frontend:v2.1. 10:12 Normal Pulled: Successfully pulled."
)
_RAW_FINDINGS_BODY = (
    "kubectl get pods returned 12 pods across 4 deployments. "
    "kubectl top pods shows app-backend-xyz789 using 450Mi/512Mi memory "
    "(88% utilization). kubectl describe node gke-pool-1 shows allocatable "
    "CPU: 3800m, requests: 3200m (84%). No eviction signals active."
)
_CLOUD_LOGGING_BODY = (
    "Cloud Logging query for severity>=WARNING returned 3 entries in last "
    "1h. Entry 1: 'Connection pool exhausted' from app-backend at 09:55. "
    "Entry 2: 'Slow query detected (2.3s)' from app-backend at 10:02. "
    "Entry 3: 'Health check timeout' from redis-cache at 10:08. No ERROR "
    "or CRITICAL entries found."
)
_INVESTIGATION_CHECKLIST_BODY = (
    "- [x] Step 1: Cluster-wide resource snapshot — collected\n"
    "- [x] Step 2: Service-level status — collected\n"
    "- [x] Step 3: Event timeline — collected\n"
    "- [x] Step 4: Deployment deep-dive — investigated\n"
    "- [x] Step 5: Pod-level diagnostics — investigated\n"
    "- [x] Step 6: HPA & scaling — investigated\n"
    "- [x] Step 7a: Cloud Logging query — completed\n"
    "- [x] Step 7b: Cross-reference findings — completed"
)


def _make_complete_output(
    *,
    include_raw: bool = False,
    include_logging: bool = False,
    include_checklist: bool = True,
) -> str:
    """Build gatherer output that passes validation (all sections >200 chars)."""
    parts = [
        f"## Cluster Overview\n{_CLUSTER_BODY}\n",
        f"## Service Status\n{_SERVICE_BODY}\n",
        f"## Events Timeline\n{_EVENTS_BODY}\n",
    ]
    if include_raw:
        parts.append(f"## Raw Findings\n{_RAW_FINDINGS_BODY}\n")
    if include_logging:
        parts.append(f"## Cloud Logging Findings\n{_CLOUD_LOGGING_BODY}\n")
    if include_checklist:
        parts.append(f"### Investigation Checklist\n{_INVESTIGATION_CHECKLIST_BODY}\n")
    return "\n".join(parts)


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
    settings.agents.max_iterations_retry = 10
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
                    "model": "gemini-2.5-pro",
                    "requires_tools": True,
                },
                {
                    "name": "tools-2",
                    "role": "Worker B",
                    "system_instruction": "B.",
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
            agent_name="agent-2", content="Final output.", success=True,
            usage={"prompt_tokens": 8, "completion_tokens": 12, "total_tokens": 20},
        )

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = orchestrator.execute_with_tools(
                "do something", StubToolSkill(), registry, strategy="sequential",
            )

        assert result.success is True
        assert len(result.agent_results) == 2
        assert result.agent_results[0].content == "First output"
        assert result.agent_results[1].content == "Final output."
        assert result.synthesized_output == "Final output."

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
        output = _make_complete_output()
        required = ["Cluster Overview", "Service Status", "Events Timeline"]
        result = orch._validate_gatherer_output(output, required)
        assert result.missing_sections == []
        assert result.shallow_sections == []
        assert not result.needs_retry

    def test_missing_sections_detected(self) -> None:
        """Missing sections are returned in missing_sections list."""
        orch = self._make_orchestrator()
        output = f"## Cluster Overview\n{_CLUSTER_BODY}\n"
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
            f"## CLUSTER OVERVIEW\n{_CLUSTER_BODY}\n"
            f"## service status\n{_SERVICE_BODY}\n"
            f"## events timeline\n{_EVENTS_BODY}\n"
            f"### Investigation Checklist\n{_INVESTIGATION_CHECKLIST_BODY}\n"
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
        output = f"anything\n### Investigation Checklist\n{_INVESTIGATION_CHECKLIST_BODY}\n"
        result = orch._validate_gatherer_output(output, [])
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
                content=f"## Cluster Overview\n{_CLUSTER_BODY}\n",
                # Missing Service Status, Events Timeline
                success=True,
                usage={"total_tokens": 100},
            ),
            AgentResult(
                agent_name="gatherer",
                content=_make_complete_output(),
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

    def test_always_retries_with_deepening_prompt_when_complete(self) -> None:
        """When gatherer output is complete with sufficient depth, the
        orchestrator STILL retries with a deepening prompt (mandatory 2nd pass)."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        complete_output = _make_complete_output()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1.execute.side_effect = [
            AgentResult(
                agent_name="gatherer",
                content=complete_output,
                success=True,
                usage={"total_tokens": 100},
            ),
            AgentResult(
                agent_name="gatherer",
                content=complete_output + "\n## Extra Findings\nMore data collected.\n",
                success=True,
                usage={"total_tokens": 150},
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
        # Agent1 should have been called twice (original + deepening pass)
        assert agent1.execute.call_count == 2
        # Deepening pass should NOT call reset (incremental — keeps history)
        agent1.reset.assert_not_called()
        # Retry prompt should be a DEEPENING prompt, not a missing-sections prompt
        retry_call = agent1.execute.call_args_list[1]
        retry_prompt = retry_call.args[0]
        assert "first diagnostic pass" in retry_prompt
        assert "Do NOT repeat tool calls" in retry_prompt
        assert "Second Pass" in retry_prompt
        # Should NOT mention missing sections
        assert "missing the following sections" not in retry_prompt
        # Result should be MERGED (first + second pass content)
        final_content = result.agent_results[0].content
        assert complete_output in final_content

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
                content=f"## Cluster Overview\n{_CLUSTER_BODY}\n",
                # Missing Service Status, Events Timeline
                success=True,
                usage={"total_tokens": 100},
            ),
            AgentResult(
                agent_name="gatherer",
                content=_make_complete_output(),
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
            f"## Service Status\n{_SERVICE_BODY}\n"
            f"## Events Timeline\n{_EVENTS_BODY}\n"
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
            f"## Service Status\n{_SERVICE_BODY}\n"
            f"## Events Timeline\n{_EVENTS_BODY}\n"
        )
        required = ["Cluster Overview", "Service Status", "Events Timeline"]
        result = orch._validate_gatherer_output(output, required)
        assert "Cluster Overview" in result.shallow_sections
        assert result.needs_retry

    def test_data_not_available_is_shallow(self) -> None:
        """Section body is 'Data not available' → shallow."""
        orch = self._make_orchestrator()
        output = (
            f"## Cluster Overview\n{_CLUSTER_BODY}\n"
            "## Service Status\n"
            "Data not available\n"
            f"## Events Timeline\n{_EVENTS_BODY}\n"
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
        # "OK" is only 2 chars, well below the 200-char default
        output = (
            "## Cluster Overview\n"
            "OK\n"
            f"## Service Status\n{_SERVICE_BODY}\n"
            f"## Events Timeline\n{_EVENTS_BODY}\n"
        )
        required = ["Cluster Overview", "Service Status", "Events Timeline"]
        result = orch._validate_gatherer_output(output, required)
        assert "Cluster Overview" in result.shallow_sections

    def test_custom_min_content_chars(self) -> None:
        """Custom min_content_chars threshold is respected."""
        orch = self._make_orchestrator()
        # 30 chars of content: "This is some moderate content."
        body_text = "This is some moderate content."
        checklist = f"\n### Investigation Checklist\n{_INVESTIGATION_CHECKLIST_BODY}\n"
        output = f"## Cluster Overview\n{body_text}\n{checklist}"
        required = ["Cluster Overview"]

        # With default (200), it should be shallow
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
            f"## Cluster Overview\n{_CLUSTER_BODY}\n"
            f"### Investigation Checklist\n{_INVESTIGATION_CHECKLIST_BODY}\n"
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
            f"## Service Status\n{_SERVICE_BODY}\n"
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
            f"## Service Status\n{_SERVICE_BODY}\n"
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
                    f"## Service Status\n{_SERVICE_BODY}\n"
                    f"## Events Timeline\n{_EVENTS_BODY}\n"
                ),
                success=True,
                usage={"total_tokens": 100},
            ),
            AgentResult(
                agent_name="gatherer",
                content=_make_complete_output(),
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
                    f"## Service Status\n{_SERVICE_BODY}\n"
                    # Events Timeline is completely missing
                ),
                success=True,
                usage={"total_tokens": 100},
            ),
            AgentResult(
                agent_name="gatherer",
                content=_make_complete_output(),
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


# ===========================================================================
# Task 3.2 — Async orchestrator methods (asyncio.gather instead of ThreadPool)
# ===========================================================================


class TestAsyncExecuteFanout:
    """Test async_execute_fanout() using gather_with_errors."""

    async def test_fanout_executes_all_agents_concurrently(self) -> None:
        """All agents run concurrently via asyncio.gather, results merged."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        agent1 = MagicMock(spec=SpecialistAgent)
        agent1.name = "agent-1"
        agent1.role = "Analyzer"
        agent1.execute.return_value = AgentResult(
            agent_name="agent-1", content="Analysis A", success=True,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "agent-2"
        agent2.role = "Summarizer"
        agent2.execute.return_value = AgentResult(
            agent_name="agent-2", content="Summary B", success=True,
            usage={"prompt_tokens": 8, "completion_tokens": 12, "total_tokens": 20},
        )

        skill = StubToolSkill()

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = await orchestrator.async_execute_fanout(
                skill, SkillPhase.ANALYZE, "ctx", "user task",
            )

        assert result.success is True
        assert len(result.agent_results) == 2
        assert result.agent_results[0].content == "Analysis A"
        assert result.agent_results[1].content == "Summary B"
        # Merged output contains both agents
        assert "agent-1" in result.synthesized_output
        assert "agent-2" in result.synthesized_output
        # Usage accumulated
        assert result.total_usage["total_tokens"] == 35

    async def test_fanout_partial_failure_still_succeeds(self) -> None:
        """If one agent fails in fanout, overall result succeeds when others succeed."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        agent1 = MagicMock(spec=SpecialistAgent)
        agent1.name = "agent-1"
        agent1.execute.return_value = AgentResult(
            agent_name="agent-1", content="Error occurred", success=False, usage={},
        )

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "agent-2"
        agent2.execute.return_value = AgentResult(
            agent_name="agent-2", content="All good", success=True,
            usage={"total_tokens": 10},
        )

        skill = StubToolSkill()

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = await orchestrator.async_execute_fanout(
                skill, SkillPhase.ANALYZE, "ctx", "task",
            )

        assert result.success is True  # at least one succeeded
        assert len(result.agent_results) == 2

    async def test_fanout_all_failures(self) -> None:
        """If all agents fail in fanout, result.success is False."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        agent1 = MagicMock(spec=SpecialistAgent)
        agent1.name = "agent-1"
        agent1.execute.return_value = AgentResult(
            agent_name="agent-1", content="Error A", success=False, usage={},
        )

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "agent-2"
        agent2.execute.return_value = AgentResult(
            agent_name="agent-2", content="Error B", success=False, usage={},
        )

        skill = StubToolSkill()

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = await orchestrator.async_execute_fanout(
                skill, SkillPhase.ANALYZE, "ctx", "task",
            )

        assert result.success is False

    async def test_fanout_handles_agent_exception(self) -> None:
        """Agent raising an exception during fanout produces a failed AgentResult."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        agent1 = MagicMock(spec=SpecialistAgent)
        agent1.name = "agent-1"
        agent1.execute.side_effect = RuntimeError("boom")

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "agent-2"
        agent2.execute.return_value = AgentResult(
            agent_name="agent-2", content="OK", success=True,
            usage={"total_tokens": 5},
        )

        skill = StubToolSkill()

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = await orchestrator.async_execute_fanout(
                skill, SkillPhase.ANALYZE, "ctx", "task",
            )

        # One succeeded, so overall success
        assert result.success is True
        assert len(result.agent_results) == 2
        # First result is the error fallback
        assert result.agent_results[0].success is False
        assert "failed with an unexpected error" in result.agent_results[0].content

    async def test_fanout_no_agents(self) -> None:
        """Fan-out with empty agent list returns success=False."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = StubToolSkill()

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[]):
            result = await orchestrator.async_execute_fanout(
                skill, SkillPhase.ANALYZE, "ctx", "task",
            )

        assert result.success is False
        assert len(result.agent_results) == 0


class TestAsyncExecuteSequential:
    """Test async_execute_sequential()."""

    async def test_sequential_executes_in_order(self) -> None:
        """Agents run sequentially with context chaining."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        agent1 = MagicMock(spec=SpecialistAgent)
        agent1.name = "agent-1"
        agent1.role = "First"
        agent1.execute.return_value = AgentResult(
            agent_name="agent-1", content="Step 1 output", success=True,
            usage={"total_tokens": 10},
        )

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "agent-2"
        agent2.role = "Second"
        agent2.execute.return_value = AgentResult(
            agent_name="agent-2", content="Step 2 output", success=True,
            usage={"total_tokens": 15},
        )

        skill = StubToolSkill()

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = await orchestrator.async_execute_sequential(
                skill, SkillPhase.ANALYZE, "ctx", "task",
            )

        assert result.success is True
        assert len(result.agent_results) == 2
        assert result.synthesized_output == "Step 2 output"

        # Second agent gets context with first agent's output
        second_call_context = agent2.execute.call_args.kwargs["context"]
        assert "Previous Analysis" in second_call_context
        assert "Step 1 output" in second_call_context

    async def test_sequential_stops_on_failure(self) -> None:
        """Sequential execution stops at the first failed agent."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        agent1 = MagicMock(spec=SpecialistAgent)
        agent1.name = "agent-1"
        agent1.role = "First"
        agent1.execute.return_value = AgentResult(
            agent_name="agent-1", content="Error", success=False, usage={},
        )

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "agent-2"
        agent2.role = "Second"

        skill = StubToolSkill()

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = await orchestrator.async_execute_sequential(
                skill, SkillPhase.ANALYZE, "ctx", "task",
            )

        assert result.success is False
        assert len(result.agent_results) == 1
        agent2.execute.assert_not_called()


class TestAsyncExecuteSkillPhase:
    """Test async_execute_skill_phase()."""

    async def test_uses_fanout_strategy(self) -> None:
        """strategy='fanout' delegates to async_execute_fanout."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        agent1 = MagicMock(spec=SpecialistAgent)
        agent1.name = "a1"
        agent1.execute.return_value = AgentResult(
            agent_name="a1", content="Out", success=True,
            usage={"total_tokens": 5},
        )

        skill = StubToolSkill()

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1]):
            result = await orchestrator.async_execute_skill_phase(
                skill, SkillPhase.ANALYZE, "ctx", "task", strategy="fanout",
            )

        assert result.success is True
        assert result.output  # Non-empty output

    async def test_default_strategy_is_sequential(self) -> None:
        """Default strategy uses async_execute_sequential."""
        client = _make_mock_client()
        client.generate.return_value = _make_generation_result(text="Done")
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = StubToolSkill()

        # Let it create real SpecialistAgents
        result = await orchestrator.async_execute_skill_phase(
            skill, SkillPhase.ANALYZE, "ctx", "task",
        )

        assert result.success is True


class TestAsyncExecuteWithTools:
    """Test async_execute_with_tools()."""

    async def test_fanout_strategy_uses_gather(self) -> None:
        """Fanout strategy runs agents concurrently via asyncio.gather."""
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
            result = await orchestrator.async_execute_with_tools(
                "analyze this", StubToolSkill(), registry, strategy="fanout",
            )

        assert result.success is True
        assert len(result.agent_results) == 2
        # Both agents called with same query
        agent1.execute.assert_called_once_with("analyze this", context="analyze this")
        agent2.execute.assert_called_once_with("analyze this", context="analyze this")
        # Merged output
        assert "agent-1" in result.synthesized_output
        assert "agent-2" in result.synthesized_output
        # Usage accumulated
        assert result.total_usage["total_tokens"] == 25

    async def test_fanout_partial_failure(self) -> None:
        """Partial failure in async fanout with tools still succeeds."""
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
            result = await orchestrator.async_execute_with_tools(
                "go", StubToolSkill(), registry, strategy="fanout",
            )

        assert result.success is True

    async def test_sequential_strategy(self) -> None:
        """Sequential strategy runs agents in order with context chaining."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

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
            result = await orchestrator.async_execute_with_tools(
                "do something", StubToolSkill(), registry, strategy="sequential",
            )

        assert result.success is True
        assert result.synthesized_output == "Final output"
        # Context chaining works
        second_call_context = agent2.execute.call_args.kwargs["context"]
        assert "Previous Analysis" in second_call_context
        assert "First output" in second_call_context

    async def test_sequential_stops_on_failure(self) -> None:
        """Sequential stops when an agent fails."""
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

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = await orchestrator.async_execute_with_tools(
                "do something", StubToolSkill(), registry,
            )

        assert result.success is False
        assert len(result.agent_results) == 1
        agent2.execute.assert_not_called()

    async def test_single_strategy(self) -> None:
        """Single strategy runs only the first agent."""
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
            result = await orchestrator.async_execute_with_tools(
                "query", StubToolSkill(), registry, strategy="single",
            )

        assert result.success is True
        assert result.synthesized_output == "Solo output"
        agent2.execute.assert_not_called()

    async def test_single_no_agents(self) -> None:
        """Single strategy with no agents returns failure."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[]):
            result = await orchestrator.async_execute_with_tools(
                "query", StubToolSkill(), registry, strategy="single",
            )

        assert result.success is False
        assert "No agents" in result.synthesized_output

    async def test_usage_accumulated(self) -> None:
        """Token usage is accumulated across agents in async mode."""
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
            result = await orchestrator.async_execute_with_tools(
                "q", StubToolSkill(), registry,
            )

        assert result.total_usage["prompt_tokens"] == 25
        assert result.total_usage["completion_tokens"] == 45
        assert result.total_usage["total_tokens"] == 70

    async def test_gatherer_retry_on_incomplete_output(self) -> None:
        """Async sequential with tools retries gatherer when output is incomplete."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1.execute.side_effect = [
            AgentResult(
                agent_name="gatherer",
                content=f"## Cluster Overview\n{_CLUSTER_BODY}\n",
                # Missing Service Status, Events Timeline
                success=True,
                usage={"total_tokens": 100},
            ),
            AgentResult(
                agent_name="gatherer",
                content=_make_complete_output(),
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
            result = await orchestrator.async_execute_with_tools(
                "check health", skill, registry,
            )

        assert result.success is True
        # Agent1 called twice (original + retry)
        assert agent1.execute.call_count == 2
        agent1.reset.assert_called_once()

    async def test_fanout_exception_handling(self) -> None:
        """Agent raising exception in async fanout produces error result."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "agent-1"
        agent1.execute.side_effect = RuntimeError("connection lost")

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "agent-2"
        agent2.execute.return_value = AgentResult(
            agent_name="agent-2", content="OK", success=True,
            usage={"total_tokens": 5},
        )

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = await orchestrator.async_execute_with_tools(
                "query", StubToolSkill(), registry, strategy="fanout",
            )

        assert result.success is True
        assert len(result.agent_results) == 2
        assert result.agent_results[0].success is False
        assert "failed with an unexpected error" in result.agent_results[0].content


class TestAsyncSyncParity:
    """Verify that async methods produce the same results as sync methods."""

    async def test_fanout_parity(self) -> None:
        """async_execute_fanout produces same result structure as execute_fanout."""
        client = _make_mock_client()
        settings = _make_mock_settings()
        registry = _make_mock_registry()

        # Same agent setup for both
        def make_agents():
            a1 = MagicMock(spec=SpecialistAgent)
            a1.name = "agent-1"
            a1.role = "R1"
            a1.execute.return_value = AgentResult(
                agent_name="agent-1", content="Output A", success=True,
                usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            )
            a2 = MagicMock(spec=SpecialistAgent)
            a2.name = "agent-2"
            a2.role = "R2"
            a2.execute.return_value = AgentResult(
                agent_name="agent-2", content="Output B", success=True,
                usage={"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
            )
            return [a1, a2]

        skill = StubToolSkill()

        # Sync
        sync_orch = Orchestrator(client, settings)
        with patch.object(sync_orch, "create_agents_for_skill", return_value=make_agents()):
            sync_result = sync_orch.execute_fanout(
                skill, SkillPhase.ANALYZE, "ctx", "task",
            )

        # Async
        async_orch = Orchestrator(client, settings)
        with patch.object(async_orch, "create_agents_for_skill", return_value=make_agents()):
            async_result = await async_orch.async_execute_fanout(
                skill, SkillPhase.ANALYZE, "ctx", "task",
            )

        # Both should have same structure
        assert sync_result.success == async_result.success
        assert len(sync_result.agent_results) == len(async_result.agent_results)
        assert sync_result.total_usage == async_result.total_usage
        assert sync_result.synthesized_output == async_result.synthesized_output

    async def test_sequential_parity(self) -> None:
        """async_execute_sequential produces same result as execute_sequential."""
        client = _make_mock_client()
        settings = _make_mock_settings()

        def make_agents():
            a = MagicMock(spec=SpecialistAgent)
            a.name = "solo"
            a.role = "Worker"
            a.execute.return_value = AgentResult(
                agent_name="solo", content="Done", success=True,
                usage={"total_tokens": 20},
            )
            return [a]

        skill = StubToolSkill()

        sync_orch = Orchestrator(client, settings)
        with patch.object(sync_orch, "create_agents_for_skill", return_value=make_agents()):
            sync_result = sync_orch.execute_sequential(
                skill, SkillPhase.ANALYZE, "ctx", "task",
            )

        async_orch = Orchestrator(client, settings)
        with patch.object(async_orch, "create_agents_for_skill", return_value=make_agents()):
            async_result = await async_orch.async_execute_sequential(
                skill, SkillPhase.ANALYZE, "ctx", "task",
            )

        assert sync_result.success == async_result.success
        assert sync_result.synthesized_output == async_result.synthesized_output
        assert sync_result.total_usage == async_result.total_usage


# ===========================================================================
# Deepening prompt — mandatory second pass
# ===========================================================================


class TestBuildDeepeningPrompt:
    """Test Orchestrator._build_deepening_prompt()."""

    def test_references_conversation_history(self) -> None:
        """The incremental deepening prompt references conversation history."""
        first_pass = "## Cluster Overview\nSome data about the cluster.\n"
        prompt = Orchestrator._build_deepening_prompt("check health", first_pass)
        assert "first diagnostic pass" in prompt
        assert "conversation history" in prompt.lower()

    def test_includes_original_query(self) -> None:
        """The deepening prompt includes the original query."""
        prompt = Orchestrator._build_deepening_prompt(
            "check health of my-app in prod", "first pass data",
        )
        assert "check health of my-app in prod" in prompt

    def test_contains_no_repeat_instruction(self) -> None:
        """The prompt instructs NOT to repeat tool calls."""
        prompt = Orchestrator._build_deepening_prompt("query", "output")
        assert "Do NOT repeat tool calls" in prompt

    def test_contains_diagnostic_instructions(self) -> None:
        """The prompt mentions specific diagnostic areas to check."""
        prompt = Orchestrator._build_deepening_prompt("query", "output")
        assert "ReplicaSet" in prompt
        assert "HPA" in prompt
        assert "Cloud Logging" in prompt

    def test_requests_only_new_findings(self) -> None:
        """The prompt asks for ONLY new findings from second pass."""
        prompt = Orchestrator._build_deepening_prompt("my query", "my output")
        assert "Second Pass" in prompt
        assert "ONLY" in prompt

    def test_no_bracket_markers(self) -> None:
        """The deepening prompt has no bracket-prefixed markers."""
        prompt = Orchestrator._build_deepening_prompt("query", "output")
        for marker in ("[WARNING]", "[ERROR]", "[IMPORTANT]", "[CRITICAL]", "[ALERT]"):
            assert marker not in prompt

    def test_does_not_embed_first_pass_output(self) -> None:
        """The incremental prompt does NOT embed the first pass output
        inline — the agent has it in conversation history already."""
        first_pass = "UNIQUE_FIRST_PASS_CONTENT_abc123xyz"
        prompt = Orchestrator._build_deepening_prompt("query", first_pass)
        # The original content should NOT appear in the prompt itself
        assert first_pass not in prompt


# ===========================================================================
# Always-retry behavior — gatherer ALWAYS gets a second pass
# ===========================================================================


class TestAlwaysRetryGatherer:
    """Test that the gatherer ALWAYS retries regardless of validation result."""

    def test_sync_always_retries_with_deepening_when_valid(self) -> None:
        """Sync: even when validation passes, gatherer retries with deepening prompt."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        complete_output = _make_complete_output()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1.execute.side_effect = [
            AgentResult(
                agent_name="gatherer",
                content=complete_output,
                success=True,
                usage={"total_tokens": 100},
            ),
            AgentResult(
                agent_name="gatherer",
                content=complete_output + "\nMore data.\n",
                success=True,
                usage={"total_tokens": 150},
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
        # Deepening pass: NO reset (incremental)
        agent1.reset.assert_not_called()

        # Deepening prompt sent, not missing-sections prompt
        retry_prompt = agent1.execute.call_args_list[1].args[0]
        assert "first diagnostic pass" in retry_prompt
        assert "Do NOT repeat tool calls" in retry_prompt

        # Result should be MERGED
        final_content = result.agent_results[0].content
        assert complete_output in final_content

    async def test_async_always_retries_with_deepening_when_valid(self) -> None:
        """Async: even when validation passes, gatherer retries with deepening prompt."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        complete_output = _make_complete_output()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1.execute.side_effect = [
            AgentResult(
                agent_name="gatherer",
                content=complete_output,
                success=True,
                usage={"total_tokens": 100},
            ),
            AgentResult(
                agent_name="gatherer",
                content=complete_output + "\nMore data.\n",
                success=True,
                usage={"total_tokens": 150},
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
            result = await orchestrator.async_execute_with_tools(
                "check health", skill, registry,
            )

        assert result.success is True
        assert agent1.execute.call_count == 2
        # Deepening pass: NO reset (incremental)
        agent1.reset.assert_not_called()

        # Deepening prompt
        retry_prompt = agent1.execute.call_args_list[1].args[0]
        assert "first diagnostic pass" in retry_prompt

        # Result should be MERGED
        final_content = result.agent_results[0].content
        assert complete_output in final_content

    def test_sync_uses_retry_prompt_when_validation_fails(self) -> None:
        """Sync: when validation fails, uses retry prompt (not deepening)."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1.execute.side_effect = [
            AgentResult(
                agent_name="gatherer",
                content="Some random text without sections",
                success=True,
                usage={"total_tokens": 50},
            ),
            AgentResult(
                agent_name="gatherer",
                content=_make_complete_output(),
                success=True,
                usage={"total_tokens": 150},
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
        # Validation retry: reset IS called
        agent1.reset.assert_called_once()

        # Missing-sections retry prompt, NOT deepening
        retry_prompt = agent1.execute.call_args_list[1].args[0]
        assert "missing the following sections" in retry_prompt
        assert "first diagnostic pass" not in retry_prompt

    def test_deepening_retry_failure_stops_pipeline(self) -> None:
        """If the deepening retry fails (success=False), pipeline stops."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        complete_output = _make_complete_output()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1.execute.side_effect = [
            AgentResult(
                agent_name="gatherer",
                content=complete_output,
                success=True,
                usage={"total_tokens": 100},
            ),
            AgentResult(
                agent_name="gatherer",
                content="API error",
                success=False,
                usage={"total_tokens": 10},
            ),
        ]

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "reporter"
        agent2.role = "Reporter"

        skill = StubToolSkill()
        skill.get_required_output_sections = MagicMock(
            return_value=["Cluster Overview", "Service Status", "Events Timeline"],
        )

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = orchestrator.execute_with_tools("check health", skill, registry)

        assert result.success is False
        agent2.execute.assert_not_called()


# ===========================================================================
# Incremental deepening + max_iterations_retry — Causa 1 tests
# ===========================================================================


class TestIncrementalDeepening:
    """Validate the incremental second-pass behavior:
    - Deepening pass does NOT reset agent (keeps conversation history)
    - Deepening pass merges first + second pass content
    - Validation retry DOES reset agent
    - max_iterations_retry is applied during second pass
    - max_iterations is restored after second pass (try/finally)
    """

    def test_deepening_merges_content(self) -> None:
        """Deepening pass merges first-pass + second-pass content."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        complete_output = _make_complete_output()
        second_pass_output = "### Second Pass — Additional Findings\nNew data here."

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1._max_iterations = 15
        agent1.execute.side_effect = [
            AgentResult(
                agent_name="gatherer",
                content=complete_output,
                success=True,
                usage={"total_tokens": 100},
            ),
            AgentResult(
                agent_name="gatherer",
                content=second_pass_output,
                success=True,
                usage={"total_tokens": 80},
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
        # Merged content: first pass + "\n\n" + second pass
        final_content = result.agent_results[0].content
        assert complete_output in final_content
        assert second_pass_output in final_content
        assert f"{complete_output}\n\n{second_pass_output}" == final_content

    def test_validation_retry_replaces_content(self) -> None:
        """Validation retry (missing sections) replaces — does NOT merge."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        retry_output = _make_complete_output()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1._max_iterations = 15
        agent1.execute.side_effect = [
            AgentResult(
                agent_name="gatherer",
                content="incomplete — no sections",
                success=True,
                usage={"total_tokens": 50},
            ),
            AgentResult(
                agent_name="gatherer",
                content=retry_output,
                success=True,
                usage={"total_tokens": 150},
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
        # Validation retry: content is REPLACED, NOT merged
        final_content = result.agent_results[0].content
        assert final_content == retry_output
        assert "incomplete — no sections" not in final_content

    def test_max_iterations_retry_applied_during_deepening(self) -> None:
        """max_iterations_retry config is set on agent during second pass."""
        client = _make_mock_client()
        settings = _make_mock_settings()
        settings.agents.max_iterations_retry = 7
        orchestrator = Orchestrator(client, settings)
        registry = _make_mock_registry()

        complete_output = _make_complete_output()
        iterations_seen: list[int] = []

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1._max_iterations = 15

        def _capture_execute(prompt: str, **kwargs: Any) -> AgentResult:
            iterations_seen.append(agent1._max_iterations)
            return AgentResult(
                agent_name="gatherer",
                content=complete_output,
                success=True,
                usage={"total_tokens": 100},
            )

        agent1.execute.side_effect = _capture_execute

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
            orchestrator.execute_with_tools("check health", skill, registry)

        # First call: original max_iterations (15)
        # Second call: max_iterations_retry (7)
        assert iterations_seen == [15, 7]

    def test_max_iterations_restored_after_deepening(self) -> None:
        """max_iterations is restored to original value after second pass."""
        client = _make_mock_client()
        settings = _make_mock_settings()
        settings.agents.max_iterations_retry = 5
        orchestrator = Orchestrator(client, settings)
        registry = _make_mock_registry()

        complete_output = _make_complete_output()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1._max_iterations = 20

        agent1.execute.side_effect = [
            AgentResult(
                agent_name="gatherer",
                content=complete_output,
                success=True,
                usage={"total_tokens": 100},
            ),
            AgentResult(
                agent_name="gatherer",
                content="more data",
                success=True,
                usage={"total_tokens": 80},
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
            orchestrator.execute_with_tools("check health", skill, registry)

        # max_iterations should be restored to original value
        assert agent1._max_iterations == 20

    def test_max_iterations_restored_on_failure(self) -> None:
        """max_iterations is restored even when the second pass raises."""
        client = _make_mock_client()
        settings = _make_mock_settings()
        settings.agents.max_iterations_retry = 5
        orchestrator = Orchestrator(client, settings)
        registry = _make_mock_registry()

        complete_output = _make_complete_output()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1._max_iterations = 20

        agent1.execute.side_effect = [
            AgentResult(
                agent_name="gatherer",
                content=complete_output,
                success=True,
                usage={"total_tokens": 100},
            ),
            RuntimeError("API explosion"),
        ]

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "reporter"
        agent2.role = "Reporter"

        skill = StubToolSkill()
        skill.get_required_output_sections = MagicMock(
            return_value=["Cluster Overview", "Service Status", "Events Timeline"],
        )

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            with pytest.raises(VAIGError, match="Pipeline execution failed"):
                orchestrator.execute_with_tools("check health", skill, registry)

        # max_iterations should STILL be restored (try/finally)
        assert agent1._max_iterations == 20

    def test_config_default_max_iterations_retry(self) -> None:
        """AgentsConfig has max_iterations_retry with default 15."""
        from vaig.core.config import AgentsConfig
        config = AgentsConfig()
        assert config.max_iterations_retry == 15


def _make_checklist(
    steps: dict[str, tuple[str, str]],
) -> str:
    """Build an Investigation Checklist section from step definitions.

    Args:
        steps: Mapping of step_id → (status, description).
            status is "x" (completed) or " " (skipped).

    Returns:
        A markdown section string like::

            ### Investigation Checklist
            - [x] Step 1: Cluster-wide resource snapshot — collected
            - [ ] Step 4: Deployment deep-dive (SKIPPED — no unhealthy deployments)
    """
    lines = ["### Investigation Checklist"]
    for step_id, (status, desc) in steps.items():
        lines.append(f"- [{status}] Step {step_id}: {desc}")
    return "\n".join(lines)


# All steps completed — baseline for "no warnings"
_ALL_STEPS_COMPLETE: dict[str, tuple[str, str]] = {
    "1": ("x", "Cluster-wide resource snapshot — collected"),
    "2": ("x", "Service-level status — collected"),
    "3": ("x", "Event timeline — collected"),
    "4": ("x", "Deployment deep-dive — investigated"),
    "5": ("x", "Pod-level diagnostics — investigated"),
    "6": ("x", "HPA & scaling — investigated"),
    "7a": ("x", "Cloud Logging query — completed"),
    "7b": ("x", "Cross-reference findings — completed"),
}


class TestValidateInvestigationChecklist:
    """Test Orchestrator._validate_investigation_checklist()."""

    def test_all_steps_completed_no_warnings(self) -> None:
        """All steps marked [x] → no warnings."""
        checklist = _make_checklist(_ALL_STEPS_COMPLETE)
        output = f"## Cluster Overview\nSome data.\n\n{checklist}\n"
        warnings = Orchestrator._validate_investigation_checklist(output, output)
        assert warnings == []

    def test_step4_skipped_no_evidence_valid(self) -> None:
        """Step 4 skipped, no unhealthy deployment evidence → no warning."""
        steps = dict(_ALL_STEPS_COMPLETE)
        steps["4"] = (" ", "Deployment deep-dive (SKIPPED — no unhealthy deployments)")
        checklist = _make_checklist(steps)
        # Output with NO deployment problem indicators
        output = (
            "## Cluster Overview\n"
            "All 3 nodes healthy. CPU at 45%.\n\n"
            "## Service Status\n"
            "app-frontend: 3/3 replicas ready, all healthy.\n"
            "app-backend: 2/2 replicas ready.\n\n"
            f"{checklist}\n"
        )
        warnings = Orchestrator._validate_investigation_checklist(output, output)
        assert warnings == []

    def test_step4_skipped_but_zero_replicas_invalid(self) -> None:
        """Step 4 skipped BUT output mentions '0/3' replicas → invalid skip."""
        steps = dict(_ALL_STEPS_COMPLETE)
        steps["4"] = (" ", "Deployment deep-dive (SKIPPED — no unhealthy deployments)")
        checklist = _make_checklist(steps)
        output = (
            "## Service Status\n"
            "app-frontend: 0/3 replicas ready, possible issue.\n\n"
            f"{checklist}\n"
        )
        warnings = Orchestrator._validate_investigation_checklist(output, output)
        assert len(warnings) == 1
        assert "Step 4" in warnings[0]
        assert "SKIPPED" in warnings[0]

    def test_step4_skipped_but_failedcreate_invalid(self) -> None:
        """Step 4 skipped BUT output mentions 'FailedCreate' → invalid skip."""
        steps = dict(_ALL_STEPS_COMPLETE)
        steps["4"] = (" ", "Deployment deep-dive (SKIPPED — no issues detected)")
        checklist = _make_checklist(steps)
        output = (
            "## Events Timeline\n"
            "10:00 Warning FailedCreate: Error creating pod.\n\n"
            f"{checklist}\n"
        )
        warnings = Orchestrator._validate_investigation_checklist(output, output)
        assert len(warnings) == 1
        assert "Step 4" in warnings[0]

    def test_step5_skipped_but_crashloopbackoff_invalid(self) -> None:
        """Step 5 skipped BUT output mentions 'CrashLoopBackOff' → invalid skip."""
        steps = dict(_ALL_STEPS_COMPLETE)
        steps["5"] = (" ", "Pod-level diagnostics (SKIPPED — all pods running)")
        checklist = _make_checklist(steps)
        output = (
            "## Service Status\n"
            "app-backend pod is in CrashLoopBackOff state.\n\n"
            f"{checklist}\n"
        )
        warnings = Orchestrator._validate_investigation_checklist(output, output)
        assert len(warnings) == 1
        assert "Step 5" in warnings[0]
        assert "CrashLoopBackOff" in warnings[0]

    def test_step6_skipped_but_scalinglimited_invalid(self) -> None:
        """Step 6 skipped BUT output mentions 'ScalingLimited' → invalid skip."""
        steps = dict(_ALL_STEPS_COMPLETE)
        steps["6"] = (" ", "Autoscaler check (SKIPPED — no autoscaler configured)")
        checklist = _make_checklist(steps)
        output = (
            "## Events Timeline\n"
            "10:30 Warning ScalingLimited: desired 5, max 3.\n\n"
            f"{checklist}\n"
        )
        warnings = Orchestrator._validate_investigation_checklist(output, output)
        assert len(warnings) == 1
        assert "Step 6" in warnings[0]
        assert "ScalingLimited" in warnings[0]

    def test_no_checklist_found_returns_warning(self) -> None:
        """Output with no Investigation Checklist section → warning."""
        output = (
            "## Cluster Overview\nSome data.\n\n"
            "## Service Status\nAll good.\n"
        )
        warnings = Orchestrator._validate_investigation_checklist(output, output)
        assert len(warnings) == 1
        assert "missing" in warnings[0].lower()

    def test_mandatory_step_skipped_always_invalid(self) -> None:
        """Mandatory steps (1, 2, 3, 7a, 7b) skipped → always invalid."""
        for mandatory_step in ("1", "2", "3", "7a", "7b"):
            steps = dict(_ALL_STEPS_COMPLETE)
            steps[mandatory_step] = (
                " ",
                f"Step {mandatory_step} description (SKIPPED — reason)",
            )
            checklist = _make_checklist(steps)
            output = f"## Some Data\nClean output.\n\n{checklist}\n"
            warnings = Orchestrator._validate_investigation_checklist(output, output)
            assert any(
                f"Step {mandatory_step}" in w for w in warnings
            ), f"Expected warning for mandatory step {mandatory_step}, got: {warnings}"
            assert any(
                "MANDATORY" in w for w in warnings
            ), f"Expected 'MANDATORY' in warning for step {mandatory_step}"

    def test_multiple_invalid_skips_returns_multiple_warnings(self) -> None:
        """Multiple invalid skips return one warning per invalid skip."""
        steps = dict(_ALL_STEPS_COMPLETE)
        # Skip step 4 with contradicting evidence
        steps["4"] = (" ", "Deployment deep-dive (SKIPPED)")
        # Skip step 5 with contradicting evidence
        steps["5"] = (" ", "Pod diagnostics (SKIPPED)")
        checklist = _make_checklist(steps)
        output = (
            "## Service Status\n"
            "app-frontend: 0/3 replicas ready.\n"
            "app-backend pod is CrashLoopBackOff.\n\n"
            f"{checklist}\n"
        )
        warnings = Orchestrator._validate_investigation_checklist(output, output)
        assert len(warnings) == 2
        step_ids = [w.split("Step ")[1][0] for w in warnings]
        assert "4" in step_ids
        assert "5" in step_ids

    def test_step5_skipped_error_word_boundary_match(self) -> None:
        """Step 5: \\bError\\b matches standalone 'Error' but test documents
        that it also matches within sentences containing 'Error' as a word."""
        steps = dict(_ALL_STEPS_COMPLETE)
        steps["5"] = (" ", "Pod diagnostics (SKIPPED)")
        checklist = _make_checklist(steps)
        # "Error" as a standalone word in output
        output = (
            "## Events Timeline\n"
            "10:00 Pod entered Error state.\n\n"
            f"{checklist}\n"
        )
        warnings = Orchestrator._validate_investigation_checklist(output, output)
        assert len(warnings) == 1
        assert "Step 5" in warnings[0]

    def test_conditional_steps_all_validly_skipped(self) -> None:
        """Steps 4, 5, 6 all skipped with no contradicting evidence → no warnings."""
        steps = dict(_ALL_STEPS_COMPLETE)
        steps["4"] = (" ", "Deployment deep-dive (SKIPPED — all healthy)")
        steps["5"] = (" ", "Pod diagnostics (SKIPPED — all running)")
        steps["6"] = (" ", "Autoscaler check (SKIPPED — not configured)")
        checklist = _make_checklist(steps)
        # Clean output with no problem indicators
        output = (
            "## Cluster Overview\n"
            "All nodes healthy. CPU 30%, Memory 40%.\n\n"
            "## Service Status\n"
            "All services running. 3/3 replicas for each deployment.\n\n"
            f"{checklist}\n"
        )
        warnings = Orchestrator._validate_investigation_checklist(output, output)
        assert warnings == []


class TestChecklistRetryPromptIntegration:
    """Test that checklist warnings flow through to the retry prompt."""

    def _make_orchestrator(self) -> Orchestrator:
        return Orchestrator(_make_mock_client(), _make_mock_settings())

    def test_checklist_warnings_appear_in_retry_prompt(self) -> None:
        """When _validate_gatherer_output adds checklist warnings to
        shallow_sections, _build_retry_prompt renders them separately."""
        orch = self._make_orchestrator()
        shallow = [
            "Investigation Checklist: Step 4 was SKIPPED but output contains "
            "evidence matching '\\b0/\\d+' — this step should NOT be skipped.",
        ]
        prompt = orch._build_retry_prompt(
            "check health",
            [],
            shallow_sections=shallow,
        )
        assert "Investigation Checklist" in prompt
        assert "Step 4" in prompt
        assert "re-run the skipped steps" in prompt
        # Should NOT treat checklist warnings as regular shallow sections
        assert "insufficient data" not in prompt

    def test_checklist_warnings_mixed_with_regular_shallow(self) -> None:
        """Both regular shallow sections and checklist warnings render correctly."""
        orch = self._make_orchestrator()
        shallow = [
            "Cluster Overview",
            "Investigation Checklist: Step 5 was SKIPPED but output contains "
            "evidence matching 'CrashLoopBackOff' — this step should NOT be skipped.",
        ]
        prompt = orch._build_retry_prompt(
            "check health",
            [],
            shallow_sections=shallow,
        )
        # Regular shallow section feedback
        assert "insufficient data" in prompt
        assert "Cluster Overview" in prompt
        # Checklist feedback
        assert "Investigation Checklist" in prompt
        assert "Step 5" in prompt
        assert "re-run the skipped steps" in prompt

    def test_checklist_validation_triggers_retry_in_pipeline(self) -> None:
        """Integration: checklist invalid skip detected during validation
        triggers a retry with checklist-specific feedback."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        # Build output that passes section depth checks but has an invalid
        # checklist skip.  Step 4 is skipped yet output mentions "0/3".
        steps = dict(_ALL_STEPS_COMPLETE)
        steps["4"] = (" ", "Deployment deep-dive (SKIPPED — no issues)")
        checklist = _make_checklist(steps)
        # Strip the "### Investigation Checklist" header from _make_checklist
        # since the test already has "## Investigation Checklist" — a duplicate
        # heading would cause body extraction to stop too early.
        checklist_items = "\n".join(checklist.splitlines()[1:])
        # The Investigation Checklist section body must be >200 chars so
        # it passes the depth check — only then will the dedicated
        # checklist validator flag the invalid skip.
        checklist_padding = (
            "Investigation checklist summary: All mandatory steps were "
            "completed successfully. Conditional steps were evaluated "
            "based on cluster evidence collected during the diagnostic "
            "run. Steps with no trigger conditions were skipped as "
            "expected. See details below for each step."
        )
        first_pass = (
            f"## Cluster Overview\n{_CLUSTER_BODY}\n"
            f"## Service Status\n{_SERVICE_BODY}\n"
            "Details: app-frontend 0/3 replicas unavailable.\n"
            f"## Events Timeline\n{_EVENTS_BODY}\n"
            f"## Investigation Checklist\n{checklist_padding}\n\n"
            f"{checklist_items}\n"
        )

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1.execute.side_effect = [
            AgentResult(
                agent_name="gatherer",
                content=first_pass,
                success=True,
                usage={"total_tokens": 100},
            ),
            AgentResult(
                agent_name="gatherer",
                content=_make_complete_output(),
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
            return_value=[
                "Cluster Overview", "Service Status",
                "Events Timeline", "Investigation Checklist",
            ],
        )

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = orchestrator.execute_with_tools("check health", skill, registry)

        assert result.success is True
        # Gatherer was retried (checklist issue triggers needs_retry)
        assert agent1.execute.call_count == 2
        # Validation retry: reset IS called (not incremental deepening)
        agent1.reset.assert_called_once()
        retry_prompt = agent1.execute.call_args_list[1].args[0]
        # The retry prompt should include checklist feedback about Step 4
        assert "Step 4" in retry_prompt


# ===========================================================================
# P3 — Reporter output validation & retry
# ===========================================================================


class TestValidateReporterOutput:
    """Test Orchestrator._validate_reporter_output()."""

    def _make_orchestrator(self) -> Orchestrator:
        return Orchestrator(_make_mock_client(), _make_mock_settings())

    def test_clean_output_returns_empty(self) -> None:
        """Clean Markdown with proper tables and ending → no issues."""
        orch = self._make_orchestrator()
        output = (
            "# Report\n\n"
            "| Col1 | Col2 |\n"
            "|------|------|\n"
            "| a    | b    |\n\n"
            "All good."
        )
        assert orch._validate_reporter_output(output) == []

    def test_broken_table_row_detected(self) -> None:
        """Table row starting with | but not ending with | → detected."""
        orch = self._make_orchestrator()
        output = (
            "| Col1 | Col2 |\n"
            "|------|------|\n"
            "| a    | b\n"  # broken — no trailing |
            "| c    | d    |\n"
        )
        issues = orch._validate_reporter_output(output)
        assert any("Broken table row at line 3" in i for i in issues)

    def test_truncated_output_detected(self) -> None:
        """Output ending with a non-punctuation char → truncation warning."""
        orch = self._make_orchestrator()
        # Must be >100 chars to trigger truncation check
        output = (
            "# Service Health Report\n\n"
            "This report covers the cluster status including all nodes, "
            "pods, and services running in the default namespac"
        )
        issues = orch._validate_reporter_output(output)
        assert any("truncated" in i.lower() for i in issues)

    def test_unclosed_code_block_detected(self) -> None:
        """Odd number of ``` markers → unclosed code block."""
        orch = self._make_orchestrator()
        output = "```bash\nkubectl get pods\n\nDone."
        issues = orch._validate_reporter_output(output)
        assert any("Unclosed code block" in i for i in issues)

    def test_closed_code_block_ok(self) -> None:
        """Even number of ``` markers → no issue."""
        orch = self._make_orchestrator()
        output = "```bash\nkubectl get pods\n```\n\nDone."
        assert orch._validate_reporter_output(output) == []

    def test_multiple_issues_reported(self) -> None:
        """Multiple problems → all reported at once."""
        orch = self._make_orchestrator()
        output = (
            "| Col |\n"
            "|-----|\n"
            "| val\n"  # broken table row
            "```\n"  # opens code block
            "some code\n"
            "Truncated mid-wor"  # truncated + unclosed code block
        )
        issues = orch._validate_reporter_output(output)
        assert len(issues) >= 2  # at least broken row + unclosed code block

    def test_output_ending_with_backtick_ok(self) -> None:
        """Output ending with ``` is valid (closing code block)."""
        orch = self._make_orchestrator()
        output = "```bash\nkubectl get pods\n```"
        assert orch._validate_reporter_output(output) == []

    def test_output_ending_with_pipe_ok(self) -> None:
        """Output ending with | is valid (end of table)."""
        orch = self._make_orchestrator()
        output = (
            "| Col1 |\n"
            "|------|\n"
            "| val  |"
        )
        assert orch._validate_reporter_output(output) == []


class TestReporterRetryIntegration:
    """Test that reporter retry is triggered in execute_with_tools."""

    def _make_orchestrator(self) -> Orchestrator:
        return Orchestrator(_make_mock_client(), _make_mock_settings())

    def test_broken_table_triggers_retry(self) -> None:
        """Reporter output with broken table row triggers a retry."""
        orchestrator = self._make_orchestrator()
        registry = _make_mock_registry()

        broken_output = (
            "| Col1 | Col2 |\n"
            "|------|------|\n"
            "| a    | b\n"  # broken row
            "\nDone."
        )
        clean_output = (
            "| Col1 | Col2 |\n"
            "|------|------|\n"
            "| a    | b    |\n"
            "\nDone."
        )

        # Gatherer agent — passes validation (no required_sections)
        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1.execute.return_value = AgentResult(
            agent_name="gatherer",
            content=_make_complete_output(),
            success=True,
            usage={"total_tokens": 100},
        )

        # Reporter agent — first call returns broken, second returns clean
        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "reporter"
        agent2.role = "Reporter"
        agent2.execute.side_effect = [
            AgentResult(
                agent_name="reporter", content=broken_output, success=True,
                usage={"total_tokens": 50},
            ),
            AgentResult(
                agent_name="reporter", content=clean_output, success=True,
                usage={"total_tokens": 60},
            ),
        ]

        skill = StubToolSkill()
        # No required output sections → gatherer validation won't trigger
        skill.get_required_output_sections = MagicMock(return_value=[])

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = orchestrator.execute_with_tools("check health", skill, registry)

        assert result.success is True
        # Reporter was retried
        assert agent2.execute.call_count == 2
        assert agent2.reset.call_count == 1
        # Final output is the clean one
        assert "| b    |" in result.synthesized_output

    def test_max_tokens_triggers_retry(self) -> None:
        """finish_reason=MAX_TOKENS on reporter triggers a retry."""
        orchestrator = self._make_orchestrator()
        registry = _make_mock_registry()

        truncated_output = "# Report\n\nThis is truncat"
        clean_output = "# Report\n\nThis is complete."

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1.execute.return_value = AgentResult(
            agent_name="gatherer",
            content=_make_complete_output(),
            success=True,
            usage={"total_tokens": 100},
        )

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "reporter"
        agent2.role = "Reporter"
        agent2.execute.side_effect = [
            AgentResult(
                agent_name="reporter", content=truncated_output, success=True,
                usage={"total_tokens": 50},
                metadata={"finish_reason": "MAX_TOKENS"},
            ),
            AgentResult(
                agent_name="reporter", content=clean_output, success=True,
                usage={"total_tokens": 60},
            ),
        ]

        skill = StubToolSkill()
        skill.get_required_output_sections = MagicMock(return_value=[])

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = orchestrator.execute_with_tools("check health", skill, registry)

        assert result.success is True
        assert agent2.execute.call_count == 2
        assert "complete." in result.synthesized_output

    def test_clean_output_no_retry(self) -> None:
        """Clean reporter output → no retry."""
        orchestrator = self._make_orchestrator()
        registry = _make_mock_registry()

        clean_output = (
            "| Col1 | Col2 |\n"
            "|------|------|\n"
            "| a    | b    |\n\n"
            "All good."
        )

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1.execute.return_value = AgentResult(
            agent_name="gatherer",
            content=_make_complete_output(),
            success=True,
            usage={"total_tokens": 100},
        )

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "reporter"
        agent2.role = "Reporter"
        agent2.execute.return_value = AgentResult(
            agent_name="reporter", content=clean_output, success=True,
            usage={"total_tokens": 50},
        )

        skill = StubToolSkill()
        skill.get_required_output_sections = MagicMock(return_value=[])

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = orchestrator.execute_with_tools("check health", skill, registry)

        assert result.success is True
        # Reporter was NOT retried
        assert agent2.execute.call_count == 1
        assert agent2.reset.call_count == 0


# ===========================================================================
# Causa 4 — Tool call metadata injected into sequential pipeline context
# ===========================================================================


class TestBuildToolsSummary:
    """Test the _build_tools_summary() helper function."""

    def test_metadata_with_tools_executed(self) -> None:
        """When metadata has tools_executed, returns a Tools Executed section."""
        metadata: dict[str, Any] = {
            "tools_executed": [
                {"name": "kubectl_get", "args": {"resource": "pods"}, "output": "OK", "error": False},
                {"name": "kubectl_describe", "args": {"resource": "pod/x"}, "output": "Details", "error": False},
                {"name": "kubectl_get", "args": {"resource": "svc"}, "output": "OK", "error": False},
            ],
        }
        summary = _build_tools_summary("Gatherer", metadata)
        assert "## Tools Executed by Gatherer" in summary
        assert "Total tool calls: 3" in summary
        assert "kubectl_describe, kubectl_get" in summary  # sorted, unique

    def test_metadata_with_failed_tools(self) -> None:
        """When metadata has failed tools, includes failure details."""
        metadata: dict[str, Any] = {
            "tools_executed": [
                {"name": "kubectl_get", "args": {"resource": "pods"}, "output": "OK", "error": False},
                {
                    "name": "kubectl_logs",
                    "args": {"pod": "my-pod"},
                    "output": "Error: container not found in pod my-pod",
                    "error": True,
                },
            ],
        }
        summary = _build_tools_summary("Gatherer", metadata)
        assert "Failed calls: 1" in summary
        assert "kubectl_logs" in summary
        assert "tool gaps" in summary
        assert "data gaps" in summary

    def test_metadata_none_returns_empty(self) -> None:
        """When metadata is None, returns empty string."""
        summary = _build_tools_summary("Gatherer", None)
        assert summary == ""

    def test_metadata_empty_dict_returns_empty(self) -> None:
        """When metadata is empty dict, returns empty string."""
        summary = _build_tools_summary("Gatherer", {})
        assert summary == ""

    def test_metadata_no_tools_executed_key(self) -> None:
        """When metadata exists but has no tools_executed key, returns empty."""
        summary = _build_tools_summary("Gatherer", {"finish_reason": "STOP"})
        assert summary == ""

    def test_metadata_empty_tools_list(self) -> None:
        """When tools_executed is an empty list, returns empty string."""
        summary = _build_tools_summary("Gatherer", {"tools_executed": []})
        assert summary == ""

    def test_failed_tool_output_truncated_to_80_chars(self) -> None:
        """Failed tool output is truncated to 80 characters in summary."""
        long_output = "A" * 200
        metadata: dict[str, Any] = {
            "tools_executed": [
                {"name": "kubectl_get", "args": {}, "output": long_output, "error": True},
            ],
        }
        summary = _build_tools_summary("Worker", metadata)
        # The truncated portion should be at most 80 chars
        # Find the output after "→ " in the summary
        arrow_idx = summary.index("→ ")
        after_arrow = summary[arrow_idx + 2:]
        # Up to the newline
        truncated_output = after_arrow.split("\n")[0]
        assert len(truncated_output) <= 80

    def test_unique_tools_sorted(self) -> None:
        """Unique tools are listed in sorted order."""
        metadata: dict[str, Any] = {
            "tools_executed": [
                {"name": "z_tool", "args": {}, "output": "ok", "error": False},
                {"name": "a_tool", "args": {}, "output": "ok", "error": False},
                {"name": "m_tool", "args": {}, "output": "ok", "error": False},
                {"name": "a_tool", "args": {}, "output": "ok", "error": False},
            ],
        }
        summary = _build_tools_summary("Agent", metadata)
        assert "a_tool, m_tool, z_tool" in summary

    def test_failed_tool_with_none_output_uses_fallback(self) -> None:
        """Failed tool with output=None uses 'error' as fallback text."""
        metadata: dict[str, Any] = {
            "tools_executed": [
                {"name": "kubectl_get", "args": {}, "output": None, "error": True},
            ],
        }
        summary = _build_tools_summary("Worker", metadata)
        assert "Failed calls: 1" in summary
        assert "error" in summary


class TestToolMetadataInSequentialPipeline:
    """Integration: verify tool metadata is injected in sequential context."""

    def test_sync_execute_with_tools_injects_metadata(self) -> None:
        """Sync sequential pipeline injects tool metadata in downstream context."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1.execute.return_value = AgentResult(
            agent_name="gatherer",
            content="Gathered data here.",
            success=True,
            usage={"total_tokens": 100},
            metadata={
                "tools_executed": [
                    {"name": "kubectl_get", "args": {"resource": "pods"}, "output": "OK", "error": False},
                    {"name": "kubectl_logs", "args": {"pod": "x"}, "output": "Error: not found", "error": True},
                ],
            },
        )

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "reporter"
        agent2.role = "Reporter"
        agent2.execute.return_value = AgentResult(
            agent_name="reporter", content="Report.", success=True,
            usage={"total_tokens": 50},
        )

        skill = StubToolSkill()
        # No required sections to skip gatherer validation retry
        skill.get_required_output_sections = MagicMock(return_value=[])

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = orchestrator.execute_with_tools("check pods", skill, registry)

        assert result.success is True
        # Second agent should receive context with tool metadata
        second_call_context = agent2.execute.call_args.kwargs["context"]
        assert "## Tools Executed by Gatherer" in second_call_context
        assert "Total tool calls: 2" in second_call_context
        assert "kubectl_get, kubectl_logs" in second_call_context
        assert "Failed calls: 1" in second_call_context
        assert "tool gaps" in second_call_context

    def test_sync_no_metadata_no_tools_summary(self) -> None:
        """When agent result has no metadata, no tools summary is added."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1.execute.return_value = AgentResult(
            agent_name="gatherer",
            content="Gathered data.",
            success=True,
            usage={"total_tokens": 100},
            # No metadata → default empty dict
        )

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "reporter"
        agent2.role = "Reporter"
        agent2.execute.return_value = AgentResult(
            agent_name="reporter", content="Report.", success=True,
            usage={"total_tokens": 50},
        )

        skill = StubToolSkill()
        skill.get_required_output_sections = MagicMock(return_value=[])

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = orchestrator.execute_with_tools("check pods", skill, registry)

        assert result.success is True
        second_call_context = agent2.execute.call_args.kwargs["context"]
        assert "Tools Executed" not in second_call_context

    async def test_async_execute_with_tools_injects_metadata(self) -> None:
        """Async sequential pipeline injects tool metadata in downstream context."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1.execute.return_value = AgentResult(
            agent_name="gatherer",
            content="Gathered data here.",
            success=True,
            usage={"total_tokens": 100},
            metadata={
                "tools_executed": [
                    {"name": "kubectl_get", "args": {"resource": "pods"}, "output": "OK", "error": False},
                    {"name": "gcloud_logs", "args": {}, "output": "Logs retrieved", "error": False},
                ],
            },
        )

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "reporter"
        agent2.role = "Reporter"
        agent2.execute.return_value = AgentResult(
            agent_name="reporter", content="Report.", success=True,
            usage={"total_tokens": 50},
        )

        skill = StubToolSkill()
        skill.get_required_output_sections = MagicMock(return_value=[])

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = await orchestrator.async_execute_with_tools(
                "check pods", skill, registry,
            )

        assert result.success is True
        second_call_context = agent2.execute.call_args.kwargs["context"]
        assert "## Tools Executed by Gatherer" in second_call_context
        assert "Total tool calls: 2" in second_call_context
        assert "gcloud_logs, kubectl_get" in second_call_context
        # No failures in this test
        assert "Failed calls" not in second_call_context

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1.execute.return_value = AgentResult(
            agent_name="gatherer",
            content="Gathered data.",
            success=True,
            usage={"total_tokens": 100},
        )

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "reporter"
        agent2.role = "Reporter"
        agent2.execute.return_value = AgentResult(
            agent_name="reporter", content="Report.", success=True,
            usage={"total_tokens": 50},
        )

        skill = StubToolSkill()
        skill.get_required_output_sections = MagicMock(return_value=[])

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = await orchestrator.async_execute_with_tools(
                "check pods", skill, registry,
            )

        assert result.success is True
        second_call_context = agent2.execute.call_args.kwargs["context"]
        assert "Tools Executed" not in second_call_context


# ===========================================================================
# Spec 1 — Tool Metadata Injection: additional coverage
# ===========================================================================


class TestToolMetadataInPlainExecute:
    """Verify _build_tools_summary() works via the plain execute_sequential() path.

    execute_sequential() (line ~197 in orchestrator.py) calls agent.execute()
    directly — NOT through execute_with_tools().  These tests verify that
    tool metadata flows through that injection point too.
    """

    def test_plain_execute_sequential_injects_metadata(self) -> None:
        """Sync execute_sequential injects tool summary when agent.metadata
        contains tools_executed."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        agent1 = MagicMock(spec=SpecialistAgent)
        agent1.name = "infra-worker"
        agent1.role = "Infrastructure Worker"
        agent1.execute.return_value = AgentResult(
            agent_name="infra-worker",
            content="Found 3 pods running.",
            success=True,
            usage={"total_tokens": 50},
            metadata={
                "tools_executed": [
                    {"name": "kubectl_get", "args": {"resource": "pods"}, "output": "OK", "error": False},
                    {"name": "kubectl_describe", "args": {"resource": "node/gke-1"}, "output": "Details", "error": False},
                ],
            },
        )

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "report-writer"
        agent2.role = "Report Writer"
        agent2.execute.return_value = AgentResult(
            agent_name="report-writer",
            content="Infrastructure Report: All healthy.",
            success=True,
            usage={"total_tokens": 40},
        )

        skill = StubToolSkill()

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = orchestrator.execute_sequential(
                skill, SkillPhase.ANALYZE, "ctx", "check infra",
            )

        assert result.success is True
        # Second agent should have received tool metadata in its context
        second_call_context = agent2.execute.call_args.kwargs["context"]
        assert "## Tools Executed by Infrastructure Worker" in second_call_context
        assert "Total tool calls: 2" in second_call_context
        assert "kubectl_describe, kubectl_get" in second_call_context

    def test_plain_execute_sequential_no_metadata_no_summary(self) -> None:
        """Sync execute_sequential: no metadata → no tools summary injected."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        agent1 = MagicMock(spec=SpecialistAgent)
        agent1.name = "worker"
        agent1.role = "Worker"
        agent1.execute.return_value = AgentResult(
            agent_name="worker",
            content="Data collected.",
            success=True,
            usage={"total_tokens": 30},
        )

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "writer"
        agent2.role = "Writer"
        agent2.execute.return_value = AgentResult(
            agent_name="writer",
            content="Report.",
            success=True,
            usage={"total_tokens": 20},
        )

        skill = StubToolSkill()

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = orchestrator.execute_sequential(
                skill, SkillPhase.ANALYZE, "ctx", "check",
            )

        assert result.success is True
        second_call_context = agent2.execute.call_args.kwargs["context"]
        assert "Tools Executed" not in second_call_context


class TestToolMetadataInPlainAsyncExecute:
    """Verify _build_tools_summary() works via the async_execute_sequential() path.

    async_execute_sequential() (line ~882 in orchestrator.py) calls
    agent.execute() via asyncio.to_thread — NOT through
    async_execute_with_tools().
    """

    async def test_async_execute_sequential_injects_metadata(self) -> None:
        """Async execute_sequential injects tool summary when agent.metadata
        contains tools_executed."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        agent1 = MagicMock(spec=SpecialistAgent)
        agent1.name = "gatherer"
        agent1.role = "Data Gatherer"
        agent1.execute.return_value = AgentResult(
            agent_name="gatherer",
            content="Collected cluster metrics.",
            success=True,
            usage={"total_tokens": 60},
            metadata={
                "tools_executed": [
                    {"name": "gcloud_monitoring", "args": {"metric": "cpu"}, "output": "45%", "error": False},
                    {"name": "kubectl_top", "args": {"resource": "nodes"}, "output": "node stats", "error": False},
                    {"name": "gcloud_monitoring", "args": {"metric": "memory"}, "output": "60%", "error": False},
                ],
            },
        )

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "reporter"
        agent2.role = "Reporter"
        agent2.execute.return_value = AgentResult(
            agent_name="reporter",
            content="Cluster Metrics Report.",
            success=True,
            usage={"total_tokens": 30},
        )

        skill = StubToolSkill()

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = await orchestrator.async_execute_sequential(
                skill, SkillPhase.ANALYZE, "ctx", "check metrics",
            )

        assert result.success is True
        # Second agent should have received tool metadata in its context
        second_call_context = agent2.execute.call_args.kwargs["context"]
        assert "## Tools Executed by Data Gatherer" in second_call_context
        assert "Total tool calls: 3" in second_call_context
        assert "gcloud_monitoring, kubectl_top" in second_call_context

    async def test_async_execute_sequential_no_metadata_no_summary(self) -> None:
        """Async execute_sequential: no metadata → no tools summary injected."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        agent1 = MagicMock(spec=SpecialistAgent)
        agent1.name = "worker"
        agent1.role = "Worker"
        agent1.execute.return_value = AgentResult(
            agent_name="worker",
            content="Data.",
            success=True,
            usage={"total_tokens": 20},
        )

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "writer"
        agent2.role = "Writer"
        agent2.execute.return_value = AgentResult(
            agent_name="writer",
            content="Report.",
            success=True,
            usage={"total_tokens": 15},
        )

        skill = StubToolSkill()

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = await orchestrator.async_execute_sequential(
                skill, SkillPhase.ANALYZE, "ctx", "check",
            )

        assert result.success is True
        second_call_context = agent2.execute.call_args.kwargs["context"]
        assert "Tools Executed" not in second_call_context


class TestBuildToolsSummaryMultipleFailures:
    """Verify _build_tools_summary() correctly lists ALL failed tools."""

    def test_multiple_failed_tools_all_listed(self) -> None:
        """When multiple tools fail, the summary lists every failure."""
        metadata: dict[str, Any] = {
            "tools_executed": [
                {"name": "kubectl_get", "args": {"resource": "pods"}, "output": "OK", "error": False},
                {
                    "name": "kubectl_logs",
                    "args": {"pod": "app-1"},
                    "output": "Error: container not found",
                    "error": True,
                },
                {
                    "name": "kubectl_exec",
                    "args": {"pod": "app-2", "command": "ls"},
                    "output": "Error: pod not running",
                    "error": True,
                },
                {
                    "name": "gcloud_logs",
                    "args": {"filter": "severity>=ERROR"},
                    "output": "Permission denied",
                    "error": True,
                },
            ],
        }
        summary = _build_tools_summary("Gatherer", metadata)

        # All 3 failed tools should be counted
        assert "Failed calls: 3" in summary

        # Each failed tool name should appear in the failure details
        assert "kubectl_logs" in summary
        assert "kubectl_exec" in summary
        assert "gcloud_logs" in summary

        # Error outputs should be included (truncated)
        assert "container not found" in summary
        assert "pod not running" in summary
        assert "Permission denied" in summary

        # The tool gaps note should appear once
        assert "tool gaps" in summary
        assert "data gaps" in summary

    def test_multiple_failed_tools_includes_args(self) -> None:
        """Failed tool entries include their args in the summary."""
        metadata: dict[str, Any] = {
            "tools_executed": [
                {
                    "name": "kubectl_logs",
                    "args": {"pod": "my-pod"},
                    "output": "Error: not found",
                    "error": True,
                },
                {
                    "name": "kubectl_describe",
                    "args": {"resource": "deploy/frontend"},
                    "output": "Error: not found",
                    "error": True,
                },
            ],
        }
        summary = _build_tools_summary("Worker", metadata)

        assert "Failed calls: 2" in summary
        # Args are included in the format: name({args}) → output
        assert "kubectl_logs(" in summary
        assert "'pod': 'my-pod'" in summary
        assert "kubectl_describe(" in summary
        assert "'resource': 'deploy/frontend'" in summary

    def test_all_tools_failed(self) -> None:
        """When ALL tools fail, summary still correctly reports."""
        metadata: dict[str, Any] = {
            "tools_executed": [
                {"name": "tool_a", "args": {}, "output": "Error A", "error": True},
                {"name": "tool_b", "args": {}, "output": "Error B", "error": True},
            ],
        }
        summary = _build_tools_summary("Agent", metadata)

        assert "Total tool calls: 2" in summary
        assert "Failed calls: 2" in summary
        assert "tool_a" in summary
        assert "tool_b" in summary


class TestBuildToolsSummaryEmptyStringOutput:
    """Verify _build_tools_summary() handles empty string output gracefully."""

    def test_successful_tool_with_empty_output(self) -> None:
        """Tool succeeding with output='' is NOT treated as a failure."""
        metadata: dict[str, Any] = {
            "tools_executed": [
                {"name": "kubectl_get", "args": {"resource": "pods"}, "output": "", "error": False},
                {"name": "kubectl_top", "args": {"resource": "nodes"}, "output": "node stats", "error": False},
            ],
        }
        summary = _build_tools_summary("Gatherer", metadata)

        assert "Total tool calls: 2" in summary
        assert "kubectl_get, kubectl_top" in summary
        # No failures — empty output on a successful tool is fine
        assert "Failed calls" not in summary

    def test_failed_tool_with_empty_output_uses_fallback(self) -> None:
        """Failed tool with output='' uses 'error' fallback (same as None)."""
        metadata: dict[str, Any] = {
            "tools_executed": [
                {"name": "kubectl_logs", "args": {"pod": "x"}, "output": "", "error": True},
            ],
        }
        summary = _build_tools_summary("Worker", metadata)

        assert "Failed calls: 1" in summary
        assert "kubectl_logs" in summary
        # Empty string is falsy, so (output or 'error') → 'error'
        assert "error" in summary

    def test_empty_output_in_pipeline_context(self) -> None:
        """Tool with empty-string output flows through the pipeline correctly.

        The downstream agent should receive a valid tools summary even when
        a tool returned an empty string.
        """
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1.execute.return_value = AgentResult(
            agent_name="gatherer",
            content="Gathered data.",
            success=True,
            usage={"total_tokens": 100},
            metadata={
                "tools_executed": [
                    {"name": "kubectl_get", "args": {"resource": "configmaps"}, "output": "", "error": False},
                    {"name": "kubectl_get", "args": {"resource": "pods"}, "output": "3 pods found", "error": False},
                ],
            },
        )

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "reporter"
        agent2.role = "Reporter"
        agent2.execute.return_value = AgentResult(
            agent_name="reporter", content="Report.", success=True,
            usage={"total_tokens": 50},
        )

        skill = StubToolSkill()
        skill.get_required_output_sections = MagicMock(return_value=[])

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = orchestrator.execute_with_tools("check pods", skill, registry)

        assert result.success is True
        second_call_context = agent2.execute.call_args.kwargs["context"]
        # Tools summary should be present and correct
        assert "## Tools Executed by Gatherer" in second_call_context
        assert "Total tool calls: 2" in second_call_context
        # No failures — empty output is NOT an error
        assert "Failed calls" not in second_call_context

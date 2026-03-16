"""Tests for deepening non-fatal behavior (MaxIterationsError fallback).

When the first pass produces valid output and the optional deepening pass
hits MaxIterationsError, the pipeline should fall back to first-pass output
instead of crashing.  When it's a genuine retry (validation failed),
MaxIterationsError should still propagate.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vaig.agents.base import AgentResult
from vaig.agents.orchestrator import Orchestrator
from vaig.agents.specialist import SpecialistAgent
from vaig.agents.tool_aware import ToolAwareAgent
from vaig.core.client import GenerationResult
from vaig.core.exceptions import MaxIterationsError, VAIGError
from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.tools.base import ToolRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Section body with >200 chars so validation passes the depth check
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


def _make_complete_output() -> str:
    """Build gatherer output that passes validation (all sections >200 chars)."""
    parts = [
        f"## Cluster Overview\n{_CLUSTER_BODY}\n",
        f"## Service Status\n{_SERVICE_BODY}\n",
        f"## Events Timeline\n{_EVENTS_BODY}\n",
        f"### Investigation Checklist\n{_INVESTIGATION_CHECKLIST_BODY}\n",
    ]
    return "\n".join(parts)


def _make_mock_client() -> MagicMock:
    client = MagicMock()
    client.generate.return_value = GenerationResult(
        text="Agent response",
        model="gemini-2.5-pro",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        finish_reason="STOP",
    )
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
# Deepening non-fatal — MaxIterationsError during deepening falls back
# ===========================================================================


class TestDeepeningNonFatal:
    """Test that MaxIterationsError during deepening second pass is non-fatal.

    When the first pass produces valid output and the deepening pass hits
    MaxIterationsError, the pipeline should fall back to first-pass output
    instead of crashing.  When it's a genuine retry (validation failed),
    MaxIterationsError should still propagate.
    """

    def test_sync_deepening_max_iterations_falls_back(self) -> None:
        """Sync: deepening pass hits MaxIterationsError -> falls back to
        first-pass output, pipeline continues successfully."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        complete_output = _make_complete_output()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1._max_iterations = 15

        # First call: valid output. Second call: MaxIterationsError
        agent1.execute.side_effect = [
            AgentResult(
                agent_name="gatherer",
                content=complete_output,
                success=True,
                usage={"total_tokens": 100},
            ),
            MaxIterationsError(
                "Tool-use loop exceeded 10 iterations", iterations=10,
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

        # Pipeline should succeed — fell back to first-pass output
        assert result.success is True
        # The gatherer result should be the FIRST pass output (not merged)
        assert result.agent_results[0].content == complete_output
        # Reporter should still run
        assert agent2.execute.call_count == 1
        # max_iterations should be restored
        assert agent1._max_iterations == 15

    def test_sync_validation_retry_max_iterations_propagates(self) -> None:
        """Sync: genuine retry (validation failed) hitting MaxIterationsError
        should propagate the error — NOT be caught."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1._max_iterations = 15

        # First call: incomplete output (validation fails).
        # Second call: MaxIterationsError
        agent1.execute.side_effect = [
            AgentResult(
                agent_name="gatherer",
                content="Some random text without required sections",
                success=True,
                usage={"total_tokens": 50},
            ),
            MaxIterationsError(
                "Tool-use loop exceeded 10 iterations", iterations=10,
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
            # Should raise because it's a genuine retry, not a deepening
            with pytest.raises(VAIGError):
                orchestrator.execute_with_tools("check health", skill, registry)

        # max_iterations should still be restored (finally block)
        assert agent1._max_iterations == 15

    def test_sync_deepening_fallback_preserves_first_pass_result(self) -> None:
        """Sync: when deepening falls back, result.agent_results[-1] is the
        first-pass result (not replaced by a failed retry)."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        complete_output = _make_complete_output()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1._max_iterations = 15
        agent1.execute.side_effect = [
            AgentResult(
                agent_name="gatherer",
                content=complete_output,
                success=True,
                usage={"total_tokens": 200},
                metadata={"finish_reason": "STOP"},
            ),
            MaxIterationsError("limit hit", iterations=10),
        ]

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "reporter"
        agent2.role = "Reporter"
        agent2.execute.return_value = AgentResult(
            agent_name="reporter", content="Report.", success=True,
            usage={"total_tokens": 50},
        )

        skill = StubToolSkill()
        skill.get_required_output_sections = MagicMock(
            return_value=["Cluster Overview", "Service Status", "Events Timeline"],
        )

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = orchestrator.execute_with_tools("check health", skill, registry)

        # First-pass result is preserved
        gatherer_result = result.agent_results[0]
        assert gatherer_result.content == complete_output
        assert gatherer_result.success is True
        # Reporter ran with the first-pass output
        reporter_context = agent2.execute.call_args.kwargs["context"]
        assert complete_output in reporter_context

    async def test_async_deepening_max_iterations_falls_back(self) -> None:
        """Async: deepening pass hits MaxIterationsError -> falls back to
        first-pass output, pipeline continues successfully."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        complete_output = _make_complete_output()

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
            MaxIterationsError("limit exceeded", iterations=10),
        ]

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "reporter"
        agent2.role = "Reporter"
        agent2.execute.return_value = AgentResult(
            agent_name="reporter", content="Async report", success=True,
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
        assert result.agent_results[0].content == complete_output
        assert agent2.execute.call_count == 1
        assert agent1._max_iterations == 15

    async def test_async_validation_retry_max_iterations_propagates(self) -> None:
        """Async: genuine retry hitting MaxIterationsError should propagate."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "gatherer"
        agent1.role = "Gatherer"
        agent1._max_iterations = 15
        agent1.execute.side_effect = [
            AgentResult(
                agent_name="gatherer",
                content="no sections here",
                success=True,
                usage={"total_tokens": 50},
            ),
            MaxIterationsError("limit exceeded", iterations=10),
        ]

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "reporter"
        agent2.role = "Reporter"

        skill = StubToolSkill()
        skill.get_required_output_sections = MagicMock(
            return_value=["Cluster Overview", "Service Status", "Events Timeline"],
        )

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            with pytest.raises(VAIGError):
                await orchestrator.async_execute_with_tools(
                    "check health", skill, registry,
                )

        assert agent1._max_iterations == 15

    def test_sync_deepening_no_error_still_merges(self) -> None:
        """Sync: when deepening succeeds (no MaxIterationsError), the
        content is merged as before — verifying the fix doesn't break
        the happy path."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        complete_output = _make_complete_output()
        second_pass = "### New Findings\nAdditional insights."

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
                content=second_pass,
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
        # Content should be MERGED (first + second)
        merged = result.agent_results[0].content
        assert complete_output in merged
        assert second_pass in merged
        assert f"{complete_output}\n\n{second_pass}" == merged

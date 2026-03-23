"""Unit tests for Batch 3 — Orchestrator state threading (Phase 3).

Covers:
- execute_sequential() threads state through agents and returns final_state (Task 3.1)
- execute_fanout() threads state and applies patches from all agents (Task 3.2)
- _execute_with_tools_impl() threads state for all strategies (Task 3.3-3.5)
- OrchestratorResult.final_state is None when skill returns no initial state (Task 3.6)
- Backward compatibility: None state is a no-op (Task 3.7)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vaig.agents.base import AgentResult
from vaig.agents.orchestrator import Orchestrator, OrchestratorResult
from vaig.agents.specialist import SpecialistAgent
from vaig.agents.tool_aware import ToolAwareAgent
from vaig.core.models import PipelineState, apply_state_patch
from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.tools.base import ToolRegistry

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_mock_client() -> MagicMock:
    client = MagicMock()
    client.generate_content.return_value = MagicMock(text="ok")
    return client


def _make_mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.budget.max_cost_per_run = 0.0
    settings.agents.max_failures_before_fallback = 3
    settings.agents.max_agent_retries = 0
    settings.agents.specialist_model = "gemini-1.5-flash"
    settings.agents.tool_aware_model = "gemini-1.5-flash"
    settings.agents.base_model = "gemini-1.5-flash"
    return settings


def _make_mock_registry() -> MagicMock:
    return MagicMock(spec=ToolRegistry)


class _NoStateSkill(BaseSkill):
    """Skill that returns no initial state (backward-compat path)."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(name="no-state", display_name="No State", description="test")

    def get_system_instruction(self) -> str:
        return "system"

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return user_input


class _StateSkill(BaseSkill):
    """Skill that returns a real initial PipelineState."""

    def __init__(self, initial: PipelineState) -> None:
        super().__init__()
        self._initial = initial

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(name="state-skill", display_name="State Skill", description="test")

    def get_system_instruction(self) -> str:
        return "system"

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return user_input

    def get_initial_state(self) -> PipelineState | None:
        return self._initial


def _make_agent_result(
    name: str,
    *,
    state_patch: dict | PipelineState | None = None,
    success: bool = True,
) -> AgentResult:
    return AgentResult(
        agent_name=name,
        content=f"output from {name}",
        success=success,
        usage={"total_tokens": 5},
        state_patch=state_patch,
    )


# ── execute_sequential() ─────────────────────────────────────────────────────


class TestExecuteSequentialState:
    """Task 3.1 — execute_sequential() threads state through agents."""

    def test_state_passed_to_first_agent(self) -> None:
        """First agent receives the skill's initial state."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        initial_state = PipelineState(metrics={"step": "start"})
        skill = _StateSkill(initial_state)

        agent = MagicMock(spec=SpecialistAgent)
        agent.name = "agent-1"
        agent.role = "Analyst"
        agent.model = "gemini-1.5-flash"
        agent.execute.return_value = _make_agent_result("agent-1")

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent]):
            orchestrator.execute_sequential(
                skill, SkillPhase.ANALYZE, "ctx", "input"
            )

        call_kwargs = agent.execute.call_args.kwargs
        assert call_kwargs.get("state") == initial_state

    def test_state_patch_applied_between_agents(self) -> None:
        """Second agent receives state with first agent's patch applied."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        initial_state = PipelineState(metrics={"count": 0})
        skill = _StateSkill(initial_state)

        agent1 = MagicMock(spec=SpecialistAgent)
        agent1.name = "agent-1"
        agent1.role = "First"
        agent1.model = "gemini-1.5-flash"
        # agent1 patches state → count=1 via metrics merge
        agent1.execute.return_value = _make_agent_result(
            "agent-1", state_patch=PipelineState(metrics={"count": 1})
        )

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "agent-2"
        agent2.role = "Second"
        agent2.model = "gemini-1.5-flash"
        agent2.execute.return_value = _make_agent_result("agent-2")

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            orchestrator.execute_sequential(
                skill, SkillPhase.ANALYZE, "ctx", "input"
            )

        agent2_state = agent2.execute.call_args.kwargs.get("state")
        assert agent2_state is not None
        assert agent2_state.metrics["count"] == 1

    def test_final_state_stored_on_result(self) -> None:
        """OrchestratorResult.final_state holds the accumulated state."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        initial_state = PipelineState(errors=[])
        skill = _StateSkill(initial_state)

        agent = MagicMock(spec=SpecialistAgent)
        agent.name = "agent-1"
        agent.role = "Analyst"
        agent.model = "gemini-1.5-flash"
        agent.execute.return_value = _make_agent_result(
            "agent-1", state_patch=PipelineState(errors=["done"])
        )

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent]):
            result = orchestrator.execute_sequential(
                skill, SkillPhase.ANALYZE, "ctx", "input"
            )

        assert result.final_state is not None
        assert "done" in result.final_state.errors

    def test_no_initial_state_skips_patch(self) -> None:
        """When skill returns None initial state, final_state stays None."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        skill = _NoStateSkill()

        agent = MagicMock(spec=SpecialistAgent)
        agent.name = "agent-1"
        agent.role = "Analyst"
        agent.model = "gemini-1.5-flash"
        # Even if agent returns a patch, it should be ignored when state is None
        agent.execute.return_value = _make_agent_result(
            "agent-1", state_patch=PipelineState(metrics={"x": 1})
        )

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent]):
            result = orchestrator.execute_sequential(
                skill, SkillPhase.ANALYZE, "ctx", "input"
            )

        assert result.final_state is None

    def test_no_initial_state_passes_none_to_agent(self) -> None:
        """When skill returns None state, agents receive state=None."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        skill = _NoStateSkill()

        agent = MagicMock(spec=SpecialistAgent)
        agent.name = "agent-1"
        agent.role = "Analyst"
        agent.model = "gemini-1.5-flash"
        agent.execute.return_value = _make_agent_result("agent-1")

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent]):
            orchestrator.execute_sequential(skill, SkillPhase.ANALYZE, "ctx", "input")

        call_kwargs = agent.execute.call_args.kwargs
        assert call_kwargs.get("state") is None


# ── execute_fanout() ─────────────────────────────────────────────────────────


class TestExecuteFanoutState:
    """Task 3.2 — execute_fanout() threads state and applies all patches."""

    def test_all_agents_receive_initial_state(self) -> None:
        """Every agent in a fanout receives the same initial state."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        initial_state = PipelineState(metrics={"key": "value"})
        skill = _StateSkill(initial_state)

        agent1 = MagicMock(spec=SpecialistAgent)
        agent1.name = "a1"
        agent1.role = "R1"
        agent1.model = "gemini-1.5-flash"
        agent1.execute.return_value = _make_agent_result("a1")

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "a2"
        agent2.role = "R2"
        agent2.model = "gemini-1.5-flash"
        agent2.execute.return_value = _make_agent_result("a2")

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            orchestrator.execute_fanout(skill, SkillPhase.ANALYZE, "ctx", "input")

        assert agent1.execute.call_args.kwargs.get("state") == initial_state
        assert agent2.execute.call_args.kwargs.get("state") == initial_state

    def test_patches_from_all_agents_applied_to_final_state(self) -> None:
        """All agent patches are sequentially applied to produce final_state."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        initial_state = PipelineState(metrics={"a": 0, "b": 0})
        skill = _StateSkill(initial_state)

        agent1 = MagicMock(spec=SpecialistAgent)
        agent1.name = "a1"
        agent1.role = "R1"
        agent1.model = "gemini-1.5-flash"
        agent1.execute.return_value = _make_agent_result(
            "a1", state_patch=PipelineState(metrics={"a": 1})
        )

        agent2 = MagicMock(spec=SpecialistAgent)
        agent2.name = "a2"
        agent2.role = "R2"
        agent2.model = "gemini-1.5-flash"
        agent2.execute.return_value = _make_agent_result(
            "a2", state_patch=PipelineState(metrics={"b": 2})
        )

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1, agent2]):
            result = orchestrator.execute_fanout(skill, SkillPhase.ANALYZE, "ctx", "input")

        assert result.final_state is not None
        # Both patches merged — both keys should be present with updated values
        assert result.final_state.metrics.get("a") == 1
        assert result.final_state.metrics.get("b") == 2

    def test_fanout_no_state_returns_none_final_state(self) -> None:
        """Fanout with no initial state results in final_state=None."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        skill = _NoStateSkill()

        agent = MagicMock(spec=SpecialistAgent)
        agent.name = "a1"
        agent.role = "R1"
        agent.model = "gemini-1.5-flash"
        agent.execute.return_value = _make_agent_result("a1")

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent]):
            result = orchestrator.execute_fanout(skill, SkillPhase.ANALYZE, "ctx", "input")

        assert result.final_state is None


# ── execute_with_tools() ─────────────────────────────────────────────────────


class StubToolSkill(_NoStateSkill):
    """Minimal skill for execute_with_tools() tests without initial state."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(name="tool-skill", display_name="Tool Skill", description="test")

    def requires_tools(self) -> bool:
        return True


class StubToolSkillWithState(_StateSkill):
    """Skill for execute_with_tools() tests WITH initial state."""

    def requires_tools(self) -> bool:
        return True


class TestExecuteWithToolsState:
    """Tasks 3.3-3.5 — execute_with_tools() threads state for all strategies."""

    def test_sequential_strategy_threads_state(self) -> None:
        """execute_with_tools sequential strategy passes state to each agent."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        initial_state = PipelineState(metrics={"stage": "init"})
        skill = StubToolSkillWithState(initial_state)

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "a1"
        agent1.role = "R1"
        agent1.model = "gemini-1.5-flash"
        agent1.execute.return_value = _make_agent_result(
            "a1", state_patch=PipelineState(metrics={"stage": "done"})
        )

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1]):
            result = orchestrator.execute_with_tools(
                "do it", skill, registry, strategy="sequential"
            )

        assert result.final_state is not None
        assert result.final_state.metrics["stage"] == "done"

    def test_fanout_strategy_threads_state(self) -> None:
        """execute_with_tools fanout strategy passes state to each agent."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        initial_state = PipelineState(metrics={"mode": "parallel"})
        skill = StubToolSkillWithState(initial_state)

        agent1 = MagicMock(spec=ToolAwareAgent)
        agent1.name = "a1"
        agent1.role = "R1"
        agent1.model = "gemini-1.5-flash"
        agent1.execute.return_value = _make_agent_result("a1")

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent1]):
            result = orchestrator.execute_with_tools(
                "analyze", skill, registry, strategy="fanout"
            )

        state_kwarg = agent1.execute.call_args.kwargs.get("state")
        assert state_kwarg == initial_state
        assert result.final_state is not None

    def test_no_state_skill_final_state_is_none(self) -> None:
        """execute_with_tools with no-state skill results in final_state=None."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        skill = StubToolSkill()

        agent = MagicMock(spec=ToolAwareAgent)
        agent.name = "a1"
        agent.role = "R1"
        agent.model = "gemini-1.5-flash"
        agent.execute.return_value = _make_agent_result("a1")

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent]):
            result = orchestrator.execute_with_tools(
                "analyze", skill, registry, strategy="sequential"
            )

        assert result.final_state is None


# ── OrchestratorResult.final_state field ─────────────────────────────────────


class TestOrchestratorResultFinalState:
    """Task 3.6 — OrchestratorResult.final_state defaults to None."""

    def test_final_state_defaults_none(self) -> None:
        result = OrchestratorResult(skill_name="test", phase=SkillPhase.ANALYZE)
        assert result.final_state is None

    def test_final_state_can_be_set(self) -> None:
        state = PipelineState(metrics={"x": 42})
        result = OrchestratorResult(
            skill_name="test", phase=SkillPhase.ANALYZE, final_state=state
        )
        assert result.final_state is state
        assert result.final_state.metrics["x"] == 42


# ── Backward compatibility ────────────────────────────────────────────────────


class TestBackwardCompatState:
    """Task 3.7 — None initial state is zero-cost, no behaviour change."""

    def test_apply_state_patch_none_state_noop(self) -> None:
        """apply_state_patch with None state always returns None (no-op)."""
        patch_state = PipelineState(metrics={"x": 1})
        result = apply_state_patch(None, patch_state)
        assert result is None

    def test_apply_state_patch_none_patch_returns_original(self) -> None:
        """apply_state_patch with None patch returns original state unchanged."""
        original = PipelineState(metrics={"x": 1})
        result = apply_state_patch(original, None)
        assert result == original

    def test_sequential_without_state_still_succeeds(self) -> None:
        """execute_sequential works identically without state threading."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        skill = _NoStateSkill()

        agent = MagicMock(spec=SpecialistAgent)
        agent.name = "a1"
        agent.role = "R1"
        agent.model = "gemini-1.5-flash"
        agent.execute.return_value = _make_agent_result("a1")

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent]):
            result = orchestrator.execute_sequential(
                skill, SkillPhase.ANALYZE, "ctx", "input"
            )

        assert result.success is True
        assert result.final_state is None

    def test_fanout_without_state_still_succeeds(self) -> None:
        """execute_fanout works identically without state threading."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        skill = _NoStateSkill()

        agent = MagicMock(spec=SpecialistAgent)
        agent.name = "a1"
        agent.role = "R1"
        agent.model = "gemini-1.5-flash"
        agent.execute.return_value = _make_agent_result("a1")

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent]):
            result = orchestrator.execute_fanout(
                skill, SkillPhase.ANALYZE, "ctx", "input"
            )

        assert result.success is True
        assert result.final_state is None

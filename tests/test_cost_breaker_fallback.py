"""Tests for cost circuit breaker and model fallback features.

Tests the following components:
- ``_compute_step_cost()`` helper function
- Cost circuit breaker logic in ``execute_with_tools`` (sequential strategy)
- Model fallback logic in ``execute_with_tools`` (sequential strategy)
- Combined cost + fallback behaviour
- parallel_sequential strategy cost tracking
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vaig.agents.base import AgentResult
from vaig.agents.orchestrator import Orchestrator, _compute_step_cost
from vaig.agents.specialist import SpecialistAgent
from vaig.core.client import GeminiClient, GenerationResult
from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.tools.base import ToolRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_client() -> MagicMock:
    client = MagicMock(spec=GeminiClient)
    client.generate.return_value = GenerationResult(
        text="Agent response",
        model="gemini-2.5-pro",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        finish_reason="STOP",
    )
    return client


def _make_mock_settings(
    *,
    max_cost_per_run: float = 0.0,
    max_failures_before_fallback: int = 2,
    fallback_model: str | None = "gemini-2.0-flash",
) -> MagicMock:
    settings = MagicMock()
    settings.models.default = "gemini-2.5-pro"
    settings.models.fallback = fallback_model
    settings.budget.max_cost_per_run = max_cost_per_run
    settings.agents.max_failures_before_fallback = max_failures_before_fallback
    return settings


def _make_agent_result(
    name: str,
    *,
    success: bool = True,
    content: str | None = None,
    error_type: str | None = None,
) -> AgentResult:
    metadata: dict = {}
    if error_type:
        metadata["error_type"] = error_type
    return AgentResult(
        agent_name=name,
        content=content or f"Result from {name}",
        success=success,
        usage={"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
        metadata=metadata,
    )


def _make_mock_agent(
    name: str,
    *,
    parallel_group: str | None = None,
    execute_return: AgentResult | None = None,
) -> MagicMock:
    agent = MagicMock(spec=SpecialistAgent)
    agent.name = name
    agent.role = f"{name} role"
    agent.parallel_group = parallel_group
    agent.model = "gemini-2.5-pro"
    agent._config = MagicMock()
    agent._config.model = "gemini-2.5-pro"
    agent.execute.return_value = execute_return or _make_agent_result(name)
    return agent


class SimpleSkill(BaseSkill):
    """Minimal skill for testing sequential execute_with_tools."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="test_skill",
            display_name="Test Skill",
            description="A skill used in unit tests.",
        )

    def get_system_instruction(self) -> str:
        return "You are a test agent."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return f"[{phase.value}] {user_input}"

    def get_agents_config(self) -> list[dict]:
        return [
            {
                "name": "agent_a",
                "role": "primary agent",
                "system_instruction": "Agent A instruction.",
                "model": "gemini-2.5-pro",
                "requires_tools": False,
            },
            {
                "name": "agent_b",
                "role": "secondary agent",
                "system_instruction": "Agent B instruction.",
                "model": "gemini-2.5-pro",
                "requires_tools": False,
            },
        ]


class ParallelSkill(BaseSkill):
    """Skill with _gatherer agents for testing parallel_sequential strategy."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="parallel_test_skill",
            display_name="Parallel Test Skill",
            description="A parallel skill for unit tests.",
        )

    def get_system_instruction(self) -> str:
        return "You are a parallel skill."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return f"[{phase.value}] {user_input}"

    def get_agents_config(self) -> list[dict]:
        return [
            {
                "name": "node_gatherer",
                "role": "node gatherer",
                "system_instruction": "Gather node data.",
                "model": "gemini-2.5-pro",
                "requires_tools": False,
            },
            {
                "name": "analyzer",
                "role": "analyzer",
                "system_instruction": "Analyze data.",
                "model": "gemini-2.5-pro",
                "requires_tools": False,
            },
        ]


# ---------------------------------------------------------------------------
# Tests: _compute_step_cost
# ---------------------------------------------------------------------------


class TestComputeStepCost:
    """Tests for the _compute_step_cost module-level helper."""

    def test_known_model_returns_cost(self) -> None:
        """_compute_step_cost should return the value from calculate_cost."""
        agent_result = _make_agent_result("test_agent")
        with patch("vaig.agents.orchestrator.calculate_cost", return_value=0.005) as mock_calc:
            cost = _compute_step_cost(agent_result, "gemini-2.5-pro")

        assert cost == 0.005
        mock_calc.assert_called_once()

    def test_unknown_model_returns_zero(self) -> None:
        """_compute_step_cost should return 0.0 when calculate_cost returns None."""
        agent_result = _make_agent_result("test_agent")
        with patch("vaig.agents.orchestrator.calculate_cost", return_value=None):
            cost = _compute_step_cost(agent_result, "unknown-model-xyz")

        assert cost == 0.0


# ---------------------------------------------------------------------------
# Tests: Cost circuit breaker
# ---------------------------------------------------------------------------


class TestCostCircuitBreaker:
    """Tests for the cost circuit breaker in execute_with_tools (sequential)."""

    def test_cost_breaker_disabled(self) -> None:
        """When max_cost_per_run=0.0, pipeline runs to completion even with cost."""
        settings = _make_mock_settings(max_cost_per_run=0.0)
        orchestrator = Orchestrator(_make_mock_client(), settings)
        skill = SimpleSkill()
        tool_registry = MagicMock(spec=ToolRegistry)

        agent_a = _make_mock_agent("agent_a")
        agent_b = _make_mock_agent("agent_b")

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent_a, agent_b]):
            with patch("vaig.agents.orchestrator._compute_step_cost", return_value=9999.0):
                result = orchestrator.execute_with_tools("test query", skill, tool_registry)

        assert result.success is True
        assert result.budget_exceeded is False
        assert len(result.agent_results) == 2

    def test_cost_breaker_triggers_mid_pipeline(self) -> None:
        """When run_cost exceeds max_cost_per_run, pipeline halts early."""
        settings = _make_mock_settings(max_cost_per_run=0.001)
        orchestrator = Orchestrator(_make_mock_client(), settings)
        skill = SimpleSkill()
        tool_registry = MagicMock(spec=ToolRegistry)

        agent_a = _make_mock_agent("agent_a")
        agent_b = _make_mock_agent("agent_b")

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent_a, agent_b]):
            # First step costs 0.005 > 0.001 → triggers breaker after agent_a
            with patch("vaig.agents.orchestrator._compute_step_cost", return_value=0.005):
                result = orchestrator.execute_with_tools("test query", skill, tool_registry)

        assert result.success is False
        assert result.budget_exceeded is True
        assert "[WARNING]" in result.synthesized_output
        # Only agent_a ran; agent_b was skipped
        assert len(result.agent_results) == 1
        assert result.agent_results[0].agent_name == "agent_a"

    def test_run_cost_usd_populated_without_budget(self) -> None:
        """run_cost_usd should accumulate even when max_cost_per_run is disabled."""
        settings = _make_mock_settings(max_cost_per_run=0.0)
        orchestrator = Orchestrator(_make_mock_client(), settings)
        skill = SimpleSkill()
        tool_registry = MagicMock(spec=ToolRegistry)

        agent_a = _make_mock_agent("agent_a")
        agent_b = _make_mock_agent("agent_b")

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent_a, agent_b]):
            with patch("vaig.agents.orchestrator._compute_step_cost", return_value=0.003):
                result = orchestrator.execute_with_tools("test query", skill, tool_registry)

        assert result.success is True
        assert result.budget_exceeded is False
        # Two agents × 0.003 each = 0.006
        assert result.run_cost_usd == 0.006

    def test_cost_breaker_parallel_sequential(self) -> None:
        """Cost breaker should trigger after parallel phase if budget is exceeded."""
        settings = _make_mock_settings(max_cost_per_run=0.001)
        orchestrator = Orchestrator(_make_mock_client(), settings)
        skill = ParallelSkill()
        tool_registry = MagicMock(spec=ToolRegistry)

        gatherer = _make_mock_agent("node_gatherer", parallel_group="gather")
        analyzer = _make_mock_agent("analyzer", parallel_group=None)

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[gatherer, analyzer]):
            # High cost for the parallel step to trigger budget exceeded
            with patch("vaig.agents.orchestrator._compute_step_cost", return_value=0.01):
                result = orchestrator.execute_with_tools(
                    "test query", skill, tool_registry, strategy="parallel_sequential",
                )

        assert result.budget_exceeded is True
        assert result.success is False
        # Analyzer should NOT have run (budget exceeded after parallel phase)
        analyzer_names = [r.agent_name for r in result.agent_results]
        assert "analyzer" not in analyzer_names


# ---------------------------------------------------------------------------
# Tests: Model fallback
# ---------------------------------------------------------------------------


class TestModelFallback:
    """Tests for the model fallback logic in execute_with_tools (sequential)."""

    def test_no_failures_no_fallback(self) -> None:
        """When all agents succeed, no fallback is triggered."""
        settings = _make_mock_settings(max_failures_before_fallback=2)
        orchestrator = Orchestrator(_make_mock_client(), settings)
        skill = SimpleSkill()
        tool_registry = MagicMock(spec=ToolRegistry)

        agent_a = _make_mock_agent("agent_a")
        agent_b = _make_mock_agent("agent_b")

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent_a, agent_b]):
            result = orchestrator.execute_with_tools("test query", skill, tool_registry)

        assert result.success is True
        # _config.model should not have been reassigned to fallback
        assert agent_a._config.model != "gemini-2.0-flash"

    def test_failures_below_threshold_no_fallback(self) -> None:
        """One rate_limit failure on agent (threshold=2) should NOT trigger fallback.

        The failure_counts dict is local to each execute_with_tools call.
        A skill with the same agent appearing twice accumulates failures within
        a single run. With threshold=2, only 1 failure means no fallback.
        """
        settings = _make_mock_settings(max_failures_before_fallback=2)
        orchestrator = Orchestrator(_make_mock_client(), settings)
        skill = SimpleSkill()
        tool_registry = MagicMock(spec=ToolRegistry)

        fail_result = _make_agent_result("agent_a", success=False, error_type="rate_limit")
        success_result = _make_agent_result("agent_a", success=True)

        # One agent that fails once (rate_limit) then succeeds
        agent_a = _make_mock_agent("agent_a")
        agent_a.execute.side_effect = [fail_result, success_result]

        agent_b = _make_mock_agent("agent_b")

        # Pipeline: agent_a (fail, count=1 < 2), agent_a (success), agent_b (success)
        # After 1 failure, no fallback triggered
        with patch.object(
            orchestrator, "create_agents_for_skill", return_value=[agent_a, agent_a, agent_b],
        ):
            result = orchestrator.execute_with_tools("test query", skill, tool_registry)

        # Only 1 failure before success reset the counter — no fallback
        assert agent_a._config.model != "gemini-2.0-flash"
        assert result.success is True

    def test_two_failures_triggers_fallback(self) -> None:
        """Two consecutive rate_limit failures on same agent trigger model switch.

        The failure_counts dict accumulates within a single execute_with_tools call.
        A skill listing the same agent name twice (or a pipeline that passes the
        same agent object twice) reaches the threshold=2 within a single run.
        After the fallback is set, the agent runs with the new model and succeeds.
        """
        settings = _make_mock_settings(max_failures_before_fallback=2)
        orchestrator = Orchestrator(_make_mock_client(), settings)
        skill = SimpleSkill()
        tool_registry = MagicMock(spec=ToolRegistry)

        fail_result = _make_agent_result("agent_a", success=False, error_type="rate_limit")
        success_result = _make_agent_result("agent_a", success=True)

        agent_a = _make_mock_agent("agent_a")
        # Called 3 times: fail (count=1), fail (count=2 → fallback set), success
        agent_a.execute.side_effect = [fail_result, fail_result, success_result]

        agent_b = _make_mock_agent("agent_b")

        # Pass agent_a three times so the loop executes it 3 times in one call
        with patch.object(
            orchestrator, "create_agents_for_skill",
            return_value=[agent_a, agent_a, agent_a, agent_b],
        ):
            result = orchestrator.execute_with_tools("test query", skill, tool_registry)

        # After 2 failures with the same agent name, fallback model must be set
        assert agent_a._config.model == "gemini-2.0-flash"
        assert result.success is True

    def test_no_fallback_model_configured(self) -> None:
        """When fallback model is None, 2 rate_limit failures set model to None.

        The fallback logic sets ``agent._config.model = settings.models.fallback``
        unconditionally when the threshold is reached. With fallback=None, the
        model is set to None (caller's responsibility to handle).
        """
        settings = _make_mock_settings(max_failures_before_fallback=2, fallback_model=None)
        orchestrator = Orchestrator(_make_mock_client(), settings)
        skill = SimpleSkill()
        tool_registry = MagicMock(spec=ToolRegistry)

        fail_result = _make_agent_result("agent_a", success=False, error_type="rate_limit")
        success_result = _make_agent_result("agent_a", success=True)
        agent_a = _make_mock_agent("agent_a")
        # Two failures reach threshold; third call succeeds (so pipeline doesn't hang)
        agent_a.execute.side_effect = [fail_result, fail_result, success_result]

        agent_b = _make_mock_agent("agent_b")

        with patch.object(
            orchestrator, "create_agents_for_skill",
            return_value=[agent_a, agent_a, agent_a, agent_b],
        ):
            result = orchestrator.execute_with_tools("test query", skill, tool_registry)

        # When fallback=None, _config.model is set to None after threshold
        assert agent_a._config.model is None

    def test_fallback_model_also_fails(self) -> None:
        """When fallback model also fails with non-retriable error, pipeline fails."""
        settings = _make_mock_settings(max_failures_before_fallback=2)
        orchestrator = Orchestrator(_make_mock_client(), settings)
        skill = SimpleSkill()
        tool_registry = MagicMock(spec=ToolRegistry)

        rate_limit_fail = _make_agent_result("agent_a", success=False, error_type="rate_limit")
        # After fallback is set, agent fails with a generic (non-retriable) error
        hard_fail = _make_agent_result("agent_a", success=False, content="model error")

        agent_a = _make_mock_agent("agent_a")
        # fail (count=1), fail (count=2 → fallback set), hard fail (no rate_limit → break)
        agent_a.execute.side_effect = [rate_limit_fail, rate_limit_fail, hard_fail]

        agent_b = _make_mock_agent("agent_b")

        with patch.object(
            orchestrator, "create_agents_for_skill",
            return_value=[agent_a, agent_a, agent_a, agent_b],
        ):
            result = orchestrator.execute_with_tools("test query", skill, tool_registry)

        # Fallback model was set, but then the fallback call also fails (non-retriable)
        assert result.success is False
        assert agent_a._config.model == "gemini-2.0-flash"

    def test_success_resets_failure_counter(self) -> None:
        """A success after a failure resets the counter, preventing premature fallback.

        Sequence: fail (count=1), succeed (counter reset to 0), fail (count=1 again).
        With threshold=2, the second failure-run only has count=1, so no fallback.
        """
        settings = _make_mock_settings(max_failures_before_fallback=2)
        orchestrator = Orchestrator(_make_mock_client(), settings)
        skill = SimpleSkill()
        tool_registry = MagicMock(spec=ToolRegistry)

        fail_result = _make_agent_result("agent_a", success=False, error_type="rate_limit")
        success_result = _make_agent_result("agent_a", success=True)

        agent_a = _make_mock_agent("agent_a")
        # fail (count=1), succeed (reset), fail (count=1, below threshold=2)
        agent_a.execute.side_effect = [fail_result, success_result, fail_result, success_result]

        agent_b = _make_mock_agent("agent_b")

        # Pipeline: a, a, a, a, b — counter resets on success so never reaches 2
        with patch.object(
            orchestrator, "create_agents_for_skill",
            return_value=[agent_a, agent_a, agent_a, agent_a, agent_b],
        ):
            result = orchestrator.execute_with_tools("test query", skill, tool_registry)

        # Counter was reset after first success — no fallback triggered
        assert agent_a._config.model != "gemini-2.0-flash"
        assert result.success is True


# ---------------------------------------------------------------------------
# Tests: Combined cost breaker + model fallback
# ---------------------------------------------------------------------------


class TestCombined:
    """Tests for cost circuit breaker and model fallback working together."""

    def test_both_features_active(self) -> None:
        """Both features active: fallback triggered but pipeline completes within budget."""
        settings = _make_mock_settings(
            max_cost_per_run=1.0,  # generous budget
            max_failures_before_fallback=2,
        )
        orchestrator = Orchestrator(_make_mock_client(), settings)
        skill = SimpleSkill()
        tool_registry = MagicMock(spec=ToolRegistry)

        fail_result = _make_agent_result("agent_a", success=False, error_type="rate_limit")
        success_result = _make_agent_result("agent_a", success=True)

        agent_a = _make_mock_agent("agent_a")
        agent_a.execute.side_effect = [fail_result, fail_result, success_result]
        agent_b = _make_mock_agent("agent_b")

        # agent_a appears 3 times so failure_counts accumulates across iterations:
        # iter 1 → fail (count=1), iter 2 → fail (count=2 → fallback set), iter 3 → success
        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent_a, agent_a, agent_a, agent_b]):
            with patch("vaig.agents.orchestrator._compute_step_cost", return_value=0.01):
                result = orchestrator.execute_with_tools("test query", skill, tool_registry)

        assert result.success is True
        assert result.budget_exceeded is False
        assert result.run_cost_usd > 0
        assert agent_a._config.model == "gemini-2.0-flash"

    def test_cost_breaker_during_fallback(self) -> None:
        """Budget exceeded during fallback retry — budget takes precedence."""
        settings = _make_mock_settings(
            max_cost_per_run=0.001,  # very tight budget
            max_failures_before_fallback=2,
        )
        orchestrator = Orchestrator(_make_mock_client(), settings)
        skill = SimpleSkill()
        tool_registry = MagicMock(spec=ToolRegistry)

        fail_result = _make_agent_result("agent_a", success=False, error_type="rate_limit")
        agent_a = _make_mock_agent("agent_a")
        agent_a.execute.return_value = fail_result
        agent_b = _make_mock_agent("agent_b")

        with patch.object(orchestrator, "create_agents_for_skill", return_value=[agent_a, agent_b]):
            # Cost 0.005 per step → exceeds 0.001 budget after first agent
            with patch("vaig.agents.orchestrator._compute_step_cost", return_value=0.005):
                result = orchestrator.execute_with_tools("test query", skill, tool_registry)

        # Cost breaker fires first (after agent_a's first execution)
        assert result.budget_exceeded is True
        assert result.success is False
        assert "[WARNING]" in result.synthesized_output

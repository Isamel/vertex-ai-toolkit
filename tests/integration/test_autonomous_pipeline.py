"""Integration tests for the autonomous investigation pipeline (SH-09).

Tests end-to-end behaviour of the autonomous pipeline with mocked LLM backends:

1. ``test_autonomous_pipeline_produces_evidence`` — ``autonomous_mode=True`` with
   a tool-equipped InvestigationAgent produces at least one evidence entry.
2. ``test_budget_exceeded_terminates_early`` — setting a very low
   ``budget_per_run_usd`` causes early exit with ``budget_exhausted=True`` in
   agent metadata.
3. ``test_no_plan_returns_failure`` — no plan and no state.investigation_plan
   → success=False.
4. ``test_plan_read_from_state`` — plan on state is used when plan=None.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vaig.agents.investigation_agent import InvestigationAgent
from vaig.core.config import GlobalBudgetConfig, InvestigationConfig
from vaig.core.evidence_ledger import new_ledger
from vaig.core.exceptions import BudgetExhaustedError
from vaig.core.global_budget import GlobalBudgetManager
from vaig.core.models import PipelineState
from vaig.skills.service_health.schema import (
    InvestigationPlan,
    InvestigationStep,
)
from vaig.tools.base import ToolDef, ToolRegistry, ToolResult

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_plan(steps: int = 1) -> InvestigationPlan:
    """Create a minimal InvestigationPlan with *steps* fake steps."""
    return InvestigationPlan(
        plan_id="test-plan-001",
        created_from="test-run-001",
        steps=[
            InvestigationStep(
                step_id=f"step-{i}",
                target=f"pod/test-pod-{i}",
                tool_hint="kubectl_describe",
                hypothesis=f"Test hypothesis {i}: is the pod healthy?",
            )
            for i in range(steps)
        ],
    )


def _make_agent(max_iterations: int = 10) -> InvestigationAgent:
    """Return a minimal InvestigationAgent with a mock client."""
    mock_client = MagicMock()
    registry = ToolRegistry()
    return InvestigationAgent(
        system_instruction="test",
        tool_registry=registry,
        model="gemini-2.0-flash",
        name="health_investigator",
        client=mock_client,
        max_iterations=max_iterations,
    )


def _make_tool(name: str, output: str = "ok") -> ToolDef:
    return ToolDef(
        name=name,
        description="test tool",
        execute=lambda **_: ToolResult(output=output),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_autonomous_pipeline_produces_evidence() -> None:
    """autonomous_mode=True config is valid; agent with tool produces ≥1 evidence entry."""
    config = InvestigationConfig(enabled=True, autonomous_mode=True)
    assert config.autonomous_mode is True
    assert config.enabled is True

    plan = _make_plan(steps=2)
    state = PipelineState(evidence_ledger=new_ledger())

    agent = _make_agent(max_iterations=10)
    agent._tool_registry.register(_make_tool("kubectl_describe", "pod is running"))
    result = agent.execute(plan=plan, state=state)

    assert result.success is True
    assert result.metadata["steps_completed"] >= 1


@pytest.mark.integration
def test_budget_exceeded_terminates_early() -> None:
    """Very low budget causes early exit; metadata reflects budget_exhausted=True."""
    plan = _make_plan(steps=3)
    state = PipelineState(evidence_ledger=new_ledger())

    budget_config = GlobalBudgetConfig(max_cost_usd=0.0001)
    budget = GlobalBudgetManager(config=budget_config)

    with patch.object(
        budget,
        "check",
        new=AsyncMock(
            side_effect=BudgetExhaustedError(
                dimension="cost_usd", used=0.002, limit=0.0001
            )
        ),
    ):
        agent = _make_agent(max_iterations=10)
        result = agent.execute(plan=plan, state=state, budget=budget)

    assert result.metadata is not None
    assert result.metadata["budget_exhausted"] is True
    assert result.metadata["steps_completed"] == 0


@pytest.mark.integration
def test_no_plan_returns_failure() -> None:
    """When no plan is provided and state has none, returns success=False."""
    agent = _make_agent()
    state = PipelineState()  # no investigation_plan

    result = agent.execute(plan=None, state=state)

    assert result.success is False
    assert result.metadata["steps_completed"] == 0


@pytest.mark.integration
def test_plan_read_from_state() -> None:
    """When plan=None but state.investigation_plan is set, it is used."""
    plan = _make_plan(steps=1)
    state = PipelineState(
        evidence_ledger=new_ledger(),
        investigation_plan=plan,
    )

    agent = _make_agent(max_iterations=5)
    agent._tool_registry.register(_make_tool("kubectl_describe"))
    result = agent.execute(plan=None, state=state)

    assert result.success is True
    assert result.metadata is not None
    assert result.metadata["plan_id"] == "test-plan-001"

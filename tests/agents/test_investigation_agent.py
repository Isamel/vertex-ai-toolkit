"""Tests for InvestigationAgent (SPEC-SH-02)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from vaig.agents.investigation_agent import InvestigationAgent
from vaig.core.evidence_ledger import EvidenceEntry, new_ledger
from vaig.core.exceptions import BudgetExhaustedError
from vaig.skills.service_health.schema import InvestigationPlan, InvestigationStep
from vaig.tools.base import ToolDef, ToolRegistry, ToolResult

# ── Helpers ───────────────────────────────────────────────────────────────


def _make_step(step_id: str, tool: str = "kubectl_describe", target: str = "pod/web") -> InvestigationStep:
    return InvestigationStep(
        step_id=step_id,
        target=target,
        tool_hint=tool,
        hypothesis=f"Hypothesis for {step_id}",
    )


def _make_plan(*step_ids: str) -> InvestigationPlan:
    return InvestigationPlan(
        plan_id="test-plan",
        created_from="test-run",
        steps=[_make_step(sid) for sid in step_ids],
    )


def _make_agent(max_iterations: int = 10) -> InvestigationAgent:
    """Return a minimal InvestigationAgent with a mock client."""
    mock_client = MagicMock()
    registry = ToolRegistry()
    return InvestigationAgent(
        system_instruction="test",
        tool_registry=registry,
        model="gemini-2.0-flash",
        name="test-investigator",
        client=mock_client,
        max_iterations=max_iterations,
    )


def _make_tool(name: str, output: str = "ok") -> ToolDef:
    return ToolDef(
        name=name,
        description="test tool",
        execute=lambda **_: ToolResult(output=output),
    )


# ── Loop completes all steps ──────────────────────────────────────────────


class TestInvestigationAgentExecute:
    __test__ = True

    def test_all_steps_complete_with_registered_tools(self) -> None:
        """Three steps, all tools registered → all steps complete."""
        agent = _make_agent()
        agent._tool_registry.register(_make_tool("kubectl_describe", "pod info here"))

        plan = _make_plan("step-0", "step-1", "step-2")
        result = agent.execute(plan)

        assert result.success is True
        assert result.metadata["steps_completed"] == 3
        assert result.metadata["steps_skipped"] == 0
        assert result.metadata["budget_exhausted"] is False
        assert result.metadata["escalated"] is False

    def test_missing_tool_still_completes_step(self) -> None:
        """If a tool is not registered, step still completes with an error message."""
        agent = _make_agent()
        # No tools registered — tool_hint won't resolve
        plan = _make_plan("step-0")
        result = agent.execute(plan)

        assert result.success is True
        assert result.metadata["steps_completed"] == 1
        assert "tool not available" in result.content

    def test_evidence_ledger_appended_to_state_patch(self) -> None:
        """After execution, state_patch contains the updated evidence_ledger."""
        agent = _make_agent()
        agent._tool_registry.register(_make_tool("kubectl_describe", "some output"))
        plan = _make_plan("step-0")

        result = agent.execute(plan)

        ledger = result.state_patch.get("evidence_ledger")
        assert ledger is not None
        assert len(ledger.entries) == 1

    # ── Budget exhaustion ─────────────────────────────────────────────────

    def test_budget_exhaustion_stops_loop(self) -> None:
        """Budget exhausts after 2 checks → remaining steps skipped."""
        agent = _make_agent()
        agent._tool_registry.register(_make_tool("kubectl_describe"))

        call_count = 0

        async def mock_check() -> None:
            nonlocal call_count
            call_count += 1
            if call_count > 2:
                raise BudgetExhaustedError(dimension="cost_usd", used=10.0, limit=5.0)

        budget = MagicMock()
        budget.check = mock_check

        plan = _make_plan("step-0", "step-1", "step-2", "step-3", "step-4")
        result = agent.execute(plan, budget=budget)

        assert result.metadata["budget_exhausted"] is True
        assert result.metadata["steps_completed"] < 5
        assert result.metadata["steps_skipped"] > 0
        assert "Budget exhausted" in result.content

    # ── Cache check ───────────────────────────────────────────────────────

    def test_cached_hypothesis_skips_tool_call(self) -> None:
        """Step whose hypothesis is already in the ledger → cached, tool not called."""
        agent = _make_agent()

        tool_called = False

        def spy_tool(**kwargs: object) -> ToolResult:
            nonlocal tool_called
            tool_called = True
            return ToolResult(output="fresh result")

        agent._tool_registry.register(
            ToolDef(name="kubectl_describe", description="test", execute=spy_tool)
        )

        step = _make_step("step-0", tool="kubectl_describe", target="pod/web")
        plan = InvestigationPlan(
            plan_id="cached-plan",
            created_from="run-1",
            steps=[step],
        )

        # Pre-populate the ledger with an entry for the same hypothesis
        pre_entry = EvidenceEntry(
            tool_name="kubectl_describe",
            tool_args_hash="aabbccddaabbccdd",
            question=step.hypothesis,
            answer_summary="pre-cached answer",
        )
        pre_ledger = new_ledger().append(pre_entry)

        from vaig.core.models import PipelineState
        state = MagicMock(spec=PipelineState)
        state.evidence_ledger = pre_ledger

        result = agent.execute(plan, state=state)

        assert tool_called is False, "Tool should NOT be called for cached hypothesis"
        assert result.metadata["steps_completed"] == 1
        assert "CACHED" in result.content

    # ── max_iterations safeguard ──────────────────────────────────────────

    def test_max_iterations_stops_loop(self) -> None:
        """With max_iterations=1, only the first step executes."""
        agent = _make_agent(max_iterations=1)
        agent._tool_registry.register(_make_tool("kubectl_describe"))

        plan = _make_plan("step-0", "step-1", "step-2")
        result = agent.execute(plan)

        assert result.metadata["iterations"] == 1
        assert result.metadata["steps_completed"] == 1
        assert result.metadata["steps_skipped"] == 2
        assert "max_iterations=1" in result.content

    # ── async_execute delegation ──────────────────────────────────────────

    def test_async_execute_delegates_to_sync(self) -> None:
        """async_execute should return the same result as execute."""
        agent = _make_agent()
        agent._tool_registry.register(_make_tool("kubectl_describe"))
        plan = _make_plan("step-0")

        sync_result = agent.execute(plan)
        async_result = asyncio.new_event_loop().run_until_complete(agent.async_execute(plan))

        assert sync_result.metadata["steps_completed"] == async_result.metadata["steps_completed"]
        assert sync_result.metadata["steps_skipped"] == async_result.metadata["steps_skipped"]

"""Investigation Agent — autonomous hypothesis–test–refine loop (SPEC-SH-02).

Extends :class:`~vaig.agents.tool_aware.ToolAwareAgent` to implement the
structured investigation loop: for each step in an :class:`InvestigationPlan`,
the agent:

1. Checks past memory (:mod:`vaig.core.memory.memory_correction`) for prior
   failures (MEM-05).
2. Checks whether the hypothesis is already answered by the ledger cache.
3. Calls the indicated tool and appends an :class:`EvidenceEntry` to the ledger.
4. Calls :meth:`SelfCorrectionController.decide` to detect circles,
   contradictions, or stale loops (SH-06).
5. Checks the global budget and terminates if exhausted.

Loop terminates on any of:

- All steps are ``complete`` or ``skipped``.
- :exc:`~vaig.core.exceptions.BudgetExhaustedError` is raised.
- ``max_iterations`` is reached.
- :attr:`SelfCorrectionController` returns ``ESCALATE``.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from vaig.agents.base import AgentResult
from vaig.agents.tool_aware import ToolAwareAgent
from vaig.core.evidence_ledger import EvidenceEntry, EvidenceLedger, new_ledger
from vaig.core.exceptions import BudgetExhaustedError
from vaig.core.global_budget import GlobalBudgetManager
from vaig.core.models import PipelineState
from vaig.core.self_correction import SelfCorrectionAction, SelfCorrectionController
from vaig.skills.service_health.schema import (
    InvestigationPlan,
    InvestigationStep,
    StepStatus,
)

__all__ = ["InvestigationAgent"]

logger = logging.getLogger(__name__)


def _hypothesis_slug(hypothesis: str) -> str:
    """Return an 8-char slug for a hypothesis string (for fingerprinting)."""
    return hashlib.sha256(hypothesis.encode()).hexdigest()[:8]


def _step_fingerprint(step: InvestigationStep) -> str:
    """Compute a stable action fingerprint for *step*."""
    slug = _hypothesis_slug(step.hypothesis)
    raw = f"{step.tool_hint}:{step.target}:{slug}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


class InvestigationAgent(ToolAwareAgent):
    """Tool-aware agent that executes an :class:`InvestigationPlan` step by step.

    Unlike the generic :class:`ToolAwareAgent`, which hands the full LLM
    tool-use loop to Gemini, :class:`InvestigationAgent` drives each step
    deterministically:

    * Per-step memory check (MEM-05).
    * Per-step cache check (avoids redundant tool calls).
    * Per-step self-correction check (SH-06).
    * Per-step budget check.

    Args:
        max_iterations: Hard cap on investigation iterations (secondary
            safeguard, in addition to budget).  Defaults to 10.
        Other args: passed through to :class:`ToolAwareAgent`.
    """

    def __init__(self, *, max_iterations: int = 10, **kwargs: Any) -> None:
        super().__init__(max_iterations=max_iterations, **kwargs)

    # ── Private helpers ───────────────────────────────────────────────────

    def _call_step_tool(self, step: InvestigationStep) -> str:
        """Invoke the tool indicated by *step.tool_hint*.

        Passes ``target=step.target`` as the primary argument.  Returns the
        tool output string, or an error message if the tool is not found or
        raises an exception.
        """
        tool_def = self._tool_registry.get(step.tool_hint)
        if tool_def is None:
            logger.debug(
                "InvestigationAgent: tool '%s' not found in registry",
                step.tool_hint,
            )
            return f"[tool not available: {step.tool_hint}]"

        try:
            result = tool_def.execute(target=step.target)
            return result.output if not result.error else f"[tool error] {result.output}"
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "InvestigationAgent: tool '%s' raised %s",
                step.tool_hint,
                exc,
                exc_info=True,
            )
            return f"[tool exception] {exc}"

    @staticmethod
    def _build_evidence_entry(
        step: InvestigationStep,
        answer: str,
        source_agent: str,
    ) -> EvidenceEntry:
        """Construct an :class:`EvidenceEntry` from a step result."""
        args_hash = hashlib.sha256(step.target.encode()).hexdigest()[:16]
        confirmed = (
            "confirm" in answer.lower()
            or "oom" in answer.lower()
            or "error" in answer.lower()
        )
        return EvidenceEntry(
            source_agent=source_agent,
            tool_name=step.tool_hint,
            tool_args_hash=args_hash,
            question=step.hypothesis,
            answer_summary=answer[:500],
            supports=(step.hypothesis,) if confirmed else (),
        )

    # ── Execute ───────────────────────────────────────────────────────────

    def execute(  # type: ignore[override]
        self,
        plan: InvestigationPlan | None = None,
        *,
        state: PipelineState | None = None,
        controller: SelfCorrectionController | None = None,
        budget: GlobalBudgetManager | None = None,
        pattern_store: Any = None,
        fix_store: Any = None,
    ) -> AgentResult:
        """Execute the investigation plan synchronously.

        Args:
            plan: The :class:`InvestigationPlan` to execute.  When ``None``,
                the plan is read from ``state.investigation_plan``.  A warning
                is logged and an empty result is returned when no plan is
                available from either source.
            state: Optional current pipeline state (ledger is read from here
                if present).
            controller: Optional :class:`SelfCorrectionController`.  When
                ``None``, a default (permissive) controller is used.
            budget: Optional :class:`~vaig.core.global_budget.GlobalBudgetManager`.
                When ``None``, budget exhaustion checks are skipped.
            pattern_store: Optional
                :class:`~vaig.core.memory.pattern_store.PatternMemoryStore`
                for MEM-05 pre-call checks.
            fix_store: Optional
                :class:`~vaig.core.memory.outcome_store.FixOutcomeStore`
                for MEM-05 pre-call checks.

        Returns:
            :class:`AgentResult` with investigation summary in ``content``
            and updated ``evidence_ledger`` in ``state_patch``.
        """
        from vaig.core.config import SelfCorrectionConfig
        from vaig.core.memory.memory_correction import (  # noqa: PLC0415
            check_memory_before_action,
            compute_action_fingerprint,
        )

        # ── Resolve plan ──────────────────────────────────────────────────
        if plan is None:
            if state is not None and state.investigation_plan is not None:
                plan = state.investigation_plan
            else:
                logger.warning(
                    "InvestigationAgent '%s': no plan provided and none found in state — returning empty result",
                    self.name,
                )
                return AgentResult(
                    agent_name=self.name,
                    content="## Investigation Summary\n\nNo plan available.",
                    success=False,
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    metadata={"plan_id": None, "steps_completed": 0, "steps_skipped": 0, "iterations": 0, "budget_exhausted": False, "escalated": False},
                    state_patch=None,
                )

        if controller is None:
            controller = SelfCorrectionController(SelfCorrectionConfig())

        ledger: EvidenceLedger = (
            state.evidence_ledger
            if state is not None and state.evidence_ledger is not None
            else new_ledger()
        )

        # Mutable copies of step statuses (plan is frozen)
        step_statuses: dict[str, StepStatus] = {
            s.step_id: StepStatus.pending for s in plan.steps
        }

        iterations = 0
        iterations_without_progress = 0
        summary_lines: list[str] = []
        budget_exhausted = False
        escalated = False

        for step in plan.steps:
            if iterations >= self._max_iterations:
                logger.info(
                    "InvestigationAgent: max_iterations=%d reached — stopping",
                    self._max_iterations,
                )
                # Mark remaining pending steps as skipped
                for s in plan.steps:
                    if step_statuses[s.step_id] == StepStatus.pending:
                        step_statuses[s.step_id] = StepStatus.skipped
                break

            iterations += 1

            # ── Cache check ───────────────────────────────────────────────
            cached_entries = ledger.already_answered(step.hypothesis)
            if cached_entries:
                step_statuses[step.step_id] = StepStatus.complete
                summary_lines.append(
                    f"**{step.step_id}** ({step.target}): CACHED — {cached_entries[0].answer_summary[:200]}"
                )
                iterations_without_progress = 0
                continue

            # ── Budget check ──────────────────────────────────────────────
            if budget is not None:
                try:
                    import asyncio  # noqa: PLC0415
                    asyncio.run(budget.check())
                except BudgetExhaustedError:
                    logger.info(
                        "InvestigationAgent: budget exhausted before step %s",
                        step.step_id,
                    )
                    budget_exhausted = True
                    for s in plan.steps:
                        if step_statuses[s.step_id] == StepStatus.pending:
                            step_statuses[s.step_id] = StepStatus.skipped
                    break

            # ── MEM-05: memory check ──────────────────────────────────────
            if pattern_store is not None and fix_store is not None:
                fingerprint = compute_action_fingerprint(
                    tool_name=step.tool_hint,
                    target=step.target,
                    hypothesis_slug=_hypothesis_slug(step.hypothesis),
                )
                warning = check_memory_before_action(
                    fingerprint=fingerprint,
                    proposed_tool=step.tool_hint,
                    proposed_args={"target": step.target},
                    pattern_store=pattern_store,
                    fix_store=fix_store,
                )
                if warning is not None:
                    logger.info(
                        "InvestigationAgent: MEM-05 warning for step %s — %s",
                        step.step_id,
                        warning.suggestion,
                    )
                    step_statuses[step.step_id] = StepStatus.skipped
                    summary_lines.append(
                        f"**{step.step_id}** ({step.target}): SKIPPED (memory warning) — {warning.suggestion}"
                    )
                    iterations_without_progress += 1
                    continue

            # ── SH-06: self-correction check ──────────────────────────────
            action = controller.decide(ledger, iterations_without_progress)
            if action == SelfCorrectionAction.escalate:
                logger.info(
                    "InvestigationAgent: SelfCorrectionController returned ESCALATE — stopping"
                )
                escalated = True
                for s in plan.steps:
                    if step_statuses[s.step_id] == StepStatus.pending:
                        step_statuses[s.step_id] = StepStatus.skipped
                break
            if action == SelfCorrectionAction.backtrack:
                logger.info(
                    "InvestigationAgent: SelfCorrectionController returned BACKTRACK — skipping step %s",
                    step.step_id,
                )
                step_statuses[step.step_id] = StepStatus.skipped
                summary_lines.append(
                    f"**{step.step_id}** ({step.target}): SKIPPED (backtrack — repeated tool call detected)"
                )
                iterations_without_progress += 1
                continue

            # ── Tool call ─────────────────────────────────────────────────
            step_statuses[step.step_id] = StepStatus.running
            answer = self._call_step_tool(step)

            entry = self._build_evidence_entry(step, answer, self.name)
            ledger = ledger.append(entry)

            step_statuses[step.step_id] = StepStatus.complete
            summary_lines.append(
                f"**{step.step_id}** ({step.target}): {answer[:300]}"
            )
            iterations_without_progress = 0

        # ── Build summary ─────────────────────────────────────────────────
        completed = sum(1 for s in step_statuses.values() if s == StepStatus.complete)
        skipped = sum(1 for s in step_statuses.values() if s == StepStatus.skipped)
        total = len(plan.steps)

        header_lines = [
            "## Investigation Summary",
            f"**Plan ID**: {plan.plan_id}",
            f"**Steps Completed**: {completed} / {total}",
            f"**Steps Skipped**: {skipped}",
        ]
        if budget_exhausted:
            header_lines.append("**Termination reason**: Budget exhausted")
        elif escalated:
            header_lines.append("**Termination reason**: Self-correction escalation")
        elif iterations >= self._max_iterations:
            header_lines.append(f"**Termination reason**: max_iterations={self._max_iterations} reached")

        content = "\n".join(header_lines + ["", "### Evidence per Step", ""] + summary_lines)

        logger.info(
            "InvestigationAgent '%s' completed — %d/%d steps, iterations=%d",
            self.name,
            completed,
            total,
            iterations,
        )

        return AgentResult(
            agent_name=self.name,
            content=content,
            success=True,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            metadata={
                "plan_id": plan.plan_id,
                "steps_completed": completed,
                "steps_skipped": skipped,
                "iterations": iterations,
                "budget_exhausted": budget_exhausted,
                "escalated": escalated,
            },
            state_patch={"evidence_ledger": ledger},
        )

    async def async_execute(  # type: ignore[override]
        self,
        plan: InvestigationPlan | None = None,
        *,
        state: PipelineState | None = None,
        controller: SelfCorrectionController | None = None,
        budget: GlobalBudgetManager | None = None,
        pattern_store: Any = None,
        fix_store: Any = None,
    ) -> AgentResult:
        """Async version — delegates to the synchronous :meth:`execute`.

        The investigation loop is CPU-bound (tool calls are blocking I/O via
        kubectl subprocesses), so async offers no benefit here.  This method
        exists for interface compatibility with the orchestrator.
        """
        return self.execute(
            plan,
            state=state,
            controller=controller,
            budget=budget,
            pattern_store=pattern_store,
            fix_store=fix_store,
        )

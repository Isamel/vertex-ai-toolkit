"""Investigation planner for autonomous root-cause investigation (SPEC-SH-01).

Provides ``create_investigation_plan()`` which converts a ``HealthReport``
into an ``InvestigationPlan`` with causal-graph-ordered steps and proportional
budget allocation.
"""

from __future__ import annotations

import uuid

from vaig.core.config import SelfCorrectionConfig
from vaig.core.global_budget import GlobalBudgetManager
from vaig.core.hypothesis_library import HypothesisLibrary
from vaig.skills.service_health.schema import (
    Finding,
    HealthReport,
    InvestigationPlan,
    InvestigationStep,
    StepStatus,
)

__all__ = ["create_investigation_plan"]

_DEFAULT_STEP_BUDGET_CAP_USD = 0.10
_DEFAULT_TOOL_HINT = "kubectl_describe"
_DEFAULT_HYPOTHESIS = "Investigate this finding to determine the root cause."


def _remaining_usd(budget: GlobalBudgetManager) -> float:
    """Return a synchronous point-in-time estimate of remaining budget.

    ``GlobalBudgetManager`` is async — this reads internal state without the
    lock, which is safe for planning purposes (single-threaded planner stage).

    If ``max_cost_usd`` is 0 (unlimited), returns a generous default so steps
    can still receive a proportional allocation.
    """
    max_usd: float = getattr(budget._config, "max_cost_usd", 0.0)  # noqa: SLF001
    if max_usd <= 0.0:
        max_usd = 1.0  # default planning budget when unlimited
    spent: float = getattr(budget, "_cost_usd", 0.0)  # noqa: SLF001
    return max(0.0, max_usd - spent)


def _causal_order(findings: list[Finding]) -> list[Finding]:
    """Sort findings so that root causes (no ``caused_by``) come first.

    Within the same causal depth level, findings are ordered by the length of
    their ``caused_by`` list (ascending) — deeper effects come later.
    """
    return sorted(findings, key=lambda f: len(f.caused_by))


def _make_step_id(index: int) -> str:
    return f"step-{index}"


def create_investigation_plan(
    report: HealthReport,
    library: HypothesisLibrary,
    budget: GlobalBudgetManager,
    self_correction_config: SelfCorrectionConfig | None = None,
) -> InvestigationPlan:
    """Generate an ``InvestigationPlan`` from a ``HealthReport``.

    Steps are generated for every root-cause finding (``Finding.caused_by`` is
    empty) as well as dependent findings.  Steps are ordered so that root
    causes appear first (lowest priority number = highest priority).

    Budget is allocated proportionally across steps from the manager's
    remaining budget, capped at ``SelfCorrectionConfig.max_budget_per_step_usd``
    per step.

    Args:
        report: The ``HealthReport`` produced by the health_analyzer agent.
        library: A ``HypothesisLibrary`` used to derive ``tool_hint`` and
            ``hypothesis`` for each step.
        budget: The ``GlobalBudgetManager`` tracking this run's resource usage.
        self_correction_config: Optional config to read ``max_budget_per_step_usd``
            from.  Defaults to ``SelfCorrectionConfig()`` (0.10 USD cap).

    Returns:
        An ``InvestigationPlan`` with one step per finding, ordered from
        root cause to dependent finding.
    """
    cfg = self_correction_config or SelfCorrectionConfig()
    max_per_step = cfg.max_budget_per_step_usd

    # All findings contribute steps — root causes first, then dependents
    all_findings = list(report.findings)
    if not all_findings:
        return InvestigationPlan(
            plan_id=str(uuid.uuid4()),
            steps=[],
            created_from=report.metadata.generated_at or "unknown",
            total_budget_allocated=0.0,
        )

    ordered = _causal_order(all_findings)

    # Budget per step: remaining / step_count, capped at max_per_step
    remaining = _remaining_usd(budget)
    step_count = len(ordered)
    raw_per_step = remaining / step_count if step_count > 0 else 0.0
    per_step_budget = min(raw_per_step, max_per_step)

    steps: list[InvestigationStep] = []
    for idx, finding in enumerate(ordered):
        # Match hypothesis templates against symptom strings from the finding
        symptoms: list[str] = _extract_symptoms(finding)
        matches = library.match(symptoms)

        tool_hint = matches[0].investigation_strategy if matches else _DEFAULT_TOOL_HINT
        hypothesis = matches[0].hypothesis_text if matches else _DEFAULT_HYPOTHESIS

        # Determine causal depth priority (root = 1, deeper = higher number)
        priority = min(5, 1 + len(finding.caused_by))

        # Map caused_by finding IDs to step IDs — we assign IDs sequentially,
        # so we need to find the step index for each dependency
        depends_on: list[str] = []
        for dep_finding_id in finding.caused_by:
            dep_idx = next(
                (i for i, f in enumerate(ordered) if f.id == dep_finding_id),
                None,
            )
            if dep_idx is not None:
                depends_on.append(_make_step_id(dep_idx))

        steps.append(
            InvestigationStep(
                step_id=_make_step_id(idx),
                target=_extract_target(finding),
                tool_hint=tool_hint,
                hypothesis=hypothesis,
                priority=priority,
                depends_on=depends_on,
                status=StepStatus.pending,
                budget_usd=per_step_budget,
            )
        )

    total_budget = sum(s.budget_usd for s in steps)

    return InvestigationPlan(
        plan_id=str(uuid.uuid4()),
        steps=steps,
        created_from=report.metadata.generated_at or "unknown",
        total_budget_allocated=total_budget,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_symptoms(finding: Finding) -> list[str]:
    """Extract symptom-like strings from a finding for library matching."""
    parts: list[str] = []
    if finding.title:
        parts.append(finding.title)
    if finding.root_cause:
        parts.append(finding.root_cause)
    if finding.description:
        parts.append(finding.description)
    parts.extend(finding.evidence)
    return parts


def _extract_target(finding: Finding) -> str:
    """Extract the primary target resource from a finding."""
    if finding.affected_resources:
        return finding.affected_resources[0]
    if finding.service:
        return finding.service
    return finding.title or "unknown"

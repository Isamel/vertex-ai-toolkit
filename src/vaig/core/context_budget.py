"""Context budget management for LLM token usage tracking.

Provides :class:`ContextBudgetManager` to allocate token budgets across
named phases and track consumption during agent execution.

Usage::

    from vaig.core.context_budget import ContextBudgetManager

    budget = ContextBudgetManager(
        total_budget=100_000,
        phases={"tool_loop": 0.7, "summariser": 0.2},
    )
    budget.record_usage("tool_loop", 5000)
    if budget.is_over_budget("tool_loop"):
        # handle over-budget condition
        ...
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ContextBudgetManager:
    """Track and enforce token budgets across named execution phases.

    Args:
        total_budget: Total token budget available across all phases.
        phases: Mapping of phase name to its fraction of the total budget.
            Fractions must sum to ≤ 1.0.

    Raises:
        ValueError: If the sum of phase fractions exceeds 1.0.
    """

    def __init__(self, total_budget: int, phases: dict[str, float]) -> None:
        total_fraction = sum(phases.values())
        if total_fraction > 1.0:
            raise ValueError(
                f"Phase fractions sum to {total_fraction:.4f}, which exceeds 1.0. "
                "Budget fractions must sum to at most 1.0."
            )
        self._total_budget = total_budget
        self._phase_budgets: dict[str, int] = {
            phase: int(total_budget * fraction)
            for phase, fraction in phases.items()
        }
        self._usage: dict[str, int] = dict.fromkeys(phases, 0)

    def record_usage(self, phase: str, tokens: int) -> None:
        """Record token consumption for a phase.

        If the phase is not registered, it is added with zero budget (no limit
        enforcement will trigger, but usage is still tracked).

        Args:
            phase: Name of the execution phase.
            tokens: Number of tokens consumed in this call.
        """
        if phase not in self._usage:
            logger.debug("Recording usage for unregistered phase %r", phase)
            self._usage[phase] = 0
            self._phase_budgets[phase] = 0
        self._usage[phase] += tokens

    def is_over_budget(self, phase: str) -> bool:
        """Return True if the phase has exceeded its allocated budget.

        An unregistered phase (budget == 0) is considered over budget as soon
        as any tokens are recorded for it.

        Args:
            phase: Name of the execution phase.

        Returns:
            True if usage exceeds the phase budget, False otherwise.
        """
        budget = self._phase_budgets.get(phase, 0)
        usage = self._usage.get(phase, 0)
        return usage > budget

    def remaining(self, phase: str) -> int:
        """Return remaining tokens for a phase (may be negative if over budget).

        Args:
            phase: Name of the execution phase.

        Returns:
            Remaining tokens (budget − usage).  Negative means over budget.
        """
        budget = self._phase_budgets.get(phase, 0)
        usage = self._usage.get(phase, 0)
        return budget - usage

    def summary(self) -> dict[str, int]:
        """Return a snapshot of current token usage per phase.

        Returns:
            Mapping of phase name to tokens consumed so far.
        """
        return dict(self._usage)

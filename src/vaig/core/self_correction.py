"""Self-Correction Controller for autonomous investigation loops (SPEC-SH-06).

Provides :class:`SelfCorrectionController` — a stateless utility that reads
an :class:`~vaig.core.evidence_ledger.EvidenceLedger` and returns a
:class:`SelfCorrectionAction` enum telling the investigation loop what to do
next.

Detection priority (highest → lowest):
    1. Circle  → ``BACKTRACK``
    2. Contradiction → ``ESCALATE``
    3. Stale   → ``FORCE_DIFFERENT``
    4. Clean   → ``CONTINUE``
"""

from __future__ import annotations

import logging
from enum import StrEnum

from vaig.core.config import SelfCorrectionConfig
from vaig.core.evidence_ledger import EvidenceLedger

__all__ = ["SelfCorrectionAction", "SelfCorrectionController"]

logger = logging.getLogger(__name__)


class SelfCorrectionAction(StrEnum):
    """Actions the investigation loop can take based on self-correction checks."""

    continue_ = "CONTINUE"
    backtrack = "BACKTRACK"
    escalate = "ESCALATE"
    force_different = "FORCE_DIFFERENT"


class SelfCorrectionController:
    """Stateless self-correction controller for investigation loops.

    All detection is read-only against the supplied :class:`EvidenceLedger` —
    no mutable state is maintained.  This makes the controller easy to unit
    test and safe to call concurrently.

    Args:
        config: :class:`~vaig.core.config.SelfCorrectionConfig` with tuning
            thresholds.
    """

    def __init__(self, config: SelfCorrectionConfig) -> None:
        self._config = config

    # ── Public detection helpers ──────────────────────────────────────────

    def check_circles(self, ledger: EvidenceLedger) -> list[str]:
        """Return a list of ``(tool_name, tool_args_hash)`` pairs that have been
        called at least ``max_repeated_calls`` times.

        Returns:
            List of human-readable strings like ``"kubectl_describe:abc123ef"``
            for each repeated pair.  Empty list means no circles detected.
        """
        counts: dict[str, int] = {}
        for entry in ledger.entries:
            if entry.tool_name:
                key = f"{entry.tool_name}:{entry.tool_args_hash}"
                counts[key] = counts.get(key, 0) + 1

        threshold = self._config.max_repeated_calls
        circles = [key for key, count in counts.items() if count >= threshold]
        if circles:
            logger.debug(
                "SelfCorrectionController: circles detected — %s",
                circles,
            )
        return circles

    def check_contradictions(self, ledger: EvidenceLedger) -> list[tuple[str, str]]:
        """Return pairs of claim strings where one entry supports and another
        contradicts the same claim on the same ``tool_name`` target.

        The check is sensitivity-gated: when ``contradiction_sensitivity`` is
        ``0.0`` this method always returns an empty list (detection disabled).

        Returns:
            List of ``(supporting_entry_id, contradicting_entry_id)`` tuples.
            Empty list means no contradictions detected.
        """
        if self._config.contradiction_sensitivity <= 0.0:
            return []

        # Index: claim → list of entry ids that support it
        supported: dict[str, list[str]] = {}
        # Index: claim → list of entry ids that contradict it
        contradicted: dict[str, list[str]] = {}

        for entry in ledger.entries:
            for claim in entry.supports:
                supported.setdefault(claim, []).append(entry.id)
            for claim in entry.contradicts:
                contradicted.setdefault(claim, []).append(entry.id)

        pairs: list[tuple[str, str]] = []
        for claim, support_ids in supported.items():
            contra_ids = contradicted.get(claim, [])
            if contra_ids:
                # Emit one pair per (support, contra) combination
                for sid in support_ids:
                    for cid in contra_ids:
                        pairs.append((sid, cid))

        if pairs:
            logger.debug(
                "SelfCorrectionController: %d contradiction pair(s) detected",
                len(pairs),
            )
        return pairs

    def check_stale(self, iterations_without_progress: int) -> bool:
        """Return ``True`` if the investigation has been stale too long.

        Args:
            iterations_without_progress: Number of consecutive iterations
                where no new steps were completed.

        Returns:
            ``True`` when ``iterations_without_progress >= max_stale_iterations``.
        """
        is_stale = iterations_without_progress >= self._config.max_stale_iterations
        if is_stale:
            logger.debug(
                "SelfCorrectionController: stale after %d iterations (threshold=%d)",
                iterations_without_progress,
                self._config.max_stale_iterations,
            )
        return is_stale

    # ── Public decision method ────────────────────────────────────────────

    def decide(
        self,
        ledger: EvidenceLedger,
        iterations_without_progress: int,
    ) -> SelfCorrectionAction:
        """Evaluate the ledger and return the recommended action.

        Decision priority (highest → lowest):
        1. Circle detected → :attr:`SelfCorrectionAction.backtrack`
        2. Contradiction detected → :attr:`SelfCorrectionAction.escalate`
        3. Stale → :attr:`SelfCorrectionAction.force_different`
        4. Clean → :attr:`SelfCorrectionAction.continue_`

        Args:
            ledger: The current evidence ledger.
            iterations_without_progress: Consecutive iterations with no
                newly completed steps.

        Returns:
            The recommended :class:`SelfCorrectionAction`.
        """
        if self.check_circles(ledger):
            return SelfCorrectionAction.backtrack

        if self.check_contradictions(ledger):
            return SelfCorrectionAction.escalate

        if self.check_stale(iterations_without_progress):
            return SelfCorrectionAction.force_different

        return SelfCorrectionAction.continue_

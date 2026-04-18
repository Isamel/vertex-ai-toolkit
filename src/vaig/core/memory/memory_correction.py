"""Memory-aware self-correction for the investigation loop (SPEC-MEM-05).

Provides :func:`compute_action_fingerprint` and
:func:`check_memory_before_action` — called by :class:`InvestigationAgent`
before each tool call to detect past failures and suggest alternatives.

The fingerprint includes ``hypothesis_slug`` to avoid collision between steps
that share the same tool + target but investigate different hypotheses.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from vaig.core.memory.models import FixOutcome, PatternEntry
    from vaig.core.memory.outcome_store import FixOutcomeStore
    from vaig.core.memory.pattern_store import PatternMemoryStore

__all__ = [
    "MemoryWarning",
    "compute_action_fingerprint",
    "check_memory_before_action",
]

logger = logging.getLogger(__name__)


class MemoryWarning(BaseModel):
    """A warning returned when past failure memory matches a proposed action.

    Attributes:
        past_pattern: The :class:`~vaig.core.memory.models.PatternEntry` that
            was found in the pattern store for this fingerprint.
        past_outcome: The :class:`~vaig.core.memory.models.FixOutcome` that
            records the failure outcome correlated to this fingerprint.
        suggestion: Human-readable alternative approach.
    """

    past_pattern: PatternEntry
    past_outcome: FixOutcome
    suggestion: str


def compute_action_fingerprint(
    tool_name: str,
    target: str,
    hypothesis_slug: str,
) -> str:
    """Compute a stable fingerprint for a proposed investigation action.

    The fingerprint is a 16-char SHA-256 prefix of
    ``tool_name + ":" + target + ":" + hypothesis_slug``.

    Including ``hypothesis_slug`` prevents collisions between two steps that
    use the same tool and target but investigate different hypotheses — e.g.
    ``kubectl_describe`` on ``pod/web-abc`` for OOM vs high-latency hypothesis.

    Args:
        tool_name: The name of the tool to be called.
        target: The investigation target (e.g. ``"pod/my-service-abc123"``).
        hypothesis_slug: A short slug derived from the step hypothesis (e.g.
            ``"oom-kill"``).

    Returns:
        A 16-character hex string fingerprint.
    """
    raw = f"{tool_name}:{target}:{hypothesis_slug}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def check_memory_before_action(
    fingerprint: str,
    proposed_tool: str,
    proposed_args: dict[str, object],
    pattern_store: PatternMemoryStore,
    fix_store: FixOutcomeStore,
) -> MemoryWarning | None:
    """Check past memory for failures on the proposed action.

    Looks up *fingerprint* in *pattern_store*.  If an entry is found,
    searches *fix_store* for a correlated :class:`~vaig.core.memory.models.FixOutcome`
    with ``outcome == "failure"`` (or the similar value ``"worsened"`` /
    ``"persisted"``).  Returns a :class:`MemoryWarning` when a past failure
    is found; returns ``None`` otherwise.

    The check is entirely error-silent — any exception (missing store,
    I/O error) returns ``None`` so the investigation loop continues normally.

    Args:
        fingerprint: Pre-computed action fingerprint from
            :func:`compute_action_fingerprint`.
        proposed_tool: Name of the tool about to be called.
        proposed_args: Tool arguments (used only for logging).
        pattern_store: :class:`~vaig.core.memory.pattern_store.PatternMemoryStore`
            instance to look up past patterns.
        fix_store: :class:`~vaig.core.memory.outcome_store.FixOutcomeStore`
            instance to look up past outcomes.

    Returns:
        :class:`MemoryWarning` if a past failure is found, ``None`` otherwise.
    """
    _FAILURE_OUTCOMES = frozenset({"worsened", "persisted"})

    try:
        pattern = pattern_store.lookup(fingerprint)
        if pattern is None:
            return None

        # Search fix_store for an outcome correlated to this fingerprint.
        # FixOutcomeStore is indexed by fix_id, so we scan all entries.
        failure_outcome: FixOutcome | None = None
        for entry in fix_store._ensure_index().values():  # noqa: SLF001
            if entry.fingerprint == fingerprint and entry.outcome in _FAILURE_OUTCOMES:
                failure_outcome = entry
                break

        if failure_outcome is None:
            return None

        suggestion = (
            f"Past attempt with '{proposed_tool}' on fingerprint '{fingerprint}' "
            f"resulted in '{failure_outcome.outcome}'. "
            "Consider trying a different tool or approach."
        )
        logger.debug(
            "check_memory_before_action: past failure detected for fingerprint %s "
            "(tool=%s, outcome=%s)",
            fingerprint,
            proposed_tool,
            failure_outcome.outcome,
        )
        return MemoryWarning(
            past_pattern=pattern,
            past_outcome=failure_outcome,
            suggestion=suggestion,
        )

    except Exception:  # noqa: BLE001
        logger.debug(
            "check_memory_before_action: error checking memory (non-fatal); "
            "fingerprint=%s tool=%s",
            fingerprint,
            proposed_tool,
            exc_info=True,
        )
        return None

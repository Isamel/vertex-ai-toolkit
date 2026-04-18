"""Pattern history query tool — returns recurrence data for a finding fingerprint.

Allows Gemini agents to look up how many times a given finding pattern has been
seen historically, giving context for RECURRING / CHRONIC badges.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vaig.tools.base import ToolResult

if TYPE_CHECKING:
    from vaig.core.config import MemoryConfig

logger = logging.getLogger(__name__)

__all__ = ["query_pattern_history"]


def query_pattern_history(
    category: str,
    service: str,
    title: str,
    description: str,
    config: MemoryConfig,
) -> ToolResult:
    """Look up historical recurrence data for a finding pattern.

    Computes the observation fingerprint from the finding's descriptive fields
    and queries the local :class:`~vaig.core.memory.pattern_store.PatternMemoryStore`
    for all past occurrences.

    Args:
        category: Finding category, e.g. ``"pod-health"``.
        service: Service / resource the finding affects.
        title: Human-readable finding title.
        description: Finding description (PII and ephemeral tokens are stripped
            automatically during fingerprint computation).
        config: ``MemoryConfig`` instance from application settings.

    Returns:
        A :class:`~vaig.tools.base.ToolResult` containing a Markdown summary of
        historical occurrences, or a brief message when no history is found.
    """
    try:
        from vaig.core.memory.fingerprint import ObservationFingerprint
        from vaig.core.memory.models import RecurrenceSignal
        from vaig.core.memory.pattern_store import PatternMemoryStore

        fp: str = ObservationFingerprint.from_finding(
            category=category,
            service=service,
            title=title,
            description=description,
        )
        store = PatternMemoryStore(base_dir=config.store_path)
        entry = store.lookup(fp)

        if entry is None:
            return ToolResult(
                output=f"No historical data found for this pattern (fingerprint: `{fp}`).",
                error=False,
            )

        signal = RecurrenceSignal.from_entry(entry)
        lines = [
            f"**Fingerprint**: `{fp}`",
            f"**Badge**: {signal.badge}",
            f"**Occurrences**: {signal.occurrences}",
            f"**First seen**: {signal.first_seen.isoformat()}",
            f"**Last seen**: {signal.last_seen.isoformat()}",
            f"**Is recurring**: {signal.is_recurring}",
        ]
        return ToolResult(output="\n".join(lines), error=False)

    except Exception:  # noqa: BLE001
        logger.debug("query_pattern_history failed", exc_info=True)
        return ToolResult(
            output="Pattern history lookup failed — memory subsystem unavailable.",
            error=True,
        )

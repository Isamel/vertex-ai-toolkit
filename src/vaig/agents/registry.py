"""Agent registry — DEPRECATED.

Agent lifecycle is managed by ``Orchestrator`` since the
*infra-agent-orchestration* change.  This module is kept as an empty
placeholder to avoid breaking third-party imports; new code should NOT
use ``AgentRegistry``.
"""

from __future__ import annotations

import warnings


class AgentRegistry:  # pragma: no cover
    """Deprecated — use ``Orchestrator`` for agent management."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        warnings.warn(
            "AgentRegistry is deprecated — use Orchestrator instead.",
            DeprecationWarning,
            stacklevel=2,
        )

"""Global budget manager for cross-run resource governance.

Tracks aggregate usage (tokens, USD cost, tool calls, wall-clock time) across
a single pipeline run and raises :class:`~vaig.core.exceptions.BudgetExhaustedError`
when any configured limit is exceeded.

All limit fields default to ``0``, which means **unlimited**.

Usage::

    from vaig.core.config import GlobalBudgetConfig
    from vaig.core.global_budget import GlobalBudgetManager

    cfg = GlobalBudgetConfig(max_tokens=1_000_000, max_cost_usd=5.0)
    mgr = GlobalBudgetManager(cfg)

    await mgr.record_tokens(50_000)
    await mgr.record_cost(0.25)
    await mgr.record_tool_call()
    await mgr.check()   # raises BudgetExhaustedError if any limit is hit
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from vaig.core.exceptions import BudgetExhaustedError

if TYPE_CHECKING:
    from vaig.core.config import GlobalBudgetConfig

__all__ = ["GlobalBudgetManager"]


class GlobalBudgetManager:
    """Thread-safe (async) tracker for global per-run budgets.

    Args:
        config: :class:`~vaig.core.config.GlobalBudgetConfig` with limit values.
            A limit of ``0`` means no limit for that dimension.

    Raises:
        :class:`~vaig.core.exceptions.BudgetExhaustedError`: On :meth:`check`
            when any non-zero limit is exceeded.
    """

    def __init__(self, config: GlobalBudgetConfig) -> None:
        self._config = config
        self._lock = asyncio.Lock()
        self._tokens: int = 0
        self._cost_usd: float = 0.0
        self._tool_calls: int = 0
        self._start_time: float = time.monotonic()

    # ── Mutation helpers ──────────────────────────────────────────────────

    async def record_tokens(self, tokens: int) -> None:
        """Add *tokens* to the accumulated token count."""
        async with self._lock:
            self._tokens += tokens

    async def record_cost(self, cost_usd: float) -> None:
        """Add *cost_usd* to the accumulated USD cost."""
        async with self._lock:
            self._cost_usd += cost_usd

    async def record_tool_call(self) -> None:
        """Increment the tool-call counter by one."""
        async with self._lock:
            self._tool_calls += 1

    # ── Budget check ──────────────────────────────────────────────────────

    async def check(self) -> None:
        """Raise :class:`~vaig.core.exceptions.BudgetExhaustedError` if any limit is hit.

        Checks are performed in this order: tokens → cost → tool_calls → wall_seconds.
        The first exceeded limit wins.
        """
        async with self._lock:
            tokens = self._tokens
            cost = self._cost_usd
            tool_calls = self._tool_calls
            elapsed = time.monotonic() - self._start_time

        cfg = self._config

        if cfg.max_tokens and tokens > cfg.max_tokens:
            raise BudgetExhaustedError(
                dimension="tokens", used=tokens, limit=cfg.max_tokens
            )
        if cfg.max_cost_usd and cost > cfg.max_cost_usd:
            raise BudgetExhaustedError(
                dimension="cost_usd", used=cost, limit=cfg.max_cost_usd
            )
        if cfg.max_tool_calls and tool_calls > cfg.max_tool_calls:
            raise BudgetExhaustedError(
                dimension="tool_calls", used=tool_calls, limit=cfg.max_tool_calls
            )
        if cfg.max_wall_seconds and elapsed > cfg.max_wall_seconds:
            raise BudgetExhaustedError(
                dimension="wall_seconds", used=elapsed, limit=cfg.max_wall_seconds
            )

    # ── Read-only snapshot ────────────────────────────────────────────────

    async def snapshot(self) -> dict[str, float]:
        """Return a point-in-time snapshot of all tracked dimensions.

        Returns:
            Dict with keys ``tokens``, ``cost_usd``, ``tool_calls``, ``wall_seconds``.
        """
        async with self._lock:
            return {
                "tokens": self._tokens,
                "cost_usd": self._cost_usd,
                "tool_calls": self._tool_calls,
                "wall_seconds": time.monotonic() - self._start_time,
            }

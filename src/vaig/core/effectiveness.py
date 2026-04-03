"""Tool effectiveness learning — automatic scoring based on historical call data.

Computes per-tool effectiveness scores from ``ToolCallOptimizer.analyze()``
output, assigns tiers (SKIP / DEPRIORITIZE / BOOST / ALLOW) per configurable
thresholds, and caches results with a monotonic TTL.

When ``effectiveness.enabled`` is False (the default), the singleton factory
returns ``None`` and all behaviour is no-op.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING

from vaig.core.config import EffectivenessConfig, get_settings

if TYPE_CHECKING:
    from vaig.core.optimizer import ToolStats
    from vaig.core.tool_call_store import ToolCallStore

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────


class EffectivenessTier(StrEnum):
    """Tier assigned to a tool based on historical effectiveness."""

    SKIP = "skip"
    DEPRIORITIZE = "deprioritize"
    BOOST = "boost"
    ALLOW = "allow"


@dataclass(slots=True, frozen=True)
class EffectivenessScore:
    """Computed effectiveness score for a single tool."""

    tool_name: str
    tier: EffectivenessTier
    failure_rate: float
    avg_duration_s: float
    call_count: int
    reason: str


# ── Default score for unknown / cold-start tools ─────────────

_ALLOW_DEFAULT = EffectivenessScore(
    tool_name="",
    tier=EffectivenessTier.ALLOW,
    failure_rate=0.0,
    avg_duration_s=0.0,
    call_count=0,
    reason="insufficient data",
)


# ── Service ───────────────────────────────────────────────────


class ToolEffectivenessService:
    """Scores tools from historical call data and caches results."""

    def __init__(self, store: ToolCallStore, config: EffectivenessConfig) -> None:
        self._store = store
        self._config = config
        self._score_cache: dict[str, EffectivenessScore] = {}
        self._cache_timestamp: float = 0.0

    # ── Public API ────────────────────────────────────────────

    def get_tool_score(
        self,
        tool_name: str,
        agent_name: str | None = None,  # noqa: ARG002 — reserved for future agent-scoped scoring
    ) -> EffectivenessScore:
        """Return the effectiveness score for *tool_name*.

        Unknown tools (not present in historical data) return ALLOW tier.
        """
        self._ensure_cache_fresh()
        score = self._score_cache.get(tool_name)
        if score is None:
            return EffectivenessScore(
                tool_name=tool_name,
                tier=EffectivenessTier.ALLOW,
                failure_rate=0.0,
                avg_duration_s=0.0,
                call_count=0,
                reason="insufficient data",
            )
        return score

    def get_all_scores(
        self,
        agent_name: str | None = None,  # noqa: ARG002 — reserved for future agent-scoped scoring
    ) -> dict[str, EffectivenessScore]:
        """Return all cached effectiveness scores."""
        self._ensure_cache_fresh()
        return dict(self._score_cache)

    def invalidate_cache(self) -> None:
        """Force cache invalidation — next call recomputes."""
        self._score_cache.clear()
        self._cache_timestamp = 0.0

    # ── Internal ──────────────────────────────────────────────

    def _ensure_cache_fresh(self) -> None:
        """Recompute scores if the cache TTL has expired."""
        now = time.monotonic()
        if now - self._cache_timestamp > self._config.cache_ttl_seconds:
            self._compute_scores()
            self._cache_timestamp = now

    def _compute_scores(self) -> None:
        """Load stats from the optimizer and assign tiers."""
        try:
            from vaig.core.optimizer import ToolCallOptimizer

            optimizer = ToolCallOptimizer(self._store)
            date_from = datetime.now(tz=UTC) - timedelta(days=self._config.lookback_days)
            insights = optimizer.analyze(date_from=date_from)

            new_cache: dict[str, EffectivenessScore] = {}
            for tool_name, stats in insights.tools.items():
                tier, reason = self._assign_tier(stats)
                new_cache[tool_name] = EffectivenessScore(
                    tool_name=tool_name,
                    tier=tier,
                    failure_rate=stats.failure_rate,
                    avg_duration_s=stats.avg_duration_s,
                    call_count=stats.call_count,
                    reason=reason,
                )
            self._score_cache = new_cache
        except Exception:  # noqa: BLE001 — R-EFF-09 error resilience
            logger.exception(
                "Failed to compute effectiveness scores — defaulting all tools to ALLOW"
            )
            self._score_cache = {}

    def _assign_tier(self, stats: ToolStats) -> tuple[EffectivenessTier, str]:
        """Apply the 6-branch algorithm from the design spec."""
        cfg = self._config

        # 1. Insufficient data
        if stats.call_count < cfg.min_calls_for_scoring:
            return EffectivenessTier.ALLOW, "insufficient data"

        # 2. High failure → SKIP
        if stats.failure_rate > cfg.skip_threshold:
            pct = round(stats.failure_rate * 100)
            return EffectivenessTier.SKIP, f"failure rate {pct}% exceeds skip threshold"

        # 3. Moderate failure → DEPRIORITIZE
        if stats.failure_rate > cfg.deprioritize_threshold:
            pct = round(stats.failure_rate * 100)
            return EffectivenessTier.DEPRIORITIZE, f"failure rate {pct}% exceeds deprioritize threshold"

        # 4. Slow → DEPRIORITIZE
        if stats.avg_duration_s > cfg.slow_tool_threshold_s:
            avg = round(stats.avg_duration_s, 1)
            return EffectivenessTier.DEPRIORITIZE, f"avg duration {avg}s exceeds slow tool threshold"

        # 5. Reliable → BOOST
        if stats.failure_rate < cfg.boost_threshold:
            return EffectivenessTier.BOOST, "low failure rate — reliable tool"

        # 6. Default
        return EffectivenessTier.ALLOW, "within normal parameters"


# ── Singleton factory ─────────────────────────────────────────

_service: ToolEffectivenessService | None = None
_service_initialized: bool = False


def get_effectiveness_service(
    store: ToolCallStore | None = None,
) -> ToolEffectivenessService | None:
    """Return the global effectiveness service, or ``None`` when disabled.

    Mirrors the ``get_settings()`` lazy-singleton pattern.
    """
    global _service, _service_initialized  # noqa: PLW0603

    if _service_initialized:
        return _service

    settings = get_settings()
    if not settings.effectiveness.enabled:
        _service = None
        _service_initialized = True
        return None

    if store is None:
        from vaig.core.tool_call_store import ToolCallStore as _Store

        _store = _Store()
    else:
        _store = store

    _service = ToolEffectivenessService(_store, settings.effectiveness)
    _service_initialized = True
    return _service


def reset_effectiveness_service() -> None:
    """Reset the singleton (for testing)."""
    global _service, _service_initialized  # noqa: PLW0603
    _service = None
    _service_initialized = False

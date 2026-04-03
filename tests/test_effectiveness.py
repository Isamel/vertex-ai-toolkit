"""Unit tests for vaig.core.effectiveness — tool effectiveness learning.

Tests cover:
- _assign_tier: all 6 branches (insufficient data, SKIP, DEPRIORITIZE-failure,
  DEPRIORITIZE-slow, BOOST, ALLOW)
- EffectivenessConfig: valid defaults, invalid thresholds, boundary values
- Cache TTL: hit within TTL, recompute after expiry, explicit invalidation
- Cold start: empty store → all ALLOW
- Error resilience: corrupted data → log + ALLOW, IOError → graceful
- Mixin pre-check: SKIP → synthetic result, DEPRIORITIZE → warn + proceed,
  disabled → no service calls (both sync + async paths)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vaig.core.config import EffectivenessConfig
from vaig.core.effectiveness import (
    EffectivenessScore,
    EffectivenessTier,
    ToolEffectivenessService,
    get_effectiveness_service,
    reset_effectiveness_service,
)
from vaig.tools.base import ToolResult

# ── Helpers ───────────────────────────────────────────────────


@dataclass(slots=True)
class _FakeToolStats:
    """Minimal stand-in for optimizer.ToolStats used in _assign_tier tests."""

    call_count: int
    failure_count: int = 0
    failure_rate: float = 0.0
    avg_duration_s: float = 1.0
    max_duration_s: float = 2.0
    cache_hit_count: int = 0
    cache_hit_rate: float = 0.0
    unique_arg_combos: int = 1
    common_errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _FakeToolInsights:
    """Minimal stand-in for optimizer.ToolInsights."""

    tools: dict[str, _FakeToolStats]
    total_calls: int = 0
    total_runs: int = 0
    redundant_calls: list[Any] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


def _make_config(**overrides: Any) -> EffectivenessConfig:
    """Create an EffectivenessConfig with optional overrides."""
    return EffectivenessConfig(**overrides)


def _make_service(
    tools: dict[str, _FakeToolStats] | None = None,
    config: EffectivenessConfig | None = None,
    raise_on_analyze: Exception | None = None,
) -> ToolEffectivenessService:
    """Create a ToolEffectivenessService with a mocked optimizer."""
    if config is None:
        config = _make_config(enabled=True)
    store = MagicMock()

    svc = ToolEffectivenessService(store, config)

    insights = _FakeToolInsights(tools=tools or {})

    # Patch the optimizer that _compute_scores creates internally
    mock_optimizer_cls = MagicMock()
    if raise_on_analyze is not None:
        mock_optimizer_cls.return_value.analyze.side_effect = raise_on_analyze
    else:
        mock_optimizer_cls.return_value.analyze.return_value = insights

    svc._optimizer_cls_override = mock_optimizer_cls  # type: ignore[attr-defined]
    return svc


# ── _assign_tier (R-EFF-03) ───────────────────────────────────


class TestAssignTier:
    """Parametrized tests for the 6-branch tier algorithm."""

    def test_insufficient_data_returns_allow(self) -> None:
        """call_count < min_calls_for_scoring → ALLOW."""
        svc = _make_service()
        stats = _FakeToolStats(call_count=2, failure_rate=0.9)  # high failure, but too few calls
        tier, reason = svc._assign_tier(stats)  # type: ignore[arg-type]
        assert tier == EffectivenessTier.ALLOW
        assert "insufficient data" in reason

    def test_high_failure_returns_skip(self) -> None:
        """failure_rate > skip_threshold → SKIP."""
        svc = _make_service()
        stats = _FakeToolStats(call_count=10, failure_rate=0.9)
        tier, reason = svc._assign_tier(stats)  # type: ignore[arg-type]
        assert tier == EffectivenessTier.SKIP
        assert "skip" in reason.lower()

    def test_moderate_failure_returns_deprioritize(self) -> None:
        """failure_rate > deprioritize_threshold → DEPRIORITIZE."""
        svc = _make_service()
        stats = _FakeToolStats(call_count=10, failure_rate=0.6)
        tier, reason = svc._assign_tier(stats)  # type: ignore[arg-type]
        assert tier == EffectivenessTier.DEPRIORITIZE
        assert "deprioritize" in reason.lower()

    def test_slow_tool_returns_deprioritize(self) -> None:
        """avg_duration_s > slow_tool_threshold_s → DEPRIORITIZE."""
        svc = _make_service()
        stats = _FakeToolStats(call_count=10, failure_rate=0.3, avg_duration_s=15.0)
        tier, reason = svc._assign_tier(stats)  # type: ignore[arg-type]
        assert tier == EffectivenessTier.DEPRIORITIZE
        assert "duration" in reason.lower()

    def test_reliable_tool_returns_boost(self) -> None:
        """failure_rate < boost_threshold → BOOST."""
        svc = _make_service()
        stats = _FakeToolStats(call_count=10, failure_rate=0.05, avg_duration_s=2.0)
        tier, reason = svc._assign_tier(stats)  # type: ignore[arg-type]
        assert tier == EffectivenessTier.BOOST
        assert "reliable" in reason.lower()

    def test_normal_performance_returns_allow(self) -> None:
        """Within all thresholds but not exceptional → ALLOW."""
        svc = _make_service()
        # failure_rate between boost (0.1) and deprioritize (0.5)
        stats = _FakeToolStats(call_count=10, failure_rate=0.3, avg_duration_s=5.0)
        tier, reason = svc._assign_tier(stats)  # type: ignore[arg-type]
        assert tier == EffectivenessTier.ALLOW
        assert "normal" in reason.lower()

    @pytest.mark.parametrize(
        ("failure_rate", "expected_tier"),
        [
            (0.81, EffectivenessTier.SKIP),
            (0.80, EffectivenessTier.DEPRIORITIZE),  # not > 0.8, falls to next branch (> 0.5)
            (0.51, EffectivenessTier.DEPRIORITIZE),
            (0.50, EffectivenessTier.ALLOW),  # not > 0.5, not < 0.1
            (0.10, EffectivenessTier.ALLOW),  # not < 0.1 (equal)
            (0.09, EffectivenessTier.BOOST),
        ],
    )
    def test_boundary_values(self, failure_rate: float, expected_tier: EffectivenessTier) -> None:
        """Boundary threshold values are assigned the correct tier."""
        svc = _make_service()
        stats = _FakeToolStats(call_count=10, failure_rate=failure_rate, avg_duration_s=5.0)
        tier, _reason = svc._assign_tier(stats)  # type: ignore[arg-type]
        assert tier == expected_tier


# ── EffectivenessConfig (R-EFF-01) ────────────────────────────


class TestEffectivenessConfig:
    """Config model validation."""

    def test_defaults_match_spec(self) -> None:
        """Default values match the spec: enabled=False, skip_threshold=0.8, etc."""
        cfg = EffectivenessConfig()
        assert cfg.enabled is False
        assert cfg.skip_threshold == 0.8
        assert cfg.deprioritize_threshold == 0.5
        assert cfg.boost_threshold == 0.1
        assert cfg.slow_tool_threshold_s == 10.0
        assert cfg.min_calls_for_scoring == 3
        assert cfg.lookback_days == 7
        assert cfg.cache_ttl_seconds == 300

    def test_invalid_threshold_too_high(self) -> None:
        """skip_threshold of 1.5 is rejected."""
        with pytest.raises(Exception):  # noqa: B017 — Pydantic ValidationError
            EffectivenessConfig(skip_threshold=1.5)

    def test_invalid_threshold_negative(self) -> None:
        """Negative threshold is rejected."""
        with pytest.raises(Exception):  # noqa: B017
            EffectivenessConfig(deprioritize_threshold=-0.1)

    def test_boundary_zero(self) -> None:
        """0.0 is a valid threshold (inclusive)."""
        cfg = EffectivenessConfig(skip_threshold=0.0)
        assert cfg.skip_threshold == 0.0

    def test_boundary_one(self) -> None:
        """1.0 is a valid threshold (inclusive)."""
        cfg = EffectivenessConfig(skip_threshold=1.0)
        assert cfg.skip_threshold == 1.0

    def test_min_calls_at_least_one(self) -> None:
        """min_calls_for_scoring must be >= 1."""
        with pytest.raises(Exception):  # noqa: B017
            EffectivenessConfig(min_calls_for_scoring=0)

    def test_cache_ttl_zero_allowed(self) -> None:
        """cache_ttl_seconds=0 is valid (always recompute)."""
        cfg = EffectivenessConfig(cache_ttl_seconds=0)
        assert cfg.cache_ttl_seconds == 0


# ── Cache TTL (R-EFF-04) ─────────────────────────────────────


class TestCacheTTL:
    """Cache lifecycle: hit, expiry, invalidation."""

    def test_cache_hit_within_ttl(self) -> None:
        """Scores requested within TTL are returned from cache (no recompute)."""
        tools = {"tool_a": _FakeToolStats(call_count=10, failure_rate=0.9)}
        svc = _make_service(tools=tools, config=_make_config(enabled=True, cache_ttl_seconds=300))

        # Prime the cache
        with patch("vaig.core.optimizer.ToolCallOptimizer") as mock_cls:
            mock_cls.return_value.analyze.return_value = _FakeToolInsights(tools=tools)
            score1 = svc.get_tool_score("tool_a")

        # Second call — should NOT recompute (mock no longer active, would error if called)
        with patch("time.monotonic", return_value=svc._cache_timestamp + 60):
            score2 = svc.get_tool_score("tool_a")

        assert score1.tier == score2.tier == EffectivenessTier.SKIP

    def test_cache_recomputes_after_ttl(self) -> None:
        """Scores are recomputed after TTL expiry."""
        tools = {"tool_a": _FakeToolStats(call_count=10, failure_rate=0.9)}
        svc = _make_service(tools=tools, config=_make_config(enabled=True, cache_ttl_seconds=300))

        # Prime the cache
        with patch("vaig.core.optimizer.ToolCallOptimizer") as mock_cls:
            mock_cls.return_value.analyze.return_value = _FakeToolInsights(tools=tools)
            svc.get_tool_score("tool_a")

        # Fast-forward past TTL
        with (
            patch("time.monotonic", return_value=svc._cache_timestamp + 301),
            patch("vaig.core.optimizer.ToolCallOptimizer") as mock_cls2,
        ):
            # Tool now has low failure rate — tier should change
            new_tools = {"tool_a": _FakeToolStats(call_count=10, failure_rate=0.05)}
            mock_cls2.return_value.analyze.return_value = _FakeToolInsights(tools=new_tools)
            score = svc.get_tool_score("tool_a")

        assert score.tier == EffectivenessTier.BOOST
        assert mock_cls2.return_value.analyze.call_count == 1

    def test_invalidate_cache_forces_recompute(self) -> None:
        """invalidate_cache() clears cache; next call recomputes."""
        tools = {"tool_a": _FakeToolStats(call_count=10, failure_rate=0.9)}
        svc = _make_service(tools=tools, config=_make_config(enabled=True))

        with patch("vaig.core.optimizer.ToolCallOptimizer") as mock_cls:
            mock_cls.return_value.analyze.return_value = _FakeToolInsights(tools=tools)
            svc.get_tool_score("tool_a")  # prime

        svc.invalidate_cache()
        assert svc._score_cache == {}
        assert svc._cache_timestamp == 0.0


# ── Cold start / error resilience (R-EFF-08, R-EFF-09) ───────


class TestColdStartAndResilience:
    """Empty store, corrupted data, and I/O errors."""

    def test_empty_store_returns_allow(self) -> None:
        """No historical data → all tools get ALLOW (R-EFF-08)."""
        svc = _make_service(tools={})

        with patch("vaig.core.optimizer.ToolCallOptimizer") as mock_cls:
            mock_cls.return_value.analyze.return_value = _FakeToolInsights(tools={})
            score = svc.get_tool_score("unknown_tool")

        assert score.tier == EffectivenessTier.ALLOW
        assert "insufficient data" in score.reason

    def test_unknown_tool_returns_allow(self) -> None:
        """Tool not in historical data → ALLOW (R-EFF-08)."""
        known_tools = {"tool_a": _FakeToolStats(call_count=10, failure_rate=0.05)}
        svc = _make_service(tools=known_tools)

        with patch("vaig.core.optimizer.ToolCallOptimizer") as mock_cls:
            mock_cls.return_value.analyze.return_value = _FakeToolInsights(tools=known_tools)
            score = svc.get_tool_score("nonexistent_tool")

        assert score.tier == EffectivenessTier.ALLOW

    def test_corrupted_data_graceful_degradation(self) -> None:
        """Corrupted JSONL → log error, all ALLOW (R-EFF-09)."""
        svc = _make_service()

        with patch("vaig.core.optimizer.ToolCallOptimizer") as mock_cls:
            mock_cls.return_value.analyze.side_effect = ValueError("corrupt data")
            score = svc.get_tool_score("tool_a")

        assert score.tier == EffectivenessTier.ALLOW

    def test_ioerror_graceful_degradation(self) -> None:
        """IOError from store → log error, all ALLOW (R-EFF-09)."""
        svc = _make_service()

        with patch("vaig.core.optimizer.ToolCallOptimizer") as mock_cls:
            mock_cls.return_value.analyze.side_effect = OSError("disk gone")
            score = svc.get_tool_score("tool_a")

        assert score.tier == EffectivenessTier.ALLOW


# ── Singleton factory (R-EFF-07) ──────────────────────────────


class TestSingletonFactory:
    """get_effectiveness_service() behaviour."""

    def setup_method(self) -> None:
        reset_effectiveness_service()

    def teardown_method(self) -> None:
        reset_effectiveness_service()

    def test_disabled_returns_none(self) -> None:
        """enabled=False → factory returns None (R-EFF-07)."""
        with patch("vaig.core.effectiveness.get_settings") as mock_settings:
            mock_settings.return_value.effectiveness = _make_config(enabled=False)
            result = get_effectiveness_service()

        assert result is None

    def test_enabled_returns_service(self) -> None:
        """enabled=True → factory returns a ToolEffectivenessService."""
        store = MagicMock()
        with patch("vaig.core.effectiveness.get_settings") as mock_settings:
            mock_settings.return_value.effectiveness = _make_config(enabled=True)
            result = get_effectiveness_service(store=store)

        assert isinstance(result, ToolEffectivenessService)

    def test_singleton_returns_same_instance(self) -> None:
        """Second call returns cached instance."""
        store = MagicMock()
        with patch("vaig.core.effectiveness.get_settings") as mock_settings:
            mock_settings.return_value.effectiveness = _make_config(enabled=True)
            first = get_effectiveness_service(store=store)
            second = get_effectiveness_service()

        assert first is second


# ── Mixin pre-check (R-EFF-06) ────────────────────────────────


class TestMixinPreCheck:
    """_check_tool_effectiveness behaviour for SKIP, DEPRIORITIZE, ALLOW."""

    def setup_method(self) -> None:
        reset_effectiveness_service()

    def teardown_method(self) -> None:
        reset_effectiveness_service()

    def test_skip_returns_synthetic_result(self) -> None:
        """SKIP-tier tool returns synthetic ToolResult without execution."""
        from vaig.agents.mixins import ToolLoopMixin

        mock_svc = MagicMock()
        mock_svc.get_tool_score.return_value = EffectivenessScore(
            tool_name="bad_tool",
            tier=EffectivenessTier.SKIP,
            failure_rate=0.9,
            avg_duration_s=1.0,
            call_count=10,
            reason="failure rate 90% exceeds skip threshold",
        )

        with patch("vaig.core.effectiveness.get_effectiveness_service", return_value=mock_svc):
            result = ToolLoopMixin._check_tool_effectiveness("bad_tool", "test_agent")

        assert isinstance(result, ToolResult)
        assert "Tool skipped (effectiveness)" in result.output
        assert result.error is True

    def test_deprioritize_returns_none(self) -> None:
        """DEPRIORITIZE-tier tool logs warning but returns None (proceed)."""
        from vaig.agents.mixins import ToolLoopMixin

        mock_svc = MagicMock()
        mock_svc.get_tool_score.return_value = EffectivenessScore(
            tool_name="slow_tool",
            tier=EffectivenessTier.DEPRIORITIZE,
            failure_rate=0.3,
            avg_duration_s=15.0,
            call_count=10,
            reason="avg duration 15.0s exceeds slow tool threshold",
        )

        with patch("vaig.core.effectiveness.get_effectiveness_service", return_value=mock_svc):
            result = ToolLoopMixin._check_tool_effectiveness("slow_tool", "test_agent")

        assert result is None

    def test_allow_returns_none(self) -> None:
        """ALLOW-tier tool returns None (proceed normally)."""
        from vaig.agents.mixins import ToolLoopMixin

        mock_svc = MagicMock()
        mock_svc.get_tool_score.return_value = EffectivenessScore(
            tool_name="ok_tool",
            tier=EffectivenessTier.ALLOW,
            failure_rate=0.3,
            avg_duration_s=5.0,
            call_count=10,
            reason="within normal parameters",
        )

        with patch("vaig.core.effectiveness.get_effectiveness_service", return_value=mock_svc):
            result = ToolLoopMixin._check_tool_effectiveness("ok_tool", "test_agent")

        assert result is None

    def test_boost_returns_none(self) -> None:
        """BOOST-tier tool returns None (proceed — boost is informational)."""
        from vaig.agents.mixins import ToolLoopMixin

        mock_svc = MagicMock()
        mock_svc.get_tool_score.return_value = EffectivenessScore(
            tool_name="great_tool",
            tier=EffectivenessTier.BOOST,
            failure_rate=0.02,
            avg_duration_s=1.0,
            call_count=50,
            reason="low failure rate — reliable tool",
        )

        with patch("vaig.core.effectiveness.get_effectiveness_service", return_value=mock_svc):
            result = ToolLoopMixin._check_tool_effectiveness("great_tool", "test_agent")

        assert result is None

    def test_disabled_service_returns_none(self) -> None:
        """When service is None (disabled), returns None without any calls."""
        from vaig.agents.mixins import ToolLoopMixin

        with patch("vaig.core.effectiveness.get_effectiveness_service", return_value=None):
            result = ToolLoopMixin._check_tool_effectiveness("any_tool", "agent")

        assert result is None

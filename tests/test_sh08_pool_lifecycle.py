"""Unit tests for SH-08: lazy-init enrichment pool and GeminiClient lifecycle.

Verifies that ThreadPoolExecutor and GeminiClient are created lazily, reused
across calls, and cleaned up by close().
"""
from __future__ import annotations

import concurrent.futures
from unittest.mock import MagicMock, patch


def _make_skill():
    from vaig.skills.service_health.skill import ServiceHealthSkill

    return ServiceHealthSkill()


def _make_empty_report():
    from vaig.skills.service_health.schema import HealthReport

    return HealthReport.model_construct(recommendations=[])


def _run_enrichment(skill, report=None):
    """Call _enrich_report_recommendations with all I/O mocked out."""
    if report is None:
        report = _make_empty_report()

    mock_settings = MagicMock()
    mock_settings.models.default = "gemini-2.5-pro"

    mock_gemini = MagicMock()

    with (
        patch("vaig.core.config.get_settings", return_value=mock_settings),
        patch("vaig.core.client.GeminiClient", return_value=mock_gemini),
        patch(
            "vaig.skills.service_health.recommendation_enricher.enrich_recommendations",
            return_value=MagicMock(),
        ) as mock_enrich,
        patch("asyncio.run", return_value=report),
    ):
        # Make enrich_recommendations return a coroutine-like object accepted by asyncio.run
        mock_enrich.return_value = MagicMock()
        result = skill._enrich_report_recommendations(report, overall_timeout=5.0)

    return result


class TestPoolLifecycle:
    def test_initial_attrs_are_none(self):
        """T-SH08-01: After construction, both lazy attrs are None."""
        skill = _make_skill()
        assert skill._enrichment_pool is None
        assert skill._gemini_client is None

    def test_first_call_creates_pool(self):
        """T-SH08-02: After first call to _enrich_report_recommendations, pool is a ThreadPoolExecutor."""
        skill = _make_skill()
        _run_enrichment(skill)
        assert isinstance(skill._enrichment_pool, concurrent.futures.ThreadPoolExecutor)

    def test_second_call_reuses_same_pool(self):
        """T-SH08-03: Two calls use the exact same pool instance (identity check)."""
        skill = _make_skill()
        _run_enrichment(skill)
        pool_after_first = skill._enrichment_pool

        _run_enrichment(skill)
        pool_after_second = skill._enrichment_pool

        assert pool_after_first is pool_after_second

    def test_close_resets_attrs_to_none(self):
        """T-SH08-04: close() sets both _enrichment_pool and _gemini_client to None."""
        skill = _make_skill()
        _run_enrichment(skill)

        assert skill._enrichment_pool is not None
        skill.close()

        assert skill._enrichment_pool is None
        assert skill._gemini_client is None

    def test_close_when_pool_is_none_does_not_raise(self):
        """T-SH08-05: Calling close() with no pool is idempotent — does not raise."""
        skill = _make_skill()
        assert skill._enrichment_pool is None
        skill.close()  # must not raise
        skill.close()  # second call also must not raise

    def test_after_close_new_call_creates_fresh_pool(self):
        """T-SH08-06: After close(), next _enrich_report_recommendations gets a NEW pool."""
        skill = _make_skill()
        _run_enrichment(skill)
        pool_before_close = skill._enrichment_pool

        skill.close()
        _run_enrichment(skill)
        pool_after_close = skill._enrichment_pool

        assert pool_after_close is not None
        assert pool_after_close is not pool_before_close

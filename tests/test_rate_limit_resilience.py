"""Integration tests for SPEC-RATE-04 + SPEC-RATE-05 resilience features.

G-6: 429 storm end-to-end — mock GeminiClient returning 429 twice then success.
H-1: Clean run produces no Run Quality section.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors as genai_errors

from vaig.core.client import GeminiClient
from vaig.core.config import (
    GCPConfig,
    GenerationConfig,
    ModelsConfig,
    RetryConfig,
    Settings,
)
from vaig.core.exceptions import GeminiRateLimitError
from vaig.core.quality import QualityIssue, QualityIssueKind, RunQualityCollector

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture()
def settings() -> Settings:
    """Settings with a fast retry config and fallback model."""
    return Settings(
        gcp=GCPConfig(project_id="test-project", location="us-central1"),
        generation=GenerationConfig(),
        models=ModelsConfig(default="gemini-2.5-pro", fallback="gemini-2.5-flash"),
        retry=RetryConfig(
            max_retries=6,
            initial_delay=0.01,
            max_delay=0.05,
            backoff_multiplier=2.0,
            retryable_status_codes=[429, 500, 502, 503, 504],
            rate_limit_initial_delay=0.01,
        ),
    )


# ── G-6: 429 storm end-to-end ─────────────────────────────────


class TestRateLimitResilienceIntegration:
    """G-6 — Mock GeminiClient: 429 twice then success, check fallback_active."""

    @patch("vaig.core.client.time.sleep")
    def test_429_twice_then_success_triggers_fallback(
        self,
        mock_sleep: MagicMock,
        settings: Settings,
    ) -> None:
        """GeminiClient with fallback: 429×4 (attempt 3 triggers switch), then success → fallback_active=True."""
        client = GeminiClient(settings, fallback_model="gemini-2.5-flash")

        call_count = 0

        def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 4:
                raise genai_errors.ClientError(429, "Resource exhausted")
            return "result-ok"

        result = client._retry_with_backoff(fn)

        assert result == "result-ok"
        assert client.fallback_active is True

    @patch("vaig.core.client.time.sleep")
    def test_quality_issue_model_degraded_recorded_on_fallback(
        self,
        mock_sleep: MagicMock,
        settings: Settings,
    ) -> None:
        """After fallback fires, a MODEL_DEGRADED QualityIssue can be recorded."""
        client = GeminiClient(settings, fallback_model="gemini-2.5-flash")

        call_count = 0

        def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 4:
                raise genai_errors.ClientError(429, "Resource exhausted")
            return "result-ok"

        client._retry_with_backoff(fn)

        # Simulate orchestrator recording the degradation
        collector = RunQualityCollector()
        if client.fallback_active:
            collector.record_kind(
                QualityIssueKind.MODEL_DEGRADED,
                where="test-agent",
                detail="switched to gemini-2.5-flash",
            )

        assert collector.has_kind(QualityIssueKind.MODEL_DEGRADED)
        assert len(collector) == 1

    @patch("vaig.core.client.time.sleep")
    def test_no_fallback_model_raises_on_exhausted_429s(
        self,
        mock_sleep: MagicMock,
        settings: Settings,
    ) -> None:
        """With no fallback model, GeminiRateLimitError raised and fallback_active stays False."""
        client = GeminiClient(settings, fallback_model=None)
        fn = MagicMock(side_effect=genai_errors.ClientError(429, "Resource exhausted"))

        with pytest.raises(GeminiRateLimitError):
            client._retry_with_backoff(fn)

        assert client.fallback_active is False

    @patch("vaig.core.client.time.sleep")
    def test_run_quality_empty_on_clean_run(
        self,
        mock_sleep: MagicMock,
        settings: Settings,
    ) -> None:
        """Clean run (no 429s, no failures) → run_quality is empty."""
        client = GeminiClient(settings, fallback_model="gemini-2.5-flash")
        fn = MagicMock(return_value="success")

        client._retry_with_backoff(fn)

        # No fallback was needed
        assert client.fallback_active is False

        # Orchestrator would produce empty run_quality
        collector = RunQualityCollector()
        if client.fallback_active:
            collector.record_kind(QualityIssueKind.MODEL_DEGRADED, where="test-agent")

        assert collector.issues == []


# ── H-1: Clean run produces no Run Quality section ────────────


class TestCleanRunNoQualitySection:
    """H-1 — Clean run: Run Quality section absent from report output."""

    def test_empty_run_quality_produces_no_section(self) -> None:
        """run_quality == [] → reporter renders no Run Quality section."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        result = ServiceHealthSkill._render_run_quality_section([])
        assert result == ""
        assert "Run Quality" not in result

    def test_none_run_quality_produces_no_section(self) -> None:
        """run_quality=None → reporter renders no Run Quality section."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        result = ServiceHealthSkill._render_run_quality_section(None)
        assert result == ""
        assert "Run Quality" not in result

    def test_no_model_degraded_produces_no_section(self) -> None:
        """Non-degraded issues → no section prepended."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        issues = [
            QualityIssue(kind=QualityIssueKind.AGENT_FAILED, where="ag"),
            QualityIssue(kind=QualityIssueKind.CONTEXT_TRUNCATED, where="attach"),
        ]
        result = ServiceHealthSkill._render_run_quality_section(issues)
        assert result == ""

    def test_run_quality_list_empty_from_clean_run(self, settings: Settings) -> None:
        """A client that never fallbacks → collector stays empty."""
        with patch("vaig.core.client.time.sleep"):
            client = GeminiClient(settings, fallback_model="gemini-2.5-flash")
            fn = MagicMock(return_value="all-good")
            client._retry_with_backoff(fn)

        assert not client.fallback_active

        collector = RunQualityCollector()
        # Only record on actual fallback
        if client.fallback_active:
            collector.record_kind(QualityIssueKind.MODEL_DEGRADED, where="clean-agent")

        assert collector.issues == []
        assert len(collector) == 0

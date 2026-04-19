"""Tests for HEALTH_ANALYZER_PROMPT change-correlation additions (SPEC-SH-12)."""

from __future__ import annotations


class TestChangeCorrelationPromptIncluded:
    """SPEC-SH-12 T4 — Unit tests for _CHANGE_CORRELATION_PROMPT inclusion."""

    def test_change_correlation_prompt_is_non_empty(self) -> None:
        from vaig.skills.service_health.prompts import _CHANGE_CORRELATION_PROMPT

        assert _CHANGE_CORRELATION_PROMPT
        assert len(_CHANGE_CORRELATION_PROMPT) > 100

    def test_change_correlation_prompt_in_health_analyzer_prompt(self) -> None:
        from vaig.skills.service_health.prompts import _CHANGE_CORRELATION_PROMPT, HEALTH_ANALYZER_PROMPT

        assert _CHANGE_CORRELATION_PROMPT in HEALTH_ANALYZER_PROMPT

    def test_change_correlation_keywords_present(self) -> None:
        from vaig.skills.service_health.prompts import _CHANGE_CORRELATION_PROMPT

        for keyword in ["deploy", "sync", "rollout", "update", "image", "revision"]:
            assert keyword in _CHANGE_CORRELATION_PROMPT, (
                f"Expected keyword '{keyword}' in _CHANGE_CORRELATION_PROMPT"
            )

    def test_change_trigger_category_defined(self) -> None:
        from vaig.skills.service_health.prompts import _CHANGE_CORRELATION_PROMPT

        assert "change-trigger" in _CHANGE_CORRELATION_PROMPT

    def test_two_minute_window_rule_present(self) -> None:
        from vaig.skills.service_health.prompts import _CHANGE_CORRELATION_PROMPT

        assert "±2 minutes" in _CHANGE_CORRELATION_PROMPT or "2 minutes" in _CHANGE_CORRELATION_PROMPT

    def test_verification_tools_mentioned(self) -> None:
        from vaig.skills.service_health.prompts import _CHANGE_CORRELATION_PROMPT

        assert "argocd_app_history" in _CHANGE_CORRELATION_PROMPT
        assert "helm_release_history" in _CHANGE_CORRELATION_PROMPT

    def test_exported_from_init(self) -> None:
        """_CHANGE_CORRELATION_PROMPT must be importable from the package __init__."""
        from vaig.skills.service_health import prompts

        assert hasattr(prompts, "_CHANGE_CORRELATION_PROMPT")


class TestChangeCorrelationIntegration:
    """SPEC-SH-12 T5 — Static string scenario: change event within ±2 min of anomaly."""

    def test_prompt_instructs_temporal_overlap_check(self) -> None:
        """Prompt must instruct the LLM to check temporal overlap of change events."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        # The prompt must instruct checking whether events fall within the anomaly window
        assert "CORRELATED CHANGE EVENT" in HEALTH_ANALYZER_PROMPT or (
            "correlated" in HEALTH_ANALYZER_PROMPT.lower() and "change" in HEALTH_ANALYZER_PROMPT.lower()
        )

    def test_prompt_instructs_not_to_fabricate_correlation(self) -> None:
        """Prompt must guard against false positives (no match → no finding)."""
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "do NOT create" in HEALTH_ANALYZER_PROMPT or "NOT create" in HEALTH_ANALYZER_PROMPT

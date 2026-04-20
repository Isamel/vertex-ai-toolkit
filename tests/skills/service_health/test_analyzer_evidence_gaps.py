"""T9–T10 — Verify that HEALTH_ANALYZER_PROMPT contains evidence gap synthesis instructions."""


class TestAnalyzerPromptEvidenceGaps:
    """HEALTH_ANALYZER_PROMPT must include evidence_gaps and investigation_coverage sections."""

    def test_prompt_contains_evidence_gaps(self) -> None:
        from vaig.skills.service_health.prompts._analyzer import HEALTH_ANALYZER_PROMPT

        assert "evidence_gaps" in HEALTH_ANALYZER_PROMPT

    def test_prompt_contains_investigation_coverage(self) -> None:
        from vaig.skills.service_health.prompts._analyzer import HEALTH_ANALYZER_PROMPT

        assert "investigation_coverage" in HEALTH_ANALYZER_PROMPT

    def test_prompt_contains_gap_reasons(self) -> None:
        from vaig.skills.service_health.prompts._analyzer import HEALTH_ANALYZER_PROMPT

        assert "not_called" in HEALTH_ANALYZER_PROMPT
        assert "error" in HEALTH_ANALYZER_PROMPT
        assert "empty_result" in HEALTH_ANALYZER_PROMPT

    def test_evidence_gaps_prompt_exported(self) -> None:
        from vaig.skills.service_health.prompts._analyzer import _EVIDENCE_GAPS_PROMPT

        assert isinstance(_EVIDENCE_GAPS_PROMPT, str)
        assert len(_EVIDENCE_GAPS_PROMPT) > 0

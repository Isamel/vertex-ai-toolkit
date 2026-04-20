"""Tests for evidence gap tracking instructions in sub-gatherer prompts (evidence-gaps change)."""

from __future__ import annotations


class TestToolTrackingInstructionsConstant:
    """T8 — TOOL_TRACKING_INSTRUCTIONS constant exists and contains key keywords."""

    def test_constant_is_non_empty(self) -> None:
        from vaig.skills.service_health.prompts._sub_gatherers import TOOL_TRACKING_INSTRUCTIONS

        assert TOOL_TRACKING_INSTRUCTIONS
        assert len(TOOL_TRACKING_INSTRUCTIONS) > 50

    def test_constant_contains_evidence_gaps_keyword(self) -> None:
        from vaig.skills.service_health.prompts._sub_gatherers import TOOL_TRACKING_INSTRUCTIONS

        assert "evidence_gaps" in TOOL_TRACKING_INSTRUCTIONS

    def test_constant_contains_reason_values(self) -> None:
        from vaig.skills.service_health.prompts._sub_gatherers import TOOL_TRACKING_INSTRUCTIONS

        for reason in ("not_called", "error", "empty_result"):
            assert reason in TOOL_TRACKING_INSTRUCTIONS, (
                f"Expected reason '{reason}' in TOOL_TRACKING_INSTRUCTIONS"
            )


class TestNodeGathererPromptContainsGapTracking:
    """T8 — node gatherer prompt includes evidence_gaps instructions."""

    def test_standard_prompt_contains_evidence_gaps(self) -> None:
        from vaig.skills.service_health.prompts._sub_gatherers import build_node_gatherer_prompt

        prompt = build_node_gatherer_prompt(is_autopilot=False)
        assert "evidence_gaps" in prompt

    def test_autopilot_prompt_contains_evidence_gaps(self) -> None:
        from vaig.skills.service_health.prompts._sub_gatherers import build_node_gatherer_prompt

        prompt = build_node_gatherer_prompt(is_autopilot=True)
        assert "evidence_gaps" in prompt


class TestWorkloadGathererPromptContainsGapTracking:
    """T8 — workload gatherer prompt includes evidence_gaps instructions."""

    def test_prompt_contains_evidence_gaps(self) -> None:
        from vaig.skills.service_health.prompts._sub_gatherers import build_workload_gatherer_prompt

        prompt = build_workload_gatherer_prompt()
        assert "evidence_gaps" in prompt


class TestDatadogGathererPromptContainsGapTracking:
    """T8 — datadog gatherer prompt includes evidence_gaps instructions."""

    def test_prompt_contains_evidence_gaps(self) -> None:
        from vaig.skills.service_health.prompts._sub_gatherers import build_datadog_gatherer_prompt

        prompt = build_datadog_gatherer_prompt(datadog_api_enabled=True)
        assert "evidence_gaps" in prompt


class TestEventGathererPromptContainsGapTracking:
    """T8 — event gatherer prompt includes evidence_gaps instructions."""

    def test_prompt_contains_evidence_gaps(self) -> None:
        from vaig.skills.service_health.prompts._sub_gatherers import build_event_gatherer_prompt

        prompt = build_event_gatherer_prompt()
        assert "evidence_gaps" in prompt


class TestLoggingGathererPromptContainsGapTracking:
    """T8 — logging gatherer prompt includes evidence_gaps instructions."""

    def test_prompt_contains_evidence_gaps(self) -> None:
        from vaig.skills.service_health.prompts._sub_gatherers import build_logging_gatherer_prompt

        prompt = build_logging_gatherer_prompt()
        assert "evidence_gaps" in prompt

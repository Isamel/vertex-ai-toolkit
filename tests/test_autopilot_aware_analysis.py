"""Tests for autopilot-aware-analysis feature.

Verifies that:
- ``build_node_gatherer_prompt(is_autopilot=False)`` returns the standard 6-step prompt
- ``build_node_gatherer_prompt(is_autopilot=True)`` returns a lightweight 2-tool-call prompt
- The Autopilot prompt prohibits ``get_node_conditions`` and ``kubectl_top``
- The Standard prompt includes ``get_node_conditions`` and ``kubectl_top``
- ``build_autopilot_instruction()`` returns the updated directive text
- The Autopilot instruction contains "CONTEXT ONLY", "NotReady", and "WORKLOAD-LEVEL"
- The default value of ``is_autopilot`` is ``False``
"""
from __future__ import annotations

import inspect


class TestBuildNodeGathererPromptStandard:
    """Tests for the standard (non-Autopilot) path of build_node_gatherer_prompt."""

    def test_default_returns_full_standard_prompt(self) -> None:
        """Calling with no args (is_autopilot=False default) returns the standard prompt."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        result = build_node_gatherer_prompt()
        assert isinstance(result, str)
        assert len(result) > 200

    def test_false_returns_full_standard_prompt(self) -> None:
        """Calling with is_autopilot=False explicitly returns the standard prompt."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        result = build_node_gatherer_prompt(is_autopilot=False)
        assert isinstance(result, str)
        assert len(result) > 200

    def test_standard_prompt_contains_get_node_conditions(self) -> None:
        """Standard prompt MUST include get_node_conditions (6-step investigation)."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        result = build_node_gatherer_prompt(is_autopilot=False)
        assert "get_node_conditions" in result

    def test_standard_prompt_contains_kubectl_top(self) -> None:
        """Standard prompt MUST include kubectl_top for node utilisation."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        result = build_node_gatherer_prompt(is_autopilot=False)
        assert "kubectl_top" in result

    def test_standard_prompt_cluster_overview_header(self) -> None:
        """Standard prompt output section must be ## Cluster Overview (downstream dependency)."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        result = build_node_gatherer_prompt(is_autopilot=False)
        assert "## Cluster Overview" in result

    def test_standard_default_equals_false(self) -> None:
        """build_node_gatherer_prompt() and build_node_gatherer_prompt(False) must be identical."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        assert build_node_gatherer_prompt() == build_node_gatherer_prompt(is_autopilot=False)


class TestBuildNodeGathererPromptAutopilot:
    """Tests for the Autopilot-specific path of build_node_gatherer_prompt."""

    def test_autopilot_returns_lightweight_prompt(self) -> None:
        """Calling with is_autopilot=True returns a prompt (non-empty string)."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        result = build_node_gatherer_prompt(is_autopilot=True)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_autopilot_prompt_prohibits_get_node_conditions(self) -> None:
        """Autopilot prompt MUST explicitly prohibit get_node_conditions per node."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        result = build_node_gatherer_prompt(is_autopilot=True)
        # The Autopilot prompt mentions get_node_conditions only to PROHIBIT it
        assert "Do NOT call get_node_conditions" in result

    def test_autopilot_prompt_prohibits_kubectl_top(self) -> None:
        """Autopilot prompt MUST explicitly prohibit kubectl_top."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        result = build_node_gatherer_prompt(is_autopilot=True)
        # The Autopilot prompt mentions kubectl_top only to PROHIBIT it
        assert "Do NOT call kubectl_top" in result

    def test_autopilot_prompt_cluster_overview_header(self) -> None:
        """Autopilot prompt output section must be ## Cluster Overview (downstream dependency)."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        result = build_node_gatherer_prompt(is_autopilot=True)
        assert "## Cluster Overview" in result

    def test_autopilot_prompt_prohibits_get_events_kube_system(self) -> None:
        """Autopilot prompt must explicitly prohibit get_events for kube-system."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        result = build_node_gatherer_prompt(is_autopilot=True)
        assert "kube-system" in result

    def test_autopilot_and_standard_prompts_are_different(self) -> None:
        """The two prompt variants must be distinct."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        standard = build_node_gatherer_prompt(is_autopilot=False)
        autopilot = build_node_gatherer_prompt(is_autopilot=True)
        assert standard != autopilot

    def test_autopilot_prompt_shorter_than_standard(self) -> None:
        """Autopilot prompt (2 tools) must be shorter than standard prompt (6 steps)."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        standard = build_node_gatherer_prompt(is_autopilot=False)
        autopilot = build_node_gatherer_prompt(is_autopilot=True)
        assert len(autopilot) < len(standard)


class TestBuildNodeGathererPromptSignature:
    """Tests for the function signature and default values."""

    def test_is_autopilot_default_is_false(self) -> None:
        """is_autopilot parameter MUST default to False."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        sig = inspect.signature(build_node_gatherer_prompt)
        param = sig.parameters.get("is_autopilot")
        assert param is not None, "is_autopilot parameter must exist"
        assert param.default is False

    def test_is_autopilot_annotation_is_bool(self) -> None:
        """is_autopilot parameter MUST be annotated as bool."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        sig = inspect.signature(build_node_gatherer_prompt)
        param = sig.parameters.get("is_autopilot")
        assert param is not None
        assert param.annotation is bool


class TestBuildAutopilotInstruction:
    """Tests for the updated build_autopilot_instruction() in language.py."""

    def test_returns_string(self) -> None:
        """build_autopilot_instruction(True) must return a non-empty string."""
        from vaig.core.language import build_autopilot_instruction

        result = build_autopilot_instruction(True)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_false_returns_empty_string(self) -> None:
        """build_autopilot_instruction(False) must return empty string."""
        from vaig.core.language import build_autopilot_instruction

        assert build_autopilot_instruction(False) == ""

    def test_none_returns_empty_string(self) -> None:
        """build_autopilot_instruction(None) must return empty string."""
        from vaig.core.language import build_autopilot_instruction

        assert build_autopilot_instruction(None) == ""

    def test_contains_context_only(self) -> None:
        """Autopilot instruction must state node data is CONTEXT ONLY."""
        from vaig.core.language import build_autopilot_instruction

        result = build_autopilot_instruction(True)
        assert "CONTEXT ONLY" in result

    def test_contains_notready_is_normal(self) -> None:
        """Autopilot instruction must mention that NotReady is normal."""
        from vaig.core.language import build_autopilot_instruction

        result = build_autopilot_instruction(True)
        assert "NotReady" in result

    def test_contains_workload_level(self) -> None:
        """Autopilot instruction must focus analysis on WORKLOAD-LEVEL health."""
        from vaig.core.language import build_autopilot_instruction

        result = build_autopilot_instruction(True)
        assert "WORKLOAD-LEVEL" in result

    def test_contains_kubectl_top_not_available(self) -> None:
        """Autopilot instruction must state kubectl_top is NOT available."""
        from vaig.core.language import build_autopilot_instruction

        result = build_autopilot_instruction(True)
        assert "kubectl_top" in result
        assert "NOT available" in result

    def test_contains_no_node_level_actions(self) -> None:
        """Autopilot instruction must prohibit node-level actions."""
        from vaig.core.language import build_autopilot_instruction

        result = build_autopilot_instruction(True)
        assert "node-level actions" in result

    def test_contains_resource_requests_mandatory(self) -> None:
        """Autopilot instruction must flag missing resource requests."""
        from vaig.core.language import build_autopilot_instruction

        result = build_autopilot_instruction(True)
        assert "MANDATORY" in result or "mandatory" in result.lower()
        assert "resource requests" in result.lower()

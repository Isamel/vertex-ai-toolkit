"""Tests for SPEC-V2-AUDIT-14: autonomous prompt overlays.

Verifies:
- AUTONOMOUS_OVERLAY constants exist and are non-empty in both prompt modules
- Overlays are accessible via the prompts package public API
- skill.py wires the analyzer overlay when investigation.enabled is True
- skill.py wires the investigator overlay when investigation.enabled is True
- Overlays are NOT applied when investigation.enabled is False (analyzer)
"""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. Module-level constants
# ---------------------------------------------------------------------------

class TestAnalyzerOverlayConstant:
    def test_exists_in_module(self) -> None:
        from vaig.skills.service_health.prompts._analyzer import AUTONOMOUS_OVERLAY

        assert isinstance(AUTONOMOUS_OVERLAY, str)
        assert len(AUTONOMOUS_OVERLAY) > 0

    def test_mentions_investigation_plan(self) -> None:
        from vaig.skills.service_health.prompts._analyzer import AUTONOMOUS_OVERLAY

        assert "InvestigationPlan" in AUTONOMOUS_OVERLAY or "investigation_plan" in AUTONOMOUS_OVERLAY

    def test_mentions_causal_chain(self) -> None:
        from vaig.skills.service_health.prompts._analyzer import AUTONOMOUS_OVERLAY

        assert "causal" in AUTONOMOUS_OVERLAY.lower()

    def test_in_module_all(self) -> None:
        from vaig.skills.service_health.prompts import _analyzer

        assert "AUTONOMOUS_OVERLAY" in _analyzer.__all__


class TestInvestigatorOverlayConstant:
    def test_exists_in_module(self) -> None:
        from vaig.skills.service_health.prompts._investigator import AUTONOMOUS_OVERLAY

        assert isinstance(AUTONOMOUS_OVERLAY, str)
        assert len(AUTONOMOUS_OVERLAY) > 0

    def test_mentions_confidence(self) -> None:
        from vaig.skills.service_health.prompts._investigator import AUTONOMOUS_OVERLAY

        assert "confidence" in AUTONOMOUS_OVERLAY.lower()

    def test_mentions_contradiction(self) -> None:
        from vaig.skills.service_health.prompts._investigator import AUTONOMOUS_OVERLAY

        assert "contradiction" in AUTONOMOUS_OVERLAY.lower()


# ---------------------------------------------------------------------------
# 2. Package-level exports
# ---------------------------------------------------------------------------

class TestPackageExports:
    def test_analyzer_overlay_exported(self) -> None:
        from vaig.skills.service_health.prompts import ANALYZER_AUTONOMOUS_OVERLAY

        assert isinstance(ANALYZER_AUTONOMOUS_OVERLAY, str)
        assert len(ANALYZER_AUTONOMOUS_OVERLAY) > 0

    def test_investigator_overlay_exported(self) -> None:
        from vaig.skills.service_health.prompts import INVESTIGATOR_AUTONOMOUS_OVERLAY

        assert isinstance(INVESTIGATOR_AUTONOMOUS_OVERLAY, str)
        assert len(INVESTIGATOR_AUTONOMOUS_OVERLAY) > 0

    def test_both_in_package_all(self) -> None:
        import vaig.skills.service_health.prompts as pkg

        assert "ANALYZER_AUTONOMOUS_OVERLAY" in pkg.__all__
        assert "INVESTIGATOR_AUTONOMOUS_OVERLAY" in pkg.__all__


# ---------------------------------------------------------------------------
# 3. skill.py wiring — overlay applied/omitted based on settings
# ---------------------------------------------------------------------------

def _make_mock_settings(*, investigation_enabled: bool) -> MagicMock:
    """Build a MagicMock that mimics the subset of Settings used in get_agents_config."""
    settings = MagicMock()
    settings.investigation.enabled = investigation_enabled
    settings.investigation.autonomous_mode = False
    settings.investigation.max_iterations = 5
    settings.investigation.budget_per_run_usd = 0.0
    settings.datadog.enabled = False
    settings.gke.default_namespace = "default"
    settings.gke.cluster_name = "test-cluster"
    settings.gke.location = "us-central1"
    settings.gke.model_copy.return_value = settings.gke
    settings.gke.model_copy.return_value.default_namespace = "default"
    settings.gke.model_copy.return_value.cluster_name = "test-cluster"
    settings.gke.model_copy.return_value.location = "us-central1"
    return settings


def _get_agent_instruction(agents: list[dict], name: str) -> str:
    for agent in agents:
        if agent.get("name") == name:
            return agent["system_instruction"]
    raise AssertionError(f"Agent '{name}' not found in agents config")


class TestSkillAnalyzerWiring:
    def _run_get_agents(self, investigation_enabled: bool) -> list[dict]:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill.__new__(ServiceHealthSkill)
        skill._prefetched_metrics = {"nodes": "", "pods": ""}
        skill._prefetched_dd_resolution = MagicMock()
        skill._prefetched_dd_resolution.dd_service_name = None
        mock_settings = _make_mock_settings(investigation_enabled=investigation_enabled)

        with (
            patch("vaig.core.config.get_settings", return_value=mock_settings),
            patch("vaig.tools.gke._clients.detect_autopilot", return_value=False),
        ):
            return skill.get_agents_config()

    def test_overlay_appended_when_investigation_enabled(self) -> None:
        from vaig.skills.service_health.prompts import (
            ANALYZER_AUTONOMOUS_OVERLAY,
            HEALTH_ANALYZER_PROMPT,
        )

        agents = self._run_get_agents(investigation_enabled=True)
        instruction = _get_agent_instruction(agents, "health_analyzer")

        assert instruction == HEALTH_ANALYZER_PROMPT + ANALYZER_AUTONOMOUS_OVERLAY

    def test_overlay_absent_when_investigation_disabled(self) -> None:
        from vaig.skills.service_health.prompts import (
            ANALYZER_AUTONOMOUS_OVERLAY,
            HEALTH_ANALYZER_PROMPT,
        )

        agents = self._run_get_agents(investigation_enabled=False)
        instruction = _get_agent_instruction(agents, "health_analyzer")

        assert instruction == HEALTH_ANALYZER_PROMPT
        assert ANALYZER_AUTONOMOUS_OVERLAY not in instruction


class TestSkillInvestigatorWiring:
    def _run_get_agents(self) -> list[dict]:
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill.__new__(ServiceHealthSkill)
        skill._prefetched_metrics = {"nodes": "", "pods": ""}
        skill._prefetched_dd_resolution = MagicMock()
        skill._prefetched_dd_resolution.dd_service_name = None
        mock_settings = _make_mock_settings(investigation_enabled=True)

        with (
            patch("vaig.core.config.get_settings", return_value=mock_settings),
            patch("vaig.tools.gke._clients.detect_autopilot", return_value=False),
        ):
            return skill.get_agents_config()

    def test_overlay_appended_to_investigator(self) -> None:
        from vaig.skills.service_health.prompts import (
            HEALTH_INVESTIGATOR_PROMPT,
            INVESTIGATOR_AUTONOMOUS_OVERLAY,
        )

        agents = self._run_get_agents()
        instruction = _get_agent_instruction(agents, "health_investigator")

        assert instruction == HEALTH_INVESTIGATOR_PROMPT + INVESTIGATOR_AUTONOMOUS_OVERLAY

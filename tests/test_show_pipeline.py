"""Tests for the ``--show-pipeline`` preview flag on ``vaig live``.

Acceptance criteria from AUDIT-12:
- ``vaig live --show-pipeline "x"`` exits 0.
- Output contains phase headings.
- No gcloud or API calls are made (mock guard raises if called).
- ``--show-pipeline json`` emits machine-readable JSON.
- Works with and without an orchestrated skill.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from vaig.cli.app import app
from vaig.core.config import Settings
from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase

runner = CliRunner()


@pytest.fixture(autouse=True)
def _mock_settings() -> Settings:
    """Provide a default Settings object, avoiding real config files."""
    settings = Settings()
    settings.skills.auto_routing = False
    with patch("vaig.cli._helpers._get_settings", return_value=settings):
        yield settings


@pytest.fixture(autouse=True)
def _no_api_calls() -> None:
    """Guard: raise if GeminiClient or gcloud is instantiated.

    Ensures ``--show-pipeline`` never triggers real API calls.
    """
    with (
        patch(
            "vaig.core.container.build_container",
            side_effect=AssertionError("build_container must not be called in --show-pipeline mode"),
        ),
        patch(
            "vaig.cli.commands.live._register_live_tools",
            side_effect=AssertionError("_register_live_tools must not be called in --show-pipeline mode"),
        ),
    ):
        yield


class FakePipelineSkill(BaseSkill):
    """Fake skill with a two-phase parallel+sequential pipeline for testing."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="fake-pipeline",
            display_name="Fake Pipeline Skill",
            description="A test skill for --show-pipeline testing",
            requires_live_tools=True,
        )

    def get_system_instruction(self) -> str:
        return "Test system instruction."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return f"phase={phase}"

    def get_agents_config(self, **kwargs: Any) -> list[dict]:
        return [
            {
                "name": "node_gatherer",
                "role": "Node Gatherer",
                "requires_tools": True,
                "parallel_group": "gather",
                "tool_categories": ["kubernetes"],
                "model": "gemini-2.5-pro",
                "max_iterations": 15,
            },
            {
                "name": "workload_gatherer",
                "role": "Workload Gatherer",
                "requires_tools": True,
                "parallel_group": "gather",
                "tool_categories": ["kubernetes", "scaling"],
                "model": "gemini-2.5-pro",
                "max_iterations": 20,
            },
            {
                "name": "health_analyzer",
                "role": "Health Analyzer",
                "requires_tools": False,
                "tool_categories": [],
                "model": "gemini-2.5-flash",
            },
        ]


# ── Helper patches shared by most tests ────────────────────────────────────


def _patch_no_skill_registry() -> Any:
    """Patch SkillRegistry so skill lookups always fail."""
    return patch(
        "vaig.cli.commands.live._helpers._get_settings",  # already done via fixture
    )


def _patch_skill_registry(skill: BaseSkill) -> Any:
    """Patch SkillRegistry.get() to return *skill*."""
    registry_mock = MagicMock()
    registry_mock.get.return_value = skill
    registry_mock.list_names.return_value = [skill.get_metadata().name]
    return patch("vaig.skills.registry.SkillRegistry", return_value=registry_mock)


# ═══════════════════════════════════════════════════════════════════════════
# Tests — InfraAgent path (no skill)
# ═══════════════════════════════════════════════════════════════════════════


class TestShowPipelineNoSkill:
    """--show-pipeline without a skill shows InfraAgent info and exits 0."""

    def test_exits_zero(self) -> None:
        result = runner.invoke(app, ["live", "--show-pipeline", "x"])
        assert result.exit_code == 0, result.output

    def test_contains_pipeline_preview_heading(self) -> None:
        result = runner.invoke(app, ["live", "--show-pipeline", "x"])
        assert "Pipeline Preview" in result.output

    def test_contains_infra_agent_mode(self) -> None:
        result = runner.invoke(app, ["live", "--show-pipeline", "x"])
        assert "INFRA_AGENT" in result.output

    def test_contains_config_section(self) -> None:
        result = runner.invoke(app, ["live", "--show-pipeline", "x"])
        assert "Configuration" in result.output

    def test_no_llm_calls_made(self) -> None:
        """The _no_api_calls fixture raises if build_container is called — this must not."""
        result = runner.invoke(app, ["live", "--show-pipeline", "x"])
        assert result.exit_code == 0, result.output


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Orchestrated skill path
# ═══════════════════════════════════════════════════════════════════════════


class TestShowPipelineWithSkill:
    """--show-pipeline --skill <name> shows phase headings for the skill's pipeline."""

    def test_exits_zero_with_skill(self) -> None:
        with _patch_skill_registry(FakePipelineSkill()):
            result = runner.invoke(app, ["live", "--show-pipeline", "--skill", "fake-pipeline", "x"])
        assert result.exit_code == 0, result.output

    def test_shows_phase_headings(self) -> None:
        with _patch_skill_registry(FakePipelineSkill()):
            result = runner.invoke(app, ["live", "--show-pipeline", "--skill", "fake-pipeline", "x"])
        assert "Phase 1" in result.output
        assert "Phase 2" in result.output

    def test_shows_gather_phase(self) -> None:
        with _patch_skill_registry(FakePipelineSkill()):
            result = runner.invoke(app, ["live", "--show-pipeline", "--skill", "fake-pipeline", "x"])
        assert "Gather" in result.output

    def test_shows_agent_names(self) -> None:
        with _patch_skill_registry(FakePipelineSkill()):
            result = runner.invoke(app, ["live", "--show-pipeline", "--skill", "fake-pipeline", "x"])
        assert "node_gatherer" in result.output
        assert "workload_gatherer" in result.output
        assert "health_analyzer" in result.output

    def test_no_llm_calls_with_skill(self) -> None:
        with _patch_skill_registry(FakePipelineSkill()):
            result = runner.invoke(app, ["live", "--show-pipeline", "--skill", "fake-pipeline", "x"])
        assert result.exit_code == 0, result.output


# ═══════════════════════════════════════════════════════════════════════════
# Tests — JSON output mode
# ═══════════════════════════════════════════════════════════════════════════


class TestShowPipelineJson:
    """--show-pipeline --show-pipeline-format json emits valid, machine-readable JSON."""

    def test_exits_zero_json(self) -> None:
        result = runner.invoke(app, ["live", "--show-pipeline", "--show-pipeline-format", "json", "x"])
        assert result.exit_code == 0, result.output

    def test_output_is_valid_json(self) -> None:
        result = runner.invoke(app, ["live", "--show-pipeline", "--show-pipeline-format", "json", "x"])
        parsed = json.loads(result.output.strip())
        assert isinstance(parsed, dict)

    def test_json_contains_phases_key(self) -> None:
        result = runner.invoke(app, ["live", "--show-pipeline", "--show-pipeline-format", "json", "x"])
        parsed = json.loads(result.output.strip())
        assert "phases" in parsed

    def test_json_contains_config_key(self) -> None:
        result = runner.invoke(app, ["live", "--show-pipeline", "--show-pipeline-format", "json", "x"])
        parsed = json.loads(result.output.strip())
        assert "config" in parsed

    def test_json_with_skill_has_skill_name(self) -> None:
        with _patch_skill_registry(FakePipelineSkill()):
            result = runner.invoke(
                app,
                ["live", "--show-pipeline", "--show-pipeline-format", "json", "--skill", "fake-pipeline", "x"],
            )
        parsed = json.loads(result.output.strip())
        assert parsed["skill"] == "Fake Pipeline Skill"

    def test_json_with_skill_phases_list_not_empty(self) -> None:
        with _patch_skill_registry(FakePipelineSkill()):
            result = runner.invoke(
                app,
                ["live", "--show-pipeline", "--show-pipeline-format", "json", "--skill", "fake-pipeline", "x"],
            )
        parsed = json.loads(result.output.strip())
        assert len(parsed["phases"]) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — _build_pipeline_phases helper
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildPipelinePhases:
    """Unit tests for the _build_pipeline_phases helper function."""

    def test_parallel_group_agents_share_a_phase(self) -> None:
        from vaig.cli.commands.live import _build_pipeline_phases

        agents = [
            {"name": "a", "parallel_group": "gather"},
            {"name": "b", "parallel_group": "gather"},
            {"name": "c"},
        ]
        phases = _build_pipeline_phases(agents)
        assert len(phases) == 2  # one gather phase + one sequential phase

    def test_first_phase_is_parallel(self) -> None:
        from vaig.cli.commands.live import _build_pipeline_phases

        agents = [
            {"name": "a", "parallel_group": "gather"},
            {"name": "b", "parallel_group": "gather"},
        ]
        phases = _build_pipeline_phases(agents)
        assert phases[0]["parallel"] is True

    def test_sequential_agent_becomes_own_phase(self) -> None:
        from vaig.cli.commands.live import _build_pipeline_phases

        agents = [{"name": "analyzer"}]
        phases = _build_pipeline_phases(agents)
        assert len(phases) == 1
        assert phases[0]["parallel"] is False

    def test_parallel_group_name_capitalised(self) -> None:
        from vaig.cli.commands.live import _build_pipeline_phases

        agents = [{"name": "a", "parallel_group": "gather"}]
        phases = _build_pipeline_phases(agents)
        assert phases[0]["name"] == "Gather"

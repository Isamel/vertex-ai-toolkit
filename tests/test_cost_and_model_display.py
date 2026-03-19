"""Tests for the cost+model display fixes.

Bug 1: _show_orchestrated_summary must use run_cost_usd (pre-accumulated)
       instead of recalculating cost with settings.models.default.

Bug 2: _format_models_used and _show_orchestrated_summary / _inject_report_metadata
       must show actual agent models, not settings.models.default.
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from vaig.agents.base import AgentResult
from vaig.agents.orchestrator import OrchestratorResult
from vaig.cli.commands.live import (
    _format_models_used,
    _inject_report_metadata,
    _show_orchestrated_summary,
)
from vaig.skills.base import SkillPhase

# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_orch_result(
    *,
    run_cost_usd: float = 0.0,
    models_used: list[str] | None = None,
    total_usage: dict[str, int] | None = None,
    agent_results: list[AgentResult] | None = None,
) -> OrchestratorResult:
    result = OrchestratorResult(skill_name="test", phase=SkillPhase.EXECUTE)
    result.run_cost_usd = run_cost_usd
    result.models_used = models_used or []
    result.total_usage = total_usage or {}
    result.agent_results = agent_results or []
    return result


def _capture_summary(orch_result: OrchestratorResult, *, model_id: str = "") -> str:
    """Capture terminal output of _show_orchestrated_summary."""
    buf = StringIO()
    con = Console(file=buf, force_terminal=False, width=200)
    # Monkeypatch the module-level console used by live.py
    import vaig.cli._helpers as helpers_mod
    original = helpers_mod.console
    helpers_mod.console = con  # type: ignore[assignment]

    import vaig.cli.commands.live as live_mod
    original_live = live_mod.console
    live_mod.console = con  # type: ignore[assignment]

    try:
        _show_orchestrated_summary(orch_result, model_id=model_id)
    finally:
        helpers_mod.console = original
        live_mod.console = original_live

    return buf.getvalue()


# ── _format_models_used ──────────────────────────────────────────────────────


class TestFormatModelsUsed:
    def test_empty_list_returns_empty_string(self) -> None:
        assert _format_models_used([]) == ""

    def test_single_model_returns_as_is(self) -> None:
        assert _format_models_used(["gemini-2.5-flash"]) == "gemini-2.5-flash"

    def test_same_model_repeated_shows_count(self) -> None:
        result = _format_models_used(["gemini-2.5-flash"] * 7)
        assert result == "gemini-2.5-flash ×7"

    def test_two_repeated_shows_count(self) -> None:
        result = _format_models_used(["gemini-2.5-flash", "gemini-2.5-flash"])
        assert result == "gemini-2.5-flash ×2"

    def test_mixed_models_shows_comma_separated(self) -> None:
        result = _format_models_used(["gemini-2.5-flash", "gemini-2.5-pro"])
        assert "gemini-2.5-flash" in result
        assert "gemini-2.5-pro" in result

    def test_single_occurrence_no_count_suffix(self) -> None:
        result = _format_models_used(["gemini-2.5-pro"])
        assert "×" not in result
        assert result == "gemini-2.5-pro"


# ── _show_orchestrated_summary — cost display ────────────────────────────────


class TestShowOrchestratedSummaryCost:
    def test_uses_precomputed_cost_when_available(self) -> None:
        """Bug 1: Must show run_cost_usd, not recalculate with wrong model."""
        orch = _make_orch_result(
            run_cost_usd=0.0045,  # < 0.01 → format_cost shows 4dp → "$0.0045"
            models_used=["gemini-2.5-flash"] * 3,
            total_usage={"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
        )
        output = _capture_summary(orch, model_id="gemini-2.5-pro")  # wrong default
        # Should show the pre-computed cost, not something recalculated with pro pricing
        assert "$0.0045" in output  # format_cost for < 0.01 shows 4dp

    def test_shows_flash_model_label_not_default(self) -> None:
        """Bug 2 (terminal): Must label cost with actual model, not settings.models.default."""
        orch = _make_orch_result(
            run_cost_usd=0.005,
            models_used=["gemini-2.5-flash"] * 5,
            total_usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        output = _capture_summary(orch, model_id="gemini-2.5-pro")
        assert "gemini-2.5-flash" in output
        assert "gemini-2.5-pro" not in output

    def test_falls_back_to_show_cost_line_when_zero_cost(self) -> None:
        """When run_cost_usd == 0.0, falls back to _show_cost_line recalculation."""
        orch = _make_orch_result(
            run_cost_usd=0.0,
            models_used=["gemini-2.5-flash"],
            total_usage={"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
        )
        # With fallback, cost is recalculated — just assert no crash and output has tokens
        output = _capture_summary(orch, model_id="gemini-2.5-flash")
        assert "1,500 total" in output

    def test_no_crash_on_mock_orch_result(self) -> None:
        """Regression: MagicMock OrchestratorResult must not raise TypeError."""
        mock_orch = MagicMock()
        mock_orch.agent_results = []
        mock_orch.success = True
        # run_cost_usd as MagicMock should be handled safely
        import vaig.cli.commands.live as live_mod
        original = live_mod.console
        buf = StringIO()
        con = Console(file=buf, force_terminal=False, width=200)
        live_mod.console = con  # type: ignore[assignment]
        try:
            _show_orchestrated_summary(mock_orch, model_id="gemini-2.5-pro")
        except TypeError:
            pytest.fail("_show_orchestrated_summary raised TypeError on MagicMock input")
        finally:
            live_mod.console = original


# ── _inject_report_metadata — model_used field ──────────────────────────────


class TestInjectReportMetadataModel:
    def _make_metadata(self, model_used: str = "") -> MagicMock:
        meta = MagicMock()
        meta.model_used = model_used
        meta.cost_metrics = None
        meta.tool_usage = None
        meta.cluster_name = "test-cluster"
        meta.project_id = "test-project"
        return meta

    def _make_report(self, model_used: str = "") -> MagicMock:
        report = MagicMock()
        report.metadata = self._make_metadata(model_used)
        return report

    def test_uses_actual_model_from_orch_result(self) -> None:
        """Bug 2 (HTML): Must use models from orch_result.models_used, not model_id arg."""
        orch = _make_orch_result(
            run_cost_usd=0.001,
            models_used=["gemini-2.5-flash"] * 3,
        )
        report = self._make_report(model_used="")

        _inject_report_metadata(
            report,
            model_id="gemini-2.5-pro",  # wrong default — should be ignored
            orch_result=orch,
        )

        assert report.metadata.model_used == "gemini-2.5-flash ×3"

    def test_falls_back_to_model_id_when_models_used_empty(self) -> None:
        """When models_used is empty, fall back to the provided model_id."""
        orch = _make_orch_result(models_used=[])
        report = self._make_report(model_used="")

        _inject_report_metadata(
            report,
            model_id="gemini-2.5-pro",
            orch_result=orch,
        )

        assert report.metadata.model_used == "gemini-2.5-pro"

    def test_preserves_existing_non_empty_model_used(self) -> None:
        """Existing non-empty model_used must not be overwritten."""
        orch = _make_orch_result(models_used=["gemini-2.5-flash"])
        report = self._make_report(model_used="custom-model")

        _inject_report_metadata(
            report,
            model_id="gemini-2.5-pro",
            orch_result=orch,
        )

        # _is_empty returns False for "custom-model", so it should not be changed
        assert report.metadata.model_used == "custom-model"

    def test_no_orch_result_uses_model_id_arg(self) -> None:
        """When no orch_result is provided, model_id arg is used directly."""
        report = self._make_report(model_used="")

        _inject_report_metadata(
            report,
            model_id="gemini-2.5-flash",
            orch_result=None,
        )

        assert report.metadata.model_used == "gemini-2.5-flash"

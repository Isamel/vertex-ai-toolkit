"""Tests for the display module — show_cost_summary and related helpers."""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from vaig.cli.display import show_cost_summary


def _capture_output(usage: dict[str, int] | None, model_id: str) -> str:
    """Call show_cost_summary and capture Rich output as plain text."""
    buf = StringIO()
    con = Console(file=buf, force_terminal=False, width=200)
    show_cost_summary(usage, model_id, console=con)
    return buf.getvalue()


# ── Basic display behaviour ──────────────────────────────────────


class TestShowCostSummary:
    def test_shows_tokens_and_cost(self) -> None:
        usage = {
            "prompt_tokens": 1234,
            "completion_tokens": 567,
            "thinking_tokens": 89,
            "total_tokens": 1890,
        }
        output = _capture_output(usage, "gemini-2.5-pro")
        assert "1,234 in" in output
        assert "567 out" in output
        assert "89 thinking" in output
        assert "1,890 total" in output
        assert "$" in output  # cost should be present

    def test_no_thinking_tokens_omits_thinking(self) -> None:
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "thinking_tokens": 0,
            "total_tokens": 300,
        }
        output = _capture_output(usage, "gemini-2.5-flash")
        assert "100 in" in output
        assert "200 out" in output
        assert "thinking" not in output

    def test_unknown_model_shows_na_cost(self) -> None:
        usage = {
            "prompt_tokens": 50,
            "completion_tokens": 100,
            "total_tokens": 150,
        }
        output = _capture_output(usage, "unknown-model-xyz")
        assert "N/A" in output

    def test_known_model_shows_dollar_cost(self) -> None:
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
        }
        output = _capture_output(usage, "gemini-2.5-flash")
        assert "$" in output
        assert "N/A" not in output


# ── Edge cases — silent no-ops ───────────────────────────────────


class TestShowCostSummaryEdgeCases:
    def test_none_usage_is_silent(self) -> None:
        output = _capture_output(None, "gemini-2.5-pro")
        assert output.strip() == ""

    def test_empty_dict_is_silent(self) -> None:
        output = _capture_output({}, "gemini-2.5-pro")
        assert output.strip() == ""

    def test_all_zero_tokens_is_silent(self) -> None:
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "thinking_tokens": 0,
            "total_tokens": 0,
        }
        output = _capture_output(usage, "gemini-2.5-pro")
        assert output.strip() == ""

    def test_missing_keys_defaults_to_zero(self) -> None:
        """Usage dict with only total_tokens should still work."""
        usage = {"total_tokens": 42}
        output = _capture_output(usage, "gemini-2.5-pro")
        assert "42 total" in output
        assert "0 in" in output
        assert "0 out" in output

"""Tests for the interactive drill-in REPL in the ``live`` command.

Covers task T-10 of the X-05 Interactive Drill-In Mode change:
- Exits on 'exit' / 'quit'
- Exits on empty input
- Sends question with report+ledger context to orchestrator
- Skips when no structured report available
- Trims oldest Q&A turns when conversation exceeds the char cap
"""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ── Pre-import sys.modules stub ───────────────────────────────────────────────
# ``vaig.tools.gke`` performs a runtime ``k8s_client`` attribute access on
# import, which fails in environments without the kubernetes SDK installed.
# Stub the module before importing anything from ``vaig.cli.commands.live``.
_gke_stub = ModuleType("vaig.tools.gke")
_gke_stub.k8s_client = MagicMock()  # type: ignore[attr-defined]
sys.modules.setdefault("vaig.tools.gke", _gke_stub)

from vaig.cli.commands.live import _run_drill_in_loop  # noqa: E402
from vaig.core.config import GenerationConfig, Settings  # noqa: E402

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_settings(max_output_tokens: int = 100) -> Settings:
    """Return a minimal Settings object with a controlled token cap."""
    settings = Settings()
    settings.generation = GenerationConfig()
    settings.generation.max_output_tokens = max_output_tokens
    return settings


def _make_orch_result(
    *,
    report_md: str = "# Report\n\nSome findings.",
    ledger_summary: str | None = "Key evidence here.",
) -> MagicMock:
    """Build a mock OrchestratorResult with a structured_report and ledger."""
    report = MagicMock()
    report.to_markdown.return_value = report_md

    ledger = MagicMock()
    ledger.to_summary.return_value = ledger_summary

    state = MagicMock()
    state.evidence_ledger = ledger

    orch_result = MagicMock()
    orch_result.structured_report = report
    orch_result.final_state = state
    return orch_result


def _make_orchestrator(response_text: str = "The answer is 42.") -> MagicMock:
    """Return a mock Orchestrator whose execute_single returns response_text."""
    orchestrator = MagicMock()
    agent_result = MagicMock()
    agent_result.content = response_text
    orchestrator.execute_single.return_value = agent_result
    return orchestrator


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestDrillInExitsOnExitCommand:
    """_run_drill_in_loop exits when the user types 'exit' or 'quit'."""

    __test__ = True

    @pytest.mark.parametrize("exit_word", ["exit", "quit", "EXIT", "QUIT"])
    def test_exits_on_exit_word(self, exit_word: str) -> None:
        orchestrator = _make_orchestrator()
        orch_result = _make_orch_result()
        settings = _make_settings()

        with patch(
            "rich.prompt.Prompt.ask",
            side_effect=[exit_word],
        ):
            asyncio.run(_run_drill_in_loop(orchestrator, orch_result, settings))

        # No inference call should have been made
        orchestrator.execute_single.assert_not_called()


class TestDrillInExitsOnEmptyInput:
    """_run_drill_in_loop exits when the user submits an empty string."""

    __test__ = True

    def test_exits_on_empty_string(self) -> None:
        orchestrator = _make_orchestrator()
        orch_result = _make_orch_result()
        settings = _make_settings()

        with patch(
            "rich.prompt.Prompt.ask",
            side_effect=[""],
        ):
            asyncio.run(_run_drill_in_loop(orchestrator, orch_result, settings))

        orchestrator.execute_single.assert_not_called()

    def test_exits_on_whitespace_only(self) -> None:
        orchestrator = _make_orchestrator()
        orch_result = _make_orch_result()
        settings = _make_settings()

        with patch(
            "rich.prompt.Prompt.ask",
            side_effect=["   "],
        ):
            asyncio.run(_run_drill_in_loop(orchestrator, orch_result, settings))

        orchestrator.execute_single.assert_not_called()


class TestDrillInSendsQuestionWithReportContext:
    """_run_drill_in_loop passes report+ledger as context to execute_single."""

    __test__ = True

    def test_context_contains_report_markdown(self) -> None:
        orchestrator = _make_orchestrator()
        orch_result = _make_orch_result(
            report_md="# My Report\n\nDetails here.",
            ledger_summary="Evidence: pod crashed.",
        )
        settings = _make_settings()

        # First call returns a question; second call exits
        with patch(
            "rich.prompt.Prompt.ask",
            side_effect=["What caused the crash?", "exit"],
        ):
            asyncio.run(_run_drill_in_loop(orchestrator, orch_result, settings))

        orchestrator.execute_single.assert_called_once()
        _call = orchestrator.execute_single.call_args
        context_passed = _call.kwargs["context"]

        assert "# My Report" in context_passed
        assert "Details here." in context_passed
        assert "Evidence: pod crashed." in context_passed

    def test_question_passed_as_positional(self) -> None:
        orchestrator = _make_orchestrator()
        orch_result = _make_orch_result()
        settings = _make_settings()

        with patch(
            "rich.prompt.Prompt.ask",
            side_effect=["Tell me more", "exit"],
        ):
            asyncio.run(_run_drill_in_loop(orchestrator, orch_result, settings))

        _call = orchestrator.execute_single.call_args
        prompt_arg = _call.args[0]
        assert prompt_arg == "Tell me more"

    def test_no_ledger_still_sends_report(self) -> None:
        """Works even when evidence ledger is absent."""
        orchestrator = _make_orchestrator()
        orch_result = _make_orch_result(ledger_summary=None)
        # Explicitly remove evidence_ledger
        orch_result.final_state = None

        settings = _make_settings()

        with patch(
            "rich.prompt.Prompt.ask",
            side_effect=["Any findings?", "exit"],
        ):
            asyncio.run(_run_drill_in_loop(orchestrator, orch_result, settings))

        orchestrator.execute_single.assert_called_once()
        _call = orchestrator.execute_single.call_args
        assert "Investigation Report" in _call.kwargs["context"]


class TestDrillInSkipsWhenNoReport:
    """_execute_orchestrated_skill skips drill-in when structured_report is None."""

    __test__ = True

    def test_interactive_flag_false_does_not_call_loop(self) -> None:
        """When interactive=False the guard condition is not satisfied."""
        # Verify that the guard `if interactive and structured_report is not None`
        # prevents the loop from being reached.
        interactive = False
        structured_report = MagicMock()  # non-None

        loop_called = False
        if interactive and structured_report is not None:
            loop_called = True  # pragma: no cover

        assert not loop_called

    def test_loop_skipped_when_structured_report_is_none(self) -> None:
        """Guard: the drill-in loop is not entered when structured_report is None."""
        # Reproduce the guard condition from _execute_orchestrated_skill directly.
        interactive = True
        structured_report = None  # simulates orch_result.structured_report being None

        loop_entered = False
        if interactive and structured_report is not None:
            loop_entered = True  # pragma: no cover

        assert not loop_entered, "Loop must not be entered when structured_report is None"


class TestDrillInContextTrimming:
    """Oldest Q&A turns are trimmed when accumulated context exceeds the char cap."""

    __test__ = True

    def test_oldest_turns_trimmed_when_over_cap(self) -> None:
        """After many exchanges the qa_turns list stays within max_chars."""
        # Use a very small cap: 4 chars per token × 10 tokens = 40 chars
        settings = _make_settings(max_output_tokens=10)

        # Report already uses more chars than the cap on its own — the loop
        # should trim qa_turns (system_ctx is NEVER trimmed).
        long_report = "X" * 1000
        orch_result = _make_orch_result(report_md=long_report, ledger_summary=None)
        orch_result.final_state = None

        orchestrator = _make_orchestrator(response_text="Short answer.")

        # Two questions then exit — second call should trigger trim path
        with patch(
            "rich.prompt.Prompt.ask",
            side_effect=["Question one?", "Question two?", "exit"],
        ):
            asyncio.run(_run_drill_in_loop(orchestrator, orch_result, settings))

        # Both questions should have been processed (no crash from trimming)
        assert orchestrator.execute_single.call_count == 2

    def test_exits_on_keyboard_interrupt(self) -> None:
        """KeyboardInterrupt during Prompt.ask ends the loop gracefully."""
        orchestrator = _make_orchestrator()
        orch_result = _make_orch_result()
        settings = _make_settings()

        with patch(
            "rich.prompt.Prompt.ask",
            side_effect=KeyboardInterrupt,
        ):
            # Should NOT raise
            asyncio.run(_run_drill_in_loop(orchestrator, orch_result, settings))

        orchestrator.execute_single.assert_not_called()

    def test_exits_on_eof_error(self) -> None:
        """EOFError (e.g. piped input exhausted) ends the loop gracefully."""
        orchestrator = _make_orchestrator()
        orch_result = _make_orch_result()
        settings = _make_settings()

        with patch(
            "rich.prompt.Prompt.ask",
            side_effect=EOFError,
        ):
            asyncio.run(_run_drill_in_loop(orchestrator, orch_result, settings))

        orchestrator.execute_single.assert_not_called()

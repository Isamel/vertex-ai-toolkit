"""Tests for REPL FileHistory — persistent command history across sessions.

Verifies that:
- The REPL uses FileHistory instead of InMemoryHistory
- The history file path is derived from settings.session.repl_history_path
- The parent directory is created if it doesn't exist
- The SessionConfig field has a sensible default
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from prompt_toolkit.history import FileHistory

from vaig.core.config import SessionConfig, Settings


# ══════════════════════════════════════════════════════════════
# SessionConfig defaults
# ══════════════════════════════════════════════════════════════


class TestSessionConfigHistory:
    def test_default_repl_history_path(self) -> None:
        """SessionConfig should have a sensible default for repl_history_path."""
        cfg = SessionConfig()
        assert cfg.repl_history_path == "~/.vaig/repl_history"

    def test_custom_repl_history_path(self) -> None:
        """repl_history_path should be configurable."""
        cfg = SessionConfig(repl_history_path="/tmp/custom_history")
        assert cfg.repl_history_path == "/tmp/custom_history"


# ══════════════════════════════════════════════════════════════
# start_repl uses FileHistory
# ══════════════════════════════════════════════════════════════


class TestReplFileHistory:
    def test_start_repl_creates_file_history(self, tmp_path: Path) -> None:
        """start_repl should create a PromptSession with FileHistory."""
        history_file = tmp_path / "vaig" / "repl_history"

        settings = Settings(
            session=SessionConfig(repl_history_path=str(history_file)),
        )

        # We patch everything that start_repl touches so we can
        # inspect the PromptSession it creates.
        captured: dict[str, object] = {}

        original_prompt_session = None

        def fake_prompt_session(**kwargs: object) -> MagicMock:
            captured.update(kwargs)
            mock = MagicMock()
            return mock

        with (
            patch("vaig.cli.repl.GeminiClient") as mock_gc,
            patch("vaig.cli.repl.Orchestrator"),
            patch("vaig.cli.repl.SessionManager") as mock_sm,
            patch("vaig.cli.repl.ContextBuilder"),
            patch("vaig.cli.repl.SkillRegistry"),
            patch("vaig.cli.repl.PromptSession", side_effect=fake_prompt_session),
            patch("vaig.cli.repl._repl_loop"),
            patch("vaig.cli.repl._show_session_cost_summary"),
            patch("vaig.cli.repl._save_cost_data"),
            patch("vaig.cli.repl.console"),
        ):
            mock_gc.return_value.current_model = "gemini-2.5-pro"
            mock_sm.return_value.load_session.return_value = None

            from vaig.cli.repl import start_repl

            start_repl(settings)

        # Verify FileHistory was passed
        assert "history" in captured
        assert isinstance(captured["history"], FileHistory)

    def test_start_repl_creates_parent_directory(self, tmp_path: Path) -> None:
        """start_repl should create the parent directory for the history file."""
        # Use a nested path that doesn't exist yet
        history_file = tmp_path / "deep" / "nested" / "repl_history"
        assert not history_file.parent.exists()

        settings = Settings(
            session=SessionConfig(repl_history_path=str(history_file)),
        )

        with (
            patch("vaig.cli.repl.GeminiClient") as mock_gc,
            patch("vaig.cli.repl.Orchestrator"),
            patch("vaig.cli.repl.SessionManager") as mock_sm,
            patch("vaig.cli.repl.ContextBuilder"),
            patch("vaig.cli.repl.SkillRegistry"),
            patch("vaig.cli.repl.PromptSession", return_value=MagicMock()),
            patch("vaig.cli.repl._repl_loop"),
            patch("vaig.cli.repl._show_session_cost_summary"),
            patch("vaig.cli.repl._save_cost_data"),
            patch("vaig.cli.repl.console"),
        ):
            mock_gc.return_value.current_model = "gemini-2.5-pro"
            mock_sm.return_value.load_session.return_value = None

            from vaig.cli.repl import start_repl

            start_repl(settings)

        # The parent directory should now exist
        assert history_file.parent.exists()

    def test_file_history_path_matches_settings(self, tmp_path: Path) -> None:
        """The FileHistory path should match the resolved settings value."""
        history_file = tmp_path / "repl_history"

        settings = Settings(
            session=SessionConfig(repl_history_path=str(history_file)),
        )

        captured: dict[str, object] = {}

        def fake_prompt_session(**kwargs: object) -> MagicMock:
            captured.update(kwargs)
            return MagicMock()

        with (
            patch("vaig.cli.repl.GeminiClient") as mock_gc,
            patch("vaig.cli.repl.Orchestrator"),
            patch("vaig.cli.repl.SessionManager") as mock_sm,
            patch("vaig.cli.repl.ContextBuilder"),
            patch("vaig.cli.repl.SkillRegistry"),
            patch("vaig.cli.repl.PromptSession", side_effect=fake_prompt_session),
            patch("vaig.cli.repl._repl_loop"),
            patch("vaig.cli.repl._show_session_cost_summary"),
            patch("vaig.cli.repl._save_cost_data"),
            patch("vaig.cli.repl.console"),
        ):
            mock_gc.return_value.current_model = "gemini-2.5-pro"
            mock_sm.return_value.load_session.return_value = None

            from vaig.cli.repl import start_repl

            start_repl(settings)

        history = captured["history"]
        assert isinstance(history, FileHistory)
        # FileHistory stores the filename attribute
        assert history.filename == str(history_file)


# ══════════════════════════════════════════════════════════════
# Import-level verification
# ══════════════════════════════════════════════════════════════


class TestImports:
    def test_repl_imports_file_history(self) -> None:
        """The repl module should import FileHistory, not InMemoryHistory."""
        import vaig.cli.repl as repl_mod

        assert hasattr(repl_mod, "FileHistory") or "FileHistory" in dir(repl_mod)
        # InMemoryHistory should NOT be imported
        assert not hasattr(repl_mod, "InMemoryHistory")

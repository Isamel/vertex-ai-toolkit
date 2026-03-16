"""Tests for Phase 4C session improvements: rename, search, show, auto-resume."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vaig.session.store import SessionStore

# store fixture is provided by conftest.py


@pytest.fixture
def populated_store(store: SessionStore) -> SessionStore:
    """Store with several sessions and messages."""
    sid1 = store.create_session(name="debug-auth", model="gemini-2.5-pro", skill="rca")
    store.add_message(sid1, role="user", content="Why is auth failing?")
    store.add_message(sid1, role="model", content="Let me analyze the auth logs.")

    sid2 = store.create_session(name="code-review", model="gemini-2.5-flash")
    store.add_message(sid2, role="user", content="Review this PR for me")

    sid3 = store.create_session(name="k8s-debugging", model="gemini-2.5-pro", skill="rca")
    store.add_message(sid3, role="user", content="Pods are crashing with OOM errors")
    store.add_message(sid3, role="model", content="Checking pod resource limits...")
    store.add_message(sid3, role="user", content="The auth service is affected too")
    return store


# ══════════════════════════════════════════════════════════════
# SessionStore.rename_session
# ══════════════════════════════════════════════════════════════
class TestStoreRenameSession:
    def test_rename_existing_session(self, store: SessionStore) -> None:
        sid = store.create_session(name="old-name", model="gemini-2.5-pro")
        result = store.rename_session(sid, "new-name")
        assert result is True
        session = store.get_session(sid)
        assert session is not None
        assert session["name"] == "new-name"

    def test_rename_updates_updated_at(self, store: SessionStore) -> None:
        sid = store.create_session(name="old", model="gemini-2.5-pro")
        session_before = store.get_session(sid)
        assert session_before is not None
        time.sleep(0.01)  # Ensure time difference
        store.rename_session(sid, "new")
        session_after = store.get_session(sid)
        assert session_after is not None
        assert session_after["updated_at"] >= session_before["updated_at"]

    def test_rename_nonexistent_returns_false(self, store: SessionStore) -> None:
        result = store.rename_session("nonexistent-id", "new-name")
        assert result is False

    def test_rename_preserves_other_fields(self, store: SessionStore) -> None:
        sid = store.create_session(name="original", model="gemini-2.5-flash", skill="rca")
        store.add_message(sid, role="user", content="Hello")
        store.rename_session(sid, "renamed")
        session = store.get_session(sid)
        assert session is not None
        assert session["model"] == "gemini-2.5-flash"
        assert session["skill"] == "rca"
        msgs = store.get_messages(sid)
        assert len(msgs) == 1


# ══════════════════════════════════════════════════════════════
# SessionStore.search_sessions (improved — now searches messages too)
# ══════════════════════════════════════════════════════════════
class TestStoreSearchSessions:
    def test_search_by_name(self, populated_store: SessionStore) -> None:
        results = populated_store.search_sessions("code-review")
        assert len(results) == 1
        assert results[0]["name"] == "code-review"

    def test_search_by_message_content(self, populated_store: SessionStore) -> None:
        results = populated_store.search_sessions("OOM")
        assert len(results) == 1
        assert results[0]["name"] == "k8s-debugging"

    def test_search_returns_message_count(self, populated_store: SessionStore) -> None:
        results = populated_store.search_sessions("k8s")
        assert len(results) == 1
        assert results[0]["message_count"] == 3

    def test_search_no_results(self, populated_store: SessionStore) -> None:
        results = populated_store.search_sessions("nonexistent-query-xyz")
        assert results == []

    def test_search_matches_across_sessions(self, populated_store: SessionStore) -> None:
        # "auth" appears in session name "debug-auth" AND in k8s-debugging message content
        results = populated_store.search_sessions("auth")
        names = {r["name"] for r in results}
        assert "debug-auth" in names
        assert "k8s-debugging" in names

    def test_search_case_insensitive_name(self, populated_store: SessionStore) -> None:
        results = populated_store.search_sessions("CODE")
        assert len(results) == 1
        assert results[0]["name"] == "code-review"


# ══════════════════════════════════════════════════════════════
# SessionStore.get_last_session
# ══════════════════════════════════════════════════════════════
class TestStoreGetLastSession:
    def test_get_last_session_returns_most_recent(self, store: SessionStore) -> None:
        store.create_session(name="first", model="gemini-2.5-pro")
        time.sleep(0.01)
        sid2 = store.create_session(name="second", model="gemini-2.5-flash")
        time.sleep(0.01)
        store.create_session(name="third", model="gemini-2.5-pro")

        # Update the second session to make it the most recent
        store.add_message(sid2, role="user", content="updated!")

        last = store.get_last_session()
        assert last is not None
        assert last["name"] == "second"
        assert last["id"] == sid2

    def test_get_last_session_empty_db(self, store: SessionStore) -> None:
        result = store.get_last_session()
        assert result is None

    def test_get_last_session_includes_message_count(self, store: SessionStore) -> None:
        sid = store.create_session(name="active", model="gemini-2.5-pro")
        store.add_message(sid, role="user", content="msg1")
        store.add_message(sid, role="model", content="msg2")

        last = store.get_last_session()
        assert last is not None
        assert last["message_count"] == 2


# ══════════════════════════════════════════════════════════════
# SessionManager — new methods
# ══════════════════════════════════════════════════════════════
class TestManagerRename:
    def test_rename_active_session(self, tmp_path: Path) -> None:
        from vaig.session.manager import SessionManager

        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        session = mgr.new_session("original")
        result = mgr.rename_session(session.id, "renamed")
        assert result is True
        assert mgr.active is not None
        assert mgr.active.name == "renamed"
        mgr.close()

    def test_rename_inactive_session(self, tmp_path: Path) -> None:
        from vaig.session.manager import SessionManager

        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        session1 = mgr.new_session("first")
        sid1 = session1.id
        mgr.new_session("second")  # Switch to second session
        result = mgr.rename_session(sid1, "renamed-first")
        assert result is True
        # Active session should still be "second"
        assert mgr.active is not None
        assert mgr.active.name == "second"
        mgr.close()


class TestManagerSearch:
    def test_search_delegates_to_store(self, tmp_path: Path) -> None:
        from vaig.session.manager import SessionManager

        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        mgr.new_session("rca-incident-42")
        mgr.new_session("code-review-pr-99")
        results = mgr.search_sessions("rca")
        assert len(results) == 1
        assert results[0]["name"] == "rca-incident-42"
        mgr.close()


class TestManagerResume:
    def test_resume_last_session(self, tmp_path: Path) -> None:
        from vaig.session.manager import SessionManager

        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        s1 = mgr.new_session("first")
        mgr.add_message("user", "hello from first")
        s2 = mgr.new_session("second")
        mgr.add_message("user", "hello from second")

        # Resume should load "second" (most recently updated)
        mgr2 = SessionManager(settings)
        loaded = mgr2.resume_last_session()
        assert loaded is not None
        assert loaded.name == "second"
        assert loaded.id == s2.id
        assert len(loaded.history) == 1
        mgr.close()
        mgr2.close()

    def test_resume_empty_db(self, tmp_path: Path) -> None:
        from vaig.session.manager import SessionManager

        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        result = mgr.resume_last_session()
        assert result is None
        mgr.close()

    def test_get_last_session(self, tmp_path: Path) -> None:
        from vaig.session.manager import SessionManager

        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        mgr.new_session("only-session")
        last = mgr.get_last_session()
        assert last is not None
        assert last["name"] == "only-session"
        mgr.close()


# ══════════════════════════════════════════════════════════════
# CLI commands
# ══════════════════════════════════════════════════════════════
class TestCLISessionsRename:
    def test_rename_command(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from vaig.cli.app import app

        runner = CliRunner()
        settings = _make_settings(tmp_path)

        with patch("vaig.cli._helpers._get_settings", return_value=settings):
            # Create a session first
            from vaig.session.manager import SessionManager

            mgr = SessionManager(settings)
            s = mgr._store.create_session(name="old-name", model="gemini-2.5-pro")
            mgr.close()

            result = runner.invoke(app, ["sessions", "rename", s, "new-name"])
            assert result.exit_code == 0
            assert "new-name" in result.output

    def test_rename_nonexistent(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from vaig.cli.app import app

        runner = CliRunner()
        settings = _make_settings(tmp_path)

        with patch("vaig.cli._helpers._get_settings", return_value=settings):
            result = runner.invoke(app, ["sessions", "rename", "bad-id", "name"])
            assert "not found" in result.output.lower() or result.exit_code == 0


class TestCLISessionsSearch:
    def test_search_command(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from vaig.cli.app import app

        runner = CliRunner()
        settings = _make_settings(tmp_path)

        with patch("vaig.cli._helpers._get_settings", return_value=settings):
            from vaig.session.store import SessionStore

            store = SessionStore(settings.db_path_resolved)
            store.create_session(name="rca-auth-issue", model="gemini-2.5-pro")
            store.create_session(name="code-review", model="gemini-2.5-flash")
            store.close()

            result = runner.invoke(app, ["sessions", "search", "rca"])
            assert result.exit_code == 0
            assert "rca-auth-issue" in result.output

    def test_search_no_results(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from vaig.cli.app import app

        runner = CliRunner()
        settings = _make_settings(tmp_path)

        with patch("vaig.cli._helpers._get_settings", return_value=settings):
            result = runner.invoke(app, ["sessions", "search", "nonexistent"])
            assert result.exit_code == 0
            assert "No sessions" in result.output


class TestCLISessionsShow:
    def test_show_command(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from vaig.cli.app import app

        runner = CliRunner()
        settings = _make_settings(tmp_path)

        with patch("vaig.cli._helpers._get_settings", return_value=settings):
            from vaig.session.store import SessionStore

            store = SessionStore(settings.db_path_resolved)
            sid = store.create_session(name="my-session", model="gemini-2.5-pro", skill="rca")
            store.add_message(sid, role="user", content="Hello AI!")
            store.add_message(sid, role="model", content="Hello human!")
            store.close()

            result = runner.invoke(app, ["sessions", "show", sid])
            assert result.exit_code == 0
            assert "my-session" in result.output
            assert "gemini-2.5-pro" in result.output

    def test_show_nonexistent(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from vaig.cli.app import app

        runner = CliRunner()
        settings = _make_settings(tmp_path)

        with patch("vaig.cli._helpers._get_settings", return_value=settings):
            result = runner.invoke(app, ["sessions", "show", "nonexistent-id"])
            assert "not found" in result.output.lower()


class TestCLISessionsList:
    def test_list_shows_message_count(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from vaig.cli.app import app

        runner = CliRunner()
        settings = _make_settings(tmp_path)

        with patch("vaig.cli._helpers._get_settings", return_value=settings):
            from vaig.session.store import SessionStore

            store = SessionStore(settings.db_path_resolved)
            sid = store.create_session(name="test-session", model="gemini-2.5-pro")
            store.add_message(sid, role="user", content="msg1")
            store.add_message(sid, role="model", content="msg2")
            store.close()

            result = runner.invoke(app, ["sessions", "list"])
            assert result.exit_code == 0
            assert "test-session" in result.output
            # Table should include Msgs column
            assert "Msgs" in result.output


# ══════════════════════════════════════════════════════════════
# Helper: _format_session_date
# ══════════════════════════════════════════════════════════════
class TestFormatSessionDate:
    def test_valid_iso_date(self) -> None:
        from vaig.cli.app import _format_session_date

        result = _format_session_date("2025-12-15T14:30:00+00:00")
        assert "2025-12-15" in result
        assert "14:30" in result

    def test_empty_string(self) -> None:
        from vaig.cli.app import _format_session_date

        result = _format_session_date("")
        assert result == "—"

    def test_invalid_date(self) -> None:
        from vaig.cli.app import _format_session_date

        result = _format_session_date("not-a-date-at-all")
        assert isinstance(result, str)
        assert len(result) > 0


# ══════════════════════════════════════════════════════════════
# Helper: _resolve_session_id
# ══════════════════════════════════════════════════════════════
class TestResolveSessionId:
    def test_full_id_match(self, tmp_path: Path) -> None:
        from vaig.cli.app import _resolve_session_id
        from vaig.session.manager import SessionManager

        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        s = mgr._store.create_session(name="test", model="gemini-2.5-pro")
        resolved = _resolve_session_id(mgr, s)
        assert resolved == s
        mgr.close()

    def test_prefix_match(self, tmp_path: Path) -> None:
        from vaig.cli.app import _resolve_session_id
        from vaig.session.manager import SessionManager

        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        s = mgr._store.create_session(name="test", model="gemini-2.5-pro")
        prefix = s[:8]
        resolved = _resolve_session_id(mgr, prefix)
        assert resolved == s
        mgr.close()

    def test_no_match_returns_prefix(self, tmp_path: Path) -> None:
        from vaig.cli.app import _resolve_session_id
        from vaig.session.manager import SessionManager

        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        resolved = _resolve_session_id(mgr, "nonexistent")
        assert resolved == "nonexistent"
        mgr.close()


# ══════════════════════════════════════════════════════════════
# Helper: create settings with tmp_path DB
# ══════════════════════════════════════════════════════════════
def _make_settings(tmp_path: Path) -> MagicMock:
    """Create a minimal Settings mock for session tests."""
    settings = MagicMock()
    db_path = tmp_path / "vaig_test.db"
    settings.db_path_resolved = str(db_path)
    settings.models.default = "gemini-2.5-pro"
    settings.session.auto_save = True
    settings.session.max_history_messages = 100
    return settings


# ══════════════════════════════════════════════════════════════
# Async SessionManager methods
# ══════════════════════════════════════════════════════════════
class TestAsyncSessionManager:
    async def test_async_new_session(self, tmp_path: Path) -> None:
        from vaig.session.manager import SessionManager

        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        session = await mgr.async_new_session("async-test")
        assert session.name == "async-test"
        assert session.model == "gemini-2.5-pro"
        assert mgr.active is not None
        assert mgr.active.id == session.id
        mgr.close()

    async def test_async_add_message(self, tmp_path: Path) -> None:
        from vaig.session.manager import SessionManager

        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        await mgr.async_new_session("msg-test")
        await mgr.async_add_message("user", "Hello async!")
        await mgr.async_add_message("assistant", "Hi back!", model="gemini-2.5-pro", token_count=10)

        assert mgr.active is not None
        assert len(mgr.active.history) == 2
        assert mgr.active.history[0].content == "Hello async!"
        mgr.close()

    async def test_async_add_message_no_session_raises(self, tmp_path: Path) -> None:
        from vaig.session.manager import SessionManager

        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        with pytest.raises(RuntimeError, match="No active session"):
            await mgr.async_add_message("user", "Should fail")
        mgr.close()

    async def test_async_add_message_persists(self, tmp_path: Path) -> None:
        from vaig.session.manager import SessionManager

        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        session = await mgr.async_new_session("persist-test")
        await mgr.async_add_message("user", "Persisted msg")

        messages = await mgr.async_get_session_messages(session.id)
        assert len(messages) == 1
        assert messages[0]["content"] == "Persisted msg"
        mgr.close()

    async def test_async_load_session(self, tmp_path: Path) -> None:
        from vaig.session.manager import SessionManager

        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        s = await mgr.async_new_session("to-load")
        await mgr.async_add_message("user", "msg1")
        await mgr.async_add_message("assistant", "msg2")
        sid = s.id

        # Create new manager to simulate fresh load
        mgr2 = SessionManager(settings)
        loaded = await mgr2.async_load_session(sid)
        assert loaded is not None
        assert loaded.name == "to-load"
        assert len(loaded.history) == 2
        mgr.close()
        mgr2.close()

    async def test_async_load_session_not_found(self, tmp_path: Path) -> None:
        from vaig.session.manager import SessionManager

        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        result = await mgr.async_load_session("nonexistent-id")
        assert result is None
        mgr.close()

    async def test_async_list_sessions(self, tmp_path: Path) -> None:
        from vaig.session.manager import SessionManager

        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        await mgr.async_new_session("s1")
        await mgr.async_new_session("s2")

        sessions = await mgr.async_list_sessions()
        assert len(sessions) == 2
        mgr.close()

    async def test_async_list_sessions_limit(self, tmp_path: Path) -> None:
        from vaig.session.manager import SessionManager

        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        for i in range(5):
            await mgr.async_new_session(f"session-{i}")

        sessions = await mgr.async_list_sessions(limit=2)
        assert len(sessions) == 2
        mgr.close()

    async def test_async_delete_session(self, tmp_path: Path) -> None:
        from vaig.session.manager import SessionManager

        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        s = await mgr.async_new_session("to-delete")
        result = await mgr.async_delete_session(s.id)
        assert result is True
        assert mgr.active is None  # active session cleared
        mgr.close()

    async def test_async_rename_session(self, tmp_path: Path) -> None:
        from vaig.session.manager import SessionManager

        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        s = await mgr.async_new_session("old-name")
        result = await mgr.async_rename_session(s.id, "new-name")
        assert result is True
        assert mgr.active is not None
        assert mgr.active.name == "new-name"
        mgr.close()

    async def test_async_search_sessions(self, tmp_path: Path) -> None:
        from vaig.session.manager import SessionManager

        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        await mgr.async_new_session("rca-incident")
        await mgr.async_new_session("code-review")

        results = await mgr.async_search_sessions("rca")
        assert len(results) == 1
        assert results[0]["name"] == "rca-incident"
        mgr.close()

    async def test_async_get_last_session(self, tmp_path: Path) -> None:
        from vaig.session.manager import SessionManager

        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        await mgr.async_new_session("first")
        await mgr.async_new_session("second")

        last = await mgr.async_get_last_session()
        assert last is not None
        assert last["name"] == "second"
        mgr.close()

    async def test_async_resume_last_session(self, tmp_path: Path) -> None:
        from vaig.session.manager import SessionManager

        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        await mgr.async_new_session("first")
        s2 = await mgr.async_new_session("second")
        await mgr.async_add_message("user", "latest msg")

        mgr2 = SessionManager(settings)
        loaded = await mgr2.async_resume_last_session()
        assert loaded is not None
        assert loaded.name == "second"
        assert loaded.id == s2.id
        mgr.close()
        mgr2.close()

    async def test_async_resume_last_session_empty(self, tmp_path: Path) -> None:
        from vaig.session.manager import SessionManager

        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        result = await mgr.async_resume_last_session()
        assert result is None
        mgr.close()

    async def test_async_save_and_load_cost_data(self, tmp_path: Path) -> None:
        from vaig.session.manager import SessionManager

        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        s = await mgr.async_new_session("cost-test")
        cost = {"total_cost": 0.05, "calls": 3}
        result = await mgr.async_save_cost_data(cost)
        assert result is True

        loaded = await mgr.async_load_cost_data(s.id)
        assert loaded is not None
        assert loaded["total_cost"] == 0.05
        mgr.close()

    async def test_async_save_cost_data_no_session(self, tmp_path: Path) -> None:
        from vaig.session.manager import SessionManager

        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        result = await mgr.async_save_cost_data({"total": 0})
        assert result is False
        mgr.close()

    async def test_async_close(self, tmp_path: Path) -> None:
        from vaig.session.manager import SessionManager

        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        await mgr.async_new_session("close-test")
        await mgr.async_close()
        assert mgr._store._aconn is None
        mgr.close()

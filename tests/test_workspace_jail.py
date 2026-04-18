"""Tests for WorkspaceJail context manager."""

from __future__ import annotations

from pathlib import Path

import pytest

from vaig.core.workspace_jail import WorkspaceJail, WorkspaceJailError

# ── Helpers ───────────────────────────────────────────────────


def _make_workspace(tmp_path: Path) -> Path:
    """Create a minimal workspace tree for tests."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "main.py").write_text("print('hello')\n")
    (ws / "sub").mkdir()
    (ws / "sub" / "util.py").write_text("x = 1\n")
    return ws


# ── Disabled mode ────────────────────────────────────────────


def test_disabled_returns_original_workspace(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    with WorkspaceJail(ws, enabled=False) as jail:
        assert jail.effective_path == ws.resolve()


def test_disabled_does_not_create_temp_dir(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    with WorkspaceJail(ws, enabled=False) as jail:
        assert jail._tmpdir is None


def test_disabled_no_copy_made(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    original_files = sorted(ws.rglob("*"))
    with WorkspaceJail(ws, enabled=False):
        pass
    assert sorted(ws.rglob("*")) == original_files


# ── Enabled mode — enter/exit ─────────────────────────────────


def test_enabled_creates_temp_copy(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    with WorkspaceJail(ws, enabled=True) as jail:
        effective = jail.effective_path
        assert effective.exists()
        assert effective != ws.resolve()
        assert (effective / "main.py").exists()
        assert (effective / "sub" / "util.py").exists()


def test_enabled_temp_dir_cleaned_after_exit(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    with WorkspaceJail(ws, enabled=True) as jail:
        tmpdir = jail._tmpdir
        assert tmpdir is not None
        assert tmpdir.exists()
    # After __exit__ the tmpdir should be gone
    assert not tmpdir.exists()


def test_enabled_temp_dir_cleaned_on_exception(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    captured_tmpdir: Path | None = None
    with pytest.raises(RuntimeError):
        with WorkspaceJail(ws, enabled=True) as jail:
            captured_tmpdir = jail._tmpdir
            raise RuntimeError("boom")
    assert captured_tmpdir is not None
    assert not captured_tmpdir.exists()


# ── sync_back ─────────────────────────────────────────────────


def test_sync_back_on_clean_exit(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    with WorkspaceJail(ws, enabled=True) as jail:
        # Modify a file in the jail
        (jail.effective_path / "main.py").write_text("print('modified')\n")
        # Add a new file
        (jail.effective_path / "new.py").write_text("y = 2\n")

    # After clean exit, changes should be in the original workspace
    assert (ws / "main.py").read_text() == "print('modified')\n"
    assert (ws / "new.py").read_text() == "y = 2\n"


def test_no_sync_back_on_exception(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    original_content = (ws / "main.py").read_text()
    with pytest.raises(RuntimeError):
        with WorkspaceJail(ws, enabled=True) as jail:
            (jail.effective_path / "main.py").write_text("corrupted content\n")
            raise RuntimeError("pipeline failed")

    # Original workspace must be untouched
    assert (ws / "main.py").read_text() == original_content


def test_sync_back_noop_when_disabled(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    jail = WorkspaceJail(ws, enabled=False)
    jail.__enter__()
    # sync_back with disabled jail should silently do nothing
    jail.sync_back()
    jail.__exit__(None, None, None)


# ── validate_path ─────────────────────────────────────────────


def test_validate_path_inside_jail(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    with WorkspaceJail(ws, enabled=True) as jail:
        resolved = jail.validate_path(Path("main.py"))
        assert resolved == (jail.effective_path / "main.py").resolve()


def test_validate_path_traversal_raises(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    with WorkspaceJail(ws, enabled=True) as jail:
        with pytest.raises(WorkspaceJailError, match="escapes jail boundary"):
            jail.validate_path(Path("../../etc/passwd"))


def test_validate_path_when_disabled(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    with WorkspaceJail(ws, enabled=False) as jail:
        resolved = jail.validate_path(Path("main.py"))
        assert resolved == (ws.resolve() / "main.py").resolve()


# ── ignore_patterns ───────────────────────────────────────────


def test_ignore_patterns_excluded_from_copy(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    (ws / ".git").mkdir()
    (ws / ".git" / "config").write_text("[core]\n")
    (ws / "node_modules").mkdir()
    (ws / "node_modules" / "pkg.js").write_text("module.exports = {}\n")

    with WorkspaceJail(ws, enabled=True) as jail:
        assert not (jail.effective_path / ".git").exists()
        assert not (jail.effective_path / "node_modules").exists()
        assert (jail.effective_path / "main.py").exists()


def test_custom_ignore_patterns(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    (ws / "dist").mkdir()
    (ws / "dist" / "bundle.js").write_text("/* bundle */\n")

    with WorkspaceJail(ws, enabled=True, ignore_patterns=["dist"]) as jail:
        assert not (jail.effective_path / "dist").exists()
        assert (jail.effective_path / "main.py").exists()

"""Tests for vaig.core.project — ensure_project_dir / ensure_project_subdir."""

from __future__ import annotations

from pathlib import Path

import pytest

from vaig.core.project import ensure_project_dir, ensure_project_subdir


class TestEnsureProjectDir:
    def test_creates_vaig_dir_in_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = ensure_project_dir()
        assert result == tmp_path / ".vaig"
        assert result.is_dir()

    def test_idempotent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        first = ensure_project_dir()
        second = ensure_project_dir()
        assert first == second
        assert second.is_dir()

    def test_returns_path_object(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = ensure_project_dir()
        assert isinstance(result, Path)


class TestEnsureProjectSubdir:
    def test_creates_named_subdir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = ensure_project_subdir("repo-cache")
        assert result == tmp_path / ".vaig" / "repo-cache"
        assert result.is_dir()

    def test_nested_subdir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = ensure_project_subdir("a/b")
        assert result == tmp_path / ".vaig" / "a" / "b"
        assert result.is_dir()

    def test_idempotent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        first = ensure_project_subdir("foo")
        second = ensure_project_subdir("foo")
        assert first == second
        assert second.is_dir()

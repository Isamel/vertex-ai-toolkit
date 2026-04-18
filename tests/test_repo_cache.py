"""Tests for vaig.core.repo_cache.RepoCache."""

from __future__ import annotations

from pathlib import Path

import pytest

from vaig.core.repo_cache import RepoCache


class TestRepoCache:
    def test_put_and_get(self, tmp_path: Path) -> None:
        cache = RepoCache(root=tmp_path)
        cache.put("acme", "repo", "main", "src/main.py", "content")
        assert cache.get("acme", "repo", "main", "src/main.py") == "content"

    def test_has_returns_false_for_missing(self, tmp_path: Path) -> None:
        cache = RepoCache(root=tmp_path)
        assert not cache.has("acme", "repo", "main", "src/missing.py")

    def test_has_returns_true_after_put(self, tmp_path: Path) -> None:
        cache = RepoCache(root=tmp_path)
        cache.put("acme", "repo", "main", "file.py", "x")
        assert cache.has("acme", "repo", "main", "file.py")

    def test_get_returns_none_for_missing(self, tmp_path: Path) -> None:
        cache = RepoCache(root=tmp_path)
        assert cache.get("acme", "repo", "main", "nope.py") is None

    def test_invalidate_removes_file(self, tmp_path: Path) -> None:
        cache = RepoCache(root=tmp_path)
        cache.put("acme", "repo", "main", "file.py", "data")
        result = cache.invalidate("acme", "repo", "main", "file.py")
        assert result is True
        assert not cache.has("acme", "repo", "main", "file.py")

    def test_invalidate_missing_returns_false(self, tmp_path: Path) -> None:
        cache = RepoCache(root=tmp_path)
        result = cache.invalidate("acme", "repo", "main", "ghost.py")
        assert result is False

    def test_mirrors_repo_structure(self, tmp_path: Path) -> None:
        cache = RepoCache(root=tmp_path)
        cache.put("acme", "repo", "main", "a/b/c.py", "deep")
        expected = tmp_path / "acme" / "repo" / "main" / "a" / "b" / "c.py"
        assert expected.exists()

    def test_multiple_refs_isolated(self, tmp_path: Path) -> None:
        cache = RepoCache(root=tmp_path)
        cache.put("acme", "repo", "main", "f.py", "on main")
        cache.put("acme", "repo", "dev", "f.py", "on dev")
        assert cache.get("acme", "repo", "main", "f.py") == "on main"
        assert cache.get("acme", "repo", "dev", "f.py") == "on dev"

    def test_root_property(self, tmp_path: Path) -> None:
        cache = RepoCache(root=tmp_path)
        assert cache.root == tmp_path

    def test_default_root_uses_vaig_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        cache = RepoCache()
        assert cache.root == tmp_path / ".vaig" / "repo-cache"
        assert cache.root.is_dir()

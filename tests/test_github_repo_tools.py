"""Tests for new GitHub repo tools — repo_search_code, repo_get_commits, repo_diff."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
from pydantic import SecretStr

from vaig.core.config import GitHubConfig
from vaig.tools.integrations.github import (
    repo_diff,
    repo_get_commits,
    repo_search_code,
)

# ── Fixtures / helpers ────────────────────────────────────────


def _make_config(
    token: str = "test-token",  # noqa: S107
    allowed_repos: list[str] | None = None,
) -> GitHubConfig:
    return GitHubConfig(
        enabled=True,
        token=SecretStr(token),
        allowed_repos=allowed_repos or [],
    )


def _mock_response(
    status_code: int,
    json_data: object | None = None,
    text: str | None = None,
) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = ValueError("no body")
    resp.text = text or ""
    return resp


# ── repo_search_code ──────────────────────────────────────────


class TestRepoSearchCode:
    def test_returns_matching_files(self) -> None:
        data = {
            "items": [
                {"path": "src/auth.py", "name": "auth.py"},
                {"path": "tests/test_auth.py", "name": "test_auth.py"},
            ]
        }
        with patch("httpx.get", return_value=_mock_response(200, data)):
            result = repo_search_code(
                config=_make_config(), owner="acme", repo="myrepo", query="def authenticate"
            )
        assert not result.error
        assert "src/auth.py" in result.output
        assert "tests/test_auth.py" in result.output

    def test_empty_results(self) -> None:
        with patch("httpx.get", return_value=_mock_response(200, {"items": []})):
            result = repo_search_code(
                config=_make_config(), owner="acme", repo="myrepo", query="xyz_nonexistent"
            )
        assert not result.error
        assert "No code matches" in result.output

    def test_401_unauthorized(self) -> None:
        with patch("httpx.get", return_value=_mock_response(401)):
            result = repo_search_code(
                config=_make_config(), owner="acme", repo="myrepo", query="test"
            )
        assert result.error
        assert "401" in result.output

    def test_404_not_found(self) -> None:
        with patch("httpx.get", return_value=_mock_response(404)):
            result = repo_search_code(
                config=_make_config(), owner="acme", repo="missing", query="test"
            )
        assert result.error
        assert "not found" in result.output.lower()

    def test_429_rate_limited(self) -> None:
        with patch("httpx.get", return_value=_mock_response(429)):
            result = repo_search_code(
                config=_make_config(), owner="acme", repo="myrepo", query="test"
            )
        assert result.error
        assert "429" in result.output

    def test_allowlist_rejection(self) -> None:
        result = repo_search_code(
            config=_make_config(allowed_repos=["acme/allowed"]),
            owner="acme",
            repo="forbidden",
            query="test",
        )
        assert result.error
        assert "not in the allowed_repos" in result.output

    def test_allowlist_passes_when_matched(self) -> None:
        data = {"items": [{"path": "main.py", "name": "main.py"}]}
        with patch("httpx.get", return_value=_mock_response(200, data)):
            result = repo_search_code(
                config=_make_config(allowed_repos=["acme/myrepo"]),
                owner="acme",
                repo="myrepo",
                query="test",
            )
        assert not result.error

    def test_output_wrapped_for_prompt_safety(self) -> None:
        data = {"items": [{"path": "evil.py", "name": "evil.py"}]}
        with patch("httpx.get", return_value=_mock_response(200, data)):
            result = repo_search_code(
                config=_make_config(), owner="acme", repo="myrepo", query="ignore"
            )
        # wrap_untrusted_content adds delimiters
        assert not result.error

    def test_timeout(self) -> None:
        with patch("httpx.get", side_effect=httpx.TimeoutException("timed out")):
            result = repo_search_code(
                config=_make_config(), owner="acme", repo="myrepo", query="test"
            )
        assert result.error
        assert "timed out" in result.output.lower()


# ── repo_get_commits ──────────────────────────────────────────


class TestRepoGetCommits:
    def _commit(
        self,
        sha: str = "abc1234",
        message: str = "fix bug",
        author: str = "Dev",
        date: str = "2024-01-01T00:00:00Z",
    ) -> dict:
        return {
            "sha": sha,
            "commit": {
                "message": message,
                "author": {"name": author, "date": date},
            },
        }

    def test_returns_commit_list(self) -> None:
        data = [self._commit("abc1234def", "feat: add login"), self._commit("xyz9876abc", "fix: typo")]
        with patch("httpx.get", return_value=_mock_response(200, data)):
            result = repo_get_commits(
                config=_make_config(), owner="acme", repo="myrepo"
            )
        assert not result.error
        assert "abc1234" in result.output
        assert "feat: add login" in result.output

    def test_empty_commits(self) -> None:
        with patch("httpx.get", return_value=_mock_response(200, [])):
            result = repo_get_commits(
                config=_make_config(), owner="acme", repo="myrepo"
            )
        assert not result.error
        assert "No commits found" in result.output

    def test_401_unauthorized(self) -> None:
        with patch("httpx.get", return_value=_mock_response(401)):
            result = repo_get_commits(
                config=_make_config(), owner="acme", repo="myrepo"
            )
        assert result.error
        assert "401" in result.output

    def test_404_not_found(self) -> None:
        with patch("httpx.get", return_value=_mock_response(404)):
            result = repo_get_commits(
                config=_make_config(), owner="acme", repo="missing"
            )
        assert result.error
        assert "not found" in result.output.lower()

    def test_429_rate_limited(self) -> None:
        with patch("httpx.get", return_value=_mock_response(429)):
            result = repo_get_commits(
                config=_make_config(), owner="acme", repo="myrepo"
            )
        assert result.error
        assert "429" in result.output

    def test_allowlist_rejection(self) -> None:
        result = repo_get_commits(
            config=_make_config(allowed_repos=["acme/allowed"]),
            owner="acme",
            repo="forbidden",
        )
        assert result.error
        assert "not in the allowed_repos" in result.output

    def test_path_filter_passed_as_param(self) -> None:
        captured: list[dict] = []

        def _fake_get(url: str, **kwargs: object) -> MagicMock:
            captured.append({"params": kwargs.get("params", {})})
            return _mock_response(200, [])

        with patch("httpx.get", side_effect=_fake_get):
            repo_get_commits(
                config=_make_config(), owner="acme", repo="myrepo", path="src/main.py"
            )

        assert captured[0]["params"].get("path") == "src/main.py"

    def test_limit_respected(self) -> None:
        commits = [self._commit(sha=f"sha{i:07d}", message=f"commit {i}") for i in range(50)]
        with patch("httpx.get", return_value=_mock_response(200, commits)):
            result = repo_get_commits(
                config=_make_config(), owner="acme", repo="myrepo", limit=5
            )
        # Output should contain at most 5 entries
        assert not result.error
        lines = [line for line in result.output.splitlines() if line.startswith("sha")]
        assert len(lines) <= 5


# ── repo_diff ─────────────────────────────────────────────────


class TestRepoDiff:
    _SAMPLE_DIFF = """\
diff --git a/src/main.py b/src/main.py
index abc..def 100644
--- a/src/main.py
+++ b/src/main.py
@@ -1,3 +1,4 @@
 def hello():
-    pass
+    print("hello")
+    return True
"""

    def test_returns_unified_diff(self) -> None:
        with patch("httpx.get", return_value=_mock_response(200, text=self._SAMPLE_DIFF)):
            result = repo_diff(
                config=_make_config(), owner="acme", repo="myrepo", base="main", head="feature"
            )
        assert not result.error
        assert "diff --git" in result.output

    def test_401_unauthorized(self) -> None:
        with patch("httpx.get", return_value=_mock_response(401, text="")):
            result = repo_diff(
                config=_make_config(), owner="acme", repo="myrepo", base="main", head="feature"
            )
        assert result.error
        assert "401" in result.output

    def test_404_not_found(self) -> None:
        with patch("httpx.get", return_value=_mock_response(404, text="")):
            result = repo_diff(
                config=_make_config(), owner="acme", repo="missing", base="main", head="head"
            )
        assert result.error
        assert "not found" in result.output.lower()

    def test_429_rate_limited(self) -> None:
        with patch("httpx.get", return_value=_mock_response(429, text="")):
            result = repo_diff(
                config=_make_config(), owner="acme", repo="myrepo", base="main", head="feature"
            )
        assert result.error
        assert "429" in result.output

    def test_allowlist_rejection(self) -> None:
        result = repo_diff(
            config=_make_config(allowed_repos=["acme/allowed"]),
            owner="acme",
            repo="forbidden",
            base="main",
            head="feature",
        )
        assert result.error
        assert "not in the allowed_repos" in result.output

    def test_diff_truncation(self) -> None:
        """Diff larger than 500 KB is truncated with a notice."""
        large_diff = "+" + "x" * (512 * 1024)
        with patch("httpx.get", return_value=_mock_response(200, text=large_diff)):
            result = repo_diff(
                config=_make_config(), owner="acme", repo="myrepo", base="main", head="feature"
            )
        assert not result.error
        assert "TRUNCATED" in result.output

    def test_empty_diff(self) -> None:
        with patch("httpx.get", return_value=_mock_response(200, text="   ")):
            result = repo_diff(
                config=_make_config(), owner="acme", repo="myrepo", base="main", head="main"
            )
        assert not result.error
        assert "No diff found" in result.output

    def test_path_filter_client_side(self) -> None:
        multi_diff = (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@ foo\n"
            "diff --git a/bar.py b/bar.py\n"
            "--- a/bar.py\n+++ b/bar.py\n@@ -1 +1 @@ bar\n"
        )
        with patch("httpx.get", return_value=_mock_response(200, text=multi_diff)):
            result = repo_diff(
                config=_make_config(),
                owner="acme",
                repo="myrepo",
                base="main",
                head="feature",
                path="foo.py",
            )
        assert not result.error
        assert "foo.py" in result.output
        assert "bar.py" not in result.output

    def test_timeout(self) -> None:
        with patch("httpx.get", side_effect=httpx.TimeoutException("timed out")):
            result = repo_diff(
                config=_make_config(), owner="acme", repo="myrepo", base="main", head="feature"
            )
        assert result.error
        assert "timed out" in result.output.lower()

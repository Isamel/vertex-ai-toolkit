"""Tests for vaig.tools.integrations.github — repo_list_tree and repo_read_file."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import httpx
import pytest

from vaig.core.config import GitHubConfig
from vaig.tools.integrations.github import repo_list_tree, repo_read_file


def _make_config(token: str = "test-token") -> GitHubConfig:
    return GitHubConfig(enabled=True, token=token)  # type: ignore[arg-type]


def _mock_response(status_code: int, json_data: object | None = None) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = ValueError("no body")
    return resp


# ── repo_list_tree ────────────────────────────────────────────────────────────


class TestRepoListTree:
    def test_returns_sorted_file_paths(self) -> None:
        tree_data = {
            "tree": [
                {"path": "b.py", "type": "blob"},
                {"path": "a.py", "type": "blob"},
                {"path": "src", "type": "tree"},
            ]
        }
        with patch("httpx.get", return_value=_mock_response(200, tree_data)):
            result = repo_list_tree(
                config=_make_config(), owner="acme", repo="myrepo", ref="main"
            )

        assert not result.error
        assert result.output == "a.py\nb.py"

    def test_uses_default_ref(self) -> None:
        with patch(
            "httpx.get",
            return_value=_mock_response(200, {"tree": [{"path": "x.py", "type": "blob"}]}),
        ):
            result = repo_list_tree(config=_make_config(), owner="acme", repo="myrepo")
        assert not result.error

    def test_empty_tree(self) -> None:
        with patch("httpx.get", return_value=_mock_response(200, {"tree": []})):
            result = repo_list_tree(config=_make_config(), owner="acme", repo="myrepo")
        assert not result.error
        assert "No files found" in result.output

    def test_404_returns_error(self) -> None:
        with patch("httpx.get", return_value=_mock_response(404)):
            result = repo_list_tree(config=_make_config(), owner="acme", repo="missing")
        assert result.error
        assert "not found" in result.output.lower()

    def test_401_returns_auth_error(self) -> None:
        with patch("httpx.get", return_value=_mock_response(401)):
            result = repo_list_tree(config=_make_config(), owner="acme", repo="myrepo")
        assert result.error
        assert "401" in result.output

    def test_429_rate_limited(self) -> None:
        with patch("httpx.get", return_value=_mock_response(429)):
            result = repo_list_tree(config=_make_config(), owner="acme", repo="myrepo")
        assert result.error
        assert "429" in result.output

    def test_timeout_returns_error(self) -> None:
        with patch("httpx.get", side_effect=httpx.TimeoutException("timed out")):
            result = repo_list_tree(config=_make_config(), owner="acme", repo="myrepo")
        assert result.error
        assert "timed out" in result.output.lower()

    def test_connection_error_returns_error(self) -> None:
        with patch("httpx.get", side_effect=httpx.RequestError("conn fail")):
            result = repo_list_tree(config=_make_config(), owner="acme", repo="myrepo")
        assert result.error

    def test_sends_auth_header(self) -> None:
        captured: list[dict] = []

        def _fake_get(url: str, **kwargs: object) -> MagicMock:
            captured.append({"url": url, "headers": kwargs.get("headers", {})})
            return _mock_response(200, {"tree": []})

        with patch("httpx.get", side_effect=_fake_get):
            repo_list_tree(config=_make_config(token="my-secret"), owner="acme", repo="myrepo")

        assert captured[0]["headers"].get("Authorization") == "Bearer my-secret"


# ── repo_read_file ────────────────────────────────────────────────────────────


class TestRepoReadFile:
    def _b64(self, content: str) -> str:
        return base64.b64encode(content.encode()).decode()

    def test_decodes_base64_content(self) -> None:
        with patch(
            "httpx.get",
            return_value=_mock_response(
                200, {"encoding": "base64", "content": self._b64("print('hello')\n")}
            ),
        ):
            result = repo_read_file(
                config=_make_config(), owner="acme", repo="myrepo", path="src/main.py"
            )
        assert not result.error
        assert result.output == "print('hello')\n"

    def test_404_returns_error(self) -> None:
        with patch("httpx.get", return_value=_mock_response(404)):
            result = repo_read_file(
                config=_make_config(), owner="acme", repo="myrepo", path="missing.py"
            )
        assert result.error
        assert "not found" in result.output.lower()

    def test_timeout_returns_error(self) -> None:
        with patch("httpx.get", side_effect=httpx.TimeoutException("timed out")):
            result = repo_read_file(
                config=_make_config(), owner="acme", repo="myrepo", path="main.py"
            )
        assert result.error

    def test_non_base64_encoding_returned_as_is(self) -> None:
        with patch(
            "httpx.get",
            return_value=_mock_response(200, {"encoding": "none", "content": "raw text"}),
        ):
            result = repo_read_file(
                config=_make_config(), owner="acme", repo="myrepo", path="file.txt"
            )
        assert not result.error
        assert result.output == "raw text"

    def test_403_returns_error(self) -> None:
        with patch("httpx.get", return_value=_mock_response(403)):
            result = repo_read_file(
                config=_make_config(), owner="acme", repo="myrepo", path="secret.py"
            )
        assert result.error
        assert "403" in result.output

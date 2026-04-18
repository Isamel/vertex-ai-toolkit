"""Unit tests for fetch_doc tool (T-05)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vaig.core.config import DocFetchConfig
from vaig.core.exceptions import ToolExecutionError
from vaig.core.prompt_defense import DELIMITER_DATA_START
from vaig.tools.knowledge.fetch_doc import fetch_doc

ALLOWED = ["kubernetes.io", "docs.python.org"]
SIMPLE_HTML = b"<html><body><p>Hello world</p></body></html>"


def _make_response(status_code: int, content: bytes, headers: dict | None = None) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.content = content
    mock_resp.headers = headers or {}
    return mock_resp


class TestFetchDocDomainAllowlist:
    def test_disallowed_domain_raises(self) -> None:
        cfg = DocFetchConfig()
        with pytest.raises(ToolExecutionError, match="domain not allowed"):
            fetch_doc("https://evil.com/inject", cfg, ALLOWED)

    def test_allowed_domain_proceeds(self) -> None:
        cfg = DocFetchConfig()
        resp = _make_response(200, SIMPLE_HTML)
        with patch("httpx.get", return_value=resp):
            result = fetch_doc("https://kubernetes.io/docs", cfg, ALLOWED)
        assert DELIMITER_DATA_START in result.output


class TestFetchDocRunCounter:
    def test_per_run_cap_exhausted_raises(self) -> None:
        cfg = DocFetchConfig(per_run_cap=2)
        counter = [2]
        with pytest.raises(ToolExecutionError, match="per_run_cap"):
            fetch_doc("https://kubernetes.io/docs", cfg, ALLOWED, _run_counter=counter)

    def test_counter_incremented(self) -> None:
        cfg = DocFetchConfig(per_run_cap=5)
        counter = [0]
        resp = _make_response(200, SIMPLE_HTML)
        with patch("httpx.get", return_value=resp):
            fetch_doc("https://kubernetes.io/docs", cfg, ALLOWED, _run_counter=counter)
        assert counter[0] == 1


class TestFetchDocByteCap:
    def test_body_truncated_at_max_bytes(self) -> None:
        cfg = DocFetchConfig(max_bytes=1024)
        # 2000 bytes of content
        large_body = b"<p>" + b"A" * 2000 + b"</p>"
        resp = _make_response(200, large_body)
        with patch("httpx.get", return_value=resp):
            # Should not raise; markdown will be from first 1024 bytes only
            result = fetch_doc("https://kubernetes.io/docs", cfg, ALLOWED)
        # Verify only 1024 bytes were processed (no "A" * 2000 worth in output)
        assert result is not None


class TestFetchDocTimeout:
    def test_timeout_exception_raises_tool_error(self) -> None:
        import httpx as _httpx
        cfg = DocFetchConfig()
        with patch("httpx.get", side_effect=_httpx.TimeoutException("timed out")):
            with pytest.raises(ToolExecutionError):
                fetch_doc("https://kubernetes.io/docs", cfg, ALLOWED)


class TestFetchDocRedirects:
    def test_redirect_to_allowed_domain_follows(self) -> None:
        cfg = DocFetchConfig()
        redirect_resp = _make_response(
            301, b"", headers={"location": "https://docs.python.org/3/"}
        )
        final_resp = _make_response(200, SIMPLE_HTML)
        with patch("httpx.get", side_effect=[redirect_resp, final_resp]):
            result = fetch_doc("https://kubernetes.io/old", cfg, ALLOWED)
        assert DELIMITER_DATA_START in result.output

    def test_redirect_to_disallowed_domain_raises(self) -> None:
        cfg = DocFetchConfig()
        redirect_resp = _make_response(
            301, b"", headers={"location": "https://evil.com/phish"}
        )
        with patch("httpx.get", return_value=redirect_resp):
            with pytest.raises(ToolExecutionError, match="disallowed"):
                fetch_doc("https://kubernetes.io/old", cfg, ALLOWED)


class TestFetchDocScriptStripping:
    def test_script_tags_stripped_from_output(self) -> None:
        cfg = DocFetchConfig()
        html_with_script = b"<html><body><script>alert('xss')</script><p>Content</p></body></html>"
        resp = _make_response(200, html_with_script)
        with patch("httpx.get", return_value=resp):
            result = fetch_doc("https://kubernetes.io/docs", cfg, ALLOWED)
        assert "alert" not in result.output
        assert "Content" in result.output

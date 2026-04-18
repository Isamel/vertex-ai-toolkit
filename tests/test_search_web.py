"""Unit tests for search_web tool (T-04)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vaig.core.config import WebSearchConfig
from vaig.core.exceptions import ToolExecutionError
from vaig.core.prompt_defense import DELIMITER_DATA_START
from vaig.tools.knowledge.search_web import search_web


class TestSearchWebGuards:
    def test_empty_api_key_raises(self) -> None:
        cfg = WebSearchConfig(api_key="")
        with pytest.raises(ToolExecutionError, match="api_key"):
            search_web("query", cfg)


class TestSearchWebHTTP:
    def _make_response(self, status_code: int, json_data: dict) -> MagicMock:  # type: ignore[type-arg]
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = json_data
        mock_resp.text = str(json_data)
        return mock_resp

    def test_http_429_raises_with_status(self) -> None:
        cfg = WebSearchConfig(api_key="sk-key")
        error_resp = self._make_response(429, {"error": "rate limited"})
        with patch("httpx.post", return_value=error_resp):
            with pytest.raises(ToolExecutionError, match="429"):
                search_web("query", cfg)

    def test_successful_response_has_delimiter(self) -> None:
        cfg = WebSearchConfig(api_key="sk-key")
        data = {
            "results": [
                {"title": "Kubernetes HPA", "url": "https://kubernetes.io/hpa", "content": "HPA docs"},
            ]
        }
        ok_resp = self._make_response(200, data)
        with patch("httpx.post", return_value=ok_resp):
            result = search_web("kubernetes HPA tuning", cfg)
        assert DELIMITER_DATA_START in result.output

    def test_successful_response_contains_result_info(self) -> None:
        cfg = WebSearchConfig(api_key="sk-key")
        data = {
            "results": [
                {"title": "My Title", "url": "https://kubernetes.io/hpa", "content": "some snippet"},
            ]
        }
        ok_resp = self._make_response(200, data)
        with patch("httpx.post", return_value=ok_resp):
            result = search_web("query", cfg)
        assert "My Title" in result.output
        assert "kubernetes.io" in result.output
        assert "some snippet" in result.output

    def test_max_results_override_passed_to_request(self) -> None:
        cfg = WebSearchConfig(api_key="sk-key")
        data: dict = {"results": []}
        ok_resp = self._make_response(200, data)
        with patch("httpx.post", return_value=ok_resp) as mock_post:
            search_web("query", cfg, max_results=3)
        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs.args[1] if len(call_kwargs.args) > 1 else call_kwargs.kwargs["json"]
        assert body["max_results"] == 3

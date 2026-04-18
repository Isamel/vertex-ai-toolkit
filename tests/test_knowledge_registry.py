"""Integration tests for create_knowledge_tools factory gating (T-08)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vaig.core.config import (
    ExportConfig,
    KnowledgeConfig,
    Settings,
    WebSearchConfig,
)
from vaig.tools.knowledge._registry import create_knowledge_tools


def _settings_disabled() -> Settings:
    return Settings.model_construct(
        knowledge=KnowledgeConfig(enabled=False),
        export=ExportConfig(),
    )


def _settings_web_only() -> Settings:
    """knowledge enabled via api_key, no RAG."""
    return Settings.model_construct(
        knowledge=KnowledgeConfig(web_search=WebSearchConfig(api_key="sk-key")),
        export=ExportConfig(rag_enabled=False),
    )


def _settings_no_key() -> Settings:
    """knowledge enabled manually, no Tavily key, RAG on."""
    return Settings.model_construct(
        knowledge=KnowledgeConfig(enabled=True, web_search=WebSearchConfig(api_key="")),
        export=ExportConfig(rag_enabled=True),
    )


def _settings_all_enabled() -> Settings:
    return Settings.model_construct(
        knowledge=KnowledgeConfig(web_search=WebSearchConfig(api_key="sk-key")),
        export=ExportConfig(rag_enabled=True),
    )


class TestCreateKnowledgeToolsGating:
    def test_disabled_returns_empty(self) -> None:
        tools = create_knowledge_tools(_settings_disabled())
        assert tools == []

    def test_web_only_returns_two_tools(self) -> None:
        tools = create_knowledge_tools(_settings_web_only())
        names = [t.name for t in tools]
        assert "fetch_doc" in names
        assert "search_web" in names
        assert "search_rag_knowledge" not in names

    def test_no_key_returns_without_search_web(self) -> None:
        with patch("vaig.core.rag.RAGKnowledgeBase.__init__", return_value=None):
            tools = create_knowledge_tools(_settings_no_key())
        names = [t.name for t in tools]
        assert "search_web" not in names
        assert "fetch_doc" in names

    def test_all_enabled_returns_three_tools(self) -> None:
        with patch("vaig.core.rag.RAGKnowledgeBase.__init__", return_value=None):
            tools = create_knowledge_tools(_settings_all_enabled())
        names = [t.name for t in tools]
        assert "search_web" in names
        assert "fetch_doc" in names
        assert "search_rag_knowledge" in names
        assert len(tools) == 3

    def test_all_tools_have_knowledge_category(self) -> None:
        tools = create_knowledge_tools(_settings_web_only())
        for tool in tools:
            assert "knowledge" in tool.categories

    def test_search_web_and_fetch_doc_not_cacheable(self) -> None:
        tools = create_knowledge_tools(_settings_web_only())
        non_cacheable = {t.name for t in tools if not t.cacheable}
        assert "fetch_doc" in non_cacheable
        assert "search_web" in non_cacheable

    def test_rag_tool_cacheable(self) -> None:
        with patch("vaig.core.rag.RAGKnowledgeBase.__init__", return_value=None):
            tools = create_knowledge_tools(_settings_all_enabled())
        rag_tools = [t for t in tools if t.name == "search_rag_knowledge"]
        assert len(rag_tools) == 1
        assert rag_tools[0].cacheable is True
        assert rag_tools[0].cache_ttl_seconds == 300


class TestCodingDomainsFlag:
    def test_include_coding_domains_true_adds_pypi(self) -> None:
        tools = create_knowledge_tools(_settings_web_only(), include_coding_domains=True)
        fetch_tool = next(t for t in tools if t.name == "fetch_doc")
        # Invoke with pypi.org — should NOT raise domain error
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"<p>pypi content</p>"
        mock_resp.headers = {}
        with patch("httpx.get", return_value=mock_resp):
            result = fetch_tool.execute(url="https://pypi.org/project/httpx/")
        # Should succeed (not raise)
        assert result is not None

    def test_include_coding_domains_false_blocks_pypi(self) -> None:
        from vaig.core.exceptions import ToolExecutionError
        tools = create_knowledge_tools(_settings_web_only(), include_coding_domains=False)
        fetch_tool = next(t for t in tools if t.name == "fetch_doc")
        with pytest.raises(ToolExecutionError, match="domain not allowed"):
            fetch_tool.execute(url="https://pypi.org/project/httpx/")

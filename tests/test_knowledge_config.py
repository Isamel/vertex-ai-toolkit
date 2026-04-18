"""Unit tests for KnowledgeConfig and sub-config models (T-03)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vaig.core.config import (
    DocFetchConfig,
    KnowledgeConfig,
    RagKnowledgeConfig,
    Settings,
    WebSearchConfig,
)


class TestKnowledgeConfigDefaults:
    def test_default_disabled(self) -> None:
        cfg = KnowledgeConfig()
        assert cfg.enabled is False

    def test_auto_enable_with_api_key(self) -> None:
        cfg = KnowledgeConfig(web_search=WebSearchConfig(api_key="sk-abc"))
        assert cfg.enabled is True

    def test_stays_disabled_without_api_key(self) -> None:
        cfg = KnowledgeConfig(enabled=False, web_search=WebSearchConfig(api_key=""))
        assert cfg.enabled is False

    def test_settings_has_knowledge_field(self) -> None:
        s = Settings()
        assert isinstance(s.knowledge, KnowledgeConfig)


class TestDocFetchConfigConstraints:
    def test_max_bytes_below_min_raises(self) -> None:
        with pytest.raises(ValidationError):
            DocFetchConfig(max_bytes=500)

    def test_max_bytes_above_max_raises(self) -> None:
        with pytest.raises(ValidationError):
            DocFetchConfig(max_bytes=6_000_000)

    def test_valid_max_bytes(self) -> None:
        cfg = DocFetchConfig(max_bytes=1024)
        assert cfg.max_bytes == 1024


class TestWebSearchConfigConstraints:
    def test_max_results_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            WebSearchConfig(max_results=0)

    def test_max_results_over_limit_raises(self) -> None:
        with pytest.raises(ValidationError):
            WebSearchConfig(max_results=21)

    def test_valid_max_results(self) -> None:
        cfg = WebSearchConfig(max_results=10)
        assert cfg.max_results == 10


class TestRagKnowledgeConfigConstraints:
    def test_top_k_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            RagKnowledgeConfig(top_k=0)

    def test_valid_top_k(self) -> None:
        cfg = RagKnowledgeConfig(top_k=5)
        assert cfg.top_k == 5

"""Integration tests for CodingSkillOrchestrator knowledge tool injection (T-10)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from vaig.core.config import (
    CodingConfig,
    ExportConfig,
    KnowledgeConfig,
    Settings,
    WebSearchConfig,
)
from vaig.tools.categories import KNOWLEDGE


def _make_client() -> MagicMock:
    client = MagicMock()
    client.current_model = "gemini-2.5-pro"
    return client


def _settings_with_knowledge(enabled: bool = True, api_key: str = "sk-test") -> Settings:
    return Settings.model_construct(
        knowledge=KnowledgeConfig(
            enabled=enabled,
            web_search=WebSearchConfig(api_key=api_key),
        ),
        export=ExportConfig(rag_enabled=False),
    )


def _make_orchestrator(settings: Settings) -> object:
    from vaig.agents.coding_pipeline import CodingSkillOrchestrator

    coding_cfg = CodingConfig(workspace_root=str(Path("/tmp")))
    return CodingSkillOrchestrator(_make_client(), coding_cfg, settings=settings)


class TestCodingPipelineKnowledgeInjection:
    def test_knowledge_enabled_registry_has_knowledge_tools(self) -> None:
        orch = _make_orchestrator(_settings_with_knowledge(enabled=True))
        registry = orch._registry  # type: ignore[attr-defined]
        tools_with_knowledge = [
            t for t in registry.list_tools() if KNOWLEDGE in t.categories
        ]
        assert len(tools_with_knowledge) >= 1  # at least fetch_doc

    def test_knowledge_disabled_registry_has_no_knowledge_tools(self) -> None:
        settings = _settings_with_knowledge(enabled=False, api_key="")
        # Force disabled
        settings = Settings.model_construct(
            knowledge=KnowledgeConfig(enabled=False),
            export=ExportConfig(rag_enabled=False),
        )
        orch = _make_orchestrator(settings)
        registry = orch._registry  # type: ignore[attr-defined]
        tools_with_knowledge = [
            t for t in registry.list_tools() if KNOWLEDGE in t.categories
        ]
        assert tools_with_knowledge == []

    def test_knowledge_tools_include_coding_domains(self) -> None:
        """fetch_doc should allow pypi.org (coding domains) in coding pipeline."""

        orch = _make_orchestrator(_settings_with_knowledge(enabled=True))
        registry = orch._registry  # type: ignore[attr-defined]
        fetch_tool = registry.get("fetch_doc")
        assert fetch_tool is not None

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"<p>pypi content</p>"
        mock_resp.headers = {}
        with patch("httpx.get", return_value=mock_resp):
            result = fetch_tool.execute(url="https://pypi.org/project/httpx/")
        # Should not raise ToolExecutionError — pypi.org is allowed via coding domains
        assert result is not None

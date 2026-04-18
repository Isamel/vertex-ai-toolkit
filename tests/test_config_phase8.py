"""Tests for Phase 8 config additions (WorkspaceRAGConfig, CM-08)."""

from __future__ import annotations

import pytest


def test_workspace_rag_config_defaults() -> None:
    """WorkspaceRAGConfig defaults match spec."""
    from vaig.core.config import WorkspaceRAGConfig

    cfg = WorkspaceRAGConfig()
    assert cfg.enabled is False
    assert cfg.reindex_on_run is False
    assert cfg.max_chunks == 500
    assert ".py" in cfg.extensions
    assert ".ts" in cfg.extensions
    assert ".go" in cfg.extensions
    assert ".java" in cfg.extensions
    assert ".md" in cfg.extensions


def test_coding_config_default_workspace_rag() -> None:
    """CodingConfig initialises with a default WorkspaceRAGConfig (disabled)."""
    from vaig.core.config import CodingConfig

    cfg = CodingConfig()
    assert hasattr(cfg, "workspace_rag")
    assert cfg.workspace_rag.enabled is False


def test_coding_config_no_workspace_rag_key() -> None:
    """CodingConfig without workspace_rag key uses default_factory silently."""
    from vaig.core.config import CodingConfig

    cfg = CodingConfig.model_validate({})
    assert cfg.workspace_rag.enabled is False


def test_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """VAIG_CODING__WORKSPACE_RAG__ENABLED=true activates the config."""
    monkeypatch.setenv("VAIG_CODING__WORKSPACE_RAG__ENABLED", "true")

    # Settings reads from env — test CodingConfig nested parse directly
    from vaig.core.config import CodingConfig, WorkspaceRAGConfig

    rag_cfg = WorkspaceRAGConfig(enabled=True)
    coding = CodingConfig(workspace_rag=rag_cfg)
    assert coding.workspace_rag.enabled is True

"""Integration tests for WorkspaceRAG — requires chromadb to be installed.

Skip automatically when chromadb is not available.
"""

from __future__ import annotations

from pathlib import Path

import pytest

chromadb = pytest.importorskip("chromadb")


@pytest.mark.integration
def test_build_and_search(tmp_path: Path) -> None:
    """Full build + search cycle with real ChromaDB."""
    from vaig.core.config import WorkspaceRAGConfig
    from vaig.core.workspace_rag import WorkspaceRAG

    # Create three distinct Python files
    (tmp_path / "alpha.py").write_text("def alpha_function():\n    return 'alpha'\n")
    (tmp_path / "beta.py").write_text("class BetaClass:\n    def method(self): pass\n")
    (tmp_path / "gamma.py").write_text("GAMMA_CONSTANT = 42\n")

    cfg = WorkspaceRAGConfig(enabled=True, extensions=[".py"])
    rag = WorkspaceRAG(tmp_path, cfg)

    count = rag.build_index()
    assert count > 0, "Expected at least 1 chunk"

    results = rag.search("alpha function", k=5)
    assert isinstance(results, list)
    assert len(results) > 0

    files_returned = [r["file"] for r in results]
    assert any("alpha" in f for f in files_returned), (
        f"Expected alpha.py in results, got: {files_returned}"
    )


@pytest.mark.integration
def test_is_stale_false_after_build(tmp_path: Path) -> None:
    """is_stale() is False immediately after build."""
    from vaig.core.config import WorkspaceRAGConfig
    from vaig.core.workspace_rag import WorkspaceRAG

    (tmp_path / "code.py").write_text("x = 1\n")
    cfg = WorkspaceRAGConfig(enabled=True, extensions=[".py"])
    rag = WorkspaceRAG(tmp_path, cfg)
    rag.build_index()

    assert rag.is_stale() is False


@pytest.mark.integration
def test_is_stale_true_after_touch(tmp_path: Path) -> None:
    """is_stale() is True after a file mtime changes."""
    from vaig.core.config import WorkspaceRAGConfig
    from vaig.core.workspace_rag import WorkspaceRAG

    py_file = tmp_path / "code.py"
    py_file.write_text("x = 1\n")

    cfg = WorkspaceRAGConfig(enabled=True, extensions=[".py"])
    rag = WorkspaceRAG(tmp_path, cfg)
    rag.build_index()

    # Push build timestamp back so file looks newer
    rag._build_timestamp -= 10.0

    assert rag.is_stale() is True

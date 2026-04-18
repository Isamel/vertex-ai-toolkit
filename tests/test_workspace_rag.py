"""Unit tests for WorkspaceRAG (CM-08)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_config(
    enabled: bool = True,
    reindex_on_run: bool = False,
    max_chunks: int = 500,
    extensions: list[str] | None = None,
) -> MagicMock:
    cfg = MagicMock()
    cfg.enabled = enabled
    cfg.reindex_on_run = reindex_on_run
    cfg.max_chunks = max_chunks
    cfg.extensions = extensions or [".py", ".ts", ".go", ".java", ".md"]
    return cfg


def _make_chromadb_mock() -> MagicMock:
    """Return a mock chromadb module with a PersistentClient."""
    chroma = MagicMock()
    collection = MagicMock()
    collection.count.return_value = 0
    collection.query.return_value = {
        "documents": [["def foo(): pass"]],
        "metadatas": [[{"file": "foo.py", "chunk_index": 0}]],
        "distances": [[0.1]],
    }
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    chroma.PersistentClient.return_value = client
    return chroma


# ── chromadb missing ──────────────────────────────────────────────────────────


def test_chromadb_missing_raises_import_error(tmp_path: Path) -> None:
    """When chromadb is absent, instantiating WorkspaceRAG raises ImportError."""
    with patch.dict(sys.modules, {"chromadb": None}):
        # Remove cached module if any
        sys.modules.pop("vaig.core.workspace_rag", None)
        from vaig.core.workspace_rag import _require_chromadb

        with pytest.raises(ImportError, match="pip install chromadb"):
            _require_chromadb()


# ── build_index ───────────────────────────────────────────────────────────────


def test_build_index_adds_chunks(tmp_path: Path) -> None:
    """build_index() chunks files and calls collection.add()."""
    # Create two Python files
    (tmp_path / "a.py").write_text("x = 1\n" * 10)
    (tmp_path / "b.py").write_text("y = 2\n" * 10)

    chroma_mock = _make_chromadb_mock()
    collection = chroma_mock.PersistentClient.return_value.get_or_create_collection.return_value

    with patch("vaig.core.workspace_rag._require_chromadb", return_value=chroma_mock):
        from vaig.core.workspace_rag import WorkspaceRAG

        rag = WorkspaceRAG(tmp_path, _make_config())
        count = rag.build_index()

    assert count >= 2  # at least 1 chunk per file
    collection.add.assert_called_once()
    call_kwargs = collection.add.call_args[1]
    assert "ids" in call_kwargs
    assert "documents" in call_kwargs
    assert "metadatas" in call_kwargs


def test_build_index_respects_max_chunks(tmp_path: Path) -> None:
    """build_index() does not exceed max_chunks."""
    # Create a file with many lines
    (tmp_path / "big.py").write_text("line\n" * 1000)

    chroma_mock = _make_chromadb_mock()

    with patch("vaig.core.workspace_rag._require_chromadb", return_value=chroma_mock):
        from vaig.core.workspace_rag import WorkspaceRAG

        rag = WorkspaceRAG(tmp_path, _make_config(max_chunks=2))
        count = rag.build_index()

    assert count <= 2


def test_build_index_skips_large_files(tmp_path: Path) -> None:
    """Files larger than 1 MB are skipped."""
    large = tmp_path / "large.py"
    large.write_bytes(b"x" * 1_100_000)

    chroma_mock = _make_chromadb_mock()
    collection = chroma_mock.PersistentClient.return_value.get_or_create_collection.return_value

    with patch("vaig.core.workspace_rag._require_chromadb", return_value=chroma_mock):
        from vaig.core.workspace_rag import WorkspaceRAG

        rag = WorkspaceRAG(tmp_path, _make_config())
        count = rag.build_index()

    assert count == 0
    collection.add.assert_not_called()


def test_build_index_respects_extensions_filter(tmp_path: Path) -> None:
    """Only files matching configured extensions are indexed."""
    (tmp_path / "code.py").write_text("print('hi')\n")
    (tmp_path / "ignore.txt").write_text("plain text\n")

    chroma_mock = _make_chromadb_mock()

    with patch("vaig.core.workspace_rag._require_chromadb", return_value=chroma_mock):
        from vaig.core.workspace_rag import WorkspaceRAG

        rag = WorkspaceRAG(tmp_path, _make_config(extensions=[".py"]))
        count = rag.build_index()

    # Only .py file should contribute
    assert count >= 1
    collection = chroma_mock.PersistentClient.return_value.get_or_create_collection.return_value
    if collection.add.called:
        ids = collection.add.call_args[1]["ids"]
        for chunk_id in ids:
            assert chunk_id.startswith("code.py"), f"Unexpected id: {chunk_id}"


# ── search ────────────────────────────────────────────────────────────────────


def test_search_returns_expected_format(tmp_path: Path) -> None:
    """search() returns list of dicts with file, chunk, score keys."""
    chroma_mock = _make_chromadb_mock()
    collection = chroma_mock.PersistentClient.return_value.get_or_create_collection.return_value
    collection.count.return_value = 5

    with patch("vaig.core.workspace_rag._require_chromadb", return_value=chroma_mock):
        from vaig.core.workspace_rag import WorkspaceRAG

        rag = WorkspaceRAG(tmp_path, _make_config())
        results = rag.search("foo function", k=3)

    assert isinstance(results, list)
    assert len(results) > 0
    for r in results:
        assert "file" in r
        assert "chunk" in r
        assert "score" in r
        assert isinstance(r["score"], float)


def test_search_triggers_build_when_empty(tmp_path: Path) -> None:
    """search() calls build_index when collection is empty."""
    (tmp_path / "a.py").write_text("def hello(): pass\n")

    chroma_mock = _make_chromadb_mock()
    collection = chroma_mock.PersistentClient.return_value.get_or_create_collection.return_value
    collection.count.return_value = 0

    with patch("vaig.core.workspace_rag._require_chromadb", return_value=chroma_mock):
        from vaig.core.workspace_rag import WorkspaceRAG

        rag = WorkspaceRAG(tmp_path, _make_config())
        rag.search("hello")

    collection.add.assert_called_once()


# ── is_stale ──────────────────────────────────────────────────────────────────


def test_is_stale_true_before_build(tmp_path: Path) -> None:
    """is_stale() returns True when no build has been done."""
    (tmp_path / "a.py").write_text("x = 1\n")

    chroma_mock = _make_chromadb_mock()

    with patch("vaig.core.workspace_rag._require_chromadb", return_value=chroma_mock):
        from vaig.core.workspace_rag import WorkspaceRAG

        rag = WorkspaceRAG(tmp_path, _make_config())
        assert rag.is_stale() is True


def test_is_stale_false_after_build(tmp_path: Path) -> None:
    """is_stale() returns False immediately after build_index."""
    (tmp_path / "a.py").write_text("x = 1\n")

    chroma_mock = _make_chromadb_mock()

    with patch("vaig.core.workspace_rag._require_chromadb", return_value=chroma_mock):
        from vaig.core.workspace_rag import WorkspaceRAG

        rag = WorkspaceRAG(tmp_path, _make_config())
        rag.build_index()
        assert rag.is_stale() is False


def test_is_stale_true_after_file_modified(tmp_path: Path) -> None:
    """is_stale() returns True when a file is modified after build."""

    py_file = tmp_path / "a.py"
    py_file.write_text("x = 1\n")

    chroma_mock = _make_chromadb_mock()

    with patch("vaig.core.workspace_rag._require_chromadb", return_value=chroma_mock):
        from vaig.core.workspace_rag import WorkspaceRAG

        rag = WorkspaceRAG(tmp_path, _make_config())
        rag.build_index()

        # Simulate file modification by pushing _build_timestamp into the past
        rag._build_timestamp -= 10.0

        assert rag.is_stale() is True

"""Workspace RAG — local vector-search over workspace files (CM-08).

ChromaDB is an optional dependency. If not installed, attempting to use this
module raises a clear ImportError with install instructions.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CHROMADB_MISSING_MSG = (
    "chromadb is required for Workspace RAG. Install it with: pip install chromadb"
)

_CHUNK_LINES = 200
_OVERLAP_LINES = 20
_MAX_FILE_BYTES = 1_000_000  # 1 MB


def _require_chromadb() -> Any:
    """Import and return chromadb, raising ImportError with help if absent."""
    try:
        import chromadb as _chromadb  # noqa: WPS433

        return _chromadb
    except ImportError as exc:
        raise ImportError(_CHROMADB_MISSING_MSG) from exc


class WorkspaceRAG:
    """Local vector-search index over workspace files using ChromaDB."""

    def __init__(self, workspace: Path, config: Any) -> None:
        """Initialise the WorkspaceRAG instance.

        Args:
            workspace: Root directory of the workspace to index.
            config: :class:`~vaig.core.config.WorkspaceRAGConfig` instance.
        """
        chromadb = _require_chromadb()
        self._workspace = workspace
        self._config = config
        self._build_timestamp: float = 0.0

        index_dir = workspace / ".vaig" / "workspace-index"
        index_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(index_dir))
        self._collection = self._client.get_or_create_collection(name="workspace")

    # ── Public API ────────────────────────────────────────────

    def build_index(self) -> int:
        """Walk workspace, chunk files, and store in ChromaDB.

        Returns:
            Total number of chunks indexed.
        """
        files = self._collect_files()

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for path in files:
            rel = path.relative_to(self._workspace)
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                logger.debug("workspace_rag: skip unreadable file %s", path)
                continue

            for chunk_idx, chunk in enumerate(self._chunk_text(content)):
                if len(ids) >= self._config.max_chunks:
                    break
                ids.append(f"{rel}::{chunk_idx}")
                documents.append(chunk)
                metadatas.append({"file": str(rel), "chunk_index": chunk_idx})

            if len(ids) >= self._config.max_chunks:
                break

        if ids:
            # Clear previous data and re-add
            self._collection.delete(where={"chunk_index": {"$gte": 0}})
            self._collection.add(ids=ids, documents=documents, metadatas=metadatas)

        self._build_timestamp = time.time()
        logger.debug("workspace_rag: indexed %d chunks", len(ids))
        return len(ids)

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Search the index for chunks relevant to *query*.

        If the index is empty or stale (and ``reindex_on_run`` is set),
        :meth:`build_index` is called automatically.

        Args:
            query: Natural-language or code search query.
            k: Number of results to return.

        Returns:
            List of ``{"file": str, "chunk": str, "score": float}`` dicts,
            sorted from most to least relevant.
        """
        count = self._collection.count()
        if count == 0 or (self._config.reindex_on_run and self.is_stale()):
            self.build_index()

        results = self._collection.query(query_texts=[query], n_results=min(k, max(self._collection.count(), 1)))

        output: list[dict[str, Any]] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances, strict=False):
            # ChromaDB returns L2 distance; convert to similarity score (0-1)
            score = 1.0 / (1.0 + dist) if dist is not None else 0.0
            output.append({"file": meta.get("file", ""), "chunk": doc, "score": score})

        return output

    def is_stale(self) -> bool:
        """Return True if workspace files have been modified since last index build."""
        if self._build_timestamp == 0.0:
            return True

        files = self._collect_files()
        if not files:
            return False

        max_mtime = max(f.stat().st_mtime for f in files)
        return max_mtime > self._build_timestamp

    # ── Private helpers ───────────────────────────────────────

    def _collect_files(self) -> list[Path]:
        """Return all workspace files matching configured extensions."""
        exts = set(self._config.extensions)
        result: list[Path] = []
        for path in self._workspace.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in exts:
                continue
            if path.stat().st_size > _MAX_FILE_BYTES:
                logger.debug("workspace_rag: skip large file %s", path)
                continue
            result.append(path)
        return result

    @staticmethod
    def _chunk_text(content: str) -> list[str]:
        """Split *content* into overlapping line-based chunks."""
        lines = content.splitlines(keepends=True)
        if not lines:
            return [""]

        chunks: list[str] = []
        start = 0
        while start < len(lines):
            end = min(start + _CHUNK_LINES, len(lines))
            chunks.append("".join(lines[start:end]))
            if end == len(lines):
                break
            start += _CHUNK_LINES - _OVERLAP_LINES

        return chunks or [""]

"""Vertex AI RAG Engine integration — knowledge base wrapper.

Provides a clean interface for creating, ingesting into, querying,
and managing Vertex AI RAG corpora.  All ``vertexai`` imports are
lazy (inside methods) so the module works without the optional
dependency installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vaig.core.config import ExportConfig

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────


@dataclass(slots=True)
class RetrievedChunk:
    """A single chunk returned by a RAG retrieval query."""

    text: str
    score: float = 0.0
    source: str = ""


@dataclass(slots=True)
class RetrievalResult:
    """Result of a RAG retrieval operation."""

    chunks: list[RetrievedChunk] = field(default_factory=list)
    query: str = ""

    @property
    def has_results(self) -> bool:
        """Return True if any chunks were retrieved."""
        return len(self.chunks) > 0

    def format_context(self) -> str:
        """Format retrieved chunks as a context string for injection."""
        if not self.chunks:
            return ""
        parts: list[str] = []
        for i, chunk in enumerate(self.chunks, 1):
            source_info = f" (source: {chunk.source})" if chunk.source else ""
            parts.append(f"[{i}]{source_info} {chunk.text}")
        return "\n\n".join(parts)


# ── Lazy import helpers ───────────────────────────────────────


def _import_vertexai_rag() -> Any:
    """Lazily import vertexai.preview.rag, raising ImportError if missing."""
    try:
        from vertexai.preview import rag

        return rag
    except ImportError:
        raise ImportError(
            "RAG Engine features require the vertexai SDK. "
            "Install with: pip install google-cloud-aiplatform[rag]"
        ) from None


def _import_vertexai() -> Any:
    """Lazily import the vertexai module."""
    try:
        import vertexai

        return vertexai
    except ImportError:
        raise ImportError(
            "RAG Engine features require the vertexai SDK. "
            "Install with: pip install google-cloud-aiplatform[rag]"
        ) from None


# ── RAG Knowledge Base ────────────────────────────────────────


class RAGKnowledgeBase:
    """Wrapper around the Vertex AI RAG Engine API.

    All Vertex AI SDK imports are lazy — they happen inside methods,
    not at module level.  This allows the module to be imported even
    when ``vertexai`` is not installed.

    Parameters
    ----------
    config:
        The :class:`ExportConfig` instance carrying RAG-related settings.
    project:
        GCP project ID.  Falls back to ``config.gcp_project_id``.
    location:
        GCP region.  Defaults to ``"us-central1"``.
    """

    def __init__(
        self,
        config: ExportConfig,
        project: str = "",
        location: str = "us-central1",
    ) -> None:
        self._config = config
        self._project = project or config.gcp_project_id
        self._location = location
        self._initialized = False
        self._org_corpus_cache: dict[str, str] = {}

    # ── Properties ────────────────────────────────────────────

    @property
    def is_configured(self) -> bool:
        """Return True if RAG is enabled and a corpus name is set."""
        return bool(self._config.rag_enabled and self._config.rag_corpus_name)

    @property
    def corpus_name(self) -> str:
        """Return the configured RAG corpus name/ID."""
        return self._config.rag_corpus_name

    # ── SDK initialisation ────────────────────────────────────

    def _ensure_initialized(self) -> None:
        """Initialize the Vertex AI SDK if not already done."""
        if self._initialized:
            return
        vertexai = _import_vertexai()
        vertexai.init(project=self._project, location=self._location)
        self._initialized = True
        logger.debug(
            "Vertex AI SDK initialized: project=%s location=%s",
            self._project,
            self._location,
        )

    # ── Corpus management ─────────────────────────────────────

    def create_corpus(self, display_name: str, description: str = "") -> str:
        """Create a new RAG corpus and return its resource name.

        Parameters
        ----------
        display_name:
            Human-readable name for the corpus.
        description:
            Optional description.

        Returns
        -------
        str
            The full resource name of the created corpus.
        """
        self._ensure_initialized()
        rag = _import_vertexai_rag()
        corpus = rag.create_corpus(
            display_name=display_name,
            description=description,
        )
        logger.info("Created RAG corpus: %s", corpus.name)
        return str(corpus.name)

    def delete_corpus(self, corpus_name: str = "") -> bool:
        """Delete a RAG corpus.

        Parameters
        ----------
        corpus_name:
            The resource name of the corpus.  Defaults to the configured
            corpus name.

        Returns
        -------
        bool
            True if deletion succeeded.
        """
        target = corpus_name or self.corpus_name
        if not target:
            logger.warning("delete_corpus called but no corpus name configured.")
            return False

        self._ensure_initialized()
        rag = _import_vertexai_rag()
        try:
            rag.delete_corpus(name=target)
            logger.info("Deleted RAG corpus: %s", target)
            return True
        except Exception:
            logger.exception("Failed to delete RAG corpus: %s", target)
            return False

    def list_corpora(self) -> list[dict[str, Any]]:
        """List all RAG corpora in the current project.

        Returns
        -------
        list[dict[str, Any]]
            A list of dicts with ``name`` and ``display_name`` keys.

        Raises
        ------
        RuntimeError
            When the underlying API call fails.
        """
        self._ensure_initialized()
        rag = _import_vertexai_rag()
        try:
            corpora = rag.list_corpora()
            return [
                {"name": str(c.name), "display_name": str(c.display_name)}
                for c in corpora
            ]
        except Exception as exc:
            logger.exception("Failed to list RAG corpora.")
            raise RuntimeError("Failed to list RAG corpora") from exc

    def resolve_corpus(self, org_id: str) -> str:
        """Find or create the RAG corpus for an organization.

        Parameters
        ----------
        org_id:
            Organization identifier.  When empty, returns the global
            ``vertex_rag_corpus_id`` from config.

        Returns
        -------
        str
            The full resource name of the resolved corpus.
        """
        if not org_id:
            return self._config.vertex_rag_corpus_id

        if org_id in self._org_corpus_cache:
            return self._org_corpus_cache[org_id]

        display_name = f"vaig-{org_id}"
        try:
            corpora = self.list_corpora()
            for corpus in corpora:
                if corpus.get("display_name") == display_name:
                    resolved: str = str(corpus["name"])
                    self._org_corpus_cache[org_id] = resolved
                    logger.info("Resolved org corpus %s → %s", display_name, resolved)
                    return resolved

            # Corpus does not exist — create it.
            name = self.create_corpus(
                display_name=display_name,
                description=f"Per-org RAG corpus for organization '{org_id}'",
            )
            self._org_corpus_cache[org_id] = name
            return name
        except (OSError, ValueError, RuntimeError):
            logger.exception(
                "Failed to resolve org corpus for '%s' — falling back to global",
                org_id,
            )
            return self._config.vertex_rag_corpus_id

    # ── Ingestion ─────────────────────────────────────────────

    def ingest_reports(self, gcs_paths: list[str]) -> bool:
        """Ingest GCS files into the configured RAG corpus.

        Parameters
        ----------
        gcs_paths:
            List of ``gs://`` URIs to import.

        Returns
        -------
        bool
            True if ingestion succeeded.
        """
        if not self.is_configured:
            logger.debug("RAG not configured — skipping ingest.")
            return False

        if not gcs_paths:
            logger.debug("No GCS paths provided — skipping ingest.")
            return False

        self._ensure_initialized()
        rag = _import_vertexai_rag()
        try:
            rag.import_files(
                corpus_name=self.corpus_name,
                paths=gcs_paths,
                chunk_size=self._config.rag_chunk_size,
                chunk_overlap=self._config.rag_chunk_overlap,
            )
            logger.info(
                "Ingested %d file(s) into corpus %s",
                len(gcs_paths),
                self.corpus_name,
            )
            return True
        except Exception:
            logger.exception("Failed to ingest reports into RAG corpus.")
            return False

    # ── Retrieval ─────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Retrieve relevant chunks from the RAG corpus.

        Parameters
        ----------
        query:
            The search query.
        top_k:
            Maximum number of chunks to return.

        Returns
        -------
        RetrievalResult
            Contains the retrieved chunks, or empty if RAG is not
            configured or the query fails.
        """
        if not self.is_configured:
            logger.debug("RAG not configured — returning empty retrieval.")
            return RetrievalResult(query=query)

        if not query.strip():
            logger.debug("Empty query — returning empty retrieval.")
            return RetrievalResult(query=query)

        self._ensure_initialized()
        rag = _import_vertexai_rag()
        try:
            response = rag.retrieval_query(
                rag_resources=[
                    rag.RagResource(rag_corpus=self.corpus_name),
                ],
                text=query,
                similarity_top_k=top_k,
            )
            chunks = [
                RetrievedChunk(
                    text=str(ctx.text),
                    score=float(getattr(ctx, "distance", 0.0)),
                    source=str(getattr(ctx, "source_uri", "")),
                )
                for ctx in (response.contexts.contexts if response.contexts else [])
            ]
            logger.info(
                "RAG retrieval: %d chunk(s) for query of length %d",
                len(chunks),
                len(query),
            )
            return RetrievalResult(chunks=chunks, query=query)
        except Exception:
            logger.exception(
                "RAG retrieval failed for query of length %d",
                len(query),
            )
            return RetrievalResult(query=query)

    def retrieve_with_fallback(
        self,
        query: str,
        org_id: str = "",
        top_k: int = 5,
        min_results: int = 3,
    ) -> RetrievalResult:
        """Query org corpus first, fall back to global when results are sparse.

        Parameters
        ----------
        query:
            The search query.
        org_id:
            Organization identifier.  When empty, delegates to
            :meth:`retrieve` on the global corpus.
        top_k:
            Maximum number of chunks to return per corpus query.
        min_results:
            Minimum chunk count from the org corpus before the global
            corpus is also queried.

        Returns
        -------
        RetrievalResult
            Merged and deduplicated results.
        """
        if not org_id:
            return self.retrieve(query, top_k=top_k)

        if not query.strip():
            logger.debug("Empty query — returning empty retrieval.")
            return RetrievalResult(query=query)

        org_corpus = self.resolve_corpus(org_id)
        if not org_corpus:
            return self.retrieve(query, top_k=top_k)

        # Query the org-specific corpus.
        self._ensure_initialized()
        rag = _import_vertexai_rag()
        org_chunks: list[RetrievedChunk] = []
        try:
            response = rag.retrieval_query(
                rag_resources=[rag.RagResource(rag_corpus=org_corpus)],
                text=query,
                similarity_top_k=top_k,
            )
            org_chunks = [
                RetrievedChunk(
                    text=str(ctx.text),
                    score=float(getattr(ctx, "distance", 0.0)),
                    source=str(getattr(ctx, "source_uri", "")),
                )
                for ctx in (response.contexts.contexts if response.contexts else [])
            ]
        except (OSError, ValueError, RuntimeError):
            logger.exception("Org corpus retrieval failed for org '%s'", org_id)

        if len(org_chunks) >= min_results:
            logger.info(
                "Org corpus '%s' returned %d chunks (>= min %d) — no global fallback",
                org_id,
                len(org_chunks),
                min_results,
            )
            return RetrievalResult(chunks=org_chunks, query=query)

        # Insufficient org results — also query the global corpus.
        global_result = self.retrieve(query, top_k=top_k)

        # Merge: org chunks first, then global, dedup by source URI.
        seen_sources: set[str] = set()
        merged: list[RetrievedChunk] = []
        for chunk in [*org_chunks, *global_result.chunks]:
            key = chunk.source
            if key and key in seen_sources:
                continue
            if key:
                seen_sources.add(key)
            merged.append(chunk)

        logger.info(
            "Org corpus '%s' returned %d chunks (< min %d) — merged with %d global → %d total",
            org_id,
            len(org_chunks),
            min_results,
            len(global_result.chunks),
            len(merged),
        )
        return RetrievalResult(chunks=merged, query=query)

    # ── Memory RAG helpers ────────────────────────────────────

    def retrieve_from_corpus(
        self,
        corpus_name: str,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks from an arbitrary *corpus_name* (not necessarily the default).

        Used by :class:`~vaig.core.memory.memory_rag.MemoryRAGIndex` to query
        the separate memory corpus.

        Parameters
        ----------
        corpus_name:
            The full Vertex AI RAG corpus resource name to query.
        query:
            Free-text search query.
        top_k:
            Maximum number of chunks to return.

        Returns
        -------
        list[RetrievedChunk]
            Retrieved chunks, or empty list on error.
        """
        if not corpus_name or not query.strip():
            return []
        self._ensure_initialized()
        rag = _import_vertexai_rag()
        try:
            response = rag.retrieval_query(
                rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
                text=query,
                similarity_top_k=top_k,
            )
            return [
                RetrievedChunk(
                    text=str(ctx.text),
                    score=float(getattr(ctx, "distance", 0.0)),
                    source=str(getattr(ctx, "source_uri", "")),
                )
                for ctx in (response.contexts.contexts if response.contexts else [])
            ]
        except Exception:  # noqa: BLE001
            logger.warning("retrieve_from_corpus failed for corpus '%s'", corpus_name)
            return []

    def ingest_narratives(self, corpus_name: str, narratives: list[str]) -> bool:
        """Ingest plain-text narratives into *corpus_name* via the Vertex AI RAG API.

        Uses ``rag.upload_file`` (direct text upload) when available in the
        installed SDK version; otherwise silently no-ops and returns ``False``.

        Parameters
        ----------
        corpus_name:
            Target Vertex AI RAG corpus resource name.
        narratives:
            List of plain-text strings to ingest.

        Returns
        -------
        bool
            ``True`` if ingestion succeeded (or was no-op due to empty input).
        """
        if not corpus_name or not narratives:
            return True
        self._ensure_initialized()
        rag = _import_vertexai_rag()
        try:
            for text in narratives:
                if not text.strip():
                    continue
                rag.upload_file(
                    corpus_name=corpus_name,
                    path=None,
                    display_name="memory-narrative",
                    description=text[:200],
                )
            return True
        except Exception:  # noqa: BLE001
            logger.warning(
                "ingest_narratives: upload to corpus '%s' failed — narratives not persisted",
                corpus_name,
            )
            return False

"""Repo knowledge indexing — on-demand RAG corpus per repository.

Provides:
- :class:`RepoChunk` — a chunk of repository content with relevance score.
- :class:`RepoKnowledgeResult` — result from searching repo knowledge base.
- :class:`RepoIndexManager` — manages per-repo RAG knowledge indices.
- :func:`search_repo_knowledge` — tool function to search repo knowledge.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from vaig.core.prompt_defense import wrap_untrusted_content
from vaig.tools.base import ToolResult
from vaig.tools.repo.batch import TreeTriageReport

if TYPE_CHECKING:
    from vaig.core.config import Settings
    from vaig.core.rag import RAGKnowledgeBase, RetrievedChunk

logger = logging.getLogger(__name__)

# Default index storage root
_DEFAULT_INDEX_ROOT = Path(".vaig") / "repo-index"


class RepoChunk(BaseModel):
    """A chunk of repository content with relevance score."""

    path: str
    content: str
    score: float


class RepoKnowledgeResult(BaseModel):
    """Result from searching repo knowledge base."""

    repo: str
    ref: str
    chunks: list[RepoChunk]
    index_built_at: datetime
    from_cache: bool

    __test__ = False  # Prevent pytest collection


class RepoIndexManager:
    """Manages per-repo RAG knowledge indices.

    Stores indices at ``.vaig/repo-index/{owner_repo}/{ref}/``.
    Builds the index on first call (index-on-demand) using the existing
    :class:`~vaig.core.rag.RAGKnowledgeBase`.  Subsequent calls use the
    cached version.  Cache is invalidated when *ref* changes or new commits
    are detected.

    Parameters
    ----------
    rag:
        A configured :class:`~vaig.core.rag.RAGKnowledgeBase` instance.
    index_root:
        Directory under which per-repo index data is stored.
        Defaults to ``.vaig/repo-index``.
    """

    __test__ = False  # Prevent pytest collection

    def __init__(
        self,
        rag: RAGKnowledgeBase,
        index_root: Path | None = None,
    ) -> None:
        self._rag = rag
        self._index_root = index_root or _DEFAULT_INDEX_ROOT
        # In-memory cache: corpus_name -> (built_at, ref, commit_sha)
        self._cache: dict[str, tuple[datetime, str, str]] = {}

    # ── Helpers ───────────────────────────────────────────────

    def _corpus_name(self, owner: str, repo: str, ref: str) -> str:
        """Return a stable corpus name for the given repo + ref."""
        owner_repo = f"{owner}_{repo}".lower().replace("/", "_")
        safe_ref = ref.replace("/", "_").replace(".", "_")
        return f"vaig-repo-{owner_repo}-{safe_ref}"

    def _index_path(self, owner: str, repo: str, ref: str) -> Path:
        """Return the filesystem path for the index directory."""
        owner_repo = f"{owner}/{repo}".lower()
        return self._index_root / owner_repo / ref

    def _is_cached(self, corpus_name: str, ref: str, latest_sha: str) -> bool:
        """Return True if a valid cache entry exists for this corpus."""
        if corpus_name not in self._cache:
            return False
        _built_at, cached_ref, cached_sha = self._cache[corpus_name]
        # Invalidate if ref changed
        if cached_ref != ref:
            return False
        # Invalidate if commit SHA changed (new commits detected)
        if latest_sha and cached_sha and cached_sha != latest_sha:
            return False
        return True

    # ── Public API ────────────────────────────────────────────

    def build_index(
        self,
        owner: str,
        repo: str,
        ref: str,
        triage_report: TreeTriageReport,
        content_fetcher: dict[str, str],
        latest_sha: str = "",
    ) -> tuple[str, bool]:
        """Build (or return cached) RAG index for the repo.

        Parameters
        ----------
        owner:
            Repository owner.
        repo:
            Repository name.
        ref:
            Branch, tag, or commit SHA.
        triage_report:
            :class:`~vaig.tools.repo.batch.TreeTriageReport` — only Tier 1
            entries are indexed.
        content_fetcher:
            Mapping of ``path -> content`` for all Tier 1 files.
        latest_sha:
            Latest commit SHA for cache invalidation.  Pass empty string if
            unknown.

        Returns
        -------
        tuple[str, bool]
            ``(corpus_name, from_cache)`` — corpus_name is the Vertex AI RAG
            corpus resource name, from_cache indicates whether the index was
            already available.
        """
        corpus_name = self._corpus_name(owner, repo, ref)

        if self._is_cached(corpus_name, ref, latest_sha):
            logger.debug("Cache hit for corpus '%s'", corpus_name)
            return corpus_name, True

        # Build index — Tier 1 files only
        tier1_entries = triage_report.tier_1
        narratives: list[str] = []

        try:
            from rich.progress import Progress, SpinnerColumn, TextColumn

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task(
                    f"[cyan]Building knowledge index for {owner}/{repo}@{ref}…",
                    total=len(tier1_entries),
                )
                for entry in tier1_entries:
                    raw = content_fetcher.get(entry.path, "")
                    if raw.strip():
                        wrapped = wrap_untrusted_content(raw)
                        narratives.append(f"# {entry.path}\n\n{wrapped}")
                    progress.advance(task)
        except ImportError:
            # rich not available — build without progress bar
            for entry in tier1_entries:
                raw = content_fetcher.get(entry.path, "")
                if raw.strip():
                    wrapped = wrap_untrusted_content(raw)
                    narratives.append(f"# {entry.path}\n\n{wrapped}")

        self._rag.ingest_narratives(corpus_name, narratives)

        built_at = datetime.now(tz=UTC)
        self._cache[corpus_name] = (built_at, ref, latest_sha)

        # Persist a marker on disk for recovery across process restarts
        index_path = self._index_path(owner, repo, ref)
        try:
            index_path.mkdir(parents=True, exist_ok=True)
            marker = index_path / "built_at.txt"
            marker.write_text(built_at.isoformat(), encoding="utf-8")
        except OSError:
            logger.debug("Could not write index marker at %s", index_path)

        logger.info(
            "Built repo knowledge index for %s/%s@%s (%d Tier-1 files)",
            owner,
            repo,
            ref,
            len(tier1_entries),
        )
        return corpus_name, False

    def search(
        self,
        owner: str,
        repo: str,
        ref: str,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Search the knowledge index for *query*.

        Returns an empty list when the index has not been built yet.

        Parameters
        ----------
        owner:
            Repository owner.
        repo:
            Repository name.
        ref:
            Branch, tag, or commit SHA.
        query:
            Free-text search query.
        top_k:
            Maximum number of chunks to return.
        """
        corpus_name = self._corpus_name(owner, repo, ref)
        return self._rag.retrieve_from_corpus(corpus_name, query, top_k)

    def invalidate(self, owner: str, repo: str, ref: str) -> None:
        """Explicitly invalidate the cache for the given repo + ref."""
        corpus_name = self._corpus_name(owner, repo, ref)
        self._cache.pop(corpus_name, None)
        logger.debug("Invalidated cache for corpus '%s'", corpus_name)

    def built_at(self, owner: str, repo: str, ref: str) -> datetime | None:
        """Return the build timestamp for the given repo + ref, or None."""
        corpus_name = self._corpus_name(owner, repo, ref)
        entry = self._cache.get(corpus_name)
        return entry[0] if entry else None


# ── Tool function ──────────────────────────────────────────────


async def search_repo_knowledge(
    *,
    settings: Settings,
    owner: str,
    repo: str,
    query: str,
    ref: str = "HEAD",
    top_k: int = 5,
    triage_report: TreeTriageReport | None = None,
    content_fetcher: dict[str, str] | None = None,
    index_manager: RepoIndexManager | None = None,
) -> ToolResult:
    """Search a repository's on-demand RAG knowledge index.

    Builds the index on first call; subsequent calls use the cached
    version.  Only Tier 1 files from *triage_report* are indexed.

    Parameters
    ----------
    settings:
        Application settings (used to configure the RAG knowledge base).
    owner:
        Repository owner.
    repo:
        Repository name.
    query:
        Natural-language search query.
    ref:
        Branch, tag, or commit SHA.  Defaults to ``"HEAD"``.
    top_k:
        Maximum number of result chunks to return (default: 5).
    triage_report:
        Optional pre-built :class:`~vaig.tools.repo.batch.TreeTriageReport`.
        When ``None``, an empty report is used (no files indexed).
    content_fetcher:
        Mapping of ``path -> file_content`` for files to index.
        When ``None``, no content is indexed.
    index_manager:
        Optional pre-constructed :class:`RepoIndexManager`.  When ``None``,
        one is created using ``settings.export`` for the RAG config.

    Returns
    -------
    ToolResult
        JSON-serialisable result with ranked chunks.
    """
    if not query.strip():
        return ToolResult(output="Query must not be empty.", error=True)

    # Build or reuse the index manager
    if index_manager is None:
        from vaig.core.rag import RAGKnowledgeBase

        rag = RAGKnowledgeBase(config=settings.export)
        index_manager = RepoIndexManager(rag=rag)

    # Build triage report if not supplied
    if triage_report is None:
        triage_report = TreeTriageReport(owner=owner, repo=repo, ref=ref)

    if content_fetcher is None:
        content_fetcher = {}

    try:
        corpus_name, from_cache = index_manager.build_index(
            owner=owner,
            repo=repo,
            ref=ref,
            triage_report=triage_report,
            content_fetcher=content_fetcher,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to build repo knowledge index for %s/%s@%s", owner, repo, ref)
        return ToolResult(output=f"Failed to build knowledge index: {exc}", error=True)

    try:
        raw_chunks = index_manager.search(
            owner=owner,
            repo=repo,
            ref=ref,
            query=query,
            top_k=top_k,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to search repo knowledge index for %s/%s@%s", owner, repo, ref)
        return ToolResult(output=f"Failed to search knowledge index: {exc}", error=True)

    built_at = index_manager.built_at(owner, repo, ref) or datetime.now(tz=UTC)

    result = RepoKnowledgeResult(
        repo=f"{owner}/{repo}",
        ref=ref,
        chunks=[
            RepoChunk(
                path=chunk.source,
                content=chunk.text,
                score=chunk.score,
            )
            for chunk in raw_chunks
        ],
        index_built_at=built_at,
        from_cache=from_cache,
    )

    if not result.chunks:
        return ToolResult(
            output=f"No results found for query: {query!r} in {owner}/{repo}@{ref}"
        )

    lines: list[str] = [
        f"Repo knowledge search: {owner}/{repo}@{ref}",
        f"Query: {query!r}",
        f"Index built at: {result.index_built_at.isoformat()} (from_cache={result.from_cache})",
        "",
    ]
    for i, chunk in enumerate(result.chunks, 1):
        lines.append(f"[{i}] {chunk.path} (score={chunk.score:.4f})")
        lines.append(chunk.content[:500] + ("…" if len(chunk.content) > 500 else ""))
        lines.append("")

    return ToolResult(output="\n".join(lines))

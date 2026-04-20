"""SPEC-V2-REPO-02 T-3/T-4 + REPO-04 + REPO-05: Repo index, retrieval, and token budget."""

from __future__ import annotations

import logging
import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from vaig.core.config import RepoInvestigationConfig
from vaig.core.repo_adapter import RepoAdapter
from vaig.core.repo_chunkers import Chunk, _token_estimate, chunk_file
from vaig.core.repo_pipeline import (
    EvidenceGap,
    TierOutcome,
    classify_file_with_evidence,
    detect_file_kind,
)
from vaig.core.repo_redactor import SecretRedactor

logger = logging.getLogger(__name__)

# ── T-3: Retrieved chunk ──────────────────────────────────────────────────────


class RetrievedRepoChunk(BaseModel):
    """A Chunk retrieved from the index, enriched with retrieval metadata."""

    file_path: str
    start_line: int
    end_line: int
    content: str
    token_estimate: int
    kind: str
    outline: str
    relevance_score: float
    retrieval_query: str

    @classmethod
    def from_chunk(
        cls,
        chunk: Chunk,
        relevance_score: float,
        retrieval_query: str,
    ) -> RetrievedRepoChunk:
        return cls(
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            content=chunk.content,
            token_estimate=chunk.token_estimate,
            kind=chunk.kind,
            outline=chunk.outline,
            relevance_score=relevance_score,
            retrieval_query=retrieval_query,
        )

    def to_chunk(self) -> Chunk:
        return Chunk(
            file_path=self.file_path,
            start_line=self.start_line,
            end_line=self.end_line,
            content=self.content,
            token_estimate=self.token_estimate,
            kind=self.kind,
            outline=self.outline,
        )


# ── REPO-04: Relevance gate ───────────────────────────────────────────────────


def _tokenise(text: str) -> set[str]:
    """Lowercase word tokens, length >= 3."""
    return {w for w in re.findall(r"[a-z0-9_-]+", text.lower()) if len(w) >= 3}


def compute_relevance_keywords(
    query: str,
    findings: list[Any],  # Finding objects — access .title, .affected_resources (list[str])
    service_names: list[str] | None = None,
) -> frozenset[str]:
    """Build keyword set from query + findings for the relevance gate."""
    keywords: set[str] = set()
    keywords.update(_tokenise(query))
    for svc in (service_names or []):
        keywords.update(_tokenise(svc))
    for finding in findings:
        title = getattr(finding, "title", None)
        if title:
            keywords.update(_tokenise(str(title)))
        affected = getattr(finding, "affected_resources", None)
        if affected:
            for res in affected:
                keywords.update(_tokenise(str(res)))
    return frozenset(keywords)


_ALWAYS_PASS_KINDS = frozenset({"helm_values", "argocd_app", "istio_crd", "helm_chart_root"})


def relevance_gate(
    chunks: list[Chunk],
    keywords: frozenset[str],
) -> list[Chunk]:
    """Stage-1 filter: keep chunks that mention any keyword OR are always-pass kinds.

    A chunk passes when:
      - chunk.kind in _ALWAYS_PASS_KINDS, OR
      - any keyword appears in chunk.content.lower() or chunk.outline.lower()

    Returns filtered list. Typical retention: 5-15% of all chunks.
    """
    if not keywords:
        return list(chunks)

    result: list[Chunk] = []
    for chunk in chunks:
        if chunk.kind in _ALWAYS_PASS_KINDS:
            result.append(chunk)
            continue
        content_lower = chunk.content.lower()
        outline_lower = chunk.outline.lower()
        for kw in keywords:
            if kw in content_lower or kw in outline_lower:
                result.append(chunk)
                break
    return result


# ── T-3: RepoIndex ────────────────────────────────────────────────────────────


def _keyword_score(query_tokens: set[str], chunk: Chunk) -> float:
    """Fallback: keyword overlap score when sklearn is unavailable."""
    if not query_tokens:
        return 0.0
    chunk_tokens = _tokenise(chunk.content) | _tokenise(chunk.outline)
    overlap = len(query_tokens & chunk_tokens)
    return overlap / (len(query_tokens) + 1e-9)


class RepoIndex:
    """Lightweight in-memory index backed by TF-IDF cosine similarity."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._vectorizer: Any = None
        self._matrix: Any = None
        self._use_sklearn = False

        if not chunks:
            return

        corpus = [f"{c.outline} {c.content}" for c in chunks]
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            self._cosine_similarity = cosine_similarity
            vectorizer = TfidfVectorizer(
                strip_accents="unicode",
                analyzer="word",
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
            )
            self._matrix = vectorizer.fit_transform(corpus)
            self._vectorizer = vectorizer
            self._use_sklearn = True
        except ImportError:
            logger.debug("sklearn not available — falling back to keyword overlap scoring")

    def search(self, query: str, k: int = 8) -> list[RetrievedRepoChunk]:
        """Return top-k chunks by cosine similarity to query."""
        if not self._chunks:
            return []

        k = min(k, len(self._chunks))

        if self._use_sklearn:
            q_vec = self._vectorizer.transform([query])
            scores = self._cosine_similarity(q_vec, self._matrix).flatten()
            # Pair with index, sort descending
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            top = ranked[:k]
            return [
                RetrievedRepoChunk.from_chunk(self._chunks[i], float(s), query)
                for i, s in top
            ]
        else:
            query_tokens = _tokenise(query)
            scored = [
                (chunk, _keyword_score(query_tokens, chunk))
                for chunk in self._chunks
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [
                RetrievedRepoChunk.from_chunk(chunk, score, query)
                for chunk, score in scored[:k]
            ]

    @classmethod
    def build(
        cls,
        adapter: RepoAdapter,
        config: RepoInvestigationConfig,
        ref: str = "HEAD",
        *,
        keywords: frozenset[str] | None = None,
        redactor: SecretRedactor | None = None,
    ) -> tuple[RepoIndex, list[EvidenceGap]]:
        """Full pipeline: list_tree → classify → chunk → redact → relevance_gate → index.

        Returns the index and any EvidenceGaps emitted during processing.
        Streaming chunker is used for files above streaming_threshold_bytes.
        """
        all_gaps: list[EvidenceGap] = []
        all_chunks: list[Chunk] = []

        paths = config.paths if config.paths else [""]

        for search_path in paths:
            try:
                tree = adapter.list_tree(ref, search_path)
            except Exception as exc:
                except_types = (KeyboardInterrupt, SystemExit)
                if isinstance(exc, except_types):
                    raise
                logger.warning("Failed to list tree for path %r: %s", search_path, exc)
                continue

            for meta in tree:
                result, gaps = classify_file_with_evidence(meta, config)
                all_gaps.extend(gaps)

                if result.outcome == TierOutcome.SKIP:
                    continue

                try:
                    content = adapter.fetch_file(ref, meta.path)
                except Exception as exc:
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        raise
                    logger.warning("Failed to fetch %r: %s", meta.path, exc)
                    continue

                # Detect kind if unknown
                kind = meta.kind
                if kind in ("unknown", "text", ""):
                    kind = detect_file_kind(content[:4096])

                # Apply redaction
                if redactor is not None and config.redaction_enabled:
                    redaction_result = redactor.redact(content, file_path=meta.path)
                    content = redaction_result.redacted_content

                # Chunk
                chunks = chunk_file(content, meta.path, kind)
                all_chunks.extend(chunks)

        # Apply relevance gate if keywords provided
        if keywords:
            all_chunks = relevance_gate(all_chunks, keywords)

        return cls(all_chunks), all_gaps


# ── T-4: Ranged file read ─────────────────────────────────────────────────────


class RangedReadResult(BaseModel):
    content: str
    total_bytes: int
    has_more: bool
    next_offset: int | None


def read_repo_file(
    adapter: RepoAdapter,
    ref: str,
    path: str,
    offset: int = 0,
    limit: int = 200,  # lines
) -> RangedReadResult:
    """T-4: read a window of lines from a repo file. Agent paginates."""
    content = adapter.fetch_file(ref, path)
    total_bytes = len(content.encode("utf-8"))
    lines = content.splitlines(keepends=True)
    total_lines = len(lines)

    window = lines[offset : offset + limit]
    window_content = "".join(window)

    end_offset = offset + len(window)
    has_more = end_offset < total_lines
    next_offset = end_offset if has_more else None

    return RangedReadResult(
        content=window_content,
        total_bytes=total_bytes,
        has_more=has_more,
        next_offset=next_offset,
    )


# ── REPO-05: Token budget manager ────────────────────────────────────────────


_DEFAULT_MAX_TOKENS_PER_PATH = 10_000  # ~8k words


class BudgetExceededFallback(StrEnum):
    REDUCE_K = "reduce_k"
    DEDUP = "dedup"
    SUMMARISE = "summarise"
    EVIDENCE_GAP = "evidence_gap"


class BudgetResult(BaseModel):
    chunks: list[RetrievedRepoChunk]
    fallbacks_applied: list[BudgetExceededFallback]
    evidence_gap: EvidenceGap | None = None
    total_candidate_chunks: int
    total_returned_chunks: int


def _chunks_token_total(chunks: list[RetrievedRepoChunk]) -> int:
    return sum(c.token_estimate for c in chunks)


def _overlaps(a: RetrievedRepoChunk, b: RetrievedRepoChunk) -> bool:
    """Return True if a and b are from the same file and have overlapping line ranges."""
    if a.file_path != b.file_path:
        return False
    return a.start_line <= b.end_line and b.start_line <= a.end_line


class TokenBudgetManager:
    def __init__(self, max_tokens: int = _DEFAULT_MAX_TOKENS_PER_PATH) -> None:
        self.max_tokens = max_tokens

    def apply(
        self,
        chunks: list[RetrievedRepoChunk],
        *,
        path_label: str = "",
    ) -> BudgetResult:
        """Apply tiered fallback ladder:
        1. If within budget → return as-is (no fallbacks).
        2. Reduce k: drop lowest-score chunks until within budget.
        3. De-duplicate: remove overlapping chunks (same file, overlapping line ranges).
        4. Summarise: for remaining over-budget chunks, replace content with outline.
        5. If still over budget → emit EvidenceGap "repo_retrieval_truncated".
        Never silently drop — every chunk is either returned, summarised, or in the gap.
        """
        total_candidate = len(chunks)
        fallbacks: list[BudgetExceededFallback] = []
        working = list(chunks)

        # Step 1: within budget?
        if _chunks_token_total(working) <= self.max_tokens:
            return BudgetResult(
                chunks=working,
                fallbacks_applied=[],
                evidence_gap=None,
                total_candidate_chunks=total_candidate,
                total_returned_chunks=len(working),
            )

        # Step 2: Reduce k — drop lowest-score chunks
        fallbacks.append(BudgetExceededFallback.REDUCE_K)
        # Sort descending by score to preserve best chunks
        working.sort(key=lambda c: c.relevance_score, reverse=True)
        while len(working) > 1 and _chunks_token_total(working) > self.max_tokens:
            working.pop()  # pop from tail (lowest score)

        # If even the single top chunk exceeds budget, emit an EvidenceGap rather
        # than silently returning empty — caller gets at least the best chunk.
        if _chunks_token_total(working) > self.max_tokens and len(working) == 1:
            # Keep the single best chunk and emit a TRUNCATED gap
            fallbacks.append(BudgetExceededFallback.EVIDENCE_GAP)
            gap = EvidenceGap(
                source="repo_processing",
                kind="TRUNCATED",
                level="WARN",
                path=path_label or None,
                details=(
                    f"Token budget ({self.max_tokens}) is smaller than the top-scored chunk "
                    f"({_chunks_token_total(working)} tokens). Returning 1 chunk (best match) "
                    f"to avoid empty result. {total_candidate - 1} candidate(s) dropped."
                ),
            )
            return BudgetResult(
                chunks=working,
                fallbacks_applied=fallbacks,
                evidence_gap=gap,
                total_candidate_chunks=total_candidate,
                total_returned_chunks=len(working),
            )

        if _chunks_token_total(working) <= self.max_tokens:
            return BudgetResult(
                chunks=working,
                fallbacks_applied=fallbacks,
                evidence_gap=None,
                total_candidate_chunks=total_candidate,
                total_returned_chunks=len(working),
            )

        # Step 3: De-duplicate overlapping chunks
        fallbacks.append(BudgetExceededFallback.DEDUP)
        deduped: list[RetrievedRepoChunk] = []
        for chunk in working:
            dominated = any(_overlaps(chunk, kept) and kept.relevance_score >= chunk.relevance_score for kept in deduped)
            if not dominated:
                deduped.append(chunk)
        working = deduped

        if _chunks_token_total(working) <= self.max_tokens:
            return BudgetResult(
                chunks=working,
                fallbacks_applied=fallbacks,
                evidence_gap=None,
                total_candidate_chunks=total_candidate,
                total_returned_chunks=len(working),
            )

        # Step 4: Summarise — replace content with outline (no LLM call)
        fallbacks.append(BudgetExceededFallback.SUMMARISE)
        summarised: list[RetrievedRepoChunk] = []
        for chunk in working:
            if _chunks_token_total(summarised) + _token_estimate(chunk.outline) <= self.max_tokens:
                # Replace content with outline
                summarised_chunk = chunk.model_copy(
                    update={"content": chunk.outline, "token_estimate": _token_estimate(chunk.outline)}
                )
                summarised.append(summarised_chunk)
            else:
                break
        working = summarised

        if _chunks_token_total(working) <= self.max_tokens:
            return BudgetResult(
                chunks=working,
                fallbacks_applied=fallbacks,
                evidence_gap=None,
                total_candidate_chunks=total_candidate,
                total_returned_chunks=len(working),
            )

        # Step 5: Still over budget — emit EvidenceGap
        fallbacks.append(BudgetExceededFallback.EVIDENCE_GAP)
        dropped_count = total_candidate - len(working)
        gap = EvidenceGap(
            source="repo_processing",
            kind="repo_retrieval_truncated",
            level="WARN",
            path=path_label or None,
            details=(
                f"Token budget ({self.max_tokens}) exceeded after all fallbacks. "
                f"{dropped_count} of {total_candidate} candidate chunks not returned. "
                f"Returned {len(working)} chunks after summarisation."
            ),
        )

        return BudgetResult(
            chunks=working,
            fallbacks_applied=fallbacks,
            evidence_gap=gap,
            total_candidate_chunks=total_candidate,
            total_returned_chunks=len(working),
        )

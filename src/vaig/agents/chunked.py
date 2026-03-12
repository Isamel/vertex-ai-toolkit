"""Chunked file processor — Map-Reduce for files exceeding the context window.

Splits oversized content into line-boundary chunks with configurable overlap,
processes each chunk independently through GeminiClient.generate(), and
consolidates partial results into a final answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vaig.core.exceptions import ChunkedProcessingError, TokenBudgetError

if TYPE_CHECKING:
    from collections.abc import Callable

    from vaig.core.client import GeminiClient
    from vaig.core.config import Settings

logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────


@dataclass
class ChunkResult:
    """Result from processing a single chunk."""

    chunk_index: int
    total_chunks: int
    content: str
    success: bool = True
    error: str = ""
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class TokenBudget:
    """Token budget calculation for chunked processing.

    Computes how many tokens each chunk can use after reserving space
    for system prompt, user prompt, max output, and a safety margin.
    """

    context_window: int
    system_prompt_tokens: int
    user_prompt_tokens: int
    max_output_tokens: int
    safety_margin: float = 0.1

    @property
    def chunk_budget(self) -> int:
        """Available tokens per chunk for file content.

        Formula:
            usable = context_window * (1 - safety_margin)
            chunk_budget = usable - system_prompt_tokens - user_prompt_tokens - max_output_tokens
        """
        usable = int(self.context_window * (1 - self.safety_margin))
        budget = usable - self.system_prompt_tokens - self.user_prompt_tokens - self.max_output_tokens
        return max(budget, 0)


# ── ChunkedProcessor ─────────────────────────────────────────


class ChunkedProcessor:
    """Orchestrates chunked file analysis via Map-Reduce.

    Usage:
        processor = ChunkedProcessor(client, settings)
        budget = processor.calculate_budget(system_instruction, user_prompt, model_id)
        if processor.needs_chunking(content, budget):
            result = processor.process_ask(content, user_prompt, system_instruction, budget)
    """

    def __init__(self, client: GeminiClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    # ── Budget ────────────────────────────────────────────────

    def calculate_budget(
        self,
        system_instruction: str,
        user_prompt: str,
        model_id: str | None = None,
    ) -> TokenBudget:
        """Compute the token budget for chunking.

        Uses count_tokens() to measure system instruction and user prompt.
        Falls back to len(text)//4 if count_tokens() fails.
        """
        mid = model_id or self._client.current_model
        model_info = self._settings.get_model_info(mid)

        context_window = model_info.context_window if model_info else 1_048_576
        max_output = model_info.max_output_tokens if model_info else 65_536
        safety_margin = self._settings.chunking.token_safety_margin

        # Measure system instruction tokens
        sys_tokens = self._count_tokens_safe(system_instruction, model_id=mid)

        # Measure user prompt tokens
        prompt_tokens = self._count_tokens_safe(user_prompt, model_id=mid)

        budget = TokenBudget(
            context_window=context_window,
            system_prompt_tokens=sys_tokens,
            user_prompt_tokens=prompt_tokens,
            max_output_tokens=max_output,
            safety_margin=safety_margin,
        )

        if budget.chunk_budget <= 0:
            raise TokenBudgetError(
                f"Token budget is zero or negative. context_window={context_window}, "
                f"system_tokens={sys_tokens}, prompt_tokens={prompt_tokens}, "
                f"max_output={max_output}, safety_margin={safety_margin}. "
                "The system prompt and user prompt consume the entire context window."
            )

        logger.info(
            "Token budget: context_window=%d, system=%d, prompt=%d, "
            "max_output=%d, safety=%.0f%%, chunk_budget=%d",
            context_window,
            sys_tokens,
            prompt_tokens,
            max_output,
            safety_margin * 100,
            budget.chunk_budget,
        )
        return budget

    # ── Detection ─────────────────────────────────────────────

    def needs_chunking(self, content: str, budget: TokenBudget) -> bool:
        """Check whether content exceeds the chunk budget and needs splitting."""
        content_tokens = self._count_tokens_safe(content)
        needs = content_tokens > budget.chunk_budget
        logger.info(
            "needs_chunking: content_tokens=%d, chunk_budget=%d → %s",
            content_tokens,
            budget.chunk_budget,
            needs,
        )
        return needs

    # ── Splitting ─────────────────────────────────────────────

    def split_into_chunks(
        self,
        content: str,
        budget: TokenBudget,
        overlap_ratio: float | None = None,
    ) -> list[str]:
        """Split content into chunks at line boundaries with overlap.

        Args:
            content: The full text content to split.
            budget: Token budget that determines chunk size.
            overlap_ratio: Fraction of lines to overlap between chunks.
                           Defaults to settings.chunking.chunk_overlap_ratio.

        Returns:
            List of text chunks.
        """
        if overlap_ratio is None:
            overlap_ratio = self._settings.chunking.chunk_overlap_ratio

        chunk_budget = budget.chunk_budget
        lines = content.splitlines(keepends=True)

        if not lines:
            return [content] if content else []

        # Estimate tokens per line using len//4 heuristic (fast, no API calls)
        line_token_estimates = [max(len(line) // 4, 1) for line in lines]

        chunks: list[str] = []
        start_idx = 0

        while start_idx < len(lines):
            # Accumulate lines until we hit the chunk budget
            accumulated_tokens = 0
            end_idx = start_idx

            while end_idx < len(lines) and accumulated_tokens + line_token_estimates[end_idx] <= chunk_budget:
                accumulated_tokens += line_token_estimates[end_idx]
                end_idx += 1

            # Handle case where a single line exceeds the budget
            if end_idx == start_idx:
                # Force-include at least one line (will be a large chunk)
                end_idx = start_idx + 1
                logger.warning(
                    "Single line at index %d exceeds chunk budget (%d tokens est.)",
                    start_idx,
                    line_token_estimates[start_idx],
                )

            chunk_text = "".join(lines[start_idx:end_idx])
            chunks.append(chunk_text)

            # Calculate overlap: go back by overlap_ratio of the chunk's line count
            chunk_line_count = end_idx - start_idx
            overlap_lines = int(chunk_line_count * overlap_ratio)

            # Next chunk starts after current minus overlap
            start_idx = end_idx - overlap_lines
            if start_idx >= end_idx:
                # Ensure forward progress
                start_idx = end_idx

        logger.info(
            "Split content into %d chunks (budget=%d tokens/chunk, overlap=%.0f%%)",
            len(chunks),
            chunk_budget,
            overlap_ratio * 100,
        )
        return chunks

    # ── Map-Reduce: Ask Mode ──────────────────────────────────

    def process_ask(
        self,
        content: str,
        user_prompt: str,
        system_instruction: str,
        budget: TokenBudget,
        *,
        model_id: str | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> str:
        """Process oversized content for ask mode using Map-Reduce.

        Map phase: Send each chunk + user prompt to the model.
        Reduce phase: Consolidate all partial results into a final answer.

        Args:
            content: The full file content that exceeds context window.
            user_prompt: The user's question/prompt.
            system_instruction: System instruction for the model.
            budget: Pre-computed token budget.
            model_id: Optional model override.
            on_progress: Callback(chunk_index, total_chunks) for progress updates.

        Returns:
            The consolidated final answer.

        Raises:
            ChunkedProcessingError: If all chunks fail.
        """
        chunks = self.split_into_chunks(content, budget)
        total = len(chunks)

        if total == 0:
            return ""

        if total == 1:
            # No need for map-reduce if it fits in one chunk
            logger.info("Content fits in a single chunk — no map-reduce needed")
            if on_progress:
                on_progress(1, 1)
            result = self._client.generate(
                f"{user_prompt}\n\n{chunks[0]}",
                system_instruction=system_instruction,
                model_id=model_id,
            )
            return result.text

        # ── Map phase ─────────────────────────────────────────
        chunk_results: list[ChunkResult] = []

        for i, chunk in enumerate(chunks):
            if on_progress:
                on_progress(i + 1, total)

            map_prompt = (
                f"{user_prompt}\n\n"
                f"--- CHUNK {i + 1}/{total} ---\n"
                f"(This is part {i + 1} of {total} chunks from a large file. "
                f"Analyze this portion and provide your findings.)\n\n"
                f"{chunk}"
            )

            try:
                result = self._client.generate(
                    map_prompt,
                    system_instruction=system_instruction,
                    model_id=model_id,
                )
                chunk_results.append(
                    ChunkResult(
                        chunk_index=i,
                        total_chunks=total,
                        content=result.text,
                        success=True,
                        usage=result.usage,
                    )
                )
                logger.info("Chunk %d/%d processed successfully", i + 1, total)
            except Exception as exc:
                logger.warning("Chunk %d/%d failed: %s", i + 1, total, exc)
                chunk_results.append(
                    ChunkResult(
                        chunk_index=i,
                        total_chunks=total,
                        content="",
                        success=False,
                        error=str(exc),
                    )
                )

        # Check if ALL chunks failed
        successful = [r for r in chunk_results if r.success]
        if not successful:
            raise ChunkedProcessingError(
                f"All {total} chunks failed during processing",
                total_chunks=total,
                failed_chunks=[r.chunk_index for r in chunk_results],
                partial_results=chunk_results,
            )

        # ── Reduce phase ──────────────────────────────────────
        return self._consolidate(
            chunk_results,
            user_prompt,
            system_instruction,
            model_id=model_id,
        )

    # ── Map-Reduce: Chat Mode ─────────────────────────────────

    def process_chat_summary(
        self,
        content: str,
        budget: TokenBudget,
        *,
        model_id: str | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> str:
        """Process oversized content for chat mode — generate a dense summary.

        Used when loading files into a chat session. The summary replaces
        the full file content in the chat context.

        Args:
            content: The full file content.
            budget: Pre-computed token budget.
            model_id: Optional model override.
            on_progress: Callback(chunk_index, total_chunks) for progress updates.

        Returns:
            A dense summary of the entire file content.

        Raises:
            ChunkedProcessingError: If all chunks fail.
        """
        summary_instruction = (
            "You are a technical document analyzer. Your job is to create a "
            "comprehensive, detailed summary that preserves all key information, "
            "data points, patterns, and technical details from the content. "
            "The summary should be dense enough that someone reading it can "
            "answer detailed questions about the original content."
        )
        summary_prompt = (
            "Create a comprehensive technical summary of the following content. "
            "Preserve all key data points, patterns, errors, timestamps, "
            "and technical details. Be thorough — this summary will be used "
            "as context for follow-up questions."
        )

        return self.process_ask(
            content,
            summary_prompt,
            summary_instruction,
            budget,
            model_id=model_id,
            on_progress=on_progress,
        )

    # ── Private helpers ───────────────────────────────────────

    def _count_tokens_safe(self, text: str, *, model_id: str | None = None) -> int:
        """Count tokens, falling back to len(text)//4 on failure."""
        if not text:
            return 0
        try:
            return self._client.count_tokens(text, model_id=model_id)
        except Exception:
            logger.debug("count_tokens() failed — using len//4 fallback")
            return len(text) // 4

    def _consolidate(
        self,
        chunk_results: list[ChunkResult],
        user_prompt: str,
        system_instruction: str,
        *,
        model_id: str | None = None,
    ) -> str:
        """Reduce phase — consolidate partial chunk results into a final answer."""
        successful = [r for r in chunk_results if r.success]
        failed = [r for r in chunk_results if not r.success]

        # Build consolidation prompt
        parts: list[str] = []
        parts.append(
            f"You previously analyzed a large file in {len(chunk_results)} chunks. "
            f"Below are the partial analyses from each chunk."
        )

        if failed:
            parts.append(
                f"\nNote: {len(failed)} chunk(s) failed to process "
                f"(indices: {[r.chunk_index for r in failed]}). "
                "Work with the available results."
            )

        parts.append(f"\nOriginal question: {user_prompt}\n")

        for r in successful:
            parts.append(
                f"--- ANALYSIS FROM CHUNK {r.chunk_index + 1}/{r.total_chunks} ---\n"
                f"{r.content}\n"
            )

        parts.append(
            "\n--- CONSOLIDATION TASK ---\n"
            "Synthesize all the partial analyses above into a single, coherent, "
            "comprehensive response to the original question. "
            "Resolve any contradictions between chunks, merge related findings, "
            "and present a unified answer."
        )

        consolidation_prompt = "\n".join(parts)

        result = self._client.generate(
            consolidation_prompt,
            system_instruction=system_instruction,
            model_id=model_id,
        )
        return result.text

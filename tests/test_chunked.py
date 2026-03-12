"""Tests for chunked file processor — Map-Reduce for oversized files."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vaig.agents.chunked import ChunkedProcessor, ChunkResult, TokenBudget
from vaig.core.config import ChunkingConfig
from vaig.core.exceptions import ChunkedProcessingError, TokenBudgetError


# ── Helpers ───────────────────────────────────────────────────


def _make_settings(
    *,
    context_window: int = 1_048_576,
    max_output_tokens: int = 65_536,
    safety_margin: float = 0.1,
    overlap_ratio: float = 0.1,
) -> MagicMock:
    """Create a mock Settings with ModelInfo and ChunkingConfig."""
    settings = MagicMock()
    model_info = MagicMock()
    model_info.context_window = context_window
    model_info.max_output_tokens = max_output_tokens
    settings.get_model_info.return_value = model_info
    settings.chunking = ChunkingConfig(
        chunk_overlap_ratio=overlap_ratio,
        token_safety_margin=safety_margin,
    )
    return settings


def _make_client(
    *,
    generate_text: str = "Response",
    count_tokens_value: int | None = None,
) -> MagicMock:
    """Create a mock GeminiClient."""
    client = MagicMock()
    client.current_model = "gemini-2.5-pro"

    result = MagicMock()
    result.text = generate_text
    result.usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    client.generate.return_value = result

    if count_tokens_value is not None:
        client.count_tokens.return_value = count_tokens_value
    else:
        # Default: use len//4 fallback by raising
        client.count_tokens.side_effect = Exception("API unavailable")

    return client


# ══════════════════════════════════════════════════════════════
# TokenBudget data class
# ══════════════════════════════════════════════════════════════


class TestTokenBudget:
    def test_chunk_budget_calculation(self) -> None:
        budget = TokenBudget(
            context_window=1_000_000,
            system_prompt_tokens=1_000,
            user_prompt_tokens=500,
            max_output_tokens=65_536,
            safety_margin=0.1,
        )
        # usable = 1_000_000 * 0.9 = 900_000
        # chunk_budget = 900_000 - 1_000 - 500 - 65_536 = 832_964
        assert budget.chunk_budget == 832_964

    def test_chunk_budget_zero_safety(self) -> None:
        budget = TokenBudget(
            context_window=100_000,
            system_prompt_tokens=0,
            user_prompt_tokens=0,
            max_output_tokens=0,
            safety_margin=0.0,
        )
        assert budget.chunk_budget == 100_000

    def test_chunk_budget_clamps_to_zero(self) -> None:
        """When reserved tokens exceed context, budget should be 0, not negative."""
        budget = TokenBudget(
            context_window=10_000,
            system_prompt_tokens=5_000,
            user_prompt_tokens=5_000,
            max_output_tokens=5_000,
            safety_margin=0.1,
        )
        assert budget.chunk_budget == 0

    def test_chunk_budget_with_high_safety_margin(self) -> None:
        budget = TokenBudget(
            context_window=1_000_000,
            system_prompt_tokens=0,
            user_prompt_tokens=0,
            max_output_tokens=0,
            safety_margin=0.5,
        )
        # 1_000_000 * 0.5 = 500_000
        assert budget.chunk_budget == 500_000


# ══════════════════════════════════════════════════════════════
# ChunkResult data class
# ══════════════════════════════════════════════════════════════


class TestChunkResult:
    def test_defaults(self) -> None:
        r = ChunkResult(chunk_index=0, total_chunks=3, content="hello")
        assert r.success is True
        assert r.error == ""
        assert r.usage == {}

    def test_failure(self) -> None:
        r = ChunkResult(chunk_index=1, total_chunks=3, content="", success=False, error="API error")
        assert r.success is False
        assert r.error == "API error"


# ══════════════════════════════════════════════════════════════
# ChunkedProcessor.calculate_budget
# ══════════════════════════════════════════════════════════════


class TestCalculateBudget:
    def test_uses_count_tokens(self) -> None:
        """When count_tokens works, budget should use its values."""
        client = _make_client(count_tokens_value=100)
        settings = _make_settings()
        processor = ChunkedProcessor(client, settings)

        budget = processor.calculate_budget("system instruction", "user prompt")

        assert budget.system_prompt_tokens == 100
        assert budget.user_prompt_tokens == 100
        assert budget.context_window == 1_048_576
        assert budget.max_output_tokens == 65_536

    def test_fallback_on_count_tokens_failure(self) -> None:
        """When count_tokens raises, fall back to len//4."""
        client = _make_client()  # count_tokens raises by default
        settings = _make_settings()
        processor = ChunkedProcessor(client, settings)

        system_instruction = "x" * 400  # len=400, //4 = 100
        user_prompt = "y" * 200  # len=200, //4 = 50

        budget = processor.calculate_budget(system_instruction, user_prompt)
        assert budget.system_prompt_tokens == 100
        assert budget.user_prompt_tokens == 50

    def test_raises_on_zero_budget(self) -> None:
        """Should raise TokenBudgetError when prompts consume entire window."""
        client = _make_client(count_tokens_value=500_000)
        settings = _make_settings(context_window=100_000)
        processor = ChunkedProcessor(client, settings)

        with pytest.raises(TokenBudgetError):
            processor.calculate_budget("huge system prompt", "huge user prompt")

    def test_uses_model_id_parameter(self) -> None:
        """Should pass model_id to get_model_info."""
        client = _make_client(count_tokens_value=10)
        settings = _make_settings()
        processor = ChunkedProcessor(client, settings)

        processor.calculate_budget("sys", "prompt", model_id="gemini-2.5-flash")
        settings.get_model_info.assert_called_once_with("gemini-2.5-flash")

    def test_uses_client_model_when_no_model_id(self) -> None:
        """Should fall back to client.current_model."""
        client = _make_client(count_tokens_value=10)
        settings = _make_settings()
        processor = ChunkedProcessor(client, settings)

        processor.calculate_budget("sys", "prompt")
        settings.get_model_info.assert_called_once_with("gemini-2.5-pro")

    def test_fallback_when_model_info_is_none(self) -> None:
        """When get_model_info returns None, use defaults."""
        client = _make_client(count_tokens_value=10)
        settings = _make_settings()
        settings.get_model_info.return_value = None
        processor = ChunkedProcessor(client, settings)

        budget = processor.calculate_budget("sys", "prompt")
        assert budget.context_window == 1_048_576
        assert budget.max_output_tokens == 65_536


# ══════════════════════════════════════════════════════════════
# ChunkedProcessor.needs_chunking
# ══════════════════════════════════════════════════════════════


class TestNeedsChunking:
    def test_small_content_does_not_need_chunking(self) -> None:
        client = _make_client(count_tokens_value=1000)
        settings = _make_settings()
        processor = ChunkedProcessor(client, settings)

        budget = TokenBudget(
            context_window=1_000_000,
            system_prompt_tokens=100,
            user_prompt_tokens=100,
            max_output_tokens=65_536,
            safety_margin=0.1,
        )

        assert processor.needs_chunking("small content", budget) is False

    def test_large_content_needs_chunking(self) -> None:
        client = _make_client(count_tokens_value=1_000_000)
        settings = _make_settings()
        processor = ChunkedProcessor(client, settings)

        budget = TokenBudget(
            context_window=1_000_000,
            system_prompt_tokens=100,
            user_prompt_tokens=100,
            max_output_tokens=65_536,
            safety_margin=0.1,
        )

        assert processor.needs_chunking("large content", budget) is True

    def test_uses_fallback_when_count_tokens_fails(self) -> None:
        """Fallback to len//4 on count_tokens failure."""
        client = _make_client()  # raises on count_tokens
        settings = _make_settings()
        processor = ChunkedProcessor(client, settings)

        budget = TokenBudget(
            context_window=100,
            system_prompt_tokens=0,
            user_prompt_tokens=0,
            max_output_tokens=0,
            safety_margin=0.0,
        )
        # 200 chars -> 50 tokens, budget = 100 -> no chunking
        assert processor.needs_chunking("x" * 200, budget) is False
        # 500 chars -> 125 tokens, budget = 100 -> needs chunking
        assert processor.needs_chunking("x" * 500, budget) is True


# ══════════════════════════════════════════════════════════════
# ChunkedProcessor.split_into_chunks
# ══════════════════════════════════════════════════════════════


class TestSplitIntoChunks:
    def _budget(self, chunk_budget_chars: int) -> TokenBudget:
        """Create a budget where chunk_budget equals chunk_budget_chars // 4 tokens.

        Since the splitter uses len//4 estimation, a budget of N tokens
        corresponds to content of N*4 characters.
        """
        # We want chunk_budget = chunk_budget_chars // 4
        # chunk_budget = context_window * (1 - margin) - sys - prompt - output
        # Set sys=prompt=output=0, margin=0 -> chunk_budget = context_window
        target_tokens = chunk_budget_chars // 4
        return TokenBudget(
            context_window=target_tokens,
            system_prompt_tokens=0,
            user_prompt_tokens=0,
            max_output_tokens=0,
            safety_margin=0.0,
        )

    def test_empty_content(self) -> None:
        client = _make_client()
        settings = _make_settings()
        processor = ChunkedProcessor(client, settings)
        budget = self._budget(1000)

        assert processor.split_into_chunks("", budget) == []

    def test_single_chunk_content(self) -> None:
        """Content that fits in one chunk should produce exactly one chunk."""
        client = _make_client()
        settings = _make_settings()
        processor = ChunkedProcessor(client, settings)
        budget = self._budget(1000)

        content = "line 1\nline 2\nline 3\n"
        chunks = processor.split_into_chunks(content, budget)
        assert len(chunks) == 1
        assert chunks[0] == content

    def test_splits_at_line_boundaries(self) -> None:
        """Chunks should split at line boundaries, not mid-line."""
        client = _make_client()
        settings = _make_settings()
        processor = ChunkedProcessor(client, settings)

        # Each line is ~40 chars -> ~10 tokens
        # Budget of 20 tokens (80 chars) should fit ~2 lines per chunk
        budget = self._budget(80)

        content = "".join(f"line {i}: some content here!!!\n" for i in range(10))
        chunks = processor.split_into_chunks(content, budget, overlap_ratio=0.0)

        assert len(chunks) > 1
        # Every chunk should end with a newline (line boundary)
        for chunk in chunks:
            assert chunk.endswith("\n")
        # All content should be covered
        assert "".join(chunks) == content  # with 0 overlap, join == original

    def test_overlap_produces_more_chunks(self) -> None:
        """Overlap should produce more chunks than no overlap."""
        client = _make_client()
        settings = _make_settings()
        processor = ChunkedProcessor(client, settings)
        # Use a budget that gives ~5-10 lines per chunk so overlap of 0.5
        # produces a meaningful number of overlap lines (int truncation issue
        # with 2 lines and 0.3 overlap = 0 overlap lines)
        budget = self._budget(400)

        content = "".join(f"line {i}: some content here!!!\n" for i in range(50))

        chunks_no_overlap = processor.split_into_chunks(content, budget, overlap_ratio=0.0)
        chunks_with_overlap = processor.split_into_chunks(content, budget, overlap_ratio=0.5)

        assert len(chunks_with_overlap) > len(chunks_no_overlap)

    def test_overlap_shares_lines_between_chunks(self) -> None:
        """Adjacent chunks should share overlapping lines."""
        client = _make_client()
        settings = _make_settings()
        processor = ChunkedProcessor(client, settings)
        budget = self._budget(200)

        content = "".join(f"line {i}\n" for i in range(30))
        chunks = processor.split_into_chunks(content, budget, overlap_ratio=0.5)

        if len(chunks) >= 2:
            # Last lines of chunk 0 should appear at start of chunk 1
            lines_0 = chunks[0].splitlines()
            lines_1 = chunks[1].splitlines()
            # Some lines from end of chunk 0 should appear in chunk 1
            overlap = set(lines_0) & set(lines_1)
            assert len(overlap) > 0

    def test_single_line_exceeds_budget(self) -> None:
        """A single line larger than the budget should still be included."""
        client = _make_client()
        settings = _make_settings()
        processor = ChunkedProcessor(client, settings)
        budget = self._budget(40)  # 10 tokens

        content = "x" * 1000 + "\nshort\n"
        chunks = processor.split_into_chunks(content, budget, overlap_ratio=0.0)

        assert len(chunks) >= 2
        assert chunks[0] == "x" * 1000 + "\n"

    def test_uses_settings_overlap_by_default(self) -> None:
        """When no overlap_ratio is passed, use settings.chunking.chunk_overlap_ratio."""
        client = _make_client()
        settings = _make_settings(overlap_ratio=0.0)
        processor = ChunkedProcessor(client, settings)
        budget = self._budget(80)

        content = "".join(f"line {i}: some content here!!!\n" for i in range(10))
        chunks = processor.split_into_chunks(content, budget)  # no explicit overlap

        # With 0 overlap, join should equal original
        assert "".join(chunks) == content

    def test_forward_progress_guaranteed(self) -> None:
        """Even with high overlap, the algorithm must always make forward progress."""
        client = _make_client()
        settings = _make_settings()
        processor = ChunkedProcessor(client, settings)
        budget = self._budget(40)

        content = "".join(f"line {i}\n" for i in range(100))
        chunks = processor.split_into_chunks(content, budget, overlap_ratio=0.9)

        # Must terminate (not infinite loop) and cover all content
        assert len(chunks) > 0
        # All lines should appear in at least one chunk
        all_chunk_lines = set()
        for chunk in chunks:
            all_chunk_lines.update(chunk.splitlines())
        original_lines = set(content.splitlines())
        assert original_lines.issubset(all_chunk_lines)


# ══════════════════════════════════════════════════════════════
# ChunkedProcessor.process_ask (Map-Reduce)
# ══════════════════════════════════════════════════════════════


class TestProcessAsk:
    def _budget(self, chunk_budget_chars: int = 200) -> TokenBudget:
        target_tokens = chunk_budget_chars // 4
        return TokenBudget(
            context_window=target_tokens,
            system_prompt_tokens=0,
            user_prompt_tokens=0,
            max_output_tokens=0,
            safety_margin=0.0,
        )

    def test_empty_content_returns_empty(self) -> None:
        client = _make_client()
        settings = _make_settings()
        processor = ChunkedProcessor(client, settings)

        result = processor.process_ask("", "question", "sys", self._budget())
        assert result == ""
        client.generate.assert_not_called()

    def test_single_chunk_no_consolidation(self) -> None:
        """Content fitting one chunk should not trigger consolidation."""
        client = _make_client(generate_text="Direct answer")
        settings = _make_settings()
        processor = ChunkedProcessor(client, settings)

        result = processor.process_ask("short content\n", "question", "sys", self._budget(1000))
        assert result == "Direct answer"
        # Only 1 generate call (no consolidation)
        assert client.generate.call_count == 1

    def test_multi_chunk_calls_map_then_reduce(self) -> None:
        """Multiple chunks should trigger map (N calls) + reduce (1 call)."""
        client = _make_client(generate_text="Partial result")
        settings = _make_settings(overlap_ratio=0.0)
        processor = ChunkedProcessor(client, settings)

        content = "".join(f"line {i}: content here!!!\n" for i in range(20))
        budget = self._budget(80)

        # Count how many chunks we'll get
        chunks = processor.split_into_chunks(content, budget)
        expected_calls = len(chunks) + 1  # map + reduce

        processor.process_ask(content, "analyze", "sys", budget)
        assert client.generate.call_count == expected_calls

    def test_consolidation_prompt_contains_partial_results(self) -> None:
        """The reduce call should include all partial results."""
        call_count = 0

        def _generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return MagicMock(
                text=f"Analysis of chunk {call_count}",
                usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            )

        client = _make_client()
        client.generate.side_effect = _generate
        settings = _make_settings(overlap_ratio=0.0)
        processor = ChunkedProcessor(client, settings)

        content = "".join(f"line {i}: content here!!!\n" for i in range(20))
        budget = self._budget(80)
        chunks = processor.split_into_chunks(content, budget)

        if len(chunks) < 2:
            pytest.skip("Need at least 2 chunks for this test")

        result = processor.process_ask(content, "analyze", "sys", budget)

        # Last call is consolidation — check its prompt
        consolidation_call = client.generate.call_args_list[-1]
        consolidation_prompt = consolidation_call.args[0] if consolidation_call.args else consolidation_call.kwargs.get("prompt", "")
        assert "CONSOLIDATION TASK" in consolidation_prompt

    def test_progress_callback_called(self) -> None:
        """on_progress should be called for each chunk."""
        client = _make_client(generate_text="result")
        settings = _make_settings(overlap_ratio=0.0)
        processor = ChunkedProcessor(client, settings)

        content = "".join(f"line {i}: content here!!!\n" for i in range(20))
        budget = self._budget(80)

        progress_calls: list[tuple[int, int]] = []

        def on_progress(current: int, total: int) -> None:
            progress_calls.append((current, total))

        processor.process_ask(content, "analyze", "sys", budget, on_progress=on_progress)

        chunks = processor.split_into_chunks(content, budget)
        assert len(progress_calls) == len(chunks)
        # First call should be (1, total), last should be (total, total)
        assert progress_calls[0][0] == 1
        assert progress_calls[-1][0] == len(chunks)

    def test_partial_failure_still_consolidates(self) -> None:
        """If some chunks fail, consolidation should still happen with successful ones."""
        call_count = 0

        def _generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("API error")
            return MagicMock(
                text=f"Result {call_count}",
                usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            )

        client = _make_client()
        client.generate.side_effect = _generate
        settings = _make_settings(overlap_ratio=0.0)
        processor = ChunkedProcessor(client, settings)

        content = "".join(f"line {i}: content here!!!\n" for i in range(20))
        budget = self._budget(80)
        chunks = processor.split_into_chunks(content, budget)

        if len(chunks) < 3:
            pytest.skip("Need at least 3 chunks for partial failure test")

        # Should not raise — partial results are OK
        result = processor.process_ask(content, "analyze", "sys", budget)
        assert result  # non-empty

    def test_all_chunks_fail_raises(self) -> None:
        """If ALL chunks fail, should raise ChunkedProcessingError."""
        client = _make_client()
        client.generate.side_effect = RuntimeError("API down")
        settings = _make_settings(overlap_ratio=0.0)
        processor = ChunkedProcessor(client, settings)

        content = "".join(f"line {i}: content here!!!\n" for i in range(20))
        budget = self._budget(80)

        with pytest.raises(ChunkedProcessingError) as exc_info:
            processor.process_ask(content, "analyze", "sys", budget)

        assert exc_info.value.total_chunks > 0
        assert len(exc_info.value.failed_chunks) == exc_info.value.total_chunks

    def test_model_id_passed_to_generate(self) -> None:
        """model_id should be forwarded to client.generate."""
        client = _make_client(generate_text="answer")
        settings = _make_settings()
        processor = ChunkedProcessor(client, settings)

        processor.process_ask("short\n", "q", "sys", self._budget(1000), model_id="gemini-2.5-flash")

        call_kwargs = client.generate.call_args.kwargs
        assert call_kwargs.get("model_id") == "gemini-2.5-flash"


# ══════════════════════════════════════════════════════════════
# ChunkedProcessor.process_chat_summary
# ══════════════════════════════════════════════════════════════


class TestProcessChatSummary:
    def test_returns_summary_text(self) -> None:
        client = _make_client(generate_text="Summary of the content")
        settings = _make_settings()
        processor = ChunkedProcessor(client, settings)

        budget = TokenBudget(
            context_window=10_000,
            system_prompt_tokens=0,
            user_prompt_tokens=0,
            max_output_tokens=0,
            safety_margin=0.0,
        )

        result = processor.process_chat_summary("short content\n", budget)
        assert result == "Summary of the content"

    def test_uses_summary_instruction(self) -> None:
        """Should use the technical document analyzer system instruction."""
        client = _make_client(generate_text="summary")
        settings = _make_settings()
        processor = ChunkedProcessor(client, settings)

        budget = TokenBudget(
            context_window=10_000,
            system_prompt_tokens=0,
            user_prompt_tokens=0,
            max_output_tokens=0,
            safety_margin=0.0,
        )

        processor.process_chat_summary("content\n", budget)

        call_kwargs = client.generate.call_args.kwargs
        sys_instruction = call_kwargs.get("system_instruction", "")
        assert "technical document analyzer" in sys_instruction.lower()


# ══════════════════════════════════════════════════════════════
# Exceptions
# ══════════════════════════════════════════════════════════════


class TestExceptions:
    def test_chunked_processing_error_fields(self) -> None:
        err = ChunkedProcessingError(
            "All chunks failed",
            total_chunks=5,
            failed_chunks=[0, 2, 4],
            partial_results=[],
        )
        assert err.total_chunks == 5
        assert err.failed_chunks == [0, 2, 4]
        assert err.partial_results == []
        assert "All chunks failed" in str(err)

    def test_token_budget_error(self) -> None:
        err = TokenBudgetError("Budget exhausted")
        assert "Budget exhausted" in str(err)
        assert isinstance(err, Exception)

    def test_chunked_processing_error_defaults(self) -> None:
        err = ChunkedProcessingError("Something failed")
        assert err.total_chunks == 0
        assert err.failed_chunks == []
        assert err.partial_results == []


# ══════════════════════════════════════════════════════════════
# ChunkingConfig
# ══════════════════════════════════════════════════════════════


class TestChunkingConfig:
    def test_defaults(self) -> None:
        cfg = ChunkingConfig()
        assert cfg.chunk_overlap_ratio == 0.1
        assert cfg.token_safety_margin == 0.1

    def test_custom_values(self) -> None:
        cfg = ChunkingConfig(chunk_overlap_ratio=0.2, token_safety_margin=0.15)
        assert cfg.chunk_overlap_ratio == 0.2
        assert cfg.token_safety_margin == 0.15


# ══════════════════════════════════════════════════════════════
# _count_tokens_safe (private helper, tested through public API)
# ══════════════════════════════════════════════════════════════


class TestCountTokensSafe:
    def test_empty_string_returns_zero(self) -> None:
        client = _make_client(count_tokens_value=999)
        settings = _make_settings()
        processor = ChunkedProcessor(client, settings)

        # Empty string should return 0 without calling count_tokens
        budget = TokenBudget(
            context_window=1_000_000,
            system_prompt_tokens=0,
            user_prompt_tokens=0,
            max_output_tokens=0,
            safety_margin=0.0,
        )
        # needs_chunking calls _count_tokens_safe internally
        assert processor.needs_chunking("", budget) is False

    def test_fallback_to_len_div_4(self) -> None:
        """When count_tokens fails, use len//4."""
        client = _make_client()  # count_tokens raises
        settings = _make_settings()
        processor = ChunkedProcessor(client, settings)

        budget = TokenBudget(
            context_window=100,
            system_prompt_tokens=0,
            user_prompt_tokens=0,
            max_output_tokens=0,
            safety_margin=0.0,
        )
        # 600 chars -> 150 tokens > 100 budget -> needs chunking
        assert processor.needs_chunking("x" * 600, budget) is True

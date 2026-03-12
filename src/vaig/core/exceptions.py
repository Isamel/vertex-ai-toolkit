"""Custom exceptions for the VAIG toolkit."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vaig.agents.chunked import ChunkResult


class VAIGError(Exception):
    """Base exception for all VAIG errors."""


class GeminiClientError(VAIGError):
    """Error raised when the Gemini client encounters an unrecoverable failure.

    Wraps the original exception and includes retry context when applicable.
    """

    def __init__(
        self,
        message: str,
        *,
        original_error: Exception | None = None,
        retries_attempted: int = 0,
    ) -> None:
        super().__init__(message)
        self.original_error = original_error
        self.retries_attempted = retries_attempted


class GeminiRateLimitError(GeminiClientError):
    """Raised when the API returns 429 (ResourceExhausted) after all retries."""


class GeminiConnectionError(GeminiClientError):
    """Raised when the API returns a server/connection error (500/502/503/504) after all retries."""


class ToolExecutionError(VAIGError):
    """Raised when a tool fails to execute."""

    def __init__(self, message: str, *, tool_name: str) -> None:
        super().__init__(message)
        self.tool_name = tool_name


class MaxIterationsError(VAIGError):
    """Raised when the tool-use loop exceeds the configured maximum iterations."""

    def __init__(self, message: str, *, iterations: int) -> None:
        super().__init__(message)
        self.iterations = iterations


class ChunkedProcessingError(VAIGError):
    """Raised when chunked file processing fails (partial or total failure).

    Carries context about which chunks failed and any partial results
    collected before the failure.
    """

    def __init__(
        self,
        message: str,
        *,
        total_chunks: int = 0,
        failed_chunks: list[int] | None = None,
        partial_results: list[ChunkResult] | None = None,
    ) -> None:
        super().__init__(message)
        self.total_chunks = total_chunks
        self.failed_chunks = failed_chunks or []
        self.partial_results = partial_results or []


class TokenBudgetError(VAIGError):
    """Raised when the token budget cannot be computed or is invalid.

    Examples: context_window is too small to fit even the system prompt,
    or count_tokens() fails and no fallback is possible.
    """

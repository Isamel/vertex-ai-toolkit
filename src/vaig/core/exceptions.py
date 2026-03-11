"""Custom exceptions for the VAIG toolkit."""

from __future__ import annotations


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

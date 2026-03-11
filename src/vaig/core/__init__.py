"""Core package."""

from vaig.core.client import ChatMessage, GeminiClient, GenerationResult
from vaig.core.config import Settings, get_settings
from vaig.core.exceptions import (
    GeminiClientError,
    GeminiConnectionError,
    GeminiRateLimitError,
    VAIGError,
)

__all__ = [
    "ChatMessage",
    "GeminiClient",
    "GeminiClientError",
    "GeminiConnectionError",
    "GeminiRateLimitError",
    "GenerationResult",
    "Settings",
    "VAIGError",
    "get_settings",
]

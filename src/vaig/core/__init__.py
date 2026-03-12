"""Core package."""

from vaig.core.client import ChatMessage, GeminiClient, GenerationResult, ToolCallResult
from vaig.core.config import CodingConfig, Settings, get_settings
from vaig.core.exceptions import (
    GeminiClientError,
    GeminiConnectionError,
    GeminiRateLimitError,
    MaxIterationsError,
    ToolExecutionError,
    VAIGError,
)

__all__ = [
    "ChatMessage",
    "CodingConfig",
    "GeminiClient",
    "GeminiClientError",
    "GeminiConnectionError",
    "GeminiRateLimitError",
    "GenerationResult",
    "MaxIterationsError",
    "Settings",
    "ToolCallResult",
    "ToolExecutionError",
    "VAIGError",
    "get_settings",
]

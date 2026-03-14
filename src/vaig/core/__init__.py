"""Core package."""

from vaig.core.cache import CacheStats, ResponseCache
from vaig.core.client import ChatMessage, GeminiClient, GenerationResult, StreamResult, ToolCallResult
from vaig.core.config import BudgetConfig, CacheConfig, CodingConfig, Settings, get_settings
from vaig.core.config_switcher import SwitchResult, get_config_snapshot, switch_cluster, switch_location, switch_project
from vaig.core.cost_tracker import BudgetStatus, CostRecord, CostTracker
from vaig.core.exceptions import (
    GeminiClientError,
    GeminiConnectionError,
    GeminiRateLimitError,
    MaxIterationsError,
    ToolExecutionError,
    VAIGError,
)

__all__ = [
    "BudgetConfig",
    "BudgetStatus",
    "CacheConfig",
    "CacheStats",
    "ChatMessage",
    "CodingConfig",
    "CostRecord",
    "CostTracker",
    "GeminiClient",
    "GeminiClientError",
    "GeminiConnectionError",
    "GeminiRateLimitError",
    "GenerationResult",
    "MaxIterationsError",
    "ResponseCache",
    "Settings",
    "StreamResult",
    "SwitchResult",
    "ToolCallResult",
    "ToolExecutionError",
    "VAIGError",
    "get_config_snapshot",
    "get_settings",
    "switch_cluster",
    "switch_location",
    "switch_project",
]

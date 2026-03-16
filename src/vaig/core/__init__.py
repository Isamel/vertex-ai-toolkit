"""Core package."""

from vaig.core.async_utils import gather_with_errors, run_sync, to_async
from vaig.core.cache import CacheStats, ResponseCache
from vaig.core.client import ChatMessage, GeminiClient, GenerationResult, StreamResult, ToolCallResult
from vaig.core.config import BudgetConfig, CacheConfig, CodingConfig, Settings, get_settings
from vaig.core.config_switcher import SwitchResult, get_config_snapshot, switch_cluster, switch_location, switch_project
from vaig.core.cost_tracker import BudgetStatus, CostRecord, CostTracker
from vaig.core.exceptions import (
    GCPAuthError,
    GCPPermissionError,
    GeminiClientError,
    GeminiConnectionError,
    GeminiRateLimitError,
    K8sAuthError,
    MaxIterationsError,
    ToolExecutionError,
    VaigAuthError,
    VAIGError,
    format_error_for_user,
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
    "GCPAuthError",
    "GCPPermissionError",
    "GeminiClient",
    "GeminiClientError",
    "GeminiConnectionError",
    "GeminiRateLimitError",
    "GenerationResult",
    "K8sAuthError",
    "MaxIterationsError",
    "ResponseCache",
    "Settings",
    "StreamResult",
    "SwitchResult",
    "ToolCallResult",
    "ToolExecutionError",
    "VAIGError",
    "VaigAuthError",
    "format_error_for_user",
    "gather_with_errors",
    "get_config_snapshot",
    "get_settings",
    "run_sync",
    "switch_cluster",
    "switch_location",
    "switch_project",
    "to_async",
]

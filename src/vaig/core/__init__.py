"""Core package."""

from vaig.core.client import ChatMessage, GeminiClient, GenerationResult, StreamResult, ToolCallResult
from vaig.core.config import BudgetConfig, CodingConfig, Settings, get_settings
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

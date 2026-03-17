"""Core package."""

from vaig.core.async_utils import gather_with_errors, run_sync, to_async
from vaig.core.cache import CacheStats, ResponseCache
from vaig.core.client import ChatMessage, GeminiClient, GenerationResult, StreamResult, ToolCallResult
from vaig.core.config import BudgetConfig, CacheConfig, CodingConfig, Settings, get_settings
from vaig.core.config_switcher import SwitchResult, get_config_snapshot, switch_cluster, switch_location, switch_project
from vaig.core.cost_tracker import BudgetStatus, CostRecord, CostTracker
from vaig.core.event_bus import EventBus
from vaig.core.events import (
    ApiCalled,
    BudgetChecked,
    CliCommandTracked,
    ErrorOccurred,
    Event,
    SessionEnded,
    SessionStarted,
    SkillUsed,
    ToolExecuted,
)
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
from vaig.core.subscribers import TelemetrySubscriber

__all__ = [
    "ApiCalled",
    "BudgetChecked",
    "BudgetConfig",
    "BudgetStatus",
    "CacheConfig",
    "CacheStats",
    "ChatMessage",
    "CliCommandTracked",
    "CodingConfig",
    "CostRecord",
    "CostTracker",
    "ErrorOccurred",
    "Event",
    "EventBus",
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
    "SessionEnded",
    "SessionStarted",
    "Settings",
    "SkillUsed",
    "StreamResult",
    "SwitchResult",
    "TelemetrySubscriber",
    "ToolCallResult",
    "ToolExecuted",
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

"""Per-session cost tracking for token budget enforcement."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from vaig.core.config import BudgetConfig
from vaig.core.pricing import calculate_cost


class BudgetStatus(StrEnum):
    """Budget check result."""

    OK = "ok"
    WARNING = "warning"
    EXCEEDED = "exceeded"


@dataclass
class CostRecord:
    """A single API call's token usage and cost."""

    model_id: str
    prompt_tokens: int
    completion_tokens: int
    thinking_tokens: int
    cost: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CostTracker:
    """Thread-safe, per-session cost accumulator.

    Records individual API call costs and provides budget checking
    against a ``BudgetConfig``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[CostRecord] = []
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_thinking_tokens: int = 0
        self._total_cost: float = 0.0

    # ── Recording ─────────────────────────────────────────────

    def record(
        self,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        thinking_tokens: int = 0,
    ) -> CostRecord:
        """Record an API call and accumulate totals.

        Args:
            model_id: The model identifier used for the call.
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.
            thinking_tokens: Number of thinking/reasoning tokens.

        Returns:
            The ``CostRecord`` that was appended.
        """
        cost = calculate_cost(model_id, prompt_tokens, completion_tokens, thinking_tokens)
        # If model is unknown, cost is None — treat as 0.
        cost_value = cost if cost is not None else 0.0

        rec = CostRecord(
            model_id=model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            thinking_tokens=thinking_tokens,
            cost=cost_value,
        )

        with self._lock:
            self._records.append(rec)
            self._total_prompt_tokens += prompt_tokens
            self._total_completion_tokens += completion_tokens
            self._total_thinking_tokens += thinking_tokens
            self._total_cost += cost_value

        return rec

    # ── Queries ───────────────────────────────────────────────

    @property
    def total_cost(self) -> float:
        """Total accumulated cost in USD."""
        with self._lock:
            return self._total_cost

    @property
    def total_tokens(self) -> int:
        """Total tokens across all categories."""
        with self._lock:
            return (
                self._total_prompt_tokens
                + self._total_completion_tokens
                + self._total_thinking_tokens
            )

    @property
    def request_count(self) -> int:
        """Number of recorded API calls."""
        with self._lock:
            return len(self._records)

    def summary(self) -> dict[str, Any]:
        """Return a summary dict with full breakdown."""
        with self._lock:
            return {
                "total_cost": self._total_cost,
                "total_prompt_tokens": self._total_prompt_tokens,
                "total_completion_tokens": self._total_completion_tokens,
                "total_thinking_tokens": self._total_thinking_tokens,
                "total_tokens": (
                    self._total_prompt_tokens
                    + self._total_completion_tokens
                    + self._total_thinking_tokens
                ),
                "request_count": len(self._records),
            }

    # ── Budget checking ───────────────────────────────────────

    def check_budget(self, config: BudgetConfig) -> tuple[BudgetStatus, str | None]:
        """Check current spend against the budget configuration.

        Args:
            config: Budget settings to check against.

        Returns:
            A tuple of ``(status, message)`` where *message* is ``None``
            when status is ``OK``.
        """
        if not config.enabled:
            return BudgetStatus.OK, None

        with self._lock:
            cost = self._total_cost

        if cost >= config.max_cost_usd:
            return (
                BudgetStatus.EXCEEDED,
                f"Budget exceeded: ${cost:.4f} >= ${config.max_cost_usd:.2f}",
            )

        warn_at = config.max_cost_usd * config.warn_threshold
        if cost >= warn_at:
            pct = (cost / config.max_cost_usd) * 100
            return (
                BudgetStatus.WARNING,
                f"Budget warning: ${cost:.4f} ({pct:.0f}% of ${config.max_cost_usd:.2f} limit)",
            )

        return BudgetStatus.OK, None

    # ── Lifecycle ─────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all accumulated data."""
        with self._lock:
            self._records.clear()
            self._total_prompt_tokens = 0
            self._total_completion_tokens = 0
            self._total_thinking_tokens = 0
            self._total_cost = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize tracker state for persistence."""
        with self._lock:
            return {
                "total_cost": self._total_cost,
                "total_prompt_tokens": self._total_prompt_tokens,
                "total_completion_tokens": self._total_completion_tokens,
                "total_thinking_tokens": self._total_thinking_tokens,
                "request_count": len(self._records),
                "records": [
                    {
                        "model_id": r.model_id,
                        "prompt_tokens": r.prompt_tokens,
                        "completion_tokens": r.completion_tokens,
                        "thinking_tokens": r.thinking_tokens,
                        "cost": r.cost,
                        "timestamp": r.timestamp.isoformat(),
                    }
                    for r in self._records
                ],
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostTracker:
        """Restore tracker state from a serialized dict."""
        tracker = cls()
        for rec_data in data.get("records", []):
            rec = CostRecord(
                model_id=rec_data["model_id"],
                prompt_tokens=rec_data["prompt_tokens"],
                completion_tokens=rec_data["completion_tokens"],
                thinking_tokens=rec_data["thinking_tokens"],
                cost=rec_data["cost"],
                timestamp=datetime.fromisoformat(rec_data["timestamp"]),
            )
            tracker._records.append(rec)
            tracker._total_prompt_tokens += rec.prompt_tokens
            tracker._total_completion_tokens += rec.completion_tokens
            tracker._total_thinking_tokens += rec.thinking_tokens
            tracker._total_cost += rec.cost
        return tracker

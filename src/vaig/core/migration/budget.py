"""Budget tracker: enforces migration_budget_usd and per-iteration cost limits."""
from __future__ import annotations

import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["BudgetEventKind", "BudgetEvent", "MigrationBudgetManager"]


class BudgetEventKind(StrEnum):
    LLM_CALL = "llm_call"
    GATE_CHECK = "gate_check"
    FILE_PROCESSED = "file_processed"
    RETRY = "retry"
    SUMMARY = "summary"  # synthetic compaction summary event


class BudgetEvent(BaseModel):
    kind: BudgetEventKind
    tokens_used: int = 0
    cost_usd: float = 0.0
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )
    metadata: dict[str, object] = Field(default_factory=dict)


class MigrationBudgetManager(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens: int = 100_000
    max_cost_usd: float = 10.0
    events: list[BudgetEvent] = Field(default_factory=list)

    def record(self, event: BudgetEvent) -> None:
        """Append event."""
        self.events.append(event)

    def total_tokens(self) -> int:
        return sum(e.tokens_used for e in self.events)

    def total_cost(self) -> float:
        return sum(e.cost_usd for e in self.events)

    def is_over_budget(self) -> bool:
        """True if total_tokens > max_tokens OR total_cost > max_cost_usd."""
        return self.total_tokens() > self.max_tokens or self.total_cost() > self.max_cost_usd

    def remaining_tokens(self) -> int:
        return self.max_tokens - self.total_tokens()

    def remaining_cost(self) -> float:
        return self.max_cost_usd - self.total_cost()

    def compact_history(self, keep_last: int = 10) -> MigrationBudgetManager:
        """Return new manager with only last `keep_last` events, same limits, totals preserved via a synthetic summary event."""
        total_tok = self.total_tokens()
        total_cost = self.total_cost()
        tail = self.events[-keep_last:] if len(self.events) > keep_last else list(self.events)

        # Tokens/cost already in tail
        tail_tok = sum(e.tokens_used for e in tail)
        tail_cost = sum(e.cost_usd for e in tail)

        # Build a synthetic summary for the compacted portion
        compacted_tok = total_tok - tail_tok
        compacted_cost = total_cost - tail_cost

        new_events: list[BudgetEvent] = []
        if compacted_tok > 0 or compacted_cost > 0.0:
            summary_event = BudgetEvent(
                kind=BudgetEventKind.SUMMARY,
                tokens_used=compacted_tok,
                cost_usd=compacted_cost,
                metadata={"compacted": True, "original_event_count": len(self.events) - len(tail)},
            )
            new_events.append(summary_event)

        new_events.extend(tail)

        return MigrationBudgetManager(
            max_tokens=self.max_tokens,
            max_cost_usd=self.max_cost_usd,
            events=new_events,
        )

    def summary(self) -> dict[str, object]:
        """Return dict with total_tokens, total_cost, event_count, is_over_budget, remaining_tokens, remaining_cost."""
        return {
            "total_tokens": self.total_tokens(),
            "total_cost": self.total_cost(),
            "event_count": len(self.events),
            "is_over_budget": self.is_over_budget(),
            "remaining_tokens": self.remaining_tokens(),
            "remaining_cost": self.remaining_cost(),
        }

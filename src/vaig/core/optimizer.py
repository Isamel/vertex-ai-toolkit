"""Tool call optimizer — analyzes tool call efficiency across runs.

Loads historical tool call records from ``ToolCallStore``, computes
per-tool statistics, detects redundant calls (same tool + same args
within a single run), and generates actionable optimization suggestions.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from vaig.core.tool_call_store import ToolCallStore

logger = logging.getLogger(__name__)


# ── Suggestion thresholds ─────────────────────────────────────
FAILURE_RATE_THRESHOLD = 0.8
MIN_CALLS_FOR_ANALYSIS = 3
SLOW_TOOL_DURATION_S = 10.0
LOW_CACHE_HIT_RATE = 0.1
MIN_CALLS_FOR_CACHE_ANALYSIS = 10
REDUNDANT_CALL_THRESHOLD = 3


# ── Data models ───────────────────────────────────────────────


@dataclass(slots=True)
class ToolStats:
    """Aggregated statistics for a single tool across analyzed runs."""

    call_count: int
    failure_count: int
    failure_rate: float  # 0.0–1.0
    avg_duration_s: float
    max_duration_s: float
    cache_hit_count: int
    cache_hit_rate: float  # 0.0–1.0
    unique_arg_combos: int
    common_errors: list[str] = field(default_factory=list)  # top 3


@dataclass(slots=True)
class RedundantCall:
    """A tool invocation that was repeated with identical arguments in a run."""

    tool_name: str
    args_hash: str
    count: int  # number of times called with same args in same run
    run_id: str


@dataclass(slots=True)
class ToolInsights:
    """Full analysis result across multiple runs."""

    total_runs: int
    total_calls: int
    total_duration_s: float
    avg_calls_per_run: float
    tools: dict[str, ToolStats] = field(default_factory=dict)
    redundant_calls: list[RedundantCall] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


# ── Optimizer ─────────────────────────────────────────────────


def _hash_args(tool_name: str, tool_args: dict[str, Any]) -> str:
    """Deterministic hash of tool name + arguments for dedup detection."""
    try:
        canonical = json.dumps(
            {"tool": tool_name, "args": tool_args},
            sort_keys=True,
            default=str,
        )
    except (TypeError, ValueError):
        canonical = f"{tool_name}:{tool_args!r}"
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


class ToolCallOptimizer:
    """Analyzes tool call history and suggests optimizations."""

    def __init__(self, store: ToolCallStore) -> None:
        self._store = store

    def analyze(self, last_n_runs: int = 50, date_from: datetime | None = None) -> ToolInsights:
        """Load last *N* runs, compute per-tool stats, detect redundancies.

        Args:
            last_n_runs: Maximum number of recent runs to include.
            date_from: When provided, only include runs from this date
                onward (based on the run's date directory name).

        Returns:
            A :class:`ToolInsights` with per-tool stats, redundant calls,
            and optimization suggestions.
        """
        # Runs are ordered lexicographically by date directory name
        # (YYYY-MM-DD), which is chronological at the day level but not
        # necessarily within a single day.
        runs = self._store.list_runs()

        # ── Optional lookback filter (apply BEFORE slicing) ───
        if date_from is not None:
            date_from = date_from.replace(hour=0, minute=0, second=0, microsecond=0)
            runs = [(rid, d) for rid, d in runs if d >= date_from]

        selected_runs = runs[-last_n_runs:]

        if not selected_runs:
            insights = ToolInsights(
                total_runs=0,
                total_calls=0,
                total_duration_s=0.0,
                avg_calls_per_run=0.0,
            )
            insights.suggestions = self.suggest(insights)
            return insights

        # Collect all records from selected runs
        all_records: list[dict[str, Any]] = []

        for run_id, _date in selected_runs:
            records = self._store.read_records(run_id=run_id)
            all_records.extend(records)

        total_runs = len(selected_runs)
        total_calls = len(all_records)
        total_duration = sum(
            float(r.get("duration_s", 0.0)) for r in all_records
        )

        # ── Per-tool aggregation ──────────────────────────────
        tool_records: dict[str, list[dict[str, Any]]] = {}
        for rec in all_records:
            name = rec.get("tool_name", "unknown")
            tool_records.setdefault(name, []).append(rec)

        tools: dict[str, ToolStats] = {}
        for tool_name, records in tool_records.items():
            count = len(records)
            failures = [r for r in records if r.get("error")]
            durations = [float(r.get("duration_s", 0.0)) for r in records]
            cached = [r for r in records if r.get("cached")]

            # Unique argument combos
            arg_hashes: set[str] = set()
            for r in records:
                args = r.get("tool_args", {})
                if isinstance(args, dict):
                    arg_hashes.add(_hash_args(tool_name, args))
                else:
                    arg_hashes.add(_hash_args(tool_name, {}))

            # Common errors — top 3
            error_messages = [
                r.get("error_message", "")
                for r in failures
                if r.get("error_message")
            ]
            common_errors = [
                msg for msg, _count in Counter(error_messages).most_common(3)
            ]

            tools[tool_name] = ToolStats(
                call_count=count,
                failure_count=len(failures),
                failure_rate=len(failures) / count if count else 0.0,
                avg_duration_s=sum(durations) / count if count else 0.0,
                max_duration_s=max(durations) if durations else 0.0,
                cache_hit_count=len(cached),
                cache_hit_rate=len(cached) / count if count else 0.0,
                unique_arg_combos=len(arg_hashes),
                common_errors=common_errors,
            )

        # ── Redundant call detection ──────────────────────────
        redundant_calls = self._detect_redundant_calls(all_records)

        avg_calls = total_calls / total_runs if total_runs else 0.0

        insights = ToolInsights(
            total_runs=total_runs,
            total_calls=total_calls,
            total_duration_s=round(total_duration, 2),
            avg_calls_per_run=round(avg_calls, 1),
            tools=tools,
            redundant_calls=redundant_calls,
        )
        insights.suggestions = self.suggest(insights)
        return insights

    def suggest(self, insights: ToolInsights) -> list[str]:
        """Generate rule-based suggestions from analysis stats.

        Rules:
        - Tool with >80% failure rate → verify detection/configuration
        - Tool with avg_duration > 10s → consider caching or timeout
        - Redundant calls > 3 → verify checklist
        - Cache hit rate < 10% with high call count → enable caching
        - No data → prompt to collect data
        """
        suggestions: list[str] = []

        if insights.total_calls == 0:
            suggestions.append(
                "No tool call data found. Run some analyses first to collect data."
            )
            return suggestions

        for name, stats in insights.tools.items():
            # High failure rate
            if stats.failure_rate > FAILURE_RATE_THRESHOLD and stats.call_count >= MIN_CALLS_FOR_ANALYSIS:
                pct = round(stats.failure_rate * 100)
                suggestions.append(
                    f"{name} fails {pct}% of the time "
                    f"({stats.failure_count}/{stats.call_count}) "
                    f"— verify detection/configuration"
                )

            # Slow tool
            if stats.avg_duration_s > SLOW_TOOL_DURATION_S:
                avg = round(stats.avg_duration_s, 1)
                suggestions.append(
                    f"{name} averages {avg}s per call "
                    f"— consider caching or timeout tuning"
                )

            # Low cache hit rate with high volume
            if (
                stats.cache_hit_rate < LOW_CACHE_HIT_RATE
                and stats.call_count >= MIN_CALLS_FOR_CACHE_ANALYSIS
                and stats.cache_hit_count == 0
            ):
                suggestions.append(
                    f"{name} has 0% cache hit rate across "
                    f"{stats.call_count} calls — enable caching"
                )

        # Redundant calls
        for rc in insights.redundant_calls:
            if rc.count > REDUNDANT_CALL_THRESHOLD:
                suggestions.append(
                    f"{rc.tool_name} called {rc.count} times with "
                    f"same args in run {rc.run_id} — verify checklist"
                )

        return suggestions

    @staticmethod
    def _detect_redundant_calls(
        records: list[dict[str, Any]],
    ) -> list[RedundantCall]:
        """Find tool calls repeated with identical args within the same run."""
        # Key: (run_id, tool_name, args_hash) → count
        call_counts: dict[tuple[str, str, str], int] = {}

        for rec in records:
            run_id = rec.get("run_id", "unknown")
            tool_name = rec.get("tool_name", "unknown")
            args = rec.get("tool_args", {})
            if not isinstance(args, dict):
                args = {}
            h = _hash_args(tool_name, args)
            key = (run_id, tool_name, h)
            call_counts[key] = call_counts.get(key, 0) + 1

        redundant: list[RedundantCall] = []
        for (run_id, tool_name, h), count in call_counts.items():
            if count > 1:
                redundant.append(
                    RedundantCall(
                        tool_name=tool_name,
                        args_hash=h,
                        count=count,
                        run_id=run_id,
                    )
                )

        # Sort by count descending for readability
        redundant.sort(key=lambda r: r.count, reverse=True)
        return redundant

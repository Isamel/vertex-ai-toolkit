"""SPEC-V2-AUDIT-13 — MemoryRecallMixin acceptance tests.

Acceptance criteria:
1. Empty PatternMemoryStore → agents behave identically (byte-for-byte prompt equality).
2. Populated store → system instruction contains '## Prior-run memory' block.
3. Budget enforced: recalls exceeding recall_budget_tokens are truncated (tail dropped).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vaig.agents.mixins import MemoryRecallMixin
from vaig.core.memory.models import PatternEntry, RecalledPattern


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_entry(title: str, service: str = "", occurrences: int = 1) -> PatternEntry:
    now = datetime(2025, 4, 12, 14, 22, tzinfo=UTC)
    return PatternEntry(
        fingerprint="abc123def456abcd",
        first_seen=now,
        last_seen=now,
        occurrences=occurrences,
        title=title,
        service=service,
    )


class _ConcreteRecallMixin(MemoryRecallMixin):
    """Minimal concrete subclass for testing."""
    pass


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestMemoryRecallMixinEmpty:
    """AC-1: empty store → base instruction returned unchanged."""

    def test_empty_store_returns_base_unchanged(self, tmp_path: Any) -> None:
        mixin = _ConcreteRecallMixin()

        with patch("vaig.agents.mixins.get_settings") as mock_settings:
            cfg = MagicMock()
            cfg.memory.enabled = True
            cfg.memory.store_path = str(tmp_path)
            cfg.memory.recall_budget_tokens = 800
            mock_settings.return_value = cfg

            base = "You are a service health gatherer."
            result = mixin._augment_system_instruction(base, "istio-cni-node")

        assert result == base, "Empty store must return base instruction unchanged."

    def test_memory_disabled_returns_base_unchanged(self, tmp_path: Any) -> None:
        mixin = _ConcreteRecallMixin()

        with patch("vaig.agents.mixins.get_settings") as mock_settings:
            cfg = MagicMock()
            cfg.memory.enabled = False
            cfg.memory.store_path = str(tmp_path)
            cfg.memory.recall_budget_tokens = 800
            mock_settings.return_value = cfg

            base = "You are a service health gatherer."
            result = mixin._augment_system_instruction(base, "istio-cni-node")

        assert result == base

    def test_recall_budget_zero_returns_base_unchanged(self, tmp_path: Any) -> None:
        mixin = _ConcreteRecallMixin()

        with patch("vaig.agents.mixins.get_settings") as mock_settings:
            cfg = MagicMock()
            cfg.memory.enabled = True
            cfg.memory.store_path = str(tmp_path)
            cfg.memory.recall_budget_tokens = 0
            mock_settings.return_value = cfg

            base = "Base instruction."
            result = mixin._augment_system_instruction(base, "anything")

        assert result == base


class TestMemoryRecallMixinPopulated:
    """AC-2: populated store → ## Prior-run memory block in result."""

    def test_populated_store_injects_recall_block(self, tmp_path: Any) -> None:
        mixin = _ConcreteRecallMixin()

        recall = RecalledPattern.from_entry(
            _make_entry("istio-cni-node readiness 503", service="prd-gke-a"),
            resolution="GKE release-channel rollback to v1.29.5-gke.1200",
            fix_outcome="CONFIRMED",
        )

        with patch.object(mixin, "_recall_patterns", return_value=[recall]):
            with patch("vaig.agents.mixins.get_settings") as mock_settings:
                cfg = MagicMock()
                cfg.memory.recall_budget_tokens = 800
                mock_settings.return_value = cfg

                base = "You are a health analyzer."
                result = mixin._augment_system_instruction(base, "istio-cni-node")

        assert "## Prior-run memory" in result
        assert base in result
        assert "istio-cni-node readiness 503" in result

    def test_recall_block_contains_resolution(self, tmp_path: Any) -> None:
        mixin = _ConcreteRecallMixin()

        recall = RecalledPattern.from_entry(
            _make_entry("istio-cni unhealthy", service="prod"),
            resolution="Rollback to v1.29.5",
            fix_outcome="CONFIRMED",
        )

        with patch.object(mixin, "_recall_patterns", return_value=[recall]):
            with patch("vaig.agents.mixins.get_settings") as mock_settings:
                cfg = MagicMock()
                cfg.memory.recall_budget_tokens = 800
                mock_settings.return_value = cfg

                result = mixin._augment_system_instruction("base", "istio")

        assert "Rollback to v1.29.5" in result
        assert "CONFIRMED" in result


class TestMemoryRecallMixinBudget:
    """AC-3: budget enforcement — tail entries dropped when budget exceeded."""

    def test_large_recalls_truncated_to_budget(self) -> None:
        mixin = _ConcreteRecallMixin()

        # Create many recalls with long titles
        recalls = [
            RecalledPattern.from_entry(_make_entry(f"Finding number {i} " + "x" * 200))
            for i in range(10)
        ]

        # Very small budget (50 tokens ≈ 200 chars)
        block = mixin._format_recall_block(recalls, budget_tokens=50)

        # Should not contain all 10 entries
        count = block.count("Finding number")
        assert count < 10, f"Expected truncation but got {count} entries."
        assert count >= 1, "At least one entry should be included."

    def test_normal_budget_includes_all_recalls(self) -> None:
        mixin = _ConcreteRecallMixin()

        recalls = [
            RecalledPattern.from_entry(_make_entry(f"Short finding {i}"))
            for i in range(3)
        ]

        block = mixin._format_recall_block(recalls, budget_tokens=800)

        assert block.count("Short finding") == 3


class TestRecalledPatternModel:
    """Unit tests for RecalledPattern model and from_entry factory."""

    def test_from_entry_maps_fields_correctly(self) -> None:
        entry = _make_entry("cache overflow", service="payment-svc", occurrences=3)
        recalled = RecalledPattern.from_entry(entry, resolution="Restart pod", fix_outcome="resolved")

        assert recalled.title == "cache overflow"
        assert recalled.cluster == "payment-svc"
        assert recalled.resolution == "Restart pod"
        assert recalled.fix_outcome == "resolved"
        assert recalled.occurrences == 3

    def test_from_entry_empty_service_maps_to_empty_cluster(self) -> None:
        entry = _make_entry("some finding", service="")
        recalled = RecalledPattern.from_entry(entry)
        assert recalled.cluster == ""

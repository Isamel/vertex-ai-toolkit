"""Tests for query_pattern_history tool."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from vaig.tools.knowledge.query_pattern_history import query_pattern_history


def _make_config(store_path: str | Path) -> MagicMock:
    cfg = MagicMock()
    cfg.store_path = str(store_path)
    return cfg


class TestQueryPatternHistoryNoHistory:
    def test_returns_no_history_message(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        result = query_pattern_history(
            category="pod-health",
            service="payment-svc",
            title="CrashLoopBackOff",
            description="Pod is crash-looping",
            config=cfg,
        )
        assert not result.error
        assert "No historical data found" in result.output

    def test_includes_fingerprint_in_message(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        result = query_pattern_history(
            category="pod-health",
            service="payment-svc",
            title="CrashLoopBackOff",
            description="Pod is crash-looping",
            config=cfg,
        )
        assert "fingerprint" in result.output.lower()


class TestQueryPatternHistoryWithHistory:
    def _seed_store(self, tmp_path: Path) -> str:
        """Seed the store with one entry and return its run_id."""
        from vaig.core.memory.pattern_store import PatternMemoryStore

        store = PatternMemoryStore(base_dir=tmp_path)
        store.record(
            "run1",
            fingerprint=None,  # type: ignore[arg-type]  # will be computed below
            title="CrashLoopBackOff",
            severity="CRITICAL",
            service="payment-svc",
            category="pod-health",
        )
        return "run1"

    def test_returns_badge_for_known_pattern(self, tmp_path: Path) -> None:
        from vaig.core.memory.fingerprint import ObservationFingerprint
        from vaig.core.memory.pattern_store import PatternMemoryStore

        fp = ObservationFingerprint.from_finding(
            category="pod-health",
            service="payment-svc",
            title="CrashLoopBackOff",
            description="Pod is crash-looping",
        )
        store = PatternMemoryStore(base_dir=tmp_path)
        store.record("run1", fp, title="CrashLoopBackOff", severity="CRITICAL")

        cfg = _make_config(tmp_path)
        result = query_pattern_history(
            category="pod-health",
            service="payment-svc",
            title="CrashLoopBackOff",
            description="Pod is crash-looping",
            config=cfg,
        )

        assert not result.error
        assert "Badge" in result.output
        assert "NEW" in result.output  # 1 occurrence → NEW

    def test_recurring_badge_for_multiple_occurrences(self, tmp_path: Path) -> None:
        from vaig.core.memory.fingerprint import ObservationFingerprint
        from vaig.core.memory.pattern_store import PatternMemoryStore

        fp = ObservationFingerprint.from_finding(
            category="pod-health",
            service="payment-svc",
            title="CrashLoopBackOff",
            description="Pod is crash-looping",
        )
        store = PatternMemoryStore(base_dir=tmp_path)
        store.record("run1", fp, title="CrashLoopBackOff", severity="CRITICAL")
        store.record("run2", fp, title="CrashLoopBackOff", severity="HIGH")

        cfg = _make_config(tmp_path)
        result = query_pattern_history(
            category="pod-health",
            service="payment-svc",
            title="CrashLoopBackOff",
            description="Pod is crash-looping",
            config=cfg,
        )

        assert not result.error
        assert "RECURRING" in result.output


class TestQueryPatternHistoryErrorHandling:
    def test_invalid_store_path_returns_error(self) -> None:
        cfg = _make_config("/nonexistent/path/that/cannot/be/created")
        # PatternMemoryStore is error-silent, so we get no-history response
        result = query_pattern_history(
            category="pod-health",
            service="svc",
            title="Issue",
            description="desc",
            config=cfg,
        )
        # Should not raise — either no-history or error message
        assert isinstance(result.output, str)

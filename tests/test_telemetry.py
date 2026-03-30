"""Tests for TelemetryEvent, TelemetryCollector, and singleton lifecycle."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from vaig.core.telemetry import (
    TelemetryCollector,
    TelemetryEvent,
    get_telemetry_collector,
    reset_telemetry_collector,
)

# ══════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════
# db_path and collector are provided by conftest.py


@pytest.fixture()
def disabled_collector(db_path: Path) -> TelemetryCollector:
    """Create a disabled collector."""
    c = TelemetryCollector(db_path=db_path, enabled=False, buffer_size=5)
    yield c
    c.close()


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    """Ensure the global singleton is reset between tests."""
    reset_telemetry_collector()
    yield
    reset_telemetry_collector()


# ══════════════════════════════════════════════════════════════
# TelemetryEvent
# ══════════════════════════════════════════════════════════════


class TestTelemetryEvent:
    """Tests for the TelemetryEvent dataclass."""

    def test_to_row_returns_correct_tuple(self) -> None:
        event = TelemetryEvent(
            event_type="tool_call",
            event_name="get_pods",
            session_id="sess-123",
            timestamp="2025-01-01T00:00:00+00:00",
            duration_ms=42.5,
            metadata_json='{"ns": "default"}',
            tokens_in=100,
            tokens_out=200,
            cost_usd=0.005,
            model="gemini-2.5-pro",
            tool_name="get_pods",
            error_type="",
            error_message="",
        )
        row = event.to_row()

        assert isinstance(row, tuple)
        assert len(row) == 13
        assert row == (
            "tool_call",
            "get_pods",
            "sess-123",
            "2025-01-01T00:00:00+00:00",
            42.5,
            '{"ns": "default"}',
            100,
            200,
            0.005,
            "gemini-2.5-pro",
            "get_pods",
            "",
            "",
        )

    def test_defaults_are_applied(self) -> None:
        event = TelemetryEvent(event_type="cli_command", event_name="ask")
        assert event.session_id == ""
        assert event.duration_ms == 0.0
        assert event.tokens_in == 0
        assert event.tokens_out == 0
        assert event.cost_usd == 0.0
        assert event.model == ""
        assert event.tool_name == ""
        assert event.error_type == ""
        assert event.error_message == ""
        # timestamp should be auto-generated
        assert event.timestamp != ""

    def test_timestamp_auto_generated(self) -> None:
        before = datetime.now(UTC).isoformat()
        event = TelemetryEvent(event_type="test", event_name="check")
        after = datetime.now(UTC).isoformat()
        assert before <= event.timestamp <= after


# ══════════════════════════════════════════════════════════════
# TelemetryCollector — Disabled mode
# ══════════════════════════════════════════════════════════════


class TestDisabledCollector:
    """All emit methods are no-ops when disabled."""

    def test_emit_tool_call_is_noop(self, disabled_collector: TelemetryCollector) -> None:
        disabled_collector.emit_tool_call("get_pods", duration_ms=10.0)
        disabled_collector.flush()
        # No DB should be created at all
        assert disabled_collector._conn is None

    def test_emit_api_call_is_noop(self, disabled_collector: TelemetryCollector) -> None:
        disabled_collector.emit_api_call("gemini-2.5-pro", tokens_in=100, tokens_out=50)
        disabled_collector.flush()
        assert disabled_collector._conn is None

    def test_emit_cli_command_is_noop(self, disabled_collector: TelemetryCollector) -> None:
        disabled_collector.emit_cli_command("ask")
        disabled_collector.flush()
        assert disabled_collector._conn is None

    def test_emit_skill_use_is_noop(self, disabled_collector: TelemetryCollector) -> None:
        disabled_collector.emit_skill_use("rca")
        disabled_collector.flush()
        assert disabled_collector._conn is None

    def test_emit_error_is_noop(self, disabled_collector: TelemetryCollector) -> None:
        disabled_collector.emit_error("ValueError", "bad input")
        disabled_collector.flush()
        assert disabled_collector._conn is None

    def test_emit_generic_is_noop(self, disabled_collector: TelemetryCollector) -> None:
        disabled_collector.emit("custom", "event")
        disabled_collector.flush()
        assert disabled_collector._conn is None

    def test_query_returns_empty_when_disabled(self, disabled_collector: TelemetryCollector) -> None:
        # Even query_events should work (returns []) — DB won't be created
        # because no events were buffered, but flush is called which is safe
        disabled_collector.emit("test", "noop")
        result = disabled_collector.query_events()
        assert result == []

    def test_get_summary_returns_empty_when_disabled(self, disabled_collector: TelemetryCollector) -> None:
        disabled_collector.emit("test", "noop")
        summary = disabled_collector.get_summary()
        assert summary["total_events"] == 0


# ══════════════════════════════════════════════════════════════
# TelemetryCollector — Buffer & Flush
# ══════════════════════════════════════════════════════════════


class TestBufferAndFlush:
    """Buffer accumulation and flush behavior."""

    def test_buffer_does_not_write_before_threshold(self, collector: TelemetryCollector) -> None:
        # buffer_size=5, emit 4 events → should stay in buffer
        for i in range(4):
            collector.emit_tool_call(f"tool_{i}")
        # No DB writes yet — connection might not even be open
        assert len(collector._buffer) == 4

    def test_buffer_auto_flushes_at_threshold(self, collector: TelemetryCollector, db_path: Path) -> None:
        # buffer_size=5, emit 5 events → should auto-flush
        for i in range(5):
            collector.emit_tool_call(f"tool_{i}")

        # Buffer should be empty after auto-flush
        assert len(collector._buffer) == 0

        # Verify data is in the DB
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM telemetry_events").fetchone()[0]
        conn.close()
        assert count == 5

    def test_manual_flush_writes_to_db(self, collector: TelemetryCollector, db_path: Path) -> None:
        collector.emit_tool_call("get_pods", duration_ms=15.0)
        collector.emit_api_call("gemini-2.5-pro", tokens_in=100, tokens_out=50, cost_usd=0.001)
        collector.flush()

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM telemetry_events ORDER BY id").fetchall()
        conn.close()

        assert len(rows) == 2
        assert rows[0]["event_type"] == "tool_call"
        assert rows[0]["event_name"] == "get_pods"
        assert rows[0]["duration_ms"] == 15.0
        assert rows[1]["event_type"] == "api_call"
        assert rows[1]["tokens_in"] == 100
        assert rows[1]["tokens_out"] == 50
        assert rows[1]["cost_usd"] == 0.001

    def test_flush_is_idempotent(self, collector: TelemetryCollector) -> None:
        collector.emit_tool_call("t1")
        collector.flush()
        collector.flush()  # Second flush should be a no-op
        events = collector.query_events()
        assert len(events) == 1

    def test_close_flushes_remaining(self, db_path: Path) -> None:
        c = TelemetryCollector(db_path=db_path, enabled=True, buffer_size=100)
        c.emit_tool_call("tool_a")
        c.emit_tool_call("tool_b")
        c.close()

        # Verify data was flushed on close
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM telemetry_events").fetchone()[0]
        conn.close()
        assert count == 2


# ══════════════════════════════════════════════════════════════
# TelemetryCollector — Fire-and-forget
# ══════════════════════════════════════════════════════════════


class TestFireAndForget:
    """Emit methods must never raise, even with broken state."""

    def test_emit_with_broken_db_path_does_not_raise(self, tmp_path: Path) -> None:
        # Point to a path that can't be created
        bad_path = tmp_path / "nonexistent" / "deep" / "path" / "test.db"
        c = TelemetryCollector(db_path=bad_path, enabled=True, buffer_size=1)
        # This should NOT raise — fire-and-forget
        c.emit_tool_call("get_pods")
        # No exception means success

    def test_emit_after_close_does_not_raise(self, db_path: Path) -> None:
        c = TelemetryCollector(db_path=db_path, enabled=True, buffer_size=1)
        c.close()
        # Emitting after close should be silent
        c.emit_tool_call("tool_after_close")
        # No exception

    def test_flush_after_close_does_not_raise(self, db_path: Path) -> None:
        c = TelemetryCollector(db_path=db_path, enabled=True, buffer_size=100)
        c.emit_tool_call("t1")
        c.close()
        c.flush()  # Should not raise


# ══════════════════════════════════════════════════════════════
# TelemetryCollector — Thread safety
# ══════════════════════════════════════════════════════════════


class TestThreadSafety:
    """Concurrent emits from multiple threads."""

    def test_concurrent_emits(self, db_path: Path) -> None:
        c = TelemetryCollector(db_path=db_path, enabled=True, buffer_size=10)
        errors: list[Exception] = []

        def emit_batch(thread_id: int) -> None:
            try:
                for i in range(20):
                    c.emit_tool_call(f"thread_{thread_id}_tool_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=emit_batch, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        c.flush()
        c.close()

        assert errors == [], f"Thread errors: {errors}"

        # All 100 events (5 threads × 20 each) should be persisted
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM telemetry_events").fetchone()[0]
        conn.close()
        assert count == 100


# ══════════════════════════════════════════════════════════════
# TelemetryCollector — Session ID
# ══════════════════════════════════════════════════════════════


class TestSessionId:
    """set_session_id propagates to emitted events."""

    def test_session_id_propagates(self, collector: TelemetryCollector) -> None:
        collector.set_session_id("session-abc-123")
        collector.emit_tool_call("get_pods")
        collector.flush()

        events = collector.query_events()
        assert len(events) == 1
        assert events[0]["session_id"] == "session-abc-123"

    def test_session_id_changes(self, collector: TelemetryCollector) -> None:
        collector.set_session_id("s1")
        collector.emit_tool_call("t1")
        collector.set_session_id("s2")
        collector.emit_tool_call("t2")
        collector.flush()

        events = collector.query_events()
        # Ordered by timestamp DESC, so s2 first
        names = {e["session_id"]: e["event_name"] for e in events}
        assert names["s1"] == "t1"
        assert names["s2"] == "t2"


# ══════════════════════════════════════════════════════════════
# TelemetryCollector — Query
# ══════════════════════════════════════════════════════════════


class TestQueryEvents:
    """query_events with filters."""

    def test_query_all(self, collector: TelemetryCollector) -> None:
        collector.emit_tool_call("t1")
        collector.emit_api_call("model-a", tokens_in=10, tokens_out=5)
        collector.emit_error("ValueError", "bad")
        collector.flush()

        events = collector.query_events()
        assert len(events) == 3

    def test_query_by_type(self, collector: TelemetryCollector) -> None:
        collector.emit_tool_call("t1")
        collector.emit_api_call("model-a", tokens_in=10, tokens_out=5)
        collector.emit_tool_call("t2")
        collector.flush()

        events = collector.query_events(event_type="tool_call")
        assert len(events) == 2
        assert all(e["event_type"] == "tool_call" for e in events)

    def test_query_by_since(self, collector: TelemetryCollector) -> None:
        # Emit an event with an old timestamp
        old_event = TelemetryEvent(
            event_type="tool_call",
            event_name="old_tool",
            timestamp="2020-01-01T00:00:00+00:00",
        )
        collector._append(old_event)
        collector.emit_tool_call("new_tool")
        collector.flush()

        events = collector.query_events(since="2024-01-01T00:00:00+00:00")
        assert len(events) == 1
        assert events[0]["event_name"] == "new_tool"

    def test_query_by_until(self, collector: TelemetryCollector) -> None:
        old_event = TelemetryEvent(
            event_type="tool_call",
            event_name="old_tool",
            timestamp="2020-01-01T00:00:00+00:00",
        )
        collector._append(old_event)
        collector.emit_tool_call("new_tool")
        collector.flush()

        events = collector.query_events(until="2021-01-01T00:00:00+00:00")
        assert len(events) == 1
        assert events[0]["event_name"] == "old_tool"

    def test_query_with_limit(self, collector: TelemetryCollector) -> None:
        for i in range(10):
            collector.emit_tool_call(f"tool_{i}")
        collector.flush()

        events = collector.query_events(limit=3)
        assert len(events) == 3

    def test_query_combined_filters(self, collector: TelemetryCollector) -> None:
        collector.emit_tool_call("recent_tool")
        collector.emit_api_call("model-a", tokens_in=10, tokens_out=5)
        collector.flush()

        events = collector.query_events(
            event_type="tool_call",
            since="2024-01-01T00:00:00+00:00",
        )
        assert len(events) == 1
        assert events[0]["event_name"] == "recent_tool"


# ══════════════════════════════════════════════════════════════
# TelemetryCollector — Summary
# ══════════════════════════════════════════════════════════════


class TestGetSummary:
    """get_summary returns correct aggregations."""

    def test_summary_empty_db(self, collector: TelemetryCollector) -> None:
        summary = collector.get_summary()
        assert summary["total_events"] == 0
        assert summary["by_type"] == {}
        assert summary["top_tools"] == {}
        assert summary["api_calls"]["count"] == 0
        assert summary["error_count"] == 0

    def test_summary_with_events(self, collector: TelemetryCollector) -> None:
        collector.emit_tool_call("get_pods")
        collector.emit_tool_call("get_pods")
        collector.emit_tool_call("get_logs")
        collector.emit_api_call("gemini-2.5-pro", tokens_in=500, tokens_out=200, cost_usd=0.01)
        collector.emit_api_call("gemini-2.5-flash", tokens_in=100, tokens_out=50, cost_usd=0.001)
        collector.emit_error("TimeoutError", "timed out")
        collector.emit_cli_command("ask")
        collector.emit_skill_use("rca")
        collector.flush()

        summary = collector.get_summary()

        assert summary["total_events"] == 8
        assert summary["by_type"]["tool_call"] == 3
        assert summary["by_type"]["api_call"] == 2
        assert summary["by_type"]["error"] == 1
        assert summary["by_type"]["cli_command"] == 1
        assert summary["by_type"]["skill_use"] == 1

        assert summary["top_tools"]["get_pods"] == 2
        assert summary["top_tools"]["get_logs"] == 1

        assert summary["api_calls"]["count"] == 2
        assert summary["api_calls"]["total_tokens_in"] == 600
        assert summary["api_calls"]["total_tokens_out"] == 250
        assert summary["api_calls"]["total_cost_usd"] == pytest.approx(0.011)

        assert summary["error_count"] == 1

    def test_summary_with_time_filter(self, collector: TelemetryCollector) -> None:
        # Insert old event directly
        old_event = TelemetryEvent(
            event_type="tool_call",
            event_name="old_tool",
            tool_name="old_tool",
            timestamp="2020-01-01T00:00:00+00:00",
        )
        collector._append(old_event)
        collector.emit_tool_call("new_tool")
        collector.flush()

        summary = collector.get_summary(since="2024-01-01T00:00:00+00:00")
        assert summary["total_events"] == 1
        assert summary["top_tools"].get("old_tool") is None


# ══════════════════════════════════════════════════════════════
# TelemetryCollector — Clear events
# ══════════════════════════════════════════════════════════════


class TestClearEvents:
    """clear_events removes old data, keeps recent."""

    def test_clear_old_events(self, collector: TelemetryCollector) -> None:
        # Insert events with old timestamps
        for i in range(3):
            old_event = TelemetryEvent(
                event_type="tool_call",
                event_name=f"old_{i}",
                timestamp="2020-01-01T00:00:00+00:00",
            )
            collector._append(old_event)

        # Insert recent events
        collector.emit_tool_call("recent")
        collector.flush()

        deleted = collector.clear_events(older_than_days=30)
        assert deleted == 3

        # Only recent event should remain
        events = collector.query_events()
        assert len(events) == 1
        assert events[0]["event_name"] == "recent"

    def test_clear_keeps_recent(self, collector: TelemetryCollector) -> None:
        collector.emit_tool_call("fresh")
        collector.flush()

        deleted = collector.clear_events(older_than_days=1)
        assert deleted == 0

        events = collector.query_events()
        assert len(events) == 1


# ══════════════════════════════════════════════════════════════
# Singleton — get_telemetry_collector / reset_telemetry_collector
# ══════════════════════════════════════════════════════════════


class TestSingleton:
    """Singleton pattern via get/reset_telemetry_collector."""

    def test_get_returns_same_instance(self, tmp_path: Path) -> None:
        from vaig.core.config import Settings, TelemetryConfig

        settings = Settings(
            session={"db_path": str(tmp_path / "sessions.db")},
            telemetry=TelemetryConfig(enabled=True, buffer_size=10),
        )

        c1 = get_telemetry_collector(settings)
        c2 = get_telemetry_collector(settings)
        assert c1 is c2

    def test_reset_clears_singleton(self, tmp_path: Path) -> None:
        from vaig.core.config import Settings, TelemetryConfig

        settings = Settings(
            session={"db_path": str(tmp_path / "sessions.db")},
            telemetry=TelemetryConfig(enabled=True, buffer_size=10),
        )

        c1 = get_telemetry_collector(settings)
        reset_telemetry_collector()
        c2 = get_telemetry_collector(settings)
        assert c1 is not c2

    def test_env_var_disables_collection(self, tmp_path: Path) -> None:
        from vaig.core.config import Settings, TelemetryConfig

        settings = Settings(
            session={"db_path": str(tmp_path / "sessions.db")},
            telemetry=TelemetryConfig(enabled=True, buffer_size=10),
        )

        with patch.dict(os.environ, {"VAIG_TELEMETRY_ENABLED": "false"}):
            reset_telemetry_collector()
            c = get_telemetry_collector(settings)
            assert c._enabled is False

    def test_env_var_zero_disables_collection(self, tmp_path: Path) -> None:
        from vaig.core.config import Settings, TelemetryConfig

        settings = Settings(
            session={"db_path": str(tmp_path / "sessions.db")},
            telemetry=TelemetryConfig(enabled=True, buffer_size=10),
        )

        with patch.dict(os.environ, {"VAIG_TELEMETRY_ENABLED": "0"}):
            reset_telemetry_collector()
            c = get_telemetry_collector(settings)
            assert c._enabled is False


# ══════════════════════════════════════════════════════════════
# TelemetryCollector — Metadata handling
# ══════════════════════════════════════════════════════════════


class TestMetadata:
    """Metadata dict is serialized as JSON."""

    def test_metadata_serialized_as_json(self, collector: TelemetryCollector) -> None:
        meta = {"namespace": "kube-system", "cluster": "prod-1"}
        collector.emit_tool_call("get_pods", metadata=meta)
        collector.flush()

        events = collector.query_events()
        assert len(events) == 1
        stored = json.loads(events[0]["metadata"])
        assert stored["namespace"] == "kube-system"
        assert stored["cluster"] == "prod-1"

    def test_no_metadata_stores_empty_string(self, collector: TelemetryCollector) -> None:
        collector.emit_tool_call("get_pods")
        collector.flush()

        events = collector.query_events()
        assert events[0]["metadata"] == ""


# ══════════════════════════════════════════════════════════════
# TelemetryConfig
# ══════════════════════════════════════════════════════════════


class TestTelemetryConfig:
    """Tests for TelemetryConfig in Settings."""

    def test_default_config(self) -> None:
        from vaig.core.config import TelemetryConfig

        cfg = TelemetryConfig()
        assert cfg.enabled is True
        assert cfg.buffer_size == 50

    def test_config_in_settings(self) -> None:
        from vaig.core.config import Settings

        settings = Settings()
        assert settings.telemetry.enabled is True
        assert settings.telemetry.buffer_size == 50

    def test_config_override(self) -> None:
        from vaig.core.config import Settings, TelemetryConfig

        settings = Settings(telemetry=TelemetryConfig(enabled=False, buffer_size=100))
        assert settings.telemetry.enabled is False
        assert settings.telemetry.buffer_size == 100


# ══════════════════════════════════════════════════════════════
# Hook Integration Tests (Phase 2)
# ══════════════════════════════════════════════════════════════


class TestToolCallHook:
    """Test that _execute_single_tool emits tool_call telemetry."""

    def test_emit_on_success(self, monkeypatch: pytest.MonkeyPatch, db_path: Path) -> None:
        """Successful tool execution emits a tool_call event with timing."""
        from vaig.core.telemetry import TelemetryCollector

        # Set up singleton with a real collector
        collector = TelemetryCollector(db_path=db_path, enabled=True, buffer_size=1)
        monkeypatch.setattr("vaig.core.telemetry._collector", collector)
        monkeypatch.setattr(
            "vaig.core.telemetry.get_telemetry_collector", lambda *a, **kw: collector,
        )

        # Wire TelemetrySubscriber so events flow to collector
        from vaig.core.subscribers.telemetry_subscriber import TelemetrySubscriber

        subscriber = TelemetrySubscriber(collector)

        # Minimal tool and registry mocks
        from unittest.mock import MagicMock

        from vaig.agents.mixins import ToolLoopMixin
        from vaig.tools.base import ToolResult

        mock_tool = MagicMock()
        mock_tool.execute.return_value = ToolResult(output="ok", error=False)
        mock_tool.parameters = []  # Satisfy _pre_validate_tool_args schema check

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_tool

        mixin = ToolLoopMixin()
        result = mixin._execute_single_tool(mock_registry, "get_pods", {"namespace": "default"})

        assert result.output == "ok"
        assert result.error is False

        # Flush and verify
        subscriber.unsubscribe_all()
        collector.flush()
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT event_type, event_name FROM telemetry_events").fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0] == ("tool_call", "get_pods")

    def test_emit_on_unknown_tool(self, monkeypatch: pytest.MonkeyPatch, db_path: Path) -> None:
        """Unknown tool emits a tool_call event with error_type=UnknownTool."""
        from vaig.core.telemetry import TelemetryCollector

        collector = TelemetryCollector(db_path=db_path, enabled=True, buffer_size=1)
        monkeypatch.setattr("vaig.core.telemetry._collector", collector)
        monkeypatch.setattr(
            "vaig.core.telemetry.get_telemetry_collector", lambda *a, **kw: collector,
        )

        # Wire TelemetrySubscriber so events flow to collector
        from vaig.core.subscribers.telemetry_subscriber import TelemetrySubscriber

        subscriber = TelemetrySubscriber(collector)

        from unittest.mock import MagicMock

        from vaig.agents.mixins import ToolLoopMixin

        mock_registry = MagicMock()
        mock_registry.get.return_value = None
        mock_registry.list_tools.return_value = []

        mixin = ToolLoopMixin()
        result = mixin._execute_single_tool(mock_registry, "no_such_tool", {})

        assert result.error is True

        subscriber.unsubscribe_all()
        collector.flush()
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, error_type FROM telemetry_events",
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0] == ("tool_call", "UnknownTool")

    def test_emit_on_exception(self, monkeypatch: pytest.MonkeyPatch, db_path: Path) -> None:
        """Tool raising an exception still emits telemetry with error info."""
        from vaig.core.telemetry import TelemetryCollector

        collector = TelemetryCollector(db_path=db_path, enabled=True, buffer_size=1)
        monkeypatch.setattr("vaig.core.telemetry._collector", collector)
        monkeypatch.setattr(
            "vaig.core.telemetry.get_telemetry_collector", lambda *a, **kw: collector,
        )

        # Wire TelemetrySubscriber so events flow to collector
        from vaig.core.subscribers.telemetry_subscriber import TelemetrySubscriber

        subscriber = TelemetrySubscriber(collector)

        from unittest.mock import MagicMock

        from vaig.agents.mixins import ToolLoopMixin

        mock_tool = MagicMock()
        mock_tool.execute.side_effect = RuntimeError("connection timeout")
        mock_tool.parameters = []  # Satisfy _pre_validate_tool_args schema check

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_tool

        mixin = ToolLoopMixin()
        result = mixin._execute_single_tool(mock_registry, "bad_tool", {"x": 1})

        assert result.error is True
        assert "connection timeout" in result.output

        subscriber.unsubscribe_all()
        collector.flush()
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, error_type, error_msg FROM telemetry_events",
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] == "tool_call"
        assert rows[0][1] == "RuntimeError"
        assert "connection timeout" in rows[0][2]


class TestApiCallHook:
    """Test that CostTracker.record() emits api_call telemetry."""

    def test_emit_on_record(self, monkeypatch: pytest.MonkeyPatch, db_path: Path) -> None:
        """CostTracker.record() emits an api_call event."""
        from vaig.core.telemetry import TelemetryCollector

        collector = TelemetryCollector(db_path=db_path, enabled=True, buffer_size=1)
        monkeypatch.setattr("vaig.core.telemetry._collector", collector)
        monkeypatch.setattr(
            "vaig.core.telemetry.get_telemetry_collector", lambda *a, **kw: collector,
        )

        # Wire TelemetrySubscriber so events flow to collector
        from vaig.core.subscribers.telemetry_subscriber import TelemetrySubscriber

        subscriber = TelemetrySubscriber(collector)

        from vaig.core.cost_tracker import CostTracker

        tracker = CostTracker()
        rec = tracker.record("gemini-2.5-pro", 100, 200, thinking_tokens=50)

        assert rec.model_id == "gemini-2.5-pro"
        assert rec.prompt_tokens == 100

        subscriber.unsubscribe_all()
        collector.flush()
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, event_name, tokens_in, tokens_out FROM telemetry_events",
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] == "api_call"
        assert rows[0][1] == "gemini-2.5-pro"
        assert rows[0][2] == 100
        assert rows[0][3] == 200


class TestSessionLifecycleHook:
    """Test that SessionManager emits session_start/session_end telemetry."""

    def test_new_session_emits_start(self, monkeypatch: pytest.MonkeyPatch, db_path: Path) -> None:
        """SessionManager.new_session() emits session_start and sets session_id."""
        from vaig.core.telemetry import TelemetryCollector

        collector = TelemetryCollector(db_path=db_path, enabled=True, buffer_size=1)
        monkeypatch.setattr("vaig.core.telemetry._collector", collector)
        monkeypatch.setattr(
            "vaig.core.telemetry.get_telemetry_collector", lambda *a, **kw: collector,
        )

        # Wire TelemetrySubscriber so events flow to collector
        from vaig.core.subscribers.telemetry_subscriber import TelemetrySubscriber

        subscriber = TelemetrySubscriber(collector)

        from unittest.mock import MagicMock

        from vaig.session.manager import SessionManager

        mock_settings = MagicMock()
        mock_settings.models.default = "gemini-2.5-pro"
        mock_settings.db_path_resolved = str(db_path.parent / "sessions.db")

        manager = SessionManager(mock_settings)
        # Mock the store
        manager._store = MagicMock()
        manager._store.create_session.return_value = "test-session-123"

        session = manager.new_session("test-session", model="gemini-2.5-pro")

        assert session.id == "test-session-123"
        assert collector._session_id == "test-session-123"

        subscriber.unsubscribe_all()
        collector.flush()
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, event_name FROM telemetry_events",
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0] == ("session", "session_start")

    def test_close_emits_end(self, monkeypatch: pytest.MonkeyPatch, db_path: Path) -> None:
        """SessionManager.close() emits session_end and flushes."""
        from vaig.core.telemetry import TelemetryCollector

        collector = TelemetryCollector(db_path=db_path, enabled=True, buffer_size=50)
        monkeypatch.setattr("vaig.core.telemetry._collector", collector)
        monkeypatch.setattr(
            "vaig.core.telemetry.get_telemetry_collector", lambda *a, **kw: collector,
        )

        # Wire TelemetrySubscriber so events flow to collector
        from vaig.core.subscribers.telemetry_subscriber import TelemetrySubscriber

        subscriber = TelemetrySubscriber(collector)

        from unittest.mock import MagicMock

        from vaig.session.manager import SessionManager

        mock_settings = MagicMock()
        mock_settings.db_path_resolved = str(db_path.parent / "sessions.db")

        manager = SessionManager(mock_settings)
        manager._store = MagicMock()

        manager.close()

        subscriber.unsubscribe_all()

        # After close, the session_end event should have been flushed to DB
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, event_name FROM telemetry_events",
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0] == ("session", "session_end")


class TestSkillUseHook:
    """Test that SkillRegistry._register() emits skill_use telemetry."""

    def test_emit_on_register(self, monkeypatch: pytest.MonkeyPatch, db_path: Path) -> None:
        """Registering a skill emits a skill_use event."""
        from vaig.core.telemetry import TelemetryCollector

        collector = TelemetryCollector(db_path=db_path, enabled=True, buffer_size=1)
        monkeypatch.setattr("vaig.core.telemetry._collector", collector)
        monkeypatch.setattr(
            "vaig.core.telemetry.get_telemetry_collector", lambda *a, **kw: collector,
        )

        # Wire TelemetrySubscriber so events flow to collector
        from vaig.core.subscribers.telemetry_subscriber import TelemetrySubscriber

        subscriber = TelemetrySubscriber(collector)

        from unittest.mock import MagicMock

        from vaig.skills.registry import SkillRegistry

        mock_skill = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.name = "gke_diagnostics"
        mock_skill.get_metadata.return_value = mock_metadata

        registry = SkillRegistry.__new__(SkillRegistry)
        registry._skills = {}
        registry._metadata_cache = {}

        registry._register(mock_skill)

        assert "gke_diagnostics" in registry._skills

        subscriber.unsubscribe_all()
        collector.flush()
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, event_name FROM telemetry_events",
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0] == ("skill_use", "gke_diagnostics")


class TestCliCommandHook:
    """Test the track_command decorator."""

    def test_track_command_emits_event(self, monkeypatch: pytest.MonkeyPatch, db_path: Path) -> None:
        """track_command decorator emits cli_command event via EventBus."""
        from vaig.core.telemetry import TelemetryCollector

        collector = TelemetryCollector(db_path=db_path, enabled=True, buffer_size=1)
        monkeypatch.setattr("vaig.core.telemetry._collector", collector)
        monkeypatch.setattr(
            "vaig.core.telemetry.get_telemetry_collector", lambda *a, **kw: collector,
        )

        # Wire TelemetrySubscriber so events flow to collector
        from vaig.core.subscribers.telemetry_subscriber import TelemetrySubscriber

        subscriber = TelemetrySubscriber(collector)

        from vaig.cli.app import track_command

        @track_command
        def my_command() -> str:
            return "done"

        result = my_command()
        assert result == "done"

        subscriber.unsubscribe_all()
        collector.flush()
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, event_name, duration_ms FROM telemetry_events",
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] == "cli_command"
        assert rows[0][1] == "my_command"
        assert rows[0][2] >= 0  # duration should be non-negative

    def test_track_command_emits_even_on_exception(
        self, monkeypatch: pytest.MonkeyPatch, db_path: Path,
    ) -> None:
        """track_command still emits telemetry when the command raises."""
        from vaig.core.telemetry import TelemetryCollector

        collector = TelemetryCollector(db_path=db_path, enabled=True, buffer_size=1)
        monkeypatch.setattr("vaig.core.telemetry._collector", collector)
        monkeypatch.setattr(
            "vaig.core.telemetry.get_telemetry_collector", lambda *a, **kw: collector,
        )

        # Wire TelemetrySubscriber so events flow to collector
        from vaig.core.subscribers.telemetry_subscriber import TelemetrySubscriber

        subscriber = TelemetrySubscriber(collector)

        from vaig.cli.app import track_command

        @track_command
        def failing_command() -> None:
            msg = "boom"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="boom"):
            failing_command()

        subscriber.unsubscribe_all()
        collector.flush()
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, event_name FROM telemetry_events",
        ).fetchall()
        conn.close()

        # Event is emitted in finally block, so even on exception
        assert len(rows) == 1
        assert rows[0] == ("cli_command", "failing_command")


class TestErrorHook:
    """Test that error hooks don't break when telemetry is unavailable."""

    def test_error_emit_is_silent(self, monkeypatch: pytest.MonkeyPatch, db_path: Path) -> None:
        """Error hook emits an error event without breaking the caller."""
        from vaig.core.telemetry import TelemetryCollector

        collector = TelemetryCollector(db_path=db_path, enabled=True, buffer_size=1)
        monkeypatch.setattr("vaig.core.telemetry._collector", collector)
        monkeypatch.setattr(
            "vaig.core.telemetry.get_telemetry_collector", lambda *a, **kw: collector,
        )

        collector.emit_error("ValueError", "test error", metadata={"source": "test"})

        collector.flush()
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, event_name, error_type, error_msg FROM telemetry_events",
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] == "error"
        assert rows[0][1] == "ValueError"
        assert rows[0][2] == "ValueError"
        assert rows[0][3] == "test error"


class TestTelemetryNeverBreaksHost:
    """Verify that telemetry failures never propagate to host functions."""

    def test_tool_call_with_broken_collector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If the collector raises, _execute_single_tool still works normally."""
        def broken_collector(*args: object, **kwargs: object) -> None:
            msg = "DB is locked"
            raise OSError(msg)

        monkeypatch.setattr(
            "vaig.core.telemetry.get_telemetry_collector", broken_collector,
        )

        from unittest.mock import MagicMock

        from vaig.agents.mixins import ToolLoopMixin
        from vaig.tools.base import ToolResult

        mock_tool = MagicMock()
        mock_tool.execute.return_value = ToolResult(output="success", error=False)
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_tool

        mixin = ToolLoopMixin()
        result = mixin._execute_single_tool(mock_registry, "some_tool", {})

        # Tool still works despite telemetry failure
        assert result.output == "success"
        assert result.error is False

    def test_cost_tracker_with_broken_collector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If the collector raises, CostTracker.record() still works."""
        def broken_collector(*args: object, **kwargs: object) -> None:
            msg = "DB is locked"
            raise OSError(msg)

        monkeypatch.setattr(
            "vaig.core.telemetry.get_telemetry_collector", broken_collector,
        )

        from vaig.core.cost_tracker import CostTracker

        tracker = CostTracker()
        rec = tracker.record("gemini-2.5-pro", 50, 100)

        # CostTracker still works
        assert rec.model_id == "gemini-2.5-pro"
        assert tracker.total_cost >= 0


# ══════════════════════════════════════════════════════════════
# Async methods
# ══════════════════════════════════════════════════════════════


class TestAsyncFlush:
    """async_flush() drains the buffer via aiosqlite."""

    async def test_async_flush_writes_to_db(self, collector: TelemetryCollector, db_path: Path) -> None:
        collector.emit_tool_call("get_pods", duration_ms=15.0)
        collector.emit_api_call("gemini-2.5-pro", tokens_in=100, tokens_out=50, cost_usd=0.001)
        await collector.async_flush()

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM telemetry_events ORDER BY id").fetchall()
        conn.close()

        assert len(rows) == 2
        assert rows[0]["event_type"] == "tool_call"
        assert rows[0]["event_name"] == "get_pods"
        assert rows[0]["duration_ms"] == 15.0
        assert rows[1]["event_type"] == "api_call"
        assert rows[1]["tokens_in"] == 100

    async def test_async_flush_is_idempotent(self, collector: TelemetryCollector) -> None:
        collector.emit_tool_call("t1")
        await collector.async_flush()
        await collector.async_flush()  # Second flush is a no-op
        events = await collector.async_query_events()
        assert len(events) == 1

    async def test_async_flush_empty_buffer(self, collector: TelemetryCollector) -> None:
        # Should not raise on empty buffer
        await collector.async_flush()

    async def test_async_flush_after_close(self, db_path: Path) -> None:
        c = TelemetryCollector(db_path=db_path, enabled=True, buffer_size=100)
        c.emit_tool_call("t1")
        c.close()
        # async_flush after sync close — buffer was already flushed + closed flag set
        await c.async_flush()


class TestAsyncQueryEvents:
    """async_query_events() with filters."""

    async def test_async_query_all(self, collector: TelemetryCollector) -> None:
        collector.emit_tool_call("t1")
        collector.emit_api_call("model-a", tokens_in=10, tokens_out=5)
        collector.emit_error("ValueError", "bad")

        events = await collector.async_query_events()
        assert len(events) == 3

    async def test_async_query_by_type(self, collector: TelemetryCollector) -> None:
        collector.emit_tool_call("t1")
        collector.emit_api_call("model-a", tokens_in=10, tokens_out=5)
        collector.emit_tool_call("t2")

        events = await collector.async_query_events(event_type="tool_call")
        assert len(events) == 2
        assert all(e["event_type"] == "tool_call" for e in events)

    async def test_async_query_by_since(self, collector: TelemetryCollector) -> None:
        old_event = TelemetryEvent(
            event_type="tool_call",
            event_name="old_tool",
            timestamp="2020-01-01T00:00:00+00:00",
        )
        collector._append(old_event)
        collector.emit_tool_call("new_tool")

        events = await collector.async_query_events(since="2024-01-01T00:00:00+00:00")
        assert len(events) == 1
        assert events[0]["event_name"] == "new_tool"

    async def test_async_query_by_until(self, collector: TelemetryCollector) -> None:
        old_event = TelemetryEvent(
            event_type="tool_call",
            event_name="old_tool",
            timestamp="2020-01-01T00:00:00+00:00",
        )
        collector._append(old_event)
        collector.emit_tool_call("new_tool")

        events = await collector.async_query_events(until="2021-01-01T00:00:00+00:00")
        assert len(events) == 1
        assert events[0]["event_name"] == "old_tool"

    async def test_async_query_with_limit(self, collector: TelemetryCollector) -> None:
        for i in range(10):
            collector.emit_tool_call(f"tool_{i}")

        events = await collector.async_query_events(limit=3)
        assert len(events) == 3

    async def test_async_query_combined_filters(self, collector: TelemetryCollector) -> None:
        collector.emit_tool_call("recent_tool")
        collector.emit_api_call("model-a", tokens_in=10, tokens_out=5)

        events = await collector.async_query_events(
            event_type="tool_call",
            since="2024-01-01T00:00:00+00:00",
        )
        assert len(events) == 1
        assert events[0]["event_name"] == "recent_tool"


class TestAsyncGetSummary:
    """async_get_summary() returns correct aggregations."""

    async def test_async_summary_empty_db(self, collector: TelemetryCollector) -> None:
        summary = await collector.async_get_summary()
        assert summary["total_events"] == 0
        assert summary["by_type"] == {}
        assert summary["top_tools"] == {}
        assert summary["api_calls"]["count"] == 0
        assert summary["error_count"] == 0

    async def test_async_summary_with_events(self, collector: TelemetryCollector) -> None:
        collector.emit_tool_call("get_pods")
        collector.emit_tool_call("get_pods")
        collector.emit_tool_call("get_logs")
        collector.emit_api_call("gemini-2.5-pro", tokens_in=500, tokens_out=200, cost_usd=0.01)
        collector.emit_api_call("gemini-2.5-flash", tokens_in=100, tokens_out=50, cost_usd=0.001)
        collector.emit_error("TimeoutError", "timed out")
        collector.emit_cli_command("ask")
        collector.emit_skill_use("rca")

        summary = await collector.async_get_summary()

        assert summary["total_events"] == 8
        assert summary["by_type"]["tool_call"] == 3
        assert summary["by_type"]["api_call"] == 2
        assert summary["by_type"]["error"] == 1
        assert summary["by_type"]["cli_command"] == 1
        assert summary["by_type"]["skill_use"] == 1

        assert summary["top_tools"]["get_pods"] == 2
        assert summary["top_tools"]["get_logs"] == 1

        assert summary["api_calls"]["count"] == 2
        assert summary["api_calls"]["total_tokens_in"] == 600
        assert summary["api_calls"]["total_tokens_out"] == 250
        assert summary["api_calls"]["total_cost_usd"] == pytest.approx(0.011)

        assert summary["error_count"] == 1

    async def test_async_summary_with_time_filter(self, collector: TelemetryCollector) -> None:
        old_event = TelemetryEvent(
            event_type="tool_call",
            event_name="old_tool",
            tool_name="old_tool",
            timestamp="2020-01-01T00:00:00+00:00",
        )
        collector._append(old_event)
        collector.emit_tool_call("new_tool")

        summary = await collector.async_get_summary(since="2024-01-01T00:00:00+00:00")
        assert summary["total_events"] == 1
        assert summary["top_tools"].get("old_tool") is None


class TestAsyncClearEvents:
    """async_clear_events() removes old data, keeps recent."""

    async def test_async_clear_old_events(self, collector: TelemetryCollector) -> None:
        for i in range(3):
            old_event = TelemetryEvent(
                event_type="tool_call",
                event_name=f"old_{i}",
                timestamp="2020-01-01T00:00:00+00:00",
            )
            collector._append(old_event)
        collector.emit_tool_call("recent")
        await collector.async_flush()

        deleted = await collector.async_clear_events(older_than_days=30)
        assert deleted == 3

        events = await collector.async_query_events()
        assert len(events) == 1
        assert events[0]["event_name"] == "recent"

    async def test_async_clear_keeps_recent(self, collector: TelemetryCollector) -> None:
        collector.emit_tool_call("fresh")
        await collector.async_flush()

        deleted = await collector.async_clear_events(older_than_days=1)
        assert deleted == 0

        events = await collector.async_query_events()
        assert len(events) == 1


class TestAsyncClose:
    """async_close() flushes and tears down connections."""

    async def test_async_close_flushes_remaining(self, db_path: Path) -> None:
        c = TelemetryCollector(db_path=db_path, enabled=True, buffer_size=100)
        c.emit_tool_call("tool_a")
        c.emit_tool_call("tool_b")
        await c.async_close()

        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM telemetry_events").fetchone()[0]
        conn.close()
        assert count == 2
        assert c._aconn is None

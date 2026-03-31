"""Tests for the AuditSubscriber (vaig.core.subscribers.audit_subscriber).

Covers:
- Subscribes to all 7 event types
- ApiCalled → correct audit record fields
- Buffer flushes at configured size
- SessionEnded triggers immediate flush
- BigQuery failure → WARNING logged, Cloud Logging proceeds
- Cloud Logging failure → WARNING logged, BigQuery proceeds
- Identity fields present on every record
- unsubscribe_all() flushes remaining
- Lazy import guard → ImportError with install instructions
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vaig.core.event_bus import EventBus
from vaig.core.events import (
    ApiCalled,
    CliCommandTracked,
    ErrorOccurred,
    SessionEnded,
    SessionStarted,
    SkillUsed,
    ToolExecuted,
)
from vaig.core.subscribers.audit_subscriber import (
    AuditSubscriber,
    _require_audit_deps,
)

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_event_bus() -> None:
    """Reset EventBus singleton between tests."""
    EventBus.get().reset()


@pytest.fixture()
def mock_settings() -> MagicMock:
    """Create mock Settings with audit config."""
    settings = MagicMock()
    settings.audit.enabled = True
    settings.audit.bigquery_dataset = "test_dataset"
    settings.audit.bigquery_table = "test_table"
    settings.audit.cloud_logging_log_name = "test-audit"
    settings.audit.buffer_size = 20
    settings.audit.flush_interval_seconds = 30
    settings.gcp.project_id = "test-project"
    return settings


@pytest.fixture()
def mock_bq_client() -> MagicMock:
    """Create mock BigQuery client."""
    client = MagicMock()
    client.project = "test-project"
    client.insert_rows_json.return_value = []  # no errors
    return client


@pytest.fixture()
def mock_logging_client() -> MagicMock:
    """Create mock Cloud Logging client."""
    client = MagicMock()
    return client


@pytest.fixture()
def subscriber(
    mock_settings: MagicMock,
    mock_bq_client: MagicMock,
    mock_logging_client: MagicMock,
) -> AuditSubscriber:
    """Create an AuditSubscriber with mocked dependencies."""
    with patch("vaig.core.subscribers.audit_subscriber.resolve_identity") as mock_identity:
        mock_identity.return_value = ("testuser", "test@example.com", "testuser:test@example.com")
        with patch("vaig.core.subscribers.audit_subscriber.get_app_version", return_value="1.0.0"):
            sub = AuditSubscriber(
                mock_settings,
                bq_client=mock_bq_client,
                logging_client=mock_logging_client,
            )
    return sub


# ── Lazy import guard ────────────────────────────────────────


class TestRequireAuditDeps:
    """Tests for the _require_audit_deps guard."""

    def test_raises_with_install_message(self) -> None:
        """Raises ImportError with pip install instructions when deps missing."""
        with patch.dict("sys.modules", {"google.cloud.bigquery": None, "google.cloud.logging": None}):
            with pytest.raises(ImportError, match="pip install 'vertex-ai-toolkit\\[audit\\]'"):
                _require_audit_deps()


# ── Subscription tests ───────────────────────────────────────


class TestAuditSubscriberSubscriptions:
    """Tests for event subscriptions."""

    def test_subscribes_to_7_event_types(self, subscriber: AuditSubscriber) -> None:
        """AuditSubscriber registers 7 unsubscribe callbacks."""
        assert len(subscriber._unsubscribers) == 7

    def test_unsubscribe_all_clears_handlers(self, subscriber: AuditSubscriber) -> None:
        """unsubscribe_all() removes all handlers."""
        subscriber.unsubscribe_all()
        assert len(subscriber._unsubscribers) == 0


# ── Identity enrichment ──────────────────────────────────────


class TestAuditSubscriberIdentity:
    """Tests for identity fields on audit records."""

    def test_os_user_on_record(self, subscriber: AuditSubscriber) -> None:
        """Every audit record includes os_user."""
        bus = EventBus.get()
        bus.emit(CliCommandTracked(command_name="diagnose", duration_ms=100.0))

        # Trigger flush
        subscriber._flush()

        assert subscriber._os_user == "testuser"

    def test_gcp_user_on_record(self, subscriber: AuditSubscriber) -> None:
        """Every audit record includes gcp_user."""
        assert subscriber._gcp_user == "test@example.com"

    def test_app_version_on_record(self, subscriber: AuditSubscriber) -> None:
        """Every audit record includes app_version."""
        assert subscriber._app_version == "1.0.0"


# ── Event handler tests ──────────────────────────────────────


class TestAuditSubscriberApiCalled:
    """Tests for the ApiCalled event handler."""

    def test_api_called_record_fields(
        self,
        subscriber: AuditSubscriber,
        mock_bq_client: MagicMock,
    ) -> None:
        """ApiCalled produces a record with model, tokens, duration."""
        bus = EventBus.get()
        bus.emit(
            ApiCalled(
                model="gemini-2.5-pro",
                tokens_in=100,
                tokens_out=50,
                cost_usd=0.01,
                duration_ms=500.0,
                metadata=(("thinking_tokens", 25),),
            )
        )

        # Flush to trigger writes
        subscriber._flush()

        # Check the record
        assert len(mock_bq_client.insert_rows_json.call_args_list) == 1
        records = mock_bq_client.insert_rows_json.call_args[0][1]
        assert len(records) == 1
        record = records[0]
        assert record["event_type"] == "api.called"
        assert record["model"] == "gemini-2.5-pro"
        assert record["tokens_in"] == 100
        assert record["tokens_out"] == 50
        assert record["tokens_thinking"] == 25
        assert record["duration_ms"] == 500.0
        assert record["result"] == "success"
        assert record["os_user"] == "testuser"
        assert record["gcp_user"] == "test@example.com"
        assert record["app_version"] == "1.0.0"


class TestAuditSubscriberCliCommand:
    """Tests for the CliCommandTracked event handler."""

    def test_cli_command_record(
        self,
        subscriber: AuditSubscriber,
        mock_bq_client: MagicMock,
    ) -> None:
        """CliCommandTracked produces record with command name."""
        bus = EventBus.get()
        bus.emit(CliCommandTracked(command_name="diagnose", duration_ms=3000.0))
        subscriber._flush()

        records = mock_bq_client.insert_rows_json.call_args[0][1]
        record = records[0]
        assert record["event_type"] == "cli.command"
        assert record["command"] == "diagnose"
        assert record["duration_ms"] == 3000.0
        assert record["result"] == "success"


class TestAuditSubscriberToolExecuted:
    """Tests for the ToolExecuted event handler."""

    def test_tool_success(
        self,
        subscriber: AuditSubscriber,
        mock_bq_client: MagicMock,
    ) -> None:
        """Successful tool execution → result='success'."""
        bus = EventBus.get()
        bus.emit(ToolExecuted(tool_name="kubectl_get_pods", duration_ms=200.0))
        subscriber._flush()

        records = mock_bq_client.insert_rows_json.call_args[0][1]
        record = records[0]
        assert record["event_type"] == "tool.executed"
        assert record["command"] == "kubectl_get_pods"
        assert record["result"] == "success"
        assert record["error_message"] is None

    def test_tool_failure(
        self,
        subscriber: AuditSubscriber,
        mock_bq_client: MagicMock,
    ) -> None:
        """Failed tool execution → result='fail' with error message."""
        bus = EventBus.get()
        bus.emit(
            ToolExecuted(
                tool_name="kubectl_get_pods",
                duration_ms=200.0,
                error=True,
                error_message="Connection refused",
            )
        )
        subscriber._flush()

        records = mock_bq_client.insert_rows_json.call_args[0][1]
        record = records[0]
        assert record["result"] == "fail"
        assert record["error_message"] == "Connection refused"


class TestAuditSubscriberSkillUsed:
    """Tests for the SkillUsed event handler."""

    def test_skill_used_record(
        self,
        subscriber: AuditSubscriber,
        mock_bq_client: MagicMock,
    ) -> None:
        """SkillUsed produces record with skill name."""
        bus = EventBus.get()
        bus.emit(SkillUsed(skill_name="gke_diagnosis", duration_ms=100.0))
        subscriber._flush()

        records = mock_bq_client.insert_rows_json.call_args[0][1]
        record = records[0]
        assert record["event_type"] == "skill.used"
        assert record["skill"] == "gke_diagnosis"


class TestAuditSubscriberSession:
    """Tests for session start/end handlers."""

    def test_session_started_sets_id(
        self,
        subscriber: AuditSubscriber,
        mock_bq_client: MagicMock,
    ) -> None:
        """SessionStarted sets the current session_id and creates a record."""
        bus = EventBus.get()
        bus.emit(SessionStarted(session_id="sess-123", model="gemini-2.5-pro", skill="gke"))
        subscriber._flush()

        assert subscriber._current_session_id == "sess-123"
        records = mock_bq_client.insert_rows_json.call_args[0][1]
        record = records[0]
        assert record["event_type"] == "session.started"
        assert record["session_id"] == "sess-123"
        assert record["model"] == "gemini-2.5-pro"
        assert record["skill"] == "gke"

    def test_session_ended_immediate_flush(
        self,
        subscriber: AuditSubscriber,
        mock_bq_client: MagicMock,
    ) -> None:
        """SessionEnded triggers an immediate flush."""
        bus = EventBus.get()

        # Add some records to buffer
        bus.emit(CliCommandTracked(command_name="diagnose", duration_ms=100.0))
        assert len(subscriber._buffer) == 1

        # SessionEnded should flush immediately
        bus.emit(SessionEnded(session_id="sess-123", duration_ms=5000.0))

        # Buffer should be empty after flush
        assert len(subscriber._buffer) == 0
        # BQ should have received records
        assert mock_bq_client.insert_rows_json.called


class TestAuditSubscriberError:
    """Tests for the ErrorOccurred event handler."""

    def test_error_record(
        self,
        subscriber: AuditSubscriber,
        mock_bq_client: MagicMock,
    ) -> None:
        """ErrorOccurred produces record with result='fail' and error details."""
        bus = EventBus.get()
        bus.emit(ErrorOccurred(error_type="GeminiClientError", error_message="API timeout", source="client"))
        subscriber._flush()

        records = mock_bq_client.insert_rows_json.call_args[0][1]
        record = records[0]
        assert record["event_type"] == "error.occurred"
        assert record["result"] == "fail"
        assert record["error_message"] == "API timeout"
        assert record["command"] == "client"


# ── Buffer and flush tests ───────────────────────────────────


class TestAuditSubscriberBuffer:
    """Tests for buffer behavior and flushing."""

    def test_buffer_flushes_at_size(
        self,
        mock_settings: MagicMock,
        mock_bq_client: MagicMock,
        mock_logging_client: MagicMock,
    ) -> None:
        """Buffer flushes when it reaches the configured buffer_size."""
        mock_settings.audit.buffer_size = 3  # Small buffer for testing

        with patch("vaig.core.subscribers.audit_subscriber.resolve_identity") as mock_id:
            mock_id.return_value = ("u", "u@co.com", "u:u@co.com")
            with patch("vaig.core.subscribers.audit_subscriber.get_app_version", return_value="1.0.0"):
                sub = AuditSubscriber(
                    mock_settings,
                    bq_client=mock_bq_client,
                    logging_client=mock_logging_client,
                )

        bus = EventBus.get()

        # Emit 3 events to trigger flush at buffer_size=3
        bus.emit(CliCommandTracked(command_name="cmd1", duration_ms=100.0))
        bus.emit(CliCommandTracked(command_name="cmd2", duration_ms=100.0))

        # Before reaching buffer_size — no flush
        assert not mock_bq_client.insert_rows_json.called

        bus.emit(CliCommandTracked(command_name="cmd3", duration_ms=100.0))

        # After reaching buffer_size — flushed
        assert mock_bq_client.insert_rows_json.called
        sub.unsubscribe_all()

    def test_unsubscribe_all_flushes(
        self,
        subscriber: AuditSubscriber,
        mock_bq_client: MagicMock,
    ) -> None:
        """unsubscribe_all() flushes remaining records."""
        bus = EventBus.get()
        bus.emit(CliCommandTracked(command_name="diagnose", duration_ms=100.0))

        # Record in buffer
        assert len(subscriber._buffer) == 1

        subscriber.unsubscribe_all()

        # Buffer flushed
        assert len(subscriber._buffer) == 0
        assert mock_bq_client.insert_rows_json.called

    def test_empty_flush_is_noop(
        self,
        subscriber: AuditSubscriber,
        mock_bq_client: MagicMock,
    ) -> None:
        """Flushing an empty buffer does nothing."""
        subscriber._flush()
        assert not mock_bq_client.insert_rows_json.called


# ── Sink failure tests ───────────────────────────────────────


class TestAuditSubscriberSinkFailures:
    """Tests for independent sink failure handling."""

    def test_bq_failure_logging_proceeds(
        self,
        subscriber: AuditSubscriber,
        mock_bq_client: MagicMock,
        mock_logging_client: MagicMock,
    ) -> None:
        """BigQuery failure doesn't prevent Cloud Logging write."""
        mock_bq_client.insert_rows_json.side_effect = Exception("BQ error")

        bus = EventBus.get()
        bus.emit(CliCommandTracked(command_name="diagnose", duration_ms=100.0))
        subscriber._flush()

        # Cloud Logging should still be called
        gcp_logger = mock_logging_client.logger.return_value
        assert gcp_logger.log_struct.called

    def test_logging_failure_bq_proceeds(
        self,
        subscriber: AuditSubscriber,
        mock_bq_client: MagicMock,
        mock_logging_client: MagicMock,
    ) -> None:
        """Cloud Logging failure doesn't prevent BigQuery write."""
        mock_logging_client.logger.return_value.log_struct.side_effect = Exception("Logging error")

        bus = EventBus.get()
        bus.emit(CliCommandTracked(command_name="diagnose", duration_ms=100.0))
        subscriber._flush()

        # BQ should still be called
        assert mock_bq_client.insert_rows_json.called

    def test_bq_failure_logs_warning(
        self,
        subscriber: AuditSubscriber,
        mock_bq_client: MagicMock,
    ) -> None:
        """BigQuery failure logs a WARNING."""
        mock_bq_client.insert_rows_json.side_effect = Exception("BQ error")

        bus = EventBus.get()
        bus.emit(CliCommandTracked(command_name="diagnose", duration_ms=100.0))

        with patch("vaig.core.subscribers.audit_subscriber.logger") as mock_logger:
            subscriber._flush()
            mock_logger.warning.assert_called()

    def test_logging_failure_logs_warning(
        self,
        subscriber: AuditSubscriber,
        mock_logging_client: MagicMock,
    ) -> None:
        """Cloud Logging failure logs a WARNING."""
        mock_logging_client.logger.return_value.log_struct.side_effect = Exception("Logging error")

        bus = EventBus.get()
        bus.emit(CliCommandTracked(command_name="diagnose", duration_ms=100.0))

        with patch("vaig.core.subscribers.audit_subscriber.logger") as mock_logger:
            subscriber._flush()
            mock_logger.warning.assert_called()


# ── Cloud Logging severity tests ─────────────────────────────


class TestAuditSubscriberLogSeverity:
    """Tests for Cloud Logging severity mapping."""

    def test_error_event_gets_error_severity(
        self,
        subscriber: AuditSubscriber,
        mock_logging_client: MagicMock,
    ) -> None:
        """ErrorOccurred events get ERROR severity in Cloud Logging."""
        bus = EventBus.get()
        bus.emit(ErrorOccurred(error_type="TestError", error_message="boom", source="test"))
        subscriber._flush()

        gcp_logger = mock_logging_client.logger.return_value
        _, kwargs = gcp_logger.log_struct.call_args
        assert kwargs["severity"] == "ERROR"

    def test_normal_event_gets_info_severity(
        self,
        subscriber: AuditSubscriber,
        mock_logging_client: MagicMock,
    ) -> None:
        """Non-error events get INFO severity in Cloud Logging."""
        bus = EventBus.get()
        bus.emit(CliCommandTracked(command_name="diagnose", duration_ms=100.0))
        subscriber._flush()

        gcp_logger = mock_logging_client.logger.return_value
        _, kwargs = gcp_logger.log_struct.call_args
        assert kwargs["severity"] == "INFO"

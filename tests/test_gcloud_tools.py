"""Tests for GCP observability tools — gcloud_logging_query, gcloud_monitoring_query, create_gcloud_tools."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vaig.tools.base import ToolDef

# ── gcloud_logging_query ─────────────────────────────────────


class TestGcloudLoggingQuery:
    """Tests for gcloud_logging_query function."""

    def test_empty_filter_returns_error(self) -> None:
        from vaig.tools.gcloud_tools import gcloud_logging_query

        result = gcloud_logging_query("")
        assert result.error is True
        assert "empty" in result.output.lower()

    def test_whitespace_filter_returns_error(self) -> None:
        from vaig.tools.gcloud_tools import gcloud_logging_query

        result = gcloud_logging_query("   ")
        assert result.error is True

    def test_invalid_order_by(self) -> None:
        from vaig.tools.gcloud_tools import gcloud_logging_query

        result = gcloud_logging_query("severity>=ERROR", order_by="name asc")
        assert result.error is True
        assert "Invalid order_by" in result.output

    @patch("vaig.tools.gcloud_tools._get_logging_client")
    def test_sdk_not_installed(self, mock_client: MagicMock) -> None:
        from vaig.tools.gcloud_tools import gcloud_logging_query

        mock_client.return_value = (None, "google-cloud-logging SDK is not installed")

        result = gcloud_logging_query("severity>=ERROR")
        assert result.error is True
        assert "not installed" in result.output

    @patch("vaig.tools.gcloud_tools._get_logging_client")
    def test_successful_query(self, mock_client: MagicMock) -> None:
        from vaig.tools.gcloud_tools import gcloud_logging_query

        client = MagicMock()
        mock_client.return_value = (client, None)

        # Create mock log entries
        entry = MagicMock()
        entry.timestamp = MagicMock()
        entry.timestamp.strftime.return_value = "2025-01-01 12:00:00"
        entry.severity = "ERROR"
        entry.resource.type = "k8s_container"
        entry.resource.labels = {"namespace_name": "prod"}
        entry.text_payload = "Connection timeout"
        entry.payload = "Connection timeout"
        entry.json_payload = None
        entry.proto_payload = None

        client.list_entries.return_value = [entry]

        result = gcloud_logging_query("severity>=ERROR")
        assert result.error is False
        assert "Connection timeout" in result.output
        assert "1 log entries" in result.output

    @patch("vaig.tools.gcloud_tools._get_logging_client")
    def test_no_entries_found(self, mock_client: MagicMock) -> None:
        from vaig.tools.gcloud_tools import gcloud_logging_query

        client = MagicMock()
        mock_client.return_value = (client, None)
        client.list_entries.return_value = []

        result = gcloud_logging_query("severity>=CRITICAL")
        assert result.error is False
        assert "No log entries found" in result.output

    @patch("vaig.tools.gcloud_tools._get_logging_client")
    def test_permission_denied(self, mock_client: MagicMock) -> None:
        from vaig.tools.gcloud_tools import gcloud_logging_query

        client = MagicMock()
        mock_client.return_value = (client, None)
        client.list_entries.side_effect = Exception("403 Permission denied")

        result = gcloud_logging_query("severity>=ERROR")
        assert result.error is True
        assert "Permission denied" in result.output

    @patch("vaig.tools.gcloud_tools._get_logging_client")
    def test_quota_exceeded(self, mock_client: MagicMock) -> None:
        from vaig.tools.gcloud_tools import gcloud_logging_query

        client = MagicMock()
        mock_client.return_value = (client, None)
        client.list_entries.side_effect = Exception("429 Resource exhausted quota")

        result = gcloud_logging_query("severity>=ERROR")
        assert result.error is True
        assert "quota" in result.output.lower()

    def test_limit_clamped_to_minimum(self) -> None:
        from vaig.tools.gcloud_tools import gcloud_logging_query

        with patch("vaig.tools.gcloud_tools._get_logging_client") as mock_client:
            client = MagicMock()
            mock_client.return_value = (client, None)
            client.list_entries.return_value = []

            gcloud_logging_query("severity>=ERROR", limit=0)
            # Should clamp to 1
            call_kwargs = client.list_entries.call_args
            assert call_kwargs.kwargs.get("max_results") == 1

    def test_limit_clamped_to_maximum(self) -> None:
        from vaig.tools.gcloud_tools import gcloud_logging_query

        with patch("vaig.tools.gcloud_tools._get_logging_client") as mock_client:
            client = MagicMock()
            mock_client.return_value = (client, None)
            client.list_entries.return_value = []

            gcloud_logging_query("severity>=ERROR", limit=5000)
            call_kwargs = client.list_entries.call_args
            assert call_kwargs.kwargs.get("max_results") == 1000


# ── gcloud_monitoring_query ──────────────────────────────────


class TestGcloudMonitoringQuery:
    """Tests for gcloud_monitoring_query function."""

    def test_empty_metric_type_returns_error(self) -> None:
        from vaig.tools.gcloud_tools import gcloud_monitoring_query

        result = gcloud_monitoring_query("")
        assert result.error is True
        assert "empty" in result.output.lower()

    @patch("vaig.tools.gcloud_tools._get_monitoring_client")
    def test_sdk_not_installed(self, mock_client: MagicMock) -> None:
        from vaig.tools.gcloud_tools import gcloud_monitoring_query

        mock_client.return_value = (None, "google-cloud-monitoring SDK is not installed")

        result = gcloud_monitoring_query("compute.googleapis.com/instance/cpu/utilization")
        assert result.error is True
        assert "not installed" in result.output

    @patch("vaig.tools.gcloud_tools._get_monitoring_client")
    def test_no_project_detected(self, mock_client: MagicMock) -> None:
        from vaig.tools.gcloud_tools import gcloud_monitoring_query

        client = MagicMock()
        mock_client.return_value = (client, None)

        # google.auth is imported inline — inject a mock into the module's
        # import resolution so the lazy ``import google.auth`` inside the
        # function picks it up and raises.
        fake_google = MagicMock()
        fake_google.auth.default.side_effect = Exception("no credentials")

        with patch.dict("sys.modules", {"google": fake_google, "google.auth": fake_google.auth}):
            result = gcloud_monitoring_query(
                "compute.googleapis.com/instance/cpu/utilization",
                project="",
            )

        assert result.error is True
        assert "No GCP project" in result.output


# ── _format_log_entry ────────────────────────────────────────


class TestFormatLogEntry:
    """Tests for _format_log_entry helper."""

    def test_text_payload(self) -> None:
        from vaig.tools.gcloud_tools import _format_log_entry

        entry = MagicMock()
        entry.timestamp = MagicMock()
        entry.timestamp.strftime.return_value = "2025-01-01 10:00:00"
        entry.severity = "WARNING"
        entry.resource.type = "k8s_container"
        entry.resource.labels = {}
        entry.text_payload = "Request timed out"
        entry.payload = "Request timed out"
        entry.json_payload = None
        entry.proto_payload = None

        line = _format_log_entry(entry)
        assert "2025-01-01 10:00:00" in line
        assert "WARNING" in line
        assert "Request timed out" in line

    def test_json_payload_with_message(self) -> None:
        from vaig.tools.gcloud_tools import _format_log_entry

        entry = MagicMock()
        entry.timestamp = MagicMock()
        entry.timestamp.strftime.return_value = "2025-01-01 10:00:00"
        entry.severity = "ERROR"
        entry.resource.type = "k8s_container"
        entry.resource.labels = {}
        entry.text_payload = None
        entry.payload = None
        entry.json_payload = {"message": "OOMKilled", "code": 137}
        entry.proto_payload = None

        line = _format_log_entry(entry)
        assert "OOMKilled" in line

    def test_no_payload(self) -> None:
        from vaig.tools.gcloud_tools import _format_log_entry

        entry = MagicMock()
        entry.timestamp = None
        entry.severity = None
        entry.resource = None
        entry.text_payload = None
        entry.payload = None
        entry.json_payload = None
        entry.proto_payload = None

        line = _format_log_entry(entry)
        assert "N/A" in line
        assert "empty payload" in line

    def test_long_payload_truncated(self) -> None:
        from vaig.tools.gcloud_tools import _format_log_entry

        entry = MagicMock()
        entry.timestamp = None
        entry.severity = "INFO"
        entry.resource = None
        entry.text_payload = "x" * 600
        entry.payload = "x" * 600
        entry.json_payload = None
        entry.proto_payload = None

        line = _format_log_entry(entry)
        assert len(line) < 700  # Truncation should limit length
        assert "..." in line


# ── create_gcloud_tools factory ──────────────────────────────


class TestCreateGcloudTools:
    """Tests for create_gcloud_tools factory function."""

    def test_returns_two_tool_defs(self) -> None:
        from vaig.tools.gcloud_tools import create_gcloud_tools

        tools = create_gcloud_tools()
        assert len(tools) == 2
        assert all(isinstance(t, ToolDef) for t in tools)

    def test_tool_names(self) -> None:
        from vaig.tools.gcloud_tools import create_gcloud_tools

        tools = create_gcloud_tools()
        names = {t.name for t in tools}
        assert names == {"gcloud_logging_query", "gcloud_monitoring_query"}

    def test_all_have_descriptions(self) -> None:
        from vaig.tools.gcloud_tools import create_gcloud_tools

        tools = create_gcloud_tools()
        for t in tools:
            assert t.description, f"Tool {t.name} has no description"
            assert len(t.description) > 20

    def test_all_have_parameters(self) -> None:
        from vaig.tools.gcloud_tools import create_gcloud_tools

        tools = create_gcloud_tools()
        for t in tools:
            assert len(t.parameters) >= 1, f"Tool {t.name} has no parameters"

    def test_custom_defaults_propagated(self) -> None:
        """Factory should accept custom project/limit/interval defaults."""
        from vaig.tools.gcloud_tools import create_gcloud_tools

        tools = create_gcloud_tools(
            project="my-project",
            log_limit=200,
            metrics_interval_minutes=120,
        )
        # Just verify it doesn't crash and still returns 2 tools
        assert len(tools) == 2

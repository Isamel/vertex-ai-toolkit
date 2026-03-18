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

    @patch("vaig.tools.gcloud_tools._get_monitoring_client")
    def test_resource_labels_added_to_filter(self, mock_client: MagicMock) -> None:
        """resource_labels dict should be converted to resource.labels.xxx filter clauses."""
        from vaig.tools.gcloud_tools import gcloud_monitoring_query

        client = MagicMock()
        mock_client.return_value = (client, None)

        # We need to test that the filter is built correctly.
        # Since the function tries to import monitoring_v3, we'll use a
        # simpler approach: verify filter construction by checking what
        # gets passed to list_time_series.
        fake_monitoring = MagicMock()
        fake_interval = MagicMock()
        fake_monitoring.TimeInterval.return_value = fake_interval
        fake_monitoring.ListTimeSeriesRequest = MagicMock()
        fake_monitoring.Aggregation = MagicMock()
        fake_monitoring.ListTimeSeriesRequest.TimeSeriesView.FULL = "FULL"

        fake_duration = MagicMock()

        # The function does `from google.cloud.monitoring_v3 import types`,
        # so the parent module's .types attribute must point to fake_monitoring.
        # Similarly, `from google.protobuf import duration_pb2` needs the
        # parent's .duration_pb2 to point to fake_duration.
        fake_monitoring_v3 = MagicMock()
        fake_monitoring_v3.types = fake_monitoring
        fake_protobuf = MagicMock()
        fake_protobuf.duration_pb2 = fake_duration

        fake_timestamp = MagicMock()
        fake_protobuf.timestamp_pb2 = fake_timestamp

        with patch.dict("sys.modules", {
            "google.cloud.monitoring_v3": fake_monitoring_v3,
            "google.cloud.monitoring_v3.types": fake_monitoring,
            "google.protobuf": fake_protobuf,
            "google.protobuf.duration_pb2": fake_duration,
            "google.protobuf.timestamp_pb2": fake_timestamp,
        }):
            result = gcloud_monitoring_query(
                "istio.io/service/server/request_count",
                project="my-project",
                resource_labels={"namespace_name": "production", "cluster_name": "prod-1"},
            )

        # The request should have been constructed with resource_labels in the filter
        assert fake_monitoring.ListTimeSeriesRequest.called, (
            "ListTimeSeriesRequest was never called — filter was not constructed"
        )
        call_kwargs = fake_monitoring.ListTimeSeriesRequest.call_args
        filter_used = call_kwargs.kwargs.get("filter", "") if call_kwargs.kwargs else ""
        assert 'resource.labels.namespace_name' in filter_used, (
            f"Expected 'resource.labels.namespace_name' in filter, got: {filter_used}"
        )
        assert 'resource.labels.cluster_name' in filter_used, (
            f"Expected 'resource.labels.cluster_name' in filter, got: {filter_used}"
        )

    def test_resource_labels_none_no_effect(self) -> None:
        """resource_labels=None should not modify the filter."""
        from vaig.tools.gcloud_tools import gcloud_monitoring_query

        # This should not crash — just hits the "no project" path
        with patch("vaig.tools.gcloud_tools._get_monitoring_client") as mock_client:
            client = MagicMock()
            mock_client.return_value = (client, None)

            fake_google = MagicMock()
            fake_google.auth.default.side_effect = Exception("no credentials")

            with patch.dict("sys.modules", {"google": fake_google, "google.auth": fake_google.auth}):
                result = gcloud_monitoring_query(
                    "compute.googleapis.com/instance/cpu/utilization",
                    project="",
                    resource_labels=None,
                )
            assert result.error is True

    def test_resource_labels_empty_dict_no_effect(self) -> None:
        """resource_labels={} should not modify the filter."""
        from vaig.tools.gcloud_tools import gcloud_monitoring_query

        with patch("vaig.tools.gcloud_tools._get_monitoring_client") as mock_client:
            client = MagicMock()
            mock_client.return_value = (client, None)

            fake_google = MagicMock()
            fake_google.auth.default.side_effect = Exception("no credentials")

            with patch.dict("sys.modules", {"google": fake_google, "google.auth": fake_google.auth}):
                result = gcloud_monitoring_query(
                    "compute.googleapis.com/instance/cpu/utilization",
                    project="",
                    resource_labels={},
                )
            assert result.error is True

    @patch("vaig.tools.gcloud_tools._get_monitoring_client")
    def test_resource_labels_invalid_key_rejected(self, mock_client: MagicMock) -> None:
        """resource_labels with invalid key (injection attempt) should return error."""
        from vaig.tools.gcloud_tools import gcloud_monitoring_query

        client = MagicMock()
        mock_client.return_value = (client, None)

        fake_monitoring = MagicMock()
        fake_monitoring.ListTimeSeriesRequest = MagicMock()
        fake_monitoring.ListTimeSeriesRequest.TimeSeriesView.FULL = "FULL"
        fake_monitoring.Aggregation = MagicMock()

        fake_monitoring_v3 = MagicMock()
        fake_monitoring_v3.types = fake_monitoring

        with patch.dict("sys.modules", {
            "google.cloud.monitoring_v3": fake_monitoring_v3,
            "google.cloud.monitoring_v3.types": fake_monitoring,
            "google.protobuf": MagicMock(),
            "google.protobuf.duration_pb2": MagicMock(),
            "google.protobuf.timestamp_pb2": MagicMock(),
        }):
            result = gcloud_monitoring_query(
                "compute.googleapis.com/instance/cpu/utilization",
                project="my-project",
                resource_labels={"bad.key!": "value"},
            )
        assert result.error is True
        assert "Invalid resource label key" in result.output

    @patch("vaig.tools.gcloud_tools._get_monitoring_client")
    def test_resource_labels_value_with_quotes_escaped(self, mock_client: MagicMock) -> None:
        """resource_labels values containing quotes/backslashes should be escaped."""
        from vaig.tools.gcloud_tools import gcloud_monitoring_query

        client = MagicMock()
        mock_client.return_value = (client, None)

        fake_monitoring = MagicMock()
        fake_monitoring.TimeInterval.return_value = MagicMock()
        fake_monitoring.ListTimeSeriesRequest = MagicMock()
        fake_monitoring.Aggregation = MagicMock()
        fake_monitoring.ListTimeSeriesRequest.TimeSeriesView.FULL = "FULL"

        fake_monitoring_v3 = MagicMock()
        fake_monitoring_v3.types = fake_monitoring

        with patch.dict("sys.modules", {
            "google.cloud.monitoring_v3": fake_monitoring_v3,
            "google.cloud.monitoring_v3.types": fake_monitoring,
            "google.protobuf": MagicMock(),
            "google.protobuf.duration_pb2": MagicMock(),
            "google.protobuf.timestamp_pb2": MagicMock(),
        }):
            result = gcloud_monitoring_query(
                "compute.googleapis.com/instance/cpu/utilization",
                project="my-project",
                resource_labels={"namespace_name": 'prod"inject'},
            )

        assert fake_monitoring.ListTimeSeriesRequest.called, (
            "ListTimeSeriesRequest was never called"
        )
        call_kwargs = fake_monitoring.ListTimeSeriesRequest.call_args
        filter_used = call_kwargs.kwargs.get("filter", "") if call_kwargs.kwargs else ""
        # The double quote in the value should be escaped
        assert r'prod\"inject' in filter_used, (
            f"Expected escaped quote in filter, got: {filter_used}"
        )


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

    def test_monitoring_tool_has_resource_labels_param(self) -> None:
        """gcloud_monitoring_query tool should expose resource_labels parameter."""
        from vaig.tools.gcloud_tools import create_gcloud_tools

        tools = create_gcloud_tools()
        monitoring_tool = next(t for t in tools if t.name == "gcloud_monitoring_query")
        param_names = {p.name for p in monitoring_tool.parameters}
        assert "resource_labels" in param_names

        rl_param = next(p for p in monitoring_tool.parameters if p.name == "resource_labels")
        assert rl_param.type == "object"
        assert rl_param.required is False

    def test_monitoring_tool_lambda_accepts_resource_labels(self) -> None:
        """The monitoring tool lambda should accept resource_labels without TypeError."""
        from vaig.tools.gcloud_tools import create_gcloud_tools

        tools = create_gcloud_tools()
        monitoring_tool = next(t for t in tools if t.name == "gcloud_monitoring_query")

        # Call with resource_labels — should not raise TypeError.
        # Will fail on empty metric type (expected), but NOT a TypeError.
        result = monitoring_tool.execute(
            metric_type="",
            resource_labels={"namespace_name": "test"},
        )
        assert result.error is True
        assert "empty" in result.output.lower() or "Metric type" in result.output


# ── _format_time_series None handling ────────────────────────


class TestFormatTimeSeriesNoneHandling:
    """Tests for _format_time_series defensive None checks.

    The Cloud Monitoring API can return time series with sparse/incomplete
    data — None intervals, None values, None metric/resource objects.
    These tests verify graceful degradation instead of NoneType crashes.
    """

    def test_empty_time_series_list(self) -> None:
        from vaig.tools.gcloud_tools import _format_time_series

        result = _format_time_series([], "test.metric")
        assert "No time series data" in result

    def test_none_time_series_list(self) -> None:
        from vaig.tools.gcloud_tools import _format_time_series

        result = _format_time_series(None, "test.metric")
        assert "No time series data" in result

    def test_point_with_none_interval(self) -> None:
        """point.interval is None — should show 'N/A' timestamp, not crash."""
        from vaig.tools.gcloud_tools import _format_time_series

        point = MagicMock()
        point.interval = None
        point.value = MagicMock()
        point.value._pb = MagicMock()
        point.value._pb.WhichOneof.return_value = "int64_value"
        point.value.int64_value = 42

        ts = MagicMock()
        ts.metric = MagicMock()
        ts.metric.labels = {"response_code": "200"}
        ts.resource = MagicMock()
        ts.resource.labels = {"namespace_name": "prod"}
        ts.points = [point]

        result = _format_time_series([ts], "test.metric")
        assert "N/A" in result
        assert "42" in result

    def test_point_with_none_value(self) -> None:
        """point.value is None — should show 'N/A' value, not crash."""
        from vaig.tools.gcloud_tools import _format_time_series

        point = MagicMock()
        point.interval = MagicMock()
        point.interval.end_time = MagicMock()
        point.interval.end_time.strftime.return_value = "2025-06-01 12:00:00"
        point.value = None

        ts = MagicMock()
        ts.metric = MagicMock()
        ts.metric.labels = {}
        ts.resource = MagicMock()
        ts.resource.labels = {}
        ts.points = [point]

        result = _format_time_series([ts], "test.metric")
        assert "2025-06-01 12:00:00" in result
        assert "N/A" in result

    def test_series_with_none_metric_and_resource(self) -> None:
        """ts.metric and ts.resource are None — labels should be empty, not crash."""
        from vaig.tools.gcloud_tools import _format_time_series

        ts = MagicMock()
        ts.metric = None
        ts.resource = None
        ts.points = []

        result = _format_time_series([ts], "test.metric")
        assert "Series 1" in result

    def test_series_with_none_points(self) -> None:
        """ts.points is None — should produce empty series, not crash."""
        from vaig.tools.gcloud_tools import _format_time_series

        ts = MagicMock()
        ts.metric = MagicMock()
        ts.metric.labels = {}
        ts.resource = MagicMock()
        ts.resource.labels = {}
        ts.points = None

        result = _format_time_series([ts], "test.metric")
        assert "Series 1" in result

    def test_point_with_none_interval_and_none_value(self) -> None:
        """Both interval and value are None — full graceful degradation."""
        from vaig.tools.gcloud_tools import _format_time_series

        point = MagicMock()
        point.interval = None
        point.value = None

        ts = MagicMock()
        ts.metric = None
        ts.resource = None
        ts.points = [point]

        result = _format_time_series([ts], "test.metric")
        # Both timestamp and value should degrade to N/A
        lines_with_na = [line for line in result.split("\n") if "N/A" in line]
        assert len(lines_with_na) >= 1

    def test_metric_labels_none_but_metric_object_exists(self) -> None:
        """ts.metric exists but ts.metric.labels is None."""
        from vaig.tools.gcloud_tools import _format_time_series

        ts = MagicMock()
        ts.metric = MagicMock()
        ts.metric.labels = None
        ts.resource = MagicMock()
        ts.resource.labels = None
        ts.points = []

        result = _format_time_series([ts], "test.metric")
        assert "Series 1" in result


# ── Tool description quality ─────────────────────────────────


class TestMonitoringToolDescription:
    """Verify that tool descriptions guide the LLM to use resource_labels correctly."""

    def test_filter_str_description_discourages_resource_labels(self) -> None:
        """filter_str description should NOT show resource.labels examples."""
        from vaig.tools.gcloud_tools import create_gcloud_tools

        tools = create_gcloud_tools()
        monitoring_tool = next(t for t in tools if t.name == "gcloud_monitoring_query")
        filter_param = next(p for p in monitoring_tool.parameters if p.name == "filter_str")

        # The description should NOT contain a resource.labels example
        assert "resource.labels" not in filter_param.description, (
            f"filter_str description should not show resource.labels examples, "
            f"got: {filter_param.description}"
        )

    def test_tool_description_mentions_resource_labels_preference(self) -> None:
        """Main tool description should direct users to resource_labels param."""
        from vaig.tools.gcloud_tools import create_gcloud_tools

        tools = create_gcloud_tools()
        monitoring_tool = next(t for t in tools if t.name == "gcloud_monitoring_query")

        assert "resource_labels" in monitoring_tool.description
        # Should explicitly say not to put resource.labels in filter_str
        assert "filter_str" in monitoring_tool.description

    def test_resource_labels_description_says_preferred(self) -> None:
        """resource_labels param description should indicate it's the preferred way."""
        from vaig.tools.gcloud_tools import create_gcloud_tools

        tools = create_gcloud_tools()
        monitoring_tool = next(t for t in tools if t.name == "gcloud_monitoring_query")
        rl_param = next(p for p in monitoring_tool.parameters if p.name == "resource_labels")

        assert "PREFERRED" in rl_param.description.upper()


# ── Regression: Bug #2 — timestamp_pb2.Timestamp().FromDatetime ──────────────


class TestMonitoringQueryTimestampRegression:
    """Regression tests for Bug #2: gcloud_monitoring_query crashed with
    AttributeError: 'NoneType' object has no attribute 'FromDatetime'

    Root cause: monitoring_types.TimeInterval() was being mutated instead of
    constructing timestamp_pb2.Timestamp() objects and passing them as arguments.
    Fix: use timestamp_pb2.Timestamp() with FromDatetime() before constructing
    the TimeInterval.
    """

    @patch("vaig.tools.gcloud_tools._get_monitoring_client")
    def test_timestamp_pb2_from_datetime_is_called(self, mock_client: MagicMock) -> None:
        """Bug #2 regression: Timestamp objects are constructed via timestamp_pb2.Timestamp()
        and .FromDatetime() is called on them — never on None."""
        from vaig.tools.gcloud_tools import gcloud_monitoring_query

        client = MagicMock()
        mock_client.return_value = (client, None)
        client.list_time_series.return_value = []

        # Track Timestamp() calls to verify .FromDatetime() is called on instances
        fake_ts_instance = MagicMock()
        fake_timestamp_pb2 = MagicMock()
        fake_timestamp_pb2.Timestamp.return_value = fake_ts_instance

        fake_monitoring = MagicMock()
        fake_monitoring.TimeInterval = MagicMock()
        fake_monitoring.ListTimeSeriesRequest = MagicMock()
        fake_monitoring.ListTimeSeriesRequest.TimeSeriesView.FULL = "FULL"
        fake_monitoring.Aggregation = MagicMock()

        fake_monitoring_v3 = MagicMock()
        fake_monitoring_v3.types = fake_monitoring

        fake_protobuf = MagicMock()
        fake_protobuf.duration_pb2 = MagicMock()
        fake_protobuf.timestamp_pb2 = fake_timestamp_pb2

        with patch.dict("sys.modules", {
            "google.cloud.monitoring_v3": fake_monitoring_v3,
            "google.cloud.monitoring_v3.types": fake_monitoring,
            "google.protobuf": fake_protobuf,
            "google.protobuf.duration_pb2": fake_protobuf.duration_pb2,
            "google.protobuf.timestamp_pb2": fake_timestamp_pb2,
        }):
            result = gcloud_monitoring_query(
                "compute.googleapis.com/instance/cpu/utilization",
                project="my-project",
            )

        # The fix: Timestamp() must be called (at least twice — start and end)
        assert fake_timestamp_pb2.Timestamp.call_count >= 2, (
            "Expected timestamp_pb2.Timestamp() to be called at least twice "
            f"(start + end), got {fake_timestamp_pb2.Timestamp.call_count} calls"
        )
        # .FromDatetime() must be called on the Timestamp instance — not on None
        assert fake_ts_instance.FromDatetime.called, (
            "Expected .FromDatetime() to be called on a Timestamp instance, not on None. "
            "This is the Bug #2 regression — if Timestamp() returns None, this fails with "
            "AttributeError: 'NoneType' object has no attribute 'FromDatetime'"
        )

    @patch("vaig.tools.gcloud_tools._get_monitoring_client")
    def test_no_attribute_error_on_time_interval_construction(self, mock_client: MagicMock) -> None:
        """Bug #2 regression: no AttributeError is raised when constructing the TimeInterval.

        Before the fix, monitoring_types.TimeInterval() returned a MagicMock whose
        .start_time and .end_time were also MagicMocks, and calling .FromDatetime()
        on them would fail if the real protobuf objects were None (as in production).
        The fix constructs real Timestamp objects before passing them to TimeInterval().
        """
        from vaig.tools.gcloud_tools import gcloud_monitoring_query

        client = MagicMock()
        mock_client.return_value = (client, None)
        client.list_time_series.return_value = []

        fake_monitoring = MagicMock()
        fake_monitoring.ListTimeSeriesRequest = MagicMock()
        fake_monitoring.ListTimeSeriesRequest.TimeSeriesView.FULL = "FULL"
        fake_monitoring.Aggregation = MagicMock()

        fake_monitoring_v3 = MagicMock()
        fake_monitoring_v3.types = fake_monitoring

        with patch.dict("sys.modules", {
            "google.cloud.monitoring_v3": fake_monitoring_v3,
            "google.cloud.monitoring_v3.types": fake_monitoring,
            "google.protobuf": MagicMock(),
            "google.protobuf.duration_pb2": MagicMock(),
            "google.protobuf.timestamp_pb2": MagicMock(),
        }):
            # Should NOT raise AttributeError: 'NoneType' object has no attribute 'FromDatetime'
            result = gcloud_monitoring_query(
                "compute.googleapis.com/instance/cpu/utilization",
                project="my-project",
            )

        # Any result is acceptable — we only care that no AttributeError was raised
        assert result is not None

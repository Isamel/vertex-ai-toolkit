"""Tests for GKE Cloud Monitoring metrics — get_pod_metrics, helper functions.

Covers:
- _build_metric_filter() with various inputs
- _calculate_trend() with rising, falling, and stable data
- _format_metrics_response() with mock time series data
- get_pod_metrics() end-to-end with mocked MetricServiceClient
- Error cases: PermissionDenied, empty data, invalid metric_type
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vaig.core.config import GKEConfig

# ── Helpers ──────────────────────────────────────────────────


def _make_gke_config(**kwargs: object) -> GKEConfig:
    defaults = {
        "cluster_name": "test-cluster",
        "project_id": "test-project",
        "location": "us-central1",
        "default_namespace": "default",
        "kubeconfig_path": "",
        "context": "",
        "log_limit": 100,
        "metrics_interval_minutes": 60,
        "proxy_url": "",
    }
    defaults.update(kwargs)
    return GKEConfig(**defaults)


def _make_time_series(pod_name: str, double_values: list[float]) -> MagicMock:
    """Build a mock TimeSeries object with the given pod name and values."""
    ts = MagicMock()
    ts.resource.labels = {"pod_name": pod_name, "namespace_name": "default"}

    points = []
    for v in double_values:
        point = MagicMock()
        point.value.double_value = v
        point.value.int64_value = 0
        points.append(point)

    ts.points = points
    return ts


def _make_permission_denied() -> Exception:
    """Create a PermissionDenied-like exception."""
    exc = Exception("403 Permission denied")
    exc.__class__.__name__ = "PermissionDenied"  # type: ignore[attr-defined]
    # Create a real class for isinstance checks
    PermDenied = type("PermissionDenied", (Exception,), {})
    return PermDenied("Request had insufficient authentication scopes.")


# ── Unit tests for _build_metric_filter ─────────────────────


class TestBuildMetricFilter:
    """Tests for _build_metric_filter()."""

    def test_cpu_filter_contains_metric_type(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _build_metric_filter

        result = _build_metric_filter(
            _CPU_METRIC, "my-cluster", "production", "frontend-"
        )

        assert f'metric.type = "{_CPU_METRIC}"' in result

    def test_filter_contains_resource_type(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _build_metric_filter

        result = _build_metric_filter(_CPU_METRIC, "cluster", "ns", "pod-")

        assert 'resource.type = "k8s_container"' in result

    def test_filter_contains_cluster_name(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _build_metric_filter

        result = _build_metric_filter(_CPU_METRIC, "my-cluster", "ns", "pod-")

        assert 'resource.labels.cluster_name = "my-cluster"' in result

    def test_filter_contains_namespace(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _build_metric_filter

        result = _build_metric_filter(_CPU_METRIC, "cluster", "production", "pod-")

        assert 'resource.labels.namespace_name = "production"' in result

    def test_filter_contains_pod_prefix_regex(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _build_metric_filter

        result = _build_metric_filter(_CPU_METRIC, "cluster", "ns", "frontend-")

        assert 'monitoring.regex.full_match("^frontend-.*")' in result

    def test_memory_filter_contains_memory_metric_type(self) -> None:
        from vaig.tools.gke.monitoring import _MEMORY_METRIC, _build_metric_filter

        result = _build_metric_filter(_MEMORY_METRIC, "cluster", "ns", "backend-")

        assert f'metric.type = "{_MEMORY_METRIC}"' in result
        assert 'monitoring.regex.full_match("^backend-.*")' in result

    def test_filter_all_parts_present(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _build_metric_filter

        result = _build_metric_filter(_CPU_METRIC, "prod-cluster", "prod", "api-")

        # All five components must be present
        assert 'metric.type = "' in result
        assert 'resource.type = "k8s_container"' in result
        assert 'resource.labels.cluster_name = "prod-cluster"' in result
        assert 'resource.labels.namespace_name = "prod"' in result
        assert 'resource.labels.pod_name = monitoring.regex.full_match' in result

    def test_filter_empty_prefix(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _build_metric_filter

        result = _build_metric_filter(_CPU_METRIC, "cluster", "ns", "")

        # Empty prefix should produce a "match all pods" pattern
        assert 'monitoring.regex.full_match("^.*")' in result


# ── Unit tests for _calculate_trend ──────────────────────────


class TestCalculateTrend:
    """Tests for _calculate_trend()."""

    def test_stable_values_returns_arrow(self) -> None:
        from vaig.tools.gke.monitoring import _calculate_trend

        values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        assert _calculate_trend(values) == "→"

    def test_rising_values_returns_up(self) -> None:
        from vaig.tools.gke.monitoring import _calculate_trend

        # Start low, end high — > 10% increase
        values = [1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5]
        assert _calculate_trend(values) == "↑"

    def test_falling_values_returns_down(self) -> None:
        from vaig.tools.gke.monitoring import _calculate_trend

        # Start high, end low — > 10% decrease
        values = [2.5, 2.0, 1.7, 1.5, 1.3, 1.2, 1.1, 1.0]
        assert _calculate_trend(values) == "↓"

    def test_fewer_than_4_values_returns_stable(self) -> None:
        from vaig.tools.gke.monitoring import _calculate_trend

        # Not enough data to compute a trend
        assert _calculate_trend([]) == "→"
        assert _calculate_trend([1.0]) == "→"
        assert _calculate_trend([1.0, 2.0]) == "→"
        assert _calculate_trend([1.0, 2.0, 3.0]) == "→"

    def test_exactly_10_percent_increase_is_stable(self) -> None:
        from vaig.tools.gke.monitoring import _calculate_trend

        # Exactly 10% — NOT > 10%, so stays stable
        # head_avg = 1.0, tail_avg = 1.1 → change = 10% exactly
        values = [1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.1]
        result = _calculate_trend(values)
        # 10% is NOT strictly > 10%, so → or ↑ depending on implementation
        # The spec says > 10% increase → ↑, so 10% exactly = →
        assert result == "→"

    def test_head_avg_zero_returns_stable(self) -> None:
        from vaig.tools.gke.monitoring import _calculate_trend

        values = [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0]
        assert _calculate_trend(values) == "→"

    def test_slightly_above_10_percent_increase(self) -> None:
        from vaig.tools.gke.monitoring import _calculate_trend

        # head avg = 1.0, tail avg = 1.15 → 15% increase → ↑
        values = [1.0, 1.0, 1.0, 1.0, 1.15, 1.15, 1.15, 1.15]
        assert _calculate_trend(values) == "↑"


# ── Unit tests for _format_metrics_response ──────────────────


class TestFormatMetricsResponse:
    """Tests for _format_metrics_response()."""

    def test_empty_list_returns_no_data_message(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _format_metrics_response

        result = _format_metrics_response(
            [], metric_type=_CPU_METRIC, namespace="default",
            pod_name_prefix="api-", window_minutes=60,
        )

        assert "No data returned" in result
        assert "default" in result
        assert "api-" in result

    def test_cpu_table_has_correct_headers(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _format_metrics_response

        ts = _make_time_series("api-pod-1", [0.5, 0.6, 0.7, 0.8])
        result = _format_metrics_response(
            [ts], metric_type=_CPU_METRIC, namespace="default",
            pod_name_prefix="api-", window_minutes=60,
        )

        assert "| Pod" in result
        assert "Avg" in result
        assert "Max" in result
        assert "Latest" in result
        assert "Trend" in result

    def test_memory_table_shows_memory_label(self) -> None:
        from vaig.tools.gke.monitoring import _MEMORY_METRIC, _format_metrics_response

        ts = _make_time_series("app-pod-1", [50_000_000.0, 60_000_000.0, 55_000_000.0, 65_000_000.0])
        result = _format_metrics_response(
            [ts], metric_type=_MEMORY_METRIC, namespace="staging",
            pod_name_prefix="app-", window_minutes=30,
        )

        assert "Memory" in result
        assert "staging" in result

    def test_pod_name_appears_in_table(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _format_metrics_response

        ts = _make_time_series("frontend-abc-123", [0.1, 0.2, 0.3, 0.4])
        result = _format_metrics_response(
            [ts], metric_type=_CPU_METRIC, namespace="default",
            pod_name_prefix="frontend-", window_minutes=60,
        )

        assert "frontend-abc-123" in result

    def test_multiple_pods_appear_as_separate_rows(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _format_metrics_response

        ts1 = _make_time_series("pod-a", [0.1, 0.2, 0.3, 0.4])
        ts2 = _make_time_series("pod-b", [0.5, 0.6, 0.7, 0.8])
        result = _format_metrics_response(
            [ts1, ts2], metric_type=_CPU_METRIC, namespace="default",
            pod_name_prefix="pod-", window_minutes=60,
        )

        assert "pod-a" in result
        assert "pod-b" in result

    def test_summary_line_contains_pod_count(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _format_metrics_response

        ts1 = _make_time_series("pod-a", [0.1, 0.2, 0.3, 0.4])
        ts2 = _make_time_series("pod-b", [0.5, 0.6, 0.7, 0.8])
        result = _format_metrics_response(
            [ts1, ts2], metric_type=_CPU_METRIC, namespace="default",
            pod_name_prefix="pod-", window_minutes=60,
        )

        assert "2 pod(s)" in result

    def test_rising_trend_shown_in_table(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _format_metrics_response

        # Strongly rising series
        ts = _make_time_series("pod-a", [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8])
        result = _format_metrics_response(
            [ts], metric_type=_CPU_METRIC, namespace="default",
            pod_name_prefix="pod-", window_minutes=60,
        )

        assert "↑" in result

    def test_cpu_values_formatted_as_millicores(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _format_metrics_response

        # 0.125 cores/s * 1000 = 125m
        ts = _make_time_series("pod-a", [0.125, 0.125, 0.125, 0.125])
        result = _format_metrics_response(
            [ts], metric_type=_CPU_METRIC, namespace="default",
            pod_name_prefix="pod-", window_minutes=60,
        )

        assert "125m" in result

    def test_memory_values_formatted_as_mib(self) -> None:
        from vaig.tools.gke.monitoring import _MEMORY_METRIC, _format_metrics_response

        # 100 MiB = 104_857_600 bytes
        bytes_100mib = 100.0 * 1024 * 1024
        ts = _make_time_series("pod-a", [bytes_100mib] * 4)
        result = _format_metrics_response(
            [ts], metric_type=_MEMORY_METRIC, namespace="default",
            pod_name_prefix="pod-", window_minutes=60,
        )

        assert "100.0Mi" in result


# ── Integration tests for get_pod_metrics ────────────────────


class TestGetPodMetrics:
    """End-to-end tests for get_pod_metrics() with mocked Cloud Monitoring client."""

    def _make_mock_client(self, ts_list: list[MagicMock]) -> MagicMock:
        """Build a mock MetricServiceClient that returns ts_list for any query."""
        client = MagicMock()
        client.list_time_series.return_value = ts_list
        return client

    def test_returns_tool_result_on_success(self) -> None:
        from vaig.tools.base import ToolResult
        from vaig.tools.gke.monitoring import get_pod_metrics

        cfg = _make_gke_config()
        ts = _make_time_series("frontend-pod-1", [0.5, 0.6, 0.7, 0.8])
        mock_client = self._make_mock_client([ts])

        with patch("vaig.tools.gke.monitoring.MetricServiceClient", return_value=mock_client):
            result = get_pod_metrics(
                namespace="default",
                pod_name_prefix="frontend-",
                gke_config=cfg,
            )

        assert isinstance(result, ToolResult)
        assert not result.error
        assert "frontend-pod-1" in result.output

    def test_cpu_only_mode_skips_memory(self) -> None:
        from vaig.tools.gke.monitoring import get_pod_metrics

        cfg = _make_gke_config()
        mock_client = self._make_mock_client([])

        with patch("vaig.tools.gke.monitoring.MetricServiceClient", return_value=mock_client):
            result = get_pod_metrics(
                namespace="default",
                pod_name_prefix="api-",
                gke_config=cfg,
                metric_type="cpu",
            )

        # With cpu only, list_time_series called once (for CPU only)
        assert mock_client.list_time_series.call_count == 1
        assert not result.error

    def test_memory_only_mode_skips_cpu(self) -> None:
        from vaig.tools.gke.monitoring import get_pod_metrics

        cfg = _make_gke_config()
        mock_client = self._make_mock_client([])

        with patch("vaig.tools.gke.monitoring.MetricServiceClient", return_value=mock_client):
            result = get_pod_metrics(
                namespace="default",
                pod_name_prefix="api-",
                gke_config=cfg,
                metric_type="memory",
            )

        # With memory only, list_time_series called once (for memory only)
        assert mock_client.list_time_series.call_count == 1
        assert not result.error

    def test_all_mode_queries_both_metrics(self) -> None:
        from vaig.tools.gke.monitoring import get_pod_metrics

        cfg = _make_gke_config()
        mock_client = self._make_mock_client([])

        with patch("vaig.tools.gke.monitoring.MetricServiceClient", return_value=mock_client):
            result = get_pod_metrics(
                namespace="default",
                pod_name_prefix="api-",
                gke_config=cfg,
                metric_type="all",
            )

        # Both CPU and memory queried
        assert mock_client.list_time_series.call_count == 2
        assert not result.error

    def test_invalid_metric_type_returns_error(self) -> None:
        from vaig.tools.gke.monitoring import get_pod_metrics

        cfg = _make_gke_config()

        with patch("vaig.tools.gke.monitoring.MetricServiceClient"):
            result = get_pod_metrics(
                namespace="default",
                pod_name_prefix="api-",
                gke_config=cfg,
                metric_type="invalid",
            )

        assert result.error
        assert "invalid" in result.output.lower() or "Invalid" in result.output

    def test_empty_data_returns_no_data_message(self) -> None:
        from vaig.tools.gke.monitoring import get_pod_metrics

        cfg = _make_gke_config()
        mock_client = self._make_mock_client([])

        with patch("vaig.tools.gke.monitoring.MetricServiceClient", return_value=mock_client):
            result = get_pod_metrics(
                namespace="default",
                pod_name_prefix="nonexistent-",
                gke_config=cfg,
            )

        assert not result.error
        assert "No data returned" in result.output

    def test_permission_denied_returns_error_message(self) -> None:
        from vaig.tools.gke.monitoring import get_pod_metrics

        cfg = _make_gke_config()
        mock_client = MagicMock()

        # Simulate PermissionDenied by raising exception with 403 in message
        PermDenied = type("PermissionDenied", (Exception,), {})
        mock_client.list_time_series.side_effect = PermDenied("403 Forbidden")

        with patch("vaig.tools.gke.monitoring.MetricServiceClient", return_value=mock_client):
            result = get_pod_metrics(
                namespace="default",
                pod_name_prefix="api-",
                gke_config=cfg,
                metric_type="cpu",
            )

        assert not result.error  # Tool itself doesn't error — it reports the 403 inline
        assert "403" in result.output or "denied" in result.output.lower() or "roles/monitoring.viewer" in result.output

    def test_generic_api_error_returns_query_error_message(self) -> None:
        from vaig.tools.gke.monitoring import get_pod_metrics

        cfg = _make_gke_config()
        mock_client = MagicMock()
        mock_client.list_time_series.side_effect = RuntimeError("Connection timeout")

        with patch("vaig.tools.gke.monitoring.MetricServiceClient", return_value=mock_client):
            result = get_pod_metrics(
                namespace="default",
                pod_name_prefix="api-",
                gke_config=cfg,
                metric_type="cpu",
            )

        assert not result.error
        assert "Query error" in result.output or "Connection timeout" in result.output

    def test_output_contains_header_with_namespace_and_prefix(self) -> None:
        from vaig.tools.gke.monitoring import get_pod_metrics

        cfg = _make_gke_config()
        ts = _make_time_series("svc-pod-abc", [0.1, 0.2, 0.3, 0.4])
        mock_client = self._make_mock_client([ts])

        with patch("vaig.tools.gke.monitoring.MetricServiceClient", return_value=mock_client):
            result = get_pod_metrics(
                namespace="production",
                pod_name_prefix="svc-",
                gke_config=cfg,
            )

        assert "production" in result.output
        assert "svc-" in result.output

    def test_window_minutes_passed_through(self) -> None:
        from vaig.tools.gke.monitoring import get_pod_metrics

        cfg = _make_gke_config()
        mock_client = self._make_mock_client([])

        with patch("vaig.tools.gke.monitoring.MetricServiceClient", return_value=mock_client):
            result = get_pod_metrics(
                namespace="default",
                pod_name_prefix="api-",
                gke_config=cfg,
                window_minutes=120,
            )

        assert "120" in result.output
        assert not result.error

    def test_monitoring_unavailable_returns_error(self) -> None:
        from vaig.tools.base import ToolResult
        from vaig.tools.gke import monitoring

        cfg = _make_gke_config()
        original = monitoring._MONITORING_AVAILABLE

        try:
            monitoring._MONITORING_AVAILABLE = False
            result = monitoring.get_pod_metrics(
                namespace="default",
                pod_name_prefix="api-",
                gke_config=cfg,
            )
        finally:
            monitoring._MONITORING_AVAILABLE = original

        assert isinstance(result, ToolResult)
        assert result.error
        assert "not available" in result.output

    def test_client_init_failure_returns_error(self) -> None:
        from vaig.tools.gke.monitoring import get_pod_metrics

        cfg = _make_gke_config()

        with patch(
            "vaig.tools.gke.monitoring.MetricServiceClient",
            side_effect=Exception("Auth failed"),
        ):
            result = get_pod_metrics(
                namespace="default",
                pod_name_prefix="api-",
                gke_config=cfg,
            )

        assert result.error
        assert "Auth failed" in result.output


# ── Tests for tool registration ───────────────────────────────


class TestToolRegistration:
    """Verify get_pod_metrics is registered correctly in create_gke_tools()."""

    def _make_mock_gke_clients(self) -> None:
        """Patch the kubernetes client detection to avoid real cluster calls."""
        pass

    def test_get_pod_metrics_tool_registered(self) -> None:
        from vaig.tools.gke._registry import create_gke_tools

        cfg = _make_gke_config()

        with patch("vaig.tools.gke._clients.detect_autopilot", return_value=False):
            tools = create_gke_tools(cfg)

        tool_names = [t.name for t in tools]
        assert "get_pod_metrics" in tool_names

    def test_get_pod_metrics_has_monitoring_category(self) -> None:
        from vaig.tools.categories import MONITORING
        from vaig.tools.gke._registry import create_gke_tools

        cfg = _make_gke_config()

        with patch("vaig.tools.gke._clients.detect_autopilot", return_value=False):
            tools = create_gke_tools(cfg)

        tool = next(t for t in tools if t.name == "get_pod_metrics")
        assert MONITORING in tool.categories

    def test_get_pod_metrics_has_kubernetes_category(self) -> None:
        from vaig.tools.categories import KUBERNETES
        from vaig.tools.gke._registry import create_gke_tools

        cfg = _make_gke_config()

        with patch("vaig.tools.gke._clients.detect_autopilot", return_value=False):
            tools = create_gke_tools(cfg)

        tool = next(t for t in tools if t.name == "get_pod_metrics")
        assert KUBERNETES in tool.categories

    def test_get_pod_metrics_has_required_parameters(self) -> None:
        from vaig.tools.gke._registry import create_gke_tools

        cfg = _make_gke_config()

        with patch("vaig.tools.gke._clients.detect_autopilot", return_value=False):
            tools = create_gke_tools(cfg)

        tool = next(t for t in tools if t.name == "get_pod_metrics")
        param_names = [p.name for p in tool.parameters]
        assert "namespace" in param_names
        assert "pod_name_prefix" in param_names

    def test_get_pod_metrics_has_optional_parameters(self) -> None:
        from vaig.tools.gke._registry import create_gke_tools

        cfg = _make_gke_config()

        with patch("vaig.tools.gke._clients.detect_autopilot", return_value=False):
            tools = create_gke_tools(cfg)

        tool = next(t for t in tools if t.name == "get_pod_metrics")
        optional_params = {p.name for p in tool.parameters if not p.required}
        assert "window_minutes" in optional_params
        assert "metric_type" in optional_params

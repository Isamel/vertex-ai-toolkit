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

import pytest

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
    """Build a mock TimeSeries object with the given pod name and values.

    Values should be passed in chronological order (oldest first), matching
    test intent. This helper stores them in reverse order (newest first) to
    mimic the real Cloud Monitoring API, which returns points in reverse
    chronological order.
    """
    ts = MagicMock()
    ts.resource.labels = {"pod_name": pod_name, "namespace_name": "default"}

    points = []
    for v in reversed(double_values):
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

        result = _build_metric_filter(_CPU_METRIC, "my-cluster", "production", "frontend-")

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

        # _re2_escape does NOT escape hyphens (only special outside [...] in Python re,
        # but a literal in RE2 everywhere) — so "frontend-" stays "frontend-"
        assert 'monitoring.regex.full_match("^frontend-.*")' in result

    def test_filter_hyphenated_prefix_no_backslash(self) -> None:
        """Hyphens in pod prefixes must NOT be backslash-escaped in the filter.

        ``re.escape("istio-ingressgateway")`` produces ``"istio\\-ingressgateway"``,
        which is valid Python regex but causes an RE2 parse error
        (``unsupported escape sequence: \\-``) in Cloud Monitoring, resulting
        in an HTTP 400.  ``_re2_escape`` must leave hyphens unescaped.
        """
        from vaig.tools.gke.monitoring import _CPU_METRIC, _build_metric_filter

        result = _build_metric_filter(_CPU_METRIC, "cluster", "ns", "istio-ingressgateway")

        assert "\\-" not in result
        assert 'monitoring.regex.full_match("^istio-ingressgateway.*")' in result

    def test_memory_filter_contains_memory_metric_type(self) -> None:
        from vaig.tools.gke.monitoring import _MEMORY_METRIC, _build_metric_filter

        result = _build_metric_filter(_MEMORY_METRIC, "cluster", "ns", "backend-")

        assert f'metric.type = "{_MEMORY_METRIC}"' in result
        # Hyphens left as-is (RE2-safe)
        assert 'monitoring.regex.full_match("^backend-.*")' in result

    def test_filter_all_parts_present(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _build_metric_filter

        result = _build_metric_filter(_CPU_METRIC, "prod-cluster", "prod", "api-")

        # All five components must be present
        assert 'metric.type = "' in result
        assert 'resource.type = "k8s_container"' in result
        assert 'resource.labels.cluster_name = "prod-cluster"' in result
        assert 'resource.labels.namespace_name = "prod"' in result
        assert "resource.labels.pod_name = monitoring.regex.full_match" in result

    def test_filter_empty_prefix(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _build_metric_filter

        result = _build_metric_filter(_CPU_METRIC, "cluster", "ns", "")

        # Empty prefix → matches all pods
        assert 'monitoring.regex.full_match("^.*")' in result


# ── Unit tests for _re2_escape ────────────────────────────────


class TestRe2Escape:
    """Tests for _re2_escape() — ensures RE2-safe escaping without over-escaping hyphens."""

    def test_plain_string_unchanged(self) -> None:
        from vaig.tools.gke.monitoring import _re2_escape

        assert _re2_escape("mypod") == "mypod"

    def test_hyphen_not_escaped(self) -> None:
        """Hyphens are literal in RE2 outside character classes and must not be escaped."""
        from vaig.tools.gke.monitoring import _re2_escape

        assert _re2_escape("istio-ingressgateway") == "istio-ingressgateway"

    def test_multiple_hyphens_not_escaped(self) -> None:
        from vaig.tools.gke.monitoring import _re2_escape

        assert _re2_escape("my-app-v2") == "my-app-v2"

    def test_dot_is_escaped(self) -> None:
        from vaig.tools.gke.monitoring import _re2_escape

        assert _re2_escape("my.pod") == r"my\.pod"

    def test_plus_is_escaped(self) -> None:
        from vaig.tools.gke.monitoring import _re2_escape

        assert _re2_escape("a+b") == r"a\+b"

    def test_star_is_escaped(self) -> None:
        from vaig.tools.gke.monitoring import _re2_escape

        assert _re2_escape("a*b") == r"a\*b"

    def test_parens_are_escaped(self) -> None:
        from vaig.tools.gke.monitoring import _re2_escape

        assert _re2_escape("foo(bar)") == r"foo\(bar\)"

    def test_caret_is_escaped(self) -> None:
        from vaig.tools.gke.monitoring import _re2_escape

        assert _re2_escape("^start") == r"\^start"

    def test_dollar_is_escaped(self) -> None:
        from vaig.tools.gke.monitoring import _re2_escape

        assert _re2_escape("end$") == r"end\$"

    def test_pipe_is_escaped(self) -> None:
        from vaig.tools.gke.monitoring import _re2_escape

        assert _re2_escape("a|b") == r"a\|b"

    def test_backslash_is_escaped(self) -> None:
        from vaig.tools.gke.monitoring import _re2_escape

        assert _re2_escape("a\\b") == r"a\\b"

    def test_empty_string(self) -> None:
        from vaig.tools.gke.monitoring import _re2_escape

        assert _re2_escape("") == ""

    def test_double_quote_is_escaped(self) -> None:
        """Double-quote must be escaped to prevent filter injection.

        The escaped prefix is embedded inside ``monitoring.regex.full_match("^...*")``.
        An unescaped ``"`` in the input would terminate the string literal and
        allow injection of arbitrary filter clauses.
        """
        from vaig.tools.gke.monitoring import _re2_escape

        assert _re2_escape('pod"name') == r"pod\"name"

    def test_hyphen_with_dots_mixed(self) -> None:
        """Hyphens pass through; dots are escaped — common real-world prefix."""
        from vaig.tools.gke.monitoring import _re2_escape

        result = _re2_escape("my-app.v2")
        assert result == r"my-app\.v2"


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
            [],
            metric_type=_CPU_METRIC,
            namespace="default",
            pod_name_prefix="api-",
            window_minutes=60,
        )

        assert "No data returned" in result
        assert "default" in result
        assert "api-" in result

    def test_cpu_table_has_correct_headers(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _format_metrics_response

        ts = _make_time_series("api-pod-1", [0.5, 0.6, 0.7, 0.8])
        result = _format_metrics_response(
            [ts],
            metric_type=_CPU_METRIC,
            namespace="default",
            pod_name_prefix="api-",
            window_minutes=60,
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
            [ts],
            metric_type=_MEMORY_METRIC,
            namespace="staging",
            pod_name_prefix="app-",
            window_minutes=30,
        )

        assert "Memory" in result
        assert "staging" in result

    def test_pod_name_appears_in_table(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _format_metrics_response

        ts = _make_time_series("frontend-abc-123", [0.1, 0.2, 0.3, 0.4])
        result = _format_metrics_response(
            [ts],
            metric_type=_CPU_METRIC,
            namespace="default",
            pod_name_prefix="frontend-",
            window_minutes=60,
        )

        assert "frontend-abc-123" in result

    def test_multiple_pods_appear_as_separate_rows(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _format_metrics_response

        ts1 = _make_time_series("pod-a", [0.1, 0.2, 0.3, 0.4])
        ts2 = _make_time_series("pod-b", [0.5, 0.6, 0.7, 0.8])
        result = _format_metrics_response(
            [ts1, ts2],
            metric_type=_CPU_METRIC,
            namespace="default",
            pod_name_prefix="pod-",
            window_minutes=60,
        )

        assert "pod-a" in result
        assert "pod-b" in result

    def test_summary_line_contains_pod_count(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _format_metrics_response

        ts1 = _make_time_series("pod-a", [0.1, 0.2, 0.3, 0.4])
        ts2 = _make_time_series("pod-b", [0.5, 0.6, 0.7, 0.8])
        result = _format_metrics_response(
            [ts1, ts2],
            metric_type=_CPU_METRIC,
            namespace="default",
            pod_name_prefix="pod-",
            window_minutes=60,
        )

        assert "2 pod(s)" in result

    def test_rising_trend_shown_in_table(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _format_metrics_response

        # Strongly rising series
        ts = _make_time_series("pod-a", [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8])
        result = _format_metrics_response(
            [ts],
            metric_type=_CPU_METRIC,
            namespace="default",
            pod_name_prefix="pod-",
            window_minutes=60,
        )

        assert "↑" in result

    def test_cpu_values_formatted_as_millicores(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _format_metrics_response

        # 0.125 cores/s * 1000 = 125m
        ts = _make_time_series("pod-a", [0.125, 0.125, 0.125, 0.125])
        result = _format_metrics_response(
            [ts],
            metric_type=_CPU_METRIC,
            namespace="default",
            pod_name_prefix="pod-",
            window_minutes=60,
        )

        assert "125m" in result

    def test_memory_values_formatted_as_mib(self) -> None:
        from vaig.tools.gke.monitoring import _MEMORY_METRIC, _format_metrics_response

        # 100 MiB = 104_857_600 bytes
        bytes_100mib = 100.0 * 1024 * 1024
        ts = _make_time_series("pod-a", [bytes_100mib] * 4)
        result = _format_metrics_response(
            [ts],
            metric_type=_MEMORY_METRIC,
            namespace="default",
            pod_name_prefix="pod-",
            window_minutes=60,
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


# ── Tests for _extract_points_values reverse-chronological order ──


class TestExtractPointsValuesOrder:
    """Verify _extract_points_values handles API reverse-chronological order."""

    def test_points_in_api_order_reversed_to_chronological(self) -> None:
        """API returns newest-first; extraction must return oldest-first."""
        from vaig.tools.gke.monitoring import _extract_points_values

        # Simulate API returning 3 points: newest (3.0) → oldest (1.0)
        ts = MagicMock()
        points = []
        for v in [3.0, 2.0, 1.0]:  # newest-first, as the real API does
            point = MagicMock()
            point.value.double_value = v
            point.value.int64_value = 0
            points.append(point)
        ts.points = points

        result = _extract_points_values(ts)

        # After reversal: oldest-first → [1.0, 2.0, 3.0]
        assert result == [1.0, 2.0, 3.0]

    def test_zero_double_value_included(self) -> None:
        """A point with double_value == 0.0 is valid and must not be skipped."""
        from vaig.tools.gke.monitoring import _extract_points_values

        ts = MagicMock()
        points = []
        for v in [0.0, 1.0, 2.0]:
            point = MagicMock()
            point.value.double_value = v
            point.value.int64_value = 0
            points.append(point)
        ts.points = points

        result = _extract_points_values(ts)

        assert 0.0 in result
        assert len(result) == 3

    def test_zero_int64_value_included(self) -> None:
        """A point with int64_value == 0 is valid and must not be skipped."""
        from vaig.tools.gke.monitoring import _extract_points_values

        ts = MagicMock()
        # Mock only int64_value (no double_value attribute)
        ts.points = []
        for v in [0, 5, 10]:
            point = MagicMock(spec=[])
            point.value = MagicMock(spec=["int64_value"])
            point.value.int64_value = v
            ts.points.append(point)

        result = _extract_points_values(ts)

        assert 0.0 in result
        assert len(result) == 3


# ── Tests for _format_metrics_response truncation ────────────


class TestFormatMetricsResponseTruncation:
    """Verify table output is truncated beyond 20 pods."""

    def test_21_pods_shows_truncation_line(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _format_metrics_response

        ts_list = [_make_time_series(f"pod-{i:03d}", [0.1, 0.2, 0.3, 0.4]) for i in range(21)]
        result = _format_metrics_response(
            ts_list,
            metric_type=_CPU_METRIC,
            namespace="default",
            pod_name_prefix="pod-",
            window_minutes=60,
        )

        assert "more pods (truncated)" in result
        assert "1 more" in result

    def test_20_pods_does_not_truncate(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _format_metrics_response

        ts_list = [_make_time_series(f"pod-{i:03d}", [0.1, 0.2, 0.3, 0.4]) for i in range(20)]
        result = _format_metrics_response(
            ts_list,
            metric_type=_CPU_METRIC,
            namespace="default",
            pod_name_prefix="pod-",
            window_minutes=60,
        )

        assert "truncated" not in result

    def test_summary_line_shows_total_pod_count_not_displayed_count(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _format_metrics_response

        ts_list = [_make_time_series(f"pod-{i:03d}", [0.1, 0.2, 0.3, 0.4]) for i in range(25)]
        result = _format_metrics_response(
            ts_list,
            metric_type=_CPU_METRIC,
            namespace="default",
            pod_name_prefix="pod-",
            window_minutes=60,
        )

        # Summary must reflect total (25), not just displayed (20)
        assert "25 pod(s)" in result


# ── Tests for window_minutes validation ──────────────────────


class TestWindowMinutesValidation:
    """Verify get_pod_metrics validates window_minutes before querying."""

    def test_zero_window_minutes_returns_error(self) -> None:
        from vaig.tools.gke.monitoring import get_pod_metrics

        cfg = _make_gke_config()

        with patch("vaig.tools.gke.monitoring.MetricServiceClient"):
            result = get_pod_metrics(
                namespace="default",
                pod_name_prefix="api-",
                gke_config=cfg,
                window_minutes=0,
            )

        assert result.error
        assert "window_minutes" in result.output.lower() or "Invalid" in result.output

    def test_negative_window_minutes_returns_error(self) -> None:
        from vaig.tools.gke.monitoring import get_pod_metrics

        cfg = _make_gke_config()

        with patch("vaig.tools.gke.monitoring.MetricServiceClient"):
            result = get_pod_metrics(
                namespace="default",
                pod_name_prefix="api-",
                gke_config=cfg,
                window_minutes=-10,
            )

        assert result.error

    def test_over_1440_window_minutes_returns_error(self) -> None:
        from vaig.tools.gke.monitoring import get_pod_metrics

        cfg = _make_gke_config()

        with patch("vaig.tools.gke.monitoring.MetricServiceClient"):
            result = get_pod_metrics(
                namespace="default",
                pod_name_prefix="api-",
                gke_config=cfg,
                window_minutes=1441,
            )

        assert result.error
        assert "1440" in result.output

    def test_1_window_minutes_is_valid(self) -> None:
        from vaig.tools.gke.monitoring import get_pod_metrics

        cfg = _make_gke_config()
        mock_client = MagicMock()
        mock_client.list_time_series.return_value = []

        with patch("vaig.tools.gke.monitoring.MetricServiceClient", return_value=mock_client):
            result = get_pod_metrics(
                namespace="default",
                pod_name_prefix="api-",
                gke_config=cfg,
                window_minutes=1,
                metric_type="cpu",
            )

        assert not result.error

    def test_1440_window_minutes_is_valid(self) -> None:
        from vaig.tools.gke.monitoring import get_pod_metrics

        cfg = _make_gke_config()
        mock_client = MagicMock()
        mock_client.list_time_series.return_value = []

        with patch("vaig.tools.gke.monitoring.MetricServiceClient", return_value=mock_client):
            result = get_pod_metrics(
                namespace="default",
                pod_name_prefix="api-",
                gke_config=cfg,
                window_minutes=1440,
                metric_type="cpu",
            )

        assert not result.error


# ── Tests for regex injection protection ─────────────────────


class TestRegexInjectionProtection:
    """Verify pod_name_prefix is safely escaped before regex interpolation."""

    def test_prefix_with_dot_is_escaped(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _build_metric_filter

        result = _build_metric_filter(_CPU_METRIC, "cluster", "ns", "app.v2-")

        # _re2_escape: dot → \. but hyphen stays literal (RE2-safe)
        assert r"app\.v2-" in result

    def test_prefix_with_brackets_is_escaped(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _build_metric_filter

        result = _build_metric_filter(_CPU_METRIC, "cluster", "ns", "pod[0]-")

        # _re2_escape: brackets → \[ \] but hyphen stays literal
        assert r"pod\[0\]-" in result


# ── Tests for _build_metric_filter_with_container ────────────


class TestBuildMetricFilterWithContainer:
    """Tests for the v2 container-level filter builder."""

    def test_filter_contains_metric_type(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _build_metric_filter_with_container

        result = _build_metric_filter_with_container(_CPU_METRIC, "my-cluster", "production")

        assert f'metric.type = "{_CPU_METRIC}"' in result

    def test_filter_contains_k8s_container_resource_type(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _build_metric_filter_with_container

        result = _build_metric_filter_with_container(_CPU_METRIC, "cluster", "ns")

        assert 'resource.type = "k8s_container"' in result

    def test_filter_contains_cluster_name(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _build_metric_filter_with_container

        result = _build_metric_filter_with_container(_CPU_METRIC, "my-cluster", "ns")

        assert 'resource.labels.cluster_name = "my-cluster"' in result

    def test_filter_contains_namespace(self) -> None:
        from vaig.tools.gke.monitoring import _CPU_METRIC, _build_metric_filter_with_container

        result = _build_metric_filter_with_container(_CPU_METRIC, "cluster", "production")

        assert 'resource.labels.namespace_name = "production"' in result

    def test_filter_does_not_contain_pod_name_filter(self) -> None:
        """Unlike _build_metric_filter, the container variant must NOT restrict by pod name."""
        from vaig.tools.gke.monitoring import _CPU_METRIC, _build_metric_filter_with_container

        result = _build_metric_filter_with_container(_CPU_METRIC, "cluster", "ns")

        assert "pod_name" not in result


# ── Tests for get_workload_usage_metrics ─────────────────────


def _make_container_ts(
    pod_name: str,
    container_name: str,
    double_values: list[float],
) -> MagicMock:
    """Build a mock TimeSeries for container-level metrics.

    resource.labels → pod_name, namespace_name
    metric.labels   → container_name
    """
    ts = MagicMock()
    ts.resource.labels = {"pod_name": pod_name, "namespace_name": "default"}
    ts.metric.labels = {"container_name": container_name}

    points = []
    for v in reversed(double_values):
        point = MagicMock()
        point.value.double_value = v
        point.value.int64_value = 0
        points.append(point)

    ts.points = points
    return ts


class TestGetWorkloadUsageMetrics:
    """Tests for get_workload_usage_metrics()."""

    def test_empty_workload_pod_names_returns_empty_dict(self) -> None:
        from vaig.tools.gke.monitoring import get_workload_usage_metrics

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.monitoring.MetricServiceClient"):
            result = get_workload_usage_metrics(
                namespace="default",
                workload_pod_names={},
                gke_config=cfg,
            )

        assert result == {}

    def test_monitoring_unavailable_returns_empty_dict(self) -> None:
        """When _MONITORING_AVAILABLE is False, should return {} without calling GCP."""
        from vaig.tools.gke.monitoring import get_workload_usage_metrics

        cfg = _make_gke_config()
        with (
            patch("vaig.tools.gke.monitoring._MONITORING_AVAILABLE", False),
            patch("vaig.tools.gke.monitoring.MetricServiceClient") as mock_client_cls,
        ):
            result = get_workload_usage_metrics(
                namespace="default",
                workload_pod_names={"api": ["api-pod-1"]},
                gke_config=cfg,
            )

        mock_client_cls.assert_not_called()
        assert result == {}

    def test_client_creation_failure_returns_empty_dict(self) -> None:
        """If MetricServiceClient() raises, should return {} gracefully."""
        from vaig.tools.gke.monitoring import get_workload_usage_metrics

        cfg = _make_gke_config()
        with patch(
            "vaig.tools.gke.monitoring.MetricServiceClient",
            side_effect=RuntimeError("auth error"),
        ):
            result = get_workload_usage_metrics(
                namespace="default",
                workload_pod_names={"api": ["api-pod-1"]},
                gke_config=cfg,
            )

        assert result == {}

    def test_single_container_metrics_mapped_to_workload(self) -> None:
        """A single container time-series should be aggregated into the workload entry."""
        from vaig.tools.gke.monitoring import get_workload_usage_metrics

        cfg = _make_gke_config()
        mock_client = MagicMock()

        cpu_ts = _make_container_ts("api-pod-1", "app", [0.2, 0.3, 0.25])
        mem_ts = _make_container_ts("api-pod-1", "app", [0.5 * (1024**3)] * 3)  # 0.5 GiB in bytes

        # First call → CPU, second → memory
        mock_client.list_time_series.side_effect = [[cpu_ts], [mem_ts]]

        with patch("vaig.tools.gke.monitoring.MetricServiceClient", return_value=mock_client):
            result = get_workload_usage_metrics(
                namespace="default",
                workload_pod_names={"api": ["api-pod-1"]},
                gke_config=cfg,
            )

        assert "api" in result
        workload_metrics = result["api"]
        assert "app" in workload_metrics.containers
        ct = workload_metrics.containers["app"]
        assert ct.container_name == "app"
        assert ct.avg_cpu_cores == pytest.approx(0.25)  # mean of [0.2, 0.3, 0.25]
        # Memory is stored in bytes internally; 0.5 GiB = 0.5 * (1024^3) bytes
        # The function converts to GiB: bytes / (1024^3)
        assert ct.avg_memory_gib == pytest.approx(0.5, abs=1e-3)

    def test_pod_not_in_workload_pod_names_is_ignored(self) -> None:
        """Time series for pods not in workload_pod_names must be silently skipped."""
        from vaig.tools.gke.monitoring import get_workload_usage_metrics

        cfg = _make_gke_config()
        mock_client = MagicMock()

        # Pod "unknown-pod-9" is NOT in workload_pod_names
        cpu_ts = _make_container_ts("unknown-pod-9", "app", [0.5])
        mock_client.list_time_series.side_effect = [[cpu_ts], []]

        with patch("vaig.tools.gke.monitoring.MetricServiceClient", return_value=mock_client):
            result = get_workload_usage_metrics(
                namespace="default",
                workload_pod_names={"api": ["api-pod-1"]},
                gke_config=cfg,
            )

        # "api" has no matching time series → absent from result
        assert result == {}

    def test_cpu_query_failure_falls_back_gracefully(self) -> None:
        """If the CPU query raises, memory query should still run."""
        from vaig.tools.gke.monitoring import get_workload_usage_metrics

        cfg = _make_gke_config()
        mock_client = MagicMock()

        mem_ts = _make_container_ts("api-pod-1", "app", [0.25 * (1024**3)] * 2)
        mock_client.list_time_series.side_effect = [
            RuntimeError("CPU query failed"),
            [mem_ts],
        ]

        with patch("vaig.tools.gke.monitoring.MetricServiceClient", return_value=mock_client):
            result = get_workload_usage_metrics(
                namespace="default",
                workload_pod_names={"api": ["api-pod-1"]},
                gke_config=cfg,
            )

        # With no CPU data, container is absent from result
        # (memory alone doesn't create a WorkloadUsageMetrics entry)
        # The important thing is it doesn't crash
        assert isinstance(result, dict)

    def test_multiple_workloads_mapped_independently(self) -> None:
        """Multiple workloads in the same namespace each get their own entry."""
        from vaig.tools.gke.monitoring import get_workload_usage_metrics

        cfg = _make_gke_config()
        mock_client = MagicMock()

        cpu_ts_api = _make_container_ts("api-pod-1", "app", [0.3])
        cpu_ts_worker = _make_container_ts("worker-pod-1", "worker", [0.6])
        mem_ts_api = _make_container_ts("api-pod-1", "app", [0.2 * (1024**3)])
        mem_ts_worker = _make_container_ts("worker-pod-1", "worker", [0.4 * (1024**3)])

        mock_client.list_time_series.side_effect = [
            [cpu_ts_api, cpu_ts_worker],
            [mem_ts_api, mem_ts_worker],
        ]

        with patch("vaig.tools.gke.monitoring.MetricServiceClient", return_value=mock_client):
            result = get_workload_usage_metrics(
                namespace="default",
                workload_pod_names={
                    "api": ["api-pod-1"],
                    "worker": ["worker-pod-1"],
                },
                gke_config=cfg,
            )

        assert "api" in result
        assert "worker" in result
        assert result["api"].containers["app"].avg_cpu_cores == pytest.approx(0.3)
        assert result["worker"].containers["worker"].avg_cpu_cores == pytest.approx(0.6)


# ── Tests for pod-mismatch diagnostics (Bug B fix) ──────────────────────────


class TestPodMismatchDiagnostics:
    """Tests that pod-name mismatches are diagnosed correctly."""

    def test_pod_mismatch_returns_empty_not_exception(self) -> None:
        """When monitoring returns pods that don't match expected, result is empty dict (no crash)."""
        from vaig.tools.gke.monitoring import get_workload_usage_metrics

        cfg = _make_gke_config()
        mock_client = MagicMock()

        # Monitoring returns "different-pod-xyz", but we expected "api-pod-1"
        cpu_ts = _make_container_ts("different-pod-xyz", "app", [0.2])
        mock_client.list_time_series.side_effect = [[cpu_ts], []]

        with patch("vaig.tools.gke.monitoring.MetricServiceClient", return_value=mock_client):
            result = get_workload_usage_metrics(
                namespace="default",
                workload_pod_names={"api": ["api-pod-1"]},
                gke_config=cfg,
            )

        # No match → empty result (no KeyError, no crash)
        assert result == {}

    def test_pod_mismatch_diagnostic_logged(self) -> None:
        """When monitoring returns pods that don't match expected, diagnostic is logged at DEBUG.

        Uses a direct handler on the vaig logger to avoid pytest caplog propagation
        issues (vaig_logger.propagate=False in log.py prevents root-level capture).
        """
        import logging

        from vaig.tools.gke.monitoring import get_workload_usage_metrics

        cfg = _make_gke_config()
        mock_client = MagicMock()

        # Cloud Monitoring gives us pods with a different naming pattern
        cpu_ts = _make_container_ts("monitoring-pod-abc123", "app", [0.3])
        mock_client.list_time_series.side_effect = [[cpu_ts], []]

        # Attach a handler directly to the vaig logger (propagate=False means
        # caplog/root won't capture it — we must go to the source).
        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        vaig_logger = logging.getLogger("vaig")
        handler = _Capture(level=logging.DEBUG)
        old_level = vaig_logger.level
        vaig_logger.addHandler(handler)
        vaig_logger.setLevel(logging.DEBUG)
        try:
            with patch("vaig.tools.gke.monitoring.MetricServiceClient", return_value=mock_client):
                get_workload_usage_metrics(
                    namespace="default",
                    workload_pod_names={"api": ["expected-pod-1"]},
                    gke_config=cfg,
                )
        finally:
            vaig_logger.removeHandler(handler)
            vaig_logger.setLevel(old_level)

        # Should have logged the mismatch-specific diagnostics at DEBUG
        messages = " ".join(r.getMessage() for r in records).lower()
        assert "pod name mismatch" in messages

    def test_matched_pods_no_mismatch_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """When pods match exactly, no mismatch warning is emitted."""
        import logging

        from vaig.tools.gke.monitoring import get_workload_usage_metrics

        cfg = _make_gke_config()
        mock_client = MagicMock()

        cpu_ts = _make_container_ts("api-pod-1", "app", [0.2])
        mock_client.list_time_series.side_effect = [[cpu_ts], []]

        with caplog.at_level(logging.DEBUG, logger="vaig.tools.gke.monitoring"):
            with patch("vaig.tools.gke.monitoring.MetricServiceClient", return_value=mock_client):
                get_workload_usage_metrics(
                    namespace="default",
                    workload_pod_names={"api": ["api-pod-1"]},
                    gke_config=cfg,
                )

        # Should NOT log mismatch when pods match
        assert "mismatch" not in caplog.text.lower()

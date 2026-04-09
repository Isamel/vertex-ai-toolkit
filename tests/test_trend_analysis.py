"""Tests for GKE anomaly trend detection (SPEC-1.2).

Covers:
- Phase 6: Unit tests for _compute_trend, _classify_severity,
  _project_days_to_threshold, TrendConfig validation, schema roundtrip
- Phase 7: Integration tests for fetch_anomaly_trends with mocked
  MetricServiceClient, graceful degradation, quota exceeded, feature
  disabled, and partial data scenarios
"""

from __future__ import annotations

from io import StringIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from vaig.core.config import TrendConfig
from vaig.skills.service_health.schema import (
    ExecutiveSummary,
    HealthReport,
    MetricTrend,
    OverallStatus,
    ReportMetadata,
    TrendAnalysis,
)

# ── Helpers ───────────────────────────────────────────────────


def _default_trend_config(**overrides: Any) -> TrendConfig:
    """Build a TrendConfig with sensible defaults, applying overrides."""
    defaults: dict[str, Any] = {
        "enabled": True,
        "baseline_days": [7],
        "memory_warning_pct": 10.0,
        "memory_critical_pct": 25.0,
        "cpu_warning_pct": 20.0,
        "cpu_critical_pct": 50.0,
        "restart_warning_count": 5,
        "restart_critical_count": 15,
        "memory_limit_gib": 4.0,
    }
    defaults.update(overrides)
    return TrendConfig(**defaults)


def _make_gke_config(
    *,
    project_id: str = "my-project",
    cluster_name: str = "my-cluster",
    default_namespace: str = "default",
    trend_config: TrendConfig | None = None,
) -> MagicMock:
    """Build a mock GKEConfig for integration tests."""
    cfg = MagicMock()
    cfg.project_id = project_id
    cfg.cluster_name = cluster_name
    cfg.default_namespace = default_namespace
    cfg.trends = trend_config or _default_trend_config()
    return cfg


# ── Phase 6: Unit Tests ──────────────────────────────────────


# ── 6.1 _compute_trend ───────────────────────────────────────


class TestComputeTrend:
    """Test _compute_trend for direction, rate, severity, and edge cases."""

    def test_increasing_memory_triggers_warning(self) -> None:
        """15% memory increase over baseline → warning severity."""
        from vaig.tools.gke.trend_analysis import _compute_trend

        config = _default_trend_config()
        result = _compute_trend(
            metric="kubernetes.io/container/memory/used_bytes",
            service_name="api-server",
            namespace="prod",
            current_avg=1150.0,
            baseline_avg=1000.0,
            config=config,
            window_days=7,
        )
        assert result is not None
        assert result.direction == "increasing"
        assert result.rate_of_change_percent == pytest.approx(15.0, rel=0.01)
        assert result.severity == "warning"
        assert result.service_name == "api-server"
        assert result.namespace == "prod"
        assert result.metric == "memory_usage"
        assert result.baseline_window_days == 7

    def test_increasing_memory_triggers_critical(self) -> None:
        """30% memory increase over baseline → critical severity."""
        from vaig.tools.gke.trend_analysis import _compute_trend

        config = _default_trend_config()
        result = _compute_trend(
            metric="kubernetes.io/container/memory/used_bytes",
            service_name="api-server",
            namespace="prod",
            current_avg=1300.0,
            baseline_avg=1000.0,
            config=config,
            window_days=7,
        )
        assert result is not None
        assert result.direction == "increasing"
        assert result.severity == "critical"

    def test_decreasing_returns_info(self) -> None:
        """Decreasing trend → direction='decreasing', severity='info'."""
        from vaig.tools.gke.trend_analysis import _compute_trend

        config = _default_trend_config()
        result = _compute_trend(
            metric="kubernetes.io/container/memory/used_bytes",
            service_name="api-server",
            namespace="prod",
            current_avg=800.0,
            baseline_avg=1000.0,
            config=config,
            window_days=7,
        )
        assert result is not None
        assert result.direction == "decreasing"
        assert result.rate_of_change_percent == pytest.approx(-20.0, rel=0.01)
        assert result.severity == "info"

    def test_stable_returns_info(self) -> None:
        """<1% change → direction='stable', severity='info'."""
        from vaig.tools.gke.trend_analysis import _compute_trend

        config = _default_trend_config()
        result = _compute_trend(
            metric="kubernetes.io/container/cpu/core_usage_time",
            service_name="worker",
            namespace="default",
            current_avg=1005.0,
            baseline_avg=1000.0,
            config=config,
            window_days=7,
        )
        assert result is not None
        assert result.direction == "stable"
        assert result.severity == "info"

    def test_zero_baseline_returns_new_direction(self) -> None:
        """Zero baseline → direction='new', rate=None, severity='info' (SC-10)."""
        from vaig.tools.gke.trend_analysis import _compute_trend

        config = _default_trend_config()
        result = _compute_trend(
            metric="kubernetes.io/container/memory/used_bytes",
            service_name="new-svc",
            namespace="staging",
            current_avg=500.0,
            baseline_avg=0.0,
            config=config,
            window_days=7,
        )
        assert result is not None
        assert result.direction == "new"
        assert result.rate_of_change_percent is None
        assert result.severity == "info"

    def test_none_baseline_returns_new_direction(self) -> None:
        """None baseline → same as zero baseline."""
        from vaig.tools.gke.trend_analysis import _compute_trend

        config = _default_trend_config()
        result = _compute_trend(
            metric="kubernetes.io/container/memory/used_bytes",
            service_name="new-svc",
            namespace="staging",
            current_avg=500.0,
            baseline_avg=None,
            config=config,
            window_days=14,
        )
        assert result is not None
        assert result.direction == "new"
        assert result.rate_of_change_percent is None
        assert result.baseline_window_days == 14

    def test_both_none_returns_none(self) -> None:
        """Both current and baseline None → returns None."""
        from vaig.tools.gke.trend_analysis import _compute_trend

        config = _default_trend_config()
        result = _compute_trend(
            metric="kubernetes.io/container/cpu/core_usage_time",
            service_name="svc",
            namespace="ns",
            current_avg=None,
            baseline_avg=None,
            config=config,
            window_days=7,
        )
        assert result is None

    def test_none_current_with_baseline_returns_new(self) -> None:
        """None current with valid baseline → current treated as 0.0."""
        from vaig.tools.gke.trend_analysis import _compute_trend

        config = _default_trend_config()
        result = _compute_trend(
            metric="kubernetes.io/container/memory/used_bytes",
            service_name="svc",
            namespace="ns",
            current_avg=None,
            baseline_avg=1000.0,
            config=config,
            window_days=7,
        )
        assert result is not None
        assert result.direction == "decreasing"
        assert result.current_value == 0.0

    def test_increasing_memory_has_days_to_threshold(self) -> None:
        """Increasing memory_usage should project days_to_threshold."""
        from vaig.tools.gke.trend_analysis import _compute_trend

        config = _default_trend_config(memory_limit_gib=4.0)
        # Baseline: 2GiB, Current: 3GiB over 7 days → daily rate = (3-2)/7 GiB/day
        baseline = 2.0 * (1024**3)
        current = 3.0 * (1024**3)
        result = _compute_trend(
            metric="kubernetes.io/container/memory/used_bytes",
            service_name="mem-hog",
            namespace="prod",
            current_avg=current,
            baseline_avg=baseline,
            config=config,
            window_days=7,
        )
        assert result is not None
        assert result.days_to_threshold is not None
        assert result.days_to_threshold > 0
        # Remaining: 4GiB - 3GiB = 1GiB; rate = 1GiB/7days ≈ 0.143 GiB/day → ~7 days
        assert result.days_to_threshold == pytest.approx(7.0, rel=0.01)

    def test_cpu_trend_no_days_to_threshold(self) -> None:
        """CPU trends should NOT have days_to_threshold projection."""
        from vaig.tools.gke.trend_analysis import _compute_trend

        config = _default_trend_config()
        result = _compute_trend(
            metric="kubernetes.io/container/cpu/core_usage_time",
            service_name="cpu-heavy",
            namespace="prod",
            current_avg=1500.0,
            baseline_avg=1000.0,
            config=config,
            window_days=7,
        )
        assert result is not None
        assert result.direction == "increasing"
        assert result.days_to_threshold is None


# ── 6.2 _classify_severity ───────────────────────────────────


class TestClassifySeverity:
    """Test severity classification for each metric type."""

    def test_memory_info(self) -> None:
        from vaig.tools.gke.trend_analysis import _classify_severity

        config = _default_trend_config()
        assert _classify_severity("memory_usage", 5.0, 1050.0, 1000.0, config) == "info"

    def test_memory_warning(self) -> None:
        from vaig.tools.gke.trend_analysis import _classify_severity

        config = _default_trend_config()
        assert _classify_severity("memory_usage", 15.0, 1150.0, 1000.0, config) == "warning"

    def test_memory_critical(self) -> None:
        from vaig.tools.gke.trend_analysis import _classify_severity

        config = _default_trend_config()
        assert _classify_severity("memory_usage", 30.0, 1300.0, 1000.0, config) == "critical"

    def test_memory_at_exact_warning_threshold(self) -> None:
        from vaig.tools.gke.trend_analysis import _classify_severity

        config = _default_trend_config(memory_warning_pct=10.0)
        assert _classify_severity("memory_usage", 10.0, 1100.0, 1000.0, config) == "warning"

    def test_memory_at_exact_critical_threshold(self) -> None:
        from vaig.tools.gke.trend_analysis import _classify_severity

        config = _default_trend_config(memory_critical_pct=25.0)
        assert _classify_severity("memory_usage", 25.0, 1250.0, 1000.0, config) == "critical"

    def test_cpu_info(self) -> None:
        from vaig.tools.gke.trend_analysis import _classify_severity

        config = _default_trend_config()
        assert _classify_severity("cpu_usage", 10.0, 110.0, 100.0, config) == "info"

    def test_cpu_warning(self) -> None:
        from vaig.tools.gke.trend_analysis import _classify_severity

        config = _default_trend_config()
        assert _classify_severity("cpu_usage", 25.0, 125.0, 100.0, config) == "warning"

    def test_cpu_critical(self) -> None:
        from vaig.tools.gke.trend_analysis import _classify_severity

        config = _default_trend_config()
        assert _classify_severity("cpu_usage", 55.0, 155.0, 100.0, config) == "critical"

    def test_restart_info(self) -> None:
        from vaig.tools.gke.trend_analysis import _classify_severity

        config = _default_trend_config()
        assert _classify_severity("restart_count", 50.0, 3.0, 1.0, config) == "info"

    def test_restart_warning(self) -> None:
        from vaig.tools.gke.trend_analysis import _classify_severity

        config = _default_trend_config()
        assert _classify_severity("restart_count", 100.0, 10.0, 5.0, config) == "warning"

    def test_restart_critical(self) -> None:
        from vaig.tools.gke.trend_analysis import _classify_severity

        config = _default_trend_config()
        assert _classify_severity("restart_count", 200.0, 20.0, 5.0, config) == "critical"

    def test_restart_uses_absolute_delta_not_pct(self) -> None:
        """Restart severity is based on absolute delta, not percentage (SC-03)."""
        from vaig.tools.gke.trend_analysis import _classify_severity

        config = _default_trend_config(restart_warning_count=5, restart_critical_count=15)
        # Even though % is huge (400%), absolute delta is only 4 → info
        assert _classify_severity("restart_count", 400.0, 5.0, 1.0, config) == "info"
        # Absolute delta = 6 → warning
        assert _classify_severity("restart_count", 200.0, 7.0, 1.0, config) == "warning"
        # Absolute delta = 16 → critical
        assert _classify_severity("restart_count", 800.0, 17.0, 1.0, config) == "critical"

    def test_unknown_metric_returns_info(self) -> None:
        from vaig.tools.gke.trend_analysis import _classify_severity

        config = _default_trend_config()
        assert _classify_severity("unknown_metric", 50.0, 150.0, 100.0, config) == "info"

    def test_negative_rate_always_info(self) -> None:
        """Decreasing trends should always be info regardless of metric."""
        from vaig.tools.gke.trend_analysis import _classify_severity

        config = _default_trend_config()
        assert _classify_severity("memory_usage", -30.0, 700.0, 1000.0, config) == "info"
        assert _classify_severity("cpu_usage", -50.0, 50.0, 100.0, config) == "info"


# ── 6.3 _project_days_to_threshold ───────────────────────────


class TestProjectDaysToThreshold:
    """Test linear extrapolation to resource limit."""

    def test_known_rate_projects_correctly(self) -> None:
        from vaig.tools.gke.trend_analysis import _project_days_to_threshold

        # Current: 3GiB, rate: 0.5 GiB/day, limit: 4GiB → 2 days
        current = 3.0 * (1024**3)
        rate_per_day = 0.5 * (1024**3)
        limit = 4.0 * (1024**3)
        result = _project_days_to_threshold(current, rate_per_day, limit)
        assert result == pytest.approx(2.0, rel=0.001)

    def test_zero_rate_returns_none(self) -> None:
        from vaig.tools.gke.trend_analysis import _project_days_to_threshold

        result = _project_days_to_threshold(1000.0, 0.0, 2000.0)
        assert result is None

    def test_negative_rate_returns_none(self) -> None:
        from vaig.tools.gke.trend_analysis import _project_days_to_threshold

        result = _project_days_to_threshold(1000.0, -10.0, 2000.0)
        assert result is None

    def test_current_exceeds_limit_returns_none(self) -> None:
        from vaig.tools.gke.trend_analysis import _project_days_to_threshold

        result = _project_days_to_threshold(3000.0, 100.0, 2000.0)
        assert result is None

    def test_current_equals_limit_returns_none(self) -> None:
        from vaig.tools.gke.trend_analysis import _project_days_to_threshold

        result = _project_days_to_threshold(2000.0, 100.0, 2000.0)
        assert result is None

    def test_small_rate_large_projection(self) -> None:
        from vaig.tools.gke.trend_analysis import _project_days_to_threshold

        # 1 byte/day remaining 1000 bytes → 1000 days
        result = _project_days_to_threshold(0.0, 1.0, 1000.0)
        assert result == pytest.approx(1000.0)


# ── 6.4 TrendConfig validation ───────────────────────────────


class TestTrendConfigValidation:
    """Test baseline_days validator (SC-09)."""

    def test_valid_baseline_days(self) -> None:
        cfg = TrendConfig(baseline_days=[7, 14, 30])
        assert cfg.baseline_days == [7, 14, 30]

    def test_single_valid_day(self) -> None:
        cfg = TrendConfig(baseline_days=[1])
        assert cfg.baseline_days == [1]

    def test_max_42_is_valid(self) -> None:
        cfg = TrendConfig(baseline_days=[42])
        assert cfg.baseline_days == [42]

    def test_exceeds_42_raises_validation_error(self) -> None:
        with pytest.raises(Exception, match="42"):
            TrendConfig(baseline_days=[45])

    def test_zero_raises_validation_error(self) -> None:
        with pytest.raises(Exception, match="at least 1"):
            TrendConfig(baseline_days=[0])

    def test_negative_raises_validation_error(self) -> None:
        with pytest.raises(Exception, match="at least 1"):
            TrendConfig(baseline_days=[-5])

    def test_default_baseline_days(self) -> None:
        cfg = TrendConfig()
        assert cfg.baseline_days == [7]


# ── 6.5 Schema roundtrip ─────────────────────────────────────


class TestSchemaRoundtrip:
    """Test ReportMetadata serialization with and without trends (NFR-02)."""

    def test_metadata_without_trends_roundtrips(self) -> None:
        meta = ReportMetadata(
            generated_at="2025-01-01T00:00:00Z",
            cluster_name="test-cluster",
        )
        dumped = meta.model_dump()
        assert dumped["trends"] is None
        restored = ReportMetadata.model_validate(dumped)
        assert restored.trends is None

    def test_metadata_with_trends_roundtrips(self) -> None:
        trend = MetricTrend(
            metric="memory_usage",
            service_name="api-server",
            namespace="prod",
            direction="increasing",
            rate_of_change_percent=15.0,
            current_value=1150.0,
            baseline_value=1000.0,
            baseline_window_days=7,
            days_to_threshold=42.0,
            severity="warning",
        )
        analysis = TrendAnalysis(
            trends=[trend],
            analyzed_at="2025-01-01T00:00:00Z",
            baseline_windows=[7],
            services_analyzed=1,
            anomalies_detected=1,
        )
        meta = ReportMetadata(
            generated_at="2025-01-01T00:00:00Z",
            cluster_name="test-cluster",
            trends=analysis,
        )
        dumped = meta.model_dump()
        assert dumped["trends"] is not None
        assert len(dumped["trends"]["trends"]) == 1

        restored = ReportMetadata.model_validate(dumped)
        assert restored.trends is not None
        assert len(restored.trends.trends) == 1
        assert restored.trends.trends[0].severity == "warning"
        assert restored.trends.anomalies_detected == 1

    def test_metadata_with_empty_trends_roundtrips(self) -> None:
        analysis = TrendAnalysis(
            trends=[],
            analyzed_at="2025-01-01T00:00:00Z",
            baseline_windows=[7],
            services_analyzed=0,
            anomalies_detected=0,
        )
        meta = ReportMetadata(trends=analysis)
        dumped = meta.model_dump()
        restored = ReportMetadata.model_validate(dumped)
        assert restored.trends is not None
        assert len(restored.trends.trends) == 0

    def test_metric_trend_new_direction_serializes(self) -> None:
        """Zero-baseline trend with None rate serializes correctly."""
        trend = MetricTrend(
            metric="memory_usage",
            service_name="new-svc",
            direction="new",
            rate_of_change_percent=None,
            current_value=500.0,
            baseline_value=None,
            severity="info",
        )
        dumped = trend.model_dump()
        assert dumped["rate_of_change_percent"] is None
        assert dumped["baseline_value"] is None
        restored = MetricTrend.model_validate(dumped)
        assert restored.rate_of_change_percent is None
        assert restored.direction == "new"

    def test_health_report_with_trends_serializes(self) -> None:
        """Full HealthReport with trends field round-trips."""
        trend = MetricTrend(
            metric="cpu_usage",
            service_name="worker",
            direction="stable",
            severity="info",
        )
        analysis = TrendAnalysis(trends=[trend], services_analyzed=1)
        meta = ReportMetadata(trends=analysis)
        report = HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.HEALTHY,
                scope="Cluster-wide",
                summary_text="All systems healthy",
            ),
            metadata=meta,
        )
        dumped = report.model_dump()
        assert dumped["metadata"]["trends"]["trends"][0]["metric"] == "cpu_usage"
        restored = HealthReport.model_validate(dumped)
        assert restored.metadata is not None
        assert restored.metadata.trends is not None
        assert restored.metadata.trends.trends[0].metric == "cpu_usage"


# ── Phase 6 (bonus): CLI display test ────────────────────────


class TestPrintTrendAnalysisTable:
    """Test the Rich CLI display for trend analysis."""

    def _capture(self, report: HealthReport) -> str:
        from vaig.cli.display import print_trend_analysis_table

        buf = StringIO()
        con = Console(file=buf, force_terminal=False, width=200)
        print_trend_analysis_table(report, console=con)
        return buf.getvalue()

    def _make_report(self, trends: TrendAnalysis | None = None) -> HealthReport:
        return HealthReport(
            executive_summary=ExecutiveSummary(
                overall_status=OverallStatus.HEALTHY,
                scope="Cluster-wide",
                summary_text="test",
            ),
            metadata=ReportMetadata(trends=trends),
        )

    def test_no_trends_prints_nothing(self) -> None:
        output = self._capture(self._make_report(trends=None))
        assert output.strip() == ""

    def test_empty_trends_prints_nothing(self) -> None:
        analysis = TrendAnalysis(trends=[])
        output = self._capture(self._make_report(trends=analysis))
        assert output.strip() == ""

    def test_displays_trend_rows(self) -> None:
        trend = MetricTrend(
            metric="memory_usage",
            service_name="api-server",
            direction="increasing",
            rate_of_change_percent=15.0,
            current_value=1150.0,
            baseline_value=1000.0,
            days_to_threshold=42.0,
            severity="warning",
        )
        analysis = TrendAnalysis(
            trends=[trend],
            services_analyzed=1,
            anomalies_detected=1,
        )
        output = self._capture(self._make_report(trends=analysis))
        assert "api-server" in output
        assert "memory_usage" in output
        assert "+15.0%" in output
        assert "42d" in output
        assert "WARNING" in output
        assert "Anomaly Trends" in output
        assert "1 anomaly detected across 1 service" in output

    def test_multiple_trends_and_plural_summary(self) -> None:
        trends = [
            MetricTrend(
                metric="memory_usage",
                service_name="svc-a",
                direction="increasing",
                rate_of_change_percent=12.0,
                severity="warning",
            ),
            MetricTrend(
                metric="cpu_usage",
                service_name="svc-b",
                direction="increasing",
                rate_of_change_percent=55.0,
                severity="critical",
            ),
        ]
        analysis = TrendAnalysis(
            trends=trends,
            services_analyzed=2,
            anomalies_detected=2,
        )
        output = self._capture(self._make_report(trends=analysis))
        assert "svc-a" in output
        assert "svc-b" in output
        assert "2 anomalies detected across 2 services" in output

    def test_new_direction_symbol(self) -> None:
        trend = MetricTrend(
            metric="memory_usage",
            service_name="new-svc",
            direction="new",
            rate_of_change_percent=None,
            severity="info",
        )
        analysis = TrendAnalysis(trends=[trend], anomalies_detected=0)
        output = self._capture(self._make_report(trends=analysis))
        assert "new" in output


# ── Phase 7: Integration Tests ────────────────────────────────


def _mock_time_series(double_value: float) -> MagicMock:
    """Create a mock time series with a single data point."""
    point = MagicMock()
    point.value.double_value = double_value
    point.value.int64_value = 0

    ts = MagicMock()
    ts.points = [point]
    return ts


class TestFetchAnomalyTrendsIntegration:
    """Integration tests for fetch_anomaly_trends with mocked Cloud Monitoring (7.1-7.5)."""

    @patch("vaig.tools.gke.trend_analysis._MONITORING_AVAILABLE", True)
    @patch("vaig.tools.gke.trend_analysis.MetricServiceClient")
    def test_normal_3_metric_response(self, mock_client_cls: MagicMock) -> None:
        """Normal response with 3 metrics → TrendAnalysis with correct counts (7.1)."""
        from vaig.tools.gke.trend_analysis import fetch_anomaly_trends

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Return different values for baseline and current queries
        # Each call to list_time_series returns a list of time series
        # With 1 namespace × 3 metrics × 1 baseline window = 6 queries
        # (3 baseline + 3 current)
        call_count = 0

        def mock_list_ts(request: Any) -> list[MagicMock]:
            nonlocal call_count
            call_count += 1
            # Odd calls = baseline (1000), even calls = current (1150) → 15% increase
            if call_count % 2 == 1:
                return [_mock_time_series(1000.0)]
            return [_mock_time_series(1150.0)]

        mock_client.list_time_series.side_effect = mock_list_ts

        gke_cfg = _make_gke_config()
        result = fetch_anomaly_trends(gke_cfg, namespaces=["default"])

        assert result is not None
        assert isinstance(result, TrendAnalysis)
        assert result.services_analyzed == 1
        assert len(result.trends) == 3  # cpu, memory, restart for 1 namespace × 1 window
        assert result.analyzed_at  # non-empty ISO timestamp

    @patch("vaig.tools.gke.trend_analysis._MONITORING_AVAILABLE", True)
    @patch("vaig.tools.gke.trend_analysis.MetricServiceClient")
    def test_graceful_degradation_on_api_error(self, mock_client_cls: MagicMock) -> None:
        """GoogleAPICallError → returns None, no exception (7.2, SC-04)."""
        from vaig.tools.gke.trend_analysis import fetch_anomaly_trends

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.list_time_series.side_effect = Exception("API connection refused")

        gke_cfg = _make_gke_config()
        result = fetch_anomaly_trends(gke_cfg, namespaces=["default"])

        assert result is None

    @patch("vaig.tools.gke.trend_analysis._MONITORING_AVAILABLE", True)
    @patch("vaig.tools.gke.trend_analysis.MetricServiceClient")
    def test_quota_exceeded_returns_none(self, mock_client_cls: MagicMock) -> None:
        """429 ResourceExhausted → returns None with debug log (7.3, SC-05)."""
        from vaig.tools.gke.trend_analysis import fetch_anomaly_trends

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.list_time_series.side_effect = Exception(
            "429 Quota exceeded for quota metric 'monitoring.googleapis.com/read_requests'"
        )

        gke_cfg = _make_gke_config()
        result = fetch_anomaly_trends(gke_cfg, namespaces=["default"])

        assert result is None

    def test_feature_disabled_returns_none_zero_api_calls(self) -> None:
        """TrendConfig(enabled=False) → returns None, zero API calls (7.4, SC-06)."""
        from vaig.tools.gke.trend_analysis import fetch_anomaly_trends

        disabled_config = _default_trend_config(enabled=False)
        gke_cfg = _make_gke_config(trend_config=disabled_config)

        with patch("vaig.tools.gke.trend_analysis.MetricServiceClient") as mock_cls:
            result = fetch_anomaly_trends(gke_cfg)
            mock_cls.assert_not_called()

        assert result is None

    @patch("vaig.tools.gke.trend_analysis._MONITORING_AVAILABLE", True)
    @patch("vaig.tools.gke.trend_analysis.MetricServiceClient")
    def test_partial_data_some_empty_series(self, mock_client_cls: MagicMock) -> None:
        """Some metrics return empty series → analyse available, note gaps (7.5)."""
        from vaig.tools.gke.trend_analysis import fetch_anomaly_trends

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        call_count = 0

        def mock_list_ts(request: Any) -> list[MagicMock]:
            nonlocal call_count
            call_count += 1
            # Return data only for first 2 queries (first metric), rest empty
            if call_count <= 2:
                return [_mock_time_series(1000.0 if call_count == 1 else 1200.0)]
            return []  # empty — no data for other metrics

        mock_client.list_time_series.side_effect = mock_list_ts

        gke_cfg = _make_gke_config()
        result = fetch_anomaly_trends(gke_cfg, namespaces=["default"])

        assert result is not None
        # Only the first metric should produce a trend (the rest have both None → skip)
        assert len(result.trends) >= 1
        assert result.services_analyzed >= 1

    @patch("vaig.tools.gke.trend_analysis._MONITORING_AVAILABLE", True)
    @patch("vaig.tools.gke.trend_analysis.MetricServiceClient")
    def test_multiple_namespaces(self, mock_client_cls: MagicMock) -> None:
        """Multiple namespaces produce trends for each namespace."""
        from vaig.tools.gke.trend_analysis import fetch_anomaly_trends

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        call_count = 0

        def mock_list_ts(request: Any) -> list[MagicMock]:
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 1:
                return [_mock_time_series(1000.0)]
            return [_mock_time_series(1300.0)]

        mock_client.list_time_series.side_effect = mock_list_ts

        gke_cfg = _make_gke_config()
        result = fetch_anomaly_trends(gke_cfg, namespaces=["ns-a", "ns-b"])

        assert result is not None
        assert result.services_analyzed == 2
        # 2 namespaces × 3 metrics × 1 window = 6 trends
        assert len(result.trends) == 6

    @patch("vaig.tools.gke.trend_analysis._MONITORING_AVAILABLE", False)
    def test_monitoring_unavailable_returns_none(self) -> None:
        """Missing Cloud Monitoring library → returns None."""
        from vaig.tools.gke.trend_analysis import fetch_anomaly_trends

        gke_cfg = _make_gke_config()
        result = fetch_anomaly_trends(gke_cfg, namespaces=["default"])
        assert result is None

    @patch("vaig.tools.gke.trend_analysis._MONITORING_AVAILABLE", True)
    @patch("vaig.tools.gke.trend_analysis.MetricServiceClient")
    def test_missing_project_id_returns_none(self, mock_client_cls: MagicMock) -> None:
        """Empty project_id → returns None without calling API."""
        from vaig.tools.gke.trend_analysis import fetch_anomaly_trends

        gke_cfg = _make_gke_config(project_id="")
        result = fetch_anomaly_trends(gke_cfg)
        mock_client_cls.assert_not_called()
        assert result is None

    @patch("vaig.tools.gke.trend_analysis._MONITORING_AVAILABLE", True)
    @patch("vaig.tools.gke.trend_analysis.MetricServiceClient")
    def test_client_creation_failure_returns_none(self, mock_client_cls: MagicMock) -> None:
        """MetricServiceClient() raises → returns None gracefully."""
        from vaig.tools.gke.trend_analysis import fetch_anomaly_trends

        mock_client_cls.side_effect = Exception("credentials not found")

        gke_cfg = _make_gke_config()
        result = fetch_anomaly_trends(gke_cfg, namespaces=["default"])
        assert result is None

    @patch("vaig.tools.gke.trend_analysis._MONITORING_AVAILABLE", True)
    @patch("vaig.tools.gke.trend_analysis.MetricServiceClient")
    def test_anomaly_count_matches_warning_critical(self, mock_client_cls: MagicMock) -> None:
        """anomalies_detected should count only warning + critical trends."""
        from vaig.tools.gke.trend_analysis import fetch_anomaly_trends

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        call_count = 0

        def mock_list_ts(request: Any) -> list[MagicMock]:
            nonlocal call_count
            call_count += 1
            # memory baseline=1000, current=1300 → 30% increase → critical
            # cpu baseline=1000, current=1050 → 5% increase → info (below 20% threshold)
            # restart baseline=1000, current=1010 → info
            idx = ((call_count - 1) // 2) % 3  # metric index: 0=cpu, 1=memory, 2=restart
            if call_count % 2 == 1:
                return [_mock_time_series(1000.0)]
            if idx == 0:  # cpu current
                return [_mock_time_series(1050.0)]
            if idx == 1:  # memory current
                return [_mock_time_series(1300.0)]
            return [_mock_time_series(1010.0)]  # restart current

        mock_client.list_time_series.side_effect = mock_list_ts

        gke_cfg = _make_gke_config()
        result = fetch_anomaly_trends(gke_cfg, namespaces=["default"])

        assert result is not None
        severities = [t.severity for t in result.trends]
        expected_anomalies = sum(1 for s in severities if s in ("warning", "critical"))
        assert result.anomalies_detected == expected_anomalies

"""Tests for Datadog REST API tools — metrics, monitors, and APM services."""

from __future__ import annotations

from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from vaig.core.config import DatadogAPIConfig

# ── sys.modules fake hierarchy ───────────────────────────────


def _make_dd_modules() -> dict[str, ModuleType | None]:
    """Create a fake ``datadog_api_client`` module hierarchy for sys.modules patching.

    This lets tests run without the real package installed. Each call returns a
    fresh set of mocks so tests remain independent.
    """
    # Root package
    dd_pkg = ModuleType("datadog_api_client")
    api_client_cls = MagicMock(name="ApiClient")
    ctx_mgr = MagicMock()
    api_client_cls.return_value.__enter__ = MagicMock(return_value=ctx_mgr)
    api_client_cls.return_value.__exit__ = MagicMock(return_value=False)
    dd_pkg.ApiClient = api_client_cls
    dd_pkg.Configuration = MagicMock(name="Configuration")

    # Exceptions module
    exceptions_mod = ModuleType("datadog_api_client.exceptions")
    api_exc_cls = type("ApiException", (Exception,), {"status": 0, "reason": ""})
    exceptions_mod.ApiException = api_exc_cls  # type: ignore[attr-defined]

    # v1 hierarchy
    v1_mod = ModuleType("datadog_api_client.v1")
    v1_api_mod = ModuleType("datadog_api_client.v1.api")

    metrics_api_mod = ModuleType("datadog_api_client.v1.api.metrics_api")
    metrics_api_mod.MetricsApi = MagicMock(name="MetricsApi")  # type: ignore[attr-defined]

    monitors_api_mod = ModuleType("datadog_api_client.v1.api.monitors_api")
    monitors_api_mod.MonitorsApi = MagicMock(name="MonitorsApi")  # type: ignore[attr-defined]

    # v2 hierarchy
    v2_mod = ModuleType("datadog_api_client.v2")
    v2_api_mod = ModuleType("datadog_api_client.v2.api")

    svc_def_api_mod = ModuleType("datadog_api_client.v2.api.service_definition_api")
    svc_def_api_mod.ServiceDefinitionApi = MagicMock(name="ServiceDefinitionApi")  # type: ignore[attr-defined]

    return {
        "datadog_api_client": dd_pkg,
        "datadog_api_client.exceptions": exceptions_mod,
        "datadog_api_client.v1": v1_mod,
        "datadog_api_client.v1.api": v1_api_mod,
        "datadog_api_client.v1.api.metrics_api": metrics_api_mod,
        "datadog_api_client.v1.api.monitors_api": monitors_api_mod,
        "datadog_api_client.v2": v2_mod,
        "datadog_api_client.v2.api": v2_api_mod,
        "datadog_api_client.v2.api.service_definition_api": svc_def_api_mod,
    }


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def dd_config() -> DatadogAPIConfig:
    """Return an enabled DatadogAPIConfig for tests."""
    return DatadogAPIConfig(
        enabled=True,
        api_key="test-api-key",
        app_key="test-app-key",
        site="datadoghq.com",
    )


@pytest.fixture()
def dd_config_disabled() -> DatadogAPIConfig:
    """Return a disabled DatadogAPIConfig for tests."""
    return DatadogAPIConfig(enabled=False)


# ── Data builders ─────────────────────────────────────────────


def _make_series(
    metric: str = "kubernetes.cpu.usage.total",
    scope: str = "cluster_name:my-cluster",
    points: list | None = None,
) -> MagicMock:
    """Build a mock Datadog metrics series object."""
    s = MagicMock()
    s.metric = metric
    s.scope = scope
    s.pointlist = points if points is not None else [[1700000000, 42.5], [1700000060, 44.0]]
    return s


def _make_monitor(
    mid: int = 1,
    name: str = "CPU High Alert",
    mtype: str = "metric alert",
    overall_state: str = "Alert",
) -> MagicMock:
    """Build a mock Datadog monitor object."""
    m = MagicMock()
    m.id = mid
    m.name = name
    m.type = mtype
    m.overall_state = overall_state
    return m


def _make_service(
    svc_name: str = "frontend",
    team: str = "platform",
    language: str = "python",
    tier: str = "critical",
) -> MagicMock:
    """Build a mock Datadog APM service definition object."""
    schema_dict = {
        "dd-service": svc_name,
        "team": team,
        "languages": [language],
        "tier": tier,
    }
    schema = MagicMock()
    schema.to_dict.return_value = schema_dict

    attrs = MagicMock()
    attrs.schema = schema

    svc = MagicMock()
    svc.attributes = attrs
    return svc


# ── query_datadog_metrics ─────────────────────────────────────


class TestQueryDatadogMetrics:
    """Tests for query_datadog_metrics."""

    def test_returns_metrics_with_template(self, dd_config: DatadogAPIConfig) -> None:
        """Built-in cpu template returns formatted series output."""
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        mock_api = MagicMock()
        mock_api.query_metrics.return_value = MagicMock(
            series=[_make_series(points=[[1700000000, 10.0], [1700000060, 20.0]])]
        )

        with patch.dict("sys.modules", _make_dd_modules()):
            result = query_datadog_metrics(
                cluster_name="my-cluster",
                metric="cpu",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        assert "=== Datadog Metrics: cpu ===" in result.output
        assert "my-cluster" in result.output
        assert "Total series: 1" in result.output

    def test_no_data_returns_no_data_message(self, dd_config: DatadogAPIConfig) -> None:
        """Empty series list returns a 'no data' message, not an error."""
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        mock_api = MagicMock()
        mock_api.query_metrics.return_value = MagicMock(series=[])

        with patch.dict("sys.modules", _make_dd_modules()):
            result = query_datadog_metrics(
                cluster_name="my-cluster",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        assert "No data returned" in result.output

    def test_unknown_metric_template_returns_error(self, dd_config: DatadogAPIConfig) -> None:
        """Unknown metric template name returns a descriptive error."""
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        with patch.dict("sys.modules", _make_dd_modules()):
            result = query_datadog_metrics(
                cluster_name="my-cluster",
                metric="nonexistent_metric",
                config=dd_config,
            )

        assert result.error is True
        assert "Unknown metric template" in result.output
        assert "nonexistent_metric" in result.output

    def test_disabled_config_returns_error(self, dd_config_disabled: DatadogAPIConfig) -> None:
        """Disabled config returns an error without calling the API."""
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        with patch.dict("sys.modules", _make_dd_modules()):
            result = query_datadog_metrics(
                cluster_name="my-cluster",
                config=dd_config_disabled,
            )

        assert result.error is True
        assert "disabled" in result.output.lower()

    def test_api_401_returns_auth_error(self) -> None:
        """HTTP 401 from the API maps to a human-readable auth error message."""
        from vaig.tools.gke.datadog_api import _dd_error_message

        assert "Authentication" in _dd_error_message(401)
        assert "permissions" in _dd_error_message(403).lower()
        assert "Rate limit" in _dd_error_message(429)
        assert "HTTP 500" in _dd_error_message(500)

    def test_import_error_returns_install_message(self, dd_config: DatadogAPIConfig) -> None:
        """Missing datadog-api-client returns install instructions."""
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        null_modules: dict[str, ModuleType | None] = {
            "datadog_api_client": None,
            "datadog_api_client.exceptions": None,
            "datadog_api_client.v1": None,
            "datadog_api_client.v1.api": None,
            "datadog_api_client.v1.api.metrics_api": None,
        }
        with patch.dict("sys.modules", null_modules):
            result = query_datadog_metrics(
                cluster_name="my-cluster",
                config=dd_config,
            )

        assert result.error is True
        assert "pip install" in result.output

    def test_sanitizes_cluster_name(self) -> None:
        """Cluster name with special chars raises ValueError (fail-fast)."""
        from vaig.tools.gke.datadog_api import _sanitize_service_name

        assert _sanitize_service_name("my-cluster.prod") == "my-cluster.prod"
        assert _sanitize_service_name("valid_name-123") == "valid_name-123"
        with pytest.raises(ValueError, match="Invalid service name"):
            _sanitize_service_name("my cluster!")


# ── get_datadog_monitors ──────────────────────────────────────


class TestGetDatadogMonitors:
    """Tests for get_datadog_monitors."""

    def test_returns_alerting_monitors(self, dd_config: DatadogAPIConfig) -> None:
        """Returns monitors in Alert state with tabular output."""
        from vaig.tools.gke.datadog_api import get_datadog_monitors

        mock_api = MagicMock()
        mock_api.list_monitors.return_value = [
            _make_monitor(mid=1, name="CPU High Alert", overall_state="Alert"),
            _make_monitor(mid=2, name="Memory OK", overall_state="OK"),
            _make_monitor(mid=3, name="Disk Alert", mtype="metric alert", overall_state="Alert"),
        ]

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_monitors(
                cluster_name="my-cluster",
                state="Alert",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        assert "CPU High Alert" in result.output
        assert "Disk Alert" in result.output
        assert "Memory OK" not in result.output
        assert "Total monitors scanned: 3" in result.output
        assert "Monitors in 'Alert' state: 2" in result.output

    def test_no_monitors_matching_state(self, dd_config: DatadogAPIConfig) -> None:
        """Returns appropriate message when no monitors match the requested state."""
        from vaig.tools.gke.datadog_api import get_datadog_monitors

        mock_api = MagicMock()
        mock_api.list_monitors.return_value = [
            _make_monitor(overall_state="OK"),
        ]

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_monitors(
                state="Alert",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        assert "No monitors in 'Alert' state" in result.output

    def test_empty_monitor_list(self, dd_config: DatadogAPIConfig) -> None:
        """Empty monitor list returns 'no monitors found' message."""
        from vaig.tools.gke.datadog_api import get_datadog_monitors

        mock_api = MagicMock()
        mock_api.list_monitors.return_value = []

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_monitors(
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        assert "No monitors found" in result.output

    def test_cluster_name_added_to_tag_filter(self, dd_config: DatadogAPIConfig) -> None:
        """cluster_name is appended to tag filter when provided."""
        from vaig.tools.gke.datadog_api import get_datadog_monitors

        mock_api = MagicMock()
        mock_api.list_monitors.return_value = []

        with patch.dict("sys.modules", _make_dd_modules()):
            get_datadog_monitors(
                cluster_name="prod-cluster",
                config=dd_config,
                _custom_api=mock_api,
            )

        call_kwargs = mock_api.list_monitors.call_args.kwargs
        assert "cluster_name:prod-cluster" in call_kwargs.get("monitor_tags", "")

    def test_disabled_config_returns_error(self, dd_config_disabled: DatadogAPIConfig) -> None:
        """Disabled config returns error without calling API."""
        from vaig.tools.gke.datadog_api import get_datadog_monitors

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_monitors(config=dd_config_disabled)

        assert result.error is True
        assert "disabled" in result.output.lower()

    def test_truncates_long_monitor_name(self, dd_config: DatadogAPIConfig) -> None:
        """Monitor names longer than 49 characters are truncated with ellipsis."""
        from vaig.tools.gke.datadog_api import get_datadog_monitors

        long_name = "A" * 60
        mock_api = MagicMock()
        mock_api.list_monitors.return_value = [
            _make_monitor(name=long_name, overall_state="Alert"),
        ]

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_monitors(
                state="Alert",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        assert "..." in result.output


# ── get_datadog_apm_services ──────────────────────────────────


class TestGetDatadogApmServices:
    """Tests for get_datadog_apm_services."""

    def setup_method(self) -> None:
        """Clear the discovery cache before each test."""
        from vaig.tools.gke._cache import clear_discovery_cache

        clear_discovery_cache()

    def test_returns_service_list(self, dd_config: DatadogAPIConfig) -> None:
        """Returns formatted table of APM services."""
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        mock_api = MagicMock()
        mock_api.list_service_definitions.return_value = MagicMock(
            data=[
                _make_service("frontend", "platform", "python", "critical"),
                _make_service("backend", "core", "go", "high"),
            ]
        )

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_apm_services(
                env="production",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        assert "=== Datadog APM Services ===" in result.output
        assert "frontend" in result.output
        assert "backend" in result.output
        assert "Total services: 2" in result.output

    def test_no_services_returns_appropriate_message(self, dd_config: DatadogAPIConfig) -> None:
        """Empty service list returns 'no services found' message."""
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        mock_api = MagicMock()
        mock_api.list_service_definitions.return_value = MagicMock(data=[])

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_apm_services(
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        assert "No APM service definitions found" in result.output

    def test_result_is_cached(self, dd_config: DatadogAPIConfig) -> None:
        """Second call with same args hits the cache and does NOT call the API again."""
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        mock_api = MagicMock()
        mock_api.list_service_definitions.return_value = MagicMock(data=[_make_service()])

        with patch.dict("sys.modules", _make_dd_modules()):
            # First call — populates cache
            result1 = get_datadog_apm_services(
                env="production",
                cluster_name="my-cluster",
                config=dd_config,
                _custom_api=mock_api,
            )

            # Second call — should return from cache WITHOUT calling the API again
            result2 = get_datadog_apm_services(
                env="production",
                cluster_name="my-cluster",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result1.output == result2.output
        assert mock_api.list_service_definitions.call_count == 1

    def test_different_env_not_cached(self, dd_config: DatadogAPIConfig) -> None:
        """Different env values use separate cache keys."""
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        mock_api = MagicMock()
        mock_api.list_service_definitions.return_value = MagicMock(data=[_make_service()])

        with patch.dict("sys.modules", _make_dd_modules()):
            get_datadog_apm_services(env="production", config=dd_config, _custom_api=mock_api)
            get_datadog_apm_services(env="staging", config=dd_config, _custom_api=mock_api)

        assert mock_api.list_service_definitions.call_count == 2

    def test_disabled_config_returns_error(self, dd_config_disabled: DatadogAPIConfig) -> None:
        """Disabled config returns error without calling API."""
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_apm_services(config=dd_config_disabled)

        assert result.error is True
        assert "disabled" in result.output.lower()

    def test_cluster_name_shown_in_output(self, dd_config: DatadogAPIConfig) -> None:
        """Cluster name appears in output when provided."""
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        mock_api = MagicMock()
        mock_api.list_service_definitions.return_value = MagicMock(data=[])

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_apm_services(
                env="production",
                cluster_name="prod-cluster",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert "prod-cluster" in result.output

    def test_import_error_returns_install_message(self, dd_config: DatadogAPIConfig) -> None:
        """Missing datadog-api-client returns install instructions."""
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        null_modules: dict[str, ModuleType | None] = {
            "datadog_api_client": None,
            "datadog_api_client.exceptions": None,
            "datadog_api_client.v2": None,
            "datadog_api_client.v2.api": None,
            "datadog_api_client.v2.api.service_definition_api": None,
        }
        with patch.dict("sys.modules", null_modules):
            result = get_datadog_apm_services(config=dd_config)

        assert result.error is True
        assert "pip install" in result.output


# ── _sanitize_service_name ────────────────────────────────────


class TestSanitizeServiceName:
    """Unit tests for the _sanitize_service_name helper."""

    def test_allows_alphanumeric(self) -> None:
        from vaig.tools.gke.datadog_api import _sanitize_service_name

        assert _sanitize_service_name("abc123") == "abc123"

    def test_allows_hyphen_underscore_dot(self) -> None:
        from vaig.tools.gke.datadog_api import _sanitize_service_name

        assert _sanitize_service_name("my-service_v1.2") == "my-service_v1.2"

    def test_raises_on_spaces_and_special_chars(self) -> None:
        from vaig.tools.gke.datadog_api import _sanitize_service_name

        with pytest.raises(ValueError, match="Invalid service name"):
            _sanitize_service_name("my cluster!@#")

    def test_empty_string(self) -> None:
        from vaig.tools.gke.datadog_api import _sanitize_service_name

        assert _sanitize_service_name("") == ""

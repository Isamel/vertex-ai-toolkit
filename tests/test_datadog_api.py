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


# ── DatadogAPIConfig auto-enable validator ────────────────────


class TestDatadogAPIConfigAutoEnable:
    """Tests for DatadogAPIConfig model_validator auto-enable/disable logic."""

    def test_auto_enables_when_both_keys_provided(self) -> None:
        """When enabled=False but both api_key and app_key are set, enabled is auto-set to True."""
        cfg = DatadogAPIConfig(enabled=False, api_key="my-api-key", app_key="my-app-key")
        assert cfg.enabled is True

    def test_auto_disables_when_enabled_but_no_keys(self) -> None:
        """When enabled=True but api_key or app_key is missing, enabled is set to False."""
        cfg = DatadogAPIConfig(enabled=True, api_key="", app_key="")
        assert cfg.enabled is False

    def test_auto_disables_when_only_api_key_provided(self) -> None:
        """When enabled=True but only api_key is set (no app_key), disables."""
        cfg = DatadogAPIConfig(enabled=True, api_key="my-api-key", app_key="")
        assert cfg.enabled is False

    def test_auto_disables_when_only_app_key_provided(self) -> None:
        """When enabled=True but only app_key is set (no api_key), disables."""
        cfg = DatadogAPIConfig(enabled=True, api_key="", app_key="my-app-key")
        assert cfg.enabled is False

    def test_stays_false_when_no_keys_and_disabled(self) -> None:
        """When enabled=False and no keys, stays False (no change)."""
        cfg = DatadogAPIConfig(enabled=False, api_key="", app_key="")
        assert cfg.enabled is False

    def test_stays_true_when_both_keys_and_enabled(self) -> None:
        """When enabled=True and both keys provided, stays True (no change)."""
        cfg = DatadogAPIConfig(enabled=True, api_key="my-api-key", app_key="my-app-key")
        assert cfg.enabled is True

    def test_auto_disables_emits_warning(self, caplog) -> None:
        """When auto-disabling, a WARNING is logged about missing keys."""
        import logging

        with patch.object(logging.getLogger("vaig"), "propagate", True):
            with caplog.at_level(logging.WARNING, logger="vaig.core.config"):
                DatadogAPIConfig(enabled=True, api_key="", app_key="")

        assert len(caplog.records) == 1
        assert "api_key or app_key is missing" in caplog.text


# ── Label-aware filtering ─────────────────────────────────────


class TestQueryDatadogMetricsLabelFilters:
    """Tests for the new service/env filter parameters on query_datadog_metrics."""

    def test_service_and_env_appear_in_query(self, dd_config: DatadogAPIConfig) -> None:
        """When service and env are provided, they appear as tags in the metric query."""
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        mock_api = MagicMock()
        mock_api.query_metrics.return_value = MagicMock(series=[])

        with patch.dict("sys.modules", _make_dd_modules()):
            query_datadog_metrics(
                cluster_name="my-cluster",
                metric="cpu",
                service="my-api",
                env="production",
                config=dd_config,
                _custom_api=mock_api,
            )

        call_kwargs = mock_api.query_metrics.call_args.kwargs
        query_str = call_kwargs.get("query", "")
        assert "service:my-api" in query_str
        assert "env:production" in query_str
        assert "cluster_name:my-cluster" in query_str

    def test_backward_compat_no_service_no_env(self, dd_config: DatadogAPIConfig) -> None:
        """Without service/env params, query only contains cluster_name filter."""
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        mock_api = MagicMock()
        mock_api.query_metrics.return_value = MagicMock(series=[])

        with patch.dict("sys.modules", _make_dd_modules()):
            query_datadog_metrics(
                cluster_name="my-cluster",
                metric="cpu",
                config=dd_config,
                _custom_api=mock_api,
            )

        call_kwargs = mock_api.query_metrics.call_args.kwargs
        query_str = call_kwargs.get("query", "")
        assert "cluster_name:my-cluster" in query_str
        assert "service:" not in query_str
        assert "env:" not in query_str

    def test_only_service_filter(self, dd_config: DatadogAPIConfig) -> None:
        """Providing only service (no env) includes just cluster and service tags."""
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        mock_api = MagicMock()
        mock_api.query_metrics.return_value = MagicMock(series=[])

        with patch.dict("sys.modules", _make_dd_modules()):
            query_datadog_metrics(
                cluster_name="my-cluster",
                metric="memory",
                service="worker",
                config=dd_config,
                _custom_api=mock_api,
            )

        call_kwargs = mock_api.query_metrics.call_args.kwargs
        query_str = call_kwargs.get("query", "")
        assert "service:worker" in query_str
        assert "env:" not in query_str

    def test_invalid_service_name_returns_error(self, dd_config: DatadogAPIConfig) -> None:
        """Service name with invalid characters returns an error without calling the API."""
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        mock_api = MagicMock()

        with patch.dict("sys.modules", _make_dd_modules()):
            result = query_datadog_metrics(
                cluster_name="my-cluster",
                service="bad service!",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is True
        assert "Invalid service" in result.output
        mock_api.query_metrics.assert_not_called()


class TestGetDatadogMonitorsLabelFilters:
    """Tests for the new service/env filter parameters on get_datadog_monitors."""

    def test_service_and_env_added_to_monitor_tags(self, dd_config: DatadogAPIConfig) -> None:
        """When service and env are provided, they appear in the monitor_tags filter."""
        from vaig.tools.gke.datadog_api import get_datadog_monitors

        mock_api = MagicMock()
        mock_api.list_monitors.return_value = []

        with patch.dict("sys.modules", _make_dd_modules()):
            get_datadog_monitors(
                cluster_name="prod-cluster",
                service="my-api",
                env="production",
                config=dd_config,
                _custom_api=mock_api,
            )

        call_kwargs = mock_api.list_monitors.call_args.kwargs
        tags = call_kwargs.get("monitor_tags", "")
        assert "service:my-api" in tags
        assert "env:production" in tags
        assert "cluster_name:prod-cluster" in tags

    def test_backward_compat_no_service_no_env(self, dd_config: DatadogAPIConfig) -> None:
        """Without service/env, monitor_tags only contains cluster_name (when provided)."""
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
        tags = call_kwargs.get("monitor_tags", "")
        assert "cluster_name:prod-cluster" in tags
        assert "service:" not in tags
        assert "env:" not in tags

    def test_no_cluster_service_env_omits_monitor_tags(self, dd_config: DatadogAPIConfig) -> None:
        """When no cluster, service, or env are provided, monitor_tags kwarg is omitted entirely."""
        from vaig.tools.gke.datadog_api import get_datadog_monitors

        mock_api = MagicMock()
        mock_api.list_monitors.return_value = []

        with patch.dict("sys.modules", _make_dd_modules()):
            get_datadog_monitors(
                config=dd_config,
                _custom_api=mock_api,
            )

        call_kwargs = mock_api.list_monitors.call_args.kwargs
        assert "monitor_tags" not in call_kwargs

    def test_invalid_env_returns_error(self, dd_config: DatadogAPIConfig) -> None:
        """Env value with invalid characters returns an error without calling the API."""
        from vaig.tools.gke.datadog_api import get_datadog_monitors

        mock_api = MagicMock()

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_monitors(
                env="prod env!",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is True
        assert "Invalid env" in result.output
        mock_api.list_monitors.assert_not_called()


class TestGetDatadogApmServicesLabelFilters:
    """Tests for the new service_name filter parameter on get_datadog_apm_services."""

    def setup_method(self) -> None:
        """Clear the discovery cache before each test."""
        from vaig.tools.gke._cache import clear_discovery_cache

        clear_discovery_cache()

    def test_service_name_filter_returns_only_matching_service(
        self, dd_config: DatadogAPIConfig
    ) -> None:
        """When service_name is provided, only the matching service is returned."""
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        mock_api = MagicMock()
        mock_api.list_service_definitions.return_value = MagicMock(
            data=[
                _make_service("frontend", "platform", "python", "critical"),
                _make_service("backend", "core", "go", "high"),
                _make_service("worker", "infra", "python", "low"),
            ]
        )

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_apm_services(
                env="production",
                service_name="frontend",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        assert "frontend" in result.output
        assert "backend" not in result.output
        assert "worker" not in result.output
        assert "Service filter: frontend" in result.output
        assert "Total services: 1" in result.output

    def test_service_name_filter_no_match_returns_no_services(
        self, dd_config: DatadogAPIConfig
    ) -> None:
        """When service_name matches nothing, returns 'no APM service definitions found'."""
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        mock_api = MagicMock()
        mock_api.list_service_definitions.return_value = MagicMock(
            data=[
                _make_service("frontend", "platform", "python", "critical"),
            ]
        )

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_apm_services(
                service_name="nonexistent-service",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        assert "No APM service definitions found" in result.output

    def test_backward_compat_no_service_name_returns_all(
        self, dd_config: DatadogAPIConfig
    ) -> None:
        """Without service_name, all services are returned (backward compat)."""
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
        assert "frontend" in result.output
        assert "backend" in result.output
        assert "Total services: 2" in result.output
        assert "Service filter:" not in result.output

    def test_service_name_uses_separate_cache_key(self, dd_config: DatadogAPIConfig) -> None:
        """Calls with and without service_name use different cache keys (no cross-contamination)."""
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        mock_api = MagicMock()
        mock_api.list_service_definitions.return_value = MagicMock(
            data=[
                _make_service("frontend"),
                _make_service("backend"),
            ]
        )

        with patch.dict("sys.modules", _make_dd_modules()):
            # Unfiltered call
            get_datadog_apm_services(env="production", config=dd_config, _custom_api=mock_api)
            # Filtered call — different cache key, must hit the API again
            get_datadog_apm_services(
                env="production", service_name="frontend", config=dd_config, _custom_api=mock_api
            )

        assert mock_api.list_service_definitions.call_count == 2


# ── Configurable labels — _build_tag_filter ───────────────────


class TestConfigurableLabelsTagFilter:
    """Tests for configurable tag key names in _build_tag_filter."""

    def test_custom_service_label_used_in_tag_filter(self) -> None:
        """When config.labels.service is overridden, the custom key name appears in the filter."""
        from vaig.core.config import DatadogLabelConfig
        from vaig.tools.gke.datadog_api import _build_tag_filter

        config = DatadogAPIConfig(
            enabled=True,
            api_key="k",
            app_key="k",
            labels=DatadogLabelConfig(service="svc"),
        )
        tag_filter, err = _build_tag_filter("my-cluster", "my-api", None, config)

        assert err is None
        assert "svc:my-api" in tag_filter
        assert "service:my-api" not in tag_filter

    def test_custom_cluster_name_label_used_in_tag_filter(self) -> None:
        """When config.labels.cluster_name is overridden, the custom key appears in the filter."""
        from vaig.core.config import DatadogLabelConfig
        from vaig.tools.gke.datadog_api import _build_tag_filter

        config = DatadogAPIConfig(
            enabled=True,
            api_key="k",
            app_key="k",
            labels=DatadogLabelConfig(cluster_name="kube_cluster"),
        )
        tag_filter, err = _build_tag_filter("prod-cluster", None, None, config)

        assert err is None
        assert "kube_cluster:prod-cluster" in tag_filter
        assert "cluster_name:prod-cluster" not in tag_filter

    def test_custom_env_label_used_in_tag_filter(self) -> None:
        """When config.labels.env is overridden, the custom key name appears in the filter."""
        from vaig.core.config import DatadogLabelConfig
        from vaig.tools.gke.datadog_api import _build_tag_filter

        config = DatadogAPIConfig(
            enabled=True,
            api_key="k",
            app_key="k",
            labels=DatadogLabelConfig(env="environment"),
        )
        tag_filter, err = _build_tag_filter(None, None, "production", config)

        assert err is None
        assert "environment:production" in tag_filter
        assert "env:production" not in tag_filter

    def test_custom_labels_dict_appended_to_filter(self) -> None:
        """custom label entries from config.labels.custom are appended to the tag filter."""
        from vaig.core.config import DatadogLabelConfig
        from vaig.tools.gke.datadog_api import _build_tag_filter

        config = DatadogAPIConfig(
            enabled=True,
            api_key="k",
            app_key="k",
            labels=DatadogLabelConfig(custom={"team": "platform", "region": "us-east"}),
        )
        tag_filter, err = _build_tag_filter("my-cluster", None, None, config)

        assert err is None
        assert "cluster_name:my-cluster" in tag_filter
        assert "team:platform" in tag_filter
        assert "region:us-east" in tag_filter

    def test_default_labels_used_when_config_is_none(self) -> None:
        """When config=None, standard Datadog tag names are used as defaults."""
        from vaig.tools.gke.datadog_api import _build_tag_filter

        tag_filter, err = _build_tag_filter("my-cluster", "my-svc", "prod", None)

        assert err is None
        assert "cluster_name:my-cluster" in tag_filter
        assert "service:my-svc" in tag_filter
        assert "env:prod" in tag_filter


# ── Configurable labels — _build_metric_templates ────────────


class TestConfigurableLabelsMetricTemplates:
    """Tests for configurable label names and custom metrics in _build_metric_templates."""

    def test_custom_pod_name_label_used_in_grouping(self) -> None:
        """When config.labels.pod_name is overridden, the by {} grouping uses the custom name."""
        from vaig.core.config import DatadogLabelConfig
        from vaig.tools.gke.datadog_api import _build_metric_templates

        config = DatadogAPIConfig(
            enabled=True,
            api_key="k",
            app_key="k",
            labels=DatadogLabelConfig(pod_name="pod"),
        )
        templates = _build_metric_templates(config)

        # Templates use {{ }} so the braces remain literal in the final Datadog query string.
        # "by {{pod}}" in the raw template renders to "by {pod}" after .format(filters=...).
        assert "{{pod}}" in templates["cpu"]
        assert "{{pod_name}}" not in templates["cpu"]

    def test_default_pod_name_in_metric_templates(self) -> None:
        """Default config uses 'pod_name' as the grouping dimension in all built-in templates."""
        from vaig.tools.gke.datadog_api import _build_metric_templates

        config = DatadogAPIConfig(enabled=True, api_key="k", app_key="k")
        templates = _build_metric_templates(config)

        for tmpl_name, tmpl_str in templates.items():
            # Templates use {{ }} so braces are literal; {{pod_name}} renders to {pod_name}.
            assert "{{pod_name}}" in tmpl_str, f"Template '{tmpl_name}' missing '{{pod_name}}' grouping"

    def test_custom_metrics_merged_into_templates(self) -> None:
        """Entries in config.custom_metrics are merged alongside the built-in templates."""
        from vaig.tools.gke.datadog_api import _build_metric_templates

        config = DatadogAPIConfig(
            enabled=True,
            api_key="k",
            app_key="k",
            custom_metrics={"my_metric": "avg:custom.metric{{{filters}}} by {{pod_name}}"},
        )
        templates = _build_metric_templates(config)

        assert "my_metric" in templates
        assert "custom.metric" in templates["my_metric"]
        # Built-in templates still present
        assert "cpu" in templates
        assert "memory" in templates

    def test_custom_metric_can_override_builtin(self) -> None:
        """A custom_metrics entry with the same key as a built-in template overrides it."""
        from vaig.tools.gke.datadog_api import _build_metric_templates

        custom_cpu = "avg:my.custom.cpu{{{filters}}} by {{pod_name}}"
        config = DatadogAPIConfig(
            enabled=True,
            api_key="k",
            app_key="k",
            custom_metrics={"cpu": custom_cpu},
        )
        templates = _build_metric_templates(config)

        assert templates["cpu"] == custom_cpu

    def test_custom_metric_missing_filters_placeholder_raises(self) -> None:
        """A custom_metrics template without {filters} raises ValueError."""
        from vaig.tools.gke.datadog_api import _build_metric_templates

        config = DatadogAPIConfig(
            enabled=True,
            api_key="k",
            app_key="k",
            custom_metrics={"broken": "avg:custom.metric{cluster_name:X} by {pod_name}"},
        )
        with pytest.raises(ValueError, match="missing the required"):
            _build_metric_templates(config)

    def test_query_uses_custom_pod_name_in_api_call(self, dd_config: DatadogAPIConfig) -> None:
        """query_datadog_metrics uses config.labels.pod_name in the actual API query string."""
        from vaig.core.config import DatadogAPIConfig as _DDConfig
        from vaig.core.config import DatadogLabelConfig
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        config = _DDConfig(
            enabled=True,
            api_key="k",
            app_key="k",
            labels=DatadogLabelConfig(pod_name="custom_pod"),
        )
        mock_api = MagicMock()
        mock_api.query_metrics.return_value = MagicMock(series=[])

        with patch.dict("sys.modules", _make_dd_modules()):
            query_datadog_metrics(
                cluster_name="my-cluster",
                metric="cpu",
                config=config,
                _custom_api=mock_api,
            )

        call_kwargs = mock_api.query_metrics.call_args.kwargs
        query_str = call_kwargs.get("query", "")
        assert "{custom_pod}" in query_str
        assert "{pod_name}" not in query_str

    def test_custom_metric_query_end_to_end(self) -> None:
        """query_datadog_metrics with a custom metric name calls the API without format errors.

        Verifies that .format(filters=...) doesn't raise when the custom template
        uses properly escaped braces ({{pod_name}} instead of {pod_name}).
        """
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        config = DatadogAPIConfig(
            enabled=True,
            api_key="k",
            app_key="k",
            custom_metrics={"my_custom": "avg:custom.metric{{{filters}}} by {{pod_name}}"},
        )
        mock_api = MagicMock()
        mock_api.query_metrics.return_value = MagicMock(series=[])

        with patch.dict("sys.modules", _make_dd_modules()):
            result = query_datadog_metrics(
                cluster_name="my-cluster",
                metric="my_custom",
                config=config,
                _custom_api=mock_api,
            )

        # Should succeed — no KeyError from .format(filters=...)
        assert not result.error
        call_kwargs = mock_api.query_metrics.call_args.kwargs
        query_str = call_kwargs.get("query", "")
        # filters placeholder was filled in, pod_name is literal in the result
        assert "my-cluster" in query_str
        assert "{filters}" not in query_str
        assert "{pod_name}" in query_str  # rendered literal after .format(filters=...)


# ── Detection config — _extract_dd_metadata ──────────────────


class TestDetectionConfigMetadata:
    """Tests for configurable annotation/label prefixes in _extract_dd_metadata."""

    def test_default_annotation_prefixes_detected(self) -> None:
        """Default annotation prefixes (ad.datadoghq.com/, admission.datadoghq.com/) are matched."""
        from unittest.mock import MagicMock

        from vaig.core.config import DatadogDetectionConfig
        from vaig.tools.gke.datadog import _extract_dd_metadata

        annotations = {
            "ad.datadoghq.com/check_names": '["nginx"]',
            "admission.datadoghq.com/enabled": "true",
            "other.annotation/key": "value",
        }
        labels = {
            "tags.datadoghq.com/service": "my-svc",
            "app": "nginx",
        }

        mock_settings = MagicMock()
        mock_settings.datadog.detection = DatadogDetectionConfig()

        with patch("vaig.tools.gke.datadog.get_settings", return_value=mock_settings):
            dd_ann, dd_lbl = _extract_dd_metadata(annotations, labels)

        assert "ad.datadoghq.com/check_names" in dd_ann
        assert "admission.datadoghq.com/enabled" in dd_ann
        assert "other.annotation/key" not in dd_ann
        assert "tags.datadoghq.com/service" in dd_lbl
        assert "app" not in dd_lbl

    def test_custom_annotation_prefixes_used(self) -> None:
        """Custom annotation_prefixes from config are used instead of defaults."""
        from vaig.core.config import DatadogDetectionConfig
        from vaig.tools.gke.datadog import _extract_dd_metadata

        annotations = {
            "custom.prefix/check_names": '["nginx"]',
            "ad.datadoghq.com/check_names": '["standard"]',  # default prefix — NOT matched
        }
        labels: dict[str, str] = {}

        mock_settings = MagicMock()
        mock_settings.datadog.detection = DatadogDetectionConfig(
            annotation_prefixes=["custom.prefix/"]
        )

        with patch("vaig.tools.gke.datadog.get_settings", return_value=mock_settings):
            dd_ann, _ = _extract_dd_metadata(annotations, labels)

        assert "custom.prefix/check_names" in dd_ann
        assert "ad.datadoghq.com/check_names" not in dd_ann

    def test_custom_label_prefix_used(self) -> None:
        """Custom label_prefix from config is used instead of the default."""
        from vaig.core.config import DatadogDetectionConfig
        from vaig.tools.gke.datadog import _extract_dd_metadata

        annotations: dict[str, str] = {}
        labels = {
            "myorg.datadoghq.com/service": "my-svc",
            "tags.datadoghq.com/service": "other-svc",  # default — NOT matched
        }

        mock_settings = MagicMock()
        mock_settings.datadog.detection = DatadogDetectionConfig(
            label_prefix="myorg.datadoghq.com/"
        )

        with patch("vaig.tools.gke.datadog.get_settings", return_value=mock_settings):
            _, dd_lbl = _extract_dd_metadata(annotations, labels)

        assert "myorg.datadoghq.com/service" in dd_lbl
        assert "tags.datadoghq.com/service" not in dd_lbl

    def test_fallback_to_module_defaults_when_get_settings_raises(self) -> None:
        """When get_settings() raises, the module-level fallback constants are used."""
        from vaig.tools.gke.datadog import _extract_dd_metadata

        annotations = {
            "ad.datadoghq.com/check_names": '["nginx"]',
        }
        labels = {
            "tags.datadoghq.com/service": "my-svc",
        }

        with patch("vaig.tools.gke.datadog.get_settings", side_effect=RuntimeError("no settings")):
            dd_ann, dd_lbl = _extract_dd_metadata(annotations, labels)

        # Fallback constants include ad.datadoghq.com/ and tags.datadoghq.com/
        assert "ad.datadoghq.com/check_names" in dd_ann
        assert "tags.datadoghq.com/service" in dd_lbl


# ── Detection config — _scan_deployment_for_datadog ──────────


class TestDetectionConfigEnvVars:
    """Tests for configurable env var names in _scan_deployment_for_datadog."""

    def _make_deploy_with_env(self, env_vars: dict[str, str]) -> MagicMock:
        """Build a minimal mock deployment with the given env vars on one container."""
        env_list = [MagicMock(name="EnvVar", value=v) for v in env_vars.values()]
        for mock_var, (k, _) in zip(env_list, env_vars.items(), strict=True):
            mock_var.name = k

        container = MagicMock()
        container.env = env_list

        pod_spec = MagicMock()
        pod_spec.containers = [container]

        pod_template = MagicMock()
        pod_template.spec = pod_spec
        pod_template.metadata = MagicMock()
        pod_template.metadata.annotations = {}
        pod_template.metadata.labels = {}

        deploy = MagicMock()
        deploy.metadata = MagicMock()
        deploy.metadata.name = "test-deploy"
        deploy.metadata.annotations = {}
        deploy.metadata.labels = {}
        deploy.spec = MagicMock()
        deploy.spec.template = pod_template

        return deploy

    def test_default_env_vars_trigger_detection(self) -> None:
        """Standard DD_ env vars (DD_SERVICE, DD_AGENT_HOST) are detected with default config."""
        from vaig.core.config import DatadogDetectionConfig
        from vaig.tools.gke.datadog import _scan_deployment_for_datadog

        deploy = self._make_deploy_with_env({"DD_SERVICE": "my-svc", "DD_AGENT_HOST": "datadog-agent"})

        mock_settings = MagicMock()
        mock_settings.datadog.detection = DatadogDetectionConfig()

        with patch("vaig.tools.gke.datadog.get_settings", return_value=mock_settings):
            result = _scan_deployment_for_datadog(deploy)

        assert result["has_datadog"] is True
        assert "DD_SERVICE" in result["env_vars"]
        assert "DD_AGENT_HOST" in result["env_vars"]

    def test_custom_env_vars_trigger_detection(self) -> None:
        """Custom env_vars from config are used to detect Datadog-instrumented workloads."""
        from vaig.core.config import DatadogDetectionConfig
        from vaig.tools.gke.datadog import _scan_deployment_for_datadog

        deploy = self._make_deploy_with_env({"MY_CUSTOM_DD_FLAG": "true"})

        mock_settings = MagicMock()
        mock_settings.datadog.detection = DatadogDetectionConfig(
            env_vars=["MY_CUSTOM_DD_FLAG"]
        )

        with patch("vaig.tools.gke.datadog.get_settings", return_value=mock_settings):
            result = _scan_deployment_for_datadog(deploy)

        assert result["has_datadog"] is True
        assert "MY_CUSTOM_DD_FLAG" in result["env_vars"]

    def test_non_dd_env_var_not_detected_with_default_config(self) -> None:
        """An unrecognised env var is not reported as Datadog configuration."""
        from vaig.core.config import DatadogDetectionConfig
        from vaig.tools.gke.datadog import _scan_deployment_for_datadog

        deploy = self._make_deploy_with_env({"SOME_OTHER_VAR": "value"})

        mock_settings = MagicMock()
        mock_settings.datadog.detection = DatadogDetectionConfig()

        with patch("vaig.tools.gke.datadog.get_settings", return_value=mock_settings):
            result = _scan_deployment_for_datadog(deploy)

        assert result["has_datadog"] is False
        assert "SOME_OTHER_VAR" not in result["env_vars"]

    def test_fallback_env_vars_when_get_settings_raises(self) -> None:
        """When get_settings() raises, fallback module-level env vars are used for detection."""
        from vaig.tools.gke.datadog import _scan_deployment_for_datadog

        deploy = self._make_deploy_with_env({"DD_AGENT_HOST": "datadog-agent"})

        with patch("vaig.tools.gke.datadog.get_settings", side_effect=RuntimeError("no settings")):
            result = _scan_deployment_for_datadog(deploy)

        assert result["has_datadog"] is True
        assert "DD_AGENT_HOST" in result["env_vars"]

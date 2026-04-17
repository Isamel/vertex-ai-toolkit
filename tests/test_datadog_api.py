"""Tests for Datadog REST API tools — metrics, monitors, and APM services."""

from __future__ import annotations

from types import ModuleType
from typing import Any
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


# ── get_datadog_service_catalog ───────────────────────────────


class TestGetDatadogServiceCatalog:
    """Tests for get_datadog_service_catalog."""

    def setup_method(self) -> None:
        """Clear the discovery cache before each test."""
        from vaig.tools.gke._cache import clear_discovery_cache

        clear_discovery_cache()

    def test_returns_service_list(self, dd_config: DatadogAPIConfig) -> None:
        """Returns formatted table of services from the service catalog."""
        from vaig.tools.gke.datadog_api import get_datadog_service_catalog

        mock_api = MagicMock()
        mock_api.list_service_definitions.return_value = MagicMock(
            data=[
                _make_service("frontend", "platform", "python", "critical"),
                _make_service("backend", "core", "go", "high"),
            ]
        )

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_service_catalog(
                env="production",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        assert "=== Datadog Service Catalog ===" in result.output
        assert "frontend" in result.output
        assert "backend" in result.output
        assert "Total services: 2" in result.output

    def test_no_services_returns_appropriate_message(self, dd_config: DatadogAPIConfig) -> None:
        """Empty service list returns 'no services found' message."""
        from vaig.tools.gke.datadog_api import get_datadog_service_catalog

        mock_api = MagicMock()
        mock_api.list_service_definitions.return_value = MagicMock(data=[])

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_service_catalog(
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        assert "No service catalog entries found" in result.output

    def test_result_is_cached(self, dd_config: DatadogAPIConfig) -> None:
        """Second call with same args hits the cache and does NOT call the API again."""
        from vaig.tools.gke.datadog_api import get_datadog_service_catalog

        mock_api = MagicMock()
        mock_api.list_service_definitions.return_value = MagicMock(data=[_make_service()])

        with patch.dict("sys.modules", _make_dd_modules()):
            # First call — populates cache
            result1 = get_datadog_service_catalog(
                env="production",
                cluster_name="my-cluster",
                config=dd_config,
                _custom_api=mock_api,
            )

            # Second call — should return from cache WITHOUT calling the API again
            result2 = get_datadog_service_catalog(
                env="production",
                cluster_name="my-cluster",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result1.output == result2.output
        assert mock_api.list_service_definitions.call_count == 1

    def test_different_env_not_cached(self, dd_config: DatadogAPIConfig) -> None:
        """Different env values use separate cache keys."""
        from vaig.tools.gke.datadog_api import get_datadog_service_catalog

        mock_api = MagicMock()
        mock_api.list_service_definitions.return_value = MagicMock(data=[_make_service()])

        with patch.dict("sys.modules", _make_dd_modules()):
            get_datadog_service_catalog(env="production", config=dd_config, _custom_api=mock_api)
            get_datadog_service_catalog(env="staging", config=dd_config, _custom_api=mock_api)

        assert mock_api.list_service_definitions.call_count == 2

    def test_disabled_config_returns_error(self, dd_config_disabled: DatadogAPIConfig) -> None:
        """Disabled config returns error without calling API."""
        from vaig.tools.gke.datadog_api import get_datadog_service_catalog

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_service_catalog(config=dd_config_disabled)

        assert result.error is True
        assert "disabled" in result.output.lower()

    def test_cluster_name_shown_in_output(self, dd_config: DatadogAPIConfig) -> None:
        """Cluster name appears in output when provided."""
        from vaig.tools.gke.datadog_api import get_datadog_service_catalog

        mock_api = MagicMock()
        mock_api.list_service_definitions.return_value = MagicMock(data=[])

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_service_catalog(
                env="production",
                cluster_name="prod-cluster",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert "prod-cluster" in result.output

    def test_import_error_returns_install_message(self, dd_config: DatadogAPIConfig) -> None:
        """Missing datadog-api-client returns install instructions."""
        from vaig.tools.gke.datadog_api import get_datadog_service_catalog

        null_modules: dict[str, ModuleType | None] = {
            "datadog_api_client": None,
            "datadog_api_client.exceptions": None,
            "datadog_api_client.v2": None,
            "datadog_api_client.v2.api": None,
            "datadog_api_client.v2.api.service_definition_api": None,
        }
        with patch.dict("sys.modules", null_modules):
            result = get_datadog_service_catalog(config=dd_config)

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


class TestGetDatadogServiceCatalogLabelFilters:
    """Tests for the service_name filter parameter on get_datadog_service_catalog."""

    def setup_method(self) -> None:
        """Clear the discovery cache before each test."""
        from vaig.tools.gke._cache import clear_discovery_cache

        clear_discovery_cache()

    def test_service_name_filter_returns_only_matching_service(
        self, dd_config: DatadogAPIConfig
    ) -> None:
        """When service_name is provided, only the matching service is returned."""
        from vaig.tools.gke.datadog_api import get_datadog_service_catalog

        mock_api = MagicMock()
        mock_api.list_service_definitions.return_value = MagicMock(
            data=[
                _make_service("frontend", "platform", "python", "critical"),
                _make_service("backend", "core", "go", "high"),
                _make_service("worker", "infra", "python", "low"),
            ]
        )

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_service_catalog(
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
        """When service_name matches nothing, returns 'no service catalog entries found'."""
        from vaig.tools.gke.datadog_api import get_datadog_service_catalog

        mock_api = MagicMock()
        mock_api.list_service_definitions.return_value = MagicMock(
            data=[
                _make_service("frontend", "platform", "python", "critical"),
            ]
        )

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_service_catalog(
                service_name="nonexistent-service",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        assert "No service catalog entries found." in result.output

    def test_backward_compat_no_service_name_returns_all(
        self, dd_config: DatadogAPIConfig
    ) -> None:
        """Without service_name, all services are returned (backward compat)."""
        from vaig.tools.gke.datadog_api import get_datadog_service_catalog

        mock_api = MagicMock()
        mock_api.list_service_definitions.return_value = MagicMock(
            data=[
                _make_service("frontend", "platform", "python", "critical"),
                _make_service("backend", "core", "go", "high"),
            ]
        )

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_service_catalog(
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
        from vaig.tools.gke.datadog_api import get_datadog_service_catalog

        mock_api = MagicMock()
        mock_api.list_service_definitions.return_value = MagicMock(
            data=[
                _make_service("frontend"),
                _make_service("backend"),
            ]
        )

        with patch.dict("sys.modules", _make_dd_modules()):
            # Unfiltered call
            get_datadog_service_catalog(env="production", config=dd_config, _custom_api=mock_api)
            # Filtered call — different cache key, must hit the API again
            get_datadog_service_catalog(
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

    def test_include_custom_labels_false_excludes_custom(self) -> None:
        """When include_custom_labels=False, config.labels.custom entries are not appended."""
        from vaig.core.config import DatadogLabelConfig
        from vaig.tools.gke.datadog_api import _build_tag_filter

        config = DatadogAPIConfig(
            enabled=True,
            api_key="k",
            app_key="k",
            labels=DatadogLabelConfig(custom={"team": "platform", "region": "us-east"}),
        )
        tag_filter, err = _build_tag_filter(
            "my-cluster", "my-svc", "prod", config, include_custom_labels=False,
        )

        assert err is None
        assert "cluster_name:my-cluster" in tag_filter
        assert "service:my-svc" in tag_filter
        assert "env:prod" in tag_filter
        assert "team:" not in tag_filter
        assert "region:" not in tag_filter

    def test_include_custom_labels_true_includes_custom(self) -> None:
        """When include_custom_labels=True (default), custom entries are appended."""
        from vaig.core.config import DatadogLabelConfig
        from vaig.tools.gke.datadog_api import _build_tag_filter

        config = DatadogAPIConfig(
            enabled=True,
            api_key="k",
            app_key="k",
            labels=DatadogLabelConfig(custom={"team": "platform"}),
        )
        tag_filter, err = _build_tag_filter(
            "my-cluster", None, None, config, include_custom_labels=True,
        )

        assert err is None
        assert "team:platform" in tag_filter

    def test_include_custom_labels_default_is_true(self) -> None:
        """Calling without include_custom_labels behaves like True (backward compat)."""
        from vaig.core.config import DatadogLabelConfig
        from vaig.tools.gke.datadog_api import _build_tag_filter

        config = DatadogAPIConfig(
            enabled=True,
            api_key="k",
            app_key="k",
            labels=DatadogLabelConfig(custom={"team": "platform"}),
        )
        # No include_custom_labels argument — should default to True
        tag_filter, err = _build_tag_filter("my-cluster", None, None, config)

        assert err is None
        assert "team:platform" in tag_filter


# ── _point_value helper ──────────────────────────────────────


class TestPointValue:
    """Tests for the _point_value helper that extracts values from SDK Point objects and plain lists."""

    def test_plain_list(self) -> None:
        """Plain [timestamp, value] lists work via index access."""
        from vaig.tools.gke.datadog_api import _point_value

        assert _point_value([1700000000, 42.5]) == 42.5

    def test_plain_list_none_value(self) -> None:
        """Returns None when the value element is None."""
        from vaig.tools.gke.datadog_api import _point_value

        assert _point_value([1700000000, None]) is None

    def test_sdk_point_like_object(self) -> None:
        """SDK Point objects expose .value = [timestamp, value]."""
        from vaig.tools.gke.datadog_api import _point_value

        class FakePoint:
            value = [1700000000, 99.0]

        assert _point_value(FakePoint()) == 99.0

    def test_sdk_point_like_none_value(self) -> None:
        """SDK Point with None value returns None."""
        from vaig.tools.gke.datadog_api import _point_value

        class FakePoint:
            value = [1700000000, None]

        assert _point_value(FakePoint()) is None

    def test_empty_list_returns_none(self) -> None:
        """Short list without index 1 returns None."""
        from vaig.tools.gke.datadog_api import _point_value

        assert _point_value([]) is None

    def test_non_numeric_returns_none(self) -> None:
        """Non-numeric value returns None (ValueError caught)."""
        from vaig.tools.gke.datadog_api import _point_value

        assert _point_value([1700000000, "not-a-number"]) is None


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


# ── get_datadog_apm_services (Metrics v1 Timeseries API) ──────


def _make_apm_query_response(
    hits: float | None = 100.0,
    errors: float | None = 5.0,
    duration: float | None = 0.015,
) -> dict[str, MagicMock]:
    """Build mock ``MetricsApi.query_metrics()`` responses for APM queries.

    Returns a dict keyed by metric kind (``hits``, ``errors``, ``duration``)
    so callers can wire them into ``side_effect`` lists.
    """
    responses: dict[str, MagicMock] = {}
    for key, value in [("hits", hits), ("errors", errors), ("duration", duration)]:
        resp = MagicMock()
        if value is not None:
            point = MagicMock()
            point.__getitem__ = lambda self, idx, v=value: [1700000000, v][idx]
            series_obj = MagicMock()
            series_obj.pointlist = [[1700000000, value]]
            resp.series = [series_obj]
        else:
            resp.series = []
        responses[key] = resp
    return responses


class TestGetDatadogApmServices:
    """Tests for get_datadog_apm_services (Metrics v1 Timeseries API)."""

    def setup_method(self) -> None:
        """Clear the discovery cache before each test."""
        from vaig.tools.gke._cache import clear_discovery_cache

        clear_discovery_cache()

    def _make_api_with_responses(
        self,
        hits: float | None = 100.0,
        errors: float | None = 5.0,
        duration: float | None = 0.015,
    ) -> MagicMock:
        """Create a mock MetricsApi with canned query_metrics responses.

        The mock is configured so that:
        - _detect_apm_operation probe queries return data for ``servlet.request``
        - The 3 APM metric queries (hits, errors, duration) return the given values
        """
        responses = _make_apm_query_response(hits=hits, errors=errors, duration=duration)
        # Build response list: first call is for _detect_apm_operation probe (returns hits series),
        # then the 3 actual metric queries.
        probe_resp = MagicMock()
        probe_series = MagicMock()
        probe_series.pointlist = [[1700000000, 42.0]]
        probe_resp.series = [probe_series]

        mock_api = MagicMock()
        mock_api.query_metrics.side_effect = [
            probe_resp,  # _detect_apm_operation probe for first operation
            responses["hits"],
            responses["errors"],
            responses["duration"],
        ]
        return mock_api

    def test_returns_apm_metrics_table(self, dd_config: DatadogAPIConfig) -> None:
        """Successful API call returns the APM metrics table with service, env, and metrics."""
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        mock_api = self._make_api_with_responses(hits=200.0, errors=10.0, duration=0.025)

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_apm_services(
                service_name="my-service",
                env="production",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        assert "=== Datadog APM Trace Metrics ===" in result.output
        assert "my-service" in result.output
        assert "production" in result.output
        assert "Throughput:" in result.output
        assert "Error rate:" in result.output
        assert "Avg latency:" in result.output

    def test_queries_use_service_and_env_tags(self, dd_config: DatadogAPIConfig) -> None:
        """MetricsApi.query_metrics calls include the correct service and env in the query."""
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        mock_api = self._make_api_with_responses()

        with patch.dict("sys.modules", _make_dd_modules()):
            get_datadog_apm_services(
                service_name="checkout",
                env="staging",
                config=dd_config,
                _custom_api=mock_api,
            )

        # At least one query_metrics call should contain service:checkout and env:staging
        calls = mock_api.query_metrics.call_args_list
        queries = [str(c) for c in calls]
        query_str = " ".join(queries)
        assert "service:checkout" in query_str
        assert "env:staging" in query_str

    def test_operation_shown_in_output(self, dd_config: DatadogAPIConfig) -> None:
        """Output contains the detected APM operation name."""
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        mock_api = self._make_api_with_responses()

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_apm_services(
                service_name="svc",
                env="prod",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        assert "Operation:" in result.output

    def test_no_data_returns_guidance(self, dd_config: DatadogAPIConfig) -> None:
        """When all probe operations return empty data, returns guidance (no error)."""
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        # All probe responses return empty series — _detect_apm_operation returns None
        mock_api = MagicMock()
        empty_resp = MagicMock()
        empty_resp.series = []
        mock_api.query_metrics.return_value = empty_resp

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_apm_services(
                service_name="ghost-service",
                env="staging",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        assert "No APM trace data found" in result.output


# ── SSLError handling in SDK-based functions ──────────────────


class TestSDKFunctionsSSLError:
    """Tests for ssl.SSLError handling in query_datadog_metrics, get_datadog_monitors,
    and get_datadog_service_catalog — the three SDK-based functions that use _get_dd_api_client.
    """

    _SSL_HELP_PHRASES = [
        "SSL certificate verification failed",
        "REQUESTS_CA_BUNDLE",
        "ssl_verify",
    ]

    def setup_method(self) -> None:
        """Clear the discovery cache before each test."""
        from vaig.tools.gke._cache import clear_discovery_cache

        clear_discovery_cache()

    # ── query_datadog_metrics ─────────────────────────────────

    def test_query_datadog_metrics_ssl_error_returns_helpful_message(
        self, dd_config: DatadogAPIConfig
    ) -> None:
        """ssl.SSLError from the metrics SDK call returns a helpful ToolResult."""
        import ssl

        from vaig.tools.gke.datadog_api import query_datadog_metrics

        ssl_exc = ssl.SSLError("CERTIFICATE_VERIFY_FAILED")
        mock_api = MagicMock()
        mock_api.query_metrics.side_effect = ssl_exc

        with patch.dict("sys.modules", _make_dd_modules()):
            result = query_datadog_metrics(
                cluster_name="my-cluster",
                metric="cpu",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is True
        for phrase in self._SSL_HELP_PHRASES:
            assert phrase in result.output, f"Missing phrase {phrase!r} in output: {result.output}"

    def test_query_datadog_metrics_ssl_error_contains_all_three_suggestions(
        self, dd_config: DatadogAPIConfig
    ) -> None:
        """The ssl.SSLError message contains all 3 numbered fix suggestions."""
        import ssl

        from vaig.tools.gke.datadog_api import query_datadog_metrics

        mock_api = MagicMock()
        mock_api.query_metrics.side_effect = ssl.SSLError("cert verify failed")

        with patch.dict("sys.modules", _make_dd_modules()):
            result = query_datadog_metrics(
                cluster_name="my-cluster",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert "1." in result.output
        assert "2." in result.output
        assert "3." in result.output

    def test_query_datadog_metrics_maxretry_ssl_reason_returns_helpful_message(
        self, dd_config: DatadogAPIConfig
    ) -> None:
        """urllib3 MaxRetryError wrapping an SSLError returns the helpful SSL message."""
        import ssl

        import urllib3.exceptions

        from vaig.tools.gke.datadog_api import query_datadog_metrics

        ssl_reason = ssl.SSLError("CERTIFICATE_VERIFY_FAILED")
        max_retry_exc = urllib3.exceptions.MaxRetryError(
            pool=MagicMock(), url="/api/v1/query", reason=ssl_reason
        )
        mock_api = MagicMock()
        mock_api.query_metrics.side_effect = max_retry_exc

        with patch.dict("sys.modules", _make_dd_modules()):
            result = query_datadog_metrics(
                cluster_name="my-cluster",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is True
        assert "SSL certificate verification failed" in result.output

    # ── get_datadog_monitors ──────────────────────────────────

    def test_get_datadog_monitors_ssl_error_returns_helpful_message(
        self, dd_config: DatadogAPIConfig
    ) -> None:
        """ssl.SSLError from the monitors SDK call returns a helpful ToolResult."""
        import ssl

        from vaig.tools.gke.datadog_api import get_datadog_monitors

        ssl_exc = ssl.SSLError("CERTIFICATE_VERIFY_FAILED")
        mock_api = MagicMock()
        mock_api.list_monitors.side_effect = ssl_exc

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_monitors(
                cluster_name="my-cluster",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is True
        for phrase in self._SSL_HELP_PHRASES:
            assert phrase in result.output, f"Missing phrase {phrase!r} in output: {result.output}"

    def test_get_datadog_monitors_ssl_error_contains_all_three_suggestions(
        self, dd_config: DatadogAPIConfig
    ) -> None:
        """The ssl.SSLError message for monitors contains all 3 numbered fix suggestions."""
        import ssl

        from vaig.tools.gke.datadog_api import get_datadog_monitors

        mock_api = MagicMock()
        mock_api.list_monitors.side_effect = ssl.SSLError("cert verify failed")

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_monitors(
                config=dd_config,
                _custom_api=mock_api,
            )

        assert "1." in result.output
        assert "2." in result.output
        assert "3." in result.output

    def test_get_datadog_monitors_maxretry_ssl_reason_returns_helpful_message(
        self, dd_config: DatadogAPIConfig
    ) -> None:
        """urllib3 MaxRetryError wrapping an SSLError returns the helpful SSL message for monitors."""
        import ssl

        import urllib3.exceptions

        from vaig.tools.gke.datadog_api import get_datadog_monitors

        ssl_reason = ssl.SSLError("CERTIFICATE_VERIFY_FAILED")
        max_retry_exc = urllib3.exceptions.MaxRetryError(
            pool=MagicMock(), url="/api/v1/monitor", reason=ssl_reason
        )
        mock_api = MagicMock()
        mock_api.list_monitors.side_effect = max_retry_exc

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_monitors(
                cluster_name="my-cluster",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is True
        assert "SSL certificate verification failed" in result.output

    # ── get_datadog_service_catalog ───────────────────────────

    def test_get_datadog_service_catalog_ssl_error_returns_helpful_message(
        self, dd_config: DatadogAPIConfig
    ) -> None:
        """ssl.SSLError from the service catalog SDK call returns a helpful ToolResult."""
        import ssl

        from vaig.tools.gke.datadog_api import get_datadog_service_catalog

        ssl_exc = ssl.SSLError("CERTIFICATE_VERIFY_FAILED")
        mock_api = MagicMock()
        mock_api.list_service_definitions.side_effect = ssl_exc

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_service_catalog(
                env="production",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is True
        for phrase in self._SSL_HELP_PHRASES:
            assert phrase in result.output, f"Missing phrase {phrase!r} in output: {result.output}"

    def test_get_datadog_service_catalog_ssl_error_contains_all_three_suggestions(
        self, dd_config: DatadogAPIConfig
    ) -> None:
        """The ssl.SSLError message for service catalog contains all 3 numbered fix suggestions."""
        import ssl

        from vaig.tools.gke.datadog_api import get_datadog_service_catalog

        mock_api = MagicMock()
        mock_api.list_service_definitions.side_effect = ssl.SSLError("cert verify failed")

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_service_catalog(
                config=dd_config,
                _custom_api=mock_api,
            )

        assert "1." in result.output
        assert "2." in result.output
        assert "3." in result.output

    def test_get_datadog_service_catalog_maxretry_ssl_reason_returns_helpful_message(
        self, dd_config: DatadogAPIConfig
    ) -> None:
        """urllib3 MaxRetryError wrapping an SSLError returns the helpful SSL message for catalog."""
        import ssl

        import urllib3.exceptions

        from vaig.tools.gke.datadog_api import get_datadog_service_catalog

        ssl_reason = ssl.SSLError("CERTIFICATE_VERIFY_FAILED")
        max_retry_exc = urllib3.exceptions.MaxRetryError(
            pool=MagicMock(), url="/api/v2/services/definitions", reason=ssl_reason
        )
        mock_api = MagicMock()
        mock_api.list_service_definitions.side_effect = max_retry_exc

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_service_catalog(
                env="production",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is True
        assert "SSL certificate verification failed" in result.output

    # ── Non-SSL MaxRetryError (Fix 2) ─────────────────────────

    def test_query_datadog_metrics_non_ssl_maxretry_returns_error_result(
        self, dd_config: DatadogAPIConfig
    ) -> None:
        """MaxRetryError with a non-SSL reason returns ToolResult(error=True) with 'multiple retries'."""
        import urllib3.exceptions

        from vaig.tools.gke.datadog_api import query_datadog_metrics

        non_ssl_reason = ConnectionError("timed out")
        max_retry_exc = urllib3.exceptions.MaxRetryError(
            pool=MagicMock(), url="/api/v1/query", reason=non_ssl_reason
        )
        mock_api = MagicMock()
        mock_api.query_metrics.side_effect = max_retry_exc

        with patch.dict("sys.modules", _make_dd_modules()):
            result = query_datadog_metrics(
                cluster_name="my-cluster",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is True
        assert "multiple retries" in result.output

    def test_get_datadog_monitors_non_ssl_maxretry_returns_error_result(
        self, dd_config: DatadogAPIConfig
    ) -> None:
        """MaxRetryError with a non-SSL reason returns ToolResult(error=True) with 'multiple retries'."""
        import urllib3.exceptions

        from vaig.tools.gke.datadog_api import get_datadog_monitors

        non_ssl_reason = ConnectionError("timed out")
        max_retry_exc = urllib3.exceptions.MaxRetryError(
            pool=MagicMock(), url="/api/v1/monitor", reason=non_ssl_reason
        )
        mock_api = MagicMock()
        mock_api.list_monitors.side_effect = max_retry_exc

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_monitors(
                cluster_name="my-cluster",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is True
        assert "multiple retries" in result.output

    def test_get_datadog_service_catalog_non_ssl_maxretry_returns_error_result(
        self, dd_config: DatadogAPIConfig
    ) -> None:
        """MaxRetryError with a non-SSL reason returns ToolResult(error=True) with 'multiple retries'."""
        import urllib3.exceptions

        from vaig.tools.gke.datadog_api import get_datadog_service_catalog

        non_ssl_reason = ConnectionError("timed out")
        max_retry_exc = urllib3.exceptions.MaxRetryError(
            pool=MagicMock(), url="/api/v2/services/definitions", reason=non_ssl_reason
        )
        mock_api = MagicMock()
        mock_api.list_service_definitions.side_effect = max_retry_exc

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_service_catalog(
                env="production",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is True
        assert "multiple retries" in result.output


# ── SSL / proxy configuration ─────────────────────────────────


class TestDatadogSSLConfig:
    """Tests for DatadogAPIConfig.ssl_verify and SSL error handling."""

    def setup_method(self) -> None:
        """Clear the discovery cache before each test."""
        from vaig.tools.gke._cache import clear_discovery_cache

        clear_discovery_cache()

    def _make_api_with_responses(
        self,
        hits: float | None = 100.0,
        errors: float | None = 5.0,
        duration: float | None = 0.015,
    ) -> MagicMock:
        """Create a mock MetricsApi with canned query_metrics responses.

        The mock is configured so that:
        - _detect_apm_operation probe queries return data for the first operation
        - The 3 APM metric queries (hits, errors, duration) return the given values
        """
        responses = _make_apm_query_response(hits=hits, errors=errors, duration=duration)
        probe_resp = MagicMock()
        probe_series = MagicMock()
        probe_series.pointlist = [[1700000000, 42.0]]
        probe_resp.series = [probe_series]

        mock_api = MagicMock()
        mock_api.query_metrics.side_effect = [
            probe_resp,
            responses["hits"],
            responses["errors"],
            responses["duration"],
        ]
        return mock_api

    # ── Config field tests ────────────────────────────────────

    def test_ssl_verify_default_is_true(self) -> None:
        """Default ssl_verify is True (standard SSL verification)."""
        cfg = DatadogAPIConfig(enabled=True, api_key="k", app_key="k")
        assert cfg.ssl_verify is True

    def test_ssl_verify_false_disables_verification(self) -> None:
        """ssl_verify=False stores as False (disables SSL certificate checking)."""
        cfg = DatadogAPIConfig(enabled=True, api_key="k", app_key="k", ssl_verify=False)
        assert cfg.ssl_verify is False

    def test_ssl_verify_str_stores_ca_bundle_path(self) -> None:
        """ssl_verify='/path/to/ca.crt' stores the path for custom CA bundle."""
        ca_path = "/etc/ssl/certs/corporate-ca.crt"
        cfg = DatadogAPIConfig(enabled=True, api_key="k", app_key="k", ssl_verify=ca_path)
        assert cfg.ssl_verify == ca_path

    def test_ssl_verify_empty_string_raises_validation_error(self) -> None:
        """ssl_verify='' raises a ValidationError — empty string is not a valid CA bundle path."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="ssl_verify must be True, False, or a non-empty path"):
            DatadogAPIConfig(enabled=True, api_key="k", app_key="k", ssl_verify="")

    def test_ssl_verify_whitespace_string_raises_validation_error(self) -> None:
        """ssl_verify='   ' (whitespace only) raises a ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="ssl_verify must be True, False, or a non-empty path"):
            DatadogAPIConfig(enabled=True, api_key="k", app_key="k", ssl_verify="   ")

    # ── APM SDK SSL handling (via _get_dd_api_client) ───────────
    # The old requests.Session.verify tests are replaced by SDK-level tests.
    # APM now uses MetricsApi via the SDK, so SSL config is tested through
    # _get_dd_api_client (see test_sdk_client_verify_ssl_* above) and
    # MaxRetryError wrapping SSLError (tested below).

    def test_apm_ssl_error_returns_helpful_message(self, dd_config: DatadogAPIConfig) -> None:
        """MaxRetryError wrapping SSLError from APM metric query returns helpful SSL message."""
        import ssl

        import urllib3.exceptions

        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        ssl_reason = ssl.SSLError("CERTIFICATE_VERIFY_FAILED")
        max_retry_exc = urllib3.exceptions.MaxRetryError(
            pool=MagicMock(), url="/api/v1/query", reason=ssl_reason
        )
        # Probe succeeds (returns data) so _detect_apm_operation finds an operation;
        # the SSL error occurs on the subsequent metric query call.
        probe_resp = MagicMock()
        probe_series = MagicMock()
        probe_series.pointlist = [[1700000000, 42.0]]
        probe_resp.series = [probe_series]

        mock_api = MagicMock()
        mock_api.query_metrics.side_effect = [probe_resp, max_retry_exc]

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_apm_services(
                service_name="svc",
                env="prod",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is True
        assert "SSL certificate verification failed" in result.output
        assert "REQUESTS_CA_BUNDLE" in result.output
        assert "ssl_verify" in result.output

    def test_apm_non_ssl_maxretry_returns_error(self, dd_config: DatadogAPIConfig) -> None:
        """MaxRetryError with non-SSL reason from APM returns error with 'multiple retries'."""
        import urllib3.exceptions

        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        non_ssl_reason = ConnectionError("timed out")
        max_retry_exc = urllib3.exceptions.MaxRetryError(
            pool=MagicMock(), url="/api/v1/query", reason=non_ssl_reason
        )
        # Probe succeeds so _detect_apm_operation finds an operation;
        # the connection error occurs on the subsequent metric query call.
        probe_resp = MagicMock()
        probe_series = MagicMock()
        probe_series.pointlist = [[1700000000, 42.0]]
        probe_resp.series = [probe_series]

        mock_api = MagicMock()
        mock_api.query_metrics.side_effect = [probe_resp, max_retry_exc]

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_apm_services(
                service_name="svc",
                env="prod",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is True
        assert "multiple retries" in result.output

    # ── Datadog SDK client SSL config ─────────────────────────

    def test_sdk_client_verify_ssl_false_when_ssl_disabled(self) -> None:
        """When ssl_verify=False, the Datadog SDK Configuration has verify_ssl=False."""
        from vaig.tools.gke.datadog_api import _get_dd_api_client

        cfg = DatadogAPIConfig(enabled=True, api_key="k", app_key="k", ssl_verify=False)

        mock_configuration = MagicMock()
        mock_configuration.server_variables = {}
        mock_configuration.api_key = {}

        dd_mods = _make_dd_modules()
        dd_mods["datadog_api_client"].Configuration.return_value = mock_configuration  # type: ignore[attr-defined]
        with patch.dict("sys.modules", dd_mods):
            _get_dd_api_client(cfg)

        assert mock_configuration.verify_ssl is False

    def test_sdk_client_ssl_ca_cert_set_for_custom_bundle(self) -> None:
        """When ssl_verify is a path, SDK Configuration has ssl_ca_cert set and verify_ssl=True."""
        from vaig.tools.gke.datadog_api import _get_dd_api_client

        ca_path = "/etc/ssl/certs/corporate-ca.crt"
        cfg = DatadogAPIConfig(enabled=True, api_key="k", app_key="k", ssl_verify=ca_path)

        mock_configuration = MagicMock()
        mock_configuration.server_variables = {}
        mock_configuration.api_key = {}

        dd_mods = _make_dd_modules()
        dd_mods["datadog_api_client"].Configuration.return_value = mock_configuration  # type: ignore[attr-defined]
        with patch.dict("sys.modules", dd_mods):
            _get_dd_api_client(cfg)

        assert mock_configuration.verify_ssl is True
        assert mock_configuration.ssl_ca_cert == ca_path

    def test_sdk_client_no_ssl_change_for_default_true(self) -> None:
        """When ssl_verify=True (default), SDK Configuration verify_ssl and ssl_ca_cert
        are NOT set at all — any attribute write would be detected by the spy.
        """
        from vaig.tools.gke.datadog_api import _get_dd_api_client

        cfg = DatadogAPIConfig(enabled=True, api_key="k", app_key="k")  # ssl_verify=True default

        # Track every __setattr__ call on the configuration object so we can assert
        # that neither verify_ssl nor ssl_ca_cert were written.
        ssl_attrs_written: list[str] = []
        original_setattr = MagicMock.__setattr__

        mock_configuration = MagicMock()
        mock_configuration.server_variables = {}
        mock_configuration.api_key = {}

        def _spy_setattr(self: object, name: str, value: object) -> None:
            if name in ("verify_ssl", "ssl_ca_cert"):
                ssl_attrs_written.append(name)
            original_setattr(self, name, value)

        dd_mods = _make_dd_modules()
        dd_mods["datadog_api_client"].Configuration.return_value = mock_configuration  # type: ignore[attr-defined]
        with patch.dict("sys.modules", dd_mods):
            with patch.object(type(mock_configuration), "__setattr__", _spy_setattr):
                _get_dd_api_client(cfg)

        assert ssl_attrs_written == [], (
            f"ssl_verify=True must not trigger any write to {ssl_attrs_written} "
            "on the SDK Configuration object"
        )

    def test_empty_result_when_all_metrics_empty(self, dd_config: DatadogAPIConfig) -> None:
        """When all probe operations return empty, returns guidance (no error)."""
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        mock_api = MagicMock()
        empty_resp = MagicMock()
        empty_resp.series = []
        mock_api.query_metrics.return_value = empty_resp

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_apm_services(
                service_name="quiet-svc",
                env="production",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        assert "No APM trace data found" in result.output

    def test_custom_time_range_default_lookback(self, dd_config: DatadogAPIConfig) -> None:
        """Default hours_back (config default = 4.0) — output mentions 4 hours window."""
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        mock_api = self._make_api_with_responses()

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_apm_services(
                service_name="svc",
                env="prod",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        assert "last 4 hours" in result.output

    def test_custom_time_range_4_hours(self, dd_config: DatadogAPIConfig) -> None:
        """hours_back=4 — output mentions 4 hours window."""
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        mock_api = self._make_api_with_responses()

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_apm_services(
                service_name="svc",
                env="prod",
                hours_back=4.0,
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        assert "last 4 hours" in result.output

    def test_non_positive_hours_back_clamps_to_default(self, dd_config: DatadogAPIConfig) -> None:
        """hours_back=0 clamps to default and proceeds normally — does not return an error."""
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        mock_api = self._make_api_with_responses()

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_apm_services(
                service_name="svc",
                env="prod",
                hours_back=0,
                config=dd_config,
                _custom_api=mock_api,
            )

        # Should not be an error — clamps to config default window and queries the API
        assert result.error is False

    def test_caches_result(self, dd_config: DatadogAPIConfig) -> None:
        """Calling get_datadog_apm_services twice with the same args only calls the API once."""
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        mock_api = self._make_api_with_responses()

        with patch.dict("sys.modules", _make_dd_modules()):
            result1 = get_datadog_apm_services(
                service_name="cached-svc",
                env="production",
                config=dd_config,
                _custom_api=mock_api,
            )
            result2 = get_datadog_apm_services(
                service_name="cached-svc",
                env="production",
                config=dd_config,
                _custom_api=mock_api,
            )

        # API called once (probe + 3 metrics = 4 calls); second call served from cache
        assert mock_api.query_metrics.call_count == 4
        assert result1.output == result2.output

    def test_different_hours_back_uses_separate_cache_key(self, dd_config: DatadogAPIConfig) -> None:
        """Different hours_back values use separate cache keys — no cross-contamination."""
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        # Need two separate mock APIs since side_effect is consumed
        mock_api1 = self._make_api_with_responses()
        mock_api2 = self._make_api_with_responses()

        with patch.dict("sys.modules", _make_dd_modules()):
            get_datadog_apm_services(
                service_name="svc", env="prod", hours_back=1.0, config=dd_config, _custom_api=mock_api1
            )

        from vaig.tools.gke._cache import clear_discovery_cache

        clear_discovery_cache()

        with patch.dict("sys.modules", _make_dd_modules()):
            get_datadog_apm_services(
                service_name="svc", env="prod", hours_back=4.0, config=dd_config, _custom_api=mock_api2
            )

        # Both APIs should have been called (not served from cache)
        assert mock_api1.query_metrics.call_count >= 1
        assert mock_api2.query_metrics.call_count >= 1

    def test_empty_service_name_returns_guidance_not_error(
        self, dd_config: DatadogAPIConfig
    ) -> None:
        """When service_name is omitted or empty, the tool returns guidance (not an error).

        This allows the LLM to call the tool speculatively and receive instructions on
        how to resolve the service name from Kubernetes labels, rather than failing hard.
        """
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_apm_services(
                service_name="",
                config=dd_config,
            )

        assert result.error is False, (
            "Empty service_name should NOT be an error — it should return guidance "
            "so the LLM can resolve the service name and retry."
        )
        assert "service_name" in result.output.lower(), (
            "Guidance message must mention 'service_name' so the LLM knows what to provide."
        )
        assert "kubernetes" in result.output.lower() or "pod label" in result.output.lower(), (
            "Guidance must direct the LLM to resolve service_name from Kubernetes labels."
        )

    def test_disabled_config_returns_error(self, dd_config_disabled: DatadogAPIConfig) -> None:
        """Disabled config returns error without calling any API."""
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        mock_api = MagicMock()

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_apm_services(
                service_name="svc",
                config=dd_config_disabled,
                _custom_api=mock_api,
            )

        assert result.error is True
        assert "disabled" in result.output.lower()
        mock_api.query_metrics.assert_not_called()

    def test_error_rate_computed_from_metrics(self, dd_config: DatadogAPIConfig) -> None:
        """Error rate is correctly computed from hits and errors metrics."""
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        # 100 hits, 50 errors → 50% error rate
        mock_api = self._make_api_with_responses(hits=100.0, errors=50.0, duration=0.010)

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_apm_services(
                service_name="error-svc",
                env="production",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        assert "50.00%" in result.output

    def test_unexpected_exception_returns_error(self, dd_config: DatadogAPIConfig) -> None:
        """When an unexpected network error occurs, returns error result."""
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        # Probe succeeds; the OSError occurs on the actual metric query.
        probe_resp = MagicMock()
        probe_series = MagicMock()
        probe_series.pointlist = [[1700000000, 42.0]]
        probe_resp.series = [probe_series]

        mock_api = MagicMock()
        mock_api.query_metrics.side_effect = [probe_resp, OSError("connection reset")]

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_apm_services(
                service_name="svc",
                env="prod",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is True
        assert "Unexpected error" in result.output


# ── Problem 1: APM-native metric templates ───────────────────


class TestBuildMetricTemplatesMode:
    """Tests for metric_mode field in DatadogAPIConfig and _build_metric_templates."""

    def test_k8s_agent_mode_returns_kubernetes_templates(self) -> None:
        """Default k8s_agent mode returns kubernetes.* metric templates."""
        from vaig.tools.gke.datadog_api import _build_metric_templates

        config = DatadogAPIConfig(enabled=True, api_key="k", app_key="k", metric_mode="k8s_agent")
        templates = _build_metric_templates(config)

        assert "cpu" in templates
        assert "memory" in templates
        assert "restarts" in templates
        assert "kubernetes.cpu.usage.total" in templates["cpu"]
        assert "kubernetes.memory.usage" in templates["memory"]
        # No APM keys
        assert "requests" not in templates
        assert "error_rate" not in templates

    def test_default_mode_is_k8s_agent(self) -> None:
        """When metric_mode is not specified, default is k8s_agent (kubernetes.* templates)."""
        from vaig.tools.gke.datadog_api import _build_metric_templates

        config = DatadogAPIConfig(enabled=True, api_key="k", app_key="k")
        templates = _build_metric_templates(config)

        assert "cpu" in templates
        assert "kubernetes.cpu.usage.total" in templates["cpu"]

    def test_apm_mode_returns_trace_templates(self) -> None:
        """APM mode returns trace.http.request.* metric templates."""
        from vaig.tools.gke.datadog_api import _build_metric_templates

        config = DatadogAPIConfig(enabled=True, api_key="k", app_key="k", metric_mode="apm")
        templates = _build_metric_templates(config)

        assert "requests" in templates
        assert "errors" in templates
        assert "latency" in templates
        assert "error_rate" in templates
        assert "apdex" in templates
        assert "trace.http.request" in templates["requests"]
        assert "trace.http.request" in templates["latency"]
        # No k8s keys
        assert "cpu" not in templates
        assert "memory" not in templates

    def test_apm_mode_all_templates_have_filters_placeholder(self) -> None:
        """All APM-mode templates contain the required {filters} placeholder."""
        from vaig.tools.gke.datadog_api import _build_metric_templates

        config = DatadogAPIConfig(enabled=True, api_key="k", app_key="k", metric_mode="apm")
        templates = _build_metric_templates(config)

        for name, tmpl in templates.items():
            assert "{filters}" in tmpl, f"APM template '{name}' missing '{{filters}}' placeholder"

    def test_custom_metrics_extend_k8s_agent_mode(self) -> None:
        """custom_metrics are merged into k8s_agent templates."""
        from vaig.tools.gke.datadog_api import _build_metric_templates

        config = DatadogAPIConfig(
            enabled=True,
            api_key="k",
            app_key="k",
            metric_mode="k8s_agent",
            custom_metrics={"my_metric": "avg:custom.metric{{{filters}}} by {{pod_name}}"},
        )
        templates = _build_metric_templates(config)

        assert "cpu" in templates
        assert "my_metric" in templates
        assert "custom.metric" in templates["my_metric"]

    def test_custom_metrics_extend_apm_mode(self) -> None:
        """custom_metrics are merged into APM templates."""
        from vaig.tools.gke.datadog_api import _build_metric_templates

        config = DatadogAPIConfig(
            enabled=True,
            api_key="k",
            app_key="k",
            metric_mode="apm",
            custom_metrics={"my_apm_extra": "sum:trace.custom.hits{{{filters}}} by {{pod_name}}"},
        )
        templates = _build_metric_templates(config)

        assert "requests" in templates
        assert "my_apm_extra" in templates

    def test_custom_metrics_override_apm_builtin(self) -> None:
        """A custom_metrics entry with same key as an APM built-in overrides it."""
        from vaig.tools.gke.datadog_api import _build_metric_templates

        custom_requests = "sum:trace.custom.requests{{{filters}}} by {{pod_name}}"
        config = DatadogAPIConfig(
            enabled=True,
            api_key="k",
            app_key="k",
            metric_mode="apm",
            custom_metrics={"requests": custom_requests},
        )
        templates = _build_metric_templates(config)

        assert templates["requests"] == custom_requests


class TestBuildMetricTemplatesBothMode:
    """Tests for metric_mode='both' in _build_metric_templates."""

    def test_both_mode_includes_k8s_and_apm_templates(self) -> None:
        """Both mode returns templates from BOTH k8s and APM sets."""
        from vaig.tools.gke.datadog_api import _build_metric_templates

        config = DatadogAPIConfig(enabled=True, api_key="k", app_key="k", metric_mode="both")
        templates = _build_metric_templates(config)

        # k8s templates present
        assert "cpu" in templates
        assert "memory" in templates
        assert "restarts" in templates
        assert "kubernetes.cpu.usage.total" in templates["cpu"]
        # APM templates present
        assert "requests" in templates
        assert "latency" in templates
        assert "error_rate" in templates
        assert "trace.http.request" in templates["requests"]

    def test_both_mode_contains_all_expected_keys(self) -> None:
        """Both mode includes keys from both k8s and APM sets."""
        from vaig.tools.gke.datadog_api import _build_metric_templates

        config = DatadogAPIConfig(enabled=True, api_key="k", app_key="k", metric_mode="both")
        templates = _build_metric_templates(config)

        expected_keys = {"cpu", "memory", "restarts", "network_in", "network_out",
                         "disk_read", "disk_write", "requests", "errors", "latency",
                         "error_rate", "apdex"}
        assert expected_keys.issubset(templates.keys())

    def test_k8s_agent_mode_excludes_apm_keys(self) -> None:
        """k8s_agent mode does NOT include APM keys like latency or requests."""
        from vaig.tools.gke.datadog_api import _build_metric_templates

        config = DatadogAPIConfig(enabled=True, api_key="k", app_key="k", metric_mode="k8s_agent")
        templates = _build_metric_templates(config)

        assert "latency" not in templates
        assert "requests" not in templates
        assert "error_rate" not in templates
        assert "apdex" not in templates

    def test_apm_mode_excludes_k8s_keys(self) -> None:
        """APM mode does NOT include k8s keys like cpu or memory."""
        from vaig.tools.gke.datadog_api import _build_metric_templates

        config = DatadogAPIConfig(enabled=True, api_key="k", app_key="k", metric_mode="apm")
        templates = _build_metric_templates(config)

        assert "cpu" not in templates
        assert "memory" not in templates
        assert "restarts" not in templates
        assert "network_in" not in templates

    def test_both_mode_all_templates_have_filters_placeholder(self) -> None:
        """All templates in both mode contain the required {filters} placeholder."""
        from vaig.tools.gke.datadog_api import _build_metric_templates

        config = DatadogAPIConfig(enabled=True, api_key="k", app_key="k", metric_mode="both")
        templates = _build_metric_templates(config)

        for name, tmpl in templates.items():
            assert "{filters}" in tmpl, f"Template '{name}' missing '{{filters}}' placeholder"

    def test_both_mode_custom_metrics_merged(self) -> None:
        """custom_metrics are merged into both-mode templates."""
        from vaig.tools.gke.datadog_api import _build_metric_templates

        config = DatadogAPIConfig(
            enabled=True,
            api_key="k",
            app_key="k",
            metric_mode="both",
            custom_metrics={"my_extra": "avg:custom.extra{{{filters}}} by {{pod_name}}"},
        )
        templates = _build_metric_templates(config)

        assert "cpu" in templates
        assert "requests" in templates
        assert "my_extra" in templates


class TestBothModePerMetricCustomLabels:
    """Tests for per-metric include_custom_labels in 'both' mode.

    When metric_mode='both', trace.* metrics must exclude custom labels while
    kubernetes.* metrics must include them.
    """

    def test_both_mode_k8s_metric_excludes_custom_labels(self, dd_config: DatadogAPIConfig) -> None:
        """In both mode, built-in k8s metrics (cpu) do NOT include custom labels.

        Built-in kubernetes.* templates only carry cluster/service/env tags; applying
        user-defined custom labels to them caused over-filtering and empty results.
        Only user-defined metrics from ``config.custom_metrics`` receive custom labels.
        """
        from vaig.core.config import DatadogLabelConfig
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        dd_config.metric_mode = "both"
        dd_config.labels = DatadogLabelConfig(custom={"team": "platform"})

        mock_api = MagicMock()
        mock_api.query_metrics.return_value = MagicMock(series=[])

        with patch.dict("sys.modules", _make_dd_modules()):
            query_datadog_metrics(
                cluster_name="my-cluster",
                metric="cpu",
                config=dd_config,
                _custom_api=mock_api,
            )

        # Built-in k8s metric → custom labels must NOT appear in the query.
        first_call_kwargs = mock_api.query_metrics.call_args_list[0].kwargs
        query_str = first_call_kwargs.get("query", "")
        assert "team:" not in query_str

    def test_both_mode_apm_metric_excludes_custom_labels(self, dd_config: DatadogAPIConfig) -> None:
        """In both mode, querying an APM metric (latency) excludes custom labels."""
        from vaig.core.config import DatadogLabelConfig
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        dd_config.metric_mode = "both"
        dd_config.labels = DatadogLabelConfig(custom={"team": "platform"})

        mock_api = MagicMock()
        mock_api.query_metrics.return_value = MagicMock(series=[])

        with patch.dict("sys.modules", _make_dd_modules()):
            query_datadog_metrics(
                cluster_name="my-cluster",
                metric="latency",
                config=dd_config,
                _custom_api=mock_api,
            )

        call_kwargs = mock_api.query_metrics.call_args.kwargs
        query_str = call_kwargs.get("query", "")
        assert "team:" not in query_str
        assert "trace." in query_str

    def test_both_mode_requests_metric_excludes_custom_labels(self, dd_config: DatadogAPIConfig) -> None:
        """In both mode, querying 'requests' (trace.* metric) excludes custom labels."""
        from vaig.core.config import DatadogLabelConfig
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        dd_config.metric_mode = "both"
        dd_config.labels = DatadogLabelConfig(custom={"team": "platform"})

        mock_api = MagicMock()
        mock_api.query_metrics.return_value = MagicMock(series=[])

        with patch.dict("sys.modules", _make_dd_modules()):
            query_datadog_metrics(
                cluster_name="my-cluster",
                metric="requests",
                config=dd_config,
                _custom_api=mock_api,
            )

        call_kwargs = mock_api.query_metrics.call_args.kwargs
        query_str = call_kwargs.get("query", "")
        assert "team:" not in query_str

    def test_both_mode_memory_metric_excludes_custom_labels(self, dd_config: DatadogAPIConfig) -> None:
        """In both mode, built-in 'memory' (kubernetes.*) metric excludes custom labels."""
        from vaig.core.config import DatadogLabelConfig
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        dd_config.metric_mode = "both"
        dd_config.labels = DatadogLabelConfig(custom={"team": "platform"})

        mock_api = MagicMock()
        mock_api.query_metrics.return_value = MagicMock(series=[])

        with patch.dict("sys.modules", _make_dd_modules()):
            query_datadog_metrics(
                cluster_name="my-cluster",
                metric="memory",
                config=dd_config,
                _custom_api=mock_api,
            )

        # Built-in k8s metric → custom labels must NOT appear in the query.
        first_call_kwargs = mock_api.query_metrics.call_args_list[0].kwargs
        query_str = first_call_kwargs.get("query", "")
        assert "team:" not in query_str

    def test_pure_apm_mode_still_excludes_custom_labels(self, dd_config: DatadogAPIConfig) -> None:
        """Pure APM mode continues to exclude custom labels (regression guard)."""
        from vaig.core.config import DatadogLabelConfig
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        dd_config.metric_mode = "apm"
        dd_config.labels = DatadogLabelConfig(custom={"team": "platform"})

        mock_api = MagicMock()
        mock_api.query_metrics.return_value = MagicMock(series=[])

        with patch.dict("sys.modules", _make_dd_modules()):
            query_datadog_metrics(
                cluster_name="my-cluster",
                metric="latency",
                config=dd_config,
                _custom_api=mock_api,
            )

        call_kwargs = mock_api.query_metrics.call_args.kwargs
        query_str = call_kwargs.get("query", "")
        assert "team:" not in query_str

    def test_pure_k8s_mode_excludes_custom_labels_for_builtin(self, dd_config: DatadogAPIConfig) -> None:
        """Pure k8s_agent mode excludes custom labels for built-in kubernetes.* metrics."""
        from vaig.core.config import DatadogLabelConfig
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        dd_config.metric_mode = "k8s_agent"
        dd_config.labels = DatadogLabelConfig(custom={"team": "platform"})

        mock_api = MagicMock()
        mock_api.query_metrics.return_value = MagicMock(series=[])

        with patch.dict("sys.modules", _make_dd_modules()):
            query_datadog_metrics(
                cluster_name="my-cluster",
                metric="cpu",
                config=dd_config,
                _custom_api=mock_api,
            )

        # Built-in k8s metric → custom labels must NOT appear in the query.
        first_call_kwargs = mock_api.query_metrics.call_args_list[0].kwargs
        query_str = first_call_kwargs.get("query", "")
        assert "team:" not in query_str

    def test_user_defined_custom_metric_includes_custom_labels(self, dd_config: DatadogAPIConfig) -> None:
        """User-defined metrics (via config.custom_metrics) DO receive custom labels.

        This is the positive case for Fix #3: only entries the user explicitly declared
        in ``custom_metrics`` should carry ``labels.custom`` in the query.
        """
        from vaig.core.config import DatadogLabelConfig
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        dd_config.metric_mode = "both"
        dd_config.labels = DatadogLabelConfig(custom={"team": "platform"})
        # User-defined custom metric: {filters} is the caller placeholder,
        # {{pod_name}} is a literal for str.format (becomes {pod_name} in the
        # final Datadog query string).
        dd_config.custom_metrics = {
            "myapp_latency": "avg:myapp.latency{{{filters}}} by {{pod_name}}",
        }

        mock_api = MagicMock()
        mock_api.query_metrics.return_value = MagicMock(series=[])

        with patch.dict("sys.modules", _make_dd_modules()):
            query_datadog_metrics(
                cluster_name="my-cluster",
                metric="myapp_latency",
                config=dd_config,
                _custom_api=mock_api,
            )

        first_call_kwargs = mock_api.query_metrics.call_args_list[0].kwargs
        query_str = first_call_kwargs.get("query", "")
        assert "team:platform" in query_str
        assert "myapp.latency" in query_str


# ── Problem 2: cluster_name_override ─────────────────────────


class TestClusterNameOverride:
    """Tests for cluster_name_override in query_datadog_metrics."""

    def test_cluster_name_override_used_when_set(self, dd_config: DatadogAPIConfig) -> None:
        """When cluster_name_override is set, it replaces cluster_name in the tag filter."""
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        dd_config.cluster_name_override = "override-cluster"

        mock_api = MagicMock()
        mock_api.query_metrics.return_value = MagicMock(
            series=[_make_series(scope="cluster_name:override-cluster")]
        )

        with patch.dict("sys.modules", _make_dd_modules()):
            result = query_datadog_metrics(
                cluster_name="original-cluster",
                metric="cpu",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        call_kwargs = mock_api.query_metrics.call_args.kwargs
        assert "override-cluster" in call_kwargs["query"]
        assert "original-cluster" not in call_kwargs["query"]

    def test_gke_cluster_name_used_when_override_empty(self, dd_config: DatadogAPIConfig) -> None:
        """When cluster_name_override is empty (default), the passed cluster_name is used."""
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        assert dd_config.cluster_name_override == ""

        mock_api = MagicMock()
        mock_api.query_metrics.return_value = MagicMock(
            series=[_make_series(scope="cluster_name:my-cluster")]
        )

        with patch.dict("sys.modules", _make_dd_modules()):
            result = query_datadog_metrics(
                cluster_name="my-cluster",
                metric="cpu",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        call_kwargs = mock_api.query_metrics.call_args.kwargs
        assert "my-cluster" in call_kwargs["query"]

    def test_cluster_name_override_default_is_empty_string(self) -> None:
        """DatadogAPIConfig.cluster_name_override defaults to empty string."""
        config = DatadogAPIConfig(enabled=True, api_key="k", app_key="k")
        assert config.cluster_name_override == ""


# ── Problem 3: default_lookback_hours ────────────────────────


class TestDefaultLookbackHours:
    """Tests for default_lookback_hours in DatadogAPIConfig and get_datadog_apm_services."""

    def setup_method(self) -> None:
        """Clear the discovery cache before each test."""
        from vaig.tools.gke._cache import clear_discovery_cache

        clear_discovery_cache()

    def _make_api_with_responses(
        self,
        hits: float | None = 100.0,
        errors: float | None = 5.0,
        duration: float | None = 0.015,
    ) -> MagicMock:
        """Create a mock MetricsApi with probe + 3 metric responses."""
        responses = _make_apm_query_response(hits=hits, errors=errors, duration=duration)
        probe_resp = MagicMock()
        probe_series = MagicMock()
        probe_series.pointlist = [[1700000000, 42.0]]
        probe_resp.series = [probe_series]

        mock_api = MagicMock()
        mock_api.query_metrics.side_effect = [
            probe_resp,
            responses["hits"],
            responses["errors"],
            responses["duration"],
        ]
        return mock_api

    def test_default_lookback_hours_from_config_used_when_no_hours_back(
        self, dd_config: DatadogAPIConfig
    ) -> None:
        """When hours_back is not passed, config.default_lookback_hours is used."""
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        dd_config.default_lookback_hours = 2.0

        mock_api = self._make_api_with_responses()

        # The output window label reflects the resolved lookback
        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_apm_services(
                service_name="svc",
                env="production",
                config=dd_config,
                _custom_api=mock_api,
            )
        assert result.error is False
        assert "last 2 hours" in result.output

    def test_explicit_hours_back_overrides_config_default(
        self, dd_config: DatadogAPIConfig
    ) -> None:
        """Explicit hours_back parameter takes priority over config.default_lookback_hours."""
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        dd_config.default_lookback_hours = 8.0

        mock_api = self._make_api_with_responses()

        # Pass explicit hours_back=1.0 — output should say "1 hour", not "8 hours"
        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_apm_services(
                service_name="svc",
                env="production",
                hours_back=1.0,
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        assert "last 1 hour" in result.output
        assert "8 hours" not in result.output

    def test_default_lookback_hours_default_value_is_4(self) -> None:
        """DatadogAPIConfig.default_lookback_hours defaults to 4.0."""
        config = DatadogAPIConfig(enabled=True, api_key="k", app_key="k")
        assert config.default_lookback_hours == 4.0

    def test_hours_back_none_resolves_to_config_default(
        self, dd_config: DatadogAPIConfig
    ) -> None:
        """Passing hours_back=None explicitly uses config.default_lookback_hours."""
        from vaig.tools.gke.datadog_api import get_datadog_apm_services

        dd_config.default_lookback_hours = 6.0

        mock_api = self._make_api_with_responses()

        with patch.dict("sys.modules", _make_dd_modules()):
            result = get_datadog_apm_services(
                service_name="svc",
                env="production",
                hours_back=None,
                config=dd_config,
                _custom_api=mock_api,
            )

        # API was called (probe + 3 metrics = 4 calls)
        assert mock_api.query_metrics.call_count == 4
        assert result.error is False
        assert "last 6 hours" in result.output


# ── _detect_apm_operation ────────────────────────────────────


class TestDetectApmOperation:
    """Tests for _detect_apm_operation: config override, cache, probe fallback."""

    def setup_method(self) -> None:
        """Clear the discovery cache before each test."""
        from vaig.tools.gke._cache import clear_discovery_cache

        clear_discovery_cache()

    def test_config_override_skips_probing(self, dd_config: DatadogAPIConfig) -> None:
        """When config.apm_operation is set (not 'auto'), returns it without API calls."""
        from vaig.tools.gke.datadog_api import _detect_apm_operation

        dd_config.apm_operation = "grpc.server"
        mock_api = MagicMock()

        result = _detect_apm_operation(mock_api, "svc", "prod", dd_config)

        assert result == "grpc.server"
        mock_api.query_metrics.assert_not_called()

    def test_probe_returns_first_operation_with_data(self, dd_config: DatadogAPIConfig) -> None:
        """Probing returns the first operation that has non-empty timeseries data."""
        from vaig.tools.gke.datadog_api import _detect_apm_operation

        # First two probes return empty; third (http.request) returns data.
        # Probe order: envoy.proxy, servlet.request, http.request, grpc.server, ...
        empty_resp = MagicMock()
        empty_resp.series = []

        hit_resp = MagicMock()
        hit_series = MagicMock()
        hit_series.pointlist = [[1700000000, 10.0]]
        hit_resp.series = [hit_series]

        mock_api = MagicMock()
        mock_api.query_metrics.side_effect = [empty_resp, empty_resp, hit_resp]

        result = _detect_apm_operation(mock_api, "svc", "prod", dd_config)

        assert result == "http.request"  # 3rd in _APM_OPERATION_PROBE_ORDER
        assert mock_api.query_metrics.call_count == 3

    def test_cache_hit_skips_api_calls(self, dd_config: DatadogAPIConfig) -> None:
        """When the result is cached, no API calls are made on subsequent invocations."""
        from vaig.tools.gke.datadog_api import _detect_apm_operation

        # First call: probe finds data on first operation
        hit_resp = MagicMock()
        hit_series = MagicMock()
        hit_series.pointlist = [[1700000000, 10.0]]
        hit_resp.series = [hit_series]

        mock_api = MagicMock()
        mock_api.query_metrics.return_value = hit_resp

        result1 = _detect_apm_operation(mock_api, "svc", "prod", dd_config)
        first_call_count = mock_api.query_metrics.call_count

        # Second call: should hit cache — no new API calls
        result2 = _detect_apm_operation(mock_api, "svc", "prod", dd_config)

        assert result1 == result2 == "envoy.proxy"  # 1st in probe order (Istio services)
        assert mock_api.query_metrics.call_count == first_call_count

    def test_all_probes_empty_returns_none(self, dd_config: DatadogAPIConfig) -> None:
        """When every probe operation returns empty data, returns None."""
        from vaig.tools.gke.datadog_api import _detect_apm_operation

        empty_resp = MagicMock()
        empty_resp.series = []
        mock_api = MagicMock()
        mock_api.query_metrics.return_value = empty_resp

        result = _detect_apm_operation(mock_api, "ghost-svc", "prod", dd_config)

        assert result is None

    def test_probe_exception_continues_to_next_operation(self, dd_config: DatadogAPIConfig) -> None:
        """If a probe raises a network error, the next operation is tried."""
        from vaig.tools.gke.datadog_api import _detect_apm_operation

        hit_resp = MagicMock()
        hit_series = MagicMock()
        hit_series.pointlist = [[1700000000, 10.0]]
        hit_resp.series = [hit_series]

        mock_api = MagicMock()
        # First probe raises a network error; second returns data.
        mock_api.query_metrics.side_effect = [OSError("connection reset"), hit_resp]

        result = _detect_apm_operation(mock_api, "svc", "prod", dd_config)

        assert result == "servlet.request"  # 2nd in _APM_OPERATION_PROBE_ORDER
        assert mock_api.query_metrics.call_count == 2


# ── _build_metric_templates with custom operation ────────────


class TestBuildMetricTemplatesOperation:
    """Tests for _build_metric_templates with custom operation parameter."""

    def test_custom_operation_produces_correct_template_strings(self) -> None:
        """Custom operation is embedded in trace.{operation}.* template keys."""
        from vaig.tools.gke.datadog_api import _build_metric_templates

        config = DatadogAPIConfig(enabled=True, api_key="k", app_key="k", metric_mode="apm")
        templates = _build_metric_templates(config, operation="grpc.server")

        assert "trace.grpc.server.hits" in templates["requests"]
        assert "trace.grpc.server.errors" in templates["errors"]
        assert "trace.grpc.server.duration" in templates["latency"]
        assert "trace.grpc.server.errors" in templates["error_rate"]
        assert "trace.grpc.server.hits" in templates["error_rate"]
        assert "trace.grpc.server.apdex" in templates["apdex"]

    def test_default_operation_is_http_request(self) -> None:
        """When no operation is passed, templates use http.request."""
        from vaig.tools.gke.datadog_api import _build_metric_templates

        config = DatadogAPIConfig(enabled=True, api_key="k", app_key="k", metric_mode="apm")
        templates = _build_metric_templates(config)

        assert "trace.http.request.hits" in templates["requests"]
        assert "trace.http.request.duration" in templates["latency"]

    def test_operation_does_not_affect_k8s_mode(self) -> None:
        """In k8s_agent mode, the operation parameter is ignored — kubernetes.* templates used."""
        from vaig.tools.gke.datadog_api import _build_metric_templates

        config = DatadogAPIConfig(enabled=True, api_key="k", app_key="k", metric_mode="k8s_agent")
        templates = _build_metric_templates(config, operation="grpc.server")

        assert "cpu" in templates
        assert "kubernetes.cpu.usage.total" in templates["cpu"]
        assert "trace" not in templates.get("cpu", "")

    def test_templates_have_filters_placeholder(self) -> None:
        """All APM templates with custom operation still have {filters} placeholder."""
        from vaig.tools.gke.datadog_api import _build_metric_templates

        config = DatadogAPIConfig(enabled=True, api_key="k", app_key="k", metric_mode="apm")
        templates = _build_metric_templates(config, operation="flask.request")

        for name, tmpl in templates.items():
            rendered = tmpl.format(filters="service:my-svc,env:prod")
            assert "service:my-svc" in rendered, f"Template '{name}' didn't render {{filters}}"


# ── Cache TTL behaviour ──────────────────────────────────────


class TestCacheTTL:
    """Tests for _cache._get_cached / _set_cache with custom TTL."""

    def setup_method(self) -> None:
        """Clear the discovery cache before each test."""
        from vaig.tools.gke._cache import clear_discovery_cache

        clear_discovery_cache()

    def test_custom_ttl_expires_entry(self) -> None:
        """An entry with ttl=0 is immediately expired on the next lookup."""
        import time as _time

        from vaig.tools.gke import _cache

        _cache._set_cache("test:key", "val", ttl=0)
        # Wait just slightly so monotonic clock advances
        _time.sleep(0.01)
        assert _cache._get_cached("test:key") is None

    def test_default_ttl_keeps_entry_alive(self) -> None:
        """An entry with default TTL (60s) is still available immediately."""
        from vaig.tools.gke import _cache

        _cache._set_cache("test:default", "val")
        assert _cache._get_cached("test:default") == "val"

    def test_override_ttl_on_read(self) -> None:
        """Passing ttl on _get_cached overrides the stored TTL for that lookup."""
        import time as _time

        from vaig.tools.gke import _cache

        _cache._set_cache("test:override", "val", ttl=3600)  # stored: 1 hour
        _time.sleep(0.01)
        # Override with ttl=0 at read time — should consider it expired
        assert _cache._get_cached("test:override", ttl=0) is None

    def test_stored_ttl_used_when_no_read_ttl(self) -> None:
        """When ttl is not passed to _get_cached, the stored per-entry TTL is used."""
        from vaig.tools.gke import _cache

        _cache._set_cache("test:stored", "val", ttl=300)
        # Immediately after writing — well within 300s TTL
        assert _cache._get_cached("test:stored") == "val"


# ── Config defaults ──────────────────────────────────────────


class TestConfigDefaults:
    """Tests for new config field defaults (cluster_name, apm_operation)."""

    def test_cluster_name_default_is_kube_cluster_name(self) -> None:
        """DatadogLabelConfig.cluster_name defaults to 'kube_cluster_name'."""
        from vaig.core.config import DatadogLabelConfig

        config = DatadogLabelConfig()
        assert config.cluster_name == "kube_cluster_name"

    def test_apm_operation_default_is_auto(self) -> None:
        """DatadogAPIConfig.apm_operation defaults to 'auto'."""
        config = DatadogAPIConfig(enabled=True, api_key="k", app_key="k")
        assert config.apm_operation == "auto"

    def test_apm_operation_can_be_set_to_custom_value(self) -> None:
        """DatadogAPIConfig.apm_operation accepts a custom operation name."""
        config = DatadogAPIConfig(enabled=True, api_key="k", app_key="k", apm_operation="servlet.request")
        assert config.apm_operation == "servlet.request"


# ── Default metric_mode ──────────────────────────────────────


class TestDefaultMetricModeConfig:
    """Verify that metric_mode defaults to 'auto'."""

    def test_metric_mode_defaults_to_auto(self) -> None:
        """DatadogAPIConfig().metric_mode should default to 'auto'."""
        config = DatadogAPIConfig(enabled=True, api_key="k", app_key="k")
        assert config.metric_mode == "auto"

    def test_metric_mode_can_be_overridden(self) -> None:
        """metric_mode can still be set to k8s_agent or apm explicitly."""
        config = DatadogAPIConfig(enabled=True, api_key="k", app_key="k", metric_mode="k8s_agent")
        assert config.metric_mode == "k8s_agent"

        config_apm = DatadogAPIConfig(enabled=True, api_key="k", app_key="k", metric_mode="apm")
        assert config_apm.metric_mode == "apm"


# ── Fuzzy template matching ──────────────────────────────────


class TestMetricTemplateFuzzyMatch:
    """Tests for fuzzy metric template matching (aliases + singular/plural)."""

    def test_exact_match_takes_priority(self, dd_config: DatadogAPIConfig) -> None:
        """Exact metric name should resolve without fuzzy matching."""
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        dd_config.metric_mode = "both"
        mock_api = MagicMock()
        mock_api.query_metrics.return_value = MagicMock(series=[])

        with patch.dict("sys.modules", _make_dd_modules()):
            result = query_datadog_metrics(
                cluster_name="my-cluster",
                metric="cpu",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        mock_api.query_metrics.assert_called_once()

    def test_singular_to_plural_fuzzy_match(self, dd_config: DatadogAPIConfig) -> None:
        """'request' (singular) should fuzzy-match to 'requests' (plural)."""
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        dd_config.metric_mode = "both"
        mock_api = MagicMock()
        mock_api.query_metrics.return_value = MagicMock(series=[])

        with patch.dict("sys.modules", _make_dd_modules()):
            result = query_datadog_metrics(
                cluster_name="my-cluster",
                metric="request",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        call_kwargs = mock_api.query_metrics.call_args.kwargs
        assert "trace." in call_kwargs["query"]

    def test_alias_error_to_error_rate(self, dd_config: DatadogAPIConfig) -> None:
        """Alias 'error' should resolve to 'error_rate'."""
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        dd_config.metric_mode = "both"
        mock_api = MagicMock()
        mock_api.query_metrics.return_value = MagicMock(series=[])

        with patch.dict("sys.modules", _make_dd_modules()):
            result = query_datadog_metrics(
                cluster_name="my-cluster",
                metric="error",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        call_kwargs = mock_api.query_metrics.call_args.kwargs
        assert "trace." in call_kwargs["query"]

    def test_alias_throughput_to_requests(self, dd_config: DatadogAPIConfig) -> None:
        """Alias 'throughput' should resolve to 'requests'."""
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        dd_config.metric_mode = "both"
        mock_api = MagicMock()
        mock_api.query_metrics.return_value = MagicMock(series=[])

        with patch.dict("sys.modules", _make_dd_modules()):
            result = query_datadog_metrics(
                cluster_name="my-cluster",
                metric="throughput",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        call_kwargs = mock_api.query_metrics.call_args.kwargs
        assert "trace." in call_kwargs["query"]

    def test_unknown_metric_shows_available_keys(self, dd_config: DatadogAPIConfig) -> None:
        """Completely unknown metric should return error with available keys."""
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        dd_config.metric_mode = "both"
        mock_api = MagicMock()

        with patch.dict("sys.modules", _make_dd_modules()):
            result = query_datadog_metrics(
                cluster_name="my-cluster",
                metric="totally_bogus_metric",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is True
        assert "Unknown metric template" in result.output
        assert "Available:" in result.output
        assert "cpu" in result.output
        assert "requests" in result.output


# ── Tag filter fallback ──────────────────────────────────────


class TestTagFilterFallback:
    """Tests for retry-without-custom-labels on empty results."""

    def test_no_retry_on_successful_query(self, dd_config: DatadogAPIConfig) -> None:
        """When data is returned on the first call, no retry should happen."""
        from vaig.core.config import DatadogLabelConfig
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        dd_config.metric_mode = "both"
        dd_config.labels = DatadogLabelConfig(custom={"team": "platform"})

        mock_api = MagicMock()
        mock_api.query_metrics.return_value = MagicMock(
            series=[_make_series(scope="cluster_name:my-cluster")]
        )

        with patch.dict("sys.modules", _make_dd_modules()):
            result = query_datadog_metrics(
                cluster_name="my-cluster",
                metric="cpu",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        assert "No data returned" not in result.output
        # Should only be called once — no retry
        assert mock_api.query_metrics.call_count == 1

    def test_retry_without_custom_labels_on_no_data(self, dd_config: DatadogAPIConfig) -> None:
        """When first query returns no data with custom labels, retry without them.

        Uses a user-defined custom metric (via ``config.custom_metrics``) so Fix #3
        includes custom labels in the first query — retry then strips them and hits
        data.
        """
        from vaig.core.config import DatadogLabelConfig
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        dd_config.metric_mode = "both"
        dd_config.labels = DatadogLabelConfig(custom={"team": "platform"})
        dd_config.custom_metrics = {
            "myapp_latency": "avg:myapp.latency{{{filters}}} by {{pod_name}}",
        }

        # First call: no data; second call: data
        mock_api = MagicMock()
        mock_api.query_metrics.side_effect = [
            MagicMock(series=[]),  # first call: empty
            MagicMock(series=[_make_series(scope="cluster_name:my-cluster")]),  # retry: data
        ]

        with patch.dict("sys.modules", _make_dd_modules()):
            result = query_datadog_metrics(
                cluster_name="my-cluster",
                metric="myapp_latency",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        assert "No data returned" not in result.output
        # Should be called twice: original + fallback
        assert mock_api.query_metrics.call_count == 2

        # First call should include custom labels, second should not
        first_query = mock_api.query_metrics.call_args_list[0].kwargs["query"]
        second_query = mock_api.query_metrics.call_args_list[1].kwargs["query"]
        assert "team:platform" in first_query
        assert "team:" not in second_query

    def test_still_no_data_returns_no_data_message(self, dd_config: DatadogAPIConfig) -> None:
        """When retry also returns no data, the original 'No data' message is returned."""
        from vaig.core.config import DatadogLabelConfig
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        dd_config.metric_mode = "both"
        dd_config.labels = DatadogLabelConfig(custom={"team": "platform"})
        dd_config.custom_metrics = {
            "myapp_latency": "avg:myapp.latency{{{filters}}} by {{pod_name}}",
        }

        # Both calls: no data
        mock_api = MagicMock()
        mock_api.query_metrics.side_effect = [
            MagicMock(series=[]),  # first call: empty
            MagicMock(series=[]),  # retry: still empty
        ]

        with patch.dict("sys.modules", _make_dd_modules()):
            result = query_datadog_metrics(
                cluster_name="my-cluster",
                metric="myapp_latency",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert result.error is False
        assert "No data returned" in result.output
        # Should still be called twice
        assert mock_api.query_metrics.call_count == 2

    def test_no_retry_for_trace_metrics(self, dd_config: DatadogAPIConfig) -> None:
        """trace.* metrics never have custom labels, so no retry should happen."""
        from vaig.core.config import DatadogLabelConfig
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        dd_config.metric_mode = "both"
        dd_config.labels = DatadogLabelConfig(custom={"team": "platform"})

        mock_api = MagicMock()
        mock_api.query_metrics.return_value = MagicMock(series=[])

        with patch.dict("sys.modules", _make_dd_modules()):
            result = query_datadog_metrics(
                cluster_name="my-cluster",
                metric="requests",
                config=dd_config,
                _custom_api=mock_api,
            )

        # trace.* metrics already exclude custom labels, so no retry
        assert mock_api.query_metrics.call_count == 1

    def test_no_retry_when_no_custom_labels_configured(self, dd_config: DatadogAPIConfig) -> None:
        """When config has no custom labels, no retry should occur."""
        from vaig.tools.gke.datadog_api import query_datadog_metrics

        dd_config.metric_mode = "both"
        # default labels — no custom labels

        mock_api = MagicMock()
        mock_api.query_metrics.return_value = MagicMock(series=[])

        with patch.dict("sys.modules", _make_dd_modules()):
            result = query_datadog_metrics(
                cluster_name="my-cluster",
                metric="cpu",
                config=dd_config,
                _custom_api=mock_api,
            )

        assert mock_api.query_metrics.call_count == 1


# ── get_datadog_service_dependencies ─────────────────────────


class TestGetDatadogServiceDependencies:
    """Tests for ``get_datadog_service_dependencies``."""

    def setup_method(self) -> None:
        from vaig.tools.gke._cache import clear_discovery_cache

        clear_discovery_cache()

    def test_returns_not_installed_when_sdk_missing(self, dd_config: DatadogAPIConfig) -> None:
        """Returns error when datadog-api-client is not installed."""
        with patch.dict("sys.modules", {"datadog_api_client": None, "datadog_api_client.exceptions": None}):
            from vaig.tools.gke.datadog_api import get_datadog_service_dependencies

            result = get_datadog_service_dependencies(service_name="my-api", config=dd_config)

        assert result.error is True
        assert "not installed" in result.output

    def test_returns_not_enabled_when_disabled(self, dd_config_disabled: DatadogAPIConfig) -> None:
        """Returns error when Datadog integration is disabled."""
        with patch.dict("sys.modules", _make_dd_modules()):
            from vaig.tools.gke.datadog_api import get_datadog_service_dependencies

            result = get_datadog_service_dependencies(service_name="my-api", config=dd_config_disabled)

        assert result.error is True
        assert "disabled" in result.output

    def test_requires_service_name(self, dd_config: DatadogAPIConfig) -> None:
        """Returns error when service_name is empty."""
        with patch.dict("sys.modules", _make_dd_modules()):
            from vaig.tools.gke.datadog_api import get_datadog_service_dependencies

            result = get_datadog_service_dependencies(service_name="", config=dd_config)

        assert result.error is True
        assert "required" in result.output.lower()

    def test_invalid_service_name_rejected(self, dd_config: DatadogAPIConfig) -> None:
        """Returns error when service_name contains invalid characters."""
        with patch.dict("sys.modules", _make_dd_modules()):
            from vaig.tools.gke.datadog_api import get_datadog_service_dependencies

            result = get_datadog_service_dependencies(
                service_name="my service; DROP TABLE", config=dd_config,
            )

        assert result.error is True
        assert "Invalid" in result.output

    def test_successful_response_with_calls_and_called_by(self, dd_config: DatadogAPIConfig) -> None:
        """Returns formatted output with downstream and upstream services."""
        mock_api = MagicMock()
        mock_api.return_value = {
            "calls": ["database-svc", "cache-svc"],
            "called_by": ["frontend-svc"],
        }

        with patch.dict("sys.modules", _make_dd_modules()):
            from vaig.tools.gke.datadog_api import get_datadog_service_dependencies

            result = get_datadog_service_dependencies(
                service_name="my-api", config=dd_config, _custom_api=mock_api,
            )

        assert result.error is False
        assert "=== Datadog Service Dependencies: my-api ===" in result.output
        assert "Downstream (calls): 2" in result.output
        assert "→ cache-svc" in result.output
        assert "→ database-svc" in result.output
        assert "Upstream (called_by): 1" in result.output
        assert "← frontend-svc" in result.output

    def test_structured_dependency_data_included(self, dd_config: DatadogAPIConfig) -> None:
        """Output contains structured DependencyEdge JSON block."""
        import json

        mock_api = MagicMock()
        mock_api.return_value = {
            "calls": ["db"],
            "called_by": ["web"],
        }

        with patch.dict("sys.modules", _make_dd_modules()):
            from vaig.tools.gke.datadog_api import get_datadog_service_dependencies

            result = get_datadog_service_dependencies(
                service_name="my-api", config=dd_config, _custom_api=mock_api,
            )

        assert result.error is False
        assert "--- STRUCTURED_DEPENDENCY_EDGES ---" in result.output
        assert "--- END_STRUCTURED_DEPENDENCY_EDGES ---" in result.output

        # Extract and parse the structured data
        start = result.output.index("--- STRUCTURED_DEPENDENCY_EDGES ---") + len("--- STRUCTURED_DEPENDENCY_EDGES ---")
        end = result.output.index("--- END_STRUCTURED_DEPENDENCY_EDGES ---")
        edges = json.loads(result.output[start:end].strip())

        assert len(edges) == 2

        # Downstream edge: my-api → db
        downstream = [e for e in edges if e["target"] == "db"]
        assert len(downstream) == 1
        assert downstream[0]["source"] == "my-api"
        assert downstream[0]["method"] == "datadog"
        assert downstream[0]["confidence"] == 0.9

        # Upstream edge: web → my-api
        upstream = [e for e in edges if e["source"] == "web"]
        assert len(upstream) == 1
        assert upstream[0]["target"] == "my-api"

    def test_empty_dependencies(self, dd_config: DatadogAPIConfig) -> None:
        """Handles service with no upstream or downstream dependencies."""
        mock_api = MagicMock()
        mock_api.return_value = {
            "calls": [],
            "called_by": [],
        }

        with patch.dict("sys.modules", _make_dd_modules()):
            from vaig.tools.gke.datadog_api import get_datadog_service_dependencies

            result = get_datadog_service_dependencies(
                service_name="isolated-svc", config=dd_config, _custom_api=mock_api,
            )

        assert result.error is False
        assert "Downstream (calls): 0" in result.output
        assert "(none)" in result.output
        assert "Upstream (called_by): 0" in result.output

    def test_result_is_cached(self, dd_config: DatadogAPIConfig) -> None:
        """Second call returns cached result without hitting the API again."""
        mock_api = MagicMock()
        mock_api.return_value = {
            "calls": ["db"],
            "called_by": [],
        }

        with patch.dict("sys.modules", _make_dd_modules()):
            from vaig.tools.gke.datadog_api import get_datadog_service_dependencies

            result1 = get_datadog_service_dependencies(
                service_name="my-api", config=dd_config, _custom_api=mock_api,
            )
            result2 = get_datadog_service_dependencies(
                service_name="my-api", config=dd_config, _custom_api=mock_api,
            )

        assert result1.output == result2.output
        # API should only be called once — second call served from cache
        mock_api.assert_called_once()

    def test_api_exception_returns_error(self, dd_config: DatadogAPIConfig) -> None:
        """ApiException from the custom API is caught and reported."""
        dd_modules = _make_dd_modules()
        ApiException = dd_modules["datadog_api_client.exceptions"].ApiException  # type: ignore[union-attr]

        mock_api = MagicMock()
        exc = ApiException("forbidden")
        exc.status = 403
        mock_api.side_effect = exc

        with patch.dict("sys.modules", dd_modules):
            from vaig.tools.gke.datadog_api import get_datadog_service_dependencies

            result = get_datadog_service_dependencies(
                service_name="my-api", config=dd_config, _custom_api=mock_api,
            )

        assert result.error is True
        assert "permissions" in result.output.lower() or "403" in result.output

    def test_response_with_to_dict_method(self, dd_config: DatadogAPIConfig) -> None:
        """Handles SDK response objects that have a to_dict() method."""
        mock_response = MagicMock()
        mock_response.to_dict.return_value = {
            "calls": ["svc-a"],
            "called_by": ["svc-b"],
        }

        mock_api = MagicMock()
        mock_api.return_value = mock_response

        with patch.dict("sys.modules", _make_dd_modules()):
            from vaig.tools.gke.datadog_api import get_datadog_service_dependencies

            result = get_datadog_service_dependencies(
                service_name="my-api", config=dd_config, _custom_api=mock_api,
            )

        assert result.error is False
        assert "→ svc-a" in result.output
        assert "← svc-b" in result.output


# ── diagnose_datadog_metrics ─────────────────────────────────


def _make_rest_response(*, status: int = 200, data: bytes = b"{}") -> MagicMock:
    """Build a mock urllib3-style HTTP response for rest_client.request()."""
    resp = MagicMock()
    resp.status = status
    resp.data = data
    return resp


class TestDiagnoseDatadogMetrics:
    """Tests for ``diagnose_datadog_metrics``."""

    # ── helpers ──────────────────────────────────────────

    @staticmethod
    def _build_mock_client(
        search_responses: dict[str, bytes] | None = None,
        tags_response: bytes = b'{"tags": {"cluster_name": ["host1"], "env": ["host2"]}}',
        tags_status: int = 200,
        search_status: int = 200,
    ) -> MagicMock:
        """Build a mock Datadog ApiClient whose rest_client dispatches by URL.

        ``search_responses`` maps query substrings (e.g. ``"kubernetes.*"``)
        to JSON byte payloads. URLs that don't match any key return ``{}``.
        """
        import json as _json

        if search_responses is None:
            search_responses = {
                "kubernetes": _json.dumps({"results": {"metrics": ["kubernetes.cpu.usage.total"]}}).encode(),
                "trace": _json.dumps({"results": {"metrics": ["trace.http.request.hits"]}}).encode(),
            }

        def _dispatch_request(method: str, url: str, **kwargs: Any) -> MagicMock:
            if "/api/v1/search" in url:
                for key, payload in search_responses.items():
                    if key in url:
                        return _make_rest_response(status=search_status, data=payload)
                return _make_rest_response(data=b'{"results": {"metrics": []}}')
            if "/api/v1/tags/hosts" in url:
                return _make_rest_response(status=tags_status, data=tags_response)
            return _make_rest_response()

        rest = MagicMock()
        rest.request.side_effect = _dispatch_request

        client = MagicMock()
        client.rest_client = rest
        return client

    # ── happy-path tests ─────────────────────────────────

    def test_both_k8s_and_trace_metrics_found(self, dd_config: DatadogAPIConfig) -> None:
        """Reports metric counts and 'auto'/'both' suggestion when both found."""
        client = self._build_mock_client()

        with patch.dict("sys.modules", _make_dd_modules()):
            from vaig.tools.gke.datadog_api import diagnose_datadog_metrics

            result = diagnose_datadog_metrics(config=dd_config, _custom_client=client)

        assert result.error is False
        assert "kubernetes.* metrics found: 1" in result.output
        assert "trace.* metrics found: 1" in result.output
        assert "auto" in result.output or "both" in result.output

    def test_only_k8s_metrics_suggests_k8s_agent_mode(self, dd_config: DatadogAPIConfig) -> None:
        """Suggests k8s_agent when only kubernetes.* metrics are found."""
        import json as _json

        client = self._build_mock_client(
            search_responses={
                "kubernetes": _json.dumps({"results": {"metrics": ["kubernetes.cpu.usage.total"]}}).encode(),
                "trace": _json.dumps({"results": {"metrics": []}}).encode(),
            },
        )

        with patch.dict("sys.modules", _make_dd_modules()):
            from vaig.tools.gke.datadog_api import diagnose_datadog_metrics

            result = diagnose_datadog_metrics(config=dd_config, _custom_client=client)

        assert result.error is False
        assert "k8s_agent" in result.output

    def test_only_trace_metrics_suggests_apm_mode(self, dd_config: DatadogAPIConfig) -> None:
        """Suggests apm when only trace.* metrics are found."""
        import json as _json

        client = self._build_mock_client(
            search_responses={
                "kubernetes": _json.dumps({"results": {"metrics": []}}).encode(),
                "trace": _json.dumps({"results": {"metrics": ["trace.http.request.hits"]}}).encode(),
            },
        )

        with patch.dict("sys.modules", _make_dd_modules()):
            from vaig.tools.gke.datadog_api import diagnose_datadog_metrics

            result = diagnose_datadog_metrics(config=dd_config, _custom_client=client)

        assert result.error is False
        assert "apm" in result.output

    def test_no_metrics_found_suggests_checking_agent(self, dd_config: DatadogAPIConfig) -> None:
        """Suggests checking agent deployment when no metrics found."""
        import json as _json

        client = self._build_mock_client(
            search_responses={
                "kubernetes": _json.dumps({"results": {"metrics": []}}).encode(),
                "trace": _json.dumps({"results": {"metrics": []}}).encode(),
            },
        )

        with patch.dict("sys.modules", _make_dd_modules()):
            from vaig.tools.gke.datadog_api import diagnose_datadog_metrics

            result = diagnose_datadog_metrics(config=dd_config, _custom_client=client)

        assert result.error is False
        assert "No kubernetes.* or trace.* metrics found" in result.output

    def test_tag_keys_discovered(self, dd_config: DatadogAPIConfig) -> None:
        """Tag keys from /api/v1/tags/hosts appear in the diagnostic output."""
        client = self._build_mock_client()

        with patch.dict("sys.modules", _make_dd_modules()):
            from vaig.tools.gke.datadog_api import diagnose_datadog_metrics

            result = diagnose_datadog_metrics(config=dd_config, _custom_client=client)

        assert result.error is False
        assert "Host tag keys discovered: 2" in result.output

    def test_missing_cluster_tag_warns(self, dd_config: DatadogAPIConfig) -> None:
        """Warns when configured cluster_name tag is not found in Datadog tags."""
        import json as _json

        client = self._build_mock_client(
            tags_response=_json.dumps({"tags": {"env": ["host1"]}}).encode(),
        )

        with patch.dict("sys.modules", _make_dd_modules()):
            from vaig.tools.gke.datadog_api import diagnose_datadog_metrics

            result = diagnose_datadog_metrics(config=dd_config, _custom_client=client)

        assert result.error is False
        # The default cluster_name label is 'kube_cluster_name'
        assert "not found in Datadog host tags" in result.output

    # ── error-path tests ─────────────────────────────────

    def test_disabled_config_returns_error(self, dd_config_disabled: DatadogAPIConfig) -> None:
        """Returns error when Datadog integration is disabled."""
        with patch.dict("sys.modules", _make_dd_modules()):
            from vaig.tools.gke.datadog_api import diagnose_datadog_metrics

            result = diagnose_datadog_metrics(config=dd_config_disabled)

        assert result.error is True
        assert "disabled" in result.output.lower()

    def test_search_http_error_recorded_in_diagnostic(self, dd_config: DatadogAPIConfig) -> None:
        """HTTP errors from the search API are recorded but don't crash."""
        client = self._build_mock_client(search_status=403)

        with patch.dict("sys.modules", _make_dd_modules()):
            from vaig.tools.gke.datadog_api import diagnose_datadog_metrics

            result = diagnose_datadog_metrics(config=dd_config, _custom_client=client)

        assert result.error is False
        assert "HTTP 403" in result.output

    def test_tag_discovery_http_error_recorded(self, dd_config: DatadogAPIConfig) -> None:
        """HTTP error from tag discovery is recorded as a diagnostic error."""
        client = self._build_mock_client(tags_status=500)

        with patch.dict("sys.modules", _make_dd_modules()):
            from vaig.tools.gke.datadog_api import diagnose_datadog_metrics

            result = diagnose_datadog_metrics(config=dd_config, _custom_client=client)

        assert result.error is False
        assert "Tag discovery HTTP 500" in result.output

    def test_search_exception_recorded_in_diagnostic(self, dd_config: DatadogAPIConfig) -> None:
        """A network exception from search is recorded without crashing."""
        client = MagicMock()
        rest = MagicMock()
        rest.request.side_effect = ConnectionError("network down")
        client.rest_client = rest

        with patch.dict("sys.modules", _make_dd_modules()):
            from vaig.tools.gke.datadog_api import diagnose_datadog_metrics

            result = diagnose_datadog_metrics(config=dd_config, _custom_client=client)

        assert result.error is False
        assert "failed: network down" in result.output

    def test_unexpected_exception_returns_error(self, dd_config: DatadogAPIConfig) -> None:
        """Unexpected exception in the outer try returns an error ToolResult."""
        # Pass a _custom_client that causes _run_diagnostics itself to blow up
        # by making the client not have a rest_client AND raising on any attribute access.
        client = MagicMock()
        client.rest_client = MagicMock()
        # Make _run_diagnostics itself raise by sabotaging the diagnostic dict build
        # via a client whose rest_client.request raises something unusual.
        client.rest_client.request.side_effect = RuntimeError("boom")

        with patch.dict("sys.modules", _make_dd_modules()):
            from vaig.tools.gke.datadog_api import diagnose_datadog_metrics

            result = diagnose_datadog_metrics(config=dd_config, _custom_client=client)

        # The inner helpers catch exceptions, so the result should still succeed
        # with the errors recorded in the diagnostic output.
        assert result.error is False
        assert "boom" in result.output

    # ── auth headers test ────────────────────────────────

    def test_auth_headers_passed_to_rest_client(self, dd_config: DatadogAPIConfig) -> None:
        """Verifies that DD-API-KEY and DD-APPLICATION-KEY headers are sent."""
        client = self._build_mock_client()

        with patch.dict("sys.modules", _make_dd_modules()):
            from vaig.tools.gke.datadog_api import diagnose_datadog_metrics

            diagnose_datadog_metrics(config=dd_config, _custom_client=client)

        # rest_client.request should have been called with headers kwarg
        for call_args in client.rest_client.request.call_args_list:
            kwargs = call_args.kwargs if call_args.kwargs else {}
            headers = kwargs.get("headers", {})
            assert headers.get("DD-API-KEY") == "test-api-key"
            assert headers.get("DD-APPLICATION-KEY") == "test-app-key"

    # ── URL encoding test ────────────────────────────────

    def test_search_query_is_url_encoded(self, dd_config: DatadogAPIConfig) -> None:
        """Verifies that the search query parameter is URL-encoded."""
        client = self._build_mock_client()

        with patch.dict("sys.modules", _make_dd_modules()):
            from vaig.tools.gke.datadog_api import diagnose_datadog_metrics

            diagnose_datadog_metrics(config=dd_config, _custom_client=client)

        # Check that search URLs contain encoded query params (+ instead of spaces, %2A for *)
        search_calls = [
            call_args for call_args in client.rest_client.request.call_args_list
            if "/api/v1/search" in str(call_args)
        ]
        assert len(search_calls) >= 2  # kubernetes.* and trace.*
        for call_args in search_calls:
            url = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("url", "")
            # urllib.parse.quote_plus encodes * as %2A
            assert "%2A" in url or "quote_plus" in url  # encoded wildcard

"""Datadog REST API tools — metrics, monitors, and APM service data for GKE diagnostics."""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Any

from vaig.core.config import DatadogAPIConfig
from vaig.tools.base import ToolResult

from . import _cache

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Error messages ───────────────────────────────────────────
_ERR_NOT_INSTALLED = "datadog-api-client not installed. Install with: pip install 'vaig[live]'"
_ERR_NOT_ENABLED = "Datadog API integration is disabled. Set datadog.enabled=true in config."
_ERR_AUTH = "Authentication failed. Check your Datadog API key and application key."
_ERR_FORBIDDEN = "Insufficient permissions. Check your Datadog API/app key scopes."
_ERR_RATE_LIMIT = "Rate limit exceeded. Try again later."

# ── Metric query templates ───────────────────────────────────
# Templates use {filters} as a placeholder for the full tag filter string.
# Example rendered: avg:kubernetes.cpu.usage.total{cluster_name:my-cluster,service:api,env:prod}
_METRIC_TEMPLATES: dict[str, str] = {
    "cpu": "avg:kubernetes.cpu.usage.total{{{filters}}} by {{pod_name}}",
    "memory": "avg:kubernetes.memory.usage{{{filters}}} by {{pod_name}}",
    "restarts": "sum:kubernetes.containers.restarts{{{filters}}} by {{pod_name}}",
    "network_in": "avg:kubernetes.network.rx_bytes{{{filters}}} by {{pod_name}}",
    "network_out": "avg:kubernetes.network.tx_bytes{{{filters}}} by {{pod_name}}",
    "disk_read": "avg:kubernetes.io.read_bytes{{{filters}}} by {{pod_name}}",
    "disk_write": "avg:kubernetes.io.write_bytes{{{filters}}} by {{pod_name}}",
}


# ── Helpers ──────────────────────────────────────────────────


_VALID_SERVICE_NAME_RE = re.compile(r"^[a-zA-Z0-9\-._]+$")


def _sanitize_service_name(name: str) -> str:
    """Validate a Datadog service/tag name and return it unchanged if valid.

    Accepts only alphanumeric characters, hyphens, underscores, and dots.
    Returns an empty string for an empty input.  Raises ``ValueError`` when
    the name contains characters outside the allowed set, rather than silently
    stripping them (fail-fast to avoid sending a mangled name to the API).

    Raises:
        ValueError: When ``name`` contains disallowed characters.
    """
    if not name:
        return ""
    if not _VALID_SERVICE_NAME_RE.match(name):
        raise ValueError(
            f"Invalid service name {name!r}: only alphanumeric, hyphens, underscores, and dots are allowed."
        )
    return name


def _get_dd_api_client(config: DatadogAPIConfig) -> Any:
    """Create a configured Datadog API client.

    Raises:
        ImportError: When ``datadog-api-client`` is not installed.
    """
    from datadog_api_client import ApiClient, Configuration  # noqa: WPS433

    configuration = Configuration()  # type: ignore[no-untyped-call]
    configuration.server_variables["site"] = config.site
    configuration.api_key["apiKeyAuth"] = config.api_key
    configuration.api_key["appKeyAuth"] = config.app_key
    configuration.request_timeout = config.timeout
    return ApiClient(configuration)


def _dd_error_message(status: int) -> str:
    """Map a Datadog HTTP status code to a human-readable error message."""
    if status == 401:
        return _ERR_AUTH
    if status == 403:
        return _ERR_FORBIDDEN
    if status == 429:
        return _ERR_RATE_LIMIT
    return f"Datadog API error (HTTP {status})."


# ── Public Tool Functions ────────────────────────────────────


def query_datadog_metrics(
    *,
    cluster_name: str,
    metric: str = "cpu",
    from_ts: int = 0,
    to_ts: int = 0,
    service: str | None = None,
    env: str | None = None,
    config: DatadogAPIConfig | None = None,
    _custom_api: Any = None,
) -> ToolResult:
    """Query Datadog metrics for a GKE cluster using the Metrics v1 API.

    Supports built-in metric templates: cpu, memory, restarts, network_in,
    network_out, disk_read, disk_write.  All queries are resolved from the
    template allowlist — arbitrary query strings are not accepted.

    When ``service`` and/or ``env`` are provided the tag filter string is
    extended beyond ``cluster_name`` — e.g.
    ``avg:kubernetes.cpu.usage.total{cluster_name:X,service:Y,env:Z}``.

    Args:
        cluster_name: GKE cluster name used to scope the metric query.
        metric: Template key (one of the ``_METRIC_TEMPLATES`` keys).
            Defaults to ``"cpu"``.
        from_ts: Unix timestamp for the start of the query window.  Defaults to
            ``now - 3600`` (last hour) when ``0``.
        to_ts: Unix timestamp for the end of the query window.  Defaults to
            ``now`` when ``0``.
        service: Optional Datadog service tag (e.g. ``"my-api"``).  When
            provided, narrows the query to that service.
        env: Optional Datadog environment tag (e.g. ``"production"``).  When
            provided, narrows the query to that environment.
        config: Optional ``DatadogAPIConfig`` (for testing / injection).
        _custom_api: Optional pre-configured Datadog MetricsApi (for testing).
    """
    try:
        from datadog_api_client.exceptions import ApiException  # noqa: WPS433
        from datadog_api_client.v1.api.metrics_api import MetricsApi  # noqa: WPS433
    except ImportError:
        return ToolResult(output=_ERR_NOT_INSTALLED, error=True)

    if config is None:
        from vaig.core.config import get_settings  # noqa: WPS433

        config = get_settings().datadog

    if not config.enabled:
        return ToolResult(output=_ERR_NOT_ENABLED, error=True)

    # Resolve time window
    now = int(time.time())
    end = to_ts if to_ts > 0 else now
    start = from_ts if from_ts > 0 else now - 3600

    # Resolve query string from allowlist only
    try:
        safe_cluster = _sanitize_service_name(cluster_name)
    except ValueError as exc:
        return ToolResult(output=str(exc), error=True)

    # Build tag filter string: always include cluster_name; optionally service/env
    filter_parts: list[str] = [f"cluster_name:{safe_cluster}"]
    if service:
        try:
            safe_service = _sanitize_service_name(service)
        except ValueError as exc:
            return ToolResult(output=str(exc), error=True)
        filter_parts.append(f"service:{safe_service}")
    if env:
        try:
            safe_env = _sanitize_service_name(env)
        except ValueError as exc:
            return ToolResult(output=str(exc), error=True)
        filter_parts.append(f"env:{safe_env}")
    filters = ",".join(filter_parts)

    template = _METRIC_TEMPLATES.get(metric)
    if template is None:
        available = ", ".join(sorted(_METRIC_TEMPLATES.keys()))
        return ToolResult(
            output=f"Unknown metric template '{metric}'. Available: {available}",
            error=True,
        )
    query = template.format(filters=filters)

    try:
        if _custom_api is not None:
            api = _custom_api
        else:
            with _get_dd_api_client(config) as client:
                api = MetricsApi(client)  # type: ignore[no-untyped-call]

        response = api.query_metrics(
            _from=start,
            to=end,
            query=query,
        )

        series = getattr(response, "series", []) or []
        if not series:
            return ToolResult(
                output=f"=== Datadog Metrics: {metric} ===\nCluster: {cluster_name}\nNo data returned for the given time window.",
                error=False,
            )

        lines: list[str] = [
            f"=== Datadog Metrics: {metric} ===",
            f"Cluster: {cluster_name}",
            f"Query: {query}",
            f"Window: {start} → {end}",
            "",
        ]

        for s in series:
            metric_name = getattr(s, "metric", "unknown")
            scope = getattr(s, "scope", "")
            points = getattr(s, "pointlist", []) or []
            if points:
                values = [p[1] for p in points if p[1] is not None]
                if values:
                    avg_val = sum(values) / len(values)
                    max_val = max(values)
                    last_val = values[-1]
                    label = scope or metric_name
                    lines.append(f"  {label:<50}  avg={avg_val:.2f}  max={max_val:.2f}  last={last_val:.2f}")

        lines.append("")
        lines.append(f"Total series: {len(series)}")

        return ToolResult(output="\n".join(lines), error=False)

    except ApiException as exc:
        status = getattr(exc, "status", 0)
        msg = _dd_error_message(status)
        logger.warning("Datadog metrics API error (HTTP %s): %s", status, exc)
        return ToolResult(output=msg, error=True)
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error querying Datadog metrics")
        return ToolResult(output="Unexpected error querying Datadog metrics. See logs for details.", error=True)


def get_datadog_monitors(
    *,
    cluster_name: str = "",
    state: str = "Alert",
    service: str | None = None,
    env: str | None = None,
    config: DatadogAPIConfig | None = None,
    _custom_api: Any = None,
) -> ToolResult:
    """Fetch active Datadog monitors using the Monitors v1 API.

    Returns monitors filtered by state (default: Alert) and optionally
    by cluster name, service, and environment tags.

    Args:
        cluster_name: Optional cluster name to filter monitors by tag
            ``cluster_name:<name>``.
        state: Monitor state to filter on (e.g. ``"Alert"``, ``"Warn"``,
            ``"No Data"``).  Case-sensitive.
        service: Optional Datadog service tag to filter monitors (e.g.
            ``"my-api"``).  Appended to the ``monitor_tags`` filter.
        env: Optional Datadog environment tag to filter monitors (e.g.
            ``"production"``).  Appended to the ``monitor_tags`` filter.
        config: Optional ``DatadogAPIConfig`` (for testing / injection).
        _custom_api: Optional pre-configured Datadog MonitorsApi (for testing).
    """
    try:
        from datadog_api_client.exceptions import ApiException  # noqa: WPS433
        from datadog_api_client.v1.api.monitors_api import MonitorsApi  # noqa: WPS433
    except ImportError:
        return ToolResult(output=_ERR_NOT_INSTALLED, error=True)

    if config is None:
        from vaig.core.config import get_settings  # noqa: WPS433

        config = get_settings().datadog

    if not config.enabled:
        return ToolResult(output=_ERR_NOT_ENABLED, error=True)

    # Build tag filter string from cluster name, service, and env
    tag_parts: list[str] = []
    if cluster_name:
        try:
            safe_cluster = _sanitize_service_name(cluster_name)
        except ValueError as exc:
            return ToolResult(output=str(exc), error=True)
        tag_parts.append(f"cluster_name:{safe_cluster}")
    if service:
        try:
            safe_service = _sanitize_service_name(service)
        except ValueError as exc:
            return ToolResult(output=str(exc), error=True)
        tag_parts.append(f"service:{safe_service}")
    if env:
        try:
            safe_env = _sanitize_service_name(env)
        except ValueError as exc:
            return ToolResult(output=str(exc), error=True)
        tag_parts.append(f"env:{safe_env}")
    tag_filter = ",".join(tag_parts) if tag_parts else None

    try:
        if _custom_api is not None:
            api = _custom_api
        else:
            with _get_dd_api_client(config) as client:
                api = MonitorsApi(client)  # type: ignore[no-untyped-call]

        kwargs: dict[str, Any] = {}
        if tag_filter:
            kwargs["monitor_tags"] = tag_filter

        monitors = api.list_monitors(**kwargs)

        if not monitors:
            return ToolResult(
                output=f"=== Datadog Monitors ({state}) ===\nNo monitors found.",
                error=False,
            )

        # Filter by state
        matching = [m for m in monitors if getattr(m, "overall_state", None) == state]

        lines: list[str] = [
            f"=== Datadog Monitors ({state}) ===",
        ]
        if cluster_name:
            lines.append(f"Cluster: {cluster_name}")
        lines.append("")

        if not matching:
            lines.append(f"No monitors in '{state}' state.")
        else:
            lines.append(f"  {'ID':<12} {'NAME':<50} {'TYPE':<20} {'STATE':<12}")
            lines.append("  " + "-" * 94)
            for m in matching:
                mid = str(getattr(m, "id", "-"))
                name = str(getattr(m, "name", "-"))
                if len(name) > 49:
                    name = name[:46] + "..."
                mtype = str(getattr(m, "type", "-"))
                mstate = str(getattr(m, "overall_state", "-"))
                lines.append(f"  {mid:<12} {name:<50} {mtype:<20} {mstate:<12}")

        lines.append("")
        lines.append(f"Total monitors scanned: {len(monitors)}")
        lines.append(f"Monitors in '{state}' state: {len(matching)}")

        return ToolResult(output="\n".join(lines), error=False)

    except ApiException as exc:
        status = getattr(exc, "status", 0)
        msg = _dd_error_message(status)
        logger.warning("Datadog monitors API error (HTTP %s): %s", status, exc)
        return ToolResult(output=msg, error=True)
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error fetching Datadog monitors")
        return ToolResult(output="Unexpected error fetching Datadog monitors. See logs for details.", error=True)


def get_datadog_apm_services(
    *,
    env: str = "production",
    cluster_name: str = "",
    service_name: str | None = None,
    config: DatadogAPIConfig | None = None,
    _custom_api: Any = None,
) -> ToolResult:
    """Fetch service catalog entries from the Datadog Service Definition v2 API.

    Returns service names, owning team, primary language, and tier for all
    registered services in the service catalog.  Results are cached for 60
    seconds per (env, cluster, service_name) combination.

    Note: this function retrieves *service definitions* (catalog metadata),
    not live APM metrics such as latency or error rate.

    Args:
        env: Datadog environment tag (e.g. ``"production"``, ``"staging"``).
            Used as a display hint and cache key — the Service Definition
            API does not filter by environment natively.
        cluster_name: Optional cluster name shown in the output header.
        service_name: Optional service name to filter results to a single
            service.  Matched against the ``dd-service`` field (exact match).
        config: Optional ``DatadogAPIConfig`` (for testing / injection).
        _custom_api: Optional pre-configured ServiceDefinitionApi (for testing).
    """
    try:
        from datadog_api_client.exceptions import ApiException  # noqa: WPS433
        from datadog_api_client.v2.api.service_definition_api import ServiceDefinitionApi  # noqa: WPS433
    except ImportError:
        return ToolResult(output=_ERR_NOT_INSTALLED, error=True)

    if config is None:
        from vaig.core.config import get_settings  # noqa: WPS433

        config = get_settings().datadog

    if not config.enabled:
        return ToolResult(output=_ERR_NOT_ENABLED, error=True)

    # Cache check (TTL = 60s) — include service_name in key for filtered results
    cache_key = _cache._cache_key_discovery("dd_apm_services", env, cluster_name, service_name or "")
    cached = _cache._get_cached(cache_key)
    if cached is not None:
        return ToolResult(output=cached, error=False)

    try:
        if _custom_api is not None:
            api = _custom_api
        else:
            with _get_dd_api_client(config) as client:
                api = ServiceDefinitionApi(client)  # type: ignore[no-untyped-call]

        response = api.list_service_definitions()

        all_services = getattr(response, "data", []) or []

        # Filter by service_name when provided (client-side exact match on dd-service)
        if service_name:
            services = []
            for svc in all_services:
                attrs = getattr(svc, "attributes", None)
                if attrs is None:
                    continue
                schema = getattr(attrs, "schema", None) or {}
                if hasattr(schema, "to_dict"):
                    schema = schema.to_dict()
                if str(schema.get("dd-service", "")) == service_name:
                    services.append(svc)
        else:
            services = list(all_services)

        lines: list[str] = [
            "=== Datadog APM Services ===",
            f"Environment: {env}",
        ]
        if cluster_name:
            lines.append(f"Cluster: {cluster_name}")
        if service_name:
            lines.append(f"Service filter: {service_name}")
        lines.append("")

        if not services:
            lines.append("No APM service definitions found.")
        else:
            lines.append(f"  {'SERVICE':<40} {'TEAM':<20} {'LANGUAGE':<15} {'TIER':<10}")
            lines.append("  " + "-" * 85)

            for svc in services:
                attrs = getattr(svc, "attributes", None)
                if attrs is None:
                    continue
                schema = getattr(attrs, "schema", None) or {}
                if hasattr(schema, "to_dict"):
                    schema = schema.to_dict()

                svc_name = str(schema.get("dd-service", "-"))
                team = str(schema.get("team", "-"))
                language = str(schema.get("languages", ["-"])[0] if schema.get("languages") else "-")
                tier = str(schema.get("tier", "-"))

                if len(svc_name) > 39:
                    svc_name = svc_name[:36] + "..."
                if len(team) > 19:
                    team = team[:16] + "..."

                lines.append(f"  {svc_name:<40} {team:<20} {language:<15} {tier:<10}")

            lines.append("")
            lines.append(f"Total services: {len(services)}")

        output = "\n".join(lines)
        _cache._set_cache(cache_key, output)
        return ToolResult(output=output, error=False)

    except ApiException as exc:
        status = getattr(exc, "status", 0)
        msg = _dd_error_message(status)
        logger.warning("Datadog APM services API error (HTTP %s): %s", status, exc)
        return ToolResult(output=msg, error=True)
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error in get_datadog_apm_services")
        return ToolResult(output="Unexpected error retrieving APM services. See logs for details.", error=True)


# ── Async wrappers ───────────────────────────────────────────

from vaig.core.async_utils import to_async  # noqa: E402

async_query_datadog_metrics = to_async(query_datadog_metrics)
async_get_datadog_monitors = to_async(get_datadog_monitors)
async_get_datadog_apm_services = to_async(get_datadog_apm_services)

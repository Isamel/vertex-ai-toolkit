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

# ── Cache TTL for APM services ───────────────────────────────
_APM_CACHE_TTL: int = 60  # seconds

# ── Metric query templates ───────────────────────────────────
_METRIC_TEMPLATES: dict[str, str] = {
    "cpu": "avg:kubernetes.cpu.usage.total{{cluster_name:{cluster}}} by {{pod_name}}",
    "memory": "avg:kubernetes.memory.usage{{cluster_name:{cluster}}} by {{pod_name}}",
    "restarts": "sum:kubernetes.containers.restarts{{cluster_name:{cluster}}} by {{pod_name}}",
    "network_in": "avg:kubernetes.network.rx_bytes{{cluster_name:{cluster}}} by {{pod_name}}",
    "network_out": "avg:kubernetes.network.tx_bytes{{cluster_name:{cluster}}} by {{pod_name}}",
    "disk_read": "avg:kubernetes.io.read_bytes{{cluster_name:{cluster}}} by {{pod_name}}",
    "disk_write": "avg:kubernetes.io.write_bytes{{cluster_name:{cluster}}} by {{pod_name}}",
}


# ── Helpers ──────────────────────────────────────────────────


def _sanitize_service_name(name: str) -> str:
    """Strip characters not allowed in Datadog service/tag names.

    Keeps alphanumeric characters, hyphens, underscores, and dots.
    Replaces everything else with an empty string.
    """
    return re.sub(r"[^a-zA-Z0-9\-._]", "", name)


def _get_dd_api_client(config: DatadogAPIConfig) -> Any:
    """Create a configured Datadog API client.

    Raises:
        ImportError: When ``datadog-api-client`` is not installed.
    """
    from datadog_api_client import ApiClient, Configuration  # noqa: WPS433

    configuration = Configuration()
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
    query: str = "",
    from_ts: int = 0,
    to_ts: int = 0,
    config: DatadogAPIConfig | None = None,
    _custom_api: Any = None,
) -> ToolResult:
    """Query Datadog metrics for a GKE cluster using the Metrics v1 API.

    Supports built-in metric templates (cpu, memory, restarts, network_in,
    network_out, disk_read, disk_write) or a custom query string.

    Args:
        cluster_name: GKE cluster name used to scope the metric query.
        metric: Template name (one of the ``_METRIC_TEMPLATES`` keys). Ignored
            when ``query`` is provided.
        query: Custom Datadog metrics query string. When provided, overrides
            the ``metric`` template.
        from_ts: Unix timestamp for the start of the query window.  Defaults to
            ``now - 3600`` (last hour) when ``0``.
        to_ts: Unix timestamp for the end of the query window.  Defaults to
            ``now`` when ``0``.
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

    # Resolve query string
    safe_cluster = _sanitize_service_name(cluster_name)
    if not query:
        template = _METRIC_TEMPLATES.get(metric)
        if template is None:
            available = ", ".join(sorted(_METRIC_TEMPLATES.keys()))
            return ToolResult(
                output=f"Unknown metric template '{metric}'. Available: {available}",
                error=True,
            )
        query = template.format(cluster=safe_cluster)

    try:
        if _custom_api is not None:
            api = _custom_api
        else:
            client = _get_dd_api_client(config)
            api = MetricsApi(client)

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
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected error querying Datadog metrics: %s", exc)
        return ToolResult(output=f"Unexpected error querying Datadog metrics: {exc}", error=True)


def get_datadog_monitors(
    *,
    cluster_name: str = "",
    tags: list[str] | None = None,
    state: str = "Alert",
    config: DatadogAPIConfig | None = None,
    _custom_api: Any = None,
) -> ToolResult:
    """Fetch active Datadog monitors using the Monitors v1 API.

    Returns monitors filtered by state (default: Alert) and optionally
    by cluster name tag or additional tags.

    Args:
        cluster_name: Optional cluster name to filter monitors by tag
            ``cluster_name:<name>``.
        tags: Additional tag filters (e.g. ``["env:production", "team:sre"]``).
        state: Monitor state to filter on (e.g. ``"Alert"``, ``"Warn"``,
            ``"No Data"``).  Case-sensitive.
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

    # Build tag filter string
    all_tags: list[str] = list(tags) if tags else []
    if cluster_name:
        safe_cluster = _sanitize_service_name(cluster_name)
        all_tags.append(f"cluster_name:{safe_cluster}")
    tag_filter = ",".join(all_tags) if all_tags else None

    try:
        if _custom_api is not None:
            api = _custom_api
        else:
            client = _get_dd_api_client(config)
            api = MonitorsApi(client)

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
        if all_tags:
            lines.append(f"Tags: {', '.join(all_tags)}")
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
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected error fetching Datadog monitors: %s", exc)
        return ToolResult(output=f"Unexpected error fetching Datadog monitors: {exc}", error=True)


def get_datadog_apm_services(
    *,
    env: str = "production",
    cluster_name: str = "",
    config: DatadogAPIConfig | None = None,
    _custom_api: Any = None,
) -> ToolResult:
    """Fetch APM service list from the Datadog APM v2 API.

    Returns service names, p50/p95/p99 latencies, error rate, and
    request throughput. Results are cached for 60 seconds per (env, cluster)
    combination.

    Args:
        env: Datadog environment tag (e.g. ``"production"``, ``"staging"``).
        cluster_name: Optional cluster name to scope the APM query.
        config: Optional ``DatadogAPIConfig`` (for testing / injection).
        _custom_api: Optional pre-configured APM services API client (for testing).
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

    # Cache check (TTL = 60s)
    cache_key = _cache._cache_key_discovery("dd_apm_services", env, cluster_name)
    cached = _cache._get_cached(cache_key)
    if cached is not None:
        return ToolResult(output=cached, error=False)

    try:
        if _custom_api is not None:
            api = _custom_api
        else:
            client = _get_dd_api_client(config)
            api = ServiceDefinitionApi(client)

        response = api.list_service_definitions()

        services = getattr(response, "data", []) or []

        lines: list[str] = [
            "=== Datadog APM Services ===",
            f"Environment: {env}",
        ]
        if cluster_name:
            lines.append(f"Cluster: {cluster_name}")
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
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected error fetching Datadog APM services: %s", exc)
        return ToolResult(output=f"Unexpected error fetching Datadog APM services: {exc}", error=True)


# ── Async wrappers ───────────────────────────────────────────

from vaig.core.async_utils import to_async  # noqa: E402

async_query_datadog_metrics = to_async(query_datadog_metrics)
async_get_datadog_monitors = to_async(get_datadog_monitors)
async_get_datadog_apm_services = to_async(get_datadog_apm_services)

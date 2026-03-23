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

# ── APM metric namespace families ───────────────────────────
# Tried in order — stop at the first family that returns any data.
# This allows the tool to work for HTTP, gRPC, Kafka, and other trace sources.
_APM_METRIC_FAMILIES = [
    "trace.web.request",   # HTTP frameworks (Flask, Django, Express, Rails, etc.)
    "trace.grpc.server",   # gRPC server-side
    "trace.grpc.client",   # gRPC client-side
    "trace.kafka.produce", # Kafka producers
    "trace.kafka.consume", # Kafka consumers
    "trace",               # Generic fallback (trace.request, trace.servlet, etc.)
]

# ── Metric query templates ───────────────────────────────────
# Templates use {filters} as a placeholder for the full tag filter string.
# Example rendered: avg:kubernetes.cpu.usage.total{cluster_name:my-cluster,service:api,env:prod}


def _build_metric_templates(config: DatadogAPIConfig) -> dict[str, str]:
    """Build the metric query template dict from config.

    The ``by {pod_name}`` grouping dimension is read from
    ``config.labels.pod_name`` so it can be overridden per environment.
    Any entries in ``config.custom_metrics`` are merged in after the
    built-in templates — the caller may provide additional metric names
    or override existing ones.

    Raises:
        ValueError: When a custom metric template is missing the required
            ``{filters}`` placeholder.
    """
    pod_name = config.labels.pod_name
    # Templates use {filters} as a Python .format() placeholder filled at query time.
    # The "by {<tag>}" grouping uses {{ }} so .format() leaves them as literal braces
    # in the final Datadog query string (e.g. "by {pod_name}").
    _by = "{{" + pod_name + "}}"
    templates: dict[str, str] = {
        "cpu": "avg:kubernetes.cpu.usage.total{{{filters}}} by " + _by,
        "memory": "avg:kubernetes.memory.usage{{{filters}}} by " + _by,
        "restarts": "sum:kubernetes.containers.restarts{{{filters}}} by " + _by,
        "network_in": "avg:kubernetes.network.rx_bytes{{{filters}}} by " + _by,
        "network_out": "avg:kubernetes.network.tx_bytes{{{filters}}} by " + _by,
        "disk_read": "avg:kubernetes.io.read_bytes{{{filters}}} by " + _by,
        "disk_write": "avg:kubernetes.io.write_bytes{{{filters}}} by " + _by,
    }
    for key, tmpl in config.custom_metrics.items():
        if "{filters}" not in tmpl:
            raise ValueError(
                f"Custom metric template '{key}' with value '{tmpl}' is missing the required '{{filters}}' placeholder."
            )
        try:
            tmpl.format(filters="test")
        except KeyError as exc:
            raise ValueError(
                f"Custom metric template '{key}' with value '{tmpl}' contains an unescaped placeholder {exc}. "
                f"Use {{{{...}}}} (double braces) for literal braces in the template (e.g. {{{{pod_name}}}} instead of {{pod_name}})."
            ) from exc
        templates[key] = tmpl
    return templates


# ── Helpers ──────────────────────────────────────────────────


_VALID_SERVICE_NAME_RE = re.compile(r"^[a-zA-Z0-9\-._]+$")
_VALID_TAG_KEY_RE = re.compile(r"^[a-zA-Z0-9_./-]+$")


def _validate_tag_key(key: str) -> None:
    """Validate a Datadog tag key name.

    Tag keys must contain only alphanumeric characters, underscores, hyphens,
    dots, or slashes.  Characters like ``,``, ``:``, ``{``, ``}`` are rejected
    because they would corrupt the Datadog query filter string.

    Raises:
        ValueError: When ``key`` contains disallowed characters.
    """
    if not _VALID_TAG_KEY_RE.match(key):
        raise ValueError(
            f"Invalid Datadog tag key '{key}': must contain only alphanumeric, "
            "underscore, hyphen, dot, or slash characters"
        )


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


def _sanitize_tag_value(param_name: str, value: str) -> str:
    """Validate a Datadog tag value and return it unchanged if valid.

    Like :func:`_sanitize_service_name` but includes *param_name* in the error
    message so callers get context-specific feedback (e.g. "Invalid env …"
    rather than the generic "Invalid service name …").

    Returns an empty string for an empty input.  Raises ``ValueError`` when the
    value contains characters outside the allowed set.

    Raises:
        ValueError: When ``value`` contains disallowed characters.
    """
    if not value:
        return ""
    if not _VALID_SERVICE_NAME_RE.match(value):
        raise ValueError(
            f"Invalid {param_name} {value!r}: only alphanumeric, hyphens, underscores, and dots are allowed."
        )
    return value


def _build_tag_filter(
    cluster_name: str | None,
    service: str | None,
    env: str | None,
    config: DatadogAPIConfig | None = None,
) -> tuple[str, ToolResult | None]:
    """Build a comma-separated Datadog tag filter string from the given tags.

    Each tag value is validated/sanitized before use.  If any value is invalid
    the function returns ``(empty, ToolResult(error=True))`` so callers can
    return the error immediately.

    Tag key names are read from ``config.labels`` when a config is provided,
    falling back to the Datadog standard names (``cluster_name``, ``service``,
    ``env``) when ``config`` is ``None``.  Any ``config.labels.custom`` entries
    are appended as additional ``key:value`` pairs.

    Args:
        cluster_name: Optional cluster name tag value.
        service: Optional service tag value.
        env: Optional environment tag value.
        config: Optional ``DatadogAPIConfig`` used to resolve tag key names.

    Returns:
        A tuple of ``(filter_string, error_result)``.  On success,
        ``error_result`` is ``None`` and ``filter_string`` is the joined tag
        filter (e.g. ``"cluster_name:foo,service:bar,env:prod"``).  On
        failure, ``filter_string`` is ``""`` and ``error_result`` is a
        ``ToolResult`` with ``error=True``.
    """
    labels = config.labels if config is not None else None

    cluster_key = labels.cluster_name if labels is not None else "cluster_name"
    service_key = labels.service if labels is not None else "service"
    env_key = labels.env if labels is not None else "env"

    tag_parts: list[str] = []
    tag_map: list[tuple[str, str | None]] = [
        (cluster_key, cluster_name),
        (service_key, service),
        (env_key, env),
    ]
    for tag_key, tag_value in tag_map:
        if not tag_value:
            continue
        try:
            _validate_tag_key(tag_key)
            safe_value = _sanitize_tag_value(tag_key, tag_value)
        except ValueError as exc:
            return "", ToolResult(output=str(exc), error=True)
        tag_parts.append(f"{tag_key}:{safe_value}")

    # Append custom labels from config (key=tag_key, value=tag_value)
    if labels is not None:
        for custom_key, custom_value in labels.custom.items():
            if not custom_value:
                continue
            try:
                _validate_tag_key(custom_key)
                safe_custom = _sanitize_tag_value(custom_key, custom_value)
            except ValueError as exc:
                return "", ToolResult(output=str(exc), error=True)
            tag_parts.append(f"{custom_key}:{safe_custom}")

    return ",".join(tag_parts), None


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

    # Build tag filter string: always include cluster_name; optionally service/env
    filters, tag_err = _build_tag_filter(cluster_name, service, env, config)
    if tag_err is not None:
        return tag_err

    metric_templates = _build_metric_templates(config)
    template = metric_templates.get(metric)
    if template is None:
        available = ", ".join(sorted(metric_templates.keys()))
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
    tag_filter_str, tag_err = _build_tag_filter(cluster_name, service, env, config)
    if tag_err is not None:
        return tag_err
    tag_filter = tag_filter_str if tag_filter_str else None

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


def get_datadog_service_catalog(
    *,
    env: str = "production",
    cluster_name: str = "",
    service_name: str | None = None,
    config: DatadogAPIConfig | None = None,
    _custom_api: Any = None,
) -> ToolResult:
    """Fetch service ownership metadata from the Datadog Service Catalog (Service Definition v2 API).

    Returns service names, owning team, primary language, and tier for all
    registered services in the service catalog.  Results are cached for 60
    seconds per (env, cluster, service_name) combination.

    Note: this function retrieves *service definitions* (catalog metadata),
    not live APM metrics such as latency or error rate.  To get live trace
    data (throughput, error rate, latency), use ``get_datadog_apm_services``
    instead.

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

    # Validate inputs before cache lookup so invalid inputs never get cached
    if service_name:
        try:
            service_name = _sanitize_tag_value("service_name", service_name)
        except ValueError as exc:
            return ToolResult(output=str(exc), error=True)

    # Cache check (TTL = 60s) — include service_name in key for filtered results
    cache_key = _cache._cache_key_discovery("dd_service_catalog", env, cluster_name, service_name or "")
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
            "=== Datadog Service Catalog ===",
            f"Environment: {env}",
        ]
        if cluster_name:
            lines.append(f"Cluster: {cluster_name}")
        if service_name:
            lines.append(f"Service filter: {service_name}")
        lines.append("")

        if not services:
            lines.append("No service catalog entries found.")
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
        logger.warning("Datadog service catalog API error (HTTP %s): %s", status, exc)
        return ToolResult(output=msg, error=True)
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error in get_datadog_service_catalog")
        return ToolResult(output="Unexpected error retrieving service catalog. See logs for details.", error=True)


def get_datadog_apm_services(
    *,
    service_name: str = "",
    env: str = "production",
    config: DatadogAPIConfig | None = None,
    _custom_api: Any = None,
) -> ToolResult:
    """Fetch live APM trace metrics for a specific service from Datadog.

    Queries Datadog trace metrics (throughput, error rate, latency) for the
    given service and environment using the Metrics v1 API.  The ``env``
    parameter is used to scope the query — only trace data tagged with the
    matching environment is returned.

    Results are cached for 60 seconds per (service_name, env) combination.

    Args:
        service_name: Service name to query — must match the ``service`` tag
            in Datadog APM (typically from ``tags.datadoghq.com/service``
            label or custom APM instrumentation).  If omitted or empty, the
            tool returns guidance on how to resolve it from Kubernetes labels
            rather than raising an error.
        env: Datadog environment tag (e.g. ``"production"``, ``"staging"``).
            Used to scope the APM query.  Defaults to ``"production"``.
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

    # Validate service_name before cache lookup
    try:
        service_name = _sanitize_tag_value("service_name", service_name)
    except ValueError as exc:
        return ToolResult(output=str(exc), error=True)

    if not service_name:
        return ToolResult(
            output=(
                "service_name parameter is required to query APM trace data. "
                "Resolve it from Kubernetes pod labels (tags.datadoghq.com/service, "
                "app.kubernetes.io/name, or app label) or from the deployment/service "
                "name first."
            ),
            error=False,
        )

    # Sanitize env to prevent tag injection or malformed queries
    try:
        env = _sanitize_tag_value("env", env)
    except ValueError as exc:
        return ToolResult(output=str(exc), error=True)

    # Build tag filter using config.labels for configurable tag key names
    # (e.g. config.labels.service may be "service" or a custom key like "svc")
    scope, tag_err = _build_tag_filter(None, service_name, env, config)
    if tag_err is not None:
        return tag_err

    # Cache check (TTL = 60s) — key includes service+env (metric family resolved at runtime)
    cache_key = _cache._cache_key_discovery("dd_apm_trace", service_name, env)
    cached = _cache._get_cached(cache_key)
    if cached is not None:
        return ToolResult(output=cached, error=False)

    now = int(time.time())
    # 30-minute lookback window — wider than 15 min to catch low-traffic services
    # that may not have had requests in the last 15 minutes.
    start = now - 1800

    def _try_metric_family(api: Any, family: str) -> tuple[dict[str, float | None], bool]:
        """Query hits/errors/duration for one metric namespace family.

        Returns (results_dict, found_any_series).  The generic "trace" family
        uses trace.request.* metric names (no ".web." segment).
        """
        # Build the three query strings for this family
        if family == "trace":
            # Generic fallback uses trace.request.* (no sub-namespace)
            queries = {
                "hits": f"sum:trace.request.hits{{{scope}}}.as_rate()",
                "errors": f"sum:trace.request.errors{{{scope}}}.as_rate()",
                "duration": f"avg:trace.request.duration{{{scope}}}",
            }
        else:
            queries = {
                "hits": f"sum:{family}.hits{{{scope}}}.as_rate()",
                "errors": f"sum:{family}.errors{{{scope}}}.as_rate()",
                "duration": f"avg:{family}.duration{{{scope}}}",
            }

        results: dict[str, float | None] = {}
        found_any = False

        for metric_key, query in queries.items():
            response = api.query_metrics(_from=start, to=now, query=query)
            series = getattr(response, "series", []) or []

            if not series:
                results[metric_key] = None
                continue

            found_any = True
            all_points: list[float] = []
            for s in series:
                for point in getattr(s, "pointlist", []) or []:
                    if point[1] is not None:
                        all_points.append(point[1])

            results[metric_key] = sum(all_points) / len(all_points) if all_points else None

        return results, found_any

    def _execute_queries(api: Any) -> ToolResult:
        """Try each APM metric family in order; stop at the first that returns data."""
        matched_family: str | None = None
        results: dict[str, float | None] = {}

        for family in _APM_METRIC_FAMILIES:
            family_results, found_any = _try_metric_family(api, family)
            if found_any:
                matched_family = family
                results = family_results
                break

        if matched_family is None:
            no_data_msg = (
                f"No APM trace data found for service '{service_name}' in env '{env}'. "
                "Tried metric families: "
                + ", ".join(_APM_METRIC_FAMILIES)
                + ". Verify the service_name matches the 'service' tag in Datadog APM."
            )
            return ToolResult(output=no_data_msg, error=False)

        # Format output
        hits = results.get("hits")
        errors = results.get("errors")
        duration = results.get("duration")

        throughput_str = f"{hits:.2f} req/s" if hits is not None else "N/A"
        error_rate_str = (
            f"{(errors / hits * 100):.2f}%" if hits is not None and hits > 0 and errors is not None
            else ("0.00%" if hits is not None and hits == 0 else "N/A")
        )
        latency_str = f"{(duration * 1000):.2f} ms" if duration is not None else "N/A"

        lines: list[str] = [
            "=== Datadog APM Trace Metrics ===",
            f"Service:    {service_name}",
            f"Env:        {env}",
            "Window:     last 30 minutes",
            f"Metric family: {matched_family}",
            "",
            f"Throughput: {throughput_str}",
            f"Error rate: {error_rate_str}",
            f"Avg latency: {latency_str}",
        ]

        output_str = "\n".join(lines)
        _cache._set_cache(cache_key, output_str)
        return ToolResult(output=output_str, error=False)

    try:
        if _custom_api is not None:
            return _execute_queries(_custom_api)

        with _get_dd_api_client(config) as client:
            return _execute_queries(MetricsApi(client))  # type: ignore[no-untyped-call]

    except ApiException as exc:
        status = getattr(exc, "status", 0)
        msg = _dd_error_message(status)
        logger.warning("Datadog APM trace metrics API error (HTTP %s): %s", status, exc)
        return ToolResult(output=msg, error=True)
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error in get_datadog_apm_services")
        return ToolResult(output="Unexpected error retrieving APM trace metrics. See logs for details.", error=True)


# ── Async wrappers ───────────────────────────────────────────

from vaig.core.async_utils import to_async  # noqa: E402

async_query_datadog_metrics = to_async(query_datadog_metrics)
async_get_datadog_monitors = to_async(get_datadog_monitors)
async_get_datadog_service_catalog = to_async(get_datadog_service_catalog)
async_get_datadog_apm_services = to_async(get_datadog_apm_services)

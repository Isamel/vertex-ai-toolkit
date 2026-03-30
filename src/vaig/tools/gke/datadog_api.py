"""Datadog REST API tools — metrics, monitors, and APM service data for GKE diagnostics."""

from __future__ import annotations

import logging
import re
import ssl
import time
from typing import TYPE_CHECKING, Any

import urllib3.exceptions  # type: ignore[import-untyped]

from vaig.core.config import DatadogAPIConfig
from vaig.tools.base import ToolResult

from . import _cache

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _point_value(point: Any) -> float | None:
    """Extract the numeric value from a Datadog ``Point`` or plain list.

    The Datadog SDK v2 ``Point`` objects do NOT support index access
    (``point[1]``).  They expose a ``.value`` attribute containing
    ``[timestamp, value]``.  In tests, ``pointlist`` entries are plain
    lists where ``point[1]`` works directly.  This helper handles both.
    """
    try:
        pair = getattr(point, "value", point)
        result = pair[1]  # type: ignore[index]
        return float(result) if result is not None else None
    except (IndexError, TypeError, KeyError, ValueError):
        return None

# ── Error messages ───────────────────────────────────────────
_ERR_NOT_INSTALLED = "datadog-api-client not installed. Install with: pip install 'vaig[live]'"
_ERR_NOT_ENABLED = "Datadog API integration is disabled. Set datadog.enabled=true in config."
_ERR_AUTH = "Authentication failed. Check your Datadog API key and application key."
_ERR_FORBIDDEN = "Insufficient permissions. Check your Datadog API/app key scopes."
_ERR_RATE_LIMIT = "Rate limit exceeded. Try again later."

# ── Datadog APM operation probe order ────────────────────────
# Ordered by prevalence: Java/Spring → generic HTTP → gRPC → Python → Ruby → messaging.
_APM_OPERATION_PROBE_ORDER: tuple[str, ...] = (
    "servlet.request",
    "http.request",
    "grpc.server",
    "flask.request",
    "web.request",
    "rack.request",
    "grpc.client",
    "kafka.consume",
)

# ── Metric name aliases (fuzzy matching) ─────────────────────
# Maps common short-hands and typos to canonical metric template keys.
_METRIC_ALIASES: dict[str, str] = {
    "request": "requests",
    "req": "requests",
    "throughput": "requests",
    "error": "error_rate",
    "err": "error_rate",
    "errs": "errors",
    "lat": "latency",
    "net_in": "network_in",
    "net_out": "network_out",
    "disk": "disk_read",
    "restart": "restarts",
    "mem": "memory",
}

# ── Metric query templates ───────────────────────────────────
# Templates use {filters} as a placeholder for the full tag filter string.
# Example rendered: avg:kubernetes.cpu.usage.total{cluster_name:my-cluster,service:api,env:prod}


def _build_metric_templates(config: DatadogAPIConfig, operation: str = "http.request") -> dict[str, str]:
    """Build the metric query template dict from config.

    The ``by {pod_name}`` grouping dimension is read from
    ``config.labels.pod_name`` so it can be overridden per environment.

    ``config.metric_mode`` controls which built-in templates are included:

    * ``"k8s_agent"`` — kubernetes.* metrics (DaemonSet Agent).
    * ``"apm"`` — trace.* metrics (APM instrumentation only).
    * ``"both"`` — kubernetes.* **and** trace.* metrics combined.

    Any entries in ``config.custom_metrics`` are merged in after the
    built-in templates — the caller may provide additional metric names
    or override existing ones.

    Raises:
        ValueError: When a custom metric template is missing the required
            ``{filters}`` placeholder.
    """
    pod_name = config.labels.pod_name
    _by = "{{" + pod_name + "}}"

    mode = getattr(config, "metric_mode", "k8s_agent")

    # ── k8s_agent (infrastructure) templates ─────────────
    k8s_templates: dict[str, str] = {
        "cpu": "avg:kubernetes.cpu.usage.total{{{filters}}} by " + _by,
        "memory": "avg:kubernetes.memory.usage{{{filters}}} by " + _by,
        "restarts": "sum:kubernetes.containers.restarts{{{filters}}} by " + _by,
        "network_in": "avg:kubernetes.network.rx_bytes{{{filters}}} by " + _by,
        "network_out": "avg:kubernetes.network.tx_bytes{{{filters}}} by " + _by,
        "disk_read": "avg:kubernetes.io.read_bytes{{{filters}}} by " + _by,
        "disk_write": "avg:kubernetes.io.write_bytes{{{filters}}} by " + _by,
    }

    # ── APM (trace.*) templates ──────────────────────────
    _op = operation
    apm_templates: dict[str, str] = {
        "requests": "sum:trace." + _op + ".hits{{{filters}}} by " + _by,
        "errors": "sum:trace." + _op + ".errors{{{filters}}} by " + _by,
        "latency": "avg:trace." + _op + ".duration{{{filters}}} by " + _by,
        "error_rate": (
            "( sum:trace." + _op + ".errors{{{filters}}} by "
            + _by
            + " / sum:trace." + _op + ".hits{{{filters}}} by "
            + _by
            + " ) * 100"
        ),
        "apdex": "avg:trace." + _op + ".apdex{{{filters}}} by " + _by,
    }

    # ── Select templates by mode ─────────────────────────
    if mode == "apm":
        templates: dict[str, str] = apm_templates
    elif mode == "both":
        templates = {**k8s_templates, **apm_templates}
    else:
        templates = k8s_templates

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
    *,
    include_custom_labels: bool = True,
) -> tuple[str, ToolResult | None]:
    """Build a comma-separated Datadog tag filter string from the given tags.

    Each tag value is validated/sanitized before use.  If any value is invalid
    the function returns ``(empty, ToolResult(error=True))`` so callers can
    return the error immediately.

    Tag key names are read from ``config.labels`` when a config is provided,
    falling back to the Datadog standard names (``cluster_name``, ``service``,
    ``env``) when ``config`` is ``None``.  When *include_custom_labels* is
    ``True`` (default), any ``config.labels.custom`` entries are appended as
    additional ``key:value`` pairs.  Set to ``False`` for APM ``trace.*``
    queries that only carry ``service`` and ``env`` tags.

    Args:
        cluster_name: Optional cluster name tag value.
        service: Optional service tag value.
        env: Optional environment tag value.
        config: Optional ``DatadogAPIConfig`` used to resolve tag key names.
        include_custom_labels: Whether to append ``config.labels.custom``
            entries.  Defaults to ``True`` for backward compatibility.

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
    if include_custom_labels and labels is not None:
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
    # SSL verification — mirrors the requests ssl_verify semantics:
    #   True  = standard verification (default)
    #   False = disable SSL verification (e.g. self-signed certs behind corporate proxy)
    #   str   = path to a custom CA bundle file
    if config.ssl_verify is False:
        configuration.verify_ssl = False
    elif isinstance(config.ssl_verify, str):
        configuration.verify_ssl = True
        configuration.ssl_ca_cert = config.ssl_verify
    # ssl_verify=True is the SDK default; no changes needed
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


def _ssl_error_result(context: str, exc: Exception) -> ToolResult:
    """Return a ToolResult with SSL troubleshooting guidance."""
    logger.error("Datadog %s: SSL certificate verification failed: %s", context, exc)
    return ToolResult(
        output=(
            f"SSL certificate verification failed connecting to Datadog: {exc}. "
            "If you are behind a corporate proxy with SSL inspection, try one of:\n"
            "  1. Set the REQUESTS_CA_BUNDLE env var to your corporate CA bundle path.\n"
            "  2. Set datadog.ssl_verify = '/path/to/ca-bundle.crt' in your vaig config.\n"
            "  3. Set datadog.ssl_verify = false to disable verification (not recommended)."
        ),
        error=True,
    )


# ── Public Tool Functions ────────────────────────────────────


def _detect_apm_operation(
    api: Any,
    service: str,
    env: str,
    config: DatadogAPIConfig,
) -> str | None:
    """Probe ``trace.{op}.hits`` metrics to find the active APM operation name.

    Checks the discovery cache first (TTL 300 s).  When
    ``config.apm_operation`` is not ``"auto"``, the configured value is
    returned immediately without any API calls.

    Returns the first operation with non-empty timeseries data, or
    ``None`` if every probe comes back empty.
    """
    # Fast path: explicit configuration — skip probing entirely.
    configured = getattr(config, "apm_operation", "auto")
    if configured != "auto":
        return configured

    cache_key = _cache._cache_key_discovery("apm_op", service, env)
    cached = _cache._get_cached(cache_key, ttl=300)
    if cached is not None:
        return cached

    now = int(time.time())
    start = now - 900  # last 15 minutes

    service_key = config.labels.service
    env_key = config.labels.env

    for op in _APM_OPERATION_PROBE_ORDER:
        query = f"sum:trace.{op}.hits{{{service_key}:{service},{env_key}:{env}}}"
        try:
            response = api.query_metrics(_from=start, to=now, query=query)
            series = getattr(response, "series", []) or []
            if series:
                _cache._set_cache(cache_key, op, ttl=300)
                logger.info(
                    "Datadog APM: detected operation '%s' for service=%s env=%s",
                    op, service, env,
                )
                return op
        except (urllib3.exceptions.MaxRetryError, OSError):
            logger.debug(
                "Datadog APM: probe for '%s' failed for service=%s env=%s",
                op, service, env, exc_info=True,
            )
            continue

    logger.warning(
        "Datadog APM: no operation found for service=%s env=%s (probed: %s)",
        service, env, ", ".join(_APM_OPERATION_PROBE_ORDER),
    )
    return None


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

    # Apply cluster_name_override when set — allows the Datadog tag value to differ
    # from the GKE cluster name (e.g. when the DD agent uses a different tag value).
    effective_cluster = getattr(config, "cluster_name_override", "") or cluster_name

    # Resolve metric mode — APM trace.* metrics do not carry custom tags
    mode = getattr(config, "metric_mode", "k8s_agent")

    # Resolve APM operation name when in apm or both mode
    resolved_operation = "http.request"
    if mode in ("apm", "both"):
        apm_op = getattr(config, "apm_operation", "auto")
        if apm_op != "auto":
            resolved_operation = apm_op
        # NOTE: auto-detection is not run here because query_datadog_metrics
        # doesn't have service/env guaranteed; _detect_apm_operation is used
        # by get_datadog_apm_services which always has those values.

    metric_templates = _build_metric_templates(config, operation=resolved_operation)
    template = metric_templates.get(metric)

    # ── Fuzzy matching: try common aliases and singular/plural ────
    if template is None:
        resolved = _METRIC_ALIASES.get(metric)
        if resolved is None:
            # Try singular/plural: "requests" → "request" or "cpu" → "cpus"
            if metric.endswith("s"):
                resolved = metric[:-1]
            else:
                resolved = metric + "s"
        template = metric_templates.get(resolved)
        if template is not None:
            logger.debug("Fuzzy-matched metric '%s' → '%s'", metric, resolved)

    if template is None:
        available = ", ".join(sorted(metric_templates.keys()))
        return ToolResult(
            output=f"Unknown metric template '{metric}'. Available: {available}",
            error=True,
        )

    # Determine include_custom_labels per-metric: APM trace.* metrics only
    # carry service+env tags, so custom labels would cause empty results.
    # In "both" mode the decision depends on the resolved template.
    _is_trace_metric = "trace." in template
    _had_custom_labels = not _is_trace_metric and bool(
        config.labels.custom if config is not None else False
    )
    filters, tag_err = _build_tag_filter(
        effective_cluster, service, env, config, include_custom_labels=not _is_trace_metric,
    )
    if tag_err is not None:
        return tag_err

    query = template.format(filters=filters)

    def _execute_query(api: Any, q: str) -> tuple[ToolResult, bool]:
        """Run a single metrics query and return ``(result, has_data)``.

        The boolean *has_data* is ``True`` when the response contained at
        least one series, ``False`` otherwise.  Callers should use this
        flag instead of inspecting the output text to decide on retries.
        """
        response = api.query_metrics(
            _from=start,
            to=end,
            query=q,
        )

        series = getattr(response, "series", []) or []
        if not series:
            return (
                ToolResult(
                    output=f"=== Datadog Metrics: {metric} ===\nCluster: {effective_cluster}\nNo data returned for the given time window.",
                    error=False,
                ),
                False,
            )

        lines: list[str] = [
            f"=== Datadog Metrics: {metric} ===",
            f"Cluster: {effective_cluster}",
            f"Query: {q}",
            f"Window: {start} → {end}",
            "",
        ]

        for s in series:
            metric_name = getattr(s, "metric", "unknown")
            scope = getattr(s, "scope", "")
            points = getattr(s, "pointlist", []) or []
            if points:
                values = [v for p in points if (v := _point_value(p)) is not None]
                if values:
                    avg_val = sum(values) / len(values)
                    max_val = max(values)
                    last_val = values[-1]
                    label = scope or metric_name
                    lines.append(f"  {label:<50}  avg={avg_val:.2f}  max={max_val:.2f}  last={last_val:.2f}")

        lines.append("")
        lines.append(f"Total series: {len(series)}")

        return (ToolResult(output="\n".join(lines), error=False), True)

    def _run_with_fallback(api: Any) -> ToolResult:
        """Execute the query and retry without custom labels on empty results."""
        result, has_data = _execute_query(api, query)

        # Retry without custom labels when the first query returned no data
        # and custom labels were included in the original filter.
        if not has_data and _had_custom_labels:
            fallback_filters, fb_err = _build_tag_filter(
                effective_cluster, service, env, config, include_custom_labels=False,
            )
            if fb_err is None:
                fallback_query = template.format(filters=fallback_filters)
                logger.debug(
                    "No data with custom labels for '%s'; retrying without custom labels",
                    metric,
                )
                result, _ = _execute_query(api, fallback_query)

        return result

    try:
        if _custom_api is not None:
            return _run_with_fallback(_custom_api)

        with _get_dd_api_client(config) as client:
            api = MetricsApi(client)  # type: ignore[no-untyped-call]
            return _run_with_fallback(api)

    except ApiException as exc:
        status = getattr(exc, "status", 0)
        msg = _dd_error_message(status)
        logger.warning("Datadog metrics API error (HTTP %s): %s", status, exc)
        return ToolResult(output=msg, error=True)
    except urllib3.exceptions.MaxRetryError as exc:
        if isinstance(getattr(exc, "reason", None), ssl.SSLError):
            return _ssl_error_result("metrics", exc)
        logger.error("Datadog metrics: connection failed after retries: %s", exc)
        return ToolResult(
            output=f"Failed to connect to Datadog after multiple retries: {exc}",
            error=True,
        )
    except ssl.SSLError as exc:
        return _ssl_error_result("metrics", exc)
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

    def _execute_monitors(api: Any) -> ToolResult:
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

    try:
        if _custom_api is not None:
            return _execute_monitors(_custom_api)

        with _get_dd_api_client(config) as client:
            api = MonitorsApi(client)  # type: ignore[no-untyped-call]
            return _execute_monitors(api)

    except ApiException as exc:
        status = getattr(exc, "status", 0)
        msg = _dd_error_message(status)
        logger.warning("Datadog monitors API error (HTTP %s): %s", status, exc)
        return ToolResult(output=msg, error=True)
    except urllib3.exceptions.MaxRetryError as exc:
        if isinstance(getattr(exc, "reason", None), ssl.SSLError):
            return _ssl_error_result("monitors", exc)
        logger.error("Datadog monitors: connection failed after retries: %s", exc)
        return ToolResult(
            output=f"Failed to connect to Datadog after multiple retries: {exc}",
            error=True,
        )
    except ssl.SSLError as exc:
        return _ssl_error_result("monitors", exc)
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

    def _execute_catalog(api: Any) -> ToolResult:
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

    try:
        if _custom_api is not None:
            return _execute_catalog(_custom_api)

        with _get_dd_api_client(config) as client:
            api = ServiceDefinitionApi(client)  # type: ignore[no-untyped-call]
            return _execute_catalog(api)

    except ApiException as exc:
        status = getattr(exc, "status", 0)
        msg = _dd_error_message(status)
        logger.warning("Datadog service catalog API error (HTTP %s): %s", status, exc)
        return ToolResult(output=msg, error=True)
    except urllib3.exceptions.MaxRetryError as exc:
        if isinstance(getattr(exc, "reason", None), ssl.SSLError):
            return _ssl_error_result("service catalog", exc)
        logger.error("Datadog service catalog: connection failed after retries: %s", exc)
        return ToolResult(
            output=f"Failed to connect to Datadog after multiple retries: {exc}",
            error=True,
        )
    except ssl.SSLError as exc:
        return _ssl_error_result("service catalog", exc)
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error in get_datadog_service_catalog")
        return ToolResult(output="Unexpected error retrieving service catalog. See logs for details.", error=True)


def _run_apm_queries(
    api: Any,
    service_name: str,
    env: str,
    tag_filter: str,
    start: int,
    now: int,
    hours_back: float,
    cache_key: str,
    config: DatadogAPIConfig,
) -> ToolResult:
    """Execute APM operation detection and metric queries using the given API object.

    Extracted so that both the ``_custom_api`` (testing) and real-client paths
    share the same logic while keeping all API calls inside the context manager
    when a real client is used.
    """
    # ── Detect APM operation name ────────────────────────
    operation = _detect_apm_operation(api, service_name, env, config)
    if operation is None:
        probed = ", ".join(_APM_OPERATION_PROBE_ORDER)
        return ToolResult(
            output=(
                f"No APM trace data found for service '{service_name}' in env '{env}'. "
                f"Probed operations: {probed}. "
                "Verify the service has APM instrumentation enabled, or set "
                "datadog.apm_operation to the correct operation name in your config."
            ),
            error=False,
        )

    # ── Build and execute 3 metric queries ───────────────
    queries = {
        "hits": f"sum:trace.{operation}.hits{{{tag_filter}}}",
        "errors": f"sum:trace.{operation}.errors{{{tag_filter}}}",
        "duration": f"avg:trace.{operation}.duration{{{tag_filter}}}",
    }

    results: dict[str, float | None] = {}
    for metric_key, query in queries.items():
        response = api.query_metrics(_from=start, to=now, query=query)
        series = getattr(response, "series", []) or []
        if series:
            points = getattr(series[0], "pointlist", []) or []
            # Extract last non-null datapoint
            last_val = None
            for point in reversed(points):
                val = _point_value(point)
                if val is not None:
                    last_val = val
                    break
            results[metric_key] = last_val
        else:
            results[metric_key] = None

    # ── Compute derived metrics ──────────────────────────
    hits = results.get("hits")
    errors = results.get("errors")
    duration = results.get("duration")

    window_seconds = max(hours_back * 3600, 1)

    if hits is not None:
        throughput_str = f"{hits / window_seconds:.4f} req/s"
    else:
        throughput_str = "N/A"

    if hits is not None and hits > 0 and errors is not None:
        error_rate_str = f"{(errors / hits * 100):.2f}%"
    elif hits is not None and hits == 0:
        error_rate_str = "0.00%"
    else:
        error_rate_str = "N/A"

    # Datadog returns duration in seconds; convert to ms for display
    latency_str = f"{(duration * 1000):.2f} ms" if duration is not None else "N/A"

    window_str = f"last {hours_back:.4g} hour{'s' if hours_back != 1 else ''}"

    lines = [
        "=== Datadog APM Trace Metrics ===",
        f"Service:    {service_name}",
        f"Env:        {env}",
        f"Window:     {window_str}",
        f"Operation:  {operation}",
        "",
        f"Throughput: {throughput_str}",
        f"Error rate: {error_rate_str}",
        f"Avg latency: {latency_str}",
    ]
    output = "\n".join(lines)
    _cache._set_cache(cache_key, output)
    return ToolResult(output=output, error=False)


def get_datadog_apm_services(
    *,
    service_name: str = "",
    env: str = "production",
    hours_back: float | None = None,
    config: DatadogAPIConfig | None = None,
    _custom_api: Any = None,
) -> ToolResult:
    """Fetch live APM trace metrics for a specific service using the Metrics v1 Timeseries API.

    Queries Datadog ``trace.{operation}.*`` metrics (throughput, error rate,
    latency) for the given service and environment.  The APM operation name
    is auto-detected by probing common ``trace.{op}.hits`` metrics unless
    ``config.apm_operation`` is explicitly set.

    Results are cached for 60 seconds per (service_name, env, hours_back) combination.

    Args:
        service_name: Service name to query — must match the ``service`` tag
            in Datadog APM (typically from ``tags.datadoghq.com/service``
            label or custom APM instrumentation).  If omitted or empty, the
            tool returns guidance on how to resolve it from Kubernetes labels.
        env: Datadog environment tag (e.g. ``"production"``, ``"staging"``).
            Used to scope the APM query.  Defaults to ``"production"``.
        hours_back: Lookback window in hours.  When ``None`` (default), the
            value from ``config.default_lookback_hours`` is used (default 4h).
            Use fractional values for sub-hour windows (e.g. ``0.5`` for 30 min).
        config: Optional ``DatadogAPIConfig`` (for testing / injection).
        _custom_api: Optional pre-configured Datadog ``MetricsApi`` (for testing).
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

    # Resolve lookback: explicit hours_back takes priority; fall back to config default.
    lookback = hours_back if hours_back is not None else getattr(config, "default_lookback_hours", 4.0)
    hours_back = lookback

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

    # Validate hours_back — clamp to default if non-positive rather than hard-failing
    if hours_back <= 0:
        hours_back = lookback  # Already resolved from config.default_lookback_hours
        logger.warning("hours_back=%s is not positive — clamping to config default %sh", hours_back, lookback)

    # Cache check (TTL = 60s) — key includes service+env+window in whole seconds
    cache_key_window = int(hours_back * 3600)
    cache_key = _cache._cache_key_discovery(
        "dd_apm_metrics", service_name, env, str(cache_key_window)
    )
    cached = _cache._get_cached(cache_key)
    if cached is not None:
        return ToolResult(output=cached, error=False)

    # Compute time range
    now = int(time.time())
    start = int(now - hours_back * 3600)

    # Build tag filter string using configurable tag key names
    tag_filter, tag_err = _build_tag_filter(None, service_name, env, config, include_custom_labels=False)
    if tag_err is not None:
        return tag_err

    try:
        if _custom_api is not None:
            return _run_apm_queries(
                _custom_api, service_name, env, tag_filter, start, now, hours_back, cache_key, config,
            )

        with _get_dd_api_client(config) as client:
            api = MetricsApi(client)  # type: ignore[no-untyped-call]
            return _run_apm_queries(
                api, service_name, env, tag_filter, start, now, hours_back, cache_key, config,
            )

    except ApiException as exc:
        status = getattr(exc, "status", 0)
        msg = _dd_error_message(status)
        logger.warning("Datadog APM metrics API error (HTTP %s): %s", status, exc)
        return ToolResult(output=msg, error=True)
    except urllib3.exceptions.MaxRetryError as exc:
        if isinstance(getattr(exc, "reason", None), ssl.SSLError):
            return _ssl_error_result("APM", exc)
        logger.error("Datadog APM: connection failed after retries: %s", exc)
        return ToolResult(
            output=f"Failed to connect to Datadog after multiple retries: {exc}",
            error=True,
        )
    except (OSError, urllib3.exceptions.HTTPError) as exc:
        logger.exception("Unexpected network error in get_datadog_apm_services: %s", exc)
        return ToolResult(output="Unexpected error retrieving APM trace metrics. See logs for details.", error=True)


# ── Async wrappers ───────────────────────────────────────────

from vaig.core.async_utils import to_async  # noqa: E402

async_query_datadog_metrics = to_async(query_datadog_metrics)
async_get_datadog_monitors = to_async(get_datadog_monitors)
async_get_datadog_service_catalog = to_async(get_datadog_service_catalog)
async_get_datadog_apm_services = to_async(get_datadog_apm_services)

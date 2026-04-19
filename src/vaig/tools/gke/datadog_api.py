"""Datadog REST API tools — metrics, monitors, and APM service data for GKE diagnostics."""

from __future__ import annotations

import logging
import re
import ssl
import time
import warnings
from typing import TYPE_CHECKING, Any

import urllib3.exceptions  # type: ignore[import-untyped]

from vaig.core.config import DatadogAPIConfig
from vaig.tools.base import ToolResult

from . import _cache

if TYPE_CHECKING:
    from vaig.core.config import GKEConfig

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
# Ordered by prevalence: Istio/Envoy (newest then legacy) → Java/Spring →
# gRPC → Ruby → Node/Express → Python → .NET → generic HTTP (catch-all).
# envoy.proxy MUST remain at position #1 (primary istio-ingressgateway pattern).
_APM_OPERATION_PROBE_ORDER: tuple[str, ...] = (
    "envoy.proxy",       # Istio/Envoy sidecar + ingressgateway (keep #1)
    "envoy.envoy",       # Older Envoy span naming (stale-cache scenario)
    "servlet.request",   # Java/Spring/Tomcat
    "grpc.server",       # gRPC server-side
    "grpc.client",       # gRPC client-side
    "rack.request",      # Ruby on Rails
    "express.request",   # Node/Express
    "django.request",    # Python/Django
    "flask.request",     # Python/Flask
    "fastapi.request",   # Python/FastAPI
    "aspnet.request",    # .NET
    "http.request",      # generic HTTP (last-resort catch-all)
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

    * ``"auto"`` — starts with kubernetes.* templates; at query time,
      ``query_datadog_metrics`` falls back to trace.* APM templates when
      the k8s queries return empty results.
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

    mode = getattr(config, "metric_mode", "auto")

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
    # "auto" starts with k8s_agent templates; query_datadog_metrics handles
    # the fallback to apm when all k8s queries return empty results.
    if mode == "apm":
        templates: dict[str, str] = apm_templates
    elif mode == "both":
        templates = {**k8s_templates, **apm_templates}
    else:
        # k8s_agent and auto both use k8s_templates as initial set
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
        # Suppress InsecureRequestWarning only when SSL verification is
        # deliberately disabled — avoids noisy warnings for expected configs
        # (e.g. self-signed certs behind corporate proxy).
        warnings.filterwarnings(
            "ignore", category=urllib3.exceptions.InsecureRequestWarning,
        )
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


# ── APM Discovery Helpers ────────────────────────────────────

_APM_OP_FROM_METRIC_RE: re.Pattern[str] = re.compile(
    r"^trace\.([^.]+(?:\.[^.]+)?)\.hits$"
)


def _dd_raw_get(
    client_ctx: Any,
    config: DatadogAPIConfig,
    path: str,
    params: dict[str, str] | None = None,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Raw GET or POST against the Datadog API.

    Mirrors the inline auth+GET pattern used in ``diagnose_datadog_metrics``
    (rest_client, ``DD-API-KEY`` / ``DD-APPLICATION-KEY`` headers, JSON decode).

    When *method* is ``"POST"``, *payload* is serialised to JSON and sent as
    the request body.  The ``Content-Type: application/json`` header is added
    automatically.

    Returns an empty dict on any failure (non-200, exception, non-dict body).
    Never raises.
    """
    import json as _json  # noqa: WPS433
    import urllib.parse  # noqa: WPS433

    try:
        auth_headers: dict[str, str] = {
            "DD-API-KEY": config.api_key,
            "DD-APPLICATION-KEY": config.app_key,
        }
        rest = getattr(client_ctx, "rest_client", client_ctx)
        qs = ""
        if params:
            qs = "?" + urllib.parse.urlencode(params)
        url = f"https://api.{config.site}{path}{qs}"
        if method.upper() == "POST":
            auth_headers["Content-Type"] = "application/json"
            body_bytes = _json.dumps(payload or {}).encode("utf-8")
            resp = rest.request("POST", url, headers=auth_headers, body=body_bytes, timeout=5)
        else:
            resp = rest.request("GET", url, headers=auth_headers, timeout=5)
        resp_status = getattr(resp, "status", 200)
        if resp_status and resp_status >= 400:  # noqa: PLR2004
            logger.debug("_dd_raw_get: HTTP %s for %s", resp_status, path)
            return {}
        if hasattr(resp, "data"):
            raw_data = resp.data
            data = _json.loads(
                raw_data.decode("utf-8") if isinstance(raw_data, bytes) else raw_data
            )
        else:
            data = {}
        return data if isinstance(data, dict) else {}
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001
        logger.debug("_dd_raw_get: request to %s failed: %s", path, exc)
        return {}


def _discover_apm_operation(
    api: Any,
    service: str,
    env: str,
    config: DatadogAPIConfig,
) -> str | None:
    """Query ``/api/v1/search`` for ``trace.*.hits`` metrics emitted by *service*.

    Parses the returned metric names with :data:`_APM_OP_FROM_METRIC_RE` and
    returns the first matched operation name.  Returns ``None`` on empty
    results, no regex match, or any API failure.

    Result is **not** cached here — callers (``_detect_apm_operation``) own
    the cache lifecycle.
    """
    client_ctx = api.api_client
    q = f"metrics:trace.*.hits service:{service} env:{env}"
    data = _dd_raw_get(client_ctx, config, "/api/v1/search", {"q": q})
    metrics: list[str] = data.get("results", {}).get("metrics", [])
    for m in metrics:
        match = _APM_OP_FROM_METRIC_RE.match(m)
        if match:
            return match.group(1)
    return None


# ── Public Tool Functions ────────────────────────────────────


def _detect_apm_operation(
    api: Any,
    service: str,
    env: str,
    config: DatadogAPIConfig,
) -> tuple[str | None, str]:
    """Resolve the active APM operation name using a 5-step fallback chain.

    1. **Override dict**: if ``config.apm_operation_overrides`` has the
       sanitized service name, return that value immediately.
    2. **Explicit config**: if ``config.apm_operation != "auto"``, return it.
    3. **Cache hit**: if ``apm_op:{service}:{env}`` is cached (TTL 300 s),
       return the cached value.
    4. **Discovery** (opt-in): if ``config.apm_discovery_enabled`` is True,
       call ``_discover_apm_operation``; on success cache and return.
    5. **Probe order**: iterate ``_APM_OPERATION_PROBE_ORDER``; return first
       operation with non-empty timeseries; cache the winner.

    Returns a ``(operation, source)`` tuple where *source* is one of
    ``"override"``, ``"config"``, ``"cache"``, ``"discovery"``, ``"probe"``,
    or ``"none"`` (when every step fails and ``None`` is returned).
    """
    # Step 1: per-service override dict (highest precedence, no API calls)
    sanitized = _sanitize_tag_value("service_name", service)
    override = config.apm_operation_overrides.get(sanitized)
    if override:
        return override, "override"

    # Step 2: explicit config.apm_operation
    configured = getattr(config, "apm_operation", "auto")
    if configured != "auto":
        return configured, "config"

    # Step 3: cache hit
    cache_key = _cache._cache_key_discovery("apm_op", service, env)
    cached = _cache._get_cached(cache_key, ttl=300)
    if cached is not None:
        return cached, "cache"

    # Step 4: discovery (feature-flagged, default off)
    if getattr(config, "apm_discovery_enabled", False):
        discovered = _discover_apm_operation(api, service, env, config)
        if discovered:
            _cache._set_cache(cache_key, discovered, ttl=300)
            logger.info(
                "Datadog APM: discovered operation '%s' for service=%s env=%s",
                discovered, service, env,
            )
            return discovered, "discovery"

    # Step 5: probe order
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
                return op, "probe"
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
    return None, "none"


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
    mode = getattr(config, "metric_mode", "auto")

    # Resolve APM operation name when in apm or both mode.
    # When apm_operation == "auto" the actual operation is probed lazily:
    #   - apm/both mode: probed here (before building the primary query) when
    #     service+env are available.  Without probing, Istio/Envoy services
    #     silently fall back to "http.request" and return no data.
    #   - auto mode: probed inside the k8s→apm fallback (api client is built
    #     there) — see _run_with_fallback below.
    resolved_operation = "http.request"
    _apm_needs_detection = False
    if mode in ("apm", "both"):
        apm_op = getattr(config, "apm_operation", "auto")
        if apm_op != "auto":
            resolved_operation = apm_op
        elif service and env:
            _apm_needs_detection = True  # probed inside _run_with_fallback

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

    # Determine include_custom_labels per-metric.  Built-in templates
    # (kubernetes.* and APM trace.*) only carry cluster/service/env tags —
    # applying user-defined custom labels to them causes over-filtering and
    # empty results.  Only user-defined entries from ``config.custom_metrics``
    # should receive custom labels.
    template_str: str = template  # narrow for nonlocal rebinding inside closure
    _is_trace_metric = "trace." in template_str
    _is_custom_metric = metric in (getattr(config, "custom_metrics", {}) or {})
    # Include custom labels when the metric is user-defined OR when it is not
    # a trace.* metric (kubernetes.* built-ins first attempt with custom labels;
    # the retry path strips them on empty results).  Only trace.* metrics are
    # hard-blocked because their tag schema never includes user-defined labels.
    _include_custom_labels = _is_custom_metric or not _is_trace_metric
    _had_custom_labels = _include_custom_labels and bool(
        config.labels.custom if config is not None else False
    )
    filters, tag_err = _build_tag_filter(
        effective_cluster, service, env, config, include_custom_labels=_include_custom_labels,
    )
    if tag_err is not None:
        return tag_err

    query = template_str.format(filters=filters)

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
            diag_lines = [
                f"=== Datadog Metrics: {metric} ===",
                f"Cluster: {effective_cluster}",
                "No data returned for the given time window.",
                "",
                "Diagnostic context:",
                f"  metric_mode: {mode}",
                f"  query: {q}",
                f"  tag_filter: {filters}",
                f"  time_window: {start} → {end} ({(end - start) // 60} min)",
                f"  metrics_queried: {', '.join(sorted(metric_templates.keys()))}",
                "",
                "Possible causes:",
                "  - Cluster or service tags may not match Datadog agent config",
                "  - Time window may predate metric collection",
                "  - If using k8s_agent mode, the Datadog DaemonSet Agent may not be installed",
                "  - If APM-only setup: set metric_mode='apm' or 'auto' in config",
                "",
                "Next step:",
                "  Run diagnose_datadog_metrics(config=config) to inspect available "
                "tags and validate label config.",
            ]
            return (
                ToolResult(output="\n".join(diag_lines), error=False),
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
        """Execute the query and retry without custom labels on empty results.

        When ``mode == "auto"``, tries the k8s_agent template first.  If that
        returns no data, rebuilds the query using APM templates and retries.
        """
        nonlocal template_str, query, resolved_operation, metric_templates

        # ── APM operation auto-detection (apm/both mode) ──────
        # Probe the APM backend for the actual operation name — required for
        # non-HTTP workloads like Istio/Envoy which use "envoy.proxy" rather
        # than "http.request".  Skip for non-trace metrics (e.g. kubernetes.*)
        # to avoid wasted API calls.
        if _apm_needs_detection and _is_trace_metric and service and env:
            try:
                detected, _det_source = _detect_apm_operation(api, service, env, config)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "APM operation auto-detection failed for service=%s env=%s: %s",
                    service, env, exc,
                )
                detected = None
            if detected and detected != resolved_operation:
                logger.debug(
                    "APM operation auto-detected: '%s' for service=%s env=%s",
                    detected, service, env,
                )
                resolved_operation = detected
                metric_templates = _build_metric_templates(config, operation=resolved_operation)
                new_template = metric_templates.get(metric)
                if new_template is not None:
                    template_str = new_template
                    query = template_str.format(filters=filters)

        result, has_data = _execute_query(api, query)

        # Retry without custom labels when the first query returned no data
        # and custom labels were included in the original filter.
        if not has_data and _had_custom_labels:
            fallback_filters, fb_err = _build_tag_filter(
                effective_cluster, service, env, config, include_custom_labels=False,
            )
            if fb_err is None:
                fallback_query = template_str.format(filters=fallback_filters)
                logger.info(
                    "No data with custom labels for '%s'; retrying without custom labels",
                    metric,
                )
                result, has_data = _execute_query(api, fallback_query)

        # ── auto-mode fallback: k8s_agent → apm ─────────────
        if not has_data and mode == "auto":
            # Re-use _build_metric_templates with apm mode to avoid duplication.
            # Copy config so we don't mutate the caller's object.
            _fb_config = config.model_copy(update={"metric_mode": "apm"})
            apm_only = _build_metric_templates(_fb_config, operation=resolved_operation)
            apm_template = apm_only.get(metric)
            if apm_template is not None:
                # Auto-detect APM operation before executing the fallback —
                # the default "http.request" misses Istio/Envoy and other
                # non-HTTP workloads.  Only probed when an apm template
                # exists for this metric (avoids wasted calls).
                if service and env:
                    try:
                        detected, _det_source = _detect_apm_operation(api, service, env, config)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "auto mode: APM operation detection failed: %s", exc,
                        )
                        detected = None
                    if detected and detected != resolved_operation:
                        logger.debug(
                            "auto mode: APM operation auto-detected: '%s' for service=%s env=%s",
                            detected, service, env,
                        )
                        resolved_operation = detected
                        apm_only = _build_metric_templates(_fb_config, operation=resolved_operation)
                        apm_template = apm_only.get(metric) or apm_template

                # APM trace.* metrics don't carry custom labels — use minimal filter
                apm_filters, apm_err = _build_tag_filter(
                    effective_cluster, service, env, config, include_custom_labels=False,
                )
                if apm_err is None:
                    apm_query = apm_template.format(filters=apm_filters)
                    logger.info(
                        "auto mode: k8s_agent returned no data for '%s'; "
                        "falling back to apm (operation=%s)",
                        metric, resolved_operation,
                    )
                    result, has_data = _execute_query(api, apm_query)
                    if has_data:
                        logger.info("auto mode: apm fallback succeeded for '%s'", metric)

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
    *,
    _retried: bool = False,
) -> ToolResult:
    """Execute APM operation detection and metric queries using the given API object.

    Extracted so that both the ``_custom_api`` (testing) and real-client paths
    share the same logic while keeping all API calls inside the context manager
    when a real client is used.

    ``_retried`` is an internal guard: when ``True`` and all metric queries
    return empty, the results are returned as-is without another cache
    invalidation + retry (prevents infinite loops for services with no traffic).
    """
    # ── Detect APM operation name ────────────────────────
    operation, op_source = _detect_apm_operation(api, service_name, env, config)
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

    # ── Cache invalidation + single-shot retry ────────────
    # If all 3 metric queries returned empty AND this is not already a retry,
    # the cached operation may be stale.  Invalidate it and re-run once.
    all_empty = (
        results.get("hits") is None
        and results.get("errors") is None
        and results.get("duration") is None
    )
    # Only invalidate when the op came from cache (not override, config, discovery, or probe).
    if all_empty and op_source == "cache" and not _retried:
        apm_op_cache_key = _cache._cache_key_discovery("apm_op", service_name, env)
        _cache._DISCOVERY_CACHE.pop(apm_op_cache_key, None)
        logger.debug(
            "Datadog APM: all metrics empty for op=%s service=%s env=%s — "
            "invalidated cache, retrying operation detection",
            operation, service_name, env,
        )
        return _run_apm_queries(
            api,
            service_name,
            env,
            tag_filter,
            start,
            now,
            hours_back,
            cache_key,
            config,
            _retried=True,
        )

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


# ── Service Dependencies ─────────────────────────────────────


def get_datadog_service_dependencies(
    *,
    service_name: str,
    config: DatadogAPIConfig | None = None,
    _custom_api: Any = None,
) -> ToolResult:
    """Fetch upstream and downstream service dependencies from the Datadog Service Dependencies v1 API.

    Calls ``GET /api/v1/service_dependencies/{service}`` which returns
    ``calls`` (downstream services this service depends on) and
    ``called_by`` (upstream services that depend on this service).

    The response is formatted as a human-readable text summary **and**
    includes a structured ``STRUCTURED_DEPENDENCY_EDGES`` JSON block
    containing :class:`DependencyEdge` records for programmatic
    consumption by the dependency graph renderer.

    Args:
        service_name: The Datadog service name to look up dependencies for.
            Required — must match the ``service`` tag in Datadog APM.
        config: Optional ``DatadogAPIConfig`` (for testing / injection).
        _custom_api: Optional pre-configured API callable for testing.
            When provided, should be a callable accepting ``(service_name)``
            and returning a dict with ``calls`` and ``called_by`` keys.
    """
    import json as _json  # noqa: WPS433

    try:
        from datadog_api_client.exceptions import ApiException  # noqa: WPS433
    except ImportError:
        return ToolResult(output=_ERR_NOT_INSTALLED, error=True)

    if config is None:
        from vaig.core.config import get_settings  # noqa: WPS433

        config = get_settings().datadog

    if not config.enabled:
        return ToolResult(output=_ERR_NOT_ENABLED, error=True)

    # Validate service_name (required for this endpoint)
    if not service_name:
        return ToolResult(
            output="service_name is required for get_datadog_service_dependencies.",
            error=True,
        )
    try:
        service_name = _sanitize_tag_value("service_name", service_name)
    except ValueError as exc:
        return ToolResult(output=str(exc), error=True)

    # Cache check (TTL = 60s)
    cache_key = _cache._cache_key_discovery("dd_service_deps", service_name)
    cached = _cache._get_cached(cache_key)
    if cached is not None:
        return ToolResult(output=cached, error=False)

    def _execute_deps(api: Any) -> ToolResult:
        from vaig.skills.service_health.schema import DependencyEdge  # noqa: WPS433

        # The SDK class for v1 service_dependencies may not exist (public beta).
        # _custom_api is a callable for tests; the raw-HTTP fallback also
        # passes a lambda (with _custom_api=None), so dispatch on the
        # object's actual type rather than gating on _custom_api.
        if callable(api):
            response = api(service_name)
        elif hasattr(api, "get_service_dependencies"):
            # Use the API instance — GET /api/v1/service_dependencies/{service}
            response = api.get_service_dependencies(service_name)
        else:
            raise TypeError(f"Unsupported Datadog service dependencies API object: {type(api)!r}")

        # Normalise response to dict
        if hasattr(response, "to_dict"):
            data = response.to_dict()
        elif isinstance(response, dict):
            data = response
        else:
            data = {}

        calls: list[str] = data.get("calls", []) or []
        called_by: list[str] = data.get("called_by", []) or []

        # Build human-readable output
        lines: list[str] = [
            f"=== Datadog Service Dependencies: {service_name} ===",
            "",
        ]

        lines.append(f"Downstream (calls): {len(calls)}")
        if calls:
            for svc in sorted(calls):
                lines.append(f"  → {svc}")
        else:
            lines.append("  (none)")

        lines.append("")
        lines.append(f"Upstream (called_by): {len(called_by)}")
        if called_by:
            for svc in sorted(called_by):
                lines.append(f"  ← {svc}")
        else:
            lines.append("  (none)")

        # Build structured DependencyEdge data
        edges: list[dict[str, Any]] = []
        for downstream in calls:
            edge = DependencyEdge(
                source=service_name,
                target=downstream,
                evidence=f"datadog:service_dependencies:{service_name}",
                confidence=0.9,
                method="datadog",
            )
            edges.append(edge.model_dump())

        for upstream in called_by:
            edge = DependencyEdge(
                source=upstream,
                target=service_name,
                evidence=f"datadog:service_dependencies:{service_name}",
                confidence=0.9,
                method="datadog",
            )
            edges.append(edge.model_dump())

        lines.append("")
        lines.append("--- STRUCTURED_DEPENDENCY_EDGES ---")
        lines.append(_json.dumps(edges, indent=2))
        lines.append("--- END_STRUCTURED_DEPENDENCY_EDGES ---")

        output = "\n".join(lines)
        _cache._set_cache(cache_key, output)
        return ToolResult(output=output, error=False)

    try:
        if _custom_api is not None:
            return _execute_deps(_custom_api)

        with _get_dd_api_client(config) as client:
            # The SDK class may or may not exist — try lazy import first
            try:
                from datadog_api_client.v1.api.service_dependencies_api import (  # noqa: WPS433
                    ServiceDependenciesApi,
                )
                api = ServiceDependenciesApi(client)  # type: ignore[no-untyped-call]
            except ImportError:
                # SDK class doesn't exist yet (v1 public beta) — fall back to
                # raw HTTP via the client's rest_client.
                logger.info(
                    "ServiceDependenciesApi SDK class not available; using raw HTTP fallback."
                )
                rest = getattr(client, "rest_client", client)
                url = f"https://api.{config.site}/api/v1/service_dependencies/{service_name}"
                resp = rest.request("GET", url)

                # Check response status before parsing (API may return 4xx/5xx).
                resp_status = getattr(resp, "status", 200)
                if resp_status and resp_status >= 400:  # noqa: PLR2004
                    msg = _dd_error_message(resp_status)
                    logger.warning(
                        "Datadog service dependencies raw HTTP error (HTTP %s)",
                        resp_status,
                    )
                    return ToolResult(output=msg, error=True)

                import json as _json_inner  # noqa: WPS433

                response_data: dict[str, Any] = _json_inner.loads(resp.data) if hasattr(resp, "data") else {}
                return _execute_deps(lambda _sn: response_data)

            return _execute_deps(api)

    except ApiException as exc:
        status = getattr(exc, "status", 0)
        msg = _dd_error_message(status)
        logger.warning("Datadog service dependencies API error (HTTP %s): %s", status, exc)
        return ToolResult(output=msg, error=True)
    except urllib3.exceptions.MaxRetryError as exc:
        if isinstance(getattr(exc, "reason", None), ssl.SSLError):
            return _ssl_error_result("service dependencies", exc)
        logger.error("Datadog service dependencies: connection failed after retries: %s", exc)
        return ToolResult(
            output=f"Failed to connect to Datadog after multiple retries: {exc}",
            error=True,
        )
    except ssl.SSLError as exc:
        return _ssl_error_result("service dependencies", exc)
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error in get_datadog_service_dependencies")
        return ToolResult(output="Unexpected error retrieving service dependencies. See logs for details.", error=True)


def diagnose_datadog_metrics(
    *,
    config: DatadogAPIConfig | None = None,
    _custom_client: Any = None,
) -> ToolResult:
    """Probe Datadog to discover available metrics and tag keys.

    Calls the Datadog search API to discover ``kubernetes.*`` and ``trace.*``
    metrics, then calls the hosts API to discover available tag keys.  Returns
    a structured diagnostic report that helps the LLM agent understand the
    metric landscape and suggest the correct ``metric_mode`` configuration.

    Args:
        config: Optional ``DatadogAPIConfig`` (for testing / injection).
        _custom_client: Optional pre-configured Datadog ApiClient (for testing).
    """
    import json as _json  # noqa: WPS433

    if config is None:
        from vaig.core.config import get_settings  # noqa: WPS433

        config = get_settings().datadog

    if not config.enabled:
        return ToolResult(output=_ERR_NOT_ENABLED, error=True)

    diagnostic: dict[str, Any] = {
        "current_config": {
            "metric_mode": getattr(config, "metric_mode", "auto"),
            "site": config.site,
            "apm_operation": getattr(config, "apm_operation", "auto"),
            "cluster_name_override": getattr(config, "cluster_name_override", ""),
            "labels": {
                "cluster_name": config.labels.cluster_name,
                "pod_name": config.labels.pod_name,
            },
        },
        "kubernetes_metrics": [],
        "trace_metrics": [],
        "tag_keys": [],
        "suggestions": [],
        "errors": [],
    }

    # Build auth headers from config so raw HTTP requests include credentials.
    _auth_headers: dict[str, str] = {
        "DD-API-KEY": config.api_key,
        "DD-APPLICATION-KEY": config.app_key,
    }

    def _search_metrics(client_ctx: Any, query: str) -> list[str]:
        """Search Datadog for metrics matching a query prefix."""
        try:
            import urllib.parse  # noqa: WPS433

            rest = getattr(client_ctx, "rest_client", client_ctx)
            encoded_q = urllib.parse.quote_plus(f"metrics:{query}")
            url = f"https://api.{config.site}/api/v1/search?q={encoded_q}"
            resp = rest.request("GET", url, headers=_auth_headers)
            resp_status = getattr(resp, "status", 200)
            if resp_status and resp_status >= 400:  # noqa: PLR2004
                diagnostic["errors"].append(f"Metric search '{query}' HTTP {resp_status}")
                return []
            if hasattr(resp, "data"):
                raw_data = resp.data
                data = _json.loads(raw_data.decode("utf-8") if isinstance(raw_data, bytes) else raw_data)
            else:
                data = {}
            results = data.get("results", {})
            return results.get("metrics", [])  # type: ignore[no-any-return]
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001
            diagnostic["errors"].append(f"Metric search '{query}' failed: {exc}")
            return []

    def _get_tag_keys(client_ctx: Any) -> list[str]:
        """Discover available host tag keys from Datadog."""
        try:
            rest = getattr(client_ctx, "rest_client", client_ctx)
            url = f"https://api.{config.site}/api/v1/tags/hosts"
            resp = rest.request("GET", url, headers=_auth_headers)
            resp_status = getattr(resp, "status", 200)
            if resp_status and resp_status >= 400:  # noqa: PLR2004
                diagnostic["errors"].append(f"Tag discovery HTTP {resp_status}")
                return []
            if hasattr(resp, "data"):
                raw_data = resp.data
                data = _json.loads(raw_data.decode("utf-8") if isinstance(raw_data, bytes) else raw_data)
            else:
                data = {}
            tags = data.get("tags", {})
            return sorted(tags.keys()) if isinstance(tags, dict) else []
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001
            diagnostic["errors"].append(f"Tag discovery failed: {exc}")
            return []

    def _run_diagnostics(client_ctx: Any) -> ToolResult:
        k8s_metrics = _search_metrics(client_ctx, "kubernetes.*")
        diagnostic["kubernetes_metrics"] = k8s_metrics[:50]  # cap for output size

        trace_metrics = _search_metrics(client_ctx, "trace.*")
        diagnostic["trace_metrics"] = trace_metrics[:50]

        diagnostic["tag_keys"] = _get_tag_keys(client_ctx)

        # ── Generate suggestions ─────────────────────────
        has_k8s = bool(k8s_metrics)
        has_trace = bool(trace_metrics)

        if has_k8s and has_trace:
            diagnostic["suggestions"].append(
                "Both kubernetes.* and trace.* metrics found. "
                "metric_mode='auto' or 'both' recommended."
            )
        elif has_k8s and not has_trace:
            diagnostic["suggestions"].append(
                "Only kubernetes.* metrics found (no trace.* APM metrics). "
                "metric_mode='k8s_agent' recommended. "
                "APM may not be enabled — check DD_APM_ENABLED on the Datadog agent."
            )
        elif has_trace and not has_k8s:
            diagnostic["suggestions"].append(
                "Only trace.* APM metrics found (no kubernetes.* infra metrics). "
                "metric_mode='apm' recommended. "
                "The Datadog DaemonSet Agent may not be deployed or may lack cluster-level checks."
            )
        else:
            diagnostic["suggestions"].append(
                "No kubernetes.* or trace.* metrics found. "
                "Verify Datadog agent is deployed and sending metrics. "
                "Check API key scopes and site configuration."
            )

        # Check if configured tag keys exist in Datadog
        tag_keys = diagnostic["tag_keys"]
        configured_cluster_tag = config.labels.cluster_name
        if tag_keys and configured_cluster_tag not in tag_keys:
            diagnostic["suggestions"].append(
                f"Configured cluster_name tag '{configured_cluster_tag}' not found in Datadog host tags. "
                f"Available tag keys: {', '.join(tag_keys[:20])}. "
                "This may cause empty query results."
            )

        lines = [
            "=== Datadog Metrics Diagnostic ===",
            "",
            f"kubernetes.* metrics found: {len(k8s_metrics)}",
            f"trace.* metrics found: {len(trace_metrics)}",
            f"Host tag keys discovered: {len(tag_keys)}",
            "",
        ]

        if diagnostic["suggestions"]:
            lines.append("Suggestions:")
            for s in diagnostic["suggestions"]:
                lines.append(f"  • {s}")
            lines.append("")

        if diagnostic["errors"]:
            lines.append("Errors during diagnostic:")
            for e in diagnostic["errors"]:
                lines.append(f"  ⚠ {e}")
            lines.append("")

        lines.append("Full diagnostic (JSON):")
        lines.append(_json.dumps(diagnostic, indent=2, default=str))

        return ToolResult(output="\n".join(lines), error=False)

    try:
        if _custom_client is not None:
            return _run_diagnostics(_custom_client)

        with _get_dd_api_client(config) as client:
            return _run_diagnostics(client)

    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in diagnose_datadog_metrics")
        return ToolResult(
            output=f"Unexpected error running Datadog diagnostics: {exc}",
            error=True,
        )


def query_datadog_error_spans(
    service: str,
    env: str,
    start_time: str,
    end_time: str,
    *,
    limit: int = 50,
    config: DatadogAPIConfig | None = None,
    _custom_client_ctx: Any = None,
) -> ToolResult:
    """Return top error spans for a service, grouped by endpoint / status / exception / upstream.

    Queries the Datadog Spans Search v2 API (``POST /api/v2/spans/events/search``)
    with ``@error:true`` to fetch error spans for the given service and environment,
    then groups them by HTTP route, status code, exception class, and upstream service.

    Args:
        service: Datadog service name (e.g. ``"my-api"``).
        env: Datadog environment tag value (e.g. ``"production"``).
        start_time: ISO-8601 start timestamp for the query window
            (e.g. ``"2024-01-01T00:00:00Z"``).
        end_time: ISO-8601 end timestamp for the query window
            (e.g. ``"2024-01-01T01:00:00Z"``).
        limit: Maximum number of spans to retrieve (capped at 200).
        config: Optional :class:`DatadogAPIConfig` (for testing / injection).
        _custom_client_ctx: Optional pre-configured client context (for testing).

    Returns:
        :class:`ToolResult` with grouped error spans or an error message.
    """
    try:
        from datadog_api_client.exceptions import ApiException  # noqa: WPS433
    except ImportError:
        return ToolResult(output=_ERR_NOT_INSTALLED, error=True)

    if config is None:
        from vaig.core.config import get_settings  # noqa: WPS433

        config = get_settings().datadog

    if not config.enabled:
        return ToolResult(output=_ERR_NOT_ENABLED, error=True)

    try:
        sanitized_service = _sanitize_service_name(service)
        sanitized_env = _sanitize_tag_value("env", env)
    except ValueError as exc:
        return ToolResult(output=str(exc), error=True)

    actual_limit = min(limit, 200)

    body: dict[str, Any] = {
        "data": {
            "attributes": {
                "filter": {
                    "query": f"service:{sanitized_service} env:{sanitized_env} @error:true",
                    "from": start_time,
                    "to": end_time,
                },
                "page": {"limit": actual_limit},
                "sort": "-timestamp",
            },
            "type": "search_request",
        }
    }

    def _run(client_ctx: Any) -> ToolResult:
        response = _dd_raw_get(
            client_ctx,
            config,  # type: ignore[arg-type]
            "/api/v2/spans/events/search",
            method="POST",
            payload=body,
        )

        spans = response.get("data", [])
        if not spans:
            return ToolResult(output="No error spans found in the specified time range.")

        # Group by (route, status_code, exception_class, upstream)
        groups: dict[tuple[str, str, str, str], int] = {}
        for span in spans:
            attrs = span.get("attributes", {}).get("attributes", {})
            route = str(attrs.get("http.route") or attrs.get("http.url") or "unknown")
            status = str(attrs.get("http.status_code", "?"))
            exc_class = str(attrs.get("error.type") or attrs.get("error.kind") or "unknown")
            upstream = str(attrs.get("@upstream_cluster") or attrs.get("upstream_cluster") or "n/a")
            key = (route, status, exc_class, upstream)
            groups[key] = groups.get(key, 0) + 1

        lines = [
            f"Error spans for service={sanitized_service} env={sanitized_env} "
            f"({len(spans)} spans, grouped by endpoint | status | exception | upstream | count):"
        ]
        for (route, status, exc_class, upstream), count in sorted(
            groups.items(), key=lambda x: -x[1]
        ):
            lines.append(f"  {route} | {status} | {exc_class} | {upstream} | {count}")
        return ToolResult(output="\n".join(lines))

    try:
        if _custom_client_ctx is not None:
            return _run(_custom_client_ctx)
        with _get_dd_api_client(config) as client:
            return _run(client)
    except ApiException as exc:
        status = getattr(exc, "status", 0)
        return ToolResult(output=_dd_error_message(status), error=True)
    except ssl.SSLError as exc:
        return _ssl_error_result("query_datadog_error_spans", exc)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in query_datadog_error_spans")
        return ToolResult(
            output=f"Unexpected error querying Datadog error spans: {exc}",
            error=True,
        )


# ── Async wrappers ───────────────────────────────────────────

from vaig.core.async_utils import to_async  # noqa: E402

async_query_datadog_metrics = to_async(query_datadog_metrics)
async_get_datadog_monitors = to_async(get_datadog_monitors)
async_get_datadog_service_catalog = to_async(get_datadog_service_catalog)
async_get_datadog_apm_services = to_async(get_datadog_apm_services)
async_get_datadog_service_dependencies = to_async(get_datadog_service_dependencies)
async_diagnose_datadog_metrics = to_async(diagnose_datadog_metrics)
async_query_datadog_error_spans = to_async(query_datadog_error_spans)


# ── Workload usage for cost pipeline (Layer 3) ───────────────

def get_datadog_workload_usage(
    namespace: str,
    workload_names: list[str],
    *,
    gke_config: GKEConfig,
    config: DatadogAPIConfig | None = None,
) -> dict[str, Any]:
    """Fetch per-workload CPU and memory usage from Datadog for cost estimation.

    This is Layer 3 in the multi-source cost measurement pipeline.  Called only
    when Cloud Monitoring (L1) and Metrics Server (L2) both return no data.

    Queries:
    - ``avg:kubernetes.cpu.usage.total{...} by {pod_name}`` (nanocores/s → vCPU)
    - ``avg:kubernetes.memory.rss{...} by {pod_name}`` (bytes → GiB)

    Args:
        namespace: Kubernetes namespace to filter on.
        workload_names: List of workload names (used to build pod-name prefix matching).
        gke_config: GKE cluster config (provides cluster name for tag filter).
        config: :class:`DatadogAPIConfig`; when ``None`` uses ``DatadogAPIConfig()``
            (disabled by default → returns ``{}`` immediately).

    Returns:
        Dict mapping workload_name → :class:`WorkloadUsageMetrics`.
        Returns ``{}`` when Datadog is disabled, not installed, or on any error.
    """
    from vaig.tools.gke.monitoring import ContainerUsageMetrics, WorkloadUsageMetrics  # noqa: WPS433

    _cfg = config if config is not None else DatadogAPIConfig()
    if not _cfg.enabled:
        return {}

    # Determine cluster tag value
    cluster_tag = _cfg.cluster_name_override or getattr(gke_config, "cluster_name", "")

    try:
        from datadog_api_client.v1.api.metrics_api import MetricsApi  # noqa: WPS433
    except ImportError:  # pragma: no cover
        logger.debug("get_datadog_workload_usage: datadog-api-client not installed")
        return {}

    import time as _time  # noqa: WPS433

    now = int(_time.time())
    start = now - 3600  # 1-hour window

    cpu_query = (
        f"avg:kubernetes.cpu.usage.total"
        f"{{kube_namespace:{namespace},kube_cluster_name:{cluster_tag}}} by {{pod_name}}"
    )
    mem_query = (
        f"avg:kubernetes.memory.rss"
        f"{{kube_namespace:{namespace},kube_cluster_name:{cluster_tag}}} by {{pod_name}}"
    )

    def _avg_series(series: Any) -> dict[str, float]:
        """Return pod_name → average metric value from a Datadog series list."""
        result: dict[str, float] = {}
        for s in series or []:
            scope = getattr(s, "scope", "") or ""
            # scope looks like "pod_name:my-pod-abc12"
            pod_name = ""
            for part in scope.split(","):
                part = part.strip()
                if part.startswith("pod_name:"):
                    pod_name = part[len("pod_name:"):]
                    break
            if not pod_name:
                continue
            points = getattr(s, "pointlist", []) or []
            vals = [v for p in points if (v := _point_value(p)) is not None]
            if vals:
                result[pod_name] = sum(vals) / len(vals)
        return result

    try:
        with _get_dd_api_client(_cfg) as client:
            api = MetricsApi(client)  # type: ignore[no-untyped-call]
            cpu_resp = api.query_metrics(_from=start, to=now, query=cpu_query)
            mem_resp = api.query_metrics(_from=start, to=now, query=mem_query)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001
        logger.debug("get_datadog_workload_usage: query failed: %s", exc)
        return {}

    cpu_by_pod = _avg_series(getattr(cpu_resp, "series", []))
    mem_by_pod = _avg_series(getattr(mem_resp, "series", []))

    if not cpu_by_pod and not mem_by_pod:
        return {}

    # Group pod metrics back to workload names (prefix match)
    # workload_names = ["my-api", "worker"]
    # pod_name = "my-api-abc12-xyz78" → matched to "my-api" (longest prefix)
    def _match_workload(pod_name: str) -> str | None:
        best: str | None = None
        best_len = 0
        for wl in workload_names:
            # Pod names are typically workload_name + "-" + hash
            if pod_name.startswith(wl + "-") or pod_name == wl:
                if len(wl) > best_len:
                    best = wl
                    best_len = len(wl)
        return best

    # workload → {cpu_sum, mem_sum, count}
    wl_cpu: dict[str, list[float]] = {}
    wl_mem: dict[str, list[float]] = {}

    for pod_name, cpu_nanocores in cpu_by_pod.items():
        wl = _match_workload(pod_name)
        if wl:
            wl_cpu.setdefault(wl, []).append(cpu_nanocores / 1e9)

    for pod_name, mem_bytes in mem_by_pod.items():
        wl = _match_workload(pod_name)
        if wl:
            wl_mem.setdefault(wl, []).append(mem_bytes / (1024**3))

    all_wl = set(wl_cpu) | set(wl_mem)
    if not all_wl:
        return {}

    output: dict[str, WorkloadUsageMetrics] = {}
    for wl_name in all_wl:
        cpu_vals = wl_cpu.get(wl_name, [])
        mem_vals = wl_mem.get(wl_name, [])
        avg_cpu = sum(cpu_vals) / len(cpu_vals) if cpu_vals else None
        avg_mem = sum(mem_vals) / len(mem_vals) if mem_vals else None
        c_metrics = {
            "_datadog": ContainerUsageMetrics(
                container_name="_datadog",
                avg_cpu_cores=avg_cpu,
                avg_memory_gib=avg_mem,
            )
        }
        output[wl_name] = WorkloadUsageMetrics(namespace=namespace, workload_name=wl_name, containers=c_metrics)

    return output  # type: ignore[return-value]

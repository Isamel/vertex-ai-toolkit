"""OpsGenie alert correlation tool — list open alerts via OpsGenie v2 API."""

from __future__ import annotations

import logging
from typing import Any

from vaig.core.config import OpsGenieConfig
from vaig.tools.base import ToolResult
from vaig.tools.integrations._cache import _cache_key, _get_cached, _set_cache
from vaig.tools.integrations._http import api_request

logger = logging.getLogger(__name__)

_OG_CACHE_TTL: int = 60  # seconds


def list_opsgenie_alerts(
    *,
    config: OpsGenieConfig,
    status: str = "open",
    priority: str = "",
    limit: int = 25,
) -> ToolResult:
    """Fetch open alerts from OpsGenie v2 API.

    Args:
        config: OpsGenie configuration with api_key and base_url.
        status: Alert status filter — ``"open"``, ``"closed"``, or ``"all"``
            (default: ``"open"``).
        priority: Optional priority filter — ``"P1"`` through ``"P5"``
            (default: all priorities).
        limit: Maximum number of alerts to return (default 25).

    Returns:
        ``ToolResult`` with a formatted Markdown table of alerts or an
        error message on failure.
    """
    effective_limit = min(limit, config.alert_fetch_limit)

    # ── Cache check ──────────────────────────────────────────
    cache_parts = ["og", "alerts", status, priority, str(effective_limit)]
    if config.team_ids:
        cache_parts.append(",".join(sorted(config.team_ids)))
    ck = _cache_key(*cache_parts)
    cached = _get_cached(ck, ttl=_OG_CACHE_TTL)
    if cached is not None:
        return ToolResult(output=cached)

    # ── Build request ────────────────────────────────────────
    url = f"{config.base_url.rstrip('/')}/v2/alerts"
    api_key = config.api_key.get_secret_value()
    headers = {
        "Authorization": f"GenieKey {api_key}",
        "Content-Type": "application/json",
    }

    # OpsGenie uses a query string for filtering
    query_parts: list[str] = []
    if status and status != "all":
        query_parts.append(f"status={status}")
    if priority:
        query_parts.append(f"priority={priority}")
    if config.team_ids:
        # OpsGenie search query supports responders
        for tid in config.team_ids:
            query_parts.append(f'teams="{tid}"')

    params: dict[str, Any] = {
        "limit": str(effective_limit),
        "sort": "createdAt",
        "order": "desc",
    }
    if query_parts:
        params["query"] = " AND ".join(query_parts)

    # ── Execute ──────────────────────────────────────────────
    data, error = api_request(
        "GET", url, headers=headers, params=params, service_name="OpsGenie",
    )
    if error is not None:
        return error

    # ── Parse and format ─────────────────────────────────────
    alerts: list[dict[str, Any]] = (data or {}).get("data", [])

    if not alerts:
        output = "No open OpsGenie alerts found."
        _set_cache(ck, output, ttl=_OG_CACHE_TTL)
        return ToolResult(output=output)

    lines = [
        "| # | Message | Status | Priority | Source | Created |",
        "|---|---------|--------|----------|--------|---------|",
    ]
    for i, alert in enumerate(alerts, 1):
        message = alert.get("message", "N/A")
        alert_status = alert.get("status", "N/A")
        alert_priority = alert.get("priority", "N/A")
        source = alert.get("source", "N/A")
        created = alert.get("createdAt", "N/A")
        lines.append(
            f"| {i} | {message} | {alert_status} | {alert_priority} | {source} | {created} |"
        )

    output = "\n".join(lines)
    _set_cache(ck, output, ttl=_OG_CACHE_TTL)
    return ToolResult(output=output)

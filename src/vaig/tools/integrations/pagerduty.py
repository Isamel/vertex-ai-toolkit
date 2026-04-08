"""PagerDuty alert correlation tool — list active incidents via REST API v2."""

from __future__ import annotations

import logging
from typing import Any

from vaig.core.config import PagerDutyConfig
from vaig.tools.base import ToolResult
from vaig.tools.integrations._cache import _cache_key, _get_cached, _set_cache
from vaig.tools.integrations._http import api_request

logger = logging.getLogger(__name__)

_PD_CACHE_TTL: int = 60  # seconds


def list_pagerduty_incidents(
    *,
    config: PagerDutyConfig,
    status: str = "triggered,acknowledged",
    service_name: str = "",
    limit: int = 25,
) -> ToolResult:
    """Fetch active incidents from PagerDuty REST API v2.

    Args:
        config: PagerDuty configuration with api_token and base_url.
        status: Comma-separated incident statuses to filter (default:
            ``"triggered,acknowledged"``).
        service_name: Optional free-text filter — only return incidents whose
            service name contains this substring (case-insensitive).
        limit: Maximum number of incidents to return (default 25).

    Returns:
        ``ToolResult`` with a formatted Markdown table of incidents or an
        error message on failure.
    """
    effective_limit = min(limit, config.alert_fetch_limit)

    # ── Cache check ──────────────────────────────────────────
    cache_parts = ["pd", "incidents", config.base_url, status, service_name, str(effective_limit)]
    if config.alert_service_ids:
        cache_parts.append(",".join(sorted(config.alert_service_ids)))
    ck = _cache_key(*cache_parts)
    cached = _get_cached(ck, ttl=_PD_CACHE_TTL)
    if cached is not None:
        return ToolResult(output=cached)

    # ── Build request ────────────────────────────────────────
    url = f"{config.base_url.rstrip('/')}/incidents"
    headers = {
        "Authorization": f"Token token={config.api_token}",
        "Content-Type": "application/json",
        "Accept": "application/vnd.pagerduty+json;version=2",
    }
    params: dict[str, Any] = {
        "statuses[]": [s.strip() for s in status.split(",")],
        "limit": str(effective_limit),
        "sort_by": "created_at:desc",
    }
    if config.alert_service_ids:
        params["service_ids[]"] = config.alert_service_ids

    # ── Execute ──────────────────────────────────────────────
    data, error = api_request(
        "GET", url, headers=headers, params=params, service_name="PagerDuty",
    )
    if error is not None:
        return error

    # ── Parse and format ─────────────────────────────────────
    incidents: list[dict[str, Any]] = (data or {}).get("incidents", [])

    # Apply optional service_name substring filter
    if service_name:
        sn_lower = service_name.lower()
        incidents = [
            inc for inc in incidents
            if sn_lower in (inc.get("service", {}).get("summary", "")).lower()
        ]

    if not incidents:
        output = "No active PagerDuty incidents found."
        _set_cache(ck, output, ttl=_PD_CACHE_TTL)
        return ToolResult(output=output)

    lines = [
        "| # | Title | Status | Urgency | Service | URL |",
        "|---|-------|--------|---------|---------|-----|",
    ]
    for i, inc in enumerate(incidents, 1):
        title = inc.get("title", "N/A")
        inc_status = inc.get("status", "N/A")
        urgency = inc.get("urgency", "N/A")
        service = inc.get("service", {}).get("summary", "N/A")
        html_url = inc.get("html_url", "N/A")
        lines.append(f"| {i} | {title} | {inc_status} | {urgency} | {service} | {html_url} |")

    output = "\n".join(lines)
    _set_cache(ck, output, ttl=_PD_CACHE_TTL)
    return ToolResult(output=output)

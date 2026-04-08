"""Slack alert correlation tool — search channel messages via conversations.history."""

from __future__ import annotations

import datetime
import logging
from typing import Any

from vaig.core.config import SlackConfig
from vaig.tools.base import ToolResult
from vaig.tools.integrations._cache import _cache_key, _get_cached, _set_cache
from vaig.tools.integrations._http import api_request

logger = logging.getLogger(__name__)

_SLACK_CACHE_TTL: int = 30  # seconds


def search_slack_messages(
    *,
    config: SlackConfig,
    channel_id: str,
    query: str = "",
    limit: int = 20,
    hours_back: int = 4,
) -> ToolResult:
    """Fetch recent messages from a Slack channel via conversations.history.

    Args:
        config: Slack configuration with bot_token.
        channel_id: Slack channel ID to search (e.g. ``"C01ABCDEF"``).
        query: Optional keyword to filter messages (case-insensitive
            substring match applied client-side).
        limit: Maximum number of messages to return (default 20).
        hours_back: How many hours back to search (default 4).

    Returns:
        ``ToolResult`` with a formatted Markdown list of messages or an
        error message on failure.
    """
    if not channel_id:
        return ToolResult(output="channel_id is required.", error=True)

    # ── Cache check ──────────────────────────────────────────
    ck = _cache_key("slack", "history", channel_id, query, str(limit), str(hours_back))
    cached = _get_cached(ck, ttl=_SLACK_CACHE_TTL)
    if cached is not None:
        return ToolResult(output=cached)

    # ── Build request ────────────────────────────────────────
    url = "https://slack.com/api/conversations.history"
    bot_token = config.bot_token.get_secret_value()
    headers = {
        "Authorization": f"Bearer {bot_token}",
        "Content-Type": "application/json",
    }

    oldest = datetime.datetime.now(tz=datetime.UTC) - datetime.timedelta(hours=hours_back)
    params: dict[str, Any] = {
        "channel": channel_id,
        "oldest": str(oldest.timestamp()),
        "limit": str(min(limit * 2, 200)),  # fetch extra to allow client-side filtering
        "inclusive": "true",
    }

    # ── Execute ──────────────────────────────────────────────
    data, error = api_request(
        "GET", url, headers=headers, params=params, service_name="Slack",
    )
    if error is not None:
        return error

    # Slack API returns ok=false for application-level errors
    if not (data or {}).get("ok", False):
        slack_error = (data or {}).get("error", "unknown_error")
        return ToolResult(
            output=f"Slack API error: {slack_error}",
            error=True,
        )

    # ── Parse and format ─────────────────────────────────────
    messages: list[dict[str, Any]] = (data or {}).get("messages", [])

    # Apply optional keyword filter
    if query:
        q_lower = query.lower()
        messages = [m for m in messages if q_lower in m.get("text", "").lower()]

    messages = messages[:limit]

    if not messages:
        output = f"No messages found in channel {channel_id} within the last {hours_back}h."
        _set_cache(ck, output, ttl=_SLACK_CACHE_TTL)
        return ToolResult(output=output)

    lines = [f"**Recent messages in channel {channel_id}** (last {hours_back}h):\n"]
    for i, msg in enumerate(messages, 1):
        text = msg.get("text", "")
        ts = msg.get("ts", "")
        user = msg.get("user", "bot")
        # Convert Unix timestamp to readable format
        try:
            dt = datetime.datetime.fromtimestamp(float(ts), tz=datetime.UTC)
            time_str = dt.strftime("%Y-%m-%d %H:%M UTC")
        except (ValueError, OSError):
            time_str = ts
        # Truncate long messages
        if len(text) > 200:
            text = text[:200] + "…"
        lines.append(f"{i}. **[{time_str}]** ({user}): {text}")

    output = "\n".join(lines)
    _set_cache(ck, output, ttl=_SLACK_CACHE_TTL)
    return ToolResult(output=output)

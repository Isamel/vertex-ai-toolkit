"""Alert correlation tool registry — factory for incident management tools."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vaig.tools.base import ToolDef, ToolParam
from vaig.tools.categories import INCIDENT_MANAGEMENT

if TYPE_CHECKING:
    from vaig.core.config import Settings

logger = logging.getLogger(__name__)


def create_alert_correlation_tools(settings: Settings) -> list[ToolDef]:
    """Create alert correlation tools gated by per-integration config.

    Each tool is only created when its integration is enabled AND has
    valid credentials configured.  Returns an empty list when nothing
    is enabled.
    """
    tools: list[ToolDef] = []

    # ── PagerDuty ────────────────────────────────────────────
    if settings.pagerduty.enabled and settings.pagerduty.api_token:
        from vaig.tools.integrations.pagerduty import list_pagerduty_incidents

        pd_cfg = settings.pagerduty
        tools.append(
            ToolDef(
                name="list_pagerduty_incidents",
                description=(
                    "List active PagerDuty incidents (triggered/acknowledged). "
                    "Use this to check if there are related incidents when diagnosing issues."
                ),
                parameters=[
                    ToolParam(
                        name="status",
                        type="string",
                        description="Comma-separated statuses to filter: triggered, acknowledged, resolved (default: triggered,acknowledged)",
                        required=False,
                    ),
                    ToolParam(
                        name="service_name",
                        type="string",
                        description="Filter incidents by service name substring (case-insensitive)",
                        required=False,
                    ),
                    ToolParam(
                        name="limit",
                        type="integer",
                        description="Maximum number of incidents to return (default: 25)",
                        required=False,
                    ),
                ],
                execute=lambda status="triggered,acknowledged", service_name="", limit=25, _cfg=pd_cfg: list_pagerduty_incidents(
                    config=_cfg, status=status, service_name=service_name, limit=int(limit),
                ),
                categories=frozenset({INCIDENT_MANAGEMENT}),
                cacheable=False,  # Tool handles its own caching
            )
        )

    # ── OpsGenie ─────────────────────────────────────────────
    if settings.opsgenie.enabled and settings.opsgenie.api_key.get_secret_value():
        from vaig.tools.integrations.opsgenie import list_opsgenie_alerts

        og_cfg = settings.opsgenie
        tools.append(
            ToolDef(
                name="list_opsgenie_alerts",
                description=(
                    "List open OpsGenie alerts. "
                    "Use this to check if there are related alerts when diagnosing issues."
                ),
                parameters=[
                    ToolParam(
                        name="status",
                        type="string",
                        description="Alert status filter: open, closed, all (default: open)",
                        required=False,
                    ),
                    ToolParam(
                        name="priority",
                        type="string",
                        description="Filter by priority: P1, P2, P3, P4, P5 (default: all)",
                        required=False,
                    ),
                    ToolParam(
                        name="limit",
                        type="integer",
                        description="Maximum number of alerts to return (default: 25)",
                        required=False,
                    ),
                ],
                execute=lambda status="open", priority="", limit=25, _cfg=og_cfg: list_opsgenie_alerts(
                    config=_cfg, status=status, priority=priority, limit=int(limit),
                ),
                categories=frozenset({INCIDENT_MANAGEMENT}),
                cacheable=False,
            )
        )

    # ── Slack ────────────────────────────────────────────────
    if settings.slack.enabled and settings.slack.bot_token.get_secret_value():
        from vaig.tools.integrations.slack import search_slack_messages

        slack_cfg = settings.slack
        tools.append(
            ToolDef(
                name="search_slack_messages",
                description=(
                    "Search recent Slack channel messages for incident-related alerts. "
                    "Use this to find Slack notifications about ongoing issues."
                ),
                parameters=[
                    ToolParam(
                        name="channel_id",
                        type="string",
                        description="Slack channel ID to search (e.g. C01ABCDEF)",
                        required=True,
                    ),
                    ToolParam(
                        name="query",
                        type="string",
                        description="Optional keyword to filter messages (case-insensitive substring match)",
                        required=False,
                    ),
                    ToolParam(
                        name="limit",
                        type="integer",
                        description="Maximum number of messages to return (default: 20)",
                        required=False,
                    ),
                    ToolParam(
                        name="hours_back",
                        type="integer",
                        description="How many hours back to search (default: 4)",
                        required=False,
                    ),
                ],
                execute=lambda channel_id, query="", limit=20, hours_back=4, _cfg=slack_cfg: search_slack_messages(
                    config=_cfg, channel_id=channel_id, query=query, limit=int(limit), hours_back=int(hours_back),
                ),
                categories=frozenset({INCIDENT_MANAGEMENT}),
                cacheable=False,
            )
        )

    if tools:
        logger.info(
            "Registered %d alert correlation tool(s): %s",
            len(tools),
            ", ".join(t.name for t in tools),
        )

    return tools

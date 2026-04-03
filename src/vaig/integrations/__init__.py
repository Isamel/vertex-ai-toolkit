"""Notification integrations — PagerDuty, Google Chat, Slack, Email, webhook server, and dispatch orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vaig.integrations.dispatcher import AlertContext, DispatchResult, NotificationDispatcher

if TYPE_CHECKING:
    from vaig.integrations.email_sender import EmailSender
    from vaig.integrations.google_chat import GoogleChatWebhook
    from vaig.integrations.pagerduty import PagerDutyClient
    from vaig.integrations.slack import SlackWebhook

__all__ = [
    "AlertContext",
    "DispatchResult",
    "EmailSender",
    "GoogleChatWebhook",
    "NotificationDispatcher",
    "PagerDutyClient",
    "SlackWebhook",
    "create_webhook_app",
]


def __getattr__(name: str) -> object:
    """Lazy-import modules that depend on optional ``requests`` package."""
    if name == "create_webhook_app":
        from vaig.integrations.webhook_server import create_webhook_app

        return create_webhook_app
    if name == "GoogleChatWebhook":
        from vaig.integrations.google_chat import GoogleChatWebhook

        return GoogleChatWebhook
    if name == "PagerDutyClient":
        from vaig.integrations.pagerduty import PagerDutyClient

        return PagerDutyClient
    if name == "SlackWebhook":
        from vaig.integrations.slack import SlackWebhook

        return SlackWebhook
    if name == "EmailSender":
        from vaig.integrations.email_sender import EmailSender

        return EmailSender
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

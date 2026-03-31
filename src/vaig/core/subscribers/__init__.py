"""Event bus subscribers — adapters that translate domain events into side effects."""

from vaig.core.subscribers.telemetry_subscriber import TelemetrySubscriber

__all__ = [
    "AuditSubscriber",
    "TelemetrySubscriber",
]


def __getattr__(name: str) -> object:
    """Lazy import for AuditSubscriber to avoid loading [audit] deps eagerly."""
    if name == "AuditSubscriber":
        from vaig.core.subscribers.audit_subscriber import AuditSubscriber

        return AuditSubscriber
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)

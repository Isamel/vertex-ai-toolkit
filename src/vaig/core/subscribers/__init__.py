"""Event bus subscribers — adapters that translate domain events into side effects."""

from vaig.core.subscribers.telemetry_subscriber import TelemetrySubscriber

__all__ = [
    "AuditSubscriber",
    "MemorySubscriber",
    "TelemetrySubscriber",
]


def __getattr__(name: str) -> object:
    """Lazy imports to avoid loading optional deps eagerly."""
    if name == "AuditSubscriber":
        from vaig.core.subscribers.audit_subscriber import AuditSubscriber

        return AuditSubscriber
    if name == "MemorySubscriber":
        from vaig.core.subscribers.memory_subscriber import MemorySubscriber

        return MemorySubscriber
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)

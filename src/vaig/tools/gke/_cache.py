"""Discovery cache — generic TTL-based cache for GKE discovery tools."""

from __future__ import annotations

import time

# ── Discovery cache ───────────────────────────────────────────
# Generic TTL-based cache for discovery tools.  Keyed on a
# colon-delimited string built by ``_cache_key_discovery()``.
_DISCOVERY_CACHE: dict[str, tuple[float, str]] = {}
_DISCOVERY_TTL: int = 60  # seconds


def _cache_key_discovery(*parts: str) -> str:
    """Build a colon-delimited cache key from arbitrary string parts."""
    return ":".join(parts)


def _get_cached(key: str) -> str | None:
    """Return the cached value for *key* if it exists and is within TTL, else ``None``."""
    entry = _DISCOVERY_CACHE.get(key)
    if entry is None:
        return None
    ts, value = entry
    if time.monotonic() - ts > _DISCOVERY_TTL:
        _DISCOVERY_CACHE.pop(key, None)
        return None
    return value


def _set_cache(key: str, value: str) -> None:
    """Store *value* in the discovery cache under *key* with the current timestamp."""
    _DISCOVERY_CACHE[key] = (time.monotonic(), value)


def clear_discovery_cache() -> None:
    """Clear the discovery cache (useful for testing)."""
    _DISCOVERY_CACHE.clear()

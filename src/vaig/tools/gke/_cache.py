"""Discovery cache — generic TTL-based cache for GKE discovery tools."""

from __future__ import annotations

import time

# ── Discovery cache ───────────────────────────────────────────
# Generic TTL-based cache for discovery tools.  Keyed on a
# colon-delimited string built by ``_cache_key_discovery()``.
_DISCOVERY_CACHE: dict[str, tuple[float, str, int]] = {}
_DISCOVERY_TTL: int = 60  # seconds


def _cache_key_discovery(*parts: str) -> str:
    """Build a colon-delimited cache key from arbitrary string parts."""
    return ":".join(parts)


def _get_cached(key: str, ttl: int | None = None) -> str | None:
    """Return the cached value for *key* if it exists and is within TTL, else ``None``.

    When *ttl* is ``None`` the per-entry TTL stored at write time is used.
    When *ttl* is provided it overrides the stored TTL for this lookup.
    """
    entry = _DISCOVERY_CACHE.get(key)
    if entry is None:
        return None
    ts, value, stored_ttl = entry
    effective_ttl = ttl if ttl is not None else stored_ttl
    if time.monotonic() - ts > effective_ttl:
        _DISCOVERY_CACHE.pop(key, None)
        return None
    return value


def _set_cache(key: str, value: str, ttl: int | None = None) -> None:
    """Store *value* in the discovery cache under *key* with the current timestamp.

    When *ttl* is ``None`` the module-level ``_DISCOVERY_TTL`` (60 s) is used.
    """
    _DISCOVERY_CACHE[key] = (time.monotonic(), value, ttl if ttl is not None else _DISCOVERY_TTL)


def clear_discovery_cache() -> None:
    """Clear the discovery cache (useful for testing)."""
    _DISCOVERY_CACHE.clear()

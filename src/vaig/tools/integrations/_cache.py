"""TTL-based cache for alert correlation tools."""

from __future__ import annotations

import time

# ── Alert correlation cache ───────────────────────────────────
# Generic TTL-based cache for integration tools.  Keyed on a
# colon-delimited string built by ``_cache_key()``.
_CACHE: dict[str, tuple[float, str, int]] = {}
_DEFAULT_TTL: int = 60  # seconds


def _cache_key(*parts: str) -> str:
    """Build a colon-delimited cache key from arbitrary string parts."""
    return ":".join(parts)


def _get_cached(key: str, ttl: int | None = None) -> str | None:
    """Return the cached value for *key* if it exists and is within TTL, else ``None``.

    When *ttl* is ``None`` the per-entry TTL stored at write time is used.
    When *ttl* is provided it overrides the stored TTL for this lookup.
    """
    entry = _CACHE.get(key)
    if entry is None:
        return None
    ts, value, stored_ttl = entry
    effective_ttl = ttl if ttl is not None else stored_ttl
    if time.monotonic() - ts > effective_ttl:
        _CACHE.pop(key, None)
        return None
    return value


def _set_cache(key: str, value: str, ttl: int | None = None) -> None:
    """Store *value* in the cache under *key* with the current timestamp.

    When *ttl* is ``None`` the module-level ``_DEFAULT_TTL`` (60 s) is used.
    """
    _CACHE[key] = (time.monotonic(), value, ttl if ttl is not None else _DEFAULT_TTL)


def clear_cache() -> None:
    """Clear the integration cache (useful for testing)."""
    _CACHE.clear()

"""Cache module — thread-safe LRU caches with TTL for Gemini API responses
and tool result deduplication.

``ResponseCache`` caches non-streaming, non-tool-use ``GenerationResult``
objects to avoid redundant API calls when users ask the same (or identical)
questions.

``ToolResultCache`` caches ``ToolResult`` objects keyed by
``(tool_name, tool_args)`` to deduplicate identical tool calls within and
across orchestrator passes.  Per-entry TTL allows tools with different
freshness requirements to coexist in a single cache.

Disabled by default — must be explicitly opted-in via ``CacheConfig.enabled``
(for ResponseCache) or by passing a ``ToolResultCache`` instance to the
tool loop (for tool dedup).

Thread-safety: all public methods acquire ``_lock`` before mutating state.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vaig.tools.base import ToolResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheStats:
    """Immutable snapshot of cache statistics."""

    hits: int
    misses: int
    evictions: int
    size: int
    max_size: int

    @property
    def hit_rate(self) -> float:
        """Hit rate as a fraction (0.0–1.0). Returns 0.0 if no lookups."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


def _make_cache_key(
    prompt: str,
    model_id: str,
    system_instruction: str | None = None,
) -> str:
    """Build a deterministic cache key from query parameters.

    Uses SHA-256 to produce a fixed-length key regardless of prompt size.
    The key incorporates the prompt text, model ID, and system instruction
    so that the same question asked with different models or system
    instructions gets separate cache entries.
    """
    parts = [
        f"model:{model_id}",
        f"system:{system_instruction or ''}",
        f"prompt:{prompt}",
    ]
    raw = "\n".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ResponseCache:
    """Thread-safe LRU cache with TTL for ``GenerationResult`` objects.

    The cache uses an ``OrderedDict`` for O(1) LRU eviction and stores
    entries as ``(value, timestamp)`` tuples for TTL expiration.

    Usage::

        cache = ResponseCache(max_size=128, ttl_seconds=300)

        # Try cache first
        key = _make_cache_key(prompt, model_id, system_instruction)
        cached = cache.get(key)
        if cached is not None:
            return cached

        # API call...
        result = client.generate(...)
        cache.put(key, result)
        return result
    """

    def __init__(self, max_size: int = 128, ttl_seconds: int = 300) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        if ttl_seconds < 0:
            raise ValueError(f"ttl_seconds must be >= 0, got {ttl_seconds}")

        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        # OrderedDict preserves insertion order; we move-to-end on access
        # and pop from the front (oldest) on eviction.
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def max_size(self) -> int:
        """Maximum number of entries the cache can hold."""
        return self._max_size

    @property
    def ttl_seconds(self) -> int:
        """Time-to-live in seconds for cache entries."""
        return self._ttl_seconds

    def get(self, key: str) -> Any | None:
        """Look up a cache entry by key.

        Returns the cached value if found and not expired, otherwise
        ``None``.  On hit, the entry is moved to the end (most-recently
        used).  On TTL expiry, the entry is evicted silently.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None

            value, timestamp = entry

            # Check TTL (0 means no expiration)
            if self._ttl_seconds > 0:
                age = time.monotonic() - timestamp
                if age > self._ttl_seconds:
                    # Expired — evict
                    del self._store[key]
                    self._misses += 1
                    self._evictions += 1
                    logger.debug("Cache entry expired (age=%.1fs): %s…", age, key[:12])
                    return None

            # Hit — move to end (most-recently used)
            self._store.move_to_end(key)
            self._hits += 1
            logger.debug("Cache HIT: %s…", key[:12])
            return value

    def put(self, key: str, value: Any) -> None:
        """Store a value in the cache.

        If the cache is full, the least-recently-used entry is evicted.
        If the key already exists, the entry is updated and moved to the
        end.
        """
        with self._lock:
            if key in self._store:
                # Update existing entry
                self._store[key] = (value, time.monotonic())
                self._store.move_to_end(key)
                return

            # Evict LRU if at capacity
            while len(self._store) >= self._max_size:
                evicted_key, _ = self._store.popitem(last=False)
                self._evictions += 1
                logger.debug("Cache LRU eviction: %s…", evicted_key[:12])

            self._store[key] = (value, time.monotonic())

    def clear(self) -> int:
        """Remove all entries from the cache.

        Returns the number of entries that were cleared.
        """
        with self._lock:
            count = len(self._store)
            self._store.clear()
            logger.info("Cache cleared (%d entries removed)", count)
            return count

    def stats(self) -> CacheStats:
        """Return an immutable snapshot of cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                size=len(self._store),
                max_size=self._max_size,
            )

    @property
    def size(self) -> int:
        """Current number of entries in the cache."""
        with self._lock:
            return len(self._store)

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"ResponseCache(size={len(self._store)}/{self._max_size}, "
                f"ttl={self._ttl_seconds}s, "
                f"hits={self._hits}, misses={self._misses})"
            )


# ── Tool result cache ────────────────────────────────────────


def _make_tool_cache_key(tool_name: str, tool_args: dict[str, Any]) -> str:
    """Build a deterministic cache key from tool name and arguments.

    Uses SHA-256 to produce a fixed-length key.  ``tool_args`` is serialized
    with ``sort_keys=True`` so that ``{"a": 1, "b": 2}`` and
    ``{"b": 2, "a": 1}`` produce the same key.
    """
    raw = tool_name + json.dumps(tool_args, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ToolResultCache:
    """Thread-safe LRU cache with per-entry TTL for ``ToolResult`` objects.

    Deduplicates identical tool calls within and across orchestrator passes.
    Each entry stores its own TTL (supplied at ``put`` time), so tools with
    different freshness requirements can coexist in a single cache instance.

    Error results (``ToolResult.error is True``) are never cached — only
    successful results are stored.

    Usage::

        cache = ToolResultCache(default_ttl=60, max_size=256)

        key = _make_tool_cache_key(tool_name, tool_args)
        cached = cache.get(key)
        if cached is not None:
            return cached          # cache hit

        result = execute_tool(...)
        cache.put(key, result, ttl_seconds=tool_def.cache_ttl_seconds)
        return result
    """

    def __init__(self, default_ttl: int = 60, max_size: int = 256) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        if default_ttl < 0:
            raise ValueError(f"default_ttl must be >= 0, got {default_ttl}")

        self._default_ttl = default_ttl
        self._max_size = max_size
        self._lock = threading.Lock()
        # value: (ToolResult, timestamp, ttl_seconds)
        self._store: OrderedDict[str, tuple[ToolResult, float, int]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def max_size(self) -> int:
        """Maximum number of entries the cache can hold."""
        return self._max_size

    @property
    def default_ttl(self) -> int:
        """Default time-to-live in seconds for cache entries."""
        return self._default_ttl

    def get(self, key: str) -> ToolResult | None:
        """Look up a cached tool result by key.

        Returns the cached ``ToolResult`` if found and not expired, otherwise
        ``None``.  On hit the entry is moved to the end (most-recently used).
        On TTL expiry the entry is evicted silently.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None

            result, timestamp, ttl = entry

            # Check per-entry TTL (0 means no expiration)
            if ttl > 0:
                age = time.monotonic() - timestamp
                if age > ttl:
                    del self._store[key]
                    self._misses += 1
                    self._evictions += 1
                    logger.debug(
                        "Tool cache entry expired (age=%.1fs, ttl=%ds): %s…",
                        age,
                        ttl,
                        key[:12],
                    )
                    return None

            # Hit — move to end (most-recently used)
            self._store.move_to_end(key)
            self._hits += 1
            logger.debug("Tool cache HIT: %s…", key[:12])
            return result

    def get_or_none(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        ttl_override: int | None = None,
    ) -> ToolResult | None:
        """Convenience: look up by tool name + args, with optional TTL override.

        If ``ttl_override`` is given the cached entry is only considered valid
        if its age is within ``ttl_override`` seconds, regardless of the TTL
        the entry was stored with.
        """
        key = _make_tool_cache_key(tool_name, tool_args)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None

            result, timestamp, stored_ttl = entry
            effective_ttl = ttl_override if ttl_override is not None else stored_ttl

            if effective_ttl > 0:
                age = time.monotonic() - timestamp
                if age > effective_ttl:
                    del self._store[key]
                    self._misses += 1
                    self._evictions += 1
                    return None

            self._store.move_to_end(key)
            self._hits += 1
            return result

    def put(self, key: str, result: ToolResult, ttl_seconds: int | None = None) -> None:
        """Store a tool result in the cache.

        Error results (``result.error is True``) are **never** cached.
        If the cache is full the least-recently-used entry is evicted.
        ``ttl_seconds`` defaults to ``self._default_ttl`` if not provided.
        """
        # Never cache error results
        if result.error:
            return

        effective_ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl

        with self._lock:
            if key in self._store:
                self._store[key] = (result, time.monotonic(), effective_ttl)
                self._store.move_to_end(key)
                return

            # Evict LRU if at capacity
            while len(self._store) >= self._max_size:
                evicted_key, _ = self._store.popitem(last=False)
                self._evictions += 1
                logger.debug("Tool cache LRU eviction: %s…", evicted_key[:12])

            self._store[key] = (result, time.monotonic(), effective_ttl)

    def clear(self) -> int:
        """Remove all entries from the cache and reset stats.

        Returns the number of entries that were cleared.
        """
        with self._lock:
            count = len(self._store)
            self._store.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            logger.info("Tool result cache cleared (%d entries removed)", count)
            return count

    def stats(self) -> CacheStats:
        """Return an immutable snapshot of cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                size=len(self._store),
                max_size=self._max_size,
            )

    @property
    def size(self) -> int:
        """Current number of entries in the cache."""
        with self._lock:
            return len(self._store)

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"ToolResultCache(size={len(self._store)}/{self._max_size}, "
                f"default_ttl={self._default_ttl}s, "
                f"hits={self._hits}, misses={self._misses})"
            )
